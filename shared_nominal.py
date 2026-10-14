"""Shared nominal substrate and sole 22-field projection for makers 1--6.

The nominal event catalog and its bounded sensor models originate in maker 1 v1.2.
Episode/context/benign sampling is independent of attack scheduling and execution.
Attack types, labels and phases are audit metadata, never projection inputs.
Ranges and rates are synthetic experimental assumptions, not field calibration.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Dict, List
import zipfile

import numpy as np
import pandas as pd
import shared_context_v4 as shared_context_module

from shared_context_v4 import (
    CONTEXT_FIELD_NAMES,
    GLOBAL_CONTEXT_PROFILE_POOL,
    is_command_context,
    nominal_semantics_for_context,
    sample_shared_context_profile,
    validate_context_profile,
    validate_global_context_pool,
)


# Version 1.1: raw-event consistency revision (2026-10-02).
# - One-second windows only; counts/rates use that same interval.
# - Randomness lives in scenario/message generation, never in normalization.
# - Current command-result flags describe one representative received command;
#   there is no cross-window queue or authenticated packet/replay implementation.
# - Retains the specified shared context pool and bounded benign event model.
# - Default internal-link RF/decoder/retry categories are synthetic convention,
#   not a claim of observed RF measurements on an onboard internal path.
# - Numeric ranges/thresholds remain controlled assumptions, not field calibration.
# - Distribution reports do not select/reject seeds to achieve a target accuracy.
# Model comparisons and cross-class leakage checks follow after makers 2-6.
#
# Design note for the journal experiment:
# NORMAL windows are not modeled as idealized clean observations.
# They include operationally plausible ambiguous variability caused by
# transient link fluctuation, GNSS geometry effects, contact-window
# transitions, decoder warnings, and scheduled command delays. These
# effects may overlap with threat symptoms in a single window, but they
# remain authorized, transient, recoverable, and non-escalating.
#
# ============================================================
# 0. Basic settings
# ============================================================



CODE_VERSION = "2.0-shared-nominal-substrate"
DEFAULT_NUM_ROWS = 50000
DEFAULT_WINDOW_SECONDS = 1
DEFAULT_OUTPUT_PATH = "dataset/normal_observable_dataset.csv"
DEFAULT_RAW_AUDIT_OUTPUT_PATH = "dataset/normal_observable_dataset_with_raw.csv"
DEFAULT_SAMPLE_TIMELINE_OUTPUT_PATH = "dataset/normal_sample_episode_timeline.csv"
DEFAULT_RANDOM_SEED = 42

# Shared episode-length sampler for the six-class benchmark.
# This reduces class-specific episode-length leakage when the normal and
# attack makers are later concatenated into the full controlled benchmark.
# The same distribution should be reused in all attack makers.
COMMON_EPISODE_LENGTH_BUCKETS = {
    "SHORT": (20, 40),
    "MEDIUM": (40, 80),
    "LONG": (80, 140),
}
COMMON_EPISODE_LENGTH_BUCKET_PROBS = {
    "SHORT": 0.35,
    "MEDIUM": 0.45,
    "LONG": 0.20,
}

from contextvars import ContextVar
_RANDOM_SOURCE = ContextVar("nominal_random_source", default=None)
class _RandomSource:
    def __getattr__(self, name):
        source = _RANDOM_SOURCE.get()
        if source is None:
            raise RuntimeError("Nominal sampling must run inside a NominalSubstrate.")
        return getattr(source, name)
RNG = _RandomSource()

LABEL_NORMAL = "NORMAL"

METADATA_FIELDS = [
    "scenario_id",
    "episode_id",
    "episode_type",
    "episode_step",
    "window_id",
    "timestamp",
    "label",
]

# CSV contract: seven metadata columns + 22 observables. Metadata is retained
# for ordering/splitting; only OBSERVABLE_FIELDS may enter model serialization.
MODEL_OUTPUT_FIELDS = METADATA_FIELDS



# ============================================================
# 1. The 22 observable fields
# ============================================================

OBSERVABLE_FIELDS: List[str] = [
    # Observation Context
    "observation_stage",
    "path_type",
    "direction",
    "source_role",
    "destination_role",
    "protocol_context",
    "operational_phase",

    # Message and Control Semantics
    "message_type",
    "service_context",
    "command_intent",
    "payload_or_parameter_context",
    "sequence_freshness_status",
    "control_security_state",

    # RF/GNSS Receiver Evidence
    "signal_quality_state",
    "tracking_stability",
    "navigation_consistency",
    "timing_consistency",

    # Operational and Communication Trace
    "decoder_status",
    "command_execution_result",
    "retry_timeout_state",
    "session_state",
    "state_transition_consistency",
]


FIELD_MEANING: Dict[str, str] = {
    "observation_stage": "Representative observation stage in the monitored LEO-centered window.",
    "path_type": "Dominant or evidence-bearing communication path in the window.",
    "direction": "Dominant or evidence-bearing communication direction in the window.",
    "source_role": "Representative source role related to the observed evidence.",
    "destination_role": "Representative destination role, usually the monitored LEO or its onboard module.",
    "protocol_context": "Dominant or evidence-bearing protocol context in the window.",
    "operational_phase": "Operational phase of the monitored LEO during the window.",

    "message_type": "Representative message type or mixed nominal traffic condition.",
    "service_context": "Representative service context of the observed communication/control activity.",
    "command_intent": "Representative command intent in the window.",
    "payload_or_parameter_context": "Representative payload or parameter context.",
    "sequence_freshness_status": "Freshness status of command/message sequence information.",
    "control_security_state": "Authentication, integrity, and authorization state of control evidence.",

    "signal_quality_state": "Window-level signal quality state derived from RF/link/GNSS measurements.",
    "tracking_stability": "Receiver or GNSS tracking stability during the window.",
    "navigation_consistency": "Consistency of GNSS-derived navigation outputs.",
    "timing_consistency": "Consistency of GNSS or onboard timing outputs.",

    "decoder_status": "Decoder/parser outcome over monitored communication inputs.",
    "command_execution_result": "Command handling or execution result during the window.",
    "retry_timeout_state": "Communication reliability state based on retries, timeouts, and resets.",
    "session_state": "Command/communication/session state during the window.",
    "state_transition_consistency": "Consistency of operational state transition in the monitored LEO.",
}


# ============================================================
# 2. Normal raw observation ranges
# ============================================================
# These are NORMAL baseline ranges for controlled dataset generation.
# They are not universal physical constants. Later, they can be calibrated
# using real receiver/telemetry traces.

NORMAL_RAW_RANGES: Dict[str, Dict[str, float]] = {
    # Gateway/GCS <-> LEO TT&C link
    "ttc_snr_db": {"mean": 18.5, "std": 2.2, "min": 12.0, "max": 25.0},
    "ttc_frame_loss_rate": {"mean": 0.006, "std": 0.004, "min": 0.0, "max": 0.020},
    "ttc_rtt_ms": {"mean": 320.0, "std": 120.0, "min": 80.0, "max": 800.0},

    # LEO <-> LEO ISL
    "isl_snr_db": {"mean": 17.0, "std": 2.5, "min": 10.0, "max": 24.0},
    "isl_frame_loss_rate": {"mean": 0.008, "std": 0.005, "min": 0.0, "max": 0.030},
    "isl_rtt_ms": {"mean": 70.0, "std": 35.0, "min": 10.0, "max": 200.0},

    # IoT -> LEO MIOTY-like uplink
    "iot_snr_db": {"mean": 0.0, "std": 4.0, "min": -12.0, "max": 10.0},
    "iot_frame_loss_rate": {"mean": 0.025, "std": 0.020, "min": 0.0, "max": 0.080},
    "iot_rtt_ms": {"mean": 1800.0, "std": 900.0, "min": 500.0, "max": 5000.0},

    # UAV -> LEO 3GPP NTN-like path
    "uav_rsrp_dbm": {"mean": -94.0, "std": 6.0, "min": -110.0, "max": -80.0},
    "uav_sinr_db": {"mean": 10.0, "std": 4.0, "min": 0.0, "max": 20.0},
    "uav_frame_loss_rate": {"mean": 0.015, "std": 0.015, "min": 0.0, "max": 0.050},
    "uav_rtt_ms": {"mean": 500.0, "std": 250.0, "min": 100.0, "max": 1500.0},

    # GNSS constellation -> LEO GNSS receiver
    "gnss_cn0_mean_dbhz": {"mean": 43.0, "std": 3.0, "min": 35.0, "max": 50.0},
    "gnss_cn0_std_db": {"mean": 1.5, "std": 0.6, "min": 0.5, "max": 3.0},
    "gnss_tracking_jitter": {"mean": 0.08, "std": 0.04, "min": 0.0, "max": 0.20},
    "gnss_carrier_lock_ratio": {"mean": 0.985, "std": 0.010, "min": 0.95, "max": 1.0},
    "position_jump_m": {"mean": 2.5, "std": 2.0, "min": 0.0, "max": 10.0},
    "velocity_jump_mps": {"mean": 0.15, "std": 0.12, "min": 0.0, "max": 0.5},
    "pseudorange_residual_m": {"mean": 2.5, "std": 1.8, "min": 0.0, "max": 8.0},
    "time_residual_ns": {"mean": 35.0, "std": 25.0, "min": 0.0, "max": 100.0},
    "clock_drift_ppm": {"mean": 0.015, "std": 0.012, "min": 0.0, "max": 0.05},

    # Command/session/operation behavior
    "session_age_s": {"mean": 180.0, "std": 120.0, "min": 1.0, "max": 600.0},
    "command_rate_per_min": {"mean": 1.2, "std": 0.9, "min": 0.0, "max": 3.0},
    "parameter_deviation_ratio": {"mean": 0.03, "std": 0.03, "min": 0.0, "max": 0.10},
}

NORMAL_EPISODES = {
    "ROUTINE_MONITORING": {
        "duration_range": (20, 80),
        "observation_stage": "MULTI_STAGE_MONITORING",
        "path_type": "MULTI_PATH",
        "direction": "MIXED_DIRECTION",
        "source_role": "MULTIPLE_AUTHORIZED_SOURCES",
        "destination_role": "MONITORED_LEO",
        "protocol_context": "MULTI_PROTOCOL",
        "operational_phase": "ROUTINE_MONITORING",
        "message_type": "NOMINAL_MIXED_TRAFFIC",
        "service_context": "HOUSEKEEPING",
        "command_intent": "NO_COMMAND",
    },

    "CONTACT_WINDOW": {
        "duration_range": (20, 60),
        "observation_stage": "PROTOCOL_HANDLER",
        "path_type": "GATEWAY_LEO",
        "direction": "SPACE_TO_GROUND",
        "source_role": "MONITORED_LEO",
        "destination_role": "GROUND_GATEWAY",
        "protocol_context": "CCSDS_TTC",
        "operational_phase": "CONTACT_WINDOW",
        "message_type": "TELEMETRY",
        "service_context": "TTC_TELEMETRY",
        "command_intent": "NO_COMMAND",
    },

    "COMMAND_SESSION": {
        "duration_range": (12, 30),
        "observation_stage": "COMMAND_HANDLER",
        "path_type": "GATEWAY_LEO",
        "direction": "GROUND_TO_SPACE",
        "source_role": "GROUND_GATEWAY",
        "destination_role": "MONITORED_LEO",
        "protocol_context": "CCSDS_TTC",
        "operational_phase": "COMMAND_SESSION",
        "message_type": "TELECOMMAND",
        "service_context": "TTC_CONTROL",
        "command_intent": "NOMINAL_CONTROL",
    },

    "GNSS_NAVIGATION": {
        "duration_range": (20, 80),
        "observation_stage": "GNSS_RECEIVER",
        "path_type": "GNSS_LEO",
        "direction": "GNSS_TO_SPACE",
        "source_role": "GNSS_CONSTELLATION",
        "destination_role": "MONITORED_LEO",
        "protocol_context": "GNSS_NAV",
        "operational_phase": "GNSS_NAVIGATION",
        "message_type": "GNSS_OBSERVATION",
        "service_context": "GNSS_PNT",
        "command_intent": "NO_COMMAND",
    },

    "IOT_REPORT_COLLECTION": {
        "duration_range": (12, 45),
        "observation_stage": "PROTOCOL_HANDLER",
        "path_type": "IOT_LEO",
        "direction": "USER_TO_SPACE",
        "source_role": "IOT_TERMINAL",
        "destination_role": "MONITORED_LEO",
        "protocol_context": "MIOTY",
        "operational_phase": "DATA_RELAY",
        "message_type": "IOT_REPORT",
        "service_context": "IOT_UPLINK",
        "command_intent": "NO_COMMAND",
    },

    "UAV_NTN_SESSION": {
        "duration_range": (12, 45),
        "observation_stage": "SESSION_MANAGER",
        "path_type": "UAV_LEO",
        "direction": "USER_TO_SPACE",
        "source_role": "UAV_TERMINAL",
        "destination_role": "MONITORED_LEO",
        "protocol_context": "3GPP_NTN",
        "operational_phase": "DATA_RELAY",
        "message_type": "UAV_TELEMETRY",
        "service_context": "UAV_NTN_SERVICE",
        "command_intent": "NO_COMMAND",
    },

    "ISL_RELAY": {
        "duration_range": (12, 45),
        "observation_stage": "PROTOCOL_HANDLER",
        "path_type": "LEO_LEO",
        "direction": "SPACE_TO_SPACE",
        "source_role": "PEER_LEO",
        "destination_role": "MONITORED_LEO",
        "protocol_context": "CCSDS_ISL",
        "operational_phase": "DATA_RELAY",
        "message_type": "TELEMETRY",
        "service_context": "ISL_ROUTING",
        "command_intent": "NO_COMMAND",
    },

    "PAYLOAD_OPERATION": {
        "duration_range": (20, 70),
        "observation_stage": "OBC_OPERATIONAL_TRACE",
        "path_type": "INTERNAL_ONBOARD",
        "direction": "INTERNAL",
        "source_role": "MONITORED_LEO",
        "destination_role": "OBC",
        "protocol_context": "ONBOARD_INTERNAL",
        "operational_phase": "PAYLOAD_OPERATION",
        "message_type": "TELEMETRY",
        "service_context": "PAYLOAD_DATA",
        "command_intent": "NO_COMMAND",
    },
}

# ============================================================
# 2-1. Benign normal-operation variability profiles
# ============================================================
# These profiles intentionally keep the label as NORMAL while introducing
# non-adversarial degradation and operational variability. The purpose is to
# prevent the detector from learning a trivial shortcut such as
# "any degraded link token = attack." These conditions should be described
# as controlled, protocol-consistent benign impairments, not as real flight logs.

NORMAL_CONDITION_PROFILES: Dict[str, Dict[str, List[Any]]] = {
    # The probabilities below intentionally reduce the proportion of perfectly
    # clean normal windows. They model operationally plausible ambiguous
    # normal windows caused by transient link fluctuation, GNSS geometry
    # effects, contact-window transition, decoder warnings, scheduled
    # maintenance/control activity, command retransmission, session resync,
    # and authentication retry. The label remains NORMAL; the difference from
    # attacks should be temporal: benign effects are short, authorized,
    # recoverable, and non-escalating rather than sustained, recurrent, or
    # progressively worsening.
    "ROUTINE_MONITORING": {
        "conditions": [
            "NOMINAL",
            "BENIGN_LINK_FLUCTUATION",
            "CONTACT_HANDOVER_TRANSITION",
            "TEMPORARY_SESSION_DELAY",
            "OBC_LOAD_TRANSIENT",
            "BENIGN_GNSS_MONITORING_VARIATION",
            "SCHEDULED_MAINTENANCE_ACTIVITY",
            "ROUTING_UPDATE",
            "GCS_GATEWAY_SYNC",
        ],
        "probs": [0.42, 0.10, 0.06, 0.05, 0.07, 0.08, 0.10, 0.08, 0.04],
    },
    "CONTACT_WINDOW": {
        "conditions": [
            "NOMINAL",
            "LOW_ELEVATION_CONTACT",
            "BENIGN_LINK_FLUCTUATION",
            "CONTACT_HANDOVER_TRANSITION",
            "BENIGN_DECODER_WARNING",
            "BENIGN_RF_RECEIVER_VARIATION",
            "SESSION_RESYNC",
            "TEMPORARY_SIGNAL_FLUCTUATION",
        ],
        "probs": [0.40, 0.15, 0.12, 0.11, 0.08, 0.07, 0.04, 0.03],
    },
    "COMMAND_SESSION": {
        "conditions": [
            "NOMINAL",
            "LOW_ELEVATION_CONTACT",
            "TEMPORARY_SESSION_DELAY",
            "BENIGN_DECODER_WARNING",
            "CONTACT_HANDOVER_TRANSITION",
            "ANTENNA_POINTING_VARIATION",
            "SCHEDULED_CONFIG_UPDATE",
            "VALID_REPEATED_STATUS_REQUEST",
            "BENIGN_COMMAND_HOLD",
            "BENIGN_OUT_OF_ORDER_DELIVERY",
            "AUTH_RETRY",
            "COMMAND_RETRANSMISSION",
            "SESSION_RESYNC",
        ],
        # More benign command/session overlap is needed under the shared
        # episode-length distribution; otherwise long command sessions become
        # too clean and message-only/replay-vs-normal shortcuts are less
        # effectively controlled.
        "probs": [0.20, 0.05, 0.11, 0.05, 0.04, 0.02, 0.12, 0.12, 0.11, 0.05, 0.08, 0.09, 0.02],
    },
    "GNSS_NAVIGATION": {
        "conditions": [
            "NOMINAL",
            "GNSS_GEOMETRY_OR_MULTIPATH",
            "BENIGN_GNSS_MONITORING_VARIATION",
            "BENIGN_PNT_RESIDUAL_VARIATION",
            "BENIGN_LINK_FLUCTUATION",
            "TEMPORARY_SESSION_DELAY",
        ],
        "probs": [0.35, 0.26, 0.13, 0.16, 0.05, 0.05],
    },
    "IOT_REPORT_COLLECTION": {
        "conditions": [
            "NOMINAL",
            "BENIGN_LINK_FLUCTUATION",
            "BENIGN_DECODER_WARNING",
            "CONTACT_HANDOVER_TRANSITION",
            "TEMPORARY_SESSION_DELAY",
            "BENIGN_RF_RECEIVER_VARIATION",
            "INTERMITTENT_PACKET_LOSS",
        ],
        "probs": [0.45, 0.20, 0.12, 0.08, 0.06, 0.05, 0.04],
    },
    "UAV_NTN_SESSION": {
        "conditions": [
            "NOMINAL",
            "BENIGN_LINK_FLUCTUATION",
            "ANTENNA_POINTING_VARIATION",
            "CONTACT_HANDOVER_TRANSITION",
            "TEMPORARY_SESSION_DELAY",
            "BENIGN_DECODER_WARNING",
            "BENIGN_RF_RECEIVER_VARIATION",
            "TEMPORARY_SIGNAL_FLUCTUATION",
        ],
        "probs": [0.44, 0.15, 0.11, 0.08, 0.06, 0.05, 0.07, 0.04],
    },
    "ISL_RELAY": {
        "conditions": [
            "NOMINAL",
            "BENIGN_LINK_FLUCTUATION",
            "ANTENNA_POINTING_VARIATION",
            "CONTACT_HANDOVER_TRANSITION",
            "BENIGN_DECODER_WARNING",
            "BENIGN_RF_RECEIVER_VARIATION",
            "ROUTING_UPDATE",
        ],
        "probs": [0.45, 0.18, 0.10, 0.09, 0.08, 0.07, 0.03],
    },
    "PAYLOAD_OPERATION": {
        "conditions": [
            "NOMINAL",
            "OBC_LOAD_TRANSIENT",
            "TEMPORARY_SESSION_DELAY",
            "BENIGN_GNSS_MONITORING_VARIATION",
            "SCHEDULED_MAINTENANCE_ACTIVITY",
            "CONFIG_SYNC",
        ],
        "probs": [0.58, 0.20, 0.06, 0.06, 0.06, 0.04],
    },
}
BENIGN_CONDITION_PHASES = [
    "BASELINE",
    "ONSET",
    "TRANSIENT_IMPAIRMENT",
    "RECOVERY",
    "STABILIZED",
]


# Conditions grouped by their intended benign overlap role. These sets are used
# only by the generator and audit outputs; they are not exported as learning
# features. Their purpose is to keep NORMAL windows realistic but transient.
BENIGN_LINK_CONDITIONS = {
    "BENIGN_LINK_FLUCTUATION",
    "LOW_ELEVATION_CONTACT",
    "ANTENNA_POINTING_VARIATION",
    "BENIGN_RF_RECEIVER_VARIATION",
    "TEMPORARY_SIGNAL_FLUCTUATION",
    "INTERMITTENT_PACKET_LOSS",
}

BENIGN_GNSS_CONDITIONS = {
    "GNSS_GEOMETRY_OR_MULTIPATH",
    "BENIGN_GNSS_MONITORING_VARIATION",
    "BENIGN_PNT_RESIDUAL_VARIATION",
}

BENIGN_COMMAND_SEMANTIC_CONDITIONS = {
    "SCHEDULED_CONFIG_UPDATE",
    "VALID_REPEATED_STATUS_REQUEST",
    "BENIGN_COMMAND_HOLD",
    "BENIGN_OUT_OF_ORDER_DELIVERY",
    "AUTH_RETRY",
    "COMMAND_RETRANSMISSION",
    "SESSION_RESYNC",
}

BENIGN_MAINTENANCE_CONDITIONS = {
    "SCHEDULED_MAINTENANCE_ACTIVITY",
    "ROUTING_UPDATE",
    "CONFIG_SYNC",
    "GCS_GATEWAY_SYNC",
    "FIREWALL_POLICY_UPDATE",
}


# ============================================================
# 3. Helper sampling functions
# ============================================================

def clipped_normal(name: str, decimals: int = 4) -> float:
    spec = NORMAL_RAW_RANGES[name]
    value = RNG.normal(spec["mean"], spec["std"])
    value = float(np.clip(value, spec["min"], spec["max"]))
    return round(value, decimals)


def weighted_choice(items: List[Any], probs: List[float]) -> Any:
    probs_array = np.array(probs, dtype=float)
    probs_array = probs_array / probs_array.sum()
    selected_index = int(RNG.choice(len(items), p=probs_array))
    return items[selected_index]


def count_choice(values: List[int], probs: List[float]) -> int:
    return int(weighted_choice(values, probs))


def flag_choice(prob_one: float) -> int:
    return int(RNG.random() < prob_one)


def uniform_float(low: float, high: float, decimals: int = 4) -> float:
    return round(float(RNG.uniform(low, high)), decimals)


def uniform_int(low: int, high: int) -> int:
    return int(RNG.integers(low, high + 1))


def choose_normal_condition(episode_type: str) -> str:
    profile = NORMAL_CONDITION_PROFILES[episode_type]
    return weighted_choice(profile["conditions"], profile["probs"])


def sample_benign_event_plan(condition: str, duration: int) -> Dict[str, int]:
    """Create a short benign event plan within a normal episode.

    Earlier versions spread a benign condition over most of an episode. That
    made NORMAL look persistently degraded, which was not a good operational
    assumption. The revised generator creates one short event burst inside an
    otherwise normal episode. The raw audit keeps the event location so that
    Step 10/11 can later inspect temporal progression without exposing this
    metadata to the classifier.
    """
    if condition == "NOMINAL":
        return {
            "benign_event_start": -1,
            "benign_event_duration": 0,
            "benign_event_end": -1,
        }

    if condition in BENIGN_COMMAND_SEMANTIC_CONDITIONS:
        candidate_durations = [4, 5, 6, 7, 8, 9]
        duration_probs = [0.12, 0.22, 0.26, 0.20, 0.14, 0.06]
    elif condition in BENIGN_GNSS_CONDITIONS:
        candidate_durations = [4, 5, 6, 8, 10]
        duration_probs = [0.12, 0.22, 0.28, 0.26, 0.12]
    elif condition in BENIGN_MAINTENANCE_CONDITIONS:
        candidate_durations = [4, 5, 6, 7, 8]
        duration_probs = [0.14, 0.24, 0.28, 0.22, 0.12]
    else:
        candidate_durations = [3, 4, 5, 6, 7]
        duration_probs = [0.12, 0.22, 0.30, 0.24, 0.12]

    event_duration = int(weighted_choice(candidate_durations, duration_probs))

    # The benchmark-wide episode length distribution can create longer normal
    # episodes than the original maker.  Keep benign events transient, but scale
    # their duration with episode length so that overlap evidence is not diluted
    # away in long episodes.  These values are audit/control variables only and
    # are not serialized into the learning-ready 22-field input.
    if condition in BENIGN_COMMAND_SEMANTIC_CONDITIONS or condition in BENIGN_MAINTENANCE_CONDITIONS:
        min_scaled = max(4, int(round(duration * 0.12)))
        max_scaled = max(min_scaled, int(round(duration * 0.25)))
    elif condition in BENIGN_GNSS_CONDITIONS:
        min_scaled = max(4, int(round(duration * 0.10)))
        max_scaled = max(min_scaled, int(round(duration * 0.22)))
    else:
        min_scaled = max(3, int(round(duration * 0.08)))
        max_scaled = max(min_scaled, int(round(duration * 0.18)))

    event_duration = max(event_duration, min_scaled)
    event_duration = min(event_duration, max_scaled, max(1, duration))

    if duration <= event_duration + 2:
        event_start = 0
    else:
        # Keep at least one ordinary window around most benign events.
        event_start = int(RNG.integers(1, duration - event_duration))

    return {
        "benign_event_start": int(event_start),
        "benign_event_duration": int(event_duration),
        "benign_event_end": int(event_start + event_duration - 1),
    }


def get_benign_condition_phase(
    step: int,
    duration: int,
    condition: str,
    event_start: int | None = None,
    event_duration: int | None = None,
) -> str:
    if condition == "NOMINAL":
        return "BASELINE"

    if event_start is None or event_duration is None:
        ratio = step / max(duration - 1, 1)
        if ratio < 0.18:
            return "BASELINE"
        if ratio < 0.33:
            return "ONSET"
        if ratio < 0.70:
            return "TRANSIENT_IMPAIRMENT"
        if ratio < 0.86:
            return "RECOVERY"
        return "STABILIZED"

    if event_start < 0 or event_duration <= 0:
        return "BASELINE"

    event_end = event_start + event_duration - 1

    if step < event_start:
        return "BASELINE"
    if step == event_start:
        return "ONSET"
    if event_start < step < event_end:
        return "TRANSIENT_IMPAIRMENT"
    if step == event_end:
        return "RECOVERY"
    if step <= event_end + 2:
        return "STABILIZED"
    return "BASELINE"


def phase_to_benign_severity(condition: str, phase: str) -> str:
    if condition == "NOMINAL" or phase in ["BASELINE", "STABILIZED"]:
        return "NONE"

    if phase == "ONSET":
        return weighted_choice(["MILD", "MODERATE"], [0.85, 0.15])

    if phase == "TRANSIENT_IMPAIRMENT":
        # Keep benign impairments mostly mild, with limited moderate windows
        # to create overlap with attack observables without making normal
        # windows look like sustained attacks.
        if condition in [
            "LOW_ELEVATION_CONTACT",
            "GNSS_GEOMETRY_OR_MULTIPATH",
            "ANTENNA_POINTING_VARIATION",
            "TEMPORARY_SIGNAL_FLUCTUATION",
            "INTERMITTENT_PACKET_LOSS",
        ]:
            return weighted_choice(["MILD", "MODERATE"], [0.65, 0.35])

        if condition in [
            "BENIGN_DECODER_WARNING",
            "TEMPORARY_SESSION_DELAY",
            "CONTACT_HANDOVER_TRANSITION",
            "SCHEDULED_CONFIG_UPDATE",
            "VALID_REPEATED_STATUS_REQUEST",
            "BENIGN_COMMAND_HOLD",
            "BENIGN_OUT_OF_ORDER_DELIVERY",
            "AUTH_RETRY",
            "COMMAND_RETRANSMISSION",
            "SESSION_RESYNC",
            "SCHEDULED_MAINTENANCE_ACTIVITY",
            "ROUTING_UPDATE",
            "CONFIG_SYNC",
            "GCS_GATEWAY_SYNC",
            "FIREWALL_POLICY_UPDATE",
        ]:
            return weighted_choice(["MILD", "MODERATE"], [0.75, 0.25])

        return weighted_choice(["MILD", "MODERATE"], [0.80, 0.20])

    if phase == "RECOVERY":
        return weighted_choice(["NONE", "MILD"], [0.35, 0.65])

    return "NONE"





def sample_common_episode_duration() -> tuple[str, int]:
    """Sample episode duration from the benchmark-wide length distribution.

    The same SHORT/MEDIUM/LONG mixture should be used in all six maker files.
    Keeping comparable episode lengths prevents a TST model from exploiting
    class-specific generator duration templates instead of temporal evidence.
    """
    buckets = list(COMMON_EPISODE_LENGTH_BUCKETS.keys())
    probs = [COMMON_EPISODE_LENGTH_BUCKET_PROBS[b] for b in buckets]
    bucket = weighted_choice(buckets, probs)
    low, high = COMMON_EPISODE_LENGTH_BUCKETS[bucket]
    duration = int(RNG.integers(low, high + 1))
    return bucket, duration


# Revision 1.2: legitimate operations that earlier versions reserved, by
# construction, to attack generators.  A normal mission does change modes,
# reconfigure payloads, adjust attitude and update routing, and operators do
# issue commands that are rejected (parameter error) or blocked by the phase
# policy.  Without these, command-intent vocabulary alone separates the
# command-related threat classes from NORMAL even in windows without any
# observed attack evidence.  Values below reuse tokens already defined by
# makers 4-6; no new vocabulary is introduced.
LEGITIMATE_OPERATION_MODES = {
    "SCHEDULED_MODE_TRANSITION": "MODE_CHANGE",
    "PAYLOAD_RECONFIGURATION": "PAYLOAD_CONFIG",
    "ATTITUDE_MANEUVER_SEQUENCE": "ATTITUDE_ADJUST",
    "ROUTING_TABLE_UPDATE": "ROUTE_UPDATE",
}
LEGITIMATE_OPERATION_INTENTS = set(LEGITIMATE_OPERATION_MODES.values())
OPERATOR_PARAMETER_ERROR_PROB = 0.015   # representative command REJECTED (parameter out of range)
PHASE_POLICY_BLOCK_PROB = 0.025         # representative command BLOCKED by the phase policy


def choose_normal_command_mode(normal_condition: str) -> str:
    """Sample a benign command mode only after a shared context is known.

    This is intentionally independent of the class label and does not affect
    Observation Context fields.  It only controls NORMAL evidence values when
    the already-sampled operational context supports command activity.
    """
    if normal_condition in BENIGN_COMMAND_SEMANTIC_CONDITIONS:
        return normal_condition

    if normal_condition in BENIGN_MAINTENANCE_CONDITIONS:
        return "SCHEDULED_CONFIG_UPDATE"

    return weighted_choice(
        [
            "MIXED_NOMINAL_COMMAND",
            "STATUS_POLLING",
            "TIME_SYNC_SEQUENCE",
            "NOMINAL_CONTROL_SEQUENCE",
            "ROUTINE_HOUSEKEEPING",
            "ORBIT_MAINTENANCE",
            "SCHEDULED_MODE_TRANSITION",
            "PAYLOAD_RECONFIGURATION",
            "ATTITUDE_MANEUVER_SEQUENCE",
            "ROUTING_TABLE_UPDATE",
        ],
        [0.36, 0.17, 0.12, 0.09, 0.07, 0.05, 0.04, 0.04, 0.03, 0.03],
    )


def _build_nominal_timeline(num_rows: int) -> List[Dict[str, Any]]:
    timeline = []
    episode_index = 0
    window_id = 0

    # Keep an audit-only nominal cycle for benign-condition diversity.  The
    # learning-visible context is NOT taken from this episode_type; it is sampled
    # per window from shared_context_v4.GLOBAL_CONTEXT_PROFILE_POOL.  This breaks
    # the previous normal-only context template and prevents the TST from using
    # an episode-type-specific context trajectory as a label proxy.
    episode_order = [
        "ROUTINE_MONITORING",
        "CONTACT_WINDOW",
        "COMMAND_SESSION",
        "GNSS_NAVIGATION",
        "CONTACT_WINDOW",
        "COMMAND_SESSION",
        "GNSS_NAVIGATION",
        "IOT_REPORT_COLLECTION",
        "UAV_NTN_SESSION",
        "ISL_RELAY",
        "PAYLOAD_OPERATION",
    ]

    validate_global_context_pool()

    while window_id < num_rows:
        episode_type = episode_order[episode_index % len(episode_order)]
        template = dict(NORMAL_EPISODES[episode_type])
        episode_length_bucket, duration = sample_common_episode_duration()

        normal_condition = choose_normal_condition(episode_type)
        benign_event_plan = sample_benign_event_plan(normal_condition, duration)
        episode_id = f"NORMAL_EP_{episode_index:05d}"

        for step in range(duration):
            if window_id >= num_rows:
                break

            # Label-agnostic context sampling.  Do not pass label/scenario here.
            # compatible_family=None keeps the same global family mixture that
            # all six makers should reuse.
            context_profile = sample_shared_context_profile(RNG, compatible_family=None)
            validate_context_profile(context_profile)
            semantic_defaults = nominal_semantics_for_context(context_profile, RNG)

            row_template = dict(template)
            row_template.update({k: v for k, v in context_profile.items() if k in CONTEXT_FIELD_NAMES})
            row_template.update(semantic_defaults)

            condition_phase = get_benign_condition_phase(
                step=step,
                duration=duration,
                condition=normal_condition,
                event_start=benign_event_plan["benign_event_start"],
                event_duration=benign_event_plan["benign_event_duration"],
            )
            event_start = benign_event_plan["benign_event_start"]
            event_duration = benign_event_plan["benign_event_duration"]
            event_end = benign_event_plan["benign_event_end"]
            event_active = int(event_start <= step <= event_end) if event_start >= 0 else 0
            event_local_step = int(step - event_start) if event_active else -1
            condition_severity = phase_to_benign_severity(
                condition=normal_condition,
                phase=condition_phase,
            )

            if is_command_context(context_profile):
                active_condition = normal_condition if condition_severity != "NONE" else "NOMINAL"
                episode_command_mode = choose_normal_command_mode(active_condition)
            else:
                episode_command_mode = "NO_COMMAND_CONTEXT"
                row_template["command_intent"] = "NO_COMMAND"

            episode = {
                "scenario_id": "NORMAL_SCENARIO_0001",
                "episode_id": episode_id,
                "episode_type": episode_type,
                "episode_step": step,
                "episode_length": duration,
                "episode_length_bucket": episode_length_bucket,
                "elapsed_seconds": step * DEFAULT_WINDOW_SECONDS,
                "normal_condition": normal_condition,
                "condition_phase": condition_phase,
                "condition_severity": condition_severity,
                "benign_event_start": event_start,
                "benign_event_duration": event_duration,
                "benign_event_end": event_end,
                "benign_event_active": event_active,
                "benign_event_local_step": event_local_step,
                "episode_command_mode": episode_command_mode,
                "shared_context_profile": context_profile.get("shared_context_profile", "UNKNOWN"),
                "shared_context_family": context_profile.get("shared_context_family", "UNKNOWN"),
                # Legacy audit key retained so older inspection scripts still work.
                "normal_context_profile": context_profile.get("shared_context_profile", "UNKNOWN"),
                **row_template,
            }

            timeline.append(episode)
            window_id += 1

        episode_index += 1

    return timeline


# ============================================================
# 3-1. Raw-to-state helper functions
# ============================================================

SIGNAL_SEVERITY = {
    "NOMINAL": 0,
    "FLUCTUATING": 1,
    "DEGRADED": 2,
    "SEVERELY_DEGRADED": 3,
    "LOST": 4,
}

def worst_signal_state(states: List[str]) -> str:
    return max(states, key=lambda x: SIGNAL_SEVERITY[x])


def snr_to_state(value: float, nominal_min: float, fluctuating_min: float, degraded_min: float) -> str:
    if value >= nominal_min:
        return "NOMINAL"
    if value >= fluctuating_min:
        return "FLUCTUATING"
    if value >= degraded_min:
        return "DEGRADED"
    return "SEVERELY_DEGRADED"


def loss_rate_to_decoder_state(frame_loss_rate: float, crc_count: int, decode_fail_count: int) -> str:
    if decode_fail_count >= 2:
        return "DECODE_FAIL"
    if frame_loss_rate >= 0.15:
        return "FRAME_LOSS"
    if crc_count >= 4:
        return "CRC_FAIL"
    if decode_fail_count == 1 or crc_count >= 2 or frame_loss_rate >= 0.07:
        return "DECODE_WARN"
    return "DECODE_OK"


def retry_timeout_to_state(total_retry: int, total_timeout: int) -> str:
    if total_timeout >= 3:
        return "BURST_TIMEOUT"
    if total_timeout >= 2:
        return "TIMEOUT"
    if total_retry >= 4:
        return "RETRY_INCREASED"
    return "NORMAL"


# ============================================================
# 3-2. Benign impairment injection functions
# ============================================================

def apply_ttc_benign_impairment(raw: Dict[str, Any], severity: str) -> None:
    if severity == "NONE":
        return

    if severity == "MILD":
        raw["ttc_snr_db"] = uniform_float(10.5, 14.5, 3)
        raw["ttc_frame_loss_rate"] = uniform_float(0.018, 0.055, 5)
        raw["ttc_crc_error_count"] = uniform_int(0, 2)
        raw["ttc_decode_fail_count"] = weighted_choice([0, 1], [0.88, 0.12])
        raw["ttc_retry_count"] = uniform_int(1, 4)
        raw["ttc_timeout_count"] = weighted_choice([0, 1], [0.85, 0.15])
        raw["ttc_rtt_ms"] = uniform_float(450, 1100, 2)

    elif severity == "MODERATE":
        raw["ttc_snr_db"] = uniform_float(7.0, 11.5, 3)
        raw["ttc_frame_loss_rate"] = uniform_float(0.045, 0.115, 5)
        raw["ttc_crc_error_count"] = uniform_int(1, 4)
        raw["ttc_decode_fail_count"] = weighted_choice([0, 1], [0.70, 0.30])
        raw["ttc_retry_count"] = uniform_int(3, 7)
        raw["ttc_timeout_count"] = weighted_choice([0, 1, 2], [0.60, 0.30, 0.10])
        raw["ttc_rtt_ms"] = uniform_float(900, 1800, 2)


def apply_isl_benign_impairment(raw: Dict[str, Any], severity: str) -> None:
    if severity == "NONE":
        return

    if severity == "MILD":
        raw["isl_snr_db"] = uniform_float(9.5, 13.5, 3)
        raw["isl_frame_loss_rate"] = uniform_float(0.020, 0.060, 5)
        raw["isl_crc_error_count"] = uniform_int(0, 2)
        raw["isl_decode_fail_count"] = weighted_choice([0, 1], [0.90, 0.10])
        raw["isl_retry_count"] = uniform_int(1, 4)
        raw["isl_timeout_count"] = weighted_choice([0, 1], [0.88, 0.12])
        raw["isl_rtt_ms"] = uniform_float(120, 260, 2)

    elif severity == "MODERATE":
        raw["isl_snr_db"] = uniform_float(6.5, 10.5, 3)
        raw["isl_frame_loss_rate"] = uniform_float(0.050, 0.120, 5)
        raw["isl_crc_error_count"] = uniform_int(1, 4)
        raw["isl_decode_fail_count"] = weighted_choice([0, 1], [0.72, 0.28])
        raw["isl_retry_count"] = uniform_int(3, 7)
        raw["isl_timeout_count"] = weighted_choice([0, 1, 2], [0.62, 0.30, 0.08])
        raw["isl_rtt_ms"] = uniform_float(200, 420, 2)


def apply_iot_benign_impairment(raw: Dict[str, Any], severity: str) -> None:
    if severity == "NONE":
        return

    if severity == "MILD":
        raw["iot_snr_db"] = uniform_float(-11.5, -5.5, 3)
        raw["iot_frame_loss_rate"] = uniform_float(0.055, 0.120, 5)
        raw["iot_crc_error_count"] = uniform_int(1, 4)
        raw["iot_decode_fail_count"] = weighted_choice([0, 1], [0.85, 0.15])
        raw["iot_retry_count"] = uniform_int(2, 6)
        raw["iot_timeout_count"] = weighted_choice([0, 1], [0.78, 0.22])
        raw["iot_rtt_ms"] = uniform_float(2500, 6500, 2)

    elif severity == "MODERATE":
        raw["iot_snr_db"] = uniform_float(-17.5, -10.5, 3)
        raw["iot_frame_loss_rate"] = uniform_float(0.100, 0.200, 5)
        raw["iot_crc_error_count"] = uniform_int(2, 6)
        raw["iot_decode_fail_count"] = weighted_choice([0, 1], [0.65, 0.35])
        raw["iot_retry_count"] = uniform_int(5, 10)
        raw["iot_timeout_count"] = weighted_choice([0, 1, 2], [0.52, 0.36, 0.12])
        raw["iot_rtt_ms"] = uniform_float(4500, 9000, 2)


def apply_uav_benign_impairment(raw: Dict[str, Any], severity: str) -> None:
    if severity == "NONE":
        return

    if severity == "MILD":
        raw["uav_sinr_db"] = uniform_float(0.5, 6.0, 3)
        raw["uav_rsrp_dbm"] = uniform_float(-112.0, -100.0, 3)
        raw["uav_frame_loss_rate"] = uniform_float(0.035, 0.090, 5)
        raw["uav_crc_error_count"] = uniform_int(1, 4)
        raw["uav_decode_fail_count"] = weighted_choice([0, 1], [0.88, 0.12])
        raw["uav_retry_count"] = uniform_int(2, 6)
        raw["uav_timeout_count"] = weighted_choice([0, 1], [0.82, 0.18])
        raw["uav_rtt_ms"] = uniform_float(800, 1900, 2)

    elif severity == "MODERATE":
        raw["uav_sinr_db"] = uniform_float(-4.5, 2.5, 3)
        raw["uav_rsrp_dbm"] = uniform_float(-122.0, -108.0, 3)
        raw["uav_frame_loss_rate"] = uniform_float(0.080, 0.180, 5)
        raw["uav_crc_error_count"] = uniform_int(2, 7)
        raw["uav_decode_fail_count"] = weighted_choice([0, 1], [0.68, 0.32])
        raw["uav_retry_count"] = uniform_int(5, 11)
        raw["uav_timeout_count"] = weighted_choice([0, 1, 2], [0.56, 0.34, 0.10])
        raw["uav_rtt_ms"] = uniform_float(1400, 3200, 2)


def apply_gnss_benign_variability(raw: Dict[str, Any], severity: str) -> None:
    """
    Apply benign GNSS geometry/multipath or receiver-monitoring variation.

    This function represents normal receiver/environmental variability, not
    spoofing. It may produce weak C/N0 fluctuation, occasional cycle-slip-like
    tracking, and bounded residual variation, but it should not create frequent
    jump-like, unavailable, or lock-loss states.
    """
    if severity == "NONE":
        return

    if severity == "MILD":
        raw["gnss_cn0_mean_dbhz"] = uniform_float(36.0, 41.0, 3)
        raw["gnss_cn0_std_db"] = uniform_float(1.6, 3.4, 3)
        raw["gnss_tracking_jitter"] = uniform_float(0.10, 0.20, 4)
        raw["gnss_carrier_lock_ratio"] = uniform_float(0.955, 0.990, 4)
        raw["gnss_cycle_slip_count"] = weighted_choice([0, 1, 2], [0.45, 0.45, 0.10])
        raw["gnss_lock_loss_count"] = weighted_choice([0, 1], [0.97, 0.03])
        raw["position_jump_m"] = uniform_float(4.5, 9.5, 3)
        raw["pseudorange_residual_m"] = uniform_float(4.5, 8.5, 3)
        raw["time_residual_ns"] = uniform_float(55.0, 115.0, 3)
        raw["clock_drift_ppm"] = uniform_float(0.015, 0.045, 5)
        raw["fix_valid"] = weighted_choice([1, 0], [0.99, 0.01])

    elif severity == "MODERATE":
        raw["gnss_cn0_mean_dbhz"] = uniform_float(32.0, 38.0, 3)
        raw["gnss_cn0_std_db"] = uniform_float(2.8, 5.2, 3)
        raw["gnss_tracking_jitter"] = uniform_float(0.16, 0.32, 4)
        raw["gnss_carrier_lock_ratio"] = uniform_float(0.900, 0.965, 4)
        raw["gnss_cycle_slip_count"] = weighted_choice([0, 1, 2, 3], [0.20, 0.38, 0.30, 0.12])
        raw["gnss_lock_loss_count"] = weighted_choice([0, 1], [0.90, 0.10])
        raw["position_jump_m"] = uniform_float(6.5, 13.5, 3)
        raw["pseudorange_residual_m"] = uniform_float(6.0, 10.5, 3)
        raw["time_residual_ns"] = uniform_float(75.0, 160.0, 3)
        raw["clock_drift_ppm"] = uniform_float(0.025, 0.050, 5)
        raw["fix_valid"] = weighted_choice([1, 0], [0.97, 0.03])


def apply_gnss_benign_pnt_residual_variation(raw: Dict[str, Any], severity: str) -> None:
    """
    Apply normal PNT/filter residual variation without RF-like signal collapse.

    This is the key benign overlap with GNSS spoofing: the receiver signal and
    tracking remain mostly healthy, while the derived PNT solution shows a
    bounded, transient residual caused by normal geometry, filter settling,
    smoothing, or clock-estimation variation. It must not create attack-like
    jump, unavailable-fix, or lock-loss states.
    """
    if severity == "NONE":
        return

    # Keep signal and tracking healthy. This prevents normal PNT residual
    # windows from becoming RF-jamming-like or receiver-failure-like.
    raw["gnss_cn0_mean_dbhz"] = uniform_float(39.0, 46.5, 3)
    raw["gnss_cn0_std_db"] = uniform_float(0.8, 2.4, 3)
    raw["gnss_tracking_jitter"] = uniform_float(0.04, 0.13, 4)
    raw["gnss_carrier_lock_ratio"] = uniform_float(0.975, 1.000, 4)
    raw["gnss_cycle_slip_count"] = weighted_choice([0, 1], [0.94, 0.06])
    raw["gnss_lock_loss_count"] = 0
    raw["fix_valid"] = 1

    if severity == "MILD":
        raw["position_jump_m"] = uniform_float(3.5, 7.8, 3)
        raw["pseudorange_residual_m"] = uniform_float(3.5, 6.8, 3)
        raw["time_residual_ns"] = uniform_float(45.0, 95.0, 3)
        raw["clock_drift_ppm"] = uniform_float(0.012, 0.040, 5)

    elif severity == "MODERATE":
        # Still bounded: enough to overlap single-window spoofing residuals,
        # but not enough to create jump/offset/unavailable states.
        raw["position_jump_m"] = uniform_float(6.0, 10.8, 3)
        raw["pseudorange_residual_m"] = uniform_float(5.5, 8.8, 3)
        raw["time_residual_ns"] = uniform_float(70.0, 125.0, 3)
        raw["clock_drift_ppm"] = uniform_float(0.020, 0.048, 5)

def apply_decoder_warning(raw: Dict[str, Any], path_type: str, severity: str) -> None:
    if severity == "NONE":
        return

    # A benign decoder warning should create overlap with attack data,
    # but should rarely escalate into frame loss or decode failure.
    if path_type == "GATEWAY_LEO":
        raw["ttc_crc_error_count"] = max(raw["ttc_crc_error_count"], uniform_int(2, 4))
        raw["ttc_frame_loss_rate"] = max(raw["ttc_frame_loss_rate"], uniform_float(0.070, 0.105, 5))
        raw["ttc_decode_fail_count"] = weighted_choice([0, 1], [0.78, 0.22])
    elif path_type == "LEO_LEO":
        raw["isl_crc_error_count"] = max(raw["isl_crc_error_count"], uniform_int(2, 4))
        raw["isl_frame_loss_rate"] = max(raw["isl_frame_loss_rate"], uniform_float(0.070, 0.105, 5))
        raw["isl_decode_fail_count"] = weighted_choice([0, 1], [0.78, 0.22])
    elif path_type == "IOT_LEO":
        raw["iot_crc_error_count"] = max(raw["iot_crc_error_count"], uniform_int(2, 5))
        raw["iot_frame_loss_rate"] = max(raw["iot_frame_loss_rate"], uniform_float(0.075, 0.130, 5))
        raw["iot_decode_fail_count"] = weighted_choice([0, 1], [0.75, 0.25])
    elif path_type == "UAV_LEO":
        raw["uav_crc_error_count"] = max(raw["uav_crc_error_count"], uniform_int(2, 5))
        raw["uav_frame_loss_rate"] = max(raw["uav_frame_loss_rate"], uniform_float(0.070, 0.125, 5))
        raw["uav_decode_fail_count"] = weighted_choice([0, 1], [0.75, 0.25])
    elif path_type == "MULTI_PATH":
        selected_path = weighted_choice(
            ["GATEWAY_LEO", "IOT_LEO", "UAV_LEO", "LEO_LEO"],
            [0.30, 0.25, 0.25, 0.20],
        )
        apply_decoder_warning(raw, selected_path, severity)


COMMAND_OUTCOME_FLAGS = {
    "EXECUTED": "executed_flag", "HELD": "held_flag", "TIMEOUT": "command_timeout_flag",
    "UNKNOWN": "command_unknown_flag", "REJECTED": "rejected_flag",
    "BLOCKED": "blocked_flag", "FAILED": "failed_flag",
}


def set_command_outcome(raw: Dict[str, Any], outcome: str) -> None:
    """Record one representative command's current outcome in this window.

    raw_command_count is the total received count. Outcome flags describe a
    representative command, not counts of all outcomes. No queued command from
    an earlier window is modeled. ACCEPTED means accepted/pending.
    """
    if int(raw["raw_command_count"]) <= 0:
        raise ValueError("Cannot assign a command outcome without command activity.")
    if outcome not in {"ACCEPTED", *COMMAND_OUTCOME_FLAGS}:
        raise ValueError(f"Unknown command outcome: {outcome}")
    for flag in COMMAND_OUTCOME_FLAGS.values():
        raw[flag] = 0
    raw["accepted_flag"] = int(outcome in {"ACCEPTED", "EXECUTED", "HELD", "TIMEOUT"})
    if outcome in COMMAND_OUTCOME_FLAGS:
        raw[COMMAND_OUTCOME_FLAGS[outcome]] = 1


def sample_benign_command_outcome(raw: Dict[str, Any], executed: float, held: float = 0., timeout: float = 0.) -> None:
    """One categorical draw: timeout, then hold, then execution or pending.

    Arguments are conditional probabilities in that order, not masses to sum.
    """
    probabilities = [timeout, (1-timeout)*held,
                     (1-timeout)*(1-held)*executed,
                     (1-timeout)*(1-held)*(1-executed)]
    set_command_outcome(raw, weighted_choice(["TIMEOUT", "HELD", "EXECUTED", "ACCEPTED"], probabilities))


def apply_retry_delay(raw: Dict[str, Any], path_type: str, severity: str) -> None:
    if severity == "NONE":
        return

    retry_low, retry_high = (4, 7) if severity == "MILD" else (6, 10)
    timeout_probs = [0.70, 0.24, 0.06] if severity == "MILD" else [0.50, 0.35, 0.15]
    timeout_value = weighted_choice([0, 1, 2], timeout_probs)

    if path_type == "GATEWAY_LEO":
        raw["ttc_retry_count"] = max(raw["ttc_retry_count"], uniform_int(retry_low, retry_high))
        raw["ttc_timeout_count"] = max(raw["ttc_timeout_count"], timeout_value)
        raw["ttc_rtt_ms"] = max(raw["ttc_rtt_ms"], uniform_float(800, 1800 if severity == "MILD" else 2600, 2))
    elif path_type == "LEO_LEO":
        raw["isl_retry_count"] = max(raw["isl_retry_count"], uniform_int(retry_low, retry_high))
        raw["isl_timeout_count"] = max(raw["isl_timeout_count"], timeout_value)
        raw["isl_rtt_ms"] = max(raw["isl_rtt_ms"], uniform_float(120, 300 if severity == "MILD" else 500, 2))
    elif path_type == "IOT_LEO":
        raw["iot_retry_count"] = max(raw["iot_retry_count"], uniform_int(retry_low + 1, retry_high + 3))
        raw["iot_timeout_count"] = max(raw["iot_timeout_count"], timeout_value)
        raw["iot_rtt_ms"] = max(raw["iot_rtt_ms"], uniform_float(3500, 7500 if severity == "MILD" else 10000, 2))
    elif path_type == "UAV_LEO":
        raw["uav_retry_count"] = max(raw["uav_retry_count"], uniform_int(retry_low, retry_high + 2))
        raw["uav_timeout_count"] = max(raw["uav_timeout_count"], timeout_value)
        raw["uav_rtt_ms"] = max(raw["uav_rtt_ms"], uniform_float(1000, 2500 if severity == "MILD" else 4200, 2))
    elif path_type == "MULTI_PATH":
        for selected_path in weighted_choice(
            [["GATEWAY_LEO"], ["IOT_LEO"], ["UAV_LEO"], ["GATEWAY_LEO", "UAV_LEO"]],
            [0.30, 0.25, 0.25, 0.20],
        ):
            apply_retry_delay(raw, selected_path, severity)

    raw["delayed_transition_flag"] = 1 if RNG.random() < (0.25 if severity == "MILD" else 0.45) else 0

    if raw["raw_command_count"] > 0:
        # A valid command may be accepted but held or delayed due to benign
        # contact/link scheduling. It should not be blocked, rejected, or failed.
        if RNG.random() < (0.25 if severity == "MILD" else 0.40):
            set_command_outcome(raw, "HELD")

        if RNG.random() < (0.05 if severity == "MILD" else 0.12):
            set_command_outcome(raw, "TIMEOUT")


def apply_obc_load_transient(raw: Dict[str, Any], severity: str) -> None:
    if severity == "NONE":
        return

    raw["session_active"] = 1
    raw["delayed_transition_flag"] = 1
    raw["state_changed_flag"] = weighted_choice([0, 1], [0.70, 0.30])

    if severity == "MODERATE":
        raw["session_reset_count"] = weighted_choice([0, 1], [0.85, 0.15])
        raw["incomplete_transition_flag"] = weighted_choice([0, 1], [0.88, 0.12])


def apply_path_specific_benign_impairment(raw: Dict[str, Any], path_type: str, severity: str) -> None:
    if path_type == "GATEWAY_LEO":
        apply_ttc_benign_impairment(raw, severity)
    elif path_type == "LEO_LEO":
        apply_isl_benign_impairment(raw, severity)
    elif path_type == "IOT_LEO":
        apply_iot_benign_impairment(raw, severity)
    elif path_type == "UAV_LEO":
        apply_uav_benign_impairment(raw, severity)
    elif path_type == "GNSS_LEO":
        apply_gnss_benign_variability(raw, severity)
    elif path_type == "MULTI_PATH":
        affected_paths = weighted_choice(
            [
                ["GATEWAY_LEO"],
                ["IOT_LEO"],
                ["UAV_LEO"],
                ["GNSS_LEO"],
                ["LEO_LEO"],
                ["GATEWAY_LEO", "IOT_LEO"],
                ["UAV_LEO", "GNSS_LEO"],
            ],
            [0.18, 0.18, 0.18, 0.16, 0.12, 0.10, 0.08],
        )
        for affected_path in affected_paths:
            apply_path_specific_benign_impairment(raw, affected_path, severity)


def apply_benign_normal_effects(raw: Dict[str, Any], episode: Dict[str, Any]) -> None:
    condition = episode["normal_condition"]
    severity = episode["condition_severity"]
    path_type = episode["path_type"]

    if condition == "NOMINAL" or severity == "NONE":
        return

    command_only = {"SCHEDULED_CONFIG_UPDATE", "VALID_REPEATED_STATUS_REQUEST",
                    "BENIGN_COMMAND_HOLD", "COMMAND_RETRANSMISSION", "AUTH_RETRY",
                    "BENIGN_OUT_OF_ORDER_DELIVERY"}
    if condition in command_only and raw["raw_command_count"] == 0:
        return
    if condition in (BENIGN_LINK_CONDITIONS | {"BENIGN_DECODER_WARNING", "CONTACT_HANDOVER_TRANSITION"}) and path_type == "INTERNAL_ONBOARD":
        return

    raw["benign_impairment_flag"] = 1

    if condition in [
        "BENIGN_LINK_FLUCTUATION",
        "LOW_ELEVATION_CONTACT",
        "ANTENNA_POINTING_VARIATION",
        "BENIGN_RF_RECEIVER_VARIATION",
        "TEMPORARY_SIGNAL_FLUCTUATION",
        "INTERMITTENT_PACKET_LOSS",
    ]:
        target_path = "GATEWAY_LEO" if condition == "LOW_ELEVATION_CONTACT" else path_type
        apply_path_specific_benign_impairment(raw, target_path, severity)
        # A targeted TT&C impairment cannot degrade an unrelated selected link.
        raw["link_degraded_flag"] = int(target_path == path_type and severity == "MODERATE" and RNG.random() < 0.35)

    elif condition in ["GNSS_GEOMETRY_OR_MULTIPATH", "BENIGN_GNSS_MONITORING_VARIATION"]:
        apply_gnss_benign_variability(raw, severity)
        raw["link_degraded_flag"] = int(path_type == "GNSS_LEO" and severity == "MODERATE" and RNG.random() < 0.15)

    elif condition == "BENIGN_PNT_RESIDUAL_VARIATION":
        apply_gnss_benign_pnt_residual_variation(raw, severity)
        # PNT residual variation is not a link-degradation event.
        raw["link_degraded_flag"] = 0

    elif condition == "BENIGN_DECODER_WARNING":
        apply_decoder_warning(raw, path_type, severity)

    elif condition == "CONTACT_HANDOVER_TRANSITION":
        apply_path_specific_benign_impairment(raw, path_type, severity)
        apply_retry_delay(raw, path_type, severity)
        raw["session_reset_count"] = weighted_choice([0, 1], [0.90, 0.10]) if severity == "MODERATE" else 0
        raw["link_degraded_flag"] = 1 if severity == "MODERATE" and RNG.random() < 0.30 else 0

    elif condition == "TEMPORARY_SESSION_DELAY":
        apply_retry_delay(raw, path_type, severity)
        raw["session_active"] = 1
        raw["link_degraded_flag"] = 1 if severity == "MODERATE" and RNG.random() < 0.20 else 0

    elif condition == "SESSION_RESYNC":
        apply_retry_delay(raw, path_type, severity)
        raw["session_active"] = 1
        raw["session_reset_count"] = weighted_choice([0, 1], [0.78, 0.22]) if severity == "MILD" else weighted_choice([0, 1], [0.58, 0.42])
        raw["delayed_transition_flag"] = 1 if RNG.random() < (0.35 if severity == "MILD" else 0.55) else 0
        raw["state_changed_flag"] = weighted_choice([0, 1], [0.55, 0.45])

    elif condition == "OBC_LOAD_TRANSIENT":
        apply_obc_load_transient(raw, severity)

    elif condition in BENIGN_MAINTENANCE_CONDITIONS:
        # Legitimate scheduled maintenance/control-plane housekeeping.
        # These conditions create overlap with ground-control compromise, but
        # remain scheduled, authorized, and non-escalating in NORMAL.
        raw["session_active"] = 1
        raw["state_changed_flag"] = 1
        raw["delayed_transition_flag"] = 1 if RNG.random() < (0.18 if severity == "MILD" else 0.32) else 0
        raw["incomplete_transition_flag"] = 1 if severity == "MODERATE" and RNG.random() < 0.04 else 0

    elif condition == "SCHEDULED_CONFIG_UPDATE":
        # A legitimate scheduled configuration update.
        # It creates semantic overlap with injection/compromise,
        # but remains fresh, authorized, and phase-consistent.
        sample_benign_command_outcome(raw, executed=0.85)
        raw["delayed_transition_flag"] = 1 if severity == "MODERATE" and RNG.random() < 0.20 else 0
        raw["state_changed_flag"] = 1

    elif condition == "VALID_REPEATED_STATUS_REQUEST":
        # Normal status polling can repeat across windows.
        # This should not be treated as replay because freshness and authorization remain valid.
        sample_benign_command_outcome(raw, executed=0.90)
        raw["state_changed_flag"] = 0
        raw["delayed_transition_flag"] = 0

    elif condition == "BENIGN_COMMAND_HOLD":
        # A valid command may be held due to normal scheduling or temporary contact constraints.
        sample_benign_command_outcome(raw, executed=0.,
            held=0.52 if severity == "MILD" else 0.70,
            timeout=0.06 if severity == "MILD" else 0.14)
        raw["delayed_transition_flag"] = 1 if RNG.random() < (0.40 if severity == "MILD" else 0.60) else 0

    elif condition == "COMMAND_RETRANSMISSION":
        # A benign command retransmission remains fresh or out-of-order but
        # authorization and freshness eventually remain acceptable.
        apply_retry_delay(raw, path_type, severity)
        sample_benign_command_outcome(raw, executed=0.82 if severity == "MILD" else 0.72,
            held=0.15 if severity == "MILD" else 0.25,
            timeout=0.05 if severity == "MILD" else 0.10)
        raw["delayed_transition_flag"] = 1 if RNG.random() < (0.20 if severity == "MILD" else 0.38) else 0

    elif condition == "AUTH_RETRY":
        # Auth retry is still a valid operator/session condition in NORMAL.
        # It should resemble authorization friction without becoming failed or
        # unauthorized.
        sample_benign_command_outcome(raw, executed=0.78 if severity == "MILD" else 0.68,
            held=0.16 if severity == "MILD" else 0.28)
        raw["delayed_transition_flag"] = 1 if RNG.random() < (0.22 if severity == "MILD" else 0.35) else 0

    # Security semantics should remain benign for the NORMAL label.
    raw["auth_present"] = 1
    raw["auth_valid"] = 1
    raw["integrity_valid"] = 1
    raw["source_authorized"] = 1
    raw["session_valid"] = 1
    raw["duplicate_flag"] = 0
    raw["accepted_sequence_flag"] = 0
    raw["source_unverified_benign_flag"] = 0
    raw["valid_looking_benign_flag"] = 0
    raw["sequence_gap"] = 1
    raw["rejected_flag"] = 0
    raw["blocked_flag"] = 0
    raw["failed_flag"] = 0

    # Command/session ambiguous-normal flags. These are applied after the
    # benign-security reset above so that NORMAL stays authorized while still
    # reflecting realistic scheduling, contact-window, and status-polling
    # irregularities in a single window.
    if raw["raw_command_count"] > 0:
        if condition == "VALID_REPEATED_STATUS_REQUEST":
            raw["sequence_gap"] = weighted_choice([1, 2], [0.78, 0.22])
            raw["accepted_sequence_flag"] = 1 if RNG.random() < 0.10 else 0
            raw["source_unverified_benign_flag"] = 1 if RNG.random() < 0.05 else 0

        elif condition == "SCHEDULED_CONFIG_UPDATE":
            raw["sequence_gap"] = weighted_choice([1, 2], [0.78, 0.22])
            raw["accepted_sequence_flag"] = 1 if RNG.random() < 0.20 else 0
            raw["source_unverified_benign_flag"] = 1 if RNG.random() < 0.08 else 0
            raw["valid_looking_benign_flag"] = 1 if RNG.random() < 0.05 else 0

        elif condition == "BENIGN_COMMAND_HOLD":
            raw["sequence_gap"] = weighted_choice([1, 2], [0.82, 0.18])
            raw["accepted_sequence_flag"] = 1 if RNG.random() < 0.18 else 0
            raw["source_unverified_benign_flag"] = 1 if RNG.random() < 0.04 else 0

        elif condition == "BENIGN_OUT_OF_ORDER_DELIVERY":
            raw["sequence_gap"] = weighted_choice([2, 1], [0.62, 0.38])
            raw["accepted_sequence_flag"] = 1 if RNG.random() < 0.22 else 0
            raw["source_unverified_benign_flag"] = 1 if RNG.random() < 0.06 else 0
            raw["delayed_transition_flag"] = 1 if RNG.random() < 0.45 else raw["delayed_transition_flag"]

        elif condition == "COMMAND_RETRANSMISSION":
            raw["sequence_gap"] = weighted_choice([1, 2], [0.62, 0.38])
            raw["accepted_sequence_flag"] = 1 if RNG.random() < 0.24 else 0
            raw["source_unverified_benign_flag"] = 1 if RNG.random() < 0.04 else 0

        elif condition == "AUTH_RETRY":
            raw["sequence_gap"] = weighted_choice([1, 2], [0.90, 0.10])
            raw["accepted_sequence_flag"] = 1 if RNG.random() < 0.06 else 0
            raw["source_unverified_benign_flag"] = 1 if RNG.random() < (0.26 if severity == "MILD" else 0.42) else 0
            raw["valid_looking_benign_flag"] = 1 if RNG.random() < 0.08 else 0

        elif condition == "SESSION_RESYNC":
            raw["sequence_gap"] = weighted_choice([1, 2], [0.72, 0.28])
            raw["accepted_sequence_flag"] = 1 if RNG.random() < 0.14 else 0
            raw["source_unverified_benign_flag"] = 1 if RNG.random() < 0.04 else 0

        elif condition in BENIGN_MAINTENANCE_CONDITIONS:
            raw["sequence_gap"] = weighted_choice([1, 2], [0.80, 0.20])
            raw["accepted_sequence_flag"] = 1 if RNG.random() < 0.16 else 0
            raw["source_unverified_benign_flag"] = 1 if RNG.random() < 0.06 else 0
            raw["valid_looking_benign_flag"] = 1 if RNG.random() < 0.06 else 0

        elif condition in ["TEMPORARY_SESSION_DELAY", "CONTACT_HANDOVER_TRANSITION"]:
            raw["sequence_gap"] = weighted_choice([1, 2], [0.86, 0.14])
            raw["accepted_sequence_flag"] = 1 if RNG.random() < 0.08 else 0


# ============================================================
# 4. Generate one normal raw observation window
# ============================================================

def _generate_nominal_window(
    window_id: int,
    timestamp: datetime,
    episode: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate raw numeric/flag observations for one normal LEO-centered window."""

    operational_phase = episode["operational_phase"]
    episode_type = episode["episode_type"]

    # Command activity follows the already-sampled operational context, not
    # the audit-only episode_type.  This prevents COMMAND_SESSION episode labels
    # from becoming a hidden context/sequence shortcut.
    command_count = 0
    if is_command_context(episode):
        command_count = count_choice([1, 2, 3], [0.65, 0.25, 0.10])

    telemetry_count = count_choice([5, 10, 15, 20], [0.20, 0.40, 0.30, 0.10])
    iot_report_count = count_choice([0, 5, 10, 20, 30], [0.15, 0.25, 0.30, 0.20, 0.10])
    uav_message_count = count_choice([0, 5, 10, 15], [0.30, 0.30, 0.25, 0.15])
    gnss_observation_count = count_choice([10, 20, 30, 40, 50], [0.10, 0.25, 0.35, 0.20, 0.10])

    raw = {
        "scenario_id": episode["scenario_id"],
        "episode_id": episode["episode_id"],
        "episode_type": episode["episode_type"],
        "episode_step": episode["episode_step"],
        "episode_length": episode["episode_length"],
        "raw_episode_length_bucket": episode.get("episode_length_bucket", "UNKNOWN"),
        "elapsed_seconds": episode.get("elapsed_seconds", episode["episode_step"] * DEFAULT_WINDOW_SECONDS),
        "window_seconds": DEFAULT_WINDOW_SECONDS,

        "window_id": window_id,
        "timestamp": timestamp.isoformat(timespec="seconds"),
        "label": LABEL_NORMAL,

        # phase / traffic counts
        "raw_operational_phase": operational_phase,
        "raw_normal_condition": episode["normal_condition"],
        "raw_condition_phase": episode["condition_phase"],
        "raw_condition_severity": episode["condition_severity"],
        "raw_benign_event_start": episode.get("benign_event_start", -1),
        "raw_benign_event_duration": episode.get("benign_event_duration", 0),
        "raw_benign_event_end": episode.get("benign_event_end", -1),
        "raw_benign_event_active": episode.get("benign_event_active", 0),
        "raw_benign_event_local_step": episode.get("benign_event_local_step", -1),
        "raw_episode_command_mode": episode.get("episode_command_mode", "MIXED_NOMINAL_COMMAND"),
        "raw_context_profile": episode.get("shared_context_profile", episode.get("normal_context_profile", "UNKNOWN")),
        "raw_context_family": episode.get("shared_context_family", "UNKNOWN"),
        "benign_impairment_flag": 0,
        "raw_command_count": command_count,
        "raw_telemetry_count": telemetry_count,
        "raw_iot_report_count": iot_report_count,
        "raw_uav_message_count": uav_message_count,
        "raw_gnss_observation_count": gnss_observation_count,

        # TT&C link raw observations
        "ttc_snr_db": clipped_normal("ttc_snr_db", 3),
        "ttc_frame_loss_rate": clipped_normal("ttc_frame_loss_rate", 5),
        "ttc_crc_error_count": count_choice([0, 1], [0.98, 0.02]),
        "ttc_decode_fail_count": 0,
        "ttc_parse_error_count": 0,
        "ttc_retry_count": count_choice([0, 1], [0.85, 0.15]),
        "ttc_timeout_count": 0,
        "ttc_rtt_ms": clipped_normal("ttc_rtt_ms", 2),

        # ISL raw observations
        "isl_snr_db": clipped_normal("isl_snr_db", 3),
        "isl_frame_loss_rate": clipped_normal("isl_frame_loss_rate", 5),
        "isl_crc_error_count": count_choice([0, 1], [0.98, 0.02]),
        "isl_decode_fail_count": 0,
        "isl_retry_count": count_choice([0, 1], [0.88, 0.12]),
        "isl_timeout_count": 0,
        "isl_rtt_ms": clipped_normal("isl_rtt_ms", 2),

        # IoT / MIOTY raw observations
        "iot_snr_db": clipped_normal("iot_snr_db", 3),
        "iot_frame_loss_rate": clipped_normal("iot_frame_loss_rate", 5),
        "iot_crc_error_count": count_choice([0, 1, 2], [0.93, 0.06, 0.01]),
        "iot_decode_fail_count": count_choice([0, 1], [0.985, 0.015]),
        "iot_retry_count": count_choice([0, 1, 2], [0.70, 0.25, 0.05]),
        "iot_timeout_count": count_choice([0, 1], [0.95, 0.05]),
        "iot_rtt_ms": clipped_normal("iot_rtt_ms", 2),

        # UAV / 3GPP NTN raw observations
        "uav_rsrp_dbm": clipped_normal("uav_rsrp_dbm", 3),
        "uav_sinr_db": clipped_normal("uav_sinr_db", 3),
        "uav_frame_loss_rate": clipped_normal("uav_frame_loss_rate", 5),
        "uav_crc_error_count": count_choice([0, 1, 2], [0.94, 0.05, 0.01]),
        "uav_decode_fail_count": count_choice([0, 1], [0.99, 0.01]),
        "uav_retry_count": count_choice([0, 1, 2], [0.75, 0.20, 0.05]),
        "uav_timeout_count": count_choice([0, 1], [0.96, 0.04]),
        "uav_rtt_ms": clipped_normal("uav_rtt_ms", 2),

        # GNSS receiver raw observations
        "gnss_cn0_mean_dbhz": clipped_normal("gnss_cn0_mean_dbhz", 3),
        "gnss_cn0_std_db": clipped_normal("gnss_cn0_std_db", 3),
        "gnss_lock_loss_count": 0,
        "gnss_cycle_slip_count": count_choice([0, 1], [0.93, 0.07]),
        "gnss_tracking_jitter": clipped_normal("gnss_tracking_jitter", 4),
        "gnss_carrier_lock_ratio": clipped_normal("gnss_carrier_lock_ratio", 4),
        "position_jump_m": clipped_normal("position_jump_m", 3),
        "velocity_jump_mps": clipped_normal("velocity_jump_mps", 4),
        "pseudorange_residual_m": clipped_normal("pseudorange_residual_m", 3),
        "time_residual_ns": clipped_normal("time_residual_ns", 3),
        "clock_drift_ppm": clipped_normal("clock_drift_ppm", 5),
        "fix_valid": 1,

        # command / control / session raw observations
        "sequence_gap": 1,
        "duplicate_flag": 0,
        "accepted_sequence_flag": 0,
        "source_unverified_benign_flag": 0,
        "valid_looking_benign_flag": 0,
        "auth_present": 1,
        "auth_valid": 1,
        "integrity_valid": 1,
        "source_authorized": 1,
        "session_valid": 1,
        "accepted_flag": 1 if command_count > 0 else 0,
        "executed_flag": flag_choice(0.80) if command_count > 0 else 0,
        "held_flag": 0,
        "command_timeout_flag": 0,
        "command_unknown_flag": 0,
        "rejected_flag": 0,
        "blocked_flag": 0,
        "failed_flag": 0,
        "parameter_range_violation_flag": 0,
        "phase_policy_block_flag": 0,
        "raw_operator_event": "NONE",

        "session_active": 1 if operational_phase in ["CONTACT_WINDOW", "COMMAND_SESSION", "DATA_RELAY"] else flag_choice(0.35),
        "session_age_s": clipped_normal("session_age_s", 2),
        "session_reset_count": 0,
        "session_expired_flag": 0,
        "link_degraded_flag": 0,
        "link_loss_flag": 0,

        # Extrapolated from THIS one-second window, not a rolling history.
        "command_rate_per_min": float(command_count * 60),
        "critical_command_count": count_choice([0, 1], [0.92, 0.08]) if command_count > 0 else 0,
        "parameter_deviation_ratio": clipped_normal("parameter_deviation_ratio", 4),

        "transition_allowed": 1,
        "unexpected_transition_flag": 0,
        "forbidden_transition_flag": 0,
        "delayed_transition_flag": 0,
        "incomplete_transition_flag": 0,
        "state_changed_flag": flag_choice(0.20),

        "raw_observation_stage": episode["observation_stage"],
        "raw_path_type": episode["path_type"],
        "raw_direction": episode["direction"],
        "raw_source_role": episode["source_role"],
        "raw_destination_role": episode["destination_role"],
        "raw_protocol_context": episode["protocol_context"],

        "raw_message_type": episode["message_type"],
        "raw_service_context": episode["service_context"],
        "raw_command_intent": episode["command_intent"],
    }

    apply_benign_normal_effects(raw, episode)
    apply_operator_error_outcome(raw)
    # Message properties are sampled ONCE and saved with the numeric evidence.
    semantics = sample_normal_message_semantics(raw)
    raw.update({"raw_" + field: value for field, value in semantics.items()})
    if semantics["command_intent"] in STATE_CHANGING_INTENTS:
        # A legitimate state-changing operation changes state only when it is executed.
        raw["state_changed_flag"] = int(raw.get("executed_flag", 0) == 1)
    return raw


def apply_operator_error_outcome(raw: Dict[str, Any]) -> None:
    """Rare benign operator errors on an otherwise authorized command.

    A representative command is REJECTED when the operator sent an out-of-range
    parameter, or BLOCKED when the phase policy does not permit it now.  Both
    remain authenticated, authorized, fresh and session-valid, change no state,
    and are independent of the benign condition of the episode.  Windows whose
    condition already produced a HELD or TIMEOUT representative command are left
    unchanged.
    """
    if int(raw.get("raw_command_count", 0)) <= 0:
        return
    if raw.get("held_flag", 0) == 1 or raw.get("command_timeout_flag", 0) == 1:
        return
    draw = RNG.random()
    if draw < OPERATOR_PARAMETER_ERROR_PROB:
        set_command_outcome(raw, "REJECTED")
        raw["parameter_range_violation_flag"] = 1
        raw["raw_operator_event"] = "OPERATOR_PARAMETER_ERROR"
    elif draw < OPERATOR_PARAMETER_ERROR_PROB + PHASE_POLICY_BLOCK_PROB:
        set_command_outcome(raw, "BLOCKED")
        raw["phase_policy_block_flag"] = 1
        raw["raw_operator_event"] = "PHASE_POLICY_BLOCK"
    else:
        return
    raw["state_changed_flag"] = 0
    raw["delayed_transition_flag"] = 0
    raw["incomplete_transition_flag"] = 0


# ============================================================
# 5. Normalize raw window into the 22-field observable record
# ============================================================

def normalize_context(raw: Dict[str, Any]) -> Dict[str, str]:
    """Return the shared, label-agnostic context exactly as sampled.

    Earlier versions re-derived context from episode_type and benign-condition
    phase.  That made context fields encode generator-specific trajectories.
    In v4, context is sampled before evidence generation and is not overwritten
    by NORMAL condition logic.
    """
    profile = {
        "observation_stage": raw["raw_observation_stage"],
        "path_type": raw["raw_path_type"],
        "direction": raw["raw_direction"],
        "source_role": raw["raw_source_role"],
        "destination_role": raw["raw_destination_role"],
        "protocol_context": raw["raw_protocol_context"],
        "operational_phase": raw["raw_operational_phase"],
    }
    validate_context_profile(profile)
    return profile


def has_command_activity(raw: Dict[str, Any]) -> bool:
    return int(raw.get("raw_command_count", 0)) > 0

def normalize_sequence_freshness(raw: Dict[str, Any]) -> str:
    if not has_command_activity(raw):
        return "NOT_APPLICABLE"

    if raw.get("unknown_sequence_flag", 0) == 1:
        return "UNKNOWN"

    # Common deterministic mapping. The generator, not this extractor,
    # determines which sequence events exist in a scenario.
    if raw.get("accepted_sequence_flag", 0) == 1:
        return "ACCEPTED"

    if raw["duplicate_flag"] == 1:
        return "REPLAYED"

    if raw["sequence_gap"] == 1:
        return "FRESH"

    if raw["sequence_gap"] == 0:
        return "DUPLICATE"

    if raw["sequence_gap"] < 0:
        return "STALE"

    if raw["sequence_gap"] > 1:
        return "OUT_OF_ORDER"

    return "UNKNOWN"


def normalize_control_security(raw: Dict[str, Any]) -> str:
    if not has_command_activity(raw):
        return "NOT_APPLICABLE"

    if raw["auth_present"] == 0:
        return "AUTH_MISSING"

    if raw.get("auth_unverified_flag", 0) == 1:
        return "AUTH_UNVERIFIED"

    if raw["auth_valid"] == 0:
        return "AUTH_FAILED"

    if raw.get("integrity_unverified_flag", 0) == 1:
        return "INTEGRITY_UNVERIFIED"

    if raw["integrity_valid"] == 0:
        return "INTEGRITY_FAILED"

    if raw.get("source_unverified_flag", 0) == 1:
        return "SOURCE_UNVERIFIED"

    if raw["source_authorized"] == 0:
        return "UNAUTHORIZED"

    # Benign overlap states: the command remains valid, but a transient
    # scheduling/source-verification condition makes the window look less clean.
    if raw.get("source_unverified_benign_flag", 0) == 1:
        return "SOURCE_UNVERIFIED"

    if (raw["session_valid"] == 0 or raw.get("valid_looking_benign_flag", 0) == 1
            or raw.get("policy_inconsistent_flag", 0) == 1
            or (raw.get("session_required_flag", 0) == 1 and raw["session_active"] == 0)):
        return "VALID_LOOKING_ANOMALOUS"

    return "AUTH_VALID"

def sample_normal_message_semantics(raw: Dict[str, Any]) -> Dict[str, str]:
    """Generate legitimate categorical message properties once.

    Event metadata can choose generated properties. Extractors cannot consult
    that metadata or resample it. Payload diagnostics use measured residuals
    and decoder outcomes, not an episode's class/phase tag.
    """
    message = raw["raw_message_type"]
    service = raw["raw_service_context"]
    intent = "NO_COMMAND"
    payload = "NOMINAL_PARAMETER"
    if has_command_activity(raw):
        mode = raw["raw_episode_command_mode"]
        fixed_intents = {
            "SCHEDULED_CONFIG_UPDATE": "CONFIG_UPDATE",
            "VALID_REPEATED_STATUS_REQUEST": "STATUS_REQUEST",
            "STATUS_POLLING": "STATUS_REQUEST", "TIME_SYNC_SEQUENCE": "TIME_SYNC",
            "NOMINAL_CONTROL_SEQUENCE": "NOMINAL_CONTROL",
            "ROUTINE_HOUSEKEEPING": "HOUSEKEEPING", "ORBIT_MAINTENANCE": "ORBIT_CONTROL",
            "AUTH_RETRY": "SESSION_REAUTH",
            **LEGITIMATE_OPERATION_MODES,
        }
        intent = fixed_intents.get(mode)
        if intent is None:
            intent = weighted_choice(
                ["STATUS_REQUEST", "TIME_SYNC", "CONFIG_UPDATE", "NOMINAL_CONTROL", "HOUSEKEEPING", "ORBIT_CONTROL",
                 "MODE_CHANGE", "PAYLOAD_CONFIG", "ATTITUDE_ADJUST", "ROUTE_UPDATE"],
                [0.30, 0.19, 0.12, 0.14, 0.07, 0.05, 0.04, 0.04, 0.025, 0.025])
        message, service, payload = sample_command_envelope(intent, RNG)
        if raw.get("parameter_range_violation_flag", 0) == 1:
            # Operator parameter error: the rejected command carried an out-of-range value.
            payload = "PARAMETER_RANGE_VIOLATION"
    elif raw["raw_path_type"] == "GNSS_LEO":
        # Residuals are nonnegative magnitudes, in the units in their names.
        if raw["time_residual_ns"] > 90 or raw["clock_drift_ppm"] > 0.045:
            payload = "TIME_RESIDUAL_VARIATION"
        elif raw["position_jump_m"] > 8:
            payload = "POSITION_RESIDUAL_VARIATION"
        elif raw["pseudorange_residual_m"] > 7:
            payload = "NAV_RESIDUAL_VARIATION"
    elif raw["raw_path_type"] in {"GATEWAY_LEO", "LEO_LEO", "IOT_LEO", "UAV_LEO"}:
        decoder = normalize_decoder_status(raw)
        if decoder != "DECODE_OK":
            message = "PARTIAL_FRAME"
            payload = "PARTIAL_PAYLOAD"
        if decoder != "DECODE_OK" or normalize_signal_quality(raw) in {"DEGRADED", "SEVERELY_DEGRADED", "LOST"}:
            service = "DEGRADED_SERVICE"
    return {"message_type": message, "service_context": service,
            "command_intent": intent, "payload_or_parameter_context": payload}


def normalize_message_control(raw: Dict[str, Any]) -> Dict[str, str]:
    """Deterministic projection of stored properties and control evidence."""
    result = {field: raw["raw_" + field] for field in (
        "message_type", "service_context", "command_intent", "payload_or_parameter_context")}
    result["sequence_freshness_status"] = normalize_sequence_freshness(raw)
    result["control_security_state"] = normalize_control_security(raw)
    return result


def normalize_signal_quality(raw: Dict[str, Any]) -> str:
    ttc_state = snr_to_state(
        raw["ttc_snr_db"],
        nominal_min=15.0,
        fluctuating_min=10.5,
        degraded_min=3.0,
    )

    isl_state = snr_to_state(
        raw["isl_snr_db"],
        nominal_min=14.0,
        fluctuating_min=9.5,
        degraded_min=3.0,
    )

    iot_state = snr_to_state(
        raw["iot_snr_db"],
        nominal_min=-5.0,
        fluctuating_min=-12.0,
        degraded_min=-18.0,
    )

    uav_state = snr_to_state(
        raw["uav_sinr_db"],
        nominal_min=6.0,
        fluctuating_min=0.0,
        degraded_min=-5.0,
    )

    gnss_state = snr_to_state(
        raw["gnss_cn0_mean_dbhz"],
        nominal_min=39.0,
        fluctuating_min=35.0,
        degraded_min=28.0,
    )

    path_type = raw["raw_path_type"]

    # Path-specific evidence selection.
    # This avoids assigning IoT/GNSS fluctuations to a Gateway-LEO command window.
    if path_type == "GATEWAY_LEO":
        return ttc_state

    if path_type == "LEO_LEO":
        return isl_state

    if path_type == "IOT_LEO":
        return iot_state

    if path_type == "UAV_LEO":
        return uav_state

    if path_type == "GNSS_LEO":
        return gnss_state

    if path_type == "INTERNAL_ONBOARD":
        return "NOMINAL"

    if path_type == "MULTI_PATH":
        return worst_signal_state([ttc_state, isl_state, iot_state, uav_state, gnss_state])
    raise ValueError(f"Unsupported raw_path_type: {path_type}")


def normalize_tracking_stability(raw: Dict[str, Any]) -> str:
    if raw["gnss_lock_loss_count"] > 0:
        return "LOCK_LOSS"

    # Avoid interpretation-heavy learning tokens such as SUSPICIOUSLY_STABLE.
    # Receiver smoothing or short-term low variance is represented as STABLE;
    # only observable tracking disturbances become CYCLE_SLIP_LIKE or UNSTABLE.
    if raw["gnss_cycle_slip_count"] > 0:
        return "CYCLE_SLIP_LIKE"

    if raw["gnss_tracking_jitter"] > 0.15 or raw["gnss_carrier_lock_ratio"] < 0.96:
        return "UNSTABLE"

    return "STABLE"

def normalize_navigation_consistency(raw: Dict[str, Any]) -> str:
    if raw["fix_valid"] == 0:
        return "UNAVAILABLE"

    if raw["position_jump_m"] > 50.0 or raw["pseudorange_residual_m"] > 30.0:
        return "JUMP_DETECTED"

    if raw["position_jump_m"] > 15.0 or raw["velocity_jump_mps"] > 1.0:
        return "INCONSISTENT"

    if raw["position_jump_m"] > 8.0 or raw["pseudorange_residual_m"] > 7.0:
        return "DRIFTING"

    return "CONSISTENT"


def normalize_timing_consistency(raw: Dict[str, Any]) -> str:
    if raw["time_residual_ns"] > 5000.0:
        return "OFFSET_JUMP"

    if raw["time_residual_ns"] > 500.0 or raw["clock_drift_ppm"] > 0.5:
        return "INCONSISTENT"

    if raw["time_residual_ns"] > 90.0 or raw["clock_drift_ppm"] > 0.045:
        return "DRIFTING"

    return "CONSISTENT"


def normalize_decoder_status(raw: Dict[str, Any]) -> str:
    ttc_decoder_state = loss_rate_to_decoder_state(
        raw["ttc_frame_loss_rate"],
        raw["ttc_crc_error_count"],
        raw["ttc_decode_fail_count"],
    )

    isl_decoder_state = loss_rate_to_decoder_state(
        raw["isl_frame_loss_rate"],
        raw["isl_crc_error_count"],
        raw["isl_decode_fail_count"],
    )

    iot_decoder_state = loss_rate_to_decoder_state(
        raw["iot_frame_loss_rate"],
        raw["iot_crc_error_count"],
        raw["iot_decode_fail_count"],
    )

    uav_decoder_state = loss_rate_to_decoder_state(
        raw["uav_frame_loss_rate"],
        raw["uav_crc_error_count"],
        raw["uav_decode_fail_count"],
    )

    path_type = raw["raw_path_type"]

    # Path-specific decoder evidence selection.
    if path_type == "GATEWAY_LEO":
        return ttc_decoder_state

    if path_type == "LEO_LEO":
        return isl_decoder_state

    if path_type == "IOT_LEO":
        return iot_decoder_state

    if path_type == "UAV_LEO":
        return uav_decoder_state

    # GNSS receiver and internal onboard traces are not packet-decoder dominated
    # in this controlled normal dataset.
    if path_type in ["GNSS_LEO", "INTERNAL_ONBOARD"]:
        return "DECODE_OK"

    # MULTI_PATH or unknown path: use conservative aggregate state.
    link_states = [
        ttc_decoder_state,
        isl_decoder_state,
        iot_decoder_state,
        uav_decoder_state,
    ]

    priority = {
        "DECODE_OK": 0,
        "DECODE_WARN": 1,
        "CRC_FAIL": 2,
        "FRAME_LOSS": 3,
        "DECODE_FAIL": 4,
    }

    return max(link_states, key=lambda x: priority[x])


def normalize_command_execution(raw: Dict[str, Any]) -> str:
    if not has_command_activity(raw):
        return "NO_COMMAND"

    if raw.get("command_unknown_flag", 0) == 1:
        return "UNKNOWN"

    if raw.get("command_timeout_flag", 0) == 1:
        return "TIMEOUT"

    if raw.get("held_flag", 0) == 1:
        return "HELD"

    if raw.get("failed_flag", 0) == 1:
        return "FAILED"

    if raw.get("blocked_flag", 0) == 1:
        return "BLOCKED"

    if raw.get("rejected_flag", 0) == 1:
        return "REJECTED"

    if raw.get("executed_flag", 0) == 1:
        return "EXECUTED"

    if raw.get("accepted_flag", 0) == 1:
        return "ACCEPTED"
    return "UNKNOWN"


def normalize_retry_timeout(raw: Dict[str, Any]) -> str:
    path_type = raw["raw_path_type"]

    if path_type == "GATEWAY_LEO":
        return retry_timeout_to_state(
            total_retry=raw["ttc_retry_count"],
            total_timeout=raw["ttc_timeout_count"],
        )

    if path_type == "LEO_LEO":
        return retry_timeout_to_state(
            total_retry=raw["isl_retry_count"],
            total_timeout=raw["isl_timeout_count"],
        )

    if path_type == "IOT_LEO":
        return retry_timeout_to_state(
            total_retry=raw["iot_retry_count"],
            total_timeout=raw["iot_timeout_count"],
        )

    if path_type == "UAV_LEO":
        return retry_timeout_to_state(
            total_retry=raw["uav_retry_count"],
            total_timeout=raw["uav_timeout_count"],
        )

    if path_type in ["GNSS_LEO", "INTERNAL_ONBOARD"]:
        return "NORMAL"

    # MULTI_PATH or unknown path: aggregate retries and timeouts.
    total_retry = (
        raw["ttc_retry_count"]
        + raw["isl_retry_count"]
        + raw["iot_retry_count"]
        + raw["uav_retry_count"]
    )

    total_timeout = (
        raw["ttc_timeout_count"]
        + raw["isl_timeout_count"]
        + raw["iot_timeout_count"]
        + raw["uav_timeout_count"]
    )

    return retry_timeout_to_state(total_retry, total_timeout)


def normalize_session_state(raw: Dict[str, Any]) -> str:
    if raw.get("link_loss_flag", 0) == 1:
        return "LINK_LOSS"

    if raw.get("session_active", 0) == 0:
        return "NO_SESSION"

    if raw.get("session_valid", 1) == 0:
        return "ABNORMAL"

    if raw.get("session_expired_flag", 0) == 1:
        return "EXPIRED"

    if raw.get("session_reset_count", 0) > 0:
        return "RESET"

    if raw.get("link_degraded_flag", 0) == 1:
        return "DEGRADED"

    return "ACTIVE"


def normalize_state_transition(raw: Dict[str, Any]) -> str:
    if raw.get("forbidden_transition_flag", 0) == 1:
        return "FORBIDDEN_TRANSITION"

    if raw.get("unexpected_transition_flag", 0) == 1:
        return "UNEXPECTED_TRANSITION"

    if raw.get("incomplete_transition_flag", 0) == 1:
        return "INCOMPLETE_TRANSITION"

    if raw.get("delayed_transition_flag", 0) == 1:
        return "DELAYED_TRANSITION"

    if raw.get("transition_allowed", 1) == 0:
        return "INCONSISTENT"

    if raw.get("state_changed_flag", 0) == 0:
        return "NO_CHANGE"

    return "CONSISTENT"


def normalize_raw_to_observable(raw: Dict[str, Any]) -> Dict[str, str]:
    observable = {}

    observable.update(normalize_context(raw))
    observable.update(normalize_message_control(raw))

    observable["signal_quality_state"] = normalize_signal_quality(raw)
    observable["tracking_stability"] = normalize_tracking_stability(raw)
    observable["navigation_consistency"] = normalize_navigation_consistency(raw)
    observable["timing_consistency"] = normalize_timing_consistency(raw)

    observable["decoder_status"] = normalize_decoder_status(raw)
    observable["command_execution_result"] = normalize_command_execution(raw)
    observable["retry_timeout_state"] = normalize_retry_timeout(raw)
    observable["session_state"] = normalize_session_state(raw)
    observable["state_transition_consistency"] = normalize_state_transition(raw)

    missing = [field for field in OBSERVABLE_FIELDS if field not in observable]
    if missing:
        raise ValueError(f"Missing observable fields: {missing}")

    return observable


# ============================================================
# 6. Build normal dataset
# ============================================================


OBSERVABLE_DOMAINS = {'observation_stage': {'SESSION_MANAGER', 'ONBOARD_MONITOR', 'COMMAND_HANDLER', 'RF_RECEIVER', 'PROTOCOL_HANDLER', 'GNSS_RECEIVER'}, 'path_type': {'INTERNAL_ONBOARD', 'UAV_LEO', 'LEO_LEO', 'IOT_LEO', 'GATEWAY_LEO', 'GNSS_LEO'}, 'direction': {'GROUND_TO_SPACE', 'USER_TO_SPACE', 'SPACE_TO_SPACE', 'INTERNAL', 'SPACE_TO_GROUND', 'GNSS_TO_SPACE'}, 'source_role': {'MONITORED_LEO', 'GNSS_CONSTELLATION', 'UAV_TERMINAL', 'PEER_LEO', 'GROUND_CONTROL_SERVICE', 'IOT_TERMINAL', 'GROUND_GATEWAY'}, 'destination_role': {'OBC', 'MONITORED_LEO', 'GROUND_GATEWAY'}, 'protocol_context': {'CCSDS_ISL', '3GPP_NTN', 'CCSDS_TTC', 'ONBOARD_INTERNAL', 'MIOTY', 'GNSS_NAV'}, 'operational_phase': {'CONTACT_WINDOW', 'ROUTINE_MONITORING', 'DATA_RELAY', 'COMMAND_SESSION', 'PAYLOAD_OPERATION', 'GNSS_NAVIGATION'}, 'message_type': {'GNSS_TIMING_UPDATE', 'IOT_REPORT', 'GNSS_OBSERVATION', 'NAVIGATION_STATE_UPDATE', 'PARTIAL_FRAME', 'NOMINAL_MIXED_TRAFFIC', 'NAVIGATION_MESSAGE', 'TELEMETRY', 'PNT_SOLUTION', 'TELECOMMAND', 'UAV_TELEMETRY', 'COMMAND_UPLINK'}, 'service_context': {'IOT_UPLINK', 'PAYLOAD_DATA', 'HOUSEKEEPING', 'NAVIGATION_FILTERING', 'PNT_MONITORING', 'UAV_NTN_SERVICE', 'CONTROL_SESSION', 'ISL_ROUTING', 'DEGRADED_SERVICE', 'TTC_CONTROL', 'TIMING_SYNCHRONIZATION', 'GNSS_PNT'}, 'command_intent': {'NOMINAL_CONTROL', 'SESSION_REAUTH', 'STATUS_REQUEST', 'MODE_CHANGE', 'ROUTE_UPDATE', 'HOUSEKEEPING', 'NO_COMMAND', 'ATTITUDE_ADJUST', 'TIME_SYNC', 'PAYLOAD_CONFIG', 'CONFIG_UPDATE', 'ORBIT_CONTROL'}, 'payload_or_parameter_context': {'PHASE_TRANSITION_PARAMETER', 'CONTEXT_MISMATCH_PARAMETER', 'POSITION_RESIDUAL_VARIATION', 'SUSPICIOUS_PARAMETER_PATTERN', 'POLICY_SYNC_CONTEXT', 'ROUTINE_PARAMETER', 'TIME_SYNC_PARAMETER', 'NAV_RESIDUAL_VARIATION', 'TIME_RESIDUAL_VARIATION', 'PARTIAL_PAYLOAD', 'SCHEDULED_CONFIG_PARAMETER', 'STATUS_PARAMETER', 'CONFIG_PARAMETER', 'PARAMETER_RANGE_VIOLATION', 'MISSION_PHASE_MISMATCH_PARAMETER', 'OUT_OF_POLICY_PARAMETER', 'CONTROL_PARAMETER', 'NOMINAL_PARAMETER'}, 'sequence_freshness_status': {'DUPLICATE', 'STALE', 'FRESH', 'OUT_OF_ORDER', 'REPLAYED', 'ACCEPTED', 'UNKNOWN', 'NOT_APPLICABLE'}, 'control_security_state': {'AUTH_VALID', 'AUTH_MISSING', 'INTEGRITY_FAILED', 'VALID_LOOKING_ANOMALOUS', 'AUTH_UNVERIFIED', 'NOT_APPLICABLE', 'UNAUTHORIZED', 'INTEGRITY_UNVERIFIED', 'SOURCE_UNVERIFIED', 'AUTH_FAILED'}, 'signal_quality_state': {'LOST', 'DEGRADED', 'SEVERELY_DEGRADED', 'NOMINAL', 'FLUCTUATING'}, 'tracking_stability': {'STABLE', 'LOCK_LOSS', 'CYCLE_SLIP_LIKE', 'UNSTABLE'}, 'navigation_consistency': {'CONSISTENT', 'UNAVAILABLE', 'JUMP_DETECTED', 'INCONSISTENT', 'DRIFTING'}, 'timing_consistency': {'CONSISTENT', 'OFFSET_JUMP', 'INCONSISTENT', 'DRIFTING'}, 'decoder_status': {'FRAME_LOSS', 'DECODE_FAIL', 'CRC_FAIL', 'DECODE_WARN', 'DECODE_OK'}, 'command_execution_result': {'REJECTED', 'NO_COMMAND', 'ACCEPTED', 'HELD', 'TIMEOUT', 'EXECUTED', 'BLOCKED'}, 'retry_timeout_state': {'RETRY_INCREASED', 'NORMAL', 'TIMEOUT', 'BURST_TIMEOUT'}, 'session_state': {'DEGRADED', 'ACTIVE', 'EXPIRED', 'LINK_LOSS', 'RESET', 'NO_SESSION', 'ABNORMAL'}, 'state_transition_consistency': {'INCOMPLETE_TRANSITION', 'NO_CHANGE', 'CONSISTENT', 'UNEXPECTED_TRANSITION', 'FORBIDDEN_TRANSITION', 'DELAYED_TRANSITION'}}

# ---------- Shared runtime: no class-specific nominal branch ----------
STATE_CHANGING_INTENTS = {'CONFIG_UPDATE','NOMINAL_CONTROL','ORBIT_CONTROL','TIME_SYNC',
                         'PAYLOAD_CONFIG','ATTITUDE_ADJUST','ROUTE_UPDATE','MODE_CHANGE'}
CANONICAL_COLUMNS = METADATA_FIELDS + OBSERVABLE_FIELDS


def validate_generation_request(num_rows, window_seconds, start_time, seed, background_seed, attacks_enabled):
    for name, value in [('num_rows',num_rows),('seed',seed),('background_seed',background_seed)]:
        if value is None and name == 'background_seed': continue
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value < (1 if name=='num_rows' else 0):
            raise ValueError(f'{name} must be a {"positive" if name=="num_rows" else "non-negative"} integer.')
    if isinstance(window_seconds,(bool,np.bool_)) or window_seconds != 1:
        raise ValueError('The observable contract requires one-second windows.')
    if start_time is not None and (not isinstance(start_time,datetime) or start_time.microsecond):
        raise ValueError('start_time must be a datetime on a whole-second boundary.')
    if not isinstance(attacks_enabled,(bool,np.bool_)):
        raise ValueError('attacks_enabled must be Boolean.')


class NominalSubstrate:
    """Independent timeline and measurement streams, shared by all six classes.

    Explicit background_seed deliberately yields paired counterfactuals. Without
    it the maker number selects a disjoint RNG stream, not a different law.
    Attack RNG consumption cannot change any nominal draw.
    """
    def __init__(self, num_rows, seed, maker_number, background_seed=None):
        entropy = [int(seed), 0x4E4F4D, int(maker_number)] if background_seed is None else [int(background_seed),0x50414952]
        self.stream_id = ':'.join(map(str,entropy))
        timeline_seed, raw_seed = np.random.SeedSequence(entropy).spawn(2)
        token = _RANDOM_SOURCE.set(np.random.default_rng(timeline_seed))
        try: self.timeline = _build_nominal_timeline(num_rows)
        finally: _RANDOM_SOURCE.reset(token)
        self.raw_rng = np.random.default_rng(raw_seed)
        self.next_window = 0

    def window(self, window_id, timestamp):
        if window_id != self.next_window:
            raise ValueError('Nominal windows must be consumed once, in consecutive order.')
        token = _RANDOM_SOURCE.set(self.raw_rng)
        try:
            raw = _generate_nominal_window(window_id,timestamp,self.timeline[window_id])
        finally: _RANDOM_SOURCE.reset(token)
        self.next_window += 1
        # The same primitive vocabulary exists in NORMAL and all attack makers.
        for name in ('unknown_sequence_flag','auth_unverified_flag','integrity_unverified_flag',
                     'source_unverified_flag','raw_policy_violation_flag'):
            raw.setdefault(name,0)
        for prefix in ('ttc','isl','iot','uav'):
            raw.setdefault(prefix+'_sync_loss_count',0)
        raw['session_required_flag'] = int(raw['raw_command_count']>0)
        raw['policy_inconsistent_flag'] = int(raw['parameter_range_violation_flag'] or raw['phase_policy_block_flag'])
        raw['raw_policy_violation_flag'] = raw['policy_inconsistent_flag']
        raw['raw_command_permitted_in_context'] = 1-raw['policy_inconsistent_flag']
        raw['raw_nominal_command_count'] = raw['raw_command_count']
        # This scalar is a synthetic representative parameter; its relationship
        # to deviation is shared, including benign rejected operator errors.
        raw['raw_command_parameter_value'] = (1.05 if raw['parameter_range_violation_flag'] else
                                              round(.5+raw['parameter_deviation_ratio']/2,6))
        raw['parameter_deviation_ratio'] = round(2*abs(raw['raw_command_parameter_value']-.5),6)
        raw['raw_nominal_condition_type'] = self.timeline[window_id]['episode_type']
        raw['raw_shared_context_profile'] = raw['raw_context_profile']
        raw['raw_shared_context_family'] = raw['raw_context_family']
        raw['raw_background_stream'] = self.stream_id
        raw['raw_window_seconds'] = 1
        raw['raw_gnss_monitor_available'] = 1
        return raw


def copy_nominal_window(nominal_raw, episode, label):
    """Copy already generated primitives; do not resample or infer from a label."""
    if nominal_raw is None:
        raise ValueError('Use build_*_dataset: raw-window overlays require a shared nominal_raw window.')
    raw = dict(nominal_raw)
    for f in ('scenario_id','episode_id','episode_type','episode_step'):
        raw[f] = episode[f]
    raw['label'] = label
    raw['raw_label_scope'] = 'EPISODE_SCENARIO_WITH_CONTEXT_WINDOWS'
    raw['raw_episode_length'] = raw['raw_episode_length_seconds'] = episode['episode_length']
    raw['raw_episode_length_type'] = episode.get('episode_length_type',episode.get('episode_length_bucket'))
    raw['raw_elapsed_seconds'] = episode['episode_step']
    raw['raw_episode_progress'] = episode['episode_step']/max(episode['episode_length']-1,1)
    for f in CONTEXT_FIELD_NAMES:
        if raw['raw_'+f] != episode[f]:
            raise ValueError(f'Attack timeline changed the shared context: {f}')
    for f,v in episode.items():
        if f.startswith('phase_boundary_'): raw['raw_'+f] = v
    for f,v in episode.get('phase_boundaries',{}).items():raw['raw_phase_boundary_'+f] = v
    return raw


def begin_command_attack(raw):
    """A new attack command replaces the representative nominal record.

    Keep physical/session transport observations. Do not inherit a benign
    operator error, accepted-sequence exception or verification flag from a
    different command. The attack mechanism then fills its actual checks.
    """
    for key in ('duplicate_flag','accepted_sequence_flag','unknown_sequence_flag',
                'auth_unverified_flag','integrity_unverified_flag','source_unverified_flag',
                'source_unverified_benign_flag','valid_looking_benign_flag','policy_inconsistent_flag',
                'parameter_range_violation_flag','phase_policy_block_flag','session_expired_flag','raw_policy_violation_flag'):
        raw[key]=0
    raw.update(auth_present=1,auth_valid=1,integrity_valid=1,source_authorized=1,
               session_valid=1,session_active=1,session_required_flag=1,sequence_gap=1,
               raw_command_permitted_in_context=1)


def refresh_environment_semantics(raw):
    """Deterministic consequences of changed receiver evidence, with no new draw.

    These are the same rules used by the nominal message sampler. A command's
    contents are not replaced by a physical-layer diagnostic.
    """
    if has_command_activity(raw):return
    if raw['raw_path_type']=='GNSS_LEO':
        if raw['time_residual_ns']>90 or raw['clock_drift_ppm']>.045:
            raw['raw_payload_or_parameter_context']='TIME_RESIDUAL_VARIATION'
        elif raw['position_jump_m']>8:
            raw['raw_payload_or_parameter_context']='POSITION_RESIDUAL_VARIATION'
        elif raw['pseudorange_residual_m']>7:
            raw['raw_payload_or_parameter_context']='NAV_RESIDUAL_VARIATION'
    elif raw['raw_path_type'] in {'GATEWAY_LEO','LEO_LEO','IOT_LEO','UAV_LEO'}:
        decoder=normalize_decoder_status(raw)
        if decoder!='DECODE_OK':
            raw['raw_message_type']='PARTIAL_FRAME';raw['raw_payload_or_parameter_context']='PARTIAL_PAYLOAD'
        if decoder!='DECODE_OK' or normalize_signal_quality(raw) in {'DEGRADED','SEVERELY_DEGRADED','LOST'}:
            raw['raw_service_context']='DEGRADED_SERVICE'


def build_dataset(maker,num_rows,window_seconds,start_time,seed,return_raw,background_seed,attacks_enabled):
    validate_generation_request(num_rows,window_seconds,start_time,seed,background_seed,attacks_enabled)
    if not isinstance(return_raw,(bool,np.bool_)):raise ValueError('return_raw must be Boolean.')
    start_time = datetime(2026,1,1) if start_time is None else start_time
    number=maker['MAKER_NUMBER']; label=maker['CLASS_LABEL']
    substrate=NominalSubstrate(num_rows,seed,number,background_seed)
    if number==1: timeline=substrate.timeline
    else:
        # Scheduling and per-window intervention draws are independent too.
        schedule_seed, effect_seed=np.random.SeedSequence([int(seed),0x415454,number]).spawn(2)
        maker['RNG']=np.random.default_rng(schedule_seed)
        timeline=maker[maker['TIMELINE_BUILDER']](num_rows,nominal_timeline=substrate.timeline)
        maker['RNG']=np.random.default_rng(effect_seed)
    rows=[];state=None;last_episode=None
    for i,episode in enumerate(timeline):
        timestamp=start_time+timedelta(seconds=i)
        base=substrate.window(i,timestamp)
        if number in (5,6) and episode['episode_id']!=last_episode:
            state=maker['new_replay_state']() if number==5 else maker['new_compromise_state'](episode)
            last_episode=episode['episode_id']
        if number==1:raw=copy_nominal_window(base,episode,label)
        else:
            options={'nominal_raw':base,'attacks_enabled':bool(attacks_enabled)}
            if number==5:options['replay_state']=state
            if number==6:options['compromise_state']=state
            raw=maker[maker['RAW_GENERATOR']](i,timestamp,episode,**options)
        # Audit baselines include every original primitive, including benign
        # effects. They are never appended to the 22-field learning record.
        raw.update({'baseline_'+k:v for k,v in base.items() if k not in METADATA_FIELDS})
        baseline_obs=normalize_raw_to_observable(base)
        observable=normalize_raw_to_observable(raw)
        raw.update({'baseline_observable_'+k:v for k,v in baseline_obs.items()})
        changes=[f for f in OBSERVABLE_FIELDS if observable[f]!=baseline_obs[f]]
        raw['raw_observable_difference_count']=len(changes)
        raw['raw_observable_difference_fields']='|'.join(changes) or 'NONE'
        raw['raw_observable_effect_present']=int(bool(changes))
        raw['raw_attacks_enabled']=int(bool(attacks_enabled) and number!=1)
        raw['raw_paired_audit_background']=int(background_seed is not None)
        raw.setdefault('raw_persistent_effect_present',0)
        raw.setdefault('raw_attack_active',0)
        raw.update(observable); rows.append(raw)
    full=pd.DataFrame(rows)
    return full if return_raw else full[CANONICAL_COLUMNS].copy()


def validate_learning_dataset(df,label,expected_rows=None,expected_window_seconds=1):
    if list(df.columns)!=CANONICAL_COLUMNS:raise ValueError('Expected exactly seven metadata + 22 observable columns.')
    if len(df)==0 or expected_rows is not None and len(df)!=expected_rows:raise ValueError('Unexpected row count.')
    if expected_window_seconds!=1:raise ValueError('One-second windows are required.')
    if df.isna().any().any():raise ValueError('Missing learning values.')
    if not df.label.eq(label).all():raise ValueError('Episode labels must be retained, including quiet windows.')
    for f,allowed in OBSERVABLE_DOMAINS.items():
        if not df[f].isin(allowed).all():raise ValueError(f'Unsupported observable category in {f}.')
    valid={tuple(entry['profile'][f] for f in CONTEXT_FIELD_NAMES) for entry in GLOBAL_CONTEXT_PROFILE_POOL}
    if any(t not in valid for t in df[CONTEXT_FIELD_NAMES].itertuples(index=False,name=None)):
        raise ValueError('Invalid shared context profile.')
    if not np.array_equal(df.window_id.to_numpy(),np.arange(len(df))):raise ValueError('Nonconsecutive windows.')
    times=pd.to_datetime(df.timestamp,utc=True,errors='raise')
    if times.isna().any() or not times.diff().dropna().dt.total_seconds().eq(1).all():raise ValueError('Nonconsecutive timestamps.')
    eid=df.episode_id.to_numpy();starts=np.r_[True,eid[1:]!=eid[:-1]]
    if len(set(eid[starts]))!=int(starts.sum()):raise ValueError('Noncontiguous episode.')
    steps=np.arange(len(df))-np.maximum.accumulate(np.where(starts,np.arange(len(df)),0))
    if not np.array_equal(df.episode_step.to_numpy(),steps):raise ValueError('Invalid episode steps.')
    if df.groupby('episode_id')[['episode_type','scenario_id']].nunique().gt(1).any().any():
        raise ValueError('Episode metadata changed within an episode.')


def validate_raw_dataset(df,maker,expected_rows=None,expected_window_seconds=1):
    validate_learning_dataset(df[CANONICAL_COLUMNS],maker['CLASS_LABEL'],expected_rows,expected_window_seconds)
    if df.isna().any().any():
        raise ValueError('Missing raw audit values: '+str(df.columns[df.isna().any()].tolist()))
    actions=['accepted_flag',*COMMAND_OUTCOME_FLAGS.values()]
    if df.loc[df.raw_command_count.eq(0),actions].ne(0).any().any():raise ValueError('Command outcome without a command.')
    if df[list(COMMAND_OUTCOME_FLAGS.values())].sum(axis=1).gt(1).any():raise ValueError('Conflicting current command outcomes.')
    if not np.allclose(df.command_rate_per_min,df.raw_command_count*60):raise ValueError('Invalid one-second command rate.')
    if not np.allclose(df.parameter_deviation_ratio,2*(df.raw_command_parameter_value-.5).abs(),atol=2e-6):
        raise ValueError('Parameter deviation contradicts the represented value.')
    if not df.raw_policy_violation_flag.eq(df.policy_inconsistent_flag).all():
        raise ValueError('Current policy audit flag contradicts its primitive.')
    for field in df.select_dtypes(include='number'):
        if not np.isfinite(df[field]).all():raise ValueError(f'Nonfinite raw value: {field}')
        if field.endswith('_flag') and not df[field].isin([0,1]).all():raise ValueError(f'Nonbinary flag: {field}')
    for field in ('ttc_frame_loss_rate','isl_frame_loss_rate','iot_frame_loss_rate','uav_frame_loss_rate','gnss_carrier_lock_ratio'):
        if not df[field].between(0,1).all():raise ValueError(f'Invalid probability: {field}')
    columns=list(df.columns)
    for values in df.itertuples(index=False,name=None):
        record=dict(zip(columns,values))
        actual=normalize_raw_to_observable(record)
        base={k[len('baseline_'):]:v for k,v in record.items() if k.startswith('baseline_') and not k.startswith('baseline_observable_')}
        counterfactual=normalize_raw_to_observable(base)
        differences=0
        for f in OBSERVABLE_FIELDS:
            if actual[f]!=record[f]:raise ValueError(f'Raw projection mismatch at window {record["window_id"]}: {f}')
            if counterfactual[f]!=record['baseline_observable_'+f]:raise ValueError('Invalid nominal counterfactual projection.')
            differences+=actual[f]!=counterfactual[f]
        if differences!=record['raw_observable_difference_count']:raise ValueError('Invalid difference audit.')
        if not record['raw_attacks_enabled'] and differences:raise ValueError('Attack-disabled window differs from common nominal substrate.')
    if maker['MAKER_NUMBER']!=1:maker['validate_attack_rows'](df)


def validate_observed_command_gate(df,observed_column):
    observed=df[observed_column].gt(0)
    if not observed.any():return
    subset=df.loc[observed]
    bad=(subset.raw_enforcement_mode.eq('FAIL_CLOSED') & subset.raw_checks_passed.eq(0) & subset.executed_flag.eq(1))
    if bad.any():raise ValueError('Fail-closed command executed after a failed check.')
    if (subset.state_changed_flag.eq(1)&subset.executed_flag.eq(0)).any():raise ValueError('Attack command changed state without execution.')


def require_nominal_when(df,mask):
    if df.loc[mask,'raw_observable_effect_present'].ne(0).any():
        raise ValueError('Unexplained observable change in a nominal-only window.')


def save_csv_atomic(df,path):
    import os,tempfile
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8-sig',newline='') as stream:
            df.to_csv(stream,index=False);stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def run_cli(maker,builder_name):
    import argparse,sys
    parser=argparse.ArgumentParser(description='Shared nominal substrate with scenario-specific causal interventions.')
    parser.add_argument('--rows',type=int,default=maker['DEFAULT_NUM_ROWS'])
    parser.add_argument('--seed',type=int,default=maker['DEFAULT_RANDOM_SEED'])
    parser.add_argument('--output-dir',type=Path,default=Path(maker['DEFAULT_OUTPUT_PATH']).parent)
    args=parser.parse_args()
    print(f'{maker["CLASS_LABEL"]}: {CODE_VERSION}; {args.rows:,} one-second windows; seed={args.seed}',flush=True)
    full=maker[builder_name](num_rows=args.rows,seed=args.seed,return_raw=True)
    validate_raw_dataset(full,maker,args.rows,1)
    output=args.output_dir/Path(maker['DEFAULT_OUTPUT_PATH']).name
    audit=args.output_dir/Path(maker['DEFAULT_RAW_AUDIT_OUTPUT_PATH']).name
    sample=args.output_dir/Path(maker.get('DEFAULT_SAMPLE_TIMELINE_OUTPUT_PATH',output.stem+'_sample_episode_timeline.csv')).name
    save_csv_atomic(full[CANONICAL_COLUMNS],output);save_csv_atomic(full,audit)
    ids=full.episode_id.drop_duplicates().iloc[:3]
    save_csv_atomic(full.loc[full.episode_id.isin(ids)],sample)
    print(f'Saved: {output.resolve()}\nSaved: {audit.resolve()}',flush=True)
    # Preserve review archives produced during execution. These do not replace
    # the individual source files and contain the shared dependencies too.
    now=datetime.now(timezone(timedelta(hours=9)))
    archive_path=args.output_dir/f'{now:%Y-%m-%d_%H-%M-%S_%f}_{maker["MAKER_NUMBER"]:02d}_REVIEW.zip'
    summary={'code_version':CODE_VERSION,'label':maker['CLASS_LABEL'],'rows':len(full),'seed':args.seed,
             'episodes':int(full.episode_id.nunique()),'window_seconds':1,
             'model_input_columns':OBSERVABLE_FIELDS,'all_rows_raw_reprojected':True,
             'common_nominal_engine':True,'formal_detector_evaluation':False,
             'observable_effect_windows':int(full.raw_observable_effect_present.sum()),
             'limitations':['Controlled synthetic distributions; no field calibration.',
                'Attack-disabled equality is a structural check, not a detector performance result.',
                'No-current-attempt and no-observable-effect are different concepts.',
                'Explicit background_seed creates paired audit data; do not independently split its replicas.']}
    with zipfile.ZipFile(archive_path,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in (output,audit,sample):archive.write(p,p.name)
        for p in (Path(maker['__file__']).resolve(),Path(__file__).resolve(),Path(shared_context_module.__file__).resolve()):
            archive.write(p,'source/'+p.name)
        archive.writestr('run_summary.json',json.dumps(summary,ensure_ascii=False,indent=2))
    print('PASS: schema, raw projection, nominal counterfactual and causal checks.',flush=True)
    print(f'UPLOAD_THIS={archive_path.resolve()}',flush=True)


def sample_command_envelope(intent,rng):
    """Common legitimate envelope/payload law, conditional on command intent."""
    def choose(values,p):return str(rng.choice(values,p=p))
    message=choose(['TELECOMMAND','COMMAND_UPLINK'],[.82,.18])
    service=choose(['TTC_CONTROL','CONTROL_SESSION'],[.76,.24])
    options={'STATUS_REQUEST':'STATUS_PARAMETER','TIME_SYNC':'TIME_SYNC_PARAMETER',
             'CONFIG_UPDATE':'SCHEDULED_CONFIG_PARAMETER','HOUSEKEEPING':'ROUTINE_PARAMETER',
             'ORBIT_CONTROL':'PHASE_TRANSITION_PARAMETER','SESSION_REAUTH':'POLICY_SYNC_CONTEXT'}
    payload='NOMINAL_PARAMETER'
    if intent in options:payload=choose(['NOMINAL_PARAMETER',options[intent]],[.8,.2])
    elif intent in {'PAYLOAD_CONFIG','ROUTE_UPDATE','MODE_CHANGE'}:
        payload=choose(['CONFIG_PARAMETER','CONTROL_PARAMETER','NOMINAL_PARAMETER','ROUTINE_PARAMETER'],[.45,.26,.20,.09])
    elif intent=='ATTITUDE_ADJUST':payload=choose(['NOMINAL_PARAMETER','CONFIG_PARAMETER','ROUTINE_PARAMETER'],[.50,.32,.18])
    return message,service,payload


def record_nominal_checks(raw):
    """Audit the already simulated benign transaction without re-handling it.

    Benign positive sequence gaps are admissible reordering/skips in the common
    substrate. Attack replay checks separately compare a captured old record to
    receiver history. No nominal operation acquires a stricter class-specific gate.
    """
    active=has_command_activity(raw)
    results={'authentication':'VALID','integrity_check':'VALID','authorization':'AUTHORIZED',
             'policy_check':'DENIED' if raw['policy_inconsistent_flag'] else 'ALLOWED',
             'session_check':'VALID' if raw['session_valid'] and raw['session_active'] else 'INVALID'}
    for key,value in results.items():raw['raw_'+key+'_result']=value if active else 'NOT_APPLICABLE'
    raw['raw_checks_passed']=int(active and not raw['policy_inconsistent_flag'] and raw['session_valid'] and raw['session_active'])
    raw['raw_security_gate_bypassed']=0
    raw['raw_handler_dispatched']=int(active and raw['accepted_flag'])
    outcome=normalize_command_execution(raw)
    raw['raw_command_handler_action']={'EXECUTED':'EXECUTE','HELD':'HOLD','ACCEPTED':'ACCEPT',
        'REJECTED':'REJECT','BLOCKED':'BLOCK','TIMEOUT':'TIMEOUT'}.get(outcome,'NO_ACTION')
