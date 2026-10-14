# Revision: common nominal background + shared observation mapping.
# Research objective: reviewer revision on synthetic-rule dependence and temporal evidence.
"""Controlled synthetic ground TT&C control-path compromise, 2026-10-02.

One second selects one shared observation context; only the 22 observable fields
are model inputs. Metadata, attack schedules and state histories are audit only.
This is downstream evidence consistent with misuse of a trusted ground path,
not an observation of a ground intrusion, real CCSDS packets or cryptography.
All commands are newly created with valid credentials and integrity. Session
binding and operational policy checks remain independent of authentication.

The local guard checks parameter bounds, command phase, batch/rate and approval.
A separate reference mission envelope exposes an intentionally incomplete local
policy: a locally permitted command can still change the configuration unsafely.
A scalar configuration is a synthetic surrogate, not spacecraft dynamics. Only
observed, executed TT&C state commands update it. No RF/GNSS damage is modeled.
Unobserved scheduled opportunities do not fabricate commands, receiver history
or state changes. The simulated stream is the selected observation stream, not
a complete wire trace. Batch items are distinct new commands with homogeneous
parameters and one representative outcome; no cross-window queue is modeled.

Episode labels include pre, quiet, unobserved and containment windows. Stopping
the attacker never silently restores state. Context samples and episode lengths
are shared with makers 1-5. Legacy HARD/EASY case names are audit labels, not
measured difficulty. No desired token proportions or classifier scores are
enforced. Historical A/A-prime results do not apply to this revised generator.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pathlib import Path

from typing import Any, Dict, List

import csv, hashlib, io, json, os, platform, tempfile, zipfile

import numpy as np

import pandas as pd

import shared_context_v4 as shared_context_module

from shared_context_v4 import (CONTEXT_FIELD_NAMES, GLOBAL_CONTEXT_PROFILE_POOL,
    is_command_context, nominal_semantics_for_context, sample_shared_context_profile,
    validate_context_profile, validate_global_context_pool)

CODE_VERSION='ground-compromise-causal-state-v1-20261002'

LABEL_SCOPE='EPISODE_SCENARIO'

TARGET_PATH='GATEWAY_LEO'

TARGET_PROTOCOL='CCSDS_TTC'

TARGET_DIRECTION='GROUND_TO_SPACE'

ENFORCEMENT_MODES=['FAIL_CLOSED','AUDIT_ONLY']

ENFORCEMENT_MODE_PROBS=[.80,.20]

PHASE_BOUNDARY_KEYS=['normal_end','establishment_end','drift_end','abnormal_end','critical_end','inconsistency_end']

ACTIVE_PHASES={'VALID_LOOKING_COMMAND_DRIFT','ABNORMAL_COMMAND_PATTERN','CRITICAL_CONTROL_MISUSE','ONBOARD_STATE_INCONSISTENCY'}

DEFAULT_SAMPLE_TIMELINE_OUTPUT_PATH='dataset/ground_segment_ttc_control_path_compromise_sample_episode_timeline.csv'

DEFAULT_NUM_ROWS = 50000

DEFAULT_WINDOW_SECONDS = 1

DEFAULT_OUTPUT_PATH = "dataset/ground_segment_ttc_control_path_compromise_observable_dataset.csv"

DEFAULT_RAW_AUDIT_OUTPUT_PATH = "dataset/ground_segment_ttc_control_path_compromise_observable_dataset_with_raw.csv"

DEFAULT_RANDOM_SEED = 342

LABEL_GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE = "GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE"

METADATA_FIELDS = [
    "scenario_id",
    "episode_id",
    "episode_type",
    "episode_step",
    "window_id",
    "timestamp",
    "label",
]

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

COMMON_EPISODE_LENGTH_BUCKETS = ["SHORT", "MEDIUM", "LONG"]

COMMON_EPISODE_LENGTH_PROBS = [0.35, 0.45, 0.20]

COMMON_EPISODE_LENGTH_RANGES = {
    "SHORT": (20, 40),
    "MEDIUM": (40, 80),
    "LONG": (80, 140),
}

TTC_COMPROMISE_EPISODE_TYPES = [
    "COMPROMISED_OPERATOR_SESSION",
    "VALID_AUTH_ABNORMAL_COMMAND",
    "COMMAND_RATE_ABUSE",
    "CROSS_PHASE_CONTROL_MISUSE",
    "CRITICAL_CONTROL_PATH_ABUSE",
    "GATEWAY_SESSION_HIJACK",
    "POLICY_VALID_BUT_MISSION_INCONSISTENT_COMMAND",
]

TTC_COMPROMISE_EPISODE_TYPE_PROBS = [
    0.25,
    0.20,
    0.15,
    0.15,
    0.10,
    0.10,
    0.05,
]

TTC_COMPROMISE_PHASES = [
    "NORMAL_AUTHORIZED_CONTACT",
    "COMPROMISED_SESSION_ESTABLISHMENT",
    "VALID_LOOKING_COMMAND_DRIFT",
    "ABNORMAL_COMMAND_PATTERN",
    "CRITICAL_CONTROL_MISUSE",
    "ONBOARD_STATE_INCONSISTENCY",
    "CONTAINMENT_OR_OPERATOR_INTERVENTION",
]

TTC_COMPROMISE_CASE_TYPES = [
    "HARD_TRUSTED_FRESH_COMMAND_DRIFT",
    "HARD_AUTH_VALID_RATE_INCREASE",
    "HARD_CROSS_PHASE_VALID_CONTROL",
    "HARD_ACCEPTED_EXECUTED_ABNORMAL_TRANSITION",
    "BENIGN_OVERLAP_SCHEDULED_CONFIG_UPDATE",
    "BENIGN_OVERLAP_COMMAND_HOLD",
    "EASY_VALID_LOOKING_ANOMALOUS_SESSION",
    "EASY_CRITICAL_CONTROL_MISUSE",
]

TTC_COMPROMISE_CASE_TYPE_PROBS = [
    # Retained legacy case priors. Names do not establish measured difficulty;
    # observable evidence is computed independently from the command records.
    0.26,
    0.18,
    0.17,
    0.13,
    0.12,
    0.10,
    0.03,
    0.01,
]

BENIGN_OVERLAP_COMPROMISE_CASE_TYPES = {
    "BENIGN_OVERLAP_SCHEDULED_CONFIG_UPDATE",
    "BENIGN_OVERLAP_COMMAND_HOLD",
}

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

SIGNAL_SEVERITY = {
    "NOMINAL": 0,
    "FLUCTUATING": 1,
    "DEGRADED": 2,
    "SEVERELY_DEGRADED": 3,
    "LOST": 4,
}

STATE_CHANGING_INTENTS = {"CONFIG_UPDATE", "NOMINAL_CONTROL", "ORBIT_CONTROL",
    "TIME_SYNC", "PAYLOAD_CONFIG", "ATTITUDE_ADJUST", "ROUTE_UPDATE", "MODE_CHANGE"}

COMMAND_OUTCOME_FLAGS = {
    "EXECUTED": "executed_flag", "HELD": "held_flag", "TIMEOUT": "command_timeout_flag",
    "UNKNOWN": "command_unknown_flag", "REJECTED": "rejected_flag",
    "BLOCKED": "blocked_flag", "FAILED": "failed_flag",
}

OBSERVABLE_DOMAINS = {
    **{field: {entry["profile"][field] for entry in GLOBAL_CONTEXT_PROFILE_POOL}
       for field in CONTEXT_FIELD_NAMES},
    "message_type": {"TELECOMMAND", "COMMAND_UPLINK", "GNSS_OBSERVATION", "PNT_SOLUTION",
                     "NAVIGATION_MESSAGE", "GNSS_TIMING_UPDATE", "NAVIGATION_STATE_UPDATE",
                     "IOT_REPORT", "UAV_TELEMETRY", "TELEMETRY", "NOMINAL_MIXED_TRAFFIC", "PARTIAL_FRAME"},
    "service_context": {"TTC_CONTROL", "CONTROL_SESSION", "GNSS_PNT", "PNT_MONITORING",
                        "TIMING_SYNCHRONIZATION", "NAVIGATION_FILTERING", "IOT_UPLINK",
                        "UAV_NTN_SERVICE", "ISL_ROUTING", "PAYLOAD_DATA", "HOUSEKEEPING", "DEGRADED_SERVICE"},
    "command_intent": {"NO_COMMAND", "STATUS_REQUEST", "TIME_SYNC", "CONFIG_UPDATE",
                       "NOMINAL_CONTROL", "HOUSEKEEPING", "ORBIT_CONTROL", "SESSION_REAUTH"},
    "payload_or_parameter_context": {"NOMINAL_PARAMETER", "STATUS_PARAMETER", "TIME_SYNC_PARAMETER",
        "SCHEDULED_CONFIG_PARAMETER", "ROUTINE_PARAMETER", "PHASE_TRANSITION_PARAMETER",
        "POLICY_SYNC_CONTEXT", "TIME_RESIDUAL_VARIATION", "POSITION_RESIDUAL_VARIATION",
        "NAV_RESIDUAL_VARIATION", "PARTIAL_PAYLOAD"},
    "sequence_freshness_status": {"NOT_APPLICABLE", "ACCEPTED", "FRESH", "OUT_OF_ORDER"},
    "control_security_state": {"NOT_APPLICABLE", "AUTH_VALID", "SOURCE_UNVERIFIED", "VALID_LOOKING_ANOMALOUS"},
    "signal_quality_state": {"NOMINAL", "FLUCTUATING", "DEGRADED", "SEVERELY_DEGRADED", "LOST"},
    "tracking_stability": {"STABLE", "CYCLE_SLIP_LIKE", "UNSTABLE", "LOCK_LOSS"},
    "navigation_consistency": {"CONSISTENT", "DRIFTING", "INCONSISTENT", "JUMP_DETECTED", "UNAVAILABLE"},
    "timing_consistency": {"CONSISTENT", "DRIFTING", "INCONSISTENT", "OFFSET_JUMP"},
    "decoder_status": {"DECODE_OK", "DECODE_WARN", "CRC_FAIL", "FRAME_LOSS", "DECODE_FAIL"},
    "command_execution_result": {"NO_COMMAND", "ACCEPTED", "EXECUTED", "HELD", "TIMEOUT"},
    "retry_timeout_state": {"NORMAL", "RETRY_INCREASED", "TIMEOUT", "BURST_TIMEOUT"},
    "session_state": {"NO_SESSION", "ACTIVE", "DEGRADED", "RESET"},
    "state_transition_consistency": {"NO_CHANGE", "CONSISTENT", "DELAYED_TRANSITION", "INCOMPLETE_TRANSITION"},
}

RNG=np.random.default_rng(DEFAULT_RANDOM_SEED)

_CANONICAL_LEARNING_READY_COLUMNS=METADATA_FIELDS+OBSERVABLE_FIELDS

def uniform_float(low: float, high: float, decimals: int = 4) -> float:
    return round(float(RNG.uniform(low, high)), decimals)

def uniform_int(low: int, high: int) -> int:
    return int(RNG.integers(low, high + 1))

def clipped_normal(name: str, decimals: int = 4) -> float:
    spec = NORMAL_RAW_RANGES[name]
    value = RNG.normal(spec["mean"], spec["std"])
    value = float(np.clip(value, spec["min"], spec["max"]))
    return round(value, decimals)

def weighted_choice(items: List[Any], probs: List[float]) -> Any:
    """Sample one item using the shared RNG.

    The manual cumulative sampler keeps generation deterministic and is faster
    than repeatedly calling np.random.choice for every observable token.
    """
    if len(items) != len(probs):
        raise ValueError("items and probs must have the same length.")

    probs_array = np.array(probs, dtype=float)
    probs_sum = float(probs_array.sum())
    if probs_sum <= 0:
        raise ValueError("Probability sum must be positive.")

    threshold = float(RNG.random()) * probs_sum
    cumulative = 0.0
    for idx, prob in enumerate(probs_array):
        cumulative += float(prob)
        if threshold <= cumulative:
            return items[idx]
    return items[-1]

def count_choice(values: List[int], probs: List[float]) -> int:
    return int(weighted_choice(values, probs))

def flag_choice(prob_one: float) -> int:
    return int(RNG.random() < prob_one)

def reset_rng(seed: int = DEFAULT_RANDOM_SEED) -> None:
    """Reset the global random number generator for reproducible dataset generation."""
    global RNG
    RNG = np.random.default_rng(seed=seed)

def validate_generation_args(num_rows: int, window_seconds: int) -> None:
    if isinstance(num_rows, (bool, np.bool_)) or not isinstance(num_rows, (int, np.integer)) or num_rows <= 0:
        raise ValueError("num_rows must be a positive integer.")
    if isinstance(window_seconds, (bool, np.bool_)) or window_seconds != 1:
        raise ValueError("Only one-second windows are supported by this maker.")

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

def has_command_activity(raw: Dict[str, Any]) -> bool:
    return int(raw.get("raw_command_count", 0)) > 0

def record_command_check_results(raw: Dict[str, Any]) -> None:
    """Store independent check outcomes, never infer authentication from authorization."""
    if not has_command_activity(raw):
        for name in ('authentication','integrity_check','authorization','policy_check','session_check'):
            raw['raw_'+name+'_result'] = 'NOT_APPLICABLE'
        raw['raw_checks_passed'] = 0
        return
    raw['raw_authentication_result'] = ('MISSING' if not raw['auth_present'] else
        'UNVERIFIED' if raw['auth_unverified_flag'] else 'VALID' if raw['auth_valid'] else 'FAILED')
    raw['raw_integrity_check_result'] = ('UNVERIFIED' if raw['integrity_unverified_flag'] else
        'VALID' if raw['integrity_valid'] else 'FAILED')
    raw['raw_authorization_result'] = ('UNVERIFIED' if raw['source_unverified_flag'] else
        'AUTHORIZED' if raw['source_authorized'] else 'UNAUTHORIZED')
    raw['raw_policy_check_result'] = 'DENIED' if raw['policy_inconsistent_flag'] else 'ALLOWED'
    raw['raw_session_check_result'] = ('NOT_REQUIRED' if not raw['session_required_flag'] else
        'MISSING' if not raw['session_active'] else 'VALID' if raw['session_valid'] else 'INVALID')
    raw['raw_checks_passed'] = int(command_checks_pass(raw))

def sample_common_episode_length() -> tuple[str, int]:
    """Sample a shared episode length bucket and duration.

    The same bucket/range design is used by the other context-balanced makers
    so that sequence length itself does not become a class proxy.
    """
    bucket = weighted_choice(COMMON_EPISODE_LENGTH_BUCKETS, COMMON_EPISODE_LENGTH_PROBS)
    low, high = COMMON_EPISODE_LENGTH_RANGES[bucket]
    return bucket, int(RNG.integers(low, high + 1))

def build_jittered_ttc_compromise_phase_boundaries(
    duration: int,
    compromise_type: str,
    case_type: str,
) -> Dict[str, float]:
    """Return per-episode phase boundaries with small random jitter.

    The values are ratios in (0, 1). They are raw/audit control variables only.
    They prevent the temporal model from memorizing fixed phase ratios while
    preserving the intended progression from trusted contact to downstream
    onboard inconsistency and containment.
    """
    base = np.array([0.08, 0.18, 0.34, 0.54, 0.72, 0.90], dtype=float)
    jitter = RNG.normal(0.0, 0.025, size=base.shape)

    if case_type == "HARD_AUTH_VALID_RATE_INCREASE":
        jitter += np.array([0.00, 0.00, -0.02, -0.01, 0.02, 0.02])
    elif case_type == "HARD_CROSS_PHASE_VALID_CONTROL":
        jitter += np.array([0.00, 0.01, -0.03, -0.02, 0.01, 0.02])
    elif case_type in BENIGN_OVERLAP_COMPROMISE_CASE_TYPES:
        jitter += np.array([0.02, 0.02, 0.02, 0.00, -0.01, -0.01])
    elif case_type == "EASY_CRITICAL_CONTROL_MISUSE":
        jitter += np.array([-0.01, -0.01, -0.02, -0.02, -0.01, 0.01])

    if compromise_type == "COMMAND_RATE_ABUSE":
        jitter += np.array([0.00, 0.00, -0.01, -0.02, 0.00, 0.01])
    elif compromise_type == "POLICY_VALID_BUT_MISSION_INCONSISTENT_COMMAND":
        jitter += np.array([0.01, 0.01, 0.00, -0.01, 0.00, 0.01])

    boundaries = np.clip(
        base + jitter,
        [0.04, 0.12, 0.24, 0.42, 0.60, 0.80],
        [0.14, 0.26, 0.44, 0.66, 0.82, 0.96],
    )

    min_gap = max(0.035, 1.5 / max(duration, 20))
    for idx in range(1, len(boundaries)):
        if boundaries[idx] <= boundaries[idx - 1] + min_gap:
            boundaries[idx] = boundaries[idx - 1] + min_gap
    boundaries = np.clip(boundaries, 0.02, 0.98)

    return {
        "normal_end": round(float(boundaries[0]), 4),
        "establishment_end": round(float(boundaries[1]), 4),
        "drift_end": round(float(boundaries[2]), 4),
        "abnormal_end": round(float(boundaries[3]), 4),
        "critical_end": round(float(boundaries[4]), 4),
        "inconsistency_end": round(float(boundaries[5]), 4),
    }

def get_ttc_compromise_phase(
    step: int,
    duration: int,
    phase_boundaries: Dict[str, float] | None = None,
) -> str:
    ratio = step / max(duration - 1, 1)
    if phase_boundaries is None:
        phase_boundaries = {
            "normal_end": 0.08,
            "establishment_end": 0.18,
            "drift_end": 0.34,
            "abnormal_end": 0.54,
            "critical_end": 0.72,
            "inconsistency_end": 0.90,
        }

    if ratio < phase_boundaries["normal_end"]:
        return "NORMAL_AUTHORIZED_CONTACT"
    if ratio < phase_boundaries["establishment_end"]:
        return "COMPROMISED_SESSION_ESTABLISHMENT"
    if ratio < phase_boundaries["drift_end"]:
        return "VALID_LOOKING_COMMAND_DRIFT"
    if ratio < phase_boundaries["abnormal_end"]:
        return "ABNORMAL_COMMAND_PATTERN"
    if ratio < phase_boundaries["critical_end"]:
        return "CRITICAL_CONTROL_MISUSE"
    if ratio < phase_boundaries["inconsistency_end"]:
        return "ONBOARD_STATE_INCONSISTENCY"
    return "CONTAINMENT_OR_OPERATOR_INTERVENTION"

def choose_compromise_case_type(compromise_type: str) -> str:
    if compromise_type == "COMMAND_RATE_ABUSE":
        return weighted_choice(
            TTC_COMPROMISE_CASE_TYPES,
            [0.18, 0.34, 0.11, 0.13, 0.09, 0.10, 0.035, 0.025],
        )

    if compromise_type == "CROSS_PHASE_CONTROL_MISUSE":
        return weighted_choice(
            TTC_COMPROMISE_CASE_TYPES,
            [0.17, 0.13, 0.38, 0.12, 0.09, 0.07, 0.025, 0.015],
        )

    if compromise_type == "CRITICAL_CONTROL_PATH_ABUSE":
        return weighted_choice(
            TTC_COMPROMISE_CASE_TYPES,
            [0.15, 0.13, 0.11, 0.24, 0.05, 0.05, 0.10, 0.17],
        )

    if compromise_type == "POLICY_VALID_BUT_MISSION_INCONSISTENT_COMMAND":
        return weighted_choice(
            TTC_COMPROMISE_CASE_TYPES,
            [0.24, 0.14, 0.23, 0.12, 0.17, 0.07, 0.02, 0.01],
        )

    if compromise_type == "GATEWAY_SESSION_HIJACK":
        return weighted_choice(
            TTC_COMPROMISE_CASE_TYPES,
            [0.24, 0.19, 0.13, 0.15, 0.07, 0.07, 0.10, 0.05],
        )

    return weighted_choice(TTC_COMPROMISE_CASE_TYPES, TTC_COMPROMISE_CASE_TYPE_PROBS)

LOCAL_PARAMETER_MIN=0.0

LOCAL_PARAMETER_MAX=1.0

MISSION_PARAMETER_MIN=.35

MISSION_PARAMETER_MAX=.65

APPROVED_CONTROL_PHASES={'COMMAND_SESSION'}

LOCAL_BATCH_LIMIT=2

LOCAL_ROLLING_RATE_LIMIT=90.0

COMMAND_PAYLOAD_OPTIONS={'STATUS_REQUEST':'STATUS_PARAMETER','TIME_SYNC':'TIME_SYNC_PARAMETER',
 'CONFIG_UPDATE':'SCHEDULED_CONFIG_PARAMETER','HOUSEKEEPING':'ROUTINE_PARAMETER','ORBIT_CONTROL':'PHASE_TRANSITION_PARAMETER',
 'SESSION_REAUTH':'POLICY_SYNC_CONTEXT'}

EVENT_FIELDS=['raw_command_id','raw_command_last_id','raw_command_sequence_id','raw_command_last_sequence_id',
 'raw_command_created_step','raw_command_parameter_value','raw_command_controller_id','raw_command_session_id']

OBSERVABLE_DOMAINS['command_intent'].update(STATE_CHANGING_INTENTS)

OBSERVABLE_DOMAINS['payload_or_parameter_context'].update({'PARAMETER_RANGE_VIOLATION','MISSION_PHASE_MISMATCH_PARAMETER',
 'OUT_OF_POLICY_PARAMETER','SUSPICIOUS_PARAMETER_PATTERN','CONTEXT_MISMATCH_PARAMETER','CONFIG_PARAMETER'})

OBSERVABLE_DOMAINS['session_state'].add('ABNORMAL')

OBSERVABLE_DOMAINS['command_execution_result'].add('BLOCKED')

OBSERVABLE_DOMAINS['state_transition_consistency'].update({'FORBIDDEN_TRANSITION','UNEXPECTED_TRANSITION'})

def phase_to_compromise_severity(phase: str, compromise_type: str) -> str:
    # Scenario schedule only; never consulted by the observable mapper.
    return {'NORMAL_AUTHORIZED_CONTACT':'NONE','COMPROMISED_SESSION_ESTABLISHMENT':'LOW',
        'VALID_LOOKING_COMMAND_DRIFT':'LOW','ABNORMAL_COMMAND_PATTERN':'MODERATE',
        'CRITICAL_CONTROL_MISUSE':'HIGH','ONBOARD_STATE_INCONSISTENCY':'HIGH',
        'CONTAINMENT_OR_OPERATOR_INTERVENTION':'RECOVERY'}[phase]

def build_ttc_compromise_episode_timeline(num_rows: int, *, nominal_timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    validate_generation_args(num_rows,1);validate_global_context_pool()
    timeline=[];index=0
    while len(timeline)<num_rows:
        kind=weighted_choice(TTC_COMPROMISE_EPISODE_TYPES,TTC_COMPROMISE_EPISODE_TYPE_PROBS)
        case=choose_compromise_case_type(kind)
        duration=nominal_timeline[len(timeline)]["episode_length"];bucket=nominal_timeline[len(timeline)]["episode_length_bucket"]
        boundaries=build_jittered_ttc_compromise_phase_boundaries(duration,kind,case)
        mode=weighted_choice(ENFORCEMENT_MODES,ENFORCEMENT_MODE_PROBS)
        # Case names retain old API provenance; they do not force token outcomes.
        increment=uniform_float(.015,.03,6) if case in BENIGN_OVERLAP_COMPROMISE_CASE_TYPES else uniform_float(.04,.08,6)
        sign=weighted_choice([-1,1],[.5,.5]);strength=uniform_float(.65,.90,6)
        for step in range(duration):
            if len(timeline)>=num_rows:break
            context={f: nominal_timeline[len(timeline)][f] for f in (*CONTEXT_FIELD_NAMES,"shared_context_profile","shared_context_family")}
            validate_context_profile(context);nominal={f: nominal_timeline[len(timeline)][f] for f in ("message_type","service_context","command_intent")}
            phase=get_ttc_compromise_phase(step,duration,boundaries)
            timeline.append({'scenario_id':'GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE_SCENARIO_0001',
                'episode_id':f'GTTC_COMP_EP_{index:05d}','episode_type':kind,'episode_step':step,
                'episode_length':duration,'episode_length_type':bucket,'phase_boundaries':boundaries,
                'compromise_type':kind,'compromise_case_type':case,'compromise_phase':phase,
                'compromise_severity':phase_to_compromise_severity(phase,kind),'enforcement_mode':mode,
                'drift_increment':increment,'drift_sign':sign,'attempt_strength':strength,
                **context,**{'nominal_'+k:v for k,v in nominal.items()}})
        index+=1
    return timeline

def new_compromise_state(episode: Dict[str, Any]) -> Dict[str, Any]:
    return {'episode_id':episode['episode_id'],'next_step':0,'sequence':0,'config':.5,'rate_history':[],
            'observed_commands':0,'executed_commands':0,'last_attack_window':-1,'last_attack_intent':'NONE'}

def is_compromise_target_context(context: Dict[str, Any]) -> bool:
    return (context['path_type']==TARGET_PATH and context['protocol_context']==TARGET_PROTOCOL
        and context['direction']==TARGET_DIRECTION and context['destination_role']=='MONITORED_LEO')

def sample_compromise_attempt_count(episode: Dict[str, Any]) -> int:
    if episode['compromise_phase'] not in ACTIVE_PHASES:return 0
    pressure={'VALID_LOOKING_COMMAND_DRIFT':.45,'ABNORMAL_COMMAND_PATTERN':.75,
        'CRITICAL_CONTROL_MISUSE':1.,'ONBOARD_STATE_INCONSISTENCY':.85}[episode['compromise_phase']]
    if RNG.random()>=episode['attempt_strength']*pressure:return 0
    return uniform_int(2,5) if episode['compromise_type']=='COMMAND_RATE_ABUSE' else 1

def command_checks_pass(raw: Dict[str, Any]) -> bool:
    return bool(raw['auth_present'] and raw['auth_valid'] and raw['integrity_valid'] and raw['source_authorized']
        and not any(raw[k] for k in ('auth_unverified_flag','integrity_unverified_flag','source_unverified_flag','policy_inconsistent_flag'))
        and (not raw['session_required_flag'] or raw['session_active'] and raw['session_valid'])
        and raw['sequence_gap']==1 and not raw['duplicate_flag'] and not raw['unknown_sequence_flag'])

def record_policy_evidence(raw: Dict[str, Any]) -> None:
    """Compute checks from command record, selected context and receiver history.

    The local guard intentionally does not enforce the reference mission
    envelope. Keeping those two checks distinct is central to class 6.
    """
    active=has_command_activity(raw);target=bool(raw['raw_observation_on_target'])
    raw['raw_payload_or_parameter_context']=raw['raw_base_payload_context']
    scoped=active and target;v=raw['raw_command_parameter_value']
    raw['raw_parameter_range_violation']=int(scoped and not LOCAL_PARAMETER_MIN<=v<=LOCAL_PARAMETER_MAX)
    raw['raw_phase_policy_violation']=int(scoped and raw['raw_command_intent']=='MODE_CHANGE'
        and raw['raw_operational_phase'] not in APPROVED_CONTROL_PHASES)
    raw['raw_rate_policy_violation']=int(scoped and (raw['raw_command_count']>LOCAL_BATCH_LIMIT
        or raw['raw_rolling_command_rate_per_min']>LOCAL_ROLLING_RATE_LIMIT))
    raw['raw_approval_policy_violation']=int(scoped and raw['raw_approval_required'] and not raw['raw_approval_present'])
    raw['raw_session_binding_violation']=int(scoped and (raw['raw_command_controller_id']!=1 or raw['raw_command_session_id']!=1))
    raw['session_valid']=1-raw['raw_session_binding_violation']
    raw['policy_inconsistent_flag']=int(any(raw[k] for k in ('raw_parameter_range_violation','raw_phase_policy_violation',
        'raw_rate_policy_violation','raw_approval_policy_violation')))
    raw['raw_command_permitted_in_context']=1-raw['policy_inconsistent_flag']
    changing=raw['raw_command_intent'] in STATE_CHANGING_INTENTS
    raw['raw_mission_candidate_violation']=int(scoped and changing and not MISSION_PARAMETER_MIN<=v<=MISSION_PARAMETER_MAX)
    if not scoped:return
    if raw['raw_parameter_range_violation']:payload='PARAMETER_RANGE_VIOLATION'
    elif raw['raw_phase_policy_violation']:payload='MISSION_PHASE_MISMATCH_PARAMETER'
    elif raw['raw_approval_policy_violation']:payload='OUT_OF_POLICY_PARAMETER'
    elif raw['raw_rate_policy_violation']:payload='SUSPICIOUS_PARAMETER_PATTERN'
    elif raw['raw_mission_candidate_violation']:payload='CONTEXT_MISMATCH_PARAMETER'
    else:return  # Preserve the already sampled nominal property.
    raw['raw_payload_or_parameter_context']=payload

def handle_representative_command(raw: Dict[str, Any]) -> None:
    """Independent checks, then one homogeneous-batch handler outcome.

    HELD/ACCEPTED do not execute later; no queue exists. AUDIT_ONLY records
    failed checks and explicit bypass. It never relabels a failure as valid.
    """
    for f in {'accepted_flag',*COMMAND_OUTCOME_FLAGS.values(),'state_changed_flag','unexpected_transition_flag',
              'forbidden_transition_flag','delayed_transition_flag','incomplete_transition_flag'}:raw[f]=0
    raw.update(transition_allowed=1,raw_security_gate_bypassed=0,raw_handler_dispatched=0)
    record_command_check_results(raw)
    if not has_command_activity(raw):raw['raw_command_handler_action']='NO_ACTION';return
    checks=command_checks_pass(raw)
    if not checks and raw['raw_enforcement_mode']=='FAIL_CLOSED':outcome='BLOCKED'
    else:
        raw['raw_handler_dispatched']=1;raw['raw_security_gate_bypassed']=int(not checks)
        outcome=weighted_choice(['EXECUTED','HELD','ACCEPTED'],[.75,.15,.10])
    set_command_outcome(raw,outcome)
    raw['raw_command_handler_action']={'EXECUTED':'EXECUTE','HELD':'HOLD','ACCEPTED':'ACCEPT','BLOCKED':'BLOCK'}[outcome]
    changing=raw['raw_command_intent'] in STATE_CHANGING_INTENTS
    changed=outcome=='EXECUTED' and changing
    if raw['raw_observation_on_target']:changed=changed and raw['raw_command_parameter_value']!=raw['raw_config_before']
    raw['state_changed_flag']=int(changed)
    raw['forbidden_transition_flag']=int(changed and raw['policy_inconsistent_flag'])
    raw['unexpected_transition_flag']=int(changed and not raw['forbidden_transition_flag']
        and (not checks or raw['raw_mission_candidate_violation']))
    raw['transition_allowed']=1-raw['forbidden_transition_flag']
    raw['delayed_transition_flag']=int(outcome=='HELD' and changing)

BASELINE_FIELDS=[f for f in NORMAL_RAW_RANGES if f not in {'session_age_s','command_rate_per_min','parameter_deviation_ratio'}]

BASELINE_FIELDS+=['fix_valid','gnss_lock_loss_count','gnss_cycle_slip_count','ttc_parse_error_count']

BASELINE_FIELDS+=[f'{p}_{k}_count' for p in ('ttc','isl','iot','uav') for k in ('crc_error','decode_fail','sync_loss','retry','timeout')]

RAW_FLAGS=['fix_valid','duplicate_flag','accepted_sequence_flag','unknown_sequence_flag','auth_present','auth_valid',
 'integrity_valid','source_authorized','auth_unverified_flag','integrity_unverified_flag','source_unverified_flag',
 'source_unverified_benign_flag','valid_looking_benign_flag','policy_inconsistent_flag','raw_command_permitted_in_context',
 'session_required_flag','session_active','session_valid','session_expired_flag','link_degraded_flag','link_loss_flag',
 'accepted_flag',*COMMAND_OUTCOME_FLAGS.values(),'state_changed_flag','unexpected_transition_flag','forbidden_transition_flag',
 'delayed_transition_flag','incomplete_transition_flag','transition_allowed','raw_security_gate_bypassed','raw_handler_dispatched',
 'raw_checks_passed','raw_attack_active','raw_observed_compromise','raw_observation_on_target','raw_is_command_context',
 'raw_config_applied','raw_mission_state_violation','raw_mission_candidate_violation','raw_parameter_range_violation',
 'raw_phase_policy_violation','raw_rate_policy_violation','raw_approval_policy_violation','raw_session_binding_violation',
 'raw_approval_present','raw_approval_required','raw_reused_intent_flag','raw_operator_intervention_flag']

RAW_COUNTS=['raw_window_seconds','raw_elapsed_seconds','raw_episode_length_seconds','raw_command_count','raw_nominal_command_count',
 'raw_compromised_command_count','raw_attempted_compromise_command_count','raw_telemetry_count','raw_iot_report_count',
 'raw_uav_message_count','raw_gnss_observation_count','critical_command_count','session_reset_count','raw_command_sequence_id',
 'raw_command_last_sequence_id','raw_command_controller_id','raw_command_session_id','raw_receiver_sequence_before','raw_receiver_sequence_after',
 'raw_rolling_command_count','raw_rolling_elapsed_seconds','raw_observed_compromise_commands_cumulative',
 'raw_executed_compromise_commands_cumulative','raw_attack_interval_s','raw_local_batch_limit']

RAW_TEXT=['raw_label_scope','raw_shared_context_profile','raw_shared_context_family','raw_target_path','raw_target_protocol',
 'raw_target_direction','raw_episode_length_type','raw_compromise_type','raw_compromise_case_type','raw_compromise_phase',
 'raw_compromise_severity','raw_enforcement_mode','raw_message_type','raw_service_context','raw_command_intent',
 'raw_payload_or_parameter_context','raw_base_payload_context','raw_representative_command_origin','raw_window_training_role','raw_command_handler_action',
 'raw_authentication_result','raw_integrity_check_result','raw_authorization_result','raw_policy_check_result','raw_session_check_result',
 'raw_command_id','raw_command_last_id','raw_episode_command_mode']

RAW_FLOATS=['raw_episode_progress','raw_drift_increment','raw_drift_sign','raw_attempt_strength','raw_command_parameter_value',
 'raw_config_before','raw_config_after','raw_rolling_command_rate_per_min','raw_local_parameter_min','raw_local_parameter_max',
 'raw_mission_parameter_min','raw_mission_parameter_max','raw_local_rolling_rate_limit','parameter_deviation_ratio','command_rate_per_min','session_age_s']

def select_sample_episode(df_full: pd.DataFrame) -> str:
    # An illustrative observed incident, not an unbiased prevalence estimate.
    for _,g in df_full.groupby('episode_id',sort=False):
        if len(g)==g.raw_episode_length_seconds.iloc[0] and g.raw_observed_compromise.sum()>=2 and g.raw_mission_state_violation.any():
            return str(g.episode_id.iloc[0])
    observed=df_full.loc[df_full.raw_observed_compromise.eq(1),'episode_id']
    return str(observed.iloc[0] if len(observed) else df_full.episode_id.iloc[0])

import shared_nominal as nominal
from shared_nominal import (normalize_context, normalize_sequence_freshness, normalize_control_security, normalize_message_control, normalize_signal_quality, normalize_tracking_stability, normalize_navigation_consistency, normalize_timing_consistency, normalize_decoder_status, normalize_command_execution, normalize_retry_timeout, normalize_session_state, normalize_state_transition, normalize_raw_to_observable)

# All benign batches contain at most three commands. Their existence alone is
# not a class-6 policy violation. The rolling rule uses a fixed 60-second window,
# not an inflated extrapolation from the first second of a new episode.
LOCAL_BATCH_LIMIT=3


def make_current_command(raw,state,attack):
    count=raw['raw_command_count'];target=bool(raw['raw_observation_on_target'])
    if not count:return
    first=(state['sequence'] if target else 0)+(1 if attack else max(1,int(raw['sequence_gap'])))
    raw.update(raw_command_id=f"{raw['episode_id']}:W{raw['window_id']}:C0",
               raw_command_last_id=f"{raw['episode_id']}:W{raw['window_id']}:C{count-1}",
               raw_command_sequence_id=first,raw_command_last_sequence_id=first+count-1,
               raw_command_created_step=raw['episode_step'],raw_command_controller_id=1,raw_command_session_id=1)
    if not attack:return  # The nominal parameter and every outcome are retained.
    kind=raw['raw_compromise_type']
    intents={'COMPROMISED_OPERATOR_SESSION':'CONFIG_UPDATE','VALID_AUTH_ABNORMAL_COMMAND':'CONFIG_UPDATE',
             'COMMAND_RATE_ABUSE':'STATUS_REQUEST','CROSS_PHASE_CONTROL_MISUSE':'MODE_CHANGE',
             'CRITICAL_CONTROL_PATH_ABUSE':'MODE_CHANGE','GATEWAY_SESSION_HIJACK':'ROUTE_UPDATE',
             'POLICY_VALID_BUT_MISSION_INCONSISTENT_COMMAND':'CONFIG_UPDATE'}
    raw['raw_command_intent']=intents[kind]
    message,service,payload=nominal.sample_command_envelope(raw['raw_command_intent'],RNG)
    raw.update(raw_message_type=message,raw_service_context=service,raw_base_payload_context=payload)
    value=round(float(np.clip(state['config']+raw['raw_drift_sign']*raw['raw_drift_increment'],.05,.95)),6)
    if kind=='VALID_AUTH_ABNORMAL_COMMAND':value=round(1.+raw['raw_drift_increment'],6)
    elif kind=='COMMAND_RATE_ABUSE':value=.5
    elif kind=='CRITICAL_CONTROL_PATH_ABUSE':
        value=.9 if raw['raw_drift_sign']>0 else .1;raw['raw_approval_required']=1;raw['raw_approval_present']=0
    elif kind=='COMPROMISED_OPERATOR_SESSION':raw['raw_command_controller_id']=2
    elif kind=='GATEWAY_SESSION_HIJACK':raw['raw_command_session_id']=2
    raw['raw_command_parameter_value']=value
    raw['parameter_deviation_ratio']=round(2*abs(value-.5),6)


def generate_ttc_compromise_raw_window(window_id,timestamp,episode,compromise_state=None,
                                      *,nominal_raw=None,attacks_enabled=True):
    if compromise_state is None:raise ValueError('Compromise requires its episode receiver state.')
    state=compromise_state
    if state['episode_id']!=episode['episode_id'] or state['next_step']!=episode['episode_step']:
        raise ValueError('Compromise receiver state must follow consecutive episode windows.')
    raw=nominal.copy_nominal_window(nominal_raw,episode,LABEL_GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE)
    target=is_compromise_target_context(episode);step=episode['episode_step']
    for f in ('compromise_type','compromise_case_type','compromise_phase','compromise_severity',
              'enforcement_mode','drift_increment','drift_sign','attempt_strength'):raw['raw_'+f]=episode[f]
    raw.update(raw_target_path=TARGET_PATH,raw_target_protocol=TARGET_PROTOCOL,raw_target_direction=TARGET_DIRECTION,
        raw_observation_on_target=int(target),raw_is_command_context=int(is_command_context(episode)),
        raw_command_id='NONE',raw_command_last_id='NONE',raw_command_sequence_id=0,raw_command_last_sequence_id=0,
        raw_command_created_step=-1,raw_command_controller_id=0,raw_command_session_id=0,
        raw_approval_required=0,raw_approval_present=1,raw_mission_candidate_violation=0,
        raw_config_before=state['config'],raw_receiver_sequence_before=state['sequence'],
        raw_local_parameter_min=LOCAL_PARAMETER_MIN,raw_local_parameter_max=LOCAL_PARAMETER_MAX,
        raw_mission_parameter_min=MISSION_PARAMETER_MIN,raw_mission_parameter_max=MISSION_PARAMETER_MAX,
        raw_local_batch_limit=LOCAL_BATCH_LIMIT,raw_local_rolling_rate_limit=LOCAL_ROLLING_RATE_LIMIT,
        raw_operator_intervention_flag=int(episode['compromise_phase']=='CONTAINMENT_OR_OPERATOR_INTERVENTION'),
        raw_parameter_range_violation=int(raw['parameter_range_violation_flag']),
        raw_phase_policy_violation=int(raw['phase_policy_block_flag']),raw_rate_policy_violation=0,
        raw_approval_policy_violation=0,raw_session_binding_violation=0)
    raw['raw_base_payload_context']=raw['raw_payload_or_parameter_context']
    nominal.record_nominal_checks(raw)
    make_current_command(raw,state,False)
    attempts=sample_compromise_attempt_count(episode) if attacks_enabled else 0
    observed=attempts if target else 0
    raw.update(raw_attempted_compromise_command_count=attempts,raw_compromised_command_count=observed,
               raw_attack_active=int(attempts>0),raw_observed_compromise=int(observed>0))
    if observed:
        nominal.begin_command_attack(raw)
        raw['raw_nominal_command_count']=0;raw['raw_command_count']=observed
        make_current_command(raw,state,True)
    history=[(s,n) for s,n in state['rate_history'] if s>=step-59]
    nominal_history=[(s,n) for s,n in state.get('nominal_rate_history',[]) if s>=step-59]
    if target:
        if raw['raw_command_count']:history.append((step,raw['raw_command_count']))
        if nominal_raw['raw_command_count']:nominal_history.append((step,nominal_raw['raw_command_count']))
    raw['raw_rolling_command_count']=sum(n for _,n in history)
    raw['raw_nominal_rolling_command_count']=sum(n for _,n in nominal_history)
    raw['raw_rolling_elapsed_seconds']=min(step+1,60)
    raw['raw_rolling_command_rate_per_min']=float(raw['raw_rolling_command_count'])
    raw['raw_rolling_window_seconds']=60
    raw['raw_rolling_warmup_flag']=int(step<59)
    # Retain a real after-effect if earlier attack traffic alone trips the guard.
    residual=bool(not observed and target and raw['raw_command_count'] and
                  raw['raw_rolling_command_count']>LOCAL_ROLLING_RATE_LIMIT>=raw['raw_nominal_rolling_command_count'])
    raw['raw_persistent_effect_present']=int(residual)
    if observed:
        raw['critical_command_count']=raw['raw_command_count'] if raw['raw_command_intent']=='MODE_CHANGE' else 0
        record_policy_evidence(raw);handle_representative_command(raw)
    elif residual:
        raw['raw_rate_policy_violation']=1;raw['policy_inconsistent_flag']=1;raw['raw_command_permitted_in_context']=0
        raw['raw_payload_or_parameter_context']='SUSPICIOUS_PARAMETER_PATTERN'
        handle_representative_command(raw)
    raw['command_rate_per_min']=raw['raw_command_count']*60
    state['rate_history']=history;state['nominal_rate_history']=nominal_history
    raw['raw_config_applied']=int(target and raw['executed_flag'] and raw['raw_command_intent'] in STATE_CHANGING_INTENTS)
    if raw['raw_config_applied']:state['config']=raw['raw_command_parameter_value']
    if target and raw['raw_command_count']:state['sequence']=raw['raw_command_last_sequence_id']
    state['observed_commands']+=observed;state['executed_commands']+=observed*raw['executed_flag']
    raw['raw_previous_attack_step']=state['last_attack_window']
    raw['raw_attack_interval_s']=step-state['last_attack_window'] if observed and state['last_attack_window']>=0 else 0
    raw['raw_reused_intent_flag']=int(observed and state['last_attack_intent']==raw['raw_command_intent'])
    if observed:state['last_attack_window']=step;state['last_attack_intent']=raw['raw_command_intent']
    raw.update(raw_config_after=state['config'],raw_receiver_sequence_after=state['sequence'],
        raw_observed_compromise_commands_cumulative=state['observed_commands'],
        raw_executed_compromise_commands_cumulative=state['executed_commands'],
        raw_mission_state_violation=int(not MISSION_PARAMETER_MIN<=state['config']<=MISSION_PARAMETER_MAX),
        raw_representative_command_origin='COMPROMISED_PATH' if observed else 'NOMINAL' if raw['raw_command_count'] else 'NONE',
        raw_window_training_role='OBSERVED_COMPROMISE' if observed else 'UNOBSERVED_ATTEMPT' if attempts else 'CONTEXT_ONLY')
    state['next_step']+=1
    raw['raw_policy_violation_flag']=int(raw['policy_inconsistent_flag'])
    return raw


def validate_attack_rows(df):
    quiet=df.raw_compromised_command_count.eq(0)&df.raw_persistent_effect_present.eq(0)
    nominal.require_nominal_when(df,quiet)
    nominal.validate_observed_command_gate(df,'raw_compromised_command_count')
    if (df.raw_compromised_command_count.gt(0)&df.raw_observation_on_target.eq(0)).any():raise ValueError('Off-target compromise evidence.')
    for _,episode in df.groupby('episode_id',sort=False):
        config=.5;sequence=0;history=[]
        for r in episode.to_dict('records'):
            if r['raw_config_before']!=config or r['raw_receiver_sequence_before']!=sequence:
                raise ValueError('Receiver state does not carry the preceding state.')
            if r['raw_config_applied']:
                if not (r['executed_flag'] and r['raw_observation_on_target']):raise ValueError('Configuration changed without target execution.')
                config=r['raw_command_parameter_value']
            if r['raw_observation_on_target'] and r['raw_command_count']:sequence=r['raw_command_last_sequence_id']
            if r['raw_config_after']!=config or r['raw_receiver_sequence_after']!=sequence:raise ValueError('Invalid post-command state.')
            history=[(s,n) for s,n in history if s>=r['episode_step']-59]
            if r['raw_observation_on_target'] and r['raw_command_count']:history.append((r['episode_step'],r['raw_command_count']))
            if r['raw_rolling_command_rate_per_min']!=sum(n for _,n in history):raise ValueError('Invalid rolling traffic evidence.')

MAKER_NUMBER = 6
CLASS_LABEL = LABEL_GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE
CODE_VERSION = nominal.CODE_VERSION
TIMELINE_BUILDER = 'build_ttc_compromise_episode_timeline'
RAW_GENERATOR = 'generate_ttc_compromise_raw_window'
OBSERVABLE_DOMAINS = nominal.OBSERVABLE_DOMAINS

def build_ttc_compromise_dataset(num_rows: int = DEFAULT_NUM_ROWS, window_seconds: int = DEFAULT_WINDOW_SECONDS,
    start_time: datetime | None = None, seed: int = DEFAULT_RANDOM_SEED, return_raw: bool = False,
    *, background_seed: int | None = None, attacks_enabled: bool = True) -> pd.DataFrame:
    """Build from the common nominal substrate; explicit background_seed is for paired audits.

    Default background streams are disjoint across makers, including equal seed values.
    Do not pool replicated paired-audit backgrounds into independent train/test splits.
    """
    return nominal.build_dataset(globals(), num_rows, window_seconds, start_time, seed,
                                 return_raw, background_seed, attacks_enabled)

def validate_dataset(df, expected_rows=None, expected_window_seconds=1):
    nominal.validate_learning_dataset(df, CLASS_LABEL, expected_rows, expected_window_seconds)

def validate_learning_ready_schema(df_learning, df_full=None, expected_rows=None, expected_window_seconds=1):
    validate_dataset(df_learning, expected_rows, expected_window_seconds)
    if df_full is not None and not df_learning.reset_index(drop=True).equals(
            df_full[METADATA_FIELDS + OBSERVABLE_FIELDS].reset_index(drop=True)):
        raise ValueError("Learning and audit rows differ.")

def validate_raw_audit_dataset(df, expected_rows=None, expected_window_seconds=1):
    nominal.validate_raw_dataset(df, globals(), expected_rows, expected_window_seconds)

if __name__ == "__main__":
    nominal.run_cli(globals(), 'build_ttc_compromise_dataset')
