# Revision: common nominal background + shared observation mapping.
# Research objective: reviewer revision on synthetic-rule dependence and temporal evidence.
"""Controlled synthetic telecommand replay, revised 2026-10-02.

A one-second row selects one shared observation context. The 22 observable
fields are model inputs; IDs, event provenance and receiver state are audit only.
An authorized original command is passively captured when actually observed; its outcome is not forced. Replay
copies its entire synthetic command record; a fingerprint identifies reuse.
This is not a CCSDS packet implementation, real cryptography or calibrated FSW.
No RF/GNSS damage, command queue or persistent spacecraft dynamics is modeled.
Episode labels include pre, quiet, unobserved and post-attempt windows.

The duplicate cache may be unavailable while the sequence high-water mark remains available.
A logged reset resynchronizes sender and receiver sequence counters. Freshness is
computed from what the receiver retains, never chosen to meet a token quota.
A replay can therefore appear fresh after a reset; ground-truth reuse is still
retained separately. Old A/A-prime performance does not describe this version.
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

CODE_VERSION = 'replay-consistent-history-v1-20261002'

LABEL_SCOPE = 'EPISODE_SCENARIO'

TARGET_PATH = 'GATEWAY_LEO'

TARGET_PROTOCOL = 'CCSDS_TTC'

TARGET_DIRECTION = 'GROUND_TO_SPACE'

ENFORCEMENT_MODES = ['FAIL_CLOSED','AUDIT_ONLY']

ENFORCEMENT_MODE_PROBS = [.80,.20]

MEMORY_MODES = ['PERSISTENT','RESET_ONCE','UNAVAILABLE']

PHASE_BOUNDARY_KEYS = ['pre_end','valid_end','attempt_end','freshness_end','response_end','side_effect_end','recovery_end']

ACTIVE_PHASES = {'REPLAY_ATTEMPT','FRESHNESS_CHECK_FAILURE','DUPLICATE_COMMAND_RESPONSE','OPERATIONAL_SIDE_EFFECT'}

DEFAULT_NUM_ROWS = 50000

DEFAULT_WINDOW_SECONDS = 1

DEFAULT_OUTPUT_PATH = "dataset/telecommand_replay_observable_dataset.csv"

DEFAULT_RAW_AUDIT_OUTPUT_PATH = "dataset/telecommand_replay_observable_dataset_with_raw.csv"

DEFAULT_SAMPLE_TIMELINE_OUTPUT_PATH = "dataset/telecommand_replay_sample_episode_timeline.csv"

DEFAULT_RANDOM_SEED = 442

LABEL_TELECOMMAND_REPLAY = "TELECOMMAND_REPLAY"

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

TC_REPLAY_EPISODE_TYPES = [
    "DUPLICATE_TELECOMMAND_REPLAY",
    "STALE_SEQUENCE_REPLAY",
    "EXPIRED_SESSION_COMMAND_REPLAY",
    "DELAYED_COMMAND_REPLAY",
    "VALID_LOOKING_REPLAY",
    "CROSS_PHASE_COMMAND_REPLAY",
    "BURST_REPLAY_ATTEMPT",
]

TC_REPLAY_EPISODE_TYPE_PROBS = [
    0.25,
    0.20,
    0.18,
    0.15,
    0.12,
    0.07,
    0.03,
]

TC_REPLAY_PHASES = [
    "PRE_VALID_COMMAND_CONTEXT",
    "VALID_COMMAND_EXECUTION",
    "REPLAY_ATTEMPT",
    "FRESHNESS_CHECK_FAILURE",
    "DUPLICATE_COMMAND_RESPONSE",
    "OPERATIONAL_SIDE_EFFECT",
    "CONTAINMENT_OR_RECOVERY",
]

TC_REPLAY_CASE_TYPES = [
    "EASY_EXPLICIT_REPLAY",
    "EASY_STALE_DUPLICATE",
    "HARD_FRESH_AUTH_VALID_RECURRENCE",
    "HARD_ACCEPTED_HELD_REPLAY",
    "HARD_CROSS_PHASE_VALID_REUSE",
    "BENIGN_OVERLAP_REPEATED_STATUS",
    "PERSISTENT_LOW_RATE_REPLAY",
    "BURST_REPLAY_EVIDENCE",
]

COMMON_EPISODE_LENGTH_BUCKETS = ["SHORT", "MEDIUM", "LONG"]

COMMON_EPISODE_LENGTH_BUCKET_PROBS = [0.35, 0.45, 0.20]

COMMON_EPISODE_LENGTH_RANGES = {
    "SHORT": (20, 40),
    "MEDIUM": (40, 80),
    "LONG": (80, 140),
}

TC_SHARED_COMMAND_INTENTS = [
    "STATUS_REQUEST",
    "TIME_SYNC",
    "PAYLOAD_CONFIG",
    "ATTITUDE_ADJUST",
    "ROUTE_UPDATE",
    "MODE_CHANGE",
    "NOMINAL_CONTROL",
    "HOUSEKEEPING",
]

TC_SHARED_PARAMETER_CONTEXTS = [
    "NOMINAL_PARAMETER",
    "CONFIG_PARAMETER",
    "CONTROL_PARAMETER",
    "ROUTINE_PARAMETER",
    "STATUS_PARAMETER",
    "TIME_SYNC_PARAMETER",
    "CONTEXT_MISMATCH_PARAMETER",
]

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

RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)

_CANONICAL_LEARNING_READY_COLUMNS = METADATA_FIELDS + OBSERVABLE_FIELDS

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

def sample_common_episode_duration() -> tuple[int, str]:
    """Sample a class-balanced episode length.

    All six makers use the same SHORT/MEDIUM/LONG mixture to reduce
    class-specific episode-length leakage. The returned bucket is stored only
    as raw audit metadata, not as a learning-visible field.
    """
    bucket = weighted_choice(COMMON_EPISODE_LENGTH_BUCKETS, COMMON_EPISODE_LENGTH_BUCKET_PROBS)
    low, high = COMMON_EPISODE_LENGTH_RANGES[bucket]
    return int(RNG.integers(low, high + 1)), bucket

def build_replay_phase_counts(duration: int, replay_type: str, case_type: str) -> Dict[str, int]:
    """Retained synthetic phase/case prior; names do not establish model difficulty."""
    phases = TC_REPLAY_PHASES
    base = np.array([0.08, 0.11, 0.18, 0.19, 0.18, 0.16, 0.10], dtype=float)

    if case_type == "BENIGN_OVERLAP_REPEATED_STATUS":
        base += np.array([0.03, 0.03, -0.02, -0.03, -0.02, -0.02, 0.03])
    elif case_type == "BURST_REPLAY_EVIDENCE" or replay_type == "BURST_REPLAY_ATTEMPT":
        base += np.array([-0.02, -0.02, 0.05, 0.03, 0.02, -0.02, -0.04])
    elif case_type == "HARD_CROSS_PHASE_VALID_REUSE":
        base += np.array([0.00, -0.01, 0.01, 0.00, -0.01, 0.04, -0.03])
    elif case_type in ["HARD_FRESH_AUTH_VALID_RECURRENCE", "PERSISTENT_LOW_RATE_REPLAY"]:
        base += np.array([0.01, 0.01, 0.00, 0.02, 0.00, 0.00, -0.04])

    base = np.clip(base, 0.035, None)
    base = base / base.sum()

    min_counts = np.array([2, 2, 3, 3, 3, 3, 2], dtype=int)
    if duration < int(min_counts.sum()):
        # Defensive fallback; current common duration is >=20, so this should not trigger.
        min_counts = np.ones(len(phases), dtype=int)

    concentration = 45.0
    jittered_probs = RNG.dirichlet(base * concentration)
    remaining = int(duration - min_counts.sum())
    extra = RNG.multinomial(max(remaining, 0), jittered_probs)
    counts = min_counts + extra

    return {phase: int(count) for phase, count in zip(phases, counts)}

def phase_counts_to_boundaries(phase_counts: Dict[str, int]) -> Dict[str, int]:
    cumulative = 0
    boundaries: Dict[str, int] = {}
    boundary_keys = [
        "pre_end",
        "valid_end",
        "attempt_end",
        "freshness_end",
        "response_end",
        "side_effect_end",
        "recovery_end",
    ]
    for key, phase in zip(boundary_keys, TC_REPLAY_PHASES):
        cumulative += int(phase_counts[phase])
        boundaries[key] = cumulative
    return boundaries

def get_tc_replay_phase(step: int, duration: int, phase_boundaries: Dict[str, int] | None = None) -> str:
    if phase_boundaries is not None:
        if step < phase_boundaries["pre_end"]:
            return "PRE_VALID_COMMAND_CONTEXT"
        if step < phase_boundaries["valid_end"]:
            return "VALID_COMMAND_EXECUTION"
        if step < phase_boundaries["attempt_end"]:
            return "REPLAY_ATTEMPT"
        if step < phase_boundaries["freshness_end"]:
            return "FRESHNESS_CHECK_FAILURE"
        if step < phase_boundaries["response_end"]:
            return "DUPLICATE_COMMAND_RESPONSE"
        if step < phase_boundaries["side_effect_end"]:
            return "OPERATIONAL_SIDE_EFFECT"
        return "CONTAINMENT_OR_RECOVERY"

    ratio = step / max(duration - 1, 1)
    if ratio < 0.08:
        return "PRE_VALID_COMMAND_CONTEXT"
    if ratio < 0.19:
        return "VALID_COMMAND_EXECUTION"
    if ratio < 0.37:
        return "REPLAY_ATTEMPT"
    if ratio < 0.56:
        return "FRESHNESS_CHECK_FAILURE"
    if ratio < 0.74:
        return "DUPLICATE_COMMAND_RESPONSE"
    if ratio < 0.90:
        return "OPERATIONAL_SIDE_EFFECT"
    return "CONTAINMENT_OR_RECOVERY"

def choose_replay_case_type(replay_type: str) -> str:
    """Retained synthetic phase/case prior; names do not establish model difficulty."""
    if replay_type == "VALID_LOOKING_REPLAY":
        return weighted_choice(
            [
                "HARD_FRESH_AUTH_VALID_RECURRENCE",
                "HARD_ACCEPTED_HELD_REPLAY",
                "PERSISTENT_LOW_RATE_REPLAY",
                "BENIGN_OVERLAP_REPEATED_STATUS",
                "HARD_CROSS_PHASE_VALID_REUSE",
                "EASY_EXPLICIT_REPLAY",
            ],
            [0.46, 0.22, 0.18, 0.10, 0.035, 0.005],
        )

    if replay_type == "CROSS_PHASE_COMMAND_REPLAY":
        return weighted_choice(
            [
                "HARD_CROSS_PHASE_VALID_REUSE",
                "HARD_FRESH_AUTH_VALID_RECURRENCE",
                "PERSISTENT_LOW_RATE_REPLAY",
                "HARD_ACCEPTED_HELD_REPLAY",
                "EASY_STALE_DUPLICATE",
            ],
            [0.56, 0.24, 0.12, 0.06, 0.02],
        )

    if replay_type == "DELAYED_COMMAND_REPLAY":
        return weighted_choice(
            [
                "PERSISTENT_LOW_RATE_REPLAY",
                "HARD_ACCEPTED_HELD_REPLAY",
                "HARD_FRESH_AUTH_VALID_RECURRENCE",
                "BENIGN_OVERLAP_REPEATED_STATUS",
                "EASY_STALE_DUPLICATE",
            ],
            [0.40, 0.28, 0.22, 0.08, 0.02],
        )

    if replay_type == "EXPIRED_SESSION_COMMAND_REPLAY":
        return weighted_choice(
            [
                "HARD_ACCEPTED_HELD_REPLAY",
                "PERSISTENT_LOW_RATE_REPLAY",
                "HARD_CROSS_PHASE_VALID_REUSE",
                "HARD_FRESH_AUTH_VALID_RECURRENCE",
                "EASY_STALE_DUPLICATE",
                "EASY_EXPLICIT_REPLAY",
            ],
            [0.34, 0.25, 0.18, 0.14, 0.07, 0.02],
        )

    if replay_type == "BURST_REPLAY_ATTEMPT":
        return weighted_choice(
            [
                "BURST_REPLAY_EVIDENCE",
                "HARD_ACCEPTED_HELD_REPLAY",
                "HARD_FRESH_AUTH_VALID_RECURRENCE",
                "PERSISTENT_LOW_RATE_REPLAY",
                "EASY_EXPLICIT_REPLAY",
            ],
            [0.18, 0.30, 0.28, 0.16, 0.08],
        )

    if replay_type == "STALE_SEQUENCE_REPLAY":
        return weighted_choice(
            [
                "PERSISTENT_LOW_RATE_REPLAY",
                "HARD_CROSS_PHASE_VALID_REUSE",
                "HARD_FRESH_AUTH_VALID_RECURRENCE",
                "HARD_ACCEPTED_HELD_REPLAY",
                "EASY_STALE_DUPLICATE",
            ],
            [0.36, 0.24, 0.20, 0.12, 0.08],
        )

    if replay_type == "DUPLICATE_TELECOMMAND_REPLAY":
        return weighted_choice(
            [
                "HARD_FRESH_AUTH_VALID_RECURRENCE",
                "BENIGN_OVERLAP_REPEATED_STATUS",
                "HARD_ACCEPTED_HELD_REPLAY",
                "PERSISTENT_LOW_RATE_REPLAY",
                "EASY_EXPLICIT_REPLAY",
            ],
            [0.42, 0.22, 0.18, 0.10, 0.08],
        )

    return weighted_choice(
        TC_REPLAY_CASE_TYPES,
        [0.035, 0.045, 0.34, 0.18, 0.16, 0.12, 0.095, 0.025],
    )

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

def handle_representative_command(raw: Dict[str, Any], force_original: bool = False) -> None:
    """One representative current command, no cross-window queue or residual action.

    accepted_flag means admitted for processing. It may coexist with EXECUTED
    or HELD; the current outcome flags themselves are mutually exclusive.
    Checks are evaluated before the handler. AUDIT_ONLY explicitly does not
    enforce a negative check result; it does not turn that result into success.
    """
    raw['accepted_flag'] = 0
    for flag in COMMAND_OUTCOME_FLAGS.values(): raw[flag] = 0
    for flag in ('state_changed_flag','unexpected_transition_flag','forbidden_transition_flag',
                 'delayed_transition_flag','incomplete_transition_flag'):
        raw[flag] = 0
    raw['transition_allowed'] = 1
    raw['raw_security_gate_bypassed'] = 0
    raw['raw_handler_dispatched'] = 0
    record_command_check_results(raw)
    if not has_command_activity(raw):
        raw['raw_command_handler_action'] = 'NO_ACTION'
        return
    checks = command_checks_pass(raw)
    if not checks and raw['raw_enforcement_mode'] == 'FAIL_CLOSED':
        failed_crypto = not raw['auth_present'] or not raw['auth_valid'] or not raw['integrity_valid']
        pending = any(raw[k] for k in ('auth_unverified_flag','integrity_unverified_flag','source_unverified_flag'))
        # Unverified commands stay held before admission, not simulated execution.
        if pending:
            outcome = 'HELD'
            raw['held_flag'] = 1
        else:
            outcome = 'REJECTED' if failed_crypto else 'BLOCKED'
            set_command_outcome(raw, outcome)
    else:
        raw['raw_handler_dispatched'] = 1
        raw['raw_security_gate_bypassed'] = int(not checks)
        # This models handler processing, not a calibrated execution-success rate.
        outcome = 'EXECUTED' if force_original and checks else weighted_choice(['EXECUTED','HELD','ACCEPTED'], [.75,.15,.10])
        set_command_outcome(raw,outcome)
    raw['raw_command_handler_action'] = {
        'EXECUTED':'EXECUTE','HELD':'HOLD','ACCEPTED':'ACCEPT','REJECTED':'REJECT','BLOCKED':'BLOCK'}[outcome]
    changes_state = raw['raw_command_intent'] in STATE_CHANGING_INTENTS
    if outcome == 'EXECUTED' and changes_state:
        raw['state_changed_flag'] = 1
        if raw['policy_inconsistent_flag']:
            raw['forbidden_transition_flag'] = 1
            raw['transition_allowed'] = 0
        elif not checks:
            raw['unexpected_transition_flag'] = 1
    elif outcome == 'HELD' and changes_state:
        raw['delayed_transition_flag'] = 1

def phase_to_replay_severity(phase: str, replay_type: str) -> str:
    """Audit stratum; not a measured security consequence."""
    return {'REPLAY_ATTEMPT':'LOW','FRESHNESS_CHECK_FAILURE':'MODERATE',
            'DUPLICATE_COMMAND_RESPONSE':'MODERATE','OPERATIONAL_SIDE_EFFECT':'HIGH'}.get(phase,'NONE')

def build_tc_replay_episode_timeline(num_rows: int, *, nominal_timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    validate_generation_args(num_rows,1)
    validate_global_context_pool()
    rows=[]; episode_index=0
    while len(rows)<num_rows:
        kind=weighted_choice(TC_REPLAY_EPISODE_TYPES,TC_REPLAY_EPISODE_TYPE_PROBS)
        case=choose_replay_case_type(kind)
        duration=nominal_timeline[len(rows)]["episode_length"];bucket=nominal_timeline[len(rows)]["episode_length_bucket"]
        boundaries=phase_counts_to_boundaries(build_replay_phase_counts(duration,kind,case))
        intent=weighted_choice(TC_SHARED_COMMAND_INTENTS,[.18,.16,.17,.14,.13,.08,.06,.08])
        payload={'STATUS_REQUEST':'STATUS_PARAMETER','HOUSEKEEPING':'ROUTINE_PARAMETER',
                 'TIME_SYNC':'TIME_SYNC_PARAMETER'}.get(intent,'CONFIG_PARAMETER')
        # No command-body anomaly is added simply because a message is replayed.
        payload=weighted_choice([payload,'NOMINAL_PARAMETER'],[.6,.4])
        enforcement=weighted_choice(ENFORCEMENT_MODES,ENFORCEMENT_MODE_PROBS)
        memory=('RESET_ONCE' if kind=='VALID_LOOKING_REPLAY' else
                weighted_choice(['PERSISTENT','UNAVAILABLE'],[.90,.10]))
        ttl=uniform_int(5,15) if kind=='DELAYED_COMMAND_REPLAY' else 10000
        strength=uniform_float(.65,.90,3)
        for step in range(duration):
            if len(rows)>=num_rows:break
            phase=get_tc_replay_phase(step,duration,boundaries)
            context={f: nominal_timeline[len(rows)][f] for f in (*CONTEXT_FIELD_NAMES,"shared_context_profile","shared_context_family")}
            validate_context_profile(context)
            nominal={f: nominal_timeline[len(rows)][f] for f in ("message_type","service_context","command_intent")}
            rows.append({
                'scenario_id':'TELECOMMAND_REPLAY_SCENARIO_0001','episode_id':f'TCREPLAY_EP_{episode_index:05d}',
                'episode_type':kind,'episode_step':step,'episode_length':duration,'episode_length_bucket':bucket,
                'phase_boundaries':boundaries,'replay_phase':phase,'replay_type':kind,'replay_case_type':case,
                'replay_severity':phase_to_replay_severity(phase,kind),'original_command_intent':intent,
                'original_parameter_profile':payload,'episode_recurrence_strength':strength,
                'enforcement_mode':enforcement,'memory_mode':memory,'command_ttl_s':ttl,
                'shared_context_profile':context['shared_context_profile'],'shared_context_family':context['shared_context_family'],
                'nominal_message_type':nominal['message_type'],'nominal_service_context':nominal['service_context'],
                'nominal_command_intent':nominal['command_intent'],
                **{f:context[f] for f in CONTEXT_FIELD_NAMES}})
        episode_index+=1
    return rows

def new_replay_state() -> Dict[str, Any]:
    """One episode, one monitored TT&C stream. No history crosses episodes."""
    return {'original':None,'highest':0,'issued':0,'session':0,'epoch':0,'seen':set(),
            'changes_done':False,'previous_attempt_step':-1,'observed_replay_windows':0}

def is_replay_target_context(context: Dict[str, Any]) -> bool:
    return (context['path_type']==TARGET_PATH and context['protocol_context']==TARGET_PROTOCOL
            and context['direction']==TARGET_DIRECTION and context['destination_role']=='MONITORED_LEO')

def command_fingerprint(command: Dict[str, Any]) -> str:
    """Synthetic record identity, not a wire-packet hash or authentication result."""
    content={k:(int(command[k]) if k in {'sequence_id','session_id','context_epoch','created_step','window_id'}
             else float(command[k]) if k=='parameter_value' else str(command[k])) for k in EVENT_KEYS}
    return hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':')).encode()).hexdigest()

EVENT_KEYS=['id','sequence_id','session_id','context_epoch','created_step','window_id','intent','payload','parameter_value']

EVENT_RAW_FIELDS={k:'raw_command_'+k for k in EVENT_KEYS}

EVENT_RAW_FIELDS.update({'intent':'raw_command_intent','payload':'raw_payload_or_parameter_context'})

def event_from_raw(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {k:raw[v] for k,v in EVENT_RAW_FIELDS.items()}

def advance_receiver_state(state: Dict[str, Any], episode: Dict[str, Any]) -> tuple[int,int,int]:
    """Apply a logged receiver/session/policy change after capture. Reset also resynchronizes the sender sequence counter; UNAVAILABLE disables only the duplicate cache."""
    reset=session_change=epoch_change=0
    if state['original'] is not None and not state['changes_done'] and episode['replay_phase'] in ACTIVE_PHASES:
        state['changes_done']=True
        if episode['replay_type']=='EXPIRED_SESSION_COMMAND_REPLAY':
            state['session']+=1;state['highest']=0;state['issued']=0;state['seen'].clear();session_change=1
        if episode['replay_type']=='CROSS_PHASE_COMMAND_REPLAY':
            state['epoch']+=1;epoch_change=1
        if episode['memory_mode']=='RESET_ONCE':
            state['highest']=0;state['issued']=0;state['seen'].clear();reset=1
    return reset,session_change,epoch_change

def replay_eligible(state: Dict[str, Any], episode: Dict[str, Any]) -> bool:
    original=state['original']
    if original is None or episode['replay_phase'] not in ACTIVE_PHASES:return False
    age=episode['episode_step']-original['created_step']
    if age<=0:return False
    if episode['replay_type']=='DELAYED_COMMAND_REPLAY' and age<=episode['command_ttl_s']:return False
    if episode['replay_type']=='STALE_SEQUENCE_REPLAY' and state['issued']<=original['sequence_id']:return False
    return True

def sample_replay_attempt_count(episode: Dict[str, Any], eligible: bool) -> int:
    if not eligible:return 0
    probability=episode['episode_recurrence_strength']
    if episode['replay_case_type']=='PERSISTENT_LOW_RATE_REPLAY':probability*=.65
    if RNG.random()>=probability:return 0
    if episode['replay_type']=='BURST_REPLAY_ATTEMPT':return count_choice([2,3,4],[.5,.35,.15])
    return count_choice([1,2],[.88,.12])

def command_checks_pass(raw: Dict[str, Any]) -> bool:
    return (all(raw[k]==1 for k in ('auth_present','auth_valid','integrity_valid','source_authorized'))
        and not any(raw[k] for k in ('auth_unverified_flag','integrity_unverified_flag','source_unverified_flag','policy_inconsistent_flag'))
        and (not raw['session_required_flag'] or (raw['session_active'] and raw['session_valid']))
        and bool(raw['raw_freshness_passed']) and bool(raw['raw_command_age_valid']))

COMMAND_EVIDENCE_FIELDS=['raw_command_count','raw_message_type','raw_service_context','raw_command_intent',
 'raw_payload_or_parameter_context','sequence_gap','duplicate_flag','unknown_sequence_flag',
 'auth_present','auth_valid','integrity_valid','source_authorized','session_required_flag','session_active','session_valid',
 'session_expired_flag','policy_inconsistent_flag','raw_command_permitted_in_context','parameter_deviation_ratio',
 'accepted_flag',*COMMAND_OUTCOME_FLAGS.values(),'state_changed_flag','transition_allowed','unexpected_transition_flag',
 'forbidden_transition_flag','delayed_transition_flag','incomplete_transition_flag','raw_checks_passed',
 'raw_handler_dispatched','raw_security_gate_bypassed','raw_command_handler_action',
 'raw_authentication_result','raw_integrity_check_result','raw_authorization_result','raw_policy_check_result','raw_session_check_result',
 'raw_command_hash','raw_command_age_s','raw_command_age_valid','raw_freshness_passed','raw_freshness_check_result']

COMMAND_EVIDENCE_FIELDS=list(dict.fromkeys(COMMAND_EVIDENCE_FIELDS+list(EVENT_RAW_FIELDS.values())))

BASELINE_FIELDS=[f for f in NORMAL_RAW_RANGES if f not in {'session_age_s','command_rate_per_min','parameter_deviation_ratio'}]

BASELINE_FIELDS+=['fix_valid','gnss_lock_loss_count','gnss_cycle_slip_count','ttc_parse_error_count']

BASELINE_FIELDS+=[f'{p}_{k}_count' for p in ('ttc','isl','iot','uav') for k in ('crc_error','decode_fail','sync_loss','retry','timeout')]

OBSERVABLE_DOMAINS['command_intent'].update(TC_SHARED_COMMAND_INTENTS)

OBSERVABLE_DOMAINS['payload_or_parameter_context'].update(TC_SHARED_PARAMETER_CONTEXTS)

OBSERVABLE_DOMAINS['sequence_freshness_status'].update({'UNKNOWN','REPLAYED','STALE','DUPLICATE'})

OBSERVABLE_DOMAINS['control_security_state'].update({'AUTH_MISSING','AUTH_FAILED','INTEGRITY_FAILED','UNAUTHORIZED','AUTH_UNVERIFIED','INTEGRITY_UNVERIFIED'})

OBSERVABLE_DOMAINS['session_state'].update({'EXPIRED','ABNORMAL','LINK_LOSS'})

OBSERVABLE_DOMAINS['command_execution_result'].update({'BLOCKED','REJECTED'})

OBSERVABLE_DOMAINS['state_transition_consistency'].update({'UNEXPECTED_TRANSITION','FORBIDDEN_TRANSITION'})

RAW_FLAGS=['fix_valid','duplicate_flag','unknown_sequence_flag','accepted_sequence_flag','auth_present','auth_valid',
 'integrity_valid','source_authorized','auth_unverified_flag','integrity_unverified_flag','source_unverified_flag',
 'source_unverified_benign_flag','valid_looking_benign_flag','policy_inconsistent_flag','raw_command_permitted_in_context',
 'session_required_flag','session_active','session_valid','session_expired_flag','link_degraded_flag','link_loss_flag',
 'accepted_flag',*COMMAND_OUTCOME_FLAGS.values(),'state_changed_flag','transition_allowed','unexpected_transition_flag',
 'forbidden_transition_flag','delayed_transition_flag','incomplete_transition_flag','raw_checks_passed','raw_handler_dispatched',
 'raw_security_gate_bypassed','raw_command_age_valid','raw_freshness_passed','raw_replay_eligible','raw_attack_active',
 'raw_observed_replay','raw_original_command_seen_flag','raw_command_hash_reuse_flag','raw_reused_intent_flag','raw_reused_payload_flag',
 'raw_observation_on_target','raw_is_command_context','raw_receiver_reset_event','raw_receiver_session_change_event',
 'raw_receiver_context_change_event','raw_receiver_memory_available']

RAW_COUNTS=['raw_command_count','raw_nominal_command_count','raw_original_command_count','raw_replayed_command_count',
 'raw_attempted_replayed_command_count','raw_telemetry_count','raw_iot_report_count','raw_uav_message_count','raw_gnss_observation_count',
 'critical_command_count','gnss_lock_loss_count','gnss_cycle_slip_count','ttc_parse_error_count','session_reset_count',
 'raw_command_age_s','raw_original_command_age_s','raw_replay_interval_s','raw_recurrence_window_index','raw_receiver_session_id',
 'raw_receiver_context_epoch','raw_receiver_highest_sequence_before','raw_receiver_highest_sequence_after',
 'raw_sender_issued_sequence_before','raw_sender_issued_sequence_after','raw_command_sequence_id','raw_command_session_id',
 'raw_command_context_epoch','raw_command_ttl_s']

RAW_COUNTS+=[f'{p}_{k}_count' for p in ('ttc','isl','iot','uav') for k in ('crc_error','decode_fail','sync_loss','retry','timeout')]

RAW_COUNTS+=['raw_phase_boundary_'+k for k in PHASE_BOUNDARY_KEYS]

RAW_HISTORY_FIELDS=['raw_label_scope','raw_target_path','raw_target_protocol','raw_target_direction','raw_command_hash',
 'raw_original_command_hash','raw_original_parameter_profile','raw_representative_command_origin','raw_window_training_role',
 'raw_shared_context_profile','raw_shared_context_family','raw_previous_attempt_step']

RAW_HISTORY_FIELDS+=['raw_original_command_'+k for k in EVENT_KEYS]

def print_one_second_dataset_summary(df: pd.DataFrame) -> None:
    print(json.dumps({'rows':len(df),'episodes':int(df.episode_id.nunique()),'window_seconds':1}))

import shared_nominal as nominal
from shared_nominal import (normalize_context, normalize_sequence_freshness, normalize_control_security, normalize_message_control, normalize_signal_quality, normalize_tracking_stability, normalize_navigation_consistency, normalize_timing_consistency, normalize_decoder_status, normalize_command_execution, normalize_retry_timeout, normalize_session_state, normalize_state_transition, normalize_raw_to_observable)

# A replay copies the complete representative command properties observed at
# capture, including its legitimate envelope. Capture itself changes no field.
EVENT_KEYS=['id','sequence_id','session_id','context_epoch','created_step','window_id',
            'intent','payload','parameter_value','message_type','service_context']
EVENT_RAW_FIELDS={k:'raw_command_'+k for k in EVENT_KEYS}
EVENT_RAW_FIELDS.update(intent='raw_command_intent',payload='raw_payload_or_parameter_context',
                        message_type='raw_message_type',service_context='raw_service_context')


def make_new_command(raw,state,intent,payload):
    return {'id':f"{raw['episode_id']}/{raw['window_id']}/NEW",
            'sequence_id':max(state['issued'],state['highest'])+int(raw['sequence_gap']),
            'session_id':state['session'],'context_epoch':state['epoch'],
            'created_step':raw['episode_step'],'window_id':raw['window_id'],
            'intent':intent,'payload':payload,'parameter_value':raw['raw_command_parameter_value'],
            'message_type':raw['raw_message_type'],'service_context':raw['raw_service_context']}


def load_command_evidence(raw,command,state):
    raw.update({EVENT_RAW_FIELDS[k]:v for k,v in command.items()})
    raw['raw_command_hash']=command_fingerprint(command)
    raw['sequence_gap']=int(command['sequence_id']-state['highest'])
    available=bool(raw['raw_receiver_memory_available'])
    raw['duplicate_flag']=int(available and raw['raw_command_hash'] in state['seen'])
    raw['unknown_sequence_flag']=0  # The sequence high-water mark remains available.
    raw['session_valid']=int(command['session_id']==state['session'])
    raw['session_expired_flag']=1-raw['session_valid']
    raw['policy_inconsistent_flag']=int(command['context_epoch']!=state['epoch'])
    raw['raw_command_permitted_in_context']=1-raw['policy_inconsistent_flag']
    raw['parameter_deviation_ratio']=round(2*abs(command['parameter_value']-.5),6)
    raw['raw_command_age_s']=int(raw['episode_step']-command['created_step'])
    raw['raw_command_age_valid']=int(raw['raw_command_age_s']<=raw['raw_command_ttl_s'])
    raw['raw_freshness_check_result']=('REUSED' if raw['duplicate_flag'] else
                                      'PASS' if raw['sequence_gap']>0 else 'SEQUENCE_MISMATCH')
    raw['raw_freshness_passed']=int(raw['raw_freshness_check_result']=='PASS')


def generate_tc_replay_raw_window(window_id,timestamp,episode,window_seconds=DEFAULT_WINDOW_SECONDS,
                                 replay_state=None,*,nominal_raw=None,attacks_enabled=True):
    if replay_state is None:raise ValueError('Replay needs the episode receiver state from build_tc_replay_dataset.')
    state=replay_state
    reset,session_change,epoch_change=advance_receiver_state(state,episode) if attacks_enabled else (0,0,0)
    raw=nominal.copy_nominal_window(nominal_raw,episode,LABEL_TELECOMMAND_REPLAY)
    target=is_replay_target_context(episode)
    for f in ('replay_type','replay_case_type','replay_phase','replay_severity','enforcement_mode','memory_mode'):
        raw['raw_'+f]=episode[f]
    raw.update(raw_observation_on_target=int(target),raw_target_path=TARGET_PATH,raw_target_protocol=TARGET_PROTOCOL,
               raw_target_direction=TARGET_DIRECTION,raw_command_ttl_s=episode['command_ttl_s'],
               raw_receiver_memory_available=int(episode['memory_mode']!='UNAVAILABLE'),
               raw_duplicate_cache_available=int(episode['memory_mode']!='UNAVAILABLE'),
               raw_sequence_state_available=1,raw_sender_resynchronization_flag=int(reset or session_change),
               raw_receiver_memory_reset_flag=reset,raw_receiver_session_change_flag=session_change,
               raw_receiver_context_change_flag=epoch_change,raw_receiver_highest_sequence_before=state['highest'],
               raw_sender_issued_sequence_before=state['issued'],raw_receiver_session_id=state['session'],
               raw_receiver_context_epoch=state['epoch'],raw_command_age_s=0,raw_command_age_valid=1,
               raw_freshness_passed=int(raw['raw_command_count']>0),
               raw_freshness_check_result='PASS' if raw['raw_command_count'] else 'NOT_APPLICABLE',
               raw_security_gate_bypassed=0,raw_handler_dispatched=int(raw['accepted_flag']))
    for key in EVENT_KEYS:
        if key in {'intent','payload','message_type','service_context','parameter_value'}:continue
        raw[EVENT_RAW_FIELDS[key]]='NONE' if key=='id' else -1
    raw['raw_command_hash']='NONE';raw['raw_command_last_sequence_id']=-1
    command=None
    if raw['raw_command_count']:
        command=make_new_command(raw,state,raw['raw_command_intent'],raw['raw_payload_or_parameter_context'])
        if not target:command.update(sequence_id=int(raw['sequence_gap']),session_id=0,context_epoch=0)
        # Attach identity only. Normal checks/outcomes come from the substrate.
        raw.update({EVENT_RAW_FIELDS[k]:v for k,v in command.items()})
        raw['raw_command_hash']=command_fingerprint(command)
        raw['raw_command_last_sequence_id']=command['sequence_id']+raw['raw_command_count']-1
    nominal.record_nominal_checks(raw)
    raw['raw_command_handler_action']=normalize_command_execution(raw)
    eligible=bool(attacks_enabled and replay_eligible(state,episode))
    attempts=sample_replay_attempt_count(episode,eligible)
    observed=attempts if target else 0
    # Passive capture of an actually observed authorized, policy-valid record.
    # Neither intent, payload, envelope nor handler outcome is forced for capture.
    capture=bool(attacks_enabled and target and command is not None and state['original'] is None
                 and episode['replay_phase'] not in {'PRE_VALID_COMMAND_CONTEXT','CONTAINMENT_OR_RECOVERY'}
                 and raw['auth_present'] and raw['auth_valid'] and raw['integrity_valid']
                 and raw['source_authorized'] and raw['session_valid'] and not raw['policy_inconsistent_flag'])
    previous=state['previous_attempt_step']
    raw.update(raw_replay_eligible=int(eligible),raw_attempted_replayed_command_count=attempts,
               raw_replayed_command_count=observed,raw_attack_active=int(attempts>0),raw_observed_replay=int(observed>0),
               raw_original_command_count=int(capture),raw_previous_attempt_step=previous,
               raw_replay_interval_s=(episode['episode_step']-(previous if previous>=0 else state['original']['created_step'])) if attempts else 0,
               raw_representative_command_origin='NOMINAL' if command else 'NONE')
    if attempts:state['previous_attempt_step']=episode['episode_step']
    if capture:
        state['original']=dict(command);raw['raw_representative_command_origin']='ORIGINAL'
    if observed:
        nominal.begin_command_attack(raw)
        raw['raw_nominal_command_count']=0;raw['raw_command_count']=observed
        load_command_evidence(raw,dict(state['original']),state)
        raw['raw_command_last_sequence_id']=raw['raw_command_sequence_id']  # identical copies
        raw['raw_representative_command_origin']='REPLAY'
        handle_representative_command(raw)
        raw['critical_command_count']=observed if raw['raw_command_intent']=='MODE_CHANGE' else 0
    raw['command_rate_per_min']=raw['raw_command_count']*60
    raw['raw_replay_burst_rate']=observed*60
    if target and raw['raw_command_count']:
        if not observed:state['issued']=int(raw['raw_command_last_sequence_id'])
        if raw['session_valid']:
            state['highest']=max(state['highest'],int(raw['raw_command_last_sequence_id']))
            if raw['raw_duplicate_cache_available']:state['seen'].add(raw['raw_command_hash'])
    if observed:state['observed_replay_windows']+=1
    original=state['original']
    for key in EVENT_KEYS:
        default='NONE' if key in {'id','intent','payload','message_type','service_context'} else -1
        raw['raw_original_command_'+key]=original[key] if original else default
    raw['raw_original_command_hash']=command_fingerprint(original) if original else 'NONE'
    raw['raw_original_command_seen_flag']=int(original is not None)
    raw['raw_original_command_age_s']=episode['episode_step']-original['created_step'] if original else 0
    raw['raw_command_hash_reuse_flag']=int(observed>0)
    raw['raw_reused_intent_flag']=int(observed>0 and raw['raw_command_intent']==original['intent'])
    raw['raw_reused_payload_flag']=int(observed>0 and raw['raw_payload_or_parameter_context']==original['payload'])
    raw['raw_original_parameter_profile']=original['payload'] if original else 'NOMINAL_PARAMETER'
    raw['raw_receiver_highest_sequence_after']=state['highest'];raw['raw_sender_issued_sequence_after']=state['issued']
    raw['raw_recurrence_window_index']=state['observed_replay_windows']
    raw['raw_window_training_role']=('OBSERVED_REPLAY' if observed else 'ORIGINAL_OBSERVED' if capture else
                                     'UNOBSERVED_ATTEMPT' if attempts else 'CONTEXT_ONLY')
    raw['raw_policy_violation_flag']=int(raw['policy_inconsistent_flag'])
    return raw


def validate_attack_rows(df):
    nominal.require_nominal_when(df,df.raw_replayed_command_count.eq(0))
    nominal.validate_observed_command_gate(df,'raw_replayed_command_count')
    target=df.raw_observation_on_target.eq(1)&df.raw_command_count.gt(0)
    gap=df.raw_command_sequence_id-df.raw_receiver_highest_sequence_before
    if not df.loc[target,'sequence_gap'].eq(gap[target]).all():
        raise ValueError('Sequence gap contradicts receiver history.')
    for _,episode in df.groupby('episode_id',sort=False):
        records={int(r['window_id']):r for r in episode.to_dict('records')}
        for row in records.values():
            if not row['raw_replayed_command_count']:continue
            original=records.get(int(row['raw_original_command_window_id']))
            if original is None or original['window_id']>=row['window_id']:raise ValueError('Replay lacks an earlier observed original.')
            if original['raw_command_hash']!=row['raw_command_hash']:raise ValueError('Replay did not reuse the observed record.')
            for key,field in EVENT_RAW_FIELDS.items():
                if row[field]!=original[field]:raise ValueError(f'Replay modified the captured command property: {key}')
            if row['raw_command_age_s']!=row['episode_step']-original['episode_step']:raise ValueError('Invalid replay age.')
    if (df.raw_replayed_command_count.gt(0)&df.raw_observation_on_target.eq(0)).any():raise ValueError('Off-target replay evidence.')

MAKER_NUMBER = 5
CLASS_LABEL = LABEL_TELECOMMAND_REPLAY
CODE_VERSION = nominal.CODE_VERSION
TIMELINE_BUILDER = 'build_tc_replay_episode_timeline'
RAW_GENERATOR = 'generate_tc_replay_raw_window'
OBSERVABLE_DOMAINS = nominal.OBSERVABLE_DOMAINS

def build_tc_replay_dataset(num_rows: int = DEFAULT_NUM_ROWS, window_seconds: int = DEFAULT_WINDOW_SECONDS,
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
    nominal.run_cli(globals(), 'build_tc_replay_dataset')
