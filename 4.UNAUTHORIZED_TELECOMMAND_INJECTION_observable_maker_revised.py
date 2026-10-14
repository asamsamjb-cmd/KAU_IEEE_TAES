# Revision: common nominal background + shared observation mapping.
# Research objective: reviewer revision on synthetic-rule dependence and temporal evidence.
"""Controlled synthetic telecommand-injection episodes, revised 2026-10-02.

One row is a one-second selected observation context. The 22 observable fields
are model input; all other fields are metadata or raw audit evidence. Injection
targets ground-to-space CCSDS TT&C only. Attempts on an unobserved target are
retained in audit counts, never inserted into GNSS/IoT/UAV/downlink messages.

This is a NEW data version, not a reproduction of historical A/A-prime scores.
Numerical distributions, case priors and enforcement modes are assumptions,
not calibrated probabilities or a cFS implementation. Message/check/handler
records are simulated categorical evidence, not independently measured sensors.
Labels describe whole episodes, including quiet, pre and post-attempt windows.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pathlib import Path

from typing import Any, Dict, List

import hashlib

import json

import platform

import zipfile

import os

import tempfile

import csv

import io

import numpy as np

import pandas as pd

import shared_context_v4 as shared_context_module

from shared_context_v4 import (
    CONTEXT_FIELD_NAMES, GLOBAL_CONTEXT_PROFILE_POOL, is_command_context,
    nominal_semantics_for_context, sample_shared_context_profile,
    validate_context_profile, validate_global_context_pool,
)

CODE_VERSION = "injection-consistent-events-v1-20261002"

LABEL_SCOPE = "EPISODE_SCENARIO"

ENFORCEMENT_MODES = ["FAIL_CLOSED", "AUDIT_ONLY"]

ENFORCEMENT_MODE_PROBS = [0.80, 0.20]

TARGET_PATH = "GATEWAY_LEO"

TARGET_PROTOCOL = "CCSDS_TTC"

TARGET_DIRECTION = "GROUND_TO_SPACE"

PHASE_BOUNDARY_KEYS = ("pre_end", "attempt_end", "security_end", "response_end", "impact_end")

STATE_CHANGING_INTENTS = {"CONFIG_UPDATE", "NOMINAL_CONTROL", "ORBIT_CONTROL",
    "TIME_SYNC", "PAYLOAD_CONFIG", "ATTITUDE_ADJUST", "ROUTE_UPDATE", "MODE_CHANGE"}

DEFAULT_NUM_ROWS = 50000

DEFAULT_WINDOW_SECONDS = 1

DEFAULT_OUTPUT_PATH = "dataset/unauthorized_telecommand_injection_observable_dataset.csv"

DEFAULT_RAW_AUDIT_OUTPUT_PATH = "dataset/unauthorized_telecommand_injection_observable_dataset_with_raw.csv"

DEFAULT_SAMPLE_TIMELINE_OUTPUT_PATH = "dataset/unauthorized_telecommand_injection_sample_episode_timeline.csv"

DEFAULT_RANDOM_SEED = 314

RNG = np.random.default_rng(seed=DEFAULT_RANDOM_SEED)

LABEL_UNAUTHORIZED_TELECOMMAND_INJECTION = "UNAUTHORIZED_TELECOMMAND_INJECTION"

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

TC_INJECTION_EPISODE_TYPES = [
    "TTC_COMMAND_INJECTION",
    "UNAUTHORIZED_SOURCE_INJECTION",
    "AUTH_INTEGRITY_FAILURE_INJECTION",
    "OUT_OF_POLICY_COMMAND_INJECTION",
    "SESSIONLESS_COMMAND_INJECTION",
    "MULTI_STAGE_COMMAND_INJECTION",
]

TC_INJECTION_EPISODE_TYPE_PROBS = [
    0.30,
    0.20,
    0.20,
    0.15,
    0.10,
    0.05,
]

TC_INJECTION_TYPES = [
    "UNAUTHORIZED_SOURCE_COMMAND",
    "AUTHENTICATION_BYPASS_ATTEMPT",
    "INTEGRITY_VIOLATION_COMMAND",
    "OUT_OF_POLICY_COMMAND",
    "CRITICAL_MODE_CHANGE_COMMAND",
    "PARAMETER_TAMPERING_COMMAND",
    "SESSIONLESS_COMMAND_INSERTION",
]

TC_INJECTION_TYPE_PROBS_BY_EPISODE_TYPE: Dict[str, List[float]] = {
    "TTC_COMMAND_INJECTION": [0.18, 0.18, 0.14, 0.16, 0.14, 0.12, 0.08],
    "UNAUTHORIZED_SOURCE_INJECTION": [0.55, 0.15, 0.08, 0.08, 0.05, 0.04, 0.05],
    "AUTH_INTEGRITY_FAILURE_INJECTION": [0.08, 0.36, 0.34, 0.08, 0.04, 0.06, 0.04],
    "OUT_OF_POLICY_COMMAND_INJECTION": [0.08, 0.08, 0.06, 0.48, 0.16, 0.10, 0.04],
    "SESSIONLESS_COMMAND_INJECTION": [0.12, 0.12, 0.08, 0.06, 0.04, 0.05, 0.53],
    "MULTI_STAGE_COMMAND_INJECTION": [0.20, 0.18, 0.18, 0.14, 0.12, 0.10, 0.08],
}

TC_INJECTION_PHASES = [
    "PRE_COMMAND_CONTEXT",
    "INJECTION_ATTEMPT",
    "SECURITY_CHECK_FAILURE",
    "COMMAND_HANDLER_RESPONSE",
    "OPERATIONAL_IMPACT",
    "CONTAINMENT_OR_RECOVERY",
]

TC_INJECTION_CASE_TYPES = [
    "EASY_AUTH_FAILURE",
    "EASY_INTEGRITY_FAILURE",
    "EASY_UNAUTHORIZED_SOURCE",
    "HARD_VALID_LOOKING_CONTEXT_MISMATCH",
    "HARD_SESSIONLESS_VALID_COMMAND",
    "HARD_SOURCE_PATH_SEMANTIC_MISMATCH",
    "BENIGN_OVERLAP_COMMAND_HOLD",
    "PERSISTENT_SEMANTIC_DRIFT",
]

TC_INJECTION_CASE_TYPE_PROBS = [
    # Legacy relative weights, renormalized over cases compatible with the
    # selected injection type. These are not final episode probabilities and
    # do not establish sparsity or classifier difficulty.
    0.020,  # EASY_AUTH_FAILURE
    0.020,  # EASY_INTEGRITY_FAILURE
    0.025,  # EASY_UNAUTHORIZED_SOURCE
    0.300,  # HARD_VALID_LOOKING_CONTEXT_MISMATCH
    0.200,  # HARD_SESSIONLESS_VALID_COMMAND
    0.165,  # HARD_SOURCE_PATH_SEMANTIC_MISMATCH
    0.180,  # BENIGN_OVERLAP_COMMAND_HOLD
    0.090,  # PERSISTENT_SEMANTIC_DRIFT
]

EASY_INJECTION_CASE_TYPES = {
    "EASY_AUTH_FAILURE",
    "EASY_INTEGRITY_FAILURE",
    "EASY_UNAUTHORIZED_SOURCE",
}

HARD_INJECTION_CASE_TYPES = {
    "HARD_VALID_LOOKING_CONTEXT_MISMATCH",
    "HARD_SESSIONLESS_VALID_COMMAND",
    "HARD_SOURCE_PATH_SEMANTIC_MISMATCH",
    "BENIGN_OVERLAP_COMMAND_HOLD",
    "PERSISTENT_SEMANTIC_DRIFT",
}

VALID_LOOKING_HARD_CASE_TYPES = {
    "HARD_VALID_LOOKING_CONTEXT_MISMATCH",
    "HARD_SESSIONLESS_VALID_COMMAND",
    "HARD_SOURCE_PATH_SEMANTIC_MISMATCH",
    "BENIGN_OVERLAP_COMMAND_HOLD",
    "PERSISTENT_SEMANTIC_DRIFT",
}

COMMON_EPISODE_LENGTH_BUCKETS = ["SHORT", "MEDIUM", "LONG"]

COMMON_EPISODE_LENGTH_PROBS = [0.35, 0.45, 0.20]

COMMON_EPISODE_LENGTH_RANGES = {
    "SHORT": (20, 40),
    "MEDIUM": (40, 80),
    "LONG": (80, 140),
}

TC_SHARED_SOURCE_ROLES = [
    "GROUND_GATEWAY",
    "GCS_OPERATOR",
    "GROUND_CONTROL_SERVICE",
    "MAINTENANCE_OPERATOR",
    "MULTIPLE_AUTHORIZED_SOURCES",
]

TC_SHARED_CONTEXT_INTENTS = [
    "STATUS_REQUEST",
    "TIME_SYNC",
    "PAYLOAD_CONFIG",
    "ATTITUDE_ADJUST",
    "ROUTE_UPDATE",
    "MODE_CHANGE",
    "NOMINAL_CONTROL",
    "HOUSEKEEPING",
 ]

TC_SHARED_PAYLOAD_CONTEXTS = [
    "NOMINAL_PARAMETER",
    "CONFIG_PARAMETER",
    "CONTROL_PARAMETER",
    "ROUTINE_PARAMETER",
    "STATUS_PARAMETER",
    "TIME_SYNC_PARAMETER",
    "SCHEDULED_CONFIG_PARAMETER",
    "POLICY_SYNC_CONTEXT",
]

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

def choose_common_episode_duration() -> tuple[str, int]:
    """Sample an episode length from the cross-maker common distribution."""
    bucket = weighted_choice(COMMON_EPISODE_LENGTH_BUCKETS, COMMON_EPISODE_LENGTH_PROBS)
    low, high = COMMON_EPISODE_LENGTH_RANGES[bucket]
    return bucket, int(RNG.integers(low, high + 1))

def choose_injection_phase_boundaries(case_type: str) -> Dict[str, float]:
    """Sample ordered episode phase boundaries from the retained synthetic prior.

    Phase jitter avoids one fixed template; it does not prove model-independent
    difficulty or calibrated temporal dynamics.
    """
    # Base cumulative boundaries roughly preserve the previous phase structure.
    pre_end = float(np.clip(RNG.normal(0.10, 0.025), 0.05, 0.16))
    attempt_end = float(np.clip(RNG.normal(0.30, 0.045), pre_end + 0.10, 0.42))
    security_end = float(np.clip(RNG.normal(0.55, 0.055), attempt_end + 0.12, 0.67))
    response_end = float(np.clip(RNG.normal(0.74, 0.045), security_end + 0.08, 0.84))
    impact_end = float(np.clip(RNG.normal(0.90, 0.030), response_end + 0.06, 0.96))

    # Case-specific jitter changes where the evidence emerges without changing
    # the learning-visible schema.
    if case_type == "BENIGN_OVERLAP_COMMAND_HOLD":
        security_end = min(0.70, security_end + float(RNG.uniform(0.02, 0.06)))
        response_end = min(0.86, response_end + float(RNG.uniform(0.01, 0.04)))
    elif case_type == "PERSISTENT_SEMANTIC_DRIFT":
        pre_end = max(0.06, pre_end - float(RNG.uniform(0.00, 0.03)))
        attempt_end = max(pre_end + 0.12, attempt_end - float(RNG.uniform(0.00, 0.04)))
        impact_end = min(0.96, impact_end + float(RNG.uniform(0.01, 0.04)))
    elif case_type == "HARD_SESSIONLESS_VALID_COMMAND":
        attempt_end = min(0.45, attempt_end + float(RNG.uniform(0.00, 0.05)))
    elif case_type in EASY_INJECTION_CASE_TYPES:
        security_end = max(attempt_end + 0.10, security_end - float(RNG.uniform(0.02, 0.05)))

    values = [pre_end, attempt_end, security_end, response_end, impact_end]
    # Enforce monotonic margins after case-specific perturbation.
    min_gaps = [0.09, 0.11, 0.08, 0.06]
    for idx, gap in enumerate(min_gaps, start=1):
        if values[idx] <= values[idx - 1] + gap:
            values[idx] = values[idx - 1] + gap
    values[-1] = min(values[-1], 0.97)

    return {
        "pre_end": round(values[0], 4),
        "attempt_end": round(values[1], 4),
        "security_end": round(values[2], 4),
        "response_end": round(values[3], 4),
        "impact_end": round(values[4], 4),
    }

def get_tc_injection_phase(
    step: int,
    duration: int,
    phase_boundaries: Dict[str, float] | None = None,
) -> str:
    ratio = step / max(duration - 1, 1)
    bounds = phase_boundaries or {
        "pre_end": 0.10,
        "attempt_end": 0.30,
        "security_end": 0.55,
        "response_end": 0.75,
        "impact_end": 0.90,
    }

    if ratio < bounds["pre_end"]:
        return "PRE_COMMAND_CONTEXT"
    if ratio < bounds["attempt_end"]:
        return "INJECTION_ATTEMPT"
    if ratio < bounds["security_end"]:
        return "SECURITY_CHECK_FAILURE"
    if ratio < bounds["response_end"]:
        return "COMMAND_HANDLER_RESPONSE"
    if ratio < bounds["impact_end"]:
        return "OPERATIONAL_IMPACT"
    return "CONTAINMENT_OR_RECOVERY"

def phase_to_injection_severity(phase: str, injection_type: str, case_type: str = "") -> str:
    # Metadata stratum only; it cannot override a check or handler outcome.
    if phase in {"PRE_COMMAND_CONTEXT", "CONTAINMENT_OR_RECOVERY"}:
        return "NONE"
    if case_type in ["BENIGN_OVERLAP_COMMAND_HOLD", "HARD_VALID_LOOKING_CONTEXT_MISMATCH", "HARD_SESSIONLESS_VALID_COMMAND"]:
        if phase == "PRE_COMMAND_CONTEXT":
            return weighted_choice(["LOW", "MODERATE"], [0.94, 0.06])
        if phase == "INJECTION_ATTEMPT":
            return weighted_choice(["LOW", "MODERATE"], [0.52, 0.48])
        if phase in ["SECURITY_CHECK_FAILURE", "COMMAND_HANDLER_RESPONSE"]:
            return weighted_choice(["LOW", "MODERATE", "HIGH"], [0.24, 0.62, 0.14])
        if phase == "OPERATIONAL_IMPACT":
            return weighted_choice(["MODERATE", "HIGH"], [0.68, 0.32])
        if phase == "CONTAINMENT_OR_RECOVERY":
            return weighted_choice(["LOW", "MODERATE"], [0.46, 0.54])

    if case_type == "PERSISTENT_SEMANTIC_DRIFT":
        if phase == "PRE_COMMAND_CONTEXT":
            return weighted_choice(["LOW", "MODERATE"], [0.92, 0.08])
        if phase == "INJECTION_ATTEMPT":
            return weighted_choice(["LOW", "MODERATE"], [0.40, 0.60])
        if phase in ["SECURITY_CHECK_FAILURE", "COMMAND_HANDLER_RESPONSE", "OPERATIONAL_IMPACT"]:
            return weighted_choice(["MODERATE", "HIGH"], [0.66, 0.34])
        if phase == "CONTAINMENT_OR_RECOVERY":
            return weighted_choice(["LOW", "MODERATE", "HIGH"], [0.30, 0.62, 0.08])

    if phase == "PRE_COMMAND_CONTEXT":
        return weighted_choice(["LOW", "MODERATE"], [0.90, 0.10])

    if phase == "INJECTION_ATTEMPT":
        return weighted_choice(["LOW", "MODERATE", "HIGH"], [0.25, 0.60, 0.15])

    if phase == "SECURITY_CHECK_FAILURE":
        return weighted_choice(["MODERATE", "HIGH", "CRITICAL"], [0.35, 0.55, 0.10])

    if phase == "COMMAND_HANDLER_RESPONSE":
        return weighted_choice(["MODERATE", "HIGH", "CRITICAL"], [0.45, 0.45, 0.10])

    if phase == "OPERATIONAL_IMPACT":
        if injection_type in ["CRITICAL_MODE_CHANGE_COMMAND", "PARAMETER_TAMPERING_COMMAND"]:
            return weighted_choice(["HIGH", "CRITICAL"], [0.60, 0.40])
        return weighted_choice(["MODERATE", "HIGH", "CRITICAL"], [0.25, 0.55, 0.20])

    if phase == "CONTAINMENT_OR_RECOVERY":
        return weighted_choice(["LOW", "MODERATE", "HIGH"], [0.35, 0.55, 0.10])

    return "MODERATE"

def choose_injection_type(episode_type: str) -> str:
    return weighted_choice(
        TC_INJECTION_TYPES,
        TC_INJECTION_TYPE_PROBS_BY_EPISODE_TYPE[episode_type],
    )

def choose_episode_source_profile(case_type: str, episode_type: str) -> str:
    """Retain a claimed-source descriptor in audit metadata only.

    It does not overwrite the shared observation source_role or establish
    sender authorization. Actual primitive authorization is generated by type.
    """
    roles = TC_SHARED_SOURCE_ROLES

    if case_type == "HARD_SOURCE_PATH_SEMANTIC_MISMATCH":
        return weighted_choice(roles, [0.50, 0.16, 0.13, 0.10, 0.11])

    if case_type == "HARD_SESSIONLESS_VALID_COMMAND":
        return weighted_choice(roles, [0.62, 0.14, 0.10, 0.09, 0.05])

    if case_type in ["HARD_VALID_LOOKING_CONTEXT_MISMATCH", "BENIGN_OVERLAP_COMMAND_HOLD", "PERSISTENT_SEMANTIC_DRIFT"]:
        return weighted_choice(roles, [0.64, 0.13, 0.09, 0.08, 0.06])

    if episode_type == "UNAUTHORIZED_SOURCE_INJECTION":
        return weighted_choice(roles, [0.50, 0.18, 0.13, 0.10, 0.09])

    return weighted_choice(roles, [0.64, 0.13, 0.09, 0.08, 0.06])

def choose_episode_command_family(injection_type: str, case_type: str) -> str:
    """Choose a repeated command family, not a repeated message identity."""
    common_intents = TC_SHARED_CONTEXT_INTENTS

    if injection_type == "CRITICAL_MODE_CHANGE_COMMAND":
        return "MODE_CHANGE"
    if injection_type == "PARAMETER_TAMPERING_COMMAND":
        return "PAYLOAD_CONFIG"
    if injection_type == "OUT_OF_POLICY_COMMAND":
        return weighted_choice(common_intents, [0.08, 0.08, 0.24, 0.16, 0.18, 0.16, 0.06, 0.04])
    if case_type == "BENIGN_OVERLAP_COMMAND_HOLD":
        return weighted_choice(common_intents, [0.24, 0.22, 0.16, 0.12, 0.09, 0.07, 0.05, 0.05])
    return weighted_choice(common_intents, [0.14, 0.14, 0.22, 0.16, 0.16, 0.12, 0.04, 0.02])

def choose_episode_payload_profile(command_family: str, case_type: str) -> str:
    """Choose a repeated categorical message property, not a numeric measurement."""
    if command_family == "TIME_SYNC":
        return weighted_choice(["TIME_SYNC_PARAMETER", "NOMINAL_PARAMETER", "STATUS_PARAMETER"], [0.58, 0.30, 0.12])
    if command_family in ["PAYLOAD_CONFIG", "ROUTE_UPDATE", "MODE_CHANGE"]:
        return weighted_choice(["CONFIG_PARAMETER", "CONTROL_PARAMETER", "NOMINAL_PARAMETER", "ROUTINE_PARAMETER"], [0.45, 0.26, 0.20, 0.09])
    if command_family in ["STATUS_REQUEST", "HOUSEKEEPING"]:
        return weighted_choice(["STATUS_PARAMETER", "NOMINAL_PARAMETER", "ROUTINE_PARAMETER"], [0.44, 0.42, 0.14])
    if case_type == "BENIGN_OVERLAP_COMMAND_HOLD":
        return weighted_choice(["NOMINAL_PARAMETER", "CONFIG_PARAMETER", "ROUTINE_PARAMETER"], [0.50, 0.32, 0.18])
    return weighted_choice(["NOMINAL_PARAMETER", "CONFIG_PARAMETER", "CONTROL_PARAMETER", "ROUTINE_PARAMETER"], [0.42, 0.32, 0.20, 0.06])

def choose_injection_case_type(episode_type: str, injection_type: str) -> str:
    """Choose a compatible mixture stratum; never substitute for check evidence.

    Renormalized legacy weights keep explicit failure cases attached to their
    matching violation. 'Hard' remains a legacy stratum name, not a measured
    classification difficulty or a requirement to hide actual failures.
    """
    options = {
        "AUTHENTICATION_BYPASS_ATTEMPT": ["EASY_AUTH_FAILURE"],
        "INTEGRITY_VIOLATION_COMMAND": ["EASY_INTEGRITY_FAILURE"],
        "UNAUTHORIZED_SOURCE_COMMAND": ["EASY_UNAUTHORIZED_SOURCE", "HARD_SOURCE_PATH_SEMANTIC_MISMATCH"],
        "SESSIONLESS_COMMAND_INSERTION": ["HARD_SESSIONLESS_VALID_COMMAND"],
    }.get(injection_type, ["HARD_VALID_LOOKING_CONTEXT_MISMATCH", "BENIGN_OVERLAP_COMMAND_HOLD", "PERSISTENT_SEMANTIC_DRIFT"])
    weights = [TC_INJECTION_CASE_TYPE_PROBS[TC_INJECTION_CASE_TYPES.index(c)] for c in options]
    return weighted_choice(options, weights)

def temporal_injection_pressure_score(episode: Dict[str, Any]) -> float:
    """Legacy audit index, not physical pressure or a detection score."""
    return {"PRE_COMMAND_CONTEXT":0., "INJECTION_ATTEMPT":0.2,
            "SECURITY_CHECK_FAILURE":0.4, "COMMAND_HANDLER_RESPONSE":0.6,
            "OPERATIONAL_IMPACT":0.8, "CONTAINMENT_OR_RECOVERY":0.}.get(episode["injection_phase"],0.)

def build_tc_injection_episode_timeline(
    num_rows: int,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    *, nominal_timeline: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build synthetic phase strata and iid shared observation contexts.

    Context draws do not receive labels, case names or severity. A shared
    sampler alone does not establish absence of joint-distribution shortcuts.
    Phase names are retained for compatibility; actual check/handler records
    determine whether a failure, execution, or state change occurred.
    """
    validate_generation_args(num_rows, window_seconds)
    timeline: List[Dict[str, Any]] = []
    episode_index = 0
    window_id = 0

    validate_global_context_pool()

    while window_id < num_rows:
        episode_type = weighted_choice(
            TC_INJECTION_EPISODE_TYPES,
            TC_INJECTION_EPISODE_TYPE_PROBS,
        )
        injection_type = choose_injection_type(episode_type)
        injection_case_type = choose_injection_case_type(episode_type, injection_type)

        episode_length_type, duration = nominal_timeline[window_id]["episode_length_bucket"], nominal_timeline[window_id]["episode_length"]

        # Raw/audit temporal recurrence controls.  These do not set any of the
        # seven Observation Context fields in A-prime.
        episode_source_profile = choose_episode_source_profile(injection_case_type, episode_type)
        episode_command_family = choose_episode_command_family(injection_type, injection_case_type)
        episode_payload_profile = choose_episode_payload_profile(episode_command_family, injection_case_type)
        phase_boundaries = choose_injection_phase_boundaries(injection_case_type)

        enforcement_mode = weighted_choice(ENFORCEMENT_MODES, ENFORCEMENT_MODE_PROBS)
        episode_id = f"TCINJ_EP_{episode_index:05d}"

        for step in range(duration):
            if window_id >= num_rows:
                break

            phase = get_tc_injection_phase(step, duration, phase_boundaries)
            severity = phase_to_injection_severity(phase, injection_type, injection_case_type)

            # Benchmark A-prime: all seven context fields come from the same
            # label-agnostic global pool used by NORMAL/RF/GNSS/Ground-TT&C.
            # No label, attack type, case, phase, or severity is passed here.
            context_profile = nominal_timeline[window_id]
            validate_context_profile(context_profile)
            semantic_defaults = {f: context_profile[f] for f in ("message_type", "service_context", "command_intent")}

            episode = {
                "scenario_id": "UNAUTHORIZED_TELECOMMAND_INJECTION_SCENARIO_0001",
                "episode_id": episode_id,
                "episode_type": episode_type,
                "enforcement_mode": enforcement_mode,
                "episode_step": step,
                "episode_length": duration,
                "episode_length_type": episode_length_type,
                "phase_boundaries": phase_boundaries,
                "elapsed_seconds": step * window_seconds,
                "episode_progress": step / max(duration - 1, 1),
                "injection_type": injection_type,
                "injection_case_type": injection_case_type,
                "injection_phase": phase,
                "injection_severity": severity,
                "episode_source_profile": episode_source_profile,
                "episode_command_family": episode_command_family,
                "episode_payload_profile": episode_payload_profile,
                "raw_temporal_pressure": temporal_injection_pressure_score({
                    "injection_phase": phase,
                    "injection_case_type": injection_case_type,
                }),
                "shared_context_profile": context_profile.get("shared_context_profile", "UNKNOWN"),
                "shared_context_family": context_profile.get("shared_context_family", "UNKNOWN"),
                "is_command_context": int(is_command_context(context_profile)),
                "nominal_message_type": str(semantic_defaults.get("message_type", "NOMINAL_MIXED_TRAFFIC")),
                "nominal_service_context": str(semantic_defaults.get("service_context", "HOUSEKEEPING")),
                "nominal_command_intent": str(semantic_defaults.get("command_intent", "NO_COMMAND")),
                **{field: context_profile[field] for field in CONTEXT_FIELD_NAMES},
            }

            timeline.append(episode)
            window_id += 1

        episode_index += 1

    return timeline

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

def has_command_activity(raw: Dict[str, Any]) -> bool:
    return int(raw.get("raw_command_count", 0)) > 0

def validate_generation_args(num_rows: int, window_seconds: int) -> None:
    if isinstance(num_rows, (bool, np.bool_)) or not isinstance(num_rows, (int, np.integer)) or num_rows <= 0:
        raise ValueError("num_rows must be a positive integer.")
    if isinstance(window_seconds, (bool, np.bool_)) or window_seconds != 1:
        raise ValueError("Only one-second windows are supported by this maker.")

def is_injection_target_context(context: Dict[str, Any]) -> bool:
    """TT&C uplink observations can carry a command even outside COMMAND_SESSION."""
    return (context['path_type'] == TARGET_PATH and context['protocol_context'] == TARGET_PROTOCOL
            and context['direction'] == TARGET_DIRECTION and context['destination_role'] == 'MONITORED_LEO')

def sample_attempt_count(episode: Dict[str, Any]) -> int:
    """Off-target attempts exist in audit independently of observation selection."""
    phase = episode['injection_phase']
    if phase in {'PRE_COMMAND_CONTEXT', 'CONTAINMENT_OR_RECOVERY'}:
        return 0
    quiet = {'INJECTION_ATTEMPT':.06, 'SECURITY_CHECK_FAILURE':.08,
             'COMMAND_HANDLER_RESPONSE':.10, 'OPERATIONAL_IMPACT':.12}[phase]
    case = episode['injection_case_type']
    quiet += .06 if case == 'BENIGN_OVERLAP_COMMAND_HOLD' else .03 if case == 'PERSISTENT_SEMANTIC_DRIFT' else 0.
    if RNG.random() < quiet:
        return 0
    return count_choice([1,2,3,4], [.62,.28,.08,.02])

def command_checks_pass(raw: Dict[str, Any]) -> bool:
    """Synthetic local check decision, independent of episode label and phase."""
    return (all(raw[k] == 1 for k in ('auth_present','auth_valid','integrity_valid','source_authorized'))
            and all(raw[k] == 0 for k in ('auth_unverified_flag','integrity_unverified_flag',
                                         'source_unverified_flag','policy_inconsistent_flag'))
            and (raw['session_required_flag'] == 0 or (raw['session_active'] == 1 and raw['session_valid'] == 1)))

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

def handle_representative_command(raw: Dict[str, Any]) -> None:
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
        outcome = weighted_choice(['EXECUTED','HELD','ACCEPTED'], [.75,.15,.10])
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

def apply_telecommand_injection_effects(raw: Dict[str, Any], episode: Dict[str, Any]) -> None:
    """Generate observed command evidence, then check/handle it causally.

    RF/GNSS measurements and other paths are left at their saved baseline.
    No numeric footprint is reverse-generated from desired observable tokens.
    """
    attempts = sample_attempt_count(episode) if episode.get("_attacks_enabled",True) else 0
    on_target = is_injection_target_context(episode)
    observed = attempts if on_target else 0
    raw['raw_attempted_injected_command_count'] = attempts
    raw['raw_injected_command_count'] = observed
    raw['raw_attack_active'] = int(attempts > 0)
    raw['raw_observation_on_target'] = int(on_target)
    raw['raw_observed_injection'] = int(observed > 0)
    raw['raw_command_count'] = raw['raw_nominal_command_count'] + observed
    raw['raw_command_burst_rate'] = float(observed * 60)
    raw['command_rate_per_min'] = float(raw['raw_command_count'] * 60)
    raw['raw_representative_command_origin'] = 'INJECTED' if observed else 'NOMINAL' if raw['raw_command_count'] else 'NONE'
    raw['raw_command_id'] = (f"{raw['episode_id']}/{raw['window_id']}/"
                             f"{raw['raw_representative_command_origin']}/0" if raw['raw_command_count'] else 'NONE')
    # Every attempt is a newly constructed message. Repeated command families
    # are not evidence of replay; no old-command history is claimed here.
    raw['raw_window_training_role'] = ('OBSERVED_INJECTION' if observed else
        'UNOBSERVED_ATTEMPT' if attempts else 'CONTEXT_ONLY')
    if not observed:
        return
    nominal.begin_command_attack(raw)
    kind = episode['injection_type']
    # The target is the ground uplink; source_role remains its observed/claimed
    # role. A claimed ground identity does not establish authorization.
    raw['raw_message_type'] = 'TELECOMMAND'
    raw['raw_service_context'] = 'TTC_CONTROL'
    intent = episode['episode_command_family']
    raw['raw_command_intent'] = intent
    raw['raw_payload_or_parameter_context'] = episode['episode_payload_profile']
    # Primitive checks, generated before either category mapping or dispatch.
    raw['session_active'] = 1
    raw['session_valid'] = 1
    raw['session_required_flag'] = 1
    if kind == 'UNAUTHORIZED_SOURCE_COMMAND':
        raw['source_authorized'] = 0
    elif kind == 'AUTHENTICATION_BYPASS_ATTEMPT':
        raw['auth_present'] = flag_choice(.65)
        raw['auth_valid'] = 0
    elif kind == 'INTEGRITY_VIOLATION_COMMAND':
        raw['integrity_valid'] = 0
    elif kind in {'OUT_OF_POLICY_COMMAND','CRITICAL_MODE_CHANGE_COMMAND'}:
        raw['policy_inconsistent_flag'] = 1
        raw['raw_command_permitted_in_context'] = 0
    elif kind == 'PARAMETER_TAMPERING_COMMAND':
        raw['parameter_deviation_ratio'] = uniform_float(.20,.80,4)
        raw['raw_command_parameter_value'] = round(.5 + raw['parameter_deviation_ratio']/2,6)
        raw['policy_inconsistent_flag'] = int(raw['parameter_deviation_ratio'] > raw['raw_parameter_limit_ratio'])
        raw['raw_command_permitted_in_context'] = int(not raw['policy_inconsistent_flag'])
        raw['raw_payload_or_parameter_context'] = 'PARAMETER_RANGE_VIOLATION'
    elif kind == 'SESSIONLESS_COMMAND_INSERTION':
        raw['session_active'] = 0
    else:
        raise ValueError(f'Unsupported injection type: {kind}')
    raw['raw_policy_violation_flag'] = raw['policy_inconsistent_flag']
    raw['raw_critical_command_flag'] = int(intent == 'MODE_CHANGE')
    raw['critical_command_count'] = observed if intent == 'MODE_CHANGE' else 0
    handle_representative_command(raw)

COMMAND_EVIDENCE_FIELDS = [
    'raw_command_count','raw_message_type','raw_service_context','raw_command_intent','raw_payload_or_parameter_context',
    'auth_present','auth_valid','integrity_valid','source_authorized','policy_inconsistent_flag',
    'session_required_flag','session_active','session_valid','parameter_deviation_ratio',
    'raw_command_permitted_in_context','accepted_flag',*COMMAND_OUTCOME_FLAGS.values(),
    'transition_allowed','state_changed_flag','unexpected_transition_flag','forbidden_transition_flag',
    'delayed_transition_flag','incomplete_transition_flag','raw_checks_passed','raw_handler_dispatched',
    'raw_security_gate_bypassed', 'raw_command_handler_action',
    'raw_authentication_result','raw_integrity_check_result','raw_authorization_result',
    'raw_policy_check_result','raw_session_check_result',
]

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

OBSERVABLE_DOMAINS["session_state"].add("LINK_LOSS")

_CANONICAL_LEARNING_READY_COLUMNS = METADATA_FIELDS + OBSERVABLE_FIELDS

RAW_COUNT_FIELDS = [
    "raw_command_count", "raw_telemetry_count", "raw_iot_report_count",
    "raw_uav_message_count", "raw_gnss_observation_count", "critical_command_count",
    "gnss_lock_loss_count", "gnss_cycle_slip_count", "session_reset_count",
    "ttc_parse_error_count",
] + [f"{path}_{metric}_count" for path in ("ttc", "isl", "iot", "uav")
     for metric in ("crc_error", "decode_fail", "retry", "timeout")]

RAW_COUNT_FIELDS += [p + "_sync_loss_count" for p in ("ttc", "isl", "iot", "uav")]

RAW_FLAG_FIELDS = [
    "fix_valid", "duplicate_flag", "accepted_sequence_flag",
    "source_unverified_benign_flag", "valid_looking_benign_flag", "auth_present",
    "auth_valid", "integrity_valid", "source_authorized", "session_valid",
    "accepted_flag", *COMMAND_OUTCOME_FLAGS.values(), "session_active",
    "session_expired_flag", "link_degraded_flag", "link_loss_flag", "transition_allowed",
    "unexpected_transition_flag", "forbidden_transition_flag", "delayed_transition_flag",
    "incomplete_transition_flag", "state_changed_flag", "raw_attack_active", "raw_observed_injection", "raw_observation_on_target",
]

OBSERVABLE_DOMAINS['command_intent'].update(TC_SHARED_CONTEXT_INTENTS)

OBSERVABLE_DOMAINS['payload_or_parameter_context'].update(TC_SHARED_PAYLOAD_CONTEXTS + ['PARAMETER_RANGE_VIOLATION'])

OBSERVABLE_DOMAINS['control_security_state'].update({'AUTH_MISSING','AUTH_FAILED','INTEGRITY_FAILED','UNAUTHORIZED',
    'AUTH_UNVERIFIED','INTEGRITY_UNVERIFIED'})

OBSERVABLE_DOMAINS['sequence_freshness_status'].add('UNKNOWN')

OBSERVABLE_DOMAINS['command_execution_result'].update({'BLOCKED','REJECTED'})

OBSERVABLE_DOMAINS['state_transition_consistency'].update({'UNEXPECTED_TRANSITION','FORBIDDEN_TRANSITION'})

RAW_COUNT_FIELDS += ['raw_attempted_injected_command_count','raw_injected_command_count','raw_nominal_command_count']

RAW_FLAG_FIELDS += ['auth_unverified_flag','integrity_unverified_flag','source_unverified_flag','policy_inconsistent_flag',
    'session_required_flag','raw_policy_violation_flag','raw_command_permitted_in_context','raw_critical_command_flag',
    'raw_checks_passed','raw_security_gate_bypassed','raw_handler_dispatched','unknown_sequence_flag']

BASELINE_FIELDS = [f for f in NORMAL_RAW_RANGES if f not in {
    'session_age_s', 'command_rate_per_min', 'parameter_deviation_ratio'}]

BASELINE_FIELDS += ['fix_valid', 'gnss_lock_loss_count', 'gnss_cycle_slip_count', 'ttc_parse_error_count']

BASELINE_FIELDS += [f'{p}_{k}_count' for p in ('ttc', 'isl', 'iot', 'uav')
                    for k in ('crc_error', 'decode_fail', 'sync_loss', 'retry', 'timeout')]

import shared_nominal as nominal
from shared_nominal import (normalize_context, normalize_sequence_freshness, normalize_control_security, normalize_message_control, normalize_signal_quality, normalize_tracking_stability, normalize_navigation_consistency, normalize_timing_consistency, normalize_decoder_status, normalize_command_execution, normalize_retry_timeout, normalize_session_state, normalize_state_transition, normalize_raw_to_observable)

def generate_tc_injection_raw_window(window_id, timestamp, episode, *, nominal_raw=None, attacks_enabled=True):
    raw=nominal.copy_nominal_window(nominal_raw,episode,LABEL_UNAUTHORIZED_TELECOMMAND_INJECTION)
    for f in ('injection_type','injection_case_type','injection_phase','injection_severity',
              'episode_source_profile','episode_command_family','episode_payload_profile','enforcement_mode'):
        raw['raw_'+f]=episode[f]
    raw.update(raw_target_path=TARGET_PATH,raw_target_protocol=TARGET_PROTOCOL,raw_target_direction=TARGET_DIRECTION,
               raw_temporal_pressure=temporal_injection_pressure_score(episode),raw_parameter_limit_ratio=.10,
               raw_critical_command_flag=0)
    # Check audit does not run a second normal handler or change normal outcomes.
    nominal.record_nominal_checks(raw)
    raw['raw_security_gate_bypassed']=0;raw['raw_handler_dispatched']=int(raw['accepted_flag'])
    raw['raw_command_handler_action']=normalize_command_execution(raw)
    raw['raw_policy_violation_flag']=raw['policy_inconsistent_flag']
    intervention=dict(episode)
    intervention['_attacks_enabled']=attacks_enabled
    apply_telecommand_injection_effects(raw,intervention)
    raw['raw_policy_violation_flag']=int(raw['policy_inconsistent_flag'])
    return raw


def validate_attack_rows(df):
    nominal.require_nominal_when(df,df.raw_injected_command_count.eq(0))
    if not df.raw_command_count.eq(df.raw_nominal_command_count+df.raw_injected_command_count).all():
        raise ValueError('Injection command count does not equal nominal + injected traffic.')
    if (df.raw_injected_command_count.gt(0)&df.raw_observation_on_target.eq(0)).any():raise ValueError('Off-target injection evidence.')
    nominal.validate_observed_command_gate(df,'raw_injected_command_count')
    for field in [c for c in df if c.startswith(('ttc_','isl_','iot_','uav_','gnss_')) and 'baseline_'+c in df]:
        if not df[field].equals(df['baseline_'+field]):raise ValueError('Injection altered unrelated physical evidence.')

MAKER_NUMBER = 4
CLASS_LABEL = LABEL_UNAUTHORIZED_TELECOMMAND_INJECTION
CODE_VERSION = nominal.CODE_VERSION
TIMELINE_BUILDER = 'build_tc_injection_episode_timeline'
RAW_GENERATOR = 'generate_tc_injection_raw_window'
OBSERVABLE_DOMAINS = nominal.OBSERVABLE_DOMAINS

def build_tc_injection_dataset(num_rows: int = DEFAULT_NUM_ROWS, window_seconds: int = DEFAULT_WINDOW_SECONDS,
    start_time: datetime | None = None, seed: int = DEFAULT_RANDOM_SEED, return_raw: bool = False,
    *, background_seed: int | None = None, attacks_enabled: bool = True) -> pd.DataFrame:
    """Build from the common nominal substrate; explicit background_seed is for paired audits.

    Default background streams are disjoint across makers, including equal seed values.
    Do not pool replicated paired-audit backgrounds into independent train/test splits.
    """
    return nominal.build_dataset(globals(), num_rows, window_seconds, start_time, seed,
                                 return_raw, background_seed, attacks_enabled)

build_unauthorized_telecommand_injection_dataset = build_tc_injection_dataset

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
    nominal.run_cli(globals(), 'build_tc_injection_dataset')
