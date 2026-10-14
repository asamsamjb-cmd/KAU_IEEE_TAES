# Revision: common nominal background + shared observation mapping.
# Research objective: reviewer revision on synthetic-rule dependence and temporal evidence.
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import os

import tempfile

import io

import csv

import hashlib

import json

import platform

import zipfile

import shared_context_v4 as shared_context_module

from pathlib import Path

from typing import Any, Dict, List

import numpy as np

import pandas as pd

from shared_context_v4 import (
    CONTEXT_FIELD_NAMES,
    GLOBAL_CONTEXT_PROFILE_POOL,
    is_command_context,
    nominal_semantics_for_context,
    sample_shared_context_profile,
    validate_context_profile,
    validate_global_context_pool,
)

DEFAULT_NUM_ROWS = 50000

DEFAULT_WINDOW_SECONDS = 1

DEFAULT_OUTPUT_PATH = "dataset/gnss_spoofing_observable_dataset.csv"

DEFAULT_RAW_AUDIT_OUTPUT_PATH = "dataset/gnss_spoofing_observable_dataset_with_raw.csv"

DEFAULT_SAMPLE_TIMELINE_OUTPUT_PATH = "dataset/gnss_spoofing_sample_episode_timeline.csv"

DEFAULT_RANDOM_SEED = 242

RNG = np.random.default_rng(seed=DEFAULT_RANDOM_SEED)

CODE_VERSION = "gnss-consistent-raw-v1-20261002"

LABEL_SCOPE = "EPISODE_SCENARIO"

LABEL_GNSS_SPOOFING = "GNSS_SPOOFING"

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

GNSS_SPOOFING_CONTEXT_SCENARIOS = [
    "GNSS_RECEIVER_SPOOFING",
    "ONBOARD_NAVIGATION_IMPACT",
    "MULTI_STAGE_GNSS_SPOOFING",
]

GNSS_SPOOFING_CONTEXT_PROBS = [0.72, 0.10, 0.18]

GNSS_SPOOFING_TYPES = [
    "GRADUAL_PULL_OFF",
    "MEACONING_REPLAY",
    "TIME_SYNCHRONIZATION_SPOOFING",
    "NAVIGATION_MESSAGE_SPOOFING",
    "AGGRESSIVE_POSITION_JUMP",
    "MULTI_SOURCE_SPOOFING",
]

GNSS_SPOOFING_TYPE_PROBS = [0.46, 0.06, 0.18, 0.09, 0.02, 0.19]

GNSS_SPOOFING_PHASES = [
    "PRE_SPOOF",
    "SIGNAL_ALIGNMENT",
    "SIGNAL_CAPTURE",
    "PULL_OFF",
    "FALSE_NAVIGATION",
    "RECOVERY",
]

GNSS_SPOOFING_CASE_TYPES = [
    "BENIGN_OVERLAP_DRIFT",
    "GRADUAL_PULL_OFF_HARD",
    "TIMING_DOMINANT_HARD",
    "OBVIOUS_FALSE_NAVIGATION",
    "RECOVERY_OVERLAP",
]

GNSS_SPOOFING_CASE_TYPE_PROBS = [
    0.44,   # BENIGN_OVERLAP_DRIFT: strongest normal-overlap case
    0.36,   # GRADUAL_PULL_OFF_HARD: temporal residual drift
    0.08,   # TIMING_DOMINANT_HARD: clock/timing persistence
    0.005,  # OBVIOUS_FALSE_NAVIGATION: retained but very rare
    0.115,  # RECOVERY_OVERLAP: post-drift benign-looking recovery
]

COMMON_EPISODE_LENGTH_BUCKETS = ["SHORT", "MEDIUM", "LONG"]

COMMON_EPISODE_LENGTH_PROBS = [0.35, 0.45, 0.20]

COMMON_EPISODE_LENGTH_RANGES = {
    "SHORT": (20, 40),
    "MEDIUM": (40, 80),
    "LONG": (80, 140),
}

MESSAGE_CONTROL_FIELDS = [
    "message_type",
    "service_context",
    "command_intent",
    "payload_or_parameter_context",
    "sequence_freshness_status",
    "control_security_state",
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
    """Sample one item using the shared RNG without repeatedly calling np.random.choice.

    This keeps the generator deterministic while avoiding unnecessary overhead
    in large 50,000-window dataset generation.
    """
    probs_array = np.array(probs, dtype=float)
    probs_sum = probs_array.sum()
    if probs_sum <= 0:
        raise ValueError("weighted_choice requires a positive probability sum.")
    probs_array = probs_array / probs_sum
    threshold = float(RNG.random())
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
    global RNG
    RNG = np.random.default_rng(seed=seed)

def sample_common_episode_length() -> tuple[str, int]:
    """Sample the shared episode-length prior (20--140 seconds).

    A common prior removes a deliberately class-specific duration setting;
    realized samples and downstream splits still require a joint audit.
    """
    bucket = weighted_choice(COMMON_EPISODE_LENGTH_BUCKETS, COMMON_EPISODE_LENGTH_PROBS)
    low, high = COMMON_EPISODE_LENGTH_RANGES[bucket]
    # upper bound is inclusive for interpretability in the raw audit.
    duration = int(RNG.integers(low, high + 1))
    return bucket, duration

def sample_spoofing_phase_plan(case_type: str, spoofing_type: str) -> Dict[str, float]:
    """Create jittered cumulative phase boundaries for one episode.

    The phase order is fixed, but boundary positions vary per episode and
    case. This prevents a temporal model from memorizing deterministic ratios
    such as exactly 10/25/45/75/90 percent of the episode.
    """
    # Base cumulative boundaries: PRE, ALIGNMENT, CAPTURE, PULL_OFF, FALSE_NAV.
    base = np.array([0.10, 0.25, 0.45, 0.75, 0.90], dtype=float)

    if case_type == "BENIGN_OVERLAP_DRIFT":
        # Longer ambiguous alignment/capture, delayed false-navigation evidence.
        base += np.array([0.020, 0.035, 0.035, 0.020, 0.005])
    elif case_type == "TIMING_DOMINANT_HARD":
        # Timing drift accumulates slightly earlier but remains gradual.
        base += np.array([0.000, -0.010, -0.005, 0.010, 0.020])
    elif case_type == "OBVIOUS_FALSE_NAVIGATION":
        # Rare obvious cases can enter false navigation earlier.
        base += np.array([-0.005, -0.015, -0.025, -0.040, -0.030])
    elif case_type == "RECOVERY_OVERLAP":
        # Recovery should be visible but not template-fixed.
        base += np.array([0.000, 0.020, 0.020, -0.010, -0.040])

    if spoofing_type == "TIME_SYNCHRONIZATION_SPOOFING":
        base += np.array([0.000, -0.005, 0.000, 0.020, 0.020])
    elif spoofing_type == "MEACONING_REPLAY":
        base += np.array([0.000, 0.015, 0.010, 0.010, 0.000])

    jitter = RNG.normal(0.0, 0.025, size=5)
    boundaries = base + jitter

    # Enforce monotonicity and minimum phase widths.
    mins = np.array([0.05, 0.14, 0.28, 0.58, 0.78], dtype=float)
    maxs = np.array([0.18, 0.36, 0.58, 0.86, 0.97], dtype=float)
    boundaries = np.clip(boundaries, mins, maxs)
    for idx in range(1, len(boundaries)):
        boundaries[idx] = max(boundaries[idx], boundaries[idx - 1] + 0.055)
    boundaries[-1] = min(boundaries[-1], 0.97)

    return {
        "pre_end": round(float(boundaries[0]), 4),
        "alignment_end": round(float(boundaries[1]), 4),
        "capture_end": round(float(boundaries[2]), 4),
        "pull_off_end": round(float(boundaries[3]), 4),
        "false_navigation_end": round(float(boundaries[4]), 4),
    }

def get_spoofing_phase(step: int, duration: int, phase_plan: Dict[str, float] | None = None) -> str:
    ratio = step / max(duration - 1, 1)

    if phase_plan is None:
        phase_plan = {
            "pre_end": 0.10,
            "alignment_end": 0.25,
            "capture_end": 0.45,
            "pull_off_end": 0.75,
            "false_navigation_end": 0.90,
        }

    if ratio < phase_plan["pre_end"]:
        return "PRE_SPOOF"
    if ratio < phase_plan["alignment_end"]:
        return "SIGNAL_ALIGNMENT"
    if ratio < phase_plan["capture_end"]:
        return "SIGNAL_CAPTURE"
    if ratio < phase_plan["pull_off_end"]:
        return "PULL_OFF"
    if ratio < phase_plan["false_navigation_end"]:
        return "FALSE_NAVIGATION"
    return "RECOVERY"

def phase_to_spoofing_severity(
    phase: str,
    spoofing_type: str,
    case_type: str = "GRADUAL_PULL_OFF_HARD",
) -> str:
    """Sample a synthetic severity stratum conditional on phase and case.

    Existing priors are retained as explicit scenario assumptions, without
    enforcing any target single-window or sequence classification performance.
    PRE_SPOOF always uses NONE. Severity can fluctuate between adjacent windows.
    """
    if phase == "PRE_SPOOF":
        return "NONE"

    if case_type == "BENIGN_OVERLAP_DRIFT":
        if phase == "SIGNAL_ALIGNMENT":
            return weighted_choice(["NONE", "SUBTLE"], [0.68, 0.32])
        if phase == "SIGNAL_CAPTURE":
            return weighted_choice(["NONE", "SUBTLE"], [0.42, 0.58])
        if phase == "PULL_OFF":
            return weighted_choice(["SUBTLE", "MODERATE"], [0.78, 0.22])
        if phase == "FALSE_NAVIGATION":
            return weighted_choice(["SUBTLE", "MODERATE"], [0.55, 0.45])
        if phase == "RECOVERY":
            return weighted_choice(["NONE", "SUBTLE"], [0.52, 0.48])

    if case_type == "RECOVERY_OVERLAP":
        if phase == "SIGNAL_ALIGNMENT":
            return weighted_choice(["NONE", "SUBTLE"], [0.45, 0.55])
        if phase == "SIGNAL_CAPTURE":
            return weighted_choice(["SUBTLE", "MODERATE"], [0.76, 0.24])
        if phase == "PULL_OFF":
            return weighted_choice(["SUBTLE", "MODERATE", "SEVERE"], [0.42, 0.50, 0.08])
        if phase == "FALSE_NAVIGATION":
            return weighted_choice(["MODERATE", "SEVERE"], [0.76, 0.24])
        if phase == "RECOVERY":
            return weighted_choice(["NONE", "SUBTLE", "MODERATE"], [0.34, 0.56, 0.10])

    if case_type == "TIMING_DOMINANT_HARD":
        if phase == "SIGNAL_ALIGNMENT":
            return weighted_choice(["NONE", "SUBTLE"], [0.48, 0.52])
        if phase == "SIGNAL_CAPTURE":
            return weighted_choice(["SUBTLE", "MODERATE"], [0.76, 0.24])
        if phase == "PULL_OFF":
            return weighted_choice(["SUBTLE", "MODERATE"], [0.52, 0.48])
        if phase == "FALSE_NAVIGATION":
            return weighted_choice(["MODERATE", "SEVERE"], [0.72, 0.28])
        if phase == "RECOVERY":
            return weighted_choice(["NONE", "SUBTLE"], [0.42, 0.58])

    if case_type == "OBVIOUS_FALSE_NAVIGATION":
        if phase == "SIGNAL_ALIGNMENT":
            return weighted_choice(["SUBTLE", "MODERATE"], [0.72, 0.28])
        if phase == "SIGNAL_CAPTURE":
            return weighted_choice(["SUBTLE", "MODERATE"], [0.42, 0.58])
        if phase == "PULL_OFF":
            return weighted_choice(["MODERATE", "SEVERE"], [0.56, 0.44])
        if phase == "FALSE_NAVIGATION":
            return weighted_choice(["MODERATE", "SEVERE"], [0.32, 0.68])
        if phase == "RECOVERY":
            return weighted_choice(["SUBTLE", "MODERATE"], [0.64, 0.36])

    # Default: gradual pull-off hard case.
    if phase == "SIGNAL_ALIGNMENT":
        return weighted_choice(["NONE", "SUBTLE"], [0.45, 0.55])
    if phase == "SIGNAL_CAPTURE":
        return weighted_choice(["SUBTLE", "MODERATE"], [0.74, 0.26])
    if phase == "PULL_OFF":
        return weighted_choice(["SUBTLE", "MODERATE"], [0.48, 0.52])
    if phase == "FALSE_NAVIGATION":
        return weighted_choice(["MODERATE", "SEVERE"], [0.78, 0.22])
    if phase == "RECOVERY":
        return weighted_choice(["NONE", "SUBTLE"], [0.38, 0.62])

    return "SUBTLE"

def assign_window_training_role(phase: str, severity: str) -> str:
    """Audit phase stratum only; never select different endpoints by model.

    These tags are not evidence-based attack detection labels. All baselines
    must use identical, predeclared endpoint eligibility and episode splits.
    """
    if phase == "PRE_SPOOF" or severity == "NONE":
        return "CONTEXT_ONLY"
    if phase == "RECOVERY":
        return "RECOVERY_CONTEXT"
    return "SCENARIO_ACTIVE"

def build_gnss_spoofing_episode_timeline(
    num_rows: int,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    *, nominal_timeline: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build phase-stratified synthetic GNSS episodes.

    Observation contexts use the same label-agnostic per-window sampler as
    makers 1/2. This is an observation selection model, not a mode trajectory,
    and does not prove that all class shortcuts have been removed.
    """
    validate_generation_args(num_rows, window_seconds)
    timeline: List[Dict[str, Any]] = []
    episode_index = 0
    window_id = 0

    validate_global_context_pool()

    while window_id < num_rows:
        # Metadata-only GNSS scenario stratum.  It is NOT used to construct the
        # seven learning-visible context fields.
        context_type = weighted_choice(
            GNSS_SPOOFING_CONTEXT_SCENARIOS,
            GNSS_SPOOFING_CONTEXT_PROBS,
        )

        spoofing_type = weighted_choice(
            GNSS_SPOOFING_TYPES,
            GNSS_SPOOFING_TYPE_PROBS,
        )

        spoofing_case_type = weighted_choice(
            GNSS_SPOOFING_CASE_TYPES,
            GNSS_SPOOFING_CASE_TYPE_PROBS,
        )

        episode_length_type, duration = nominal_timeline[window_id]["episode_length_bucket"], nominal_timeline[window_id]["episode_length"]
        phase_plan = sample_spoofing_phase_plan(
            case_type=spoofing_case_type,
            spoofing_type=spoofing_type,
        )

        episode_id = f"GNSSSPF_EP_{episode_index:05d}"

        for step in range(duration):
            if window_id >= num_rows:
                break

            phase = get_spoofing_phase(step, duration, phase_plan=phase_plan)
            severity = phase_to_spoofing_severity(
                phase=phase,
                spoofing_type=spoofing_type,
                case_type=spoofing_case_type,
            )

            # Benchmark A-prime: sample the seven context fields from the same
            # label-agnostic pool used by NORMAL/RF/Ground-TT&C.  No attack
            # label/type/phase/severity is passed to this sampler.
            context_profile = nominal_timeline[window_id]
            validate_context_profile(context_profile)
            semantic_defaults = {f: context_profile[f] for f in ("message_type", "service_context", "command_intent")}
            command_context = is_command_context(context_profile)

            command_intent = str(semantic_defaults.get("command_intent", "NO_COMMAND"))
            if not command_context:
                command_intent = "NO_COMMAND"

            episode = {
                "scenario_id": "GNSS_SPOOFING_SCENARIO_0001",
                "episode_id": episode_id,
                "episode_type": context_type,
                "episode_step": step,
                "episode_length": duration,
                "episode_length_bucket": episode_length_type,
                "elapsed_seconds": step * window_seconds,
                "episode_progress": step / max(duration - 1, 1),
                "phase_boundary_pre_end": phase_plan["pre_end"],
                "phase_boundary_alignment_end": phase_plan["alignment_end"],
                "phase_boundary_capture_end": phase_plan["capture_end"],
                "phase_boundary_pull_off_end": phase_plan["pull_off_end"],
                "phase_boundary_false_navigation_end": phase_plan["false_navigation_end"],
                "spoofing_phase": phase,
                "spoofing_severity": severity,
                "spoofing_type": spoofing_type,
                "spoofing_case_type": spoofing_case_type,
                "shared_context_profile": context_profile.get("shared_context_profile", "UNKNOWN"),
                "shared_context_family": context_profile.get("shared_context_family", "UNKNOWN"),
                "is_command_context": int(command_context),

                **{field: context_profile[field] for field in CONTEXT_FIELD_NAMES},
                "message_type": str(semantic_defaults.get("message_type", "NOMINAL_MIXED_TRAFFIC")),
                "service_context": str(semantic_defaults.get("service_context", "HOUSEKEEPING")),
                "command_intent": command_intent,
            }

            timeline.append(episode)
            window_id += 1

        episode_index += 1

    return timeline

def phase_progress_factor(phase: str, severity: str, case_type: str) -> float:
    """Legacy audit-only severity index; not a measured or integrated drift.

    It does not modify measurements or categorical observables in this version.
    """
    if phase == "PRE_SPOOF":
        base = 0.00
    elif phase == "SIGNAL_ALIGNMENT":
        base = 0.04
    elif phase == "SIGNAL_CAPTURE":
        base = 0.11
    elif phase == "PULL_OFF":
        base = 0.42
    elif phase == "FALSE_NAVIGATION":
        base = 0.70
    elif phase == "RECOVERY":
        base = 0.20
    else:
        base = 0.10

    if severity == "NONE":
        base *= 0.25
    elif severity == "SUBTLE":
        base *= 0.68
    elif severity == "MODERATE":
        base *= 1.00
    elif severity == "SEVERE":
        base *= 1.12

    if case_type == "BENIGN_OVERLAP_DRIFT":
        base *= 0.70
    elif case_type == "TIMING_DOMINANT_HARD":
        base *= 0.88
    elif case_type == "OBVIOUS_FALSE_NAVIGATION":
        base *= 1.28
    elif case_type == "RECOVERY_OVERLAP" and phase == "RECOVERY":
        base *= 0.55

    return float(np.clip(base, 0.0, 1.0))

def temporal_spoofing_pressure_score(episode: Dict[str, Any]) -> float:
    """Record the legacy audit index; not a physical pressure or attack score."""
    return round(
        phase_progress_factor(
            phase=episode.get("spoofing_phase", "PRE_SPOOF"),
            severity=episode.get("spoofing_severity", "NONE"),
            case_type=episode.get("spoofing_case_type", "GRADUAL_PULL_OFF_HARD"),
        ),
        4,
    )

def apply_gnss_spoofing_effects(raw: Dict[str, Any], episode: Dict[str, Any]) -> None:
    severity = "NONE" if episode["spoofing_phase"] == "PRE_SPOOF" else episode["spoofing_severity"]
    spoofing_type = episode["spoofing_type"]
    spoofing_phase = episode["spoofing_phase"]
    case_type = episode.get("spoofing_case_type", "GRADUAL_PULL_OFF_HARD")

    # Raw-audit-only flags. These are intentionally not part of the 22-field
    # learning input. They document the latent spoofing process and support
    # post-generation validation.
    active = spoofing_phase not in {"PRE_SPOOF", "RECOVERY"} and severity != "NONE"
    residual = spoofing_phase == "RECOVERY" and severity != "NONE"
    raw["raw_attack_active"] = int(active)
    raw["raw_residual_effect_active"] = int(residual)
    raw["raw_effect_applied"] = int(active or residual)
    raw["raw_spoofer_present_flag"] = int(active)
    raw["raw_temporal_pressure"] = temporal_spoofing_pressure_score(episode) if active or residual else 0.0
    raw["raw_signal_alignment_flag"] = 1 if active and spoofing_phase in [
        "SIGNAL_ALIGNMENT", "SIGNAL_CAPTURE", "PULL_OFF", "FALSE_NAVIGATION"
    ] else 0
    raw["raw_signal_capture_flag"] = 1 if active and spoofing_phase in [
        "SIGNAL_CAPTURE", "PULL_OFF", "FALSE_NAVIGATION"
    ] else 0
    raw["raw_pull_off_flag"] = 1 if active and spoofing_phase in [
        "PULL_OFF", "FALSE_NAVIGATION"
    ] else 0
    raw["raw_false_pnt_flag"] = 1 if active and spoofing_phase == "FALSE_NAVIGATION" else 0

    raw["raw_code_phase_divergence_flag"] = 0
    raw["raw_carrier_phase_anomaly_flag"] = 0
    raw["raw_suspicious_stability_flag"] = 0
    raw["raw_navigation_message_anomaly_flag"] = 0
    raw["raw_ephemeris_inconsistency_flag"] = 0
    raw["raw_clock_bias_attack_flag"] = 0

    def set_gnss(
        cn0_low: float,
        cn0_high: float,
        cn0std_low: float,
        cn0std_high: float,
        jitter_low: float,
        jitter_high: float,
        lock_low: float,
        lock_high: float,
        pos_low: float,
        pos_high: float,
        vel_low: float,
        vel_high: float,
        pr_low: float,
        pr_high: float,
        time_low: float,
        time_high: float,
        drift_low: float,
        drift_high: float,
    ) -> None:
        raw["gnss_cn0_mean_dbhz"] = uniform_float(cn0_low, cn0_high, 3)
        raw["gnss_cn0_std_db"] = uniform_float(cn0std_low, cn0std_high, 3)
        raw["gnss_tracking_jitter"] = uniform_float(jitter_low, jitter_high, 4)
        raw["gnss_carrier_lock_ratio"] = uniform_float(lock_low, lock_high, 4)
        raw["position_jump_m"] = uniform_float(pos_low, pos_high, 3)
        raw["velocity_jump_mps"] = uniform_float(vel_low, vel_high, 4)
        raw["pseudorange_residual_m"] = uniform_float(pr_low, pr_high, 3)
        raw["time_residual_ns"] = uniform_float(time_low, time_high, 3)
        raw["clock_drift_ppm"] = uniform_float(drift_low, drift_high, 5)

    if severity == "NONE":
        return  # Preserve every baseline measurement, including fix validity.

    # Spoofing keeps access to apparently valid GNSS-like signals. The signal
    # quality remains normal-looking while PNT residuals grow by phase/case.
    if case_type == "BENIGN_OVERLAP_DRIFT":
        if severity == "SUBTLE":
            set_gnss(38.0, 47.5, 0.9, 3.0, 0.045, 0.155, 0.965, 0.998,
                     3.5, 15.5, 0.08, 0.48, 3.5, 15.5, 22.0, 95.0, 0.008, 0.045)
        elif severity == "MODERATE":
            set_gnss(38.5, 48.5, 0.8, 2.6, 0.04, 0.150, 0.970, 0.999,
                     12.0, 62.0, 0.30, 1.45, 12.0, 52.0, 75.0, 520.0, 0.025, 0.220)
        else:
            set_gnss(39.0, 49.0, 0.65, 2.2, 0.035, 0.140, 0.975, 0.999,
                     26.0, 135.0, 0.70, 3.10, 24.0, 105.0, 240.0, 2600.0, 0.090, 0.850)

    elif severity == "SUBTLE":
        set_gnss(38.0, 48.0, 0.8, 2.8, 0.04, 0.160, 0.965, 0.998,
                 4.0, 18.0, 0.10, 0.55, 4.0, 18.0, 28.0, 125.0, 0.010, 0.060)

    elif severity == "MODERATE":
        set_gnss(39.0, 50.0, 0.65, 2.3, 0.030, 0.150, 0.970, 1.000,
                 14.0, 76.0, 0.35, 1.80, 14.0, 60.0, 90.0, 780.0, 0.030, 0.300)

    elif severity == "SEVERE":
        if spoofing_type == "AGGRESSIVE_POSITION_JUMP" or case_type == "OBVIOUS_FALSE_NAVIGATION":
            set_gnss(40.0, 50.8, 0.50, 1.80, 0.020, 0.140, 0.980, 1.000,
                     120.0, 620.0, 2.20, 8.50, 80.0, 320.0, 1200.0, 11000.0, 0.500, 2.600)
        elif spoofing_type == "TIME_SYNCHRONIZATION_SPOOFING" or case_type == "TIMING_DOMINANT_HARD":
            set_gnss(39.5, 50.5, 0.50, 1.90, 0.025, 0.145, 0.978, 1.000,
                     18.0, 88.0, 0.35, 1.90, 16.0, 66.0, 650.0, 7200.0, 0.300, 2.200)
            raw["raw_clock_bias_attack_flag"] = 1
        else:
            set_gnss(39.5, 50.5, 0.55, 2.0, 0.025, 0.145, 0.975, 1.000,
                     42.0, 190.0, 0.90, 4.20, 32.0, 145.0, 420.0, 4200.0, 0.140, 1.300)

    # Spoofing normally preserves the receiver fix; invalid fix would make the
    # sample resemble jamming or receiver outage.
    raw["fix_valid"] = weighted_choice([1, 0], [0.995, 0.005]) if severity == "SEVERE" else 1

    # Exactly one measurement sampler; do not overwrite strong residuals or
    # invalid fixes to enforce desired token prevalence or temporal-model scores.

    # Type-specific raw evidence. Keep these mostly as raw-audit attributes;
    # these latent flags are not independent receiver measurements and are not
    # used by the common observable mapper.
    if spoofing_type == "MEACONING_REPLAY":
        raw["raw_navigation_message_anomaly_flag"] = flag_choice(0.16 if severity in ["MODERATE", "SEVERE"] else 0.05)
    elif spoofing_type == "NAVIGATION_MESSAGE_SPOOFING":
        raw["raw_navigation_message_anomaly_flag"] = 1
        raw["raw_ephemeris_inconsistency_flag"] = flag_choice(0.22 if severity != "SEVERE" else 0.42)
    elif spoofing_type == "MULTI_SOURCE_SPOOFING":
        raw["raw_code_phase_divergence_flag"] = max(
            raw["raw_code_phase_divergence_flag"],
            flag_choice(0.07 if severity in ["MODERATE", "SEVERE"] else 0.02),
        )
        raw["raw_carrier_phase_anomaly_flag"] = max(
            raw["raw_carrier_phase_anomaly_flag"],
            flag_choice(0.05 if severity in ["MODERATE", "SEVERE"] else 0.015),
        )
    elif spoofing_type == "TIME_SYNCHRONIZATION_SPOOFING":
        raw["raw_clock_bias_attack_flag"] = 1 if severity in ["MODERATE", "SEVERE"] else raw["raw_clock_bias_attack_flag"]

    if severity in ["SUBTLE", "MODERATE"]:
        raw["raw_suspicious_stability_flag"] = flag_choice(
            0.055 if case_type == "BENIGN_OVERLAP_DRIFT" else 0.075
        )
        raw["raw_code_phase_divergence_flag"] = max(
            raw["raw_code_phase_divergence_flag"],
            flag_choice(0.018 if severity == "SUBTLE" else 0.055),
        )
        raw["raw_carrier_phase_anomaly_flag"] = max(
            raw["raw_carrier_phase_anomaly_flag"],
            flag_choice(0.010 if severity == "SUBTLE" else 0.035),
        )

    elif severity == "SEVERE":
        raw["raw_suspicious_stability_flag"] = flag_choice(0.11)
        raw["raw_code_phase_divergence_flag"] = max(
            raw["raw_code_phase_divergence_flag"],
            flag_choice(0.095),
        )
        raw["raw_carrier_phase_anomaly_flag"] = max(
            raw["raw_carrier_phase_anomaly_flag"],
            flag_choice(0.065),
        )

    # Deception rather than denial: avoid jamming-like lock loss and cycle-slip bursts.
    raw["gnss_lock_loss_count"] = 0 if RNG.random() < 0.998 else 1
    raw["gnss_cycle_slip_count"] = count_choice([0, 1], [0.965, 0.035])

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
    "incomplete_transition_flag", "state_changed_flag", "raw_attack_active", "raw_residual_effect_active", "raw_effect_applied", "raw_gnss_monitor_available",
]

def validate_generation_args(num_rows: int, window_seconds: int) -> None:
    if isinstance(num_rows, (bool, np.bool_)) or not isinstance(num_rows, (int, np.integer)) or num_rows <= 0:
        raise ValueError("num_rows must be a positive integer.")
    if isinstance(window_seconds, (bool, np.bool_)) or window_seconds != 1:
        raise ValueError("Only one-second windows are supported by this maker.")

PHASE_BOUNDARY_KEYS = ('pre_end', 'alignment_end', 'capture_end', 'pull_off_end', 'false_navigation_end')

SEVERITIES = {'NONE', 'SUBTLE', 'MODERATE', 'SEVERE'}

GNSS_MEASUREMENT_FIELDS = [
    'gnss_cn0_mean_dbhz', 'gnss_cn0_std_db', 'gnss_lock_loss_count',
    'gnss_cycle_slip_count', 'gnss_tracking_jitter', 'gnss_carrier_lock_ratio',
    'position_jump_m', 'velocity_jump_mps', 'pseudorange_residual_m',
    'time_residual_ns', 'clock_drift_ppm', 'fix_valid',
]

BASELINE_FIELDS = [f for f in NORMAL_RAW_RANGES if f not in {
    'session_age_s', 'command_rate_per_min', 'parameter_deviation_ratio'}]

BASELINE_FIELDS += ['fix_valid', 'gnss_lock_loss_count', 'gnss_cycle_slip_count', 'ttc_parse_error_count']

BASELINE_FIELDS += [f'{p}_{k}_count' for p in ('ttc', 'isl', 'iot', 'uav')
                    for k in ('crc_error', 'decode_fail', 'sync_loss', 'retry', 'timeout')]

import shared_nominal as nominal
from shared_nominal import (normalize_context, normalize_sequence_freshness, normalize_control_security, normalize_message_control, normalize_signal_quality, normalize_tracking_stability, normalize_navigation_consistency, normalize_timing_consistency, normalize_decoder_status, normalize_command_execution, normalize_retry_timeout, normalize_session_state, normalize_state_transition, normalize_raw_to_observable)

def generate_gnss_spoofing_raw_window(window_id, timestamp, episode, *, nominal_raw=None, attacks_enabled=True):
    raw=nominal.copy_nominal_window(nominal_raw,episode,LABEL_GNSS_SPOOFING)
    for f in ('spoofing_type','spoofing_phase','spoofing_severity','spoofing_case_type'):raw['raw_'+f]=episode[f]
    intervention=dict(episode)
    if not attacks_enabled:intervention['spoofing_severity']='NONE'
    apply_gnss_spoofing_effects(raw,intervention)
    raw['raw_persistent_effect_present']=raw['raw_residual_effect_active']
    if raw['raw_effect_applied']:
        raw['fix_valid']=min(raw['fix_valid'],nominal_raw['fix_valid'])
        # Spoofing does not repair a coincident benign receiver outage.
        for field in ('gnss_lock_loss_count','gnss_cycle_slip_count'):
            raw[field]=max(raw[field],nominal_raw[field])
        nominal.refresh_environment_semantics(raw)
    return raw


def validate_attack_rows(df):
    nominal.require_nominal_when(df,df.raw_effect_applied.eq(0))
    fields=[c for c in df if c.startswith(('ttc_','isl_','iot_','uav_')) and 'baseline_'+c in df]
    fields+=['raw_command_count','auth_present','auth_valid','integrity_valid','source_authorized',
             'sequence_gap','duplicate_flag','session_active','session_valid','state_changed_flag',
             'raw_command_intent',*COMMAND_OUTCOME_FLAGS.values()]
    for field in fields:
        if not df[field].equals(df['baseline_'+field]):raise ValueError(f'GNSS intervention changed unrelated evidence: {field}')

MAKER_NUMBER = 3
CLASS_LABEL = LABEL_GNSS_SPOOFING
CODE_VERSION = nominal.CODE_VERSION
TIMELINE_BUILDER = 'build_gnss_spoofing_episode_timeline'
RAW_GENERATOR = 'generate_gnss_spoofing_raw_window'
OBSERVABLE_DOMAINS = nominal.OBSERVABLE_DOMAINS

def build_gnss_spoofing_dataset(num_rows: int = DEFAULT_NUM_ROWS, window_seconds: int = DEFAULT_WINDOW_SECONDS,
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
    nominal.run_cli(globals(), 'build_gnss_spoofing_dataset')
