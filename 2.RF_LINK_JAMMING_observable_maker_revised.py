# Revision: common nominal background + shared observation mapping.
# Research objective: reviewer revision on synthetic-rule dependence and temporal evidence.
"""2.1 — RF-link interference scenario maker (one-second windows).

The 29-column CSV contains seven metadata columns and 22 model-input fields.
label denotes the EPISODE SCENARIO, including pre-attack/off windows; it is
NOT an instantaneous attack-presence label. raw_attack_active records whether
interference was applied to a configured target in this window.

Targets are fixed per episode; observation contexts use the unchanged shared
pool. GNSS tracking/navigation/timing are spacecraft receiver monitors; signal,
decoder and retry fields select raw_path_type. Context profiles still vary per
window: this is not a simulated continuous spacecraft-mode trajectory.

Numeric impairment ranges and phase/case probabilities are controlled synthetic
assumptions, not a calibrated RF waveform/channel simulator. Waveform names are
scenario descriptors, not implementations of those waveforms. RECOVERY denotes
waning interference, possibly still active. No empirical waveform realism,
authentication enforcement or detector advantage is established by this maker.

All random choices belong to generation. Projection is deterministic, uses no
scenario label/phase/severity, and uses the revised maker-1 extraction thresholds.
No token is softened, capped or resampled to favor cross-window classification.
Internal-path nominal defaults follow maker 1 and do not mean RF was measured.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pathlib import Path

from typing import Any, Dict, List

import hashlib

import json

import platform

import zipfile

import numpy as np

import pandas as pd

import shared_context_v4 as shared_context_module

from shared_context_v4 import (
    CONTEXT_FIELD_NAMES, GLOBAL_CONTEXT_PROFILE_POOL, is_command_context,
    nominal_semantics_for_context, sample_shared_context_profile,
    validate_context_profile, validate_global_context_pool,
)

CODE_VERSION = "2.1-target-consistent-rf"

LABEL_SCOPE = "EPISODE_SCENARIO"

DEFAULT_NUM_ROWS = 50000

DEFAULT_WINDOW_SECONDS = 1

DEFAULT_OUTPUT_PATH = "dataset/rf_link_jamming_observable_dataset.csv"

DEFAULT_RAW_AUDIT_OUTPUT_PATH = "dataset/rf_link_jamming_observable_dataset_with_raw.csv"

DEFAULT_SAMPLE_TIMELINE_OUTPUT_PATH = "dataset/rf_jamming_sample_episode_timeline.csv"

DEFAULT_RANDOM_SEED = 142

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

RNG = np.random.default_rng(seed=DEFAULT_RANDOM_SEED)

LABEL_RF_LINK_JAMMING = "RF_LINK_JAMMING"

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

RF_JAMMING_PATH_SCENARIOS = [
    "TTC_UPLINK_JAMMING",
    "IOT_UPLINK_JAMMING",
    "UAV_NTN_JAMMING",
    "GNSS_JAMMING",
    "ISL_JAMMING",
    "MULTI_PATH_JAMMING",
]

RF_JAMMING_PATH_PROBS = [
    0.30,  # GATEWAY_LEO
    0.20,  # IOT_LEO
    0.20,  # UAV_LEO
    0.15,  # GNSS_LEO
    0.10,  # LEO_LEO
    0.05,  # MULTI_PATH
]

RF_JAMMING_TYPES = [
    "BROADBAND_NOISE",
    "NARROWBAND_TONE",
    "SWEEP_JAMMING",
    "PULSED_JAMMING",
    "INTERMITTENT_JAMMING",
    "BARRAGE_JAMMING",
]

RF_JAMMING_TYPE_PROBS = [
    0.25,
    0.15,
    0.15,
    0.15,
    0.20,
    0.10,
]

RF_JAMMING_PHASES = [
    "PRE_ATTACK",
    "ONSET",
    "MILD",
    "MODERATE",
    "SEVERE",
    "RECOVERY",
]

RF_JAMMING_CASE_TYPES = [
    "BENIGN_OVERLAP_MILD",
    "PERSISTENT_PROGRESSIVE",
    "INTERMITTENT_PULSED",
    "OBVIOUS_SEVERE",
    "RECOVERY_OVERLAP",
]

RF_JAMMING_CASE_TYPE_PROBS = [
    0.43,  # predominantly mild envelope
    0.24,  # progressive envelope
    0.22,  # intermittent envelope
    0.025, # predominantly severe envelope
    0.085, # waning-interference envelope
]

SCENARIO_TARGETS = {
    "TTC_UPLINK_JAMMING": ("GATEWAY_LEO",),
    "IOT_UPLINK_JAMMING": ("IOT_LEO",),
    "UAV_NTN_JAMMING": ("UAV_LEO",),
    "GNSS_JAMMING": ("GNSS_LEO",),
    "ISL_JAMMING": ("LEO_LEO",),
}

MULTIPATH_TARGET_SETS = (
    ("GATEWAY_LEO", "IOT_LEO"), ("UAV_LEO", "GNSS_LEO"),
    ("GATEWAY_LEO", "UAV_LEO", "IOT_LEO"),
)

MULTIPATH_TARGET_PROBS = (0.35, 0.35, 0.30)

PATH_PREFIX = {"GATEWAY_LEO": "ttc", "LEO_LEO": "isl", "IOT_LEO": "iot", "UAV_LEO": "uav"}

PHASE_BOUNDARY_KEYS = ("pre_end", "onset_end", "mild_end", "moderate_end", "severe_end")

def validate_generation_args(num_rows: int, window_seconds: int) -> None:
    if isinstance(num_rows, (bool, np.bool_)) or not isinstance(num_rows, (int, np.integer)) or num_rows <= 0:
        raise ValueError("num_rows must be a positive integer.")
    if isinstance(window_seconds, (bool, np.bool_)) or window_seconds != 1:
        raise ValueError("Only one-second windows are supported by this maker.")

def get_target_paths(episode: Dict[str, Any]) -> tuple[str, ...]:
    scenario = episode["episode_type"]
    expected = SCENARIO_TARGETS.get(scenario)
    targets = tuple(episode.get("target_paths", expected or ()))
    if scenario == "MULTI_PATH_JAMMING":
        if targets not in MULTIPATH_TARGET_SETS:
            raise ValueError("A MULTI_PATH episode must provide its fixed target set.")
    elif expected is None or targets != expected:
        raise ValueError(f"Target paths disagree with episode_type: {scenario}, {targets}")
    return targets

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
    probs_array = np.array(probs, dtype=float)
    probs_array = probs_array / probs_array.sum()

    selected_index = int(RNG.choice(len(items), p=probs_array))
    return items[selected_index]

def count_choice(values: List[int], probs: List[float]) -> int:
    return int(weighted_choice(values, probs))

def flag_choice(prob_one: float) -> int:
    return int(RNG.random() < prob_one)

def reset_rng(seed: int = DEFAULT_RANDOM_SEED) -> None:
    """Reset the global random number generator for reproducible dataset generation."""
    global RNG
    RNG = np.random.default_rng(seed=seed)

def sample_common_episode_duration() -> tuple[str, int]:
    """Sample episode duration from the benchmark-wide length distribution."""
    buckets = list(COMMON_EPISODE_LENGTH_BUCKETS.keys())
    probs = [COMMON_EPISODE_LENGTH_BUCKET_PROBS[b] for b in buckets]
    bucket = weighted_choice(buckets, probs)
    low, high = COMMON_EPISODE_LENGTH_BUCKETS[bucket]
    duration = int(RNG.integers(low, high + 1))
    return bucket, duration

def get_jamming_phase(
    step: int,
    duration: int,
    phase_boundaries: Dict[str, float] | None = None,
) -> str:
    """Return temporally ordered RF-jamming phase with episode-level jitter.

    Earlier versions used fixed phase ratios for every episode.  That makes it
    possible for a sequence model to learn generator-specific phase templates.
    The revised maker keeps the same semantic progression but lets each
    episode shift onset, evidence, severe, and recovery boundaries.
    """
    ratio = step / max(duration - 1, 1)

    if phase_boundaries is None:
        phase_boundaries = {
            "pre_end": 0.10,
            "onset_end": 0.25,
            "mild_end": 0.45,
            "moderate_end": 0.75,
            "severe_end": 0.90,
        }

    if ratio < phase_boundaries["pre_end"]:
        return "PRE_ATTACK"
    if ratio < phase_boundaries["onset_end"]:
        return "ONSET"
    if ratio < phase_boundaries["mild_end"]:
        return "MILD"
    if ratio < phase_boundaries["moderate_end"]:
        return "MODERATE"
    if ratio < phase_boundaries["severe_end"]:
        return "SEVERE"
    return "RECOVERY"

def sample_jamming_phase_boundaries(case_type: str) -> Dict[str, float]:
    """Sample monotone episode-level phase boundaries for RF jamming.

    The boundaries are raw/audit generation controls only.  They are not part
    of the 22-field observable input.  Case types change how long onset,
    evidence, and recovery last, but no case type is directly serialized.
    """
    pre_end = float(RNG.uniform(0.06, 0.14))

    if case_type == "BENIGN_OVERLAP_MILD":
        onset_span = float(RNG.uniform(0.14, 0.24))
        mild_span = float(RNG.uniform(0.20, 0.32))
        moderate_span = float(RNG.uniform(0.18, 0.28))
        severe_span = float(RNG.uniform(0.06, 0.12))
    elif case_type == "INTERMITTENT_PULSED":
        onset_span = float(RNG.uniform(0.10, 0.22))
        mild_span = float(RNG.uniform(0.16, 0.30))
        moderate_span = float(RNG.uniform(0.18, 0.34))
        severe_span = float(RNG.uniform(0.08, 0.15))
    elif case_type == "OBVIOUS_SEVERE":
        onset_span = float(RNG.uniform(0.08, 0.16))
        mild_span = float(RNG.uniform(0.12, 0.22))
        moderate_span = float(RNG.uniform(0.20, 0.32))
        severe_span = float(RNG.uniform(0.10, 0.18))
    elif case_type == "RECOVERY_OVERLAP":
        onset_span = float(RNG.uniform(0.10, 0.20))
        mild_span = float(RNG.uniform(0.14, 0.26))
        moderate_span = float(RNG.uniform(0.18, 0.30))
        severe_span = float(RNG.uniform(0.06, 0.12))
    else:
        onset_span = float(RNG.uniform(0.10, 0.20))
        mild_span = float(RNG.uniform(0.16, 0.28))
        moderate_span = float(RNG.uniform(0.22, 0.34))
        severe_span = float(RNG.uniform(0.07, 0.15))

    onset_end = pre_end + onset_span
    mild_end = onset_end + mild_span
    moderate_end = mild_end + moderate_span
    severe_end = moderate_end + severe_span

    # Clamp while preserving monotonic order and a non-trivial recovery tail.
    onset_end = float(np.clip(onset_end, pre_end + 0.07, 0.34))
    mild_end = float(np.clip(mild_end, onset_end + 0.10, 0.60))
    moderate_end = float(np.clip(moderate_end, mild_end + 0.12, 0.84))
    severe_end = float(np.clip(severe_end, moderate_end + 0.06, 0.94))

    return {
        "pre_end": pre_end,
        "onset_end": onset_end,
        "mild_end": mild_end,
        "moderate_end": moderate_end,
        "severe_end": severe_end,
    }

def phase_to_severity(
    phase: str,
    jamming_type: str,
    case_type: str = "PERSISTENT_PROGRESSIVE",
) -> str:
    """Sample a synthetic impairment envelope; never alter projected tokens.

    Historical case probabilities are explicit scenario assumptions. No result
    or model score is used to select cases. PRE_ATTACK is always interference-free.
    """
    if phase not in RF_JAMMING_PHASES or jamming_type not in RF_JAMMING_TYPES or case_type not in RF_JAMMING_CASE_TYPES:
        raise ValueError("Unknown jamming phase, waveform descriptor or case.")
    if phase == "PRE_ATTACK":
        return "NONE"
    if case_type == "BENIGN_OVERLAP_MILD":
        if phase == "ONSET":
            return weighted_choice(["NONE", "MILD"], [0.24, 0.76])
        if phase == "MILD":
            return weighted_choice(["MILD", "MODERATE"], [0.90, 0.10])
        if phase == "MODERATE":
            return weighted_choice(["MILD", "MODERATE"], [0.50, 0.50])
        if phase == "SEVERE":
            return weighted_choice(["MODERATE", "SEVERE"], [0.82, 0.18])
        if phase == "RECOVERY":
            return weighted_choice(["NONE", "MILD"], [0.48, 0.52])

    if case_type == "INTERMITTENT_PULSED":
        if phase == "ONSET":
            return weighted_choice(["NONE", "MILD", "MODERATE"], [0.24, 0.66, 0.10])
        if phase == "MILD":
            return weighted_choice(["NONE", "MILD", "MODERATE"], [0.24, 0.62, 0.14])
        if phase == "MODERATE":
            return weighted_choice(["MILD", "MODERATE", "SEVERE"], [0.30, 0.60, 0.10])
        if phase == "SEVERE":
            return weighted_choice(["NONE", "MILD", "MODERATE", "SEVERE"], [0.06, 0.22, 0.54, 0.18])
        if phase == "RECOVERY":
            return weighted_choice(["NONE", "MILD", "MODERATE"], [0.34, 0.54, 0.12])

    if case_type == "OBVIOUS_SEVERE":
        if phase == "ONSET":
            return weighted_choice(["MILD", "MODERATE"], [0.66, 0.34])
        if phase == "MILD":
            return weighted_choice(["MILD", "MODERATE"], [0.52, 0.48])
        if phase == "MODERATE":
            return weighted_choice(["MODERATE", "SEVERE"], [0.62, 0.38])
        if phase == "SEVERE":
            if jamming_type in ["BARRAGE_JAMMING", "BROADBAND_NOISE"]:
                return weighted_choice(["MODERATE", "SEVERE", "OUTAGE"], [0.12, 0.78, 0.10])
            return weighted_choice(["MODERATE", "SEVERE", "OUTAGE"], [0.16, 0.79, 0.05])
        if phase == "RECOVERY":
            return weighted_choice(["MODERATE", "MILD", "NONE"], [0.30, 0.52, 0.18])

    if case_type == "RECOVERY_OVERLAP":
        if phase == "ONSET":
            return weighted_choice(["MILD", "MODERATE"], [0.78, 0.22])
        if phase == "MILD":
            return weighted_choice(["MILD", "MODERATE"], [0.74, 0.26])
        if phase == "MODERATE":
            return weighted_choice(["MODERATE", "SEVERE"], [0.82, 0.18])
        if phase == "SEVERE":
            return weighted_choice(["MODERATE", "SEVERE"], [0.44, 0.56])
        if phase == "RECOVERY":
            return weighted_choice(["NONE", "MILD", "MODERATE"], [0.30, 0.58, 0.12])

    # Default progressive case.
    if phase == "ONSET":
        return weighted_choice(["MILD", "MODERATE"], [0.80, 0.20])
    if phase == "MILD":
        return weighted_choice(["MILD", "MODERATE"], [0.84, 0.16])
    if phase == "MODERATE":
        return weighted_choice(["MILD", "MODERATE", "SEVERE"], [0.22, 0.70, 0.08])
    if phase == "SEVERE":
        if jamming_type == "BARRAGE_JAMMING":
            return weighted_choice(["MODERATE", "SEVERE", "OUTAGE"], [0.34, 0.62, 0.04])
        return weighted_choice(["MODERATE", "SEVERE", "OUTAGE"], [0.38, 0.60, 0.02])
    if phase == "RECOVERY":
        return weighted_choice(["MODERATE", "MILD", "NONE"], [0.20, 0.58, 0.22])

    raise ValueError("Unmapped phase/case combination.")

def build_rf_jamming_episode_timeline(
    num_rows: int,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    *, nominal_timeline: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    validate_generation_args(num_rows, window_seconds)
    timeline = []
    episode_index = 0
    window_id = 0

    validate_global_context_pool()

    while window_id < num_rows:
        path_scenario = weighted_choice(
            RF_JAMMING_PATH_SCENARIOS,
            RF_JAMMING_PATH_PROBS,
        )
        target_paths = (weighted_choice(MULTIPATH_TARGET_SETS, MULTIPATH_TARGET_PROBS)
                        if path_scenario == "MULTI_PATH_JAMMING" else SCENARIO_TARGETS[path_scenario])

        jamming_type = weighted_choice(
            RF_JAMMING_TYPES,
            RF_JAMMING_TYPE_PROBS,
        )

        jamming_case_type = weighted_choice(
            RF_JAMMING_CASE_TYPES,
            RF_JAMMING_CASE_TYPE_PROBS,
        )

        # Scenario prior: intermittent/pulsed descriptors more often use intermittent envelopes.
        if jamming_type in ["INTERMITTENT_JAMMING", "PULSED_JAMMING"] and RNG.random() < 0.34:
            jamming_case_type = "INTERMITTENT_PULSED"

        episode_length_bucket, duration = nominal_timeline[window_id]["episode_length_bucket"], nominal_timeline[window_id]["episode_length"]
        phase_boundaries = sample_jamming_phase_boundaries(jamming_case_type)

        episode_id = f"RFJAM_EP_{episode_index:05d}"

        for step in range(duration):
            if window_id >= num_rows:
                break

            elapsed_seconds = step * window_seconds
            episode_progress = step / max(duration - 1, 1)

            phase = get_jamming_phase(step, duration, phase_boundaries=phase_boundaries)
            severity = phase_to_severity(
                phase=phase,
                jamming_type=jamming_type,
                case_type=jamming_case_type,
            )

            # Label-agnostic per-window Observation Context sampling.  Do not
            # derive these 7 fields from jamming phase/path/severity.  RF
            # evidence is generated below through signal/decoder/retry/session
            # fields, not through context-only shortcuts.
            context_profile = nominal_timeline[window_id]
            validate_context_profile(context_profile)
            semantic_defaults = {f: context_profile[f] for f in ("message_type", "service_context", "command_intent")}

            message_type = semantic_defaults.get("message_type", "TELEMETRY")
            service_context = semantic_defaults.get("service_context", "TTC_TELEMETRY")

            episode = {
                "scenario_id": "RF_LINK_JAMMING_SCENARIO_0001",
                "episode_id": episode_id,
                "episode_type": path_scenario,
                "target_paths": target_paths,
                "episode_step": step,
                "episode_length": duration,
                "episode_length_bucket": episode_length_bucket,
                "elapsed_seconds": elapsed_seconds,
                "episode_progress": episode_progress,
                "phase_boundary_pre_end": phase_boundaries["pre_end"],
                "phase_boundary_onset_end": phase_boundaries["onset_end"],
                "phase_boundary_mild_end": phase_boundaries["mild_end"],
                "phase_boundary_moderate_end": phase_boundaries["moderate_end"],
                "phase_boundary_severe_end": phase_boundaries["severe_end"],
                "jamming_phase": phase,
                "jamming_severity": severity,
                "jamming_type": jamming_type,
                "jamming_case_type": jamming_case_type,
                "shared_context_profile": context_profile.get("shared_context_profile", "UNKNOWN"),
                "shared_context_family": context_profile.get("shared_context_family", "UNKNOWN"),

                **{field: context_profile[field] for field in CONTEXT_FIELD_NAMES},
                "message_type": message_type,
                "service_context": service_context,
                "command_intent": semantic_defaults["command_intent"],
            }

            timeline.append(episode)
            window_id += 1

        episode_index += 1

    return timeline

def apply_ttc_jamming(raw: Dict[str, Any], severity: str) -> None:
    if severity == "MILD":
        raw["ttc_snr_db"] = uniform_float(10.5, 14.5, 3)
        raw["ttc_frame_loss_rate"] = uniform_float(0.018, 0.060, 5)
        raw["ttc_crc_error_count"] = uniform_int(0, 3)
        raw["ttc_decode_fail_count"] = weighted_choice([0, 1], [0.90, 0.10])
        raw["ttc_sync_loss_count"] = 0
        raw["ttc_retry_count"] = uniform_int(1, 5)
        raw["ttc_timeout_count"] = weighted_choice([0, 1], [0.86, 0.14])
        raw["ttc_rtt_ms"] = uniform_float(450, 1200, 2)

    elif severity == "MODERATE":
        raw["ttc_snr_db"] = uniform_float(5.0, 10.8, 3)
        raw["ttc_frame_loss_rate"] = uniform_float(0.070, 0.220, 5)
        raw["ttc_crc_error_count"] = uniform_int(2, 7)
        raw["ttc_decode_fail_count"] = weighted_choice([0, 1, 2], [0.25, 0.55, 0.20])
        raw["ttc_sync_loss_count"] = 0
        raw["ttc_retry_count"] = uniform_int(4, 10)
        raw["ttc_timeout_count"] = weighted_choice([0, 1, 2], [0.18, 0.62, 0.20])
        raw["ttc_rtt_ms"] = uniform_float(850, 2600, 2)

    elif severity == "SEVERE":
        raw["ttc_snr_db"] = uniform_float(-1.5, 5.8, 3)
        raw["ttc_frame_loss_rate"] = uniform_float(0.180, 0.520, 5)
        raw["ttc_crc_error_count"] = uniform_int(5, 15)
        raw["ttc_decode_fail_count"] = uniform_int(2, 5)
        raw["ttc_sync_loss_count"] = weighted_choice([0, 1, 2], [0.62, 0.28, 0.10])
        raw["ttc_retry_count"] = uniform_int(8, 18)
        raw["ttc_timeout_count"] = uniform_int(2, 7)
        raw["ttc_rtt_ms"] = uniform_float(2200, 6000, 2)

    elif severity == "OUTAGE":
        raw["ttc_snr_db"] = uniform_float(-6.0, 1.5, 3)
        raw["ttc_frame_loss_rate"] = uniform_float(0.380, 0.850, 5)
        raw["ttc_crc_error_count"] = uniform_int(10, 24)
        raw["ttc_decode_fail_count"] = uniform_int(4, 8)
        raw["ttc_sync_loss_count"] = weighted_choice([0, 1, 2, 3], [0.25, 0.35, 0.25, 0.15])
        raw["ttc_retry_count"] = uniform_int(14, 28)
        raw["ttc_timeout_count"] = uniform_int(6, 14)
        raw["ttc_rtt_ms"] = uniform_float(4000, 9000, 2)

def apply_iot_jamming(raw: Dict[str, Any], severity: str) -> None:
    if severity == "MILD":
        raw["iot_snr_db"] = uniform_float(-11.5, -5.5, 3)
        raw["iot_frame_loss_rate"] = uniform_float(0.055, 0.125, 5)
        raw["iot_crc_error_count"] = uniform_int(1, 4)
        raw["iot_decode_fail_count"] = weighted_choice([0, 1], [0.86, 0.14])
        raw["iot_sync_loss_count"] = 0
        raw["iot_retry_count"] = uniform_int(2, 7)
        raw["iot_timeout_count"] = weighted_choice([0, 1], [0.78, 0.22])

    elif severity == "MODERATE":
        raw["iot_snr_db"] = uniform_float(-18.0, -10.5, 3)
        raw["iot_frame_loss_rate"] = uniform_float(0.110, 0.320, 5)
        raw["iot_crc_error_count"] = uniform_int(4, 11)
        raw["iot_decode_fail_count"] = weighted_choice([0, 1, 2, 3], [0.15, 0.45, 0.30, 0.10])
        raw["iot_sync_loss_count"] = weighted_choice([0, 1], [0.90, 0.10])
        raw["iot_retry_count"] = uniform_int(6, 15)
        raw["iot_timeout_count"] = uniform_int(1, 5)

    elif severity == "SEVERE":
        raw["iot_snr_db"] = uniform_float(-26.0, -16.0, 3)
        raw["iot_frame_loss_rate"] = uniform_float(0.280, 0.650, 5)
        raw["iot_crc_error_count"] = uniform_int(8, 22)
        raw["iot_decode_fail_count"] = uniform_int(3, 7)
        raw["iot_sync_loss_count"] = weighted_choice([0, 1, 2], [0.55, 0.30, 0.15])
        raw["iot_retry_count"] = uniform_int(12, 26)
        raw["iot_timeout_count"] = uniform_int(3, 9)

    elif severity == "OUTAGE":
        raw["iot_snr_db"] = uniform_float(-32.0, -22.0, 3)
        raw["iot_frame_loss_rate"] = uniform_float(0.520, 0.950, 5)
        raw["iot_crc_error_count"] = uniform_int(14, 30)
        raw["iot_decode_fail_count"] = uniform_int(5, 10)
        raw["iot_sync_loss_count"] = weighted_choice([0, 1, 2, 3], [0.20, 0.30, 0.30, 0.20])
        raw["iot_retry_count"] = uniform_int(20, 38)
        raw["iot_timeout_count"] = uniform_int(7, 16)

def apply_uav_jamming(raw: Dict[str, Any], severity: str) -> None:
    if severity == "MILD":
        raw["uav_sinr_db"] = uniform_float(0.5, 6.0, 3)
        raw["uav_frame_loss_rate"] = uniform_float(0.035, 0.115, 5)
        raw["uav_crc_error_count"] = uniform_int(1, 4)
        raw["uav_decode_fail_count"] = weighted_choice([0, 1], [0.88, 0.12])
        raw["uav_sync_loss_count"] = 0
        raw["uav_retry_count"] = uniform_int(2, 7)
        raw["uav_timeout_count"] = weighted_choice([0, 1], [0.80, 0.20])

    elif severity == "MODERATE":
        raw["uav_sinr_db"] = uniform_float(-4.5, 1.5, 3)
        raw["uav_frame_loss_rate"] = uniform_float(0.090, 0.300, 5)
        raw["uav_crc_error_count"] = uniform_int(4, 12)
        raw["uav_decode_fail_count"] = weighted_choice([0, 1, 2, 3], [0.16, 0.46, 0.28, 0.10])
        raw["uav_sync_loss_count"] = weighted_choice([0, 1], [0.90, 0.10])
        raw["uav_retry_count"] = uniform_int(6, 15)
        raw["uav_timeout_count"] = uniform_int(1, 5)

    elif severity == "SEVERE":
        raw["uav_sinr_db"] = uniform_float(-12.0, -4.5, 3)
        raw["uav_frame_loss_rate"] = uniform_float(0.260, 0.620, 5)
        raw["uav_crc_error_count"] = uniform_int(8, 22)
        raw["uav_decode_fail_count"] = uniform_int(3, 7)
        raw["uav_sync_loss_count"] = weighted_choice([0, 1, 2], [0.58, 0.28, 0.14])
        raw["uav_retry_count"] = uniform_int(12, 26)
        raw["uav_timeout_count"] = uniform_int(3, 9)

    elif severity == "OUTAGE":
        raw["uav_sinr_db"] = uniform_float(-18.0, -9.0, 3)
        raw["uav_frame_loss_rate"] = uniform_float(0.500, 0.920, 5)
        raw["uav_crc_error_count"] = uniform_int(14, 30)
        raw["uav_decode_fail_count"] = uniform_int(5, 10)
        raw["uav_sync_loss_count"] = weighted_choice([0, 1, 2, 3], [0.22, 0.28, 0.30, 0.20])
        raw["uav_retry_count"] = uniform_int(20, 38)
        raw["uav_timeout_count"] = uniform_int(7, 16)

def apply_isl_jamming(raw: Dict[str, Any], severity: str) -> None:
    if severity == "MILD":
        raw["isl_snr_db"] = uniform_float(9.5, 13.5, 3)
        raw["isl_frame_loss_rate"] = uniform_float(0.020, 0.070, 5)
        raw["isl_crc_error_count"] = uniform_int(0, 3)
        raw["isl_decode_fail_count"] = weighted_choice([0, 1], [0.91, 0.09])
        raw["isl_sync_loss_count"] = 0
        raw["isl_retry_count"] = uniform_int(1, 5)
        raw["isl_timeout_count"] = weighted_choice([0, 1], [0.86, 0.14])

    elif severity == "MODERATE":
        raw["isl_snr_db"] = uniform_float(5.0, 10.0, 3)
        raw["isl_frame_loss_rate"] = uniform_float(0.070, 0.220, 5)
        raw["isl_crc_error_count"] = uniform_int(3, 8)
        raw["isl_decode_fail_count"] = weighted_choice([0, 1, 2], [0.22, 0.56, 0.22])
        raw["isl_sync_loss_count"] = weighted_choice([0, 1], [0.92, 0.08])
        raw["isl_retry_count"] = uniform_int(4, 11)
        raw["isl_timeout_count"] = weighted_choice([0, 1, 2], [0.16, 0.64, 0.20])

    elif severity == "SEVERE":
        raw["isl_snr_db"] = uniform_float(-1.5, 5.5, 3)
        raw["isl_frame_loss_rate"] = uniform_float(0.180, 0.500, 5)
        raw["isl_crc_error_count"] = uniform_int(6, 16)
        raw["isl_decode_fail_count"] = uniform_int(2, 5)
        raw["isl_sync_loss_count"] = weighted_choice([0, 1, 2], [0.60, 0.30, 0.10])
        raw["isl_retry_count"] = uniform_int(8, 18)
        raw["isl_timeout_count"] = uniform_int(2, 7)

    elif severity == "OUTAGE":
        raw["isl_snr_db"] = uniform_float(-6.0, 1.0, 3)
        raw["isl_frame_loss_rate"] = uniform_float(0.380, 0.850, 5)
        raw["isl_crc_error_count"] = uniform_int(10, 25)
        raw["isl_decode_fail_count"] = uniform_int(4, 8)
        raw["isl_sync_loss_count"] = weighted_choice([0, 1, 2, 3], [0.25, 0.35, 0.25, 0.15])
        raw["isl_retry_count"] = uniform_int(14, 28)
        raw["isl_timeout_count"] = uniform_int(6, 14)

def apply_gnss_jamming(raw: Dict[str, Any], severity: str) -> None:
    if severity == "MILD":
        raw["gnss_cn0_mean_dbhz"] = uniform_float(33.0, 38.5, 3)
        raw["gnss_cn0_std_db"] = uniform_float(2.3, 5.0, 3)
        raw["gnss_tracking_jitter"] = uniform_float(0.12, 0.30, 4)
        raw["gnss_carrier_lock_ratio"] = uniform_float(0.88, 0.970, 4)
        raw["gnss_lock_loss_count"] = weighted_choice([0, 1], [0.88, 0.12])
        raw["gnss_cycle_slip_count"] = uniform_int(1, 4)
        raw["fix_valid"] = weighted_choice([1, 0], [0.94, 0.06])

    elif severity == "MODERATE":
        raw["gnss_cn0_mean_dbhz"] = uniform_float(24.0, 33.0, 3)
        raw["gnss_cn0_std_db"] = uniform_float(4.5, 8.0, 3)
        raw["gnss_tracking_jitter"] = uniform_float(0.25, 0.65, 4)
        raw["gnss_carrier_lock_ratio"] = uniform_float(0.62, 0.88, 4)
        raw["gnss_lock_loss_count"] = uniform_int(0, 4)
        raw["gnss_cycle_slip_count"] = uniform_int(3, 10)
        raw["fix_valid"] = weighted_choice([1, 0], [0.52, 0.48])

    elif severity == "SEVERE":
        raw["gnss_cn0_mean_dbhz"] = uniform_float(14.0, 25.0, 3)
        raw["gnss_cn0_std_db"] = uniform_float(6.5, 12.0, 3)
        raw["gnss_tracking_jitter"] = uniform_float(0.55, 1.30, 4)
        raw["gnss_carrier_lock_ratio"] = uniform_float(0.18, 0.62, 4)
        raw["gnss_lock_loss_count"] = uniform_int(2, 12)
        raw["gnss_cycle_slip_count"] = uniform_int(8, 24)
        raw["fix_valid"] = weighted_choice([1, 0], [0.16, 0.84])

    elif severity == "OUTAGE":
        raw["gnss_cn0_mean_dbhz"] = uniform_float(5.0, 16.0, 3)
        raw["gnss_cn0_std_db"] = uniform_float(8.0, 15.0, 3)
        raw["gnss_tracking_jitter"] = uniform_float(0.85, 2.00, 4)
        raw["gnss_carrier_lock_ratio"] = uniform_float(0.00, 0.35, 4)
        raw["gnss_lock_loss_count"] = uniform_int(6, 20)
        raw["gnss_cycle_slip_count"] = uniform_int(12, 30)
        raw["fix_valid"] = 0

def apply_rf_jamming_effects(raw: Dict[str, Any], episode: Dict[str, Any]) -> None:
    """Perturb the episode's fixed targets, independently of the observed path.

    Proposed absolute impairment values cannot improve the SAME baseline
    realization. The before values are retained in baseline_* audit columns by
    the caller. This monotone counterfactual is a modeling convention, not a
    proof of a calibrated physical RF channel.
    """
    targets = get_target_paths(episode)
    severity = episode["jamming_severity"]
    if severity not in {"NONE", "MILD", "MODERATE", "SEVERE", "OUTAGE"}:
        raise ValueError(f"Unknown jamming severity: {severity}")
    if episode.get("jamming_phase") == "PRE_ATTACK" and severity != "NONE":
        raise ValueError("PRE_ATTACK cannot contain an applied jamming effect.")
    if severity == "NONE":
        return
    effect_functions = {
        "GATEWAY_LEO": apply_ttc_jamming, "LEO_LEO": apply_isl_jamming,
        "IOT_LEO": apply_iot_jamming, "UAV_LEO": apply_uav_jamming,
        "GNSS_LEO": apply_gnss_jamming,
    }
    lower_is_worse = {"ttc_snr_db", "isl_snr_db", "iot_snr_db", "uav_sinr_db",
                      "gnss_cn0_mean_dbhz", "gnss_carrier_lock_ratio", "fix_valid"}
    before = dict(raw)
    for path in targets:
        effect_functions[path](raw, severity)
    for field, value in before.items():
        if raw[field] != value:
            raw[field] = min(raw[field], value) if field in lower_is_worse else max(raw[field], value)

def selected_link_metrics(raw: Dict[str, Any]) -> tuple[float, int, int, int]:
    """Loss, decoder failures, retries, timeouts on the observed RF data link."""
    path = raw["raw_path_type"]
    prefixes = list(PATH_PREFIX.values()) if path == "MULTI_PATH" else ([PATH_PREFIX[path]] if path in PATH_PREFIX else [])
    if not prefixes:
        return 0.0, 0, 0, 0
    return (max(raw[p + "_frame_loss_rate"] for p in prefixes),
            sum(raw[p + "_decode_fail_count"] for p in prefixes),
            sum(raw[p + "_retry_count"] for p in prefixes),
            sum(raw[p + "_timeout_count"] for p in prefixes))

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

RAW_COUNT_FIELDS += [p + "_sync_loss_count" for p in PATH_PREFIX.values()]

RAW_FLAG_FIELDS = [
    "fix_valid", "duplicate_flag", "accepted_sequence_flag",
    "source_unverified_benign_flag", "valid_looking_benign_flag", "auth_present",
    "auth_valid", "integrity_valid", "source_authorized", "session_valid",
    "accepted_flag", *COMMAND_OUTCOME_FLAGS.values(), "session_active",
    "session_expired_flag", "link_degraded_flag", "link_loss_flag", "transition_allowed",
    "unexpected_transition_flag", "forbidden_transition_flag", "delayed_transition_flag",
    "incomplete_transition_flag", "state_changed_flag", "raw_attack_active", "raw_observation_on_target",
]

import shared_nominal as nominal
from shared_nominal import (normalize_context, normalize_sequence_freshness, normalize_control_security, normalize_message_control, normalize_signal_quality, normalize_tracking_stability, normalize_navigation_consistency, normalize_timing_consistency, normalize_decoder_status, normalize_command_execution, normalize_retry_timeout, normalize_session_state, normalize_state_transition, normalize_raw_to_observable)

def apply_delivery_delta(raw, before):
    """Only added selected-link impairment may worsen nominal delivery."""
    loss, failures, retries, timeouts = selected_link_metrics(raw)
    old_loss, old_failures, old_retries, old_timeouts = selected_link_metrics(before)
    if (loss,failures,retries,timeouts)==(old_loss,old_failures,old_retries,old_timeouts):return
    raw['link_degraded_flag']=max(raw['link_degraded_flag'],int(loss>=.07 or failures>0 or retries>=4 or timeouts>=2))
    raw['link_loss_flag']=max(raw['link_loss_flag'],int(loss>=.5 and timeouts>=3))
    raw['session_reset_count']=max(raw['session_reset_count'],int(raw['session_active'] and timeouts>old_timeouts and timeouts>=3 and flag_choice(.20)))
    if not has_command_activity(raw):return
    # Additional loss cannot turn a rejected, blocked or held nominal command
    # into a newly executed command. No queue or retry execution is fabricated.
    timeout_p=min(.85,.08*max(0,timeouts-old_timeouts)+.25*max(0,loss-old_loss))
    hold_p=min(.60,.04*max(0,retries-old_retries)+.05*max(0,failures-old_failures))
    if not any(raw[f] for f in ('rejected_flag','blocked_flag','failed_flag','command_timeout_flag')):
        if RNG.random()<timeout_p:nominal.set_command_outcome(raw,'TIMEOUT')
        elif not raw['held_flag'] and RNG.random()<hold_p:nominal.set_command_outcome(raw,'HELD')
        if raw['held_flag'] or raw['command_timeout_flag']:
            raw['state_changed_flag']=0;raw['delayed_transition_flag']=1
    if loss>old_loss and loss>=.07 and flag_choice(min(.9,loss-old_loss)):
        raw['sequence_gap']=max(raw['sequence_gap'],2)


def generate_rf_jamming_raw_window(window_id, timestamp, episode, *, nominal_raw=None, attacks_enabled=True):
    raw=nominal.copy_nominal_window(nominal_raw,episode,LABEL_RF_LINK_JAMMING)
    targets=get_target_paths(episode)
    for f in ('jamming_phase','jamming_severity','jamming_type','jamming_case_type'):raw['raw_'+f]=episode[f]
    raw['raw_target_paths']='|'.join(targets)
    raw['raw_observation_on_target']=int(episode['path_type'] in targets or episode['path_type']=='MULTI_PATH')
    raw['raw_attack_active']=int(attacks_enabled and episode['jamming_severity']!='NONE')
    if raw['raw_attack_active']:
        before=dict(raw);apply_rf_jamming_effects(raw,episode)
        apply_delivery_delta(raw,before)
        nominal.refresh_environment_semantics(raw)
    return raw


def validate_attack_rows(df):
    nominal.require_nominal_when(df,df.raw_attack_active.eq(0))
    groups={'GATEWAY_LEO':('ttc_',),'LEO_LEO':('isl_',),'IOT_LEO':('iot_',),'UAV_LEO':('uav_',),
            'GNSS_LEO':('gnss_','position_jump_m','velocity_jump_mps','pseudorange_residual_m','time_residual_ns','clock_drift_ppm','fix_valid')}
    for path,prefixes in groups.items():
        fields=[c for c in df if c.startswith(prefixes) and 'baseline_'+c in df]
        off=~df.raw_target_paths.str.split('|',regex=False).map(lambda p:path in p)
        if any(not df.loc[off,c].equals(df.loc[off,'baseline_'+c]) for c in fields):
            raise ValueError(f'RF intervention changed an off-target receiver: {path}')
    for field in ('auth_present','auth_valid','integrity_valid','source_authorized','session_valid','raw_command_intent'):
        if not df[field].equals(df['baseline_'+field]):raise ValueError('RF synthesized unrelated command/security evidence.')

MAKER_NUMBER = 2
CLASS_LABEL = LABEL_RF_LINK_JAMMING
CODE_VERSION = nominal.CODE_VERSION
TIMELINE_BUILDER = 'build_rf_jamming_episode_timeline'
RAW_GENERATOR = 'generate_rf_jamming_raw_window'
OBSERVABLE_DOMAINS = nominal.OBSERVABLE_DOMAINS

def build_rf_jamming_dataset(num_rows: int = DEFAULT_NUM_ROWS, window_seconds: int = DEFAULT_WINDOW_SECONDS,
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
    nominal.run_cli(globals(), 'build_rf_jamming_dataset')
