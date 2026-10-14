"""Shared context sampler for the TAES six-class observable makers (v4).

This module is intentionally label-agnostic.  All six makers should import
GLOBAL_CONTEXT_PROFILE_POOL and sample_shared_context_profile() from this file
instead of defining class-specific context templates.  A context profile is an
operational interpretation frame, not attack evidence.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

CONTEXT_FIELD_NAMES: List[str] = [
    "observation_stage",
    "path_type",
    "direction",
    "source_role",
    "destination_role",
    "protocol_context",
    "operational_phase",
]

# Keep context vocabulary strictly operational.  Do not add attack-phase,
# attack-intensity, replay-stage, spoofing-stage, or compromise-stage tokens.
ALLOWED_OBSERVATION_STAGES = {
    "RF_RECEIVER",
    "GNSS_RECEIVER",
    "PROTOCOL_HANDLER",
    "SESSION_MANAGER",
    "COMMAND_HANDLER",
    "ONBOARD_MONITOR",
}

ALLOWED_OPERATIONAL_PHASES = {
    "CONTACT_WINDOW",
    "COMMAND_SESSION",
    "ROUTINE_MONITORING",
    "GNSS_NAVIGATION",
    "DATA_RELAY",
    "PAYLOAD_OPERATION",
    "SAFE_MODE_MONITORING",
    "LINK_MAINTENANCE",
}

FORBIDDEN_CONTEXT_TOKENS = {
    "PRE_ATTACK",
    "ONSET",
    "ATTACK_ACTIVE",
    "PULL_OFF",
    "SECURITY_CHECK_FAILURE",
    "REPLAY_ATTEMPT",
    "CONTROL_DRIFT",
    "SPOOFING_DRIFT",
    "JAMMING_DEGRADATION",
    "COMPROMISE_PROGRESS",
    "NORMAL",
    "SUSPICIOUS",
    "RECOVERY",
}

# family is operational only.  It must never be a class label.
# The global weights are intentionally broad and should be reused by all six
# makers unless a separate experiment explicitly studies context shift.
GLOBAL_CONTEXT_PROFILE_POOL: List[Dict[str, Any]] = [
    {
        "name": "TTC_UPLINK_COMMAND_HANDLER",
        "family": "TTC_CONTROL",
        "weight": 0.115,
        "profile": {
            "observation_stage": "COMMAND_HANDLER",
            "path_type": "GATEWAY_LEO",
            "direction": "GROUND_TO_SPACE",
            "source_role": "GROUND_GATEWAY",
            "destination_role": "MONITORED_LEO",
            "protocol_context": "CCSDS_TTC",
            "operational_phase": "COMMAND_SESSION",
        },
    },
    {
        "name": "TTC_UPLINK_SESSION_MANAGER",
        "family": "TTC_CONTROL",
        "weight": 0.085,
        "profile": {
            "observation_stage": "SESSION_MANAGER",
            "path_type": "GATEWAY_LEO",
            "direction": "GROUND_TO_SPACE",
            "source_role": "GROUND_CONTROL_SERVICE",
            "destination_role": "MONITORED_LEO",
            "protocol_context": "CCSDS_TTC",
            "operational_phase": "COMMAND_SESSION",
        },
    },
    {
        "name": "TTC_UPLINK_PROTOCOL_CONTACT",
        "family": "TTC_CONTROL",
        "weight": 0.070,
        "profile": {
            "observation_stage": "PROTOCOL_HANDLER",
            "path_type": "GATEWAY_LEO",
            "direction": "GROUND_TO_SPACE",
            "source_role": "GROUND_GATEWAY",
            "destination_role": "MONITORED_LEO",
            "protocol_context": "CCSDS_TTC",
            "operational_phase": "CONTACT_WINDOW",
        },
    },
    {
        "name": "TTC_DOWNLINK_TELEMETRY_CONTACT",
        "family": "TTC_TELEMETRY",
        "weight": 0.075,
        "profile": {
            "observation_stage": "PROTOCOL_HANDLER",
            "path_type": "GATEWAY_LEO",
            "direction": "SPACE_TO_GROUND",
            "source_role": "MONITORED_LEO",
            "destination_role": "GROUND_GATEWAY",
            "protocol_context": "CCSDS_TTC",
            "operational_phase": "CONTACT_WINDOW",
        },
    },
    {
        "name": "TTC_RF_RECEIVER_CONTACT",
        "family": "RF_LINK",
        "weight": 0.070,
        "profile": {
            "observation_stage": "RF_RECEIVER",
            "path_type": "GATEWAY_LEO",
            "direction": "GROUND_TO_SPACE",
            "source_role": "GROUND_GATEWAY",
            "destination_role": "MONITORED_LEO",
            "protocol_context": "CCSDS_TTC",
            "operational_phase": "CONTACT_WINDOW",
        },
    },
    {
        "name": "ISL_COMMAND_RELAY",
        "family": "TTC_CONTROL",
        "weight": 0.060,
        "profile": {
            "observation_stage": "COMMAND_HANDLER",
            "path_type": "LEO_LEO",
            "direction": "SPACE_TO_SPACE",
            "source_role": "PEER_LEO",
            "destination_role": "MONITORED_LEO",
            "protocol_context": "CCSDS_ISL",
            "operational_phase": "COMMAND_SESSION",
        },
    },
    {
        "name": "ISL_PROTOCOL_RELAY",
        "family": "ISL_RELAY",
        "weight": 0.065,
        "profile": {
            "observation_stage": "PROTOCOL_HANDLER",
            "path_type": "LEO_LEO",
            "direction": "SPACE_TO_SPACE",
            "source_role": "PEER_LEO",
            "destination_role": "MONITORED_LEO",
            "protocol_context": "CCSDS_ISL",
            "operational_phase": "DATA_RELAY",
        },
    },
    {
        "name": "ISL_RF_RECEIVER_RELAY",
        "family": "RF_LINK",
        "weight": 0.055,
        "profile": {
            "observation_stage": "RF_RECEIVER",
            "path_type": "LEO_LEO",
            "direction": "SPACE_TO_SPACE",
            "source_role": "PEER_LEO",
            "destination_role": "MONITORED_LEO",
            "protocol_context": "CCSDS_ISL",
            "operational_phase": "DATA_RELAY",
        },
    },
    {
        "name": "MIOTY_ACCESS_PROTOCOL",
        "family": "IOT_ACCESS",
        "weight": 0.060,
        "profile": {
            "observation_stage": "PROTOCOL_HANDLER",
            "path_type": "IOT_LEO",
            "direction": "USER_TO_SPACE",
            "source_role": "IOT_TERMINAL",
            "destination_role": "MONITORED_LEO",
            "protocol_context": "MIOTY",
            "operational_phase": "DATA_RELAY",
        },
    },
    {
        "name": "MIOTY_ACCESS_RF_RECEIVER",
        "family": "RF_LINK",
        "weight": 0.050,
        "profile": {
            "observation_stage": "RF_RECEIVER",
            "path_type": "IOT_LEO",
            "direction": "USER_TO_SPACE",
            "source_role": "IOT_TERMINAL",
            "destination_role": "MONITORED_LEO",
            "protocol_context": "MIOTY",
            "operational_phase": "DATA_RELAY",
        },
    },
    {
        "name": "NTN_ACCESS_SESSION",
        "family": "NTN_ACCESS",
        "weight": 0.060,
        "profile": {
            "observation_stage": "SESSION_MANAGER",
            "path_type": "UAV_LEO",
            "direction": "USER_TO_SPACE",
            "source_role": "UAV_TERMINAL",
            "destination_role": "MONITORED_LEO",
            "protocol_context": "3GPP_NTN",
            "operational_phase": "DATA_RELAY",
        },
    },
    {
        "name": "NTN_ACCESS_RF_RECEIVER",
        "family": "RF_LINK",
        "weight": 0.050,
        "profile": {
            "observation_stage": "RF_RECEIVER",
            "path_type": "UAV_LEO",
            "direction": "USER_TO_SPACE",
            "source_role": "UAV_TERMINAL",
            "destination_role": "MONITORED_LEO",
            "protocol_context": "3GPP_NTN",
            "operational_phase": "DATA_RELAY",
        },
    },
    {
        "name": "GNSS_RECEIVER_NAVIGATION",
        "family": "GNSS_PNT",
        "weight": 0.090,
        "profile": {
            "observation_stage": "GNSS_RECEIVER",
            "path_type": "GNSS_LEO",
            "direction": "GNSS_TO_SPACE",
            "source_role": "GNSS_CONSTELLATION",
            "destination_role": "MONITORED_LEO",
            "protocol_context": "GNSS_NAV",
            "operational_phase": "GNSS_NAVIGATION",
        },
    },
    {
        "name": "GNSS_ONBOARD_MONITOR",
        "family": "GNSS_PNT",
        "weight": 0.055,
        "profile": {
            "observation_stage": "ONBOARD_MONITOR",
            "path_type": "GNSS_LEO",
            "direction": "GNSS_TO_SPACE",
            "source_role": "GNSS_CONSTELLATION",
            "destination_role": "MONITORED_LEO",
            "protocol_context": "GNSS_NAV",
            "operational_phase": "GNSS_NAVIGATION",
        },
    },
    {
        "name": "ONBOARD_PAYLOAD_MONITOR",
        "family": "ONBOARD_INTERNAL",
        "weight": 0.055,
        "profile": {
            "observation_stage": "ONBOARD_MONITOR",
            "path_type": "INTERNAL_ONBOARD",
            "direction": "INTERNAL",
            "source_role": "MONITORED_LEO",
            "destination_role": "OBC",
            "protocol_context": "ONBOARD_INTERNAL",
            "operational_phase": "PAYLOAD_OPERATION",
        },
    },
    {
        "name": "ONBOARD_ROUTINE_MONITOR",
        "family": "ONBOARD_INTERNAL",
        "weight": 0.045,
        "profile": {
            "observation_stage": "ONBOARD_MONITOR",
            "path_type": "INTERNAL_ONBOARD",
            "direction": "INTERNAL",
            "source_role": "MONITORED_LEO",
            "destination_role": "OBC",
            "protocol_context": "ONBOARD_INTERNAL",
            "operational_phase": "ROUTINE_MONITORING",
        },
    },
]

# Optional operational-family aliases.  These are not labels.
COMPATIBLE_FAMILY_ALIASES: Dict[str, Sequence[str]] = {
    "TTC_COMPATIBLE": ("TTC_CONTROL", "TTC_TELEMETRY", "ISL_RELAY"),
    "GNSS_COMPATIBLE": ("GNSS_PNT",),
    "RF_COMPATIBLE": ("RF_LINK", "TTC_TELEMETRY", "ISL_RELAY", "IOT_ACCESS", "NTN_ACCESS"),
    "ACCESS_COMPATIBLE": ("IOT_ACCESS", "NTN_ACCESS", "RF_LINK"),
    "ONBOARD_COMPATIBLE": ("ONBOARD_INTERNAL", "TTC_CONTROL"),
}


def _get_rng(rng: Any | None) -> Any:
    return rng if rng is not None else np.random.default_rng()


def _normalize_weights(weights: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(weights), dtype=float)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError("Weights must be a non-empty one-dimensional sequence.")
    total = float(arr.sum())
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")
    return arr / total


def _families_from_compatible(compatible_family: str | Sequence[str] | None) -> set[str] | None:
    if compatible_family is None:
        return None
    if isinstance(compatible_family, str):
        values = COMPATIBLE_FAMILY_ALIASES.get(compatible_family, (compatible_family,))
    else:
        values = []
        for item in compatible_family:
            values.extend(COMPATIBLE_FAMILY_ALIASES.get(str(item), (str(item),)))
    return set(values)


def validate_context_profile(profile: Mapping[str, Any]) -> None:
    missing = [field for field in CONTEXT_FIELD_NAMES if field not in profile]
    if missing:
        raise ValueError(f"Context profile is missing required fields: {missing}")

    for field in CONTEXT_FIELD_NAMES:
        token = str(profile[field])
        if token in FORBIDDEN_CONTEXT_TOKENS:
            raise ValueError(f"Forbidden context token in {field}: {token}")

    stage = str(profile["observation_stage"])
    phase = str(profile["operational_phase"])
    if stage not in ALLOWED_OBSERVATION_STAGES:
        raise ValueError(f"Non-operational observation_stage token: {stage}")
    if phase not in ALLOWED_OPERATIONAL_PHASES:
        raise ValueError(f"Non-operational operational_phase token: {phase}")


def validate_global_context_pool() -> None:
    names = [entry["name"] for entry in GLOBAL_CONTEXT_PROFILE_POOL]
    duplicate_names = [name for name, count in Counter(names).items() if count > 1]
    if duplicate_names:
        raise ValueError(f"Duplicate context profile names: {duplicate_names}")
    for entry in GLOBAL_CONTEXT_PROFILE_POOL:
        if "label" in entry or "scenario" in entry:
            raise ValueError("Context pool entries must not contain label/scenario keys.")
        validate_context_profile(entry["profile"])


def sample_shared_context_profile(
    rng: Any | None = None,
    compatible_family: str | Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Sample one label-agnostic context profile.

    compatible_family is an optional operational filter only.  It must never be
    used as a class label or attack-specific selector.  To minimize context-only
    leakage, all six makers should preferentially call this function with
    compatible_family=None or with the same benchmark-level family mixture.
    """
    validate_global_context_pool()
    rng = _get_rng(rng)
    allowed_families = _families_from_compatible(compatible_family)
    candidates = [
        entry for entry in GLOBAL_CONTEXT_PROFILE_POOL
        if allowed_families is None or entry["family"] in allowed_families
    ]
    if not candidates:
        raise ValueError(f"No context profiles match compatible_family={compatible_family!r}")

    probs = _normalize_weights(entry["weight"] for entry in candidates)
    idx = int(rng.choice(len(candidates), p=probs))
    selected = candidates[idx]
    profile = dict(selected["profile"])
    validate_context_profile(profile)
    profile["shared_context_profile"] = str(selected["name"])
    profile["shared_context_family"] = str(selected["family"])
    return profile


def nominal_semantics_for_context(profile: Mapping[str, Any], rng: Any | None = None) -> Dict[str, str]:
    """Return benign default message/control semantics compatible with context.

    The returned values are defaults.  Individual makers may alter evidence
    fields according to their label, but should not alter the seven context
    fields after sampling.
    """
    rng = _get_rng(rng)
    path_type = str(profile["path_type"])
    direction = str(profile["direction"])
    protocol = str(profile["protocol_context"])
    phase = str(profile["operational_phase"])

    if phase == "COMMAND_SESSION" and direction in {"GROUND_TO_SPACE", "SPACE_TO_SPACE"} and protocol in {"CCSDS_TTC", "CCSDS_ISL"}:
        command_intents = ["STATUS_REQUEST", "TIME_SYNC", "CONFIG_UPDATE", "NOMINAL_CONTROL", "HOUSEKEEPING", "ORBIT_CONTROL"]
        command_probs = _normalize_weights([0.32, 0.20, 0.14, 0.18, 0.10, 0.06])
        return {
            "message_type": str(rng.choice(["TELECOMMAND", "COMMAND_UPLINK"], p=_normalize_weights([0.82, 0.18]))),
            "service_context": str(rng.choice(["TTC_CONTROL", "CONTROL_SESSION"], p=_normalize_weights([0.76, 0.24]))),
            "command_intent": str(rng.choice(command_intents, p=command_probs)),
        }

    if path_type == "GNSS_LEO" or protocol == "GNSS_NAV":
        return {
            "message_type": str(rng.choice(
                ["GNSS_OBSERVATION", "PNT_SOLUTION", "NAVIGATION_MESSAGE", "GNSS_TIMING_UPDATE", "NAVIGATION_STATE_UPDATE"],
                p=_normalize_weights([0.34, 0.26, 0.20, 0.12, 0.08]),
            )),
            "service_context": str(rng.choice(
                ["GNSS_PNT", "PNT_MONITORING", "TIMING_SYNCHRONIZATION", "NAVIGATION_FILTERING"],
                p=_normalize_weights([0.36, 0.34, 0.18, 0.12]),
            )),
            "command_intent": "NO_COMMAND",
        }

    if path_type == "IOT_LEO":
        return {"message_type": "IOT_REPORT", "service_context": "IOT_UPLINK", "command_intent": "NO_COMMAND"}

    if path_type == "UAV_LEO":
        return {"message_type": "UAV_TELEMETRY", "service_context": "UAV_NTN_SERVICE", "command_intent": "NO_COMMAND"}

    if path_type == "LEO_LEO":
        return {"message_type": "TELEMETRY", "service_context": "ISL_ROUTING", "command_intent": "NO_COMMAND"}

    if path_type == "INTERNAL_ONBOARD":
        return {"message_type": "TELEMETRY", "service_context": "PAYLOAD_DATA", "command_intent": "NO_COMMAND"}

    return {"message_type": "NOMINAL_MIXED_TRAFFIC", "service_context": "HOUSEKEEPING", "command_intent": "NO_COMMAND"}


def is_command_context(profile: Mapping[str, Any]) -> bool:
    return (
        str(profile.get("operational_phase")) == "COMMAND_SESSION"
        and str(profile.get("direction")) in {"GROUND_TO_SPACE", "SPACE_TO_SPACE"}
        and str(profile.get("protocol_context")) in {"CCSDS_TTC", "CCSDS_ISL"}
        and str(profile.get("path_type")) in {"GATEWAY_LEO", "LEO_LEO"}
    )
