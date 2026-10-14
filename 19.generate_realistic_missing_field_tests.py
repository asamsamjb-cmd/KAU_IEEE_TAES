#!/usr/bin/env python3
"""Step 18: label-independent structured missing-field stress data.

Run beside the numbered scripts, or use --project-root PATH.
Reads the COMPLETE shared_nominal_v2 Step 7 test split. Never modifies source splits.
Writes dataset/shared_nominal_v2/realistic_missing_field_tests/.
Use a new --output-dir to change settings; completed matching outputs are verified/reused.

The legacy 'realistic_missing' artifact names are retained for compatibility.
These are synthetic burst-masking conditions, NOT field-calibrated outage models.
Severity changes field count, target affected fraction and burst proposal length jointly.
Context and message/control fields are protected by design: this does not cover all
missingness mechanisms. One fixed mask realization is used; no robustness claim is
made by generation alone. Labels/episode-ID strings never select masks or RNG seeds.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
BASE_DIR = Path(__file__).resolve().parent
INPUT_CONTRACT_VERSION = "shared_nominal_v2"
PROTOCOL_VERSION = "label_independent_structured_missing_v2"
LABELS = ["NORMAL", "RF_LINK_JAMMING", "GNSS_SPOOFING", "UNAUTHORIZED_TELECOMMAND_INJECTION",
          "TELECOMMAND_REPLAY", "GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE"]

DEFAULT_RANDOM_SEED = 20260610

MISSING_PLACEHOLDER = "UNK"

MISSING_COUNTS = [2, 5, 8]

EXPECTED_OBSERVABLE_WINDOW_SECONDS = 1

DEFAULT_AFFECTED_FRACTION_BY_K: Dict[int, float] = {
    2: 0.25,
    5: 0.35,
    8: 0.45,
}

DEFAULT_BURST_LENGTH_RANGE_BY_K: Dict[int, Tuple[int, int]] = {
    2: (2, 4),
    5: (3, 6),
    8: (4, 8),
}

OBSERVATION_CONTEXT_FIELDS: List[str] = [
    "observation_stage",
    "path_type",
    "direction",
    "source_role",
    "destination_role",
    "protocol_context",
    "operational_phase",
]

PROTECTED_MESSAGE_CONTROL_FIELDS: List[str] = [
    "message_type",
    "service_context",
    "command_intent",
    "payload_or_parameter_context",
    "sequence_freshness_status",
    "control_security_state",
]

RF_GNSS_RECEIVER_FIELDS: List[str] = [
    "signal_quality_state",
    "tracking_stability",
    "navigation_consistency",
    "timing_consistency",
]

OPERATIONAL_TRACE_FIELDS: List[str] = [
    "decoder_status",
    "command_execution_result",
    "retry_timeout_state",
    "session_state",
    "state_transition_consistency",
]

FALLBACK_METADATA_FIELDS: List[str] = [
    "scenario_id",
    "episode_id",
    "episode_type",
    "episode_step",
    "window_id",
    "timestamp",
    "label",
]

FALLBACK_OBSERVABLE_FIELDS: List[str] = OBSERVATION_CONTEXT_FIELDS + PROTECTED_MESSAGE_CONTROL_FIELDS + RF_GNSS_RECEIVER_FIELDS + OPERATIONAL_TRACE_FIELDS

@dataclass(frozen=True)
class StructuredMissingConfig:
    missing_count: int
    affected_fraction: float
    burst_length_range: Tuple[int, int]

def get_config_for_missing_count(missing_count: int) -> StructuredMissingConfig:
    if missing_count <= 0:
        raise ValueError(f"missing_count must be positive, got {missing_count}")
    affected_fraction = DEFAULT_AFFECTED_FRACTION_BY_K.get(missing_count, min(0.50, 0.15 + 0.04 * missing_count))
    burst_range = DEFAULT_BURST_LENGTH_RANGE_BY_K.get(missing_count, (3, max(4, min(10, missing_count + 2))))
    return StructuredMissingConfig(
        missing_count=int(missing_count),
        affected_fraction=float(affected_fraction),
        burst_length_range=(int(burst_range[0]), int(burst_range[1])),
    )

def build_burst_mask(num_rows: int, config: StructuredMissingConfig, rng: random.Random) -> Tuple[np.ndarray, List[Dict[str, int]]]:
    if num_rows <= 0:
        return np.zeros(0, dtype=bool), []

    target = max(1, int(round(num_rows * config.affected_fraction)))
    target = min(target, num_rows)
    min_len, max_len = config.burst_length_range
    min_len = max(1, min(min_len, num_rows))
    max_len = max(min_len, min(max_len, num_rows))

    affected = np.zeros(num_rows, dtype=bool)
    bursts: List[Dict[str, int]] = []
    attempts = 0
    max_attempts = max(20, num_rows * 3)

    while int(affected.sum()) < target and attempts < max_attempts:
        attempts += 1
        burst_len = rng.randint(min_len, max_len)
        if burst_len >= num_rows:
            start = 0
        else:
            start = rng.randint(0, num_rows - burst_len)
        end_exclusive = min(num_rows, start + burst_len)
        before = int(affected.sum())
        affected[start:end_exclusive] = True
        after = int(affected.sum())
        if after > before:
            bursts.append({
                "start_pos": int(start),
                "end_pos_inclusive": int(end_exclusive - 1),
                "length_rows": int(end_exclusive - start),
            })

    if int(affected.sum()) < target:
        # Deterministic fallback: fill remaining positions with one contiguous block
        # around the first still-unaffected region.
        missing_positions = np.where(~affected)[0]
        if len(missing_positions):
            need = target - int(affected.sum())
            selected = missing_positions[:need]
            affected[selected] = True
            bursts.append({
                "start_pos": int(selected[0]),
                "end_pos_inclusive": int(selected[-1]),
                "length_rows": int(len(selected)),
            })

    # Audit the realized disjoint contiguous runs, including merged proposals.
    padded = np.r_[False, affected, False].astype(np.int8)
    starts, ends = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
    bursts = [{"start_pos": int(a), "end_pos_inclusive": int(b-1), "length_rows": int(b-a)}
              for a, b in zip(starts, ends)]
    return affected, bursts

def build_realistic_missing_testset(
    base_df: pd.DataFrame,
    schema: Dict[str, Any],
    missing_count: int,
    placeholder: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    metadata_fields: List[str] = schema["metadata_fields"]
    observable_fields: List[str] = schema["observable_fields"]
    canonical_columns: List[str] = schema["canonical_columns"]

    config = get_config_for_missing_count(missing_count)
    degraded = base_df[canonical_columns].copy().reset_index(drop=True)
    audit_rows: List[Dict[str, Any]] = []
    burst_rows: List[Dict[str, Any]] = []

    masked_field_counts = {field: 0 for field in observable_fields}
    affected_total = 0

    grouped = degraded.groupby("episode_id", sort=False)
    for episode_ordinal, (episode_id, group) in enumerate(grouped):
        group_sorted = group.sort_values(["episode_step", "window_id"], kind="mergesort")
        row_indices = group_sorted.index.to_numpy(dtype=np.int64)
        if len(row_indices) == 0:
            continue

        label = str(group_sorted["label"].iloc[0])
        rng = make_episode_rng(DEFAULT_RANDOM_SEED, missing_count, episode_ordinal)
        field_rng = make_episode_rng(DEFAULT_RANDOM_SEED, 0, episode_ordinal)
        missing_fields = choose_episode_missing_fields(
            label=label,
            missing_count=missing_count,
            observable_fields=observable_fields,
            rng=field_rng,
        )
        affected_mask, bursts = build_burst_mask(len(row_indices), config=config, rng=rng)

        for burst_id, burst in enumerate(bursts):
            burst_rows.append({
                "condition": f"realistic_missing{missing_count}",
                "episode_id": str(episode_id),
                "label": label,
                "burst_id": int(burst_id),
                "episode_start_pos": int(burst["start_pos"]),
                "episode_end_pos_inclusive": int(burst["end_pos_inclusive"]),
                "length_rows": int(burst["length_rows"]),
                "missing_count_per_affected_row": int(missing_count),
                "masked_fields": "|".join(missing_fields),
                "outage_profile": infer_outage_profile(label, missing_fields),
            })

        affected_positions = np.where(affected_mask)[0]
        affected_total += int(len(affected_positions))

        for pos in affected_positions:
            row_index = int(row_indices[pos])
            original_values = {field: str(degraded.at[row_index, field]) for field in missing_fields}
            for field in missing_fields:
                degraded.at[row_index, field] = placeholder
                masked_field_counts[field] += 1

            audit_rows.append({
                "condition": f"realistic_missing{missing_count}",
                "row_index": row_index,
                "scenario_id": str(degraded.at[row_index, "scenario_id"]),
                "episode_id": str(episode_id),
                "episode_step": int(degraded.at[row_index, "episode_step"]),
                "window_id": int(degraded.at[row_index, "window_id"]),
                "timestamp": str(degraded.at[row_index, "timestamp"]),
                "label": label,
                "missing_count_per_affected_row": int(missing_count),
                "masked_fields": "|".join(missing_fields),
                "outage_profile": infer_outage_profile(label, missing_fields),
                "original_values_json": json.dumps(original_values, ensure_ascii=False),
            })

    audit_df = pd.DataFrame(audit_rows)
    burst_df = pd.DataFrame(burst_rows)

    # Ensure canonical output types and validation.
    degraded = degraded[canonical_columns].copy().reset_index(drop=True)
    validate_output_dataframe(base_df=base_df, degraded_df=degraded, schema=schema, condition_name=f"realistic_missing{missing_count}")

    total_rows = int(len(degraded))
    affected_ratio = float(affected_total / total_rows) if total_rows else 0.0
    average_missing_per_row = float((affected_total * missing_count) / total_rows) if total_rows else 0.0

    summary = {
        "condition": f"realistic_missing{missing_count}",
        "rows": total_rows,
        "affected_rows": int(affected_total),
        "affected_row_ratio": affected_ratio,
        "missing_count_per_affected_row": int(missing_count),
        "average_missing_fields_per_row": average_missing_per_row,
        "placeholder": placeholder,
        "affected_fraction_target": float(config.affected_fraction),
        "burst_length_range_rows": list(config.burst_length_range),
        "num_burst_records": int(len(burst_df)),
        "masked_field_counts": {k: int(v) for k, v in masked_field_counts.items() if int(v) > 0},
        "protected_context_fields": OBSERVATION_CONTEXT_FIELDS,
        "protected_message_control_fields": PROTECTED_MESSAGE_CONTROL_FIELDS,
        "generation_rule": (
            "Field loss is generated as episode-local contiguous outage bursts. "
            "Within each episode outage, the same RF/GNSS receiver evidence or "
            "operational trace fields are masked, preserving temporal consistency "
            "under a prespecified synthetic stress scenario, without operational calibration."
        ),
    }
    return degraded, audit_df, {"summary": summary, "burst_df": burst_df}

def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def read_csv(path):
    return pd.read_csv(path, encoding='utf-8-sig', keep_default_na=False,
                       dtype={key: str for key in FALLBACK_METADATA_FIELDS if key not in ('episode_step', 'window_id')})


def make_episode_rng(base_seed, missing_count, episode_ordinal):
    # The ordinal is the episode's first-appearance position in the immutable split.
    # The label-bearing episode ID and observable values never enter the RNG.
    require(isinstance(episode_ordinal, (int, np.integer)) and episode_ordinal >= 0,
            'Use an opaque nonnegative episode ordinal, not an episode ID/label')
    payload = f'{int(base_seed)}:{int(missing_count)}:{int(episode_ordinal)}'.encode('ascii')
    return random.Random(int.from_bytes(hashlib.sha256(payload).digest()[:16], 'big'))


def choose_episode_missing_fields(label, missing_count, observable_fields, rng):
    # label is retained only for compatibility with the original call signature.
    candidates = RF_GNSS_RECEIVER_FIELDS + OPERATIONAL_TRACE_FIELDS
    require(set(candidates) <= set(observable_fields), 'Missing required evidence-field schema')
    require(0 < missing_count <= len(candidates), 'Missing count must be between 1 and 9')
    candidates = candidates.copy()
    rng.shuffle(candidates)
    return candidates[:missing_count]


def infer_outage_profile(label, missing_fields):
    # Diagnostic family only; ground-truth class is deliberately unused.
    fields = set(missing_fields)
    receiver = len(fields.intersection(RF_GNSS_RECEIVER_FIELDS))
    trace = len(fields.intersection(OPERATIONAL_TRACE_FIELDS))
    return 'RECEIVER_DOMINANT_GAP' if receiver > trace else 'TRACE_DOMINANT_GAP' if trace > receiver else 'MIXED_GAP'


def validate_source_dataframe(df, schema, source_name):
    require(list(df.columns) == schema['canonical_columns'], f'{source_name}: exact canonical schema required')
    require(len(df) > 0 and not df.isna().any().any(), f'{source_name}: empty/null source')
    out = df.copy().reset_index(drop=True)
    for col in out:
        values = out[col].astype(str)
        require(values.str.len().gt(0).all() and values.eq(values.str.strip()).all(), f'Empty/whitespace {col}')
    for col in ('episode_step', 'window_id'):
        values = pd.to_numeric(out[col], errors='raise').to_numpy()
        require(np.isfinite(values).all() and np.equal(values, np.floor(values)).all()
                and ((values >= 0) & (values < 2**53)).all(), f'Invalid integer {col}')
        out[col] = values.astype(np.int64)
    require(set(out.label) <= set(schema['label_order']), 'Unknown class')
    require(not out.duplicated(['episode_id', 'episode_step']).any(), 'Duplicate episode step')
    times = pd.to_datetime(out.timestamp, utc=True, errors='raise')
    for episode, group in out.groupby('episode_id', sort=False):
        ordered = group.sort_values(['episode_step', 'window_id'], kind='mergesort')
        require(group.label.nunique() == 1, f'Mixed-label episode: {episode}')
        for col in ('episode_step', 'window_id'):
            require(np.all(np.diff(ordered[col]) == 1), f'Nonconsecutive {col}: {episode}')
        require(times.loc[ordered.index].diff().dropna().dt.total_seconds().eq(1).all(),
                f'Nonconsecutive timestamps: {episode}')
    return out


def validate_output_dataframe(base_df, degraded_df, schema, condition_name):
    require(list(degraded_df.columns) == schema['canonical_columns'] and len(base_df) == len(degraded_df),
            f'{condition_name}: changed schema/population')
    for col in schema['metadata_fields'] + OBSERVATION_CONTEXT_FIELDS + PROTECTED_MESSAGE_CONTROL_FIELDS:
        require(base_df[col].astype(str).equals(degraded_df[col].astype(str)), f'{condition_name}: changed protected {col}')
    require(not degraded_df.isna().any().any(), f'{condition_name}: null output')


def validate_mask_values(base, degraded, schema, k, placeholder):
    validate_output_dataframe(base, degraded, schema, f'missing{k}')
    fields = schema['observable_fields']
    before, after = base[fields].astype(str).to_numpy(), degraded[fields].astype(str).to_numpy()
    changed = before != after
    require(np.isin(changed.sum(1), [0, k]).all(), 'Affected rows must mask exactly k fields')
    require((after[changed] == placeholder).all(), 'A changed value is not the missing placeholder')
    require((before != placeholder).all(), 'Complete source already contains the missing placeholder')
    return changed



def validate_shared_build(build, dataset, manifest_hash):
    policy = build.get('policy', {})
    require(build.get('status') == 'complete'
            and policy.get('nominal_substrate') == INPUT_CONTRACT_VERSION
            and policy.get('columns') == FALLBACK_METADATA_FIELDS + FALLBACK_OBSERVABLE_FIELDS
            and policy.get('label_order') == LABELS and policy.get('window_seconds') == 1,
            'Use the completed shared_nominal_v2 Step 7 build with the agreed 22 observables.')
    expected_id = hashlib.sha256(json.dumps({'policy': policy, 'sources': build.get('sources')},
                                           sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    require(build.get('dataset_id') == expected_id, 'Shared nominal Step 7 dataset identity mismatch')
    spec = build.get('observability_audit', {})
    files = {s: f'audit/{s}_window_observable_effects.csv' for s in ('train', 'val', 'test')}
    hashes = {name: build.get('artifacts', {}).get(name) for name in files.values()}
    require(spec.get('schema_version') == 'counterfactual_observable_change_v1'
            and spec.get('split_files') == files and bool(spec.get('scope'))
            and all(isinstance(value, str) and len(value) == 64 for value in hashes.values()),
            'Step 7 must declare its separate observable-effect diagnostics.')
    # Record complete-data diagnostic lineage only. No sidecar values are read or
    # copied into masked observables: masking can remove an effect-bearing field.
    return {'dataset_id': build['dataset_id'], 'nominal_substrate': INPUT_CONTRACT_VERSION,
            'step7_build_manifest_sha256': manifest_hash,
            'observability_audit': {'schema_version': spec['schema_version'], 'dataset_id': build['dataset_id'],
                'source_dataset_dir': str(dataset.resolve()), 'step7_build_manifest_sha256': manifest_hash,
                'split_files': files, 'artifacts': hashes, 'scope': spec['scope']}}


def required_output_names(counts):
    names = {'test_complete_observable.csv', 'realistic_missing_field_manifest_step18.json',
             'realistic_missing_field_summary_step18.csv'}
    for k in counts:
        names.update({f'test_realistic_missing{k}_observable.csv',
            f'audit/test_realistic_missing{k}_mask_audit.csv', f'audit/test_realistic_missing{k}_burst_audit.csv'})
    return names


def verify_sources_unchanged(inputs):
    for path, digest in inputs.items():
        require(sha256(path) == digest, f'Source changed during generation: {path}')


def run(project_root, output_dir=None, missing_counts=(2, 5, 8), placeholder='UNK', seed=20260610, source_test=None):
    global DEFAULT_RANDOM_SEED
    DEFAULT_RANDOM_SEED = int(seed)
    root = Path(project_root).expanduser().resolve()
    dataset = root / 'dataset/shared_nominal_v2'
    source = dataset / 'splits/test_observable.csv'
    if source_test is not None:
        candidate = Path(source_test).expanduser()
        candidate = (candidate if candidate.is_absolute() else root / candidate).resolve()
        require(candidate == source.resolve(), 'The source must be the current revised Step 7 test split')
    output = Path(output_dir).expanduser() if output_dir is not None else dataset / 'realistic_missing_field_tests'
    output = (output if output.is_absolute() else root / output).resolve()
    counts = sorted(set(int(k) for k in missing_counts))
    require(counts and all(0 < k <= 9 for k in counts), 'Provide missing counts between 1 and 9')
    require(placeholder == 'UNK', 'This revision protocol uses the literal value UNK')
    require(seed >= 0, 'Seed must be nonnegative')
    build_path, map_path = dataset / 'build_manifest.json', dataset / 'splits/label_mapping.json'
    inputs = {str(p): sha256(p) for p in (source, map_path, build_path)}
    build = json.loads(build_path.read_text(encoding='utf-8-sig'))
    mapping = json.loads(map_path.read_text(encoding='utf-8-sig'))
    source_contract = validate_shared_build(build, dataset, inputs[str(build_path)])
    require(build.get('status') == 'complete' and build.get('dataset_id'), 'Step 7 must be complete')
    for path in (source, map_path):
        require(inputs[str(path)] == build.get('artifacts', {}).get(path.relative_to(dataset).as_posix()),
                f'Step 7 artifact hash mismatch: {path}')
    require(build.get('model_input_fields') == FALLBACK_OBSERVABLE_FIELDS, 'Unexpected Step 7 field order')
    require(mapping.get('label_order') == LABELS and mapping.get('label_to_id') == dict(zip(LABELS, range(6))),
            'Canonical label mapping mismatch')
    schema = {'metadata_fields': FALLBACK_METADATA_FIELDS, 'observable_fields': FALLBACK_OBSERVABLE_FIELDS,
              'canonical_columns': FALLBACK_METADATA_FIELDS + FALLBACK_OBSERVABLE_FIELDS,
              'label_order': LABELS, 'observable_window_seconds': 1}
    for key in ('metadata_fields', 'observable_fields', 'observable_window_seconds'):
        if key in mapping:
            require(mapping[key] == schema[key], f'Step 7 schema mismatch: {key}')
    base = validate_source_dataframe(read_csv(source), schema, str(source))
    require(len(base) == build['counts']['test']['rows'] and base.episode_id.nunique() == build['counts']['test']['episodes'],
            'Step 7 test count mismatch')
    require(set(base.label) == set(LABELS), 'Shared nominal test split must contain all six classes')
    require(not base[schema['observable_fields']].eq(placeholder).any().any(), 'Complete source already contains UNK')
    verify_sources_unchanged(inputs)
    identity = {'protocol_version': PROTOCOL_VERSION, 'input_contract_version': INPUT_CONTRACT_VERSION,
                'source_contract': source_contract, 'dataset_id': build['dataset_id'],
                'input_sha256': inputs, 'script_sha256': sha256(Path(__file__)), 'seed': int(seed),
                'missing_counts': counts, 'placeholder': placeholder,
                'scenarios': {str(k): asdict(get_config_for_missing_count(k)) for k in counts}}
    # Normalize tuples exactly as persisted JSON before equality/reuse checks.
    identity = json.loads(json.dumps(identity))
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if output.exists():
        path = output / 'run_manifest.json'
        require(path.is_file(), f'Output already exists without a revised manifest: {output}. Use a new --output-dir')
        old = json.loads(path.read_text(encoding='utf-8-sig'))
        require(old.get('status') == 'COMPLETE' and old.get('run_id') == run_id and old.get('identity') == identity,
                'Output is incomplete or belongs to different inputs/settings/code. Use a new --output-dir')
        require(set(old.get('artifacts', {})) == required_output_names(counts),
                'Incomplete or unexpected generated artifact inventory')
        for name, digest in old['artifacts'].items():
            path = (output / name).resolve()
            require(path.is_relative_to(output) and sha256(path) == digest, f'Changed generated artifact: {name}')
        verify_sources_unchanged(inputs)
        print(f'[Reuse verified Step 18] {output}', flush=True)
        return output
    output.mkdir(parents=True, exist_ok=False)
    (output / 'audit').mkdir()
    record = {'status': 'RUNNING', 'run_id': run_id, 'identity': identity,
              'dataset_id': build['dataset_id'], 'source_contract': source_contract, 'started_at_utc': datetime.now(timezone.utc).isoformat()}
    save_json(output / 'run_manifest.json', record)
    try:
        # Byte-identical control, including original row order and CSV formatting.
        (output / 'test_complete_observable.csv').write_bytes(source.read_bytes())
        summaries, paths = [], {'complete': 'test_complete_observable.csv'}
        for k in counts:
            name = f'realistic_missing{k}'
            degraded, audit, details = build_realistic_missing_testset(base, schema, k, placeholder)
            changed = validate_mask_values(base, degraded, schema, k, placeholder)
            require(int(changed.any(1).sum()) == details['summary']['affected_rows'], 'Mask audit count mismatch')
            filename = f'test_{name}_observable.csv'
            degraded.to_csv(output / filename, index=False, encoding='utf-8-sig')
            audit.to_csv(output / 'audit' / f'test_{name}_mask_audit.csv', index=False, encoding='utf-8-sig')
            details['burst_df'].to_csv(output / 'audit' / f'test_{name}_burst_audit.csv', index=False, encoding='utf-8-sig')
            summaries.append(details['summary'])
            paths[name] = filename
            print(f'{name}: {details["summary"]["affected_rows"]:,}/{len(base):,} affected windows', flush=True)
        protocol = {'protocol_version': PROTOCOL_VERSION, 'input_contract_version': INPUT_CONTRACT_VERSION,
                    'source_contract': source_contract, 'step': 18, 'run_id': run_id,
                    'dataset_id': build['dataset_id'], 'source_test_sha256': inputs[str(source)],
                    'condition_paths': paths, 'schema': schema, 'seed': int(seed), 'placeholder': placeholder,
                    'mask_selection': 'uniform permutation of nine evidence fields, independent of ground-truth labels',
                    'rng_key': 'seed, stream (k for bursts; zero for fields), opaque first-appearance episode ordinal',
                    'nested_field_sets': True, 'nested_burst_masks': False,
                    'observable_effect_audit_use': 'Step 7 effect flags describe the complete data; they are not recomputed '
                        'after masking, are not model inputs, and never select fields, bursts or evaluation endpoints.',
                    'burst_range_meaning': 'proposal lengths; overlapping proposals merge and target fractions may overshoot',
                    'scope': 'Synthetic structured stress only; one mask seed; joint severity changes k/fraction/burst length. '
                             'Thirteen protected context/control fields remain visible by design. No field-calibrated claim.',
                    'summaries': summaries}
        save_json(output / 'realistic_missing_field_manifest_step18.json', protocol)
        pd.DataFrame(summaries).to_csv(output / 'realistic_missing_field_summary_step18.csv', index=False, encoding='utf-8-sig')
        verify_sources_unchanged(inputs)
        record.update(status='COMPLETE', completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      artifacts={p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob('*'))
                                 if p.is_file() and p.name != 'run_manifest.json'})
        require(set(record['artifacts']) == required_output_names(counts), 'Generated artifact inventory mismatch')
        save_json(output / 'run_manifest.json', record)
        print(f'COMPLETE: {output}', flush=True)
        return output
    except Exception as exc:
        record.update(status='FAILED', error=str(exc))
        save_json(output / 'run_manifest.json', record)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root', type=Path, default=BASE_DIR)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--source-test', type=Path)
    parser.add_argument('--missing-counts', nargs='+', type=int, default=MISSING_COUNTS)
    parser.add_argument('--placeholder', default='UNK')
    parser.add_argument('--seed', type=int, default=20260610)
    args = parser.parse_args(argv)
    run(args.project_root, args.output_dir, args.missing_counts, args.placeholder, args.seed, args.source_test)


if __name__ == '__main__':
    main()
