#!/usr/bin/env python3
"""Step 41: shared-nominal provenance, duplicates and conditional fingerprint diagnostics.

Run after revised Step 7; does not depend on or modify Steps 8-19/model training.
Default inputs: dataset/*.csv and dataset/shared_nominal_v2/build_manifest.json.
Default outputs: outputs/shared_nominal_v2/41A_benchmark_Aprime_audit/run_<UTC>/.
Keep this script beside the six makers, shared_nominal.py and shared_context_v4.py. Both '1.name.py'
and '1. name.py' maker names are supported; conflicting copies are never guessed.

Maker review ZIPs saved by Steps 1-6 are inspected read-only in dataset/ by default.
Use --maker-review-dir PATH for an additional directory. Missing generation records
are explicitly UNVERIFIED; current source text alone cannot prove how CSVs were made.
No maker is executed, data regenerated, source changed, or old output deleted.

Convergence revision: all three logistic fits use a fixed 10,000-iteration
budget (previously 1,000). Features, class weights, tolerance, regularization,
solver, seed and splits are unchanged. A warning or an exhausted budget is
reported explicitly; held-out scores never trigger retries or model selection.

Scope: exact supplied-record/episode separation, exact observable duplicates,
static source patterns, fixed train-only context probes and 22-field no-change probes. Different seeds do not
prove independent processes; low context-probe scores do not prove no shortcuts.
No source/model/seed is selected or retuned in response to held-out audit scores.
Labels remain supplied episode/scenario classes, including quiet/unobserved windows.
Same-generator audits do not establish operational/external validity or the necessity
of DistilBERT/TST. The revision concerns representation + cross-window aggregation.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import platform
import sys
import warnings
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from zipfile import ZipFile, BadZipFile
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / 'dataset'
SPLIT_DIR = DATASET_DIR / 'shared_nominal_v2/splits'
OUTPUT_DIR = BASE_DIR / 'outputs/shared_nominal_v2/41A_benchmark_Aprime_audit'
PROTOCOL_VERSION = 'shared_nominal_seed_duplicate_context_v3_1_convergence'
LENGTHS = [1, 2, 3, 5, 10]
PROBE_SEED = 20260610
PROBE_MAX_ITER = 10000
PINNED_INPUTS = {}

LABEL_ORDER = [
    "NORMAL",
    "RF_LINK_JAMMING",
    "GNSS_SPOOFING",
    "UNAUTHORIZED_TELECOMMAND_INJECTION",
    "TELECOMMAND_REPLAY",
    "GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE",
]

SOURCE_FILES = {
    "NORMAL": "normal_observable_dataset.csv",
    "RF_LINK_JAMMING": "rf_link_jamming_observable_dataset.csv",
    "GNSS_SPOOFING": "gnss_spoofing_observable_dataset.csv",
    "UNAUTHORIZED_TELECOMMAND_INJECTION": "unauthorized_telecommand_injection_observable_dataset.csv",
    "TELECOMMAND_REPLAY": "telecommand_replay_observable_dataset.csv",
    "GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE": "ground_segment_ttc_control_path_compromise_observable_dataset.csv",
}

MAKER_FILES = {
    "NORMAL": "1. normal_observable_maker_revised.py",
    "RF_LINK_JAMMING": "2. RF_LINK_JAMMING_observable_maker_revised.py",
    "GNSS_SPOOFING": "3. GNSS_SPOOFING_observable_maker_revised.py",
    "UNAUTHORIZED_TELECOMMAND_INJECTION": "4. UNAUTHORIZED_TELECOMMAND_INJECTION_observable_maker_revised.py",
    "TELECOMMAND_REPLAY": "5. TELECOMMAND_REPLAY_observable_maker_revised.py",
    "GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE": "6. GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE_observable_maker_revised.py",
}

SPLIT_FILES = {
    "train": "train_observable.csv",
    "val": "val_observable.csv",
    "test": "test_observable.csv",
}

METADATA_FIELDS = [
    "scenario_id",
    "episode_id",
    "episode_type",
    "episode_step",
    "window_id",
    "timestamp",
    "label",
]

CONTEXT_FIELDS = [
    "observation_stage",
    "path_type",
    "direction",
    "source_role",
    "destination_role",
    "protocol_context",
    "operational_phase",
]

OBSERVABLE_FIELDS = [
    *CONTEXT_FIELDS,
    "message_type",
    "service_context",
    "command_intent",
    "payload_or_parameter_context",
    "sequence_freshness_status",
    "control_security_state",
    "signal_quality_state",
    "tracking_stability",
    "navigation_consistency",
    "timing_consistency",
    "decoder_status",
    "command_execution_result",
    "retry_timeout_state",
    "session_state",
    "state_transition_consistency",
]

CANONICAL_COLUMNS = METADATA_FIELDS + OBSERVABLE_FIELDS

def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def pin(path, expected=None):
    path = Path(path).resolve()
    digest = sha256(path)
    require(expected is None or digest == expected, f'Hash mismatch: {path}')
    require(str(path) not in PINNED_INPUTS or PINNED_INPUTS[str(path)] == digest, f'Input changed during audit: {path}')
    PINNED_INPUTS[str(path)] = digest
    return digest


def read_json(path):
    pin(path)
    value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    pin(path)
    require(isinstance(value, dict), f'Expected a JSON object: {path}')
    return value


def save_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    def scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f'Unsupported JSON value: {type(value).__name__}')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=scalar) + '\n', encoding='utf-8')
    temp.replace(path)


def save_csv(path, frame):
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def read_csv(path):
    pin(path)
    # Preserve literal NONE/NA categories and identifiers; reject actual empty cells.
    frame = pd.read_csv(path, encoding='utf-8-sig', keep_default_na=False, dtype=str)
    pin(path)
    return frame


def canonicalize_signature_frame(df, fields):
    require(set(fields) <= set(df), 'Missing signature fields')
    out = df[fields].copy()
    require(not out.isna().any().any(), 'Null values cannot enter an exact signature')
    for col in fields:
        out[col] = out[col].astype(str)
        require(out[col].str.len().gt(0).all(), f'Empty signature value: {col}')
    return out


def row_hashes(df, fields):
    """SHA-256 over unambiguous UTF-8 JSON arrays, excluding dataframe index.

    This replaces version-dependent 64-bit pandas hashes and delimiter joins.
    Hash equality is used as a computational exact-match proxy, not a semantic one.
    """
    values = canonicalize_signature_frame(df, fields)
    return pd.Series([hashlib.sha256(json.dumps(row, ensure_ascii=False, separators=(',', ':'),
                            allow_nan=False).encode('utf-8')).hexdigest()
                      for row in values.itertuples(index=False, name=None)], index=df.index, dtype=object)


def integers(values, name):
    numeric = pd.to_numeric(values, errors='raise').to_numpy()
    require(np.isfinite(numeric).all() and (numeric == np.floor(numeric)).all()
            and ((numeric >= 0) & (numeric < 2**53)).all(), f'Invalid integer: {name}')
    return numeric.astype(np.int64)


def validate_frame(frame, name, single_label=None):
    require(list(frame.columns) == CANONICAL_COLUMNS, f'{name}: exact 7 metadata + 22 observable columns required')
    require(len(frame) > 0 and not frame.isna().any().any(), f'{name}: empty/null dataset')
    out = frame.copy().reset_index(drop=True)
    for col in CANONICAL_COLUMNS:
        values = out[col].astype(str)
        require(values.str.len().gt(0).all() and values.eq(values.str.strip()).all(), f'{name}: empty/whitespace {col}')
    for col in ('episode_step', 'window_id'):
        out[col] = integers(out[col], name + ':' + col)
    require(set(out.label) <= set(LABEL_ORDER), f'{name}: unknown class')
    if single_label is not None:
        require(set(out.label) == {single_label}, f'{name}: wrong source class')
    require(not out.duplicated(['episode_id', 'episode_step']).any(), f'{name}: duplicate episode/window identity')
    timestamp = pd.to_datetime(out.timestamp, utc=True, errors='raise')
    require(not timestamp.isna().any(), f'{name}: invalid timestamp')
    for episode, group in out.groupby('episode_id', sort=False):
        ordered = group.sort_values(['episode_step', 'window_id'], kind='mergesort')
        require(group.label.nunique() == 1, f'{name}: mixed labels within {episode}')
        for col in ('episode_step', 'window_id'):
            require(np.all(np.diff(ordered[col]) == 1), f'{name}: nonconsecutive {col}: {episode}')
        require(timestamp.loc[ordered.index].diff().dropna().dt.total_seconds().eq(1).all(),
                f'{name}: nonconsecutive timestamps: {episode}')
        require(group.scenario_id.nunique() == group.episode_type.nunique() == 1,
                f'{name}: inconsistent episode metadata: {episode}')
    return out


AUDIT_SCHEMA_VERSION = "counterfactual_observable_change_v1"
AUDIT_FIELDS = [
    "audit_observable_effect_present", "audit_observable_difference_count",
    "audit_persistent_effect_present", "audit_attack_active",
]
AUDIT_SCOPE = (
    "Actual versus paired nominal 22-field projection differences; audit only. "
    "No current attack, no projection difference, and no class information are distinct. "
    "These annotations are not model inputs, an oracle bound, or a proof of identifiability."
)


def observability_audit_contract() -> Dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "split_files": {split: f"audit/{split}_window_observable_effects.csv"
                        for split in ("train", "val", "test")},
        "columns": METADATA_FIELDS + AUDIT_FIELDS,
        "scope": AUDIT_SCOPE,
    }

def raw_audit_path(source_path: Path) -> Path:
    path = source_path.with_name(source_path.stem + "_with_raw.csv")
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing companion raw audit: {path}. Run the corresponding revised maker "
            "and keep its learning CSV and *_with_raw.csv together in --dataset-dir. "
            "Do not substitute a sample timeline or an older raw file."
        )
    return path

def load_observability_audit(
    learning: pd.DataFrame, source_path: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Check the new maker contract without adding privileged data to learning.

    Read only the needed audit columns in chunks, not the entire raw-physics
    table. Validate row identity and recompute the difference of all 22 fields.
    Normalization/physics remain the maker's responsibility; this is a handoff
    and counterfactual-annotation check, not a detector or fingerprint test.
    """
    path = raw_audit_path(source_path)
    baseline = ["baseline_observable_" + f for f in OBSERVABLE_FIELDS]
    binary = ["raw_observable_effect_present", "raw_persistent_effect_present",
              "raw_attack_active", "raw_attacks_enabled", "raw_paired_audit_background"]
    additional = binary + ["raw_observable_difference_count", "raw_observable_difference_fields",
                           "raw_background_stream"]
    required = CANONICAL_COLUMNS + baseline + additional
    header = pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns.tolist()
    missing = [field for field in required if field not in header]
    if missing:
        raise ValueError(
            f"{path.name} does not have the shared-nominal audit schema: {missing}. "
            "Regenerate this class with the revised maker and shared_nominal.py."
        )
    offset, pieces, streams = 0, [], set()
    with pd.read_csv(path, encoding="utf-8-sig", usecols=required,
                     dtype=str, keep_default_na=False, chunksize=10000) as reader:
        for chunk in reader:
            chunk = chunk.reset_index(drop=True)
            expected = learning.iloc[offset:offset + len(chunk)].reset_index(drop=True)
            if len(expected) != len(chunk) or not chunk[CANONICAL_COLUMNS].equals(expected.astype(str)):
                raise ValueError(f"Learning and raw audit rows differ at offset {offset}: {path.name}")
            if chunk[required].apply(lambda s: s.str.strip().eq("")).any().any():
                raise ValueError(f"Blank audit values in {path.name} at offset {offset}.")
            numeric = {}
            for field in binary + ["raw_observable_difference_count"]:
                values = pd.to_numeric(chunk[field], errors="raise")
                allowed_max = 22 if field == "raw_observable_difference_count" else 1
                if (not np.isfinite(values).all() or (values % 1 != 0).any()
                        or not values.between(0, allowed_max).all()):
                    raise ValueError(f"Invalid integer audit field {field}: {path.name}")
                numeric[field] = values.to_numpy(dtype=np.int64)
            if numeric["raw_paired_audit_background"].any():
                raise ValueError(
                    f"Paired audit backgrounds are not permitted in the main corpus: {path.name}. "
                    "Use the maker's default independent background streams."
                )
            expected_enabled = int(expected["label"].iloc[0] != "NORMAL")
            if not (numeric["raw_attacks_enabled"] == expected_enabled).all():
                raise ValueError(f"Unexpected attacks_enabled setting for this class: {path.name}")
            changes = chunk[OBSERVABLE_FIELDS].to_numpy() != chunk[baseline].to_numpy()
            counts = changes.sum(axis=1).astype(np.int64)
            fields = np.asarray(["|".join(np.asarray(OBSERVABLE_FIELDS)[row]) or "NONE" for row in changes])
            if (not np.array_equal(counts, numeric["raw_observable_difference_count"])
                    or not np.array_equal((counts > 0).astype(np.int64), numeric["raw_observable_effect_present"])
                    or not np.array_equal(fields, chunk["raw_observable_difference_fields"].to_numpy())):
                raise ValueError(f"Counterfactual observable difference/effect annotations disagree: {path.name}")
            if not expected_enabled and (counts.any() or numeric["raw_attack_active"].any()
                                         or numeric["raw_persistent_effect_present"].any()):
                raise ValueError(f"NORMAL has an attack-related counterfactual change: {path.name}")
            streams.update(chunk["raw_background_stream"].unique().tolist())
            part = expected[METADATA_FIELDS].copy()
            for target, source in zip(AUDIT_FIELDS, ["raw_observable_effect_present", "raw_observable_difference_count",
                                                   "raw_persistent_effect_present", "raw_attack_active"]):
                part[target] = numeric[source]
            pieces.append(part)
            offset += len(chunk)
    if offset != len(learning) or not pieces:
        raise ValueError(f"Learning and raw audit row counts differ: {path.name}")
    if len(streams) != 1:
        raise ValueError(f"Expected one independent background stream per maker file: {path.name}")
    audit = pd.concat(pieces, ignore_index=True)
    return audit, {
        "raw_audit_file": path.name, "rows": len(audit), "background_stream": next(iter(streams)),
        "observable_effect_windows": int(audit["audit_observable_effect_present"].sum()),
        "persistent_effect_windows": int(audit["audit_persistent_effect_present"].sum()),
        "paired_background": False, "scope": AUDIT_SCOPE,
    }

def load_build_contract(dataset):
    build_root = dataset / 'shared_nominal_v2'
    record = read_json(build_root / 'build_manifest.json')
    require(record.get('status') == 'complete', 'Revised Step 7 must be complete')
    policy, sources = record.get('policy', {}), record.get('sources', [])
    require(record.get('dataset_id') == object_hash({'policy': policy, 'sources': sources}), 'Invalid Step 7 dataset identity')
    require(policy.get('version') == 'step7_episode_split_v2'
            and policy.get('nominal_substrate') == 'shared_nominal_v2'
            and record.get('observability_audit') == observability_audit_contract() and policy.get('columns') == CANONICAL_COLUMNS
            and policy.get('label_order') == LABEL_ORDER and policy.get('window_seconds') == 1
            and policy.get('sequence_lengths') == LENGTHS, 'Unsupported Step 7 schema/split protocol')
    require(record.get('model_input_fields') == OBSERVABLE_FIELDS
            and record.get('metadata_not_model_features') == METADATA_FIELDS, 'Step 7 feature/metadata boundary changed')
    expected = policy.get('expected_rows_per_class')
    require(type(expected) is int and expected > 0, 'Missing Step 7 source row count policy')
    require([s.get('label') for s in sources] == LABEL_ORDER, 'Six ordered class sources required')
    for source in sources:
        require(source.get('file') == SOURCE_FILES[source['label']], 'Unexpected source filename')
        path = dataset / source['file']
        pin(path, source['sha256'])
        require(path.stat().st_size == source['size_bytes'], f'Source size differs: {path}')
        raw = source.get('raw_audit', {})
        require(raw.get('file') == path.stem+'_with_raw.csv', 'Missing/mismatched shared nominal raw companion')
        raw_path = dataset/raw['file']
        pin(raw_path, raw.get('sha256'))
        require(isinstance(raw.get('sha256'), str) and len(raw['sha256']) == 64
                and raw_path.stat().st_size == raw.get('size_bytes'), 'Raw companion identity differs')
    required = ['unified_observable_dataset.csv', 'splits/label_mapping.json', 'splits/episode_split_manifest.csv']
    required += ['splits/' + name for name in SPLIT_FILES.values()]
    required += list(observability_audit_contract()['split_files'].values())
    inventory = record.get('artifacts', {})
    require(set(required) <= set(inventory), 'Step 7 completion inventory is missing audit inputs')
    for name, digest in inventory.items():
        path = (build_root / name).resolve()
        require(path.is_relative_to(build_root.resolve()) and isinstance(digest, str) and len(digest) == 64,
                f'Invalid Step 7 artifact path/digest: {name}')
        pin(path, digest)
    mapping = read_json(build_root / 'splits/label_mapping.json')
    require(mapping.get('label_order') == LABEL_ORDER and mapping.get('label_to_id') == dict(zip(LABEL_ORDER, range(6)))
            and mapping.get('metadata_fields') == METADATA_FIELDS and mapping.get('observable_fields') == OBSERVABLE_FIELDS,
            'Step 7 label/schema map mismatch')
    return record


def resolve_maker_path(directory, canonical_name):
    names = [canonical_name, canonical_name.replace('. ', '.', 1)]
    paths = sorted({(directory / name).resolve() for name in names if (directory / name).is_file()})
    require(paths, f'Maker missing: {directory / canonical_name} (space after the dot is optional)')
    digests = {pin(path) for path in paths}
    require(len(digests) == 1, f'Conflicting maker copies: {paths}. Use --maker-dir with the intended six sources')
    return paths[0]


def literal_assignments(tree):
    values = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    try:
                        values[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        values.pop(target.id, None)  # A later dynamic assignment invalidates an older literal.
    return values


def qualified_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return qualified_name(node.value) + '.' + node.attr
    return ''


def static_maker_source_audit(maker_dir=BASE_DIR):
    """Inspect shared delegation without importing or executing any maker.

    These are syntactic observations, not a proof about all runtime branches.
    Independent stream records and counterfactual annotations are checked later.
    """
    maker_dir = Path(maker_dir)
    shared_path = maker_dir/'shared_nominal.py'
    pin(shared_path)
    shared_tree = ast.parse(shared_path.read_text(encoding='utf-8-sig'))
    shared_version = literal_assignments(shared_tree).get('CODE_VERSION')
    require(isinstance(shared_version, str) and shared_version, 'Shared nominal CODE_VERSION is missing')
    rows = []
    for number, label in enumerate(LABEL_ORDER, 1):
        path = resolve_maker_path(maker_dir, MAKER_FILES[label])
        tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        values = literal_assignments(tree)
        aliases = {a.asname or a.name for n in tree.body if isinstance(n, ast.Import)
                   for a in n.names if a.name == 'shared_nominal'}
        builders = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name.startswith('build_') and n.name.endswith('_dataset')]
        delegates = [n.name for n in builders if any(isinstance(c, ast.Return)
            and isinstance(c.value, ast.Call) and qualified_name(c.value.func) in {a+'.build_dataset' for a in aliases}
            and [qualified_name(x) for x in c.value.args[1:]] ==
                ['num_rows','window_seconds','start_time','seed','return_raw','background_seed','attacks_enabled']
            and c.value.args and isinstance(c.value.args[0], ast.Call)
            and qualified_name(c.value.args[0].func) == 'globals' for c in ast.walk(n))]
        # Resolve the last top-level binding, not a stale earlier local normalizer.
        projection_shared = False
        version_ref = ''
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                for a in node.names:
                    if (a.asname or a.name) == 'normalize_raw_to_observable':
                        projection_shared = node.module == 'shared_nominal' and a.name == 'normalize_raw_to_observable'
            elif isinstance(node, ast.FunctionDef) and node.name == 'normalize_raw_to_observable':
                projection_shared = False
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(t, ast.Name) and t.id == 'CODE_VERSION' for t in targets):
                    version_ref = qualified_name(node.value)
                if any(isinstance(t, ast.Name) and t.id == 'normalize_raw_to_observable' for t in targets):
                    projection_shared = qualified_name(node.value) in {a+'.normalize_raw_to_observable' for a in aliases}
        shared_version_ref = version_ref in {a+'.CODE_VERSION' for a in aliases}
        seed = values.get('DEFAULT_RANDOM_SEED')
        rows.append({'label':label, 'maker_file':path.name, 'maker_path':str(path), 'maker_sha256':pin(path),
            'seed':seed if type(seed) is int and seed >= 0 else None,
            'maker_number':values.get('MAKER_NUMBER'), 'expected_maker_number':number,
            'code_version':shared_version if shared_version_ref else values.get('CODE_VERSION'),
            'version_from_shared_nominal':shared_version_ref, 'uses_shared_nominal_builder':bool(delegates),
            'delegating_builders':'|'.join(delegates), 'uses_shared_projection':projection_shared,
            'canonical_file_present':True})
    table = pd.DataFrame(rows)
    seeds = [r['seed'] for r in rows]
    nodes = {n.name:n for n in shared_tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    runtime = nodes.get('build_dataset')
    substrate = nodes.get('NominalSubstrate')
    runtime_calls = {qualified_name(n.func) for n in ast.walk(runtime) if isinstance(n, ast.Call)} if runtime else set()
    substrate_calls = {qualified_name(n.func) for n in ast.walk(substrate) if isinstance(n, ast.Call)} if substrate else set()
    summary = {'all_six_canonical_makers_present':True, 'all_seeds_present':all(type(v) is int for v in seeds),
        'distinct_declared_default_seeds':len(set(seeds)) == 6,
        'all_use_shared_nominal_builder':bool(table.uses_shared_nominal_builder.all()),
        'all_use_shared_projection':bool(table.uses_shared_projection.all()),
        'all_use_shared_code_version':bool(table.version_from_shared_nominal.all()),
        'maker_numbers_match':bool(table.maker_number.eq(table.expected_maker_number).all()),
        'shared_runtime_uses_substrate_and_projection':{'NominalSubstrate','normalize_raw_to_observable'} <= runtime_calls,
        'separate_seedsequence_calls_in_background_and_attack_runtime':
            'np.random.SeedSequence' in runtime_calls and 'np.random.SeedSequence' in substrate_calls,
        'shared_nominal_sha256':pin(shared_path), 'shared_nominal_code_version':shared_version,
        'scope':'Static delegation and call patterns only. Different default seeds are descriptive; '
                'no statistical independence or shortcut absence is inferred. No makers executed.'}
    return table, summary



def load_shared_pool(maker_dir):
    path = Path(maker_dir) / 'shared_context_v4.py'
    pin(path)
    constants = literal_assignments(ast.parse(path.read_text(encoding='utf-8-sig')))
    pool = constants.get('GLOBAL_CONTEXT_PROFILE_POOL')
    require(constants.get('CONTEXT_FIELD_NAMES') == CONTEXT_FIELDS and isinstance(pool, list) and pool,
            'Shared module must declare the seven context fields and a literal profile pool')
    profiles, weights = [], []
    forbidden = set(constants.get('FORBIDDEN_CONTEXT_TOKENS', set())) | set(LABEL_ORDER)
    for entry in pool:
        profile = entry.get('profile', {})
        require(set(profile) == set(CONTEXT_FIELDS), 'Shared profile contains missing/extra fields')
        require(all(isinstance(profile[k], str) and profile[k] and profile[k] not in forbidden for k in CONTEXT_FIELDS),
                'Shared context profile has invalid/label/attack-stage tokens')
        require('label' not in entry and 'scenario' not in entry, 'Label/scenario key in shared pool')
        profiles.append(tuple(profile[k] for k in CONTEXT_FIELDS))
        weights.append(float(entry['weight']))
    require(len(set(profiles)) == len(profiles) and np.isfinite(weights).all() and (np.array(weights) > 0).all(),
            'Duplicate shared profiles or invalid mixture weights')
    weights = np.asarray(weights) / np.sum(weights)
    nominal_path = Path(maker_dir)/'shared_nominal.py'
    return {'path':path, 'sha256':pin(path), 'profiles':profiles, 'weights':weights,
            'nominal_path':nominal_path, 'nominal_sha256':pin(nominal_path)}


def maker_generation_lineage(dataset, directories, build, maker_table, shared):
    """Bind current CLI review archives by their actual learning/raw/source bytes.

    Shared nominal run_summary.json deliberately has no sha256 map. Derive
    member digests directly; never report these valid archives as absent.
    """
    sources = {s['label']:s for s in build['sources']}
    matches, ignored = {label:[] for label in LABEL_ORDER}, []
    def member_hash(archive, name):
        h = hashlib.sha256()
        with archive.open(name) as stream:
            for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
        return h.hexdigest()
    archives = sorted({p.resolve() for directory in directories for p in Path(directory).glob('*REVIEW.zip')})
    for path in archives:
        try:
            with ZipFile(path) as archive:
                names = archive.namelist()
                if 'run_summary.json' not in names: continue
                require(len(names) == len(set(names)), f'Duplicate ZIP member names: {path}')
                require(archive.getinfo('run_summary.json').file_size <= 4*1024*1024, 'Oversized maker summary')
                summary = json.loads(archive.read('run_summary.json').decode('utf-8-sig'))
                require(isinstance(summary, dict), f'Invalid maker summary: {path}')
                label = summary.get('label')
                if label not in sources: continue
                source = sources[label]; name = source['file']
                if name not in names or member_hash(archive,name) != source['sha256']: continue
                # Several legitimate maker runs can emit the same 22-field CSV
                # with different raw draws. Only the current learning/raw pair
                # identifies this cohort; keep older archives without adopting them.
                raw_name = source['raw_audit']['file']
                if raw_name not in names or member_hash(archive,raw_name) != source['raw_audit']['sha256']:
                    ignored.append({'file':str(path), 'reason':'LEARNING_MATCHES_BUT_RAW_COMPANION_DIFFERS_OR_IS_ABSENT'})
                    continue
                archive_digest = pin(path)
                require(summary.get('common_nominal_engine') is True and summary.get('window_seconds') == 1
                        and summary.get('model_input_columns') == OBSERVABLE_FIELDS,
                        f'Matching archive is not the shared-nominal generation protocol: {path}')
                require(type(summary.get('seed')) is int and summary['seed'] >= 0
                        and summary.get('rows') == build['policy']['expected_rows_per_class'],
                        f'Invalid generation seed/rows: {path}')
                canonical = MAKER_FILES[label]
                maker_members = [n for n in ('source/'+canonical,'source/'+canonical.replace('. ','.',1)) if n in names]
                require(maker_members, f'Missing archived maker source: {path}')
                maker_hashes = {member_hash(archive,n) for n in maker_members}
                require(len(maker_hashes) == 1, f'Conflicting archived maker copies: {path}')
                required = ['source/shared_nominal.py','source/shared_context_v4.py']
                require(all(n in names for n in required), f'Missing raw/shared generation members: {path}')
                nominal_hash, context_hash = [member_hash(archive,n) for n in required]
                raw_hash = source['raw_audit']['sha256']
                matches[label].append({'path':str(path), 'seed':summary['seed'], 'maker_sha256':next(iter(maker_hashes)),
                    'shared_nominal_sha256':nominal_hash, 'shared_context_sha256':context_hash,
                    'raw_audit_sha256':raw_hash, 'episodes':summary.get('episodes'),
                    'code_version':summary.get('code_version'), 'effects':summary.get('observable_effect_windows')})
            if matches.get(label) and matches[label][-1]['path'] == str(path): pin(path,archive_digest)
        except (BadZipFile, json.JSONDecodeError, UnicodeDecodeError) as exc:
            ignored.append({'file':str(path), 'reason':type(exc).__name__})
    rows=[]; by_label=maker_table.set_index('label')
    for label in LABEL_ORDER:
        records=matches[label]
        if not records:
            rows.append({'label':label,'status':'UNVERIFIED_NO_MATCHING_REVIEW_ARCHIVE','recorded_seed':None})
            continue
        keys=('seed','maker_sha256','shared_nominal_sha256','shared_context_sha256','raw_audit_sha256','episodes','code_version','effects')
        require(len({tuple(r[k] for k in keys) for r in records}) == 1,
                f'Conflicting generation records for the same learning/raw files: {label}')
        r=records[0]
        current = (r['maker_sha256'] == by_label.loc[label,'maker_sha256']
                   and r['shared_nominal_sha256'] == shared['nominal_sha256'] and r['shared_context_sha256'] == shared['sha256'])
        rows.append({'label':label,'status':'VERIFIED_CSV_SOURCE_BINDING' if current else 'ARCHIVED_SOURCE_DIFFERS_FROM_CURRENT',
            'recorded_seed':r['seed'],'recorded_default_seed_matches':bool(r['seed'] == by_label.loc[label,'seed']),
            'learning_sha256':sources[label]['sha256'], **{k:r[k] for k in keys if k.endswith('sha256')},
            'archive_paths':[r['path'] for r in records], 'recorded_episodes':r['episodes'],
            'recorded_code_version':r['code_version'], 'recorded_observable_effect_windows':r['effects']})
    seeds=[r['recorded_seed'] for r in rows]
    return {'records':rows,'all_csvs_bound_to_current_sources':all(r['status']=='VERIFIED_CSV_SOURCE_BINDING' for r in rows),
        'distinct_recorded_seeds':len(set(seeds)) == 6 if all(type(x) is int for x in seeds) else None,
        'ignored_or_unreadable_candidate_archives':ignored,
        'scope':'Actual learning/raw/shared source byte binding; archived assertions are not rerun physics or causal proofs.'}

def compare_exact_records(expected, actual, description):
    keys = ['label', 'episode_id', 'episode_step', 'window_id']
    require(len(actual) == len(expected), f'{description}: different row counts')
    require(not expected.duplicated(keys).any() and not actual.duplicated(keys).any(), f'{description}: duplicate record identity')
    left, right = pd.MultiIndex.from_frame(expected[keys]), pd.MultiIndex.from_frame(actual[keys])
    positions = left.get_indexer(right)
    require((positions >= 0).all(), f'{description}: missing, extra or relabelled records')
    aligned = expected.iloc[positions].reset_index(drop=True)
    for col in CANONICAL_COLUMNS:
        require(np.array_equal(aligned[col].astype(str).to_numpy(), actual[col].astype(str).to_numpy()),
                f'{description}: source value changed in {col}')
    return positions


def reproduce_episode_split(episode_manifest, policy):
    ratios = policy.get('ratios')
    seed = policy.get('seed')
    require(type(seed) is int and seed >= 0 and isinstance(ratios, list) and len(ratios) == 3
            and all(0 < r < 1 for r in ratios) and abs(sum(ratios) - 1) < 1e-12, 'Invalid Step 7 split seed/ratios')
    rng = np.random.default_rng(seed)
    mapping = {}
    for label in LABEL_ORDER:
        ids = sorted(episode_manifest.loc[episode_manifest.label == label, 'global_episode_id'].tolist())
        require(len(ids) >= 3, f'Need at least 3 episodes for {label}')
        rng.shuffle(ids)
        n = len(ids); n_train = max(1, int(round(n * ratios[0]))); n_val = max(1, int(round(n * ratios[1])))
        n_test = n - n_train - n_val
        if n_test < 1:
            n_train = max(1, n_train - 1); n_test = n - n_train - n_val
        if n_test < 1:
            n_val = max(1, n_val - 1); n_test = n - n_train - n_val
        require(min(n_train, n_val, n_test) >= 1, 'Invalid episode split allocation')
        for name, group in [('train', ids[:n_train]), ('val', ids[n_train:n_train + n_val]), ('test', ids[n_train + n_val:])]:
            mapping.update({episode: name for episode in group})
    actual = episode_manifest.set_index('global_episode_id').split.to_dict()
    require(actual == mapping, 'Episode split does not reproduce the recorded Step 7 seed/rounding policy')


def split_integrity_audit(splits, source, build, episode_manifest, unified):
    ep_sets = {name: set(frame.episode_id) for name, frame in splits.items()}
    pairwise = {f'{a}_vs_{b}': {'overlap_count': len(ep_sets[a] & ep_sets[b]),
                              'examples': sorted(ep_sets[a] & ep_sets[b])[:10]} for a, b in combinations(splits, 2)}
    require(all(item['overlap_count'] == 0 for item in pairwise.values()), 'Episodes overlap across train/val/test')
    combined = pd.concat([frame.assign(__split=name) for name, frame in splits.items()], ignore_index=True)
    expected = source.copy()
    expected['__global_episode_id'] = expected.label + '::' + expected.episode_id
    collision = source.groupby('episode_id').label.nunique().gt(1)
    # Revised Step 7 rewrites ALL episode IDs when any cross-class collision exists.
    if collision.any():
        expected['episode_id'] = expected['__global_episode_id']
    positions = compare_exact_records(expected, combined, 'Original maker CSVs -> split union')
    compare_exact_records(expected, unified, 'Original maker CSVs -> unified CSV')
    combined['__global_episode_id'] = expected['__global_episode_id'].to_numpy()[positions]
    require(set(episode_manifest.split) == set(SPLIT_FILES) and episode_manifest.global_episode_id.is_unique,
            'Invalid/duplicate episode split manifest')
    observed = combined.groupby('__global_episode_id', sort=True).agg(
        label=('label', 'first'), split=('__split', 'first'), row_count=('label', 'size'),
        min_episode_step=('episode_step', 'min'), max_episode_step=('episode_step', 'max'),
        min_window_id=('window_id', 'min'), max_window_id=('window_id', 'max'),
        start_timestamp=('timestamp', 'min'), end_timestamp=('timestamp', 'max'))
    manifest = episode_manifest.set_index('global_episode_id').sort_index()
    require(observed.index.equals(manifest.index), 'Episode manifest does not cover the supplied episodes exactly once')
    for col in observed:
        require(col in manifest, f'Missing episode manifest column: {col}')
        require(np.array_equal(observed[col].astype(str), manifest[col].astype(str)), f'Episode manifest mismatch: {col}')
    require(np.array_equal(manifest.label + '::' + manifest.original_episode_id, manifest.index),
            'Episode manifest original/global IDs disagree')
    require(manifest.observable_window_seconds.eq('1').all(), 'Episode manifest window duration differs')
    require(np.array_equal(integers(manifest.episode_duration_seconds, 'episode duration'), observed.row_count),
            'Episode durations differ from one-second record counts')
    reproduce_episode_split(episode_manifest, build['policy'])
    counts = {}
    for name, frame in splits.items():
        lengths = frame.groupby('episode_id').size()
        counts[name] = {'rows': len(frame), 'episodes': len(lengths),
                       'sequences_stride1': {str(length): int((lengths - length + 1).clip(lower=0).sum()) for length in LENGTHS}}
        require(counts[name] == build['counts'].get(name), f'{name}: row/episode/sequence counts differ from Step 7')
    class_rows = combined.label.value_counts().reindex(LABEL_ORDER, fill_value=0).to_dict()
    expected_n = build['policy']['expected_rows_per_class']
    require(all(n == expected_n for n in class_rows.values()), 'Class totals differ from source policy')
    report = {'counts': counts, 'pairwise_episode_overlap': pairwise, 'no_episode_overlap': True,
              'total_split_rows': len(combined), 'expected_total_rows': expected_n * 6,
              'rows_by_label_across_splits': class_rows, 'all_rows_assigned': True,
              'exact_original_record_coverage': True, 'all_canonical_values_preserved': True,
              'unified_csv_matches_original_sources': True, 'episode_manifest_matches_records': True,
              'recorded_split_seed_reproduced': True, 'source_episode_id_cross_label_collisions': int(collision.sum()),
              'source_id_rewrite_applied': bool(collision.any()),
              'scope': 'Exact records and supplied episode IDs; not proof that distinct episodes are statistically independent.'}
    return report, combined


def episode_trajectory_hashes(df):
    rows = []
    work = df.copy()
    if '__obs_hash' not in work:
        work['__obs_hash'] = row_hashes(work, OBSERVABLE_FIELDS)
    for (label, episode), group in work.groupby(['label', 'episode_id'], sort=False):
        group = group.sort_values(['episode_step', 'window_id'], kind='mergesort')
        digest = hashlib.sha256(b'observable22-trajectory-v1\0' + b''.join(bytes.fromhex(v) for v in group['__obs_hash'])).hexdigest()
        rows.append({'label': label, 'episode_id': episode, 'length': len(group), 'trajectory_sha256': digest,
                     '__split': group['__split'].iloc[0] if '__split' in group else 'unspecified'})
    return pd.DataFrame(rows)


def l10_sequence_hashes(df):
    rows = []
    for (label, episode), group in df.groupby(['label', 'episode_id'], sort=False):
        group = group.sort_values(['episode_step', 'window_id'], kind='mergesort')
        digests = [bytes.fromhex(v) for v in group['__obs_hash']]
        steps = group.episode_step.to_numpy()
        for end in range(9, len(group)):
            signature = hashlib.sha256(b'observable22-L10-v1\0' + b''.join(digests[end - 9:end + 1])).hexdigest()
            rows.append({'label': label, 'episode_id': episode, '__split': group['__split'].iloc[0],
                         'sequence_end_step': int(steps[end]), 'sequence_sha256': signature})
    return pd.DataFrame(rows, columns=['label', 'episode_id', '__split', 'sequence_end_step', 'sequence_sha256'])


def cross_label_collision_summary(df, hash_col):
    distinct = df[[hash_col, 'label']].drop_duplicates()
    counts = distinct.groupby(hash_col).size()
    cross = counts[counts > 1].index
    examples = distinct[distinct[hash_col].isin(cross[:20])].groupby(hash_col).label.agg(list)
    return {'num_cross_label_hashes': len(cross),
            'rows_in_cross_label_hashes': int(df[hash_col].isin(cross).sum()),
            'examples': [{'hash': str(k), 'labels': list(v)} for k, v in examples.items()]}


def cross_split_overlap(df, hash_col):
    report = {}
    for a, b in combinations(SPLIT_FILES, 2):
        left, right = (df.loc[df['__split'] == name] for name in (a, b))
        lset, rset = set(left[hash_col]), set(right[hash_col])
        shared = lset & rset
        same_labels, different_labels = set(), set()
        llabels = left[left[hash_col].isin(shared)].groupby(hash_col).label.agg(set).to_dict()
        rlabels = right[right[hash_col].isin(shared)].groupby(hash_col).label.agg(set).to_dict()
        for signature in shared:
            if llabels[signature] & rlabels[signature]:
                same_labels.add(signature)
            if len(llabels[signature] | rlabels[signature]) > 1:
                different_labels.add(signature)
        report[f'{a}_vs_{b}'] = {'shared_signatures': len(shared), 'same_label_shared_signatures': len(same_labels),
            'cross_label_shared_signatures': len(different_labels),
            'left_rows_in_shared_signatures': int(left[hash_col].isin(shared).sum()),
            'right_rows_with_left_signature': int(right[hash_col].isin(shared).sum()),
            'right_row_overlap_fraction': float(right[hash_col].isin(shared).mean()) if len(right) else None,
            'example_signatures': sorted(shared)[:10]}
    return report


def context_distribution_table(df):
    rows = []
    for field in CONTEXT_FIELDS:
        tab = pd.crosstab(df[field], df.label).reindex(columns=LABEL_ORDER, fill_value=0)
        for value, counts in tab.iterrows():
            rows.append({'field': field, 'value': value, 'total': int(counts.sum()),
                         'dominant_label': str(counts.idxmax()), 'dominant_label_fraction': float(counts.max() / counts.sum()),
                         **{'count_' + label: int(counts[label]) for label in LABEL_ORDER}})
    return pd.DataFrame(rows)


def context_profile_audit(df, shared):
    counts = df.groupby(CONTEXT_FIELDS + ['label']).size().unstack('label', fill_value=0).reindex(columns=LABEL_ORDER, fill_value=0)
    valid = set(shared['profiles'])
    unknown = [profile for profile in counts.index if profile not in valid]
    table = counts.reset_index()
    table['in_declared_shared_pool'] = [tuple(row) in valid for row in table[CONTEXT_FIELDS].itertuples(index=False, name=None)]
    expected_weights = dict(zip(shared['profiles'], shared['weights']))
    table['declared_probability'] = [float(expected_weights.get(tuple(row), 0)) for row in table[CONTEXT_FIELDS].itertuples(index=False, name=None)]
    empirical = counts.reindex(pd.MultiIndex.from_tuples(shared['profiles'], names=CONTEXT_FIELDS), fill_value=0)
    # Include out-of-pool mass in total variation, rather than normalizing it away.
    totals = df.label.value_counts().reindex(LABEL_ORDER, fill_value=0)
    tv = {}
    for label in LABEL_ORDER:
        expected_mass_diff = np.abs(empirical[label].to_numpy() / totals[label] - shared['weights']).sum()
        outside_mass = float(counts.loc[unknown, label].sum() / totals[label]) if unknown else 0.
        tv[label] = float((expected_mass_diff + outside_mass) / 2)
    return table, {'declared_profile_count': len(valid), 'observed_profile_count': len(counts),
        'out_of_pool_profile_count': len(unknown), 'out_of_pool_rows': int(counts.loc[unknown].sum().sum()) if unknown else 0,
        'total_variation_from_declared_mixture_by_label': tv,
        'interpretation': 'Descriptive empirical association; no p-value, independence proof or score-driven regeneration rule.'}


def classification_metrics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    require(len(truth) == len(prediction) and set(prediction) <= set(LABEL_ORDER), 'Invalid probe prediction')
    rows = {}
    for label in LABEL_ORDER:
        tp = int(((truth == label) & (prediction == label)).sum())
        fp = int(((truth != label) & (prediction == label)).sum())
        fn = int(((truth == label) & (prediction != label)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.
        recall = tp / (tp + fn) if tp + fn else 0.
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
        rows[label] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': int((truth == label).sum())}
    return {'accuracy': float(np.mean(truth == prediction)) if len(truth) else None,
            'macro_f1': float(np.mean([r['f1'] for r in rows.values()])),
            'balanced_accuracy_six_class_macro_recall': float(np.mean([r['recall'] for r in rows.values()])),
            'rows': len(truth), 'per_class': rows}


def context_majority_fit(train):
    table = train.groupby(CONTEXT_FIELDS + ['label']).size().unstack('label', fill_value=0).reindex(columns=LABEL_ORDER, fill_value=0)
    mapping = {tuple(key): LABEL_ORDER[int(np.argmax(row))] for key, row in zip(table.index, table.to_numpy())}
    majority = LABEL_ORDER[int(train.label.value_counts().reindex(LABEL_ORDER, fill_value=0).to_numpy().argmax())]
    return mapping, majority


def majority_predict(frame, mapping, majority):
    keys = list(frame[CONTEXT_FIELDS].itertuples(index=False, name=None))
    return np.array([mapping.get(key, majority) for key in keys]), np.array([key not in mapping for key in keys])


def context_majority_mapping_probe(train, test):
    mapping, majority = context_majority_fit(train)
    prediction, unseen = majority_predict(test, mapping, majority)
    return {'status': 'COMPLETED', 'method': 'train-context-signature majority mapping',
            **classification_metrics(test.label, prediction), 'test_unseen_context_signature_rows': int(unseen.sum()),
            'fallback_label': majority, 'tie_rule': 'first in canonical label order'}


def logistic_fit_diagnostics(model, captured):
    """Check numerical termination, not predictive validity or absence of shortcuts."""
    from sklearn.exceptions import ConvergenceWarning
    classifier = model.named_steps['clf']
    iterations = np.asarray(classifier.n_iter_, dtype=np.int64)
    warnings_seen = [{'category': item.category.__name__, 'message': str(item.message)} for item in captured]
    convergence_warnings = sum(issubclass(item.category, ConvergenceWarning) for item in captured)
    limit_reached = bool((iterations >= classifier.max_iter).any())
    converged = bool(iterations.size > 0 and not convergence_warnings and not limit_reached)
    if convergence_warnings:
        status = 'NOT_CONVERGED'
    elif not converged:
        status = 'CONVERGENCE_NOT_CONFIRMED'
    else:
        status = 'COMPLETED_WITH_WARNINGS' if warnings_seen else 'COMPLETED'
    return {'status': status, 'converged': converged,
        'max_iter': int(classifier.max_iter), 'iterations': iterations.tolist(),
        'iteration_limit_reached': limit_reached, 'convergence_warning_count': int(convergence_warnings),
        'tol': float(classifier.tol), 'class_weight': classifier.class_weight, 'warnings': warnings_seen,
        'convergence_criterion': 'No ConvergenceWarning and every n_iter_ < max_iter at the unchanged solver tolerance.',
        'held_out_metrics_require_converged_fit': True}


def fit_logistic_context_probe(train):
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        from threadpoolctl import threadpool_limits
        import sklearn
    except ImportError as exc:
        return None, {'status': 'SKIPPED_MISSING_DEPENDENCY', 'reason': str(exc)}
    model = Pipeline([('onehot', ColumnTransformer([('context', OneHotEncoder(handle_unknown='ignore'), CONTEXT_FIELDS)], remainder='drop')),
                      ('clf', LogisticRegression(C=1.0, max_iter=PROBE_MAX_ITER, solver='lbfgs', random_state=PROBE_SEED))])
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
        warnings.simplefilter('always')
        model.fit(train[CONTEXT_FIELDS], train.label)
    diagnostics = logistic_fit_diagnostics(model, captured)
    print(f"  Context logistic: {diagnostics['status']}; iterations={diagnostics['iterations']}/{PROBE_MAX_ITER}", flush=True)
    return model, {**diagnostics,
        'method': 'train-only one-hot 7-context-field logistic regression', 'sklearn_version': sklearn.__version__,
        'C': 1.0, 'solver': 'lbfgs', 'seed': PROBE_SEED, 'test_or_val_used_for_fit_or_selection': False}


def optional_logistic_context_probe(train, test):
    model, report = fit_logistic_context_probe(train)
    return report if model is None else {**report, **classification_metrics(test.label, model.predict(test[CONTEXT_FIELDS]))}


def prior_references(train, truth):
    p = pd.Series(truth).value_counts().reindex(LABEL_ORDER, fill_value=0).to_numpy(dtype=float)
    p /= p.sum()
    q = train.label.value_counts().reindex(LABEL_ORDER, fill_value=0).to_numpy(dtype=float); q /= q.sum()
    references = {}
    for name, probabilities in [('uniform_independent_random', np.full(6, 1/6)), ('train_prior_independent_random', q)]:
        den = p + probabilities
        references[name] = {'population_limit_macro_f1': float(np.divide(2*p*probabilities, den, out=np.zeros(6), where=den > 0).mean()),
                            'expected_accuracy': float(p @ probabilities)}
    return {'references': references,
            'note': 'Population-limit F1 references, not exact finite-sample expected F1 or uncertainty intervals. '
                    'Macro-F1 is not universally 1/6 when class proportions differ.'}


def run_context_probes(splits, out):
    train = splits['train']
    mapping, majority = context_majority_fit(train)
    logistic, settings = fit_logistic_context_probe(train)
    report = {'fit_split': 'train', 'features': CONTEXT_FIELDS, 'tie_rule': 'canonical label order',
              'logistic_fit': settings, 'results': {}, 'no_hyperparameter_or_seed_selection': True,
              'scope': 'Single-window diagnostic; no time-series context is supplied to these probes. '
                       'L10-common only aligns evaluation targets, and does not make this a temporal model.'}
    summaries = []
    save_json(out / '41A_context_probe_fit.json', {'train_majority_label': majority,
        'majority_mapping': [{'context': list(key), 'predicted_label': value} for key, value in sorted(mapping.items())],
        'logistic': settings})
    if logistic is not None:
        classifier = logistic.named_steps['clf']
        np.savez_compressed(out / '41A_context_logistic_parameters.npz', coefficients=classifier.coef_, intercept=classifier.intercept_,
                            classes=classifier.classes_.astype(str), feature_names=logistic.named_steps['onehot'].get_feature_names_out().astype(str))
    for split in ('val', 'test'):
        frame = splits[split]
        majority_pred, unseen = majority_predict(frame, mapping, majority)
        predictions = {'majority_mapping': majority_pred, 'train_majority_baseline': np.full(len(frame), majority)}
        if logistic is not None:
            predictions['logistic'] = logistic.predict(frame[CONTEXT_FIELDS])
        order = frame.sort_values(['episode_id', 'episode_step', 'window_id'], kind='mergesort')
        common = pd.Series(False, index=frame.index)
        common.loc[order.index] = order.groupby('episode_id', sort=False).cumcount().ge(9).to_numpy()
        pred_frame = frame[METADATA_FIELDS].copy()
        pred_frame['common_population'] = common.astype(int)
        pred_frame['unseen_train_context_signature'] = unseen.astype(int)
        for model, prediction in predictions.items():
            pred_frame['pred_' + model] = prediction
        save_csv(out / f'41A_context_probe_predictions_{split}.csv', pred_frame)
        report['results'][split] = {}
        for population, mask in [('all_windows', np.ones(len(frame), dtype=bool)), ('common_L10_endpoints', common.to_numpy())]:
            require(mask.any(), f'{split}: empty {population} context evaluation')
            truth = frame.loc[mask, 'label'].to_numpy()
            result = {'rows': int(mask.sum()), 'unseen_context_rows': int(unseen[mask].sum()),
                      'prior_references': prior_references(train, truth), 'models': {}}
            for model, prediction in predictions.items():
                metrics = classification_metrics(truth, prediction[mask])
                metrics['fit_status'] = settings['status'] if model == 'logistic' else 'COMPLETED'
                metrics['fit_converged'] = settings.get('converged') if model == 'logistic' else None
                result['models'][model] = metrics
                summaries.append({'split': split, 'population': population, 'model': model,
                                  'rows': metrics['rows'], 'accuracy': metrics['accuracy'], 'macro_f1': metrics['macro_f1'],
                                  'fit_status': metrics['fit_status'], 'fit_converged': metrics['fit_converged']})
            report['results'][split][population] = result
    save_csv(out / '41A_context_probe_metrics.csv', pd.DataFrame(summaries))
    return report
def align_split_effects(splits, source_effects, source_collision, build_root):
    """Check separate Step 7 flags against the independently checked raw companions."""
    expected = source_effects.copy()
    if source_collision:
        expected['episode_id'] = expected.label + '::' + expected.episode_id
    keys = ['label', 'episode_id', 'episode_step', 'window_id']
    require(not expected.duplicated(keys).any(), 'Duplicate source effect identities')
    results = {}
    for split, frame in splits.items():
        path = build_root/observability_audit_contract()['split_files'][split]
        stored = read_csv(path)
        require(list(stored.columns) == METADATA_FIELDS+AUDIT_FIELDS and len(stored) == len(frame),
                f'{split}: observable-effect schema/count mismatch')
        for col in METADATA_FIELDS:
            require(np.array_equal(stored[col].astype(str), frame[col].astype(str)),
                    f'{split}: observable-effect metadata mismatch: {col}')
        positions = pd.MultiIndex.from_frame(expected[keys]).get_indexer(pd.MultiIndex.from_frame(frame[keys]))
        require((positions >= 0).all(), f'{split}: effect source record missing')
        selected = expected.iloc[positions].reset_index(drop=True)
        for col in METADATA_FIELDS:
            require(np.array_equal(stored[col].astype(str), selected[col].astype(str)),
                    f'{split}: raw/split diagnostic metadata differs: {col}')
        for col in AUDIT_FIELDS:
            values = integers(stored[col], col)
            require(np.array_equal(values, selected[col].to_numpy()), f'{split}: raw/split effect flags differ: {col}')
            stored[col] = values
        results[split] = stored
    return results


def quiet_context_records(frame, audit, length, include_features=True):
    """Build genuine contiguous causal contexts, then select no-projection-change contexts.

    Filtering quiet rows before building contexts would fabricate adjacency and
    erase changes in intervening windows. Only the original 22 fields become
    features; labels and counterfactual flags are selection/diagnostic metadata.
    """
    require(length in (1, 10), 'Fingerprint diagnostic lengths are prespecified as L1 and L10')
    frame, audit = frame.reset_index(drop=True), audit.reset_index(drop=True)
    require(len(frame) == len(audit), 'Effect annotation row count differs')
    for col in METADATA_FIELDS:
        require(np.array_equal(frame[col].astype(str), audit[col].astype(str)), f'Fingerprint annotation mismatch: {col}')
    flags = integers(audit.audit_observable_effect_present, 'observable effect flag')
    require(np.isin(flags, [0, 1]).all(), 'Invalid observable effect flag')
    eligible, endpoints, features, coverage = [], [], [], []
    # Field/value pairs prevent identical category spellings in different fields
    # from collapsing. Token counts preserve the bag of complete L windows.
    tokens = None
    if include_features:
        # Reuse identical strings across windows; these are text encodings, not
        # fitted feature IDs. DictVectorizer still learns its columns from train only.
        encodings = {field:{str(value):json.dumps([field,str(value)], ensure_ascii=False, separators=(',', ':'))
                           for value in frame[field].unique()} for field in OBSERVABLE_FIELDS}
        tokens = [[encodings[field][str(value)] for field,value in zip(OBSERVABLE_FIELDS,row)]
                  for row in frame[OBSERVABLE_FIELDS].itertuples(index=False,name=None)]
    for episode, group in frame.groupby('episode_id', sort=False):
        ordered = group.sort_values(['episode_step','window_id'], kind='mergesort')
        ids = ordered.index.to_numpy()
        require(group.label.nunique() == 1, 'Mixed labels in fingerprint episode')
        for col in ('episode_step','window_id'):
            require(np.all(np.diff(integers(ordered[col], col)) == 1), f'Noncontiguous fingerprint context: {col}')
        for end in range(length-1, len(ids)):
            context = ids[end-length+1:end+1]
            current = int(flags[context[-1]]); history = int(flags[context[:-1]].sum())
            eligible.append((str(group.label.iloc[0]), current, history))
            if current+history == 0:
                endpoints.append(int(context[-1]))
                if include_features:
                    features.append(dict(Counter(token for row in context for token in tokens[row])))
    for label in LABEL_ORDER:
        rows = [(current,history) for cls,current,history in eligible if cls == label]
        coverage.append({'label':label, 'sequence_length':length, 'all_contexts':len(rows),
            'current_changed':sum(current > 0 for current,history in rows),
            'history_only_changed':sum(current == 0 and history > 0 for current,history in rows),
            'no_observable_change':sum(current+history == 0 for current,history in rows)})
    metadata = frame.iloc[endpoints][METADATA_FIELDS].copy().reset_index(drop=True)
    metadata['sequence_length'] = length
    metadata['context_effect_window_count'] = 0
    return metadata, features, coverage


def fit_logistic_fingerprint_probe(features, labels):
    """A fixed diagnostic classifier; no validation/test vocabulary or tuning."""
    if len(labels) == 0 or len(set(labels)) < 2:
        return None, {'status':'SKIPPED_INSUFFICIENT_TRAIN_CLASSES', 'train_rows':len(labels)}
    try:
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from threadpoolctl import threadpool_limits
        import sklearn
    except ImportError as exc:
        return None, {'status':'SKIPPED_MISSING_DEPENDENCY', 'reason':str(exc)}
    model = Pipeline([('tokens', DictVectorizer(sparse=True)),
        ('clf', LogisticRegression(C=1.0, max_iter=PROBE_MAX_ITER, solver='lbfgs', random_state=PROBE_SEED))])
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
        warnings.simplefilter('always')
        model.fit(features, labels)
    diagnostics = logistic_fit_diagnostics(model, captured)
    print(f"  Fingerprint logistic: {diagnostics['status']}; iterations={diagnostics['iterations']}/{PROBE_MAX_ITER}", flush=True)
    return model, {**diagnostics,
        'features':'22 field/value token counts over original contiguous windows; no metadata or flags',
        'fit_split':'train', 'vocabulary_fit_split':'train', 'train_rows':len(labels),
        'train_class_support':{label:int(np.count_nonzero(np.asarray(labels) == label)) for label in LABEL_ORDER},
        'C':1.0, 'solver':'lbfgs', 'seed':PROBE_SEED, 'sklearn_version':sklearn.__version__,
        'test_or_val_used_for_fit_or_selection':False}


def run_fingerprint_probes(splits, effects, out, skip_probes=False):
    """Conditional residual-signature diagnostics, never a benchmark acceptance score."""
    report = {'status':'SKIPPED_BY_REQUEST' if skip_probes else 'COMPLETED', 'lengths':[1,10],
        'selection':'zero observable projection differences in every window of the original context',
        'features':OBSERVABLE_FIELDS, 'fit_split':'train', 'results':{}, 'score_based_pass_fail_threshold':None,
        'interpretation':'A residual-signature diagnostic, not an attack detector or independence test. '
            'No projection change does not imply no class information. Conditioning on no change can '
            'select different contexts by attack class/target; class priors and episode dependence matter. '
            'No oracle bound, universal chance threshold, or seed-regeneration rule is inferred. '
            'Main evaluation populations and supplied scenario labels are unchanged.'}
    all_coverage=[]
    for length in (1,10):
        print(f'  Build no-observable-change L{length} fingerprint diagnostic', flush=True)
        train_meta, train_features, coverage = quiet_context_records(splits['train'], effects['train'], length, not skip_probes)
        all_coverage.extend(dict(row,split='train') for row in coverage)
        model, settings = (None, {'status':'SKIPPED_BY_REQUEST'}) if skip_probes else fit_logistic_fingerprint_probe(
            train_features, train_meta.label.to_numpy())
        result={'fit':settings,'splits':{}}
        if model is not None:
            classifier=model.named_steps['clf']
            np.savez_compressed(out/f'41A_no_change_L{length}_logistic_parameters.npz',
                coefficients=classifier.coef_, intercept=classifier.intercept_, classes=classifier.classes_.astype(str),
                feature_names=model.named_steps['tokens'].get_feature_names_out().astype(str))
        del train_features
        for split in ('val','test'):
            meta, features, coverage = quiet_context_records(splits[split], effects[split], length, not skip_probes)
            all_coverage.extend(dict(row,split=split) for row in coverage)
            row={'rows':len(meta), 'class_support':{label:int(meta.label.eq(label).sum()) for label in LABEL_ORDER},
                 'absent_true_labels':[label for label in LABEL_ORDER if not meta.label.eq(label).any()]}
            if not len(meta): row['status']='EMPTY_CONDITIONAL_POPULATION'
            elif model is None: row['status']=settings['status']
            else:
                prediction=model.predict(features)
                row.update(status=settings['status'], fit_converged=settings.get('converged'),
                           **classification_metrics(meta.label.to_numpy(),prediction))
                normal=row['per_class']['NORMAL']
                row['normal_recall']=normal['recall'] if normal['support'] else None
                row['normal_prediction_fraction']=float(np.mean(prediction == 'NORMAL'))
                row['prior_references']=prior_references(train_meta,meta.label)
                vocabulary=model.named_steps['tokens'].vocabulary_
                row['unknown_token_occurrences']=sum(count for feature in features for token,count in feature.items() if token not in vocabulary)
                row['total_token_occurrences']=sum(sum(feature.values()) for feature in features)
                output=meta.copy();output['prediction']=prediction;output['correct']=(prediction == meta.label).astype(int)
                save_csv(out/f'41A_no_change_L{length}_probe_predictions_{split}.csv',output)
            result['splits'][split]=row
        report['results'][str(length)]=result
        del model
    save_csv(out/'41A_observable_effect_context_coverage.csv',pd.DataFrame(all_coverage))
    save_json(out/'41A_no_observable_change_fingerprint_probes.json',report)
    return report


def run(project_root=BASE_DIR, maker_dir=None, maker_review_dirs=(), output_dir=None, skip_probes=False):
    global PINNED_INPUTS
    PINNED_INPUTS = {}
    root = Path(project_root).expanduser().resolve()
    def resolve(value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else root / path).resolve()
    makers = resolve(maker_dir) if maker_dir else root
    dataset = root / 'dataset'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = resolve(output_dir) if output_dir else root / 'outputs/shared_nominal_v2/41A_benchmark_Aprime_audit' / ('run_' + stamp)
    # An import has no filesystem effects. A new run never overwrites any existing directory.
    out.mkdir(parents=True, exist_ok=False)
    protocol = {'version': PROTOCOL_VERSION, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': sha256(Path(__file__)), 'project_root': str(root), 'maker_dir': str(makers),
        'features': OBSERVABLE_FIELDS, 'context_probe_features': CONTEXT_FIELDS,
        'labels': LABEL_ORDER, 'duplicate_scopes': ['observable_row', 'whole_episode', 'causal_stride1_L10_sequence'],
        'hash_encoding': 'SHA-256 over UTF-8 compact JSON observable arrays; ordered fixed-size row digests for trajectories',
        'probe_fit_split': 'train', 'probe_evaluation_splits': ['val', 'test'],
        'probe_populations': ['all_windows', 'common_L10_endpoints'],
        'logistic_settings': {'C': 1.0, 'solver': 'lbfgs', 'max_iter': PROBE_MAX_ITER, 'random_state': PROBE_SEED,
                              'tol': 1e-4, 'class_weight': None},
        'convergence_revision': {'previous_max_iter': 1000, 'fixed_max_iter': PROBE_MAX_ITER,
            'fitting_change': 'Iteration budget only; existing sklearn defaults and preprocessing preserved.',
            'automatic_retries': False, 'held_out_score_based_adjustment': False},
        'data_or_seed_selection': False, 'score_based_pass_fail_threshold': None,
        'neural_model_training_required': False, 'diagnostic_probe_fitting_enabled':not skip_probes,
        'conditional_fingerprint_lengths':[1,10],
        'label_policy': 'retain supplied scenario/episode labels including quiet windows',
        'limitations': ['Exact duplicate audit does not detect near-duplicates or establish independent episodes.',
                       'Distinct seed integers do not prove independent random processes.',
                       'Low context-only scores do not prove absence of every shortcut.',
                       'No physical realism, external validity or encoder/aggregator necessity is established.'],
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__}}
    save_json(out / 'protocol.json', protocol)
    manifest = {'status': 'RUNNING', 'protocol_version': PROTOCOL_VERSION, 'script_sha256': protocol['script_sha256'],
                'protocol_sha256': sha256(out / 'protocol.json'), 'started_at_utc': protocol['created_at_utc']}
    save_json(out / 'run_manifest.json', manifest)
    report = {'protocol_version': PROTOCOL_VERSION, 'scope': protocol['limitations']}
    try:
        print('Step 41: current shared nominal data audit (read-only inputs)', flush=True)
        print('[1/7] Verify revised Step 7 identity, source CSVs and output hashes', flush=True)
        build = load_build_contract(dataset)
        manifest['dataset_id'] = report['dataset_id'] = build['dataset_id']
        report['step7_build_manifest_sha256'] = PINNED_INPUTS[str((dataset / 'shared_nominal_v2/build_manifest.json').resolve())]
        print('[2/7] Audit six current maker sources and recorded generation lineage', flush=True)
        maker_table, maker_summary = static_maker_source_audit(makers)
        shared = load_shared_pool(makers)
        save_csv(out / '41A_maker_static_audit.csv', maker_table)
        report['maker_static_audit'] = maker_summary
        directories = [dataset] + [resolve(value) for value in maker_review_dirs]
        require(all(path.is_dir() for path in directories), 'Maker review directory not found')
        lineage = maker_generation_lineage(dataset, directories, build, maker_table, shared)
        report['generation_lineage'] = lineage
        save_json(out / '41A_generation_lineage.json', lineage)
        print('[3/7] Validate source records and exact source/unified/split correspondence', flush=True)
        source_frames, source_stats, raw_effects, stream_records = [], [], [], []
        expected = build['policy']['expected_rows_per_class']
        for label in LABEL_ORDER:
            frame = validate_frame(read_csv(dataset / SOURCE_FILES[label]), SOURCE_FILES[label], label)
            require(len(frame) == expected, f'{label}: row count differs from Step 7 source policy')
            effects, raw_report = load_observability_audit(frame, dataset/SOURCE_FILES[label])
            # The companion was pinned by load_build_contract; recheck after its chunked read.
            pin(dataset/raw_report['raw_audit_file'])
            stream = raw_report['background_stream']
            number = LABEL_ORDER.index(label)+1
            parts = stream.split(':')
            require(len(parts) == 3 and all(p.isdecimal() for p in parts)
                    and int(parts[1]) == 0x4E4F4D and int(parts[2]) == number,
                    f'{label}: unexpected independent nominal stream record')
            require(stream not in {r['background_stream'] for r in stream_records}, 'Repeated nominal background stream across makers')
            record = next(r for r in lineage['records'] if r['label'] == label)
            if record.get('recorded_seed') is not None:
                require(int(parts[0]) == record['recorded_seed'], f'{label}: raw background and archived seed differ')
                require(record.get('recorded_observable_effect_windows') == raw_report['observable_effect_windows'],
                        f'{label}: archived observable-effect count differs')
            raw_effects.append(effects);stream_records.append(dict(raw_report,label=label))
            source_frames.append(frame)
            source_stats.append({'label': label, 'file': SOURCE_FILES[label], 'rows': len(frame), 'episodes': frame.episode_id.nunique(),
                                 'unique_context_signatures': len(frame[CONTEXT_FIELDS].drop_duplicates())})
        source = pd.concat(source_frames, ignore_index=True)
        del source_frames
        save_csv(out / '41A_source_dataset_summary.csv', pd.DataFrame(source_stats))
        report['source_dataset'] = {'total_rows': len(source), 'rows_per_class': expected,
                                    'episodes_total': sum(row['episodes'] for row in source_stats),
                                    'formal_default_300000_row_size': expected == 50000}
        for entry, stats in zip(lineage['records'], source_stats):
            if entry.get('recorded_episodes') is not None:
                require(entry['recorded_episodes'] == stats['episodes'], f"Generation episode count mismatch: {entry['label']}")
        build_root = dataset / 'shared_nominal_v2'
        splits = {name: validate_frame(read_csv(build_root / 'splits' / filename), name) for name, filename in SPLIT_FILES.items()}
        unified = validate_frame(read_csv(build_root / 'unified_observable_dataset.csv'), 'unified')
        episode_manifest = read_csv(build_root / 'splits/episode_split_manifest.csv')
        split_report, combined = split_integrity_audit(splits, source, build, episode_manifest, unified)
        report['split_integrity'] = split_report
        effects_by_split = align_split_effects(splits, pd.concat(raw_effects,ignore_index=True),
            split_report['source_id_rewrite_applied'], build_root)
        del raw_effects
        report['observable_effect_audit'] = {'source_binding':'RAW_AND_SPLITS_VERIFIED',
            'schema':observability_audit_contract(), 'streams':stream_records,
            'independent_default_stream_ids_distinct':True, 'paired_backgrounds_present':False,
            'interpretation':AUDIT_SCOPE+' Distinct recorded streams are not a proof of statistical independence.'}
        del source, unified
        print('[4/7] Audit observable-row and full-episode duplicate signatures', flush=True)
        combined['__obs_hash'] = row_hashes(combined, OBSERVABLE_FIELDS)
        report['cross_label_exact_observable_row_collisions'] = cross_label_collision_summary(combined, '__obs_hash')
        report['cross_split_exact_observable_row_overlap'] = cross_split_overlap(combined, '__obs_hash')
        trajectories = episode_trajectory_hashes(combined)
        save_csv(out / '41A_episode_trajectory_hashes.csv', trajectories.rename(columns={'__split': 'split'}))
        report['cross_label_exact_episode_trajectory_collisions'] = cross_label_collision_summary(trajectories, 'trajectory_sha256')
        report['cross_split_exact_episode_trajectory_overlap'] = cross_split_overlap(trajectories, 'trajectory_sha256')
        save_csv(out / '41A_episode_length_distribution_by_label.csv',
                 trajectories.groupby(['__split', 'label', 'length']).size().rename('episodes').reset_index().rename(columns={'__split': 'split'}))
        print('[5/7] Audit exact causal L10 duplicates across the three splits', flush=True)
        sequences = l10_sequence_hashes(combined)
        for split in SPLIT_FILES:
            require(int(sequences['__split'].eq(split).sum()) == build['counts'][split]['sequences_stride1']['10'],
                    f'{split}: L10 signature inventory count mismatch')
        save_csv(out / '41A_L10_sequence_hashes.csv', sequences.rename(columns={'__split': 'split'}))
        report['cross_split_exact_L10_sequence_overlap'] = cross_split_overlap(sequences, 'sequence_sha256')
        report['cross_label_exact_L10_sequence_collisions'] = cross_label_collision_summary(sequences, 'sequence_sha256')
        del sequences
        print('[6/7] Audit emitted context profiles and train-only fixed context probes', flush=True)
        save_csv(out / '41A_context_value_distribution_by_label.csv', context_distribution_table(combined))
        profiles, profile_report = context_profile_audit(combined, shared)
        save_csv(out / '41A_context_profile_distribution_by_label.csv', profiles)
        report['shared_context_empirical_audit'] = profile_report
        probes = ({'status':'SKIPPED_BY_REQUEST'} if skip_probes else run_context_probes(splits, out))
        report['context_only_probes'] = probes
        # Preserve recognizable legacy result keys without mislabelling successful fitting as leakage PASS.
        if not skip_probes:
            report['context_only_probe_majority_mapping'] = {'status': 'COMPLETED', **probes['results']['test']['all_windows']['models']['majority_mapping']}
            report['context_only_probe_logistic'] = probes['logistic_fit'] | probes['results']['test']['all_windows']['models'].get('logistic', {})
            for split in ('val', 'test'):
                require(probes['results'][split]['common_L10_endpoints']['rows'] == build['counts'][split]['sequences_stride1']['10'],
                        f'{split}: context probe common population mismatch')
        fingerprints = run_fingerprint_probes(splits,effects_by_split,out,skip_probes)
        report['no_observable_change_fingerprint_probes'] = fingerprints
        fits = {} if skip_probes else {'context_L1': probes['logistic_fit'],
            **{f'no_change_L{length}': value['fit'] for length, value in fingerprints['results'].items()}}
        convergence = {'status': ('SKIPPED_BY_REQUEST' if skip_probes else
            'ALL_CONVERGED' if all(fit.get('converged') is True for fit in fits.values()) else 'REVIEW_REQUIRED'),
            'fits': fits,
            'interpretation': 'Numerical convergence only. Review any other fitting warnings separately. '
                'A completed audit or converged fit does not prove chance-level performance or absence of shortcuts. '
                'Predictions from unconfirmed fits are retained for diagnosis, not as converged evidence.'}
        report['logistic_convergence'] = convergence
        save_json(out / '41A_logistic_convergence_audit.json', convergence)
        static_ok = all(maker_summary[key] for key in ('all_seeds_present',
            'all_use_shared_nominal_builder', 'all_use_shared_projection', 'all_use_shared_code_version',
            'maker_numbers_match', 'shared_runtime_uses_substrate_and_projection',
            'separate_seedsequence_calls_in_background_and_attack_runtime'))
        warnings_for_review = []
        if not static_ok:
            warnings_for_review.append('Shared nominal builder/projection/stream call patterns need review; static syntax is not an independence test.')
        if not lineage['all_csvs_bound_to_current_sources']:
            warnings_for_review.append('At least one CSV is not bound to the current maker/shared source by a matching generation archive.')
        if profile_report['out_of_pool_rows']:
            warnings_for_review.append('Some emitted seven-field context profiles are outside the current declared shared pool.')
        if any(row['shared_signatures'] for row in report['cross_split_exact_episode_trajectory_overlap'].values()):
            warnings_for_review.append('Exact whole-episode observable trajectories occur across splits; inspect before interpreting generalization.')
        if any(row['shared_signatures'] for row in report['cross_split_exact_L10_sequence_overlap'].values()):
            warnings_for_review.append('Exact L10 observable sequences occur across splits; report their extent and inspect their origin.')
        if report['cross_label_exact_episode_trajectory_collisions']['num_cross_label_hashes']:
            warnings_for_review.append('Some identical complete observable trajectories have different scenario labels.')
        if skip_probes:
            warnings_for_review.append('Diagnostic probe fitting/evaluation was skipped by request; no shortcut conclusion is available.')
        elif probes['logistic_fit']['status'] != 'COMPLETED':
            warnings_for_review.append('The logistic probe is unavailable or produced fitting warnings; inspect its status.')
        if not skip_probes and any(result['fit']['status'] != 'COMPLETED' for result in fingerprints['results'].values()):
            warnings_for_review.append('At least one conditional 22-field fingerprint probe is unavailable or has fitting warnings; inspect class support and status.')
        report['verdict'] = {'record_and_split_integrity': 'CONFIRMED',
            'logistic_convergence': convergence['status'],
            'current_maker_source_binding': 'CONFIRMED' if lineage['all_csvs_bound_to_current_sources'] else 'NOT_FULLY_VERIFIED',
            'static_rng_shared_sampler_patterns': 'MATCH' if static_ok else 'REVIEW',
            'statistical_independence': 'NOT_ESTABLISHED_BY_THIS_AUDIT',
            'context_shortcut_absence': 'NOT_PROVEN; inspect held-out diagnostics and conditional population coverage',
            'benchmark_Aprime_decision': 'INTEGRITY_CONFIRMED_REVIEW_DIAGNOSTICS' if not warnings_for_review else 'REVIEW_REQUIRED',
            'review_flags': warnings_for_review,
            'duplicate_interpretation': 'Single categorical rows may legitimately repeat. Exact overlap is reported, not automatically removed. '
                                        'Even a clean exact-duplicate check does not exclude near-duplicates or shared generator mechanisms.',
            'revision_interpretation': 'Use these diagnostics to delimit benchmark evidence; do not regenerate/select seeds to hit a desired context score.'}
        print('[7/7] Recheck inputs and finalize the audit', flush=True)
        for path, digest in PINNED_INPUTS.items():
            require(sha256(path) == digest, f'Input changed during audit: {path}')
        require(sha256(Path(__file__)) == protocol['script_sha256'], 'Audit code changed during the run')
        save_json(out / '41A_benchmark_Aprime_audit_report.json', report)
        manifest.update(status='COMPLETE', finished_at_utc=datetime.now(timezone.utc).isoformat(),
                        verdict=report['verdict'], input_sha256=dict(PINNED_INPUTS),
                        artifacts={p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'run_manifest.json'})
        save_json(out / 'run_manifest.json', manifest)
        if not skip_probes:
            display = pd.read_csv(out / '41A_context_probe_metrics.csv')
            print(display.to_string(index=False), flush=True)
        print(f"Logistic convergence: {convergence['status']}", flush=True)
        print(f"Audit COMPLETE: {report['verdict']['benchmark_Aprime_decision']}\nResults: {out}", flush=True)
        return out
    except Exception as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        save_json(out / '41A_benchmark_Aprime_audit_report.json', report)
        manifest.update(status='FAILED', error_type=type(exc).__name__, error=str(exc), input_sha256=dict(PINNED_INPUTS))
        save_json(out / 'run_manifest.json', manifest)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root', type=Path, default=BASE_DIR)
    parser.add_argument('--maker-dir', type=Path, help='Folder with six makers, shared_nominal.py and shared_context_v4.py')
    parser.add_argument('--maker-review-dir', type=Path, action='append', default=[], help='Additional folder containing original maker *_REVIEW.zip records; repeat as needed')
    parser.add_argument('--output-dir', type=Path, help='New empty output path; existing directories are preserved')
    parser.add_argument('--skip-probes', action='store_true', help='Run structural/duplicate/annotation checks without fitting or evaluating diagnostic classifiers')
    args = parser.parse_args(argv)
    run(args.project_root, args.maker_dir, args.maker_review_dir, args.output_dir, args.skip_probes)


if __name__ == '__main__':
    main()
