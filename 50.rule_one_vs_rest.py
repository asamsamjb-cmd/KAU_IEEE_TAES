#!/usr/bin/env python3
"""Step 50: retrospective independent Rule one-vs-rest on shared_nominal_v2 A.

Place beside the original 16.observable_rule_baseline.py in the project folder.
    python "50.rule_one_vs_rest.py"
    python "50.rule_one_vs_rest.py" --audit-only

Requires completed current Steps 7-10 and 16. No Step 49/B results required.
Accepts shared_nominal_v2_semantic_rules_v1 and shared_nominal_semantic_rules_v3.
Reuse the original Step 16 code/results; do not rerun calibration to upgrade them.
Uses saved pre-fusion scores and validation-frozen thresholds. No training,
score generation, threshold optimization, neural imports, GPU or network access.
When multiple matching Step 16 runs exist, select --step16-source RUN_DIRECTORY.
--project-root supports execution from another folder; --step16-source also
accepts a flat Step 16 ZIP. Existing outputs are never overwritten.
"""
import argparse
import hashlib
import io
import importlib.util
import json
import platform
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

LABELS = ('NORMAL', 'RF_LINK_JAMMING', 'GNSS_SPOOFING',
          'UNAUTHORIZED_TELECOMMAND_INJECTION', 'TELECOMMAND_REPLAY',
          'GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE')
MODELS = ('Rule_L1', 'Rule_L10')
SHORT = ('RF jamming', 'GNSS spoofing', 'TC injection', 'TC replay', 'Ground control')
SCORE_COLS = ['evidence_' + name for name in LABELS[1:]]
COUNT_NAMES = ('tp', 'fp', 'fn', 'tn', 'normal_fp', 'normal_n')
META = ['sequence_id', 'source_split', 'episode_id', 'episode_type', 'scenario_id',
        'sequence_length', 'observable_window_seconds', 'temporal_context_seconds',
        'sequence_start_row', 'sequence_end_row', 'sequence_start_step', 'sequence_end_step',
        'sequence_start_window_id', 'sequence_end_window_id', 'sequence_start_timestamp',
        'sequence_end_timestamp', 'start_label', 'label', 'label_id',
        'sequence_num_label_changes', 'sequence_has_label_transition', 'y_true', 'true_label']
ROOT = Path(__file__).resolve().parent
OUTPUT_REL = 'outputs/shared_nominal_v2/50_rule_one_vs_rest'
PROTOCOL_VERSION = 'step50_shared_nominal_independent_ovr_v2_1'
LEGACY_RULE_VERSION = 'shared_nominal_v2_semantic_rules_v1'
CURRENT_RULE_VERSION = 'shared_nominal_semantic_rules_v3'
DATA_REL = 'dataset/shared_nominal_v2'
PIPELINE_REL = 'outputs/shared_nominal_v2'
SEQ_REL = PIPELINE_REL + '/tst_sequences_finetuned_main'
EMB_REL = PIPELINE_REL + '/tst_embeddings_finetuned_main'
RULE_REL = PIPELINE_REL + '/16_observable_rule_baseline'
PROVENANCE = ['dataset_id', 'step8_run_id', 'step9_run_id', 'step10_run_id',
              'step16_run_id', 'rule_config_sha256', 'rule_spec_sha256']
EFFECT_COLUMNS = ['audit_current_observable_effect', 'audit_history_effect_window_count',
                  'audit_context_effect_window_count', 'audit_effect_stratum']
N_BOOTSTRAP = 20000
BOOTSTRAP_SEED = 2026100150
BATCH_SIZE = 200


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def object_digest(obj):
    return sha(json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8'))


def csv_bytes(data):
    return pd.read_csv(io.BytesIO(data), encoding='utf-8-sig', float_precision='round_trip',
                       dtype={c: str for c in ('episode_id', 'episode_type', 'scenario_id')}, keep_default_na=False)


def decisions(scores, thresholds):
    scores = np.asarray(scores, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    require(scores.ndim == 2 and scores.shape[1] == 5 and len(scores) > 0,
            'Evidence must have shape [N,5]')
    require(np.isfinite(scores).all() and ((scores >= 0) & (scores <= 3)).all(), 'Invalid evidence; current Step 16 scores must lie in [0,3]')
    require(thresholds.shape == (5,) and np.isfinite(thresholds).all() and
            (thresholds > 0).all(), 'Invalid threshold vector')
    # Match Step 16 arithmetic, including equality and canonical tie order.
    ratios = scores / thresholds
    binary = ratios >= 1.
    top = ratios.max(axis=1)
    fused = np.where(top >= 1., ratios.argmax(axis=1) + 1, 0)
    return binary, fused, {
        'eligible_attack_count': binary.sum(axis=1),
        'tie_at_max': ((ratios == top[:, None]).sum(axis=1) > 1) & (top >= 1.),
        'max_score_threshold_ratio': top,
    }


def align_pair(left, right, columns=META):
    result = []
    for d in (left, right):
        require(set(columns) <= set(d.columns), 'Missing paired metadata columns')
        require(not d[columns].isna().any().any(), 'Missing paired metadata values')
        require(d.sequence_id.is_unique, 'Duplicate sequence_id')
        d = d.sort_values('sequence_id').reset_index(drop=True)
        require(np.array_equal(d.sequence_id, np.arange(len(d))), 'Noncontiguous sequence IDs')
        result.append(d)
    for name in columns:
        require(np.array_equal(result[0][name], result[1][name]), 'Paired metadata mismatch: ' + name)
    return result


def episode_inventory(frame):
    require(not frame[['episode_id', 'y_true', 'sequence_end_step']].isna().any().any(),
            'Missing episode identity/label/endpoint')
    require(frame.episode_id.astype(str).str.strip().ne('').all(), 'Empty episode_id')
    require(not frame.duplicated(['episode_id', 'sequence_end_step']).any(), 'Duplicate endpoint')
    require(np.isin(frame.y_true, np.arange(6)).all(), 'Invalid ground truth')
    require(frame.groupby('episode_id').y_true.nunique().eq(1).all(),
            'Mixed-label episode: class-stratified bootstrap not applicable')
    inventory = frame.groupby('episode_id', sort=True).agg(
        label_id=('y_true', 'first'), n_endpoints=('y_true', 'size')).reset_index()
    inventory['label'] = np.array(LABELS)[inventory.label_id.to_numpy(dtype=np.int64)]
    lookup = {e: i for i, e in enumerate(inventory.episode_id)}
    codes = frame.episode_id.map(lookup).to_numpy(dtype=np.int64)
    return inventory, codes


def resolve_path(root, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root/path).resolve()


def import_rule(root, code=None):
    if code is None:
        candidates = sorted(root.glob('16.*observable_rule_baseline.py'))
        require(len(candidates) == 1,
                'Keep exactly one original 16.observable_rule_baseline.py beside Step 50, '
                'or supply --step16-code PATH. Found: ' + str(candidates))
        path = candidates[0]
    else:
        path = resolve_path(root, code)
    require(path.is_file(), 'Step 16 code missing: ' + str(path))
    name = '_step16_for_step50'
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'Cannot import Step 16 code')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True  # --audit-only must not create project __pycache__ files.
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    require(getattr(module, 'PROTOCOL_VERSION', None) in (LEGACY_RULE_VERSION, CURRENT_RULE_VERSION)
            and tuple(getattr(module, 'LABELS', ())) == LABELS,
            'Unsupported Step 16 code: expected shared_nominal_v2_semantic_rules_v1 or '
            'shared_nominal_semantic_rules_v3 with the canonical six labels. Found: '
            + str(getattr(module, 'PROTOCOL_VERSION', None)))
    return module


def resolve_source(root, value, rule, contract):
    if value is not None:
        source = resolve_path(root, value)
        require(source.exists(), 'Step 16 source not found: ' + str(source))
        return source
    candidates = []
    for path in sorted((root/rule.RESULT_REL).glob('run_*/run_manifest.json')):
        try:
            manifest = json.loads(path.read_text(encoding='utf-8-sig'))
        except (OSError, ValueError):
            continue
        source = manifest.get('source_contract', manifest.get('dataset_contract'))
        if manifest.get('status') == 'COMPLETE' and source == contract:
            candidates.append(path.parent)
    require(candidates, 'No COMPLETE Step 16 run matches the current shared_nominal_v2 data. '
            'Check the original completed results or supply --step16-source PATH; '
            'do not recalibrate just to change the protocol version.')
    require(len(candidates) == 1, 'Ambiguous matching Step 16 runs. Use --step16-source PATH; '
            'no newest/best run is selected automatically: ' + str(candidates))
    return candidates[0]


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha_file(path),
            'size_bytes': path.stat().st_size}


def safe_name(name):
    return (isinstance(name, str) and name not in ('', '.', '..') and
            '/' not in name and '\\' not in name and ':' not in name)


def source_payload(path):
    """Read a COMPLETE compatible manifest and verify every declared payload."""
    records, payload = [], {}
    archive = None
    try:
        if path.is_dir():
            def read(name):
                target = (path/name).resolve()
                require(target.is_relative_to(path), 'Step 16 member escapes source directory')
                raw = target.read_bytes()
                records.append({'path': str(target), 'sha256': sha(raw), 'size_bytes': len(raw)})
                return raw
        else:
            records.append(file_record(path))
            archive = zipfile.ZipFile(path)
            names = [i.filename for i in archive.infolist() if not i.is_dir()]
            require(len(names) == len(set(names)), 'Duplicate ZIP member')
            require(all(safe_name(n) for n in names), 'Use a flat Step 16 ZIP without nested paths')
            read = archive.read
        manifest_raw = read('run_manifest.json')
        manifest = json.loads(manifest_raw)
        legacy = isinstance(manifest.get('dataset_contract'), dict) and 'source_contract' not in manifest
        require(manifest.get('status') == 'COMPLETE' and
                (bool(manifest.get('run_id')) or (legacy and manifest.get('calibrate_only') is False)),
                'Step 16 source must be COMPLETE, not CALIBRATED or FAILED')
        for item in manifest.get('outputs', []):
            name = item.get('file')
            require(safe_name(name) and name != 'run_manifest.json' and name not in payload,
                    'Invalid/duplicate Step 16 output filename')
            raw = read(name)
            require(len(raw) == item.get('size_bytes') and sha(raw) == item.get('sha256'),
                    'Step 16 source size/hash mismatch: ' + name)
            payload[name] = raw
        required = {'frozen_rule_config.json', 'protocol.json', 'rule_spec.json',
                    'results_summary.csv', 'input_audit.json', 'data_audit.json'}
        required.update(f'{split}_predictions_{model}.csv'
                        for split in ('val', 'test') for model in MODELS)
        require(required <= payload.keys(), 'Missing required Step 16 outputs: ' + str(required-payload.keys()))
    finally:
        if archive is not None:
            archive.close()
    return manifest, sha(manifest_raw), payload, records


def compare_columns(actual, reference, columns, description):
    require(set(columns) <= set(actual.columns) and len(actual) == len(reference),
            description + ': missing columns/rows')
    for column in columns:
        # Metadata is read by Step 16 as strings before integer normalization.
        require(np.array_equal(actual[column].astype(str).to_numpy(),
                               reference[column].astype(str).to_numpy()),
                description + ': mismatch in ' + column)


def recheck_predictions(frame, meta, effects, model, split, config, expected, summary, rule,
                        legacy=False):
    require(frame.sequence_id.is_unique, 'Duplicate prediction sequence_id')
    frame = frame.sort_values('sequence_id').reset_index(drop=True)
    compare_columns(frame, meta, list(meta.columns), f'{split}/{model} current Step 10 metadata')
    if legacy:
        # v1 did not export v3 provenance, ID aliases or diagnostic sidecars.
        # Only fill absent values after manifest/input hashes and row identity pass;
        # existing values still go through all of the checks below.
        for names in (('y_true', 'true_label_id'), ('y_pred', 'pred_label_id')):
            present = [name for name in names if name in frame]
            require(present, 'Legacy Step 16 prediction IDs missing: ' + str(names))
            for name in names:
                if name not in frame: frame[name] = frame[present[0]]
        derived = {**expected, 'model': model, 'common_population': 1,
                   'rule_context_length': int(model[6:]),
                   'true_label': meta.label.to_numpy()}
        for column, value in derived.items():
            if column not in frame: frame[column] = value
        context = rule.effects_for_context(effects, int(model[6:]))
        for column in EFFECT_COLUMNS:
            if column not in frame: frame[column] = context[column].to_numpy()
    compare_columns(frame, rule.effects_for_context(effects, int(model[6:])), EFFECT_COLUMNS,
                    f'{split}/{model} observable-effect scope')
    for column, value in {**expected, 'model': model, 'common_population': 1,
                          'rule_context_length': int(model[6:])}.items():
        require(column in frame and frame[column].eq(value).all(),
                f'{split}/{model}: incorrect {column}')
    y = meta.label_id.to_numpy(dtype=np.int64)
    require(set(y) == set(range(6)), f'{split}: all six classes are required')
    for column in ('y_true', 'true_label_id', 'label_id'):
        require(column in frame and np.array_equal(frame[column], y), 'True label ID mismatch: ' + column)
    require(np.array_equal(frame.true_label, np.array(LABELS)[y]), 'True label name mismatch')
    require(set(SCORE_COLS) <= set(frame), 'Missing pre-fusion evidence scores')
    thresholds = config['models'][model]['thresholds']
    binary, fused, diag = decisions(frame[SCORE_COLS].to_numpy(dtype=np.float64), thresholds)
    original, original_aux = rule.classify(frame[SCORE_COLS].to_numpy(dtype=np.float64), thresholds)
    require(np.array_equal(original, fused), 'Step 50 fusion no longer matches current Step 16')
    for column in ('y_pred', 'pred_label_id'):
        require(np.array_equal(frame[column], fused), 'Reconstructed prediction mismatch: ' + column)
    require(np.array_equal(frame.pred_label, np.array(LABELS)[fused]), 'Predicted label mismatch')
    require(np.array_equal(frame.correct, y == fused), 'Correctness mismatch')
    for column in ('eligible_attack_count', 'tie_at_max'):
        require(np.array_equal(frame[column], diag[column]) and
                np.array_equal(original_aux[column], diag[column]), 'Diagnostic mismatch: ' + column)
    require(np.allclose(frame.max_score_threshold_ratio, diag['max_score_threshold_ratio'],
                        rtol=1e-14, atol=1e-14), 'Reconstructed ratio mismatch')
    computed = {**rule.metrics(y, fused), 'conflict_rate': float((binary.sum(1) > 1).mean()),
                'exact_tie_rate': float(diag['tie_at_max'].mean()),
                'normal_prediction_rate': float((fused == 0).mean())}
    selected = summary[(summary.split == split) & (summary.model == model)]
    require(len(selected) == 1, 'Ambiguous Step 16 summary row')
    row = selected.iloc[0]
    for column, value in {**expected, 'population': 'common_L10_endpoints',
                          'n_sequences': len(frame), 'n_episodes': frame.episode_id.nunique()}.items():
        if legacy and column in expected and column not in row.index:
            continue  # Provenance is authenticated at manifest level in v1.
        require(row[column] == value, 'Step 16 summary identity/count mismatch: ' + column)
    for column, value in computed.items():
        require(np.isclose(row[column], value, rtol=0, atol=1e-12),
                'Step 16 multiclass summary mismatch: ' + column)
    if split == 'val':
        for saved, column in [('validation_macro_f1', 'macro_f1'),
                              ('validation_normal_false_alarm_rate', 'normal_false_alarm_rate')]:
            require(np.isclose(config['models'][model][saved], computed[column], rtol=0, atol=1e-12),
                    'Frozen validation result mismatch: ' + column)
    return frame, {'split': split, 'model': model, **computed}


def calibration_signature(root, rule, contract, shared_reads):
    """Source identities from completed records; no reopening large raw/NPZ arrays."""
    signature = rule.input_signature(root, shared_reads)
    for name in ('splits/label_mapping.json', 'splits/train_observable.csv', 'splits/val_observable.csv'):
        signature[rule.DATASET_REL+'/'+name] = contract['input_sha256'][name]
    name = 'val_window_metadata.csv'
    signature[rule.OUTPUT_REL+'/tst_embeddings_finetuned_main/'+name] = contract['step9_artifacts'][name]
    for name in ('val_tst_sequence_metadata_L10.csv', 'val_tst_sequences_L10.npz'):
        signature[rule.OUTPUT_REL+'/tst_sequences_finetuned_main/'+name] = contract['step10_artifacts'][name]
    return signature


def load_metadata(root, split, rule, contract, reads):
    """Authenticate the full Step 10 population without loading embeddings or raw scores."""
    name = f'{split}_tst_sequence_metadata_L10.csv'
    frame = rule.verified_read(root/rule.OUTPUT_REL/'tst_sequences_finetuned_main'/name,
                               reads, contract['step10_artifacts'].get(name))
    rule.require_columns(frame, META[:-2], f'{split} current Step 10 metadata')
    int_columns = ['sequence_id', 'sequence_length', 'observable_window_seconds', 'temporal_context_seconds',
                   'sequence_start_row', 'sequence_end_row', 'sequence_start_step', 'sequence_end_step',
                   'sequence_start_window_id', 'sequence_end_window_id', 'label_id',
                   'sequence_num_label_changes', 'sequence_has_label_transition']
    for column in int_columns:
        frame[column] = rule.integers(frame[column], column)
    require(np.array_equal(frame.sequence_id, np.arange(len(frame))), 'Incomplete/unsorted Step 10 sequence IDs')
    common = contract['common_endpoint_report']['splits'][split]
    require(len(frame) == contract['counts'][split]['sequences_stride1']['10'] == common['common_endpoint_count'],
            'Step 10 common population count mismatch')
    y = frame.label_id.to_numpy(dtype=np.int64)
    require(set(y) == set(range(6)) and np.array_equal(frame.label, np.array(LABELS)[y])
            and np.array_equal(frame.start_label, frame.label), 'Step 10 label mapping/transition mismatch')
    for column, expected in [('source_split', split), ('sequence_length', 10), ('observable_window_seconds', 1),
                             ('temporal_context_seconds', 10), ('sequence_num_label_changes', 0),
                             ('sequence_has_label_transition', 0)]:
        require(frame[column].eq(expected).all(), 'Step 10 unexpected ' + column)
    require((frame.sequence_end_step-frame.sequence_start_step).eq(9).all()
            and (frame.sequence_end_window_id-frame.sequence_start_window_id).eq(9).all()
            and (frame.sequence_end_row >= frame.sequence_start_row).all(), 'Invalid L10 span')
    start = rule.parse_times(frame.sequence_start_timestamp)
    end = rule.parse_times(frame.sequence_end_timestamp)
    require((end-start).dt.total_seconds().eq(9).all(), 'Invalid L10 timestamp span')
    ends = frame.sequence_end_row.to_numpy(dtype=np.int64)
    require(len(np.unique(ends)) == len(ends)
            and sha(np.sort(ends).astype('<i8').tobytes()) == common['sorted_end_row_indices_sha256'],
            'Step 10 common endpoint identity mismatch')
    require(not frame.duplicated(['episode_id', 'sequence_end_step']).any()
            and frame.groupby('episode_id').label_id.nunique().eq(1).all(), 'Duplicate endpoint or mixed-label episode')
    return frame


def legacy_read(path, records, expected=None, kind='json'):
    """Read small current manifests/metadata, preserving portable hash records."""
    raw = Path(path).read_bytes()
    record = {'path': str(Path(path).resolve()), 'sha256': sha(raw), 'size_bytes': len(raw)}
    if expected is not None:
        require(record['sha256'] == expected, 'Current artifact hash mismatch: ' + str(path))
    records.append(record)
    if kind == 'json':
        return json.loads(raw)
    return pd.read_csv(io.BytesIO(raw), encoding='utf-8-sig', dtype=str, keep_default_na=False)


def legacy_verified_read(path, records, expected, kind='csv'):
    require(isinstance(expected, str) and len(expected) == 64,
            'Missing completed artifact hash: ' + str(path))
    return legacy_read(path, records, expected, kind)


def legacy_current_contract(root, records, fields):
    """Authenticate current Steps 7-10 without calling a v3-only Step 16 API."""
    build = legacy_read(root/DATA_REL/'build_manifest.json', records)
    build_sha = records[-1]['sha256']
    runs, digests = {}, {'step7': build_sha}
    for key, folder in [('step8', PIPELINE_REL+'/distilbert_window_baseline'),
                        ('step9', EMB_REL), ('step10', SEQ_REL)]:
        runs[key] = legacy_read(root/folder/'run_manifest.json', records)
        digests[key] = records[-1]['sha256']
    require(build.get('status') == 'complete' and bool(build.get('dataset_id'))
            and build.get('model_input_fields') == list(fields)
            and build.get('policy', {}).get('nominal_substrate') == 'shared_nominal_v2',
            'Step 7 is not the completed shared_nominal_v2 dataset')
    for key, run in runs.items():
        require(run.get('status') == 'COMPLETE' and bool(run.get('run_id'))
                and run.get('dataset_id') == build['dataset_id'], 'Incomplete/mixed ' + key)
    r8, r9, r10 = (runs[k] for k in ('step8', 'step9', 'step10'))
    source9, source10 = r9.get('source_contract', {}), r10.get('source', {})
    inputs = r8.get('input_sha256', {})
    required = ['splits/label_mapping.json', 'splits/episode_split_manifest.csv']
    required += ['splits/'+s+'_observable.csv' for s in ('train', 'val', 'test')]
    require(all(isinstance(inputs.get(n), str) and len(inputs[n]) == 64
                and inputs[n] == build.get('artifacts', {}).get(n) for n in required),
            'Step 7/8 split and label-map hashes differ')
    require(r8.get('step7_build_manifest_sha256') == build_sha
            and source9.get('step7_build_manifest_sha256') == build_sha
            and source9.get('step7_build') == build
            and source9.get('step8_manifest_sha256') == digests['step8']
            and source9.get('input_sha256') == inputs
            and source9.get('dataset_id') == build['dataset_id']
            and source9.get('step8_run_id') == r9.get('step8_run_id') == r8['run_id'],
            'Step 7/8/9 provenance differs')
    names9 = ['embedding_extraction_report.json']
    for split in ('train', 'val', 'test'):
        names9 += [split+'_window_embeddings.npz', split+'_window_metadata.csv']
    hashes9 = {n: r9.get('artifacts', {}).get(n) for n in names9}
    require(all(isinstance(v, str) and len(v) == 64 for v in hashes9.values())
            and source10.get('artifacts') == hashes9
            and source10.get('step9_manifest_sha256') == digests['step9']
            and source10.get('dataset_id') == build['dataset_id']
            and source10.get('step8_run_id') == r10.get('step8_run_id') == r8['run_id']
            and source10.get('step9_run_id') == r10.get('step9_run_id') == r9['run_id'],
            'Step 8/9/10 provenance differs')
    index = legacy_verified_read(root/SEQ_REL/'sequence_build_report_index.json', records,
                r10.get('artifacts', {}).get('sequence_build_report_index.json'), 'json')
    report = legacy_verified_read(root/SEQ_REL/'sequence_build_report_L10.json', records,
                r10.get('artifacts', {}).get('sequence_build_report_L10.json'), 'json')
    for record in (index, report):
        require(record.get('dataset_id') == build['dataset_id']
                and record.get('step8_run_id') == r8['run_id']
                and record.get('step9_run_id') == r9['run_id']
                and record.get('run_id') == r10['run_id']
                and record.get('source_artifact_sha256') == hashes9,
                'Step 10 report provenance differs')
    common = index.get('common_endpoint_report', {})
    require(index.get('status') == 'COMPLETE' and common.get('reference_length') == 10
            and r10.get('protocol', {}).get('stride') == 1 and report.get('sequence_length') == 10,
            'Step 10 must contain complete stride-1 L10 endpoints')
    for split in ('train', 'val', 'test'):
        count = build.get('counts', {}).get(split, {})
        detail = report.get('splits', {}).get(split, {})
        n = count.get('sequences_stride1', {}).get('10')
        require(all(type(v) is int and v > 0 for v in (count.get('rows'), count.get('episodes'), n))
                and detail.get('input_windows') == count['rows']
                and detail.get('num_episodes') == count['episodes']
                and detail.get('num_sequences') == n
                and common.get('splits', {}).get(split, {}).get('common_endpoint_count') == n,
                'Step 7/10 population differs: ' + split)
        name = split+'_tst_sequence_observable_effects_L10.csv'
        require(isinstance(r10.get('artifacts', {}).get(name), str)
                and detail.get('sequence_observable_effects_sha256') == r10['artifacts'][name],
                'Step 10 observable-effect record missing/different: ' + split)
    return {'dataset_id': build['dataset_id'], 'step8_run_id': r8['run_id'],
            'step9_run_id': r9['run_id'], 'step10_run_id': r10['run_id'],
            'manifest_sha256': digests, 'input_sha256': inputs,
            'step9_artifacts': hashes9, 'step10_artifacts': r10['artifacts'],
            'counts': build['counts'], 'common_endpoint_report': common}


def legacy_integers(values, description):
    a = pd.to_numeric(np.asarray(values), errors='raise')
    require(np.isfinite(a).all() and (a == np.floor(a)).all()
            and ((a >= 0) & (a < 2**53)).all(), 'Invalid integers: ' + description)
    return np.asarray(a, dtype=np.int64)


def legacy_require_columns(frame, columns, description):
    require(set(columns) <= set(frame), description + ': missing metadata')
    require(not frame[columns].isna().any().any()
            and all(frame[c].astype(str).str.strip().ne('').all() for c in columns),
            description + ': empty metadata')


def legacy_effects_for_context(effects, length):
    require(length in (1, 10), 'Invalid rule context length')
    result = effects[EFFECT_COLUMNS].copy()
    if length == 1:
        result[EFFECT_COLUMNS[1]] = 0
        result[EFFECT_COLUMNS[2]] = result[EFFECT_COLUMNS[0]]
        result[EFFECT_COLUMNS[3]] = np.where(result[EFFECT_COLUMNS[0]].eq(1),
                                            'current_changed', 'no_observable_change')
    return result


def legacy_load_effects(root, split, meta, records, contract):
    name = split+'_tst_sequence_observable_effects_L10.csv'
    effects = legacy_verified_read(root/SEQ_REL/name, records, contract['step10_artifacts'].get(name))
    require(list(effects) == list(meta) + EFFECT_COLUMNS, 'Unexpected Step 10 diagnostic schema')
    compare_columns(effects, meta, list(meta), split + ' observable-effect row identity')
    current, history, whole = [legacy_integers(effects[c], c) for c in EFFECT_COLUMNS[:3]]
    require(np.isin(current, [0, 1]).all() and (history < 10).all()
            and np.array_equal(whole, current+history), 'Invalid observable-effect counts')
    strata = np.where(current == 1, 'current_changed',
                      np.where(history > 0, 'history_only_changed', 'no_observable_change'))
    require(np.array_equal(effects[EFFECT_COLUMNS[3]], strata), 'Incorrect observable-effect strata')
    result = effects[EFFECT_COLUMNS].copy()
    for c, values in zip(EFFECT_COLUMNS[:3], (current, history, whole)):
        result[c] = values
    return result


def load_legacy_source(root, path, rule):
    """Compatibility for the completed v1 format; no rule upgrade or recalibration."""
    records = [file_record(rule.__file__)]
    current = legacy_current_contract(root, records, rule.FIELDS)
    legacy_contract = {'dataset_version': 'shared_nominal_v2',
                       **{k: current[k] for k in PROVENANCE[:4]},
                       'expected_counts': {s: [c['rows'], c['episodes'], c['sequences_stride1']['10']]
                                           for s, c in current['counts'].items()}}
    path = resolve_source(root, path, SimpleNamespace(RESULT_REL=RULE_REL), legacy_contract)
    manifest, manifest_sha, payload, source_records = source_payload(path)
    records.extend(source_records)
    config = json.loads(payload['frozen_rule_config.json'])
    p16, spec = (json.loads(payload[n]) for n in ('protocol.json', 'rule_spec.json'))
    require(manifest.get('dataset_contract') == p16.get('dataset_contract') == legacy_contract
            and manifest.get('calibrate_only') is False,
            'Legacy Step 16 source differs from the current shared_nominal_v2 pipeline')
    require(config.get('protocol_version') == p16.get('version') == spec.get('version') == LEGACY_RULE_VERSION
            and p16.get('available_observable_fields') == spec.get('available_fields') == list(rule.FIELDS)
            and tuple(spec.get('canonical_labels', ())) == LABELS
            and p16.get('metadata_feature_use') is False, 'Legacy Step 16 protocol/schema mismatch')
    config_sha = sha(payload['frozen_rule_config.json'])
    require(config_sha == manifest.get('frozen_config_sha256')
            and object_digest(p16) == config.get('protocol_sha256') == manifest.get('protocol_sha256')
            and object_digest(spec) == config.get('rule_spec_sha256') == p16.get('rule_spec_sha256'),
            'Legacy Step 16 configuration/protocol/specification digest mismatch')
    require(manifest.get('script_sha256') == config.get('script_sha256') == records[0]['sha256'],
            'Step 16 code differs from the completed v1 run. Use --step16-code PATH to the '
            'original producer code; do not recalibrate or replace the saved results.')
    require(config.get('label_to_id') == dict(zip(LABELS, range(6)))
            and set(config.get('models', {})) == set(MODELS), 'Legacy label/model mapping mismatch')
    for model in MODELS:
        thresholds = config['models'][model].get('thresholds', [])
        require(len(thresholds) == 5 and all(type(t) in (int, float) and np.isfinite(t)
                and t > 0 and t in spec.get('threshold_grid', []) for t in thresholds),
                'Invalid frozen legacy thresholds: ' + model)
    require(config.get('test_inputs_read_before_freeze') is False
            and manifest.get('test_opened_after_freeze') is True, 'Missing legacy freeze-before-test record')
    frozen = datetime.fromisoformat(config['frozen_at_utc'])
    require(frozen.tzinfo is not None and manifest.get('frozen_at_utc') == config['frozen_at_utc'],
            'Invalid legacy freeze timestamp')
    original = manifest.get('inputs', [])
    require(original and json.loads(payload['input_audit.json']) == original, 'Legacy input audits disagree')
    expected_inputs = {DATA_REL+'/build_manifest.json': current['manifest_sha256']['step7'],
        DATA_REL+'/splits/label_mapping.json': current['input_sha256']['splits/label_mapping.json'],
        SEQ_REL+'/sequence_build_report_index.json': current['step10_artifacts']['sequence_build_report_index.json']}
    for split in ('train', 'val', 'test'):
        name = 'splits/'+split+'_observable.csv'
        expected_inputs[DATA_REL+'/'+name] = current['input_sha256'][name]
    for split in ('val', 'test'):
        name = split+'_window_metadata.csv'
        expected_inputs[EMB_REL+'/'+name] = current['step9_artifacts'][name]
        for suffix in ('sequence_metadata_L10.csv', 'sequences_L10.npz'):
            name = split+'_tst_'+suffix
            expected_inputs[SEQ_REL+'/'+name] = current['step10_artifacts'][name]
    require(len(original) == len(expected_inputs), 'Unexpected/duplicate legacy input inventory')
    for relative, digest in expected_inputs.items():
        matches = [r for r in original if r.get('path', '').replace('\\', '/').endswith('/'+relative)]
        require(len(matches) == 1 and matches[0].get('sha256') == digest,
                'Legacy recorded input differs: ' + relative)
        live = file_record(root/relative)
        require(live['sha256'] == digest and live['size_bytes'] == matches[0].get('size_bytes'),
                'Current input differs from completed Step 16: ' + relative)
        records.append(live)
        if Path(relative).name.startswith('test_'):
            require(datetime.fromisoformat(matches[0]['read_at_utc']) > frozen,
                    'Legacy test input was opened before threshold freeze')
    calibration = {k: v for k, v in expected_inputs.items() if not Path(k).name.startswith('test_')}
    require(config.get('calibration_input_sha256') == calibration, 'Legacy calibration input signature differs')
    mapping = legacy_verified_read(root/DATA_REL/'splits/label_mapping.json', records,
                                  expected_inputs[DATA_REL+'/splits/label_mapping.json'], 'json')
    require(mapping.get('label_to_id', mapping) == dict(zip(LABELS, range(6))), 'Current label mapping differs')
    # The v1 manifest has no digest-based run_id. Retain its recorded run-folder
    # identity and bind it to the manifest/config hashes in this Step 50 audit.
    run_id = manifest.get('run_id') or str(manifest.get('output_dir', '')).replace('\\', '/').rsplit('/', 1)[-1]
    require(isinstance(run_id, str) and safe_name(run_id) and bool(run_id), 'Legacy run identity missing')
    expected = {k: current[k] for k in PROVENANCE[:4]}
    expected.update(step16_run_id=run_id, rule_config_sha256=config_sha,
                    rule_spec_sha256=config['rule_spec_sha256'])
    api = SimpleNamespace(OUTPUT_REL=PIPELINE_REL, verified_read=legacy_verified_read,
        require_columns=legacy_require_columns, integers=legacy_integers,
        parse_times=lambda values: pd.to_datetime(values, format='mixed', errors='raise', utc=True),
        effects_for_context=legacy_effects_for_context, classify=rule.classify, metrics=rule.metrics)
    summary = csv_bytes(payload['results_summary.csv'])
    require(len(summary) == 4, 'Expected both legacy Rule models on validation and test')
    data_audit = json.loads(payload['data_audit.json'])
    frames, checks, populations = [], [], {}
    for split in ('val', 'test'):
        meta = load_metadata(root, split, api, current, records)
        effects = legacy_load_effects(root, split, meta, records, current)
        populations[split] = set(meta.episode_id)
        require(len(populations[split]) == current['counts'][split]['episodes'], 'Legacy episode count differs')
        for key, value in [('sequences', len(meta)), ('episodes', len(populations[split])), ('alignment_passed', True)]:
            if key in data_audit.get(split, {}):
                require(data_audit[split][key] == value, 'Legacy population audit differs: ' + split+'/'+key)
        for model in MODELS:
            frame, check = recheck_predictions(csv_bytes(payload[f'{split}_predictions_{model}.csv']),
                meta, effects, model, split, config, expected, summary, api, legacy=True)
            checks.append(check)
            if split == 'test': frames.append(frame)
    require(populations['val'].isdisjoint(populations['test']), 'Validation/test episodes overlap')
    frames = align_pair(*frames)
    inventory, _ = episode_inventory(frames[0])
    audit = {'source_path': str(path), 'source_manifest_sha256': manifest_sha,
        'source_archive_sha256': source_records[0]['sha256'] if path.is_file() else None,
        'source_outputs': manifest['outputs'], 'verified_output_files': len(payload),
        'source_contract': current, 'step16_run_id': run_id,
        'configuration_sha256': config_sha, 'step16_script_sha256': records[0]['sha256'],
        'source_protocol_version': LEGACY_RULE_VERSION, 'contract_adapter': 'dataset_contract_v1',
        'legacy_run_identity_kind': 'saved_run_id' if manifest.get('run_id') else 'recorded_output_directory_name',
        'legacy_derived_columns': 'Absent provenance and ID aliases are added after verification; '
                                  'effect diagnostics come from hash-verified current Step 10 sidecars.',
        'reconstructed_decision_mismatches': 0, 'multiclass_reconstruction': checks,
        'test_endpoints': len(frames[0]), 'test_episodes': len(inventory),
        'strata_episode_counts': [int((inventory.label_id == c).sum()) for c in range(6)],
        'test_used_for_threshold_selection_in_step50': False, 'score_generation_in_step50': False,
        'inputs': records}
    verify_sources(audit)
    return frames, config, audit


def load_source(path=None, *, root=ROOT, step16_code=None):
    """Use current Step 16 input checks and saved scores; never call its calibrator."""
    root = Path(root).resolve()
    rule = import_rule(root, step16_code)
    if rule.PROTOCOL_VERSION == LEGACY_RULE_VERSION:
        return load_legacy_source(root, path, rule)
    records = [file_record(rule.__file__)]
    current_reads = []
    contract = rule.load_input_contract(root, current_reads)
    path = resolve_source(root, path, rule, contract)
    manifest, manifest_sha, payload, source_records = source_payload(path)
    records.extend(source_records)
    config = json.loads(payload['frozen_rule_config.json'])
    p16 = json.loads(payload['protocol.json'])
    spec16 = json.loads(payload['rule_spec.json'])
    require(manifest.get('source_contract') == config.get('source_contract') == contract,
            'Step 16 source differs from the current shared_nominal_v2 pipeline')
    require(manifest.get('script_sha256') == config.get('script_sha256') == records[0]['sha256'],
            'Step 16 code/script differs from the completed run; use the matching revised code')
    config_sha = sha(payload['frozen_rule_config.json'])
    require(config_sha == manifest.get('frozen_config_sha256'), 'Configuration digest mismatch')
    require(object_digest(p16) == config.get('protocol_sha256') == manifest.get('protocol_sha256')
            and object_digest(spec16) == config.get('rule_spec_sha256') == p16.get('rule_spec_sha256')
            and spec16 == rule.rule_specification(), 'Current rule/protocol specification mismatch')
    require(config.get('protocol_version') == p16.get('version') == rule.PROTOCOL_VERSION
            and p16.get('source_contract_sha256') == object_digest(contract)
            and p16.get('available_observable_fields') == list(rule.FIELDS)
            and p16.get('metadata_feature_use') is False, 'Step 16 protocol mismatch')
    require(config.get('label_to_id') == dict(zip(LABELS, range(6)))
            and set(config.get('models', {})) == set(MODELS), 'Label/model mapping mismatch')
    for model in MODELS:
        thresholds = config['models'][model].get('thresholds', [])
        require(len(thresholds) == 5 and all(type(t) in (int, float) and t in rule.GRID for t in thresholds),
                'Thresholds differ from the Step 16 validation grid')
    require(config.get('test_inputs_read_before_freeze') is False
            and manifest.get('test_opened_after_freeze') is True,
            'Step 16 freeze-before-test record missing')
    frozen = datetime.fromisoformat(config['frozen_at_utc'])
    require(frozen.tzinfo is not None and manifest.get('frozen_at_utc') == config['frozen_at_utc'],
            'Invalid configuration freeze timestamp')
    original_reads = manifest.get('inputs', [])
    require(json.loads(payload['input_audit.json']) == original_reads, 'Step 16 input audits disagree')
    test_reads = [item for item in original_reads
                  if item['path'].replace('\\', '/').rsplit('/', 1)[-1].startswith('test_')]
    require(test_reads and all(datetime.fromisoformat(item['read_at_utc']) > frozen for item in test_reads),
            'Step 16 test input was recorded before configuration freeze')
    run_identity = {'source_contract': contract, 'protocol_sha256': object_digest(p16),
                    'script_sha256': records[0]['sha256'], 'fixed_config_sha256': config_sha,
                    'output_dir': manifest.get('output_dir')}
    require(object_digest(run_identity) == manifest['run_id'], 'Step 16 run identity mismatch')

    # Step 16 already audited raw observations/sequence arrays before saving these
    # scores. Here the completed manifests bind those inputs, while the only
    # live cohort payloads needed are exact metadata and diagnostic sidecars.
    require(calibration_signature(root, rule, contract, current_reads) == config.get('calibration_input_sha256'),
            'Frozen calibration differs from current train/validation artifact records')
    val_meta = load_metadata(root, 'val', rule, contract, current_reads)
    val_effects = rule.load_effects(root, 'val', val_meta, current_reads, contract)
    test_meta = load_metadata(root, 'test', rule, contract, current_reads)
    test_effects = rule.load_effects(root, 'test', test_meta, current_reads, contract)
    require(set(val_meta.episode_id).isdisjoint(test_meta.episode_id), 'Validation/test episodes overlap')
    original_by_path = {item['path'].replace('\\', '/'): item['sha256'] for item in original_reads}
    for item in current_reads:
        require(original_by_path.get(item['path'].replace('\\', '/')) == item['sha256'],
                'Step 16 did not use the same current input: ' + item['path'])
    expected = {key: contract[key] for key in rule.PROVENANCE_KEYS}
    expected.update(step16_run_id=manifest['run_id'], rule_config_sha256=config_sha,
                    rule_spec_sha256=config['rule_spec_sha256'])
    summary = csv_bytes(payload['results_summary.csv'])
    require(len(summary) == 4, 'Expected both Rule models on validation and test')
    data_audit = json.loads(payload['data_audit.json'])
    frames, rechecks = [], []
    for split, meta, effects in [('val', val_meta, val_effects), ('test', test_meta, test_effects)]:
        for model in MODELS:
            frame, check = recheck_predictions(csv_bytes(payload[f'{split}_predictions_{model}.csv']),
                                              meta, effects, model, split, config, expected, summary, rule)
            rechecks.append(check)
            if split == 'test': frames.append(frame)
        require(data_audit[split].get('alignment_passed') is True
                and data_audit[split].get('sequences') == len(meta)
                and data_audit[split].get('episodes') == contract['counts'][split]['episodes'],
                'Step 16 population audit mismatch: ' + split)
    frames = align_pair(*frames)
    inventory, _ = episode_inventory(frames[0])
    require(set(inventory.label_id) == set(range(6)), 'All six episode strata are required')
    records.extend(current_reads)
    rule.verify_inputs_unchanged(records)
    audit = {'source_path': str(path), 'source_manifest_sha256': manifest_sha,
             'source_archive_sha256': source_records[0]['sha256'] if path.is_file() else None,
             'source_outputs': manifest['outputs'], 'verified_output_files': len(payload),
             'source_contract': contract, 'step16_run_id': manifest['run_id'],
             'configuration_sha256': config_sha, 'step16_script_sha256': records[0]['sha256'],
             'reconstructed_decision_mismatches': 0, 'multiclass_reconstruction': rechecks,
             'test_endpoints': len(frames[0]), 'test_episodes': len(inventory),
             'strata_episode_counts': [int((inventory.label_id == c).sum()) for c in range(6)],
             'test_used_for_threshold_selection_in_step50': False,
             'score_generation_in_step50': False, 'inputs': records}
    return frames, config, audit


def episode_counts(y, binary, codes, n_episodes):
    """Return [episode, model, detector, count]; columns are COUNT_NAMES."""
    y = np.asarray(y); binary = np.asarray(binary); codes = np.asarray(codes)
    require(y.ndim == 1 and np.isin(y, np.arange(6)).all(), 'Invalid true labels')
    require(binary.shape == (2, len(y), 5) and np.isin(binary, [0, 1]).all(), 'Invalid binary decisions')
    require(codes.shape == y.shape and np.issubdtype(codes.dtype, np.integer) and
            np.all((codes >= 0) & (codes < n_episodes)), 'Invalid episode codes')
    binary = binary.astype(bool)
    counts = np.zeros((n_episodes, 2, 5, 6), dtype=np.int64)
    for model in range(2):
        for j in range(5):
            truth = y == j + 1; pred = binary[model, :, j]
            masks = [truth & pred, ~truth & pred, truth & ~pred, ~truth & ~pred,
                     (y == 0) & pred, y == 0]
            for k, mask in enumerate(masks):
                counts[:, model, j, k] = np.bincount(codes[mask], minlength=n_episodes)
    return counts


def divide(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    return np.divide(a, b, out=np.full(np.broadcast_shapes(a.shape, b.shape), np.nan), where=b != 0)


def metrics(counts):
    tp, fp, fn, tn, normal_fp, normal_n = np.moveaxis(np.asarray(counts, dtype=np.float64), -1, 0)
    return {'f1': divide(2*tp, 2*tp+fp+fn), 'precision': divide(tp, tp+fp),
            'recall': divide(tp, tp+fn), 'specificity': divide(tn, tn+fp),
            'rest_fpr': divide(fp, fp+tn), 'normal_fpr': divide(normal_fp, normal_n)}


def draw_weights(strata, batch_size, rng):
    strata = np.asarray(strata)
    weights = np.zeros((batch_size, len(strata)), dtype=np.int64)
    for c in np.unique(strata):
        ix = np.flatnonzero(strata == c)
        weights[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1/len(ix)), size=batch_size)
    return weights


def bootstrap(counts, strata, n_resamples=N_BOOTSTRAP, seed=BOOTSTRAP_SEED,
              batch_size=BATCH_SIZE, progress=True):
    require(counts.shape == (len(strata), 2, 5, 6), 'Invalid episode counts shape')
    require(set(np.unique(strata)) == set(range(6)), 'All original six strata are required')
    require(n_resamples > 0 and batch_size > 0, 'Invalid bootstrap size')
    rng = np.random.Generator(np.random.PCG64(seed))
    flat = counts.reshape(len(strata), -1).astype(np.float64)
    dist = {key: np.empty((n_resamples, 2, 5), dtype=np.float64) for key in metrics(counts.sum(0))}
    digest = hashlib.sha256()
    start = time.perf_counter()
    for lo in range(0, n_resamples, batch_size):
        hi = min(lo+batch_size, n_resamples)
        weights = draw_weights(strata, hi-lo, rng)
        digest.update(weights.astype('<i8').tobytes())
        total = (weights.astype(np.float64) @ flat).reshape(hi-lo, 2, 5, 6)
        for key, value in metrics(total).items():
            dist[key][lo:hi] = value
        if progress and (lo == 0 or hi % 4000 == 0 or hi == n_resamples):
            print(f'Bootstrap {hi:,}/{n_resamples:,} | {time.perf_counter()-start:.1f}s', flush=True)
    return dist, digest.hexdigest()


def interval(values, lower=.025, upper=.975):
    # Do not silently condition an interval on bootstrap draws with defined precision.
    if not np.isfinite(values).all():
        return float('nan'), float('nan')
    return tuple(float(v) for v in np.quantile(values, [lower, upper], method='linear'))


def create_output(path):
    Path(path).mkdir(parents=True, exist_ok=False)


def write_json(path, obj):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_csv(path, records):
    pd.DataFrame(records).to_csv(path, index=False, encoding='utf-8-sig', float_format='%.17g')


def protocol(audit):
    return {
        'version': PROTOCOL_VERSION,
        'design_status': 'retrospective revision analysis; Step16 multiclass results previously observed',
        'review_mapping': ['R2-5 partial: scenario-wise one-vs-rest conventional heuristic evaluation',
                           'R2-1 supplemental: domain-informed rule diagnostics'],
        'labels': LABELS, 'models': MODELS, 'population': 'shared_nominal_v2 A test, complete common L10 endpoints, terminal label',
        'n_endpoints': audit['test_endpoints'], 'n_episodes': audit['test_episodes'],
        'dataset_id': audit['source_contract']['dataset_id'], 'step16_run_id': audit['step16_run_id'],
        'configuration_sha256': audit['configuration_sha256'],
        'observable_effects': 'diagnostics only; no endpoint filtering, feature use, relabeling or threshold selection',
        'effect_strata': 'model-specific: L1 current window only; L10 current and preceding nine windows',
        'decision': 'each pre-fusion evidence_j / original threshold_j >= 1; multiple alarms allowed',
        'negative_class': 'NORMAL plus the other four attacks',
        'threshold_selection': 'reuse Step16 validation multiclass calibration; no binary optimization',
        'bootstrap_replicates': N_BOOTSTRAP, 'bootstrap_seed': BOOTSTRAP_SEED, 'batch_size': BATCH_SIZE,
        'rng': 'numpy.random.PCG64', 'resampling': 'whole episodes within six true-label strata, paired across all detectors/models',
        'point_weighting': 'pooled endpoint counts, not episode-average F1',
        'interval': 'percentile linear quantiles; 95% marginal for metrics and deltas',
        'multiplicity': 'five L10-minus-L1 F1 contrasts; 99% Bonferroni marginal intervals, nominal 95% family target under bootstrap approximation',
        'undefined_precision': 'blank CSV; precision CI blank if any bootstrap draw is undefined',
        'limitations': [
            'generator-informed heuristic, not a published detector reproduction or exhaustive rules baseline',
            'not external, SIL, operational, onset-detection or simultaneous-multiattack validation',
            'L10 changes aggregation, recurrence and validation thresholds; not a pure temporal ablation',
            'conditional on original rules/validation selection; no generator or threshold-selection uncertainty',
            'episodes assumed approximately independent/exchangeable within true class',
            'family adjustment covers these five contrasts only, not the entire revision',
            'zero observed normal false alarms yield a degenerate empirical percentile CI, not a population risk upper bound',
            'does not establish TST superiority or complete R2-5 by itself',
            'no_observable_change means no change in the recorded 22-field counterfactual projection; not proof of zero information or an oracle ceiling',
            'effect-stratum summaries are descriptive; L1 and L10 strata are not identical paired subpopulations',
        ],
    }


def save_analysis(out, frames, config, counts, inventory, codes, binary, fused, dist, weight_digest):
    totals = counts.sum(0); point = metrics(totals)
    y = frames[0].y_true.to_numpy(dtype=np.int64)
    metric_rows, comparisons, fused_rows, class_rows, overlap_rows = [], [], [], [], []
    for m, model in enumerate(MODELS):
        predictions = frames[m].copy()  # Preserve verified source IDs and model-specific audit flags.
        predictions['model'] = model
        predictions['original_fused_prediction'] = fused[m]
        predictions['independent_alarm_count'] = binary[m].sum(axis=1)
        for j, attack in enumerate(LABELS[1:]):
            predictions['evidence_' + attack] = frames[m][SCORE_COLS[j]]
            predictions['binary_' + attack] = binary[m, :, j].astype(np.int8)
            row = {'model': model, 'attack': attack, 'threshold': config['models'][model]['thresholds'][j],
                   **{name: int(totals[m, j, k]) for k, name in enumerate(COUNT_NAMES)}}
            for key in point:
                ci = interval(dist[key][:, m, j])
                row.update({key: float(point[key][m, j]), key+'_ci95_low': ci[0], key+'_ci95_high': ci[1],
                            key+'_undefined_bootstrap_draws': int((~np.isfinite(dist[key][:, m, j])).sum())})
            metric_rows.append(row)
            true = y == j+1; is_fused = fused[m] == j+1
            tp = np.count_nonzero(true & is_fused); fp = np.count_nonzero(~true & is_fused)
            fn = np.count_nonzero(true & ~is_fused)
            lost_tp = np.count_nonzero(true & binary[m, :, j] & ~is_fused)
            removed_fp = np.count_nonzero(~true & binary[m, :, j] & ~is_fused)
            require(tp + lost_tp == totals[m, j, 0] and fp + removed_fp == totals[m, j, 1],
                    'Independent versus fused accounting mismatch')
            fused_rows.append({'model': model, 'attack': attack,
                'independent_binary_f1': float(point['f1'][m, j]), 'fused_class_f1': 2*tp/(2*tp+fp+fn),
                'independent_tp': int(totals[m, j, 0]), 'fused_tp': int(tp),
                'true_attack_alarms_lost_in_fusion': int(lost_tp),
                'independent_fp': int(totals[m, j, 1]), 'fused_fp': int(fp),
                'false_alarms_removed_in_fusion': int(removed_fp)})
            for c, label in enumerate(LABELS):
                mask = y == c
                class_rows.append({'model': model, 'detector': attack, 'true_class': label,
                                   'role': 'positive' if c == j+1 else 'negative',
                                   'n_endpoints': int(mask.sum()), 'n_alarms': int(binary[m, mask, j].sum()),
                                   'alarm_rate': float(binary[m, mask, j].mean())})
        for c, label in enumerate(LABELS):
            mask = y == c; n_alarm = binary[m, mask].sum(1)
            overlap_rows.append({'model': model, 'true_class': label, 'n_endpoints': int(mask.sum()),
                                 'zero_alarms': int((n_alarm == 0).sum()),
                                 'one_alarm': int((n_alarm == 1).sum()),
                                 'multiple_alarms': int((n_alarm > 1).sum()),
                                 'any_alarm_rate': float((n_alarm > 0).mean()),
                                 'multiple_alarm_rate': float((n_alarm > 1).mean())})
        predictions.to_csv(out / ('test_binary_predictions_' + model + '.csv.gz'), index=False,
                           encoding='utf-8-sig', float_format='%.17g', compression={'method': 'gzip', 'mtime': 0})
    delta = dist['f1'][:, 1] - dist['f1'][:, 0]
    for j, attack in enumerate(LABELS[1:]):
        ci = interval(delta[:, j]); adjusted = interval(delta[:, j], .005, .995)
        comparisons.append({'attack': attack, 'contrast': 'Rule_L10_minus_Rule_L1',
                            'binary_f1_delta': float(point['f1'][1, j]-point['f1'][0, j]),
                            'ci95_low': ci[0], 'ci95_high': ci[1],
                            'bonferroni_ci99_low': adjusted[0], 'bonferroni_ci99_high': adjusted[1],
                            'family_size': 5, 'adjusted_direction': 'L10_higher' if adjusted[0] > 0 else
                            ('L1_higher' if adjusted[1] < 0 else 'zero_included_not_equivalence')})
    write_csv(out/'binary_metrics.csv', metric_rows)
    write_csv(out/'paired_binary_f1_comparisons.csv', comparisons)
    write_csv(out/'independent_vs_fused_diagnostics.csv', fused_rows)
    write_csv(out/'detector_by_true_class.csv', class_rows)
    write_csv(out/'alarm_overlap_by_true_class.csv', overlap_rows)
    write_csv(out/'episode_inventory.csv', inventory)
    write_csv(out/'detector_by_observable_effect.csv', effect_breakdown(frames, binary))
    np.savez_compressed(out/'episode_sufficient_counts.npz', counts=counts,
                        episode_ids=inventory.episode_id.to_numpy(dtype=str),
                        strata=inventory.label_id.to_numpy(dtype=np.int64),
                        count_names=np.array(COUNT_NAMES), models=np.array(MODELS), detectors=np.array(LABELS[1:]))
    np.savez_compressed(out/'bootstrap_distribution.npz', **dist, f1_delta_L10_minus_L1=delta,
                        models=np.array(MODELS), detectors=np.array(LABELS[1:]))
    write_json(out/'bootstrap_audit.json', {'weight_matrix_sha256_little_endian_int64': weight_digest,
               'n_resamples': len(delta), 'seed': BOOTSTRAP_SEED, 'batch_size': BATCH_SIZE,
               'strata_episode_counts': [int((inventory.label_id == c).sum()) for c in range(6)],
               'all_models_and_detectors_use_same_draws': True})
    lines = ['Step 50: shared_nominal_v2 A 시나리오별 독립 Rule one-vs-rest 평가', '',
             f'{len(frames[0]):,} matched test endpoints / {len(inventory):,} episodes; existing Step 16 validation thresholds.',
             '독립 탐지: 각 결합 전 evidence/threshold >= 1. 다중분류 argmax 이진화와 구분.',
             '기존 multiclass 결과를 본 뒤 정한 retrospective analysis이며 임계값 재탐색 없음.', '',
             'Attack | L1 binary F1 | L10 binary F1 | delta | paired Bonferroni 99% CI']
    for j, name in enumerate(SHORT):
        row = comparisons[j]
        lines.append(f"{name} | {point['f1'][0,j]:.6f} | {point['f1'][1,j]:.6f} | "
                     f"{row['binary_f1_delta']:+.6f} | [{row['bonferroni_ci99_low']:+.6f}, {row['bonferroni_ci99_high']:+.6f}]")
    lines += ['', '정상-only FPR 및 rest FPR은 binary_metrics.csv에서 분리 보고.',
              '양성 예측이 없으면 precision은 미정의(빈 값); CI도 미정의 표본을 제거해 계산하지 않음.',
              '독립 탐지와 최종 결합은 TP/FP를 함께 바꾸므로 F1 차이로 결합 제거를 권고하지 않음.',
              'L10은 평균 evidence·replay recurrence·별도 validation 임계값을 함께 바꾼 구성 비교.',
              'R2-5 부분 대응 및 R2-1 보완. 발표된 탐지기 재현, 외부 검증, TST 우월성 증명은 아님.',
              'CI는 규칙/임계값 선택을 조건으로 한 클래스별 전체 episode bootstrap 근사.',
              '기존 여섯 클래스 macro-F1과 다섯 독립 binary F1은 같은 지표가 아니므로 대체 금지.', '']
    lines.insert(-1, '관측 정상 오탐 0의 percentile CI [0,0]은 경험적 퇴화 구간이며 모집단 위험의 상한이 아님.')
    (out/'analysis_summary_KO.txt').write_text('\n'.join(lines), encoding='utf-8')


def effect_breakdown(frames, binary):
    """Descriptive post-hoc diagnostics only; primary counts retain every endpoint."""
    rows = []
    for m, model in enumerate(MODELS):
        frame = frames[m]
        y = frame.y_true.to_numpy(dtype=np.int64)
        for stratum in ('current_changed', 'history_only_changed', 'no_observable_change'):
            mask = frame.audit_effect_stratum.eq(stratum).to_numpy()
            for j, attack in enumerate(LABELS[1:]):
                truth, alarm = y[mask] == j+1, binary[m, mask, j]
                normal = y[mask] == 0
                values = np.array([np.count_nonzero(truth & alarm), np.count_nonzero(~truth & alarm),
                                   np.count_nonzero(truth & ~alarm), np.count_nonzero(~truth & ~alarm),
                                   np.count_nonzero(normal & alarm), np.count_nonzero(normal)])
                rows.append({'model': model, 'rule_context_length': int(model[6:]),
                             'audit_effect_stratum': stratum, 'attack': attack,
                             'n_endpoints': int(mask.sum()), 'n_episodes': frame.loc[mask, 'episode_id'].nunique(),
                             'positive_n': int(truth.sum()), 'negative_n': int((~truth).sum()),
                             **dict(zip(COUNT_NAMES, map(int, values))),
                             **{key: float(value) for key, value in metrics(values).items()}})
    return rows


def verify_sources(audit):
    for item in audit['inputs']:
        path = Path(item['path'])
        require(path.is_file() and path.stat().st_size == item['size_bytes']
                and sha_file(path) == item['sha256'],
                'Source changed during Step 50: ' + str(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root', '--root', type=Path, default=ROOT,
                        help='Project containing the revised scripts, dataset and outputs')
    parser.add_argument('--step16-code', type=Path, help='Revised Step 16 code, if its filename is different')
    parser.add_argument('--step16-source', type=Path,
                        help='Completed Step 16 directory or flat ZIP; default: unique current matching run')
    parser.add_argument('--output', type=Path, help='New result directory; existing paths are rejected')
    parser.add_argument('--audit-only', action='store_true',
                        help='Read-only source/decision checks; no result folder or bootstrap')
    args = parser.parse_args(argv)
    root = args.project_root.expanduser().resolve()
    require(root.is_dir(), 'Project root not found: ' + str(root))
    require(not (args.audit_only and args.output), '--output is not used with read-only --audit-only')
    requested_output = resolve_path(root, args.output) if args.output else None
    if requested_output is not None:
        require(not requested_output.exists(), 'Output already exists; choose a new directory: ' + str(requested_output))
    started = time.perf_counter()
    script_sha = sha(Path(__file__).read_bytes())
    print('Checking the current Step 16 source, matched endpoints and saved rule decisions ...', flush=True)
    frames, config, audit = load_source(args.step16_source, root=root, step16_code=args.step16_code)
    print(f"Source audit passed: {audit['test_endpoints']:,} test endpoints / "
          f"{audit['test_episodes']:,} episodes. Validation thresholds reused unchanged.", flush=True)
    if args.audit_only:
        require(script_sha == sha(Path(__file__).read_bytes()), 'Step 50 code changed during audit')
        verify_sources(audit)
        print('AUDIT_ONLY_COMPLETE: no output files written.', flush=True)
        return 0
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = requested_output or root/OUTPUT_REL/('run_'+stamp)
    create_output(out)
    specification = protocol(audit)
    manifest = {'status': 'RUNNING', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
                'script_sha256': script_sha, 'protocol_version': PROTOCOL_VERSION,
                'source_manifest_sha256': audit['source_manifest_sha256'],
                'source_contract': audit['source_contract'], 'step16_run_id': audit['step16_run_id'],
                'configuration_sha256': audit['configuration_sha256'], 'output_dir': str(out),
                'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__}
    manifest['run_id'] = object_digest({'script_sha256': script_sha, 'protocol': specification,
                                       'source_manifest_sha256': audit['source_manifest_sha256'], 'output_dir': str(out)})
    write_json(out/'run_manifest.json', manifest)
    try:
        write_json(out/'protocol.json', specification)
        write_json(out/'input_audit.json', audit)
        write_json(out/'source_rule_configuration.json', config)
        inventory, codes = episode_inventory(frames[0])
        binary, fused = [], []
        for frame, model in zip(frames, MODELS):
            b, f, _ = decisions(frame[SCORE_COLS].to_numpy(), config['models'][model]['thresholds'])
            binary.append(b); fused.append(f)
        binary = np.stack(binary); fused = np.stack(fused)
        counts = episode_counts(frames[0].y_true.to_numpy(), binary, codes, len(inventory))
        dist, weight_digest = bootstrap(counts, inventory.label_id.to_numpy())
        save_analysis(out, frames, config, counts, inventory, codes, binary, fused, dist, weight_digest)
        verify_sources(audit)
        require(script_sha == sha(Path(__file__).read_bytes()), 'Step 50 code changed during execution')
        outputs = [{'file': p.name, 'sha256': sha(p.read_bytes()), 'size_bytes': p.stat().st_size}
                   for p in sorted(out.iterdir()) if p.is_file() and p.name != 'run_manifest.json']
        manifest.update(status='COMPLETE', completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.perf_counter()-started,
                        protocol_sha256=sha((out/'protocol.json').read_bytes()), outputs=outputs)
        write_json(out/'run_manifest.json', manifest)
        print('COMPLETE: ' + str(out), flush=True)
    except BaseException as exc:
        manifest.update(status='FAILED', error=str(exc), error_type=type(exc).__name__,
                        elapsed_seconds=time.perf_counter()-started)
        write_json(out/'run_manifest.json', manifest)
        raise
    return 0


if __name__ == '__main__':
    sys.exit(main())
