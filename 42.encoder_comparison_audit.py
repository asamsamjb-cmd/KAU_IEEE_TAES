#!/usr/bin/env python3
"""TAES revision Step 42: encoder-input and comparison-readiness audit.

Run beside the numbered scripts:
    python "42.encoder_comparison_audit.py"

Requires COMPLETE revised Steps 7, 8 and 9. Steps 10/11/12 are inspected when
available; their absence does not prevent the input/tokenizer audit. No Step
16/17 run, old cohort counts, training, model inference, or download is needed.
Uses train/validation observables and the Step 7 episode inventory. Never opens
raw/serialized test windows, predictions, embedding arrays, or test-window/
sequence metadata or observable-effect sidecars. Test hashes and aggregate
counts are DECLARED upstream records, not independently verified test contents. Existing training is not
an untouched-test experiment simply because this audit does not read test data.

AUDIT_COMPLETE means this audit completed, not an encoder comparison result.
comparison_design.json is a DRAFT: no alternative encoders are trained here.
The scientific question is the 22-field representation plus temporal evidence
aggregation, with encoder capacity/pretraining controls, not TST superiority.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path, PureWindowsPath
import sys
import time
import traceback

DEPENDENCY_ERROR = None
try:
    import numpy as np
    import pandas as pd
except (ImportError, ValueError, OSError) as exc:
    np = pd = None
    DEPENDENCY_ERROR = str(exc)

FIELDS = (
    'observation_stage','path_type','direction','source_role','destination_role',
    'protocol_context','operational_phase','message_type','service_context',
    'command_intent','payload_or_parameter_context','sequence_freshness_status',
    'control_security_state','signal_quality_state','tracking_stability',
    'navigation_consistency','timing_consistency','decoder_status',
    'command_execution_result','retry_timeout_state','session_state',
    'state_transition_consistency',
)
METADATA = ('scenario_id','episode_id','episode_type','episode_step','window_id','timestamp','label')
LABELS = ('NORMAL','RF_LINK_JAMMING','GNSS_SPOOFING',
          'UNAUTHORIZED_TELECOMMAND_INJECTION','TELECOMMAND_REPLAY',
          'GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE')
LABEL_TO_ID = dict(zip(LABELS, range(6)))
LENGTHS = [1, 2, 3, 5, 10]
SPLITS = ('train', 'val', 'test')
SERIALIZER = 'observable22_fixed_order_is_v1'
PROTOCOL = 'shared_nominal_encoder_input_audit_v3'
DATA_REL = Path('dataset/shared_nominal_v2')
BASE_OUTPUT_REL = Path('outputs/shared_nominal_v2')
MODEL_REL = BASE_OUTPUT_REL / 'distilbert_window_baseline/trained_distilbert_model'
EMBEDDING_REL = BASE_OUTPUT_REL / 'tst_embeddings_finetuned_main'
SEQUENCE_REL = BASE_OUTPUT_REL / 'tst_sequences_finetuned_main'
OUTPUT_REL = BASE_OUTPUT_REL / '42_encoder_comparison_audit'
REFERENCE_HEAD_SEED = 20260610
AUDIT_SCHEMA = 'counterfactual_observable_change_v1'
AUDIT_FIELDS = ('audit_observable_effect_present', 'audit_observable_difference_count',
                'audit_persistent_effect_present', 'audit_attack_active')
AUDIT_FILES = {split:f'audit/{split}_window_observable_effects.csv' for split in SPLITS}
STEP11_SOURCE_KEYS = ('dataset_id', 'step8_run_id', 'step9_run_id', 'step10_run_id',
                     'input_sha256', 'step8_model_sha256', 'step9_artifacts', 'step10_artifacts',
                     'common_endpoint_report', 'observability_audit')
STEP11_IDENTITY_KEYS = ('source_contract', 'script_sha256', 'settings', 'selected_sequence_lengths',
                       'source_embedding_dir', 'source_sequence_dir', 'output_root',
                       'runtime', 'seed_policy', 'comparison_policy')
STEP11_LENGTH_TABLES = ('LENGTH_AWARE_REGULARIZATION', 'MISSING_AWARE_REGULARIZATION',
                       'ROBUST_KEEP_WINDOWS_BY_LENGTH', 'ROBUST_LAST_WINDOW_KEEP_PROB_BY_LENGTH',
                       'ROBUST_DIM_DROPOUT_PROB_BY_LENGTH', 'ROBUST_GAUSSIAN_NOISE_STD_BY_LENGTH')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def object_hash(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode('utf-8')).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def artifact_path(root, name):
    require(isinstance(name, str) and name and '\\' not in name, f'Invalid artifact path: {name}')
    rel = Path(name)
    require(not rel.is_absolute() and not PureWindowsPath(name).drive and '..' not in rel.parts,
            f'Unsafe artifact path: {name}')
    root = Path(root).resolve()
    path = (root / rel).resolve()
    require(path.is_relative_to(root), f'Artifact escapes its root: {name}')
    return path


def declared_hashes(record, names, description):
    inventory = record.get('artifacts', {})
    result = {name: inventory.get(name) for name in names}
    require(all(valid_digest(v) for v in result.values()), f'{description}: missing/invalid artifact hashes')
    return result


def guard_read_scope(path):
    """No incidental test/embedding/prediction reads, including hash-only reads."""
    name = Path(path).name.lower()
    require(not name.endswith(('.npz', '.npy')) and 'predictions' not in Path(path).parts
            and not name.startswith('test_'), f'Outside Step 42 read scope: {path}')


def snapshot_file(path, audit, expected_sha=None):
    path = Path(path).resolve()
    guard_read_scope(path)
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            f'File changed during hashing: {path}')
    require(expected_sha is None or digest == expected_sha, f'Input hash mismatch: {path}')
    for item in audit:
        if item['path'] == str(path):
            require(item['sha256'] == digest, f'Input changed during audit: {path}')
            return item
    item = {'path':str(path), 'sha256':digest, 'size_bytes':before.st_size, 'read_at_utc':utc_now()}
    audit.append(item)
    return item


def read_input(path, audit, kind='json', expected_sha=None):
    item = snapshot_file(path, audit, expected_sha)
    if kind == 'json':
        value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    elif kind == 'csv':
        value = pd.read_csv(path, encoding='utf-8-sig', dtype=str, keep_default_na=False)
    else:
        raise ValueError(f'Unsupported reader: {kind}')
    require(sha256(path) == item['sha256'], f'Input changed during read: {path}')
    return value


def recorded_digest(path, audit):
    return next(x['sha256'] for x in audit if x['path'] == str(Path(path).resolve()))


def integer_array(series, name):
    x = pd.to_numeric(series, errors='raise').to_numpy(dtype=np.float64)
    require(np.isfinite(x).all() and (x == np.floor(x)).all(), f'{name}: invalid integer values')
    require(((x >= 0) & (x <= np.iinfo(np.int64).max-2048)).all(), f'{name}: out-of-range integers')
    return x.astype(np.int64)


def project_path(recorded, root):
    """Map saved Windows/POSIX paths to the selected project, never by basename."""
    root = Path(root).resolve()
    raw = str(recorded)
    parts = PureWindowsPath(raw).parts if '\\' in raw else Path(raw).parts
    if Path(raw).is_absolute() or PureWindowsPath(raw).drive:
        positions = [i for i, part in enumerate(parts) if part in ('outputs', 'dataset')]
        require(len(positions) == 1, f'Cannot map recorded project path: {recorded}')
        parts = parts[positions[0]:]
    require(parts and parts[0] in ('outputs', 'dataset') and '..' not in parts,
            f'Unsafe or unknown project-relative path: {recorded}')
    path = root.joinpath(*parts).resolve()
    require(path.is_relative_to(root), f'Recorded path escapes project: {recorded}')
    return path


def add_finding(findings, code, message, severity='REVIEW'):
    findings.append({'severity':severity, 'code':code, 'message':message})


def load_build(root, audit):
    dataset = root / DATA_REL
    path = dataset / 'build_manifest.json'
    build = read_input(path, audit)
    require(build.get('status') == 'complete', 'Revised Step 7 is not complete')
    policy, sources = build.get('policy', {}), build.get('sources', [])
    require(build.get('dataset_id') == object_hash({'policy':policy, 'sources':sources}),
            'Step 7 dataset_id does not match its source/policy record')
    require(policy.get('version') == 'step7_episode_split_v2'
            and policy.get('nominal_substrate') == 'shared_nominal_v2'
            and policy.get('observability_audit_schema') == AUDIT_SCHEMA
            and policy.get('columns') == list(METADATA+FIELDS)
            and policy.get('label_order') == list(LABELS)
            and policy.get('window_seconds') == 1 and policy.get('sequence_lengths') == LENGTHS,
            'Step 7 schema/window/split protocol differs')
    require(build.get('model_input_fields') == list(FIELDS)
            and build.get('metadata_not_model_features') == list(METADATA), 'Step 7 feature boundary differs')
    require([s.get('label') for s in sources] == list(LABELS)
            and all(valid_digest(s.get('sha256')) and valid_digest(s.get('raw_audit', {}).get('sha256'))
                    for s in sources), 'Six ordered Step 7 learning/raw-audit source hashes required')
    spec = build.get('observability_audit', {})
    require(spec.get('schema_version') == AUDIT_SCHEMA and spec.get('split_files') == AUDIT_FILES
            and spec.get('columns') == list(METADATA+AUDIT_FIELDS) and bool(spec.get('scope')),
            'Step 7 observable-effect sidecar contract differs')
    declared_hashes(build, AUDIT_FILES.values(), 'Step 7 observable-effect sidecars')
    names = [f'splits/{s}_observable.csv' for s in SPLITS]
    names += ['splits/label_mapping.json', 'splits/episode_split_manifest.csv']
    inputs = declared_hashes(build, names, 'Step 7')
    mapping = read_input(dataset/'splits/label_mapping.json', audit,
                         expected_sha=inputs['splits/label_mapping.json'])
    require(mapping.get('label_order') == list(LABELS) and mapping.get('label_to_id') == LABEL_TO_ID
            and mapping.get('observable_fields') == list(FIELDS)
            and mapping.get('metadata_fields') == list(METADATA)
            and mapping.get('observable_window_seconds') == 1, 'Step 7 label/schema map differs')
    inventory = read_input(dataset/'splits/episode_split_manifest.csv', audit, 'csv',
                           inputs['splits/episode_split_manifest.csv'])
    needed = ['label','global_episode_id','original_episode_id','row_count','split',
              'min_episode_step','max_episode_step','min_window_id','max_window_id',
              'start_timestamp','end_timestamp','observable_window_seconds','episode_duration_seconds']
    require(set(needed) <= set(inventory), 'Incomplete Step 7 episode inventory')
    require(not inventory[needed].isna().any().any() and inventory[needed].ne('').all().all(),
            'Empty episode inventory cells')
    require(set(inventory.split) == set(SPLITS) and set(inventory.label) == set(LABELS)
            and inventory.global_episode_id.is_unique, 'Episode inventory identities/coverage invalid')
    require(inventory.global_episode_id.eq(inventory.label+'::'+inventory.original_episode_id).all(),
            'Episode inventory global IDs differ from Step 7 convention')
    collision = inventory.groupby('original_episode_id').label.nunique().gt(1).any()
    inventory['episode_id'] = inventory.global_episode_id if collision else inventory.original_episode_id
    require(inventory.episode_id.is_unique, 'Step 7 output episode identities are ambiguous')
    for col in ('row_count','min_episode_step','max_episode_step','min_window_id','max_window_id'):
        inventory[col] = integer_array(inventory[col], 'episode inventory '+col)
    require(inventory.row_count.gt(0).all(), 'Nonpositive episode size')
    require(inventory.max_episode_step.sub(inventory.min_episode_step).add(1).eq(inventory.row_count).all()
            and inventory.max_window_id.sub(inventory.min_window_id).add(1).eq(inventory.row_count).all(),
            'Episode inventory step/window range differs from row counts')
    require(pd.to_numeric(inventory.observable_window_seconds).eq(1).all()
            and pd.to_numeric(inventory.episode_duration_seconds).eq(inventory.row_count).all(),
            'Episode inventory time policy differs')
    elapsed = (pd.to_datetime(inventory.end_timestamp, utc=True) -
               pd.to_datetime(inventory.start_timestamp, utc=True)).dt.total_seconds()
    require(elapsed.eq(inventory.row_count-1).all(), 'Episode inventory timestamps differ from one-second windows')
    for split in SPLITS:
        inv = inventory[inventory.split == split]
        require(set(inv.label) == set(LABELS), f'{split}: missing class in episode inventory')
        actual = {'rows':int(inv.row_count.sum()), 'episodes':len(inv),
                  'sequences_stride1':{str(l):int((inv.row_count-l+1).clip(lower=0).sum()) for l in LENGTHS}}
        require(build.get('counts', {}).get(split) == actual, f'{split}: Step 7 count record inconsistent')
    expected = policy.get('expected_rows_per_class')
    require(type(expected) is int and expected > 0 and
            inventory.groupby('label').row_count.sum().eq(expected).all(), 'Class/source policy row totals differ')
    return build, inputs, inventory


def load_encoder_contract(root, model_dir, build, inputs, audit):
    root8, root9 = model_dir.parent, root/EMBEDDING_REL
    r8 = read_input(root8/'run_manifest.json', audit)
    r9 = read_input(root9/'run_manifest.json', audit)
    for name, run in [('Step 8', r8), ('Step 9', r9)]:
        require(run.get('status') == 'COMPLETE' and run.get('run_id'), f'{name} must be COMPLETE')
        require(run.get('dataset_id') == build['dataset_id'], f'{name}: different dataset_id')
    h7 = recorded_digest(root/DATA_REL/'build_manifest.json', audit)
    h8 = recorded_digest(root8/'run_manifest.json', audit)
    require(r8.get('input_sha256') == inputs and r8.get('step7_build_manifest_sha256') == h7,
            'Step 8 was trained on different Step 7 inputs')
    require(r8.get('step7_build') == build, 'Step 8 embedded Step 7 record differs')
    p8 = r8.get('protocol', {})
    require(p8.get('sample_limits') == [None, None, None]
            and p8.get('serializer_version') == SERIALIZER
            and p8.get('selection') == 'validation macro-F1', 'Step 8 main-training protocol differs')
    s9 = r9.get('source_contract', {})
    require(s9.get('dataset_id') == build['dataset_id'] and s9.get('input_sha256') == inputs
            and s9.get('step7_build_manifest_sha256') == h7
            and s9.get('step8_run_id') == r8['run_id'] and r9.get('step8_run_id') == r8['run_id']
            and s9.get('step8_manifest_sha256') == h8, 'Step 7 -> 8 -> 9 provenance mismatch')
    require(s9.get('step7_build') == build
            and project_path(s9.get('source_dataset_dir'), root) == root/DATA_REL,
            'Step 9 embedded Step 7 build or source dataset differs')
    observability = {'schema_version':AUDIT_SCHEMA, 'dataset_id':build['dataset_id'],
        'source_dataset_dir':s9['source_dataset_dir'], 'step7_build_manifest_sha256':h7,
        'split_files':AUDIT_FILES, 'artifacts':declared_hashes(build, AUDIT_FILES.values(), 'Step 7 audit'),
        'scope':build['observability_audit']['scope']}
    model_hashes = {n:v for n,v in r8.get('artifacts', {}).items() if n.startswith('trained_distilbert_model/')}
    required = ['config.json','label_mapping.json','training_provenance.json','tokenizer.json','tokenizer_config.json']
    require(all('trained_distilbert_model/'+n in model_hashes for n in required), 'Missing recorded encoder files')
    require(any(n.endswith(('.safetensors','.bin')) and 'model' in Path(n).name for n in model_hashes),
            'No recorded encoder weights')
    present = {p.relative_to(model_dir).as_posix() for p in model_dir.rglob('*') if p.is_file()}
    require(present == {n.removeprefix('trained_distilbert_model/') for n in model_hashes},
            'Current encoder directory differs from Step 8 inventory')
    hashes8 = dict(model_hashes)
    hashes8.update(declared_hashes(r8, [f'serialized/{s}_serialized.csv' for s in SPLITS], 'Step 8 serialized inputs'))
    require(hashes8 == s9.get('step8_artifacts'), 'Step 9 consumed different Step 8 model/serialized inputs')
    for name, digest in model_hashes.items():
        require(valid_digest(digest), f'Invalid model hash: {name}')
        snapshot_file(artifact_path(model_dir, name.removeprefix('trained_distilbert_model/')), audit, digest)
    config = read_input(model_dir/'config.json', audit)
    mapping = read_input(model_dir/'label_mapping.json', audit)
    provenance = read_input(model_dir/'training_provenance.json', audit)
    max_length = p8.get('max_length')
    require(type(max_length) is int and max_length > 2, 'Invalid Step 8 max_length')
    require(mapping.get('dataset_id') == build['dataset_id'] and mapping.get('encoder_source') == 'finetuned'
            and mapping.get('serializer_version') == SERIALIZER and mapping.get('label_order') == list(LABELS)
            and mapping.get('label_to_id') == LABEL_TO_ID and mapping.get('observable_fields') == list(FIELDS)
            and mapping.get('metadata_fields') == list(METADATA)
            and mapping.get('observable_window_seconds') == 1 and mapping.get('max_length') == max_length,
            'Encoder mapping differs from current schema/dataset')
    require(config.get('model_type') == 'distilbert' and config.get('dim') == 768
            and config.get('step7_dataset_id') == build['dataset_id']
            and config.get('observable_serializer_version') == SERIALIZER
            and config.get('label2id') == LABEL_TO_ID
            and config.get('id2label') == {str(v):k for k,v in LABEL_TO_ID.items()}
            and max_length <= config.get('max_position_embeddings', 0), 'Encoder config differs')
    for key, expected in {'dataset_id':build['dataset_id'], 'run_id':r8['run_id'],
        'step7_build_manifest_sha256':h7, 'input_sha256':inputs, 'serializer_version':SERIALIZER,
        'max_length':max_length, 'encoder_source':'finetuned', 'initializer_commit':r8.get('initializer_commit'),
        'tokenizer_sha256':r8.get('tokenizer_sha256'),
        'model_artifacts':{n.removeprefix('trained_distilbert_model/'):v for n,v in model_hashes.items()
                           if not n.endswith('/training_provenance.json')}}.items():
        require(provenance.get(key) == expected, f'Encoder training provenance differs: {key}')
    report_hash = declared_hashes(r9, ['embedding_extraction_report.json'], 'Step 9 report')
    report = read_input(root9/'embedding_extraction_report.json', audit,
                        expected_sha=report_hash['embedding_extraction_report.json'])
    encoder = r9.get('encoder', {})
    require(report.get('status') == 'COMPLETE' and report.get('run_id') == r9['run_id']
            and report.get('dataset_id') == build['dataset_id'] and report.get('step8_run_id') == r8['run_id']
            and report.get('input_sha256') == inputs and report.get('step8_model_sha256') == model_hashes,
            'Step 9 report provenance differs')
    require(r9.get('encoder_source') == 'finetuned' and report.get('embedding_encoder_source') == 'finetuned'
            and encoder == report.get('encoder_source_summary')
            and encoder.get('representation_type') == 'supervised_finetuned_distilbert_cls'
            and encoder.get('pooling') == 'last_hidden_state[:, 0, :]'
            and encoder.get('normalization') == 'none' and encoder.get('weight_updates') is False
            and encoder.get('embedding_dim') == 768, 'Step 9 must be unchanged fine-tuned raw CLS')
    require(project_path(encoder.get('model_source'), root) == root/MODEL_REL
            and project_path(encoder.get('tokenizer_source'), root) == root/MODEL_REL
            and project_path(report.get('source_model_dir'), root) == root/MODEL_REL,
            'Step 9 records a different model/tokenizer source')
    require(report.get('serializer_version') == SERIALIZER and report.get('observable_fields') == list(FIELDS)
            and report.get('metadata_fields') == list(METADATA) and report.get('label_order') == list(LABELS)
            and report.get('label_to_id') == LABEL_TO_ID and report.get('max_length') == max_length
            and report.get('embedding_dim') == 768 and report.get('embedding_dtype') == 'float32'
            and report.get('observable_window_seconds') == 1 and report.get('tst_sequence_lengths') == LENGTHS,
            'Step 9 serializer/schema/extraction contract differs')
    require(r9.get('protocol', {}).get('fp16') is False and report.get('use_fp16_inference') is False,
            'Current main extraction requires FP32 without AMP')
    fp = r8.get('tokenizer_sha256')
    require(valid_digest(fp) and s9.get('tokenizer_sha256') == fp and encoder.get('tokenizer_sha256') == fp,
            'Step 8/9 tokenizer fingerprints differ')
    hashes9 = dict(report_hash)
    hashes9.update(declared_hashes(r9, [f'{s}_window_{kind}' for s in SPLITS
                                      for kind in ('embeddings.npz','metadata.csv')], 'Step 9 outputs'))
    require(set(r9.get('completed_splits', {})) == set(SPLITS)
            and set(report.get('splits', {})) == set(SPLITS), 'Step 9 split completion missing')
    for split in SPLITS:
        record, summary = r9['completed_splits'][split], report['splits'][split]
        require(record.get('summary') == summary and summary.get('rows') == build['counts'][split]['rows']
                and summary.get('embedding_shape') == [build['counts'][split]['rows'], 768]
                and summary.get('embedding_dtype') == 'float32', f'{split}: Step 9 summary differs')
        require(all(record.get('artifacts', {}).get(n) == hashes9[n] for n in
                    (f'{split}_window_embeddings.npz', f'{split}_window_metadata.csv'))
                and all(v == r9['artifacts'].get(n) for n,v in record.get('artifacts', {}).items()),
                f'{split}: Step 9 completion hashes differ')
    contract = {'dataset_id':build['dataset_id'], 'step8_run_id':r8['run_id'], 'step9_run_id':r9['run_id'],
                'manifest_sha256':{'step7':h7, 'step8':h8, 'step9':recorded_digest(root9/'run_manifest.json', audit)},
                'input_sha256':inputs, 'step8_model_sha256':model_hashes, 'step9_artifacts':hashes9,
                'tokenizer_sha256':fp, 'observability_audit':observability}
    return r8, r9, report, config, mapping, contract


def verify_episode_rows(frame, split, inventory):
    inv = inventory[inventory.split == split].set_index('episode_id')
    group = frame.groupby('episode_id')
    require(set(group.groups) == set(inv.index), f'{split}: episodes differ from Step 7 allocation')
    observed = group.agg(label=('label','first'), row_count=('label','size'),
                         min_episode_step=('episode_step','min'), max_episode_step=('episode_step','max'),
                         min_window_id=('window_id','min'), max_window_id=('window_id','max'))
    for col in observed:
        require(np.array_equal(observed[col].to_numpy(), inv.loc[observed.index,col].to_numpy()),
                f'{split}: episode inventory mismatch: {col}')
    times = frame.assign(__time=pd.to_datetime(frame.timestamp, utc=True)).groupby('episode_id').__time
    for actual, name in [(times.min(),'start_timestamp'), (times.max(),'end_timestamp')]:
        require(np.array_equal(actual.to_numpy(), pd.to_datetime(inv.loc[actual.index,name], utc=True).to_numpy()),
                f'{split}: episode inventory mismatch: {name}')


def verify_serialization(frame, serialized, split):
    require(list(serialized.columns) == list(METADATA)+['label_id','text'] and len(frame) == len(serialized),
            f'{split}: Step 8 serialized schema/rows differ')
    for col in METADATA:
        a, b = frame[col], serialized[col]
        if col in ('episode_step', 'window_id'):
            a, b = integer_array(a,col), integer_array(b,col)
        elif col == 'timestamp':
            a, b = pd.to_datetime(a,utc=True).to_numpy(), pd.to_datetime(b,utc=True).to_numpy()
        else:
            a, b = a.astype(str).to_numpy(), b.astype(str).to_numpy()
        require(np.array_equal(a,b), f'{split}: Step 8 serialized identity/order mismatch: {col}')
    require(np.array_equal(integer_array(serialized.label_id,'serialized label ID'),
                           frame.label.map(LABEL_TO_ID).to_numpy()), f'{split}: serialized label mapping differs')
    # Exact text identity with saved Step 8 input, not just presence of 22 field names.
    for i, values in enumerate(frame[list(FIELDS)].itertuples(index=False, name=None)):
        text, _ = serialize_values(values)
        require(text == serialized.text.iloc[i], f'{split}: Step 8 serialized text mismatch at row {i}')


def tokenizer_fingerprint(tokenizer):
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    backend.pop('padding', None); backend.pop('truncation', None)
    return object_hash({'backend':backend, 'special_tokens_map':tokenizer.special_tokens_map,
                        'model_input_names':tokenizer.model_input_names})


def local_tokenizer(model_dir, config, expected_hash):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError('Use the existing Step 8/9 environment with transformers; no download is performed') from exc
    tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True, local_files_only=True, trust_remote_code=False)
    require(tok.is_fast, 'The exact saved fast tokenizer is required for offset coverage')
    require(tokenizer_fingerprint(tok) == expected_hash, 'Loaded tokenizer fingerprint differs from Steps 8/9')
    vocab = tok.get_vocab()
    require(vocab and min(vocab.values()) >= 0 and max(vocab.values()) < config.get('vocab_size',0),
            'Tokenizer IDs exceed checkpoint vocabulary')
    require(tok.unk_token_id is not None, 'Tokenizer has no unknown-token identity')
    return tok
def serialize_values(values):
    require(len(values)==len(FIELDS), 'Serializer needs exactly 22 observable values')
    pieces, spans, offset = [], [], 0
    for field, value in zip(FIELDS, values):
        require(not pd.isna(value), f'{field}: null value')
        value = str(value).strip()
        require(bool(value), f'{field}: empty value')
        prefix = f'{field} is '
        piece = prefix+value
        spans.append({'field':field, 'field_start':offset,
                      'value_start':offset+len(prefix), 'end':offset+len(piece)})
        pieces.append(piece)
        offset += len(piece)+2
    return '; '.join(pieces)+'.', spans


def row_coverage(full, kept, spans):
    """Count retained subword tokens, excluding special tokens, by exact offsets."""
    def tokens(x):
        require(len(x['input_ids'])==len(x['offset_mapping'])==len(x['special_tokens_mask']),
                'Tokenizer output lengths differ')
        return Counter((int(i), int(a), int(b)) for i,(a,b),sp in zip(
            x['input_ids'],x['offset_mapping'],x['special_tokens_mask']) if not sp and b>a)
    whole, remaining = tokens(full), tokens(kept)
    require(not (remaining-whole), 'Truncated tokenizer tokens are not a subset of the full encoding')
    totals = [[0,0,0,0,False] for _ in spans]
    ends = [s['end'] for s in spans]
    for key,count in whole.items():
        token_id,start,end = key
        k = bisect_right(ends,start)
        while k<len(spans) and spans[k]['field_start']<end:
            row = totals[k]
            row[0] += count
            row[1] += remaining[key]
            if spans[k]['value_start']<end:
                row[2] += count
                row[3] += remaining[key]
                row[4] = row[4] or token_id==full.get('unk_token_id')
            k += 1
    out = []
    for span,(ct,cr,vt,vr,unknown) in zip(spans,totals):
        require(ct>0 and vt>0, f'Tokenizer produced no tokens for {span["field"]}')
        stats = {'field':span['field'], 'value_has_unk':unknown}
        for name,total,retained in [('clause',ct,cr),('value',vt,vr)]:
            stats[name+'_status'] = 'full' if retained==total else 'absent' if retained==0 else 'partial'
        out.append(stats)
    return out


def build_vocab(train, val):
    vocab = {'source':'train_only', 'unknown_id':0, 'fields':{}}
    rows = []
    for f in FIELDS:
        values = sorted(set(train[f].astype(str).str.strip()))
        mapping = {v:i+1 for i,v in enumerate(values)}
        vocab['fields'][f] = mapping
        unseen = val[f].astype(str).str.strip().map(lambda x:x not in mapping)
        rows.append({'field':f, 'train_vocabulary_size':len(mapping),
                     'val_oov_rows':int(unseen.sum()), 'val_oov_fraction':float(unseen.mean()),
                     'val_oov_values':json.dumps(sorted(set(val.loc[unseen,f].astype(str).str.strip())),
                                                ensure_ascii=False)})
    return vocab, pd.DataFrame(rows)


def validate_frame(df, split, expected):
    require(len(df)==expected['rows'], f'{split}: unexpected row count {len(df)}')
    require(set(df.columns)==set(METADATA+FIELDS), f'{split}: expected exactly metadata + 22 observable columns')
    require(not df[list(METADATA+FIELDS)].isna().any().any(), f'{split}: null input')
    for col in METADATA+FIELDS:
        require(df[col].astype(str).str.strip().ne('').all(), f'{split}: empty {col}')
    require(set(df.label)==set(LABELS), f'{split}: class coverage or unknown labels')
    for col in ['episode_step','window_id']:
        df[col] = integer_array(df[col],f'{split} {col}')
    require(not df[['episode_id','episode_step']].duplicated().any(), f'{split}: duplicate episode steps')
    require(not df[['episode_id','episode_step','window_id']].duplicated().any(),
            f'{split}: duplicate window identity')
    group = df.groupby('episode_id',sort=True)
    sizes = group.size()
    require(len(sizes)==expected['episodes'], f'{split}: episode count mismatch')
    require(all((group[col].nunique()==1).all() for col in ('label','scenario_id','episode_type')), f'{split}: mixed episode metadata')
    ordered = df.sort_values(['episode_id','episode_step'])
    for col in ['episode_step','window_id']:
        require(ordered.groupby('episode_id')[col].diff().dropna().eq(1).all(),
                f'{split}: nonconsecutive {col}')
    timestamps = pd.to_datetime(ordered.timestamp,errors='raise',utc=True)
    require(timestamps.groupby(ordered.episode_id).diff().dropna().dt.total_seconds().eq(1).all(),
            f'{split}: window timestamp spacing is not one second')
    n_l10 = int((sizes-9).clip(lower=0).sum())
    require(n_l10==expected['sequences_stride1']['10'], f'{split}: derived L10 count mismatch')
    return {'rows':len(df),'episodes':len(sizes),'derived_l10_sequences':n_l10,
            'class_rows':{k:int(v) for k,v in df.label.value_counts().items()}}


def validate_metadata(df, metadata, split):
    require(len(metadata)==len(df), f'{split}: Step 9 metadata row count mismatch')
    require(set(METADATA+('label_id','embedding_row_index')).issubset(metadata.columns),
            f'{split}: Step 9 metadata schema mismatch')
    keys=['episode_id','episode_step','window_id']
    metadata=metadata.copy()
    for col in ['episode_step','window_id']:
        metadata[col]=integer_array(metadata[col],f'{split} metadata {col}')
    base=df.set_index(keys,drop=False)
    target=pd.MultiIndex.from_frame(metadata[keys])
    require(base.index.is_unique and target.is_unique and set(base.index)==set(target),
            f'{split}: Step 9 window identities mismatch')
    reordered=not base.index.equals(target)
    aligned=base.reindex(target).reset_index(drop=True)
    for col in METADATA:
        if col in ('window_id','episode_step'):
            a=metadata[col].to_numpy();b=aligned[col].to_numpy()
        elif col=='timestamp':
            a=pd.to_datetime(metadata[col],utc=True,errors='raise').to_numpy()
            b=pd.to_datetime(aligned[col],utc=True,errors='raise').to_numpy()
        else:
            a=metadata[col].astype(str).to_numpy();b=aligned[col].astype(str).to_numpy()
        require(np.array_equal(a,b),f'{split}: Step 9 metadata/order mismatch in {col}')
    require(np.array_equal(integer_array(metadata.embedding_row_index,'embedding rows'),np.arange(len(df))),
            f'{split}: embedding row indices mismatch')
    require(np.array_equal(integer_array(metadata.label_id,'metadata labels'),aligned.label.map(LABEL_TO_ID).to_numpy()),
            f'{split}: metadata label IDs mismatch')
    return {'alignment':'PASS','keys':keys,'rows_reordered':reordered}


def audit_tokens(df, split, tokenizer, max_length, batch_size):
    counts=[Counter() for _ in FIELDS]
    lengths=[];clipped=0;all_full=0;unknown_rows=0;unknown_tokens=0;examples=[]
    for start in range(0,len(df),batch_size):
        batch=df.iloc[start:start+batch_size][list(FIELDS)]
        serialized=[serialize_values(values) for values in batch.itertuples(index=False,name=None)]
        texts=[x[0] for x in serialized]
        options=dict(add_special_tokens=True,padding=False,return_offsets_mapping=True,
                     return_special_tokens_mask=True,return_attention_mask=False,verbose=False)
        full=tokenizer(texts,truncation=False,**options)
        kept=tokenizer(texts,truncation=True,max_length=max_length,**options)
        for j,(_,spans) in enumerate(serialized):
            f={k:full[k][j] for k in ['input_ids','offset_mapping','special_tokens_mask']}
            f['unk_token_id']=tokenizer.unk_token_id
            t={k:kept[k][j] for k in ['input_ids','offset_mapping','special_tokens_mask']}
            length=len(f['input_ids']);lengths.append(length)
            n_unknown=sum(i==tokenizer.unk_token_id for i in f['input_ids'])
            unknown_rows+=int(n_unknown>0);unknown_tokens+=n_unknown
            require(len(t['input_ids'])<=max_length,'Tokenizer did not honor max_length')
            is_clipped=length>len(t['input_ids']);clipped+=is_clipped
            coverage=row_coverage(f,t,spans)
            is_full=all(x['clause_status']=='full' for x in coverage);all_full+=is_full
            for i,c in enumerate(coverage):
                counts[i]['clause_'+c['clause_status']+'_rows']+=1
                counts[i]['value_'+c['value_status']+'_rows']+=1
                counts[i]['value_contains_unk_rows']+=int(c['value_has_unk'])
            if (not is_full or n_unknown) and len(examples)<5:
                examples.append({'split':split,'row_index':start+j,'full_token_length':length,'tokenizer_unknown_tokens':n_unknown,
                                 'affected_fields':[x for x in coverage if x['clause_status']!='full']})
        if start==0 or (start//batch_size+1)%40==0 or start+batch_size>=len(df):
            print(f'  Token coverage {split}: {min(start+batch_size,len(df)):,}/{len(df):,}',flush=True)
    arr=np.asarray(lengths)
    rows=[]
    for field,c in zip(FIELDS,counts):
        row={'split':split,'field':field,'rows':len(df)}
        for kind in ['clause','value']:
            for status in ['full','partial','absent']:row[kind+'_'+status+'_rows']=c[kind+'_'+status+'_rows']
        row['value_contains_unk_rows']=c['value_contains_unk_rows'];rows.append(row)
    summary={'rows':len(df),'tokenizer_unknown_rows':unknown_rows,'tokenizer_unknown_tokens':unknown_tokens,
             'coverage_interpretation':'hypothetical truncation at max_length; real Step 8/9 reject overlength','truncated_rows':int(clipped),'truncated_fraction':clipped/len(df),
             'all_22_clauses_retained_rows':int(all_full),'min_tokens':int(arr.min()),
             'max_tokens':int(arr.max()),'mean_tokens':float(arr.mean()),
             'p50_p95_p99_tokens':[float(x) for x in np.quantile(arr,[.5,.95,.99])]}
    histogram=[{'split':split,'full_token_length':int(k),'rows':int(v)} for k,v in sorted(Counter(lengths).items())]
    return rows,summary,examples,histogram


def sequence_reference(root, build, contract, metadata, audit, findings):
    directory = root/SEQUENCE_REL
    path = directory/'run_manifest.json'
    if not path.is_file():
        add_finding(findings, 'STEP10_PENDING', 'Step 10 is absent; common endpoints are proposed from Step 7 counts only.')
        return {'status':'PENDING', 'common_reference_length':10}
    run = read_input(path, audit)
    if run.get('status') != 'COMPLETE':
        add_finding(findings, 'STEP10_PENDING', 'Step 10 is not COMPLETE; finish it before an encoder comparison.')
        return {'status':'PENDING', 'recorded_status':run.get('status'), 'common_reference_length':10}
    for key in ('dataset_id','step8_run_id','step9_run_id'):
        require(run.get(key) == contract[key], f'Step 10 belongs to a different {key}')
    source = run.get('source', {})
    require(source.get('step9_manifest_sha256') == contract['manifest_sha256']['step9']
            and source.get('artifacts') == contract['step9_artifacts']
            and source.get('observability_audit') == contract['observability_audit'],
            'Step 10 source artifacts/observable-effect audit differ')
    for key in ('dataset_id','step8_run_id','step9_run_id'):
        require(source.get(key) == contract[key], f'Step 10 source differs: {key}')
    p = run.get('protocol', {})
    require(p.get('sequence_lengths') == LENGTHS and p.get('stride') == 1
            and p.get('window_seconds') == 1 and p.get('save_metadata') is True,
            'Step 10 temporal protocol differs')
    names = ['sequence_build_report_index.json']
    for length in LENGTHS:
        names.append(f'sequence_build_report_L{length}.json')
        for split in SPLITS:
            names.extend([f'{split}_tst_sequences_L{length}.npz', f'{split}_tst_sequence_metadata_L{length}.csv',
                          f'{split}_tst_sequence_observable_effects_L{length}.csv'])
    hashes10 = declared_hashes(run, names, 'Step 10 outputs')
    index = read_input(directory/names[0], audit, expected_sha=hashes10[names[0]])
    require(index.get('status') == 'COMPLETE' and index.get('run_id') == run.get('run_id')
            and index.get('source_artifact_sha256') == contract['step9_artifacts']
            and index.get('observability_audit') == contract['observability_audit']
            and index.get('stride') == 1, 'Step 10 report index differs')
    for key in ('dataset_id','step8_run_id','step9_run_id'):
        require(index.get(key) == contract[key], f'Step 10 index differs: {key}')
    for length in LENGTHS:
        name = f'sequence_build_report_L{length}.json'
        report = read_input(directory/name, audit, expected_sha=hashes10[name])
        for key in ('dataset_id','step8_run_id','step9_run_id','run_id',
                    'source_artifact_sha256','observability_audit'):
            require(report.get(key) == index.get(key), f'L{length}: Step 10 report differs: {key}')
        require(report.get('sequence_length') == length, f'L{length}: Step 10 report length differs')
        for split in SPLITS:
            name = f'{split}_tst_sequence_observable_effects_L{length}.csv'
            summary = report.get('splits', {}).get(split, {})
            require(summary.get('sequence_observable_effects_sha256') == hashes10[name]
                    and project_path(summary.get('sequence_observable_effects_csv_path'), root) == directory/name,
                    f'{split}/L{length}: Step 10 declared observable-effect file/hash differs')
    common = index.get('common_endpoint_report', {})
    require(common.get('reference_length') == 10, 'Step 10 common endpoint reference must be L10')
    for split in SPLITS:
        info = common.get('splits', {}).get(split, {})
        require(info.get('common_endpoint_count') == build['counts'][split]['sequences_stride1']['10']
                and info.get('available_sequences_by_L') == build['counts'][split]['sequences_stride1'],
                f'{split}: Step 10 common count differs from Step 7')
        if split in metadata:
            md = metadata[split]
            steps = integer_array(md.episode_step, 'endpoint episode step')
            starts = pd.Series(steps).groupby(md.episode_id).transform('min').to_numpy()
            ends = np.flatnonzero(steps-starts >= 9)
            digest = hashlib.sha256(np.asarray(ends, dtype='<i8').tobytes()).hexdigest()
            require(info.get('sorted_end_row_indices_sha256') == digest, f'{split}: Step 10 common endpoint identity differs')
    contract.update(step10_run_id=run['run_id'], step10_artifacts=hashes10, common_endpoint_report=common)
    contract['manifest_sha256']['step10'] = recorded_digest(path,audit)
    return {'status':'VERIFIED_MANIFEST_CHAIN', 'step10_run_id':run['run_id'], 'common_endpoint_report':common,
            'train_val_endpoint_identity':'recomputed from verified Step 9 metadata',
            'test_endpoint_identity':'declared Step 10 digest only; no test metadata/arrays read',
            'observable_effect_sidecars':'Step 7/10 manifest linkage only; no sidecar contents read',
            'sequence_array_contents':'not read or independently revalidated in Step 42'}


def step11_source_contract(contract):
    """Expose precisely the fields emitted by the authoritative Step 11 loader."""
    source = {key:contract[key] for key in STEP11_SOURCE_KEYS}
    source['manifest_sha256'] = {key:contract['manifest_sha256'][key] for key in ('step8','step9','step10')}
    return source


def step11_identity_hash(run):
    # Step 11 hashes these six Python dicts before JSON converts their integer keys.
    require(all(key in run for key in STEP11_IDENTITY_KEYS), 'Incomplete Step 11 recorded identity')
    identity = {key:run[key] for key in STEP11_IDENTITY_KEYS}
    identity['settings'] = dict(run['settings'])
    for name in STEP11_LENGTH_TABLES:
        values = identity['settings'].get(name)
        require(isinstance(values, dict), f'Step 11 settings lack length table: {name}')
        restored = {int(key):value for key,value in values.items()}
        require(len(restored) == len(values), f'Ambiguous Step 11 length keys: {name}')
        identity['settings'][name] = restored
    return object_hash(identity)


def temporal_reference(root, explicit_run, contract, audit, findings):
    """Read configuration manifests only; never select runs using test scores."""
    if 'step10_run_id' not in contract:
        return {'status':'PENDING_STEP10', 'preset_head_seed':REFERENCE_HEAD_SEED}
    expected_source = step11_source_contract(contract)
    selected = Path(explicit_run).resolve() if explicit_run is not None else root/BASE_OUTPUT_REL/'tst_training_finetuned_main'
    path = selected/'run_manifest.json'
    r11 = read_input(path, audit) if path.is_file() else None
    linkage, r12 = None, None
    main_matches = (r11 is not None and r11.get('status') == 'COMPLETE'
                    and r11.get('source_contract') == expected_source
                    and r11.get('settings', {}).get('DEFAULT_RANDOM_SEED') == REFERENCE_HEAD_SEED)
    if explicit_run is None and not main_matches:
        # A COMPLETE matching main run takes precedence; partial optional runs do
        # not prevent the independent train/validation input audit from finishing.
        r11 = None
        path12 = root/BASE_OUTPUT_REL/'tst_training_finetuned_multiseed/multiseed_run_manifest.json'
        if path12.is_file():
            r12 = read_input(path12, audit)
            if r12.get('status') == 'COMPLETE':
                identity = r12.get('identity', {})
                require(identity and r12.get('job_id') == object_hash(identity), 'Step 12 recorded identity differs')
                require(project_path(identity.get('output_root'), root) == path12.parent,
                        'Step 12 recorded output root differs')
                seeds, records = identity.get('seeds', []), r12.get('runs', [])
                require(seeds and all(type(seed) is int for seed in seeds) and len(set(seeds)) == len(seeds)
                        and len(records) == len(seeds)
                        and sorted(x.get('seed') for x in records) == sorted(seeds),
                        'Step 12 COMPLETE seed inventory differs')
                matches = [x for x in records if x['seed'] == REFERENCE_HEAD_SEED]
                if matches:
                    linkage = matches[0]
                    require(linkage.get('status') in ('completed','reused_main','reused_complete')
                            and valid_digest(linkage.get('manifest_sha256'))
                            and valid_digest(linkage.get('step11_run_id')), 'Step 12 reference-seed completion identity missing')
                    selected = project_path(linkage['output_root'], root)
                    path = selected/'run_manifest.json'
                    r11 = read_input(path, audit) if path.is_file() else None
            else:
                add_finding(findings, 'STEP12_PENDING', 'Step 12 is not COMPLETE; its recorded paths are not adopted as a head reference.')
    if r11 is None or r11.get('status') != 'COMPLETE':
        if explicit_run is not None or linkage is not None:
            raise ValueError(f'Selected/recorded Step 11 reference is missing or not COMPLETE: {path}')
        add_finding(findings, 'TST_REFERENCE_PENDING', 'No COMPLETE matching preset-seed Step 11 reference is available; input audit can finish independently.')
        return {'status':'PENDING', 'preset_head_seed':REFERENCE_HEAD_SEED, 'path':str(selected)}
    require(step11_identity_hash(r11) == r11.get('run_id'), 'Step 11 recorded identity differs')
    require(project_path(r11.get('output_root'), root) == selected
            and project_path(r11.get('source_embedding_dir'), root) == root/EMBEDDING_REL
            and project_path(r11.get('source_sequence_dir'), root) == root/SEQUENCE_REL,
            'Step 11 recorded input/output paths differ')
    if linkage is not None:
        require(linkage.get('manifest_sha256') == recorded_digest(path, audit)
                and linkage.get('step11_run_id') == r11.get('run_id')
                and r12['identity'].get('train_script_sha256') == r11.get('script_sha256')
                and r12['identity'].get('ordered_lengths') == r11.get('selected_sequence_lengths')
                and (r12['identity'].get('missing_aware') is None
                     or r12['identity']['missing_aware'] == r11.get('settings', {}).get('USE_MISSING_AWARE_TRAINING')),
                'Step 12 reference-seed linkage differs')
    source = r11.get('source_contract', {})
    require(source == expected_source and r11.get('dataset_id') == contract['dataset_id'],
            'Step 11 reference uses a different dataset/model/sequence contract')
    settings = r11.get('settings', {})
    require(settings.get('DEFAULT_RANDOM_SEED') == REFERENCE_HEAD_SEED,
            'Step 42 reference is the prespecified seed 20260610, not a best-performing seed')
    require(10 in r11.get('selected_sequence_lengths', []), 'Step 11 reference has no L10 run')
    expected = {'D_MODEL':128, 'NUM_HEADS':4, 'NUM_ENCODER_LAYERS':1, 'DIM_FEEDFORWARD':256,
        'DROPOUT':0.25, 'POOLING':'last', 'MAX_EPOCHS':30, 'PATIENCE':7, 'BATCH_SIZE':256,
        'LEARNING_RATE':7e-5, 'WEIGHT_DECAY':5e-4, 'GRAD_CLIP_NORM':1., 'USE_CLASS_WEIGHTS':False,
        'LABEL_SMOOTHING':.03, 'PRIMARY_SELECTION_METRIC':'val_macro_f1',
        'USE_SHORTCUT_RESISTANT_TRAINING':False, 'USE_LENGTH_AWARE_REGULARIZATION':False,
        'USE_MISSING_AWARE_TRAINING':False, 'RUN_ROBUST_EVIDENCE_BUDGET_EVAL':False,
        'USE_ROBUST_EVAL_AS_PRIMARY_FOR_LONG_CONTEXTS':False,
        'SEQUENCE_FIT_SUBSET':'full', 'PRIMARY_TEST_SUBSET':'full'}
    mismatches = {k:{'expected':v,'recorded':settings.get(k)} for k,v in expected.items() if settings.get(k) != v}
    if mismatches:
        add_finding(findings, 'TST_CONTROL_DIFFERENCE', 'Selected head settings differ from current main defaults: '+', '.join(mismatches))
    return {'status':'VERIFIED_MANIFEST_CHAIN', 'path':str(selected), 'step11_run_id':r11['run_id'],
            'preset_head_seed':REFERENCE_HEAD_SEED, 'recorded_settings':settings,
            'expected_main_controls':expected, 'main_control_differences':mismatches,
            'selection_source':'Step 12 recorded seed path' if linkage else 'explicit/main preset seed',
            'selected_lengths_in_original_order':r11['selected_sequence_lengths'],
            'seed_policy':r11.get('seed_policy'), 'checkpoint_contents_or_predictions_read':False,
            'interpretation':'Configuration provenance only; no head reproduction or metric re-evaluation.'}


def comparison_design(reference, findings, max_length, contract, build):
    return {
        'status':'DRAFT_NOT_FROZEN', 'protocol_version':PROTOCOL,
        'dataset_id':contract['dataset_id'], 'source_contract':contract,
        'research_purpose':'TAES reviewer revision: 22-field representation and cross-window evidence aggregation',
        'questions':[
            'Does temporal evidence aggregation help when the encoder is held fixed?',
            'Can lightweight categorical/field encoders support the same observable representation?',
            'What is the contribution of pretrained weights under the same DistilBERT architecture/tokenizer?'],
        'encoder_conditions':['CategoricalEmbedding_MLP','FieldToken_Encoder',
                              'DistilBERT_RandomInit_Train','DistilBERT_Pretrained_Finetune'],
        'encoder_seed_count':5, 'proposed_encoder_seeds':list(range(20260610,20260615)),
        'proposed_fixed_head_seed':REFERENCE_HEAD_SEED, 'sequence_length':10,
        'embedding_interface_dimension':768,
        'pipeline':'window-supervised encoder -> eval/no weight updates during extraction -> independently trained temporal head',
        'aggregator':'same TST architecture/settings, new head weights per encoder run; no checkpoint sharing',
        'within_encoder_control':'same encoder single-window classifier and TST L10, both evaluated on identical L10 endpoints',
        'window_control_limit':'The window classifier and temporal head also differ in capacity/training; report existing last-only/history ablations as supporting evidence, not a pure causal estimate.',
        'aggregation_generality':'Keep the existing same-encoder pooling/LSTM comparisons; this draft does not require TST to win.',
        'inputs':{'observable_fields':list(FIELDS), 'excluded_from_features':list(METADATA),
                  'metadata_use':'episode grouping, labels and evaluation alignment only',
                  'vocabulary':'per-field train-only fitting; unknown ID 0; validation/test never add categories',
                  'label_policy':'retain supplied episode/scenario labels, including quiet/unobserved windows',
                  'tokenizer':'same pinned saved tokenizer for both DistilBERT conditions; random initialization refers to model weights, not tokenizer',
                  'no_silent_truncation':True, 'current_reference_max_length':max_length},
        'split_counts':build['counts'],
        'evaluation':{'primary':'six-class macro-F1 on common L10 endpoints',
                      'secondary':'per-class recall, confusion matrices; all-available populations reported separately',
                      'test_selection':'no seed/model/width/budget/threshold selection using test results',
                      'same_population':'match window identities and whole-episode allocation across every condition/seed'},
        'pretraining_control':'Random and pretrained DistilBERT must share architecture, tokenizer, classification head, data, selection rule and declared training/tuning budget; only initial encoder weights differ.',
        'budget_rule':'Predeclare matched update exposure and tuning allowance; report parameters, training time, memory and full encoder-plus-head inference cost. Equal epochs or a 768D interface alone do not imply equal compute/capacity.',
        'projection_rule':'Lightweight-to-768D projections must be trained on train only and counted in parameters/cost.',
        'seed_rule':'Prespecify all five encoder seeds; reset the temporal-head seed for each encoder run and keep the same selected-L order. Five head seeds from one existing Step 8 encoder are not five encoder seeds.',
        'existing_reference_rule':'The current single Step 8 run is a provenance reference. Reuse as one condition only if the final architecture/budget/selection protocol matches; never count it five times.',
        'statistical_plan':{'estimand':'mean per-run macro-F1, not majority-vote ensemble performance',
            'resampling':'paired whole episodes, class-stratified, same endpoint samples across methods; recompute nonlinear F1 per run then average',
            'uncertainty':'episode CI conditional on the fitted runs; report encoder-seed SD separately',
            'multiplicity':'declare the contrast family and adjustment before evaluating alternatives',
            'equivalence':'requires an independently justified margin and a prespecified equivalence analysis; CI including zero is insufficient'},
        'formal_training_budget_frozen':False, 'formal_test_evaluation_authorized_by_this_file':False,
        'next_decisions':['finalize lightweight widths and matched classification/projection heads',
                          'choose training budgets using train/validation pilots and convergence evidence',
                          'lock protocol, seeds, contrasts and multiplicity family before alternative-encoder test evaluation'],
        'reference_settings':reference, 'review_findings':findings,
        'interpretation_limits':['No alternative-encoder scores produced by Step 42',
            'No claim that existing test data are historically unseen',
            'No operational/NOS3 validity or universal encoder superiority established',
            'No conclusion of encoder equivalence or pretrained-weight benefit from this input audit']}


def finish_outputs(out, manifest, audit, findings):
    write_json(out/'input_audit.json', audit)
    with (out/'findings.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['severity','code','message'])
        writer.writeheader(); writer.writerows(findings)
    manifest['finished_at_utc'] = utc_now()
    manifest['output_sha256'] = {p.name:sha256(p) for p in sorted(out.iterdir())
                                 if p.is_file() and p.name != 'run_manifest.json'}
    write_json(out/'run_manifest.json', manifest)
    print(f'\n{manifest["status"]}\nReadiness: {manifest.get("readiness", "FAILED")}\nResults: {out}', flush=True)


def run(root, model_dir=None, step17_dir=None, step16_dir=None, batch_size=256,
        output_dir=None, tst_run=None):
    # Positional legacy arguments are retained solely to fail clearly, not silently ignore stale sources.
    require(step17_dir is None and step16_dir is None,
            'Step 42 reads shared-nominal Steps 7/8/9 directly. Remove obsolete --step16-run/--step17-run arguments.')
    root = Path(root).resolve()
    require(root.is_dir(), 'Project root does not exist')
    model_dir = Path(model_dir).resolve() if model_dir is not None else root/MODEL_REL
    out = Path(output_dir).resolve() if output_dir is not None else root/OUTPUT_REL/(
        'run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    require(not out.is_relative_to(root/'dataset') and not out.is_relative_to(model_dir)
            and not out.is_relative_to(root/EMBEDDING_REL) and not out.is_relative_to(root/SEQUENCE_REL),
            'Output must be outside immutable data/model/embedding/sequence inputs')
    out.mkdir(parents=True, exist_ok=False)
    audit, findings = [], []
    started = time.perf_counter()
    protocol = {'version':PROTOCOL, 'research_purpose':'TAES revision: representation plus temporal evidence aggregation',
        'required_steps':[7,8,9], 'optional_reference_steps':[10,11,12], 'no_training_or_model_inference':True,
        'observable_splits_read':['train','val'], 'episode_inventory_read':'all splits, metadata/counts only',
        'raw_or_serialized_test_read':False, 'predictions_or_embeddings_read':False,
        'observable_effect_sidecars_read':False,
        'observability_linkage':'Step 7/9/10 recorded manifest/hash linkage only; audit annotations are not features',
        'counts_source':'current Step 7 build and episode inventory; no historic hardcodes',
        'reference_head_seed':REFERENCE_HEAD_SEED, 'silent_truncation_allowed':False,
        'tokenizer_unknown_allowed':False, 'alternative_encoder_protocol_status':'DRAFT'}
    write_json(out/'protocol.json', protocol)  # Fixed before loading any data.
    manifest = {'status':'RUNNING', 'started_at_utc':utc_now(), 'protocol_version':PROTOCOL,
        'script_sha256':sha256(Path(__file__)), 'project_root':str(root),
        'python':platform.python_version(), 'numpy':getattr(np,'__version__',None), 'pandas':getattr(pd,'__version__',None),
        'raw_test_data_read':False, 'training_performed':False, 'encoder_inference_performed':False,
        'test_predictions_read':False, 'embedding_arrays_read':False}
    write_json(out/'run_manifest.json', manifest)
    print('Step 42: revised encoder input / provenance / comparison-readiness audit.\n'
          'Train/validation observables only; local tokenizer; no training or download.\n'
          f'Output: {out}', flush=True)
    try:
        require(type(batch_size) is int and 1 <= batch_size <= 2048, 'batch_size must be 1..2048')
        if DEPENDENCY_ERROR is not None:
            raise RuntimeError('Use the existing Step 8/9 environment: '+DEPENDENCY_ERROR)
        print('[1/5] Check current Step 7/8/9 contracts and saved encoder bytes', flush=True)
        build, inputs, inventory = load_build(root, audit)
        r8, r9, report, config, mapping, contract = load_encoder_contract(root, model_dir, build, inputs, audit)
        manifest.update(dataset_id=build['dataset_id'], source_contract=contract)
        write_json(out/'checkpoint_inventory.json', {
            'status':'VERIFIED_CURRENT_BYTES_AND_RECORDED_8_TO_9_CHAIN',
            'files':[x for x in audit if Path(x['path']).is_relative_to(model_dir)],
            'claim':'Current model/tokenizer bytes match Step 8 completion and the Step 9 extraction source record.',
            'limits':'No encoder inference, embedding-content reproduction, generation-lineage or test-content audit here; Step 41 covers maker/source diagnostics.'})
        frames, metadata, data_summary = {}, {}, {}
        print('[2/5] Verify train/validation episodes, exact serialization and Step 9 alignment', flush=True)
        for split in ('train','val'):
            relative = f'splits/{split}_observable.csv'
            frame = read_input(root/DATA_REL/relative, audit, 'csv', inputs[relative])
            data_summary[split] = validate_frame(frame, split, build['counts'][split])
            verify_episode_rows(frame, split, inventory)
            name = f'{split}_window_metadata.csv'
            md = read_input(root/EMBEDDING_REL/name, audit, 'csv', contract['step9_artifacts'][name])
            aligned = validate_metadata(frame, md, split)
            require(aligned['rows_reordered'] is False, f'{split}: revised Step 9 must preserve Step 7 row order')
            data_summary[split]['step9_alignment'] = aligned
            rel = f'serialized/{split}_serialized.csv'
            serialized = read_input(model_dir.parent/rel, audit, 'csv', r8['artifacts'][rel])
            verify_serialization(frame, serialized, split)
            data_summary[split]['exact_step8_serialization'] = True
            frames[split], metadata[split] = frame, md
            print(f'  {split}: {len(frame):,} windows, {data_summary[split]["episodes"]:,} episodes verified', flush=True)
        reserved = inventory[inventory.split == 'test']
        data_summary['reserved_test'] = {'raw_rows_declared':int(reserved.row_count.sum()),
            'raw_episodes':len(reserved), 'L10_eligible_episodes':int(reserved.row_count.ge(10).sum()),
            'common_L10_endpoints_declared':int((reserved.row_count-9).clip(lower=0).sum()),
            'contents_independently_read_or_verified':False, 'source':'Step 7 episode inventory and recorded split hash'}
        write_json(out/'dataset_audit.json', data_summary)
        print('[3/5] Fit train-only categorical vocabularies and audit all tokenized fields', flush=True)
        vocab, stats = build_vocab(frames['train'], frames['val'])
        write_json(out/'train_field_vocabulary.json', vocab)
        stats.to_csv(out/'vocabulary_audit.csv', index=False, encoding='utf-8-sig')
        if stats.val_oov_rows.gt(0).any():
            add_finding(findings, 'VALIDATION_CATEGORICAL_OOV', 'Validation contains train-unseen field values; lightweight encoders must map them to train-defined unknown ID 0.')
        tokenizer = local_tokenizer(model_dir, config, contract['tokenizer_sha256'])
        import transformers
        manifest['transformers'] = transformers.__version__
        max_length = mapping['max_length']
        summary = {'max_length':max_length, 'tokenizer_class':type(tokenizer).__name__,
            'tokenizer_sha256':contract['tokenizer_sha256'], 'tokenizer_source':str(model_dir),
            'local_files_only':True, 'actual_pipeline_truncation':False,
            'truncation_coverage':'diagnostic hypothetical clipping only; any loss fails this audit',
            'all_train_validation_rows_checked':True, 'test_token_coverage_checked':False, 'splits':{}}
        coverage, examples, histograms = [], [], []
        for split, frame in frames.items():
            rows, info, ex, hist = audit_tokens(frame, split, tokenizer, max_length, batch_size)
            summary['splits'][split] = info
            coverage.extend(rows); examples.extend(ex); histograms.extend(hist)
        write_json(out/'tokenization_summary.json', summary)
        write_json(out/'truncation_examples.json', examples)
        pd.DataFrame(coverage).to_csv(out/'token_coverage_by_field.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(histograms).to_csv(out/'token_length_histogram.csv', index=False, encoding='utf-8-sig')
        bad = [s for s,x in summary['splits'].items() if x['truncated_rows'] or x['tokenizer_unknown_rows']
               or x['all_22_clauses_retained_rows'] != x['rows']]
        require(not bad, 'Step 8/9 no-loss tokenizer contract failed for '+', '.join(bad)+
                '; inspect tokenization_summary.json and truncation_examples.json. Do not silently truncate or substitute a tokenizer.')
        print('[4/5] Inspect available sequence/head reference manifests without predictions', flush=True)
        sequence = sequence_reference(root, build, contract, metadata, audit, findings)
        temporal = temporal_reference(root, tst_run, contract, audit, findings)
        settings = {'encoder_settings_verified_for_current_cohort':True,
            'step8_run_id':r8['run_id'], 'step8_training_protocol':r8['protocol'],
            'step8_runtime':r8.get('runtime', {}), 'initializer_commit':r8.get('initializer_commit'),
            'step9_extraction_protocol':r9.get('protocol', {}), 'sequence_reference':sequence,
            'temporal_reference':temporal, 'encoder_test_scores_used_for_selection':False}
        write_json(out/'reference_settings.json', settings)
        write_json(out/'comparison_design.json', comparison_design(settings, findings, max_length, contract, build))
        print('[5/5] Recheck read inputs and seal audit outputs', flush=True)
        for item in list(audit):
            snapshot_file(item['path'], audit, item['sha256'])
        manifest.update(status='AUDIT_COMPLETE', readiness='REVIEW_REQUIRED' if findings else 'READY_FOR_PROTOCOL_REVIEW',
            source_contract=contract, elapsed_seconds=time.perf_counter()-started,
            audit_claim='Input/provenance checks completed; no alternative-encoder comparison performed',
            test_content_verification='NOT_PERFORMED; only upstream declared hashes and episode counts used')
        finish_outputs(out, manifest, audit, findings)
        return out
    except Exception as exc:
        manifest.update(status='FAILED', readiness='FAILED', error_type=type(exc).__name__, error=str(exc),
                        elapsed_seconds=time.perf_counter()-started)
        (out/'failure_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        finish_outputs(out, manifest, audit, findings)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--model-dir', type=Path, help='Optional identical encoder bundle, with its Step 8 manifest beside it')
    parser.add_argument('--tst-run', type=Path, help='Explicit COMPLETE preset-seed Step 11 directory')
    parser.add_argument('--output-dir', type=Path, help='New directory; existing results are never overwritten')
    parser.add_argument('--batch-size', type=int, default=256, help='CPU tokenizer audit batch size, not a training setting')
    parser.add_argument('--step17-run', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--step16-run', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        run(args.project_root, args.model_dir, args.step17_run, args.step16_run, args.batch_size,
            args.output_dir, args.tst_run)
        return 0
    except Exception as exc:
        print(f'\nAudit stopped: {exc}\nInputs and previous outputs are preserved. '
              'Review the error and the new run directory, if printed; do not retrain to bypass provenance checks.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
