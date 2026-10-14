#!/usr/bin/env python3
"""Step 44: shared-nominal random-init pilot matched to Step 43 pretrained.

Place beside the delivered revised 42 and 43 scripts (43 filename suffixes are OK).
Run AFTER the revised Step 43 main pilot has completed:
    python "44.random_init_matched_pilot.py"
Read-only preflight:
    python "44.random_init_matched_pilot.py" --check-only
If several eligible 43 runs exist, explicitly choose --source-run "FULL_RUN_PATH".
Resume the same run:
    python "44.random_init_matched_pilot.py" --resume "outputs/shared_nominal_v2/44_random_init_matched_pilot/run_..."

Only random-init DistilBERT is trained from scratch, using the source pretrained
training rules (main defaults: lr 2e-5, <=6 epochs), then a fresh L10 TST on its
selected raw CLS features. Model/training code is reused from the verified 43.
No pretrained weights are loaded into the random encoder. The official base
bundle supplies its architecture/tokenizer and is verified against source 43.
Same seeds, data, cache, batches, scheduler, early stopping and TST configuration;
actual epochs/optimizer updates may differ. Window-common and TST scores use
identical validation endpoints. All-window F1 is retained for checkpoint selection.

One train/validation diagnostic seed, not the formal encoder comparison. No test
windows, predictions or embeddings are read; upstream provenance metadata may
contain declared test counts/hashes. No automatic downloads or source-run writes.
Keep the full source 43 folder (including cache and selected validation logits).
43's epoch checkpoints preserve RNG/optimizer/scheduler/scaler state on resume.
The engine must be the exact 43 source recorded by the selected completed run;
there is no hard-coded historical code hash or historical run path.
All current shared-nominal episode labels are retained, including quiet attack
windows. Counterfactual annotations are never features, sample filters or labels.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path, PureWindowsPath
import platform
import sys
import traceback

SOURCE_VERSION = 'shared_nominal_trainval_pilot_v3'
PROTOCOL_VERSION = 'shared_nominal_matched_random_pilot_v3'
DATA_REL = 'dataset/shared_nominal_v2'
SOURCE_REL = 'outputs/shared_nominal_v2/43_encoder_comparison_trainval_pilot'
OUTPUT_REL = 'outputs/shared_nominal_v2/44_random_init_matched_pilot'
TARGET = 'DistilBERT_RandomInit_Train'
REFERENCE = 'DistilBERT_Pretrained_Finetune'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_hash(obj):
    data = json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def load_engine(path, expected_sha):
    path = Path(path).resolve()
    require(path.is_file(), 'Keep the revised Step 43 Python script beside Step 44: ' + str(path))
    require(valid_sha(expected_sha) and digest(path) == expected_sha,
            'Step 43 code differs from the completed source run. Keep the exact 43 file used for that run.')
    spec = importlib.util.spec_from_file_location('taes_step43_engine', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.PROTOCOL_VERSION == SOURCE_VERSION, 'Unexpected Step 43 protocol version')
    return module


def require_separate(output, source):
    output, source = Path(output).resolve(), Path(source).resolve()
    require(output != source and source not in output.parents and output not in source.parents,
            'Output must be separate from the Step 43 source directory')


def read_source(source):
    source = Path(source).resolve()
    require(source.is_dir(), 'Full completed revised Step 43 folder required (not only the review ZIP): ' + str(source))
    manifest = read_json(source / 'run_manifest.json')
    protocol = read_json(source / 'protocol.json')
    require(manifest.get('script_number') == 43 and manifest.get('status') == 'PILOT_COMPLETE'
            and manifest.get('conditions_completed') == 4 and manifest.get('protocol_version') == SOURCE_VERSION,
            'Run the revised Step 43 to completion first; legacy results cannot be used')
    require(valid_sha(protocol.get('script_sha256')) and
            manifest.get('script_sha256') == protocol.get('script_sha256'),
            'Source Step 43 code hash mismatch')
    body = {k: v for k, v in protocol.items() if k != 'identity'}
    require(object_hash(body) == protocol.get('identity') == manifest.get('protocol_identity'),
            'Source protocol identity mismatch')
    p = protocol['protocol']
    require(p.get('version') == SOURCE_VERSION and p.get('formal') is False
            and p.get('test_evaluation') is False and manifest.get('test_data_read') is False,
            'Only the revised train/validation pilot is a valid source')
    require(p.get('source_contract') == manifest.get('source_contract') and
            p['source_contract']['dataset_id'] == manifest.get('dataset_id'), 'Source dataset identity mismatch')
    records = manifest.get('output_sha256', {})
    needed = {'protocol.json', 'input_audit.json', 'cache/complete.json', 'pilot_summary.json'}
    for condition in (TARGET, REFERENCE):
        needed.update(condition + '/' + name for name in
                      ('encoder/complete.json', 'tst/complete.json', 'embeddings/complete.json',
                       'val_tst_predictions_L10.csv', 'val_window_predictions_common_L10.csv',
                       'window_common_L10_metrics.json'))
    require(needed <= set(records), 'Source manifest lacks revised common-endpoint review records')
    for name, expected in records.items():
        path = artifact_path(source, name)
        require(path.is_file() and digest(path) == expected, 'Source artifact hash mismatch: ' + name)
    return {'manifest': manifest, 'protocol': protocol, 'manifest_sha256': digest(source / 'run_manifest.json')}


def make_target_protocol(engine, source):
    require(source['candidate'] == 'main' and source['encoder_seed'] == 20260928,
            'Use the predeclared main Step 43 pilot (seed 20260928), not a best-score-selected candidate')
    require(source.get('version') == SOURCE_VERSION and source.get('nominal_substrate') == 'shared_nominal_v2',
            'Current shared-nominal Step 43 protocol required')
    expected = engine.make_protocol('main', source['bert_micro_batch'], source['device'], source['amp'])
    for key in ('encoder_seed', 'window_head_seed', 'encoder_shuffle_seed', 'tst_seed', 'tst_shuffle_seed',
                'conditions', 'fields', 'labels', 'embedding_dim', 'tst_model', 'candidates', 'encoder_max_epochs',
                'encoder_patience', 'encoder_min_epochs', 'encoder_plateau_delta', 'selection', 'stop',
                'evaluation_precision', 'extraction_precision', 'label_policy', 'audit_annotation_policy'):
        require(source.get(key) == expected[key], 'Source differs from the declared main Step 43 controls: ' + key)
    protocol = copy.deepcopy(source)
    index = engine.CONDITIONS.index(TARGET)
    reference = engine.training_spec(REFERENCE, source)
    protocol['version'] = PROTOCOL_VERSION
    protocol['conditions'] = [TARGET]
    protocol['purpose'] = 'RANDOM_INIT_MATCHED_PRETRAINED_TRAIN_VAL_PILOT'
    protocol['candidates']['main'][index] = reference['learning_rate']
    protocol['encoder_max_epochs'][index] = reference['max_epochs']
    protocol['formal'] = False
    protocol['test_evaluation'] = False
    protocol['comparison_reference'] = REFERENCE
    protocol['limitations'] = [
        'one diagnostic seed; exclude from formal seed statistics',
        'same training rules and maximum budget, not necessarily equal realized optimizer steps or epochs',
        'compared with the Step 43 random run BOTH learning rate and schedule horizon change',
        'validation selects encoder/TST checkpoints and informs later settings; this is not held-out evidence',
        'no full-convergence, equivalence, test-generalization or pure learning-rate causal claim',
        'window and TST heads differ in capacity/training; common endpoints alone do not isolate time aggregation',
        'all supplied shared-nominal episode labels are retained, including quiet attack windows; no evidence-based filtering or relabeling',
        'overall validation F1 alone does not establish attack-evidence use or exclude generator signatures; interpret with the benchmark/evidence audits',
    ]
    require(engine.training_spec(TARGET, protocol) == reference, 'Encoder training settings are not matched')
    require(engine.training_spec(TARGET, protocol, tst=True) ==
            engine.training_spec(REFERENCE, source, tst=True), 'TST control is not matched')
    require(protocol['tst_model'] == engine.TST_MODEL, 'Source TST architecture differs from the loaded engine')
    return protocol


def guard_protocol(path, payload):
    identity = object_hash(payload)
    expected = {'identity': identity, **payload}
    path = Path(path)
    if path.exists():
        require(read_json(path) == expected, 'Resume protocol/inputs/code/runtime changed; use a new run')
    else:
        write_json(path, expected)
    return identity


def load_readonly_cache(engine, source, source_identity, frames, max_length=None):
    cache = Path(source) / 'cache'
    record = read_json(cache / 'complete.json')
    require(record['identity'] == source_identity, 'Source cache identity mismatch')
    length = record.get('max_length')
    require(type(length) is int and length > 2 and (max_length is None or length == max_length),
            'Cached token length differs from the audited source protocol')
    require(set(frames) == {'train', 'val'}, 'Cache accepts train/validation only')
    expected = {f'{split}_{kind}.npy' for split in ('train', 'val')
                for kind in ('input_ids', 'attention_mask', 'categories', 'labels')}
    require(set(record['artifact_sha256']) == expected, 'Unexpected cache files; train/validation only required')
    for name, expected_hash in record['artifact_sha256'].items():
        path = artifact_path(cache, name)
        require(path.is_file() and digest(path) == expected_hash, 'Original Step 43 cache missing/changed: ' + str(path))
    result = {}
    for split in ('train', 'val'):
        frame = frames[split]
        item = {kind: engine.np.load(cache / f'{split}_{kind}.npy', mmap_mode='r', allow_pickle=False)
                for kind in ('input_ids', 'attention_mask', 'labels')}
        require(item['input_ids'].shape == item['attention_mask'].shape == (len(frame), length),
                'Cached token shape mismatch')
        require(item['input_ids'].dtype == engine.np.int32 and item['attention_mask'].dtype == engine.np.int8
                and item['labels'].dtype == engine.np.int64, 'Cached dtype differs from revised Step 43')
        mask = item['attention_mask']
        require(engine.np.isin(mask, [0, 1]).all() and (mask.sum(axis=1) > 2).all()
                and (engine.np.diff(mask, axis=1) <= 0).all(), 'Invalid right-padded token mask')
        require((item['input_ids'] >= 0).all(), 'Invalid negative cached token ID')
        expected_labels = frame.label.map(engine.LABEL_TO_ID).to_numpy(dtype=engine.np.int64)
        require(engine.np.array_equal(item['labels'], expected_labels), 'Cached labels do not match current rows')
        item['sequences'], item['sequence_labels'], item['metadata'] = engine.sequence_view(frame)
        result[split] = item
    return result


def prepare(engine, root, source, source_record, args):
    require(not engine.DEPENDENCY_ERROR,
            'Use the Step 43 training virtual environment. Dependency error: ' + str(engine.DEPENDENCY_ERROR))
    import transformers
    prior = source_record['protocol']
    protocol = make_target_protocol(engine, prior['protocol'])
    require(protocol['device'] in ('cpu', 'cuda'), 'Unknown source device')
    require(protocol['device'] != 'cuda' or engine.torch.cuda.is_available(),
            'Source 43 used CUDA; keep the same CUDA environment for this matched pilot')
    device = engine.torch.device(protocol['device'])
    runtime = dict(python=platform.python_version(), torch=str(engine.torch.__version__),
                   numpy=engine.np.__version__, pandas=engine.pd.__version__, transformers=transformers.__version__,
                   platform=platform.platform(), device=engine.torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                   cuda=engine.torch.version.cuda, cudnn=engine.torch.backends.cudnn.version(),
                   torch_threads=engine.torch.get_num_threads(), torch_interop_threads=engine.torch.get_num_interop_threads(),
                   cuda_device_count=engine.torch.cuda.device_count() if device.type == 'cuda' else 0,
                   numerical_policy=engine.numerical_runtime())
    require(runtime == prior['runtime'], 'Runtime differs from Step 43; retain its Python/packages/device/thread configuration')
    audit_path = args.audit_run
    if audit_path is None:
        recorded = source_record['manifest']['audit_path']
        candidate = Path(recorded)
        audit_path = candidate if candidate.exists() else engine.mapped_path(recorded, root)
    audit_path = Path(audit_path).resolve()
    if args.audit_script is None:
        recorded_script = source_record['manifest'].get('invocation', {}).get('audit_script')
        if recorded_script and Path(recorded_script).is_file():
            args.audit_script = Path(recorded_script).resolve()
    audit, frames, tokenizer_dir, _, inputs = engine.load_inputs(root, audit_path, args.audit_script)
    require(audit['manifest_sha256'] == prior['audit_manifest_sha256'], 'The selected Step 42 audit differs from source 43')
    require(engine.pilot_source_contract(audit['source_contract']) == engine.pilot_source_contract(protocol['source_contract']),
            'Current shared-nominal inputs/observability source contract differs from Step 43')
    # Exact current bytes, while allowing the project/42 script to be relocated or renamed.
    fingerprint = lambda records: sorted((x['sha256'], x['size_bytes']) for x in records)
    require(fingerprint(inputs) == fingerprint(prior['source_files']), 'Input source bytes differ from Step 43')
    inputs = [{k: x[k] for k in ('path', 'sha256', 'size_bytes')} for x in inputs]
    require(audit['tokenization_summary']['max_length'] == protocol['max_length'], 'Audited token budget changed')
    print('Verifying and opening Step 43 train/validation cache read-only...', flush=True)
    data = load_readonly_cache(engine, source, prior['identity'], frames, protocol['max_length'])
    del frames
    base_path = args.base_model_dir
    if base_path is None:
        original = Path(source_record['manifest']['initializer_path'])
        base_path = original if original.is_dir() else None
    base = engine.resolve_base_model(base_path, offline=True)
    names = {'config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt',
             'special_tokens_map.json', 'added_tokens.json'}
    base_hashes = {p.name: digest(p) for p in sorted(base.iterdir()) if p.is_file() and p.name in names}
    require(base_hashes == prior['base_hashes'], 'Base initializer/config/tokenizer differs from Step 43')
    tokenizer = engine.tokenizers_for_run(tokenizer_dir, base, protocol['source_contract']['tokenizer_sha256'])
    for split, item in data.items():
        ids, mask = item['input_ids'], item['attention_mask']
        require((ids < len(tokenizer)).all() and not (ids[mask == 1] == tokenizer.unk_token_id).any()
                and (ids[mask == 0] == tokenizer.pad_token_id).all(), split + ': invalid cached tokenizer IDs')
    reference_metrics = validate_references(engine, source, source_record, data)
    payload = dict(protocol=protocol, runtime=runtime, source_protocol_identity=prior['identity'],
                   source_manifest_sha256=source_record['manifest_sha256'], source_run=str(source),
                   source_files=inputs, engine_sha256=prior['script_sha256'], script_sha256=digest(Path(__file__)),
                   base_hashes=base_hashes, encoder_training=engine.training_spec(TARGET, protocol),
                   tst_training=engine.training_spec(TARGET, protocol, tst=True), reference_metrics=reference_metrics)
    args.audit_script = Path(audit['validator_path'])
    return payload, data, base, device, audit_path


def train_one(engine, data, base, device, payload, out, identity):
    from transformers import DistilBertConfig
    protocol = payload['protocol']
    engine.seed_all(protocol['encoder_seed'])
    config = DistilBertConfig.from_pretrained(str(base), local_files_only=True)
    config._attn_implementation = 'eager'
    encoder = engine.BertEncoder(config, weights_path=None)
    model = engine.WindowModel(encoder, head_seed=protocol['window_head_seed'])
    folder = out / TARGET
    batches = {split: engine.ArrayBatches({'input_ids': item['input_ids'],
                'attention_mask': item['attention_mask']}, item['labels']) for split, item in data.items()}
    stage = object_hash({'run': identity, 'condition': TARGET, 'stage': 'encoder'})
    print(f"Random-init encoder: lr={payload['encoder_training']['learning_rate']}, max_epochs={payload['encoder_training']['max_epochs']}; no pretrained weights.", flush=True)
    trained = engine.fit(model, batches['train'], batches['val'], payload['encoder_training'],
                         folder / 'encoder', stage, device)
    window_logits = engine.np.load(folder / 'encoder' / 'val_logits.npy', allow_pickle=False)
    common_logits = window_logits[data['val']['sequences'][:, -1]]
    common_y = data['val']['sequence_labels']
    common = engine.metrics(common_y, common_logits, 0.)
    common['loss'] = float(engine.nn.functional.cross_entropy(engine.torch.from_numpy(common_logits), engine.torch.from_numpy(common_y)))
    write_json(folder / 'window_common_L10_metrics.json', common)
    save_predictions(engine, folder / 'val_window_predictions_common_L10.csv', data['val'], common_logits,
                     identity, payload, 'window')
    feature_id = object_hash({'stage': stage, 'checkpoint': trained['artifact_sha256']['best.pt']})
    embeddings = engine.extract_features(model, batches, folder / 'embeddings',
                                          payload['encoder_training']['eval_batch_size'], device, feature_id)
    del model, encoder, batches
    if device.type == 'cuda':
        engine.torch.cuda.empty_cache()
    temporal_batches = {split: engine.SequenceBatches(embeddings[split], item['sequences'],
                         item['sequence_labels']) for split, item in data.items()}
    engine.seed_all(protocol['tst_seed'])
    tst = engine.TST()
    embedding_record = read_json(folder / 'embeddings' / 'complete.json')
    temporal_id = object_hash({'run': identity, 'condition': TARGET, 'embeddings': embedding_record['sha256']})
    print('Fresh L10 TST on fixed features from the selected encoder.', flush=True)
    temporal = engine.fit(tst, temporal_batches['train'], temporal_batches['val'], payload['tst_training'],
                          folder / 'tst', temporal_id, device)
    logits = engine.np.load(folder / 'tst' / 'val_logits.npy', allow_pickle=False)
    require(logits.shape == common_logits.shape, 'Window and TST common endpoint populations differ')
    save_predictions(engine, folder / 'val_tst_predictions_L10.csv', data['val'], logits, identity, payload, 'tst')
    return trained, temporal, embedding_record


def summarize(engine, source, out, trained, temporal, embedding_record, payload):
    rows = []
    for name in (TARGET, REFERENCE):
        rows.append(dict(run='step43_reference', condition=name, **payload['reference_metrics'][name]))
    common = read_json(out / TARGET / 'window_common_L10_metrics.json')
    row = dict(window_val_macro_f1=trained['best_metrics']['macro_f1'], window_val_count=trained['best_metrics']['n'],
               window_val_common_macro_f1=common['macro_f1'], common_val_count=common['n'],
               l10_val_macro_f1=temporal['best_metrics']['macro_f1'], l10_val_count=temporal['best_metrics']['n'],
               temporal_delta_val_common_macro_f1=temporal['best_metrics']['macro_f1']-common['macro_f1'],
               encoder_best_epoch=trained['best_epoch'], encoder_epochs=trained['epochs_completed'],
               tst_best_epoch=temporal['best_epoch'], tst_epochs=temporal['epochs_completed'])
    rows.append(dict(run='step44_matched', condition=TARGET, **row))
    for item in rows:
        item.update(dataset_id=payload['protocol']['source_contract']['dataset_id'],
                    encoder_seed=payload['protocol']['encoder_seed'], tst_seed=payload['protocol']['tst_seed'],
                    evaluation_population='common_L10_endpoints for common window/TST metrics', formal_experiment=False)
    engine.write_csv(out / 'comparison_summary.csv', rows)
    write_json(out / 'comparison_summary.json', rows)
    history = engine.pd.read_csv(out / TARGET / 'encoder' / 'history.csv', encoding='utf-8-sig')
    reference = payload['reference_metrics'][REFERENCE]
    assessment = dict(status='REVIEW_REQUIRED', formal=False, test_evaluation=False,
        encoder_convergence=trained['convergence'], tst_convergence=temporal['convergence'],
        first_epoch_f1=float(history.iloc[0].val_macro_f1), final_epoch_f1=float(history.iloc[-1].val_macro_f1),
        best_epoch=trained['best_epoch'], final_minus_first_f1=float(history.iloc[-1].val_macro_f1-history.iloc[0].val_macro_f1),
        amp_skipped_updates=int(history.amp_skipped_updates.sum()), encoder_optimizer_updates=trained['optimizer_updates'],
        tst_optimizer_updates=temporal['optimizer_updates'], encoder_training_seconds=trained['training_seconds'],
        extraction_seconds=embedding_record['elapsed_seconds'], tst_training_seconds=temporal['training_seconds'],
        delta_direction='Step 44 random-init minus Step 43 pretrained',
        random_minus_pretrained_window_common_f1=common['macro_f1']-reference['window_val_common_macro_f1'],
        random_minus_pretrained_l10_f1=temporal['best_metrics']['macro_f1']-reference['l10_val_macro_f1'],
        comparison_scope='same training rules/maximum budget; realized epochs and optimizer updates may differ',
        evaluation_precision='float32', limitations=payload['protocol']['limitations'])
    write_json(out / 'diagnostic_assessment.json', assessment)
    lines = ['# Step 44 matched random-init pilot', '',
             'TRAIN/VALIDATION ONLY; one diagnostic seed; review before formal settings are declared.', '',
             'Random-init matches the selected main Step 43 pretrained training rules and maximum budget.',
             'Step 43 models are not retrained. Actual early stopping and optimizer updates can differ.', '',
             '| Run | Encoder | All-window val F1 | Common-window val F1 | L10 val F1 | L10 minus common-window |',
             '|---|---|---:|---:|---:|---:|']
    lines += [f"| {r['run']} | {r['condition']} | {r['window_val_macro_f1']:.6f} | {r['window_val_common_macro_f1']:.6f} | {r['l10_val_macro_f1']:.6f} | {r['temporal_delta_val_common_macro_f1']:+.6f} |" for r in rows]
    lines += ['', 'Common-window and L10 scores use the exact same validation endpoints.',
              'All-window F1 uses all validation windows and selects the encoder checkpoint.',
              *payload['protocol']['limitations'],
              'Keep full 43/44 run folders; review ZIPs exclude weights, embeddings and logits.']
    (out / 'analysis_summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def artifact_path(root, name):
    require(isinstance(name, str) and not PureWindowsPath(name).drive and '\\' not in name,
            'Unsafe artifact path')
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts,
            'Unsafe artifact path')
    require(not any(p.lower().startswith('test') for p in relative.parts) and relative.suffix != '.npz',
            'Test inputs/embeddings are outside this train/validation pilot')
    path = (Path(root) / relative).resolve()
    require(Path(root).resolve() in path.parents, 'Artifact escapes run directory')
    return path


def resolve_engine(root, explicit=None, expected_sha=None):
    require(valid_sha(expected_sha), 'The completed Step 43 source must declare its engine code hash')
    paths = [Path(root) / explicit] if explicit else sorted(Path(root).glob('43*.py'))
    matches = [p.resolve() for p in paths if p.is_file() and digest(p) == expected_sha]
    require(matches, 'Put the exact 43 Python file used for the completed source run beside 44, or pass --engine-script')
    return matches[0]


def resolve_source(root, explicit=None):
    if explicit is not None:
        return (Path(root) / explicit).resolve()
    build = read_json(Path(root) / DATA_REL / 'build_manifest.json')
    candidates = []
    for path in sorted((Path(root) / SOURCE_REL).glob('run_*/run_manifest.json')):
        item = read_json(path)
        if (item.get('script_number') == 43 and item.get('status') == 'PILOT_COMPLETE'
                and item.get('protocol_version') == SOURCE_VERSION and valid_sha(item.get('script_sha256'))
                and item.get('dataset_id') == build.get('dataset_id')):
            protocol = read_json(path.parent / 'protocol.json')['protocol']
            if protocol.get('candidate') == 'main' and protocol.get('encoder_seed') == 20260928:
                candidates.append(path.parent)
    require(candidates, 'No completed revised main Step 43 for the current dataset. Finish 43 first, or specify --source-run')
    require(len(candidates) == 1,
            'Multiple eligible Step 43 runs: specify the intended --source-run. No latest/best-score selection is made.\n'
            + '\n'.join(str(p) for p in candidates))
    return candidates[0].resolve()


def validate_references(engine, source, record, data):
    """Recompute reference F1 from authenticated logits and check paired endpoint CSVs."""
    protocol = record['protocol']['protocol']
    identity = record['protocol']['identity']
    require(protocol['conditions'] == list(engine.CONDITIONS), 'Unexpected source encoder conditions')
    summary = read_json(source / 'pilot_summary.json')
    require(len(summary) == 4 and {r['condition'] for r in summary} == set(engine.CONDITIONS),
            'Source summary is incomplete or duplicated')
    result = {}
    for condition in (TARGET, REFERENCE):
        folder = source / condition
        completions = {}
        logits = {}
        for stage in ('encoder', 'tst'):
            complete = read_json(folder / stage / 'complete.json')
            cfg = engine.training_spec(condition, protocol, tst=stage == 'tst')
            if stage == 'encoder':
                stage_id = object_hash({'run': identity, 'condition': condition, 'stage': 'encoder'})
            else:
                embeddings = read_json(folder / 'embeddings/complete.json')
                stage_id = object_hash({'run': identity, 'condition': condition, 'embeddings': embeddings['sha256']})
            require(complete['contract'] == object_hash({'identity': stage_id, 'config': cfg}),
                    'Source training settings/contract differ: ' + condition + '/' + stage)
            path = artifact_path(folder / stage, 'val_logits.npy')
            require(digest(path) == complete['artifact_sha256']['val_logits.npy'], 'Source validation logits changed')
            logits[stage] = engine.np.load(path, allow_pickle=False)
            expected_rows = len(data['val']['labels' if stage == 'encoder' else 'sequence_labels'])
            require(logits[stage].shape == (expected_rows, 6) and logits[stage].dtype == engine.np.float32,
                    'Source validation logits shape/precision differs: ' + condition + '/' + stage)
            completions[stage] = complete
            require(complete.get('evaluation_precision') == 'float32', 'Source validation precision differs')
        y = data['val']['sequence_labels']
        common_logits = logits['encoder'][data['val']['sequences'][:, -1]]
        all_metric = engine.metrics(data['val']['labels'], logits['encoder'], 0.)
        common = engine.metrics(y, common_logits, 0.)
        temporal = engine.metrics(y, logits['tst'], 0.)
        for filename, values, kind in (('val_window_predictions_common_L10.csv', common_logits, 'window'),
                                        ('val_tst_predictions_L10.csv', logits['tst'], 'tst')):
            meta = data['val']['metadata']
            dtype = {c: str for c in meta.columns if meta[c].dtype == object}
            actual = engine.pd.read_csv(folder / filename, encoding='utf-8-sig', dtype=dtype)
            require(set(meta.columns) <= set(actual.columns) and len(actual) == len(meta), 'Reference endpoint columns/count mismatch')
            require(actual[list(meta.columns)].equals(meta), 'Reference validation endpoints/order differ')
            require(engine.np.array_equal(actual.y_true, y) and engine.np.array_equal(actual.y_pred, values.argmax(axis=1)),
                    'Reference predictions/labels differ from logits')
            probs = engine.np.exp(values-values.max(axis=1, keepdims=True));probs /= probs.sum(axis=1, keepdims=True)
            require(engine.np.allclose(actual[[f'prob_{i}' for i in range(6)]], probs, rtol=1e-6, atol=1e-7),
                    'Reference probabilities differ from logits')
            expected = dict(dataset_id=protocol['source_contract']['dataset_id'], step43_protocol_identity=identity,
                            condition=condition, encoder_seed=protocol['encoder_seed'],
                            head_seed=protocol['window_head_seed'] if kind == 'window' else protocol['tst_seed'],
                            head_kind=kind, candidate=protocol['candidate'], evaluation_population='common_L10_endpoints')
            for key, value in expected.items():
                require(key in actual and actual[key].eq(value).all(), 'Reference prediction provenance differs: ' + key)
        old = next(r for r in summary if r['condition'] == condition)
        enc, tst = completions['encoder'], completions['tst']
        row = dict(window_val_macro_f1=all_metric['macro_f1'], window_val_count=all_metric['n'],
                   window_val_common_macro_f1=common['macro_f1'], common_val_count=common['n'],
                   l10_val_macro_f1=temporal['macro_f1'], l10_val_count=temporal['n'],
                   temporal_delta_val_common_macro_f1=temporal['macro_f1']-common['macro_f1'],
                   encoder_best_epoch=enc['best_epoch'], encoder_epochs=enc['epochs_completed'],
                   tst_best_epoch=tst['best_epoch'], tst_epochs=tst['epochs_completed'])
        for key, value in row.items():
            require(key in old and abs(old[key]-value) <= 1e-12, 'Source summary differs from predictions: ' + key)
        for complete, metric in ((enc, all_metric), (tst, temporal)):
            require(complete['best_metrics']['n'] == metric['n'] and
                    abs(complete['best_metrics']['macro_f1']-metric['macro_f1']) <= 1e-12,
                    'Source best metrics differ from saved validation predictions')
        result[condition] = row
    return result


def save_predictions(engine, path, val, logits, identity, payload, kind):
    predictions = engine.prediction_rows(val['metadata'], val['sequence_labels'], logits, identity,
                                        TARGET, payload['protocol'], 'common_L10_endpoints', head_kind=kind)
    # Keep source 43 and current 44 identities distinct; a 44 run is not a 43 run.
    predictions['step43_protocol_identity'] = payload['source_protocol_identity']
    predictions['step44_protocol_identity'] = identity
    predictions.to_csv(path, index=False, encoding='utf-8-sig')


def verify_unchanged(engine, source, source_record, payload, base, engine_path):
    require(read_source(source)['manifest_sha256'] == source_record['manifest_sha256'], 'Source 43 changed during training')
    cache = read_json(source / 'cache/complete.json')
    for name, expected in cache['artifact_sha256'].items():
        require(digest(artifact_path(source / 'cache', name)) == expected, 'Source cache changed during training')
    for item in payload['source_files']:
        engine.checked_file(item['path'], item['sha256'], [])
    for name, expected in payload['base_hashes'].items():
        # Official Hugging Face snapshots may link into a sibling blobs directory.
        # These names were collected from the explicit initializer allowlist in prepare.
        require(Path(name).name == name and '\\' not in name and ':' not in name,
                'Unexpected initializer filename')
        require(digest(base / name) == expected, 'Base initializer changed during training')
    require(digest(Path(__file__)) == payload['script_sha256'] and digest(engine_path) == payload['engine_sha256'],
            'Code changed during training')
    for condition in (TARGET, REFERENCE):
        for stage in ('encoder', 'tst'):
            folder = source / condition / stage
            complete = read_json(folder / 'complete.json')
            require(digest(folder / 'val_logits.npy') == complete['artifact_sha256']['val_logits.npy'],
                    'Source reference logits changed during training')


def verify_completed(engine, out, manifest, payload, identity):
    require(manifest.get('conditions_completed') == 1 and manifest.get('protocol_identity') == identity,
            'Incomplete or mismatched Step 44 completion')
    needed = {'protocol.json', 'input_audit.json', 'comparison_summary.json', 'comparison_summary.csv',
              'diagnostic_assessment.json', TARGET + '/val_window_predictions_common_L10.csv',
              TARGET + '/val_tst_predictions_L10.csv', TARGET + '/window_common_L10_metrics.json',
              TARGET + '/encoder/complete.json', TARGET + '/tst/complete.json', TARGET + '/embeddings/complete.json'}
    require(needed <= set(manifest.get('output_sha256', {})), 'Completed Step 44 inventory is incomplete')
    for name, expected in manifest['output_sha256'].items():
        require(digest(artifact_path(out, name)) == expected, 'Completed Step 44 output changed: ' + name)
    for stage in ('encoder', 'tst'):
        folder = out / TARGET / stage
        record = read_json(folder / 'complete.json')
        require(set(record['artifact_sha256']) == {'best.pt', 'last.pt', 'history.csv', 'val_logits.npy'},
                'Completed training inventory differs')
        for name, expected in record['artifact_sha256'].items():
            require(digest(artifact_path(folder, name)) == expected, 'Completed checkpoint/logits changed')
    folder = out / TARGET / 'embeddings'
    record = read_json(folder / 'complete.json')
    require(set(record['sha256']) == {'train', 'val'}, 'Unexpected completed embedding splits')
    for split, expected in record['sha256'].items():
        require(digest(artifact_path(folder, split + '.npy')) == expected, 'Completed embeddings changed')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root', type=Path, help='Default: folder containing this script; inherited on resume')
    parser.add_argument('--source-run', type=Path, help='Full completed revised main Step 43 folder; auto-select only when unique')
    parser.add_argument('--engine-script', type=Path, help='Revised 43 Python file; 43*.py is discovered by its content hash')
    parser.add_argument('--audit-run', type=Path, help='Same Step 42 audit used by source 43; default inherits source')
    parser.add_argument('--audit-script', type=Path, help='Exact revised 42 Python file; filenames may be changed')
    parser.add_argument('--base-model-dir', type=Path, help='Same official base snapshot used by 43; local files only')
    parser.add_argument('--resume', type=Path, help='Existing revised Step 44 folder')
    parser.add_argument('--check-only', action='store_true', help='Read-only input/cache/runtime check; do not train or create a run')
    args = parser.parse_args(argv)
    out = None
    names = ('project_root', 'source_run', 'engine_script', 'audit_run', 'audit_script', 'base_model_dir')
    try:
        root = (args.project_root or Path(__file__).resolve().parent).resolve()
        previous = {}
        if args.resume:
            out = (root / args.resume).resolve()
            require(out.is_dir(), 'Resume folder does not exist')
            previous = read_json(out / 'run_manifest.json')
            require(previous.get('script_number') == 44 and previous.get('protocol_version') == PROTOCOL_VERSION,
                    'Resume must identify a revised Step 44 run; legacy results remain separate')
            require((out / 'protocol.json').is_file(), 'Resume protocol is missing; existing results are preserved')
            for key, value in previous.get('invocation', {}).items():
                if key in names and getattr(args, key) is None and value is not None:
                    setattr(args, key, Path(value))
            root = (args.project_root or root).resolve()
        require(root.is_dir(), 'Project root does not exist')
        args.project_root = root
        for key in names[1:]:
            if getattr(args, key) is not None:
                setattr(args, key, (root / getattr(args, key)).resolve())
        source = resolve_source(root, args.source_run)
        args.source_run = source
        if out is None:
            out = root / OUTPUT_REL / ('run_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
        require_separate(out, source)
        print('Step 44: validating current Step 43 source and matched controls...', flush=True)
        source_record = read_source(source)
        engine_path = resolve_engine(root, args.engine_script, source_record['protocol']['script_sha256'])
        args.engine_script = engine_path
        engine = load_engine(engine_path, source_record['protocol']['script_sha256'])
        payload, data, base, device, audit_path = prepare(engine, root, source, source_record, args)
        identity = object_hash(payload)
        if args.resume:
            require(read_json(out / 'protocol.json') == dict(identity=identity, **payload)
                    and previous.get('protocol_identity') == identity,
                    'Resume inputs/configuration/code/runtime changed; existing results are preserved')
        if args.check_only:
            print('CHECK_COMPLETE: source/cache/runtime verified. No training or run creation.', flush=True)
            print(json.dumps({'source_run': str(source), 'condition': TARGET,
                              'encoder_training': payload['encoder_training'], 'tst_training': payload['tst_training'],
                              'window_rows': {s: len(x['labels']) for s, x in data.items()},
                              'L10_rows': {s: len(x['sequence_labels']) for s, x in data.items()}}, indent=2))
            return 0
        if not args.resume:
            out.mkdir(parents=True, exist_ok=False)
        with engine.RunLock(out / '.run.lock'):
            identity = guard_protocol(out / 'protocol.json', payload)
            if previous.get('status') == 'DIAGNOSTIC_PILOT_COMPLETE':
                verify_completed(engine, out, previous, payload, identity)
                print('Verified completed Step 44; no retraining: ' + str(out), flush=True)
                return 0
            args.audit_run, args.base_model_dir = audit_path, base
            invocation = {k: str(getattr(args, k)) if getattr(args, k) is not None else None for k in names}
            manifest = dict(previous)
            for key in ('error_type', 'error', 'finished_at_utc', 'output_sha256', 'review_zip'):
                manifest.pop(key, None)
            now = engine.utc()
            manifest.update(script_number=44, protocol_version=PROTOCOL_VERSION, status='RUNNING',
                started_at_utc=previous.get('started_at_utc', now), invocation=invocation,
                script_sha256=payload['script_sha256'], engine_sha256=payload['engine_sha256'], protocol_identity=identity,
                source_run=str(source), source_protocol_identity=source_record['protocol']['identity'],
                source_manifest_sha256=source_record['manifest_sha256'],
                dataset_id=payload['protocol']['source_contract']['dataset_id'], source_contract=payload['protocol']['source_contract'],
                initializer_path=str(base), audit_path=str(audit_path), runtime=payload['runtime'],
                test_data_read=False, formal_experiment=False, conditions=[TARGET],
                encoder_pilot_seed=payload['protocol']['encoder_seed'], readiness='REVIEW_REQUIRED')
            if args.resume:
                manifest['last_resumed_at_utc'] = now
            write_json(out / 'run_manifest.json', manifest)
            write_json(out / 'input_audit.json', payload['source_files'])
            print('Output: ' + str(out), flush=True)
            try:
                trained, temporal, embedding_record = train_one(engine, data, base, device, payload, out, identity)
                summarize(engine, source, out, trained, temporal, embedding_record, payload)
                verify_unchanged(engine, source, source_record, payload, base, engine_path)
            except BaseException as exc:
                manifest.update(status='INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAILED',
                                error_type=type(exc).__name__, error=str(exc))
                (out / 'failure_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
                engine.finish_run(out, manifest)
                print(f'Resume: python "{Path(__file__).name}" --resume "{out}"', flush=True)
                return 130 if isinstance(exc, KeyboardInterrupt) else 1
            (out / 'failure_traceback.txt').unlink(missing_ok=True)
            manifest.update(status='DIAGNOSTIC_PILOT_COMPLETE', conditions_completed=1,
                            validation_population='common L10 endpoints for matched window/TST comparison')
            engine.finish_run(out, manifest)
            return 0
    except Exception as exc:
        print('Step 44 stopped before training: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
