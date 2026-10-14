#!/usr/bin/env python3
"""TAES Step 45: four encoders x five NEW training seeds, train/validation only.

Keep this file beside the delivered revised 42, 43 and 44 scripts, in the same
project and virtual environment. Keep the FULL Step 43 run folder/cache and the
completed revised Step 44 results. Filenames may be changed; code hashes must match. No packages/models are installed or downloaded.

Run (or resume the same experiment with the SAME command):
    py "45.encoder_comparison_formal_trainval.py"
Optional input/cache/environment check without starting a run:
    py "45.encoder_comparison_formal_trainval.py" --check-only

Default output: outputs/shared_nominal_v2/45_encoder_comparison_formal_trainval/run_formal_5seed
Existing completed pipelines are hash-checked and skipped. A partial training
stage resumes from the last fully saved epoch, including optimizer/scheduler/RNG.
An interrupted embedding extraction is repeated; it is not a training restart.
Changing code, inputs, protocol or runtime blocks resume instead of mixing runs.

Formal seeds: 20260610..20260614. Pilot seed 20260928 is excluded. Window head
and encoder shuffle seeds are fixed; TST is newly trained for EACH encoder/seed
with the same fixed head seed. This is 20 pipelines, not 5 x 10 head crossings.
No test windows/predictions/embeddings or pilot weights are used for training.
Upstream manifests/inventory supply declared test counts only. No formal-score
seed selection, hyperparameter search or ensemble. All runs also save single-window
predictions at the SAME L10 endpoints, making population-matched comparisons possible.
Keep the complete output folder. The review ZIP contains small records, not
weights, embeddings or raw logits. Test evaluation/statistics are a later step.
Current shared-nominal labels are retained even for quiet attack windows.
Counterfactual audit annotations remain provenance/diagnostics, never features,
training filters, relabeling rules or model-selection criteria.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path, PureWindowsPath
import shutil
import sys
import traceback
import zipfile

DIAGNOSTIC_VERSION = 'shared_nominal_matched_random_pilot_v3'
PROTOCOL_VERSION = 'shared_nominal_formal_encoder_trainval_v3'
DATA_REL = 'dataset/shared_nominal_v2'
DIAGNOSTIC_REL = 'outputs/shared_nominal_v2/44_random_init_matched_pilot'
OUTPUT_REL = 'outputs/shared_nominal_v2/45_encoder_comparison_formal_trainval'
SEEDS = (20260610, 20260611, 20260612, 20260613, 20260614)
CONDITIONS = ('CategoricalEmbedding_MLP', 'FieldToken_Encoder',
              'DistilBERT_RandomInit_Train', 'DistilBERT_Pretrained_Finetune')


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
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def safe_artifact(folder, name):
    require(isinstance(name, str), 'Invalid artifact name')
    relative = Path(name)
    require(not relative.is_absolute() and not PureWindowsPath(name).drive and
            '..' not in relative.parts and '\\' not in name, 'Unsafe artifact path')
    require(not any(part.lower().startswith('test') for part in relative.parts) and relative.suffix != '.npz',
            'Test artifacts are outside this train/validation stage')
    folder = Path(folder).resolve()
    path = (folder / relative).resolve()
    require(folder in path.parents, 'Unsafe artifact location')
    return path


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def load_adapter(path, expected_sha):
    path = Path(path).resolve()
    require(valid_sha(expected_sha) and path.is_file() and digest(path) == expected_sha,
            'Use the exact 44 Python file recorded by the completed diagnostic, or pass --adapter-script')
    spec = importlib.util.spec_from_file_location('taes_step44_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.PROTOCOL_VERSION == DIAGNOSTIC_VERSION, 'Unexpected Step 44 protocol')
    return module


def read_diagnostic(path):
    """Authenticate current Step 44 small records; never load its trained parameters."""
    path = Path(path).resolve()
    archive = zipfile.ZipFile(path) if path.is_file() else None
    try:
        if archive:
            names = archive.namelist()
            require(len(names) == len(set(names)), 'Duplicate diagnostic ZIP members')
            for item in archive.infolist():
                safe_artifact(Path.cwd(), item.filename)
                require(item.file_size < 50 * 1024**2 and Path(item.filename).suffix in ('.json', '.csv', '.md', '.txt'),
                        'Expected small diagnostic review records only')
            require(archive.testzip() is None, 'Diagnostic ZIP CRC failed')
            read = archive.read
        else:
            require(path.is_dir(), 'Completed revised Step 44 folder or ZIP is missing: ' + str(path))
            read = lambda name: safe_artifact(path, name).read_bytes()
        raw = read('run_manifest.json')
        manifest = json.loads(raw)
        require(manifest.get('status') == 'DIAGNOSTIC_PILOT_COMPLETE' and manifest.get('script_number') == 44
                and manifest.get('protocol_version') == DIAGNOSTIC_VERSION and manifest.get('conditions_completed') == 1,
                'Complete the revised Step 44 first; legacy diagnostics are incompatible')
        require(valid_sha(manifest.get('script_sha256')) and manifest.get('formal_experiment') is False
                and manifest.get('test_data_read') is False, 'Step 44 source/role differs')
        records = manifest.get('output_sha256', {})
        needed = {'protocol.json', 'input_audit.json', 'comparison_summary.json', 'diagnostic_assessment.json',
                  'comparison_summary.csv', CONDITIONS[2] + '/encoder/complete.json', CONDITIONS[2] + '/tst/complete.json',
                  CONDITIONS[2] + '/embeddings/complete.json', CONDITIONS[2] + '/window_common_L10_metrics.json',
                  CONDITIONS[2] + '/val_window_predictions_common_L10.csv', CONDITIONS[2] + '/val_tst_predictions_L10.csv'}
        require(needed <= set(records), 'Incomplete revised Step 44 review inventory')
        for name, expected in records.items():
            safe_artifact(Path.cwd() if archive else path, name)
            require(Path(name).suffix in ('.json', '.csv', '.md', '.txt'), 'Expected small diagnostic record')
            require(hashlib.sha256(read(name)).hexdigest() == expected, 'Step 44 artifact hash mismatch: ' + name)
        protocol = json.loads(read('protocol.json'))
        identity = object_hash({k: v for k, v in protocol.items() if k != 'identity'})
        require(protocol.get('identity') == manifest.get('protocol_identity') == identity
                and protocol.get('script_sha256') == manifest['script_sha256'], 'Step 44 protocol identity mismatch')
        require(protocol['protocol'].get('version') == DIAGNOSTIC_VERSION
                and protocol['protocol'].get('formal') is False and protocol['protocol'].get('test_evaluation') is False,
                'Unexpected diagnostic protocol scope')
        require(manifest.get('source_protocol_identity') == protocol.get('source_protocol_identity') and
                manifest.get('source_manifest_sha256') == protocol.get('source_manifest_sha256'),
                'Diagnostic source identity is inconsistent')
        require(valid_sha(protocol.get('engine_sha256')) and manifest.get('engine_sha256') == protocol['engine_sha256'],
                'Diagnostic engine identity is inconsistent')
        source = protocol['protocol'].get('source_contract', {})
        require(source == manifest.get('source_contract') and source.get('dataset_id') == manifest.get('dataset_id')
                and protocol['protocol'].get('nominal_substrate') == 'shared_nominal_v2',
                'Diagnostic shared-nominal dataset/source contract differs')
        return dict(manifest=manifest, protocol=protocol, manifest_sha256=hashlib.sha256(raw).hexdigest())
    finally:
        if archive:
            archive.close()


def formal_protocols(engine, source):
    """Use the declared matched recipe, never select hyperparameters from formal scores."""
    require(tuple(engine.CONDITIONS) == CONDITIONS, 'Encoder order changed')
    require(source['candidate'] == 'main' and source['encoder_seed'] == 20260928
            and source.get('version') == DIAGNOSTIC_VERSION and source.get('nominal_substrate') == 'shared_nominal_v2',
            'Shared-nominal matched main pilot protocol required')
    require(engine.training_spec(CONDITIONS[2], source) == engine.training_spec(CONDITIONS[3], source),
            'Source random/pretrained training rules must already be matched')
    require(source['tst_model'] == engine.TST_MODEL and source['fields'] == list(engine.FIELDS)
            and source['labels'] == list(engine.LABELS), 'Source architecture/schema differs')
    result = []
    for seed in SEEDS:
        p = copy.deepcopy(source)
        p.update(version=PROTOCOL_VERSION, purpose='FORMAL_FIVE_SEED_ENCODER_TRAIN_VAL', formal=True,
                 test_evaluation=False, conditions=list(CONDITIONS), encoder_seed=seed,
                 window_head_seed=20260929, encoder_shuffle_seed=20260930,
                 tst_seed=20260610, tst_shuffle_seed=20260611,
                 limitations=[
                     '22-field representation and cross-window evidence aggregation are the revision focus',
                     'prespecified finite training rules/maxima; full convergence or equal realized compute is not claimed',
                     'model families differ in capacity and compute; encoder results at L10 are conditional on this TST control',
                     'encoder seed varies initialization where applicable and training dropout; pretrained starting weights are identical',
                     'fixed window/TST head and shuffle seeds; no crossing with the separate ten-head-seed study',
                     'random/pretrained share tokenizer/architecture/training rules; initialization weights are the intended contrast',
                     'window-common and L10 share endpoints, but head capacity/training also differ',
                     'validation selects checkpoints; held-out test evaluation and uncertainty are a later step',
                     'all eligible shared-nominal windows/endpoints are retained, including quiet or unobserved attack windows',
                     'audit annotations remain provenance/diagnostics and never select training rows, labels or checkpoints',
                     'overall F1 alone does not demonstrate attack-evidence use or rule out generator signatures; interpret with benchmark/evidence audits',
                 ])
        p['candidates'] = {'main': copy.deepcopy(source['candidates']['main'])}
        require(engine.training_spec(CONDITIONS[2], p) == engine.training_spec(CONDITIONS[3], p),
                'BERT training rules are not matched')
        require(p['encoder_seed'] != 20260928, 'Pilot seed must not enter formal results')
        result.append(p)
    return result


def job_list(engine, protocols):
    jobs = []
    for p in protocols:
        seed = p['encoder_seed']
        for condition in CONDITIONS:
            jobs.append(dict(key=f'{seed}/{condition}', seed=seed, condition=condition,
                             relative_dir=f'seed_{seed}/{condition}', protocol=p,
                             encoder_training=engine.training_spec(condition, p),
                             tst_training=engine.training_spec(condition, p, tst=True)))
    require(len(jobs) == 20 and len({j['key'] for j in jobs}) == 20, 'Expected 20 unique formal pipelines')
    return jobs


def validate_output(root, out, sources):
    root, out = Path(root).resolve(), Path(out).resolve()
    require(out.parent == (root / OUTPUT_REL).resolve() and out.name.startswith('run_'),
            'Output must be one run directory directly inside ' + OUTPUT_REL)
    for source in sources:
        source = Path(source).resolve()
        require(out != source and out not in source.parents and source not in out.parents,
                'Output overlaps a source; original files are read-only')


def freeze_protocol(path, payload, resume):
    path = Path(path)
    expected = dict(identity=object_hash(payload), **payload)
    if resume:
        require(path.is_file(), 'Resume protocol is missing; do not reconstruct a run by guessing')
        require(read_json(path) == expected, 'Resume code/inputs/protocol/runtime changed; stop instead of mixing runs')
    else:
        require(not path.exists(), 'Protocol already exists; use the existing run')
        write_json(path, expected)
    return expected['identity']


def initializer_path(condition, base):
    require(condition in CONDITIONS, 'Unknown encoder condition')
    return Path(base) / 'model.safetensors' if condition == CONDITIONS[3] else None


def prepare(adapter, engine, root, source, diagnostic, args):
    record = adapter.read_source(source)
    d = read_diagnostic(diagnostic)
    require(d['protocol']['source_manifest_sha256'] == record['manifest_sha256'] and
            d['protocol']['source_protocol_identity'] == record['protocol']['identity'],
            'Step 44 does not refer to this Step 43 run')
    common, data, base, device, audit = adapter.prepare(engine, root, source, record, args)
    for key in ('runtime', 'base_hashes', 'engine_sha256', 'encoder_training', 'tst_training', 'protocol'):
        require(common[key] == d['protocol'][key], 'Current inputs/matched recipe differ from Step 44: ' + key)
    require(set(data) == {'train', 'val'}, 'Formal training accepts train/validation only')
    vocabulary = read_json(source / 'train_field_vocabulary.json')
    require(vocabulary['source'] == 'train_only' and vocabulary['unknown_id'] == 0 and
            set(vocabulary['fields']) == set(engine.FIELDS), 'Unexpected categorical vocabulary policy')
    audited = engine.read_audit(audit)
    require(vocabulary == audited['train_field_vocabulary'], 'Cached vocabulary differs from the verified train-only audit')
    for split, item in data.items():
        categories = engine.np.load(source / 'cache' / f'{split}_categories.npy', mmap_mode='r', allow_pickle=False)
        require(categories.shape == (len(item['labels']), 22) and categories.dtype == engine.np.int64,
                'Categorical cache shape/dtype mismatch')
        for i, field in enumerate(engine.FIELDS):
            require(int(categories[:, i].min()) >= 0 and int(categories[:, i].max()) <= len(vocabulary['fields'][field]),
                    'Cached category outside train vocabulary: ' + field)
        item['categories'] = categories
    protocols = formal_protocols(engine, d['protocol']['protocol'])
    jobs = job_list(engine, protocols)
    # adapter.prepare already establishes and records the Step 43 numerical policy.
    # Model/RNG initialization belongs to each formal job, never this read-only check.
    numeric = dict(cudnn_benchmark=engine.torch.backends.cudnn.benchmark,
                   cudnn_deterministic=engine.torch.backends.cudnn.deterministic,
                   matmul_allow_tf32=engine.torch.backends.cuda.matmul.allow_tf32,
                   cudnn_allow_tf32=engine.torch.backends.cudnn.allow_tf32,
                   torch_num_threads=engine.torch.get_num_threads(),
                   cudnn_version=engine.torch.backends.cudnn.version(),
                   cuda_device_index=engine.torch.cuda.current_device() if device.type == 'cuda' else None,
                   effective_policy=copy.deepcopy(common['runtime']['numerical_policy']))
    # The Step 7 manifest/inventory are already authenticated by 42->43->44.
    # Only their declared counts are used; no held-out windows or predictions are opened.
    build_path = root / DATA_REL / 'build_manifest.json'
    build = read_json(build_path)
    require(build['dataset_id'] == protocols[0]['source_contract']['dataset_id'], 'Step 7 cohort differs')
    inventory_path = root / DATA_REL / 'splits/episode_split_manifest.csv'
    expected_inputs = {Path(x['path']).resolve(): x['sha256'] for x in common['source_files']}
    for path in (build_path, inventory_path):
        require(path.resolve() in expected_inputs and digest(path) == expected_inputs[path.resolve()], 'Step 7 metadata changed')
    inventory = engine.pd.read_csv(inventory_path, encoding='utf-8-sig')
    test_episodes = inventory.loc[inventory['split'] == 'test', 'row_count']
    statistics = statistics_plan(build, int((test_episodes >= 10).sum()))
    require(int(test_episodes.sum()) == statistics['expected_test_windows'] and
            int((test_episodes - 9).clip(lower=0).sum()) == statistics['expected_test_l10'] and
            len(test_episodes) == statistics['expected_test_episodes'], 'Step 7 declared test counts disagree')
    validation = data['val']
    endpoints = validation['sequences'][:, -1]
    endpoint_contract = dict(population='common_L10_endpoints', window_rows=len(validation['labels']),
        common_rows=len(endpoints), eligible_episodes=int(validation['metadata'].episode_id.nunique()),
        end_row_indices_sha256=hashlib.sha256(engine.np.asarray(endpoints, dtype='<i8').tobytes()).hexdigest(),
        metadata_sha256=object_hash(validation['metadata'].to_dict(orient='records')),
        labels_sha256=hashlib.sha256(engine.np.asarray(validation['sequence_labels'], dtype='<i8').tobytes()).hexdigest())
    payload = dict(version=PROTOCOL_VERSION, purpose='FORMAL_ENCODER_COMPARISON_TRAIN_VAL_ONLY',
                   formal=True, test_evaluation=False, conditions=list(CONDITIONS), seeds=list(SEEDS),
                   planned_pipelines=20, excluded_pilot_seed=20260928, protocols=protocols, jobs=jobs,
                   statistics_plan=statistics, runtime=common['runtime'], numerical_runtime=numeric,
                   script_sha256=digest(Path(__file__)), adapter_sha256=d['protocol']['script_sha256'],
                   engine_sha256=common['engine_sha256'], source_files=common['source_files'],
                   audit_manifest_sha256=record['protocol']['audit_manifest_sha256'], base_hashes=common['base_hashes'],
                   dataset_id=build['dataset_id'], source_contract=protocols[0]['source_contract'],
                   validation_endpoint_contract=endpoint_contract,
                   source43=dict(path=str(source), identity=record['protocol']['identity'], manifest_sha256=record['manifest_sha256']),
                   source44=dict(path=str(diagnostic), identity=d['protocol']['identity'], manifest_sha256=d['manifest_sha256']),
                   source_cache=read_json(source / 'cache/complete.json'),
                   train_vocabulary_sha256=digest(source / 'train_field_vocabulary.json'),
                   selection_policy='recipe declared before formal jobs; retain every listed seed; no formal-score tuning or ensembling')
    return payload, data, base, device, audit, vocabulary


def write_predictions(engine, path, metadata, labels, logits, complete, identity, job, head_kind):
    np = engine.np
    logits, labels = np.asarray(logits), np.asarray(labels)
    require(logits.shape == (len(metadata), 6) and labels.shape == (len(metadata),),
            'Prediction shape differs from declared evaluation endpoints')
    require(np.isfinite(logits).all(), 'Non-finite logits')
    require(np.array_equal(metadata.label.map(engine.LABEL_TO_ID).to_numpy(), labels),
            'Prediction labels are not aligned with endpoints')
    measured = engine.metrics(labels, logits, complete['best_metrics']['loss'])
    require(measured['confusion_matrix'] == complete['best_metrics']['confusion_matrix'] and
            abs(measured['macro_f1'] - complete['best_metrics']['macro_f1']) < 1e-12,
            'Selected checkpoint predictions differ from its reported metrics')
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    require(np.allclose(probabilities.sum(axis=1), 1., rtol=0., atol=1e-6), 'Invalid probability sums')
    frame = metadata.copy()
    frame['y_true'], frame['y_pred'] = labels, logits.argmax(axis=1)
    for k in range(6):
        frame[f'prob_{k}'] = probabilities[:, k]
    require(head_kind in ('window', 'tst'), 'Unknown prediction head')
    frame['dataset_id'] = job['protocol']['source_contract']['dataset_id']
    frame['step45_protocol_identity'] = identity
    frame['condition'] = job['condition']
    frame['encoder_seed'] = job['seed']
    frame['head_seed'] = job['protocol']['window_head_seed'] if head_kind == 'window' else job['protocol']['tst_seed']
    frame['head_kind'] = head_kind
    frame['evaluation_population'] = 'common_L10_endpoints'
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    frame.to_csv(temporary, index=False, encoding='utf-8-sig')
    temporary.replace(path)


def pipeline_folder(out, job):
    return safe_artifact(out, job['relative_dir'])


def train_pipeline(engine, data, base, vocabulary, job, out, identity, device):
    """Same model constructors and fit/extraction functions as reviewed Step 43."""
    condition, protocol = job['condition'], job['protocol']
    folder = pipeline_folder(out, job)
    cfg = job['encoder_training']
    engine.seed_all(job['seed'])
    sizes = [len(vocabulary['fields'][f]) for f in engine.FIELDS]
    if condition == CONDITIONS[0]:
        encoder = engine.CategoricalEncoder(sizes)
    elif condition == CONDITIONS[1]:
        encoder = engine.FieldTokenEncoder(sizes)
    else:
        from transformers import DistilBertConfig
        config = DistilBertConfig.from_pretrained(str(base), local_files_only=True)
        config._attn_implementation = 'eager'
        encoder = engine.BertEncoder(config, weights_path=initializer_path(condition, base))
    encoder_parameters = sum(p.numel() for p in encoder.parameters())
    model = engine.WindowModel(encoder, head_seed=protocol['window_head_seed'])
    bert = condition in CONDITIONS[2:]
    batches = {s: engine.ArrayBatches(
        {'input_ids': x['input_ids'], 'attention_mask': x['attention_mask']} if bert else {'x': x['categories']},
        x['labels']) for s, x in data.items()}
    stage = object_hash(dict(run=identity, seed=job['seed'], condition=condition, stage='encoder'))
    trained = engine.fit(model, batches['train'], batches['val'], cfg, folder / 'encoder', stage, device)
    window_logits = engine.np.load(folder / 'encoder/val_logits.npy', allow_pickle=False)
    require(window_logits.shape == (len(data['val']['labels']), 6), 'All-window prediction count differs')
    common_logits = window_logits[data['val']['sequences'][:, -1]]
    common_y = data['val']['sequence_labels']
    loss = float(engine.nn.functional.cross_entropy(engine.torch.from_numpy(common_logits), engine.torch.from_numpy(common_y)))
    common = engine.metrics(common_y, common_logits, loss)
    write_json(folder / 'window_common_L10_metrics.json', common)
    write_predictions(engine, folder / 'val_window_predictions_common_L10.csv', data['val']['metadata'],
                      common_y, common_logits, {'best_metrics': common}, identity, job, 'window')
    feature_identity = object_hash(dict(stage=stage, checkpoint=trained['artifact_sha256']['best.pt']))
    embeddings = engine.extract_features(model, batches, folder / 'embeddings',
                                          cfg['eval_batch_size'], device, feature_identity)
    del model, encoder, batches
    if device.type == 'cuda':
        engine.torch.cuda.empty_cache()
    embedding_record = read_json(folder / 'embeddings' / 'complete.json')
    temporal_batches = {s: engine.SequenceBatches(embeddings[s], x['sequences'], x['sequence_labels'])
                        for s, x in data.items()}
    engine.seed_all(protocol['tst_seed'])
    tst = engine.TST()
    temporal_id = object_hash(dict(run=identity, seed=job['seed'], condition=condition,
                                   embeddings=embedding_record['sha256']))
    temporal = engine.fit(tst, temporal_batches['train'], temporal_batches['val'], job['tst_training'],
                          folder / 'tst', temporal_id, device)
    logits = engine.np.load(folder / 'tst' / 'val_logits.npy', allow_pickle=False)
    write_predictions(engine, folder / 'val_tst_predictions_L10.csv', data['val']['metadata'],
                      data['val']['sequence_labels'], logits, temporal, identity, job, 'tst')
    require(logits.shape == common_logits.shape, 'Window/TST endpoint counts differ')
    row = dict(seed=job['seed'], condition=condition,
               dataset_id=protocol['source_contract']['dataset_id'], step45_protocol_identity=identity,
               window_head_seed=protocol['window_head_seed'], tst_seed=protocol['tst_seed'],
               evaluation_precision='float32', extraction_precision='float32',
               window_val_count=trained['best_metrics']['n'], common_val_count=common['n'],
               l10_val_count=temporal['best_metrics']['n'], window_val_common_macro_f1=common['macro_f1'],
               temporal_delta_val_common_macro_f1=temporal['best_metrics']['macro_f1']-common['macro_f1'],
               encoder_optimizer_updates=trained['optimizer_updates'], tst_optimizer_updates=temporal['optimizer_updates'],
               encoder_parameters=encoder_parameters, window_model_parameters=trained['parameters'],
               tst_parameters=temporal['parameters'], deployment_parameters=encoder_parameters + temporal['parameters'],
               window_val_macro_f1=trained['best_metrics']['macro_f1'],
               l10_val_macro_f1=temporal['best_metrics']['macro_f1'],
               encoder_best_epoch=trained['best_epoch'], encoder_epochs=trained['epochs_completed'],
               encoder_convergence=trained['convergence'], tst_best_epoch=temporal['best_epoch'],
               tst_epochs=temporal['epochs_completed'], tst_convergence=temporal['convergence'],
               encoder_training_seconds=trained['training_seconds'],
               extraction_seconds=embedding_record['elapsed_seconds'],
               tst_training_seconds=temporal['training_seconds'],
               peak_gpu_bytes=max(trained['peak_gpu_bytes'], temporal['peak_gpu_bytes']))
    del tst, temporal_batches, embeddings, logits
    if device.type == 'cuda':
        engine.torch.cuda.empty_cache()
    return row


def job_contract(identity, job):
    return object_hash(dict(run=identity, job=job))


def load_completed(out, job, identity):
    folder = pipeline_folder(out, job)
    path = folder / 'pipeline_complete.json'
    if not path.exists():
        return None
    record = read_json(path)
    body = {k: v for k, v in record.items() if k != 'identity'}
    require(record.get('identity') == object_hash(body) and record['contract'] == job_contract(identity, job),
            'Completed pipeline identity/configuration changed: ' + job['key'])
    require(record['summary']['seed'] == job['seed'] and record['summary']['condition'] == job['condition'],
            'Completed pipeline summary belongs to a different job')
    require(record['summary'].get('dataset_id') == job['protocol']['source_contract']['dataset_id']
            and record['summary'].get('step45_protocol_identity') == identity,
            'Completed pipeline belongs to another dataset/run')
    required = {'encoder/complete.json', 'encoder/best.pt', 'encoder/last.pt', 'encoder/history.csv', 'encoder/val_logits.npy',
                'tst/complete.json', 'tst/best.pt', 'tst/last.pt', 'tst/history.csv', 'tst/val_logits.npy',
                'embeddings/complete.json', 'embeddings/train.npy', 'embeddings/val.npy',
                'val_tst_predictions_L10.csv', 'val_window_predictions_common_L10.csv', 'window_common_L10_metrics.json'}
    require(required <= set(record.get('artifact_sha256', {})), 'Completed pipeline inventory is incomplete')
    for name, expected in record['artifact_sha256'].items():
        artifact = safe_artifact(folder, name)
        require(artifact.is_file() and digest(artifact) == expected,
                'Completed pipeline artifact changed or missing: ' + str(artifact))
    return record['summary']


def aggregate_validation(engine, rows):
    keys = [(r['seed'], r['condition']) for r in rows]
    require(len(keys) == len(set(keys)), 'Duplicate condition/seed in summary')
    require(all(seed in SEEDS and condition in CONDITIONS for seed, condition in keys), 'Unexpected formal row')
    result = []
    for condition in CONDITIONS:
        selected = [r for r in rows if r['condition'] == condition]
        if not selected:
            continue
        row = dict(condition=condition, completed_seeds=len(selected), planned_seeds=5,
                   all_five_complete=len(selected) == 5)
        for metric in ('window_val_macro_f1', 'window_val_common_macro_f1', 'l10_val_macro_f1', 'temporal_delta_val_common_macro_f1'):
            values = engine.np.asarray([r[metric] for r in selected], dtype=float)
            require(engine.np.isfinite(values).all(), 'Non-finite validation score')
            row[metric + '_mean'] = float(values.mean())
            row[metric + '_sd'] = float(values.std(ddof=1)) if len(values) > 1 else None
        result.append(row)
    return result


def update_progress(engine, out, jobs, rows, current):
    completed = {(r['seed'], r['condition']) for r in rows}
    pending = [j['key'] for j in jobs if (j['seed'], j['condition']) not in completed]
    write_json(out / 'progress.json', dict(updated_at_utc=engine.utc(),
               completed_pipelines=len(rows), planned_pipelines=len(jobs), current_pipeline=current,
               pending_pipelines=pending, test_evaluation=False))
    if rows:
        engine.write_csv(out / 'formal_validation_runs.csv', rows)
        write_json(out / 'formal_validation_runs.json', rows)
        aggregate = aggregate_validation(engine, rows)
        engine.write_csv(out / 'formal_validation_summary.csv', aggregate)
        write_json(out / 'formal_validation_summary.json', aggregate)


def run_queue(engine, data, base, vocabulary, jobs, out, identity, device, verified_rows=None):
    # Validate EVERY existing completed job before starting any further training.
    rows = list(verified_rows) if verified_rows is not None else verified_pipelines(out, jobs, identity)
    completed = {f"{r['seed']}/{r['condition']}" for r in rows}
    update_progress(engine, out, jobs, rows, None)
    for index, job in enumerate(jobs, 1):
        if job['key'] in completed:
            print(f'[{index}/{len(jobs)}] Verified complete; skip {job["key"]}', flush=True)
            continue
        update_progress(engine, out, jobs, rows, job['key'])
        print(f'\n[{index}/{len(jobs)}] {job["key"]} | {len(rows)} pipelines complete', flush=True)
        summary = train_pipeline(engine, data, base, vocabulary, job, out, identity, device)
        folder = pipeline_folder(out, job)
        artifacts = {}
        for artifact in sorted(folder.rglob('*')):
            if artifact.is_file() and artifact.name != 'pipeline_complete.json' and artifact.suffix != '.tmp':
                relative = artifact.relative_to(folder).as_posix()
                safe_artifact(folder, relative)
                artifacts[relative] = digest(artifact)
        require(bool(artifacts), 'No completed pipeline artifacts')
        record = dict(contract=job_contract(identity, job), summary=summary,
                      artifact_sha256=artifacts, completed_at_utc=engine.utc())
        write_json(folder / 'pipeline_complete.json', dict(identity=object_hash(record), **record))
        rows.append(summary)
        rows.sort(key=lambda r: (r['seed'], CONDITIONS.index(r['condition'])))
        update_progress(engine, out, jobs, rows, None)
        print(f'Completed {len(rows)}/{len(jobs)} | L10 validation F1={summary["l10_val_macro_f1"]:.6f}', flush=True)
    return rows


def write_overview(engine, out, rows):
    aggregate = aggregate_validation(engine, rows)
    lines = ['# Step 45 formal encoder training', '',
             'TRAIN/VALIDATION ONLY. Test evaluation is pending.', '',
             'Five encoder-training seeds per condition; each selected encoder has a newly trained L10 TST.',
             'TST/window-head/shuffle seeds are fixed. Pilot runs are excluded. No seed selection or ensemble.', '',
             '| Encoder | Seeds | All-window val mean | Common-window val mean | L10 val mean | L10 seed SD |',
             '|---|---:|---:|---:|---:|---:|']
    for r in aggregate:
        sd = r['l10_val_macro_f1_sd']
        sd_text = 'n.a.' if sd is None else f'{sd:.6f}'
        lines.append(f'| {r["condition"]} | {r["completed_seeds"]}/5 | {r["window_val_macro_f1_mean"]:.6f} | '
                     f'{r["window_val_common_macro_f1_mean"]:.6f} | {r["l10_val_macro_f1_mean"]:.6f} | {sd_text} |')
    lines += ['', 'Prespecified finite budgets; no full-convergence, equal-compute, or equivalence claim.',
              'Common-window and L10 scores share identical endpoints; the all-window metric selects the encoder checkpoint.',
              'L10 encoder rankings are conditional on the fixed TST control; window/TST heads also differ in capacity and training.',
              'Recorded timings are training/extraction times, not deployment inference latency.',
              'Do not tune settings or pick seeds using these formal outcomes.',
              'Retain the complete folder: review ZIP excludes checkpoints, embeddings and raw logits.']
    (out / 'analysis_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def reusable_stage_bytes(out, job, data, identity):
    """Credit only fully verified reusable stages; never deserialize pilot/model weights."""
    if identity is None:
        return 0
    folder = pipeline_folder(out, job)
    encoder_id = object_hash(dict(run=identity, seed=job['seed'], condition=job['condition'], stage='encoder'))
    verified_files = set()

    def check_files(directory, hashes):
        for name, expected in hashes.items():
            path = safe_artifact(directory, name)
            require(path.is_file() and digest(path) == expected, 'Reusable stage artifact changed: ' + str(path))
            verified_files.add(path)

    def completed_stage(stage, stage_id, cfg):
        directory = folder / stage
        path = directory / 'complete.json'
        if not path.is_file():
            return None
        record = read_json(path)
        require(record.get('contract') == object_hash(dict(identity=stage_id, config=cfg)),
                'Reusable training stage belongs to a different job/configuration: ' + str(path))
        hashes = record.get('artifact_sha256', {})
        require(set(hashes) == {'best.pt', 'last.pt', 'history.csv', 'val_logits.npy'},
                'Reusable training stage inventory is incomplete')
        check_files(directory, hashes)
        verified_files.add(path)
        return record

    encoder = completed_stage('encoder', encoder_id, job['encoder_training'])
    embedding_path = folder / 'embeddings/complete.json'
    embeddings = None
    if embedding_path.is_file():
        require(encoder is not None, 'Reusable embeddings require a completed encoder')
        embeddings = read_json(embedding_path)
        expected_id = object_hash(dict(stage=encoder_id, checkpoint=encoder['artifact_sha256']['best.pt']))
        require(embeddings.get('identity') == expected_id and embeddings.get('precision') == 'float32'
                and embeddings.get('rows') == {s:len(item['labels']) for s,item in data.items()}
                and set(embeddings.get('sha256', {})) == {'train', 'val'}, 'Reusable embedding contract differs')
        check_files(embedding_path.parent, {s+'.npy':h for s,h in embeddings['sha256'].items()})
        verified_files.add(embedding_path)
    if (folder / 'tst/complete.json').is_file():
        require(embeddings is not None, 'Reusable TST requires completed encoder embeddings')
        temporal_id = object_hash(dict(run=identity, seed=job['seed'], condition=job['condition'],
                                       embeddings=embeddings['sha256']))
        completed_stage('tst', temporal_id, job['tst_training'])
    # Sparse/incomplete files must not receive credit for unallocated logical bytes.
    def allocated_bytes(path):
        info = path.stat()
        return min(info.st_size, info.st_blocks * 512) if hasattr(info, 'st_blocks') else info.st_size
    return sum(allocated_bytes(path) for path in verified_files)


def required_free_bytes(jobs, out, data, identity=None):
    # Retained embeddings follow the CURRENT cohort, not historical A-prime counts.
    # Checkpoint allowances cover FP32 model/best/AdamW states, with one atomic-save temporary.
    pending = [j for j in jobs if not (pipeline_folder(out, j) / 'pipeline_complete.json').is_file()]
    if not pending:
        return 0
    embedding_bytes = sum(len(item['labels']) for item in data.values()) * 768 * 4
    # Interrupted epochs without a stage completion stay conservatively budgeted in full.
    retained = sum(max(0, embedding_bytes + (1.65e9 if j['condition'] in CONDITIONS[2:] else .10e9)
                       - reusable_stage_bytes(out, j, data, identity)) for j in pending)
    return int(retained + (1.35e9 if any(j['condition'] in CONDITIONS[2:] for j in pending) else .15e9) + .5 * 1024**3)


def resolve_adapter(root, explicit=None, expected_sha=None):
    require(valid_sha(expected_sha), 'The completed Step 44 diagnostic must declare its code hash')
    paths = [Path(root) / explicit] if explicit else sorted(Path(root).glob('44*.py'))
    matches = [p.resolve() for p in paths if p.is_file() and digest(p) == expected_sha]
    require(matches, 'Put the exact 44 Python file used for the diagnostic beside 45, or pass --adapter-script')
    return matches[0]


def resolve_diagnostic(root, explicit=None, source_identity=None):
    if explicit is not None:
        return (Path(root) / explicit).resolve()
    build = read_json(Path(root) / DATA_REL / 'build_manifest.json')
    candidates = []
    for path in sorted((Path(root) / DIAGNOSTIC_REL).glob('run_*/run_manifest.json')):
        m = read_json(path)
        if (m.get('script_number') == 44 and m.get('status') == 'DIAGNOSTIC_PILOT_COMPLETE'
                and m.get('protocol_version') == DIAGNOSTIC_VERSION and valid_sha(m.get('script_sha256'))
                and m.get('dataset_id') == build.get('dataset_id')
                and (source_identity is None or m.get('source_protocol_identity') == source_identity)):
            candidates.append(path.parent)
    require(candidates, 'No completed revised Step 44 for this cohort/source. Finish 44 or specify --diagnostic-run (folder/ZIP)')
    require(len(candidates) == 1,
            'Multiple eligible Step 44 runs; specify --diagnostic-run. No latest/best-score selection is made.\n'
            + '\n'.join(map(str, candidates)))
    return candidates[0].resolve()


def statistics_plan(build, eligible_test_episodes):
    counts = build['counts']['test']
    require(build.get('status') == 'complete', 'Incomplete Step 7 build')
    return dict(stage='future_test_evaluation_not_executed_here',
        primary_endpoint='macro_f1_at_fixed_L10_test_endpoints', dataset_id=build['dataset_id'],
        primary_population='all eligible L10 endpoints with supplied episode labels, including quiet attack windows; no evidence-based exclusion or relabeling',
        observability_interpretation='counterfactual annotations are diagnostic only; aggregate scores do not establish attack-evidence use or an information-theoretic ceiling',
        expected_test_windows=counts['rows'], expected_test_l10=counts['sequences_stride1']['10'],
        expected_test_episodes=counts['episodes'], expected_test_l10_episodes=eligible_test_episodes,
        test_count_source='authenticated Step 7 manifest/episode inventory; no test windows opened; evaluator must independently verify',
        point_estimate='arithmetic mean of per-seed macro-F1; no probability ensemble',
        seed_variation='sample SD across five encoder-training seeds',
        contrasts=[CONDITIONS[3] + ' minus ' + c for c in CONDITIONS[:3]],
        primary_contrast_scope='encoder comparisons at L10 conditional on the same TST architecture/training recipe',
        bootstrap_unit='class-stratified whole eligible episode, paired across all models and seeds',
        bootstrap_draws=20000, marginal_confidence=.95,
        three_contrast_bonferroni_confidence=1-.05/3,
        secondary_within_encoder_temporal_contrasts=[c + ': L10 TST minus window classifier at common L10 endpoints' for c in CONDITIONS],
        secondary_four_contrast_bonferroni_confidence=1-.05/4,
        multiplicity_scope='three primary encoder contrasts and four explicitly secondary within-encoder contrasts are separate families; no joint seven-contrast coverage claim',
        temporal_interpretation='same endpoint population, but differing head capacity/training; use pooling/LSTM controls for aggregator interpretation',
        uncertainty_scope='episode CI conditional on the trained seeds; report encoder-seed SD separately; fixed TST seed',
        equivalence_claim=False, test_used_for_selection=False)


def verified_pipelines(out, jobs, identity):
    rows = []
    for job in jobs:
        summary = load_completed(out, job, identity)
        if summary is not None:
            rows.append(summary)
    return rows


def verify_final_record(out, manifest, rows):
    require(manifest.get('status') == 'FORMAL_TRAIN_VAL_COMPLETE' and manifest.get('completed_pipelines') == 20
            and manifest.get('completed_encoder_seeds') == 5 and len(rows) == 20,
            'Formal completion does not cover all 20 pipelines')
    required = {'protocol.json', 'input_audit.json', 'train_field_vocabulary.json', 'formal_statistics_plan.json',
                'source_cache_reference.json', 'formal_validation_runs.json', 'formal_validation_runs.csv',
                'formal_validation_summary.json', 'formal_validation_summary.csv', 'progress.json', 'analysis_summary.md'}
    require(required <= set(manifest.get('output_sha256', {})), 'Formal output inventory is incomplete')
    for name, expected in manifest['output_sha256'].items():
        require(digest(safe_artifact(out, name)) == expected, 'Completed formal result changed: ' + name)


def verify_sources(adapter, engine, payload, source, diagnostic, base, args):
    for item in payload['source_files']:
        engine.checked_file(item['path'], item['sha256'], [])
    for name, expected in payload['base_hashes'].items():
        # Hugging Face snapshots can link to their sibling blobs folder.
        require(Path(name).name == name and '\\' not in name and ':' not in name, 'Unsafe initializer filename')
        require(digest(base / name) == expected, 'Base initializer changed during formal training')
    require(adapter.read_source(source)['manifest_sha256'] == payload['source43']['manifest_sha256'],
            'Source Step 43 changed during formal training')
    require(read_diagnostic(diagnostic)['manifest_sha256'] == payload['source44']['manifest_sha256'],
            'Source Step 44 changed during formal training')
    for name, expected in payload['source_cache']['artifact_sha256'].items():
        require(digest(safe_artifact(source / 'cache', name)) == expected, 'Source cache changed during formal training')
    require(digest(source / 'train_field_vocabulary.json') == payload['train_vocabulary_sha256'], 'Vocabulary changed')
    require(digest(Path(__file__)) == payload['script_sha256'] and digest(args.adapter_script) == payload['adapter_sha256']
            and digest(args.engine_script) == payload['engine_sha256'], 'Source code changed during formal training')


def require_initialization_only(out):
    """Recover a interrupted initial write only when no training/result files exist.

    Any complete protocol must still match fresh authenticated inputs before use.
    Atomic-write temporary files contain no trained results and can be overwritten
    by the normal writer after acquiring the run lock.
    """
    out = Path(out)
    if not out.exists():
        return
    require(out.is_dir(), 'Output is not a directory')
    allowed = {'.run.lock', 'protocol.json', 'protocol.json.tmp', 'run_manifest.json.tmp'}
    require(all(path.is_file() and path.name in allowed for path in out.iterdir()),
            'Output has results without a run manifest; preserve it and inspect before continuing')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root', type=Path, help='Default: script directory; inherited on resume')
    parser.add_argument('--source-run', type=Path, help='Full revised Step 43 folder; defaults to the source recorded by 44')
    parser.add_argument('--diagnostic-run', type=Path, help='Completed revised Step 44 folder/ZIP; auto-select only if unique')
    parser.add_argument('--adapter-script', type=Path, help='Revised 44 Python file; otherwise discover 44*.py by content hash')
    parser.add_argument('--engine-script', type=Path, help='Revised 43 Python file; otherwise discover 43*.py by content hash')
    parser.add_argument('--audit-run', type=Path, help='Same revised Step 42 folder/ZIP used by the source pilots')
    parser.add_argument('--audit-script', type=Path, help='Exact revised 42 Python file; otherwise discover by content hash')
    parser.add_argument('--base-model-dir', type=Path, help='Same official local base initializer used by the source pilots')
    parser.add_argument('--resume', type=Path, help='Explicit revised 45 run; default run resumes automatically')
    parser.add_argument('--check-only', action='store_true', help='Read-only input/cache/environment/disk check; no training')
    args = parser.parse_args(argv)
    out = None
    names = ('project_root', 'source_run', 'diagnostic_run', 'adapter_script', 'engine_script',
             'audit_run', 'audit_script', 'base_model_dir')
    try:
        root = (args.project_root or Path(__file__).resolve().parent).resolve()
        out = (root / (args.resume or Path(OUTPUT_REL) / 'run_formal_5seed')).resolve()
        previous = {}
        if (out / 'run_manifest.json').is_file():
            previous = read_json(out / 'run_manifest.json')
            require(previous.get('script_number') == 45 and previous.get('protocol_version') == PROTOCOL_VERSION,
                    'Existing output is not a revised Step 45 run; keep legacy results separate')
            for key in names:
                value = previous.get('invocation', {}).get(key)
                if getattr(args, key) is None and value is not None:
                    setattr(args, key, Path(value))
            root = (args.project_root or root).resolve()
        else:
            require(args.resume is None or out.is_dir(), 'Explicit resume directory does not exist')
            require_initialization_only(out)
        require(root.is_dir(), 'Project root does not exist')
        args.project_root = root
        for key in names[1:]:
            if getattr(args, key) is not None:
                setattr(args, key, (root / getattr(args, key)).resolve())
        # Resolve data lineage before importing the exact scripts recorded by that lineage.
        # This identity is only a discovery filter; adapter.read_source authenticates it below.
        source_identity = read_json(args.source_run / 'protocol.json')['identity'] if args.source_run else None
        diagnostic = resolve_diagnostic(root, args.diagnostic_run, source_identity)
        d = read_diagnostic(diagnostic)
        args.adapter_script = resolve_adapter(root, args.adapter_script, d['protocol']['script_sha256'])
        adapter = load_adapter(args.adapter_script, d['protocol']['script_sha256'])
        args.engine_script = adapter.resolve_engine(root, args.engine_script, d['protocol']['engine_sha256'])
        engine = adapter.load_engine(args.engine_script, d['protocol']['engine_sha256'])
        if args.source_run is None:
            recorded = d['protocol']['source_run']
            args.source_run = Path(recorded) if Path(recorded).is_dir() else engine.mapped_path(recorded, root)
        source = args.source_run.resolve()
        args.source_run, args.diagnostic_run = source, diagnostic
        validate_output(root, out, [source, diagnostic, root / 'dataset'])
        print('Step 45: validating revised inputs and declared pilot recipe. No test windows are read.', flush=True)
        payload, data, base, device, audit, vocabulary = prepare(adapter, engine, root, source, diagnostic, args)
        validate_output(root, out, [base])
        args.audit_run, args.base_model_dir = audit, base
        jobs = payload['jobs']
        identity = object_hash(payload)
        if previous:
            identity = freeze_protocol(out / 'protocol.json', payload, resume=True)
            require(previous['protocol_identity'] == identity, 'Run manifest/protocol mismatch')
        elif (out / 'protocol.json').is_file():
            identity = freeze_protocol(out / 'protocol.json', payload, resume=True)
        needed, free = required_free_bytes(jobs, out, data, identity), shutil.disk_usage(root).free
        print(f'Disk: {free / 1024**3:.1f} GiB free; conservative remaining estimate {needed / 1024**3:.1f} GiB.', flush=True)
        require(free >= needed, 'Insufficient disk space for retained checkpoints/embeddings; preserve source runs')
        if args.check_only:
            rows = verified_pipelines(out, jobs, identity) if previous else []
            if previous.get('status') == 'FORMAL_TRAIN_VAL_COMPLETE':
                verify_final_record(out, previous, rows)
            print('CHECK_COMPLETE: 4 encoders x 5 seeds; no training or new run created.', flush=True)
            print(json.dumps(dict(seeds=list(SEEDS), pipelines=len(jobs), verified_completed_pipelines=len(rows),
                output=str(out), window_rows={s: len(x['labels']) for s, x in data.items()},
                l10_rows={s: len(x['sequence_labels']) for s, x in data.items()}), indent=2))
            return 0
        out.mkdir(parents=True, exist_ok=True)
        with engine.RunLock(out / '.run.lock'):
            if previous:
                require(read_json(out / 'run_manifest.json') == previous, 'Run changed during preparation; invoke again')
            else:
                require(not (out / 'run_manifest.json').exists(),
                        'Another process created this run; invoke the same command again to resume')
                require_initialization_only(out)
            identity = freeze_protocol(out / 'protocol.json', payload,
                                       resume=bool(previous) or (out / 'protocol.json').is_file())
            # Check every existing completed pipeline BEFORE rewriting any run status/progress.
            verified = verified_pipelines(out, jobs, identity)
            if previous.get('status') == 'FORMAL_TRAIN_VAL_COMPLETE':
                verify_final_record(out, previous, verified)
                print('Verified all 20 completed pipelines; no training or result rewrite: ' + str(out), flush=True)
                return 0
            invocation = {k: str(getattr(args, k)) if getattr(args, k) is not None else None for k in names}
            manifest = dict(previous)
            for key in ('error_type', 'error', 'finished_at_utc', 'output_sha256', 'review_zip'):
                manifest.pop(key, None)
            manifest.update(script_number=45, protocol_version=PROTOCOL_VERSION, status='RUNNING',
                started_at_utc=previous.get('started_at_utc', engine.utc()), invocation=invocation,
                script_sha256=payload['script_sha256'], engine_sha256=payload['engine_sha256'], adapter_sha256=payload['adapter_sha256'],
                protocol_identity=identity, runtime=payload['runtime'], numerical_runtime=payload['numerical_runtime'],
                initializer_path=str(base), audit_path=str(audit), source43=payload['source43'], source44=payload['source44'],
                dataset_id=payload['dataset_id'], source_contract=payload['source_contract'],
                validation_endpoint_contract=payload['validation_endpoint_contract'], formal_experiment=True,
                test_data_read=False, test_evaluation=False, conditions=list(CONDITIONS), encoder_seeds=list(SEEDS),
                planned_pipelines=20, readiness='TEST_EVALUATION_PENDING')
            if previous:
                manifest['last_resumed_at_utc'] = engine.utc()
            write_json(out / 'run_manifest.json', manifest)
            print('Output: ' + str(out), flush=True)
            print('Interruption: use the same command; completed pipelines are verified and skipped.', flush=True)
            try:
                write_json(out / 'input_audit.json', payload['source_files'])
                write_json(out / 'train_field_vocabulary.json', vocabulary)
                write_json(out / 'formal_statistics_plan.json', payload['statistics_plan'])
                write_json(out / 'source_cache_reference.json', dict(path=str(source / 'cache'), **payload['source_cache']))
                rows = run_queue(engine, data, base, vocabulary, jobs, out, identity, device, verified_rows=verified)
                require(len(rows) == 20 and all(x['all_five_complete'] for x in aggregate_validation(engine, rows)),
                        'Incomplete formal experiment; a subset is never complete')
                write_overview(engine, out, rows)
                verify_sources(adapter, engine, payload, source, diagnostic, base, args)
            except BaseException as exc:
                manifest.update(status='INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAILED',
                                error_type=type(exc).__name__, error=str(exc))
                if (out / 'progress.json').is_file():
                    manifest['completed_pipelines'] = read_json(out / 'progress.json')['completed_pipelines']
                (out / 'failure_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
                engine.finish_run(out, manifest)
                print(f'Resume: python "{Path(__file__).name}" --resume "{out}"', flush=True)
                return 130 if isinstance(exc, KeyboardInterrupt) else 1
            (out / 'failure_traceback.txt').unlink(missing_ok=True)
            manifest.update(status='FORMAL_TRAIN_VAL_COMPLETE', completed_pipelines=20, completed_encoder_seeds=5)
            engine.finish_run(out, manifest)
            return 0
    except Exception as exc:
        print('Step 45 stopped: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
