#!/usr/bin/env python3
"""TAES Step 46: held-out evaluation of the revised four-encoder/five-seed study.

Keep beside the delivered revised 42/43/44/45 scripts; filenames may be renamed.
Run/resume: python "46.encoder_comparison_frozen_test.py"
Read-only preflight: python "46.encoder_comparison_frozen_test.py" --check-only
Reuse completed predictions: python "46.encoder_comparison_frozen_test.py" --statistics-only

No training, downloads, threshold tuning, hyperparameter changes or seed selection.
All 20 validation-selected encoder/TST pairs come from the completed revised 45.
Inputs, raw CLS representation, label order and token budget follow revised 7-9.
Original data and training results are read-only. Old experiments remain separate.

Default output: outputs/shared_nominal_v2/46_encoder_comparison_frozen_test/run_formal_5seed
Keep the full output folder. Review ZIP includes both common-endpoint prediction
sets/statistics; it excludes embeddings, raw logits and all-window predictions.

Before accepting each test pipeline, VALIDATION reference batches check restored
window logits, FP32 features and TST logits against 45. Up to approximately 512
rows/sequences are used, rounded UP to complete original evaluation batches.
Both window/TST inference and feature extraction are FP32, matching revised 45;
AMP in training is not an evaluation policy. A mismatch stops, never changes tolerance.

All test counts/hashes are obtained from the declared 45/7 cohort and verified.
L10-eligible episodes can be fewer than all raw test episodes. All models/seeds
use identical L10 endpoints. Point estimate: mean of five per-seed six-class
macro-F1 values; sample seed SD is reported separately. No probability ensemble.
Paired, class-stratified whole-episode bootstrap: 20,000 draws, PCG64 seed 20261001,
batch 100, linear percentile intervals. Three PRIMARY L10 encoder contrasts
(pretrained minus alternatives): 95% and 98.3333% Bonferroni intervals. Four
SECONDARY within-encoder contrasts (TST minus common-window): 95% and 98.75%.
These are separate comparison families, not simultaneous seven-contrast coverage.
The study concerns representation and evidence aggregation; no model necessity,
full-convergence, equal-compute or equivalence claim is automatically inferred.
All eligible endpoints, including unchanged contexts, remain in the primary result.
After all predictions, Step 7 counterfactual projection-change annotations support
separate descriptive diagnostics; they never enter model inputs or model selection.
These annotations are not an oracle performance bound or an identifiability proof.
Statistics-only requires the recorded post-inference source verification to pass.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PureWindowsPath
import platform
import shutil
import sys
import time
import traceback
import zipfile

import numpy as np
import pandas as pd

PROTOCOL_VERSION = 'shared_nominal_formal_encoder_test_v3'
SOURCE_VERSION = 'shared_nominal_formal_encoder_trainval_v3'
SOURCE_REL = 'outputs/shared_nominal_v2/45_encoder_comparison_formal_trainval/run_formal_5seed'
OUTPUT_REL = 'outputs/shared_nominal_v2/46_encoder_comparison_frozen_test/run_formal_5seed'
DATA_REL = 'dataset/shared_nominal_v2'
SEEDS = tuple(range(20260610, 20260615))
CONDITIONS = ('CategoricalEmbedding_MLP', 'FieldToken_Encoder',
              'DistilBERT_RandomInit_Train', 'DistilBERT_Pretrained_Finetune')
LABELS = ('NORMAL', 'RF_LINK_JAMMING', 'GNSS_SPOOFING',
          'UNAUTHORIZED_TELECOMMAND_INJECTION', 'TELECOMMAND_REPLAY',
          'GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE')
META = ['episode_id', 'episode_step', 'window_id', 'label', 'y_true']
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, BOOTSTRAP_BATCH = 20000, 20261001, 100
CANARY_N, CANARY_ATOL, CANARY_RTOL = 512, 1e-5, 1e-5
OBSERVABILITY_FILE = 'audit/test_window_observable_effects.csv'
OBSERVABILITY_REFERENCE = 'observability_endpoints_L10.csv'
SOURCE_VERIFICATION_FILE = 'inference_sources_verified.json'
AUDIT_FIELDS = ('audit_observable_effect_present','audit_observable_difference_count',
                'audit_persistent_effect_present','audit_attack_active')
EFFECT_FIELDS = ('audit_current_observable_effect','audit_history_effect_window_count',
                 'audit_context_effect_window_count','audit_effect_stratum')
EFFECT_STRATA = ('current_changed','history_only_changed','no_observable_change')


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


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


def replace_atomic(temporary, target):
    # Bounded retry for the transient Windows sharing error seen in Step 45.
    for attempt in range(6):
        try:
            Path(temporary).replace(target)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.2 * (attempt + 1))


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    replace_atomic(temp, path)


def write_csv(path, frame):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    pd.DataFrame(frame).to_csv(temp, index=False, encoding='utf-8-sig')
    replace_atomic(temp, path)


def write_array(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as stream:
        if path.suffix == '.npz':
            np.savez_compressed(stream, **arrays)
        else:
            np.save(stream, arrays['value'], allow_pickle=False)
    replace_atomic(temp, path)


def safe_artifact(folder, name):
    relative = Path(name); folder = Path(folder).resolve()
    require(not relative.is_absolute() and not PureWindowsPath(name).drive and
            '..' not in relative.parts and '\\' not in name, 'Unsafe artifact path: ' + str(name))
    path = (folder / relative).resolve()
    require(folder in path.parents, 'Artifact escaped its run directory')
    return path


def write_completion(folder, filename, contract, result, artifacts):
    folder = Path(folder)
    body = dict(contract=contract, result=result, completed_at_utc=utc(),
                artifact_sha256={n: digest(safe_artifact(folder, n)) for n in artifacts})
    write_json(folder / filename, dict(identity=object_hash(body), **body))


def read_completion(folder, filename, contract):
    folder = Path(folder); path = folder / filename
    if not path.exists():
        return None
    r = read_json(path)
    require(r.get('identity') == object_hash({k: v for k, v in r.items() if k != 'identity'}) and
            r.get('contract') == contract, 'Completion identity/configuration changed: ' + str(path))
    require(bool(r['artifact_sha256']), 'Empty completion record')
    for name, expected in r['artifact_sha256'].items():
        artifact = safe_artifact(folder, name)
        require(artifact.is_file() and digest(artifact) == expected, 'Completed artifact changed/missing: ' + str(artifact))
    return r['result']


class RunLock:
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.stream = self.path.open('a+b')
        if self.stream.tell() == 0:
            self.stream.write(b'0'); self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            raise ValueError('Step 46 is already running in this output folder')
        return self

    def __exit__(self, *args):
        self.stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()


def macro_f1(cm):
    cm = np.asarray(cm)
    den = cm.sum(-1) + cm.sum(-2)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    return np.divide(2. * tp, den, out=np.zeros_like(tp, dtype=float), where=den > 0).mean(-1)


def mean_seed_f1(cm):
    return macro_f1(cm).mean(-1)


def metrics_from_predictions(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    require(y.ndim == 1 and y.shape == pred.shape and len(y) > 0, 'Prediction/label shape mismatch')
    require(np.isin(y, range(6)).all() and np.isin(pred, range(6)).all(), 'Invalid class IDs')
    cm = np.bincount((6*y + pred).astype(np.int64), minlength=36).reshape(6, 6)
    den = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2.*np.diag(cm), den, out=np.zeros(6), where=den > 0)
    return dict(n=len(y), macro_f1=float(f1.mean()), accuracy=float(np.trace(cm)/cm.sum()),
                weighted_f1=float(np.average(f1, weights=cm.sum(1))), class_f1=f1.tolist(),
                confusion_matrix=cm.tolist())


def validate_predictions(frame, reference, job=None, protocol=None, head_kind=None):
    require(set(META + ['y_pred'] + [f'prob_{k}' for k in range(6)]) <= set(frame), 'Incomplete prediction schema')
    require(set(reference.columns) <= set(frame) and frame[list(reference.columns)].reset_index(drop=True).equals(reference.reset_index(drop=True)), 'Prediction endpoints/labels/order changed')
    require(not frame.duplicated(['episode_id', 'episode_step']).any(), 'Duplicate prediction endpoint')
    y = frame.label.map(dict(zip(LABELS, range(6)))).to_numpy()
    require(np.array_equal(y, frame.y_true), 'Label names/IDs disagree')
    probs = frame[[f'prob_{k}' for k in range(6)]].to_numpy(dtype=float)
    require(np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all() and
            np.allclose(probs.sum(1), 1., rtol=0, atol=1e-6), 'Invalid probability values/sums')
    require(np.array_equal(probs.argmax(1), frame.y_pred), 'Probabilities/predicted classes disagree')
    if job is not None:
        expected = prediction_provenance(job, protocol, head_kind,
                    'all_test_windows' if 'sequence_end_row' not in reference else 'common_L10_endpoints')
        for key, value in expected.items():
            require(key in frame and frame[key].eq(value).all(), 'Prediction provenance differs: '+key)
    return metrics_from_predictions(frame.y_true.to_numpy(), frame.y_pred.to_numpy())


def episode_confusions(reference, predictions, n_classes=6):
    predictions = np.asarray(predictions)
    require(predictions.ndim == 2 and predictions.shape[1] == len(reference), 'Wrong prediction count')
    require(np.isfinite(predictions).all() and np.isin(predictions, range(n_classes)).all(), 'Invalid prediction IDs')
    require(reference.groupby('episode_id').y_true.nunique().max() == 1, 'Mixed class within episode')
    codes, names = pd.factorize(reference.episode_id, sort=True)
    inventory = reference.groupby('episode_id', sort=True).agg(y_true=('y_true','first'), endpoints=('y_true','size')).reset_index()
    require(np.array_equal(inventory.episode_id.to_numpy(), names.to_numpy()), 'Episode ordering mismatch')
    y = reference.y_true.to_numpy(dtype=np.int64)
    require(np.isin(y, range(n_classes)).all(), 'Invalid reference class IDs')
    counts = np.zeros((len(names), len(predictions), n_classes, n_classes), dtype=np.int64)
    for r, pred in enumerate(predictions):
        flat = (codes*n_classes + y)*n_classes + pred.astype(np.int64)
        counts[:, r] = np.bincount(flat, minlength=len(names)*n_classes*n_classes).reshape(len(names),n_classes,n_classes)
    return counts, inventory.y_true.to_numpy(dtype=np.int64), inventory


def draw_episode_weights(strata, batch, rng):
    weights = np.zeros((batch, len(strata)), dtype=np.int64)
    for c in np.unique(strata):
        ix = np.flatnonzero(strata == c)
        weights[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1./len(ix)), size=batch)
    return weights


def bootstrap_scores(counts, strata, n_seeds, n_models, n_resamples, random_seed, batch_size, progress=True):
    require(counts.shape[1] == n_seeds*n_models and len(strata) == len(counts), 'Bootstrap dimensions disagree')
    require(n_resamples > 0 and batch_size > 0, 'Invalid bootstrap count')
    sufficient = np.stack((np.diagonal(counts, axis1=-2, axis2=-1), counts.sum(-1), counts.sum(-2)), axis=-2).astype(float)
    flat = sufficient.reshape(len(strata), -1)
    rng = np.random.Generator(np.random.PCG64(random_seed))
    scores = np.empty((n_resamples, n_models)); start = time.perf_counter()
    for lo in range(0, n_resamples, batch_size):
        hi = min(lo+batch_size, n_resamples)
        weights = draw_episode_weights(strata, hi-lo, rng)
        sampled = (weights.astype(float) @ flat).reshape(hi-lo,n_models,n_seeds,3,counts.shape[-1])
        den = sampled[...,1,:] + sampled[...,2,:]
        f1 = np.divide(2*sampled[...,0,:], den, out=np.zeros_like(den), where=den > 0)
        scores[lo:hi] = f1.mean(-1).mean(-1)  # classes, then seeds, NEVER pooled confusion F1
        if progress and (lo == 0 or hi % 1000 == 0 or hi == n_resamples):
            print(f'  Bootstrap {hi:,}/{n_resamples:,} | {time.perf_counter()-start:.1f}s', flush=True)
    return scores


def contrast_intervals(scores, point_estimates):
    require(np.shape(scores)[1] == 4 and np.shape(point_estimates) == (4,), 'Expected four conditions')
    rows = []
    def direction(low, high):
        return 'pretrained_higher' if low > 0 else 'comparator_higher' if high < 0 else 'zero_included_not_equivalence'
    for i, condition in enumerate(CONDITIONS[:3]):
        delta = np.asarray(scores)[:,3] - np.asarray(scores)[:,i]
        lo, hi = np.quantile(delta, [.025,.975], method='linear')
        blo, bhi = np.quantile(delta, [.05/(2*3),1-.05/(2*3)], method='linear')
        point = float(point_estimates[3] - point_estimates[i])
        rows.append(dict(reference=CONDITIONS[3], comparator=condition, delta_macro_f1=point,
                         delta_percentage_points=100*point, ci95_low=float(lo),ci95_high=float(hi),
                         bonferroni_confidence=1-.05/3, bonferroni_low=float(blo),bonferroni_high=float(bhi),
                         ci95_interpretation=direction(lo,hi),familywise_interpretation=direction(blo,bhi)))
    return rows


def read_source(source, verify_weights=True):
    source = Path(source).resolve()
    require(source.is_dir(), 'Full completed revised Step 45 folder required; the review ZIP alone has no weights')
    m = read_json(source/'run_manifest.json'); p = read_json(source/'protocol.json')
    require(m.get('script_number') == 45 and m.get('protocol_version') == SOURCE_VERSION
            and m.get('status') == 'FORMAL_TRAIN_VAL_COMPLETE' and m.get('completed_pipelines') == 20
            and m.get('completed_encoder_seeds') == 5 and m.get('test_evaluation') is False
            and m.get('test_data_read') is False, 'Complete the revised Step 45 first; legacy/partial runs cannot be evaluated')
    identity = object_hash({k:v for k,v in p.items() if k != 'identity'})
    require(p.get('identity') == m.get('protocol_identity') == identity and p.get('version') == SOURCE_VERSION,
            'Step 45 protocol identity/version changed')
    for key in ('script_sha256','adapter_sha256','engine_sha256'):
        require(valid_sha256(p.get(key)) and p[key] == m.get(key), 'Source code chain differs: '+key)
    require(p.get('seeds') == list(SEEDS) and p.get('conditions') == list(CONDITIONS)
            and p.get('formal') is True and p.get('test_evaluation') is False, 'Formal scope/seeds changed')
    require(p.get('dataset_id') == m.get('dataset_id') == p['source_contract']['dataset_id']
            and p.get('source_contract') == m.get('source_contract'), 'Formal dataset provenance differs')
    validate_statistics_plan(p['statistics_plan'])
    required = {'protocol.json','train_field_vocabulary.json','formal_statistics_plan.json',
                'formal_validation_runs.json','formal_validation_summary.json','source_cache_reference.json'}
    require(required <= set(m.get('output_sha256',{})), 'Formal review inventory is incomplete')
    for name, expected in m['output_sha256'].items():
        require(not any(part.lower().startswith('test') for part in Path(name).parts), 'Unexpected test artifact in training source')
        f = safe_artifact(source,name)
        require(f.is_file() and digest(f) == expected, 'Step 45 small artifact changed: '+name)
    require(read_json(source/'formal_statistics_plan.json') == p['statistics_plan'], 'Statistics plan differs from training protocol')
    require([(j['seed'],j['condition']) for j in p['jobs']] == [(s,c) for s in SEEDS for c in CONDITIONS], 'Formal job set/order changed')
    require([x.get('encoder_seed') for x in p['protocols']] == list(SEEDS), 'Formal per-seed protocols differ')
    by_seed = dict(zip(SEEDS,p['protocols']))
    weights, records = {}, {}
    for job in p['jobs']:
        require(job['protocol'] == by_seed[job['seed']], 'Job/per-seed protocol mismatch')
        require(job['key'] == f"{job['seed']}/{job['condition']}" and
                job['relative_dir'] == f"seed_{job['seed']}/{job['condition']}", 'Formal job path/key differs')
        require(job['protocol']['encoder_seed'] == job['seed'] and job['protocol']['test_evaluation'] is False
                and job['protocol']['evaluation_precision'] == job['protocol']['extraction_precision'] == 'float32',
                'Formal seed/precision policy differs')
        folder = safe_artifact(source,job['relative_dir']); marker = read_json(folder/'pipeline_complete.json')
        require(marker.get('identity') == object_hash({k:v for k,v in marker.items() if k != 'identity'})
                and marker.get('contract') == object_hash(dict(run=identity,job=job)), 'Pipeline identity changed: '+job['key'])
        require(marker['summary']['seed'] == job['seed'] and marker['summary']['condition'] == job['condition']
                and marker['summary'].get('dataset_id') == p['dataset_id']
                and marker['summary'].get('step45_protocol_identity') == identity, 'Pipeline provenance mismatch')
        for name in ('pipeline_complete.json','encoder/complete.json','tst/complete.json','embeddings/complete.json'):
            require(job['relative_dir']+'/'+name in m['output_sha256'], 'Missing authenticated training record: '+name)
        embedding = read_json(folder/'embeddings/complete.json')
        for stage in ('encoder','tst'):
            complete = read_json(folder/stage/'complete.json')
            stage_id = (object_hash(dict(run=identity,seed=job['seed'],condition=job['condition'],stage='encoder'))
                        if stage == 'encoder' else object_hash(dict(run=identity,seed=job['seed'],condition=job['condition'],embeddings=embedding['sha256'])))
            cfg = job['encoder_training'] if stage == 'encoder' else job['tst_training']
            require(complete['contract'] == object_hash(dict(identity=stage_id,config=cfg))
                    and complete.get('evaluation_precision') == 'float32', 'Training stage settings/precision differ')
            expected = complete['artifact_sha256']['best.pt']
            require(marker['artifact_sha256'].get(stage+'/best.pt') == expected, 'Checkpoint cross-reference mismatch')
            name = job['relative_dir']+'/'+stage+'/best.pt'; weights[name] = expected
            if verify_weights:
                f = safe_artifact(source,name)
                require(f.is_file() and digest(f) == expected, 'Selected checkpoint missing/changed: '+str(f))
        required_pipeline = {'encoder/val_logits.npy','tst/val_logits.npy','embeddings/val.npy',
                             'val_window_predictions_common_L10.csv','val_tst_predictions_L10.csv'}
        require(required_pipeline <= set(marker['artifact_sha256']), 'Missing validation reference artifacts')
        records[job['key']] = marker
    vocabulary = read_json(source/'train_field_vocabulary.json')
    require(digest(source/'train_field_vocabulary.json') == p['train_vocabulary_sha256'], 'Source vocabulary differs')
    return dict(manifest=m,protocol=p,manifest_sha256=digest(source/'run_manifest.json'),
                vocabulary=vocabulary,weights=weights,records=records)




def prepare_arrays(frame, fields, serializer, tokenizer, vocabulary, max_length):
    # Only the explicit field allowlist is passed to the serializer.
    require(type(max_length) is int and max_length > 2, 'Invalid declared token budget')
    require(tokenizer.padding_side == 'right' and tokenizer.pad_token_id is not None and tokenizer.unk_token_id is not None,
            'Expected the same right-padded tokenizer used by Step 45')
    count = len(frame)
    categories = np.stack([frame[f].astype(str).str.strip().map(vocabulary[f]).fillna(0).to_numpy(dtype=np.int64)
                           for f in fields], axis=1)
    ids = np.full((count,max_length),tokenizer.pad_token_id,dtype=np.int32)
    mask = np.zeros((count,max_length),dtype=np.int8)
    maximum, unknown = 0, 0
    for start in range(0,count,1024):
        texts = serializer(frame.iloc[start:start+1024][list(fields)])
        token_ids = tokenizer(texts,add_special_tokens=True,truncation=False,padding=False)['input_ids']
        require(len(token_ids) == len(texts) == min(1024,count-start), 'Tokenizer changed the number of rows')
        for j, row in enumerate(token_ids):
            require(2 < len(row) <= max_length, 'Empty tokenized row or exceeded token budget; no truncation is allowed')
            maximum = max(maximum,len(row)); unknown += int(tokenizer.unk_token_id in row)
            ids[start+j,:len(row)] = row; mask[start+j,:len(row)] = 1
    require(unknown == 0, 'Actual tokenizer UNK found; stop rather than silently change the audited representation')
    labels = frame.label.map(dict(zip(LABELS,range(6)))).to_numpy(dtype=np.int64)
    data = dict(input_ids=ids,attention_mask=mask,categories=categories,labels=labels)
    audit = dict(max_length=max_length,rows=count,max_token_length=maximum,truncated_rows=0,rows_with_tokenizer_UNK=unknown,
                 categorical_unknown_id=0,categorical_oov={f:int((categories[:,i] == 0).sum()) for i,f in enumerate(fields)})
    return data,audit


def load_engine(root, code_hashes):
    paths = resolve_code_paths(root,code_hashes)
    path = paths['43']
    spec = importlib.util.spec_from_file_location('taes46_engine43', path)
    engine = importlib.util.module_from_spec(spec); spec.loader.exec_module(engine)
    require(not engine.DEPENDENCY_ERROR, 'Use the training virtual environment: '+str(engine.DEPENDENCY_ERROR))
    return engine


def prepare_environment(root, source, record, base_override=None, audit_override=None, audit_script=None):
    code_hashes = source_code_hashes(record)
    engine = load_engine(root,code_hashes)
    import transformers
    p,m = record['protocol'],record['manifest']; original = p['runtime']
    device_name = p['protocols'][0]['device']
    require(device_name in ('cpu','cuda') and (device_name != 'cuda' or engine.torch.cuda.is_available()),
            'Use the same device environment as Step 45; no automatic CPU fallback')
    device = engine.torch.device(device_name); numeric = p['numerical_runtime']
    engine.torch.set_num_threads(numeric['torch_num_threads'])
    if device.type == 'cuda':engine.torch.cuda.set_device(numeric['cuda_device_index'])
    engine.torch.backends.cuda.matmul.allow_tf32 = numeric['matmul_allow_tf32']
    engine.torch.backends.cudnn.allow_tf32 = numeric['cudnn_allow_tf32']
    policy = engine.numerical_runtime()
    require(policy == numeric.get('effective_policy') == original.get('numerical_policy'),
            'Effective numerical policy differs from Step 45; retain its arithmetic settings')
    runtime = dict(python=platform.python_version(),torch=str(engine.torch.__version__),numpy=np.__version__,
        pandas=pd.__version__,transformers=transformers.__version__,platform=platform.platform(),
        device=engine.torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',cuda=engine.torch.version.cuda,
        cudnn=engine.torch.backends.cudnn.version(),torch_threads=engine.torch.get_num_threads(),
        torch_interop_threads=engine.torch.get_num_interop_threads(),cuda_device_count=engine.torch.cuda.device_count() if device.type == 'cuda' else 0,
        numerical_policy=policy)
    require(runtime == original, 'Evaluation runtime differs from Step 45; retain its environment')
    require(engine.torch.backends.cudnn.benchmark == numeric['cudnn_benchmark'] and
            engine.torch.backends.cudnn.deterministic == numeric['cudnn_deterministic'] and
            engine.torch.backends.cudnn.version() == numeric['cudnn_version'], 'Numerical runtime differs')
    for job in p['jobs']:
        require(job['encoder_training'] == engine.training_spec(job['condition'],job['protocol'])
                and job['tst_training'] == engine.training_spec(job['condition'],job['protocol'],tst=True)
                and job['protocol']['tst_model'] == engine.TST_MODEL, 'Source job differs from the loaded training engine')
    audit_path = Path(audit_override).resolve() if audit_override else recorded_path(m['audit_path'],root,engine)
    if audit_script is None:
        recorded_script = m.get('invocation',{}).get('audit_script')
        if recorded_script and Path(recorded_script).is_file():
            audit_script = Path(recorded_script).resolve()
    audit,frames,tokenizer_dir,vocabulary,inputs = engine.load_inputs(root,audit_path,audit_script)
    require(audit['manifest_sha256'] == p['audit_manifest_sha256'], 'Input audit differs from Step 45')
    fingerprints = lambda rows: sorted((x['sha256'],x['size_bytes']) for x in rows)
    require(fingerprints(inputs) == fingerprints(p['source_files']), 'Current source bytes differ from Step 45')
    require(engine.pilot_source_contract(audit['source_contract']) == p['source_contract'],
            'Current shared-nominal cohort/source/observability contract differs from Step 45')
    require(record['vocabulary'] == dict(source='train_only',unknown_id=0,fields=vocabulary), 'Train-only vocabulary differs')
    helper,_ = engine.load_audit_module(root,audit,Path(audit['validator_path']))
    build,declared,inventory = helper.load_build(root,[])
    plan = p['statistics_plan']; test_counts = build['counts']['test']
    eligible = int(((inventory.split == 'test') & (inventory.row_count >= 10)).sum())
    require(plan['dataset_id'] == build['dataset_id'] and plan['expected_test_windows'] == test_counts['rows']
            and plan['expected_test_l10'] == test_counts['sequences_stride1']['10']
            and plan['expected_test_episodes'] == test_counts['episodes'] and plan['expected_test_l10_episodes'] == eligible,
            'Declared 45/7 test populations differ')
    forbidden = set(frames['train'].episode_id) | set(frames['val'].episode_id)
    val_indices,val_labels,val_metadata = engine.sequence_view(frames['val'])
    endpoint_contract = dict(population='common_L10_endpoints',window_rows=len(frames['val']),common_rows=len(val_indices),
        eligible_episodes=int(val_metadata.episode_id.nunique()),
        end_row_indices_sha256=hashlib.sha256(np.asarray(val_indices[:,-1],dtype='<i8').tobytes()).hexdigest(),
        metadata_sha256=object_hash(val_metadata.to_dict(orient='records')),
        labels_sha256=hashlib.sha256(np.asarray(val_labels,dtype='<i8').tobytes()).hexdigest())
    require(endpoint_contract == p['validation_endpoint_contract'], 'Validation endpoint contract differs')
    source43 = recorded_path(p['source43']['path'],root,engine)
    cache = source43/'cache'
    require(read_json(cache/'complete.json') == p['source_cache'] and p['source_cache']['identity'] == p['source43']['identity'],
            'Original train/validation cache identity changed')
    length = p['protocols'][0]['max_length']
    require(length == p['source_cache']['max_length'] == audit['tokenization_summary']['max_length']
            and all(x['max_length'] == length for x in p['protocols']), 'Token budget differs')
    expected_cache = {f'{split}_{kind}.npy' for split in ('train','val') for kind in ('input_ids','attention_mask','categories','labels')}
    require(set(p['source_cache']['artifact_sha256']) == expected_cache, 'Unexpected source cache split/files')
    for name,expected in p['source_cache']['artifact_sha256'].items():
        require(digest(safe_artifact(cache,name)) == expected, 'Source cache changed: '+name)
    canary = {kind:np.load(cache/f'val_{kind}.npy',mmap_mode='r',allow_pickle=False)
              for kind in ('input_ids','attention_mask','categories','labels')}
    n_val = len(frames['val'])
    require(canary['input_ids'].shape == canary['attention_mask'].shape == (n_val,length)
            and canary['categories'].shape == (n_val,22)
            and np.array_equal(canary['labels'],frames['val'].label.map(engine.LABEL_TO_ID).to_numpy()), 'Validation cache rows differ')
    base = Path(base_override).resolve() if base_override else Path(m['initializer_path']).resolve()
    require(base.is_dir(), 'Original base config/tokenizer folder missing; supply --base-model-dir')
    for name,expected in p['base_hashes'].items():
        require(Path(name).name == name and '\\' not in name and ':' not in name, 'Invalid base filename')
        require((base/name).is_file() and digest(base/name) == expected, 'Base bundle changed: '+name)
    tokenizer = engine.tokenizers_for_run(tokenizer_dir,base,p['source_contract']['tokenizer_sha256'])
    test_path = root/DATA_REL/'splits/test_observable.csv'
    test_sha = declared['splits/test_observable.csv']
    require(test_sha == p['source_contract']['input_sha256']['splits/test_observable.csv'] and digest(test_path) == test_sha,
            'Test bytes differ from the cohort declared before formal training')
    input_hashes = {x['path']:x['sha256'] for x in inputs}
    return dict(engine=engine,device=device,runtime=runtime,base=base,tokenizer=tokenizer,forbidden=forbidden,
                canary=canary,val_indices=val_indices,val_labels=val_labels,val_metadata=val_metadata,
                input_hashes=input_hashes,helper=helper,inventory=inventory,test_counts=test_counts,
                eligible_test_episodes=eligible,test_path=test_path,test_sha256=test_sha,max_length=length,
                cache_path=cache,source_cache=p['source_cache'],code_paths=resolve_code_paths(root,code_hashes))


def statistics_spec():
    return dict(draws=BOOTSTRAP_DRAWS,rng='PCG64',rng_seed=BOOTSTRAP_SEED,batch_size=BOOTSTRAP_BATCH,
                quantile_method='linear',primary='six-class macro-F1 at fixed L10 test endpoints',
                estimate='arithmetic mean of per-encoder-seed F1; no ensemble or pooled confusion F1',
                seed_sd_ddof=1,resampling='class-stratified whole episodes, same draw for all models and seeds',
                seed_resampling=False,confidence=.95,bonferroni_confidence=1-.05/3,
                contrasts=[CONDITIONS[3]+' minus '+c for c in CONDITIONS[:3]],
                uncertainty='episode sampling conditional on these five trained seeds and fixed TST RNG',
                secondary_contrasts=[c+': L10 TST minus window classifier at common L10 endpoints' for c in CONDITIONS],
                secondary_bonferroni_confidence=1-.05/4,
                multiplicity_scope='three primary encoder contrasts and four secondary temporal contrasts are separate families; no joint seven-contrast coverage claim',
                temporal_scope='same endpoints; different head capacity/training also contribute; interpret with pooling/LSTM controls',
                primary_population='all eligible L10 endpoints, including no observable change; no filtering or relabeling',
                equivalence_margin=None,test_used_for_selection=False)


def freeze_protocol(out, root, source, record, env, write=True):
    plan = record['protocol']['statistics_plan']
    payload = dict(version=PROTOCOL_VERSION,script_number=46,purpose='FORMAL_ENCODER_HELD_OUT_TEST_AND_BOOTSTRAP',
        script_sha256=digest(Path(__file__)),source_manifest_sha256=record['manifest_sha256'],
        source_protocol_identity=record['protocol']['identity'],source_run=str(source),project_root=str(root),
        source_code_sha256=source_code_hashes(record),checkpoint_sha256=record['weights'],dataset_id=record['protocol']['dataset_id'],
        observability_audit=record['protocol']['source_contract']['observability_audit'],
        observability_diagnostics=observability_spec(),
        test_sha256=env['test_sha256'],input_sha256=env['input_hashes'],source_cache=record['protocol']['source_cache'],
        vocabulary_sha256=record['protocol']['train_vocabulary_sha256'],runtime=env['runtime'],
        numerical_runtime=record['protocol']['numerical_runtime'],base_config_dir=str(env['base']),
        base_hashes=record['protocol']['base_hashes'],conditions=list(CONDITIONS),encoder_seeds=list(SEEDS),
        jobs=record['protocol']['jobs'],test_windows=plan['expected_test_windows'],test_l10=plan['expected_test_l10'],
        test_episodes=plan['expected_test_episodes'],test_l10_episodes=plan['expected_test_l10_episodes'],
        test_endpoint_identity=env['test_endpoint_identity'],max_length=env['max_length'],
        preprocessing='train-only categories; explicit 22-field allowlist; no truncation or actual tokenizer UNK',
        evaluation_precision='float32',feature_precision='float32',
        window_and_tst_evaluation='FP32 with original Step 45 evaluation batch boundaries; AMP training does not imply AMP inference',
        canary=dict(target_rows=CANARY_N,target_sequences=CANARY_N,round_up_to_original_batches=True,
                    atol=CANARY_ATOL,rtol=CANARY_RTOL,predictions_exact=True),
        statistics=statistics_spec(),source_statistics_plan=plan,training=False,
        checkpoint_selection='all 20 validation-selected pairs already declared by Step 45; no test selection')
    expected = dict(identity=object_hash(payload),**payload)
    path = out/'protocol.json'
    if path.exists():
        require(read_json(path) == expected, 'Resume inputs/code/protocol/runtime changed; previous results are preserved')
    elif write:
        write_json(path,expected)
    return expected


def load_test(root, env, record):
    path = env['test_path']; expected = env['test_counts']
    require(digest(path) == env['test_sha256'], 'Test file changed')
    frame = pd.read_csv(path,encoding='utf-8-sig',dtype=str,keep_default_na=False)
    engine,helper = env['engine'],env['helper']
    helper.validate_frame(frame,'test',expected)
    helper.verify_episode_rows(frame,'test',env['inventory'])
    require(not set(frame.episode_id) & env['forbidden'], 'Test episode split leakage')
    indices,labels,meta = engine.sequence_view(frame)
    require(len(indices) == expected['sequences_stride1']['10'] and meta.episode_id.nunique() == env['eligible_test_episodes'],
            'Test L10 endpoint/eligible-episode count differs')
    require(set(labels) == set(range(6)), 'L10 test endpoints must contain all six classes; do not silently drop a class')
    data,audit = prepare_arrays(frame,engine.FIELDS,engine.serialize_frame,env['tokenizer'],
                                record['vocabulary']['fields'],env['max_length'])
    require(digest(path) == env['test_sha256'], 'Test file changed while reading')
    all_meta = frame[list(engine.METADATA)].copy()
    all_meta['window_row_index'] = np.arange(len(frame),dtype=np.int64)
    all_meta['y_true'] = data['labels']; meta['y_true'] = labels
    data.update(sequences=indices,sequence_labels=labels,metadata=meta,window_metadata=all_meta)
    env['test_endpoint_identity'] = object_hash(meta.to_dict(orient='records'))
    return data,audit


def observability_spec():
    return dict(schema_version='counterfactual_observable_change_v1',length=10,
        strata=list(EFFECT_STRATA),scope='descriptive only, attached after all 20 test predictions',
        measurement='actual versus paired nominal 22-field projection changes within each causal L10 context',
        no_change_interpretation='no measured projection change; not proof of no class information or an oracle ceiling',
        outputs='per-class conditional accuracy, predicted NORMAL fraction and predicted-class counts; seed means and sample SD',
        model_inputs=False,checkpoint_selection=False,primary_endpoint_filtering=False,
        pass_fail_threshold=None,additional_significance_tests=False)


def integer_column(frame, name, low, high):
    values = pd.to_numeric(frame[name],errors='raise').to_numpy(dtype=float)
    require(np.isfinite(values).all() and np.equal(values,np.floor(values)).all() and
            ((values >= low) & (values <= high)).all(), 'Invalid integer annotation: '+name)
    frame[name] = values.astype(np.int64)


def validate_observability_reference(frame, reference):
    require(list(frame) == list(reference)+list(EFFECT_FIELDS) and len(frame) == len(reference),
            'Observable-effect endpoint schema/count differs')
    require(frame[list(reference)].reset_index(drop=True).equals(reference.reset_index(drop=True)),
            'Observable-effect endpoint metadata/order differs')
    integer_column(frame,EFFECT_FIELDS[0],0,1)
    integer_column(frame,EFFECT_FIELDS[1],0,9)
    integer_column(frame,EFFECT_FIELDS[2],0,10)
    current = frame[EFFECT_FIELDS[0]].to_numpy(); history = frame[EFFECT_FIELDS[1]].to_numpy()
    require(np.array_equal(frame[EFFECT_FIELDS[2]],current+history), 'Observable-effect context count differs')
    expected = np.where(current == 1,EFFECT_STRATA[0],np.where(history > 0,EFFECT_STRATA[1],EFFECT_STRATA[2]))
    require(np.array_equal(frame[EFFECT_FIELDS[3]],expected), 'Observable-effect stratum differs')
    return frame


def load_observability_reference(root, protocol, env, data):
    # Called only after inference. Never add these columns to model data or labels.
    spec = protocol['observability_audit']
    require(spec.get('schema_version') == 'counterfactual_observable_change_v1' and
            spec.get('dataset_id') == protocol['dataset_id'] and
            spec.get('split_files',{}).get('test') == OBSERVABILITY_FILE,
            'Shared-nominal observability provenance differs')
    expected = spec.get('artifacts',{}).get(OBSERVABILITY_FILE)
    require(valid_sha256(expected), 'Missing declared test observable-effect file')
    path = safe_artifact(root/DATA_REL,OBSERVABILITY_FILE)
    require(path.is_file() and digest(path) == expected, 'Test observable-effect source changed/missing')
    frame = pd.read_csv(path,encoding='utf-8-sig',dtype=str,keep_default_na=False)
    metadata = list(env['engine'].METADATA)
    require(list(frame) == metadata+list(AUDIT_FIELDS) and len(frame) == len(data['labels']),
            'Window observable-effect schema/count differs')
    for name in ('episode_step','window_id'):
        integer_column(frame,name,0,np.iinfo(np.int64).max)
    require(frame[metadata].equals(data['window_metadata'][metadata].reset_index(drop=True)),
            'Window observable-effect metadata/order differs')
    for name in AUDIT_FIELDS:
        integer_column(frame,name,0,22 if name == 'audit_observable_difference_count' else 1)
    flags = frame.audit_observable_effect_present.to_numpy(dtype=np.int64)
    require(np.array_equal(flags,(frame.audit_observable_difference_count.to_numpy() > 0).astype(np.int64)),
            'Observable-effect flag/difference count disagree')
    indices = data['sequences']
    require(indices.shape == (len(data['metadata']),10), 'Expected exact causal L10 indices for diagnostics')
    current = flags[indices[:,-1]]; history = flags[indices[:,:-1]].sum(1)
    result = data['metadata'].copy()
    result[EFFECT_FIELDS[0]] = current; result[EFFECT_FIELDS[1]] = history
    result[EFFECT_FIELDS[2]] = current+history
    result[EFFECT_FIELDS[3]] = np.where(current == 1,EFFECT_STRATA[0],np.where(history > 0,EFFECT_STRATA[1],EFFECT_STRATA[2]))
    require(digest(path) == expected, 'Test observable-effect source changed while reading')
    env['observability_path'],env['observability_sha256'] = path,expected
    return validate_observability_reference(result,data['metadata'])


def verify_observability_file(out, protocol, reference=None):
    reference = verify_endpoint_file(out,protocol) if reference is None else reference
    path = out/OBSERVABILITY_REFERENCE
    require(path.is_file(), 'Observable-effect endpoint reference missing; resume the normal command')
    return validate_observability_reference(read_prediction_csv(path),reference)


def observability_diagnostics(reference, predictions, protocol):
    predictions = np.asarray(predictions)
    require(predictions.shape == (40,len(reference)) and np.isin(predictions,range(6)).all(),
            'Expected all 40 matched-endpoint prediction vectors for diagnostics')
    rows = []
    y = reference.y_true.to_numpy(); strata = reference.audit_effect_stratum.to_numpy()
    for head_index,head in enumerate(('tst','window')):
        for c,condition in enumerate(CONDITIONS):
            for s,seed in enumerate(SEEDS):
                pred = predictions[(head_index*4+c)*5+s]
                for stratum in EFFECT_STRATA:
                    for label,name in enumerate(LABELS):
                        select = (strata == stratum) & (y == label); n = int(select.sum())
                        counts = np.bincount(pred[select],minlength=6)
                        rows.append(dict(dataset_id=protocol['dataset_id'],step46_protocol_identity=protocol['identity'],
                            condition=condition,encoder_seed=seed,head_kind=head,audit_effect_stratum=stratum,true_label=name,
                            n=n,correct=int(counts[label]),accuracy=float(counts[label]/n) if n else None,
                            predicted_normal_fraction=float(counts[0]/n) if n else None,
                            **{f'predicted_class_{k}_count':int(counts[k]) for k in range(6)}))
    frame = pd.DataFrame(rows); summary = []
    keys = ['condition','head_kind','audit_effect_stratum','true_label']
    for key,group in frame.groupby(keys,sort=False):
        require(len(group) == 5 and group.n.nunique() == 1, 'Diagnostic seed populations differ')
        item = dict(zip(keys,key),dataset_id=protocol['dataset_id'],step46_protocol_identity=protocol['identity'],
                    endpoints_per_seed=int(group.n.iloc[0]),encoder_seeds=5)
        for metric in ('accuracy','predicted_normal_fraction'):
            valid = group[metric].dropna()
            item[metric+'_seed_mean'] = float(valid.mean()) if len(valid) else None
            item[metric+'_seed_sd_ddof1'] = float(valid.std(ddof=1)) if len(valid) > 1 else None
        summary.append(item)
    return frame,pd.DataFrame(summary)


def compare_canary(actual, expected, description, predictions=False):
    actual, expected = np.asarray(actual),np.asarray(expected)
    require(actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all(),
            'Invalid validation canary shape/values: '+description)
    require(np.allclose(actual,expected,rtol=CANARY_RTOL,atol=CANARY_ATOL),
            'Validation canary differs from Step 45: '+description+'; stop before accepting test output')
    if predictions:
        require(np.array_equal(actual.argmax(1),expected.argmax(1)), 'Validation canary prediction mismatch: '+description)
    return dict(max_abs_error=float(np.max(np.abs(actual.astype(float)-expected.astype(float)))),
                compared_values=int(actual.size),predictions_checked=predictions)


def batches_for(engine, data, condition):
    arrays = {k:data[k] for k in ('input_ids','attention_mask')} if condition in CONDITIONS[2:] else {'x':data['categories']}
    return engine.ArrayBatches(arrays,data['labels'])


def extract_fp32(engine, model, batches, batch_size, device, path=None):
    shape = (len(batches.labels),768)
    temporary = Path(path).with_name(Path(path).name+'.tmp') if path is not None else None
    result = np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float32,shape=shape) if temporary else np.empty(shape,dtype=np.float32)
    with engine.evaluating(model):
        for lo in range(0,shape[0],batch_size):
            ix = np.arange(lo,min(lo+batch_size,shape[0]))
            args,_ = batches.tensors(ix,device)
            # Feature extraction in Step 43/45 does NOT use AMP.
            with engine.torch.autocast(device_type=device.type,enabled=False):
                features = model.features(**args).float().cpu().numpy()
            require(features.shape == (len(ix),768) and np.isfinite(features).all(), 'Invalid FP32 embeddings')
            result[lo:lo+len(ix)] = features
            if temporary and (lo == 0 or (lo+len(ix)) % 8192 == 0 or lo+len(ix) == shape[0]):
                print(f'    FP32 features {lo+len(ix):,}/{shape[0]:,}',flush=True)
    if temporary:
        result.flush(); del result
        replace_atomic(temporary,path)
        return np.load(path,mmap_mode='r',allow_pickle=False)
    return result


def prediction_frame(metadata, logits, job, protocol, head_kind, population):
    # Compute CSV probabilities in float64 to avoid introducing softmax rounding ties.
    # Stored/model logits and feature extraction remain FP32.
    logits = np.asarray(logits,dtype=np.float64)
    require(logits.shape == (len(metadata),6) and np.isfinite(logits).all(), 'Invalid prediction logits')
    prob = np.exp(logits-logits.max(1,keepdims=True)); prob /= prob.sum(1,keepdims=True)
    frame = metadata.copy(); frame['y_pred'] = logits.argmax(1)
    for k in range(6):
        frame[f'prob_{k}'] = prob[:,k]
    for key,value in prediction_provenance(job,protocol,head_kind,population).items():
        frame[key] = value
    validate_predictions(frame,metadata,job,protocol,head_kind)
    return frame


def infer_pipeline(job, source, folder, record, env, data):
    engine,device = env['engine'],env['device']; condition = job['condition']
    folder.mkdir(parents=True,exist_ok=True)
    src = safe_artifact(source,job['relative_dir']); marker = record['records'][job['key']]
    cfg,tcfg = job['encoder_training'],job['tst_training']
    # Reference bytes are checked before using them as validation canaries.
    for name in ('encoder/val_logits.npy','embeddings/val.npy','tst/val_logits.npy'):
        f = safe_artifact(src,name)
        require(f.is_file() and digest(f) == marker['artifact_sha256'][name], 'Validation reference changed/missing: '+str(f))
    val_features = np.load(src/'embeddings/val.npy',mmap_mode='r',allow_pickle=False)
    val_window_logits = np.load(src/'encoder/val_logits.npy',mmap_mode='r',allow_pickle=False)
    val_tst_logits = np.load(src/'tst/val_logits.npy',mmap_mode='r',allow_pickle=False)
    n_val = len(env['canary']['labels']); n_sequences = len(env['val_labels'])
    require(val_features.shape == (n_val,768) and val_features.dtype == np.float32 and
            val_window_logits.shape == (n_val,6) and val_tst_logits.shape == (n_sequences,6) and
            val_window_logits.dtype == val_tst_logits.dtype == np.float32, 'Validation reference dimensions/precision changed')
    n_window = canary_count(n_val,cfg['eval_batch_size'])
    n_tst = canary_count(n_sequences,tcfg['eval_batch_size'])
    sizes = [len(record['vocabulary']['fields'][f]) for f in engine.FIELDS]
    engine.seed_all(job['seed'])
    if condition == CONDITIONS[0]:
        encoder = engine.CategoricalEncoder(sizes)
    elif condition == CONDITIONS[1]:
        encoder = engine.FieldTokenEncoder(sizes)
    else:
        from transformers import DistilBertConfig
        config = DistilBertConfig.from_pretrained(str(env['base']),local_files_only=True)
        config._attn_implementation = 'eager'
        # EVERY condition is restored from its own Step 45 trained state dict.
        encoder = engine.BertEncoder(config,weights_path=None)
    model = engine.WindowModel(encoder,head_seed=job['protocol']['window_head_seed'])
    model.load_state_dict(engine.load_checkpoint(src/'encoder/best.pt'),strict=True)
    model.to(device)
    parameters = dict(encoder_parameters=sum(p.numel() for p in encoder.parameters()),
                      window_model_parameters=sum(p.numel() for p in model.parameters()))
    canary_batches = batches_for(engine,{k:v[:n_window] for k,v in env['canary'].items()},condition)
    _,canary_logits = engine.evaluate(model,canary_batches,cfg['eval_batch_size'],device,0.,False)
    canary_features = extract_fp32(engine,model,canary_batches,cfg['eval_batch_size'],device)
    canary = dict(window=compare_canary(canary_logits,val_window_logits[:n_window],'window logits',True),
                  features=compare_canary(canary_features,val_features[:n_window],'FP32 encoder features'))
    engine.seed_all(job['protocol']['tst_seed'])
    tst = engine.TST(); tst.load_state_dict(engine.load_checkpoint(src/'tst/best.pt'),strict=True); tst.to(device)
    parameters['tst_parameters'] = sum(p.numel() for p in tst.parameters())
    parameters['deployment_parameters'] = parameters['encoder_parameters'] + parameters['tst_parameters']
    for k,v in parameters.items():
        require(v == marker['summary'][k], 'Restored parameter count differs: '+k)
    val_sequence = engine.SequenceBatches(val_features,env['val_indices'][:n_tst],env['val_labels'][:n_tst])
    _,restored_tst_logits = engine.evaluate(tst,val_sequence,tcfg['eval_batch_size'],device,tcfg['label_smoothing'],False)
    canary['tst'] = compare_canary(restored_tst_logits,val_tst_logits[:n_tst],'TST logits',True)
    canary.update(status='PASS',window_rows=n_window,sequences=n_tst,atol=CANARY_ATOL,rtol=CANARY_RTOL)
    write_json(folder/'validation_canary.json',canary)
    del val_features,val_window_logits,val_tst_logits,val_sequence,canary_logits,canary_features,restored_tst_logits,canary_batches
    print('  Validation restore/precision canary: PASS; test inference starts',flush=True)
    if device.type == 'cuda':engine.torch.cuda.synchronize()
    started = time.perf_counter()
    batches = batches_for(engine,data,condition)
    _,window_logits = engine.evaluate(model,batches,cfg['eval_batch_size'],device,0.,False)
    features = extract_fp32(engine,model,batches,cfg['eval_batch_size'],device,folder/'test_embeddings.npy')
    del model,encoder,batches
    if device.type == 'cuda':engine.torch.cuda.empty_cache()
    sequence = engine.SequenceBatches(features,data['sequences'],data['sequence_labels'])
    _,tst_logits = engine.evaluate(tst,sequence,tcfg['eval_batch_size'],device,tcfg['label_smoothing'],False)
    if device.type == 'cuda':engine.torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    l1frame = prediction_frame(data['window_metadata'],window_logits,job,env['evaluation_protocol'],'window','all_test_windows')
    l10frame = prediction_frame(data['metadata'],tst_logits,job,env['evaluation_protocol'],'tst','common_L10_endpoints')
    l1 = validate_predictions(l1frame,data['window_metadata'])
    l10 = validate_predictions(l10frame,data['metadata'])
    end = data['sequences'][:,-1]
    require(np.array_equal(data['labels'][end],data['sequence_labels']), 'L1 matched-endpoint label alignment failed')
    commonframe = prediction_frame(data['metadata'],window_logits[end],job,env['evaluation_protocol'],'window','common_L10_endpoints')
    matched = validate_predictions(commonframe,data['metadata'],job,env['evaluation_protocol'],'window')
    row = dict(seed=job['seed'],condition=condition,dataset_id=record['protocol']['dataset_id'],
               step46_protocol_identity=env['evaluation_protocol']['identity'],evaluation_precision='float32',
               temporal_delta_macro_f1=l10['macro_f1']-matched['macro_f1'],l1_all=l1,l1_matched_endpoints=matched,l10=l10,
               **parameters,inference_seconds_including_feature_io=elapsed,
               encoder_best_epoch=marker['summary']['encoder_best_epoch'],
               encoder_convergence=marker['summary']['encoder_convergence'],
               tst_best_epoch=marker['summary']['tst_best_epoch'],tst_convergence=marker['summary']['tst_convergence'])
    write_csv(folder/'test_predictions_L1_all.csv',l1frame)
    write_csv(folder/'test_predictions_L10.csv',l10frame)
    write_csv(folder/'test_predictions_window_common_L10.csv',commonframe)
    write_array(folder/'test_window_logits.npy',value=window_logits)
    write_array(folder/'test_tst_logits_L10.npy',value=tst_logits)
    write_json(folder/'metrics.json',row)
    del tst,sequence,features,window_logits,tst_logits
    if device.type == 'cuda':engine.torch.cuda.empty_cache()
    artifacts = ['test_predictions_L1_all.csv','test_predictions_L10.csv','test_window_logits.npy',
                 'test_tst_logits_L10.npy','test_embeddings.npy','metrics.json','validation_canary.json','test_predictions_window_common_L10.csv']
    return row,artifacts


def pipeline_contract(protocol, job):
    return object_hash(dict(run=protocol['identity'],job=job))


def update_progress(out, jobs, rows, current):
    completed = {(r['seed'],r['condition']) for r in rows}
    write_json(out/'progress.json',dict(updated_at_utc=utc(),completed_pipelines=len(rows),planned_pipelines=len(jobs),
               current_pipeline=current,pending_pipelines=[j['key'] for j in jobs if (j['seed'],j['condition']) not in completed]))


def run_queue(out, protocol, record, source, env, data, verified_rows=None):
    jobs = protocol['jobs']
    rows = list(verified_rows) if verified_rows is not None else verify_test_pipelines(out,protocol)
    completed = {f"{r['seed']}/{r['condition']}" for r in rows}
    update_progress(out,jobs,rows,None)
    for i,job in enumerate(jobs,1):
        if job['key'] in completed:
            print(f'[{i}/{len(jobs)}] Verified complete; skip {job["key"]}',flush=True)
            continue
        update_progress(out,jobs,rows,job['key'])
        print(f'[{i}/{len(jobs)}] Test inference {job["key"]}',flush=True)
        folder = safe_artifact(out,job['relative_dir'])
        row,artifacts = infer_pipeline(job,source,folder,record,env,data)
        require((row['seed'],row['condition']) == (job['seed'],job['condition']), 'New result belongs to another job')
        write_completion(folder,'pipeline_complete.json',pipeline_contract(protocol,job),row,artifacts)
        rows.append(row); update_progress(out,jobs,rows,None)
        print(f'  Complete {len(rows)}/{len(jobs)}',flush=True)
    return rows


def finish_run(out, manifest):
    # Keep full-window CSV and arrays locally. L10 CSVs suffice for primary review.
    small = [p for p in sorted(out.rglob('*')) if p.is_file() and
             (p.suffix in ('.json','.csv','.txt') or p.name == 'bootstrap_distribution.npz') and
             p.name not in ('run_manifest.json','test_predictions_L1_all.csv')]
    manifest.update(updated_at_utc=utc(),output_sha256={p.relative_to(out).as_posix():digest(p) for p in small},
                    review_excludes=['test_predictions_L1_all.csv','*.npy','model checkpoints'],review_zip=str(out.with_suffix('.zip')))
    write_json(out/'run_manifest.json',manifest)
    target = out.with_suffix('.zip'); temp = target.with_name(target.name+'.tmp')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in small+[out/'run_manifest.json']:
            z.write(path,path.relative_to(out).as_posix())
    replace_atomic(temp,target)
    print(f'\n{manifest["status"]}\nReview ZIP: {target}',flush=True)


def run_statistics(out, protocol):
    verify_inference_sources(out,protocol)
    jobs = protocol['jobs']
    require({(j['condition'],j['seed']) for j in jobs} == {(c,s) for c in CONDITIONS for s in SEEDS} and len(jobs) == 20,
            'Statistics require all four conditions and all five seeds')
    reference = verify_endpoint_file(out,protocol)
    effect_reference = verify_observability_file(out,protocol,reference)
    predictions,window_predictions,rows,hashes = [],[],[],{}
    # MODEL-MAJOR order, repeated for the second head: [4 L10 models, 4 window models] x 5 seeds.
    # One bootstrap weight matrix resamples identical whole episodes across all 40 prediction vectors.
    for condition in CONDITIONS:
        for seed in SEEDS:
            job = next(j for j in jobs if (j['condition'],j['seed']) == (condition,seed))
            folder = safe_artifact(out,job['relative_dir'])
            marker = read_json(folder/'pipeline_complete.json')
            require(PIPELINE_ARTIFACTS <= set(marker.get('artifact_sha256',{})), 'Incomplete test prediction inventory')
            row = read_completion(folder,'pipeline_complete.json',pipeline_contract(protocol,job))
            require(row is not None and (row['condition'],row['seed']) == (condition,seed) and
                    row.get('step46_protocol_identity') == protocol['identity'] and row.get('dataset_id') == protocol['dataset_id'],
                    'Incomplete/mismatched test pipeline: '+job['key'])
            for name,head,metric,target in (
                ('test_predictions_L10.csv','tst','l10',predictions),
                ('test_predictions_window_common_L10.csv','window','l1_matched_endpoints',window_predictions)):
                path = folder/name
                frame = read_prediction_csv(path)
                measured = validate_predictions(frame,reference,job,protocol,head)
                require(measured['confusion_matrix'] == row[metric]['confusion_matrix'] and
                        abs(measured['macro_f1']-row[metric]['macro_f1']) < 1e-12, 'Stored predictions/metrics disagree: '+name)
                hashes[path.relative_to(out).as_posix()] = digest(path)
                target.append(frame.y_pred.to_numpy(dtype=np.int64))
            hashes[(folder/'pipeline_complete.json').relative_to(out).as_posix()] = digest(folder/'pipeline_complete.json')
            rows.append(row)
    hashes['evaluation_endpoints_L10.csv'] = digest(out/'evaluation_endpoints_L10.csv')
    hashes[OBSERVABILITY_REFERENCE] = digest(out/OBSERVABILITY_REFERENCE)
    contract = object_hash(dict(run=protocol['identity'],inputs=hashes,statistics=statistics_spec()))
    if (out/'statistics_complete.json').exists():
        require(STATISTICS_ARTIFACTS <= set(read_json(out/'statistics_complete.json').get('artifact_sha256',{})),
                'Incomplete statistics artifact inventory')
        completed = read_completion(out,'statistics_complete.json',contract)
        print('Statistics already complete and verified; skip bootstrap',flush=True)
        return completed
    counts,strata,inventory = episode_confusions(reference,np.stack(predictions+window_predictions))
    require(set(strata) == set(range(6)), 'Bootstrap requires all six episode strata')
    require(np.array_equal(counts.sum((0,2,3)),np.full(40,protocol['test_l10'])), 'Episode counts do not sum to test population')
    all_cm = counts.sum(0).reshape(8,5,6,6)
    all_f1 = macro_f1(all_cm); estimates = all_f1.mean(1)
    scores = bootstrap_scores(counts,strata,5,8,BOOTSTRAP_DRAWS,BOOTSTRAP_SEED,BOOTSTRAP_BATCH)
    contrasts = contrast_intervals(scores[:,:4],estimates[:4])
    temporal = temporal_intervals(scores[:,:4],scores[:,4:],estimates[:4],estimates[4:])
    cm = all_cm[:4]
    by_seed = []
    for row in rows:
        flat = {k:v for k,v in row.items() if k not in ('l1_all','l1_matched_endpoints','l10')}
        for endpoint in ('l1_all','l1_matched_endpoints','l10'):
            for metric in ('n','macro_f1','accuracy','weighted_f1'):
                flat[endpoint+'_'+metric] = row[endpoint][metric]
        by_seed.append(flat)
    models = []
    for i,condition in enumerate(CONDITIONS):
        selected = [r for r in rows if r['condition'] == condition]
        item = dict(condition=condition,dataset_id=protocol['dataset_id'],step46_protocol_identity=protocol['identity'],
            encoder_seeds=5,**{k:selected[0][k] for k in
                    ('encoder_parameters','window_model_parameters','tst_parameters','deployment_parameters')})
        for endpoint in ('l1_all','l1_matched_endpoints','l10'):
            values = np.array([r[endpoint]['macro_f1'] for r in selected])
            item[endpoint+'_macro_f1_mean'] = float(values.mean())
            item[endpoint+'_macro_f1_seed_sd_ddof1'] = float(values.std(ddof=1))
        deltas = all_f1[i]-all_f1[i+4]
        lo,hi = np.quantile(scores[:,i],[.025,.975],method='linear')
        wlo,whi = np.quantile(scores[:,i+4],[.025,.975],method='linear')
        item.update(l10_episode_ci95_low=float(lo),l10_episode_ci95_high=float(hi),
            window_common_episode_ci95_low=float(wlo),window_common_episode_ci95_high=float(whi),
            temporal_delta_macro_f1_mean=float(deltas.mean()),temporal_delta_macro_f1_seed_sd_ddof1=float(deltas.std(ddof=1)),
            temporal_delta_ci95_low=temporal[i]['ci95_low'],temporal_delta_ci95_high=temporal[i]['ci95_high'],
            temporal_delta_bonferroni_low=temporal[i]['bonferroni_low'],temporal_delta_bonferroni_high=temporal[i]['bonferroni_high'],
            encoder_budget_limited_seeds=sum(r['encoder_convergence']=='BUDGET_REVIEW_REQUIRED' for r in selected),
            tst_budget_limited_seeds=sum(r['tst_convergence']=='BUDGET_REVIEW_REQUIRED' for r in selected))
        models.append(item)
    den = cm.sum(-1,keepdims=True)
    normalized = np.divide(cm,den,out=np.zeros_like(cm,dtype=float),where=den>0).mean(1)
    confusion = [dict(condition=c,true_label=LABELS[t],predicted_label=LABELS[k],mean_seed_row_normalized=float(normalized[i,t,k]))
                 for i,c in enumerate(CONDITIONS) for t in range(6) for k in range(6)]
    den = cm.sum(-1)+cm.sum(-2)
    cf1 = np.divide(2.*np.diagonal(cm,axis1=-2,axis2=-1),den,out=np.zeros((4,5,6)),where=den>0)
    per_class = [dict(condition=c,label=LABELS[k],f1_seed_mean=float(cf1[i,:,k].mean()),
                      f1_seed_sd_ddof1=float(cf1[i,:,k].std(ddof=1))) for i,c in enumerate(CONDITIONS) for k in range(6)]
    write_csv(out/'metrics_by_seed.csv',by_seed)
    write_csv(out/'model_summary.csv',models); write_json(out/'model_summary.json',models)
    write_csv(out/'paired_comparisons.csv',contrasts); write_json(out/'paired_comparisons.json',contrasts)
    write_csv(out/'temporal_comparisons.csv',temporal); write_json(out/'temporal_comparisons.json',temporal)
    write_csv(out/'seed_mean_row_normalized_confusion.csv',confusion)
    write_csv(out/'per_class_f1.csv',per_class); write_csv(out/'episode_inventory.csv',inventory)
    diagnostic_rows,diagnostic_summary = observability_diagnostics(effect_reference,
        np.stack(predictions+window_predictions),protocol)
    write_csv(out/'observable_effect_diagnostics_by_seed.csv',diagnostic_rows)
    write_csv(out/'observable_effect_diagnostics_summary.csv',diagnostic_summary)
    write_json(out/'observable_effect_diagnostics_scope.json',dict(**observability_spec(),
        dataset_id=protocol['dataset_id'],step46_protocol_identity=protocol['identity'],
        endpoint_counts={s:int(effect_reference.audit_effect_stratum.eq(s).sum()) for s in EFFECT_STRATA}))
    write_array(out/'bootstrap_distribution.npz',conditions=np.array(CONDITIONS),encoder_seeds=np.array(SEEDS),
        seed_mean_macro_f1=scores[:,:4],pretrained_minus_comparators=scores[:,[3]]-scores[:,:3],
        window_common_seed_mean_macro_f1=scores[:,4:],tst_minus_common_window=scores[:,:4]-scores[:,4:])
    lines = ['Step 46: revised representation/evidence-aggregation test comparison','',
        f"L10 population: {protocol['test_l10']:,} endpoints / {protocol['test_l10_episodes']:,} eligible whole episodes.",
        f"Raw test: {protocol['test_windows']:,} windows / {protocol['test_episodes']:,} episodes. No quiet-window filtering.",
        'Four encoders x five encoder seeds; fixed TST/head/shuffle RNG policy from Step 45.',
        'Point estimates average per-seed six-class macro-F1, not probabilities or pooled-confusion F1.',
        'Episode CIs condition on these trained seeds. Sample encoder-seed SD is reported separately.',
        'Primary: pretrained minus alternatives at L10; three-contrast family, 95% / 98.333333% intervals.',
        'Secondary: TST minus window at identical L10 endpoints; four-contrast family, 95% / 98.75% intervals.',
        'Families are separate; there is no simultaneous seven-contrast coverage claim.',
        'Temporal differences include different head capacity/training; interpret alongside pooling/LSTM controls.',
        'No equivalence inference from zero-containing intervals; no architecture necessity or full-convergence claim.',
        'L1-all and L10 have different populations; only matched endpoints support the paired temporal contrast.',
        'Observable-effect diagnostics are descriptive and separate from the primary all-endpoint estimates.',
        'No measured counterfactual projection change does not prove no class information or define an oracle ceiling.',
        'Diagnostic annotations do not enter model inputs, checkpoint selection, exclusion rules or significance tests.',
        'Timing includes feature extraction and I/O; it is not an onboard latency benchmark.','']
    for row in models:
        lines.append(f'{row["condition"]}: L10 mean={row["l10_macro_f1_mean"]:.9f}, seed SD={row["l10_macro_f1_seed_sd_ddof1"]:.9f}')
    temp = out/'analysis_summary.txt.tmp'; temp.write_text('\n'.join(lines)+'\n',encoding='utf-8'); replace_atomic(temp,out/'analysis_summary.txt')
    result = dict(pipelines=20,encoder_seeds=5,bootstrap_draws=BOOTSTRAP_DRAWS,endpoints=protocol['test_l10'],
        episodes=protocol['test_l10_episodes'],raw_test_episodes=protocol['test_episodes'],primary_contrasts=3,secondary_contrasts=4,
        dataset_id=protocol['dataset_id'],step46_protocol_identity=protocol['identity'])
    write_completion(out,'statistics_complete.json',contract,result,sorted(STATISTICS_ARTIFACTS))
    return result


def validate_saved_protocol(out):
    protocol = read_json(out/'protocol.json')
    require(protocol['identity'] == object_hash({k:v for k,v in protocol.items() if k != 'identity'}), 'Evaluation protocol identity changed')
    require(protocol['script_sha256'] == digest(Path(__file__)) and protocol.get('version') == PROTOCOL_VERSION,
            'Saved evaluation used different code/source; do not mix results')
    validate_code_hashes(protocol.get('source_code_sha256'))
    require(protocol.get('observability_diagnostics') == observability_spec(), 'Diagnostic scope changed')
    require(protocol['statistics'] == statistics_spec() and protocol['conditions'] == list(CONDITIONS) and
            protocol['encoder_seeds'] == list(SEEDS), 'Frozen statistics/conditions changed')
    require(protocol['runtime']['numpy'] == np.__version__ and protocol['runtime']['pandas'] == pd.__version__,
            'Use the same NumPy/pandas versions for reproducible statistics')
    require(len(protocol.get('jobs',[])) == 20 and [(j['seed'],j['condition']) for j in protocol['jobs']] ==
            [(s,c) for s in SEEDS for c in CONDITIONS], 'Saved job set changed')
    require(protocol['evaluation_precision'] == 'float32' and protocol['test_l10_episodes'] <= protocol['test_episodes'],
            'Saved precision/population protocol differs')
    return protocol


PIPELINE_ARTIFACTS = {'test_predictions_L1_all.csv','test_predictions_L10.csv',
    'test_predictions_window_common_L10.csv','test_window_logits.npy','test_tst_logits_L10.npy',
    'test_embeddings.npy','metrics.json','validation_canary.json'}
STATISTICS_ARTIFACTS = {'metrics_by_seed.csv','model_summary.csv','model_summary.json',
    'paired_comparisons.csv','paired_comparisons.json','temporal_comparisons.csv','temporal_comparisons.json',
    'seed_mean_row_normalized_confusion.csv','per_class_f1.csv','episode_inventory.csv',
    'bootstrap_distribution.npz','analysis_summary.txt','observable_effect_diagnostics_by_seed.csv',
    'observable_effect_diagnostics_summary.csv','observable_effect_diagnostics_scope.json'}
FINAL_STATUS = 'FROZEN_TEST_AND_BOOTSTRAP_COMPLETE'


def valid_sha256(value):
    return isinstance(value,str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_code_hashes(hashes):
    require(isinstance(hashes,dict) and set(hashes) == {'43','44','45'} and
            all(valid_sha256(v) for v in hashes.values()), 'Invalid source code chain')


def source_code_hashes(record):
    p = record['protocol']
    hashes = {'43':p['engine_sha256'],'44':p['adapter_sha256'],'45':p['script_sha256']}
    validate_code_hashes(hashes)
    return hashes


def resolve_code_paths(root, code_hashes):
    validate_code_hashes(code_hashes)
    paths = {}
    for role,expected in code_hashes.items():
        matches = [p for p in sorted(Path(root).glob(role+'*.py')) if p.is_file() and digest(p) == expected]
        require(matches, 'Place the delivered revised Step '+role+' Python file beside this script; filename suffixes are allowed')
        paths[role] = matches[0]
    return paths


def recorded_path(value, root, engine):
    path = Path(value).expanduser()
    return path.resolve() if path.exists() else engine.mapped_path(value,root)


def validate_statistics_plan(plan):
    expected = dict(contrasts=[CONDITIONS[3]+' minus '+c for c in CONDITIONS[:3]],
        secondary_within_encoder_temporal_contrasts=[c+': L10 TST minus window classifier at common L10 endpoints' for c in CONDITIONS],
        bootstrap_draws=BOOTSTRAP_DRAWS,marginal_confidence=.95,three_contrast_bonferroni_confidence=1-.05/3,
        secondary_four_contrast_bonferroni_confidence=1-.05/4,equivalence_claim=False,test_used_for_selection=False,
        primary_endpoint='macro_f1_at_fixed_L10_test_endpoints')
    for key,value in expected.items():
        require(plan.get(key) == value, 'The test analysis differs from the Step 45 plan: '+key)
    for key in ('expected_test_windows','expected_test_l10','expected_test_episodes','expected_test_l10_episodes'):
        require(type(plan.get(key)) is int and plan[key] > 0, 'Invalid declared population: '+key)
    require(plan['expected_test_l10'] <= plan['expected_test_windows'] and
            plan['expected_test_l10_episodes'] <= plan['expected_test_episodes'] and bool(plan.get('dataset_id')),
            'Inconsistent declared test populations')


def canary_count(total, batch_size):
    require(type(total) is int and total > 0 and type(batch_size) is int and batch_size > 0, 'Invalid canary batch dimensions')
    return min(total, ((min(total,CANARY_N)+batch_size-1)//batch_size)*batch_size)


def prediction_provenance(job, protocol, head_kind, population):
    require(head_kind in ('window','tst'), 'Unknown prediction head')
    return dict(dataset_id=protocol['dataset_id'],source45_protocol_identity=protocol['source_protocol_identity'],
        step46_protocol_identity=protocol['identity'],condition=job['condition'],encoder_seed=job['seed'],
        head_seed=job['protocol']['window_head_seed' if head_kind == 'window' else 'tst_seed'],
        head_kind=head_kind,evaluation_population=population)


def read_prediction_csv(path):
    return pd.read_csv(path,encoding='utf-8-sig',keep_default_na=False,
        dtype={k:str for k in ('scenario_id','episode_id','episode_type','timestamp','label','dataset_id',
                              'source45_protocol_identity','step46_protocol_identity')})


def verify_test_pipelines(out, protocol):
    rows = []
    for job in protocol['jobs']:
        folder = safe_artifact(out,job['relative_dir'])
        path = folder/'pipeline_complete.json'
        if not path.exists():continue
        require(PIPELINE_ARTIFACTS <= set(read_json(path).get('artifact_sha256',{})), 'Incomplete test artifact inventory: '+job['key'])
        row = read_completion(folder,'pipeline_complete.json',pipeline_contract(protocol,job))
        require((row['seed'],row['condition']) == (job['seed'],job['condition']) and
            row.get('dataset_id') == protocol['dataset_id'] and row.get('step46_protocol_identity') == protocol['identity'] and
            row.get('evaluation_precision') == 'float32', 'Completed test provenance differs')
        require(row['l1_all']['n'] == protocol['test_windows'] and row['l10']['n'] ==
                row['l1_matched_endpoints']['n'] == protocol['test_l10'], 'Completed test population differs')
        require(read_json(folder/'metrics.json') == row, 'Metrics/completion record mismatch')
        rows.append(row)
    return rows


def verify_final_outputs(out, manifest, protocol):
    require(manifest.get('status') == FINAL_STATUS and manifest.get('protocol_identity') == protocol['identity'] and
            manifest.get('protocol_version') == PROTOCOL_VERSION and manifest.get('completed_pipelines') == 20,
            'Incomplete final manifest')
    required = STATISTICS_ARTIFACTS | {'protocol.json','evaluation_endpoints_L10.csv','statistics_complete.json',
                                     'test_input_audit.json','train_field_vocabulary.json',
                                     OBSERVABILITY_REFERENCE,SOURCE_VERIFICATION_FILE}
    require(required <= set(manifest.get('output_sha256',{})), 'Incomplete final artifact inventory')
    for name,expected in manifest['output_sha256'].items():
        require(digest(safe_artifact(out,name)) == expected, 'Final artifact changed: '+name)
    verify_inference_sources(out,protocol)


def verify_sources_unchanged(source, record, env):
    require(digest(source/'run_manifest.json') == record['manifest_sha256'] and
            read_json(source/'protocol.json') == record['protocol'], 'Step 45 source changed during evaluation')
    for name,expected in {**record['manifest']['output_sha256'],**record['weights']}.items():
        require(digest(safe_artifact(source,name)) == expected, 'Step 45 artifact changed during evaluation: '+name)
    for path,expected in env['input_hashes'].items():
        require(digest(path) == expected, 'Upstream input changed during evaluation: '+path)
    require(digest(env['test_path']) == env['test_sha256'], 'Test data changed during evaluation')
    require(digest(env['observability_path']) == env['observability_sha256'], 'Observable-effect source changed during evaluation')
    for name,expected in record['protocol']['base_hashes'].items():
        require(digest(env['base']/name) == expected, 'Base bundle changed during evaluation')
    for name,expected in env['source_cache']['artifact_sha256'].items():
        require(digest(safe_artifact(env['cache_path'],name)) == expected, 'Source cache changed during evaluation')
    require(read_json(env['cache_path']/'complete.json') == env['source_cache'], 'Source cache record changed')
    for role,path in env['code_paths'].items():
        require(digest(path) == source_code_hashes(record)[role], 'Dependency code changed during evaluation')
    require(digest(Path(__file__)) == env['evaluation_protocol']['script_sha256'], 'Step 46 code changed during evaluation')


def source_verification_contract(protocol):
    return object_hash(dict(run=protocol['identity'],source45=protocol['source_protocol_identity'],
        source_manifest=protocol['source_manifest_sha256'],checkpoints=protocol['checkpoint_sha256'],
        stage='all_20_predictions_then_source_verification'))


def source_verification_artifacts(protocol):
    return {'evaluation_endpoints_L10.csv',OBSERVABILITY_REFERENCE} | {
        job['relative_dir']+'/pipeline_complete.json' for job in protocol['jobs']}


def source_verification_result(protocol):
    return dict(status='SOURCE_INPUTS_VERIFIED',pipelines=20,step46_protocol_identity=protocol['identity'],
        source45_protocol_identity=protocol['source_protocol_identity'],source_manifest_sha256=protocol['source_manifest_sha256'])


def verify_inference_sources(out, protocol):
    path = out/SOURCE_VERIFICATION_FILE
    require(path.is_file(), 'Post-inference source verification is missing; resume the normal command before --statistics-only')
    require(set(read_json(path).get('artifact_sha256',{})) == source_verification_artifacts(protocol),
            'Post-inference source verification inventory differs')
    result = read_completion(out,SOURCE_VERIFICATION_FILE,source_verification_contract(protocol))
    require(result == source_verification_result(protocol), 'Post-inference source verification record differs')
    return result


def seal_inference_sources(out, protocol, source, record, env):
    require(len(verify_test_pipelines(out,protocol)) == 20, 'Source verification requires all 20 completed predictions')
    verify_observability_file(out,protocol)
    # Persist only AFTER these checks pass. Statistics-only cannot bypass a failed
    # final source check merely because the last inference pipeline was written.
    verify_sources_unchanged(source,record,env)
    if (out/SOURCE_VERIFICATION_FILE).exists():
        verify_inference_sources(out,protocol)
    else:
        write_completion(out,SOURCE_VERIFICATION_FILE,source_verification_contract(protocol),
            source_verification_result(protocol),sorted(source_verification_artifacts(protocol)))


def temporal_intervals(scores, window_scores, estimates, window_estimates):
    require(np.shape(scores) == np.shape(window_scores) and np.shape(scores)[1] == 4, 'Expected paired four-encoder scores')
    rows = []
    def direction(lo,hi):
        return 'tst_higher' if lo > 0 else 'window_higher' if hi < 0 else 'zero_included_not_equivalence'
    for i,c in enumerate(CONDITIONS):
        delta = scores[:,i]-window_scores[:,i]
        lo,hi = np.quantile(delta,[.025,.975],method='linear')
        blo,bhi = np.quantile(delta,[.05/8,1-.05/8],method='linear')
        point = float(estimates[i]-window_estimates[i])
        rows.append(dict(condition=c,reference='L10_TST',comparator='window_at_common_L10_endpoints',
            comparison_family='secondary_four_temporal_contrasts',delta_macro_f1=point,delta_percentage_points=100*point,
            ci95_low=float(lo),ci95_high=float(hi),bonferroni_confidence=1-.05/4,
            bonferroni_low=float(blo),bonferroni_high=float(bhi),ci95_interpretation=direction(lo,hi),
            familywise_interpretation=direction(blo,bhi)))
    return rows


def verify_endpoint_file(out, protocol):
    path = out/'evaluation_endpoints_L10.csv'
    require(path.is_file(), 'Evaluation endpoint reference is missing')
    frame = read_prediction_csv(path)
    require(len(frame) == protocol['test_l10'] and frame.episode_id.nunique() == protocol['test_l10_episodes'] and
        object_hash(frame.to_dict(orient='records')) == protocol['test_endpoint_identity'], 'Evaluation endpoints changed')
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root',type=Path,help='Project directory; defaults to this script directory')
    parser.add_argument('--source-run',type=Path,help='Full completed revised Step 45 run')
    parser.add_argument('--base-model-dir',type=Path,help='Original local base config/tokenizer bundle')
    parser.add_argument('--audit-run',type=Path,help='Relocated original Step 42 audit folder')
    parser.add_argument('--audit-script',type=Path,help='Explicit revised Step 42 Python file')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only',action='store_true',help='Read-only source/hash/runtime checks; no inference or output run')
    mode.add_argument('--statistics-only',action='store_true',help='Verify stored predictions and compute/reuse statistics; no CUDA/models')
    args = parser.parse_args(argv)
    root = (args.project_root or Path(__file__).resolve().parent).expanduser().resolve()
    source = (root/(args.source_run or SOURCE_REL)).resolve(); out = (root/OUTPUT_REL).resolve()
    try:
        require(root in out.parents and out != source and source not in out.parents and out not in source.parents,
                'Output overlaps original inputs/source')
        if args.statistics_only:
            require((out/'protocol.json').is_file(), 'No Step 46 evaluation exists; run inference first')
            with RunLock(out/'.run.lock'):
                protocol = validate_saved_protocol(out)
                manifest = read_json(out/'run_manifest.json')
                require(manifest.get('protocol_identity') == protocol['identity'] and
                        manifest.get('protocol_version') == PROTOCOL_VERSION, 'Run/protocol mismatch')
                require(len(verify_test_pipelines(out,protocol)) == 20, 'Complete all 20 test pipelines before statistics')
                verify_inference_sources(out,protocol)
                verify_endpoint_file(out,protocol)
                if manifest.get('status') == FINAL_STATUS:
                    verify_final_outputs(out,manifest,protocol)
                    print('COMPLETE: existing results verified; no files rewritten.',flush=True)
                    return 0
                result = run_statistics(out,protocol)
                (out/'failure_traceback.txt').unlink(missing_ok=True)
                for key in ('error_type','error'):
                    manifest.pop(key,None)
                manifest.update(status=FINAL_STATUS,completed_pipelines=20,statistics=result,
                                finished_at_utc=utc(),readiness='READY_FOR_RESULT_REVIEW')
                finish_run(out,manifest)
            return 0
        print('Step 46: verifying the revised Step 45 source and selected checkpoints...',flush=True)
        record = read_source(source)
        resolved = lambda p:(root/p).resolve() if p else None
        env = prepare_environment(root,source,record,resolved(args.base_model_dir),resolved(args.audit_run),resolved(args.audit_script))
        if args.check_only:
            print('CHECK_COMPLETE: selected checkpoints, declared cohort, preprocessing and runtime verified.',flush=True)
            print('No model inference/training, test metrics, new run or downloads.',flush=True)
            return 0
        data,input_audit = load_test(root,env,record)
        protocol = freeze_protocol(out,root,source,record,env,write=False)
        if out.exists() and not (out/'protocol.json').exists():
            require(not any(p.name != '.run.lock' for p in out.iterdir()), 'Output has files without protocol; preserve for inspection')
        out.mkdir(parents=True,exist_ok=True)
        with RunLock(out/'.run.lock'):
            protocol = freeze_protocol(out,root,source,record,env,write=False)
            old = read_json(out/'run_manifest.json') if (out/'run_manifest.json').exists() else {}
            if old:
                require(old.get('protocol_identity') == protocol['identity'] and old.get('protocol_version') == PROTOCOL_VERSION,
                        'Resume manifest/protocol mismatch')
            # Verify prior completed work before rewriting any run records.
            rows = verify_test_pipelines(out,protocol)
            if (out/'evaluation_endpoints_L10.csv').exists():verify_endpoint_file(out,protocol)
            elif rows:raise ValueError('Completed pipelines exist but their endpoint reference is missing')
            if old.get('status') == FINAL_STATUS:
                require(len(rows) == 20, 'Final run is missing pipelines')
                verify_final_outputs(out,old,protocol)
                print('COMPLETE: existing results verified; no inference/bootstrap or file rewrites.',flush=True)
                return 0
            audit_record = dict(dataset_id=protocol['dataset_id'],step46_protocol_identity=protocol['identity'],
                test_sha256=env['test_sha256'],preprocessing=input_audit,episode_split_disjoint=True,
                windows=protocol['test_windows'],l10=protocol['test_l10'],episodes=protocol['test_episodes'],
                l10_eligible_episodes=protocol['test_l10_episodes'])
            for name,value in (('test_input_audit.json',audit_record),('train_field_vocabulary.json',record['vocabulary'])):
                if (out/name).exists():require(read_json(out/name) == value, 'Saved input record changed: '+name)
            needed = (20-len(rows))*(protocol['test_windows']*(768*4+2*1024)+1024**2)+512*1024**2
            require(shutil.disk_usage(out).free >= needed,
                    f'Insufficient free disk space; need about {needed/1024**3:.2f} GiB for remaining outputs')
            freeze_protocol(out,root,source,record,env)
            env['evaluation_protocol'] = protocol
            manifest = dict(script_number=46,protocol_version=PROTOCOL_VERSION,status='RUNNING',
                dataset_id=protocol['dataset_id'],started_at_utc=old.get('started_at_utc',utc()),last_invoked_at_utc=utc(),
                protocol_identity=protocol['identity'],source_run=str(source),source_manifest_sha256=record['manifest_sha256'],
                test_data_read=True,test_evaluation=True,training=False,planned_pipelines=20,completed_pipelines=len(rows),
                readiness='TEST_INFERENCE_IN_PROGRESS')
            write_json(out/'run_manifest.json',manifest)
            try:
                if not (out/'evaluation_endpoints_L10.csv').exists():write_csv(out/'evaluation_endpoints_L10.csv',data['metadata'])
                for name,value in (('test_input_audit.json',audit_record),('train_field_vocabulary.json',record['vocabulary'])):
                    if not (out/name).exists():write_json(out/name,value)
                rows = run_queue(out,protocol,record,source,env,data,verified_rows=rows)
                require(len(rows) == 20, 'Do not report a partial test comparison')
                effects = load_observability_reference(root,protocol,env,data)
                if (out/OBSERVABILITY_REFERENCE).exists():
                    require(verify_observability_file(out,protocol).equals(effects), 'Saved diagnostic annotations changed')
                else:
                    write_csv(out/OBSERVABILITY_REFERENCE,effects)
                seal_inference_sources(out,protocol,source,record,env)
                manifest.update(status='TEST_INFERENCE_COMPLETE',completed_pipelines=20,readiness='BOOTSTRAP_PENDING')
                write_json(out/'run_manifest.json',manifest)
                result = run_statistics(out,protocol)
                (out/'failure_traceback.txt').unlink(missing_ok=True)
                manifest.update(status=FINAL_STATUS,statistics=result,finished_at_utc=utc(),readiness='READY_FOR_RESULT_REVIEW')
                finish_run(out,manifest)
            except BaseException as exc:
                manifest.update(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',
                    error_type=type(exc).__name__,error=str(exc),readiness='RESUME_SAME_COMMAND')
                if (out/'progress.json').exists():manifest['completed_pipelines'] = read_json(out/'progress.json')['completed_pipelines']
                (out/'failure_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
                try:finish_run(out,manifest)
                except Exception as packaging_error:print('Could not package review files: '+str(packaging_error),file=sys.stderr)
                print('Resume with the SAME command; completed pipelines will be verified and skipped.',flush=True)
                return 130 if isinstance(exc,KeyboardInterrupt) else 1
        return 0
    except Exception as exc:
        print('Step 46 stopped: '+str(exc),file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
