#!/usr/bin/env python3
"""Step 52: temporal-statistics MLP on the shared_nominal_v2 representation.

Keep beside the current revised 47/11/13/14/15 scripts. Run this file directly;
the identical command resumes. --audit-only verifies files/cohorts without torch.
Finish the current Step 47 first. No Step 51, support ZIP, or fixed old hash is used.

One new family, TemporalStats_MLP: concatenate mean, population std (ddof=0),
minimum and maximum of each embedding coordinate over every causal L10 window.
Reuse the frozen Step 8/9 supervised fine-tuned 22-field CLS representation,
Step 13 architecture/training recipe, and all three existing Step 47 Missing-8
embedding caches. Train all ten seeds on clean train/validation BEFORE any new
test inference. Retain every endpoint, including quiet attack contexts.

Four declared contrasts: TST and the clean-validation-selected pooling
family minus TemporalStats_MLP, on clean and mean-of-three Missing-8 scores.
20,000 paired class-stratified whole-episode bootstrap draws, Bonferroni family
size four. CIs condition on these fitted heads, encoder, data and three masks;
no head/mask resampling, score target, forced ranking, or equivalence claim.
This retrospective extension does not claim equal model capacity/compute,
external validity, or independent encoder replications.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
import zipfile
import zlib
import numpy as np
import pandas as pd

VERSION = 'shared_nominal_temporal_statistics_v2'
ENGINE_VERSION = 'shared_nominal_uniform_missing8_aggregation_v2'
OUT_REL = 'outputs/shared_nominal_v2/52_temporal_statistics/run_temporal_statistics_v2'
SEEDS = tuple(range(20260610, 20260620))
CONDITIONS = ('clean', 'missing8_20261001', 'missing8_20261002', 'missing8_20261003')
LABELS = ('NORMAL', 'RF_LINK_JAMMING', 'GNSS_SPOOFING', 'UNAUTHORIZED_TELECOMMAND_INJECTION',
          'TELECOMMAND_REPLAY', 'GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE')
TRAIN = dict(max_epochs=30, patience=7, batch_size=256, learning_rate=7e-5,
    weight_decay=5e-4, grad_clip_norm=1., hidden_dim=256, mlp_layers=2, dropout=.25,
    use_input_layernorm=True, use_standardization=False, use_amp=True)
MODEL = dict(input_dim=3072, hidden_dim=256, num_classes=6, mlp_layers=2,
             dropout=.25, use_input_layernorm=True)
N_BOOTSTRAP = 20000
BOOTSTRAP_SEED = 2026100152
BOOTSTRAP_BATCH = 100
e = None  # Load only after CLI parsing; --help never imports a neural backend.


def require(ok, message):
    if not bool(ok): raise ValueError(message)


def load_engine(root, explicit=None):
    paths = [Path(explicit)] if explicit else sorted(root.glob('47*.py'))
    unique = {hashlib.sha256(p.read_bytes()).hexdigest(): p.resolve() for p in paths if p.is_file()}
    require(len(unique) == 1, 'Place one current Step 47 beside Step 52, or use --step47-code.')
    path = next(iter(unique.values()))
    spec = importlib.util.spec_from_file_location('step52_current47', path)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    require(getattr(module, 'PROTOCOL_VERSION', None) == ENGINE_VERSION,
            'Step 52 requires the current shared_nominal_v2 Step 47.')
    require(tuple(module.SEEDS) == SEEDS and tuple(module.CONDITIONS) == CONDITIONS
            and tuple(module.LABELS) == LABELS, 'Step 47 seeds/conditions/classes differ')
    return module, path


def classes(value, shape, name):
    a = np.asarray(value)
    require(a.shape == shape and a.dtype.kind in 'iu' and np.isin(a, range(6)).all(),
            'Invalid integer classes/shape: ' + name)
    return a


def authenticated_bytes(path, expected):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, 'Input changed: ' + str(path))
    return raw


def same_frame(actual, expected, name):
    require(actual.reset_index(drop=True).astype(str).equals(expected.reset_index(drop=True).astype(str)),
            'Endpoint alignment differs: ' + name)


def audit_step47(root, folder, context, engine_path, data):
    """Authenticate current completed evidence, selection and the three caches."""
    helper = context['helper']; observed = context['observed']
    manifest = helper.load_json(folder / 'run_manifest.json', observed)
    require(manifest.get('status') == 'MISSING8_AGGREGATION_COMPLETE', 'Finish current Step 47 before Step 52')
    files = manifest.get('output_sha256', {})
    required = set(e.FINAL_FILES) | {f'cache/{c}/embeddings_complete.json' for c in CONDITIONS[1:]}
    require(required <= set(files), 'Step 47 completion misses required evidence/caches')
    helper.verify_selected(folder, files, list(files), observed)
    protocol = e.read_json(folder / 'protocol.json')
    require(protocol.get('identity') == e.object_hash({k:v for k,v in protocol.items() if k != 'identity'})
            == manifest.get('protocol_identity'), 'Step 47 protocol identity differs')
    require(protocol.get('version') == ENGINE_VERSION and protocol.get('script_sha256') == e.digest(engine_path),
            'Step 47 code differs from the completed run')
    require(protocol.get('audit_script_sha256') == e.digest(context['helper_path'])
            and e.read_json(folder / 'input_audit.json') == protocol['input_audit'],
            'Step 47 audit source/record differs')
    require(protocol['source_contract'] == context['source_contract']
            and protocol['checkpoint_contracts'] == context['heads']
            and protocol['source_hashes'] == context['code_hashes']
            and protocol['runs'] == [list(r) for r in e.RUNS]
            and protocol['head_seeds'] == list(SEEDS) and protocol['conditions'] == list(CONDITIONS)
            and protocol['fields'] == list(e.FIELDS) and protocol['labels'] == list(LABELS)
            and protocol.get('test_used_for_selection') is False, 'Step 47 data/models/protocol differ')
    for name, sha in protocol['input_audit'].items(): helper.track_file(Path(name), observed, sha)
    selection = e.read_json(folder / 'validation_selection.json')
    rows = pd.read_csv(folder / 'validation_metrics.csv').to_dict('records')
    pool = e.choose_pool(rows)
    # Recompute the saved pooling validation metrics from authenticated clean
    # validation predictions; no test result participates in the selection.
    for family in e.POOLS:
        part = [r for r in rows if r['family'] == family]
        for r in part:
            actual = e.metrics(data['val']['y'], data['val']['references'][e.run_key(family, int(r['seed']))].argmax(1))['macro_f1']
            require(abs(actual - r['macro_f1']) <= 1e-12, 'Step 47 validation metric differs from saved predictions')
        require(abs(selection['candidate_pool_mean_validation_f1'][family] - np.mean([r['macro_f1'] for r in part])) <= 1e-12,
                'Step 47 pooling selection summary differs')
    require(selection.get('selected_pool') == pool and selection.get('all_ten_head_seeds_retained') is True
            and selection.get('selection_data') == 'clean validation only; no missing-mask or test metrics',
            'Step 47 validation selection differs')
    same_frame(helper.read_csv(folder / 'evaluation_endpoints_L10.csv'), data['test']['meta'], 'Step 47 test')
    for split in ('val', 'test'):
        require(protocol['populations'][split]['endpoint_identity'] == e.object_hash(data[split]['meta'].to_dict(orient='records')),
                'Step 47 endpoint identity differs: ' + split)
    for condition in CONDITIONS[1:]:
        cache = folder / 'cache' / condition
        contract = e.object_hash(dict(protocol=protocol['identity'], stage=condition))
        done = e.read_completion(cache, 'embeddings_complete.json', contract)
        record = e.read_json(cache / 'embeddings_complete.json')
        require(done is not None and {'embeddings.npy', 'mask_audit.json'} <= set(record['artifact_sha256']),
                'Incomplete Missing-8 cache')
        helper.verify_selected(cache, record['artifact_sha256'], list(record['artifact_sha256']), observed)
        _, expected = e.apply_mask(data['test']['frame'], e.make_mask(len(data['test']['frame']), int(condition.split('_')[1])))
        require(all(done.get(k) == v for k,v in dict(mask_seed=int(condition.split('_')[1]), **expected).items()),
                'Step 47 Missing-8 realization differs')
        emb = np.load(cache / 'embeddings.npy', mmap_mode='r', allow_pickle=False)
        require(emb.shape == (len(data['test']['frame']), 768) and emb.dtype == np.float32 and np.isfinite(emb).all(),
                'Invalid Missing-8 embeddings')
    result = dict(folder=str(folder), manifest=manifest, protocol=protocol, selection=selection)
    reference_predictions(result, data['test'])
    return result


def load_train(root, context, data):
    """Add the authenticated Step 9/10 train artifacts to Step 47's cohort audit."""
    helper = context['helper']; source = context['source_contract']; observed = context['observed']
    helper.verify_selected(root / e.EMB_REL, source['step9_artifacts'],
        ['train_window_embeddings.npz', 'train_window_metadata.csv'], observed)
    helper.verify_selected(root / e.SEQ_REL, source['step10_artifacts'],
        ['train_tst_sequences_L10.npz', 'train_tst_sequence_metadata_L10.csv',
         'train_tst_sequence_observable_effects_L10.csv'], observed)
    frame = e.read_records(root / e.DATA_REL / 'splits/train_observable.csv')
    require(list(frame) == list(e.META + e.FIELDS) and set(frame.label) == set(LABELS), 'Train schema/classes differ')
    for name in ('episode_step', 'window_id'): frame[name] = helper.integers(frame[name], name)
    require(not frame[list(e.FIELDS)].astype(str).apply(lambda c:c.str.strip().eq('')).any().any(), 'Empty train fields')
    require(frame.groupby('episode_id').label.nunique().eq(1).all(), 'Mixed train episode labels')
    require(not set(frame.episode_id).intersection(set(data['val']['frame'].episode_id) | set(data['test']['frame'].episode_id)),
            'Train/validation/test episode overlap')
    with np.load(root / e.EMB_REL / 'train_window_embeddings.npz', allow_pickle=False) as z:
        emb = z['embeddings']; row = z['row_indices']; labels = z['label_ids']
    require(emb.shape == (len(frame), 768) and emb.dtype == np.float32 and np.isfinite(emb).all(), 'Invalid train embeddings')
    classes(labels, (len(frame),), 'train embedding labels')
    require(np.array_equal(row, np.arange(len(frame))) and np.array_equal(labels, frame.label.map(dict(zip(LABELS,range(6))))),
            'Train embedding alignment differs')
    wm = e.read_records(root / e.EMB_REL / 'train_window_metadata.csv')
    same_frame(wm[list(e.META)], frame[list(e.META)], 'train windows')
    with np.load(root / e.SEQ_REL / 'train_tst_sequences_L10.npz', allow_pickle=False) as z: ix = z['sequence_indices']; y = z['y']
    classes(y, (len(ix),), 'train sequence labels'); e.validate_indices(frame, ix, y)
    counts = context['build']['counts']['train']
    detail = e.read_json(root / e.SEQ_REL / 'sequence_build_report_L10.json')['splits']['train']
    require(len(frame) == counts['rows'] == detail['input_windows'] and frame.episode_id.nunique() == counts['episodes']
            and len(ix) == counts['sequences_stride1']['10'] == detail['num_sequences'], 'Wrong train population')
    meta = helper.read_csv(root / e.SEQ_REL / 'train_tst_sequence_metadata_L10.csv')
    require(len(meta) == len(ix) and np.array_equal(meta.sequence_id, np.arange(len(ix)))
            and np.array_equal(meta.sequence_start_row, ix[:,0]) and np.array_equal(meta.sequence_end_row, ix[:,-1]),
            'Train sequence row order differs')
    end = frame.iloc[ix[:,-1]]
    require(np.array_equal(meta.label_id, y) and np.array_equal(meta.label, end.label)
            and np.array_equal(meta.episode_id, end.episode_id) and np.array_equal(meta.sequence_end_step, end.episode_step)
            and meta.source_split.eq('train').all() and meta.sequence_length.eq(10).all() and set(y) == set(range(6)),
            'Train endpoint metadata differs')
    require(hashlib.sha256(np.asarray(ix[:,-1], dtype='<i8').tobytes()).hexdigest() == detail['end_row_indices_sha256'],
            'Train endpoint ordering differs from Step 10')
    effects = helper.validate_effect_frame(helper.read_csv(root / e.SEQ_REL / 'train_tst_sequence_observable_effects_L10.csv'), meta, 10)
    helper.recheck_inputs(observed)
    return dict(frame=frame, embeddings=emb, indices=ix, y=y, meta=meta, effects=effects)


def reference_predictions(previous, test):
    folder = Path(previous['folder']); hashes = previous['manifest']['output_sha256']
    raw = authenticated_bytes(folder / 'predictions.npz', hashes['predictions.npz'])
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        old = classes(z['predictions'], (4, len(e.RUNS), len(test['y'])), 'Step 47 predictions')
        require(np.array_equal(z['y'], test['y']) and tuple(z['conditions']) == CONDITIONS
                and tuple(z['families']) == tuple(f for f,s in e.RUNS)
                and np.array_equal(z['seeds'], [s if s is not None else -1 for f,s in e.RUNS])
                and z['dataset_id'].item() == previous['protocol']['source_contract']['dataset_id']
                and z['step47_protocol_identity'].item() == previous['protocol']['identity'], 'Step 47 prediction contract differs')
        result = np.stack([np.stack([old[:,e.RUNS.index((f,s))] for s in SEEDS], axis=1)
                           for f in ('TST', previous['selection']['selected_pool'])], axis=1)
    for fi, family in enumerate(('TST', previous['selection']['selected_pool'])):
        for si, seed in enumerate(SEEDS):
            require(np.array_equal(result[0,fi,si], test['references'][e.run_key(family,seed)].argmax(1)),
                    'Step 47 clean predictions differ from original saved heads')
    return result


def statistics_block(embeddings, indices):
    indices = np.asarray(indices)
    require(indices.ndim == 2 and indices.shape[1] == 10 and indices.dtype.kind in 'iu', 'Expected integer L10 indices')
    require(indices.size and indices.min() >= 0 and indices.max() < len(embeddings), 'Out-of-range sequence indices')
    x = np.asarray(embeddings[indices], dtype=np.float32)
    require(x.ndim == 3 and x.shape[-1] == 768 and np.isfinite(x).all(), 'Invalid sequence embeddings')
    result = np.concatenate((x.mean(1,dtype=np.float32), x.std(1,ddof=0,dtype=np.float32), x.min(1), x.max(1)), axis=1)
    require(np.isfinite(result).all(), 'Nonfinite temporal statistics')
    return result


def feature_cache(out, name, identity, embeddings, indices):
    folder = out / 'features' / name; folder.mkdir(parents=True, exist_ok=True)
    contract = e.object_hash(dict(protocol=identity, features=name, recipe='mean_std0_min_max_fp32_L10'))
    done = e.read_completion(folder, 'complete.json', contract)
    if done is None:
        temp = folder / 'features.npy.tmp'
        x = np.lib.format.open_memmap(temp, mode='w+', dtype=np.float32, shape=(len(indices),3072))
        for lo in range(0,len(indices),512): x[lo:lo+512] = statistics_block(embeddings,indices[lo:lo+512])
        x.flush(); del x; e.replace_atomic(temp, folder / 'features.npy')
        e.write_completion(folder,'complete.json',contract,dict(shape=[len(indices),3072]),['features.npy'])
    # Copy-on-write mmap allows torch.from_numpy without granting persistent writes.
    x = np.load(folder / 'features.npy', mmap_mode='c', allow_pickle=False)
    require(x.shape == (len(indices),3072) and x.dtype == np.float32 and np.isfinite(x).all(), 'Invalid cached features')
    return x


def check_pool_configuration(module, context, family):
    for name, value in TRAIN.items(): require(getattr(module, name.upper()) == value, 'Step 13 source training setting differs: ' + name)
    require(module.LABEL_SMOOTHING == .03 and module.NUM_WORKERS == 0, 'Step 13 loss/loader configuration differs')
    for seed in SEEDS:
        head = context['heads'][e.run_key(family, seed)]
        require(head['train_config'] == TRAIN and head['model_config'] == {**MODEL, 'input_dim':1536 if family == 'MeanMax_MLP' else 768},
                'Step 13 saved configuration differs from the declared comparison recipe')
        path = Path(head['checkpoint']); context['helper'].track_file(path, context['observed'], context['observed'][str(path.resolve())])
        ck = e.read_checkpoint(path); e.validate_checkpoint(ck, family, seed, context)


def training_backend(context, selected):
    path = context['code_paths']['pool']
    require(e.digest(path) == context['code_hashes']['pool'], 'Step 13 source changed')
    module = e.import_script(path, 'step52_current13'); torch = module.torch
    require(torch.cuda.is_available(), 'Use the original CUDA environment; full CPU training is disabled')
    torch.cuda.set_device(0)
    record = context['training_records']['pool']['runtime']
    torch.set_num_threads(record.get('threads', record.get('torch_num_threads')))
    # Step 13 explicitly enables benchmark, and inherits all other backend flags.
    # Record those flags and reject drift on resume; do not impose old Step 52 flags.
    torch.backends.cudnn.benchmark = True
    runtime = e.evaluation_runtime(context)
    runtime.update(cudnn_benchmark=torch.backends.cudnn.benchmark)
    module.progress_bar = lambda iterable, total=None, desc=None: iterable
    check_pool_configuration(module, context, selected)
    return module, torch.device('cuda:0'), runtime


def job_identity(identity, seed): return e.object_hash(dict(protocol=identity, head_seed=seed))


def training_gate(out, identity):
    index = e.read_json(out / 'training_index.json')
    require(set(index) == set(map(str,SEEDS)), 'All ten clean-trained heads must complete before test inference')
    for seed in SEEDS:
        folder = out / f'seed_{seed}'
        require(e.digest(folder / 'training_complete.json') == index[str(seed)], 'Training index hash differs')
        done = e.read_completion(folder,'training_complete.json',job_identity(identity,seed))
        require(done is not None and done['seed'] == seed, 'Unfinished or incorrect seed')
        record = e.read_json(folder / 'training_complete.json')
        require({'best.pt','history.csv','validation_predictions.npz'} <= set(record['artifact_sha256']), 'Training evidence incomplete')
    return index


def prediction_check(pred, y):
    classes(y, (len(y),), 'truth')
    require(np.array_equal(pred['y_true'], y), 'Prediction truth differs')
    logits = np.asarray(pred['logits'])
    require(logits.shape == (len(y),6) and logits.dtype.kind == 'f' and np.isfinite(logits).all(), 'Invalid prediction logits')
    p = classes(pred['y_pred'], (len(y),), 'predictions')
    require(np.array_equal(p, logits.argmax(1)), 'Predictions disagree with logits')
    return p


def cpu_state(model): return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def save_checkpoint(torch, path, payload):
    temp = path.with_name(path.name + '.tmp'); torch.save(payload,temp); e.replace_atomic(temp,path)


def restore_rng(torch, checkpoint):
    torch.set_rng_state(checkpoint['torch_rng']); torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    random.setstate(checkpoint['python_rng']); n = checkpoint['numpy_rng']
    np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))


def resume_checkpoint(torch, folder, contract):
    pointer = folder / 'resume.json'
    if not pointer.exists(): return None
    saved = e.read_json(pointer)
    require(saved.get('identity') == e.object_hash({k:v for k,v in saved.items() if k != 'identity'})
            and saved.get('contract') == contract, 'Resume identity differs')
    path = e.safe_artifact(folder, saved['file'])
    require(e.digest(path) == saved['sha256'], 'Resume checkpoint hash differs')
    ck = torch.load(path,map_location='cpu',weights_only=True)
    require(ck['identity'] == contract and type(ck['epoch']) is int and 1 <= ck['epoch'] <= TRAIN['max_epochs']
            and len(ck['history']) == ck['epoch'] and 1 <= ck['best_epoch'] <= ck['epoch']
            and 0 <= ck['bad'] <= TRAIN['patience'] and np.isfinite(ck['best']) and 0 <= ck['best'] <= 1,
            'Invalid resume state')
    return ck


def commit_epoch(torch, folder, contract, payload):
    # Commit a new filename and THEN an atomic hash-bound pointer. A crash before
    # the pointer leaves the previous epoch resumable, including its RNG state.
    name = f'resume/epoch_{payload["epoch"]:03d}.pt'; path = folder / name
    path.parent.mkdir(exist_ok=True); save_checkpoint(torch,path,payload)
    body = dict(contract=contract,file=name,sha256=e.digest(path))
    e.write_json(folder / 'resume.json',dict(identity=e.object_hash(body),**body))
    for previous in path.parent.glob('epoch_*.pt'):
        if previous != path: previous.unlink()


def train_head(module, out, identity, seed, X, y, device):
    torch = module.torch; folder = out / f'seed_{seed}'; folder.mkdir(parents=True,exist_ok=True)
    contract = job_identity(identity,seed)
    done = e.read_completion(folder,'training_complete.json',contract)
    if done is not None: return done
    module.set_all_seeds(seed)
    train_loader = module.make_loader(X['train'],y['train'],TRAIN['batch_size'],True,device)
    val_loader = module.make_loader(X['val'],y['val'],TRAIN['batch_size']*2,False,device)
    model = module.MLPClassifier(module.ModelConfig(**MODEL)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=TRAIN['learning_rate'],weight_decay=TRAIN['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=2,min_lr=1e-6)
    criterion = module.nn.CrossEntropyLoss(label_smoothing=.03); scaler = module.make_grad_scaler(device)
    best=-1.; best_epoch=0; bad=0; history=[]; best_state=None; start=1
    ck = resume_checkpoint(torch,folder,contract)
    if ck is not None:
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
        scaler.load_state_dict(ck['scaler']); best=ck['best']; best_epoch=ck['best_epoch']; bad=ck['bad']
        history=ck['history']; best_state=ck['best_state']; start=ck['epoch']+1; restore_rng(torch,ck); del ck
        print(f'Resume head {seed} at epoch {start}',flush=True)
    for epoch in ([] if bad >= TRAIN['patience'] else range(start,TRAIN['max_epochs']+1)):
        tick=time.perf_counter(); trained=module.train_one_epoch(model,train_loader,optimizer,criterion,device,scaler)
        with torch.inference_mode(): val=module.predict_model(model,val_loader,device)
        p=prediction_check(val,y['val']); metrics=e.metrics(y['val'],p); f1=metrics['macro_f1']; scheduler.step(f1)
        if f1 > best+1e-12: best=f1; best_epoch=epoch; bad=0; best_state=cpu_state(model)
        else: bad+=1
        row=dict(epoch=epoch,train_loss=trained['loss'],train_accuracy=trained['accuracy'],val_macro_f1=f1,
                 val_accuracy=metrics['accuracy'],learning_rate=optimizer.param_groups[0]['lr'],elapsed_seconds=time.perf_counter()-tick)
        require(all(np.isfinite(v) for v in row.values()), 'Nonfinite training history')
        history.append(row); n=np.random.get_state()
        payload=dict(identity=contract,epoch=epoch,model=cpu_state(model),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),best=best,best_epoch=best_epoch,bad=bad,
            best_state=best_state,history=history,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            python_rng=random.getstate(),numpy_rng=[n[0],n[1].tolist(),n[2],n[3],n[4]])
        commit_epoch(torch,folder,contract,payload); e.write_csv(folder/'history.csv',history)
        print(f'  {seed} epoch {epoch}/30: val F1={f1:.6f}; best={best_epoch}',flush=True)
        del payload,val
        if bad >= TRAIN['patience']: break
    require(best_state is not None,'No validation-selected model')
    model.load_state_dict(best_state)
    save_checkpoint(torch,folder/'best.pt',dict(model_state_dict=best_state,model_config=MODEL,train_config=TRAIN,
        identity=contract,head_seed=seed,best_epoch=best_epoch,best_val_macro_f1=best,label_order=list(LABELS)))
    with torch.inference_mode(): val=module.predict_model(model,val_loader,device)
    p=prediction_check(val,y['val'])
    require(abs(e.metrics(y['val'],p)['macro_f1']-best) < 1e-12,'Selected validation score changed')
    e.write_csv(folder/'history.csv',history)
    e.write_array(folder/'validation_predictions.npz',**{k:val[k] for k in ('logits','y_true','y_pred')})
    summary=dict(seed=seed,best_epoch=best_epoch,epochs_completed=len(history),val_macro_f1=best,
        convergence='PLATEAU_OBSERVED' if bad >= TRAIN['patience'] else 'BUDGET_REVIEW_REQUIRED',
        parameters=sum(p.numel() for p in model.parameters()),training_seconds=sum(r['elapsed_seconds'] for r in history))
    last=e.read_json(folder/'resume.json')['file']
    e.write_completion(folder,'training_complete.json',contract,summary,
        ['best.pt','history.csv','validation_predictions.npz','resume.json',last])
    del model,optimizer,scheduler,train_loader,val_loader,best_state,val; gc.collect(); torch.cuda.empty_cache()
    return summary


def evaluate_head(module, out, identity, seed, condition, x, y, endpoint_id, device):
    index=training_gate(out,identity)  # Enforce the family-level gate even for a direct call.
    folder=out/f'seed_{seed}'; target=folder/condition; target.mkdir(parents=True,exist_ok=True)
    contract=e.object_hash(dict(protocol=identity,seed=seed,condition=condition,training=index,endpoint_identity=endpoint_id))
    done=e.read_completion(target,'complete.json',contract)
    if done is None:
        ck=module.torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        require(ck['identity']==job_identity(identity,seed) and ck['head_seed']==seed and ck['model_config']==MODEL
                and ck['train_config']==TRAIN and ck['label_order']==list(LABELS),'Selected checkpoint contract differs')
        model=module.MLPClassifier(module.ModelConfig(**MODEL)); model.load_state_dict(ck['model_state_dict']); model.to(device)
        loader=module.make_loader(x,y,TRAIN['batch_size']*2,False,device)
        with module.torch.inference_mode(): pred=module.predict_model(model,loader,device)
        prediction_check(pred,y)
        e.write_array(target/'predictions.npz',**{k:pred[k] for k in ('logits','y_true','y_pred')})
        e.write_completion(target,'complete.json',contract,dict(endpoint_identity=endpoint_id),['predictions.npz'])
        del model,loader,ck; gc.collect(); module.torch.cuda.empty_cache()
    with np.load(target/'predictions.npz',allow_pickle=False) as saved:
        return prediction_check(saved,y).astype(np.int8)


def comparisons(draws, points, families):
    rows=[]
    for condition,ix in [('clean',[0]),('missing8_mean',[1,2,3])]:
        for fi,family in enumerate(families[:2]):
            diff=(draws[:,ix,fi]-draws[:,ix,2]).mean(1)
            lo,hi,alo,ahi=np.quantile(diff,[.025,.975,.05/8,1-.05/8],method='linear')
            rows.append(dict(condition=condition,reference=family,comparator='TemporalStats_MLP',
                delta_macro_f1=float((points[ix,fi]-points[ix,2]).mean()),ci95_low=float(lo),ci95_high=float(hi),
                bonferroni_low=float(alo),bonferroni_high=float(ahi),family_size=4,marginal_confidence=.9875,
                direction='reference_higher' if alo>0 else 'statistics_higher' if ahi<0 else 'zero_included_not_equivalence'))
    return rows


def statistics(out, previous, test, new_predictions):
    families=('TST',previous['selection']['selected_pool'],'TemporalStats_MLP')
    y=classes(test['y'],(len(test['indices']),),'test labels')
    require(set(y)==set(range(6)),'All six test classes required')
    new_predictions=classes(new_predictions,(4,10,len(y)),'new predictions')
    pred=np.concatenate((reference_predictions(previous,test),new_predictions[:,None]),axis=1)
    counts,strata=e.episode_counts(y,test['meta'].episode_id,pred.reshape(4,30,len(y)))
    require(set(strata)==set(range(6)),'All six episode strata required')
    draws=e.bootstrap_scores(counts,strata,n_resamples=N_BOOTSTRAP,seed=BOOTSTRAP_SEED,batch=BOOTSTRAP_BATCH)
    draws=draws.reshape(N_BOOTSTRAP,4,3,10).mean(-1)
    total=counts.sum(0).reshape(4,3,10,6,6)
    seed_f1=e.macro_f1(total); points=seed_f1.mean(-1)
    seed_acc=np.trace(total,axis1=-2,axis2=-1)/len(y)
    normal=total[:,:,:,0,:]; seed_fpr=(normal.sum(-1)-normal[:,:,:,0])/normal.sum(-1)
    rows=[]; by_seed=[]; class_rows=[]; conf_rows=[]
    for condition,ix in [(c,[i]) for i,c in enumerate(CONDITIONS)]+[('missing8_mean',[1,2,3])]:
        for fi,family in enumerate(families):
            values=seed_f1[ix,fi].mean(0); lo,hi=np.quantile(draws[:,ix,fi].mean(1),[.025,.975],method='linear')
            rows.append(dict(condition=condition,family=family,n_head_seeds=10,n_masks=0 if ix==[0] else len(ix),
                macro_f1_mean=values.mean(),macro_f1_head_seed_sd_ddof1=values.std(ddof=1),ci95_low=lo,ci95_high=hi,
                accuracy_mean=seed_acc[ix,fi].mean(),normal_fpr_mean=seed_fpr[ix,fi].mean()))
            for si,seed in enumerate(SEEDS):
                by_seed.append(dict(condition=condition,family=family,seed=seed,macro_f1=values[si],
                    accuracy=seed_acc[ix,fi,si].mean(),normal_fpr=seed_fpr[ix,fi,si].mean()))
            cms=total[ix,fi]; den=cms.sum(-1)+cms.sum(-2); tp=np.diagonal(cms,axis1=-2,axis2=-1)
            cf=np.divide(2.*tp,den,out=np.zeros_like(tp,dtype=float),where=den>0).mean((0,1))
            norm=(cms/cms.sum(-1,keepdims=True)).mean((0,1))
            for li,label in enumerate(LABELS):
                class_rows.append(dict(condition=condition,family=family,label=label,f1_mean=cf[li]))
                for pi,pl in enumerate(LABELS):
                    conf_rows.append(dict(condition=condition,family=family,true_label=label,predicted_label=pl,
                                          seed_mask_mean_row_normalized=norm[li,pi]))
    inv=test['meta'].assign(y_true=y).groupby('episode_id',sort=True).agg(label_id=('y_true','first'),n_endpoints=('y_true','size')).reset_index()
    e.write_csv(out/'model_summary.csv',rows); e.write_csv(out/'metrics_by_seed.csv',by_seed)
    e.write_csv(out/'paired_comparisons.csv',comparisons(draws,points,families))
    e.write_csv(out/'per_class_f1.csv',class_rows); e.write_csv(out/'row_normalized_confusion.csv',conf_rows)
    e.write_csv(out/'episode_inventory.csv',inv); e.write_csv(out/'evaluation_endpoints_L10.csv',test['meta'])
    e.write_csv(out/'clean_context_observable_effects_L10.csv',pd.concat([test['meta'],test['effects']],axis=1))
    e.write_array(out/'predictions.npz',predictions=pred,y=y,conditions=np.array(CONDITIONS),families=np.array(families),seeds=np.array(SEEDS))
    e.write_array(out/'episode_confusions.npz',counts=counts.reshape(len(inv),4,3,10,6,6),episodes=inv.episode_id.to_numpy(dtype=str),strata=strata)
    e.write_array(out/'bootstrap_distribution.npz',macro_f1_seed_mean=draws,conditions=np.array(CONDITIONS),families=np.array(families))
    e.write_json(out/'statistics_audit.json',dict(draws=N_BOOTSTRAP,seed=BOOTSTRAP_SEED,batch_size=BOOTSTRAP_BATCH,
        sampling='class-stratified whole episode; identical draw paired across clean, masks, families and fitted head seeds',
        estimand='per-head endpoint six-class macro-F1, mean heads, then equal mean of three masks; no probability ensemble',
        CI_scope='conditional on dataset, encoder, fitted heads and three masks; no head/mask resampling',
        population='all eligible endpoints, including no_observable_change; effects are diagnostics only',
        multiplicity='four reference-minus-statistics contrasts; Bonferroni 98.75% marginal intervals',equivalence_claim=False))


def make_protocol(context, previous):
    body=dict(version=VERSION,script_sha256=e.digest(Path(__file__)),engine47_sha256=e.digest(Path(e.__file__)),
        pool13_sha256=context['code_hashes']['pool'],source_contract=context['source_contract'],input_audit=dict(context['observed']),
        source47_identity=previous['protocol']['identity'],source47_manifest_sha256=e.digest(Path(previous['folder'])/'run_manifest.json'),
        selected_pool=previous['selection']['selected_pool'],head_seeds=list(SEEDS),conditions=list(CONDITIONS),
        test_endpoint_identity=previous['protocol']['populations']['test']['endpoint_identity'],
        families=['TST',previous['selection']['selected_pool'],'TemporalStats_MLP'],model=MODEL,training=TRAIN,label_smoothing=.03,
        features='mean,std(ddof=0),min,max per embedding coordinate, concatenated in that order; FP32; causal stride-1 L10',
        training_selection='clean validation macro-F1 improves by >1e-12; earliest tied epoch; no test/missingness selection',
        gate='all ten clean-trained heads complete before any new test inference',fields=list(e.FIELDS),labels=list(LABELS),
        bootstrap=dict(draws=N_BOOTSTRAP,seed=BOOTSTRAP_SEED,batch=BOOTSTRAP_BATCH,family_size=4),
        design='retrospective extension; original reference test results already known; one new head family, no encoder retraining',
        limitations=['CIs condition on dataset, encoder, fitted heads and three existing masks',
            'ten head seeds are not ten encoder replications; inspect finite-budget convergence flags',
            'different feature dimension and parameter count; no equal-capacity/equal-compute claim',
            'Step13 training RNG/backend policy; bitwise GPU determinism not promised',
            'order-insensitive statistics summarize 22-field learned representations; no metadata/labels/effect flags in features',
            'all endpoints retained; no target score, forced ranking, equivalence or external-validity claim'])
    return dict(identity=e.object_hash(body),**body)


FINAL_FILES=['protocol.json','input_audit.json','training_runtime.json','runtime_complete.json','training_index.json','training_ready.json',
    'training_summary.csv','model_summary.csv','metrics_by_seed.csv','paired_comparisons.csv','per_class_f1.csv',
    'row_normalized_confusion.csv','episode_inventory.csv','evaluation_endpoints_L10.csv',
    'clean_context_observable_effects_L10.csv','predictions.npz','episode_confusions.npz',
    'bootstrap_distribution.npz','statistics_audit.json','analysis_summary.txt']


def package_review(out, manifest):
    # Include only the current manifest inventory: partial/stale files cannot
    # silently enter a completed review archive. Large local caches stay on disk.
    names=list(manifest['output_sha256'])+['run_manifest.json']
    archive=out.parent/(out.name+'_review.zip'); temp=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in names: z.write(e.safe_artifact(out,name),name)
    with zipfile.ZipFile(temp) as z: require(z.testzip() is None,'Review archive integrity failure')
    e.replace_atomic(temp,archive); return archive


def verify_final(out, protocol):
    manifest=e.read_json(out/'run_manifest.json')
    require(manifest.get('status')=='TEMPORAL_STATISTICS_COMPLETE' and manifest.get('protocol_identity')==protocol['identity'],
            'Incomplete/mismatched Step 52 completion')
    require(set(FINAL_FILES)<=set(manifest.get('output_sha256',{})), 'Incomplete final artifact inventory')
    for name,sha in manifest['output_sha256'].items():
        require(e.digest(e.safe_artifact(out,name))==sha,'Completed output changed/missing: '+name)
    index=training_gate(out,protocol['identity'])
    for name in ('train','val',*CONDITIONS):
        folder=out/'features'/name
        contract=e.object_hash(dict(protocol=protocol['identity'],features=name,recipe='mean_std0_min_max_fp32_L10'))
        done=e.read_completion(folder,'complete.json',contract)
        require(done is not None and 'features.npy' in e.read_json(folder/'complete.json')['artifact_sha256'],
                'Completed feature cache missing: '+name)
    for seed in SEEDS:
        for condition in CONDITIONS:
            folder=out/f'seed_{seed}'/condition
            contract=e.object_hash(dict(protocol=protocol['identity'],seed=seed,condition=condition,training=index,
                                       endpoint_identity=protocol['test_endpoint_identity']))
            done=e.read_completion(folder,'complete.json',contract)
            require(done is not None and 'predictions.npz' in e.read_json(folder/'complete.json')['artifact_sha256'],
                    'Completed test predictions missing')
    archive=out.parent/(out.name+'_review.zip'); valid=False
    if archive.is_file():
        try:
            with zipfile.ZipFile(archive) as z:
                valid=(len(z.namelist())==len(set(z.namelist()))
                    and set(z.namelist())==set(manifest['output_sha256'])|{'run_manifest.json'} and z.testzip() is None
                    and z.read('run_manifest.json')==(out/'run_manifest.json').read_bytes()
                    and all(archive_digest(z,n)==h for n,h in manifest['output_sha256'].items()))
        except (OSError,KeyError,RuntimeError,EOFError,zlib.error,zipfile.BadZipFile): valid=False
    return archive if valid else package_review(out,manifest)


def archive_digest(archive, name):
    h=hashlib.sha256()
    with archive.open(name) as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def run(module, device, out, protocol, context, data, previous):
    identity=protocol['identity']
    X={s:feature_cache(out,s,identity,data[s]['embeddings'],data[s]['indices']) for s in ('train','val')}
    y={s:data[s]['y'] for s in ('train','val')}
    index=e.read_json(out/'training_index.json') if (out/'training_index.json').exists() else {}
    require(set(index)<=set(map(str,SEEDS)),'Unknown training-index seed')
    for seed,sha in index.items(): require(e.digest(out/f'seed_{seed}/training_complete.json')==sha,'Training index changed')
    summaries=[]
    for i,seed in enumerate(SEEDS,1):
        print(f'[{i}/10] Clean training/validation head {seed}',flush=True)
        done=train_head(module,out,identity,seed,X,y,device)
        index[str(seed)]=e.digest(out/f'seed_{seed}/training_complete.json'); e.write_json(out/'training_index.json',index)
        summaries.append(done); e.write_csv(out/'training_summary.csv',summaries)
    index=training_gate(out,identity)
    e.save_matching_json(out/'training_ready.json',dict(all_ten_heads_complete_before_new_test_inference=True,
                         training_completion_sha256=index))
    context['helper'].recheck_inputs(context['observed'])
    del X,y; gc.collect(); module.torch.cuda.empty_cache()
    test=data['test']; endpoint=e.object_hash(test['meta'].to_dict(orient='records'))
    predictions=np.empty((4,10,len(test['y'])),dtype=np.int8)
    for ci,condition in enumerate(CONDITIONS):
        print('Test condition: '+condition,flush=True)
        if condition=='clean': emb=test['embeddings']
        else:
            path=Path(previous['folder'])/'cache'/condition/'embeddings.npy'
            require(e.digest(path)==context['observed'][str(path.resolve())],'Missing-8 cache changed during training')
            emb=np.load(path,mmap_mode='r',allow_pickle=False)
        x=feature_cache(out,condition,identity,emb,test['indices'])
        for si,seed in enumerate(SEEDS):
            predictions[ci,si]=evaluate_head(module,out,identity,seed,condition,x,test['y'],endpoint,device)
        del x,emb; gc.collect()
    context['helper'].recheck_inputs(context['observed'])
    statistics(out,previous,test,predictions)


def execute(root, out, context, data, previous):
    protocol=make_protocol(context,previous); out.mkdir(parents=True,exist_ok=True)
    with e.RunLock(out/'.run.lock'):
        e.initialize_protocol(out,protocol)
        context['helper'].recheck_inputs(context['observed'])
        if (out/'run_manifest.json').exists(): return verify_final(out,protocol)
        e.save_matching_json(out/'input_audit.json',context['observed'])
        try:
            runtime_saved=e.read_completion(out,'runtime_complete.json',protocol['identity'])
            require(runtime_saved is not None or not ((out/'features').exists() or any(out.glob('seed_*'))
                    or (out/'training_index.json').exists()), 'Partial training has no authenticated runtime record')
            module,device,runtime=training_backend(context,previous['selection']['selected_pool'])
            e.save_matching_json(out/'training_runtime.json',runtime)
            if runtime_saved is None:
                e.write_completion(out,'runtime_complete.json',protocol['identity'],runtime,['training_runtime.json'])
            else: require(runtime_saved==runtime,'Training runtime changed on resume')
            run(module,device,out,protocol,context,data,previous)
            context['helper'].recheck_inputs(context['observed'])
            require(e.digest(Path(__file__))==protocol['script_sha256'],'Step 52 code changed during execution')
            (out/'analysis_summary.txt').write_text(
                'TemporalStats_MLP uses mean, population std, min, max over the same causal L10 embeddings.\n'
                'The pooling reference was selected on clean validation: '+protocol['selected_pool']+'.\n'
                'All ten new heads completed clean train/validation before new test inference.\n'
                'Every eligible endpoint, including quiet contexts, is retained. No target score or ranking is imposed.\n'
                'Four reference-minus-statistics contrasts use paired whole-episode class-stratified bootstrap; Bonferroni family four.\n'
                'CIs condition on the fitted heads, encoder, data and three masks. Zero-containing intervals are not equivalence.\n'
                'This retrospective extension does not establish equal capacity/compute or external validity.\n',encoding='utf-8')
            include=list(FINAL_FILES)
            for seed in SEEDS:
                include += [f'seed_{seed}/'+n for n in ('training_complete.json','history.csv','validation_predictions.npz')]
                include += [f'seed_{seed}/{c}/complete.json' for c in CONDITIONS]
            include += [f'features/{name}/complete.json' for name in ('train','val',*CONDITIONS)]
            manifest=dict(status='TEMPORAL_STATISTICS_COMPLETE',protocol_identity=protocol['identity'],completed_at_utc=e.utc(),
                output_sha256={n:e.digest(out/n) for n in include},
                review_archive_omits='Local selected/resume weights, feature caches and redundant per-head test arrays; hashes retained.')
            e.write_json(out/'run_manifest.json',manifest)
            if (out/'failure.json').exists(): (out/'failure.json').unlink()
            return package_review(out,manifest)
        except BaseException as exc:
            # No COMPLETE marker for failed work. Keep the last valid archive;
            # failure reporting must not replace it with an incomplete archive.
            e.write_json(out/'failure.json',dict(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',
                error=str(exc),traceback=traceback.format_exc(),at_utc=e.utc()))
            raise


def validate_output_location(out, protected):
    out=out.resolve()
    for source in protected:
        source=Path(source).resolve()
        require(out!=source and source not in out.parents and out not in source.parents, 'Output overlaps input/source: '+str(source))


def main(argv=None):
    global e
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--step47-code',type=Path)
    parser.add_argument('--step15-code',type=Path,help='Current backend-free validator already required by Step 47')
    parser.add_argument('--reference47',type=Path,help='Completed current Step 47 folder; default is the shared_nominal_v2 run')
    parser.add_argument('--tst-root',type=Path); parser.add_argument('--pooling-root',type=Path); parser.add_argument('--lstm-root',type=Path)
    parser.add_argument('--audit-only','--preflight-only',action='store_true',help='Authenticate sources, complete Step 47, all cohorts and masks; no torch or output run')
    args=parser.parse_args(argv); root=args.project_root.expanduser().resolve()
    def rooted(value):
        if value is None: return None
        path=Path(value).expanduser(); return (path if path.is_absolute() else root/path).resolve()
    e,engine_path=load_engine(root,rooted(args.step47_code))
    out=rooted(args.output or OUT_REL); reference=rooted(args.reference47 or e.OUT_REL)
    context=e.audit_sources(root,rooted(args.step15_code),rooted(args.tst_root),rooted(args.pooling_root),rooted(args.lstm_root))
    context['helper'].track_file(engine_path,context['observed'])
    data=e.load_cohorts(root,context); data['train']=load_train(root,context,data)
    previous=audit_step47(root,reference,context,engine_path,data)
    validate_output_location(out,[root/e.DATA_REL,root/e.EMB_REL,root/e.SEQ_REL,root/e.ENC_REL,reference,
                                 *map(Path,context['roots'].values()),*map(Path,context['observed'])])
    if args.audit_only:
        print('Step 52 audit complete; no training/inference. Validation-selected pool: '+previous['selection']['selected_pool'],flush=True)
        return 0
    archive=execute(root,out,context,data,previous)
    print('Verified Step 52 review archive: '+str(archive),flush=True); return 0


if __name__=='__main__': sys.exit(main())
