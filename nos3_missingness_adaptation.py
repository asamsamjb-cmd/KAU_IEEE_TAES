#!/usr/bin/env python3
"""Bounded, post-hoc NOS3 missingness adaptation experiment.

Four heads only: TST L10 / MeanMax MLP x clean / 50% fixed mask14.
The reviewed fine-tuned DistilBERT is FROZEN. No live NOS3 commands.
The reviewed result ZIP supplies the exact hash-pinned evaluator implementation.
Default --stage all resumes the SAME sealed experiment after interruption.
Run from the original Windows project using its existing .venv interpreter.
"""
from __future__ import annotations
import argparse
import collections
import contextlib
import datetime as dt
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import sys
import time
import traceback
import zipfile

VERSION = '1.1-nos3-step7-manifest-fix'
PRE_MANIFEST_FIX_SHA = 'b1cfe721b094a2228656dde10709acf3845ed2adacf9315aeffc9d4b1ef29866'
SEED = 20260610
REFERENCE_NAME = 'run_20261005T144929_212195Z.zip'
REFERENCE_SHA = '68d3cbd8c407ca8366e096b757a34848767512aabe9482bc827e29773bc821db'
EVALUATOR_SHA = 'c7401887ce703831187b7a6d47244f3686f2cb8608dd6ecb686aec16d775d069'
CAMPAIGN_NAME = '2026-10-05_20-23-23_413130_09_NOS3_REPEATABILITY.zip'
KINDS = ('TST_L10', 'MeanMax_MLP')
ARMS = ('clean', 'mixed_mask14')


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_npy(path, value):
    import numpy as np
    path = Path(path); tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.save(f, value, allow_pickle=False); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch_save(value, path):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


@contextlib.contextmanager
def exclusive_workspace(root):
    """Kernel lock releases automatically after a crash, including on Windows."""
    root.mkdir(parents=True, exist_ok=True)
    with (root/'experiment.lock').open('a+b') as f:
        f.seek(0, 2)
        if f.tell() == 0:
            f.write(b'0'); f.flush()
        f.seek(0)
        if os.name == 'nt':
            import msvcrt
            try: msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as ex: raise RuntimeError('Another process is using this experiment folder') from ex
        else:
            import fcntl
            try: fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as ex: raise RuntimeError('Another process is using this experiment folder') from ex
        try:
            yield
        finally:
            f.seek(0)
            if os.name == 'nt': msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_reference(path, source_dir):
    """Execute only the pre-reviewed evaluator bytes, never arbitrary ZIP code."""
    path = Path(path)
    require(path.is_file(), f'Reviewed result ZIP not found: {path}')
    require(sha(path) == REFERENCE_SHA, 'Reviewed result ZIP SHA256 mismatch; use the completed 14:49:29 UTC result')
    source_dir = Path(source_dir); source_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        prefix = REFERENCE_NAME[:-4] + '/'
        manifest = json.loads(z.read(prefix+'result_manifest.json'))
        names = [i.filename for i in z.infolist() if not i.is_dir()]
        require(len(names) == len(set(names)), 'Duplicate reviewed ZIP members')
        require(set(names) == {prefix+x for x in manifest} | {prefix+'result_manifest.json'}, 'Reviewed ZIP inventory differs')
        for rel, checksum in manifest.items():
            require(hashlib.sha256(z.read(prefix+rel)).hexdigest() == checksum, f'Reviewed artifact is corrupt: {rel}')
        summary = json.loads(z.read(prefix+'summary.json'))
        require(summary['status'] == 'EXTENDED_COMPLETE' and not summary['errors'] and not summary['rule_errors'],
                'Reference is not the completed extended diagnostic')
        source = z.read(prefix+'nos3_detector_evaluation.py')
        require(hashlib.sha256(source).hexdigest() == EVALUATOR_SHA, 'Evaluator source SHA256 mismatch')
        provenance = json.loads(z.read(prefix+'model_provenance.json'))
    source_path = source_dir/'nos3_detector_evaluation.py'
    source_path.write_bytes(source)
    spec = importlib.util.spec_from_file_location('nos3_adaptation_verified_evaluator', source_path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module, dict(path=str(path.resolve()), sha256=REFERENCE_SHA, provenance=provenance)


def head_config(e, settings):
    return dict(input_dim=768, num_classes=6, sequence_length=10, temporal_context_seconds=10,
                d_model=int(settings['D_MODEL']), num_heads=int(settings['NUM_HEADS']),
                num_encoder_layers=int(settings['NUM_ENCODER_LAYERS']),
                dim_feedforward=int(settings['DIM_FEEDFORWARD']), dropout=float(settings['DROPOUT']),
                pooling=settings['POOLING'])


def assert_disjoint(groups):
    keys = list(groups)
    for i, left in enumerate(keys):
        for right in keys[i+1:]:
            overlap = set(groups[left]) & set(groups[right])
            require(not overlap, f'Episode split overlap: {left}/{right}: {sorted(overlap)[:5]}')


def load_split(e, context, split):
    """Read exactly the requested split; do not reconstruct or filter endpoints."""
    import numpy as np
    import pandas as pd
    require(split in ('train', 'val', 'test'), 'Unknown split')
    c = context['contract']; r9 = context['roots']['step9']; r10 = context['roots']['step10']
    raw = context['root']/f'dataset/shared_nominal_v2/splits/{split}_observable.csv'
    e.pin(raw, c['input_sha256'][f'splits/{split}_observable.csv'])
    original = pd.read_csv(raw, keep_default_na=False)
    require(list(original.columns) == e.META+e.FIELDS and not original.duplicated(e.KEYS).any(), f'{split}: invalid input schema/keys')
    wm_path = r9/f'{split}_window_metadata.csv'; e.pin(wm_path, c['step9_artifacts'][wm_path.name])
    windows = pd.read_csv(wm_path, keep_default_na=False)
    require(not windows.duplicated(e.KEYS).any() and np.array_equal(windows.embedding_row_index, np.arange(len(windows))),
            f'{split}: embedding metadata ordering differs')
    positions = pd.MultiIndex.from_frame(original[e.KEYS]).get_indexer(pd.MultiIndex.from_frame(windows[e.KEYS]))
    require(len(original) == len(windows) and (positions >= 0).all(), f'{split}: metadata row identity differs')
    frame = original.iloc[positions].reset_index(drop=True)
    for key in e.META:
        require(frame[key].astype(str).equals(windows[key].astype(str)), f'{split}: metadata differs: {key}')
    emb_path = r9/f'{split}_window_embeddings.npz'; e.pin(emb_path, c['step9_artifacts'][emb_path.name])
    with np.load(emb_path, allow_pickle=False) as z:
        require(np.array_equal(z['row_indices'], np.arange(len(frame))), f'{split}: cached embedding indices differ')
        clean = z['embeddings'].copy()
    require(clean.dtype == np.float32 and clean.shape == (len(frame),768) and np.isfinite(clean).all(), f'{split}: invalid embeddings')
    sm_path = r10/f'{split}_tst_sequence_metadata_L10.csv'; e.pin(sm_path, c['step10_artifacts'][sm_path.name])
    sm = pd.read_csv(sm_path, keep_default_na=False).sort_values('sequence_id').reset_index(drop=True)
    sq_path = r10/f'{split}_tst_sequences_L10.npz'; e.pin(sq_path, c['step10_artifacts'][sq_path.name])
    with np.load(sq_path, allow_pickle=False) as z:
        indices, y = z['sequence_indices'].copy(), z['y'].copy()
    require(indices.dtype.kind in 'iu' and y.dtype.kind in 'iu' and indices.shape == (len(sm),10)
            and y.shape == (len(sm),), f'{split}: invalid sequence arrays')
    expected = c['common_endpoint_report']['splits'][split]
    require(len(sm) == expected['common_endpoint_count'] and len(sm) > 0
            and np.array_equal(sm.sequence_id, np.arange(len(sm))), f'{split}: incomplete endpoint population')
    require((indices >= 0).all() and (indices < len(frame)).all() and (np.diff(indices,axis=1)==1).all(), f'{split}: invalid sequence span')
    ends = indices[:,-1]
    require(np.array_equal(indices[:,0], sm.sequence_start_row) and np.array_equal(ends, sm.sequence_end_row)
            and len(np.unique(ends)) == len(ends), f'{split}: endpoint identity mismatch')
    require(e.sha_bytes(np.sort(ends).astype('<i8',copy=False).tobytes()) == expected['sorted_end_row_indices_sha256'],
            f'{split}: endpoint digest differs')
    episodes = frame.episode_id.to_numpy()
    require((episodes[indices] == episodes[ends,None]).all() and np.array_equal(episodes[ends],sm.episode_id), f'{split}: sequence crosses an episode')
    complete = np.flatnonzero(frame.groupby('episode_id',sort=False).cumcount().to_numpy() >= 9)
    require(np.array_equal(np.sort(ends),complete), f'{split}: omitted eligible endpoints')
    require((np.diff(frame.episode_step.to_numpy()[indices],axis=1)==1).all(), f'{split}: nonconsecutive episode steps')
    times = pd.to_datetime(frame.timestamp, errors='raise', utc=True)
    require(times.notna().all() and times.diff().eq(pd.Timedelta(seconds=1)).to_numpy()[indices[:,1:]].all(),
            f'{split}: timestamps are not consecutive seconds')
    true = frame.label.map(dict(zip(e.LABELS,range(6)))).to_numpy()
    require(np.isin(true,range(6)).all() and np.array_equal(y,true[ends]) and np.array_equal(sm.label_id,y)
            and np.array_equal(sm.label,frame.label.iloc[ends]), f'{split}: terminal labels differ')
    if 'source_split' in sm:
        require((sm.source_split == split).all(), f'{split}: source_split mismatch')
    require(set(y.tolist()) == set(range(6)), f'{split}: expected all six classes')
    return dict(frame=frame, clean=clean, indices=indices.astype(np.int64,copy=False), y=y.astype(np.int64,copy=False),
                metadata=sm, episodes=set(episodes), input_sha256=c['input_sha256'][f'splits/{split}_observable.csv'])


def split_manifest(e, context):
    import pandas as pd
    rel = 'splits/episode_split_manifest.csv'
    p = context['root']/'dataset/shared_nominal_v2'/rel
    e.pin(p, context['contract']['input_sha256'][rel])
    frame = pd.read_csv(p, keep_default_na=False)
    # Step 7 writes an inventory with original/global IDs, not episode_id.
    # Match its rewrite_duplicate_episode_ids_if_needed() and the Step 42 audit:
    # if any original ID occurs in multiple classes, ALL output IDs are global.
    step7 = {'label','original_episode_id','global_episode_id','split'}
    if step7 <= set(frame.columns):
        require(frame[list(step7)].notna().all().all()
                and frame[list(step7)].astype(str).ne('').all().all(), 'Empty Step 7 episode inventory identities')
        original = frame.original_episode_id.astype(str)
        labels = frame.label.astype(str)
        global_ids = frame.global_episode_id.astype(str)
        require(labels.isin(e.LABELS).all(), 'Unknown label in Step 7 episode inventory')
        require(global_ids.eq(labels+'::'+original).all() and global_ids.is_unique,
                'Step 7 global episode IDs are invalid or duplicated')
        collision = frame.assign(original_episode_id=original).groupby('original_episode_id').label.nunique().gt(1).any()
        canonical = global_ids if collision else original
        if 'episode_id' in frame:
            require(frame.episode_id.astype(str).equals(canonical), 'Explicit episode_id conflicts with Step 7 identity policy')
        frame['episode_id'] = canonical
        print('  Episode inventory: Step 7 original/global schema; output IDs = '
              + ('global_episode_id' if collision else 'original_episode_id'), flush=True)
    else:
        require({'episode_id','split'} <= set(frame.columns),
                f'Unrecognized episode inventory schema. Columns: {list(frame.columns)}')
    require(not frame.episode_id.duplicated().any(), 'Episode split manifest has duplicate assignments')
    require(set(frame.split) == {'train','val','test'}, 'Episode split manifest has unexpected split names')
    groups = {s:set(frame.loc[frame.split==s,'episode_id']) for s in ('train','val','test')}
    assert_disjoint(groups)
    return groups


def observable_texts(e, frame):
    # Exclude all metadata, labels, campaign identifiers and intervention labels.
    return [e.serial(dict(zip(e.FIELDS, values))) for values in frame[e.FIELDS].itertuples(index=False,name=None)]


def factor_texts(texts):
    import numpy as np
    unique = {}; indices = np.empty(len(texts),dtype=np.int64)
    for i, text in enumerate(texts):
        if text not in unique: unique[text] = len(unique)
        indices[i] = unique[text]
    return list(unique), indices


def mask_cache(e, frame, engine, root, identity):
    import numpy as np
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    texts, mapping = factor_texts(observable_texts(e,e.mask14(frame)))
    contract = dict(identity=identity, ordered_unique_texts_sha256=digest(texts), row_mapping_sha256=e.sha_bytes(mapping.astype('<i8').tobytes()),
                    encoder='Frozen reviewed supervised-finetuned DistilBERT CLS FP32', vectors=len(texts), rows=len(frame))
    record_path = root/'cache.json'
    if record_path.is_file():
        record = read_json(record_path)
        require(record['contract'] == contract, 'Cached masked input identity differs')
        for name, checksum in record['artifacts'].items(): require(sha(root/name)==checksum, f'Masked cache changed: {name}')
        vectors = np.load(root/'vectors.npy',allow_pickle=False)
        row_map = np.load(root/'row_map.npy',allow_pickle=False)
    else:
        parts = []
        for lo in range(0,len(texts),128):
            parts.append(engine.embed(texts[lo:lo+128])); engine.cache.clear()
            print(f'  Mask cache: {min(lo+128,len(texts)):,}/{len(texts):,} unique inputs',flush=True)
        vectors = np.concatenate(parts).astype('float32',copy=False); row_map = mapping
        atomic_npy(root/'vectors.npy',vectors); atomic_npy(root/'row_map.npy',row_map)
        atomic_json(root/'unique_texts.json',texts)
        atomic_json(record_path,dict(contract=contract,artifacts={n:sha(root/n) for n in ('vectors.npy','row_map.npy','unique_texts.json')}))
    require(vectors.dtype == np.float32 and vectors.shape == (len(texts),768) and np.isfinite(vectors).all()
            and np.array_equal(row_map,mapping), 'Invalid masked cache content')
    return dict(vectors=vectors,row_map=row_map)


def encoder_control(e, data, engine, output):
    import numpy as np
    # Development control uses validation only. No test labels/inputs are read.
    ids = np.unique(data['indices'][:16])
    texts = observable_texts(e,data['frame'].iloc[ids])
    live = engine.embed(texts); engine.cache.clear()
    expected = data['clean'][ids]
    report = dict(population='First 16 validation L10 endpoints, without label/performance selection',
                  windows=len(ids),atol=e.ATOL,rtol=e.RTOL,max_abs_difference=float(np.abs(live-expected).max()))
    report['status'] = 'PASS' if np.allclose(live,expected,atol=e.ATOL,rtol=e.RTOL) else 'FAIL'
    atomic_json(output,report)
    require(report['status']=='PASS', 'Frozen encoder does not reproduce cached validation embeddings')


def epoch_schedule(n, epoch, seed):
    import numpy as np
    # Independent from torch dropout RNG and from labels. Same schedule in both arms.
    rng = np.random.default_rng(np.random.SeedSequence([int(seed),int(epoch),9127]))
    return rng.permutation(n), rng.random(n) < .5


def batch_inputs(data, rows, masked):
    import numpy as np
    ids = data['indices'][rows]
    x = np.asarray(data['clean'][ids],dtype=np.float32).copy()
    flags = np.broadcast_to(np.asarray(masked,dtype=bool),(len(rows),))
    if flags.any(): x[flags] = data['vectors'][data['row_map'][ids[flags]]]
    return x


def new_head(e, kind, config, seed):
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    require(kind in KINDS, 'Unknown head architecture')
    if kind == 'TST_L10': return e.make_tst(config,torch)
    class MeanMaxMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Proposed supplementary comparator, not a claim to reproduce another numbered baseline.
            self.input_norm = torch.nn.LayerNorm(config['input_dim'])
            self.net = torch.nn.Sequential(torch.nn.Linear(2*config['input_dim'],config['d_model']),
                torch.nn.GELU(),torch.nn.Dropout(config['dropout']),torch.nn.Linear(config['d_model'],6))
        def forward(self,x):
            if x.ndim != 3 or x.shape[1:] != (10,config['input_dim']): raise ValueError('Expected [batch,10,768] input')
            x = self.input_norm(x)
            return self.net(torch.cat((x.mean(dim=1),x.max(dim=1).values),dim=1))
    return MeanMaxMLP()


def state_digest(state):
    h = hashlib.sha256()
    for key,value in sorted(state.items()):
        h.update(key.encode()); h.update(str(tuple(value.shape)).encode())
        h.update(str(value.dtype).encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def predict_head(e, model, data, masked, device, batch_size=256):
    import numpy as np
    import torch
    model.eval(); parts=[]
    with torch.inference_mode():
        for lo in range(0,len(data['indices']),batch_size):
            rows=np.arange(lo,min(lo+batch_size,len(data['indices'])))
            x=torch.from_numpy(batch_inputs(data,rows,masked)).to(device)
            parts.append(torch.softmax(model(x).float(),dim=1).cpu().numpy())
    probs=np.concatenate(parts);e.validate_probs(probs,len(data['indices']))
    return probs


def train_head(e, kind, arm, config, recipe, train, val, folder, device):
    import numpy as np
    import torch
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    require(arm in ARMS,'Unknown training arm')
    run_identity=dict(kind=kind,arm=arm,config=config,recipe=recipe,device=device)
    complete_path=folder/'complete.json'
    if complete_path.is_file():
        complete=read_json(complete_path)
        require(complete['identity']==run_identity and sha(folder/'best.pt')==complete['best_checkpoint_sha256'], 'Completed head identity/weights differ')
        print(f'  Reusing completed head: {kind}/{arm}',flush=True)
        return complete
    model=new_head(e,kind,config,recipe['seed']).to(device).float()
    initial_digest=state_digest(model.state_dict())
    optimizer=torch.optim.AdamW(model.parameters(),lr=recipe['learning_rate'],weight_decay=recipe['weight_decay'])
    loss_fn=torch.nn.CrossEntropyLoss(label_smoothing=recipe['label_smoothing'])
    history=[];best_score=-1.;best_epoch=0;best_state=None;start_epoch=1;steps=0;masked_count=0
    last=folder/'last.pt'
    if last.is_file():
        saved=torch.load(last,map_location='cpu',weights_only=True)
        require(saved['identity']==run_identity and saved['initial_state_sha256']==initial_digest,'Resume checkpoint belongs to another protocol/runtime')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        for state in optimizer.state.values():
            for key,value in state.items():
                if torch.is_tensor(value): state[key]=value.to(device)
        history=saved['history'];best_score=saved['best_score'];best_epoch=saved['best_epoch'];best_state=saved['best_state']
        start_epoch=saved['epoch']+1;steps=saved['optimizer_steps'];masked_count=saved['masked_sequence_presentations']
        torch.set_rng_state(saved['cpu_rng'])
        if device.startswith('cuda'): torch.cuda.set_rng_state_all(saved['cuda_rng'])
        print(f'  Resuming {kind}/{arm} after epoch {start_epoch-1}',flush=True)
    for epoch in range(start_epoch,recipe['epochs']+1):
        order,flags=epoch_schedule(len(train['indices']),epoch,recipe['seed'])
        active=flags if arm=='mixed_mask14' else np.zeros(len(flags),dtype=bool)
        model.train();loss_sum=0.;n_seen=0;last_progress=time.monotonic()
        for lo in range(0,len(order),recipe['batch_size']):
            rows=order[lo:lo+recipe['batch_size']];select=active[lo:lo+len(rows)]
            x=torch.from_numpy(batch_inputs(train,rows,select)).to(device)
            y=torch.from_numpy(train['y'][rows]).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(x);loss=loss_fn(logits,y)
            require(torch.isfinite(loss).item(),'Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),recipe['grad_clip_norm'],error_if_nonfinite=True)
            optimizer.step();steps+=1;n_seen+=len(rows);masked_count+=int(select.sum());loss_sum+=float(loss.detach())*len(rows)
            if time.monotonic()-last_progress>=25:
                print(f'  {kind}/{arm} epoch {epoch}/{recipe["epochs"]}: {n_seen:,}/{len(order):,}',flush=True);last_progress=time.monotonic()
        p_clean=predict_head(e,model,val,False,device,recipe['batch_size'])
        p_mask=predict_head(e,model,val,True,device,recipe['batch_size'])
        _,clean_metrics=e.classification_metrics(val['y'],p_clean.argmax(1))
        _,mask_metrics=e.classification_metrics(val['y'],p_mask.argmax(1))
        score=(clean_metrics['macro_f1']+mask_metrics['macro_f1'])/2
        history.append(dict(epoch=epoch,loss=loss_sum/n_seen,val_clean_macro_f1=clean_metrics['macro_f1'],
            val_mask14_macro_f1=mask_metrics['macro_f1'],selection_score=score,optimizer_steps=steps,
            masked_presentations_this_epoch=int(active.sum()),sequence_presentations=n_seen))
        if score>best_score:
            best_score=score;best_epoch=epoch;best_state=cpu_state(model)
        # Best state, optimizer and RNG are committed together, avoiding mismatched sidecar checkpoints.
        saved=dict(identity=run_identity,initial_state_sha256=initial_digest,epoch=epoch,model=cpu_state(model),
            optimizer=optimizer.state_dict(),cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [],history=history,
            best_state=best_state,best_score=best_score,best_epoch=best_epoch,optimizer_steps=steps,
            masked_sequence_presentations=masked_count)
        atomic_torch_save(saved,last);atomic_json(folder/'history.json',history)
        print(f'  {kind}/{arm} epoch {epoch}/{recipe["epochs"]}: val clean={clean_metrics["macro_f1"]:.6f}, mask14={mask_metrics["macro_f1"]:.6f}, selected={best_epoch}',flush=True)
    require(best_state is not None and len(history)==recipe['epochs'],'Incomplete training history')
    # A crash can occur after last.pt was committed but before its history sidecar.
    # Restore it even when resuming an already finished final epoch.
    atomic_json(folder/'history.json',history)
    final=dict(identity=run_identity,model_state_dict=best_state,selected_epoch=best_epoch,selection_score=best_score,
               initial_state_sha256=initial_digest,label_order=e.LABELS,training_complete=True)
    atomic_torch_save(final,folder/'best.pt')
    result=dict(identity=run_identity,status='COMPLETE',initial_state_sha256=initial_digest,
        selected_epoch=best_epoch,selection_score=best_score,optimizer_steps=steps,
        masked_sequence_presentations=masked_count,total_sequence_presentations=len(train['indices'])*recipe['epochs'],
        best_checkpoint_sha256=sha(folder/'best.pt'),parameter_count=sum(p.numel() for p in model.parameters()))
    atomic_json(complete_path,result)
    del model,optimizer
    if device.startswith('cuda'):torch.cuda.empty_cache()
    return result


def collision_audit(window_keys, indices, y, labels):
    import numpy as np
    groups={}
    for seq,label in zip(indices,y):
        key=tuple(int(x) for x in window_keys[seq])
        if key not in groups:groups[key]=np.zeros(len(labels),dtype=np.int64)
        groups[key][int(label)]+=1
    rows=[];ambiguous=0;ambiguous_n=0;majority=0
    for key,counts in groups.items():
        conflicting=int(np.count_nonzero(counts))>1;count=int(counts.sum())
        ambiguous+=int(conflicting);ambiguous_n+=count if conflicting else 0;majority+=int(counts.max())
        rows.append(dict(input_id=digest(key),endpoints=count,conflicting=conflicting,counts=counts.tolist()))
    report=dict(endpoints=len(y),unique_inputs=len(groups),conflicting_groups=ambiguous,
        endpoints_in_conflicting_groups=ambiguous_n,empirical_accuracy_ceiling=majority/len(y),
        interpretation='Finite-sample deterministic accuracy ceiling for exact ordered observable inputs; not a macro-F1 ceiling or an estimate of population Bayes accuracy.')
    return report,rows


def training_seal(root, protocol_id):
    """All four chosen checkpoints must be immutable BEFORE opening test inputs."""
    root=Path(root); records={}; artifacts={}
    for kind in KINDS:
        for arm in ARMS:
            name=kind+'__'+arm; folder=root/'heads'/name
            require((folder/'complete.json').is_file(), 'All four heads must be complete; missing '+name)
            r=read_json(folder/'complete.json')
            require(r['status']=='COMPLETE' and r['identity']['recipe']['protocol_id']==protocol_id, 'Incomplete or different-protocol head '+name)
            require(r['identity']['kind']==kind and r['identity']['arm']==arm, 'Head identity differs')
            require(sha(folder/'best.pt')==r['best_checkpoint_sha256'], 'Selected head changed: '+name)
            hist=read_json(folder/'history.json')
            require(len(hist)==r['identity']['recipe']['epochs'] and hist[-1]['optimizer_steps']==r['optimizer_steps'], 'Training history incomplete')
            require(all(h['sequence_presentations']==hist[0]['sequence_presentations'] for h in hist), 'Unequal epoch presentation budgets')
            require(sum(h['masked_presentations_this_epoch'] for h in hist)==r['masked_sequence_presentations'], 'Masked presentation audit differs')
            records[name]=r
            for file in ('best.pt','complete.json','history.json'):
                rel=f'heads/{name}/{file}';artifacts[rel]=sha(root/rel)
        x,y=[records[kind+'__'+arm] for arm in ARMS]
        require(x['initial_state_sha256']==y['initial_state_sha256'] and x['optimizer_steps']==y['optimizer_steps']
                and x['total_sequence_presentations']==y['total_sequence_presentations'], 'Unmatched clean/masked training budgets or initialization')
        require(x['masked_sequence_presentations']==0 and y['masked_sequence_presentations']>0, 'Training arm contamination')
    value=dict(protocol_id=protocol_id,artifacts=artifacts,heads=records)
    target=root/'training_seal.json'
    if target.exists(): require(read_json(target)==value,'Frozen training artifacts changed after selection')
    else: atomic_json(target,value)
    return value


def load_selected_heads(e, root, seal, device):
    import torch
    heads={}
    for name,r in seal['heads'].items():
        p=Path(root)/'heads'/name/'best.pt'
        require(sha(p)==r['best_checkpoint_sha256'],'Head checksum differs before evaluation')
        checkpoint=torch.load(p,map_location='cpu',weights_only=True)
        require(checkpoint['identity']==r['identity'] and checkpoint['label_order']==e.LABELS and checkpoint['training_complete'], 'Invalid selected checkpoint')
        require(checkpoint['selected_epoch']==r['selected_epoch'],'Selected epoch differs')
        model=new_head(e,r['identity']['kind'],r['identity']['config'],r['identity']['recipe']['seed'])
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        model.float().to(device).eval();heads[name]=model
    return heads


def evaluate_benchmark(e, data, heads, out, device):
    import numpy as np
    import pandas as pd
    summaries=[];perclass=[];episode_rows=[];confusions=[];predictions=[];decisions={}
    y=data['y'];metadata=data['metadata']
    for condition,mask in (('clean',False),('fixed_mask14',True)):
        for name,model in heads.items():
            p=predict_head(e,model,data,mask,device);pred=p.argmax(1)
            pc,overall=e.classification_metrics(y,pred)
            summaries.append(dict(model=name,condition=condition,**overall,replay_recall=pc[4]['recall'],
                                  normal_to_ground_count=int(((y==0)&(pred==5)).sum())))
            perclass.extend(dict(model=name,condition=condition,**r) for r in pc)
            cm=np.bincount(6*y+pred,minlength=36).reshape(6,6)
            confusions.extend(dict(model=name,condition=condition,true_label=e.LABELS[i],prediction=e.LABELS[j],count=int(cm[i,j])) for i in range(6) for j in range(6))
            record=metadata.copy();record['model']=name;record['condition']=condition
            record['predicted_label_id']=pred;record['prediction']=[e.LABELS[int(x)] for x in pred]
            for j,label in enumerate(e.LABELS):record['prob_'+label.lower()]=p[:,j]
            predictions.append(record);decisions[(condition,name)]=pred
            for episode,group in metadata.groupby('episode_id',sort=False):
                idx=group.index.to_numpy();counts=np.bincount(6*y[idx]+pred[idx],minlength=36)
                row=dict(model=name,condition=condition,episode_id=episode,endpoint_count=len(idx))
                row.update({f'cm_{i}_{j}':int(counts[6*i+j]) for i in range(6) for j in range(6)});episode_rows.append(row)
            print(f'  Benchmark {name}/{condition}: macro-F1={overall["macro_f1"]:.6f}, normal FAR={overall["normal_false_alarm_fraction"]:.6f}, replay recall={pc[4]["recall"]:.6f}',flush=True)
            e.write_csv(out/'benchmark_summary.csv',summaries)
            e.write_csv(out/'benchmark_per_class.csv',perclass)
            e.write_csv(out/'benchmark_confusion_counts.csv',confusions)
            # Preserve completed model predictions if a subsequent model fails.
            pd.concat(predictions,ignore_index=True).to_csv(out/'benchmark_predictions.csv',index=False,encoding='utf-8-sig')
            e.write_csv(out/'benchmark_episode_confusions.csv',episode_rows)
    contrasts=[];changes=[]
    for kind in KINDS:
        before=kind+'__clean';after=kind+'__mixed_mask14'
        if before not in heads or after not in heads:continue
        for condition in ('clean','fixed_mask14'):
            left,right=[next(x for x in summaries if x['model']==name and x['condition']==condition) for name in (before,after)]
            contrasts.append(dict(architecture=kind,condition=condition,direction='mixed_mask14 training minus clean training',
                delta_macro_f1=right['macro_f1']-left['macro_f1'],
                delta_normal_false_alarm_fraction=right['normal_false_alarm_fraction']-left['normal_false_alarm_fraction'],
                delta_replay_recall=right['replay_recall']-left['replay_recall']))
            x,z=decisions[(condition,before)],decisions[(condition,after)]
            rows=metadata[['sequence_id','episode_id','sequence_end_row','label']].copy()
            rows['architecture']=kind;rows['condition']=condition
            rows['clean_trained_prediction']=[e.LABELS[int(k)] for k in x]
            rows['mixed_trained_prediction']=[e.LABELS[int(k)] for k in z]
            rows['decision_changed']=x!=z;changes.append(rows)
    if contrasts:
        e.write_csv(out/'adaptation_paired_differences.csv',contrasts)
        pd.concat(changes,ignore_index=True).to_csv(out/'adaptation_endpoint_changes.csv',index=False,encoding='utf-8-sig')
    return summaries


def evaluate_collisions(e, data, out):
    reports=[];details=[]
    for condition in ('clean','fixed_mask14'):
        frame=data['frame'] if condition=='clean' else e.mask14(data['frame'])
        texts,keys=factor_texts(observable_texts(e,frame))
        report,groups=collision_audit(keys,data['indices'],data['y'],e.LABELS)
        reports.append(dict(condition=condition,**report))
        for r in groups:
            row=dict(condition=condition,input_id=r['input_id'],endpoints=r['endpoints'],conflicting=r['conflicting'])
            row.update({'count_'+label.lower():r['counts'][j] for j,label in enumerate(e.LABELS)})
            details.append(row)
    e.write_csv(out/'input_collision_summary.csv',reports)
    e.write_csv(out/'input_collision_groups.csv',details)
    return reports


def evaluate_nos3(e, runs, engine, heads, out, device):
    import numpy as np
    import pandas as pd
    endpoints=e.prepare_inputs(runs,out);trajectory=e.trajectory_endpoints(runs)
    require(len(trajectory)==809,'Reviewed NOS3 trajectory population changed')
    predictions=[]
    for run in runs:
        points=[r for r in trajectory if r['run']==run['name']]
        ends=[r['endpoint_row'] for r in points]
        starts=[r['window_start'] for r in run['view']['index']]
        for view,key in zip(e.VIEWS,('features','without_history')):
            texts=[e.serial(row) for row in run['view'][key]]
            embeddings=engine.embed(texts)
            data=dict(clean=embeddings,indices=np.array(e.seq_indices(ends,10,starts),dtype=np.int64))
            for name,model in heads.items():
                p=predict_head(e,model,data,False,device)
                for point,vector in zip(points,p):
                    pred=int(vector.argmax());row=dict(point,model=name,view=view,serialization='fixed_unk',
                        prediction=e.LABELS[pred],predicted_label_id=pred,confidence=float(vector[pred]))
                    row.update({'prob_'+label.lower():float(vector[j]) for j,label in enumerate(e.LABELS)})
                    predictions.append(row)
        engine.cache.clear()
    events=[r for r in predictions if r['activity_group']=='current_command']
    require(len(events)==len(heads)*60 and len(predictions)==len(heads)*1618,'Incomplete NOS3 adapted predictions')
    e.write_csv(out/'event_predictions.csv',events);e.write_csv(out/'trajectory_predictions.csv',predictions)
    summary=e.summarize_predictions(events,endpoints,out)
    paired=[]
    lookup={(r['model'],r['run'],r['endpoint_row'],r['view']):r for r in events}
    for r in events:
        if r['view'] != 'observed_partial':continue
        other=lookup[(r['model'],r['run'],r['endpoint_row'],'freshness_unk')]
        paired.append(dict(model=r['model'],run=r['run'],pair=r['pair'],endpoint_row=r['endpoint_row'],
            population=r['population'],intervention_truth=r['intervention_truth'],
            observed_prediction=r['prediction'],freshness_unk_prediction=other['prediction'],
            decision_changed=r['prediction']!=other['prediction'],
            replay_probability_observed=r['prob_telecommand_replay'],
            replay_probability_freshness_unk=other['prob_telecommand_replay'],
            replay_probability_difference=r['prob_telecommand_replay']-other['prob_telecommand_replay']))
    e.write_csv(out/'history_ablation_paired.csv',paired)
    grouped=collections.defaultdict(collections.Counter)
    for r in predictions:
        for scope in (r['run'],'ALL_RUNS_DESCRIPTIVE'):
            for activity in (r['activity_group'],'ALL_ENDPOINTS'):
                grouped[(scope,r['model'],r['view'],activity)][r['prediction']]+=1
    rows=[]
    for (run,model,view,activity),counts in grouped.items():
        total=sum(counts.values())
        item=dict(run=run,model=model,view=view,activity_group=activity,endpoint_count=total,
                  interpretation='Descriptive output distribution; unlabelled quiet/recent endpoints have no assigned class truth')
        item.update({'count_'+label.lower():counts[label] for label in e.LABELS});rows.append(item)
    e.write_csv(out/'trajectory_class_distribution.csv',rows)
    return dict(endpoint_summary=summary,common_endpoints=809,event_predictions=len(events),trajectory_predictions=len(predictions))


def historical_references(reference_path, out, e, data):
    """Read held-out historical results only in the final evaluation stage."""
    import numpy as np
    import pandas as pd
    prefix=REFERENCE_NAME[:-4]+'/'
    with zipfile.ZipFile(reference_path) as z:
        for filename in ('benchmark_summary.csv','benchmark_per_class.csv','endpoint_summary.csv',
                         'event_predictions.csv','trajectory_class_distribution.csv','benchmark_clean_reproduction.json'):
            (out/('historical_'+filename)).write_bytes(z.read(prefix+filename))
        old=pd.read_csv(io.BytesIO(z.read(prefix+'benchmark_clean_TST_L10_predictions.csv')),keep_default_na=False)
        require(len(old)==len(data['y']) and np.array_equal(old.sequence_id,data['metadata'].sequence_id)
                and np.array_equal(old.episode_id,data['metadata'].episode_id)
                and np.array_equal(old.sequence_end_row,data['metadata'].sequence_end_row),
                'Benchmark population differs from the previously reviewed diagnostic')
        old_features=pd.read_csv(io.BytesIO(z.read(prefix+'observable_windows.csv')),keep_default_na=False)
    fresh=pd.read_csv(out/'observable_windows.csv',keep_default_na=False)
    require(old_features.equals(fresh),'Regenerated NOS3 observables differ from the reviewed mapping')
    atomic_json(out/'historical_reference_note.json',dict(
        source_sha256=REFERENCE_SHA,models=['TST_L1','TST_L10','Rule_L10'],
        purpose='Unmodified earlier frozen-detector references. Adaptation effects use the newly matched clean-trained heads, not unmatched historical training.',
        independent_external_validation=False))


def runtime_config(torch, device, threads):
    require(device in ('cpu','cuda'),'Unsupported device')
    if device=='cuda':require(torch.cuda.is_available(),'CUDA is unavailable in this interpreter; use --device cpu')
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if hasattr(torch.backends.cuda.matmul,'allow_tf32'):torch.backends.cuda.matmul.allow_tf32=False
    if hasattr(torch.backends.cudnn,'allow_tf32'):torch.backends.cudnn.allow_tf32=False
    # Stable eager attention path for these short sequences; no autocast/AMP.
    if hasattr(torch.backends.cuda,'enable_flash_sdp'):torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda,'enable_mem_efficient_sdp'):torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda,'enable_math_sdp'):torch.backends.cuda.enable_math_sdp(True)
    return dict(python=platform.python_version(),torch=torch.__version__,numpy=__import__('numpy').__version__,
        pandas=__import__('pandas').__version__,transformers=__import__('transformers').__version__,
        device=device,threads=threads,precision='float32',amp=False,tf32=False,deterministic_algorithms=True,
        gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,cuda=torch.version.cuda if device=='cuda' else None)


def build_protocol(e, context, reference, runtime):
    settings=context['r11']['settings']
    for key,value in dict(MAX_EPOCHS=30,BATCH_SIZE=256,LEARNING_RATE=7e-5,WEIGHT_DECAY=.0005,
                          LABEL_SMOOTHING=.03,GRAD_CLIP_NORM=1.).items():
        require(settings[key]==value,f'Reviewed training default changed: {key}')
    recipe=dict(seed=SEED,epochs=30,batch_size=256,learning_rate=7e-5,weight_decay=.0005,
                label_smoothing=.03,grad_clip_norm=1.)
    protocol=dict(version=VERSION,script_sha256=sha(Path(__file__)),source_evaluator_sha256=EVALUATOR_SHA,
        previous_result_sha256=REFERENCE_SHA,campaign_sha256=e.REVIEWED_CAMPAIGN_SHA,
        source_contract=context['contract'],step11_run_id=context['r11']['run_id'],tokenizer_sha256=context['tokenizer_sha'],
        recipe=recipe,head_config=head_config(e,settings),runtime=runtime,architectures=list(KINDS),training_arms=list(ARMS),
        encoder='Frozen reviewed fine-tuned DistilBERT CLS, evaluated on CPU FP32',encoder_training=False,
        selection='Arithmetic mean of clean and fixed-mask14 validation six-class macro-F1; earliest epoch on exact ties',
        epoch_policy='Exactly 30 epochs in all four runs; no early stopping or learning-rate search',
        other_input_augmentations=False,checkpoint_initialization='Same per-architecture random initialization in both arms; no warm start',
        augmentation=dict(probability=.5,unit='whole L10 sequence per epoch',mode='fixed field-name mask',
            class_independent=True,available_fields=list(e.AVAILABLE_FIELDS),masked_fields=list(e.UNAVAILABLE_FIELDS),
            serialization='observable22_fixed_order_is_v1',placeholder='UNK'),
        pooling_comparator='Per-window LayerNorm(768); concatenate temporal mean and max; Linear(1536,128), GELU, Dropout(.25), Linear(128,6). New supplementary comparator.',
        label_order=e.LABELS,phase_policy='Train/validation only before four checkpoints are sealed; test/NOS3 inference only afterward',
        additional_live_commands=0,formal_external_validation=False,independent_seed_replications=1,
        limitations=[
            'Post-hoc targeted adaptation motivated by already observed NOS3 and benchmark masking failures; not a preregistered external test.',
            'One fixed seed per architecture and arm; no claim of statistical superiority or training-variance robustness.',
            'Five normal/replay pairs belong to one continuous NOS3 session; overlapping windows are not independent trials.',
            'Fixed mask14 differs from the NOS3 conditional 14/15/18 missingness pattern and field-value distribution.',
            'Freshness is computed by an external within-run frame-history observer; the original frame is outside the ten-second model input.',
            'Missing fields may remove class-discriminating evidence; six-class recovery is not guaranteed.',
            'Quiet and recent-history NOS3 endpoints carry no assigned class truth; distributions are not false-alarm rates.',
            'No claim of RF/GNSS attack coverage, cryptographic anti-replay bypass, full 22-field validation or operational readiness.'])
    protocol['protocol_id']=digest(protocol)
    return protocol


def seal_protocol(root, protocol):
    path=Path(root)/'protocol.json'
    if not path.exists():
        atomic_json(path,protocol);return
    previous=read_json(path)
    if previous==protocol:return
    # Only upgrade the known v1.0 preflight failure. Prepared data or any head
    # artifacts forbid migration; training/evaluation contracts stay immutable.
    without_identity=lambda x:{k:v for k,v in x.items() if k not in ('version','script_sha256','protocol_id')}
    inactive=not any((Path(root)/name).exists() for name in
        ('prepared.json','cache','heads','training_seal.json','validation_encoder_control.json'))
    compatible=(previous.get('version')=='1.0-nos3-frozen-encoder-adaptation'
        and previous.get('script_sha256')==PRE_MANIFEST_FIX_SHA
        and previous.get('protocol_id')==digest({k:v for k,v in previous.items() if k!='protocol_id'})
        and protocol.get('version')==VERSION
        and without_identity(previous)==without_identity(protocol))
    require(inactive and compatible,
            'Experiment protocol/runtime changed after preparation or is not the known v1.0 preflight failure. Existing artifacts were preserved.')
    history=Path(root)/'protocol_history'/f"{previous['protocol_id']}.json"
    if history.exists():require(read_json(history)==previous,'Archived preflight protocol differs')
    else:atomic_json(history,previous)
    atomic_json(path,protocol)
    print('  Recovered the v1.0 preflight-only failure; prior protocol and result ZIPs retained.',flush=True)


def copy_training_review(workspace, out):
    for name in ('protocol.json','prepared.json','training_seal.json','validation_encoder_control.json'):
        p=workspace/name
        if p.is_file():shutil.copy2(p,out/name)
    history=workspace/'protocol_history'
    if history.is_dir():shutil.copytree(history,out/'protocol_history',dirs_exist_ok=True)
    source=workspace/'source/nos3_detector_evaluation.py'
    if source.is_file():shutil.copy2(source,out/'verified_evaluator.py')
    for kind in KINDS:
        for arm in ARMS:
            name=kind+'__'+arm
            for file in ('complete.json','history.json','best.pt'):
                p=workspace/'heads'/name/file
                if p.is_file():
                    target=out/'heads'/name/file;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
            last=workspace/'heads'/name/'last.pt'
            if last.exists():
                # Large optimizer/cache files stay local for resume; the review records their checksums.
                atomic_json(out/(name+'_resume_reference.json'),dict(path=str(last),sha256=sha(last)))


def seal_review(out):
    out=Path(out)
    manifest={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*'))
              if p.is_file() and p.name!='result_manifest.json' and '__pycache__' not in p.parts}
    atomic_json(out/'result_manifest.json',manifest)
    target=out.with_suffix('.zip')
    require(not target.exists(),'Refusing to overwrite a previous review ZIP')
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for rel in list(manifest)+['result_manifest.json']:z.write(out/rel,out.name+'/'+rel)
    with zipfile.ZipFile(target) as z:
        require(z.testzip() is None,'Review ZIP integrity check failed')
        for rel,checksum in manifest.items():require(hashlib.sha256(z.read(out.name+'/'+rel)).hexdigest()==checksum,'Review ZIP member checksum differs')
    return target


def execute(args, workspace, out, report):
    import numpy as np
    root=args.project_root.resolve()
    reference_path=args.previous_result.resolve() if args.previous_result else root/'outputs/shared_nominal_v2/nos3_detector_evaluation'/REFERENCE_NAME
    print('[1/5] Verify the reviewed evaluator, encoder and data lineage.',flush=True)
    e,reference=load_reference(reference_path,workspace/'source')
    nos3_path=args.nos3_zip.resolve() if args.nos3_zip else root/CAMPAIGN_NAME
    e.pin(nos3_path,e.REVIEWED_CAMPAIGN_SHA)
    report['input_sha256']=e.PINS
    context=e.model_context(root,args.tst_root)
    old=reference['provenance']
    require(old['source_contract']==context['contract'] and old['step11_run_id']==context['r11']['run_id']
            and old['tokenizer_sha256']==context['tokenizer_sha'],'Reviewed model/data lineage changed')
    groups=split_manifest(e,context)
    # Import dependencies only after source validation. Offline loading only.
    engine=e.FrozenEngine(context,args.threads)
    for parameter in engine.encoder.parameters():parameter.requires_grad_(False)
    runtime=runtime_config(engine.torch,args.device,args.threads)
    protocol=build_protocol(e,context,reference,runtime);seal_protocol(workspace,protocol)
    report.update(protocol_id=protocol['protocol_id'],runtime=runtime,limitations=protocol['limitations'])
    recipe=dict(protocol['recipe'],protocol_id=protocol['protocol_id'])
    if args.stage in ('all','prepare','train'):
        print('[2/5] Prepare train/validation only. No test windows or live NOS3 commands.',flush=True)
        train=load_split(e,context,'train');val=load_split(e,context,'val')
        require(train['episodes']==groups['train'] and val['episodes']==groups['val'],'Observed episodes differ from sealed split manifest')
        assert_disjoint(dict(train=train['episodes'],val=val['episodes'],test=groups['test']))
        encoder_control(e,val,engine,workspace/'validation_encoder_control.json')
        for split,data in (('train',train),('val',val)):
            identity=dict(protocol_id=protocol['protocol_id'],split=split,input_sha256=data['input_sha256'])
            data.update(mask_cache(e,data['frame'],engine,workspace/'cache'/split,identity))
        prepared=dict(protocol_id=protocol['protocol_id'],status='PREPARED',
            splits={s:dict(windows=len(d['frame']),endpoints=len(d['indices']),episodes=len(d['episodes']),
                masked_unique_inputs=len(d['vectors'])) for s,d in (('train',train),('val',val))},test_windows_read=False)
        atomic_json(workspace/'prepared.json',prepared)
        report['stages']['prepare']='COMPLETE'
        if args.stage=='prepare':
            for path,checksum in e.PINS.items():require(sha(path)==checksum,f'Source changed: {path}')
            report['status']='PREPARED';return
        print('[3/5] Train four heads with matched initialization and 30 epochs each.',flush=True)
        for kind in KINDS:
            for arm in ARMS:
                name=kind+'__'+arm
                train_head(e,kind,arm,protocol['head_config'],recipe,train,val,workspace/'heads'/name,args.device)
        report['stages']['train']='COMPLETE'
        del train,val
    for path,checksum in e.PINS.items():require(sha(path)==checksum,f'Source changed: {path}')
    seal=training_seal(workspace,protocol['protocol_id'])
    report['stages']['train']='COMPLETE';report['trained_head_count']=4
    if args.stage=='train':report['status']='TRAINING_COMPLETE';return
    # This is the only gate after which held-out observable rows can be opened.
    atomic_json(out/'evaluation_gate.json',dict(training_seal_sha256=sha(workspace/'training_seal.json'),
        protocol_id=protocol['protocol_id'],opened_utc=now(),head_hashes={k:v['best_checkpoint_sha256'] for k,v in seal['heads'].items()}))
    print('[4/5] Evaluate clean and mask14 benchmark on the same common endpoints.',flush=True)
    report['stages']['benchmark']='RUNNING'
    test=load_split(e,context,'test')
    require(test['episodes']==groups['test'],'Test episodes differ from declared split')
    require(len(test['indices'])==38078,'Reviewed common test endpoint count differs')
    # Evaluation-only encoder check protects resumed/evaluate-only invocations.
    control_ids=np.unique(test['indices'][:16]);live=engine.embed(observable_texts(e,test['frame'].iloc[control_ids]));engine.cache.clear()
    control=dict(windows=len(control_ids),atol=e.ATOL,rtol=e.RTOL,
                 max_abs_difference=float(np.abs(live-test['clean'][control_ids]).max()))
    control['status']='PASS' if np.allclose(live,test['clean'][control_ids],atol=e.ATOL,rtol=e.RTOL) else 'FAIL'
    atomic_json(out/'test_encoder_control.json',control)
    require(control['status']=='PASS','Frozen encoder test-control reproduction failed')
    test.update(mask_cache(e,test['frame'],engine,workspace/'cache/test',dict(protocol_id=protocol['protocol_id'],split='test',input_sha256=test['input_sha256'])))
    heads=load_selected_heads(e,workspace,seal,args.device)
    report['benchmark_summary']=evaluate_benchmark(e,test,heads,out,args.device)
    report['input_collisions']=evaluate_collisions(e,test,out)
    report['stages']['benchmark']='COMPLETE'
    print('[5/5] Evaluate the existing NOS3 campaign and freshness ablation; no new commands.',flush=True)
    report['stages']['nos3']='RUNNING'
    runs,mapping=e.verify_campaign(nos3_path,out/'verified_campaign')
    report['nos3']=evaluate_nos3(e,runs,engine,heads,out,args.device)
    atomic_json(out/'schema_contract.json',mapping)
    historical_references(reference_path,out,e,test)
    shutil.rmtree(out/'verified_campaign')
    report['stages']['nos3']='COMPLETE'
    for path,checksum in e.PINS.items():require(sha(path)==checksum,f'Source input changed during the experiment: {path}')
    for rel,checksum in seal['artifacts'].items():require(sha(workspace/rel)==checksum,f'Selected model changed during evaluation: {rel}')
    atomic_json(out/'input_sha256.json',e.PINS)
    report['status']='COMPLETE'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--nos3-zip',type=Path,help='Reviewed Step 09 campaign ZIP; defaults to the known filename in project root')
    parser.add_argument('--previous-result',type=Path,help='Completed v1.3 result ZIP; defaults to the reviewed filename in the original output folder')
    parser.add_argument('--output-root',type=Path,help='Persistent experiment folder; reuse the same folder to resume')
    parser.add_argument('--tst-root',type=Path,help='Optional explicit original Step11 main-seed run folder')
    parser.add_argument('--stage',choices=('all','prepare','train','evaluate'),default='all')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu',help='Head training/inference device; encoder stays on CPU FP32')
    parser.add_argument('--threads',type=int,choices=range(1,17),default=2)
    args=parser.parse_args(argv)
    workspace=(args.output_root.resolve() if args.output_root else args.project_root.resolve()/'outputs/shared_nominal_v2/nos3_missingness_adaptation')
    # Must be set before importing torch or initializing CUDA.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    print(f'NOS3 missingness adaptation: {VERSION}',flush=True)
    print(f'Stage: {args.stage} | Frozen encoder | Four heads | Seed {SEED} | Head device: {args.device} | No live commands',flush=True)
    # Never create experiment artifacts in an unrelated existing directory.
    if workspace.exists() and not (workspace/'protocol.json').exists():
        allowed={'source','reviews','experiment.lock'}
        require(set(p.name for p in workspace.iterdir())<=allowed,'Output folder contains unrelated files; choose an empty experiment folder')
    try:
        with exclusive_workspace(workspace):
            stamp=dt.datetime.now(dt.timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
            out=workspace/'reviews'/stamp;out.mkdir(parents=True,exist_ok=False)
            shutil.copy2(Path(__file__),out/Path(__file__).name)
            report=dict(version=VERSION,status='STARTED',started_utc=now(),stage=args.stage,
                protocol_id=None,trained_head_count=0,errors=[],formal_external_validation=False,
                post_hoc_supplementary_experiment=True,encoder_retrained=False,additional_live_commands_sent=0,
                stages=dict(prepare='NOT_RUN',train='NOT_RUN',benchmark='NOT_RUN',nos3='NOT_RUN'))
            code=1
            try:
                execute(args,workspace,out,report);code=0
            except (Exception,KeyboardInterrupt) as error:
                report['status']='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAILED'
                code=130 if isinstance(error,KeyboardInterrupt) else 1
                report['errors'].append(f'{type(error).__name__}: {error}')
                (out/'failure_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
                print(f'Experiment stopped: {error}',file=sys.stderr,flush=True)
                if any((workspace/'heads').rglob('last.pt')):
                    print('Completed epoch checkpoints remain local. Repeat the SAME command to resume.',flush=True)
                else:
                    print('No head-training checkpoint was created. See the recorded preflight error.',flush=True)
            finally:
                report['finished_utc']=now();atomic_json(out/'summary.json',report)
                copy_training_review(workspace,out)
                # Failed campaign reconstruction can be large; preserve its identity, not duplicate the original evidence ZIP.
                campaign=out/'verified_campaign'
                if campaign.exists():shutil.rmtree(campaign)
                archive=seal_review(out)
                print(json.dumps({k:report[k] for k in ('status','trained_head_count','stages','errors')},indent=2),flush=True)
                print(f'ZIP_SHA256={sha(archive)}',flush=True)
                print(f'UPLOAD_THIS={archive}',flush=True)
            return code
    except (OSError,RuntimeError) as error:
        print(f'Workspace could not be locked or written: {error}',file=sys.stderr,flush=True)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
