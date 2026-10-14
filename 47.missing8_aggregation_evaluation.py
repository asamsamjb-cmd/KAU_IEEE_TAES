#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 47: shared-nominal Missing-8, shared fine-tuned encoder / aggregation evaluation.

Keep this file and taes_shared_nominal_audit.py beside the current revised
11/13/14 scripts. The companion is imported automatically; do not run it.
Run: python "47.missing8_aggregation_evaluation.py"
Optional: --preflight-only (CPU file/cohort/raw-audit check, no inference).
Re-run the identical command to resume; completed stages are hash-verified.

Protocol: label-blind uniform 8-of-22 field selection for EVERY record,
three predetermined PCG64 seeds; literal UNK replacement, then re-embedding.
This uniform all-window stress protocol is separate from Step 18/19 structured
group/burst masking. All clean and masked conditions keep the same L10 endpoints.
No training, no checkpoint/seed selection on test. Validation selects one of
Mean/Max/MeanMax MLP (mean over all ten seeds) and EWMA alpha. Main estimand:
TST minus that selected pooling family, averaged over ten trained head seeds
and three masks. Paired class-stratified episode bootstrap, 20,000 draws.
CI is conditional on these trained heads, encoder and three observed masks.
A zero-containing difference interval does NOT demonstrate equivalence.

Step 45/46 encoder-comparison weights are deliberately not used here.
Revision 3 adds a matched CURRENT-window classifier reference, post-mask
paired-nominal projection diagnostics, and clause-omission sensitivity using
the SAME three masks and all saved heads. Literal UNK remains the primary
protocol; omission is supplementary, never a test-selected replacement.
No new training or mask search. Paired-nominal diagnostics are privileged
annotations used AFTER inference, never model features or an information bound.
This experiment compares aggregation with ONE fixed encoder; it cannot establish
DistilBERT superiority over other encoders or a unique temporal-order advantage.
Step 52 consumes the unchanged primary v2 contract/caches from this new run.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
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

BASE_DIR=Path(__file__).resolve().parent
FIELDS=('observation_stage','path_type','direction','source_role','destination_role',
        'protocol_context','operational_phase','message_type','service_context',
        'command_intent','payload_or_parameter_context','sequence_freshness_status',
        'control_security_state','signal_quality_state','tracking_stability',
        'navigation_consistency','timing_consistency','decoder_status',
        'command_execution_result','retry_timeout_state','session_state','state_transition_consistency')
LABELS=('NORMAL','RF_LINK_JAMMING','GNSS_SPOOFING','UNAUTHORIZED_TELECOMMAND_INJECTION',
        'TELECOMMAND_REPLAY','GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE')
SEEDS=tuple(range(20260610,20260620))
MASK_SEEDS=(20261001,20261002,20261003)
ALPHAS=(0.1,0.25,0.5,0.75,1.0)
POOLS=('Mean_MLP','Max_MLP','MeanMax_MLP')
FAMILIES=('TST',*POOLS,'LSTM','ProbabilityMean','MajorityVote','EWMA')
TRAINED=FAMILIES[:5]
MODES={'Mean_MLP':'mean','Max_MLP':'max','MeanMax_MLP':'meanmax'}
CONDITIONS=('clean',*(f'missing8_{s}' for s in MASK_SEEDS))
RUNS=tuple((f,s) for f in TRAINED for s in SEEDS)+tuple((f,None) for f in FAMILIES[5:])
PROTOCOL_VERSION='shared_nominal_uniform_missing8_aggregation_v2'
CODE_VERSION='3.1-audit-loader-fix'
AUDIT_MODULE_FILENAME='taes_shared_nominal_audit.py'
AUDIT_MODULE_SHA256='182f03eff1d58a923ecdeac3c71a8b9398baff2f6397604ecd04ae1d2f39dd2f'
SUPPLEMENT_VERSION='missing8_fixed_masks_omission_and_current_window_v1'
PROJECTION_STRATA=('current_retained','history_only_retained','all_changes_masked','no_clean_change')
PREFIX='outputs/shared_nominal_v2'
OUT_REL=PREFIX+'/47_missing8_aggregation/run_missing8_uniform_v3'
DATA_REL='dataset/shared_nominal_v2'
ENC_REL=PREFIX+'/distilbert_window_baseline/trained_distilbert_model'
EMB_REL=PREFIX+'/tst_embeddings_finetuned_main'
SEQ_REL=PREFIX+'/tst_sequences_finetuned_main'
TST_REL=PREFIX+'/tst_training_finetuned_multiseed'
POOL_REL=PREFIX+'/39_pooling_mlp_finetuned_main'
LSTM_REL=PREFIX+'/lstm_finetuned_distilbert_l10'
META=('scenario_id','episode_id','episode_type','episode_step','window_id','timestamp','label')
SOURCE_ROLES={'TST':11,'pool':13,'LSTM':14}
PROVENANCE_KEYS=('dataset_id','step8_run_id','step9_run_id','step10_run_id')
N_BOOTSTRAP=20000
BOOTSTRAP_SEED=20261004
BOOTSTRAP_BATCH=100

def require(ok,message):
    if not ok: raise ValueError(message)

def utc():return datetime.now(timezone.utc).isoformat()


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
            raise ValueError('Step 47 is already running in this output folder')
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


def metrics(y,pred):
    y,pred=np.asarray(y),np.asarray(pred)
    require(y.ndim==1 and y.shape==pred.shape and len(y)>0,'Wrong prediction shape')
    require(np.isin(y,range(6)).all() and np.isin(pred,range(6)).all(),'Invalid class IDs')
    cm=np.bincount((6*y+pred).astype(np.int64),minlength=36).reshape(6,6)
    den=cm.sum(0)+cm.sum(1)
    cf=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0)
    return dict(macro_f1=float(cf.mean()),accuracy=float(np.trace(cm)/cm.sum()),
                class_f1=cf.tolist(),confusion_matrix=cm.tolist())


def make_mask(n_rows,seed):
    """Depends ONLY on row count and seed, never labels, IDs or field contents."""
    require(n_rows>0,'Empty mask population')
    rng=np.random.Generator(np.random.PCG64(seed))
    result=np.zeros((n_rows,len(FIELDS)),dtype=bool)
    # Independent uniform subset without replacement per record, canonical CSV order.
    for i in range(n_rows):result[i,rng.choice(len(FIELDS),size=8,replace=False)]=True
    return result


def apply_mask(frame,mask):
    mask=np.asarray(mask)
    require(mask.dtype==bool and mask.shape==(len(frame),22) and (mask.sum(1)==8).all(),'Invalid Missing-8 mask')
    out=frame.copy(deep=True)
    values=frame[list(FIELDS)].astype(str).to_numpy(copy=True)
    changed=mask & (values!='UNK')
    values[mask]='UNK'
    out[list(FIELDS)]=values
    return out,dict(rows=len(frame),selected_cells=int(mask.sum()),newly_replaced_cells=int(changed.sum()),
                    already_UNK_selected_cells=int((mask & ~changed).sum()),
                    selected_per_field=dict(zip(FIELDS,mask.sum(0).astype(int).tolist())),
                    newly_replaced_per_row_histogram=np.bincount(changed.sum(1),minlength=9).tolist(),
                    metadata_unchanged=True,mask_uses_labels_or_values=False)


def serialize_frame(frame,omit_mask=None):
    if omit_mask is None:
        return ['; '.join(f'{f} is {str(v).strip()}' for f,v in zip(FIELDS,row))+'.'
                for row in frame[list(FIELDS)].itertuples(index=False,name=None)]
    omit_mask=np.asarray(omit_mask)
    require(omit_mask.dtype==bool and omit_mask.shape==(len(frame),22) and (omit_mask.sum(1)==8).all(),
            'Omission requires the exact eight selected fields per row')
    return ['; '.join(f'{f} is {str(v).strip()}' for f,v,drop in zip(FIELDS,row,omitted) if not drop)+'.'
            for row,omitted in zip(frame[list(FIELDS)].itertuples(index=False,name=None),omit_mask)]


def check_probs(p):
    p=np.asarray(p)
    require(p.ndim==2 and p.shape[1]==6 and np.isfinite(p).all(),'Invalid probability shape/NaN')
    require((p>=0).all() and (p<=1).all() and np.allclose(p.sum(1),1,atol=2e-6,rtol=0),'Invalid probabilities')
    return p


def aggregate_probabilities(p,indices,method,alpha=None):
    p=check_probs(p); result=np.empty((len(indices),6),dtype=np.float64)
    for lo in range(0,len(indices),2048):
        q=p[indices[lo:lo+2048]].astype(np.float64)
        if method=='mean':result[lo:lo+len(q)]=q.mean(1)
        elif method=='ewma':
            require(alpha is not None and 0<alpha<=1,'Invalid EWMA alpha')
            a=q[:,0].copy()
            for j in range(1,q.shape[1]):a=(1-alpha)*a+alpha*q[:,j]
            result[lo:lo+len(q)]=a
        elif method=='vote':
            vote=np.eye(6,dtype=np.int64)[q.argmax(-1)].sum(1)
            # Restrict ties to max-vote classes, then mean probability, then lowest ID.
            tie=vote==vote.max(1,keepdims=True)
            chosen=np.where(tie,q.mean(1),-1).argmax(1)
            result[lo:lo+len(q)]=np.eye(6)[chosen]
        else:raise ValueError('Unknown probability aggregator: '+method)
    return result


def choose_pool(rows):
    scores=[]
    for family in POOLS:
        part=[r for r in rows if r['family']==family]
        require(len(part)==10 and sorted(r['seed'] for r in part)==list(SEEDS),'Need all ten validation head seeds')
        values=np.asarray([r['macro_f1'] for r in part],float)
        require(np.isfinite(values).all() and ((values>=0)&(values<=1)).all(),'Invalid validation score')
        scores.append(values.mean())
    return POOLS[int(np.argmax(scores))]


def choose_alpha(scores):
    require(set(scores)==set(ALPHAS),'Incomplete EWMA validation grid')
    require(all(np.isfinite(v) and 0<=v<=1 for v in scores.values()),'Invalid EWMA validation score')
    return ALPHAS[int(np.argmax([scores[a] for a in ALPHAS]))]


def validate_indices(frame,indices,y):
    indices=np.asarray(indices);y=np.asarray(y)
    require(indices.ndim==2 and indices.shape[1]==10 and len(indices)==len(y) and len(indices)>0,'Invalid L10 shape')
    require(np.issubdtype(indices.dtype,np.integer) and indices.min()>=0 and indices.max()<len(frame),'Invalid sequence indices')
    require(not frame.duplicated(['episode_id','episode_step']).any(),'Duplicate episode/step')
    ep=frame.episode_id.to_numpy()[indices]
    steps=frame.episode_step.to_numpy(dtype=np.int64)[indices]
    require((ep==ep[:,:1]).all() and (np.diff(steps,axis=1)==1).all(),'Sequence crosses episode or missing step')
    times=pd.to_datetime(frame.timestamp,utc=True,errors='raise')
    require(not times.isna().any() and
            (np.diff(times.to_numpy(dtype='datetime64[ns]')[indices],axis=1)==np.timedelta64(1,'s')).all(),
            'Sequence timestamps are not consecutive one-second observations')
    labels=frame.label.map(dict(zip(LABELS,range(6)))).to_numpy()[indices]
    require((labels==labels[:,:1]).all() and np.array_equal(y,labels[:,-1]),'Wrong sequence labels')
    keys=pd.DataFrame({'episode_id':ep[:,-1],'end_step':steps[:,-1]})
    require(not keys.duplicated().any(),'Duplicate endpoint')
    complete=[]
    for _,positions in frame.groupby('episode_id',sort=False).indices.items():
        positions=np.asarray(positions)[np.argsort(frame.iloc[positions].episode_step.to_numpy(),kind='stable')]
        require(np.all(np.diff(frame.iloc[positions].episode_step.to_numpy(dtype=np.int64))==1),'Nonconsecutive episode')
        if len(positions)>=10:complete.extend(np.lib.stride_tricks.sliding_window_view(positions,10))
    require(np.array_equal(indices,np.asarray(complete,dtype=np.int64)),
            'Sequence indices must retain the complete causal stride-1 L10 population in Step 10 order')


def check_predictions(d,reference):
    require(len(d)==len(reference),'Prediction count differs')
    for name in ('episode_id','sequence_end_step','sequence_end_row'):
        require(name in d and np.array_equal(d[name].to_numpy(),reference[name].to_numpy()),'Prediction alignment: '+name)
    require(np.array_equal(d.true_label_id,reference.label_id),'Prediction label mismatch')
    p=check_probs(d[['prob_'+l.lower() for l in LABELS]].to_numpy(dtype=float))
    require(np.array_equal(p.argmax(1),d.pred_label_id),'Prediction/probability mismatch')
    return p


def validate_checkpoint(c,family,seed,context=None):
    cfg,tc=c['model_config'],c['train_config']
    require(cfg['num_classes']==6 and cfg['input_dim']==(1536 if family=='MeanMax_MLP' else 768),'Checkpoint dimensions differ')
    require(np.isfinite(c['best_val_summary']['macro_f1']),'Missing checkpoint validation metric')
    if family=='TST':
        require(tc['random_seed']==seed and cfg['sequence_length']==10,'Wrong TST seed/L')
        require(c['label_order']==list(LABELS),'Wrong TST class order')
        require(not tc.get('use_robust_eval_as_primary_for_long_contexts',False),'Legacy TST robust-primary checkpoint unsupported')
        require(tc.get('use_missing_aware_training') is False,'Expected clean-trained TST heads')
    else:
        require(c['seed']==seed and c['sequence_length']==10,'Wrong checkpoint seed/L')
        if family in POOLS:
            require(c['pooling_mode']==MODES[family] and not tc['use_standardization'],'Wrong pooling mode or unsupported standardized features')
        else:require(cfg['sequence_length']==10,'Wrong LSTM length')
    if context is not None:
        head=context['heads'][run_key(family,seed)]
        require(c.get('source_contract')==context['source_contract'] and c.get('provenance')==head['provenance'],
                'Checkpoint dataset/encoder/sequence/run provenance differs')
        require(cfg==head['model_config'] and tc==head['train_config'],'Checkpoint model/training configuration differs')


def paths_for(family,seed,context=None,split='test'):
    if context is not None:
        item=context['heads'][run_key(family,seed)]
        return item['checkpoint'],item['predictions'][split]
    if family=='TST':
        base=f'{TST_REL}/seed_{seed}/L10'
        return f'{base}/checkpoints/best_tst_model_L10.pt',f'{base}/predictions/test_predictions_L10.csv'
    if family=='LSTM':
        return (f'{LSTM_REL}/models/best_lstm_finetuned_distilbert_L10_seed{seed}.pt',
                f'{LSTM_REL}/predictions/seed{seed}_test_predictions.csv')
    mode=MODES[family]
    return (f'{POOL_REL}/models/best_pooling_mlp_L10_{mode}_seed{seed}.pt',
            f'{POOL_REL}/predictions/{mode}_seed{seed}_test_predictions.csv')


def episode_counts(y,episodes,predictions):
    y=np.asarray(y);p=np.asarray(predictions)
    require(p.ndim==3 and p.shape[-1]==len(y),'Prediction tensor must be [conditions,runs,endpoints]')
    require(np.isin(y,range(6)).all() and np.isin(p,range(6)).all(),'Invalid labels in bootstrap')
    codes,names=pd.factorize(np.asarray(episodes),sort=True)
    require(len(codes)==len(y) and (codes>=0).all(),'Invalid episodes')
    strata=np.empty(len(names),dtype=np.int64)
    for i in range(len(names)):
        labels=np.unique(y[codes==i]);require(len(labels)==1,'Mixed label episode');strata[i]=labels[0]
    counts=np.zeros((len(names),*p.shape[:2],6,6),dtype=np.int64)
    for ci in range(p.shape[0]):
        for ri in range(p.shape[1]):
            flat=(codes*6+y)*6+p[ci,ri]
            counts[:,ci,ri]=np.bincount(flat.astype(np.int64),minlength=len(names)*36).reshape(-1,6,6)
    return counts,strata


def bootstrap_scores(counts,strata,n_resamples=N_BOOTSTRAP,seed=BOOTSTRAP_SEED,batch=BOOTSTRAP_BATCH,progress=True):
    require(counts.ndim==5 and counts.shape[-2:]==(6,6) and len(strata)==len(counts),'Invalid confusion tensor')
    require(n_resamples>0 and batch>0,'Invalid bootstrap parameters')
    sufficient=np.stack((np.diagonal(counts,axis1=-2,axis2=-1),counts.sum(-1),counts.sum(-2)),axis=-2).astype(float)
    flat=sufficient.reshape(len(strata),-1);rng=np.random.Generator(np.random.PCG64(seed))
    out=np.empty((n_resamples,*counts.shape[1:3]),dtype=float)
    for lo in range(0,n_resamples,batch):
        n=min(batch,n_resamples-lo);w=np.zeros((n,len(strata)))
        for c in np.unique(strata):
            ix=np.flatnonzero(strata==c);w[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=n)
        sampled=(w@flat).reshape(n,*counts.shape[1:3],3,6)
        den=sampled[...,1,:]+sampled[...,2,:]
        f1=np.divide(2*sampled[...,0,:],den,out=np.zeros_like(den),where=den>0)
        out[lo:lo+n]=f1.mean(-1)
        if progress and (lo==0 or (lo+n)%1000==0 or lo+n==n_resamples):print(f'  Bootstrap {lo+n:,}/{n_resamples:,}',flush=True)
    return out


def initialize_protocol(out,protocol):
    path=out/'protocol.json'
    if path.exists():require(read_json(path)==protocol,'Protocol/input/runtime changed. Do not mix runs; inspect the discrepancy.')
    else:
        require(not any(p.name not in {'.run.lock'} for p in out.iterdir()),'Output contains foreign/unfinished protocol files')
        write_json(path,protocol)


def import_script(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def read_records(path):
    return pd.read_csv(path,encoding='utf-8-sig',keep_default_na=False,
        dtype={k:str for k in (*FIELDS,'scenario_id','episode_id','episode_type','timestamp','label',
                              'sequence_start_timestamp','sequence_end_timestamp')})


def resolve_code(root,number,expected=None,explicit=None):
    candidates=[Path(explicit)] if explicit else sorted(root.glob(str(number)+'*.py'))
    candidates=[p.resolve() for p in candidates if p.is_file() and (expected is None or digest(p)==expected)]
    # Duplicate downloads of identical bytes are harmless; different versions are ambiguous.
    unique={digest(p):p for p in candidates}
    require(len(unique)==1,f'Place the matching current Step {number} script beside Step 47; found {len(unique)} matching versions')
    return next(iter(unique.values()))


def load_audit_helper(root,support=None):
    """Use the shipped, reviewed validator instead of an arbitrary Step 15 version."""
    if support is not None:
        helper_path=Path(support).expanduser()
        helper_path=(helper_path if helper_path.is_absolute() else Path(root)/helper_path).resolve()
        require(helper_path.is_file(),'Audit module not found: '+str(helper_path))
    else:
        helper_path=(Path(root)/AUDIT_MODULE_FILENAME).resolve()
        if helper_path.is_file():
            require(digest(helper_path)==AUDIT_MODULE_SHA256,
                    'Audit companion hash changed; extract both original files from step47_fix_v3_1.zip')
        else:
            candidates=[p.resolve() for p in Path(root).glob('15*.py')
                        if p.is_file() and digest(p)==AUDIT_MODULE_SHA256]
            require(bool(candidates),
                    'Missing '+AUDIT_MODULE_FILENAME+'. Extract BOTH files from step47_fix_v3_1.zip '
                    'into the project folder. Existing Step 15 scripts/results do not need to be replaced or rerun.')
            helper_path=sorted(candidates)[0]
    helper=import_script(helper_path,'step47_audit15')
    required=('FIELDS','LABELS','SEEDS','REVISED_OUTPUT_REL','check_inputs','validate_effect_frame',
              'track_file','load_json','verify_selected','integers','read_csv','load_prediction',
              'verify_prediction_metrics','recheck_inputs')
    missing=[name for name in required if not hasattr(helper,name)]
    require(not missing,'Incompatible audit module '+str(helper_path)+'; missing: '+', '.join(missing)+
            '. Use the supplied '+AUDIT_MODULE_FILENAME)
    require(tuple(helper.FIELDS)==FIELDS and tuple(helper.LABELS)==LABELS and tuple(helper.SEEDS)==SEEDS
            and helper.REVISED_OUTPUT_REL==PREFIX,'Audit module schema/classes/seeds differ from shared_nominal_v2')
    require(all(callable(getattr(helper,name)) for name in required[4:]),'Audit module has non-callable validation functions')
    print('Using verified audit module: '+str(helper_path),flush=True)
    return helper,helper_path


def audit_sources(root,support=None,tst_root=None,pooling_root=None,lstm_root=None):
    # Reuse the already revised, backend-free Step 15 validator; its OUTPUT run is
    # not a dependency. Models are authenticated directly against Steps 12/13/14.
    helper,helper_path=load_audit_helper(root,support)
    _,_,context=helper.check_inputs(root,SEEDS,tst_root,pooling_root,lstm_root,return_context=True)
    observed=context['observed'];helper.track_file(helper_path,observed)
    heads={};code_hashes={};records={}
    for family in TRAINED:
        for seed in SEEDS:
            expected=context['expected'][(family,seed)]
            manifest=Path(expected['source_manifest']);folder=manifest.parent
            run=helper.load_json(manifest,observed)
            key='pool' if family in POOLS else family
            record=run if family=='TST' else run['identity']
            sha=record.get('script_sha256')
            require(isinstance(sha,str) and len(sha)==64,'Missing recorded model source: '+key)
            require(key not in code_hashes or code_hashes[key]==sha,'Mixed model code across seeds')
            code_hashes[key]=sha;records[key]=record
            if family=='TST':
                checkpoint='L10/checkpoints/best_tst_model_L10.pt'
                names={s:f'L10/predictions/{s}_predictions_L10.csv' for s in ('val','test')}
                cfg=helper.load_json(folder/'L10/reports/evaluation_metrics_L10.json',observed)
                model_config=cfg['model_config'];train_config=cfg['train_config']
            else:
                prefix=f'{MODES[family]}_seed{seed}' if family in POOLS else f'seed{seed}'
                checkpoint=(f'models/best_pooling_mlp_L10_{MODES[family]}_seed{seed}.pt' if family in POOLS
                            else f'models/best_lstm_finetuned_distilbert_L10_seed{seed}.pt')
                names={s:f'predictions/{prefix}_{s}_predictions.csv' for s in ('val','test')}
                cfg=record['configuration'];train_config=cfg['train_config']
                model_config=(dict(input_dim=1536 if family=='MeanMax_MLP' else 768,num_classes=6,
                    **{k:train_config[k] for k in ('hidden_dim','mlp_layers','dropout','use_input_layernorm')})
                    if family in POOLS else cfg['model_config'])
            helper.verify_selected(folder,run['artifacts'],[checkpoint,*names.values()],observed)
            if family!='TST':
                trial=run['completed_runs'][f'{MODES[family]}:{seed}' if family in POOLS else str(seed)]
                require(all(trial['artifacts'].get(n)==run['artifacts'][n] for n in names.values()),
                        'Validation/test trial artifact mismatch')
            provenance={k:context['source_contract'][k] for k in PROVENANCE_KEYS}
            provenance[f'step{SOURCE_ROLES[key]}_run_id']=run['run_id']
            heads[run_key(family,seed)]=dict(checkpoint=str(folder/checkpoint),
                predictions={s:str(folder/n) for s,n in names.items()},provenance=provenance,
                model_config=model_config,train_config=train_config,
                runtime=record['runtime'],settings=record['settings'])
    code_paths={k:resolve_code(root,SOURCE_ROLES[k],sha) for k,sha in code_hashes.items()}
    for path in code_paths.values():helper.track_file(path,observed)
    source=context['source_contract'];dataset=root/DATA_REL
    helper.verify_selected(dataset,source['input_sha256'],
        [f'splits/{s}_observable.csv' for s in ('train','val','test')]+['splits/label_mapping.json'],observed)
    root8=(root/ENC_REL).parent
    helper.verify_selected(root8,source['step8_model_sha256'],list(source['step8_model_sha256']),observed)
    for name in ('config.json','label_mapping.json','tokenizer_config.json'):
        require('trained_distilbert_model/'+name in source['step8_model_sha256'],'Missing encoder metadata: '+name)
    require(any('model' in n and n.endswith(('.safetensors','.bin')) for n in source['step8_model_sha256']),
            'Missing authenticated trained encoder weights')
    # No unrecorded tokenizer/weight companions may shadow the saved bundle.
    actual={p.relative_to(root8).as_posix() for p in (root/ENC_REL).rglob('*') if p.is_file()}
    require(actual==set(source['step8_model_sha256']),'Encoder bundle contains unrecorded/missing files')
    embedding_names=['embedding_extraction_report.json']
    sequence_names=['sequence_build_report_L10.json','sequence_build_report_index.json']
    for split in ('val','test'):
        embedding_names += [f'{split}_window_embeddings.npz',f'{split}_window_metadata.csv']
        sequence_names += [f'{split}_tst_sequences_L10.npz',f'{split}_tst_sequence_metadata_L10.csv',
                           f'{split}_tst_sequence_observable_effects_L10.csv']
    helper.verify_selected(root/EMB_REL,source['step9_artifacts'],embedding_names,observed)
    helper.verify_selected(root/SEQ_REL,source['step10_artifacts'],sequence_names,observed)
    context.update(helper=helper,helper_path=helper_path,heads=heads,code_paths=code_paths,code_hashes=code_hashes,
                   training_records=records,build=helper.load_json(dataset/'build_manifest.json',observed))
    return context


def load_cohorts(root,context):
    helper=context['helper'];source=context['source_contract']
    report=read_json(root/EMB_REL/'embedding_extraction_report.json')
    run9=read_json(root/EMB_REL/'run_manifest.json');model_map=read_json(root/ENC_REL/'label_mapping.json')
    require(report['embedding_encoder_source']=='finetuned' and report['use_fp16_inference'] is False
            and run9['encoder_source']=='finetuned','This shared-CLS comparison requires the completed FP32 Step 9 extraction')
    require(report['observable_fields']==list(FIELDS) and report['metadata_fields']==list(META)
            and report['label_order']==list(LABELS) and report['embedding_dtype']=='float32', 'Step 9 schema differs')
    require(all(report.get(k)==source[k] for k in PROVENANCE_KEYS[:2]) and report['run_id']==source['step9_run_id'],
            'Embedding report run provenance differs')
    require(model_map.get('dataset_id')==source['dataset_id'] and model_map.get('observable_fields')==list(FIELDS)
            and model_map.get('label_order')==list(LABELS) and model_map.get('label_to_id')==dict(zip(LABELS,range(6)))
            and model_map.get('serializer_version')=='observable22_fixed_order_is_v1'
            and report['max_length']==model_map.get('max_length'), 'Encoder schema/serializer/token budget differs')
    require(type(report['max_length']) is int and report['max_length']>2 and type(report['batch_size']) is int
            and report['batch_size']>0 and run9['protocol']['batch_size']==report['batch_size'], 'Invalid extraction batch/token budget')
    require(report['encoder_source_summary']['tokenizer_sha256']==run9['source_contract']['tokenizer_sha256'],
            'Step 9 tokenizer identity differs')
    context.update(extraction=report,encoder_runtime=run9['runtime'],max_length=report['max_length'],
                   encoder_batch_size=report['batch_size'],tokenizer_sha256=run9['source_contract']['tokenizer_sha256'])
    train_eps=set(read_records(root/DATA_REL/'splits/train_observable.csv').episode_id)
    sequence_report=read_json(root/SEQ_REL/'sequence_build_report_L10.json')
    result={};previous=train_eps
    for split in ('val','test'):
        frame=read_records(root/DATA_REL/f'splits/{split}_observable.csv')
        for name in ('episode_step','window_id'):frame[name]=helper.integers(frame[name],name)
        counts=context['build']['counts'][split];detail=sequence_report['splits'][split]
        require(len(frame)==counts['rows'] and frame.episode_id.nunique()==counts['episodes'],'Wrong shared-nominal population')
        require(list(frame)==list(META)+list(FIELDS) and set(frame.label)==set(LABELS),'Wrong exact 22-field schema/classes')
        require(not frame[list(FIELDS)].astype(str).apply(lambda col:col.str.strip().eq('')).any().any(),'Empty fields')
        require(frame.groupby('episode_id').label.nunique().eq(1).all(),'Mixed labels in episode')
        require(not previous.intersection(frame.episode_id),'Episode split leakage')
        previous=previous.union(frame.episode_id)
        with np.load(root/f'{EMB_REL}/{split}_window_embeddings.npz',allow_pickle=False) as z:
            e=z['embeddings'];row=z['row_indices'];labels=z['label_ids']
        require(e.shape==(len(frame),768) and e.dtype==np.float32 and np.isfinite(e).all(),'Invalid stored embeddings')
        require(np.array_equal(row,np.arange(len(frame))) and np.array_equal(labels,frame.label.map(dict(zip(LABELS,range(6))))),'Embedding row/label mismatch')
        wm=read_records(root/f'{EMB_REL}/{split}_window_metadata.csv')
        for name in META:
            require(np.array_equal(wm[name],frame[name]),'Window metadata alignment: '+name)
        with np.load(root/f'{SEQ_REL}/{split}_tst_sequences_L10.npz',allow_pickle=False) as z:
            ix=z['sequence_indices'];y=z['y']
        validate_indices(frame,ix,y)
        require(len(ix)==counts['sequences_stride1']['10']==detail['num_sequences']
                and len(frame)==detail['input_windows'], 'Wrong current Step 7/10 L10 count')
        meta=helper.read_csv(root/f'{SEQ_REL}/{split}_tst_sequence_metadata_L10.csv')
        require(len(meta)==len(ix) and np.array_equal(meta.sequence_id,np.arange(len(ix))),'Wrong sequence metadata order')
        require(np.array_equal(meta.sequence_start_row,ix[:,0]) and np.array_equal(meta.sequence_end_row,ix[:,-1]),'Sequence rows differ')
        require(np.array_equal(meta.label_id,y) and np.array_equal(meta.episode_id,frame.episode_id.to_numpy()[ix[:,-1]]) and
                np.array_equal(meta.sequence_end_step,frame.episode_step.to_numpy()[ix[:,-1]]),'Sequence endpoints differ')
        require(set(y)==set(range(6)) and meta.source_split.eq(split).all() and meta.sequence_length.eq(10).all(),
                'Incomplete classes or wrong split/length')
        require(hashlib.sha256(np.asarray(ix[:,-1],dtype='<i8').tobytes()).hexdigest()==detail['end_row_indices_sha256'],
                'Endpoint ordering differs from Step 10')
        effects=helper.read_csv(root/SEQ_REL/f'{split}_tst_sequence_observable_effects_L10.csv')
        effect_columns=helper.validate_effect_frame(effects,meta,10)
        ref=pd.concat([meta,effect_columns],axis=1)
        references={}
        for family in TRAINED:
            for seed in SEEDS:
                head=context['heads'][run_key(family,seed)];path=Path(head['predictions'][split])
                fields={**head['provenance'],'source_split':split}
                if family!='TST':fields['seed']=seed
                if family in POOLS:fields['pooling_mode']=MODES[family]
                pred,_=helper.load_prediction(path,ref,dict(sha256=context['observed'][str(path.resolve())],prediction_fields=fields))
                if split=='test':helper.verify_prediction_metrics(y,pred,context['expected'][(family,seed)])
                saved=helper.read_csv(path).sort_values('sequence_id').reset_index(drop=True)
                references[run_key(family,seed)]=check_predictions(saved,meta)
        result[split]=dict(frame=frame,embeddings=e,indices=ix,y=y,meta=meta,effects=effect_columns,references=references)
    helper.recheck_inputs(context['observed'])
    return result


def import_models(context):
    modules={}
    for key,path in context['code_paths'].items():
        require(digest(path)==context['code_hashes'][key],'Model source changed: '+key)
        modules[key]=import_script(path,'step47_source_'+key)
    return modules


def load_field_changes(root,context,cohort):
    """Read existing hash-bound paired nominal columns, without executing makers.

    Only test row identities are retained. Counterfactual values are never
    serialized, embedded, used to choose masks, or used to select a model.
    """
    frame=cohort['frame'];keys=['episode_id','episode_step']
    target=pd.MultiIndex.from_frame(frame[keys].astype(str))
    require(target.is_unique,'Duplicate diagnostic endpoint keys')
    changes=np.zeros((len(frame),22),bool);seen=np.zeros(len(frame),bool)
    baseline=['baseline_observable_'+f for f in FIELDS]
    audit=['raw_observable_difference_count','raw_observable_effect_present','raw_observable_difference_fields']
    columns=list(META)+list(FIELDS)+baseline+audit
    for source in context['build']['sources']:
        record=source.get('raw_audit',{})
        require(isinstance(record.get('sha256'),str) and len(record['sha256'])==64,
                'Step 7 lacks the original raw audit hash; do not regenerate data for Step 47')
        path=safe_artifact(root/'dataset',record['file'])
        require(path.is_file(),f'Missing original paired-nominal audit: {path}. Restore the existing file; do not regenerate it.')
        context['helper'].track_file(path,context['observed'],record['sha256'])
        require(path.stat().st_size==record['size_bytes'],'Raw audit size differs: '+str(path))
        header=pd.read_csv(path,encoding='utf-8-sig',nrows=0).columns
        require(set(columns)<=set(header),'Raw audit lacks paired-nominal projection columns: '+str(path))
        with pd.read_csv(path,encoding='utf-8-sig',usecols=columns,dtype=str,
                         keep_default_na=False,chunksize=10000) as chunks:
            for chunk in chunks:
                positions=target.get_indexer(pd.MultiIndex.from_frame(chunk[keys]))
                take=positions>=0
                if not take.any():continue
                part=chunk.loc[take].reset_index(drop=True);positions=positions[take]
                require(len(np.unique(positions))==len(positions) and not seen[positions].any(),
                        'Duplicate raw diagnostic rows across files/chunks')
                expected=frame.iloc[positions][list(META)+list(FIELDS)].astype(str).reset_index(drop=True)
                require(part[list(META)+list(FIELDS)].equals(expected) and part.label.eq(source['label']).all(),
                        'Raw audit rows do not match the evaluated observations')
                diff=part[list(FIELDS)].to_numpy()!=part[baseline].to_numpy()
                count=pd.to_numeric(part[audit[0]],errors='raise').to_numpy()
                flag=pd.to_numeric(part[audit[1]],errors='raise').to_numpy()
                names=np.array(['|'.join(np.asarray(FIELDS)[row]) or 'NONE' for row in diff])
                require(np.array_equal(diff.sum(1),count) and np.array_equal(diff.any(1).astype(int),flag)
                        and np.array_equal(names,part[audit[2]].to_numpy()),'Raw projection difference audit disagrees')
                require(not diff[part.label.eq('NORMAL').to_numpy()].any(),'NORMAL has paired-nominal projection changes')
                changes[positions]=diff;seen[positions]=True
    require(seen.all(),'Some test observations are absent from the authenticated raw companions')
    d=projection_diagnostics(changes,np.zeros_like(changes),cohort['indices'])
    for actual,name in [('retained_current_change','audit_current_observable_effect'),
                        ('retained_history_window_count','audit_history_effect_window_count'),
                        ('retained_context_window_count','audit_context_effect_window_count')]:
        require(np.array_equal(d[actual].astype(int),cohort['effects'][name].astype(int)),
                'Paired-nominal field differences disagree with Step 10: '+name)
    return changes


def projection_diagnostics(changes,mask,indices):
    changes=np.asarray(changes);mask=np.asarray(mask);indices=np.asarray(indices)
    require(changes.dtype==bool and mask.dtype==bool and changes.shape==mask.shape
            and changes.ndim==2 and changes.shape[1]==22,'Invalid field-change/mask matrices')
    require(indices.ndim==2 and indices.shape[1]==10 and indices.dtype.kind in 'iu'
            and indices.size>0 and indices.min()>=0 and indices.max()<len(changes),'Invalid diagnostic L10 indices')
    before=changes.any(1)[indices];after=(changes & ~mask).any(1)[indices]
    current=after[:,-1];history=after[:,:-1].sum(1);whole=after.sum(1)
    group=np.where(current,'current_retained',np.where(history>0,'history_only_retained',
                   np.where(before.any(1),'all_changes_masked','no_clean_change')))
    lag=np.where(whole>0,after[:,::-1].argmax(1),-1)
    return pd.DataFrame(dict(projection_stratum=group,clean_current_change=before[:,-1].astype(int),
        clean_context_window_count=before.sum(1),retained_current_change=current.astype(int),
        retained_history_window_count=history,retained_context_window_count=whole,
        retained_context_field_count=(changes & ~mask).sum(1)[indices].sum(1),
        latest_retained_lag=lag,current_change_masked_history_retained=before[:,-1] & ~current & (history>0)))


def diagnostic_rows(y,episodes,predictions,runs,diagnostics,condition,serialization):
    """Descriptive accuracy/class recall with support; no subset macro-F1 claim."""
    y=np.asarray(y);episodes=np.asarray(episodes);p=np.asarray(predictions)
    require(p.shape==(len(runs),len(y)) and len(diagnostics)==len(y),'Diagnostic alignment differs')
    groups={s:diagnostics.projection_stratum.eq(s).to_numpy() for s in PROJECTION_STRATA}
    groups['current_change_masked_history_retained']=diagnostics.current_change_masked_history_retained.to_numpy(bool)
    for lag in range(10):groups['latest_retained_lag_'+str(lag)]=diagnostics.latest_retained_lag.eq(lag).to_numpy()
    rows=[]
    for name,selected in groups.items():
        for label_id,label in [(-1,'ALL'),*enumerate(LABELS)]:
            ix=selected if label_id<0 else selected & (y==label_id)
            n=int(ix.sum());nep=len(np.unique(episodes[ix]))
            for family in dict.fromkeys(f for f,s in runs):
                ri=[i for i,(f,s) in enumerate(runs) if f==family]
                accuracy=(p[ri][:,ix]==y[ix]).mean(1) if n else None
                rows.append(dict(condition=condition,serialization=serialization,stratum=name,class_label=label,
                    family=family,head_seeds=len(ri),endpoints=n,episodes=nep,
                    mean_seed_accuracy=float(accuracy.mean()) if n else None,
                    head_seed_accuracy_sd=float(accuracy.std(ddof=1)) if n and len(ri)>1 else None,
                    interpretation='descriptive; class row is recall; no projection change is not proof of no class information'))
    return rows


def supplemental_summary(y,episodes,predictions,conditions,runs,serialization):
    y=np.asarray(y);p=np.asarray(predictions)
    require(p.shape==(len(conditions),len(runs),len(y)),'Supplemental prediction alignment differs')
    rows=[]
    for ci,condition in enumerate(conditions):
        for family in dict.fromkeys(f for f,s in runs):
            ri=[i for i,(f,s) in enumerate(runs) if f==family]
            scores=[metrics(y,p[ci,i]) for i in ri]
            normal=y==0;replay=y==4
            rows.append(dict(condition=condition,serialization=serialization,family=family,head_seeds=len(ri),
                endpoints=len(y),episodes=len(np.unique(episodes)),
                macro_f1=float(np.mean([s['macro_f1'] for s in scores])),
                accuracy=float(np.mean([s['accuracy'] for s in scores])),
                normal_false_alarm_rate=float((p[ci,ri][:,normal]!=0).mean()) if normal.any() else None,
                replay_recall=float((p[ci,ri][:,replay]==4).mean()) if replay.any() else None,
                role='supplementary fixed-checkpoint analysis; not selection'))
    return rows


def serialization_comparison(y,literal,omission,conditions,runs):
    literal=np.asarray(literal);omission=np.asarray(omission);y=np.asarray(y)
    require(literal.shape==omission.shape==(len(conditions),len(runs),len(y)),
            'Serialization comparison must use identical conditions, runs and endpoints')
    rows=[]
    for ci,condition in enumerate(conditions):
        for family in dict.fromkeys(f for f,s in runs):
            ri=[i for i,(f,s) in enumerate(runs) if f==family]
            delta=[metrics(y,omission[ci,i])['macro_f1']-metrics(y,literal[ci,i])['macro_f1'] for i in ri]
            changed=(omission[ci,ri]!=literal[ci,ri]).sum(1)
            rows.append(dict(condition=condition,family=family,endpoints=len(y),head_seeds=len(ri),
                mean_seed_decision_changes=float(changed.mean()),
                mean_seed_decision_change_fraction=float(changed.mean()/len(y)),
                omit_minus_literal_macro_f1=float(np.mean(delta)),
                role='paired descriptive serialization sensitivity; neither view selected on test'))
    return rows


def supplemental_paired_statistics(out,meta,selection):
    with np.load(out/'supplemental_predictions.npz',allow_pickle=False) as z:
        p=z['predictions'];y=z['y'];families=z['families'];styles=z['serializations']
        require(tuple(z['conditions'])==CONDITIONS and len(meta)==len(y)
                and tuple(styles)==('literal_UNK','omit_selected_clauses'), 'Supplementary statistics grid differs')
    selected=['TST',selection['selected_pool'],'WindowClassifier']
    take=np.flatnonzero(np.isin(families,selected));family=families[take]
    require([int((family==f).sum()) for f in selected]==[10,10,1], 'Missing fixed heads/current-window reference')
    pred=p[:,:,take].reshape(2*len(CONDITIONS),len(take),len(y))
    counts,strata=episode_counts(y,meta.episode_id.to_numpy(),pred)
    point=macro_f1(counts.sum(0));samples=bootstrap_scores(counts,strata)
    point=np.stack([point[:,family==f].mean(-1) for f in selected],axis=-1).reshape(2,4,3)
    samples=np.stack([samples[:,:,family==f].mean(-1) for f in selected],axis=-1).reshape(len(samples),2,4,3)
    rows=[]
    for si,style in enumerate(styles):
        for fi,comparator in enumerate(selected[1:],1):
            value=float((point[si,1:,0]-point[si,1:,fi]).mean())
            delta=(samples[:,si,1:,0]-samples[:,si,1:,fi]).mean(1)
            lo,hi=np.quantile(delta,[.025,.975]);blo,bhi=np.quantile(delta,[.05/8,1-.05/8])
            rows.append(dict(serialization=style,reference='TST',comparator=comparator,delta_macro_f1=value,
                ci95_low=float(lo),ci95_high=float(hi),bonferroni_low=float(blo),bonferroni_high=float(bhi),
                supplementary_family_size=4,primary_contrast=False,
                ci95_interpretation='reference_higher' if lo>0 else 'comparator_higher' if hi<0 else 'zero_included_not_equivalence',
                estimand='equal mean of three mask scores; per-head F1 mean, not ensemble',
                scope='paired class-stratified episodes; conditional on fixed heads/masks; supplementary family'))
    write_csv(out/'supplemental_paired_comparisons_missing8.csv',rows)
    write_array(out/'supplemental_bootstrap_distribution.npz',family_scores=samples,serializations=styles,
                conditions=np.array(CONDITIONS),families=np.array(selected))


def remaining_disk_bytes(out,n_windows,n_endpoints):
    # Six FP32 masked embeddings, primary/supplementary probabilities, bootstrap,
    # temporary atomic writes and review archive. An estimate, not a reservation.
    planned=int(1.5*(6*n_windows*768*4 + 8*50*n_endpoints*6*4 + 6*n_windows*6*4)
                + 512*1024**2)
    existing=sum(p.stat().st_size for p in (Path(out)/'cache').rglob('*') if p.is_file())
    return max(0,planned-existing)


def package_failure(out,exc):
    out=Path(out);folder=out.parent/'failures';folder.mkdir(parents=True,exist_ok=True)
    archive=folder/('step47_failed_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'.zip')
    evidence=dict(status='FAILED',code_version=CODE_VERSION,error_type=type(exc).__name__,error=str(exc),
        timestamp_utc=utc(),traceback=traceback.format_exc(),resume='Repeat the identical command after resolving the cause; keep completed caches.')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('failure.json',json.dumps(evidence,ensure_ascii=False,indent=2))
        z.write(Path(__file__),'source/'+Path(__file__).name)
        for name in ('protocol.json','input_audit.json','encoder_canary.json','validation_selection.json','progress.json'):
            if (out/name).is_file():z.write(out/name,name)
    return archive


def read_checkpoint(path):
    import torch
    # All checkpoints are local user-created and SHA-bound before loading.
    # Full dicts from these scripts are plain tensors/config/metrics.
    return torch.load(path,map_location='cpu',weights_only=True)


def restore_head(root,modules,family,seed,context):
    path=Path(paths_for(family,seed,context)[0])
    context['helper'].track_file(path,context['observed'],context['observed'][str(path)])
    c=read_checkpoint(path);validate_checkpoint(c,family,seed,context)
    key='pool' if family in POOLS else family;module=modules[key]
    import torch
    head=context['heads'][run_key(family,seed)]
    torch.set_num_threads(head['runtime'].get('torch_num_threads',head['runtime'].get('threads')))
    # Steps 13/14 set benchmark=True, while Step 11 records its deterministic
    # cuDNN setting. Do not leak Step 9's deterministic flag into the LSTM.
    torch.backends.cudnn.deterministic=(bool(head['settings']['DETERMINISTIC_CUDNN']) if family=='TST'
                                      else context['startup_cudnn_deterministic'])
    module.set_all_seeds(seed)
    cls={'TST':'TimeSeriesTransformerClassifier','pool':'MLPClassifier','LSTM':'FixedFineTunedEmbeddingLSTMClassifier'}[key]
    model=getattr(module,cls)(module.ModelConfig(**c['model_config']))
    model.load_state_dict(c['model_state_dict'],strict=True);model.eval().to('cuda')
    for param in model.parameters():param.requires_grad_(False)
    return model,c


def predict_head(model,checkpoint,family,e,indices):
    import torch
    b=int(checkpoint['train_config']['batch_size'])*(2 if family in POOLS else 1)
    # Step 11 evaluates in FP32; Steps 13/14 evaluate under autocast.
    amp=family!='TST' and bool(checkpoint['train_config']['use_amp'])
    result=np.empty((len(indices),6),dtype=np.float32)
    with torch.inference_mode():
        for lo in range(0,len(indices),b):
            x=e[indices[lo:lo+b]]
            if family in POOLS:
                mean=x.mean(1,dtype=np.float32);maximum=x.max(1)
                x=mean if family=='Mean_MLP' else maximum if family=='Max_MLP' else np.concatenate((mean,maximum),axis=1)
            x=torch.from_numpy(np.ascontiguousarray(x)).to('cuda',dtype=torch.float32)
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp):logits=model(x)
            # Match original NumPy FP32 softmax rather than half softmax.
            a=logits.float().cpu().numpy();a=a-a.max(1,keepdims=True);a=np.exp(a);a/=a.sum(1,keepdims=True)
            result[lo:lo+len(a)]=a
    return check_probs(result)


def encoder_probabilities(model,embeddings):
    import torch
    p=np.empty((len(embeddings),6),dtype=np.float32)
    with torch.inference_mode():
        for lo in range(0,len(embeddings),512):
            x=torch.from_numpy(np.asarray(embeddings[lo:lo+512]).copy()).to('cuda')
            x=model.dropout(torch.relu(model.pre_classifier(x)))
            logits=model.classifier(x)
            p[lo:lo+len(x)]=torch.softmax(logits,dim=-1).cpu().numpy()
    return check_probs(p)


def tokenizer_fingerprint(tokenizer):
    backend=json.loads(tokenizer.backend_tokenizer.to_str())
    backend.pop('padding',None);backend.pop('truncation',None)
    return object_hash(dict(backend=backend,special_tokens_map=tokenizer.special_tokens_map,
                            model_input_names=tokenizer.model_input_names))


def canary_count(total,batch,target=512):
    require(type(batch) is int and batch>0 and total>0,'Invalid encoder canary batch/population')
    return min(total,((min(total,target)+batch-1)//batch)*batch)


def encode_records(frame,tokenizer,model,context,callback=None,omit_mask=None):
    import torch
    output=np.empty((len(frame),768),dtype=np.float32);maximum=0
    batch=context['encoder_batch_size'];max_length=context['max_length']
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
        for lo in range(0,len(frame),batch):
            texts=serialize_frame(frame.iloc[lo:lo+batch],None if omit_mask is None else omit_mask[lo:lo+batch])
            tokens=tokenizer(texts,truncation=False,padding=True,return_tensors='pt')
            length=int(tokens['input_ids'].shape[1]);maximum=max(maximum,length)
            require(length<=max_length and len(tokens['input_ids'])==len(texts),'Token budget/row count differs; no truncation allowed')
            require(tokenizer.cls_token_id is not None and (tokens['input_ids'][:,0]==tokenizer.cls_token_id).all(),
                    'First token must be CLS')
            require(tokenizer.unk_token_id is None or not (tokens['input_ids']==tokenizer.unk_token_id).any(),
                    'Actual tokenizer [UNK] is forbidden; literal field value UNK uses the saved vocabulary')
            z=model.distilbert(input_ids=tokens['input_ids'].to('cuda'),attention_mask=tokens['attention_mask'].to('cuda'))
            output[lo:lo+len(texts)]=z.last_hidden_state[:,0,:].float().cpu().numpy()
            if callback and (lo==0 or (lo//batch)%100==0):callback(lo+len(texts),len(frame))
    require(np.isfinite(output).all(),'Nonfinite re-embeddings')
    return output,dict(rows=len(frame),batch_size=batch,max_length=max_length,precision='FP32',dynamic_padding=True,
                       serialization='literal_UNK' if omit_mask is None else 'omit_selected_clauses',
                       max_token_length=maximum,truncated_rows=0,tokenizer_UNK_rows=0)

def run_key(family,seed):return family+('/seed_'+str(seed) if seed is not None else '/deterministic')


def cached_probs(out,relative,identity,producer,validator=None,expected_rows=None):
    folder=out/relative;folder.mkdir(parents=True,exist_ok=True)
    contract=object_hash(dict(protocol=identity,stage=relative))
    done=read_completion(folder,'complete.json',contract)
    if done is None:
        probs=check_probs(producer())
        evidence=validator(probs) if validator else {}
        write_array(folder/'probabilities.npy',value=probs)
        write_completion(folder,'complete.json',contract,evidence,['probabilities.npy'])
    else:probs=check_probs(np.load(folder/'probabilities.npy',allow_pickle=False))
    require(expected_rows is None or len(probs)==expected_rows,'Cached probability endpoint count differs')
    if done is not None and validator:validator(probs)
    return probs


def family_scores(scores):
    """scores [...,condition,run] -> [...,condition,family], average seed F1s."""
    return np.stack([scores[...,[i for i,(f,s) in enumerate(RUNS) if f==family]].mean(-1) for family in FAMILIES],axis=-1)


def interval_row(reference,comparator,point,samples,primary=False):
    lo,hi=np.quantile(samples,[.025,.975],method='linear')
    blo,bhi=np.quantile(samples,[.05/14,1-.05/14],method='linear')
    return dict(reference=reference,comparator=comparator,delta_macro_f1=float(point),delta_percentage_points=float(100*point),
                ci95_low=float(lo),ci95_high=float(hi),primary_contrast=primary,
                exploratory_family_size=7,bonferroni_low=float(blo),bonferroni_high=float(bhi),
                ci95_interpretation='reference_higher' if lo>0 else 'comparator_higher' if hi<0 else 'zero_included_not_equivalence')


def make_statistics(out,data,predictions,selection):
    y=data['test']['y'];meta=data['test']['meta']
    counts,strata=episode_counts(y,meta.episode_id.to_numpy(),predictions)
    point=macro_f1(counts.sum(0));group=family_scores(point)
    sampled=bootstrap_scores(counts,strata);group_sampled=family_scores(sampled)
    missing=group[1:].mean(0);missing_sampled=group_sampled[:,1:].mean(1)
    rows=[]
    for ci,condition in enumerate(CONDITIONS):
        for ri,(family,seed) in enumerate(RUNS):
            met=metrics(y,predictions[ci,ri])
            rows.append(dict(condition=condition,family=family,seed=seed,macro_f1=met['macro_f1'],accuracy=met['accuracy']))
    write_csv(out/'metrics_by_run.csv',rows)
    summary=[]
    for fi,family in enumerate(FAMILIES):
        ix=[i for i,(f,s) in enumerate(RUNS) if f==family]
        mask_averaged_by_seed=point[1:][:,ix].mean(0)
        summary.append(dict(family=family,head_seeds=len(ix),mask_realizations=3,
                            clean_macro_f1=float(group[0,fi]),missing8_macro_f1=float(missing[fi]),
                            missing_minus_clean=float(missing[fi]-group[0,fi]),
                            clean_head_seed_sd=float(point[0,ix].std(ddof=1)) if len(ix)>1 else None,
                            missing8_head_seed_sd=float(mask_averaged_by_seed.std(ddof=1)) if len(ix)>1 else None,
                            missing8_mask_sd=float(group[1:,fi].std(ddof=1)),
                            val_selected_pool=family==selection['selected_pool']))
    write_csv(out/'model_summary.csv',summary)
    contrasts=[]
    for fi,family in enumerate(FAMILIES[1:],1):
        contrasts.append(interval_row('TST',family,missing[0]-missing[fi],missing_sampled[:,0]-missing_sampled[:,fi],family==selection['selected_pool']))
    write_csv(out/'paired_comparisons_missing8.csv',contrasts)
    degradation=[]
    for fi,family in enumerate(FAMILIES):
        samples=missing_sampled[:,fi]-group_sampled[:,0,fi];lo,hi=np.quantile(samples,[.025,.975])
        degradation.append(dict(family=family,missing_minus_clean=float(missing[fi]-group[0,fi]),ci95_low=float(lo),ci95_high=float(hi),role='descriptive_paired_degradation'))
    write_csv(out/'clean_to_missing8.csv',degradation)
    write_array(out/'bootstrap_distribution.npz',family_scores=group_sampled,conditions=np.array(CONDITIONS),families=np.array(FAMILIES))
    write_array(out/'episode_confusions.npz',counts=counts,class_strata=strata,
                episodes=np.array(sorted(meta.episode_id.unique()),dtype=str))
    perclass=[]
    cms=counts.sum(0)
    for ci,condition in enumerate(CONDITIONS):
        for fi,family in enumerate(FAMILIES):
            ix=[i for i,(f,s) in enumerate(RUNS) if f==family];cm=cms[ci,ix]
            den=cm.sum(-1)+cm.sum(-2);tp=np.diagonal(cm,axis1=-2,axis2=-1)
            cf=np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den>0).mean(0)
            for k,label in enumerate(LABELS):perclass.append(dict(condition=condition,family=family,label=label,mean_seed_f1=float(cf[k])))
    write_csv(out/'per_class_f1.csv',perclass)
    text=(f"Step 47 complete: uniform label-blind Missing-8; 3 masks, 50 trained heads + 3 probability controls.\n"
          f"Validation-selected pooling family: {selection['selected_pool']}; EWMA alpha: {selection['ewma_alpha']}.\n"
          "Primary contrast: TST minus selected pooling, Missing-8 mean over masks and head seeds.\n"
          "20,000 paired class-stratified episode bootstrap draws; fixed heads/masks, not seed resampling.\n"
          "95% CI containing zero is not equivalence. Seven-comparator Bonferroni intervals are also saved.\n"
          "Uniform all-row masking is separate from Step18/19 structured masking. Selected slots already UNK need not change.\n"
          "Post-mask projection diagnostics and a current-window classifier are supplied separately; no endpoint filtering in primary metrics.\n"
          "Clause-omission sensitivity uses identical masks, checkpoints and endpoints; literal UNK remains primary.\n"
          "Projection changes are privileged paired-nominal diagnostics, not model features, a bound or a proof of identifiability.\n"
          "One encoder is fixed throughout; encoder superiority and a unique need for temporal order are not tested here.\n"
          "No NOS3, Generator B, fresh realization or learned temporal-statistics head in this run.\n")
    (out/'analysis_summary.txt').write_text(text,encoding='utf-8')


FINAL_FILES=['protocol.json','input_audit.json','validation_selection.json','validation_metrics.csv','encoder_canary.json',
             'evaluation_endpoints_L10.csv','window_mask_alignment.csv','mask_audits.json','masks.npz','predictions.npz',
             'metrics_by_run.csv','model_summary.csv','paired_comparisons_missing8.csv','clean_to_missing8.csv',
             'bootstrap_distribution.npz','episode_confusions.npz','per_class_f1.csv','analysis_summary.txt',
             'inference_sources_verified.json','clean_context_observable_effects_L10.csv',
             'supplemental_predictions.npz','omission_mask_audits.json','paired_nominal_field_changes.npz',
             'supplemental_model_summary.csv','postmask_projection_metrics.csv','postmask_projection_coverage.csv',
             'postmask_endpoint_diagnostics.csv','serialization_sensitivity.csv','supplemental_interpretation.json',
             'supplemental_paired_comparisons_missing8.csv','supplemental_bootstrap_distribution.npz']


def finish_run(out,identity):
    include=list(FINAL_FILES)
    include+=['source/'+p.name for p in sorted((out/'source').iterdir()) if p.is_file()]
    # Replay evidence and stage hashes are tiny; raw probability/embedding caches stay on disk.
    include += [p.relative_to(out).as_posix() for p in sorted((out/'cache').rglob('*complete.json'))]
    manifest=dict(status='MISSING8_AGGREGATION_COMPLETE',protocol_identity=identity,completed_at_utc=utc(),
                  code_version=CODE_VERSION,supplement_version=SUPPLEMENT_VERSION,
                  protocol_version=PROTOCOL_VERSION,dataset_id=read_json(out/'protocol.json')['source_contract']['dataset_id'],
                  trained_heads=50,deterministic_controls=3,mask_realizations=3,training=False,
                  test_used_for_selection=False,output_sha256={n:digest(out/n) for n in include},
                  review_archive_omits='Selected weights, probability arrays and re-embeddings remain in project/run directories.')
    write_json(out/'run_manifest.json',manifest)
    return package_review(out,manifest)


def package_review(out,manifest):
    include=list(manifest['output_sha256'])
    zip_path=out.parent/(out.name+'_review.zip');temporary=zip_path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name in include+['run_manifest.json']:archive.write(out/name,name)
    with zipfile.ZipFile(temporary) as archive:require(archive.testzip() is None,'Review ZIP integrity failure')
    replace_atomic(temporary,zip_path)
    return zip_path


def evaluation_runtime(context):
    import torch
    import transformers
    require(torch.cuda.is_available(),'Use the original CUDA environment; CPU full inference is disabled')
    runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,torch=str(torch.__version__),
        transformers=transformers.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(torch.cuda.current_device()),device_index=torch.cuda.current_device(),
        threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        startup_cudnn_deterministic=torch.backends.cudnn.deterministic,
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    for record in [context['encoder_runtime'],*[r['runtime'] for r in context['training_records'].values()]]:
        for key in ('python','numpy','pandas','torch','transformers','cuda','cudnn'):
            if key in record:require(str(record[key])==str(runtime[key]),'Original runtime differs: '+key)
        require(record.get('gpu',record.get('device_name'))==runtime['gpu'],'Use the original GPU environment')
        if 'device' in record:require(str(record['device']).startswith('cuda'),'The recorded head was not trained/evaluated on CUDA')
        threads=record.get('torch_num_interop_threads',record.get('interop_threads'))
        require(threads is None or threads==runtime['interop_threads'],'Original interop thread policy differs')
    context['startup_cudnn_deterministic']=bool(runtime['startup_cudnn_deterministic'])
    return runtime


def make_protocol(context,data,runtime):
    protocol=dict(version=PROTOCOL_VERSION,script_sha256=digest(Path(__file__)),
        code_version=CODE_VERSION,supplement_version=SUPPLEMENT_VERSION,
        source_hashes=context['code_hashes'],audit_script_sha256=digest(context['helper_path']),
        source_contract=context['source_contract'],input_audit=context['observed'],runtime=runtime,
        checkpoint_contracts=context['heads'],fields=list(FIELDS),labels=list(LABELS),head_seeds=list(SEEDS),
        mask_seeds=list(MASK_SEEDS),conditions=list(CONDITIONS),runs=[list(r) for r in RUNS],
        populations={s:dict(windows=len(data[s]['frame']),episodes=int(data[s]['frame'].episode_id.nunique()),
            l10=len(data[s]['y']),eligible_episodes=int(data[s]['meta'].episode_id.nunique()),
            endpoint_identity=object_hash(data[s]['meta'].to_dict(orient='records'))) for s in ('val','test')},
        mask_rule='PCG64 uniform 8-of-22 without replacement independently per canonical CSV row; all rows; literal UNK',
        legacy_mask_equivalent=False,mask_uses_labels=False,training=False,test_used_for_selection=False,
        primary='TST minus clean-validation-selected best of Mean/Max/MeanMax MLP; all ten head seeds, all three masks',
        pool_selection='maximum mean per-seed clean validation Macro-F1; ties: Mean, Max, MeanMax',
        ewma_alpha_candidates=list(ALPHAS),ewma_selection='clean validation Macro-F1; ties: ascending alpha',
        ewma_initialization='first probability vector of each L10 window; recurrence reset each window',
        vote_tie_rule='majority count, then mean probability among tied classes, then lowest class ID',
        inference=dict(encoder='FP32 dynamic-padded batches; raw last-layer CLS from Step8 trained model',
            encoder_batch_size=context['encoder_batch_size'],max_length=context['max_length'],
            TST='FP32 original batch',pooling_and_LSTM='original checkpoint batch and AMP; pooling batch=2*train batch',
            probability_controls='Step8 classifier head on the SAME FP32 Step9 CLS; not Step8 Trainer FP16 replay',
            clean_replay='all 50 saved heads: validation and test decisions exact; original fixed probability tolerances'),
        diagnostics=dict(source='Hash-bound existing raw companions: actual minus paired-nominal 22 fields',
            groups=list(PROJECTION_STRATA),mask_both_records=True,counterfactual_values_are_model_inputs=False,
            metrics='Descriptive accuracy and per-class recall with endpoint/episode support, plus last retained lag 0..9',
            extra_group='Current projection change masked but some history change retained',
            no_information_bound=True,subgroups_select_nothing=True),
        supplement=dict(serializations=['literal_UNK','omit_selected_clauses'],same_masks=True,
            same_checkpoints=True,same_endpoints=True,same_validation_selection=True,
            reference='Original Step8 classifier at current L10 endpoint, not TST L1',
            omission='Remove only selected clauses, preserve field order and all other values; no new special tokens',
            primary_serialization_unchanged=True,posthoc_extension=True,
            contrasts='TST minus selected pooling and current-window classifier in both serializations',
            statistics='20,000 paired class-stratified episode draws; 95% and four-contrast Bonferroni 98.75% intervals; supplementary'),
        statistics=dict(resamples=N_BOOTSTRAP,seed=BOOTSTRAP_SEED,batch=BOOTSTRAP_BATCH,
            unit='class-stratified whole episode; same draw for all models, seeds, masks and clean condition',
            estimand='mean of per-head-seed six-class Macro-F1s, then mean of three masks; no probability ensemble',
            ci='95% linear percentile, conditional on encoder, trained heads, three masks and generator realization',
            multiplicity='one validation-selected primary contrast; seven-comparator Bonferroni intervals supplementary',
            population='all eligible L10 endpoints, including contexts with no observed clean projection change',
            seed_resampling=False,mask_resampling=False,equivalence_claim=False))
    return dict(identity=object_hash(protocol),**protocol)


def verify_clean_replay(probs,cohort,family,seed):
    ref=cohort['references'][run_key(family,seed)]
    require(probs.shape==ref.shape and np.array_equal(probs.argmax(1),ref.argmax(1)),
            f'{family}/{seed}: clean decisions do not reproduce; check source/runtime/precision')
    error=float(np.max(np.abs(probs-ref)))
    require(np.allclose(probs,ref,rtol=1e-4,atol=2e-5),
            f'{family}/{seed}: clean probabilities differ, max_abs={error}; do not tune tolerances on test')
    return dict(status='PASS',rows=len(probs),prediction_mismatches=0,max_abs_probability_error=error)


def compute_head_probabilities(root,context,modules,family,seed,cohort,embeddings):
    import torch
    model,checkpoint=restore_head(root,modules,family,seed,context)
    try:return predict_head(model,checkpoint,family,embeddings,cohort['indices'])
    finally:
        del model
        torch.cuda.empty_cache()


def head_probabilities(root,out,context,modules,family,seed,cohort,embeddings,split,condition,identity):
    def produce():return compute_head_probabilities(root,context,modules,family,seed,cohort,embeddings)
    validator=(lambda p:verify_clean_replay(p,cohort,family,seed)) if condition=='clean' else None
    return cached_probs(out,f'cache/{split}/{condition}/{run_key(family,seed)}',identity,produce,validator,len(cohort['y']))


def run_supplement(root,out,context,data,modules,tokenizer,encoder,identity,primary,window_predictions,selection):
    """Fixed supplementary views; preserve the primary grid consumed by Step 52."""
    cohort=data['test'];frame=cohort['frame'];ix=cohort['indices'];y=cohort['y']
    runs=(*RUNS,('WindowClassifier',None))
    literal=np.concatenate([primary,np.stack(window_predictions)[:,None,:]],axis=1)
    omitted=[literal[0].copy()];omission_audits=[]
    for condition in CONDITIONS[1:]:
        seed=int(condition.removeprefix('missing8_'));mask=make_mask(len(frame),seed)
        masked,audit=apply_mask(frame,mask);tag='omit_clauses_'+condition
        cache=out/'cache'/tag;cache.mkdir(parents=True,exist_ok=True)
        contract=object_hash(dict(protocol=identity,stage=tag))
        done=read_completion(cache,'embeddings_complete.json',contract)
        write_json(out/'progress.json',dict(stage='supplementary_omission',condition=condition,updated_at_utc=utc()))
        if done is None:
            print(f'Supplementary clause omission: re-embedding {condition}',flush=True)
            embeddings,encoding=encode_records(masked,tokenizer,encoder,context,
                lambda n,t:print(f'  {tag}: {n:,}/{t:,}',flush=True),omit_mask=mask)
            done=dict(mask_seed=seed,**audit,encoding=encoding)
            write_array(cache/'embeddings.npy',value=embeddings);write_json(cache/'mask_audit.json',done)
            write_completion(cache,'embeddings_complete.json',contract,done,['embeddings.npy','mask_audit.json'])
        else:embeddings=np.load(cache/'embeddings.npy',mmap_mode='r',allow_pickle=False)
        require(all(done.get(k)==v for k,v in dict(mask_seed=seed,**audit).items())
                and done['encoding']['serialization']=='omit_selected_clauses','Omission mask/serialization differs')
        require(embeddings.shape==(len(frame),768) and embeddings.dtype==np.float32 and np.isfinite(embeddings).all(),
                'Invalid supplementary embeddings')
        omission_audits.append(done);pred=[]
        for family in TRAINED:
            for head_seed in SEEDS:
                print(f'{tag}: {family} seed={head_seed}',flush=True)
                p=head_probabilities(root,out,context,modules,family,head_seed,cohort,embeddings,'test',tag,identity)
                pred.append(p.argmax(1).astype(np.int8))
        wp=cached_probs(out,f'cache/test/{tag}/window_classifier',identity,
            lambda:encoder_probabilities(encoder,embeddings),expected_rows=len(frame))
        for method in ('mean','vote','ewma'):
            pred.append(aggregate_probabilities(wp,ix,method,selection['ewma_alpha']).argmax(1).astype(np.int8))
        pred.append(wp[ix[:,-1]].argmax(1).astype(np.int8));omitted.append(np.stack(pred))
        del embeddings,masked
    omitted=np.stack(omitted)
    write_array(out/'supplemental_predictions.npz',predictions=np.stack([literal,omitted]),y=y,
        serializations=np.array(['literal_UNK','omit_selected_clauses']),conditions=np.array(CONDITIONS),
        families=np.array([f for f,s in runs]),seeds=np.array([s if s is not None else -1 for f,s in runs]),
        dataset_id=np.array(context['source_contract']['dataset_id']),step47_protocol_identity=np.array(identity))
    write_json(out/'omission_mask_audits.json',omission_audits)
    write_array(out/'paired_nominal_field_changes.npz',changes=cohort['field_changes'],fields=np.array(FIELDS),
                diagnostic_only=np.array(True))
    summary=[];diagnostic=[];coverage=[];endpoint_diagnostics=[]
    for style,pred in [('literal_UNK',literal),('omit_selected_clauses',omitted)]:
        summary.extend(supplemental_summary(y,cohort['meta'].episode_id,pred,CONDITIONS,runs,style))
        for ci,condition in enumerate(CONDITIONS):
            mask=np.zeros((len(frame),22),bool) if ci==0 else make_mask(len(frame),MASK_SEEDS[ci-1])
            d=projection_diagnostics(cohort['field_changes'],mask,ix)
            diagnostic.extend(diagnostic_rows(y,cohort['meta'].episode_id,pred[ci],runs,d,condition,style))
            if style=='literal_UNK':
                aligned=pd.concat([cohort['meta'].reset_index(drop=True),d],axis=1)
                aligned.insert(0,'condition',condition);endpoint_diagnostics.append(aligned)
                for stratum in PROJECTION_STRATA:
                    for li,label in enumerate(LABELS):
                        take=d.projection_stratum.eq(stratum).to_numpy() & (y==li)
                        coverage.append(dict(condition=condition,projection_stratum=stratum,label=label,
                            endpoints=int(take.sum()),episodes=int(cohort['meta'].loc[take,'episode_id'].nunique())))
    write_csv(out/'supplemental_model_summary.csv',summary)
    write_csv(out/'postmask_projection_metrics.csv',diagnostic)
    write_csv(out/'postmask_projection_coverage.csv',coverage)
    write_csv(out/'postmask_endpoint_diagnostics.csv',pd.concat(endpoint_diagnostics,ignore_index=True))
    write_csv(out/'serialization_sensitivity.csv',serialization_comparison(y,literal,omitted,CONDITIONS,runs))
    write_json(out/'supplemental_interpretation.json',dict(version=SUPPLEMENT_VERSION,
        primary_serialization='literal_UNK',omission_role='paired supplementary sensitivity; no result-based selection',
        current_window_reference='Original Step 8 classifier on the same frozen CLS at each L10 endpoint. Not a newly trained TST L1.',
        diagnostics='Paired-nominal projection changes after removing the same fields from both records; not a truth-derived predictor.',
        subgroup_metrics='Descriptive endpoint accuracy and per-class recall, with support and head-seed dispersion; no independent-window inference.',
        scope='Aggregation under missing observations with one fixed encoder. Neither encoder superiority nor temporal-order necessity is established.',
        no_change='No retained paired-nominal field change does not prove absence of class information.',
        chronology='Supplement defined after reviewing clean encoder comparison and NOS3 sensitivity; not preregistered research.',
        main_result='All eligible endpoints remain in primary comparisons. Subgroups never select checkpoints, masks, thresholds or families.'))


def load_encoder(root,context):
    import transformers
    tokenizer=transformers.AutoTokenizer.from_pretrained(root/ENC_REL,use_fast=True,local_files_only=True)
    require(tokenizer.padding_side=='right' and tokenizer_fingerprint(tokenizer)==context['tokenizer_sha256'],
            'Original tokenizer fingerprint/padding differs')
    encoder,loading=transformers.DistilBertForSequenceClassification.from_pretrained(
        root/ENC_REL,local_files_only=True,output_loading_info=True)
    require(not any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),
            'Trained encoder did not restore exactly')
    require(encoder.config.num_labels==6 and encoder.config.dim==768 and
        [encoder.config.id2label[i] for i in range(6)]==list(LABELS) and
        encoder.config.label2id==dict(zip(LABELS,range(6))) and
        getattr(encoder.config,'step7_dataset_id',None)==context['source_contract']['dataset_id'] and
        getattr(encoder.config,'observable_serializer_version',None)=='observable22_fixed_order_is_v1' and
        context['max_length']<=encoder.config.max_position_embeddings,'Encoder identity/schema differs')
    encoder.eval().to('cuda')
    for param in encoder.parameters():param.requires_grad_(False)
    return tokenizer,encoder


def save_matching_json(path,value):
    if path.exists():require(read_json(path)==value,'Saved selection/input record changed: '+str(path))
    else:write_json(path,value)


def verify_final(out,identity):
    manifest=read_json(out/'run_manifest.json')
    require(manifest.get('status')=='MISSING8_AGGREGATION_COMPLETE' and manifest.get('protocol_identity')==identity,
            'Incomplete or mismatched completed run')
    require(set(FINAL_FILES)<=set(manifest.get('output_sha256',{})),'Incomplete final artifact inventory')
    for name,sha in manifest['output_sha256'].items():
        require(digest(safe_artifact(out,name))==sha,'Completed review artifact changed: '+name)
    archive=out.parent/(out.name+'_review.zip')
    valid=False
    if archive.is_file():
        try:
            with zipfile.ZipFile(archive) as saved:
                valid=(set(saved.namelist())==set(manifest['output_sha256'])|{'run_manifest.json'}
                    and saved.read('run_manifest.json')==(out/'run_manifest.json').read_bytes()
                    and saved.testzip() is None)
        except (OSError,KeyError,zipfile.BadZipFile):
            valid=False
    if not valid:
        print('Rebuilding the review archive from verified completed outputs; no inference/statistics rerun.',flush=True)
        return package_review(out,manifest)
    return archive


def execute(root,out,support,context,data):
    runtime=evaluation_runtime(context)
    protocol=make_protocol(context,data,runtime);identity=protocol['identity']
    initialize_protocol(out,protocol)
    if (out/'run_manifest.json').exists():
        archive=verify_final(out,identity)
        print('All completed outputs verified; no inference/statistics or artifact rewrites.',flush=True)
        return archive
    save_matching_json(out/'input_audit.json',context['observed'])
    write_json(out/'progress.json',dict(stage='restore_and_clean_replay',updated_at_utc=utc()))
    (out/'source').mkdir(exist_ok=True)
    for path in (Path(__file__),context['helper_path'],*context['code_paths'].values()):
        target=out/'source'/path.name
        if target.exists():require(digest(target)==digest(path),'Saved model source changed')
        else:shutil.copyfile(path,target)
    modules=import_models(context)
    # Check every selected checkpoint's configuration/provenance before masked work.
    for family in TRAINED:
        for seed in SEEDS:validate_checkpoint(read_checkpoint(paths_for(family,seed,context)[0]),family,seed,context)
    tokenizer,encoder=load_encoder(root,context)
    folder=out/'cache/encoder_canary';folder.mkdir(parents=True,exist_ok=True)
    contract=object_hash(dict(protocol=identity,stage='encoder_canary'))
    canary=read_completion(folder,'complete.json',contract)
    if canary is None:
        n=canary_count(len(data['val']['frame']),context['encoder_batch_size'])
        fresh,info=encode_records(data['val']['frame'].iloc[:n],tokenizer,encoder,context)
        ref=data['val']['embeddings'][:n];error=float(np.max(np.abs(fresh-ref)))
        require(fresh.shape==ref.shape and np.allclose(fresh,ref,rtol=1e-5,atol=2e-5),
                f'Step9 encoder canary differs: max_abs={error}; stop before masked evaluation')
        canary=dict(status='PASS',max_abs_error=error,**info)
        write_json(folder/'evidence.json',canary)
        write_completion(folder,'complete.json',contract,canary,['evidence.json'])
    save_matching_json(out/'encoder_canary.json',canary)
    validation_rows=[]
    for family in TRAINED:
        for seed in SEEDS:
            print(f'Clean validation replay: {family} seed={seed}',flush=True)
            p=head_probabilities(root,out,context,modules,family,seed,data['val'],data['val']['embeddings'],'val','clean',identity)
            validation_rows.append(dict(family=family,seed=seed,macro_f1=metrics(data['val']['y'],p.argmax(1))['macro_f1']))
    valwp=cached_probs(out,'cache/val/clean/window_classifier',identity,
        lambda:encoder_probabilities(encoder,data['val']['embeddings']),expected_rows=len(data['val']['frame']))
    alpha_scores={a:metrics(data['val']['y'],aggregate_probabilities(valwp,data['val']['indices'],'ewma',a).argmax(1))['macro_f1'] for a in ALPHAS}
    selection=dict(selected_pool=choose_pool(validation_rows),ewma_alpha=choose_alpha(alpha_scores),
        ewma_validation_scores=[dict(alpha=a,macro_f1=alpha_scores[a]) for a in ALPHAS],
        candidate_pool_mean_validation_f1={f:float(np.mean([r['macro_f1'] for r in validation_rows if r['family']==f])) for f in POOLS},
        selection_data='clean validation only; no missing-mask or test metrics',all_ten_head_seeds_retained=True)
    save_matching_json(out/'validation_selection.json',selection)
    for family,method in [('ProbabilityMean','mean'),('MajorityVote','vote'),('EWMA','ewma')]:
        p=aggregate_probabilities(valwp,data['val']['indices'],method,selection['ewma_alpha'])
        validation_rows.append(dict(family=family,seed=None,macro_f1=metrics(data['val']['y'],p.argmax(1))['macro_f1']))
    write_csv(out/'validation_metrics.csv',validation_rows)
    print(f"Validation selections: pooling={selection['selected_pool']}, EWMA alpha={selection['ewma_alpha']}",flush=True)
    predictions=[];masks=[];mask_audits=[];window_predictions=[]
    for condition in CONDITIONS:
        write_json(out/'progress.json',dict(stage='primary_inference',condition=condition,updated_at_utc=utc()))
        if condition=='clean':embeddings=data['test']['embeddings']
        else:
            mask_seed=int(condition.removeprefix('missing8_'))
            mask=make_mask(len(data['test']['frame']),mask_seed)
            masked,mask_audit=apply_mask(data['test']['frame'],mask);masks.append(mask)
            cache=out/'cache'/condition;cache.mkdir(parents=True,exist_ok=True)
            contract=object_hash(dict(protocol=identity,stage=condition))
            done=read_completion(cache,'embeddings_complete.json',contract)
            if done is None:
                print(f'Re-embedding {condition}: {len(masked):,} records',flush=True)
                embeddings,encoding=encode_records(masked,tokenizer,encoder,context,
                    lambda n,t:print(f'  {condition}: {n:,}/{t:,}',flush=True))
                evidence=dict(mask_seed=mask_seed,**mask_audit,encoding=encoding)
                write_array(cache/'embeddings.npy',value=embeddings);write_json(cache/'mask_audit.json',evidence)
                write_completion(cache,'embeddings_complete.json',contract,evidence,['embeddings.npy','mask_audit.json'])
                done=evidence
            else:embeddings=np.load(cache/'embeddings.npy',mmap_mode='r',allow_pickle=False)
            require(embeddings.shape==(len(masked),768) and embeddings.dtype==np.float32 and np.isfinite(embeddings).all(),
                    'Cached missing embeddings have wrong shape/precision/values')
            require(all(done.get(k)==v for k,v in dict(mask_seed=mask_seed,**mask_audit).items()),'Cached mask realization differs')
            mask_audits.append(done);del masked
        condition_predictions=[]
        for family in TRAINED:
            for seed in SEEDS:
                print(f'{condition}: {family} seed={seed}',flush=True)
                p=head_probabilities(root,out,context,modules,family,seed,data['test'],embeddings,'test',condition,identity)
                condition_predictions.append(p.argmax(1).astype(np.int8))
        wp=cached_probs(out,f'cache/test/{condition}/window_classifier',identity,
            lambda:encoder_probabilities(encoder,embeddings),expected_rows=len(data['test']['frame']))
        window_predictions.append(wp[data['test']['indices'][:,-1]].argmax(1).astype(np.int8))
        for method in ('mean','vote','ewma'):
            condition_predictions.append(aggregate_probabilities(wp,data['test']['indices'],method,selection['ewma_alpha']).argmax(1).astype(np.int8))
        predictions.append(np.stack(condition_predictions));del embeddings
    predictions=np.stack(predictions)
    require(predictions.shape==(len(CONDITIONS),len(RUNS),len(data['test']['y'])),'Incomplete prediction grid')
    write_array(out/'predictions.npz',predictions=predictions,y=data['test']['y'],conditions=np.array(CONDITIONS),
        families=np.array([f for f,s in RUNS]),seeds=np.array([s if s is not None else -1 for f,s in RUNS]),
        dataset_id=np.array(context['source_contract']['dataset_id']),step47_protocol_identity=np.array(identity))
    write_array(out/'masks.npz',masks=np.stack(masks),mask_seeds=np.array(MASK_SEEDS),fields=np.array(FIELDS))
    write_json(out/'mask_audits.json',mask_audits)
    write_csv(out/'evaluation_endpoints_L10.csv',data['test']['meta'])
    write_csv(out/'clean_context_observable_effects_L10.csv',pd.concat([data['test']['meta'],data['test']['effects']],axis=1))
    wm=data['test']['frame'][list(META)].copy();wm.insert(0,'embedding_row_index',np.arange(len(wm)))
    write_csv(out/'window_mask_alignment.csv',wm)
    run_supplement(root,out,context,data,modules,tokenizer,encoder,identity,predictions,window_predictions,selection)
    # No statistics/final marker can certify outputs if any source changed during inference.
    context['helper'].recheck_inputs(context['observed'])
    require(digest(Path(__file__))==protocol['script_sha256'],'Step47 code changed during evaluation')
    write_completion(out,'inference_sources_verified.json',identity,
        dict(status='PASS',heads=50,primary_conditions=4,additional_omission_conditions=3,current_window_reference=True),
        ['predictions.npz','masks.npz','evaluation_endpoints_L10.csv','clean_context_observable_effects_L10.csv',
         'validation_selection.json','supplemental_predictions.npz','paired_nominal_field_changes.npz'])
    write_json(out/'progress.json',dict(stage='paired_episode_statistics',updated_at_utc=utc()))
    make_statistics(out,data,predictions,selection)
    print('Supplementary paired episode intervals: TST vs current window and selected pooling, both serializations.',flush=True)
    supplemental_paired_statistics(out,data['test']['meta'],selection)
    return finish_run(out,identity)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root',type=Path,default=BASE_DIR)
    parser.add_argument('--audit-script',type=Path,help='Optional compatible validator override; default uses the supplied taes_shared_nominal_audit.py')
    parser.add_argument('--tst-root',type=Path,help='Completed Step12 multiseed output root')
    parser.add_argument('--pooling-root',type=Path,help='Completed Step13 output root')
    parser.add_argument('--lstm-root',type=Path,help='Completed Step14 output root')
    parser.add_argument('--preflight-only',action='store_true',help='Audit files, hashes, cohorts, alignment; no GPU inference or output changes')
    args=parser.parse_args(argv);root=args.project_root.expanduser().resolve()
    support=(root/args.audit_script).resolve() if args.audit_script else None
    print('Step 47: '+CODE_VERSION+' | No training | Literal UNK primary + matched omission sensitivity',flush=True)
    print('Checking current 7-14 inputs and all 50 saved head checkpoints...',flush=True)
    out=root/OUT_REL
    try:
        context=audit_sources(root,support,args.tst_root,args.pooling_root,args.lstm_root);data=load_cohorts(root,context)
        print('Checking existing raw companions for post-mask diagnostic alignment; no data generation...',flush=True)
        data['test']['field_changes']=load_field_changes(root,context,data['test'])
        print(f"PASS: {len(context['observed'])} input files; val L10={len(data['val']['y']):,}; test L10={len(data['test']['y']):,}; all 50 heads found.",flush=True)
        estimate=remaining_disk_bytes(out,len(data['test']['frame']),len(data['test']['y']))
        free=shutil.disk_usage(root).free
        print(f'Disk free: {free/1024**3:.2f} GiB; conservative remaining estimate: {estimate/1024**3:.2f} GiB.',flush=True)
        require(free>=estimate,'Insufficient disk space; preserve source runs and use an existing drive with room')
        if args.preflight_only:
            print('PREFLIGHT ONLY: no inference or output changes. Tensor restore and clean replay remain to be checked on CUDA.',flush=True);return 0
        out.mkdir(parents=True,exist_ok=True)
        with RunLock(out/'.run.lock'):archive=execute(root,out,support,context,data)
    except Exception as exc:
        if not args.preflight_only:
            try:print('FAILURE_REVIEW_ZIP='+str(package_failure(out,exc)),flush=True)
            except OSError as archive_error:print('Could not package failure: '+str(archive_error),file=sys.stderr,flush=True)
        raise
    print(f'COMPLETE. UPLOAD_THIS={archive}',flush=True)
    return 0


if __name__=='__main__':
    try:sys.exit(main())
    except Exception as exc:
        print(f'\nSTOPPED: {type(exc).__name__}: {exc}\nExisting inputs were not modified. After resolving the cause, rerun the same command; completed stages are verified and reused.',file=sys.stderr,flush=True)
        raise
