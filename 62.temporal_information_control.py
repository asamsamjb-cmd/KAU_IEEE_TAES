#!/usr/bin/env python3
"""Step 62: paired temporal-information control, conditional on one frozen encoder.

Save beside the numbered scripts, then run in the project's existing .venv:
    python 62.temporal_information_control.py --preflight-only
    python 62.temporal_information_control.py
An interrupted run resumes at the last committed epoch in the SAME output folder.
Use --step49-run PATH if more than one completed Step49 parameter-stress run exists.

Both conditions train independently on ALL causal Step10 L10 endpoints:
actual_history = [e(t-9), ..., e(t)]; repeat_current = [e(t)] * 10.
Same frozen Step9 FP32 embeddings, original hash-verified Step11 model, ten paired
head seeds, initial tensors, shuffles, dropout streams, fixed30epoch budget and
validation-only checkpoint selection. No encoder training, input-label features,
augmentation, early stopping, condition-dependent LR scheduler, or test selection.
The original Step11 model is reused; this is a NEW matched head-training protocol
(FP32, fixed LR), not a replay of historical early-stopped/AMP checkpoints.

Test metadata and hashes are audited during preflight; no test prediction/metric
is produced until every train/validation job has a verified completion record.
Inference uses the same condition transform as its training. Intervals resample
whole episodes within class, paired across all heads/conditions; heads remain
fixed. They exclude encoder/head-training/generator uncertainty. This measures
additional history information in this synthetic distribution, not TST superiority,
external validity, temporal-order necessity, or a causal physical intervention.
Weights and cached features are never included in the review ZIP.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PureWindowsPath
import platform
import random
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

VERSION = 'shared_nominal_temporal_information_control_v1'
DATASET_ID = '478f782ceb970d7a2de73319ccc4e929be06cfea684a054aa72d932ed65ae83b'
TST_SOURCE_SHA = 'a2101c08cfecbd6af80975195a1bfde265462c0e7939c15ec06ca8d2e4d3967d'
PREFIX = Path('outputs/shared_nominal_v2')
DATA = Path('dataset/shared_nominal_v2')
ENCODER = PREFIX/'distilbert_window_baseline'
EMBEDDINGS = PREFIX/'tst_embeddings_finetuned_main'
SEQUENCES = PREFIX/'tst_sequences_finetuned_main'
OUTPUT = PREFIX/'62_temporal_information_control/run_information_control_v1'
LABELS = ('NORMAL','RF_LINK_JAMMING','GNSS_SPOOFING','UNAUTHORIZED_TELECOMMAND_INJECTION',
          'TELECOMMAND_REPLAY','GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE')
FIELDS = ('observation_stage','path_type','direction','source_role','destination_role',
          'protocol_context','operational_phase','message_type','service_context',
          'command_intent','payload_or_parameter_context','sequence_freshness_status',
          'control_security_state','signal_quality_state','tracking_stability',
          'navigation_consistency','timing_consistency','decoder_status',
          'command_execution_result','retry_timeout_state','session_state','state_transition_consistency')
META = ('scenario_id','episode_id','episode_type','episode_step','window_id','timestamp','label')
SPLITS = ('train','val','test')
CONDITIONS = ('actual_history','repeat_current')
SEEDS = tuple(range(20260610,20260620))
ARCHITECTURE = dict(input_dim=768,num_classes=6,sequence_length=10,temporal_context_seconds=10,
                    d_model=128,num_heads=4,num_encoder_layers=1,dim_feedforward=256,dropout=.25,pooling='last')
TRAINING = dict(epochs=30,batch_size=256,learning_rate=7e-5,weight_decay=5e-4,
                grad_clip_norm=1.,label_smoothing=.03,precision='FP32',amp=False,
                optimizer='AdamW',scheduler='none; fixed learning rate',early_stopping=False,
                selection='highest validation six-class Macro-F1; then lowest validation loss; then earliest epoch',
                augmentation=False,class_weights=False)


def require(condition, message):
    if not condition: raise ValueError(message)


def utc(): return datetime.now(timezone.utc).isoformat()


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def write_csv(path,frame):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    pd.DataFrame(frame).to_csv(temp,index=False,encoding='utf-8-sig',float_format='%.10g')
    temp.replace(path)


def safe_path(root,relative):
    root=Path(root).resolve();path=(root/relative).resolve()
    require(path!=root and root in path.parents,'Unsafe artifact path: '+str(relative))
    return path


def checked(path,expected,observed=None):
    path=Path(path).resolve()
    require(path.is_file(),'Required source is missing: '+str(path))
    actual=digest(path)
    require(actual==expected,'Source checksum mismatch: '+str(path))
    if observed is not None: observed[str(path)]=actual
    return path


def write_completion(folder,name,contract,result,artifacts):
    folder=Path(folder)
    write_json(folder/name,dict(contract=contract,result=result,
                               artifact_sha256={rel:digest(safe_path(folder,rel)) for rel in artifacts}))


def read_completion(folder,name,contract):
    folder=Path(folder);path=folder/name
    if not path.exists(): return None
    record=read_json(path)
    require(record.get('contract')==contract,'Completion contract mismatch: '+str(path))
    require(bool(record.get('artifact_sha256')),'Empty completion inventory: '+str(path))
    for rel,sha in record['artifact_sha256'].items(): checked(safe_path(folder,rel),sha)
    return record


def initialize_protocol(out,protocol):
    out=Path(out)
    if (out/'protocol.json').exists():
        require(read_json(out/'protocol.json')==protocol,
                'Protocol changed. Existing output is read-only to this run; choose a NEW --output-dir.')
    else:
        require(not out.exists() or not [p for p in out.iterdir() if p.name!='.step62.lock'],
                'Refusing a nonempty output folder without a matching Step62 protocol.')
        out.mkdir(parents=True,exist_ok=True)
        write_json(out/'protocol.json',protocol)


@contextlib.contextmanager
def run_lock(out):
    """OS lock automatically releases after interruption, including Windows termination."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    path=out/'.step62.lock'
    with path.open('a+b') as f:
        f.seek(0);f.write(b'0');f.flush();f.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc: raise ValueError('Another Step62 process owns this output folder.') from exc
        try: yield
        finally:
            f.seek(0)
            if os.name=='nt': msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(f,fcntl.LOCK_UN)


def find_step49(root,explicit=None):
    if explicit:
        p=Path(explicit);p=p if p.is_absolute() else root/p
        return p.resolve()
    candidates=[]
    for p in (root/PREFIX).glob('49*/run*/protocol.json'):
        try:
            protocol=read_json(p);manifest=read_json(p.parent/'run_manifest.json')
            if protocol.get('version')=='shared_nominal_A_parameter_stress_v1' and manifest.get('status')=='PARAMETER_STRESS_COMPLETE':
                candidates.append(p.parent.resolve())
        except (OSError,ValueError): pass
    require(len(candidates)==1,
            f'Expected one completed Step49 parameter-stress run, found {len(candidates)}. Use --step49-run PATH.')
    return candidates[0]


def audit_sources(root,step49_run=None,tst_source=None):
    """Authenticate existing source/encoder/artifacts without loading any model weights."""
    root=Path(root).resolve();folder=find_step49(root,step49_run)
    p=read_json(folder/'protocol.json');m=read_json(folder/'run_manifest.json');observed={}
    require(p.get('version')=='shared_nominal_A_parameter_stress_v1' and m.get('status')=='PARAMETER_STRESS_COMPLETE',
            'Step49 parameter-stress source is not complete.')
    require(p.get('identity')==m.get('protocol_identity')==object_hash({k:v for k,v in p.items() if k!='identity'}),
            'Step49 protocol identity mismatch')
    checked(folder/'protocol.json',m['output_sha256']['protocol.json'],observed)
    observed[str((folder/'run_manifest.json').resolve())]=digest(folder/'run_manifest.json')
    sc=p['source_contract'];require(sc['dataset_id']==DATASET_ID,'Wrong dataset identity')
    expected_runs=dict(step8_run_id='059d070ff62e5ddae8b3a596f6cd666b0bb0939ed9cf3098da1f4b78e8aaabb8',
                       step9_run_id='e39bea134697a8a44101bf111b7555b1cfd1f38a15eb118dd1067c9790ecad9b',
                       step10_run_id='5c49ef8835eb0e50d1825a4f39cbdeb08c40c4366744722b5ce3aac5dce6c266')
    require(all(sc[k]==v for k,v in expected_runs.items()),'Wrong fixed encoder/embedding/sequence run identity')
    hashes={sha for path,sha in p['input_audit'].items()
            if PureWindowsPath(path).name.startswith('11.') and path.endswith('.py')}
    require(hashes=={TST_SOURCE_SHA},'Step49 recorded TST implementation differs')
    code_candidates=([Path(tst_source) if Path(tst_source).is_absolute() else root/tst_source] if tst_source
                     else list(root.glob('11*.py'))+list((folder/'source').glob('11*.py')))
    matches=sorted({x.resolve() for x in code_candidates if x.is_file() and digest(x)==TST_SOURCE_SHA})
    require(bool(matches),'Cannot find hash-matched Step11 model source; use --tst-source PATH.')
    code=checked(matches[0],TST_SOURCE_SHA,observed)
    original=p['checkpoint_contracts']['TST/seed_20260610']['model_config']
    require(original==ARCHITECTURE,'Recorded original L10 architecture differs')
    for rel,sha in sc['input_sha256'].items(): checked(safe_path(root/DATA,rel),sha,observed)
    for rel,sha in sc['step8_model_sha256'].items(): checked(safe_path(root/ENCODER,rel),sha,observed)
    for role,directory in [('step8',ENCODER),('step9',EMBEDDINGS),('step10',SEQUENCES)]:
        path=checked(root/directory/'run_manifest.json',sc['manifest_sha256'][role],observed)
        require(read_json(path).get('run_id')==sc[role+'_run_id'],'Upstream run identity differs: '+role)
    embedding_names=['embedding_extraction_report.json']
    sequence_names=['sequence_build_report_L10.json','sequence_build_report_index.json']
    for split in SPLITS:
        embedding_names += [f'{split}_window_embeddings.npz',f'{split}_window_metadata.csv']
        sequence_names += [f'{split}_tst_sequences_L10.npz',f'{split}_tst_sequence_metadata_L10.csv']
    for directory,key,names in [(EMBEDDINGS,'step9_artifacts',embedding_names),(SEQUENCES,'step10_artifacts',sequence_names)]:
        for name in names: checked(root/directory/name,sc[key][name],observed)
    report=read_json(root/EMBEDDINGS/'embedding_extraction_report.json')
    require(report['embedding_encoder_source']=='finetuned' and report['use_fp16_inference'] is False
            and report['embedding_dtype']=='float32','Requires the existing frozen FP32 Step9 extraction')
    require(report['observable_fields']==list(FIELDS) and report['metadata_fields']==list(META)
            and report['label_order']==list(LABELS),'Step9 metadata/observable/label schema differs')
    require(all(report[k]==sc[k] for k in ['dataset_id','step8_run_id']) and report['run_id']==sc['step9_run_id'],
            'Embedding extraction provenance differs')
    mapping=read_json(root/ENCODER/'trained_distilbert_model/label_mapping.json')
    require(mapping.get('dataset_id')==DATASET_ID and mapping.get('observable_fields')==list(FIELDS)
            and mapping.get('label_order')==list(LABELS) and mapping.get('label_to_id')==dict(zip(LABELS,range(6)))
            and mapping.get('serializer_version')=='observable22_fixed_order_is_v1'
            and mapping.get('max_length')==report['max_length'],'Fixed encoder schema differs')
    return dict(source_contract=sc,step49_run=str(folder),step49_identity=p['identity'],
                model_source=str(code),input_sha256=observed,extraction=report)


def integer_array(value,name):
    a=np.asarray(value)
    require(np.issubdtype(a.dtype,np.integer),name+' must contain integers')
    return a.astype(np.int64,copy=False)


def validate_population(frame,indices,y,split):
    require(set(META)<=set(frame),'Missing episode metadata')
    indices=integer_array(indices,'sequence_indices');y=integer_array(y,'labels')
    require(indices.ndim==2 and indices.shape[1]==10 and len(indices)>0 and y.shape==(len(indices),),'Invalid L10 population shape')
    require(indices.min()>=0 and indices.max()<len(frame),'Sequence index out of range')
    require(not frame[['episode_id','episode_step']].duplicated().any(),'Duplicate episode/step')
    steps=integer_array(frame.episode_step.to_numpy(),'episode_step')
    require(frame.episode_id.astype(str).str.strip().ne('').all(),'Blank episode ID')
    labels=frame.label.map(dict(zip(LABELS,range(6))))
    require(labels.notna().all(),'Unknown labels')
    labels=labels.to_numpy(dtype=np.int64)
    require(frame.groupby('episode_id').label.nunique().eq(1).all(),'Mixed labels within episode')
    ep=frame.episode_id.to_numpy()[indices];seq_steps=steps[indices]
    require((ep==ep[:,:1]).all() and (np.diff(seq_steps,axis=1)==1).all(),'Noncausal or cross-episode sequence')
    times=pd.to_datetime(frame.timestamp,utc=True,errors='raise')
    require(not times.isna().any() and
            (np.diff(times.to_numpy(dtype='datetime64[ns]')[indices],axis=1)==np.timedelta64(1,'s')).all(),
            'Sequence timestamps must be consecutive one-second observations')
    require(np.array_equal(y,labels[indices[:,-1]]),'Sequence labels differ from current endpoint')
    complete=[]
    for positions in frame.groupby('episode_id',sort=False).indices.values():
        positions=np.asarray(positions);positions=positions[np.argsort(steps[positions],kind='stable')]
        require((np.diff(steps[positions])==1).all(),'Nonconsecutive episode steps')
        if len(positions)>=10: complete.append(np.lib.stride_tricks.sliding_window_view(positions,10))
    require(complete and np.array_equal(indices,np.concatenate(complete)),
            'Must retain ALL causal stride-1 L10 endpoints in original Step10 order')
    meta=frame.iloc[indices[:,-1]][list(META)].reset_index(drop=True).copy()
    meta.insert(0,'sequence_id',np.arange(len(indices),dtype=np.int64))
    meta['sequence_start_row']=indices[:,0];meta['sequence_end_row']=indices[:,-1]
    meta['sequence_end_step']=seq_steps[:,-1];meta['label_id']=y;meta['source_split']=split
    return meta


def check_split_disjoint(data):
    seen=set();audit={}
    for split in SPLITS:
        episodes=set(data[split]['frame'].episode_id.astype(str))
        require(not seen.intersection(episodes),'Episode split leakage: '+split)
        seen.update(episodes);audit[split]=len(episodes)
    return audit


def load_cohorts(root,context):
    root=Path(root);result={};report=read_json(root/SEQUENCES/'sequence_build_report_L10.json')
    for split in SPLITS:
        frame=pd.read_csv(root/DATA/f'splits/{split}_observable.csv',encoding='utf-8-sig',keep_default_na=False)
        require(list(frame)==list(META)+list(FIELDS) and set(frame.label)==set(LABELS),'Wrong observable schema/classes: '+split)
        wm=pd.read_csv(root/EMBEDDINGS/f'{split}_window_metadata.csv',encoding='utf-8-sig',keep_default_na=False)
        require(len(wm)==len(frame),'Window metadata length differs')
        for name in META: require(np.array_equal(wm[name],frame[name]),'Window metadata alignment: '+name)
        with np.load(root/EMBEDDINGS/f'{split}_window_embeddings.npz',allow_pickle=False) as z:
            # Test is verified for shape/finite values here but never scored or used for selection.
            embeddings=z['embeddings'];rows=z['row_indices'];labels=z['label_ids']
        require(embeddings.shape==(len(frame),768) and embeddings.dtype==np.float32 and np.isfinite(embeddings).all(),
                'Invalid FP32 frozen embedding matrix: '+split)
        require(np.array_equal(rows,np.arange(len(frame))) and np.array_equal(labels,frame.label.map(dict(zip(LABELS,range(6))))),
                'Embedding rows/labels differ')
        with np.load(root/SEQUENCES/f'{split}_tst_sequences_L10.npz',allow_pickle=False) as z:
            ix=z['sequence_indices'];y=z['y']
            for key,expected in [('start_row_indices',ix[:,0]),('end_row_indices',ix[:,-1]),('label_ids',y)]:
                require(np.array_equal(z[key],expected),'Step10 array mismatch: '+key)
        meta=validate_population(frame,ix,y,split)
        stored=pd.read_csv(root/SEQUENCES/f'{split}_tst_sequence_metadata_L10.csv',encoding='utf-8-sig',keep_default_na=False)
        require(len(stored)==len(meta),'Sequence metadata count differs')
        for key in ['sequence_id','sequence_start_row','sequence_end_row','sequence_end_step','episode_id','label_id','source_split']:
            require(np.array_equal(stored[key],meta[key]),'Step10 metadata alignment: '+key)
        require(stored.sequence_length.eq(10).all() and set(y)==set(range(6)),'Wrong L10/class coverage')
        detail=report['splits'][split]
        require(len(ix)==detail['num_sequences'] and len(frame)==detail['input_windows'],'Wrong Step10 population count')
        require(hashlib.sha256(np.asarray(ix[:,-1],dtype='<i8').tobytes()).hexdigest()==detail['end_row_indices_sha256'],
                'Step10 endpoint hash differs')
        result[split]=dict(frame=frame,embeddings=embeddings,indices=ix,y=y,meta=meta)
    check_split_disjoint(result)
    return result


def transform_batch(embeddings,indices,batch_ids,condition):
    require(condition in CONDITIONS,'Unknown temporal-information condition: '+str(condition))
    ix=indices[batch_ids]
    if condition=='repeat_current': ix=np.repeat(ix[:,-1:],10,axis=1)
    return np.asarray(embeddings[ix],dtype=np.float32)


def load_model_source(path,expected=TST_SOURCE_SHA):
    path=checked(path,expected)
    spec=importlib.util.spec_from_file_location('_step62_verified_step11',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    require(hasattr(module,'ModelConfig') and hasattr(module,'TimeSeriesTransformerClassifier'),'Missing original TST model')
    return module


def seed_all(seed):
    import torch
    random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def state_hash(state):
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        a=tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def cpu_state(model): return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def save_checkpoint(path,value):
    import torch
    path=Path(path);temp=path.with_name(path.name+'.tmp');torch.save(value,temp);temp.replace(path)


def load_checkpoint(path,expected):
    import torch
    checked(path,expected)
    with zipfile.ZipFile(path) as archive: require(archive.testzip() is None,'Checkpoint CRC error: '+str(path))
    return torch.load(path,map_location='cpu',weights_only=True)


def scores_from_counts(cm):
    """Return [..., macro-F1, normal FPR, six recalls]; fixed six-class zero division."""
    cm=np.asarray(cm,dtype=np.float64);tp=np.diagonal(cm,axis1=-2,axis2=-1)
    support=cm.sum(axis=-1);predicted=cm.sum(axis=-2);den=support+predicted
    f1=np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0)
    recall=np.divide(tp,support,out=np.zeros_like(tp),where=support>0)
    fpr=np.divide(support[...,0]-tp[...,0],support[...,0],out=np.zeros_like(support[...,0]),where=support[...,0]>0)
    return np.concatenate([f1.mean(axis=-1)[...,None],fpr[...,None],recall],axis=-1)


def prediction_metrics(y,pred):
    y=np.asarray(y);pred=np.asarray(pred)
    require(y.shape==pred.shape and y.ndim==1 and len(y)>0 and np.isin(y,range(6)).all() and np.isin(pred,range(6)).all(),
            'Invalid prediction shape or class ID')
    cm=np.bincount((y*6+pred).astype(np.int64),minlength=36).reshape(6,6);s=scores_from_counts(cm)
    return dict(macro_f1=float(s[0]),normal_fpr=float(s[1]),per_class_recall=dict(zip(LABELS,s[2:].tolist())),
                confusion_matrix=cm.tolist(),n=len(y))


def predict(model,cohort,condition,batch_size,device,label_smoothing=.03):
    import torch
    model.eval();all_logits=[];loss_sum=0.
    criterion=torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing,reduction='sum')
    with torch.no_grad():
        for start in range(0,len(cohort['y']),batch_size):
            ids=np.arange(start,min(start+batch_size,len(cohort['y'])))
            x=torch.from_numpy(transform_batch(cohort['embeddings'],cohort['indices'],ids,condition)).to(device)
            y=torch.from_numpy(cohort['y'][ids].astype(np.int64)).to(device)
            logits=model(x);require(bool(torch.isfinite(logits).all()),'Nonfinite inference logits')
            loss_sum+=float(criterion(logits,y).item());all_logits.append(logits.cpu().numpy())
    logits=np.concatenate(all_logits);pred=logits.argmax(axis=1)
    result=prediction_metrics(cohort['y'],pred);result['loss']=loss_sum/len(pred)
    return logits,result


def prediction_frame(cohort,logits,seed,condition):
    frame=cohort['meta'].copy();frame['seed']=seed;frame['condition']=condition
    frame['y_true']=cohort['y'];frame['y_pred']=logits.argmax(axis=1)
    shifted=logits-logits.max(axis=1,keepdims=True);probs=np.exp(shifted);probs/=probs.sum(axis=1,keepdims=True)
    for i in range(6): frame[f'prob_{i}']=probs[:,i]
    return frame


def job_contract(identity,seed,condition): return object_hash(dict(protocol=identity,seed=int(seed),condition=condition))


def job_folder(out,seed,condition): return Path(out)/f'seed_{seed}'/condition


def train_job(factory,data,config,folder,seed,condition,identity,device):
    """Transactional epoch resume. A interrupted partial epoch is restarted exactly."""
    import torch
    folder=Path(folder);contract=job_contract(identity,seed,condition)
    done=read_completion(folder,'trainval_complete.json',contract)
    if done: return done['result']
    folder.mkdir(parents=True,exist_ok=True)
    seed_all(seed);model=factory().to(device);initial=state_hash(model.state_dict())
    parameter_count=sum(p.numel() for p in model.parameters())
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    criterion=torch.nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    history=[];best_state=None;best_score=(-math.inf,-math.inf);best_epoch=0;start_epoch=1;order_hashes=[]
    pointer=folder/'resume.json'
    if pointer.exists():
        resume=read_json(pointer);require(resume['contract']==contract,'Resume protocol mismatch')
        state=load_checkpoint(safe_path(folder,resume['checkpoint']),resume['sha256'])
        require(state['contract']==contract and state['initial_state_sha256']==initial,'Resume initialization differs')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        for opt_state in optimizer.state.values():
            for k,v in opt_state.items():
                if isinstance(v,torch.Tensor): opt_state[k]=v.to(device)
        history=state['history'];best_state=state['best_model'];best_score=tuple(state['best_score'])
        best_epoch=state['best_epoch'];start_epoch=state['epoch']+1;order_hashes=state['order_hashes']
        require(len(history)==state['epoch'] and start_epoch<=config['epochs']+1,'Invalid resumed epoch count')
    for epoch in range(start_epoch,config['epochs']+1):
        # Separate deterministic epoch RNG makes partial-epoch restart independent of
        # model reconstruction, validation calls, and previous condition execution.
        dropout_seed=(seed*1000003+epoch*9176+31)%(2**63-1)
        seed_all(dropout_seed)
        order=np.random.default_rng(np.random.SeedSequence([seed,epoch,6201])).permutation(len(data['train']['y']))
        order_hash=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest();order_hashes.append(order_hash)
        model.train();loss_sum=0.;t0=time.perf_counter();updates=0
        for start in range(0,len(order),config['batch_size']):
            ids=order[start:start+config['batch_size']]
            x=torch.from_numpy(transform_batch(data['train']['embeddings'],data['train']['indices'],ids,condition)).to(device)
            y=torch.from_numpy(data['train']['y'][ids].astype(np.int64)).to(device)
            optimizer.zero_grad(set_to_none=True);logits=model(x);loss=criterion(logits,y)
            require(bool(torch.isfinite(loss)),'Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config['grad_clip_norm'],error_if_nonfinite=True)
            optimizer.step();loss_sum+=float(loss.item())*len(ids);updates+=1
        _,validation=predict(model,data['val'],condition,config['batch_size'],device,config['label_smoothing'])
        score=(validation['macro_f1'],-validation['loss'])
        if score>best_score: best_score=score;best_epoch=epoch;best_state=cpu_state(model)
        history.append(dict(epoch=epoch,train_loss=loss_sum/len(order),val_loss=validation['loss'],
                            val_macro_f1=validation['macro_f1'],best_epoch=best_epoch,
                            learning_rate=config['learning_rate'],updates=updates,
                            order_sha256=order_hash,dropout_seed=dropout_seed,elapsed_seconds=time.perf_counter()-t0))
        name=f'epoch_{epoch:03d}.pt'
        payload=dict(contract=contract,epoch=epoch,model=cpu_state(model),optimizer=optimizer.state_dict(),
                     best_model=best_state,best_score=list(best_score),best_epoch=best_epoch,history=history,
                     order_hashes=order_hashes,initial_state_sha256=initial,
                     torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        save_checkpoint(folder/name,payload)
        write_json(pointer,dict(contract=contract,checkpoint=name,sha256=digest(folder/name)))
        # Keep current and previous committed epoch; never delete upstream outputs.
        for old in folder.glob('epoch_*.pt'):
            if int(old.stem.split('_')[-1])<epoch-1: old.unlink()
        print(f'[seed {seed} {condition}] epoch {epoch}/{config["epochs"]} val Macro-F1={validation["macro_f1"]:.6f}',flush=True)
    require(best_state is not None and len(history)==config['epochs'],'Incomplete fixed-epoch training')
    model.load_state_dict(best_state)
    logits,metrics=predict(model,data['val'],condition,config['batch_size'],device,config['label_smoothing'])
    require(abs(metrics['macro_f1']-best_score[0])<1e-12,'Best validation checkpoint replay differs')
    save_checkpoint(folder/'best.pt',dict(contract=contract,model=best_state,best_epoch=best_epoch))
    write_csv(folder/'training_history.csv',history)
    write_csv(folder/'val_predictions.csv',prediction_frame(data['val'],logits,seed,condition))
    result=dict(seed=seed,condition=condition,epochs_completed=len(history),best_epoch=best_epoch,
                updates=sum(row['updates'] for row in history),parameter_count=parameter_count,
                initial_state_sha256=initial,best_state_sha256=state_hash(best_state),
                epoch_order_sha256=order_hashes,dropout_seeds=[row['dropout_seed'] for row in history],validation=metrics)
    write_json(folder/'validation_metrics.json',metrics)
    write_completion(folder,'trainval_complete.json',contract,result,
                     ['best.pt','training_history.csv','val_predictions.csv','validation_metrics.json'])
    return result


def verify_trainval_gate(out,seeds,identity):
    jobs=[];pairs=[]
    for seed in seeds:
        pair=[]
        for condition in CONDITIONS:
            record=read_completion(job_folder(out,seed,condition),'trainval_complete.json',job_contract(identity,seed,condition))
            require(record is not None,f'Test gate closed: train/validation incomplete for seed {seed} {condition}')
            pair.append(record['result']);jobs.append(dict(seed=seed,condition=condition,completion_sha256=digest(job_folder(out,seed,condition)/'trainval_complete.json')))
        for key in ['initial_state_sha256','epoch_order_sha256','dropout_seeds','epochs_completed','updates','parameter_count']:
            require(pair[0][key]==pair[1][key],f'Unmatched paired training: seed {seed}, {key}')
        pairs.append(dict(seed=seed,initial_state_sha256=pair[0]['initial_state_sha256'],
                          epoch_orders_sha256=object_hash(pair[0]['epoch_order_sha256']),
                          dropout_seeds_sha256=object_hash(pair[0]['dropout_seeds']),
                          epochs=pair[0]['epochs_completed'],updates=pair[0]['updates'],parameter_count=pair[0]['parameter_count'],
                          actual_best_epoch=pair[0]['best_epoch'],repeat_best_epoch=pair[1]['best_epoch'],matched=True))
    return jobs,pairs


def test_job(factory,cohort,config,folder,seed,condition,identity,device):
    import torch
    folder=Path(folder);contract=job_contract(identity,seed,condition)
    done=read_completion(folder,'test_complete.json',contract)
    if done: return done['result']
    trained=read_completion(folder,'trainval_complete.json',contract)
    require(trained is not None,'Test job has no completed training')
    state=load_checkpoint(folder/'best.pt',trained['artifact_sha256']['best.pt'])
    require(state['contract']==contract,'Best checkpoint contract differs')
    model=factory().to(device);model.load_state_dict(state['model'])
    logits,metrics=predict(model,cohort,condition,config['batch_size'],device,config['label_smoothing'])
    write_csv(folder/'test_predictions.csv',prediction_frame(cohort,logits,seed,condition))
    write_json(folder/'test_metrics.json',metrics)
    write_completion(folder,'test_complete.json',contract,metrics,['test_predictions.csv','test_metrics.json'])
    del model
    if str(device).startswith('cuda'): torch.cuda.empty_cache()
    return metrics


def episode_bootstrap(y,episodes,predictions,draws=20000,seed=2026100762):
    """Confusion counts per whole episode; shared stratified draws across fixed heads."""
    y=integer_array(y,'y');predictions=integer_array(predictions,'predictions');episodes=np.asarray(episodes).astype(str)
    require(y.ndim==1 and episodes.shape==y.shape and predictions.ndim==3 and predictions.shape[1:]==(2,len(y))
            and predictions.shape[0]>0 and set(y)==set(range(6)) and np.isin(predictions,range(6)).all(),'Invalid bootstrap inputs')
    unique,inverse=np.unique(episodes,return_inverse=True);strata=np.empty(len(unique),dtype=np.int64)
    for i in range(len(unique)):
        labels=np.unique(y[inverse==i]);require(len(labels)==1,'Mixed labels in bootstrap episode');strata[i]=labels[0]
    require(all((strata==c).sum()>=2 for c in range(6)),'Need at least two test episodes per class for episode bootstrap')
    counts=np.zeros((len(unique),predictions.shape[0],2,6,6),dtype=np.int64)
    for head in range(predictions.shape[0]):
        for condition in range(2): np.add.at(counts[:,head,condition],(inverse,y,predictions[head,condition]),1)
    point_by_head=scores_from_counts(counts.sum(axis=0));point=point_by_head.mean(axis=0)
    rng=np.random.default_rng(seed);samples=np.empty((draws,2,8),dtype=np.float64)
    flat=counts.reshape(len(unique),-1).astype(np.float64)
    for start in range(0,draws,64):
        n=min(64,draws-start);weights=np.zeros((n,len(unique)),dtype=np.float64)
        for c in range(6):
            ix=np.flatnonzero(strata==c)
            weights[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=n)
        cms=(weights@flat).reshape(n,predictions.shape[0],2,6,6)
        samples[start:start+n]=scores_from_counts(cms).mean(axis=1)
    difference=samples[:,0]-samples[:,1]
    def interval(value,draw):
        low,high=np.quantile(draw,[.025,.975])
        return dict(estimate=float(value),ci95_low=float(low),ci95_high=float(high))
    names=['macro_f1','normal_fpr',*['recall_'+name for name in LABELS]]
    result=dict(bootstrap_unit='whole episode, stratified by class',draws=draws,bootstrap_seed=seed,
                head_seed_resampling=False,estimator='mean of per-head metrics; no ensemble or pooled-head confusion',
                episodes=len(unique),episodes_per_class={LABELS[c]:int((strata==c).sum()) for c in range(6)},
                interval_scope='Conditional on this fixed encoder, these trained heads and this generator realization. '
                               'No encoder/head-training/generator/external-validity uncertainty; descriptive percentile intervals.',
                primary_contrast='actual_history minus repeat_current Macro-F1',multiplicity='one primary contrast; FPR and recalls descriptive')
    for i,name in enumerate(names):
        result[name+'_difference']=interval(point[0,i]-point[1,i],difference[:,i])
    result['conditions']={condition:{name:interval(point[c,i],samples[:,c,i]) for i,name in enumerate(names)}
                          for c,condition in enumerate(CONDITIONS)}
    result['head_seed_sd']={condition:{name:float(point_by_head[:,c,i].std(ddof=1)) if predictions.shape[0]>1 else 0.
                                      for i,name in enumerate(names)} for c,condition in enumerate(CONDITIONS)}
    return result,samples


def numeric_runtime(args):
    import torch
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    device='cuda:0' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
    require(not device.startswith('cuda') or torch.cuda.is_available(),'CUDA requested but unavailable; use --device cpu.')
    if device.startswith('cuda'): torch.cuda.set_device(torch.device(device))
    return device,dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,
                       device=device,device_name=torch.cuda.get_device_name(torch.device(device)) if device.startswith('cuda') else platform.processor(),
                       cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),threads=torch.get_num_threads(),
                       deterministic_algorithms=True,precision='FP32',amp=False,tf32=False,
                       cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


def package_review(out):
    """Only small source/audit/prediction/statistical files; no .pt/.npy/.npz arrays."""
    out=Path(out)
    include=sorted(p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()
                   and p.suffix in {'.json','.csv','.py','.txt'} and p.name not in {'resume.json','run_manifest.json'}
                   and not p.name.endswith('.tmp'))
    inventory={rel:digest(out/rel) for rel in include}
    manifest=dict(status='TEMPORAL_INFORMATION_CONTROL_COMPLETE',protocol_identity=read_json(out/'protocol.json')['identity'],
                  completed_at_utc=utc(),output_sha256=inventory,review_excludes='weights, optimizer states, embeddings, temporary files',
                  training=True,encoder_training=False,test_used_for_selection=False)
    write_json(out/'run_manifest.json',manifest)
    write_review_archive(out,manifest)


def write_review_archive(out,manifest):
    out=Path(out);temporary=out/'review_STEP62.zip.tmp'
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for rel in sorted(manifest['output_sha256'])+['run_manifest.json']: archive.write(out/rel,rel)
    temporary.replace(out/'review_STEP62.zip')


def review_archive_valid(out,manifest):
    out=Path(out);expected={**manifest['output_sha256'],'run_manifest.json':digest(out/'run_manifest.json')}
    try:
        with zipfile.ZipFile(out/'review_STEP62.zip') as archive:
            return (set(archive.namelist())==set(expected) and archive.testzip() is None and
                    all(hashlib.sha256(archive.read(rel)).hexdigest()==sha for rel,sha in expected.items()))
    except (OSError,ValueError,zipfile.BadZipFile,RuntimeError,EOFError):
        return False


def verify_final(out,identity):
    path=Path(out)/'run_manifest.json'
    if not path.exists(): return False
    manifest=read_json(path)
    require(manifest.get('status')=='TEMPORAL_INFORMATION_CONTROL_COMPLETE' and manifest.get('protocol_identity')==identity,
            'Existing final manifest differs')
    for rel,sha in manifest['output_sha256'].items(): checked(safe_path(out,rel),sha)
    protocol=read_json(Path(out)/'protocol.json')
    verify_trainval_gate(out,protocol['seeds'],identity)
    for seed in protocol['seeds']:
        for condition in CONDITIONS:
            require(read_completion(job_folder(out,seed,condition),'test_complete.json',
                                    job_contract(identity,seed,condition)) is not None,
                    'Completed run has an incomplete test job')
    if not review_archive_valid(out,manifest):
        # ZIP is a derived artifact. An interrupted finalization is repairable only
        # after the source inventory AND excluded local model weights verify above.
        write_review_archive(out,manifest)
        require(review_archive_valid(out,manifest),'Rebuilt review archive failed verification')
        print('Rebuilt review ZIP from fully verified completed artifacts.',flush=True)
    return True


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--step49-run',type=Path,help='Completed Step49 parameter-stress run (source hashes only).')
    p.add_argument('--tst-source',type=Path,help='Original hash-matched Step11 Python source.')
    p.add_argument('--output-dir',type=Path,help='Default: '+str(OUTPUT))
    p.add_argument('--preflight-only',action='store_true',help='Verify inputs, model and full populations; do not train or score test.')
    p.add_argument('--device',default='auto',help='auto, cpu, cuda:0, etc.; no WDDM process-name blocking.')
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--head-seeds',type=int,choices=[5,10],default=10,help='First5 or all10 predefined paired head seeds. Default10.')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    require(args.threads>0,'--threads must be positive')
    root=args.project_root.resolve();out=args.output_dir or root/OUTPUT
    if not out.is_absolute(): out=root/out
    out=out.resolve()
    require(out!=root and out not in [root/DATA,root/PREFIX,root/ENCODER,root/EMBEDDINGS,root/SEQUENCES],
            'Output must be a separate Step62 run folder')
    print('Step62: authenticating fixed encoder lineage, original TST, and all causal L10 endpoints.',flush=True)
    context=audit_sources(root,args.step49_run,args.tst_source)
    require(not any(out==Path(path) or out in Path(path).parents for path in context['input_sha256']),
            'Output cannot contain upstream source artifacts')
    data=load_cohorts(root,context)
    try: device,runtime=numeric_runtime(args)
    except ImportError as exc: raise ValueError('PyTorch is required. Run inside the existing project .venv.') from exc
    module=load_model_source(context['model_source']);factory=lambda:module.TimeSeriesTransformerClassifier(module.ModelConfig(**ARCHITECTURE))
    seed_all(SEEDS[0]);model=factory();parameter_count=sum(p.numel() for p in model.parameters());del model
    seeds=SEEDS[:args.head_seeds]
    audit={split:dict(windows=len(d['frame']),episodes=d['frame'].episode_id.nunique(),
                     endpoints=len(d['y']),endpoint_episodes=d['meta'].episode_id.nunique(),
                     endpoint_sha256=hashlib.sha256(np.asarray(d['indices'][:,-1],dtype='<i8').tobytes()).hexdigest(),
                     class_counts={LABELS[c]:int((d['y']==c).sum()) for c in range(6)}) for split,d in data.items()}
    protocol=dict(version=VERSION,script_sha256=digest(Path(__file__)),dataset_id=DATASET_ID,
                  source_contract=context['source_contract'],source_step49_identity=context['step49_identity'],
                  source_model_sha256=TST_SOURCE_SHA,input_sha256=context['input_sha256'],
                  encoder='One frozen original Step8 fine-tuned DistilBERT; existing FP32 Step9 CLS',
                  encoder_training=False,architecture=ARCHITECTURE,parameter_count=parameter_count,training=TRAINING,
                  conditions=dict(actual_history='embeddings[original causal L10 indices]',repeat_current='endpoint embedding repeated10; positional embeddings unchanged'),
                  seeds=list(seeds),pairing='same initialization, every epoch order, per-epoch dropout RNG reset, fixed budget; independent optimizers and validation selection',
                  resume='Restart an interrupted partial epoch from last SHA-checked commit; deterministic epoch RNG resets both conditions. Runtime must match.',
                  test_gate='Verify ALL trainval jobs before any test prediction or metric. Metadata/hashes audited beforehand.',
                  populations=audit,runtime=runtime,statistics=dict(draws=20000,seed=2026100762,unit='class-stratified whole episode',
                  estimator='mean per-head six-class Macro-F1',head_seed_resampling=False,contrast='actual_history minus repeat_current'),
                  scope='Conditional fixed-encoder/fixed-trained-head within-generator information control. No temporal-order, external validity or TST superiority claim.')
    protocol['identity']=object_hash(protocol)
    with run_lock(out):
        initialize_protocol(out,protocol)
        if verify_final(out,protocol['identity']):
            print('Verified completed run; nothing was overwritten. '+str(out/'review_STEP62.zip'));return 0
        write_json(out/'input_audit.json',dict(source_files=context['input_sha256'],populations=audit,episode_split_overlap=0,
                                              inputs='embeddings only; metadata and labels used for alignment/evaluation only'))
        for split in SPLITS: write_csv(out/f'endpoint_audit_{split}.csv',data[split]['meta'])
        (out/'source').mkdir(exist_ok=True)
        for path in [Path(__file__),Path(context['model_source'])]: shutil.copyfile(path,out/'source'/path.name)
        print(f"Preflight PASS: {parameter_count:,} parameters per condition; {len(seeds)} paired seeds; fixed{TRAINING['epochs']}epochs; {device}; FP32.",flush=True)
        for split,row in audit.items(): print(f'  {split}: {row["endpoints"]:,} complete causal L10 endpoints / {row["episodes"]:,} disjoint episodes',flush=True)
        if args.preflight_only:
            write_json(out/'preflight.json',dict(status='PREFLIGHT_PASS',protocol_identity=protocol['identity'],training=False,test_scoring=False))
            print('Preflight complete. Run the same command without --preflight-only to train.');return 0
        for seed in seeds:
            for condition in CONDITIONS:
                train_job(factory,data,TRAINING,job_folder(out,seed,condition),seed,condition,protocol['identity'],device)
        jobs,pairs=verify_trainval_gate(out,seeds,protocol['identity'])
        write_json(out/'trainval_gate.json',dict(status='ALL_TRAINVAL_JOBS_VERIFIED',jobs=jobs,opened_at_utc=utc()))
        write_csv(out/'pairing_audit.csv',pairs)
        predictions=np.empty((len(seeds),2,len(data['test']['y'])),dtype=np.int64);metric_rows=[]
        for i,seed in enumerate(seeds):
            for j,condition in enumerate(CONDITIONS):
                folder=job_folder(out,seed,condition)
                metrics=test_job(factory,data['test'],TRAINING,folder,seed,condition,protocol['identity'],device)
                pred=pd.read_csv(folder/'test_predictions.csv',encoding='utf-8-sig')
                require(np.array_equal(pred.sequence_id,data['test']['meta'].sequence_id) and np.array_equal(pred.y_true,data['test']['y']),
                        'Test prediction endpoint alignment differs')
                predictions[i,j]=pred.y_pred.to_numpy(dtype=np.int64)
                metric_rows.append(dict(seed=seed,condition=condition,macro_f1=metrics['macro_f1'],normal_fpr=metrics['normal_fpr'],
                                        **{'recall_'+label:value for label,value in metrics['per_class_recall'].items()}))
        summary,samples=episode_bootstrap(data['test']['y'],data['test']['meta'].episode_id.to_numpy(),predictions)
        summary.update(dataset_id=DATASET_ID,encoder=protocol['encoder'],head_seeds=list(seeds),parameter_count=parameter_count,
                       interpretation='Difference isolates access to actual additional historical embeddings for this matched TST head. '
                                      'A positive estimate is not evidence of temporal-order necessity or architecture superiority.')
        write_json(out/'summary.json',summary);write_csv(out/'metrics_by_seed.csv',metric_rows)
        names=['macro_f1','normal_fpr',*['recall_'+label for label in LABELS]]
        draws={'draw':np.arange(len(samples))}
        for k,name in enumerate(names):
            for j,condition in enumerate(CONDITIONS):draws[condition+'_'+name]=samples[:,j,k]
            draws['difference_'+name]=samples[:,0,k]-samples[:,1,k]
        write_csv(out/'episode_bootstrap.csv',draws)
        for path,sha in context['input_sha256'].items():checked(path,sha)
        require(digest(Path(__file__))==protocol['script_sha256'],'Step62 source changed during execution')
        package_review(out)
        effect=summary['macro_f1_difference']
        print('COMPLETE: actual-minus-repeat Macro-F1 = {estimate:.6f}; conditional episode95%CI [{ci95_low:.6f}, {ci95_high:.6f}]'.format(**effect))
        print('Review ZIP: '+str(out/'review_STEP62.zip'))
    return 0


if __name__=='__main__':
    try: raise SystemExit(main())
    except (ValueError,FileNotFoundError,KeyError,OSError) as exc:
        print('Step62 stopped safely: '+str(exc),file=sys.stderr);raise SystemExit(2)
