#!/usr/bin/env python3
"""TAES Step 51: shared_nominal_v2 train-time field-group comparison, five encoder seeds.

Place beside the revised 42-46 scripts in the existing project:
    python "51.field_group_train_ablation.py"
The same command resumes this run. Completed work is authenticated and skipped.
Output: outputs/shared_nominal_v2/51_field_group_train_ablation/run_shared_nominal_v2_review.zip

R2-2 / revised Table X: use the existing CategoricalEmbedding-MLP + TST family.
Reuse the authenticated Full predictions from Step46. Train seven original
Table-X field subsets, with identical Step45 budgets, vocabulary and RNG policy.
Dropped fields map to constant padding ID 0 at TRAIN/VAL/TEST, preserving model
shape and paired initialization. Metadata/labels never enter feature tensors.
All 35 new pipelines finish train/validation selection before new test inference.
Use --audit-only for a read-only check without loading the training engine.
Both feature extraction and evaluation use FP32, matching revised Steps45/46.

This is a retrospective, architecture-conditional information-content analysis.
It does not establish raw-vs-semantic superiority, causal sensor contribution,
TST necessity, operational validity, or information-theoretic necessity. A small
or nonsignificant Full/subset difference does not establish equivalence.
Encoder seed varies; temporal-head/shuffle seeds stay fixed as in Step45.
No hyperparameter search, BERT retraining, extra regimes or test-time masking.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
import importlib.metadata
import io
import json
import os
from pathlib import Path, PureWindowsPath
import platform
import shutil
import sys
import time
import traceback
import zipfile
import zlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

FIELDS=('observation_stage','path_type','direction','source_role','destination_role',
        'protocol_context','operational_phase','message_type','service_context',
        'command_intent','payload_or_parameter_context','sequence_freshness_status',
        'control_security_state','signal_quality_state','tracking_stability',
        'navigation_consistency','timing_consistency','decoder_status',
        'command_execution_result','retry_timeout_state','session_state','state_transition_consistency')
LABELS=('NORMAL','RF_LINK_JAMMING','GNSS_SPOOFING','UNAUTHORIZED_TELECOMMAND_INJECTION',
        'TELECOMMAND_REPLAY','GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE')
METADATA=('scenario_id','episode_id','episode_type','episode_step','window_id','timestamp','label')
SETTINGS={
    'Full':FIELDS,
    'ContextOnly':FIELDS[:7],
    'ContextService':FIELDS[:7]+('service_context',),
    'EvidenceOnly':FIELDS[7:],
    'EvidenceNoService':tuple(f for f in FIELDS[7:] if f!='service_context'),
    'MessageOnly':FIELDS[7:13],
    'ReceiverOnly':FIELDS[13:17],
    'TraceOnly':FIELDS[17:],
}
SEEDS=tuple(range(20260610,20260615))
PROTOCOL_VERSION='shared_nominal_train_group_v2'
DATA_REL='dataset/shared_nominal_v2'
FORMAL_REL='outputs/shared_nominal_v2/45_encoder_comparison_formal_trainval/run_formal_5seed'
FULL_TEST_REL='outputs/shared_nominal_v2/46_encoder_comparison_frozen_test/run_formal_5seed'
OUTPUT_REL='outputs/shared_nominal_v2/51_field_group_train_ablation/run_shared_nominal_v2'
SOURCE_KEYS=('dataset_id','step8_run_id','step9_run_id','input_sha256',
             'step8_model_sha256','step9_artifacts','tokenizer_sha256','observability_audit')
BOOTSTRAP_N=20000
BOOTSTRAP_SEED=2026100151


def require(ok,message):
    if not bool(ok):raise ValueError(message)


def utc():return datetime.now(timezone.utc).isoformat()


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def object_hash(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def replace(temp,path):
    for attempt in range(12):
        try:Path(temp).replace(path);return
        except PermissionError:
            if attempt==11:raise
            time.sleep(.25*(attempt+1))


def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    replace(temp,path)


def write_csv(path,rows,columns=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    pd.DataFrame(rows,columns=columns).to_csv(temp,index=False,encoding='utf-8-sig')
    replace(temp,path)


def write_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as stream:np.savez_compressed(stream,**arrays)
    replace(temp,path)


def safe(folder,name):
    relative=Path(name);folder=Path(folder).resolve()
    require(not relative.is_absolute() and not PureWindowsPath(name).drive and
            '..' not in relative.parts and '\\' not in name,'Unsafe relative artifact path')
    path=(folder/relative).resolve()
    require(folder in path.parents,'Artifact is outside its folder')
    return path


def checked(path,expected,log=None):
    path=Path(path)
    require(path.is_file(),'Required original file missing: '+str(path))
    require(digest(path)==expected,'Input/artifact hash mismatch: '+str(path))
    if log is not None:log.append({'path':str(path),'sha256':expected,'size_bytes':path.stat().st_size})


def authenticated_bytes(path,expected):
    content=Path(path).read_bytes()
    require(hashlib.sha256(content).hexdigest()==expected,'Input/artifact hash mismatch: '+str(path))
    return content


def verify_runtime(runtime,source):
    require(runtime==source['runtime'],
            'Step45 runtime mismatch; use the same environment/device/numerical policy. '
            +'; '.join(k for k in set(runtime)|set(source['runtime']) if runtime.get(k)!=source['runtime'].get(k)))


def prepare_runtime(engine,source):
    torch=engine.torch;numeric=source['numerical_runtime']
    require(source['protocols'][0]['device']=='cuda' and torch.cuda.is_available(),
            'This matched experiment requires the original Step45 CUDA environment; no CPU fallback')
    torch.cuda.set_device(numeric['cuda_device_index']);torch.set_num_threads(numeric['torch_num_threads'])
    torch.backends.cuda.matmul.allow_tf32=numeric['matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32=numeric['cudnn_allow_tf32']
    policy=engine.numerical_runtime()
    require(policy==numeric['effective_policy']==source['runtime']['numerical_policy'],
            'Effective numerical policy differs from the Full training source')
    device=torch.device('cuda',numeric['cuda_device_index'])
    runtime=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,
        pandas=pd.__version__,transformers=importlib.metadata.version('transformers'),platform=platform.platform(),
        device=torch.cuda.get_device_name(device),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),
        cuda_device_count=torch.cuda.device_count(),numerical_policy=policy)
    verify_runtime(runtime,source)
    return device,runtime


def seal(folder,name,identity,files,extra=None):
    data={'identity':identity,'artifact_sha256':{str(n):digest(safe(folder,str(n))) for n in files}}
    data.update(extra or {});write_json(Path(folder)/name,data);return data


def verify_record(folder,name,identity):
    path=Path(folder)/name
    if not path.exists():return None
    data=read_json(path);require(data['identity']==identity,'Completed-stage identity changed: '+str(path))
    require(bool(data.get('artifact_sha256')), 'Empty completion inventory: '+str(path))
    for n,h in data['artifact_sha256'].items():checked(safe(folder,n),h)
    return data


def bind_protocol(out,body):
    out=Path(out);path=out/'protocol.json'
    payload={'identity':object_hash(body),**body}
    if path.exists():require(read_json(path)==payload,'Resume protocol changed; original code/inputs/runtime required')
    else:
        require(not any(p.name!='.run.lock' for p in out.iterdir()),'Existing unrecognized output directory; no overwrite')
        write_json(path,payload)
    return payload['identity']


def mask_categories(categories,setting):
    require(setting in SETTINGS,'Unknown field setting')
    result=np.asarray(categories,dtype=np.int64).copy()
    require(result.ndim==2 and result.shape[1]==22,'Expected 22 categorical input slots')
    result[:,[i for i,f in enumerate(FIELDS) if f not in SETTINGS[setting]]]=0
    return result


def encode(frame,vocabulary):
    return np.stack([frame[f].astype(str).str.strip().map(vocabulary[f]).fillna(0).to_numpy(dtype=np.int64)
                     for f in FIELDS],axis=1)


def verify_vocab(train,vocabulary):
    require(set(vocabulary)==set(FIELDS),'Vocabulary field mismatch')
    for f in FIELDS:
        expected={v:i+1 for i,v in enumerate(sorted(train[f].astype(str).str.strip().unique()))}
        require(vocabulary[f]==expected,'Vocabulary is not exactly train-only: '+f)


def sequence_view(frame):
    require(not frame[['episode_id','episode_step']].duplicated().any(),'Duplicate episode/step')
    blocks=[]
    for _,positions in frame.groupby('episode_id',sort=False).indices.items():
        positions=np.asarray(positions)
        positions=positions[np.argsort(frame.iloc[positions].episode_step.to_numpy(),kind='stable')]
        part=frame.iloc[positions]
        require(part.label.nunique()==1,'Mixed-label episode')
        require(np.all(np.diff(part.episode_step.to_numpy(dtype=np.int64))==1),'Nonconsecutive episode steps')
        if len(positions)>=10:blocks.append(np.lib.stride_tricks.sliding_window_view(positions,10).copy())
    require(bool(blocks),'No complete L10 sequences')
    indices=np.concatenate(blocks).astype(np.int64)
    meta=frame.iloc[indices[:,-1]][list(METADATA)].reset_index(drop=True)
    meta['sequence_id']=np.arange(len(indices),dtype=np.int64)
    meta['sequence_start_row']=indices[:,0];meta['sequence_end_row']=indices[:,-1]
    meta['sequence_length']=10
    labels=meta.label.map({name:i for i,name in enumerate(LABELS)}).to_numpy(dtype=np.int64)
    return indices,labels,meta


def import_script(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    require(spec is not None and spec.loader is not None,'Cannot import '+str(path))
    module=importlib.util.module_from_spec(spec)
    previous=sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode=True
        spec.loader.exec_module(module)
    finally:sys.dont_write_bytecode=previous
    return module


def resolve_code(root,number,expected):
    candidates=[p for p in sorted(root.glob(str(number)+'*.py')) if p.is_file() and digest(p)==expected]
    require(candidates,'Place the revised Step '+str(number)+' code used by the source run beside Step51')
    return candidates[0]


def load_split(root,split,context):
    path=root/DATA_REL/'splits'/f'{split}_observable.csv'
    expected=context['build']['counts'][split];wanted=context['declared'][f'splits/{split}_observable.csv']
    content=authenticated_bytes(path,wanted)
    frame=pd.read_csv(io.BytesIO(content),encoding='utf-8-sig',dtype=str,keep_default_na=False)
    helper=context['helper']
    helper.validate_frame(frame,split,expected)
    helper.verify_episode_rows(frame,split,context['inventory'])
    ix,y,meta=sequence_view(frame)
    require(len(ix)==expected['sequences_stride1']['10'] and set(y)==set(range(6)),
            'Incomplete common L10 population: '+split)
    meta['y_true']=y
    return {'frame':frame,'indices':ix,'sequence_labels':y,'metadata':meta,
            'labels':frame.label.map(dict(zip(LABELS,range(6)))).to_numpy(dtype=np.int64)}


def preflight(root,formal,full_test):
    """Read-only NumPy/pandas source checks. No test-row parsing or model imports."""
    root,formal,full_test=map(lambda p:Path(p).resolve(),(root,formal,full_test))
    p46=read_json(full_test/'protocol.json')
    evaluator_path=resolve_code(root,46,p46.get('script_sha256'))
    evaluator=import_script(evaluator_path,'_step46_for51')
    require(evaluator.PROTOCOL_VERSION=='shared_nominal_formal_encoder_test_v3','Use current Step46')
    p46=evaluator.validate_saved_protocol(full_test)
    full_manifest=read_json(full_test/'run_manifest.json')
    full_manifest_sha=digest(full_test/'run_manifest.json')
    evaluator.verify_final_outputs(full_test,full_manifest,p46)
    record=evaluator.read_source(formal,verify_weights=False)
    source=record['protocol']
    require(p46['source_manifest_sha256']==record['manifest_sha256']
            and p46['source_protocol_identity']==source['identity'] and p46['jobs']==source['jobs']
            and p46['dataset_id']==source['dataset_id'], 'Full test and training source differ')
    paths=evaluator.resolve_code_paths(root,evaluator.source_code_hashes(record))
    known={x['sha256'] for x in source['source_files']}
    helpers=[p for p in sorted(root.glob('42*.py')) if p.is_file() and digest(p) in known]
    require(helpers,'Current Step42 input validator used by Step45 is required beside Step51')
    helper=import_script(helpers[0],'_step42_for51')
    require(helper.PROTOCOL=='shared_nominal_encoder_input_audit_v3','Use the revised shared-nominal validator')
    log=[]
    build,declared,inventory=helper.load_build(root,log)
    _,_,_,_,_,contract=helper.load_encoder_contract(root,root/helper.MODEL_REL,build,declared,log)
    require({k:contract[k] for k in SOURCE_KEYS}=={k:source['source_contract'][k] for k in SOURCE_KEYS}
            and contract['manifest_sha256']==source['source_contract']['manifest_sha256'],
            'Current dataset/encoder source differs from Step45')
    plan=source['statistics_plan'];counts=build['counts']['test']
    require(plan['expected_test_windows']==p46['test_windows']==counts['rows']
            and plan['expected_test_l10']==p46['test_l10']==counts['sequences_stride1']['10']
            and plan['expected_test_episodes']==p46['test_episodes']==counts['episodes']
            and p46['test_l10_episodes']==int(((inventory.split=='test')&(inventory.row_count>=10)).sum()),
            'Current test population differs from completed Full source')
    require(p46['test_sha256']==declared['splits/test_observable.csv']
            and p46['observability_audit']==contract['observability_audit'], 'Full test source differs')
    jobs={j['seed']:j for j in source['jobs'] if j['condition']=='CategoricalEmbedding_MLP'}
    require(set(jobs)==set(SEEDS) and len(jobs)==5,'Five paired Full encoder seeds required')
    vocabulary=record['vocabulary']
    require(vocabulary.get('source')=='train_only' and vocabulary.get('unknown_id')==0,
            'Unexpected train vocabulary policy')
    context=dict(root=root,formal=formal,full_test=full_test,source=source,record=record,
                 full_protocol=p46,full_manifest=full_manifest,full_manifest_sha256=full_manifest_sha,
                 jobs=jobs,vocab=vocabulary['fields'],
                 build=build,declared=declared,inventory=inventory,helper=helper,evaluator=evaluator,
                 engine_path=paths['43'],code_paths={**paths,'42':helpers[0],'46':evaluator_path},log=log)
    data={split:load_split(root,split,context) for split in ('train','val')}
    require(set(data['train']['frame'].episode_id).isdisjoint(data['val']['frame'].episode_id),
            'Train/validation episode leakage')
    verify_vocab(data['train']['frame'],context['vocab'])
    for split,item in data.items():
        name=f'{split}_window_metadata.csv'
        md=helper.read_input(root/helper.EMBEDDING_REL/name,log,'csv',contract['step9_artifacts'][name])
        require(helper.validate_metadata(item['frame'],md,split)['rows_reordered'] is False,
                'Source row order differs from the shared pipeline')
        item['categories']=encode(item['frame'],context['vocab'])
        checked(root/DATA_REL/f'splits/{split}_observable.csv',declared[f'splits/{split}_observable.csv'],log)
    val=data['val'];metadata=val['metadata'].drop(columns='y_true')
    endpoint=dict(population='common_L10_endpoints',window_rows=len(val['labels']),common_rows=len(val['indices']),
        eligible_episodes=int(metadata.episode_id.nunique()),
        end_row_indices_sha256=hashlib.sha256(val['indices'][:,-1].astype('<i8').tobytes()).hexdigest(),
        metadata_sha256=object_hash(metadata.to_dict(orient='records')),
        labels_sha256=hashlib.sha256(val['sequence_labels'].astype('<i8').tobytes()).hexdigest())
    require(endpoint==source['validation_endpoint_contract'],'Validation endpoints differ from Full training')
    # Bound source records and used payloads are rechecked before/after long work.
    for folder,manifest,wanted in ((formal,record['manifest'],record['manifest_sha256']),
                                    (full_test,full_manifest,full_manifest_sha)):
        checked(folder/'run_manifest.json',wanted,log)
        for name,wanted in manifest['output_sha256'].items():checked(safe(folder,name),wanted,log)
    code_hashes={**evaluator.source_code_hashes(record),'42':digest(helpers[0]),'46':p46['script_sha256']}
    for role,path in context['code_paths'].items():checked(path,code_hashes[role],log)
    context['data']=data
    return context


def load_engine(path,expected):
    checked(path,expected)
    engine=import_script(path,'_step43_for51')
    require(engine.DEPENDENCY_ERROR is None,'Step43 runtime dependency unavailable: '+str(engine.DEPENDENCY_ERROR))
    require(tuple(engine.FIELDS)==FIELDS and tuple(engine.LABELS)==LABELS,'Engine schema changed')
    # Preserve the source training implementation; only atomic I/O retries change.
    engine.write_json=write_json;engine.write_csv=write_csv
    def save_checkpoint(path,payload):
        path=Path(path);temp=path.with_name(path.name+'.tmp')
        engine.torch.save(payload,temp);replace(temp,path)
    engine.save_checkpoint=save_checkpoint
    return engine


def check_registered(out,index):
    for job,h in index.items():checked(safe(out,job+'/training_complete.json'),h)


def training_files(folder):
    names=[]
    for stage in ('encoder','tst'):
        complete=read_json(folder/stage/'complete.json')
        names += [stage+'/complete.json']+[stage+'/'+n for n in complete['artifact_sha256']]
    return names


def trained_identity(identity,seed,setting):return object_hash({'run':identity,'seed':seed,'setting':setting})


def train_one(engine,data,vocab,job,setting,out,identity,device):
    seed=job['seed'];folder=out/f'seed_{seed}'/setting
    stage=trained_identity(identity,seed,setting)
    completed=verify_record(folder,'training_complete.json',stage)
    if completed is not None:
        verify_training_record(folder,completed)
        return completed
    folder.mkdir(parents=True,exist_ok=True)
    engine.seed_all(seed)
    encoder=engine.CategoricalEncoder([len(vocab[f]) for f in FIELDS])
    model=engine.WindowModel(encoder,head_seed=job['protocol']['window_head_seed'])
    batches={s:engine.ArrayBatches({'x':mask_categories(d['categories'],setting)},d['labels']) for s,d in data.items()}
    enc=engine.fit(model,batches['train'],batches['val'],job['encoder_training'],folder/'encoder',stage+'/encoder',device)
    embeddings=engine.extract_features(model,batches,folder/'embeddings',job['encoder_training']['eval_batch_size'],
                                       device,object_hash({'stage':stage,'encoder_best':enc['artifact_sha256']['best.pt']}))
    del model,encoder,batches;engine.torch.cuda.empty_cache()
    temporal={s:engine.SequenceBatches(embeddings[s],d['indices'],d['sequence_labels']) for s,d in data.items()}
    engine.seed_all(job['protocol']['tst_seed']);tst=engine.TST()
    tst_identity=object_hash({'stage':stage,'embeddings':read_json(folder/'embeddings/complete.json')['sha256']})
    temporal_result=engine.fit(tst,temporal['train'],temporal['val'],job['tst_training'],folder/'tst',tst_identity,device)
    summary={'seed':seed,'setting':setting,'n_fields':len(SETTINGS[setting]),
             'encoder_val_macro_f1':enc['best_metrics']['macro_f1'],
             'l10_val_macro_f1':temporal_result['best_metrics']['macro_f1'],
             'encoder_best_epoch':enc['best_epoch'],'tst_best_epoch':temporal_result['best_epoch'],
             'encoder_convergence':enc['convergence'],'tst_convergence':temporal_result['convergence'],
             'encoder_training_seconds':enc['training_seconds'],'tst_training_seconds':temporal_result['training_seconds'],
             'evaluation_precision':'float32','extraction_precision':'float32',
             'window_head_seed':job['protocol']['window_head_seed'],'tst_seed':job['protocol']['tst_seed'],
             'encoder_optimizer_updates':enc['optimizer_updates'],'tst_optimizer_updates':temporal_result['optimizer_updates'],
             'encoder_epochs_completed':enc['epochs_completed'],'tst_epochs_completed':temporal_result['epochs_completed']}
    result=seal(folder,'training_complete.json',stage,training_files(folder),
                {'summary':summary,'completed_at_utc':utc(),
                 'embedding_provenance':read_json(folder/'embeddings/complete.json')})
    del tst,temporal,embeddings;gc.collect();engine.torch.cuda.empty_cache()
    # Only this new job's reproducible intermediate feature cache is removed.
    shutil.rmtree(folder/'embeddings')
    return result


def verify_training_record(folder,record):
    needed={stage+'/'+name for stage in ('encoder','tst')
            for name in ('complete.json','best.pt','last.pt','history.csv','val_logits.npy')}
    require(set(record['artifact_sha256'])==needed,'Incomplete training artifact inventory')
    require(record.get('embedding_provenance',{}).get('precision')=='float32'
            and set(record['embedding_provenance'].get('sha256',{}))=={'train','val'},
            'Training embedding provenance differs')
    for stage in ('encoder','tst'):
        complete=read_json(folder/stage/'complete.json')
        require(complete.get('evaluation_precision')=='float32','Selected checkpoint was not evaluated in FP32')
        for name,wanted in complete['artifact_sha256'].items():
            require(record['artifact_sha256'].get(stage+'/'+name)==wanted,'Training artifact cross-reference mismatch')


def aligned_prediction(frame,meta,y):
    require(len(frame)==len(meta),'Prediction endpoint count mismatch')
    require(set(meta)<=set(frame),'Prediction metadata columns missing')
    for name in meta:
        require(np.array_equal(frame[name],meta[name]),'Prediction endpoint mismatch: '+name)
    require(np.array_equal(frame.y_true,y),'Prediction truth mismatch')
    pred=frame.y_pred.to_numpy()
    require(np.isin(pred,np.arange(6)).all(),'Invalid predicted class')
    return pred.astype(np.int64)


def load_full_prediction(path,expected,meta,y,context,seed):
    content=authenticated_bytes(path,expected)
    frame=pd.read_csv(io.BytesIO(content),encoding='utf-8-sig',float_precision='round_trip',keep_default_na=False,
        dtype={k:str for k in ('scenario_id','episode_id','episode_type','timestamp','label','dataset_id',
                              'source45_protocol_identity','step46_protocol_identity')})
    context['evaluator'].validate_predictions(frame,meta,context['jobs'][seed],context['full_protocol'],'tst')
    return aligned_prediction(frame,meta,y)


def training_gate(out,identity,index,write=False):
    expected={f'seed_{seed}/{setting}' for seed in SEEDS for setting in list(SETTINGS)[1:]}
    require(set(index)==expected,'All 35 train/validation pipelines must be complete before test')
    path=out/'training_ready.json'
    body={'identity':identity,'all_35_selected_without_new_test_inference':True,
          'training_completion_sha256':index}
    if path.exists():
        saved=read_json(path)
        require({k:saved.get(k) for k in body}==body,'Training-ready source/index changed')
        return saved
    require(write,'Test blocked: training_ready.json is missing')
    body['ready_at_utc']=utc();write_json(path,body);return body


def test_identity(folder,stage,context,ready):
    return object_hash({'stage':stage,'training':digest(folder/'training_complete.json'),
                        'test':context['declared']['splits/test_observable.csv'],
                        'training_ready':object_hash(ready),'evaluation_precision':'float32'})


def test_one(engine,data,vocab,job,setting,out,identity,device,context):
    require(set(data)=={'val','test'},'Validation replay and test only at inference')
    seed=job['seed'];folder=out/f'seed_{seed}'/setting;stage=trained_identity(identity,seed,setting)
    trained=verify_record(folder,'training_complete.json',stage)
    require(trained is not None,'Test is blocked until training is complete')
    verify_training_record(folder,trained)
    ready=training_gate(out,identity,read_json(out/'training_index.json'))
    test_id=test_identity(folder,stage,context,ready)
    if verify_record(folder,'test_complete.json',test_id) is not None:return
    model=engine.WindowModel(engine.CategoricalEncoder([len(vocab[f]) for f in FIELDS]),
                             head_seed=job['protocol']['window_head_seed'])
    model.load_state_dict(engine.load_checkpoint(folder/'encoder/best.pt'),strict=True);model.to(device)
    batches={s:engine.ArrayBatches({'x':mask_categories(d['categories'],setting)},d['labels']) for s,d in data.items()}
    cfg=job['encoder_training'];tcfg=job['tst_training'];evaluator=context['evaluator']
    # Replay selected validation logits at the original FP32 batch boundaries.
    # This detects inference/restore mismatches; it never selects new checkpoints.
    _,window_logits=engine.evaluate(model,batches['val'],cfg['eval_batch_size'],device,cfg['label_smoothing'],False)
    replay={'window':evaluator.compare_canary(window_logits,np.load(folder/'encoder/val_logits.npy',allow_pickle=False),
                                             'Step51 selected window validation',True)}
    cache=folder/'test_cache';cache.mkdir(exist_ok=True)
    # Step43.extract_features completion contracts are train/val-only. Inference
    # uses the existing Step46 FP32 extractor, recomputing incomplete caches.
    embeddings={split:evaluator.extract_fp32(engine,model,batch,cfg['eval_batch_size'],device,cache/(split+'.npy'))
                for split,batch in batches.items()}
    tst=engine.TST();tst.load_state_dict(engine.load_checkpoint(folder/'tst/best.pt'),strict=True);tst.to(device)
    sequences={split:engine.SequenceBatches(embeddings[split],d['indices'],d['sequence_labels']) for split,d in data.items()}
    _,val_logits=engine.evaluate(tst,sequences['val'],tcfg['eval_batch_size'],device,tcfg['label_smoothing'],False)
    replay['tst']=evaluator.compare_canary(val_logits,np.load(folder/'tst/val_logits.npy',allow_pickle=False),
                                         'Step51 selected TST validation',True)
    score,logits=engine.evaluate(tst,sequences['test'],tcfg['eval_batch_size'],device,tcfg['label_smoothing'],False)
    d=data['test'];pred=logits.argmax(1).astype(np.int64)
    require(logits.shape==(len(d['sequence_labels']),6) and np.isfinite(logits).all(),'Invalid test logits')
    write_npz(folder/'test_predictions_L10.npz',y_true=d['sequence_labels'],y_pred=pred,logits=logits,
              sequence_id=d['metadata'].sequence_id.to_numpy(dtype=np.int64),
              sequence_end_row=d['metadata'].sequence_end_row.to_numpy(dtype=np.int64))
    write_json(folder/'test_metrics.json',score);write_json(folder/'validation_replay.json',replay)
    seal(folder,'test_complete.json',test_id,['test_predictions_L10.npz','test_metrics.json','validation_replay.json'],
         {'endpoint_identity':object_hash(d['metadata'].to_dict(orient='records')),
          'dataset_id':context['build']['dataset_id'],'seed':seed,'setting':setting,
          'evaluation_precision':'float32','completed_at_utc':utc()})
    del model,tst,batches,sequences,embeddings,window_logits,val_logits;gc.collect();engine.torch.cuda.empty_cache()
    shutil.rmtree(cache)


def macro(cm):
    tp=np.diagonal(cm,axis1=-2,axis2=-1);den=cm.sum(-1)+cm.sum(-2)
    return np.divide(2.*tp,den,out=np.zeros_like(tp,dtype=float),where=den>0).mean(-1)


def bootstrap(counts,strata,n_draws=BOOTSTRAP_N,batch=100,seed=BOOTSTRAP_SEED,progress=True):
    require(counts.ndim==5 and counts.shape[0]==len(strata) and counts.shape[-2:]==(6,6),'Invalid episode counts')
    require(set(np.unique(strata))==set(range(6)),'Six true-label strata required')
    require(n_draws>0 and batch>0 and np.isfinite(counts).all() and (counts>=0).all(),'Invalid bootstrap counts/size')
    nmodels,nseeds=counts.shape[1:3]
    sufficient=np.stack((np.diagonal(counts,axis1=-2,axis2=-1),counts.sum(-1),counts.sum(-2)),axis=-2)
    flat=sufficient.reshape(len(strata),-1).astype(float)
    rng=np.random.Generator(np.random.PCG64(seed));samples=np.empty((n_draws,nmodels));h=hashlib.sha256()
    for first in range(0,n_draws,batch):
        last=min(first+batch,n_draws);w=np.zeros((last-first,len(strata)),dtype=np.int64)
        for c in range(6):
            ix=np.flatnonzero(strata==c);w[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=last-first)
        h.update(w.astype('<i8').tobytes())
        pooled=(w.astype(float)@flat).reshape(last-first,nmodels,nseeds,3,6)
        den=pooled[...,1,:]+pooled[...,2,:]
        f1=np.divide(2*pooled[...,0,:],den,out=np.zeros_like(den),where=den>0)
        samples[first:last]=f1.mean(-1).mean(-1)
        if progress and (first==0 or last%4000==0 or last==n_draws):print(f'Bootstrap {last:,}/{n_draws:,}',flush=True)
    return samples,h.hexdigest()


def verify_training_family(out,identity):
    index=read_json(out/'training_index.json') if (out/'training_index.json').exists() else {}
    expected={f'seed_{seed}/{setting}' for seed in SEEDS for setting in list(SETTINGS)[1:]}
    require(set(index)==expected,'All 35 train/validation pipelines must complete before new test inference')
    check_registered(out,index)
    for seed in SEEDS:
        for setting in list(SETTINGS)[1:]:
            folder=out/f'seed_{seed}'/setting
            record=verify_record(folder,'training_complete.json',trained_identity(identity,seed,setting))
            require(record is not None,'Missing training completion')
            verify_training_record(folder,record)
    return index


def collect_statistics(out,full_test,test_data,identity,context):
    reference_manifest=context['full_manifest']
    ready=training_gate(out,identity,verify_training_family(out,identity))
    names=list(SETTINGS);meta=test_data['metadata'];y=test_data['sequence_labels']
    inventory=meta.assign(y_true=y).groupby('episode_id',sort=True).agg(
        label_id=('y_true','first'),n_labels=('y_true','nunique'),n_endpoints=('y_true','size')).reset_index()
    require(inventory.n_labels.eq(1).all(),'Mixed endpoint labels in episode')
    lookup={name:i for i,name in enumerate(inventory.episode_id)}
    codes=meta.episode_id.map(lookup).to_numpy(dtype=np.int64)
    counts=np.zeros((len(inventory),len(names),len(SEEDS),6,6),dtype=np.int64)
    for m,setting in enumerate(names):
        for s,seed in enumerate(SEEDS):
            folder=out/f'seed_{seed}'/setting
            if setting=='Full':
                relative=f'seed_{seed}/CategoricalEmbedding_MLP/test_predictions_L10.csv'
                path=full_test/relative;expected=reference_manifest['output_sha256'][relative]
                pred=load_full_prediction(path,expected,meta,y,context,seed)
                folder.mkdir(parents=True,exist_ok=True)
                write_npz(folder/'test_predictions_L10.npz',y_true=y,y_pred=pred,
                          sequence_id=meta.sequence_id.to_numpy(dtype=np.int64),
                          sequence_end_row=meta.sequence_end_row.to_numpy(dtype=np.int64))
                write_json(folder/'reused_full_reference.json',{'source':str(path),'sha256':expected,
                           'seed':seed,'training':False,'source_manifest_sha256':context['full_manifest_sha256']})
            else:
                verify_test_record(folder,identity,seed,setting,context,ready,meta)
                with np.load(folder/'test_predictions_L10.npz',allow_pickle=False) as a:
                    require(np.array_equal(a['y_true'],y),'Test endpoint truth changed')
                    for key in ('sequence_id','sequence_end_row'):
                        require(np.array_equal(a[key],meta[key]),'Test endpoint identity changed: '+key)
                    pred=a['y_pred'].copy()
                    require(np.isfinite(a['logits']).all() and a['logits'].shape==(len(y),6),'Invalid test logits')
                    require(np.array_equal(pred,a['logits'].argmax(1)),'Stored prediction/logit mismatch')
                require(np.isin(pred,np.arange(6)).all(),'Invalid test prediction')
            counts[:,m,s]=np.bincount((codes*6+y)*6+pred,minlength=len(inventory)*36).reshape(-1,6,6)
    total=counts.sum(0);per_seed=macro(total)
    draws,weight_hash=bootstrap(counts,inventory.label_id.to_numpy(dtype=np.int64))
    seed_rows=[];summary=[];contrasts=[];class_rows=[];conf_rows=[]
    for m,name in enumerate(names):
        accuracy=np.trace(total[m],axis1=-2,axis2=-1)/total[m].sum(axis=(-1,-2))
        normal_fpr=(total[m,:,0].sum(-1)-total[m,:,0,0])/total[m,:,0].sum(-1)
        lo,hi=np.quantile(draws[:,m],[.025,.975],method='linear')
        summary.append({'setting':name,'n_fields':len(SETTINGS[name]),'n_encoder_seeds':5,
                        'macro_f1_mean':per_seed[m].mean(),'macro_f1_seed_sd_ddof1':per_seed[m].std(ddof=1),
                        'macro_f1_ci95_low':lo,'macro_f1_ci95_high':hi,
                        'accuracy_mean':accuracy.mean(),'accuracy_seed_sd_ddof1':accuracy.std(ddof=1),
                        'normal_fpr_mean':normal_fpr.mean()})
        for s,seed in enumerate(SEEDS):
            seed_rows.append({'setting':name,'seed':seed,'macro_f1':per_seed[m,s],
                              'accuracy':accuracy[s],'normal_fpr':normal_fpr[s]})
        den=total[m].sum(-1)+total[m].sum(-2)
        f1=np.divide(2*np.diagonal(total[m],axis1=-2,axis2=-1),den,out=np.zeros_like(den,dtype=float),where=den>0)
        row_norm=np.divide(total[m],total[m].sum(-1,keepdims=True),out=np.zeros_like(total[m],dtype=float),
                           where=total[m].sum(-1,keepdims=True)>0).mean(0)
        for c,label in enumerate(LABELS):
            class_rows.append({'setting':name,'label':label,'f1_mean':f1[:,c].mean(),'f1_seed_sd_ddof1':f1[:,c].std(ddof=1)})
            for pred,label_pred in enumerate(LABELS):
                conf_rows.append({'setting':name,'true_label':label,'predicted_label':label_pred,
                                  'seed_mean_row_normalized':row_norm[c,pred]})
        if m:
            difference=draws[:,0]-draws[:,m]
            lo95,hi95=np.quantile(difference,[.025,.975],method='linear')
            lo_adj,hi_adj=np.quantile(difference,[.05/(2*7),1-.05/(2*7)],method='linear')
            contrasts.append({'contrast':'Full_minus_'+name,'delta_macro_f1':per_seed[0].mean()-per_seed[m].mean(),
                              'ci95_low':lo95,'ci95_high':hi95,'bonferroni_low':lo_adj,'bonferroni_high':hi_adj,
                              'family_size':7,'marginal_confidence':1-.05/7,
                              'direction':'Full_higher' if lo_adj>0 else ('Subset_higher' if hi_adj<0 else 'zero_included_not_equivalence')})
    write_csv(out/'metrics_by_seed.csv',seed_rows);write_csv(out/'field_group_summary.csv',summary)
    write_csv(out/'paired_comparisons.csv',contrasts);write_csv(out/'per_class_f1.csv',class_rows)
    write_csv(out/'row_normalized_confusion.csv',conf_rows)
    write_csv(out/'evaluation_endpoints_L10.csv',meta.assign(y_true=y))
    save_effect_diagnostics(out,context,meta)
    write_csv(out/'episode_inventory.csv',inventory.drop(columns='n_labels'))
    write_npz(out/'episode_confusions.npz',counts=counts,episode_ids=inventory.episode_id.to_numpy(dtype=str),
              strata=inventory.label_id.to_numpy(dtype=np.int64),settings=np.array(names),seeds=np.array(SEEDS),labels=np.array(LABELS))
    write_npz(out/'bootstrap_distribution.npz',macro_f1_seed_mean=draws,settings=np.array(names))
    write_json(out/'statistics_audit.json',{'draws':BOOTSTRAP_N,'seed':BOOTSTRAP_SEED,'batch_size':100,
               'weights_sha256_little_endian_int64':weight_hash,'pairing':'same whole-episode draws for all settings and seeds',
               'estimand':'endpoint macro-F1 for each seed, then average seeds; not ensemble or pooled-seed confusion',
               'CI_scope':'conditional on fitted checkpoints and observed encoder seeds; no threshold/model/generator reselection',
               'family':'seven Full-minus-subset contrasts only; nominal 95% family coverage under percentile bootstrap approximation'})


def finish(out,manifest):
    # Checkpoints remain on the user's computer for resume; ship review evidence only.
    files=[p for p in sorted(out.rglob('*')) if p.is_file() and p.suffix in ('.json','.csv','.npz')
           and p.name!='run_manifest.json' and not any(part in ('embeddings','test_cache') for part in p.relative_to(out).parts)]
    manifest={**manifest,'finished_at_utc':utc(),
              'output_sha256':{p.relative_to(out).as_posix():digest(p) for p in files},
              'review_excludes':'model .pt files and reproducible embedding .npy caches; original sources read-only'}
    write_json(out/'run_manifest.json',manifest)
    return ensure_review_archive(out,manifest)


def ensure_review_archive(out,manifest):
    """Repair packaging after a hard interruption without rewriting any results."""
    archive=out.with_name(out.name+'_review.zip');temp=archive.with_suffix('.zip.tmp')
    expected={**manifest['output_sha256'],'run_manifest.json':digest(out/'run_manifest.json')}
    if archive.is_file():
        try:
            with zipfile.ZipFile(archive) as z:
                names=z.namelist()
                valid=len(names)==len(expected) and set(names)==set(expected)
                if valid:
                    for name,wanted in expected.items():
                        h=hashlib.sha256()
                        with z.open(name) as stream:
                            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
                        if h.hexdigest()!=wanted:valid=False;break
                if valid:return archive
        except (OSError,ValueError,RuntimeError,EOFError,zipfile.BadZipFile,zlib.error):pass
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in expected:z.write(safe(out,name),name)
    replace(temp,archive)
    print('\nUPLOAD THIS FILE: '+str(archive),flush=True)
    return archive


def verify_test_record(folder,identity,seed,setting,context,ready,meta):
    stage=trained_identity(identity,seed,setting)
    result=verify_record(folder,'test_complete.json',test_identity(folder,stage,context,ready))
    require(result is not None and set(result['artifact_sha256'])==
            {'test_predictions_L10.npz','test_metrics.json','validation_replay.json'},'Missing/incomplete test record')
    require(result.get('endpoint_identity')==object_hash(meta.to_dict(orient='records'))
            and result.get('dataset_id')==context['build']['dataset_id']
            and result.get('seed')==seed and result.get('setting')==setting
            and result.get('evaluation_precision')=='float32','Test provenance/endpoint identity changed')
    require(datetime.fromisoformat(result['completed_at_utc'])>=datetime.fromisoformat(ready['ready_at_utc']),
            'Test completion precedes whole-family training completion')
    return result


def verify_sources(context):
    for item in context['log']:checked(Path(item['path']),item['sha256'])


def save_effect_diagnostics(out,context,meta):
    # Keep the source annotation with an explicit scope. It describes changes in
    # ALL 22 fields and is NOT recomputed or relabelled as a subset-specific flag.
    path=context['full_test']/context['evaluator'].OBSERVABILITY_REFERENCE
    expected=context['full_manifest']['output_sha256'][path.name]
    checked(path,expected)
    reference=context['evaluator'].read_prediction_csv(path)
    context['evaluator'].validate_observability_reference(reference,meta)
    write_csv(out/'observability_endpoints_L10.csv',reference)
    write_json(out/'observability_scope.json',{
        'source':'completed Step46 full-22-field counterfactual projection-change diagnostics',
        'subset_specific_change_flags':False,'model_inputs':False,'primary_population_filter':False,
        'scope':'Full-schema annotation only; dropped-field counterfactuals are not available here. '
                'Do not interpret these flags as evidence availability for each retained subset or an oracle ceiling.'})


def execute(engine,out,identity,data,vocab,jobs,root,formal,full_test,device,context):
    index=read_json(out/'training_index.json') if (out/'training_index.json').exists() else {}
    check_registered(out,index);summaries=[];number=0
    for seed in SEEDS:
        for setting in list(SETTINGS)[1:]:
            number+=1;key=f'seed_{seed}/{setting}'
            print(f'[{number}/35] Train/validation: {seed} / {setting}',flush=True)
            result=train_one(engine,data,vocab,jobs[seed],setting,out,identity,device)
            index[key]=digest(out/key/'training_complete.json');write_json(out/'training_index.json',index)
            summaries.append(result['summary']);write_csv(out/'training_summary.csv',summaries)
    index=verify_training_family(out,identity)
    verify_sources(context)
    training_gate(out,identity,index,write=True)  # Preserve the first ready time when resuming.
    # First read of NEW test observations. No new test rows guide any of the 35 fits.
    test=load_split(root,'test',context);episodes=set(test['frame'].episode_id)
    require(all(not episodes&set(d['frame'].episode_id) for d in data.values()),'Test/train/validation episode leakage')
    require(object_hash(test['metadata'].to_dict(orient='records'))==context['full_protocol']['test_endpoint_identity'],
            'Test endpoints differ from Full Step46')
    test['categories']=encode(test['frame'],vocab)
    checked(root/DATA_REL/'splits/test_observable.csv',context['declared']['splits/test_observable.csv'],context['log'])
    write_json(out/'test_input_audit.json',{'rows':len(test['frame']),'episodes':len(episodes),
             'endpoints':len(test['indices']),'sha256':context['declared']['splits/test_observable.csv'],
             'oov_counts':{f:int((test['categories'][:,i]==0).sum()) for i,f in enumerate(FIELDS)},
             'all_fits_completed_before_new_test_inference':True})
    number=0
    for seed in SEEDS:
        for setting in list(SETTINGS)[1:]:
            number+=1;print(f'[{number}/35] Test: {seed} / {setting}',flush=True)
            test_one(engine,{'val':data['val'],'test':test},vocab,jobs[seed],setting,out,identity,device,context)
    verify_sources(context)
    collect_statistics(out,full_test,test,identity,context)


def protocol_body(context,runtime):
    source=context['source']
    return {'version':PROTOCOL_VERSION,'script_sha256':digest(__file__),
          'engine_sha256':source['engine_sha256'],
          'source45_manifest_sha256':context['record']['manifest_sha256'],
          'source46_manifest_sha256':context['full_manifest_sha256'],
          'source45_protocol_identity':source['identity'],'source46_protocol_identity':context['full_protocol']['identity'],
          'source_contract':source['source_contract'],'counts':context['build']['counts'],
          'settings':{k:list(v) for k,v in SETTINGS.items()},'encoder_seeds':list(SEEDS),'labels':list(LABELS),
          'jobs':{str(k):v for k,v in context['jobs'].items()},'runtime':runtime,
          'split_sha256':{s:context['declared'][f'splits/{s}_observable.csv'] for s in ('train','val','test')},
          'vocabulary':context['vocab'],'evaluation_precision':'float32','extraction_precision':'float32',
          'validation_replay':'selected window/TST validation logits; original FP32 batch boundaries; Step46 tolerance and exact predictions',
          'bootstrap':{'draws':BOOTSTRAP_N,'seed':BOOTSTRAP_SEED,'batch':100,'family_size':7},
          'design':'retrospective revision analysis; existing Full results already observed',
          'comparison':'same CategoricalEmbedding-MLP + TST; only available field values change',
          'mask':'constant padding ID0 at every train/validation/test row; original 22-slot architecture retained',
          'selection':'existing Step45 validation checkpoint rules; no retuning by subset or test',
          'scope':'shared_nominal_v2 A structured-encoder train-time field-group comparison; not a DistilBERT ablation',
          'limitations':['finite inherited budgets; retain and report BUDGET_REVIEW_REQUIRED; do not extend budgets from test scores',
                         'group-only tests do not prove each group is necessary or isolate causal sensor contributions',
                         'canonical categorical inputs, not raw-versus-semantic validation',
                         'schema groups are not exclusive physical sensor sources',
                         'five encoder seeds and fixed TST/head/shuffle seeds, not five fully independent pipeline seeds',
                         'CIs condition on observed fitted seeds; seed SD reported separately, no equivalence claim',
                         'retain quiet/unchanged attack endpoints; audit flags never select features, rows, labels or checkpoints',
                         'Full22 observable-effect annotations do not identify changes in each field subset',
                         'zero empirical NORMAL false alarms do not establish a population risk bound']}


def verify_completed(out,previous,identity,context):
    require(previous.get('protocol_identity')==identity and previous.get('status')=='COMPLETE'
            and previous.get('protocol_version')==PROTOCOL_VERSION,'Completed run identity/version differs')
    required={'protocol.json','training_index.json','training_ready.json','training_summary.csv','input_audit.json',
              'metrics_by_seed.csv','field_group_summary.csv','paired_comparisons.csv',
              'evaluation_endpoints_L10.csv','episode_inventory.csv','episode_confusions.npz',
              'bootstrap_distribution.npz','statistics_audit.json','observability_endpoints_L10.csv','observability_scope.json'}
    require(required<=set(previous.get('output_sha256',{})),'Completed result inventory is incomplete')
    for name,wanted in previous['output_sha256'].items():checked(safe(out,name),wanted)
    ready=training_gate(out,identity,verify_training_family(out,identity))
    meta=context['evaluator'].read_prediction_csv(out/'evaluation_endpoints_L10.csv')
    require(object_hash(meta.to_dict(orient='records'))==context['full_protocol']['test_endpoint_identity'],
            'Completed endpoint set differs from Full')
    for seed in SEEDS:
        for setting in list(SETTINGS)[1:]:
            verify_test_record(out/f'seed_{seed}'/setting,identity,seed,setting,context,ready,meta)


def remaining_disk_bytes(out,context,identity):
    pending=[]
    for seed in SEEDS:
        for setting in list(SETTINGS)[1:]:
            folder=out/f'seed_{seed}'/setting
            if not (folder/'training_complete.json').exists():pending.append((seed,setting))
    # Current Full checkpoint parameter counts estimate retained FP32 model/Adam
    # states. Feature caches are sequential and removed after each completed job.
    params=[]
    for seed in SEEDS:
        folder=context['formal']/f'seed_{seed}'/'CategoricalEmbedding_MLP'
        params.append(sum(read_json(folder/stage/'complete.json')['parameters'] for stage in ('encoder','tst')))
    cache_rows=sum(len(d['labels']) for d in context['data'].values())+context['build']['counts']['test']['rows']
    return int(len(pending)*max(params)*24 + cache_rows*768*4 + .5*1024**3)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--formal-run',type=Path,default=None)
    parser.add_argument('--full-test-run',type=Path,default=None)
    parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--audit-only',action='store_true',help='Read-only source/train-val checks; no model import, training or output folder')
    args=parser.parse_args(argv);root=args.project_root.expanduser().resolve()
    def rooted(value,default):
        path=Path(value or default).expanduser()
        return (path if path.is_absolute() else root/path).resolve()
    formal=rooted(args.formal_run,FORMAL_REL);full_test=rooted(args.full_test_run,FULL_TEST_REL)
    out=rooted(args.output,OUTPUT_REL)
    protected=[formal,full_test,root/'dataset']
    for source in protected:
        require(out!=source and source not in out.parents and out not in source.parents,'Output overlaps original inputs')
    require(out!=root and out not in root.parents,'Output cannot be the project directory or its parent')
    print('Step51: 7 field subsets x 5 encoder seeds; current Full predictions reused.',flush=True)
    print('Checking current shared-nominal sources before any new training ...',flush=True)
    context=preflight(root,formal,full_test)
    # Reject nesting inside any earlier pipeline output, even when --output is explicit.
    for item in context['log']:
        path=Path(item['path']).resolve()
        require(out!=path and out not in path.parents,'Output overlaps a source input')
    if args.audit_only:
        print('AUDIT_ONLY_COMPLETE: current sources verified; no models loaded or files written.',flush=True)
        return 0
    source=context['source'];jobs=context['jobs'];data=context['data'];vocab=context['vocab']
    body=protocol_body(context,source['runtime']);identity=object_hash(body)
    if (out/'protocol.json').exists():
        require(read_json(out/'protocol.json')==dict(identity=identity,**body),
                'Resume protocol/source changed; preserve existing results and use a new output directory')
    elif out.exists():
        require(not any(out.iterdir()),'Existing unrecognized output directory; no overwrite')
    previous=read_json(out/'run_manifest.json') if (out/'run_manifest.json').exists() else {}
    if previous.get('status')=='COMPLETE':
        verify_completed(out,previous,identity,context);verify_sources(context)
        ensure_review_archive(out,previous)
        print('COMPLETE: all saved results verified; no retraining or result rewrite. '+str(out),flush=True)
        return 0
    engine=load_engine(context['engine_path'],source['engine_sha256'])
    for job in jobs.values():
        require(job['encoder_training']==engine.training_spec(job['condition'],job['protocol'])
                and job['tst_training']==engine.training_spec(job['condition'],job['protocol'],tst=True)
                and job['protocol']['tst_model']==engine.TST_MODEL,'Full source training recipe/model differs')
        require(job['protocol']['evaluation_precision']==job['protocol']['extraction_precision']=='float32',
                'Use the revised FP32 Full source')
    device,runtime=prepare_runtime(engine,source)
    require(runtime==body['runtime'],'Runtime changed during preparation')
    out.mkdir(parents=True,exist_ok=True)
    with engine.RunLock(out/'.run.lock'):
        # Recheck after acquiring the lock; another process may have just finished.
        identity=bind_protocol(out,body)
        previous=read_json(out/'run_manifest.json') if (out/'run_manifest.json').exists() else {}
        if previous.get('status')=='COMPLETE':
            verify_completed(out,previous,identity,context)
            ensure_review_archive(out,previous)
            print('COMPLETE: existing run verified.',flush=True);return 0
        manifest={'script_number':51,'protocol_version':PROTOCOL_VERSION,'status':'RUNNING',
                  'started_at_utc':previous.get('started_at_utc',utc()),'protocol_identity':identity,
                  'script_sha256':body['script_sha256'],'dataset_id':context['build']['dataset_id'],
                  'source45_protocol_identity':source['identity'],
                  'source46_protocol_identity':context['full_protocol']['identity'],
                  'runtime':runtime,'training_pipelines_planned':35,'full_reference_reused_seeds':list(SEEDS),
                  'new_training_family':'CategoricalEmbedding_MLP + TST'}
        write_json(out/'run_manifest.json',manifest)
        try:
            needed=remaining_disk_bytes(out,context,identity);free=shutil.disk_usage(out).free
            require(free>=needed,f'Insufficient disk: {free/1024**3:.2f} GiB free; estimated remaining {needed/1024**3:.2f} GiB')
            write_json(out/'input_audit.json',context['log'])
            execute(engine,out,identity,data,vocab,jobs,root,formal,full_test,device,context)
            verify_sources(context)
            require(digest(__file__)==body['script_sha256'],'Step51 code changed during execution')
            manifest['status']='COMPLETE';finish(out,manifest)
        except BaseException as exc:
            manifest.update(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',
                            error=str(exc),traceback=traceback.format_exc())
            finish(out,manifest);raise
    return 0


if __name__=='__main__':sys.exit(main())
