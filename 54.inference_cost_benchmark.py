#!/usr/bin/env python3
"""Measure existing TAES checkpoints; no training and no edits to earlier outputs.

Place beside the numbered scripts, then run after other GPU work finishes:
    python "54.inference_cost_benchmark.py"
Inventory files while Step49 is running (does not load models/use CUDA):
    python "54.inference_cost_benchmark.py" --preflight-only

Default: fixed seed 20260610, four Step46 encoder+TST pipelines and the original
Step8 encoder with four Step12/13 heads authenticated by completed Step48.
--all-seeds measures all recorded seeds; no best/fastest-seed selection.
Each timing call produces ONE L10 decision (not a batch of ten decisions).
FP32, eval/inference_mode. Same 64 validation endpoints; 3 rounds x 3 repetitions.
Each scope/round starts with at least 50 warmup calls and 0.5 seconds of warmup;
other samples receive 2 warmups. No GC/cache flush or memory probes in timing.
Memory is measured in a separate pass. All observations and round diagnostics
are retained; no trimming or fastest-round selection. Labels/flags are not features.
Full10 encodes ten windows together; stream1 assumes nine prior CLS embeddings
are resident on the GPU and measures one update. Initial cache filling is the
full10 case. Head-only excludes encoding. Acquisition of ten seconds of telemetry
and raw-to-observable extraction are outside this 22-field-input benchmark.
Formal DistilBERT keeps fixed padding; shared Step8 keeps dynamic padding.
Pooling heads are standardized to FP32 here (their historic evaluation used AMP).
Historic F1 is therefore not advertised as a new FP32 accuracy measurement.
CUDA memory is PyTorch allocator memory, not whole-device or process memory.
Windows desktop/unknown GPU contexts are logged without blocking. Python/known
compute executable names trigger a conservative pause, not a claim of active
work. No processes are killed; unavailable names remain explicitly unresolved.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PureWindowsPath
import platform
import subprocess
import sys
import time
import traceback
import zipfile

VERSION = 'shared_nominal_inference_cost_v3_windows_activity'
DATASET_ID = '478f782ceb970d7a2de73319ccc4e929be06cfea684a054aa72d932ed65ae83b'
PREFIX = Path('outputs/shared_nominal_v2')
DATA = Path('dataset/shared_nominal_v2')
ENCODER = PREFIX/'distilbert_window_baseline/trained_distilbert_model'
EMBEDDINGS = PREFIX/'tst_embeddings_finetuned_main'
DEFAULT_SEED = 20260610
SAMPLE_SEED = 2026101501
FIELDS = ('observation_stage','path_type','direction','source_role','destination_role',
          'protocol_context','operational_phase','message_type','service_context',
          'command_intent','payload_or_parameter_context','sequence_freshness_status',
          'control_security_state','signal_quality_state','tracking_stability',
          'navigation_consistency','timing_consistency','decoder_status',
          'command_execution_result','retry_timeout_state','session_state','state_transition_consistency')
LABELS = ('NORMAL','RF_LINK_JAMMING','GNSS_SPOOFING','UNAUTHORIZED_TELECOMMAND_INJECTION',
          'TELECOMMAND_REPLAY','GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE')
CONDITIONS = ('CategoricalEmbedding_MLP','FieldToken_Encoder',
              'DistilBERT_RandomInit_Train','DistilBERT_Pretrained_Finetune')
FAMILIES = ('TST','Mean_MLP','Max_MLP','MeanMax_MLP')
SCOPES = ('full10_model_only','stream1_model_only','full10_end_to_end',
          'stream1_end_to_end','head_only')
RUNS = {46: ('46_encoder_comparison_frozen_test','FROZEN_TEST_AND_BOOTSTRAP_COMPLETE'),
        48: ('48_fresh_realizations','FRESH_REALIZATIONS_COMPLETE')}
VERIFIED = {}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    temporary = Path(str(path)+'.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def checked(path, expected):
    path = Path(path).resolve()
    require(path.is_file(), 'Missing existing project file: '+str(path))
    require(isinstance(expected, str) and len(expected)==64, 'Missing recorded SHA256: '+str(path))
    actual = digest(path)
    require(actual==expected, 'File hash differs from the recorded experiment: '+str(path))
    VERIFIED[str(path)] = actual
    return path


def safe_relative(root, value):
    rel = Path(value)
    require(not rel.is_absolute() and not PureWindowsPath(value).drive and
            '..' not in rel.parts and '\\' not in value, 'Unsafe recorded relative path: '+value)
    result = (Path(root)/rel).resolve()
    require(Path(root).resolve() in result.parents, 'Relative path escapes source folder')
    return result


def resolve_recorded(root, value):
    """Rebase project outputs/dataset paths first; never silently use an old copy."""
    root = Path(root).resolve()
    parts = list(PureWindowsPath(str(value)).parts) if '\\' in str(value) else list(Path(value).parts)
    for anchor in ('outputs','dataset'):
        if anchor in parts:
            tail = parts[parts.index(anchor):]
            require('..' not in tail, 'Unsafe relocated path')
            return root.joinpath(*tail).resolve()
    path = Path(value).expanduser()
    if not path.is_absolute() and not PureWindowsPath(str(value)).drive:
        return safe_relative(root, str(value))
    return path.resolve()


def find_run(root, number, explicit=None):
    folder_name, status = RUNS[number]
    if explicit:
        candidates = [Path(explicit).resolve()]
    else:
        base = Path(root)/PREFIX/folder_name
        candidates = sorted(p.parent for p in base.glob('*/run_manifest.json'))
    matches = []
    for folder in candidates:
        try:
            m = read_json(folder/'run_manifest.json')
            if m.get('status') != status:
                continue
            dataset = m.get('dataset_id')
            if dataset is None:
                dataset = read_json(folder/'protocol.json').get('source_contract',{}).get('dataset_id')
            if dataset == DATASET_ID:
                matches.append(folder)
        except (OSError, ValueError, KeyError):
            continue
    require(matches, f'No completed current Step{number} folder found. Use --run{number} PATH.')
    require(len(matches)==1, f'Multiple completed Step{number} runs found: '+
            ', '.join(map(str,matches))+f'. Specify --run{number}; no score/date selection is made.')
    return matches[0]


def load_run(folder, number):
    m, p = read_json(folder/'run_manifest.json'), read_json(folder/'protocol.json')
    require(m.get('status')==RUNS[number][1], f'Step{number} did not complete')
    require(p['identity']==m['protocol_identity']==object_hash({k:v for k,v in p.items() if k!='identity'}),
            f'Step{number} protocol identity mismatch')
    checked(folder/'protocol.json', m['output_sha256']['protocol.json'])
    VERIFIED[str((folder/'run_manifest.json').resolve())] = digest(folder/'run_manifest.json')
    return m,p


def resolve_code(root, number, expected, search_folders=()):
    candidates = list(Path(root).glob(str(number)+'*.py'))
    for folder in search_folders:
        candidates.extend(Path(folder).glob(str(number)+'*.py'))
    matches = [p.resolve() for p in candidates if digest(p)==expected]
    require(matches, f'Cannot find original Step{number} Python code with SHA256 {expected} beside this script.')
    return checked(sorted(set(matches))[0], expected)


def import_code(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sequence_indices(frame):
    import numpy as np
    require(not frame[['episode_id','episode_step']].duplicated().any(), 'Duplicate episode steps')
    groups=[]
    for _, positions in frame.groupby('episode_id',sort=False).indices.items():
        positions=np.asarray(positions)
        positions=positions[np.argsort(frame.iloc[positions].episode_step.to_numpy(),kind='stable')]
        part=frame.iloc[positions]
        require(part.label.nunique()==1 and np.all(np.diff(part.episode_step.to_numpy(dtype='int64'))==1)
                and np.all(np.diff(part.window_id.to_numpy(dtype='int64'))==1), 'Nonconsecutive or mixed-label episode')
        if len(positions)>=10:
            groups.append(np.lib.stride_tricks.sliding_window_view(positions,10).copy())
    require(groups, 'No consecutive L10 contexts in validation data')
    return np.concatenate(groups)


def select_samples(indices, count, seed=SAMPLE_SEED):
    import numpy as np
    require(1<=count<=len(indices), 'Sample count exceeds available L10 endpoints')
    return np.random.default_rng(seed).choice(len(indices),size=count,replace=False)


def base_config(root, p46, explicit=None):
    candidates=[]
    if explicit:
        candidates.append(Path(explicit).expanduser()/'config.json')
    candidates.append(Path(p46['base_config_dir']).expanduser()/'config.json')
    cache=Path(os.environ.get('HF_HUB_CACHE',str(Path.home()/'.cache/huggingface/hub')))
    candidates.extend(cache.glob('models--*distilbert*/snapshots/*/config.json'))
    matches=[p for p in candidates if p.is_file() and digest(p)==p46['base_hashes']['config.json']]
    require(matches, 'Pinned DistilBERT config missing; use --base-model-dir PATH. No download/training is performed.')
    return checked(matches[0],p46['base_hashes']['config.json'])


def prepare_sources(root, args):
    """Only small records/CSV/hash I/O. Safe to run while another job uses CUDA."""
    import numpy as np
    import pandas as pd
    r46=find_run(root,46,args.run46);r48=find_run(root,48,args.run48)
    m46,p46=load_run(r46,46);m48,p48=load_run(r48,48)
    require(p46['dataset_id']==p48['source_contract']['dataset_id']==DATASET_ID, 'Source datasets differ')
    p45dir=resolve_recorded(root,p46['source_run'])
    checked(p45dir/'run_manifest.json',p46['source_manifest_sha256'])
    m45=read_json(p45dir/'run_manifest.json');p45=read_json(p45dir/'protocol.json')
    checked(p45dir/'protocol.json',m45['output_sha256']['protocol.json'])
    require(m45['status']=='FORMAL_TRAIN_VAL_COMPLETE' and m45['completed_pipelines']==20
            and p45['identity']==p46['source_protocol_identity']==object_hash({k:v for k,v in p45.items() if k!='identity'}),
            'Step45 source identity/completion differs from Step46')
    require(p45['jobs']==p46['jobs'] and p45['dataset_id']==DATASET_ID, 'Formal job/source mismatch')
    vocab_path=checked(r46/'train_field_vocabulary.json',m46['output_sha256']['train_field_vocabulary.json'])
    require(digest(vocab_path)==p46['vocabulary_sha256'], 'Vocabulary provenance differs')
    vocab=read_json(vocab_path)
    require(vocab['source']=='train_only' and vocab['unknown_id']==0 and set(vocab['fields'])==set(FIELDS), 'Wrong vocabulary schema')
    engine_path=resolve_code(root,43,p46['source_code_sha256']['43'])
    # Authenticated Step48 records contain the original Step8/12/13 lineage.
    audit=p48['input_audit']
    codes={}
    for role,number in [('TST',11),('pool',13)]:
        hashes={sha for path,sha in audit.items() if PureWindowsPath(path).name.startswith(str(number)+'.') and path.endswith('.py')}
        require(len(hashes)==1, f'Ambiguous original Step{number} source hashes')
        codes[role]=resolve_code(root,number,hashes.pop(),[r48/'source'])
    sc=p48['source_contract'];formal_sc=p46['jobs'][0]['protocol']['source_contract']
    for key in ['dataset_id','step8_run_id','step9_run_id','step8_model_sha256']:
        require(sc[key]==formal_sc[key], 'Formal/shared source differs: '+key)
    encoder_dir=root/ENCODER
    for name,sha in sc['step8_model_sha256'].items():
        checked(safe_relative(encoder_dir.parent,name),sha)
    mapping=read_json(encoder_dir/'label_mapping.json')
    require(mapping['observable_fields']==list(FIELDS) and mapping['label_order']==list(LABELS)
            and mapping['serializer_version']=='observable22_fixed_order_is_v1', 'Step8 input/label schema differs')
    val_path=checked(root/DATA/'splits/val_observable.csv',sc['input_sha256']['splits/val_observable.csv'])
    require(sc['input_sha256']['splits/val_observable.csv']==formal_sc['input_sha256']['splits/val_observable.csv'], 'Validation sources differ')
    frame=pd.read_csv(val_path,encoding='utf-8-sig',keep_default_na=False)
    require(set(FIELDS)<=set(frame) and set(frame.label)==set(LABELS), 'Validation schema/classes differ')
    require(not frame[list(FIELDS)].astype(str).apply(lambda c:c.str.strip().eq('')).any().any(), 'Empty observable fields')
    indices=sequence_indices(frame);selected=select_samples(indices,args.samples)
    shared_ref=checked(root/EMBEDDINGS/'val_window_embeddings.npz',sc['step9_artifacts']['val_window_embeddings.npz'])
    seq_path=root/PREFIX/'tst_sequences_finetuned_main/val_tst_sequences_L10.npz'
    checked(seq_path,sc['step10_artifacts']['val_tst_sequences_L10.npz'])
    with np.load(seq_path,allow_pickle=False) as arr:
        require(np.array_equal(indices,arr['sequence_indices']), 'Step10 validation sequence order differs')
    max_length=p46['max_length']
    require(mapping['max_length']==max_length, 'Token budgets differ')
    cases=[]
    for job in p46['jobs']:
        if not args.all_seeds and job['seed']!=DEFAULT_SEED:
            continue
        folder=safe_relative(p45dir,job['relative_dir'])
        checked(folder/'pipeline_complete.json',m45['output_sha256'][job['relative_dir']+'/pipeline_complete.json'])
        marker=read_json(folder/'pipeline_complete.json')
        require(marker['identity']==object_hash({k:v for k,v in marker.items() if k!='identity'}), 'Formal pipeline marker changed')
        require(marker['summary']['seed']==job['seed'] and marker['summary']['condition']==job['condition'], 'Wrong formal pipeline')
        for stage in ['encoder','tst']:
            rel=job['relative_dir']+'/'+stage+'/best.pt'
            require(p46['checkpoint_sha256'][rel]==marker['artifact_sha256'][stage+'/best.pt'], 'Checkpoint inventories disagree')
            checked(folder/stage/'best.pt',p46['checkpoint_sha256'][rel])
        for rel in ['embeddings/val.npy','tst/val_logits.npy']:
            checked(folder/rel,marker['artifact_sha256'][rel])
        cases.append(dict(key=f"formal46/{job['condition']}/seed_{job['seed']}",track='formal46',
            condition=job['condition'],family='TST',seed=job['seed'],job=job,folder=str(folder),
            encoder_checkpoint=str(folder/'encoder/best.pt'),head_checkpoint=str(folder/'tst/best.pt'),
            expected_parameters=marker['summary']['deployment_parameters'],padding='fixed_max_length'))
    for key,head in p48['checkpoint_contracts'].items():
        family=key.split('/')[0]
        if family not in FAMILIES:
            continue
        seed=int(key.rsplit('_',1)[1])
        if not args.all_seeds and seed!=DEFAULT_SEED:
            continue
        checkpoint=checked(resolve_recorded(root,head['checkpoint']),audit[head['checkpoint']])
        predictions=checked(resolve_recorded(root,head['predictions']['val']),audit[head['predictions']['val']])
        cases.append(dict(key='shared8/'+key,track='shared8',condition='Step8_Finetuned_DistilBERT',
            family=family,seed=seed,head_checkpoint=str(checkpoint),head_record=head,
            validation_predictions=str(predictions),padding='dynamic_batch'))
    require(len(cases)==(60 if args.all_seeds else 8), 'Unexpected number of existing model configurations')
    return dict(root=root,r46=r46,r48=r48,p46=p46,p48=p48,vocab=vocab['fields'],engine_path=engine_path,
        codes=codes,encoder_dir=encoder_dir,base_config=base_config(root,p46,args.base_model_dir),
        frame=frame,indices=indices,selected=selected,max_length=max_length,cases=cases,
        shared_ref=shared_ref,shared_contract=sc,tokenizer_sha256=formal_sc['tokenizer_sha256'])


def serialize_values(rows):
    return ['; '.join(f'{field} is {str(value).strip()}' for field,value in zip(FIELDS,row))+'.' for row in rows]


def summarize_latencies(values):
    import numpy as np
    a=np.asarray(values,dtype=float)
    require(len(a)>0 and np.isfinite(a).all() and (a>=0).all(), 'Invalid timing observations')
    return dict(measurements=len(a),mean_ms=float(a.mean()),median_ms=float(np.median(a)),
                p95_ms=float(np.quantile(a,.95)),min_ms=float(a.min()),max_ms=float(a.max()),
                sd_ms=float(a.std(ddof=1)) if len(a)>1 else 0.)


def configure_numeric(torch, threads):
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')


def cuda_snapshot(torch, device):
    if device.type!='cuda':
        return dict(allocated_bytes=None,reserved_bytes=None)
    return dict(allocated_bytes=int(torch.cuda.memory_allocated(device)),
                reserved_bytes=int(torch.cuda.memory_reserved(device)))


def synchronize(torch, device):
    if device.type=='cuda':
        torch.cuda.synchronize(device)


def time_call(call, sync, clock=time.perf_counter_ns):
    """Completion fence is within the interval; setup/warmup is outside."""
    sync()
    start=clock()
    result=call()
    sync()
    elapsed=(clock()-start)/1_000_000
    return elapsed,result


def windows_process_names(pids):
    """Best-effort read-only name lookup; no elevation, termination or command-line capture."""
    if platform.system()!='Windows' or not pids: return {}
    ids=','.join(str(int(pid)) for pid in sorted(set(pids)))
    command=("[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new(); "
             "Get-Process -Id "+ids+" -ErrorAction SilentlyContinue | "
             "Select-Object Id,ProcessName | ConvertTo-Json -Compress")
    try:
        result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',command],
            capture_output=True,encoding='utf-8',errors='replace',timeout=5,check=False)
        values=json.loads(result.stdout.strip()) if result.stdout.strip() else []
        if isinstance(values,dict): values=[values]
        return {int(v['Id']):str(v['ProcessName']) for v in values
                if isinstance(v,dict) and v.get('Id') is not None and v.get('ProcessName')}
    except (OSError,subprocess.SubprocessError,ValueError,TypeError):
        return {}


def gpu_process_kind(name):
    """Windows process-name heuristic only; presence never proves active GPU utilization."""
    import re
    stem=PureWindowsPath(str(name)).name.lower().removesuffix('.exe')
    if re.fullmatch(r'python(?:w|\d+(?:\.\d+)*)?',stem) or stem in {
        'ipython','jupyter','jupyter-lab','torchrun','ollama','llama-server','llama-cli',
        'vllm','tritonserver','matlab','octave','nvidia-cuda-mps-server'}:
        return 'suspected_compute'
    if stem in {'explorer','dwm','searchapp','searchhost','textinputhost','shellexperiencehost',
        'microsoft.notes','phoneexperiencehost','nvidia share','notion','systemsettings',
        'cef_server','powerpnt','msedgewebview2','msedge','chrome','firefox',
        'startmenuexperiencehost','applicationframehost','runtimebroker','sihost',
        'csrss','winlogon','services','svchost','fontdrvhost','audiodg','taskhostw','system'}:
        return 'desktop_or_system_name'
    return 'unresolved'


def gpu_busy_message(activity):
    rows=activity.get('suspected_compute_processes',activity.get('other_compute_processes',[]))
    names='; '.join(f"{r.get('resolved_name') or r.get('name','unknown')} (PID {r['pid']})" for r in rows)
    return ('Possible GPU compute workload detected: '+(names or 'see GPU activity JSON')+
            '. Check that process before benchmarking. Process presence alone does not prove active computation.')


def gpu_activity(torch, device):
    if device.type!='cuda':
        return dict(status='CPU_REQUESTED',other_compute_processes=[],background_review_required=False)
    try:
        result=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10,check=True)
        uuid=str(getattr(torch.cuda.get_device_properties(device),'uuid','')).strip()
        normalized_uuid=uuid.lower().removeprefix('gpu-');rows=[]
        for row in csv.reader(result.stdout.splitlines()):
            if len(row)<4 or not row[1].strip().isdigit(): continue
            if normalized_uuid and row[0].strip().lower().removeprefix('gpu-')!=normalized_uuid: continue
            if int(row[1])!=os.getpid():
                rows.append(dict(gpu_uuid=row[0].strip(),pid=int(row[1]),name=row[2].strip(),memory_mib=row[3].strip()))
        windows=platform.system()=='Windows'
        missing=[r['pid'] for r in rows if not r['name'] or r['name'].startswith('[')]
        resolved=windows_process_names(missing) if windows else {}
        for row in rows:
            row['resolved_name']=resolved.get(row['pid'])
            row['classification']=(gpu_process_kind(row['resolved_name'] or row['name'])
                                   if windows else 'compute_context_reported')
        suspected=[r for r in rows if r['classification'] in ('suspected_compute','compute_context_reported')]
        unknown=[r for r in rows if r['classification']=='unresolved']
        status='BUSY' if suspected else ('BACKGROUND_ACTIVITY_RECORDED' if rows else 'NO_OTHER_COMPUTE_PROCESS_REPORTED')
        return dict(status=status,gpu_uuid=uuid or None,other_compute_processes=rows,
            suspected_compute_processes=suspected,unresolved_processes=unknown,
            background_review_required=bool(rows),
            detection_policy=('Windows process-name heuristic: suspected compute contexts pause by default; desktop and unresolved entries are recorded without blocking.'
                              if windows else 'Other compute contexts reported by nvidia-smi pause by default.'),
            limitation='A process/context listing is not utilization evidence. Desktop and unresolved processes may interfere. WDDM memory N/A does not mean zero usage; no idle-device proof.')
    except (OSError,subprocess.SubprocessError) as exc:
        return dict(status='UNAVAILABLE',reason=str(exc),other_compute_processes=[],background_review_required=True)


class Pipeline:
    """Actual deployed encoder+aggregator; unused window classifier stays off GPU."""
    def __init__(self, encoder, head, condition, family, tokenizer, vocabulary,
                 max_length, padding, device, torch):
        self.encoder=encoder.eval().float().to(device)
        self.head=head.eval().float().to(device)
        self.condition,self.family=condition,family
        self.tokenizer,self.vocabulary=tokenizer,vocabulary
        self.max_length,self.padding=max_length,padding
        self.device,self.torch=device,torch
        for module in (self.encoder,self.head):
            for param in module.parameters():
                param.requires_grad_(False)

    def cpu_inputs(self, rows):
        import numpy as np
        if self.condition in CONDITIONS[:2]:
            values=np.array([[self.vocabulary[field].get(str(value),0)
                for field,value in zip(FIELDS,row)] for row in rows],dtype=np.int64)
            return {'x':self.torch.from_numpy(values)}
        tokens=self.tokenizer(serialize_values(rows),add_special_tokens=True,truncation=False,
            padding='max_length' if self.padding=='fixed_max_length' else True,
            max_length=self.max_length if self.padding=='fixed_max_length' else None,return_tensors='pt')
        require(tokens['input_ids'].shape[1]<=self.max_length, 'Input exceeds original token budget; no truncation allowed')
        require(not bool(tokens['input_ids'].eq(self.tokenizer.unk_token_id).any()), 'Unexpected tokenizer UNK')
        return {k:tokens[k] for k in ('input_ids','attention_mask')}

    def to_device(self, inputs):
        return {k:v.to(self.device) for k,v in inputs.items()}

    def encode(self, inputs):
        result=self.encoder(**inputs)
        return result.last_hidden_state[:,0,:] if hasattr(result,'last_hidden_state') else result

    def aggregate(self, features):
        x=features.reshape(1,10,768)
        if self.family=='Mean_MLP':
            x=x.mean(dim=1)
        elif self.family=='Max_MLP':
            x=x.max(dim=1).values
        elif self.family=='MeanMax_MLP':
            x=self.torch.cat((x.mean(dim=1),x.max(dim=1).values),dim=1)
        return self.head(x)

    def prepare_call(self, scope, rows, cached_features):
        torch=self.torch
        if scope=='head_only':
            features=torch.as_tensor(cached_features,device=self.device)
            return lambda:self.aggregate(features)
        streaming=scope.startswith('stream1')
        selected=rows[-1:] if streaming else rows
        history=torch.as_tensor(cached_features[:9],device=self.device) if streaming else None
        def infer(inputs):
            features=self.encode(inputs)
            if streaming:
                features=torch.cat((history,features),dim=0)
            return self.aggregate(features)
        if scope.endswith('model_only'):
            inputs=self.to_device(self.cpu_inputs(selected))
            return lambda:infer(inputs)
        def end_to_end():
            logits=infer(self.to_device(self.cpu_inputs(selected)))
            return int(logits.argmax(dim=1).item())
        return end_to_end


def checkpoint_seed(checkpoint):
    return checkpoint['seed'] if 'seed' in checkpoint else checkpoint['train_config']['random_seed']


def restore_pipeline(case, env, device, modules, tokenizer):
    import torch
    from transformers import DistilBertConfig, DistilBertForSequenceClassification
    engine=modules['engine']
    if case['track']=='formal46':
        condition=case['condition']
        sizes=[len(env['vocab'][f]) for f in FIELDS]
        if condition==CONDITIONS[0]:
            encoder=engine.CategoricalEncoder(sizes)
        elif condition==CONDITIONS[1]:
            encoder=engine.FieldTokenEncoder(sizes)
        else:
            config=DistilBertConfig.from_json_file(str(env['base_config']))
            config._attn_implementation='eager'
            encoder=engine.BertEncoder(config,weights_path=None)
        window=engine.WindowModel(encoder,head_seed=case['job']['protocol']['window_head_seed'])
        window.load_state_dict(engine.load_checkpoint(case['encoder_checkpoint']),strict=True)
        encoder=window.encoder
        del window
        head=engine.TST()
        head.load_state_dict(engine.load_checkpoint(case['head_checkpoint']),strict=True)
    else:
        loaded,loading=DistilBertForSequenceClassification.from_pretrained(str(env['encoder_dir']),
            local_files_only=True,output_loading_info=True)
        require(not any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),
                'Step8 trained encoder did not restore exactly')
        require(loaded.config.num_labels==6 and loaded.config.dim==768 and
                [loaded.config.id2label[i] for i in range(6)]==list(LABELS) and
                getattr(loaded.config,'step7_dataset_id',None)==DATASET_ID, 'Step8 encoder identity/schema differs')
        encoder=loaded.distilbert
        del loaded
        checkpoint=torch.load(case['head_checkpoint'],map_location='cpu',weights_only=False)
        record=case['head_record']
        require(checkpoint['model_config']==record['model_config'] and checkpoint['train_config']==record['train_config']
                and checkpoint['source_contract']==env['shared_contract'] and checkpoint['provenance']==record['provenance'],
                'Shared head checkpoint configuration/provenance mismatch')
        require(checkpoint_seed(checkpoint)==case['seed'], 'Shared head seed mismatch')
        if case['family']=='TST':
            module=modules['TST'];cls=module.TimeSeriesTransformerClassifier
            require(checkpoint['label_order']==list(LABELS), 'TST class order differs')
        else:
            module=modules['pool'];cls=module.MLPClassifier
            mode={'Mean_MLP':'mean','Max_MLP':'max','MeanMax_MLP':'meanmax'}[case['family']]
            require(checkpoint['pooling_mode']==mode and not checkpoint['train_config']['use_standardization'], 'Pooling preprocessing differs')
        head=cls(module.ModelConfig(**checkpoint['model_config']))
        head.load_state_dict(checkpoint['model_state_dict'],strict=True)
        del checkpoint
    pipeline=Pipeline(encoder,head,case['condition'],case['family'],tokenizer,env['vocab'],
                      env['max_length'],case['padding'],device,torch)
    n_encoder=sum(p.numel() for p in pipeline.encoder.parameters())
    n_head=sum(p.numel() for p in pipeline.head.parameters())
    if 'expected_parameters' in case:
        require(n_encoder+n_head==case['expected_parameters'], 'Deployed parameter count differs from Step46')
    configuration=getattr(getattr(encoder,'model',encoder),'config',None)
    return pipeline,dict(encoder_parameters=n_encoder,head_parameters=n_head,deployment_parameters=n_encoder+n_head,
        attention_implementation=getattr(configuration,'_attn_implementation',None))


def check_probability_replay(actual, expected):
    import numpy as np
    require(actual.shape==expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all(),
            'Historical head probability schema differs')
    require(np.allclose(actual,expected,atol=1e-4,rtol=1e-4) and
            np.array_equal(actual.argmax(1),expected.argmax(1)), 'Historical head prediction replay failed')
    return float(np.max(np.abs(actual-expected)))


def replay_shared_head(pipeline, case, env, references, table):
    """Replay ORIGINAL precision, pooling implementation and original batch boundaries."""
    import numpy as np
    torch=pipeline.torch
    cfg=case['head_record']['train_config']
    batch=cfg['batch_size']*(1 if case['family']=='TST' else 2)
    amp=case['family']!='TST' and bool(cfg['use_amp'])
    require(not amp or pipeline.device.type=='cuda', 'Original pooling AMP replay requires CUDA')
    probabilities=table[['prob_'+label.lower() for label in LABELS]].to_numpy(dtype=float)
    require(np.array_equal(probabilities.argmax(1),table.pred_label_id.to_numpy()), 'Historical probability/decision mismatch')
    maximum=0.;replayed=0
    # Compute the complete original batches containing measured samples, not a
    # new 64-example batch whose AMP rounding may differ from the saved evaluator.
    for block in sorted(set((env['selected']//batch).tolist())):
        lo=block*batch;hi=min(lo+batch,len(env['indices']))
        x=np.asarray(references[env['indices'][lo:hi]],dtype=np.float32)
        if case['family'] in FAMILIES[1:]:
            mean=x.mean(1,dtype=np.float32);maximum_values=x.max(1)
            x=mean if case['family']=='Mean_MLP' else maximum_values if case['family']=='Max_MLP' else np.concatenate((mean,maximum_values),axis=1)
        x=torch.from_numpy(np.ascontiguousarray(x)).to(pipeline.device,dtype=torch.float32)
        with torch.autocast(device_type=pipeline.device.type,dtype=torch.float16,enabled=amp):
            logits=pipeline.head(x)
        values=logits.float().cpu().numpy()
        values=np.exp(values-values.max(1,keepdims=True));values/=values.sum(1,keepdims=True)
        maximum=max(maximum,check_probability_replay(values,probabilities[lo:hi]))
        replayed+=hi-lo
        del x,logits
    return dict(status='PASS',precision='AMP_FP16' if amp else 'FP32',original_batch_size=batch,
                replayed_sequences=replayed,max_abs_probability_error=maximum,predictions_exact=True)


def validate_pipeline(pipeline, case, env, sample_rows):
    """Authenticate loaded weights, check stored encoder references and cache equality."""
    import numpy as np
    import pandas as pd
    torch=pipeline.torch
    chosen=env['selected'];indices=env['indices'][chosen]
    if case['track']=='formal46':
        references=np.load(Path(case['folder'])/'embeddings/val.npy',mmap_mode='r',allow_pickle=False)
        stored_logits=np.load(Path(case['folder'])/'tst/val_logits.npy',mmap_mode='r',allow_pickle=False)
        prediction_table=None
    else:
        with np.load(env['shared_ref'],allow_pickle=False) as arr:
            require(np.array_equal(arr['row_indices'],np.arange(len(env['frame']))) and
                    np.array_equal(arr['label_ids'],env['frame'].label.map(dict(zip(LABELS,range(6)))).to_numpy()),
                    'Shared reference row/label order changed')
            references=arr['embeddings']
        prediction_table=pd.read_csv(case['validation_predictions'],encoding='utf-8-sig').sort_values('sequence_id')
        ends=env['indices'][:,-1]
        require(len(prediction_table)==len(env['indices']) and
                np.array_equal(prediction_table.sequence_id.to_numpy(),np.arange(len(ends))) and
                np.array_equal(prediction_table.episode_id.to_numpy(),env['frame'].iloc[ends].episode_id.to_numpy()) and
                np.array_equal(prediction_table.sequence_end_row.to_numpy(),ends) and
                np.array_equal(prediction_table.sequence_end_step.to_numpy(),env['frame'].iloc[ends].episode_step.to_numpy()),
                'Shared head validation endpoints differ')
        stored_logits=None
    native_replay=(replay_shared_head(pipeline,case,env,references,prediction_table)
                   if prediction_table is not None else None)
    maximum_feature_error=0.;maximum_stream_error=0.;maximum_head_error=0.
    reference_disagreements=0;features=[]
    for number,(rows,ix) in enumerate(zip(sample_rows,indices)):
        full=pipeline.encode(pipeline.to_device(pipeline.cpu_inputs(rows)))
        values=full.detach().cpu().numpy()
        expected=np.asarray(references[ix])
        maximum_feature_error=max(maximum_feature_error,float(np.max(np.abs(values-expected))))
        require(np.isfinite(values).all() and np.allclose(values,expected,rtol=1e-4,atol=1e-4),
                'Stored encoder feature canary failed: '+case['key'])
        logits=pipeline.aggregate(full).detach().cpu().numpy()
        stream=pipeline.prepare_call('stream1_model_only',rows,values)().detach().cpu().numpy()
        maximum_stream_error=max(maximum_stream_error,float(np.max(np.abs(logits-stream))))
        require(np.allclose(logits,stream,rtol=1e-4,atol=1e-4) and np.array_equal(logits.argmax(1),stream.argmax(1)),
                'Full10 and cached stream1 predictions differ: '+case['key'])
        if stored_logits is not None:
            original=np.asarray(stored_logits[chosen[number]])
            maximum_head_error=max(maximum_head_error,float(np.max(np.abs(logits[0]-original))))
            require(np.allclose(logits[0],original,rtol=1e-4,atol=1e-4) and int(logits.argmax())==int(original.argmax()),
                    'Formal TST reference canary failed: '+case['key'])
        else:
            saved=prediction_table.iloc[int(chosen[number])]
            reference_disagreements+=int(int(logits.argmax())!=int(saved.pred_label_id))
        features.append(values.copy())
        del full
    return np.stack(features),dict(status='PASS',samples=len(features),
        encoder_max_abs_error=maximum_feature_error,stream_vs_full_max_abs_logit_error=maximum_stream_error,
        formal_reference_max_abs_logit_error=maximum_head_error if stored_logits is not None else None,
        shared_historical_native_precision_replay=native_replay,
        shared_historical_prediction_disagreements=reference_disagreements if prediction_table is not None else None,
        tolerance=dict(atol=1e-4,rtol=1e-4),
        historical_precision_note='Native historical precision/batch replay must pass first. New FP32 per-sample differences are reported separately; no score-based selection.')


def measurement_schedule(cases, rounds):
    """Fixed order changes; never select the fastest round/model/sample."""
    import numpy as np
    require(len(cases)>0 and rounds>0, 'Empty measurement schedule')
    rng=np.random.default_rng(SAMPLE_SEED+54)
    base=rng.permutation(len(cases));scopes=list(SCOPES);result=[]
    for r in range(rounds):
        order=np.roll(base,r//2)
        if r%2: order=order[::-1]
        shift=r%len(scopes);scope_order=scopes[shift:]+scopes[:shift]
        if r%2: scope_order=scope_order[::-1]
        result.append(dict(round=r,case_indices=order.tolist(),scopes=scope_order))
    return result


def benchmark_scope(pipeline, scope, sample_rows, features, repeats, warmup,
                    *, round_index=0, sample_warmup=1, warmup_seconds=0.):
    """Single-decision wall latency only; no memory probes/GC/cache flush in this loop."""
    import numpy as np
    require(repeats>0 and warmup>0 and sample_warmup>0 and warmup_seconds>=0,
            'Invalid latency warmup/repetition settings')
    require(len(sample_rows)==len(features)>0 and scope in SCOPES, 'Invalid scope/input cohort')
    torch,device=pipeline.torch,pipeline.device
    sync=lambda:synchronize(torch,device)
    # Same sample permutation for every case and scope within a round.
    order=np.random.default_rng(SAMPLE_SEED+1000+round_index).permutation(len(sample_rows))
    raw=[];warm_calls=0;gc_was_enabled=gc.isenabled()
    gc.disable()
    try:
        for position,index in enumerate(order):
            i=int(index)
            call=pipeline.prepare_call(scope,sample_rows[i],features[i])
            minimum=warmup if position==0 else sample_warmup
            seconds=warmup_seconds if position==0 else 0.
            start=time.perf_counter();n=0
            while n<minimum or time.perf_counter()-start<seconds:
                result=call();sync();del result;n+=1
            warm_calls+=n
            for repeat in range(repeats):
                elapsed,result=time_call(call,sync)
                require(np.isfinite(elapsed) and elapsed>=0, 'Invalid measured duration')
                raw.append(dict(round=round_index,sample_index=i,sample_order=position,
                                repeat=repeat,latency_ms=elapsed))
                del result
            del call
    finally:
        if gc_was_enabled: gc.enable()
    stats=summarize_latencies([r['latency_ms'] for r in raw])
    stats['warmup_calls']=warm_calls
    # Memory is populated from the separate pass, not measured here.
    stats.update(peak_allocated_mib=None,peak_reserved_mib=None,
                 peak_incremental_allocated_mib=None,resident_baseline_allocated_mib=None)
    return stats,raw


def summarize_memory(rows):
    require(rows, 'No memory observations')
    def mib(key):
        values=[r[key] for r in rows if r[key] is not None]
        return max(values)/1024**2 if values else None
    return dict(peak_allocated_mib=mib('peak_allocated_bytes'),
                peak_reserved_mib=mib('peak_reserved_bytes'),
                peak_incremental_allocated_mib=mib('incremental_allocated_bytes'),
                resident_baseline_allocated_mib=mib('baseline_allocated_bytes'))


def measure_memory_scope(pipeline, scope, sample_rows, features, repeats=2, warmup=2):
    """Separate untimed pass. Archive every baseline/peak in bytes for independent replay."""
    require(repeats>0 and warmup>0 and len(sample_rows)==len(features)>0, 'Invalid memory pass')
    torch,device=pipeline.torch,pipeline.device;raw=[]
    # Reset cached allocator state only at the boundary of this untimed scope.
    gc.collect()
    if device.type=='cuda': torch.cuda.empty_cache()
    for i,rows in enumerate(sample_rows):
        call=pipeline.prepare_call(scope,rows,features[i])
        for _ in range(warmup):
            result=call();del result
        synchronize(torch,device)
        for repeat in range(repeats):
            baseline=cuda_snapshot(torch,device)
            if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
            result=call();synchronize(torch,device)
            if device.type=='cuda':
                peak=int(torch.cuda.max_memory_allocated(device))
                reserved=int(torch.cuda.max_memory_reserved(device))
                incremental=peak-baseline['allocated_bytes']
                require(incremental>=0, 'Peak memory fell below resident baseline')
            else: peak=reserved=incremental=None
            raw.append(dict(sample_index=i,repeat=repeat,
                baseline_allocated_bytes=baseline['allocated_bytes'],
                baseline_reserved_bytes=baseline['reserved_bytes'],peak_allocated_bytes=peak,
                peak_reserved_bytes=reserved,incremental_allocated_bytes=incremental))
            del result
        del call
    return summarize_memory(raw),raw


def gpu_observation(torch, device):
    """Boundary snapshots only; never issue nvidia-smi inside a latency block."""
    activity=gpu_activity(torch,device)
    result=dict(observed_at_utc=utc(),activity=activity)
    if device.type!='cuda': return result
    fields='uuid,pstate,temperature.gpu,utilization.gpu,clocks.sm,clocks.mem,power.draw'
    try:
        p=subprocess.run(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits'],
                         capture_output=True,text=True,timeout=10,check=False)
        if p.returncode!=0: raise RuntimeError(p.stderr.strip()[:300])
        target=str(getattr(torch.cuda.get_device_properties(device),'uuid','')).lower().removeprefix('gpu-')
        rows=list(csv.reader(p.stdout.splitlines()))
        matched=[r for r in rows if len(r)==7 and r[0].strip().lower().removeprefix('gpu-')==target]
        result['gpu_state']=(dict(zip(fields.split(','),[v.strip() for v in matched[0]]))
                             if len(matched)==1 else dict(status='DEVICE_NOT_RESOLVED'))
    except (OSError,subprocess.SubprocessError,RuntimeError) as exc:
        result['gpu_state']=dict(status='UNAVAILABLE',reason=str(exc))
    return result


def timing_quality(observations):
    """Descriptive warnings, not CIs or automatic outlier removal."""
    import numpy as np
    from collections import defaultdict
    buckets=defaultdict(list)
    for row in observations: buckets[(row['case'],row['scope'],row['round'])].append(row)
    round_rows=[];groups=defaultdict(list)
    for (case,scope,r),rows in sorted(buckets.items()):
        stats=summarize_latencies([x['latency_ms'] for x in rows])
        by_sample=defaultdict(list)
        for x in rows: by_sample[x['sample_order']].append(x['latency_ms'])
        sequence=[float(np.median(by_sample[i])) for i in sorted(by_sample)]
        halves=np.array_split(sequence,2)
        first=float(np.median(halves[0]));last=float(np.median(halves[1])) if len(halves[1]) else first
        ratio=max(first,last)/min(first,last) if min(first,last)>0 else None
        item=dict(case=case,scope=scope,round=r,**stats,
                  first_half_sample_median_ms=first,last_half_sample_median_ms=last,
                  chronological_half_ratio=ratio)
        round_rows.append(item);groups[(case,scope)].append(item)
    details=[]
    for (case,scope),rows in sorted(groups.items()):
        medians=[x['median_ms'] for x in rows];ratio=max(medians)/min(medians) if min(medians)>0 else None
        flags=[]
        if len(rows)<3: flags.append('fewer_than_three_rounds')
        if ratio is None or ratio>1.25: flags.append('round_median_spread_over_25_percent')
        if any(x['chronological_half_ratio'] is None or x['chronological_half_ratio']>1.25 for x in rows):
            flags.append('within_round_half_drift_over_25_percent')
        if any(x['p95_ms']>2*x['median_ms'] for x in rows): flags.append('p95_over_twice_median')
        details.append(dict(case=case,scope=scope,rounds=len(rows),round_medians_ms=medians,
                            round_median_max_over_min=ratio,observations=sum(x['measurements'] for x in rows),flags=flags))
    status='REVIEW_TIMING_VARIABILITY' if any(x['flags'] for x in details) else 'NO_LARGE_VARIABILITY_FLAG'
    return round_rows,dict(status=status,groups=details,
        heuristic='25% round/half median spread, p95/median >2, or fewer than three rounds. Descriptive flags, not statistical tests.',
        retention='All samples and rounds retained, including slow observations. No best-round selection.',
        scope='Warm in-process microbenchmark; no claim of idle-device proof, hard real-time bounds, or accuracy uncertainty.')


def run_measurements(env, out, modules, tokenizer, sample_rows, args, torch, device):
    """One resident pipeline at a time; change model order across complete rounds."""
    from collections import defaultdict
    cases=env['cases'];schedule=measurement_schedule(cases,args.rounds)
    write_json(out/'measurement_schedule.json',schedule)
    raw=[];memory=[];gpu=[];cached_by_case={};parameters_by_case={}
    memory_stats={};quality={};round_summaries=[]
    def snapshot(phase,key,round_index=None):
        state=dict(phase=phase,case=key,round=round_index,**gpu_observation(torch,device))
        gpu.append(state);write_json(out/'gpu_observations.json',gpu)
        require(state['activity']['status']!='BUSY' or args.allow_busy_gpu,
                gpu_busy_message(state['activity'])+' See gpu_observations.json.')
    def restore(case):
        gc.collect()
        if device.type=='cuda': torch.cuda.empty_cache()
        return restore_pipeline(case,env,device,modules,tokenizer)
    def write_latency():
        write_csv(out/'latency_samples.csv',raw)
        rounds,report=timing_quality(raw)
        write_csv(out/'round_latency_summary.csv',rounds)
        write_json(out/'timing_quality.json',report)
        return rounds,report
    # Collect all latency rounds before any dedicated memory measurement.
    for block in schedule:
        r=block['round']
        for case_index in block['case_indices']:
            case=cases[case_index];key=case['key']
            print(f"Round {r+1}/{args.rounds}: {key}",flush=True)
            pipeline,parameters=restore(case)
            with torch.inference_mode():
                if key not in cached_by_case:
                    cached,canary=validate_pipeline(pipeline,case,env,sample_rows)
                    cached_by_case[key]=cached;parameters_by_case[key]=parameters
                    write_json(out/f'canary_{case_index+1:02d}.json',dict(case=key,**canary))
                else:
                    require(parameters==parameters_by_case[key], 'Restored pipeline parameter/backend changed')
                snapshot('latency_before_warmup',key,r)
                for scope in block['scopes']:
                    stats,observations=benchmark_scope(pipeline,scope,sample_rows,cached_by_case[key],
                        args.repeats,args.warmup,round_index=r,sample_warmup=args.sample_warmup,
                        warmup_seconds=args.warmup_seconds)
                    raw.extend(dict(case=key,scope=scope,**row) for row in observations)
                    round_summaries,quality=write_latency()
                    print(f"  {scope}: median {stats['median_ms']:.3f} ms, p95 {stats['p95_ms']:.3f} ms",flush=True)
                snapshot('latency_after',key,r)
            del pipeline
    # Separate pass retains only the current input/history on device.
    for case in cases:
        key=case['key'];print('Memory pass: '+key,flush=True)
        pipeline,parameters=restore(case)
        require(parameters==parameters_by_case[key], 'Memory-pass pipeline differs')
        with torch.inference_mode():
            for scope in SCOPES:
                stats,observations=measure_memory_scope(pipeline,scope,sample_rows,cached_by_case[key],repeats=2,warmup=2)
                memory_stats[(key,scope)]=stats
                memory.extend(dict(case=key,scope=scope,**row) for row in observations)
                write_csv(out/'memory_samples.csv',memory)
        del pipeline
    gc.collect()
    if device.type=='cuda': torch.cuda.empty_cache()
    buckets=defaultdict(list)
    for row in raw: buckets[(row['case'],row['scope'])].append(row['latency_ms'])
    summaries=[]
    for case in cases:
        for scope in SCOPES:
            key=(case['key'],scope);values=buckets[key]
            require(len(values)==args.rounds*len(sample_rows)*args.repeats,'Incomplete latency grid')
            details=dict(case=case['key'],track=case['track'],condition=case['condition'],family=case['family'],
                seed=case['seed'],scope=scope,padding=case['padding'],precision='FP32',device=str(device),
                samples=len(sample_rows),repeats=args.repeats,rounds=args.rounds,**parameters_by_case[case['key']])
            summaries.append(dict(**details,**summarize_latencies(values),**memory_stats[key]))
    write_csv(out/'cost_summary.csv',summaries)
    return dict(completed_pipelines=len(cases),summary_rows=len(summaries),timing_observations=len(raw),
        memory_observations=len(memory),timing_quality=quality['status'],
        interference_reported=any(x['activity']['status']=='BUSY' for x in gpu),
        gpu_background_review_required=any(x['activity'].get('background_review_required',False) for x in gpu))


def make_inventory(env, args):
    return dict(version=VERSION,created_at_utc=utc(),dataset_id=DATASET_ID,
        project_root=str(env['root']),step46_run=str(env['r46']),step48_run=str(env['r48']),
        step46_identity=env['p46']['identity'],step48_identity=env['p48']['identity'],
        seed_policy='all recorded seeds' if args.all_seeds else 'prespecified seed 20260610; no score or latency selection',
        samples=args.samples,sample_seed=SAMPLE_SEED,source_split='validation',
        selected_sequence_indices=env['selected'].tolist(),
        model_cases=[{k:v for k,v in c.items() if k not in ('job','head_record')} for c in env['cases']],
        input_sha256=dict(VERIFIED),script_sha256=digest(Path(__file__)))


def finish_archive(out, manifest):
    manifest['finished_at_utc']=utc()
    manifest['output_sha256']={p.relative_to(out).as_posix():digest(p)
        for p in sorted(out.rglob('*')) if p.is_file() and p.name!='run_manifest.json' and p.suffix!='.tmp'}
    write_json(out/'run_manifest.json',manifest)
    suffix='_FAILED_review.zip' if manifest['status']=='FAILED' else '_review.zip'
    archive=out.with_name(out.name+suffix)
    temporary=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix!='.tmp':
                z.write(p,p.relative_to(out).as_posix())
    temporary.replace(archive)
    return archive


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--run46',type=Path,help='Explicit completed Step46 folder if several exist')
    p.add_argument('--run48',type=Path,help='Explicit completed Step48 folder if several exist')
    p.add_argument('--base-model-dir',type=Path,help='Existing pinned DistilBERT config directory')
    p.add_argument('--preflight-only',action='store_true',help='Verify paths/hashes and inventory; no torch/CUDA/model loading')
    p.add_argument('--all-seeds',action='store_true',help='All 20 formal encoder pairs and 40 shared encoder/head pairs')
    p.add_argument('--samples',type=int,default=64)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--warmup',type=int,default=50,help='Minimum warmup calls at each scope/round start')
    p.add_argument('--warmup-seconds',type=float,default=.5,help='Also warm each scope/round for at least this duration')
    p.add_argument('--sample-warmup',type=int,default=2,help='Warmups when switching to each remaining input')
    p.add_argument('--rounds',type=int,default=3,help='Complete measurement rounds; all are retained')
    p.add_argument('--cuda-device',type=int,default=0)
    p.add_argument('--threads',type=int,default=None)
    p.add_argument('--allow-busy-gpu',action='store_true',help='Exploratory only: allow suspected compute contexts and mark results; Windows desktop/unresolved contexts are already logged without blocking')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    require(args.samples>0 and args.repeats>0 and args.warmup>0 and args.rounds>0 and args.sample_warmup>0,
            'samples/repeats/warmup/rounds/sample-warmup must be positive')
    import math
    require(math.isfinite(args.warmup_seconds) and args.warmup_seconds>=0, 'warmup-seconds must be finite and nonnegative')
    require(args.threads is None or args.threads>0, 'threads must be positive')
    root=args.project_root.expanduser().resolve()
    require((root/DATA).is_dir(), 'Place this script beside the existing numbered scripts, or supply --project-root PATH')
    out=root/PREFIX/'54_inference_cost'/('run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='RUNNING',version=VERSION,started_at_utc=utc(),training=False,
                  test_used_for_selection=False,output_directory=str(out))
    write_json(out/'run_manifest.json',manifest)
    source_copy=out/'source'/Path(__file__).name;source_copy.parent.mkdir()
    source_copy.write_bytes(Path(__file__).read_bytes())
    try:
        print('Step54 '+VERSION+': locate existing recorded code/checkpoints and verify source hashes ...',flush=True)
        env=prepare_sources(root,args)
        inventory=make_inventory(env,args);write_json(out/'source_inventory.json',inventory)
        endpoint=env['frame'].iloc[env['indices'][env['selected'][:],-1]].copy()
        endpoint=endpoint[['episode_id','episode_step','window_id','label']]
        endpoint.insert(0,'selected_sequence_index',env['selected'])
        endpoint.to_csv(out/'measurement_endpoints.csv',index=False,encoding='utf-8-sig')
        print(f"Verified {len(env['cases'])} existing pipelines; {args.samples} common validation L10 endpoints.",flush=True)
        for case in env['cases']:
            print('  '+case['key'],flush=True)
        if args.preflight_only:
            manifest['status']='PREFLIGHT_COMPLETE_NO_TIMING'
            archive=finish_archive(out,manifest)
            print('Preflight complete. No CUDA or model execution.\nReview ZIP: '+str(archive),flush=True)
            return 0
        import numpy as np
        import torch
        import transformers
        from transformers import AutoTokenizer
        require(torch.cuda.is_available(), 'CUDA unavailable; use the training environment. --preflight-only needs no CUDA.')
        device=torch.device(f'cuda:{args.cuda_device}')
        if device.type=='cuda':
            torch.cuda.set_device(device)
        threads=args.threads or env['p46']['numerical_runtime']['torch_num_threads']
        configure_numeric(torch,threads)
        activity=gpu_activity(torch,device)
        write_json(out/'gpu_activity_before.json',activity)
        require(activity['status']!='BUSY' or args.allow_busy_gpu,
                gpu_busy_message(activity))
        if activity.get('background_review_required'):
            print('GPU background entries recorded; this is not proof of a competing training job. '+
                  'See gpu_activity_before.json; timing variability will be reported.',flush=True)
        runtime=dict(python=platform.python_version(),platform=platform.platform(),torch=str(torch.__version__),
            transformers=transformers.__version__,numpy=np.__version__,device=str(device),
            device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor(),
            cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),torch_threads=torch.get_num_threads(),
            torch_interop_threads=torch.get_num_interop_threads(),cudnn_benchmark=False,cudnn_deterministic=True,
            cuda_matmul_tf32=False,cudnn_tf32=False,precision='FP32',gpu_activity=activity)
        write_json(out/'runtime.json',runtime)
        protocol=dict(version=VERSION,inventory_sha256=digest(out/'source_inventory.json'),runtime=runtime,
            samples=args.samples,repeats=args.repeats,rounds=args.rounds,
            warmup_first_sample=args.warmup,warmup_minimum_seconds=args.warmup_seconds,
            warmup_each_other_sample=args.sample_warmup,
            timing_method='Single-decision synchronized wall clock; cyclic GC disabled during latency scopes; no allocator cleanup or probes inside latency blocks.',
            ordering='Fixed seed permutations of common inputs; reversed/rotated case and scope orders across rounds; measurement_schedule.json records actual order.',
            aggregation='All rounds/samples/repeats pooled; no trimming, fastest-round selection, or adaptive stopping.',
            memory_method='Separate untimed pass after all latency rounds; 2 warmups and 2 recorded calls for every input/scope; baseline and peak bytes archived.',
            gpu_monitoring='Boundary snapshots; on Windows, known desktop and unresolved process entries are logged without blocking. Suspected compute executable names pause by default; name classification is heuristic, not utilization evidence.',
            sample_seed=SAMPLE_SEED,scopes=list(SCOPES),decision_batch_size=1,sequence_length=10,
            full10_encoder_window_batch=10,stream1_encoder_window_batch=1,
            end_to_end='in-memory 22 field values -> serialization/tokenization or category lookup -> CPU-to-device -> encoder -> aggregator -> CPU class id',
            model_only='preprocessed device tensors -> encoder and aggregator -> device logits; synchronized wall clock',
            head_only='resident cached [1,10,768] CLS features -> pooling when required and classifier -> device logits',
            memory_residency='encoder plus head remain loaded in every scope, including head_only; only the current sample/cache is on GPU',
            stream1='nine previously encoded CLS vectors already resident on device; encode one new window and concatenate/update context',
            excludes=['disk I/O','model loading','raw-to-22-field extraction','telemetry acquisition/wait time','network transport'],
            limitations=['Hardware-specific measurements; not satellite onboard latency.',
                'PyTorch allocated/reserved GPU memory; not CPU RAM, whole-process GPU memory or whole-device usage.',
                'Formal encoders use original fixed padding; shared Step8 uses original dynamic padding. Compare aggregators within the shared track.',
                'All timed heads use FP32. Historical pooling accuracy used AMP; no new full FP32 accuracy evaluation is claimed.',
                'Repeated timings share samples/checkpoints; they are not independent training seeds or accuracy confidence intervals.',
                'Warm steady-state in-process timings exclude cyclic-GC pauses; no hard real-time or cold-start claim.',
                'timing_quality.json contains descriptive variability flags, not a statistical equivalence or stability proof.'],
            canary=dict(atol=1e-4,rtol=1e-4,formal_predictions_exact=True))
        write_json(out/'protocol.json',protocol)
        modules={'engine':import_code(env['engine_path'],'cost54_engine')}
        for role,path in env['codes'].items():
            modules[role]=import_code(path,'cost54_'+role)
        engine=modules['engine']
        require(not engine.DEPENDENCY_ERROR and tuple(engine.FIELDS)==FIELDS and tuple(engine.LABELS)==LABELS, 'Existing Step43 dependency/schema error')
        tokenizer=AutoTokenizer.from_pretrained(str(env['encoder_dir']),local_files_only=True,use_fast=True)
        require(tokenizer.is_fast and tokenizer.padding_side=='right' and tokenizer.pad_token_id is not None
                and tokenizer.unk_token_id is not None, 'Expected original fast right-padded tokenizer')
        backend=json.loads(tokenizer.backend_tokenizer.to_str());backend.pop('padding',None);backend.pop('truncation',None)
        fingerprint=object_hash(dict(backend=backend,special_tokens_map=tokenizer.special_tokens_map,
            model_input_names=tokenizer.model_input_names))
        require(fingerprint==env['tokenizer_sha256'], 'Original tokenizer fingerprint differs')
        values=env['frame'][list(FIELDS)].to_numpy(dtype=str)
        sample_rows=[values[ix].tolist() for ix in env['indices'][env['selected']]]
        measured=run_measurements(env,out,modules,tokenizer,sample_rows,args,torch,device)
        ending=gpu_activity(torch,device);write_json(out/'gpu_activity_after.json',ending)
        # Consumed sources must still match the authenticated bytes after benchmarking.
        for path,sha in list(VERIFIED.items()):
            checked(path,sha)
        measured['interference_reported'] = measured['interference_reported'] or activity['status']=='BUSY' or ending['status']=='BUSY'
        measured['gpu_background_review_required'] = measured['gpu_background_review_required'] or activity.get('background_review_required',False) or ending.get('background_review_required',False)
        quality=read_json(out/'timing_quality.json')
        quality['gpu_background_review_required']=measured['gpu_background_review_required']
        quality['gpu_interference_reported']=measured['interference_reported']
        quality['gpu_activity_note']='Background or unresolved GPU contexts are not an idle-device proof; inspect GPU logs together with timing distributions.'
        write_json(out/'timing_quality.json',quality)
        manifest.update(status='COST_MEASUREMENT_COMPLETE',**measured,
            gpu_idle_detection_available=activity['status'] not in ('UNAVAILABLE','CPU_REQUESTED'),
            input_sha256=dict(VERIFIED))
        archive=finish_archive(out,manifest)
        print('Timing quality: '+measured['timing_quality']+' (see timing_quality.json)',flush=True)
        print('\nCompleted. Summary: '+str(out/'cost_summary.csv')+'\nReview ZIP: '+str(archive),flush=True)
        return 0
    except Exception as exc:
        manifest.update(status='FAILED',error_type=type(exc).__name__,error=str(exc),input_sha256=dict(VERIFIED))
        (out/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        archive=finish_archive(out,manifest)
        print('\nSTOPPED: '+str(exc)+'\nDiagnostic ZIP: '+str(archive),file=sys.stderr,flush=True)
        return 1


if __name__=='__main__':
    sys.exit(main())
