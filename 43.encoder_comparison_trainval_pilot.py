#!/usr/bin/env python3
"""TAES shared-nominal encoder train/validation pilot (script 43).

Save beside the existing numbered scripts and run:
    python "43.encoder_comparison_trainval_pilot.py"
Resume the same protocol at its last completed epoch:
    python "43.encoder_comparison_trainval_pilot.py" --resume "outputs/shared_nominal_v2/43_encoder_comparison_trainval_pilot/run_..."

Four encoders, one pilot seed, retrained L10 TST control. No test evaluation.
The default run uses main candidates only. This is NOT the formal five-seed study.
Requires the completed revised 42 audit and its exact Python source beside this
script (filename changes are allowed; content is hash-checked). Train/validation
inputs come from shared_nominal_v2. No test windows, predictions or embeddings
are read. Only metadata/counts already in upstream provenance are inspected.
Window and L10 scores include a matched L10 endpoint comparison. Pilot encoder
budgets are retained to inspect convergence, not to assert a causal pretraining
benefit. Validation/extraction are FP32; AMP applies to training only.
All supplied episode labels are retained, including quiet attack windows.
Counterfactual evidence annotations are provenance/diagnostics, never features,
sample filters, relabeling rules or checkpoint-selection criteria in this pilot.
Steps 10/11/12 are optional references, so an ongoing 11/12 run is not an input.
The pinned public pretrained initializer is distinct from the fine-tuned Step 8
checkpoint. Default is local/cache only; --allow-download permits its download.
Original inputs/results are read-only. Model weights/embeddings stay outside Review ZIP.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import platform
import random
import sys
import time
import traceback
from types import SimpleNamespace
import zipfile

DEPENDENCY_ERROR = None
try:
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
except (ImportError, OSError, ValueError) as exc:
    DEPENDENCY_ERROR = str(exc)
    np = pd = torch = nn = None
Module = nn.Module if nn is not None else object

FIELDS = ('observation_stage','path_type','direction','source_role','destination_role',
          'protocol_context','operational_phase','message_type','service_context',
          'command_intent','payload_or_parameter_context','sequence_freshness_status',
          'control_security_state','signal_quality_state','tracking_stability',
          'navigation_consistency','timing_consistency','decoder_status',
          'command_execution_result','retry_timeout_state','session_state','state_transition_consistency')
METADATA = ('scenario_id','episode_id','episode_type','episode_step','window_id','timestamp','label')
LABELS = ('NORMAL','RF_LINK_JAMMING','GNSS_SPOOFING','UNAUTHORIZED_TELECOMMAND_INJECTION',
          'TELECOMMAND_REPLAY','GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE')
LABEL_TO_ID = dict(zip(LABELS,range(6)))
CONDITIONS = ('CategoricalEmbedding_MLP','FieldToken_Encoder',
              'DistilBERT_RandomInit_Train','DistilBERT_Pretrained_Finetune')
DATA_REL = 'dataset/shared_nominal_v2'
OUTPUT_REL = 'outputs/shared_nominal_v2/43_encoder_comparison_trainval_pilot'
AUDIT_REL = 'outputs/shared_nominal_v2/42_encoder_comparison_audit'
AUDIT_PROTOCOL = 'shared_nominal_encoder_input_audit_v3'
PROTOCOL_VERSION = 'shared_nominal_trainval_pilot_v3'
PILOT_SOURCE_KEYS = ('dataset_id','step8_run_id','step9_run_id','input_sha256',
                     'step8_model_sha256','step9_artifacts','tokenizer_sha256','observability_audit')
BASE_ID = 'distilbert/distilbert-base-uncased'
BASE_REVISION = '12040accade4e8a0f71eabdb258fecc2e7e948be'
BASE_WEIGHT_SHA = '5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063'
TST_MODEL = dict(input_dim=768,num_classes=6,sequence_length=10,temporal_context_seconds=10,
                 d_model=128,num_heads=4,num_encoder_layers=1,dim_feedforward=256,dropout=.25,pooling='last')


def require(ok, message):
    if not ok: raise ValueError(message)


def utc(): return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def object_hash(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_csv(path,rows,columns=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns or list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    temp.replace(path)


def save_checkpoint(path,payload):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    torch.save(payload,temp);temp.replace(path)


def load_checkpoint(path):
    """torch.load does not verify tensor-storage CRCs; validate the complete archive first."""
    path=Path(path)
    try:
        with zipfile.ZipFile(path) as archive:
            require(archive.testzip() is None,f'Checkpoint integrity/CRC failed: {path}')
    except (zipfile.BadZipFile,EOFError,OSError) as exc:
        raise ValueError(f'Checkpoint integrity/CRC failed: {path}') from exc
    return torch.load(path,map_location='cpu',weights_only=True)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def serialize_frame(df):
    return ['; '.join(f'{f} is {str(value).strip()}' for f,value in zip(FIELDS,row))+'.'
            for row in df[list(FIELDS)].itertuples(index=False,name=None)]


def encode_fields(df,vocab):
    return np.stack([df[f].astype(str).str.strip().map(vocab[f]).fillna(0).to_numpy(dtype=np.int64)
                     for f in FIELDS],axis=1)


def sequence_view(df,length=10):
    """Indices refer to the input row order; windows never cross an episode."""
    require(not df[['episode_id','episode_step']].duplicated().any(),'Duplicate episode/step')
    blocks=[]
    for _,positions in df.groupby('episode_id',sort=False).indices.items():
        positions=np.asarray(positions)
        positions=positions[np.argsort(df.iloc[positions].episode_step.to_numpy(),kind='stable')]
        part=df.iloc[positions]
        require(part.label.nunique()==1,'Mixed labels within episode')
        require(np.all(np.diff(part.episode_step.to_numpy(dtype=np.int64))==1),'Nonconsecutive episode steps')
        if len(positions)>=length:blocks.append(np.lib.stride_tricks.sliding_window_view(positions,length).copy())
    require(bool(blocks),'No complete L10 sequences')
    indices=np.concatenate(blocks).astype(np.int64)
    meta=df.iloc[indices[:,-1]][list(METADATA)].reset_index(drop=True)
    meta['sequence_id']=np.arange(len(indices),dtype=np.int64)
    meta['sequence_start_row']=indices[:,0];meta['sequence_end_row']=indices[:,-1]
    meta['sequence_length']=length
    y=meta.label.map(LABEL_TO_ID).to_numpy(dtype=np.int64)
    return indices,y,meta


def metrics(y,logits,loss):
    logits=np.asarray(logits);y=np.asarray(y,dtype=np.int64)
    require(logits.ndim==2 and logits.shape==(len(y),6) and y.ndim==1 and ((y>=0)&(y<6)).all(),'Invalid classification shape/labels')
    pred=logits.argmax(axis=1)
    require(len(y)>0 and np.isfinite(logits).all(),'Invalid/empty predictions')
    cm=np.bincount(y*6+pred,minlength=36).reshape(6,6)
    tp=np.diag(cm);den=cm.sum(axis=0)+cm.sum(axis=1)
    f1=np.divide(2.*tp,den,out=np.zeros(6),where=den>0)
    return dict(loss=float(loss),accuracy=float(tp.sum()/len(y)),macro_f1=float(f1.mean()),
                weighted_f1=float((f1*cm.sum(axis=1)).sum()/len(y)),
                class_f1=f1.tolist(),confusion_matrix=cm.tolist(),n=len(y))


class CategoricalEncoder(Module):
    def __init__(self,sizes,output_dim=768):
        super().__init__();self.value_embeddings=nn.ModuleList([nn.Embedding(n+1,16,padding_idx=0) for n in sizes])
        self.net=nn.Sequential(nn.Linear(len(sizes)*16,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,output_dim))
    def forward(self,x):return self.net(torch.cat([e(x[:,i]) for i,e in enumerate(self.value_embeddings)],dim=1))


class FieldTokenEncoder(Module):
    def __init__(self,sizes,output_dim=768):
        super().__init__();self.value_embeddings=nn.ModuleList([nn.Embedding(n+1,64,padding_idx=0) for n in sizes])
        self.field_embedding=nn.Parameter(torch.empty(1,len(sizes),64))
        self.cls=nn.Parameter(torch.empty(1,1,64))
        self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(64,4,128,.1,activation='gelu',batch_first=True),2)
        # PyTorch clones layers with identical initial values; initialize each layer independently.
        for layer in self.encoder.layers:
            for name,p in layer.named_parameters():
                if p.ndim>1:nn.init.xavier_uniform_(p)
                elif 'weight' in name:nn.init.ones_(p)
                else:nn.init.zeros_(p)
        self.norm=nn.LayerNorm(64);self.projection=nn.Linear(64,output_dim)
        nn.init.normal_(self.cls,std=.02);nn.init.normal_(self.field_embedding,std=.02)
    def forward(self,x):
        values=torch.stack([e(x[:,i]) for i,e in enumerate(self.value_embeddings)],dim=1)+self.field_embedding
        h=self.encoder(torch.cat([self.cls.expand(x.shape[0],-1,-1),values],dim=1))
        return self.projection(self.norm(h[:,0,:]))


class BertEncoder(Module):
    def __init__(self,config,weights_path=None):
        super().__init__()
        from transformers import DistilBertModel
        self.model=DistilBertModel(config)
        if weights_path is not None:
            from safetensors.torch import load_file
            state=load_file(str(weights_path),device='cpu')
            if any(k.startswith('distilbert.') for k in state):
                state={k[len('distilbert.'):]:v for k,v in state.items() if k.startswith('distilbert.')}
            self.model.load_state_dict(state,strict=True)
    def forward(self,input_ids,attention_mask):
        return self.model(input_ids=input_ids,attention_mask=attention_mask,return_dict=True).last_hidden_state[:,0,:]


class WindowModel(Module):
    def __init__(self,encoder,dim=768,head_seed=20260929):
        super().__init__();self.encoder=encoder
        # CPU-only construction; fork RNG keeps encoder/dropout streams independent of head creation.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(head_seed)
            self.head=nn.Sequential(nn.Linear(dim,dim),nn.ReLU(),nn.Dropout(.2),nn.Linear(dim,6))
            for layer in self.head:
                if isinstance(layer,nn.Linear):nn.init.normal_(layer.weight,std=.02);nn.init.zeros_(layer.bias)
    def features(self,**batch):return self.encoder(**batch)
    def forward(self,**batch):return self.head(self.features(**batch))


class TST(Module):
    """Architecture/state-dict keys follow the user's existing script 11."""
    def __init__(self):
        super().__init__();c=SimpleNamespace(**TST_MODEL);self.config=c
        self.input_norm=nn.LayerNorm(c.input_dim);self.input_projection=nn.Linear(c.input_dim,c.d_model)
        self.position_embedding=nn.Parameter(torch.zeros(1,c.sequence_length,c.d_model));self.dropout=nn.Dropout(c.dropout)
        layer=nn.TransformerEncoderLayer(c.d_model,c.num_heads,c.dim_feedforward,c.dropout,
                                        activation='gelu',batch_first=True,norm_first=False)
        self.encoder=nn.TransformerEncoder(layer,c.num_encoder_layers);self.output_norm=nn.LayerNorm(c.d_model)
        self.classifier=nn.Sequential(nn.Dropout(c.dropout),nn.Linear(c.d_model,c.num_classes))
        nn.init.trunc_normal_(self.position_embedding,std=.02)
        nn.init.xavier_uniform_(self.input_projection.weight);nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.classifier[-1].weight);nn.init.zeros_(self.classifier[-1].bias)
    def forward(self,x):
        require(x.ndim==3 and tuple(x.shape[1:])==(10,768),'Expected TST [B,10,768]')
        h=self.input_projection(self.input_norm(x))+self.position_embedding
        h=self.encoder(self.dropout(h))
        return self.classifier(self.output_norm(h[:,-1,:]))


class ArrayBatches:
    def __init__(self,arrays,labels):
        self.arrays=arrays;self.labels=np.asarray(labels,dtype=np.int64)
        require(all(len(a)==len(self.labels) for a in arrays.values()),'Input/label count mismatch')
    def tensors(self,indices,device):
        data={k:torch.from_numpy(np.asarray(a[indices]).copy()).to(device) for k,a in self.arrays.items()}
        for k in ('input_ids','attention_mask'):
            if k in data:data[k]=data[k].long()
        if 'attention_mask' in data:
            n=int(data['attention_mask'].sum(dim=1).max().item())
            data={k:v[:,:n] for k,v in data.items()}
        return data,torch.as_tensor(self.labels[indices],dtype=torch.long,device=device)


class SequenceBatches:
    def __init__(self,embeddings,indices,labels):self.embeddings=embeddings;self.indices=indices;self.labels=labels
    def tensors(self,indices,device):
        x=np.asarray(self.embeddings[self.indices[indices]],dtype=np.float32)
        return {'x':torch.from_numpy(x).to(device)},torch.as_tensor(self.labels[indices],dtype=torch.long,device=device)


def amp_context(device,amp):
    return torch.autocast(device_type='cuda',dtype=torch.float16) if amp and device.type=='cuda' else contextlib.nullcontext()


def update_batch(model,batches,indices,optimizer,micro_batch,device,label_smoothing,amp,scaler,clip=1.):
    optimizer.zero_grad(set_to_none=True);total=0.;count=len(indices)
    for start in range(0,count,micro_batch):
        args,y=batches.tensors(indices[start:start+micro_batch],device)
        with amp_context(device,amp):
            logits=model(**args)
            loss=nn.functional.cross_entropy(logits,y,label_smoothing=label_smoothing,reduction='sum')/count
        require(bool(torch.isfinite(loss).item()),'Non-finite training loss')
        if scaler is not None:scaler.scale(loss).backward()
        else:loss.backward()
        total+=float(loss.detach())
    if scaler is not None:scaler.unscale_(optimizer)
    norm=nn.utils.clip_grad_norm_(model.parameters(),clip)
    if scaler is None:require(bool(torch.isfinite(norm).item()),'Non-finite gradient')
    updated=True
    if scaler is not None:
        old=scaler.get_scale();scaler.step(optimizer);scaler.update();updated=scaler.get_scale()>=old
    else:optimizer.step()
    return total,updated


@contextlib.contextmanager
def evaluating(model):
    was_training=model.training;model.eval()
    try:
        with torch.no_grad():yield
    finally:model.train(was_training)


def evaluate(model,batches,batch_size,device,smoothing=0.,amp=False):
    outputs=[];loss_sum=0.
    with evaluating(model):
        for start in range(0,len(batches.labels),batch_size):
            args,y=batches.tensors(np.arange(start,min(start+batch_size,len(batches.labels))),device)
            with amp_context(device,amp):
                logits=model(**args)
                loss=nn.functional.cross_entropy(logits,y,label_smoothing=smoothing,reduction='sum')
            loss_sum+=float(loss);outputs.append(logits.float().cpu().numpy())
    logits=np.concatenate(outputs)
    return metrics(batches.labels,logits,loss_sum/len(batches.labels)),logits


def better(current,best):
    return best is None or current['macro_f1']>best['macro_f1']+1e-12 or (
        abs(current['macro_f1']-best['macro_f1'])<=1e-12 and current['loss']<best['loss'])


def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def fit(model,train,val,cfg,out,identity,device):
    """Atomic epoch checkpoint includes current AND best state, optimizer, scheduler and RNG."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    contract=object_hash({'identity':identity,'config':cfg});model.to(device)
    if (out/'complete.json').exists():
        result=read_json(out/'complete.json');require(result['contract']==contract,'Training protocol/identity changed')
        verify_stage_artifacts(out,result)
        model.load_state_dict(load_checkpoint(out/'best.pt'),strict=True)
        return result
    seed_all(cfg['seed']);model.to(device)
    decay=[];no_decay=[]
    for name,p in model.named_parameters():
        if p.requires_grad:(no_decay if cfg['scheduler']=='linear' and (p.ndim==1 or name.endswith('bias')) else decay).append(p)
    optimizer=torch.optim.AdamW([{'params':decay,'weight_decay':cfg['weight_decay']},
                                 {'params':no_decay,'weight_decay':0.}],lr=cfg['learning_rate'])
    steps=math.ceil(len(train.labels)/cfg['batch_size'])*cfg['max_epochs']
    warmup=math.ceil(steps*cfg['warmup_ratio'])
    if cfg['scheduler']=='linear':
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:s/max(1,warmup) if s<warmup else max(0.,(steps-s)/max(1,steps-warmup)))
    else:scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=2,min_lr=1e-6)
    amp=cfg['amp'] and device.type=='cuda'
    scaler=torch.amp.GradScaler('cuda') if amp else None
    history=[];best=None;best_state=None;best_epoch=0;bad=0;plateau_best=-1.;start_epoch=1
    if (out/'last.pt').exists():
        ck=load_checkpoint(out/'last.pt')
        require(ck['contract']==contract,'Checkpoint protocol/identity changed')
        model.load_state_dict(ck['model'],strict=True);optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        if scaler is not None:scaler.load_state_dict(ck['scaler'])
        history=ck['history'];best=ck['best'];best_state=ck['best_model'];best_epoch=ck['best_epoch']
        bad=ck['bad'];plateau_best=ck['plateau_best'];start_epoch=ck['epoch']+1
        restore_rng_state(ck)
        torch.set_rng_state(ck['torch_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(ck['cuda_rng'])
        print(f'  Resume {out.name} from epoch {start_epoch}',flush=True)
    should_stop=bool(history) and len(history)>=cfg['min_epochs'] and bad>=cfg['patience']
    for epoch in ([] if should_stop else range(start_epoch,cfg['max_epochs']+1)):
        t=time.perf_counter();model.train();total=0.;overflows=0;successful_updates=0
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        order=np.random.default_rng(cfg['shuffle_seed']+epoch).permutation(len(train.labels))
        for step,start in enumerate(range(0,len(order),cfg['batch_size']),1):
            ids=order[start:start+cfg['batch_size']]
            loss,updated=update_batch(model,train,ids,optimizer,cfg['micro_batch_size'],device,cfg['label_smoothing'],amp,scaler)
            total+=loss*len(ids);overflows+=int(not updated);successful_updates+=int(updated)
            if updated and cfg['scheduler']=='linear':scheduler.step()
            if step%500==0:print(f'    epoch {epoch}: {start+len(ids):,}/{len(order):,} rows',flush=True)
        require(successful_updates>0,'All optimizer updates were skipped; no trained epoch can be reported')
        summary,_=evaluate(model,val,cfg['eval_batch_size'],device,cfg['label_smoothing'],False)
        if cfg['scheduler']!='linear':scheduler.step(summary['macro_f1'])
        improved=better(summary,best)
        if improved:best=summary;best_epoch=epoch;best_state=cpu_state(model)
        if cfg['scheduler']=='linear':
            if summary['macro_f1']>plateau_best+cfg['plateau_delta']:plateau_best=summary['macro_f1'];bad=0
            else:bad+=1
        else:bad=0 if improved else bad+1
        row=dict(epoch=epoch,train_loss=total/len(order),val_loss=summary['loss'],val_macro_f1=summary['macro_f1'],
                 val_accuracy=summary['accuracy'],learning_rate=optimizer.param_groups[0]['lr'],
                 elapsed_seconds=time.perf_counter()-t,amp_skipped_updates=overflows,optimizer_updates=successful_updates,
                 peak_gpu_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else 0)
        history.append(row)
        ck=dict(contract=contract,epoch=epoch,model=cpu_state(model),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict() if scaler else None,history=history,best=best,best_model=best_state,
                best_epoch=best_epoch,bad=bad,plateau_best=plateau_best,torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],**save_rng_state())
        save_checkpoint(out/'last.pt',ck);write_csv(out/'history.csv',history)
        print(f'  epoch {epoch}/{cfg["max_epochs"]} | train loss {row["train_loss"]:.5f} | val F1 {summary["macro_f1"]:.6f} | best {best_epoch}',flush=True)
        del ck
        if epoch>=cfg['min_epochs'] and bad>=cfg['patience']:break
    require(best_state is not None,'No trained checkpoint')
    model.load_state_dict(best_state,strict=True);save_checkpoint(out/'best.pt',best_state)
    write_csv(out/'history.csv',history)
    evaluated,logits=evaluate(model,val,cfg['eval_batch_size'],device,cfg['label_smoothing'],False)
    np.save(out/'val_logits.npy',logits,allow_pickle=False)
    convergence='PLATEAU_OBSERVED' if bad>=cfg['patience'] else 'BUDGET_REVIEW_REQUIRED'
    result=dict(contract=contract,best_epoch=best_epoch,best_metrics=evaluated,selection_metrics=best,
                epochs_completed=len(history),convergence=convergence,
                training_seconds=sum(r['elapsed_seconds'] for r in history),
                evaluation_precision='float32',optimizer_updates=sum(r['optimizer_updates'] for r in history),
                parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                peak_gpu_bytes=max(r['peak_gpu_bytes'] for r in history),
                artifact_sha256={name:sha256(out/name) for name in ['best.pt','last.pt','history.csv','val_logits.npy']})
    write_json(out/'complete.json',result)
    return result


def read_audit(path):
    """Verify a revised 42 result; completion is an input audit, not a formal protocol."""
    path=Path(path).resolve()
    with contextlib.ExitStack() as stack:
        if path.is_file():
            archive=stack.enter_context(zipfile.ZipFile(path))
            require(archive.testzip() is None,'42 ZIP CRC failed')
            require(len(archive.namelist())==len(set(archive.namelist())),'Duplicate 42 ZIP names')
            read=lambda name:archive.read(name)
        else:
            require(path.is_dir(),f'Step 42 run not found: {path}')
            read=lambda name:(path/name).read_bytes()
        raw=read('run_manifest.json');manifest=json.loads(raw)
        require(manifest.get('status')=='AUDIT_COMPLETE' and
                manifest.get('protocol_version')==AUDIT_PROTOCOL,
                'Run the shared_nominal_v2 Step 42 first; a legacy/incomplete audit cannot initialize this pilot')
        result={'manifest':manifest,'manifest_sha256':hashlib.sha256(raw).hexdigest(),'path':str(path)}
        hashes=manifest.get('output_sha256',{})
        required={'protocol.json','input_audit.json','dataset_audit.json','reference_settings.json',
                  'tokenization_summary.json','train_field_vocabulary.json','checkpoint_inventory.json','comparison_design.json'}
        require(required<=set(hashes),'Step 42 result lacks required verified files')
        for name,digest in hashes.items():
            require(Path(name).name==name and '\\' not in name and ':' not in name,'Unsafe 42 output name')
            data=read(name);require(hashlib.sha256(data).hexdigest()==digest,f'42 output hash mismatch: {name}')
            if name.endswith('.json'):result[name[:-5]]=json.loads(data)
        summary=result['tokenization_summary']
        require(result['protocol'].get('version')==AUDIT_PROTOCOL
                and summary.get('actual_pipeline_truncation') is False
                and summary.get('all_train_validation_rows_checked') is True,'Step 42 tokenizer protocol differs')
        require(set(summary.get('splits',{}))=={'train','val'},'Step 42 train/validation token audit missing')
        for split,x in summary['splits'].items():
            require(x.get('rows',0)>0 and x.get('truncated_rows')==0 and x.get('tokenizer_unknown_rows')==0
                    and x.get('all_22_clauses_retained_rows')==x['rows'],f'{split}: failed Step 42 no-loss input audit')
        require(result['reference_settings'].get('encoder_settings_verified_for_current_cohort') is True,
                'Step 42 encoder provenance was not verified')
        require(result['comparison_design'].get('status')=='DRAFT_NOT_FROZEN',
                'Step 43 expects the revised comparison draft, not an old formal result')
        source=manifest.get('source_contract',{})
        pilot_source_contract(source)
        design=result['comparison_design']
        require(manifest.get('dataset_id')==source['dataset_id']
                and design.get('dataset_id')==source['dataset_id']
                and design.get('source_contract')==source,'Step 42 source contract differs between outputs')
        require(design.get('protocol_version')==AUDIT_PROTOCOL
                and design.get('encoder_conditions')==list(CONDITIONS)
                and design.get('sequence_length')==10
                and design.get('inputs',{}).get('observable_fields')==list(FIELDS),
                'Step 42 comparison inputs differ from this pilot')
        require(result['protocol'].get('observable_splits_read')==['train','val']
                and all(result['protocol'].get(k) is False for k in
                        ('raw_or_serialized_test_read','predictions_or_embeddings_read','observable_effect_sidecars_read')),
                'Step 42 input scope differs from this train/validation pilot')
        return result


def pilot_source_contract(source):
    """Project to actual pilot dependencies; optional Step 10/11/12 progress is irrelevant."""
    require(all(key in source for key in PILOT_SOURCE_KEYS),
            'Step 42 source contract lacks shared-nominal inputs or observability linkage')
    hashes=source.get('manifest_sha256',{})
    require(all(key in hashes for key in ('step7','step8','step9')),'Incomplete Step 42 source contract manifests')
    return {**{key:source[key] for key in PILOT_SOURCE_KEYS},
            'manifest_sha256':{key:hashes[key] for key in ('step7','step8','step9')}}


def mapped_path(text,root):
    parts=PureWindowsPath(text).parts if '\\' in text else Path(text).parts
    hits=[i for i,p in enumerate(parts) if p in ('dataset','outputs')]
    require(len(hits)==1 and '..' not in parts,'Unrecognized/ambiguous audited project path')
    path=root.joinpath(*parts[hits[0]:]).resolve()
    require(path.is_relative_to(root.resolve()),'Audited path escapes project')
    return path


def checked_file(path,expected,log):
    path=Path(path).resolve();guard_source_read(path)
    actual=sha256(path);require(actual==expected,f'Input hash mismatch: {path}')
    previous=[x for x in log if x['path']==str(path)]
    require(not previous or previous[0]['sha256']==actual,f'Input changed during pilot: {path}')
    if not previous:log.append(dict(path=str(path),sha256=actual,size_bytes=path.stat().st_size))


def load_inputs(root,audit_path,audit_script=None):
    root=Path(root).resolve();audit=read_audit(audit_path);log=[]
    helper,script=load_audit_module(root,audit,audit_script)
    build,inputs,inventory=helper.load_build(root,log)
    model_dir=root/helper.MODEL_REL
    r8,r9,report,config,mapping,contract=helper.load_encoder_contract(root,model_dir,build,inputs,log)
    source=audit['manifest'].get('source_contract',{})
    # 10/11 may have finished after the input audit. Only 7/8/9 determine these pilot inputs.
    for key in PILOT_SOURCE_KEYS:
        require(source.get(key)==contract[key],f'Current pipeline differs from selected Step 42: {key}')
    for step in ('step7','step8','step9'):
        require(source.get('manifest_sha256',{}).get(step)==contract['manifest_sha256'][step],f'Step 42 {step} source changed')
    frames={};counts={}
    for split in ('train','val'):
        relative=f'splits/{split}_observable.csv';p=root/helper.DATA_REL/relative
        df=helper.read_input(p,log,'csv',inputs[relative])
        counts[split]=helper.validate_frame(df,split,build['counts'][split])
        helper.verify_episode_rows(df,split,inventory)
        name=f'{split}_window_metadata.csv'
        md=helper.read_input(root/helper.EMBEDDING_REL/name,log,'csv',contract['step9_artifacts'][name])
        require(helper.validate_metadata(df,md,split)['rows_reordered'] is False,'Step 9 reordered current source rows')
        rel=f'serialized/{split}_serialized.csv'
        serialized=helper.read_input(model_dir.parent/rel,log,'csv',r8['artifacts'][rel])
        helper.verify_serialization(df,serialized,split)
        ix,truth,meta=sequence_view(df)
        require(len(ix)==build['counts'][split]['sequences_stride1']['10'],f'{split}: L10 count differs')
        require(set(truth)==set(range(6)),f'{split}: one or more classes have no L10 endpoint; a six-class pilot requires all classes')
        seen=audit['tokenization_summary']['splits'][split]
        require(seen['rows']==len(df) and audit['dataset_audit'][split]['episodes']==df.episode_id.nunique(),
                f'{split}: Step 42 population differs')
        frames[split]=df
    vocab=audit['train_field_vocabulary']
    require(vocab.get('source')=='train_only' and vocab.get('unknown_id')==0 and set(vocab.get('fields',{}))==set(FIELDS),'Vocabulary policy mismatch')
    for f in FIELDS:
        values=sorted(frames['train'][f].astype(str).str.strip().unique())
        require(vocab['fields'][f]=={value:i+1 for i,value in enumerate(values)},f'Train-only vocabulary changed: {f}')
    require(audit['tokenization_summary']['max_length']==mapping['max_length']
            and audit['tokenization_summary']['tokenizer_sha256']==contract['tokenizer_sha256'],'Token budget/identity changed')
    helper.local_tokenizer(model_dir,config,contract['tokenizer_sha256'])
    checked_file(script,audit['manifest']['script_sha256'],log)
    path=Path(audit_path)
    if path.is_file():checked_file(path,sha256(path),log)
    else:
        checked_file(path/'run_manifest.json',audit['manifest_sha256'],log)
        for name,digest in audit['manifest']['output_sha256'].items():checked_file(path/name,digest,log)
    audit['source_contract']=contract;audit['counts']=build['counts'];audit['validator_path']=str(script)
    audit['input_audit']=log
    return audit,frames,model_dir,vocab['fields'],log


def verify_base_weights(path):
    path=Path(path)
    require(sha256(path/'model.safetensors')==BASE_WEIGHT_SHA,
            'Base initializer hash mismatch. Use the pinned official pretrained base, never the fine-tuned Step 8 checkpoint.')


def resolve_base_model(path=None,offline=False):
    if path is None:
        from huggingface_hub import snapshot_download
        arguments=dict(repo_id=BASE_ID,revision=BASE_REVISION,
                       allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.txt'])
        try:path=Path(snapshot_download(**arguments,local_files_only=True))
        except Exception:
            if offline:raise ValueError('Pinned pretrained base is not cached. Use --base-model-dir with the official bundle, or --allow-download. The fine-tuned Step 8 model is not a valid initializer.')
            print('Pinned public base model not cached; downloading config/tokenizer/weights only (about 268 MB weights).',flush=True)
            path=Path(snapshot_download(**arguments,local_files_only=False))
    path=Path(path).resolve();verify_base_weights(path)
    c=read_json(path/'config.json')
    for k,v in dict(model_type='distilbert',dim=768,n_layers=6,n_heads=12,hidden_dim=3072,
                    vocab_size=30522,max_position_embeddings=512,dropout=.1,attention_dropout=.1,activation='gelu').items():
        require(c.get(k)==v,f'Unexpected base config {k}')
    return path


def make_protocol(candidate,bert_micro_batch,device,amp):
    require(candidate in ('main','alternate'),'Unknown pilot candidate')
    require(type(bert_micro_batch) is int and 1<=bert_micro_batch<=16,'BERT microbatch must be 1..16')
    return dict(version=PROTOCOL_VERSION,purpose='TRAIN_VAL_PILOT_ONLY',formal=False,test_evaluation=False,
                research_purpose='TAES reviewer revision: 22-field representation and cross-window evidence aggregation',
                nominal_substrate='shared_nominal_v2',
                label_policy='retain supplied episode labels, including quiet/unobserved attack windows',
                audit_annotation_policy='provenance only; never features, filtering, relabeling or model selection',
                candidate=candidate,conditions=list(CONDITIONS),encoder_seed=20260928,window_head_seed=20260929,
                encoder_shuffle_seed=20260930,tst_seed=20260610,tst_shuffle_seed=20260611,
                bert_micro_batch=bert_micro_batch,device=device,amp=amp,max_length=None,evaluation_precision="float32",extraction_precision="float32",
                embedding_dim=768,fields=list(FIELDS),labels=list(LABELS),tst_model=TST_MODEL,
                candidates={'main':[1e-3,3e-4,1e-4,2e-5],'alternate':[3e-4,1e-4,3e-4,5e-5]},
                encoder_max_epochs=[12,12,12,6],encoder_patience=4,encoder_min_epochs=3,
                encoder_plateau_delta=1e-4,
                selection='val macro_f1; equal within 1e-12 -> lower val loss',
                stop='encoder meaningful-F1 plateau; TST existing strict best metric patience',
                initializer=dict(repo=BASE_ID,revision=BASE_REVISION,weight_sha256=BASE_WEIGHT_SHA),
                limitations=['one pilot seed, not formal result','different model budgets: not equal-compute comparison',
                             'no equivalence claim from a small validation gap','cost timings include training/extraction overhead', 'random/pretrained pilot budgets differ; formal pretraining contrasts need a declared controlled protocol', 'TST architecture and hyperparameters follow Step 11, but this standalone L10 pilot uses separately recorded RNG/shuffle streams'])


def training_spec(condition,protocol,tst=False):
    if tst:
        return dict(max_epochs=30,min_epochs=1,patience=7,plateau_delta=0.,learning_rate=7e-5,
                    weight_decay=5e-4,batch_size=256,micro_batch_size=256,label_smoothing=.03,
                    warmup_ratio=0.,scheduler='plateau',eval_batch_size=256,seed=protocol['tst_seed'],
                    shuffle_seed=protocol['tst_shuffle_seed'],amp=protocol['amp'])
    i=CONDITIONS.index(condition);bert=i>=2
    return dict(max_epochs=protocol['encoder_max_epochs'][i],min_epochs=protocol['encoder_min_epochs'],
                patience=protocol['encoder_patience'],plateau_delta=protocol['encoder_plateau_delta'],
                learning_rate=protocol['candidates'][protocol['candidate']][i],weight_decay=.01,
                batch_size=16 if bert else 256,micro_batch_size=protocol['bert_micro_batch'] if bert else 256,
                label_smoothing=0.,warmup_ratio=.06,scheduler='linear',eval_batch_size=protocol['bert_micro_batch'] if bert else 512,
                seed=protocol['encoder_seed'],shuffle_seed=protocol['encoder_shuffle_seed'],amp=protocol['amp'])


def tokenizers_for_run(local_dir,base_dir,expected_hash=None):
    from transformers import AutoTokenizer
    kwargs=dict(local_files_only=True,trust_remote_code=False,use_fast=True)
    tok=AutoTokenizer.from_pretrained(str(local_dir),**kwargs)
    base=AutoTokenizer.from_pretrained(str(base_dir),**kwargs)
    def fingerprint(t):
        backend=json.loads(t.backend_tokenizer.to_str());backend.pop('padding',None);backend.pop('truncation',None)
        return object_hash({'backend':backend,'special_tokens_map':t.special_tokens_map,'model_input_names':t.model_input_names})
    require(tok.is_fast and base.is_fast,'Fast tokenizers required')
    require(fingerprint(tok)==fingerprint(base),'Saved and initializer tokenizer behavior differs (not just vocabulary)')
    require(expected_hash is None or fingerprint(tok)==expected_hash,'Audited tokenizer fingerprint differs')
    require(tok.padding_side=='right' and tok.pad_token_id is not None and tok.unk_token_id is not None,'Expected right-padded tokenizer with pad/UNK IDs')
    return tok


def prepare_cache(frames,vocab,tokenizer,out,identity,max_length):
    require(type(max_length) is int and max_length>2,'Invalid audited token budget')
    require(set(frames)=={'train','val'},'Cache accepts train/validation only')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);meta_path=out/'complete.json';started=time.perf_counter()
    if meta_path.exists():
        meta=read_json(meta_path);require(meta['identity']==identity and meta.get('max_length')==max_length,'Preprocessing identity/length changed')
        require(set(meta.get('artifact_sha256',{}))==cache_artifact_names(),'Cache artifact inventory is incomplete or unexpected')
        for name,digest in meta['artifact_sha256'].items():require(sha256(safe_output(out,name))==digest,f'Cache hash mismatch {name}')
    else:
        for split,df in frames.items():
            count=len(df)
            ids=np.lib.format.open_memmap(out/f'{split}_input_ids.npy',mode='w+',dtype=np.int32,shape=(count,max_length))
            mask=np.lib.format.open_memmap(out/f'{split}_attention_mask.npy',mode='w+',dtype=np.int8,shape=(count,max_length))
            ids[:]=tokenizer.pad_token_id;mask[:]=0
            for start in range(0,count,1024):
                chunk=df.iloc[start:start+1024];texts=serialize_frame(chunk)
                encoded=tokenizer(texts,add_special_tokens=True,truncation=False,padding=False)['input_ids']
                require(len(encoded)==len(chunk),'Tokenizer changed the input row count')
                for j,row in enumerate(encoded):
                    require(2<len(row)<=max_length,'Empty token sequence or truncation risk; stop instead of dropping fields')
                    require(tokenizer.unk_token_id not in row,'Unexpected tokenizer UNK after the reviewed audit')
                    ids[start+j,:len(row)]=row;mask[start+j,:len(row)]=1
                if start%20480==0:print(f'  Tokenize {split}: {min(start+len(chunk),count):,}/{count:,}',flush=True)
            ids.flush();mask.flush();del ids,mask
            np.save(out/f'{split}_categories.npy',encode_fields(df,vocab),allow_pickle=False)
            np.save(out/f'{split}_labels.npy',df.label.map(LABEL_TO_ID).to_numpy(dtype=np.int64),allow_pickle=False)
        meta=dict(identity=identity,max_length=max_length,elapsed_seconds=time.perf_counter()-started,
                  artifact_sha256={name:sha256(out/name) for name in sorted(cache_artifact_names())})
        write_json(meta_path,meta)
    data={}
    for split,df in frames.items():
        entry={name:np.load(out/f'{split}_{name}.npy',mmap_mode='r',allow_pickle=False)
               for name in ('input_ids','attention_mask','categories','labels')}
        validate_cache_arrays(entry,df,vocab,tokenizer,max_length)
        entry['sequences'],entry['sequence_labels'],entry['metadata']=sequence_view(df)
        data[split]=entry
    return data


def cache_artifact_names():
    return {f'{split}_{name}.npy' for split in ('train','val')
            for name in ('input_ids','attention_mask','categories','labels')}


def validate_cache_arrays(entry,df,vocab,tokenizer,max_length):
    expected={'input_ids':((len(df),max_length),np.dtype('int32')),
              'attention_mask':((len(df),max_length),np.dtype('int8')),
              'categories':((len(df),len(FIELDS)),np.dtype('int64')),
              'labels':((len(df),),np.dtype('int64'))}
    for name,(shape,dtype) in expected.items():
        require(entry[name].shape==shape and entry[name].dtype==dtype,f'Cache shape/dtype mismatch: {name}')
    require(np.array_equal(entry['labels'],df.label.map(LABEL_TO_ID).to_numpy(dtype=np.int64)),
            'Cache labels do not match current source rows')
    require(np.array_equal(entry['categories'],encode_fields(df,vocab)),'Cache categories do not match train-only vocabulary/source rows')
    for start in range(0,len(df),4096):
        ids=entry['input_ids'][start:start+4096];mask=entry['attention_mask'][start:start+4096]
        lengths=mask.sum(axis=1)
        require(((mask==0)|(mask==1)).all() and (lengths>2).all()
                and np.array_equal(mask,np.arange(max_length)[None,:]<lengths[:,None]),'Cache attention masks must be nonempty right-padded sequences')
        require((ids>=0).all() and (ids[mask==0]==tokenizer.pad_token_id).all()
                and not (ids[mask==1]==tokenizer.unk_token_id).any(),'Cache token IDs/padding/UNK mismatch')


def extract_features(model,batches,out,batch_size,device,identity):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);record=out/'complete.json';model.to(device)
    if record.exists():
        meta=read_json(record);require(meta['identity']==identity,'Embedding checkpoint identity changed')
        require(set(meta.get('sha256',{}))==set(batches)=={'train','val'}
                and meta.get('rows')=={s:len(b.labels) for s,b in batches.items()}
                and meta.get('precision')=='float32','Embedding split/row/precision contract differs')
        for s,digest in meta['sha256'].items():require(sha256(out/f'{s}.npy')==digest,'Embedding hash mismatch')
    else:
        start=time.perf_counter()
        with evaluating(model):
            for split,b in batches.items():
                path=out/f'{split}.npy';e=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(b.labels),768))
                for first in range(0,len(b.labels),batch_size):
                    rows=np.arange(first,min(first+batch_size,len(b.labels)))
                    args,_=b.tensors(rows,device);features=model.features(**args).float().cpu().numpy()
                    require(features.shape==(len(rows),768) and np.isfinite(features).all(),'Invalid encoder embeddings')
                    e[rows]=features
                e.flush();del e
        meta=dict(identity=identity,precision='float32',elapsed_seconds=time.perf_counter()-start,
                  rows={s:len(b.labels) for s,b in batches.items()},sha256={s:sha256(out/f'{s}.npy') for s in batches})
        write_json(record,meta)
    arrays={s:np.load(out/f'{s}.npy',mmap_mode='r',allow_pickle=False) for s in batches}
    for split,array in arrays.items():
        require(array.shape==(len(batches[split].labels),768) and array.dtype==np.dtype('float32'),
                f'{split}: embedding shape/dtype mismatch')
    return arrays


def run_pipeline(data,sizes,base_dir,protocol,out,identity,device):
    from transformers import DistilBertConfig
    out=Path(out);out.mkdir(parents=True,exist_ok=True);summary=[]
    for i,condition in enumerate(CONDITIONS):
        print(f'\n[{i+1}/4] {condition} | {protocol["candidate"]}',flush=True)
        seed_all(protocol['encoder_seed'])
        if i==0:encoder=CategoricalEncoder(sizes)
        elif i==1:encoder=FieldTokenEncoder(sizes)
        else:
            config=DistilBertConfig.from_pretrained(str(base_dir),local_files_only=True)
            config._attn_implementation='eager'
            encoder=BertEncoder(config,Path(base_dir)/'model.safetensors' if i==3 else None)
        model=WindowModel(encoder,head_seed=protocol['window_head_seed'])
        encoder_params=sum(p.numel() for p in encoder.parameters())
        cfg=training_spec(condition,protocol);folder=out/condition
        batches={s:ArrayBatches(({'input_ids':x['input_ids'],'attention_mask':x['attention_mask']} if i>=2 else {'x':x['categories']}),x['labels']) for s,x in data.items()}
        stage_identity=object_hash({'run':identity,'condition':condition,'stage':'encoder'})
        trained=fit(model,batches['train'],batches['val'],cfg,folder/'encoder',stage_identity,device)
        window_logits=np.load(folder/'encoder/val_logits.npy',allow_pickle=False)
        require(window_logits.shape==(len(data['val']['labels']),6),'Saved window prediction shape differs')
        ends=data['val']['sequences'][:,-1];common_logits=window_logits[ends]
        common_y=data['val']['sequence_labels']
        loss=float(nn.functional.cross_entropy(torch.from_numpy(common_logits),torch.from_numpy(common_y)))
        window_common=metrics(common_y,common_logits,loss)
        prediction_rows(data['val']['metadata'],common_y,common_logits,identity,condition,protocol,'common_L10_endpoints',head_kind='window').to_csv(
            folder/'val_window_predictions_common_L10.csv',index=False,encoding='utf-8-sig')
        write_json(folder/'window_common_L10_metrics.json',window_common)
        feature_identity=object_hash({'stage':stage_identity,'checkpoint':trained['artifact_sha256']['best.pt']})
        embeddings=extract_features(model,batches,folder/'embeddings',cfg['eval_batch_size'],device,feature_identity)
        del model,encoder
        if device.type=='cuda':torch.cuda.empty_cache()
        tb={s:SequenceBatches(embeddings[s],x['sequences'],x['sequence_labels']) for s,x in data.items()}
        seed_all(protocol['tst_seed']);tst=TST()
        temporal_identity=object_hash({'run':identity,'condition':condition,'embeddings':read_json(folder/'embeddings/complete.json')['sha256']})
        temporal=fit(tst,tb['train'],tb['val'],training_spec(condition,protocol,tst=True),folder/'tst',temporal_identity,device)
        logits=np.load(folder/'tst/val_logits.npy',allow_pickle=False)
        require(logits.shape==common_logits.shape,'Window and TST common endpoint populations differ')
        prediction_rows(data['val']['metadata'],common_y,logits,identity,condition,protocol,'common_L10_endpoints').to_csv(
            folder/'val_tst_predictions_L10.csv',index=False,encoding='utf-8-sig')
        summary.append(dict(condition=condition,candidate=protocol['candidate'],dataset_id=protocol['source_contract']['dataset_id'],
            step43_protocol_identity=identity,encoder_seed=protocol['encoder_seed'],tst_seed=protocol['tst_seed'],
            encoder_parameters=encoder_params,window_model_parameters=trained['parameters'],tst_parameters=temporal['parameters'],
            window_val_macro_f1=trained['best_metrics']['macro_f1'],window_val_count=trained['best_metrics']['n'],
            window_val_common_macro_f1=window_common['macro_f1'],common_val_count=window_common['n'],
            l10_val_macro_f1=temporal['best_metrics']['macro_f1'],l10_val_count=temporal['best_metrics']['n'],
            temporal_delta_val_common_macro_f1=temporal['best_metrics']['macro_f1']-window_common['macro_f1'],
            encoder_best_epoch=trained['best_epoch'],encoder_epochs=trained['epochs_completed'],encoder_convergence=trained['convergence'],
            tst_best_epoch=temporal['best_epoch'],tst_epochs=temporal['epochs_completed'],tst_convergence=temporal['convergence'],
            encoder_training_seconds=trained['training_seconds'],tst_training_seconds=temporal['training_seconds'],
            peak_gpu_bytes=max(trained['peak_gpu_bytes'],temporal['peak_gpu_bytes']),evaluation_precision='float32',formal_experiment=False))
        write_csv(out/'pilot_summary.csv',summary);write_json(out/'pilot_summary.json',summary)
        del tst,tb,embeddings
        if device.type=='cuda':torch.cuda.empty_cache()
    return summary


class RunLock:
    """OS advisory lock automatically releases on process exit, including hard interruption."""
    def __init__(self,path):self.path=Path(path);self.file=None
    def __enter__(self):
        self.file=self.path.open('a+b')
        if self.file.tell()==0:self.file.write(b'0');self.file.flush()
        self.file.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.file.close();raise ValueError('This run is already active. Do not launch a second process on the same run.')
        return self
    def __exit__(self,*_):
        self.file.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(self.file.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(self.file.fileno(),fcntl.LOCK_UN)
        self.file.close()


def finish_run(out,manifest):
    small=[p for p in sorted(out.rglob('*')) if p.is_file() and p.suffix in ('.json','.csv','.md','.txt') and p.name!='run_manifest.json']
    manifest['finished_at_utc']=utc()
    manifest['output_sha256']={p.relative_to(out).as_posix():sha256(p) for p in small}
    archive=out.with_suffix('.zip');manifest['review_zip']=str(archive)
    write_json(out/'run_manifest.json',manifest)
    temp=archive.with_name(archive.name+'.tmp')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in small+[out/'run_manifest.json']:z.write(p,p.relative_to(out).as_posix())
    temp.replace(archive)
    print(f'\n{manifest["status"]}\nReadiness: {manifest.get("readiness","REVIEW_REQUIRED")}\nResults: {out}\nReview ZIP: {archive}',flush=True)


def execute_run(root,out,args,manifest,prepared=None):
    ctx=prepared if prepared is not None else preflight_run(root,out,args)
    audit,protocol,identity=ctx['audit'],ctx['protocol'],ctx['identity']
    if not (out/'protocol.json').exists():write_json(out/'protocol.json',dict(identity=identity,**ctx['payload']))
    manifest.update(protocol_identity=identity,dataset_id=protocol['source_contract']['dataset_id'],
        source_contract=protocol['source_contract'],runtime=ctx['runtime'],initializer_path=str(ctx['base']),audit_path=str(ctx['audit_path']))
    write_json(out/'run_manifest.json',manifest)
    write_json(out/'input_audit.json',ctx['inputs'])
    write_json(out/'train_field_vocabulary.json',dict(source='train_only',unknown_id=0,fields=ctx['vocab']))
    write_json(out/'reference_settings.json',audit['reference_settings'])
    print(f'Device: {ctx["runtime"]["device"]}; candidate: {protocol["candidate"]}; pilot seed: {protocol["encoder_seed"]}',flush=True)
    data=prepare_cache(ctx['frames'],ctx['vocab'],ctx['tokenizer'],out/'cache',identity,max_length=protocol['max_length'])
    summary=run_pipeline(data,[len(ctx['vocab'][f]) for f in FIELDS],ctx['base'],protocol,out,identity,ctx['device'])
    # Long runs must finish on exactly the read inputs and initializer with which they began.
    for item in ctx['inputs']:checked_file(item['path'],item['sha256'],[])
    for name,digest in ctx['payload']['base_hashes'].items():require(sha256(ctx['base']/name)==digest,'Initializer changed during pilot')
    require(sha256(Path(__file__))==ctx['payload']['script_sha256'],'Pilot source code changed during training')
    notes={'status':'TRAIN_VAL_PILOT_ONLY','formal':False,'encoder_seeds':1,'test_evaluation':False,
        'matched_comparison':'window_val_common_macro_f1 and l10_val_macro_f1 use the same L10 endpoints',
        'all_window_metric':'window_val_macro_f1 uses all available validation windows and selects the encoder checkpoint',
        'confounding':'Window classifier and TST also differ in head capacity/training; matched endpoints alone do not isolate a pure temporal effect.',
        'pretraining_limit':'Random/pretrained pilots retain different learning-rate/epoch budgets to diagnose convergence; these pilot scores alone are not a controlled causal pretraining comparison.',
        'uncertainty':'No formal seed uncertainty, bootstrap significance or equivalence claim from this pilot.',
        'shared_nominal_scope':'All conditions use the same current shared-nominal train/validation episodes and 22 fields; metadata and counterfactual audit annotations are not features.',
        'quiet_windows':'Episode/scenario labels are retained even without observable evidence. This pilot does not filter/relabel them or tune to an expected score.',
        'evidence_limit':'Overall validation F1 alone does not establish attack-evidence use or rule out generator signatures; interpret alongside the current benchmark audits and evidence-availability diagnostics.',
        'cost_limit':'Timing includes validation/checkpoint overhead; it is not an onboard inference benchmark.',
        'next_use':'Use train/validation convergence evidence to specify the formal encoder comparison before test evaluation.'}
    write_json(out/'interpretation.json',notes)
    manifest.update(status='PILOT_COMPLETE',readiness='REVIEW_REQUIRED',conditions_completed=len(summary),formal_experiment=False,
                    validation_population='common L10 endpoints for matched window/TST comparison',test_data_read=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root',type=Path,help='Default: script directory; inherited on resume')
    p.add_argument('--resume',type=Path,help='Existing revised 43 run; resumes last completed epoch')
    p.add_argument('--audit-run',type=Path,help='Completed revised 42 folder or ZIP; auto-select only equivalent current-dataset audits')
    p.add_argument('--audit-script',type=Path,help='Exact revised 42 Python file used to create the selected audit')
    p.add_argument('--base-model-dir',type=Path,help='Pinned official base initializer; never the fine-tuned Step 8 model')
    network=p.add_mutually_exclusive_group()
    network.add_argument('--offline',action='store_true',default=None,help='Use local/cache only (default); inherited on resume')
    network.add_argument('--allow-download',dest='offline',action='store_false',help='Allow pinned public base download if absent')
    p.add_argument('--candidate',choices=['main','alternate'])
    p.add_argument('--device',choices=['auto','cuda','cpu'])
    p.add_argument('--bert-micro-batch',type=int,help='1..16; effective BERT training batch stays 16; recorded for strict resume')
    amp=p.add_mutually_exclusive_group();amp.add_argument('--amp',dest='amp',action='store_true');amp.add_argument('--no-amp',dest='amp',action='store_false');p.set_defaults(amp=None)
    args=p.parse_args(argv);root=(args.project_root or Path(__file__).resolve().parent).resolve();out=None;old={}
    names=('project_root','audit_run','audit_script','base_model_dir','offline','candidate','device','bert_micro_batch','amp')
    paths=('project_root','audit_run','audit_script','base_model_dir')
    try:
        if args.resume:
            out=(root/args.resume).resolve() if not args.resume.is_absolute() else args.resume.resolve()
            require(out.is_dir() and (out/'run_manifest.json').is_file(),'Resume must identify an existing 43 run directory')
            old=read_json(out/'run_manifest.json')
            require(old.get('script_number')==43 and old.get('protocol_version')==PROTOCOL_VERSION,
                    'This is not a revised 43 pilot; old experiments remain separate')
            for name,value in old.get('invocation',{}).items():
                if name in names and getattr(args,name) is None:setattr(args,name,Path(value) if name in paths and value is not None else value)
            root=(args.project_root or root).resolve()
        require(root.is_dir(),'Project root does not exist');args.project_root=root
        args.offline=True if args.offline is None else args.offline
        for name in paths[1:]:
            value=getattr(args,name)
            if value is not None:setattr(args,name,(root/value).resolve() if not value.is_absolute() else value.resolve())
        if out is None:out=root/OUTPUT_REL/('run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'));out.mkdir(parents=True,exist_ok=False)
        with RunLock(out/'.run.lock'):
            # Preflight rejects altered resumes without rewriting an existing completion/failed record.
            ctx=preflight_run(root,out,args)
            if old.get('status')=='PILOT_COMPLETE':
                verify_complete_run(out,old);print(f'Verified completed pilot; no retraining: {out}',flush=True);return 0
            args.audit_run=ctx['audit_path'];args.audit_script=Path(ctx['audit']['validator_path']);args.base_model_dir=ctx['base']
            invocation={name:(str(getattr(args,name)) if isinstance(getattr(args,name),Path) else getattr(args,name)) for name in names}
            manifest=dict(old)
            for key in ('error_type','error','finished_at_utc','output_sha256','review_zip'):manifest.pop(key,None)
            manifest.update(script_number=43,protocol_version=PROTOCOL_VERSION,status='RUNNING',
                started_at_utc=old.get('started_at_utc',utc()),last_started_at_utc=utc(),script_sha256=sha256(Path(__file__)),
                invocation=invocation,test_data_read=False,formal_experiment=False,encoder_pilot_seed=ctx['protocol']['encoder_seed'])
            if args.resume:manifest['previous_status']=old['status']
            write_json(out/'run_manifest.json',manifest)
            print(f'Step 43: four-encoder train/validation pilot.\nOutput: {out}\nNo test evaluation; keep the entire run directory.',flush=True)
            try:execute_run(root,out,args,manifest,ctx)
            except BaseException as exc:
                manifest.update(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',readiness='REVIEW_REQUIRED',error_type=type(exc).__name__,error=str(exc))
                (out/'failure_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
                finish_run(out,manifest)
                print(f'Resume: python "{Path(__file__).name}" --resume "{out}"',flush=True)
                return 130 if isinstance(exc,KeyboardInterrupt) else 1
            else:
                (out/'failure_traceback.txt').unlink(missing_ok=True);finish_run(out,manifest);return 0
    except Exception as exc:
        print(f'Pilot stopped before execution: {exc}',file=sys.stderr)
        return 1




def resolve_audit(root,explicit=None):
    if explicit is not None:
        path=Path(explicit);return path.resolve() if path.is_absolute() else (root/path).resolve()
    root=Path(root).resolve()
    build=read_json(root/DATA_REL/'build_manifest.json')
    candidates=[]
    for path in sorted((root/AUDIT_REL).glob('run_*/run_manifest.json')):
        record=read_json(path)
        if (record.get('status')=='AUDIT_COMPLETE' and record.get('dataset_id')==build.get('dataset_id')
                and record.get('protocol_version')==AUDIT_PROTOCOL):
            candidates.append((path.parent,record))
    require(candidates,'No completed revised Step 42 for the current dataset. Run 42 first or pass --audit-run.')
    # Equivalent audits may be repeated without choosing on scores or file modification time.
    signatures={object_hash({'source':pilot_source_contract(x.get('source_contract',{})),'script':x.get('script_sha256'),
                             'protocol':x.get('output_sha256',{}).get('protocol.json')}) for _,x in candidates}
    require(len(signatures)==1,'Multiple different Step 42 audits match this dataset. Specify the intended --audit-run; no latest/best-score selection is made.')
    return candidates[0][0]


def load_audit_module(root,audit,explicit=None):
    """Reuse the exact 42 validators, authenticated by the completed audit's code hash."""
    import importlib.util
    wanted=audit['manifest'].get('script_sha256')
    paths=[Path(explicit).resolve()] if explicit else sorted(root.glob('42*.py'))
    matching=[p for p in paths if p.is_file() and sha256(p)==wanted]
    require(matching,'Put the revised Step 42 Python file used for this audit beside script 43, or pass --audit-script. Its bytes must match the completed audit.')
    path=matching[0];spec=importlib.util.spec_from_file_location('_step42_pilot_input_checks',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    require(module.PROTOCOL==AUDIT_PROTOCOL,'Unexpected Step 42 validator version')
    return module,path


def guard_source_read(path):
    path=Path(path);name=path.name.lower()
    require(not name.startswith('test_') and path.stem.lower()!='test'
            and 'predictions' not in [part.lower() for part in path.parts],
            f'Test/prediction input is outside this train/validation pilot: {path}')


def save_rng_state():
    state=np.random.get_state()
    return {'python_rng':random.getstate(),'numpy_rng':[state[0],state[1].tolist(),int(state[2]),int(state[3]),float(state[4])]}


def restore_rng_state(checkpoint):
    random.setstate(checkpoint['python_rng'])
    state=checkpoint['numpy_rng']
    np.random.set_state((state[0],np.asarray(state[1],dtype=np.uint32),state[2],state[3],state[4]))


def prediction_rows(metadata,truth,logits,identity,condition,protocol,population,head_kind='tst'):
    require(head_kind in ('window','tst'),'Unknown prediction head kind')
    require(len(metadata)==len(truth)==len(logits),'Prediction/endpoint row count mismatch')
    preds=metadata.copy();preds['y_true']=truth;preds['y_pred']=logits.argmax(axis=1)
    probability=np.exp(logits-logits.max(axis=1,keepdims=True));probability/=probability.sum(axis=1,keepdims=True)
    for k in range(6):preds[f'prob_{k}']=probability[:,k]
    preds['dataset_id']=protocol['source_contract']['dataset_id'];preds['step43_protocol_identity']=identity
    preds['condition']=condition;preds['encoder_seed']=protocol['encoder_seed']
    preds['head_seed']=protocol['window_head_seed'] if head_kind=='window' else protocol['tst_seed']
    preds['head_kind']=head_kind
    preds['candidate']=protocol['candidate'];preds['evaluation_population']=population
    return preds


def preflight_run(root,out,args):
    """Read-only with respect to a prior pilot; reject changed resumes before mutating it."""
    if DEPENDENCY_ERROR:raise RuntimeError('Use the existing training environment. Dependency error: '+DEPENDENCY_ERROR)
    import transformers
    existing=read_json(out/'protocol.json') if (out/'protocol.json').is_file() else None
    prior=existing['protocol'] if existing else {}
    audit_path=resolve_audit(root,args.audit_run)
    audit,frames,tokenizer_dir,vocab,inputs=load_inputs(root,audit_path,args.audit_script)
    device_name=args.device or prior.get('device','auto')
    if device_name=='auto':
        require(torch.cuda.is_available(),'CUDA unavailable. To intentionally run the pilot on CPU use --device cpu.')
        device_name='cuda'
    require(device_name!='cuda' or torch.cuda.is_available(),'Requested CUDA is unavailable')
    device=torch.device(device_name)
    amp=(prior.get('amp',True) if args.amp is None else args.amp) and device.type=='cuda'
    protocol=make_protocol(args.candidate or prior.get('candidate','main'),
        args.bert_micro_batch if args.bert_micro_batch is not None else prior.get('bert_micro_batch',16),device_name,amp)
    protocol.update(max_length=audit['tokenization_summary']['max_length'],source_contract=audit['source_contract'],
                    split_counts={s:audit['counts'][s] for s in ('train','val')})
    base=resolve_base_model(args.base_model_dir,args.offline)
    base_hashes={p.name:sha256(p) for p in sorted(base.iterdir()) if p.is_file() and p.name in
                 ('config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.txt','special_tokens_map.json','added_tokens.json')}
    tokenizer=tokenizers_for_run(tokenizer_dir,base,audit['source_contract']['tokenizer_sha256'])
    runtime=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,pandas=pd.__version__,
        transformers=transformers.__version__,platform=platform.platform(),device=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
        cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),torch_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(),cuda_device_count=torch.cuda.device_count() if device.type=='cuda' else 0,
        numerical_policy=numerical_runtime())
    # Audit read timestamps do not enter identity. Paths and hashes of actual read inputs do.
    source_files=[{k:x[k] for k in ('path','sha256','size_bytes')} for x in inputs]
    payload=dict(protocol=protocol,runtime=runtime,source_files=source_files,audit_manifest_sha256=audit['manifest_sha256'],
                 script_sha256=sha256(Path(__file__)),base_hashes=base_hashes)
    identity=object_hash(payload)
    if existing:require(existing.get('identity')==identity and existing==dict(identity=identity,**payload),
        'Resume inputs/configuration/code/runtime changed. Existing results are preserved; use a new run.')
    return dict(audit=audit,frames=frames,vocab=vocab,inputs=inputs,base=base,tokenizer=tokenizer,
                runtime=runtime,protocol=protocol,payload=payload,identity=identity,device=device,audit_path=audit_path)


def numerical_runtime():
    """Record effective arithmetic flags so changed settings cannot silently resume.

    Apply the same cuDNN policy already used by seed_all; preserve the user's
    other backend defaults and bind their effective values to the run identity.
    This records reproducibility conditions, not a cross-hardware bitwise guarantee.
    """
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    cuda=torch.backends.cuda
    return dict(cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
        cuda_matmul_allow_tf32=bool(cuda.matmul.allow_tf32),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        fp16_reduced_precision_reduction=getattr(cuda.matmul,'allow_fp16_reduced_precision_reduction',None),
        bf16_reduced_precision_reduction=getattr(cuda.matmul,'allow_bf16_reduced_precision_reduction',None),
        flash_sdp_enabled=cuda.flash_sdp_enabled(),mem_efficient_sdp_enabled=cuda.mem_efficient_sdp_enabled(),
        math_sdp_enabled=cuda.math_sdp_enabled(),
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        nvidia_tf32_override=os.environ.get('NVIDIA_TF32_OVERRIDE'))


def verify_stage_artifacts(folder,record):
    hashes=record.get('artifact_sha256',{})
    require(set(hashes)=={'best.pt','last.pt','history.csv','val_logits.npy'},
            'Completed training artifact inventory is incomplete or unexpected')
    for name,digest in hashes.items():
        require(sha256(safe_output(folder,name))==digest,f'Completed artifact changed: {name}')


def verify_complete_run(out,manifest):
    require(manifest.get('status')=='PILOT_COMPLETE' and manifest.get('conditions_completed')==4,'Incomplete pilot')
    required={'protocol.json','input_audit.json','train_field_vocabulary.json','reference_settings.json',
              'pilot_summary.csv','pilot_summary.json','interpretation.json','cache/complete.json'}
    for condition in CONDITIONS:
        required.update(f'{condition}/{name}' for name in ('encoder/complete.json','tst/complete.json','embeddings/complete.json',
            'val_window_predictions_common_L10.csv','val_tst_predictions_L10.csv','window_common_L10_metrics.json'))
    require(required<=set(manifest.get('output_sha256',{})),'Completed pilot output inventory is incomplete')
    for name,digest in manifest.get('output_sha256',{}).items():
        path=safe_output(out,name);require(sha256(path)==digest,f'Completed pilot output changed: {name}')
    for condition in CONDITIONS:
        for stage in ('encoder','tst'):
            folder=out/condition/stage;record=read_json(folder/'complete.json')
            verify_stage_artifacts(folder,record)
        folder=out/condition/'embeddings';record=read_json(folder/'complete.json')
        require(set(record.get('sha256',{}))=={'train','val'},'Completed embedding inventory is incomplete')
        for split,digest in record['sha256'].items():
            require(split in ('train','val'),'Unexpected embedding split')
            require(sha256(folder/f'{split}.npy')==digest,f'Completed {condition} embeddings changed')
    record=read_json(out/'cache/complete.json')
    require(set(record.get('artifact_sha256',{}))==cache_artifact_names(),'Cache artifact inventory is incomplete or unexpected')
    for name,digest in record['artifact_sha256'].items():require(sha256(safe_output(out/'cache',name))==digest,f'Completed cache changed: {name}')


def safe_output(root,name):
    require(isinstance(name,str) and not PureWindowsPath(name).drive and '\\' not in name,'Invalid artifact path')
    path=(Path(root)/name).resolve()
    require(not Path(name).is_absolute() and '..' not in Path(name).parts and path.is_relative_to(Path(root).resolve()),'Artifact path escapes run')
    return path

if __name__=='__main__':raise SystemExit(main())
