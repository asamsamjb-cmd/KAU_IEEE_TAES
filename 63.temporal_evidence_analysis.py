#!/usr/bin/env python3
"""Step63: audit-only decomposition of Step62's paired temporal information control.

Run from the existing project (NumPy + pandas; no GPU, training or new inference):
    python "63.temporal_evidence_analysis.py"
    python "63.temporal_evidence_analysis.py" --preflight-only

Step64 imports this file's authenticated reader; keep both files together.
The original 62 outputs and dataset are read-only. New timestamped output folders
include all analysis tables, protocol, hash manifest, and a review ZIP.

This is a post-hoc mechanistic analysis of previously inspected test predictions,
not a new independent confirmatory experiment. Counterfactual changes identify
changes in the 22-field projection, NOT guaranteed class-identifying evidence.
Neither fine temporal order, BERT necessity, nor TST > LSTM is tested here.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path, PurePosixPath
import shutil
import sys
import traceback
import zipfile

import numpy as np
import pandas as pd

READER_API = 'step62_followup_reader_v1'
VERSION = 'shared_nominal_temporal_evidence_analysis_v1'
DATASET_ID = '478f782ceb970d7a2de73319ccc4e929be06cfea684a054aa72d932ed65ae83b'
STEP62_SHA = 'b5095475f0ff9aedd0b9797b24e39636041c0d20e914cf2d7a98fd9eab56cb81'
SEEDS = tuple(range(20260610,20260620))
CONDITIONS = ('actual_history','repeat_current')
LABELS = ('NORMAL','RF_LINK_JAMMING','GNSS_SPOOFING','UNAUTHORIZED_TELECOMMAND_INJECTION',
          'TELECOMMAND_REPLAY','GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE')
META = ('scenario_id','episode_id','episode_type','episode_step','window_id','timestamp','label')
AUDIT_FIELDS = ('audit_observable_effect_present','audit_observable_difference_count',
                'audit_persistent_effect_present','audit_attack_active')
DEFAULT_RUN = Path('outputs/shared_nominal_v2/62_temporal_information_control/run_information_control_v1')

def require(condition, message):
    if not condition:
        raise ValueError(message)

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def obj_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def json_safe(value):
    if isinstance(value,dict): return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_safe(v) for v in value]
    if isinstance(value,np.ndarray): return json_safe(value.tolist())
    if isinstance(value,np.generic): return json_safe(value.item())
    if isinstance(value,float) and not np.isfinite(value): return None
    if isinstance(value,Path): return str(value)
    return value

def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(json_safe(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)

def write_csv(path,frame):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False,encoding='utf-8-sig',float_format='%.17g')

def safe_child(base,relative):
    require(isinstance(relative,str) and '\\' not in relative and ':' not in relative,'Unsafe relative path: '+str(relative))
    p=PurePosixPath(relative)
    require(not p.is_absolute() and '..' not in p.parts and relative!='','Unsafe relative path: '+relative)
    result=(Path(base)/relative).resolve()
    require(result.is_relative_to(Path(base).resolve()),'Path escapes source directory')
    return result

def verify_file(path,expected,audit):
    path=Path(path)
    require(path.is_file(),'Required file missing: '+str(path))
    got=digest(path)
    require(got==expected,'Source SHA256 mismatch (do not bypass): '+str(path))
    audit[str(path.resolve())]=got

def create_output(root,number,name):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out=Path(root)/'outputs/shared_nominal_v2'/f'{number}_{name}'/('run_'+stamp)
    out.mkdir(parents=True,exist_ok=False)
    return out

def finish_output(out,status,details=None):
    out=Path(out)
    hashes={p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*'))
            if p.is_file() and p.name!='run_manifest.json' and p.suffix!='.zip'}
    write_json(out/'run_manifest.json',dict(status=status,completed_at_utc=utc_now(),
        output_sha256=hashes,details=details or {},training=False,new_inference=False))
    archive=out.parent/(out.name+('_FAILED' if 'FAILED' in status else '')+'_review.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(out).as_posix())
    return archive

def add_common_arguments(parser):
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--step62-run',type=Path,default=None,help='Explicit completed Step62 run directory (not ZIP).')
    parser.add_argument('--dataset-dir',type=Path,default=None)
    parser.add_argument('--bootstrap-draws',type=int,default=20000)
    parser.add_argument('--bootstrap-seed',type=int,default=2026100763)
    parser.add_argument('--preflight-only',action='store_true')

def _resolve(root,path,default):
    p=Path(path) if path is not None else Path(default)
    return p.resolve() if p.is_absolute() else (root/p).resolve()

def verify_step62(run):
    """Verify review artifacts only: model weights/embeddings need not be loaded."""
    run=Path(run).resolve(); files={}
    require((run/'run_manifest.json').is_file(),'Completed Step62 not found: '+str(run))
    manifest=read_json(run/'run_manifest.json'); files[str(run/'run_manifest.json')]=digest(run/'run_manifest.json')
    require(manifest.get('status')=='TEMPORAL_INFORMATION_CONTROL_COMPLETE','Step62 is not complete')
    require(manifest.get('test_used_for_selection') is False,'Step62 test-selection contract differs')
    hashes=manifest['output_sha256']
    require('protocol.json' in hashes and 'trainval_gate.json' in hashes,'Step62 missing protocol/gate hashes')
    for name,expected in hashes.items(): verify_file(safe_child(run,name),expected,files)
    p=read_json(run/'protocol.json')
    require(obj_hash({k:v for k,v in p.items() if k!='identity'})==p['identity']==manifest['protocol_identity'],
            'Step62 protocol identity invalid')
    require(p['version']=='shared_nominal_temporal_information_control_v1' and p['dataset_id']==DATASET_ID,
            'Wrong Step62 protocol or dataset')
    require(p['script_sha256']==STEP62_SHA,'Use the reviewed current Step62 source')
    verify_file(run/'source/62.temporal_information_control.py',STEP62_SHA,files)
    require(tuple(p['seeds'])==SEEDS and set(p['conditions'])==set(CONDITIONS),'Step62 seeds/conditions differ')
    require(p['architecture']['sequence_length']==10 and p['encoder_training'] is False,'Wrong Step62 architecture/encoder contract')
    source=p['source_contract']; spec=source['observability_audit']
    require(source['dataset_id']==DATASET_ID==spec['dataset_id'] and
            spec['schema_version']=='counterfactual_observable_change_v1','Wrong observability audit contract')
    endpoints={s:pd.read_csv(run/f'endpoint_audit_{s}.csv',keep_default_na=False) for s in ('train','val','test')}
    for s,m in endpoints.items():
        pop=p['populations'][s]
        require(len(m)==pop['endpoints'] and m.episode_id.nunique()==pop['endpoint_episodes'],'Wrong endpoint population: '+s)
        require(np.array_equal(m.sequence_id,np.arange(len(m))) and m.source_split.eq(s).all(),'Endpoint ID/split mismatch')
        require(np.array_equal(m.label.map(dict(zip(LABELS,range(6)))),m.label_id),'Endpoint labels differ')
        require((m.sequence_end_row-m.sequence_start_row).eq(9).all() and
                m.sequence_end_step.eq(m.episode_step).all(),'Endpoint L10 alignment differs')
        require(hashlib.sha256(m.sequence_end_row.to_numpy(dtype='<i8').tobytes()).hexdigest()==pop['endpoint_sha256'],
                'Endpoint index hash differs')
        for label in LABELS: require(int(m.label.eq(label).sum())==pop['class_counts'][label],'Class population mismatch')
    for a,b in [('train','val'),('train','test'),('val','test')]:
        require(not (set(endpoints[a].episode_id)&set(endpoints[b].episode_id)),'Episode leakage across splits')
    gate=read_json(run/'trainval_gate.json')
    jobs={(j['seed'],j['condition']):j for j in gate['jobs']}
    require(gate['status']=='ALL_TRAINVAL_JOBS_VERIFIED' and len(gate['jobs'])==20 and
            set(jobs)=={(s,c) for s in SEEDS for c in CONDITIONS},'Invalid train/validation gate')
    for seed in SEEDS:
        pair=[]
        for cond in CONDITIONS:
            folder=run/f'seed_{seed}'/cond
            done=read_json(folder/'trainval_complete.json'); test=read_json(folder/'test_complete.json')
            verify_file(folder/'trainval_complete.json',jobs[(seed,cond)]['completion_sha256'],files)
            expected=obj_hash(dict(protocol=p['identity'],seed=seed,condition=cond))
            require(done['contract']==test['contract']==expected,'Completion contract mismatch')
            for record in [done,test]:
                for name,h in record['artifact_sha256'].items():
                    if name=='best.pt': continue  # Review ZIP intentionally omits checkpoint; no inference occurs.
                    verify_file(safe_child(folder,name),h,files)
            r=done['result']; pair.append(r)
            require(r['seed']==seed and r['condition']==cond and r['epochs_completed']==30 and
                    r['parameter_count']==234758,'Training budget/source differs')
        for key in ('initial_state_sha256','epoch_order_sha256','dropout_seeds','epochs_completed','updates','parameter_count'):
            require(pair[0][key]==pair[1][key],'Unmatched paired training: '+key)
    return dict(run=run,protocol=p,seeds=list(SEEDS),endpoints=endpoints,
                source_audit=dict(step62_protocol_identity=p['identity'],dataset_id=DATASET_ID,verified_files=files))

def validate_windows(windows,endpoints,split,pop):
    require(list(windows)==list(META)+list(AUDIT_FIELDS),'Step7 audit column schema differs')
    require(len(windows)==pop['windows'] and windows.episode_id.nunique()==pop['episodes'],'Window population differs: '+split)
    require(not windows[list(META)].isna().any().any() and
            not windows[['episode_id','episode_step']].duplicated().any(),'Missing/duplicated window metadata')
    for col in ('episode_step','window_id')+AUDIT_FIELDS:
        a=windows[col].to_numpy()
        require(np.issubdtype(a.dtype,np.integer),col+' must be integer')
    for col in ('audit_observable_effect_present','audit_persistent_effect_present','audit_attack_active'):
        require(windows[col].isin([0,1]).all(),'Nonbinary audit flag: '+col)
    counts=windows.audit_observable_difference_count
    require(counts.between(0,22).all() and np.array_equal(counts.gt(0),windows.audit_observable_effect_present.astype(bool)),
            'Observable difference count/flag disagreement')
    require(windows.loc[windows.label.eq('NORMAL'),list(AUDIT_FIELDS)].eq(0).all().all(),'NORMAL counterfactual audit must be zero')
    require(windows.label.isin(LABELS).all(),'Unknown window label')
    times=pd.to_datetime(windows.timestamp,utc=True,errors='raise').to_numpy(dtype='datetime64[ns]')
    require(not np.isnat(times).any(),'Missing timestamp')
    ends=[]
    for episode,ix in windows.groupby('episode_id',sort=False).indices.items():
        ix=np.asarray(ix); w=windows.iloc[ix]
        require(str(episode).strip()!='' and np.array_equal(ix,np.arange(ix[0],ix[-1]+1)),
                'Episode rows must be contiguous: '+str(episode))
        require(w.label.nunique()==1 and w.scenario_id.nunique()==1 and w.episode_type.nunique()==1,'Mixed episode metadata')
        require(np.array_equal(w.episode_step,np.arange(len(w))),'Episode steps must start at0 and be consecutive')
        require((np.diff(times[ix])==np.timedelta64(1,'s')).all(),'Expected recorded one-second cadence')
        ends.extend(ix[9:])
    ends=np.asarray(ends,dtype=np.int64)
    require(np.array_equal(ends,endpoints.sequence_end_row),'Must keep every causal stride1 L10 endpoint')
    require(np.array_equal(ends-9,endpoints.sequence_start_row),'Sequence begins outside causal episode window')
    observed=windows.iloc[ends][list(META)].reset_index(drop=True)
    for col in META: require(np.array_equal(observed[col],endpoints[col]),'Audit/endpoint metadata mismatch: '+col)

def load_context(args):
    require(args.bootstrap_draws>=2000,'Use at least2000 episode-bootstrap draws (default20000)')
    require(0<=args.bootstrap_seed<2**63,'bootstrap seed out of range')
    root=Path(args.project_root).resolve()
    run=_resolve(root,args.step62_run,DEFAULT_RUN)
    print('Verifying completed Step62, all recorded result hashes and paired training ...',flush=True)
    ctx=verify_step62(run)
    dataset=_resolve(root,args.dataset_dir,Path('dataset/shared_nominal_v2'))
    source=ctx['protocol']['source_contract']; spec=source['observability_audit']
    files=ctx['source_audit']['verified_files']
    verify_file(dataset/'build_manifest.json',spec['step7_build_manifest_sha256'],files)
    build=read_json(dataset/'build_manifest.json')
    require(build['dataset_id']==DATASET_ID,'Step7 dataset identity differs')
    ctx['windows']={}
    for split in ('val','test'):
        relative=spec['split_files'][split]; path=safe_child(dataset,relative)
        verify_file(path,spec['artifacts'][relative],files)
        splitname=f'splits/{split}_observable.csv'; splitpath=dataset/splitname
        verify_file(splitpath,source['input_sha256'][splitname],files)
        windows=pd.read_csv(path,keep_default_na=False)
        splitmeta=pd.read_csv(splitpath,usecols=list(META),keep_default_na=False)
        for col in META: require(np.array_equal(windows[col],splitmeta[col]),'Step7 learning/audit mismatch: '+col)
        validate_windows(windows,ctx['endpoints'][split],split,ctx['protocol']['populations'][split])
        ctx['windows'][split]=windows
    require(not(set(ctx['windows']['val'].episode_id)&set(ctx['windows']['test'].episode_id)), 'Window split leakage')
    ctx['dataset']=dataset
    ctx['source_audit'].update(dataset_dir=str(dataset),verified_at_utc=utc_now(),
        audit_role='descriptive evaluation only; never supplied to a predictor or threshold selector',
        prediction_loading='val/test probabilities loaded only when requested by evaluator')
    return ctx

def load_predictions(ctx,split):
    require(split in ('val','test'),'Only validation/test prediction files are supported')
    meta=ctx['endpoints'][split]; out=np.empty((len(ctx['seeds']),2,len(meta),6),np.float64)
    for si,seed in enumerate(ctx['seeds']):
        for ci,cond in enumerate(CONDITIONS):
            folder=ctx['run']/f'seed_{seed}'/cond; path=folder/f'{split}_predictions.csv'
            # Recheck just before reading, including the gate-selected validation artifacts.
            verify_file(path,ctx['source_audit']['verified_files'][str(path.resolve())],{})
            f=pd.read_csv(path,keep_default_na=False)
            require(len(f)==len(meta),'Prediction length differs')
            for col in meta: require(np.array_equal(f[col],meta[col]),'Prediction endpoint alignment: '+col)
            require(f.seed.eq(seed).all() and f.condition.eq(cond).all() and np.array_equal(f.y_true,meta.label_id),
                    'Prediction seed/condition/labels differ')
            prob=f[[f'prob_{i}' for i in range(6)]].to_numpy(dtype=np.float64)
            require(np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all() and
                    np.max(np.abs(prob.sum(1)-1))<5e-7,'Invalid saved probabilities')
            require(np.array_equal(prob.argmax(1),f.y_pred),'Probability/prediction argmax differs')
            cm=np.bincount(f.y_true.to_numpy(dtype=int)*6+f.y_pred.to_numpy(dtype=int),minlength=36).reshape(6,6)
            metrics=read_json(folder/('validation_metrics.json' if split=='val' else 'test_metrics.json'))
            require(np.array_equal(cm,metrics['confusion_matrix']) and metrics['n']==len(f),'Prediction confusion matrix differs')
            out[si,ci]=prob
    return out

def evidence_endpoints(windows,meta):
    changed=windows.audit_observable_effect_present.to_numpy(dtype=bool)
    ends=meta.sequence_end_row.to_numpy(dtype=np.int64)
    require(np.array_equal(ends-9,meta.sequence_start_row),'Expected causal L10')
    require((ends>=9).all() and (ends<len(changed)).all(),'L10 indices out of bounds')
    past=changed[ends[:,None]-np.arange(1,10)]
    count=past.sum(1); current=changed[ends]
    lag=np.where(count>0,past.argmax(1)+1,0)
    strata=np.where(current,'current_changed',np.where(count>0,'history_only_changed','no_change_in_L10'))
    strata=np.where(meta.label_id.to_numpy()==0,'NORMAL',strata)
    result=meta.copy()
    result['current_changed']=current.astype(int)
    result['past_changed_count']=count
    result['most_recent_past_change_lag']=lag
    result['evidence_stratum']=strata
    return result

def bootstrap_weights(episode_labels,draws,seed):
    labels=np.asarray(episode_labels); rng=np.random.default_rng(seed)
    weights=np.zeros((draws,len(labels)),dtype=np.float64)
    for c in np.unique(labels):
        ix=np.flatnonzero(labels==c)
        weights[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=draws)
    return weights

def ratio_summary(num,den,episode_labels,target_classes,weights):
    """Paired mean-head class-macro ratios; whole-episode weights are shared.

    num[E,2] is the mean correct count over FIXED head seeds; den[E] is shared
    across those heads. Linearity then equals averaging each head's class recall.
    CI is unavailable if any target class is absent, has <2 eligible episodes,
    or any bootstrap replicate has an empty target subgroup. Never silently
    average over fewer classes or drop undefined replicates.
    """
    num=np.asarray(num,float); den=np.asarray(den,float); labels=np.asarray(episode_labels)
    classes=list(target_classes); point=[]; boot=[]; supported=[]
    for c in classes:
        ix=np.flatnonzero(labels==c); d=den[ix].sum(); supported.append(int((den[ix]>0).sum()))
        point.append(num[ix].sum(0)/d if d else np.full(2,np.nan))
        bd=weights[:,ix]@den[ix]; bn=weights[:,ix]@num[ix]
        boot.append(np.divide(bn,bd[:,None],out=np.full_like(bn,np.nan),where=bd[:,None]>0))
    point=np.mean(point,axis=0); b=np.mean(boot,axis=0)
    samples=np.column_stack([b,b[:,0]-b[:,1]])
    status=('missing_target_class' if min(supported,default=0)==0 else
            'fewer_than_two_eligible_episodes_in_a_target_class' if min(supported)<2 else
            'empty_subgroup_in_resamples' if not np.isfinite(samples).all() else 'ok')
    q=np.quantile(samples,[.025,.975],axis=0) if status=='ok' else np.full((2,3),np.nan)
    return dict(actual_history=float(point[0]),repeat_current=float(point[1]),difference=float(point[0]-point[1]),
        actual_ci95_low=q[0,0],actual_ci95_high=q[1,0],repeat_ci95_low=q[0,1],repeat_ci95_high=q[1,1],
        ci95_low=q[0,2],ci95_high=q[1,2],ci_status=status,
        finite_bootstrap_draws=int(np.isfinite(samples).all(1).sum()),bootstrap_draws=len(weights),
        eligible_episodes_per_target_class=supported),samples

def group_masks(table):
    attack=table.label_id.to_numpy()>0; current=table.current_changed.to_numpy().astype(bool)
    groups=[('all_attack_endpoints',attack),('NORMAL',~attack)]
    for name in ('current_changed','history_only_changed','no_change_in_L10'):
        groups.append((name,table.evidence_stratum.eq(name).to_numpy()))
    for count in range(10):
        groups.append((f'current_unchanged_past_count_{count}',attack&~current&(table.past_changed_count.to_numpy()==count)))
    for lag in range(1,10):
        groups.append((f'history_only_recent_lag_{lag}',table.evidence_stratum.eq('history_only_changed').to_numpy()&
                       (table.most_recent_past_change_lag.to_numpy()==lag)))
    return groups

def analyze_evidence(ctx,prob,out,draws,seed):
    table=evidence_endpoints(ctx['windows']['test'],ctx['endpoints']['test'])
    write_csv(out/'endpoint_evidence_test.csv',table)
    y=table.label_id.to_numpy(dtype=int); pred=prob.argmax(-1)
    episodes,ei=np.unique(table.episode_id.to_numpy(),return_inverse=True)
    labels=np.array([y[np.flatnonzero(ei==i)[0]] for i in range(len(episodes))])
    weights=bootstrap_weights(labels,draws,seed)
    correct=(pred==y[None,None,:]); mean_correct=correct.mean(0)
    rows=[]; by_seed=[]; primary_samples=None; primary=None
    for group,mask in group_masks(table):
        den=np.bincount(ei,weights=mask,minlength=len(episodes))
        num=np.column_stack([np.bincount(ei,weights=mask*mean_correct[c],minlength=len(episodes)) for c in range(2)])
        targets=[('NORMAL',[0])] if group=='NORMAL' else [(label,[c]) for c,label in enumerate(LABELS) if c>0]+[('ATTACK_MACRO',list(range(1,6)))]
        for target,classes in targets:
            estimate,boot=ratio_summary(num,den,labels,classes,weights)
            per_head=[]
            for si,head_seed in enumerate(ctx['seeds']):
                per_class=np.array([correct[si][:,mask&(y==c)].mean(1) if (mask&(y==c)).any() else np.full(2,np.nan) for c in classes])
                value=per_class.mean(0); per_head.append(value)
                for ci,cond in enumerate(CONDITIONS):
                    by_seed.append(dict(group=group,target=target,seed=head_seed,condition=cond,recall=value[ci]))
            per_head=np.asarray(per_head)
            included=np.isin(labels,classes)
            row=dict(group=group,target=target,metric='class_recall' if len(classes)==1 else 'five_attack_class_macro_recall',
                endpoints=int(den[included].sum()),episodes=int(((den>0)&included).sum()),
                actual_head_seed_sd=float(per_head[:,0].std(ddof=1)),repeat_head_seed_sd=float(per_head[:,1].std(ddof=1)),
                paired_head_seed_sd=float((per_head[:,0]-per_head[:,1]).std(ddof=1)),
                **{k:v for k,v in estimate.items() if k!='eligible_episodes_per_target_class'})
            rows.append(row)
            if group=='history_only_changed' and target=='ATTACK_MACRO':
                primary=row; primary_samples=boot
    results=pd.DataFrame(rows); write_csv(out/'evidence_recall_summary.csv',results)
    write_csv(out/'evidence_recall_by_head_seed.csv',pd.DataFrame(by_seed))
    write_csv(out/'primary_episode_bootstrap.csv',pd.DataFrame(primary_samples,columns=['actual_history','repeat_current','difference']))
    # Whole-population reference is six-class Macro-F1, not an attack-only F1.
    references=[]
    for si,head_seed in enumerate(ctx['seeds']):
        for ci,cond in enumerate(CONDITIONS):
            cm=np.bincount(y*6+pred[si,ci],minlength=36).reshape(6,6)
            denom=cm.sum(0)+cm.sum(1); f1=np.divide(2*np.diag(cm),denom,out=np.zeros(6),where=denom>0).mean()
            references.append(dict(seed=head_seed,condition=cond,macro_f1=f1,normal_fpr=1-cm[0,0]/cm[0].sum()))
    write_csv(out/'step62_metric_replay.csv',pd.DataFrame(references))
    summary=dict(primary=primary,all_groups_reported=True,training=False,new_inference=False,
        conclusion_scope='Mechanistic association within the inspected Step62 dataset; not independent confirmation or causal mediation.',
        primary_interpretation='Positive history-only contrast supports use of past observable projection changes when current projection is unchanged.',
        limitations=[
            'Observable changes are paired nominal-versus-actual 22-field differences; they are not guaranteed class-identifying evidence.',
            'No-change predictions alone cannot prove leakage, impossibility or a Bayes error bound.',
            'Count/recency strata can differ in episode difficulty and attack stage; trends are descriptive, not randomized effects.',
            'CIs condition on the fixed ten head seeds, one encoder and generator; no seed, encoder or generator resampling.',
            'Secondary CIs are pointwise exploratory intervals, without multiplicity correction.',
            'No result here proves BERT necessity, 22-field superiority over raw, fine order sensitivity or TST superiority over LSTM.'])
    write_json(out/'summary.json',summary)
    return summary

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser); args=parser.parse_args(argv)
    out=create_output(Path(args.project_root).resolve(),63,'temporal_evidence_analysis')
    try:
        shutil.copy2(Path(__file__),out/Path(__file__).name)
        ctx=load_context(args)
        write_json(out/'input_audit.json',ctx['source_audit'])
        protocol=dict(version=VERSION,created_at_utc=utc_now(),script_sha256=digest(__file__),
            source_step62_identity=ctx['protocol']['identity'],dataset_id=DATASET_ID,
            labels=LABELS,conditions=CONDITIONS,seeds=ctx['seeds'],post_hoc=True,training=False,
            primary='history_only_changed: actual_history minus repeat_current five-attack-class macro recall',
            groups='all attack; normal; current/history-only/no change; current-unchanged past count0..9; history-only recent lag1..9',
            bootstrap=dict(draws=args.bootstrap_draws,seed=args.bootstrap_seed,unit='class-stratified whole episode',
                fixed_head_seeds=True,paired=True,primary_ci=.95,secondary_intervals='pointwise exploratory'),
            runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
            interpretation='Observed 22-field projection change is audit information, never a model feature or an oracle bound.')
        protocol['identity']=obj_hash(protocol); write_json(out/'protocol.json',protocol)
        if args.preflight_only:
            archive=finish_output(out,'STEP63_PREFLIGHT_PASS')
            print('Preflight passed; no analysis or prediction loading.\nReview ZIP: '+str(archive)); return 0
        print('Loading authenticated test predictions and analyzing whole-episode strata ...',flush=True)
        prob=load_predictions(ctx,'test')
        try:
            from threadpoolctl import threadpool_limits
            limits=threadpool_limits(limits=4)
        except ImportError: limits=nullcontext()
        with limits: summary=analyze_evidence(ctx,prob,out,args.bootstrap_draws,args.bootstrap_seed)
        archive=finish_output(out,'STEP63_TEMPORAL_EVIDENCE_COMPLETE',dict(protocol_identity=protocol['identity']))
        print('Step63 complete. Primary paired recall difference: '+str(summary['primary']['difference']))
        print('Results: '+str(out)+'\nReview ZIP: '+str(archive)); return 0
    except Exception as exc:
        write_json(out/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()))
        archive=finish_output(out,'STEP63_FAILED')
        print('STOPPED: '+str(exc)+'\nDiagnostic ZIP: '+str(archive),file=sys.stderr); return 1

if __name__=='__main__':
    sys.exit(main())
