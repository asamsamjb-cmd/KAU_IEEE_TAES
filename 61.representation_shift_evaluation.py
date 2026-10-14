#!/usr/bin/env python3
"""Step61: frozen Raw versus Semantic22 under the EXISTING Step49 regimes.

Place beside existing 59.raw_semantic_representation_baseline.py and
60.paired_attack_response_audit.py in the original project directory.
    python "61.representation_shift_evaluation.py"
    python "61.representation_shift_evaluation.py" --preflight-only

CPU inference only: original Step59 L10 HGB/MLP, both representations, all five
saved seeds; no fitting, new generator, threshold adjustment or row filtering.
Step49 reference/mild/moderate/severe all retained. Raw means simulated monitor
measurements/properties, not RF waveforms. Saved train-fitted vocabularies and
scalers are immutable. Fixed validation samples replay original predictions
before any shift inference. User's ORIGINAL Step59 directory is needed because
its review ZIP intentionally omits model.joblib files.

Outputs: outputs/shared_nominal_v2/61_representation_shift/run_<UTC>/ and ZIP.
Each invocation creates a NEW run; previous outputs are never overwritten.
Statistics: common nominal-episode bootstrap paired across six class replicas,
conditions, representations and fixed trained seeds. Primary family has 16
contrasts (4 conditions x 2 learners x Macro-F1/NORMAL-FPR). No seed resampling.
Interpretation: retrospective internal sensitivity, not independent external
validation, confirmatory preregistration, equivalence, or architecture superiority.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
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
import warnings
import zipfile
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
import numpy as np
import pandas as pd

VERSION='raw_semantic_frozen_parameter_shift_v2'
S60_SHA='250c5d7513638c978ef2126b3e39c2575df68ad721d36d4761bd8d46b21f9102'
CONDITIONS=('P_reference','P_mild','P_moderate','P_severe')
GROUPS=(('Raw','HGB'),('RawScoped','HGB'),('Semantic22','HGB'),('Raw','MLP'),('RawScoped','MLP'),('Semantic22','MLP'))
REPRESENTATIONS=('Raw','RawScoped','Semantic22')
PAIRS=(('matched_scope','Semantic22','RawScoped'),('scope_effect','RawScoped','Raw'),('all_link','Semantic22','Raw'))
STEP59_VERSION='shared_nominal_raw_semantic_v2'
SEEDS=tuple(range(20260610,20260615))
METRICS=('macro_f1','normal_false_alarm_rate')+tuple('recall_'+x for x in (
    'NORMAL','RF_LINK_JAMMING','GNSS_SPOOFING','UNAUTHORIZED_TELECOMMAND_INJECTION',
    'TELECOMMAND_REPLAY','GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE'))


def require(ok,message):
    if not bool(ok):raise ValueError(message)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def object_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def source_code(root,number,expected,extra=()):
    candidates=list(Path(root).glob(f'{number}.*.py'))
    for folder in extra:candidates.extend(Path(folder).glob(f'{number}.*.py'))
    matched=sorted({p.resolve() for p in candidates if p.is_file() and digest(p)==expected})
    require(matched,f'Missing exact recorded Step{number} code SHA256 {expected}; restore the original code, do not edit its hash.')
    return matched[0]

def prepare_output(out,sources):
    out=Path(out).resolve()
    for s in sources:
        s=Path(s).resolve()
        require(out!=s and s not in out.parents and out not in s.parents,'Output overlaps a source directory')
    out.mkdir(parents=True,exist_ok=False)
    return out

def restore_encoder(base,record,validate_fields=True):
    numeric=record['numeric'];categorical=record['categorical'];categories=record['categories']
    require(len(numeric+categorical)==len(set(numeric+categorical)),'Repeated feature field')
    require(set(categories)==set(categorical),'Vocabulary inventory mismatch')
    if validate_fields:
        raw=numeric==list(base.RAW_NUMERIC) and categorical==list(base.RAW_CATEGORICAL)
        scoped=numeric==list(base.SCOPED_NUMERIC) and categorical==list(base.RAW_CATEGORICAL)
        semantic=numeric==[] and categorical==list(base.FIELDS)
        require(raw or scoped or semantic,'Unexpected feature allowlist')
    names=list(numeric)
    for field in categorical:
        vals=categories[field]
        require(isinstance(vals,list) and vals==sorted(set(vals)) and all(isinstance(v,str) and v for v in vals),'Invalid saved vocabulary')
        names.extend([field+'='+v for v in vals]+[field+'=<UNSEEN>'])
    require(names==record['window_feature_names'] and record['unknown_policy']=='one explicit unseen category per field','Saved feature order/policy differs')
    enc=base.WindowEncoder(numeric,categorical);enc.categories=categories;enc.names=names
    require(enc.record()==record,'Feature encoder restoration differs')
    return enc

def validate_raw_alignment(raw,frame,fields,meta):
    require(len(raw)==len(frame),'Raw/observable record count mismatch')
    renamed=raw[list(meta)+list(fields)].copy()
    for k in ('scenario_id','episode_id'):renamed[k]='P_'+renamed[k].astype(str)
    for k in tuple(meta)+tuple(fields):
        require(np.array_equal(renamed[k].astype(str).to_numpy(),frame[k].astype(str).to_numpy()),'Raw/observable alignment differs: '+k)
    return raw

def cluster_confusions(y,predictions,codes,cluster_ids):
    y=np.asarray(y);p=np.asarray(predictions);codes=np.asarray(codes);cluster_ids=np.asarray(cluster_ids)
    require(y.ndim==1 and p.ndim==2 and p.shape[1]==len(y) and codes.shape==y.shape,'Confusion population mismatch')
    require(y.dtype.kind in 'iu' and p.dtype.kind in 'iu' and np.isin(y,np.arange(6)).all() and np.isin(p,np.arange(6)).all(),'Invalid prediction/target labels')
    require(len(cluster_ids)>=2 and len(set(cluster_ids.tolist()))==len(cluster_ids),'Invalid cluster inventory')
    mapping={v:i for i,v in enumerate(cluster_ids.tolist())}
    require(set(codes.tolist())==set(mapping),'Cluster population mismatch')
    indices=np.asarray([mapping[v] for v in codes],dtype=np.int64)
    out=np.empty((p.shape[0],len(cluster_ids),6,6),dtype=np.int64)
    for i,row in enumerate(p):out[i]=np.bincount(indices*36+y*6+row,minlength=len(cluster_ids)*36).reshape(len(cluster_ids),6,6)
    return out

def metrics_from_confusions(cm):
    cm=np.asarray(cm,dtype=np.float64);tp=np.diagonal(cm,axis1=-2,axis2=-1)
    support=cm.sum(-1);den=support+cm.sum(-2)
    require(np.all(support>0),'Missing class in resampled cohort; cannot silently discard a class')
    f=np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0).mean(-1)
    recall=tp/support
    return np.concatenate([f[...,None],(1-recall[...,0])[...,None],recall],axis=-1)

def bootstrap_metrics(counts,weights):
    counts=np.asarray(counts);weights=np.asarray(weights)
    require(counts.ndim==4 and counts.shape[-2:]==(6,6) and weights.ndim==2 and weights.shape[1]==counts.shape[1],'Invalid bootstrap shape')
    require(np.isfinite(weights).all() and (weights>=0).all() and np.all(weights==np.floor(weights)) and np.all(weights.sum(1)==counts.shape[1]),'Invalid episode multiplicities')
    out=np.empty((len(weights),counts.shape[0],8),dtype=np.float64)
    for start in range(0,len(weights),256):
        cm=np.einsum('bk,skij->bsij',weights[start:start+256].astype(float),counts,optimize=True)
        out[start:start+256]=metrics_from_confusions(cm)
    return out

def interval(x,alpha=.05):
    require(np.isfinite(x).all(),'Nonfinite bootstrap statistic')
    lo,hi=np.quantile(x,[alpha/2,1-alpha/2],method='linear')
    return float(lo),float(hi)

def make_contrasts(points,samples):
    """Three prespecified pairs per learner; Bonferroni 16 within each pair family (4 conditions x 2 learners x 2 metrics), 12 for changes."""
    comparisons=[];changes=[];index={g:i for i,g in enumerate(GROUPS)}
    require(points.shape==(4,len(GROUPS),8) and samples.shape[1:]==points.shape,'Unexpected model summary shape')
    for ci,condition in enumerate(CONDITIONS):
        for family in ('HGB','MLP'):
            for pair,first,second in PAIRS:
                si=index[(first,family)];ri=index[(second,family)]
                for mi,metric in enumerate(METRICS[:2]):
                    delta=samples[:,ci,si,mi]-samples[:,ci,ri,mi]
                    lo,hi=interval(delta);blo,bhi=interval(delta,.05/16)
                    row=dict(condition=condition,family=family,pair=pair,first=first,second=second,metric=metric,first_mean=float(points[ci,si,mi]),second_mean=float(points[ci,ri,mi]),first_minus_second=float(points[ci,si,mi]-points[ci,ri,mi]),ci95_low=lo,ci95_high=hi,bonferroni16_low=blo,bonferroni16_high=bhi,favorable_direction='positive' if mi==0 else 'negative',scope='Retrospective conditional interval within one pair family; zero included is not equivalence')
                    comparisons.append(row)
                    if ci:
                        d=delta-(samples[:,0,si,mi]-samples[:,0,ri,mi]);dl,dh=interval(d);bl,bh=interval(d,.05/12)
                        changes.append(dict(condition=condition,family=family,pair=pair,first=first,second=second,metric=metric,first_change=float(points[ci,si,mi]-points[0,si,mi]),second_change=float(points[ci,ri,mi]-points[0,ri,mi]),difference_in_changes=float(points[ci,si,mi]-points[0,si,mi]-points[ci,ri,mi]+points[0,ri,mi]),ci95_low=dl,ci95_high=dh,bonferroni12_low=bl,bonferroni12_high=bh,scope='Secondary change relative to Step49 reference; smaller loss does not imply higher absolute performance'))
    return comparisons,changes

def open_step59(folder,helper):
    path=Path(folder)/'run_manifest.json';m=read_json(path)
    require(m.get('status')=='RAW_SEMANTIC_BASELINE_COMPLETE' and m.get('test_used_for_selection') is False,'Need a completed original Step59 run, not an incomplete/preflight run')
    reader=helper.BoundReader(folder,dict(m['output_sha256']));reader.hashes['run_manifest.json']=digest(path);reader.path('run_manifest.json')
    p=reader.json('protocol.json');helper.check_identity(p,'Step59 protocol')
    require(p.get('version')==STEP59_VERSION and p['identity']==m['protocol_identity'] and p['dataset_id']==m['dataset_id'],'Step59 protocol/manifest mismatch (need the v2 run with RawScoped)')
    require(p['seeds']==list(SEEDS) and p['families']==['HGB','MLP'] and p['lengths']==[1,10] and set(REPRESENTATIONS)<=set(p['representations']),'Step59 model/seed inventory differs')
    require(digest(reader.path('source/59.raw_semantic_representation_baseline.py'))==p['script_sha256'],'Step59 producer source differs')
    marker=reader.json('all_training_complete.json')
    require(marker['protocol_identity']==p['identity'] and marker['jobs']==len(p['representations'])*len(SEEDS)*4 and marker['test_scoring_started'] is False,'Step59 training-completion barrier missing')
    return reader,p

def collect_models(reader,protocol,encoders):
    cases=[]
    for rep,family in GROUPS:
        for seed in SEEDS:
            key=f'{rep}_L10_{family}_seed{seed}';folder='jobs/'+key
            done=reader.json(folder+'/complete.json');report=reader.json(folder+'/training_report.json')
            expected=object_hash(dict(protocol=protocol['identity'],key=key,encoder=encoders[rep].record()))
            require(done['identity']==expected,'Step59 job identity differs: '+key)
            require(set(done['files'])=={'model.joblib','history.csv','training_report.json','validation_predictions.npz'},'Incomplete saved training job')
            for name,h in done['files'].items():
                require(reader.hashes.get(folder+'/'+name)==h,'Manifest/job hash mismatch: '+key+'/'+name)
            local=reader.root/folder/'model.joblib'
            require(local.is_file(),'Missing model.joblib: '+str(local)+'. Use the ORIGINAL Step59 run on the training PC; the review ZIP excludes models. No automatic retraining.')
            for name in done['files']:reader.path(folder+'/'+name)
            require(report['features']==len(encoders[rep].names)*5 and report['family']==family and report['seed']==seed and report.get('validation_replay')=='PASS','Saved model report differs: '+key)
            cases.append(dict(key=key,representation=rep,family=family,seed=seed,path=str(local),features=report['features']))
    return cases

def load_raw_condition(reader,protocol,condition,cohort,base,projection):
    folder='generation/'+condition;frame=reader.csv(folder+'/evaluation_records.csv')
    done=reader.json(folder+'/complete.json');parts=[];changed_raw=0;changed_semantic=0;changed_scoped=0
    columns=list(dict.fromkeys(list(base.META)+list(base.FIELDS)+list(base.RAW_KEYS)+['baseline_'+k for k in base.RAW_KEYS]+['baseline_observable_'+k for k in base.FIELDS]+['raw_observable_effect_present']))
    n=protocol['generator']['rows_per_class']
    for i,label in enumerate(base.LABELS,1):
        name=f'raw_class_{i}.csv.gz';rel=folder+'/'+name
        require(done['artifact_sha256'].get(name)==reader.hashes.get(rel),'Unbound raw generation artifact')
        raw=pd.read_csv(reader.path(rel),usecols=columns,keep_default_na=False,encoding='utf-8-sig')
        wanted=frame.iloc[(i-1)*n:i*n].reset_index(drop=True)
        require(len(raw)==n and raw.label.eq(label).all(),'Raw class inventory differs')
        validate_raw_alignment(raw,wanted,base.FIELDS,base.META)
        clean,semantic_same,raw_same,scoped_same=base.audit_projection(raw,projection)
        parts.append(clean);changed_raw+=int((~raw_same).sum());changed_semantic+=int((~semantic_same).sum());changed_scoped+=int((~scoped_same).sum())
        del raw
    raw_frame=pd.concat(parts,ignore_index=True)
    seq=reader.arrays(folder+'/sequences_L10.npz',('indices','y'))
    require(np.array_equal(seq['y'],cohort['y']),'Raw evaluation target alignment differs')
    require(len(raw_frame)==len(frame) and not raw_frame.isna().any().any(),'Missing raw records')
    return frame,raw_frame,seq['indices'],dict(condition=condition,records=len(frame),endpoints=len(seq['y']),shared_nominal_episodes=len(cohort['block_ids']),raw_fields=len(base.RAW_FIELDS),actual_and_nominal_projection='PASS',metadata_and_observable_alignment='PASS',raw_changed_windows=changed_raw,scoped_raw_changed_windows=changed_scoped,semantic_changed_windows=changed_semantic,endpoint_filtering=False)

def resolve_recorded_input(root,recorded):
    p=Path(recorded)
    if p.is_file():return p.resolve()
    parts=PureWindowsPath(recorded).parts
    require('dataset' in parts,'Cannot relocate recorded dataset input: '+recorded)
    p=Path(root).joinpath(*parts[parts.index('dataset'):])
    require(p.is_file(),'Missing original dataset input: '+str(p))
    return p.resolve()

def make_validation_canary(root,reader,protocol,base,projection,per_class=16):
    """Label-balanced deterministic validation replay; labels choose audit rows only."""
    meta=reader.csv('endpoints_val_L10.csv')
    require(meta.sample_index.tolist()==list(range(len(meta))) and set(meta.label)==set(base.LABELS),'Validation endpoint order/schema changed')
    selected=[]
    for label in base.LABELS:
        ids=np.flatnonzero(meta.label.to_numpy()==label)
        require(len(ids)>=per_class,'Insufficient validation endpoints for replay')
        selected.extend(ids[np.linspace(0,len(ids)-1,per_class,dtype=int)].tolist())
    selected=np.asarray(sorted(selected),dtype=np.int64);end=meta.iloc[selected].reset_index(drop=True)
    keys=[(str(row.episode_id),int(row.episode_step)+offset) for row in end.itertuples() for offset in range(-9,1)]
    require(all(step>=0 for _,step in keys),'Validation endpoint is not a complete L10 context')
    episodes={e for e,_ in keys};pieces=[];observed={}
    paths=[(recorded,h) for recorded,h in protocol['inputs'].items() if recorded.endswith('_with_raw.csv')]
    require(len(paths)==6,'Expected six original raw source companions')
    cols=list(dict.fromkeys(list(base.META)+list(base.FIELDS)+list(base.RAW_KEYS)+['baseline_'+k for k in base.RAW_KEYS]+['baseline_observable_'+k for k in base.FIELDS]+['raw_observable_effect_present']))
    for recorded,expected in sorted(paths):
        path=resolve_recorded_input(root,recorded);require(digest(path)==expected,'Original raw source changed: '+str(path));observed[str(path)]=expected
        for chunk in pd.read_csv(path,usecols=cols,chunksize=10000,keep_default_na=False,encoding='utf-8-sig'):
            chosen=chunk[chunk.episode_id.astype(str).isin(episodes)]
            if len(chosen):pieces.append(chosen.copy())
    require(pieces,'No validation source records found')
    pool=pd.concat(pieces,ignore_index=True)
    index=pd.MultiIndex.from_arrays([pool.episode_id.astype(str),pool.episode_step.astype(int)])
    require(index.is_unique,'Duplicate validation source key')
    positions=index.get_indexer(pd.MultiIndex.from_tuples(keys));require((positions>=0).all(),'Missing validation history row')
    rows=pool.iloc[positions].reset_index(drop=True);indices=np.arange(len(rows)).reshape(-1,10)
    for k in base.META:require(np.array_equal(rows.iloc[indices[:,-1]][k].astype(str).to_numpy(),end[k].astype(str).to_numpy()),'Validation replay endpoint metadata differs: '+k)
    raw,_,_,_=base.audit_projection(rows,projection)
    y=end.label.map(dict(zip(base.LABELS,range(6)))).to_numpy(dtype=np.int64)
    return dict(raw=raw,semantic=rows[list(base.FIELDS)],indices=indices,selected=selected,y=y,metadata=end,observed=observed)

def load_model(case,protocol):
    import sklearn,joblib
    require(sklearn.__version__==protocol['runtime']['sklearn'],f"Saved sklearn {protocol['runtime']['sklearn']} differs from installed {sklearn.__version__}. Run in the original Step59 .venv; no silent cross-version pickle loading.")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');model=joblib.load(case['path'])
    require(not caught,'Saved model raised deserialization warnings: '+str([str(w.message) for w in caught]))
    require(np.array_equal(model.classes_,np.arange(6)) and model.n_features_in_==case['features'],'Restored model label/feature schema differs')
    return model

def replay_validation(reader,protocol,base,encoders,cases,canary,helper,out):
    rows=[]
    for rep in REPRESENTATIONS:
        source=canary['raw'] if rep=='Raw' else base.scope_raw(canary['raw']) if rep=='RawScoped' else canary['semantic']
        matrix=base.aggregate_windows(encoders[rep].transform(source),canary['indices'],10)
        for case in (c for c in cases if c['representation']==rep):
            saved=reader.arrays('jobs/'+case['key']+'/validation_predictions.npz',('y_true','y_pred'))
            sel=canary['selected'];require(np.array_equal(saved['y_true'][sel],canary['y']),'Saved validation label order differs')
            model=load_model(case,protocol);pred=base.predict_batched(model,matrix)
            require(np.array_equal(pred,saved['y_pred'][sel]),'Validation replay changed for '+case['key']+'; stop before stress scoring')
            rows.append(dict(key=case['key'],endpoints=len(sel),hard_prediction_mismatches=0,model_sha256=digest(case['path'])))
            del model
        del matrix
    helper.write_csv(out/'validation_replay.csv',rows);helper.write_csv(out/'validation_replay_endpoints.csv',canary['metadata'])
    return rows

def evaluate_models(base,protocol,encoders,cases,frame,raw,indices,memory_gib):
    expected=[(r,f,s) for r,f in GROUPS for s in SEEDS]
    require([(c['representation'],c['family'],c['seed']) for c in cases]==expected,'Missing/reordered model family or seed')
    predictions=np.empty((len(cases),len(indices)),dtype=np.int8);feature_report=[]
    for rep in REPRESENTATIONS:
        enc=encoders[rep];data=raw if rep=='Raw' else base.scope_raw(raw) if rep=='RawScoped' else frame[list(base.FIELDS)]
        estimate=(len(frame)*len(enc.names)+4*len(indices)*len(enc.names)*5)*8/2**30
        require(estimate<=memory_gib,f'Estimated feature working set {estimate:.2f} GiB > --memory-budget-gib {memory_gib}; no row subsampling.')
        features=base.aggregate_windows(enc.transform(data),indices,10)
        require(np.isfinite(features).all(),'Nonfinite aggregated features')
        for k in enc.categorical:
            feature_report.append(dict(representation=rep,field=k,unseen_windows=int((~data[k].astype(str).isin(enc.categories[k])).sum()),windows=len(data)))
        for j,case in enumerate(cases):
            if case['representation']!=rep:continue
            model=load_model(case,protocol);pred=base.predict_batched(model,features)
            require(pred.shape==(len(indices),) and np.isin(pred,np.arange(6)).all(),'Invalid frozen predictions')
            predictions[j]=pred;del model
        del features;gc.collect()
    return predictions,feature_report

def locate_step59(root,explicit):
    if explicit:
        p=Path(explicit).expanduser();p=p if p.is_absolute() else root/p
        require(p.is_dir(),'--step59-run is not an existing run directory');return p.resolve()
    base=root/'outputs/shared_nominal_v2/59_raw_semantic';matches=[]
    for p in sorted(base.glob('run_*/run_manifest.json')):
        try:
            if read_json(p).get('status')=='RAW_SEMANTIC_BASELINE_COMPLETE' and read_json(p.parent/'protocol.json').get('version')==STEP59_VERSION:matches.append(p.parent.resolve())
        except (OSError,ValueError):continue
    require(len(matches)==1,'Expected exactly one completed Step59 '+STEP59_VERSION+' run in '+str(base)+'; specify --step59-run PATH when missing/ambiguous. Review ZIP lacks models.')
    return matches[0]

def package(helper,out,status,extra):
    files=sorted(p for p in out.rglob('*') if p.is_file() and p.name!='run_manifest.json' and not p.name.endswith('.tmp'))
    m=dict(status=status,version=VERSION,completed_at_utc=utc(),**extra,output_sha256={p.relative_to(out).as_posix():digest(p) for p in files})
    m['identity']=object_hash(m);helper.write_json(out/'run_manifest.json',m)
    dest=out.with_name(out.name+('_FAILED_review.zip' if status=='FAILED' else '_review.zip'));tmp=dest.with_name(dest.name+'.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in files+[out/'run_manifest.json']:z.write(path,path.relative_to(out).as_posix())
    helper.atomic_replace(tmp,dest);return dest

def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--step49-run',type=Path)
    p.add_argument('--step59-run',type=Path)
    p.add_argument('--output-root',type=Path)
    p.add_argument('--preflight-only',action='store_true',help='Verify sources, raw projections and validation replay; do not score shift models or bootstrap')
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--memory-budget-gib',type=float,default=8.)
    p.add_argument('--bootstrap-draws',type=int,default=20000)
    p.add_argument('--bootstrap-seed',type=int,default=2026100761)
    return p

def main(argv=None):
    args=parser().parse_args(argv);out=None;helper=None;created=False;started=time.perf_counter()
    try:
        require(args.threads>0 and args.bootstrap_draws>=1000 and args.bootstrap_seed>=0 and np.isfinite(args.memory_budget_gib) and args.memory_budget_gib>0,'Invalid CPU/memory/bootstrap setting')
        root=args.project_root.expanduser().resolve();source59=locate_step59(root,args.step59_run)
        helper_path=source_code(root,60,S60_SHA);helper=load_module(helper_path,'step61_existing60')
        source49=helper.locate_source(root,args.step49_run)
        reader59,p59=open_step59(source59,helper);reader49,m49,p49=helper.open_source(source49)
        require(p59['dataset_id']==p49['source_contract']['dataset_id'],'Step49 and Step59 use different original training datasets')
        base_path=source_code(root,59,p59['script_sha256'],[source59/'source']);base=load_module(base_path,'step61_existing59')
        require(tuple(p59['raw_fields'])==base.RAW_FIELDS and tuple(p59['scoped_fields'])==base.SCOPED_NUMERIC and tuple(p59['semantic_fields'])==base.FIELDS and tuple(p59['labels'])==base.LABELS,'Original Step59 input allowlists differ')
        projection=base.load_projection(root)
        require(digest(reader49.path('source/shared_nominal.py'))==base.PROJECTION_SHA and digest(reader49.path('source/shared_context_v4.py'))==base.CONTEXT_SHA,'Step49 mapper differs from Step59 mapper')
        encoders={rep:restore_encoder(base,reader59.json('encoder_'+rep+'.json')) for rep in REPRESENTATIONS}
        cases=collect_models(reader59,p59,encoders)
        parent=args.output_root.expanduser() if args.output_root else root/'outputs/shared_nominal_v2/61_representation_shift'
        parent=parent if parent.is_absolute() else root/parent
        out=prepare_output(parent/datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ'),[source49,source59]);created=True
        from threadpoolctl import threadpool_limits
        import sklearn,joblib
        print(f'Step61 {VERSION}: source contracts verified; {len(GROUPS)*len(SEEDS)} saved L10 models (all-link raw, single-path scoped raw, Semantic22), four matched parameter regimes. CPU only.',flush=True)
        protocol=dict(version=VERSION,script_sha256=digest(__file__),dataset_id=p59['dataset_id'],source49=str(source49),source59=str(source59),step49_protocol_identity=p49['identity'],step59_protocol_identity=p59['identity'],step60_code_sha256=S60_SHA,step59_code_sha256=p59['script_sha256'],conditions=list(CONDITIONS),groups=[dict(representation=r,family=f) for r,f in GROUPS],seeds=list(SEEDS),length=10,training=False,normalization='Original Step59 train-fitted vocabulary, explicit UNSEEN bins and saved MLP scaler; no fit on shift',aggregation=p59['temporal_features'],raw_monitor_policy=p59['raw_monitor_policy'],preflight_only=args.preflight_only,validation_replay='16 evenly spaced validation endpoints per true class; audit only, no model selection',bootstrap=dict(draws=args.bootstrap_draws,seed=args.bootstrap_seed,unit='whole common nominal episode including all six class replicas',pairing='same multiplicities across four regimes, representations, learners and fixed seeds',estimator='mean of per-seed endpoint Macro-F1/FPR/recall; no probability ensemble or pooled-seed confusion',resample_training_seeds=False,resample_generator_seeds=False,intervals='conditional linear-percentile95; Bonferroni16 within each of three pair families (matched_scope: Semantic22-RawScoped, scope_effect: RawScoped-Raw, all_link: Semantic22-Raw) over 4 conditions x2 learners x2 metrics; change families Bonferroni12'),scope_policy=p59['scope_policy'],interpretation=['Retrospective analysis using previously observed source regimes; not preregistered confirmation.','Same generator with joint parameter changes; no external validity or isolated factor-causality claim.','Raw simulated measurements and monitor outputs; not raw RF waveforms.','Deterministic projection does not create information; differences are learner/feature interactions.','Raw and semantic temporal statistics do not preserve every ordering detail.','Report absolute scores and NORMAL false alarms as well as changes; smaller loss alone is not superiority.','All endpoints including quiet/unchanged inputs retained; no label, provenance, baseline or cause fields enter model.','Intervals conditional on these trained models and one paired nominal realization; not formal equivalence.'],runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__,threads=args.threads,platform=platform.platform()),source59_runtime=p59['runtime'])
        protocol['identity']=object_hash(protocol);helper.write_json(out/'protocol.json',protocol)
        src=out/'source';src.mkdir();
        for path in [Path(__file__),base_path,helper_path,root/'shared_nominal.py',root/'shared_context_v4.py']:shutil.copyfile(path,src/path.name)
        for label,reader in [('step49',reader49),('step59',reader59)]:
            folder=out/'source_metadata'/label;folder.mkdir(parents=True)
            for name in ('run_manifest.json','protocol.json'):shutil.copyfile(reader.path(name),folder/name)
        for rep in encoders:helper.write_json(out/('encoder_'+rep+'.json'),encoders[rep].record())
        helper.write_csv(out/'model_inventory.csv',cases)
        audits=[];all_summaries=[];seed_rows=[];feature_rows=[];points=[];samples=[]
        common=None;signature=None;weights=None;canary=None
        with threadpool_limits(limits=args.threads):
            print('  Replay 96 fixed original validation endpoints for every saved model ...',flush=True)
            canary=make_validation_canary(root,reader59,p59,base,projection)
            replay_validation(reader59,p59,base,encoders,cases,canary,helper,out)
            for condition in CONDITIONS:
                print('  '+condition+': validate full causal population and actual/nominal raw-to-token replay ...',flush=True)
                cohort=helper.load_condition(reader49,p49,condition)
                frame,raw,indices,audit=load_raw_condition(reader49,p49,condition,cohort,base,projection)
                audits.append(audit)
                if common is None:
                    common=cohort['block_ids'].copy();signature=cohort['endpoint_signature']
                    if not args.preflight_only:
                        weights=helper.bootstrap_weights(len(common),args.bootstrap_draws,args.bootstrap_seed)
                        helper.write_npz(out/'bootstrap_design.npz',block_ids=common,weights=weights)
                else:require(np.array_equal(common,cohort['block_ids']) and signature==cohort['endpoint_signature'],'Conditions do not share identical endpoint/episode population')
                folder=out/condition;folder.mkdir()
                helper.write_csv(folder/'endpoints.csv',cohort['meta'].assign(block_id=common[cohort['block_codes']]))
                if not args.preflight_only:
                    print(f'    Frozen prediction for all {len(GROUPS)*len(SEEDS)} models, then paired episode bootstrap ...',flush=True)
                    pred,fr=evaluate_models(base,p59,encoders,cases,frame,raw,indices,args.memory_budget_gib)
                    feature_rows.extend(dict(condition=condition,**r) for r in fr)
                    helper.write_npz(folder/'predictions.npz',predictions=pred,y=cohort['y'],sequence_id=cohort['meta'].sequence_id.to_numpy(),block_id=common[cohort['block_codes']],representations=np.array([c['representation'] for c in cases]),families=np.array([c['family'] for c in cases]),seeds=np.array([c['seed'] for c in cases]))
                    counts=cluster_confusions(cohort['y'],pred,common[cohort['block_codes']],common)
                    helper.write_npz(folder/'cluster_confusions.npz',counts=counts,block_ids=common)
                    ps=metrics_from_confusions(counts.sum(axis=1)).reshape(len(GROUPS),5,8)
                    bs=bootstrap_metrics(counts,weights).reshape(args.bootstrap_draws,len(GROUPS),5,8).mean(axis=2)
                    point=ps.mean(axis=1);points.append(point);samples.append(bs)
                    for gi,(rep,family) in enumerate(GROUPS):
                        for mi,metric in enumerate(METRICS):
                            lo,hi=interval(bs[:,gi,mi])
                            all_summaries.append(dict(condition=condition,representation=rep,family=family,metric=metric,mean=float(point[gi,mi]),seed_sd_ddof1=float(ps[gi,:,mi].std(ddof=1)),episode_ci95_low=lo,episode_ci95_high=hi,seeds=5,endpoints=len(indices),normal_endpoints=int((cohort['y']==0).sum()),shared_nominal_episodes=len(common)))
                        for si,seed in enumerate(SEEDS):seed_rows.append(dict(condition=condition,representation=rep,family=family,seed=seed,**{metric:float(ps[gi,si,mi]) for mi,metric in enumerate(METRICS)}))
                    del pred,counts,bs
                del frame,raw,indices,cohort;gc.collect()
        helper.write_csv(out/'cohort_audit.csv',audits)
        if not args.preflight_only:
            point=np.stack(points);sample=np.stack(samples,axis=1)
            comparisons,changes=make_contrasts(point,sample)
            helper.write_csv(out/'model_summary.csv',all_summaries);helper.write_csv(out/'per_seed_metrics.csv',seed_rows)
            helper.write_csv(out/'representation_comparisons.csv',comparisons);helper.write_csv(out/'paired_changes_from_reference.csv',changes)
            helper.write_csv(out/'unknown_category_audit.csv',feature_rows)
            helper.write_npz(out/'bootstrap_metrics.npz',point=point,samples=sample,conditions=np.array(CONDITIONS),representations=np.array([r for r,f in GROUPS]),families=np.array([f for r,f in GROUPS]),metrics=np.array(METRICS))
        reader49.recheck();reader59.recheck()
        for path,h in canary['observed'].items():require(digest(path)==h,'Original validation source changed during execution')
        require(digest(__file__)==protocol['script_sha256'] and digest(helper_path)==S60_SHA and digest(base_path)==p59['script_sha256'],'Experiment/helper code changed during execution')
        require(digest(root/'shared_nominal.py')==base.PROJECTION_SHA and digest(root/'shared_context_v4.py')==base.CONTEXT_SHA,'Mapper source changed during execution')
        helper.write_json(out/'input_audit.json',dict(step49_consumed_sha256=reader49.observed,step59_consumed_sha256=reader59.observed,original_validation_sources=canary['observed'],all_rechecked_at_end=True,source_integrity_scope='Consumed artifacts bound to stored manifests; SHA integrity is not an independent authenticity or external-validity guarantee'))
        status='REPRESENTATION_SHIFT_PREFLIGHT_COMPLETE' if args.preflight_only else 'REPRESENTATION_SHIFT_COMPLETE'
        archive=package(helper,out,status,dict(protocol_identity=protocol['identity'],elapsed_seconds=time.perf_counter()-started,training=False,test_used_for_selection=False))
        print(status+'\nReview ZIP: '+str(archive),flush=True);return 0
    except Exception as exc:
        print('STOPPED: '+type(exc).__name__+': '+str(exc),file=sys.stderr,flush=True)
        if created and out is not None and helper is not None:
            try:
                helper.write_json(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
                archive=package(helper,out,'FAILED',dict(elapsed_seconds=time.perf_counter()-started))
                print('Diagnostic ZIP: '+str(archive),file=sys.stderr)
            except Exception as packaging:print('Diagnostic packaging failed: '+str(packaging),file=sys.stderr)
        return 1

if __name__=='__main__':sys.exit(main())
