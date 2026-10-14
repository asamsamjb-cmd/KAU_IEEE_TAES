#!/usr/bin/env python3
"""Step59 v2: matched raw-monitor versus Semantic22 baselines (CPU only).

v2 adds RawScoped: the same raw allowlist restricted to the per-window
single-path observation contract of the pinned mapper (link channels kept only
for the link the window path selects; other links' channels masked to 0 with
link_observed_* indicators). Raw (all four links) is kept and reported as the
all-link arm. Prespecified L10 contrasts: Semantic22-RawScoped (matched scope),
RawScoped-Raw (scope effect), Semantic22-Raw (all-link, as in v1).

Place beside revised41, shared_nominal.py and the existing dataset directory.
    python "59.raw_semantic_representation_baseline.py"
    python "59.raw_semantic_representation_baseline.py" --preflight-only

No regeneration, GPU, edits to earlier outputs, test-based tuning or filtering.
Default core: Raw / RawScoped / Semantic22 x HGB / MLP x L1 / L10 x five fixed seeds.
Raw is simulated measurements + observable monitor properties, NOT RF I/Q.
Two benign-origin bits are coalesced into cause-neutral monitor properties.
Every actual AND counterfactual row must reconstruct the original 22 tokens.
Counterfactuals and scenario metadata are audit-only, never learning inputs.
L10 uses mean/std(ddof=0)/min/max/last of numeric and one-hot window features.
This compares matched tabular pipelines; it does not isolate TST or preserve
all temporal order. Raw summary statistics are not an information upper bound.

Default fixed output: outputs/shared_nominal_v2/59_raw_semantic/run_raw_semantic_v2 (v1 is left untouched)
Rerun the identical command to reuse verified completed training jobs.
--include-combined must be chosen before test evaluation; it changes identity.
--run-name selects a separate run if deliberately changing the protocol.
Models are retained locally; the review ZIP excludes joblib model files.
Runtime is hardware/data dependent; no 30-60-minute completion promise.
"""
from __future__ import annotations
import argparse, ast, copy, gc, hashlib, importlib.util, json, os, platform
import sys, time, traceback, warnings, zipfile, tempfile, shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

VERSION='shared_nominal_raw_semantic_v2'
SEEDS=tuple(range(20260610,20260615))
LABELS=('NORMAL','RF_LINK_JAMMING','GNSS_SPOOFING','UNAUTHORIZED_TELECOMMAND_INJECTION',
        'TELECOMMAND_REPLAY','GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE')
META=('scenario_id','episode_id','episode_type','episode_step','window_id','timestamp','label')
FIELDS=tuple('observation_stage path_type direction source_role destination_role protocol_context operational_phase message_type service_context command_intent payload_or_parameter_context sequence_freshness_status control_security_state signal_quality_state tracking_stability navigation_consistency timing_consistency decoder_status command_execution_result retry_timeout_state session_state state_transition_consistency'.split())
RAW_CATEGORICAL=tuple('raw_'+f for f in FIELDS[:11])
RAW_KEYS=tuple(sorted('accepted_flag accepted_sequence_flag auth_present auth_unverified_flag auth_valid blocked_flag clock_drift_ppm command_timeout_flag command_unknown_flag delayed_transition_flag duplicate_flag executed_flag failed_flag fix_valid forbidden_transition_flag gnss_carrier_lock_ratio gnss_cn0_mean_dbhz gnss_cycle_slip_count gnss_lock_loss_count gnss_tracking_jitter held_flag incomplete_transition_flag integrity_unverified_flag integrity_valid iot_crc_error_count iot_decode_fail_count iot_frame_loss_rate iot_retry_count iot_snr_db iot_timeout_count isl_crc_error_count isl_decode_fail_count isl_frame_loss_rate isl_retry_count isl_snr_db isl_timeout_count link_degraded_flag link_loss_flag policy_inconsistent_flag position_jump_m pseudorange_residual_m raw_command_count raw_command_intent raw_destination_role raw_direction raw_message_type raw_observation_stage raw_operational_phase raw_path_type raw_payload_or_parameter_context raw_protocol_context raw_service_context raw_source_role rejected_flag sequence_gap session_active session_expired_flag session_required_flag session_reset_count session_valid source_authorized source_unverified_benign_flag source_unverified_flag state_changed_flag time_residual_ns transition_allowed ttc_crc_error_count ttc_decode_fail_count ttc_frame_loss_rate ttc_retry_count ttc_snr_db ttc_timeout_count uav_crc_error_count uav_decode_fail_count uav_frame_loss_rate uav_retry_count uav_sinr_db uav_timeout_count unexpected_transition_flag unknown_sequence_flag valid_looking_benign_flag velocity_jump_mps'.split()))
ORIGIN_FLAGS=('source_unverified_benign_flag','valid_looking_benign_flag')
RAW_FIELDS=tuple(k for k in RAW_KEYS if k not in ORIGIN_FLAGS)
RAW_NUMERIC=tuple(k for k in RAW_FIELDS if k not in RAW_CATEGORICAL)
# Single-path observation contract of the pinned mapper (normalize_signal_quality,
# normalize_decoder_status, normalize_retry_timeout read ONE link chosen by raw_path_type;
# GNSS_LEO / INTERNAL_ONBOARD read no link channel). GNSS monitors, command/session/
# transition flags, counts and context categories are read for every window.
LINK_OF_PATH={'GATEWAY_LEO':'ttc','LEO_LEO':'isl','IOT_LEO':'iot','UAV_LEO':'uav'}
NO_LINK_PATHS=('GNSS_LEO','INTERNAL_ONBOARD')
LINK_COLUMNS={'ttc':('ttc_snr_db','ttc_frame_loss_rate','ttc_crc_error_count','ttc_decode_fail_count','ttc_retry_count','ttc_timeout_count'),
              'isl':('isl_snr_db','isl_frame_loss_rate','isl_crc_error_count','isl_decode_fail_count','isl_retry_count','isl_timeout_count'),
              'iot':('iot_snr_db','iot_frame_loss_rate','iot_crc_error_count','iot_decode_fail_count','iot_retry_count','iot_timeout_count'),
              'uav':('uav_sinr_db','uav_frame_loss_rate','uav_crc_error_count','uav_decode_fail_count','uav_retry_count','uav_timeout_count')}
SCOPE_INDICATORS=tuple('link_observed_'+p for p in ('ttc','isl','iot','uav'))
SCOPED_NUMERIC=RAW_NUMERIC+SCOPE_INDICATORS
SCOPE_POLICY=('Per-window single-path observation contract of the pinned mapper: the six channels of the four links '
              '(snr/sinr, frame_loss_rate, crc_error_count, decode_fail_count, retry_count, timeout_count) are kept only for the link '
              'selected by raw_path_type (GATEWAY_LEO->ttc, LEO_LEO->isl, IOT_LEO->iot, UAV_LEO->uav); channels of non-selected links are '
              'set to 0 and four link_observed_* indicators are added (all 0 for GNSS_LEO and INTERNAL_ONBOARD). GNSS monitors, '
              'command/session/transition flags, counts and the eleven context categories are common to every window and unchanged. '
              'This matches the information the 22-field projection can read, not a physical claim that other links are unobservable.')
PROJECTION_SHA='adfe6e2ea4268287efa593fb63e16ca84bc2dddc1bf2d22cd95ad360ca17d71d'
AUDIT41_SHA='bf5701921ffbf1fef79c38e57cfb7131b4a3406bd423a033b34744be76b6f729'
CONTEXT_SHA='b228affd6dbcd22c6a68143e5640b0dafbe5528bd77a52d420f1fcf066eea05c'
JOB_FILES=('model.joblib','history.csv','training_report.json','validation_predictions.npz')
DEFAULT_TRAINING=dict(hgb_max_iter=150,hgb_learning_rate=.08,hgb_max_leaf_nodes=31,
    hgb_min_samples_leaf=20,hgb_l2=1.,mlp_hidden=[128,64],mlp_batch_size=512,
    mlp_learning_rate=.001,mlp_alpha=.0001,mlp_max_epochs=120,mlp_min_epochs=10,
    mlp_patience=12,mlp_min_delta=.0001)

def require(value,message):
    if not bool(value): raise ValueError(message)
def utc():return datetime.now(timezone.utc).isoformat()
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def object_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    def scalar(x):
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=scalar)+'\n',encoding='utf-8');os.replace(tmp,path)
def write_csv(path,frame):
    path=Path(path);tmp=path.with_name(path.name+'.tmp');frame.to_csv(tmp,index=False,encoding='utf-8-sig');os.replace(tmp,path)
def write_npz(path,**arrays):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    os.replace(tmp,path)
def read_csv(path,**kwargs):return pd.read_csv(path,keep_default_na=False,encoding='utf-8-sig',**kwargs)
def import_path(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

def load_projection(root):
    root=Path(root);p=root/'shared_nominal.py'
    require(p.is_file() and sha256(p)==PROJECTION_SHA,'Current shared_nominal.py source hash differs; do not adapt to an unknown mapper silently.')
    require((root/'shared_context_v4.py').is_file() and sha256(root/'shared_context_v4.py')==CONTEXT_SHA,'Current shared_context_v4.py source hash differs')
    import_path(root/'shared_context_v4.py','shared_context_v4')
    module=import_path(p,'step59_shared_nominal')
    require(tuple(module.OBSERVABLE_FIELDS)==FIELDS,'Projection field order differs')
    # Validate transitive raw-key access; dynamic four message keys are explicit.
    tree=ast.parse(p.read_text(encoding='utf-8-sig'));functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    stack=['normalize_raw_to_observable'];seen=set();keys=set()
    while stack:
        fn=stack.pop()
        if fn in seen:continue
        seen.add(fn)
        for node in ast.walk(functions[fn]):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in functions:stack.append(node.func.id)
            if isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name) and node.value.id=='raw' and isinstance(node.slice,ast.Constant):keys.add(node.slice.value)
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='raw' and node.func.attr=='get':keys.add(node.args[0].value)
    keys.update('raw_'+f for f in FIELDS[7:11]);require(keys==set(RAW_KEYS),'Raw dependency allowlist differs from pinned mapper')
    return module

def sanitize_raw(frame,prefix=''):
    require(set(prefix+k for k in RAW_KEYS)<=set(frame),'Missing raw mapper dependencies: '+str(sorted(set(prefix+k for k in RAW_KEYS)-set(frame))))
    out=pd.DataFrame(index=frame.index)
    for k in RAW_KEYS:
        value=frame[prefix+k]
        if k in RAW_CATEGORICAL:
            require(value.notna().all(),'Null raw categorical: '+k)
            value=value.astype(str);require(value.str.len().gt(0).all() and value.eq(value.str.strip()).all(),'Blank/whitespace raw category: '+k)
        else:
            value=pd.to_numeric(value,errors='raise').astype(np.float64)
            require(np.isfinite(value).all(),'Nonfinite raw value: '+k)
            if k.endswith('_flag') or k in ('auth_present','auth_valid','integrity_valid','source_authorized','session_active','session_valid','fix_valid','transition_allowed'):
                require(value.isin([0,1]).all(),'Non-binary monitor status: '+k)
        out[k]=value
    out['source_unverified_flag']=np.maximum(out.source_unverified_flag,out.source_unverified_benign_flag)
    out['policy_inconsistent_flag']=np.maximum(out.policy_inconsistent_flag,out.valid_looking_benign_flag)
    return out[list(RAW_FIELDS)].reset_index(drop=True)

def scope_raw(frame):
    """Mask link channels outside the window's selected path; add observed-link indicators."""
    require(set(RAW_FIELDS)<=set(frame),'Scoped raw requires the sanitized raw allowlist')
    out=frame[list(RAW_FIELDS)].copy().reset_index(drop=True)
    path=out.raw_path_type.astype(str).to_numpy()
    unknown=sorted(set(path.tolist())-set(LINK_OF_PATH)-set(NO_LINK_PATHS))
    require(not unknown,'raw_path_type outside the scope policy (MULTI_PATH or unknown): '+str(unknown))
    for prefix,cols in LINK_COLUMNS.items():
        observed=np.array([LINK_OF_PATH.get(p)==prefix for p in path],dtype=bool)
        out['link_observed_'+prefix]=observed.astype(np.float64)
        out.loc[~observed,list(cols)]=0.
    return out[list(SCOPED_NUMERIC)+list(RAW_CATEGORICAL)]

def restore_projection_row(row):
    out=dict(row);out.update({k:0 for k in ORIGIN_FLAGS});return out

def audit_projection(frame,module):
    actual=sanitize_raw(frame);baseline=sanitize_raw(frame,'baseline_')
    need=list(FIELDS)+['baseline_observable_'+f for f in FIELDS]
    require(set(need)<=set(frame),'Missing counterfactual semantic projection')
    for prefix,data in [('',actual),('baseline_observable_',baseline)]:
        expected=frame[[prefix+f for f in FIELDS]].astype(str).to_numpy()
        for i,row in enumerate(data.to_dict('records')):
            obs=module.normalize_raw_to_observable(restore_projection_row(row))
            require(all(str(obs[f])==expected[i,j] for j,f in enumerate(FIELDS)),f'22-field projection mismatch in {prefix or "actual"} row {i}')
    semantic_same=(frame[list(FIELDS)].astype(str).to_numpy()==frame[['baseline_observable_'+f for f in FIELDS]].astype(str).to_numpy()).all(axis=1)
    raw_same=(actual.to_numpy()==baseline.to_numpy()).all(axis=1)
    require(not np.any(raw_same & ~semantic_same),'Identical allowed raw produced different semantic tokens')
    scoped_same=(scope_raw(actual).to_numpy()==scope_raw(baseline).to_numpy()).all(axis=1)
    require(not np.any(scoped_same & ~semantic_same),'Identical in-scope raw produced different semantic tokens; the scope policy misses a mapper input')
    require(not np.any(raw_same & ~scoped_same),'Identical allowed raw must also be identical in scope')
    if 'raw_observable_effect_present' in frame:
        require(np.array_equal(pd.to_numeric(frame.raw_observable_effect_present).to_numpy(),(~semantic_same).astype(int)),'Stored observable-effect flag differs')
    return actual,semantic_same,raw_same,scoped_same

def sequence_indices(frame,length):
    require(length>0 and len(frame)>0,'Empty/invalid sequence request');pieces=[]
    for _,g in frame.groupby('episode_id',sort=False):
        g=g.sort_values('episode_step',kind='stable');ix=g.index.to_numpy(dtype=np.int64)
        require(g.label.nunique()==1,'Mixed episode labels')
        require(np.all(np.diff(g.episode_step.to_numpy(dtype=np.int64))==1) and np.all(np.diff(g.window_id.to_numpy(dtype=np.int64))==1),'Sequence would bridge a gap')
        if len(g)>=length:pieces.append(np.lib.stride_tricks.sliding_window_view(ix,length).copy())
    return np.concatenate(pieces) if pieces else np.empty((0,length),dtype=np.int64)

class WindowEncoder:
    """Train-only vocabulary; explicit unknown channel, never ordinal averaging."""
    def __init__(self,numeric,categorical):self.numeric=list(numeric);self.categorical=list(categorical)
    def fit(self,frame):
        self.categories={k:sorted(frame[k].astype(str).unique().tolist()) for k in self.categorical}
        self.names=list(self.numeric)
        for k in self.categorical:self.names.extend([k+'='+v for v in self.categories[k]]+[k+'=<UNSEEN>'])
        return self
    def transform(self,frame):
        out=np.zeros((len(frame),len(self.names)),dtype=np.float64)
        if self.numeric:
            values=frame[self.numeric].to_numpy(dtype=np.float64);require(np.isfinite(values).all(),'Nonfinite numeric model input');out[:,:len(self.numeric)]=values
        j=len(self.numeric);rows=np.arange(len(frame))
        for k in self.categorical:
            vocab=self.categories[k];mapping={v:i for i,v in enumerate(vocab)}
            ix=frame[k].astype(str).map(mapping).fillna(len(vocab)).to_numpy(dtype=np.int64)
            out[rows,j+ix]=1.;j+=len(vocab)+1
        return out
    def record(self):return dict(numeric=self.numeric,categorical=self.categorical,categories=self.categories,window_feature_names=self.names,unknown_policy='one explicit unseen category per field')

def aggregate_windows(window_matrix,indices,length):
    require(len(indices)>0,'No eligible windows')
    if length==1:return window_matrix[indices[:,0]].copy()
    width=window_matrix.shape[1];out=np.empty((len(indices),width*5),dtype=np.float64)
    for start in range(0,len(indices),512):
        ix=indices[start:start+512];v=window_matrix[ix]
        out[start:start+len(ix)]=np.concatenate([v.mean(axis=1),v.std(axis=1,ddof=0),v.min(axis=1),v.max(axis=1),v[:,-1]],axis=1)
    return out

def confusion(y,p):return np.bincount(np.asarray(y,dtype=int)*6+np.asarray(p,dtype=int),minlength=36).reshape(6,6)
def macro_f1(cm):
    cm=np.asarray(cm,dtype=float);tp=np.diagonal(cm,axis1=-2,axis2=-1);den=cm.sum(axis=-2)+cm.sum(axis=-1)
    return np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0).mean(axis=-1)
def mean_seed_f1(cms):return float(np.mean(macro_f1(cms)))
def scores(y,p):
    cm=confusion(y,p);support=cm.sum(axis=1);recall=np.divide(np.diag(cm),support,out=np.zeros(6),where=support>0)
    return dict(n=len(y),macro_f1=float(macro_f1(cm)),balanced_accuracy=float(recall.mean()),accuracy=float(np.trace(cm)/max(1,cm.sum())),normal_false_alarm_rate=float(1-recall[0]) if support[0] else None,support=support.tolist(),confusion=cm.astype(int).tolist())

def fit_model(family,x,y,xv,yv,seed,settings):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import log_loss
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    require(set(np.unique(y))==set(range(6)),'Training requires all six classes')
    start=time.perf_counter();history=[];notes=[]
    if family=='HGB':
        model=HistGradientBoostingClassifier(max_iter=settings['hgb_max_iter'],learning_rate=settings['hgb_learning_rate'],max_leaf_nodes=settings['hgb_max_leaf_nodes'],min_samples_leaf=settings['hgb_min_samples_leaf'],l2_regularization=settings['hgb_l2'],early_stopping=False,random_state=seed)
        model.fit(x,y);prob=model.predict_proba(xv)
        history.append(dict(iteration=int(model.n_iter_),val_macro_f1=float(macro_f1(confusion(yv,prob.argmax(1)))),val_log_loss=float(log_loss(yv,prob,labels=np.arange(6)))))
        report=dict(selection='Fixed iteration budget; no validation or test selection',best_iteration=int(model.n_iter_),budget_exhausted=False,stopping='PRESPECIFIED_FIXED_BUDGET')
    else:
        require(family=='MLP','Unknown model family')
        scale=StandardScaler();xt=scale.fit_transform(x);vt=scale.transform(xv)
        net=MLPClassifier(hidden_layer_sizes=tuple(settings['mlp_hidden']),activation='relu',solver='adam',alpha=settings['mlp_alpha'],batch_size=min(settings['mlp_batch_size'],len(y)),learning_rate_init=settings['mlp_learning_rate'],random_state=seed,early_stopping=False,max_iter=1,shuffle=True)
        best=None;best_f=-np.inf;best_loss=np.inf;plateau_best=-np.inf;stale=0;best_epoch=0
        for epoch in range(1,settings['mlp_max_epochs']+1):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');net.partial_fit(xt,y,classes=np.arange(6))
            notes.extend(dict(category=w.category.__name__,message=str(w.message)) for w in caught)
            prob=net.predict_proba(vt);f=float(macro_f1(confusion(yv,prob.argmax(1))));loss=float(log_loss(yv,prob,labels=np.arange(6)))
            require(np.isfinite(loss) and np.isfinite(net.loss_),'Nonfinite MLP fit')
            history.append(dict(iteration=epoch,train_loss=float(net.loss_),val_macro_f1=f,val_log_loss=loss))
            if f>best_f+1e-12 or (abs(f-best_f)<=1e-12 and loss<best_loss):best=copy.deepcopy(net);best_f=f;best_loss=loss;best_epoch=epoch
            if f>plateau_best+settings['mlp_min_delta']:plateau_best=f;stale=0
            else:stale+=1
            if epoch==1 or epoch%5==0:print(f'    epoch {epoch}: val F1={f:.6f}',flush=True)
            if epoch>=settings['mlp_min_epochs'] and stale>=settings['mlp_patience']:break
        model=Pipeline([('scale',scale),('classifier',best)])
        plateau=epoch>=settings['mlp_min_epochs'] and stale>=settings['mlp_patience']
        report=dict(selection='External episode-separated validation macro-F1; tie lower validation logloss',best_iteration=best_epoch,epochs_completed=epoch,budget_exhausted=not plateau,stopping='VALIDATION_PLATEAU' if plateau else 'MAX_EPOCHS')
    require(np.array_equal(model.classes_,np.arange(6)),'Unexpected classifier probability order')
    report.update(internal_random_validation=False,training_seconds=time.perf_counter()-start,warnings=notes,seed=int(seed),family=family)
    return model,history,report

def verify_job(folder,identity):
    p=Path(folder)/'complete.json'
    if not p.exists():return False
    d=read_json(p);require(d['identity']==identity,'Completed job protocol differs: '+str(folder))
    require(isinstance(d.get('files'),dict) and set(d['files'])==set(JOB_FILES),'Completed job inventory is incomplete or unexpected: '+str(folder))
    for rel,h in d['files'].items():
        require(isinstance(h,str) and len(h)==64 and all(c in '0123456789abcdef' for c in h),'Invalid checkpoint SHA256')
        f=(Path(folder)/rel).resolve();require(f.is_relative_to(Path(folder).resolve()) and f.is_file() and sha256(f)==h,'Completed job file hash differs: '+str(f))
    return True

def episode_bootstrap(y,episodes,predictions,contrasts,draws,seed):
    """Whole class-stratified episodes, shared draws, average per-seed F1."""
    y=np.asarray(y,dtype=int);ids,ei=np.unique(np.asarray(episodes),return_inverse=True);n=len(ids)
    strata=[]
    for cls in range(6):
        chosen=np.unique(ei[y==cls]);require(len(chosen)>0,'No bootstrap episodes in class '+str(cls));strata.append(chosen)
        require(all(np.all(y[ei==e]==cls) for e in chosen),'Mixed-label bootstrap episode')
    names=list(predictions);counts=[];sizes=[]
    for name in names:
        pred=np.asarray(predictions[name]);require(pred.ndim==2 and pred.shape[1]==len(y),'Prediction shape mismatch')
        sizes.append(len(pred));per=[]
        for row in pred:
            require(np.isin(row,np.arange(6)).all(),'Invalid predicted class')
            per.append(np.bincount(ei*36+y*6+row.astype(int),minlength=n*36).reshape(n,36))
        counts.append(np.stack(per,axis=1))
    packed=np.concatenate(counts,axis=1);flat=packed.reshape(n,-1).astype(float);cuts=np.cumsum([0]+sizes)
    rng=np.random.default_rng(seed);dist=np.empty((draws,len(names)))
    for start in range(0,draws,100):
        b=min(100,draws-start);w=np.zeros((b,n))
        for ix in strata:w[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=b)
        cm=(w@flat).reshape(b,sum(sizes),6,6);f=macro_f1(cm)
        for i in range(len(names)):dist[start:start+b,i]=f[:,cuts[i]:cuts[i+1]].mean(axis=1)
    delta=np.column_stack([dist[:,names.index(a)]-dist[:,names.index(b)] for _,a,b in contrasts]) if contrasts else np.empty((draws,0))
    return dict(names=names,metrics=dist,contrast_names=[n for n,_,_ in contrasts],deltas=delta,episodes=n,seed_resampling=False)

def freeze_protocol(out,payload):
    protocol=dict(identity=object_hash(payload),**payload);p=Path(out)/'protocol.json'
    if p.exists():require(read_json(p)==protocol,'Resume protocol/input/source/runtime changed. Keep this run; use a new --run-name for a deliberate new protocol.')
    else:write_json(p,protocol)
    return protocol

def match_l1(predictions,l10_indices):return np.asarray(predictions)[l10_indices[:,-1]]

@contextmanager
def run_lock(path):
    """OS advisory lock: released by OS after a crash, no stale-PID killing."""
    f=Path(path).open('a+b')
    try:
        if f.tell()==0:f.write(b'0');f.flush()
        f.seek(0)
        if os.name=='nt':
            import msvcrt
            try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            except OSError as e:raise RuntimeError('This Step59 run is already open in another process.') from e
        else:
            import fcntl
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except OSError as e:raise RuntimeError('This Step59 run is already open in another process.') from e
        yield
    finally:f.close()

def prepare_sources(root,out):
    """Existing Step41 validator, plus full cause-neutral raw/semantic audit."""
    candidates=[p for p in root.glob('41*.py') if sha256(p)==AUDIT41_SHA]
    require(candidates,'Place the current revised 41.benchmark_A_seed_duplicate_context_audit.py beside Step59.')
    audit_path=sorted(candidates)[0];audit=import_path(audit_path,'step59_audit41');audit.PINNED_INPUTS={}
    projection=load_projection(root)
    for p in (audit_path,root/'shared_nominal.py',root/'shared_context_v4.py'):audit.pin(p)
    dataset=root/'dataset';buildroot=dataset/'shared_nominal_v2'
    print('[Preflight] Verify existing Step7 sources, raw companions and split hashes ...',flush=True)
    build=audit.load_build_contract(dataset);source=[];features=[];sem_same=[];raw_same=[];scoped_same=[];catalog=[];source_reports=[]
    for entry in build['sources']:
        path=dataset/entry['file'];learning=audit.validate_frame(audit.read_csv(path),entry['file'],entry['label'])
        rawpath=dataset/entry['raw_audit']['file'];header=read_csv(rawpath,nrows=0).columns.tolist()
        need=set(META+FIELDS+RAW_KEYS)|{'baseline_'+k for k in RAW_KEYS}|{'baseline_observable_'+f for f in FIELDS}|{'raw_observable_effect_present','raw_paired_audit_background'}
        require(need<=set(header),'Raw source missing columns: '+str(sorted(need-set(header))))
        for k in header:
            if k in ORIGIN_FLAGS:role='COALESCED_NOT_SEPARATE_INPUT';reason='Cause provenance removed; OR merged into generic monitor status'
            elif k in RAW_FIELDS:role='MODEL_RAW';reason='Pinned mapper dependency; simulated measurement or observable monitor/context property'
            elif k in FIELDS:role='MODEL_SEMANTIC_ONLY';reason='Never copied into raw feature matrix'
            elif k.startswith('baseline_'):role='AUDIT_ONLY';reason='Privileged paired nominal counterfactual'
            elif k in META:role='METADATA_ONLY';reason='Labels, ordering or identity; not a model feature'
            else:role='EXCLUDED';reason='Outside the explicit mapper-input allowlist; no implicit numeric-column selection'
            catalog.append(dict(source=entry['file'],column=k,role=role,reason=reason))
        offset=0;effect_count=0;raw_change=0;scoped_change=0
        for chunk in read_csv(rawpath,usecols=lambda c:c in need,dtype=str,chunksize=5000):
            chunk=chunk.reset_index(drop=True);expected=learning.iloc[offset:offset+len(chunk)].reset_index(drop=True)
            require(len(expected)==len(chunk),'Raw companion has extra rows')
            for k in META+FIELDS:require(np.array_equal(chunk[k].astype(str),expected[k].astype(str)),'Raw/learning identity or value mismatch: '+k)
            require(pd.to_numeric(chunk.raw_paired_audit_background).eq(0).all(),'Main A requires independently generated backgrounds, not paired stress-test data')
            clean,sem,prim,scoped=audit_projection(chunk,projection)
            features.append(clean);sem_same.append(sem);raw_same.append(prim);scoped_same.append(scoped);offset+=len(chunk)
            effect_count+=int((~sem).sum());raw_change+=int((~prim).sum());scoped_change+=int((~scoped).sum())
        require(offset==len(learning),'Raw companion missing rows')
        source.append(learning);source_reports.append(dict(label=entry['label'],rows=offset,semantic_changed=effect_count,raw_changed=raw_change,scoped_raw_changed=scoped_change,roundtrip='PASS'))
        print('  '+entry['label']+f': {offset:,} rows; raw/22-field roundtrip PASS',flush=True)
    source=pd.concat(source,ignore_index=True);raw=pd.concat(features,ignore_index=True);del features
    sem_same=np.concatenate(sem_same);raw_same=np.concatenate(raw_same);scoped_same=np.concatenate(scoped_same)
    splits={s:audit.validate_frame(audit.read_csv(buildroot/'splits'/f),s) for s,f in audit.SPLIT_FILES.items()}
    unified=audit.validate_frame(audit.read_csv(buildroot/'unified_observable_dataset.csv'),'unified')
    episode_manifest=audit.read_csv(buildroot/'splits/episode_split_manifest.csv')
    integrity,_=audit.split_integrity_audit(splits,source,build,episode_manifest,unified)
    expected=source.copy()
    if integrity['source_id_rewrite_applied']:expected['episode_id']=expected.label+'::'+expected.episode_id
    arrays={};scoped={};indices={};masks={};alignment={}
    for split,frame in splits.items():
        keys=['label','episode_id','episode_step','window_id']
        positions=pd.MultiIndex.from_frame(expected[keys]).get_indexer(pd.MultiIndex.from_frame(frame[keys]))
        require((positions>=0).all(),'Split contains an unknown raw-source record')
        audit.compare_exact_records(expected.iloc[positions].reset_index(drop=True),frame,'Step59 source -> '+split)
        arrays[split]=raw.iloc[positions].reset_index(drop=True);scoped[split]=scope_raw(arrays[split])
        indices[split]={n:sequence_indices(frame,n) for n in (1,10)}
        require(len(indices[split][10])==build['counts'][split]['sequences_stride1']['10'],'L10 endpoint count differs')
        require(len(indices[split][1])==len(frame) and np.array_equal(indices[split][1][:,0],np.arange(len(frame))), 'Existing split row order must follow episode/time order')
        masks[split]=dict(semantic_no_change=sem_same[positions],raw_no_change=raw_same[positions],scoped_no_change=scoped_same[positions])
        stored=audit.read_csv(buildroot/'audit'/f'{split}_window_observable_effects.csv')
        for k in META:require(np.array_equal(stored[k].astype(str),frame[k].astype(str)),'Step7 audit identity mismatch')
        require(np.array_equal(pd.to_numeric(stored.audit_observable_effect_present),~sem_same[positions]),'Step7 audit mask differs')
        alignment[split]=dict(rows=len(frame),episodes=frame.episode_id.nunique(),l10_endpoints=len(indices[split][10]),semantic_no_change=int(sem_same[positions].sum()),raw_no_change=int(raw_same[positions].sum()),scoped_no_change=int(scoped_same[positions].sum()))
        endpoint=frame.iloc[indices[split][10][:,-1]][list(META)].reset_index(drop=True)
        endpoint.insert(0,'sample_index',np.arange(len(endpoint)));endpoint['source_row_index']=indices[split][10][:,-1]
        endpoint['sequence_start_step']=frame.iloc[indices[split][10][:,0]].episode_step.to_numpy()
        write_csv(out/f'endpoints_{split}_L10.csv',endpoint)
        write_npz(out/f'audit_masks_{split}.npz',**masks[split])
    write_csv(out/'raw_column_policy.csv',pd.DataFrame(catalog))
    report=dict(dataset_id=build['dataset_id'],original_mapper_dependencies=len(RAW_KEYS),model_raw_fields=len(RAW_FIELDS),numeric_fields=len(RAW_NUMERIC),categorical_fields=len(RAW_CATEGORICAL),raw_fields=list(RAW_FIELDS),scoped_numeric_fields=len(SCOPED_NUMERIC),scoped_fields=list(SCOPED_NUMERIC)+list(RAW_CATEGORICAL),scope_policy=SCOPE_POLICY,link_of_path=LINK_OF_PATH,link_columns={k:list(v) for k,v in LINK_COLUMNS.items()},coalescing={'source_unverified_flag':'source_unverified_flag OR source_unverified_benign_flag','policy_inconsistent_flag':'policy_inconsistent_flag OR valid_looking_benign_flag'},sources=source_reports,alignment=alignment,split_integrity=integrity,interpretation='Synthetic monitor measurements and metadata available to the mapper, not actual RF samples or an independently validated security monitor. Raw (all-link), scoped-raw and semantic equality are distinct audit strata.')
    write_json(out/'input_audit.json',report)
    return dict(root=root,build=build,audit=audit,frames=splits,raw=arrays,scoped=scoped,indices=indices,masks=masks,report=report,inputs=dict(audit.PINNED_INPUTS))

def group_input(env,representation,split):
    if representation=='Raw':return env['raw'][split]
    if representation=='RawScoped':return env['scoped'][split]
    if representation=='Semantic22':return env['frames'][split][list(FIELDS)]
    return pd.concat([env['raw'][split],env['frames'][split][list(FIELDS)]],axis=1)

def feature_encoder(env,representation):
    numeric=list(RAW_NUMERIC) if representation in ('Raw','RawPlusSemantic22') else list(SCOPED_NUMERIC) if representation=='RawScoped' else []
    categorical=(list(RAW_CATEGORICAL) if representation in ('Raw','RawScoped','RawPlusSemantic22') else [])+(list(FIELDS) if representation in ('Semantic22','RawPlusSemantic22') else [])
    return WindowEncoder(numeric,categorical).fit(group_input(env,representation,'train'))

def feature_matrix(env,representation,split,length,encoder,memory_gib):
    n=len(env['indices'][split][length]);width=len(encoder.names)*(5 if length==10 else 1)
    # Lower-bound working estimate including validation/scaled/copy arrays; fail clearly.
    estimate=n*width*8*4
    require(estimate<=memory_gib*2**30,f'Estimated matrix working set {estimate/2**30:.2f} GiB exceeds --memory-budget-gib {memory_gib}. Increase only if RAM permits; no data subsampling is performed.')
    window=encoder.transform(group_input(env,representation,split))
    return aggregate_windows(window,env['indices'][split][length],length)

def resource_preflight(env,representations,out,args):
    rows=[]
    for representation in representations:
        enc=feature_encoder(env,representation)
        for length in (1,10):
            width=len(enc.names)*(5 if length==10 else 1)
            for split in ('train','val','test'):
                n=len(env['indices'][split][length]);gib=n*width*8*4/2**30
                rows.append(dict(representation=representation,length=length,split=split,rows=n,features=width,estimated_working_gib=gib))
    write_json(out/'resource_preflight.json',dict(memory_budget_gib=args.memory_budget_gib,scope='Conservative four-matrix feature working estimate, not a hard total-process RAM cap; also allow RAM for source frames, learner internals and the OS',matrices=rows))
    peak=max(r['estimated_working_gib'] for r in rows)
    require(peak<=args.memory_budget_gib,f'Preflight working estimate needs {peak:.2f} GiB, above --memory-budget-gib {args.memory_budget_gib}. Review available RAM; increase the flag only if sufficient. No training has started.')
    print(f'[Preflight] Largest feature working estimate {peak:.2f} GiB; budget {args.memory_budget_gib:g} GiB (not a total RAM cap).',flush=True)

def targets(env,split,length):return env['frames'][split].label.map(dict(zip(LABELS,range(6)))).to_numpy(dtype=np.int64)[env['indices'][split][length][:,-1]]

def align_reference(frame,env):
    """Align authenticated predictions by episode + terminal step, not row number."""
    endpoint=env['frames']['test'].iloc[env['indices']['test'][10][:,-1]].reset_index(drop=True)
    step='episode_step' if 'episode_step' in frame else 'sequence_end_step'
    require({'episode_id',step}<=set(frame),'Reference endpoint keys missing')
    observed=pd.DataFrame(dict(episode_id=frame.episode_id.astype(str),episode_step=pd.to_numeric(frame[step],errors='raise')))
    wanted=pd.DataFrame(dict(episode_id=endpoint.episode_id.astype(str),episode_step=endpoint.episode_step.astype(int)))
    index=pd.MultiIndex.from_frame(observed);target=pd.MultiIndex.from_frame(wanted)
    require(index.is_unique and len(index)==len(target),'Reference endpoint population differs or has duplicates')
    pos=index.get_indexer(target);require((pos>=0).all(),'Reference endpoint identity differs');frame=frame.iloc[pos].reset_index(drop=True)
    for key in META:
        if key in frame:require(np.array_equal(frame[key].astype(str),endpoint[key].astype(str)),'Reference metadata differs: '+key)
    for key,wanted_values in [('sequence_length',np.full(len(frame),10)),('sequence_start_step',endpoint.episode_step.astype(int).to_numpy()-9)]:
        if key in frame:require(np.array_equal(pd.to_numeric(frame[key]).to_numpy(),wanted_values),'Reference window context differs: '+key)
    truth=[k for k in ('y_true','true_label_id','label_id') if k in frame];pred=[k for k in ('y_pred','pred_label_id') if k in frame]
    require(truth and pred,'Reference class IDs missing');y=targets(env,'test',10)
    for k in truth:require(np.array_equal(pd.to_numeric(frame[k]).to_numpy(),y),'Reference true-label mapping differs')
    result=pd.to_numeric(frame[pred[0]],errors='raise').to_numpy()
    require(np.isin(result,np.arange(6)).all(),'Invalid reference predicted class')
    for k in pred:require(np.array_equal(pd.to_numeric(frame[k]).to_numpy(),result),'Reference prediction aliases differ')
    if 'pred_label' in frame:require(np.array_equal(frame.pred_label.astype(str),np.array(LABELS)[result.astype(int)]),'Reference predicted label differs')
    return result.astype(np.int8)

def discover_references(env,out,args):
    """Bind existing results before training; never select the newest/best run."""
    root=env['root'];dataset=env['build']['dataset_id'];inventory=[];jobs=[]
    def resolve(p):return (root/p).resolve() if not p.is_absolute() else p.resolve()
    def bind_file(folder,name,expected):
        p=(folder/name).resolve();require(p.is_relative_to(folder.resolve()),'Reference path escapes run')
        require(isinstance(expected,str) and len(expected)==64 and p.is_file() and sha256(p)==expected,'Reference file missing/hash differs: '+str(p))
        env['inputs'][str(p)]=expected;return p
    def remember(p):env['inputs'][str(p.resolve())]=sha256(p)
    run46=resolve(args.run46) if args.run46 else root/'outputs/shared_nominal_v2/46_encoder_comparison_frozen_test/run_formal_5seed'
    rule_paths=[resolve(args.rule_run)] if args.rule_run else []
    if not args.rule_run:
        for p in sorted((root/'outputs/shared_nominal_v2/16_observable_rule_baseline').glob('run_*/run_manifest.json')):
            try:
                m=read_json(p);contract=m.get('source_contract',m.get('dataset_contract',{}))
                if m.get('status')=='COMPLETE' and contract.get('dataset_id')==dataset:rule_paths.append(p.parent)
            except (OSError,ValueError):continue
    for kind,paths,explicit in [('Step46',[run46] if run46.is_dir() else [],bool(args.run46)),('Rule16',rule_paths,bool(args.rule_run))]:
        if len(paths)!=1:
            require(not explicit,kind+' explicit reference path does not resolve to one run')
            inventory.append(dict(source=kind,status='NOT_USED_MISSING' if not paths else 'NOT_USED_AMBIGUOUS',paths=[str(p) for p in paths]));continue
        folder=paths[0];manifest_path=folder/'run_manifest.json';m=read_json(manifest_path);remember(manifest_path)
        if kind=='Step46':
            require(m.get('status')=='FROZEN_TEST_AND_BOOTSTRAP_COMPLETE' and m.get('dataset_id')==dataset,'Step46 reference is incomplete or from another dataset')
            hashes=m.get('output_sha256',{});p=read_json(bind_file(folder,'protocol.json',hashes.get('protocol.json')))
            require(p.get('identity')==m.get('protocol_identity') and p.get('dataset_id')==dataset,'Step46 reference protocol binding differs')
            require(p['identity']==object_hash({k:v for k,v in p.items() if k!='identity'}),'Step46 reference protocol digest differs')
            test_path=root/'dataset/shared_nominal_v2/splits/test_observable.csv'
            require(p.get('test_sha256')==env['inputs'][str(test_path.resolve())],'Step46 reference test split differs')
            for condition in ('CategoricalEmbedding_MLP','DistilBERT_Pretrained_Finetune'):
                for seed in SEEDS:
                    chosen=[j for j in p['jobs'] if j['condition']==condition and j['seed']==seed];require(len(chosen)==1,'Step46 reference requires exactly the five fixed seeds')
                    name=chosen[0]['relative_dir']+'/test_predictions_L10.csv';path=bind_file(folder,name,hashes.get(name))
                    frame=read_csv(path)
                    expected=dict(dataset_id=dataset,condition=condition,encoder_seed=seed,head_kind='tst',evaluation_population='common_L10_endpoints',step46_protocol_identity=p['identity'])
                    for column,value in expected.items():require(column in frame and frame[column].astype(str).eq(str(value)).all(),'Step46 reference prediction provenance differs: '+column)
                    align_reference(frame,env)
                    jobs.append(dict(group='Reference46_'+condition+'_TST_L10',seed=seed,path=str(path),sha256=sha256(path)))
        else:
            contract=m.get('source_contract',m.get('dataset_contract',{}))
            require(m.get('status')=='COMPLETE' and contract.get('dataset_id')==dataset and m.get('calibrate_only') is not True,'Rule16 reference is incomplete or from another dataset')
            entries=[r for r in m.get('outputs',[]) if r.get('file')=='test_predictions_Rule_L10.csv'];require(len(entries)==1,'Rule16 reference prediction inventory differs')
            path=bind_file(folder,entries[0]['file'],entries[0]['sha256']);align_reference(read_csv(path),env)
            jobs.append(dict(group='Reference16_Rule_L10',seed=None,path=str(path),sha256=sha256(path)))
        inventory.append(dict(source=kind,status='VERIFIED_DESCRIPTIVE_REFERENCE',path=str(folder)))
    record=dict(inventory=inventory,jobs=jobs,role='Previously evaluated different pipelines. Descriptive context with episode CIs; excluded from primary representation contrasts and never a model-selection signal.')
    write_json(out/'reference_inventory.json',record)
    for entry in inventory:print('[Reference] '+entry['source']+': '+entry['status'],flush=True)
    return record

def evaluate_references(env,out,protocol):
    predictions={};rows=[]
    for job in protocol['references']['jobs']:
        path=Path(job['path']);require(sha256(path)==job['sha256'],'Bound reference changed: '+str(path))
        p=align_reference(read_csv(path),env);predictions.setdefault(job['group'],[]).append(p)
        rows.append(dict(group=job['group'],seed=job['seed'],**scores(targets(env,'test',10),p)))
    predictions={k:np.stack(v) for k,v in predictions.items()}
    write_json(out/'reference_metrics.json',rows)
    write_npz(out/'reference_predictions_L10.npz',**predictions,y_true=targets(env,'test',10).astype(np.int8))
    return predictions

def train_one(folder,identity,family,x,y,xv,yv,seed,settings):
    import joblib
    folder.mkdir(parents=True,exist_ok=True)
    if verify_job(folder,identity):print('  Reuse verified '+folder.name,flush=True);return read_json(folder/'training_report.json')
    model,history,report=fit_model(family,x,y,xv,yv,seed,settings)
    tmp=folder/'model.joblib.tmp';joblib.dump(model,tmp);os.replace(tmp,folder/'model.joblib')
    restored=joblib.load(folder/'model.joblib');prob=model.predict_proba(xv);replayed=restored.predict_proba(xv)
    require(np.array_equal(prob.argmax(1),replayed.argmax(1)) and np.allclose(prob,replayed,atol=1e-12,rtol=1e-12),'Saved model validation replay differs')
    report.update(validation_replay='PASS',validation=scores(yv,prob.argmax(1)),features=int(x.shape[1]),training_rows=len(y),validation_rows=len(yv))
    write_csv(folder/'history.csv',pd.DataFrame(history));write_json(folder/'training_report.json',report)
    write_npz(folder/'validation_predictions.npz',y_true=yv.astype(np.int8),y_pred=prob.argmax(1).astype(np.int8))
    files=JOB_FILES
    write_json(folder/'complete.json',dict(identity=identity,files={n:sha256(folder/n) for n in files},completed_at_utc=utc()))
    return report

def main_training(env,out,protocol,args):
    reps=protocol['representations'];settings=protocol['training'];reports=[]
    for representation in reps:
        enc=feature_encoder(env,representation);record=enc.record();write_json(out/f'encoder_{representation}.json',record)
        for length in (1,10):
            x=feature_matrix(env,representation,'train',length,enc,args.memory_budget_gib);xv=feature_matrix(env,representation,'val',length,enc,args.memory_budget_gib)
            y=targets(env,'train',length);yv=targets(env,'val',length)
            for family in ('HGB','MLP'):
                for seed in SEEDS:
                    key=f'{representation}_L{length}_{family}_seed{seed}';print('[Train] '+key,flush=True)
                    identity=object_hash(dict(protocol=protocol['identity'],key=key,encoder=record))
                    report=train_one(out/'jobs'/key,identity,family,x,y,xv,yv,seed,settings)
                    reports.append(dict(report,key=key,representation=representation,length=length,family=family,seed=seed))
            del x,xv;gc.collect()
    write_json(out/'all_training_complete.json',dict(protocol_identity=protocol['identity'],jobs=len(reports),test_scoring_started=False,reports=reports))
    return reports

def predict_batched(model,x):
    return np.concatenate([model.predict(x[i:i+4096]) for i in range(0,len(x),4096)]).astype(np.int8)

def evaluate_main(env,out,protocol,args):
    import joblib
    marker=read_json(out/'all_training_complete.json')
    require(marker['protocol_identity']==protocol['identity'] and marker['jobs']==len(protocol['representations'])*len(SEEDS)*4,'All main training must complete before test scoring')
    predictions={};metrics=[];strata_rows=[];test=env['frames']['test'];l10=env['indices']['test'][10]
    for representation in protocol['representations']:
        enc=feature_encoder(env,representation);record=enc.record()
        require(read_json(out/f'encoder_{representation}.json')==record,'Train-fitted feature vocabulary changed')
        for length in (1,10):
            x=feature_matrix(env,representation,'test',length,enc,args.memory_budget_gib);y=targets(env,'test',length)
            ix=env['indices']['test'][length]
            for family in ('HGB','MLP'):
                group=f'{representation}_L{length}_{family}';predictions[group]=[]
                for seed in SEEDS:
                    key=f'{group}_seed{seed}';folder=out/'jobs'/key
                    identity=object_hash(dict(protocol=protocol['identity'],key=key,encoder=record));require(verify_job(folder,identity),'Missing trained job')
                    print('[Test] '+key,flush=True);model=joblib.load(folder/'model.joblib');p=predict_batched(model,x)
                    write_npz(folder/'test_predictions.npz',y_true=y.astype(np.int8),y_pred=p)
                    paired=match_l1(p,l10) if length==1 else p;predictions[group].append(paired)
                    metrics.append(dict(group=group,seed=seed,population='common_L10_endpoints',**scores(targets(env,'test',10),paired)))
                    if length==1:metrics.append(dict(group=group,seed=seed,population='all_test_windows',**scores(y,p)))
                    masks={k:env['masks']['test'][k][ix].all(axis=1) for k in ('semantic_no_change','raw_no_change','scoped_no_change')}
                    masks['semantic_same_raw_changed']=masks['semantic_no_change'] & ~masks['raw_no_change']
                    masks['semantic_same_scoped_changed']=masks['semantic_no_change'] & ~masks['scoped_no_change']
                    masks['scoped_same_raw_changed']=masks['scoped_no_change'] & ~masks['raw_no_change']
                    for name,mask in masks.items():
                        strata_rows.append(dict(group=group,seed=seed,stratum=name,population='all_test_windows' if length==1 else 'common_L10_endpoints',**scores(y[mask],p[mask])))
                    del model
            del x;gc.collect()
    predictions={k:np.stack(v) for k,v in predictions.items()}
    write_json(out/'metrics_by_seed.json',metrics);write_csv(out/'metrics_by_seed.csv',pd.DataFrame([{k:v for k,v in r.items() if k not in ('support','confusion')} for r in metrics]))
    write_json(out/'conditional_main_metrics.json',strata_rows)
    write_npz(out/'paired_predictions_L10.npz',**predictions,y_true=targets(env,'test',10).astype(np.int8))
    return predictions,metrics

def fingerprint_audit(env,out,protocol,args):
    """Fixed first-seed conditional probes; no accuracy-based pass/fail or retries."""
    import joblib
    rows=[];settings=protocol['training'];seed=SEEDS[0]
    for representation,cohorts in (('Raw',('semantic_no_change','raw_no_change')),('RawScoped',('semantic_no_change','scoped_no_change'))):
        enc=feature_encoder(env,representation)
        for length in (1,10):
            matrices={s:feature_matrix(env,representation,s,length,enc,args.memory_budget_gib) for s in ('train','val','test')}
            ys={s:targets(env,s,length) for s in matrices}
            for cohort in cohorts:
                masks={s:env['masks'][s][cohort][env['indices'][s][length]].all(axis=1) for s in matrices}
                support={s:np.bincount(ys[s][masks[s]],minlength=6).tolist() for s in matrices}
                for family in ('HGB','MLP'):
                    key=f'{representation}_{cohort}_L{length}_{family}';print('[Conditional diagnostic] '+key,flush=True)
                    if min(support['train'])<5 or min(support['val'])<1 or min(support['test'])<1:
                        rows.append(dict(key=key,status='SKIPPED_INSUFFICIENT_CLASS_SUPPORT',support=support));continue
                    identity=object_hash(dict(protocol=protocol['identity'],probe=key,seed=seed,encoder=enc.record()))
                    folder=out/'probes'/key
                    report=train_one(folder,identity,family,matrices['train'][masks['train']],ys['train'][masks['train']],matrices['val'][masks['val']],ys['val'][masks['val']],seed,settings)
                    model=joblib.load(folder/'model.joblib');p=predict_batched(model,matrices['test'][masks['test']]);yt=ys['test'][masks['test']]
                    prior=np.bincount(ys['train'][masks['train']],minlength=6);majority=np.full(len(yt),prior.argmax())
                    row=dict(key=key,representation=representation,cohort=cohort,length=length,family=family,status='COMPLETED',support=support,training=report,metrics=scores(yt,p),train_majority_reference=scores(yt,majority),balanced_accuracy_chance_reference=1/6,interpretation='Above-chance performance is a diagnostic, not an automatic shortcut finding. Semantic-no-change may retain legitimate raw variation; raw-/scoped-no-change equality only concerns the allowed (all-link / in-scope) monitor inputs.')
                    rows.append(row);write_npz(folder/'test_predictions.npz',y_true=yt.astype(np.int8),y_pred=p,sequence_indices=np.flatnonzero(masks['test']))
                    write_json(out/'fingerprint_audit.json',rows);del model
            del matrices;gc.collect()
    write_json(out/'fingerprint_audit.json',rows)
    return rows

def statistical_summary(env,out,protocol,predictions,args):
    y=targets(env,'test',10);ix=env['indices']['test'][10][:,-1];episodes=env['frames']['test'].iloc[ix].episode_id.to_numpy()
    contrasts=[];families={}
    for length in (10,1):
        tier='primary' if length==10 else 'secondary'
        for model in ('HGB','MLP'):
            name=f'Semantic22_minus_RawScoped_L{length}_{model}';contrasts.append((name,f'Semantic22_L{length}_{model}',f'RawScoped_L{length}_{model}'));families[name]=f'{tier}_L{length}_matched_scope_two_models'
            name=f'RawScoped_minus_Raw_L{length}_{model}';contrasts.append((name,f'RawScoped_L{length}_{model}',f'Raw_L{length}_{model}'));families[name]=f'{tier}_L{length}_scope_effect_two_models'
            name=f'Semantic22_minus_Raw_L{length}_{model}';contrasts.append((name,f'Semantic22_L{length}_{model}',f'Raw_L{length}_{model}'));families[name]=f'{tier}_L{length}_all_link_two_models'
    if 'RawPlusSemantic22' in protocol['representations']:
        for length in (10,1):
            for model in ('HGB','MLP'):
                name=f'Combined_minus_Raw_L{length}_{model}';contrasts.append((name,f'RawPlusSemantic22_L{length}_{model}',f'Raw_L{length}_{model}'));families[name]='secondary_combined_four_contrasts'
    print(f'[Statistics] {args.bootstrap:,} paired whole-episode draws ...',flush=True)
    b=episode_bootstrap(y,episodes,predictions,contrasts,args.bootstrap,2026100759);summary=[];comparisons=[]
    for i,name in enumerate(b['names']):
        f=np.array([float(macro_f1(confusion(y,p))) for p in predictions[name]])
        lo,hi=np.quantile(b['metrics'][:,i],[.025,.975],method='linear')
        summary.append(dict(group=name,role='descriptive_existing_pipeline' if name.startswith('Reference') else 'matched_representation_baseline',seeds=len(f),macro_f1_mean=float(f.mean()),seed_sd_ddof1=float(f.std(ddof=1)) if len(f)>1 else None,episode_ci95_low=float(lo),episode_ci95_high=float(hi),endpoints=len(y),episodes=b['episodes']))
    points={r['group']:r['macro_f1_mean'] for r in summary}
    for i,(name,a,c) in enumerate(contrasts):
        size=sum(v==families[name] for v in families.values());lo,hi=np.quantile(b['deltas'][:,i],[.025,.975]);bl,bh=np.quantile(b['deltas'][:,i],[.025/size,1-.025/size])
        comparisons.append(dict(contrast=name,family=families[name],delta_macro_f1=points[a]-points[c],ci95_low=float(lo),ci95_high=float(hi),bonferroni_confidence=1-.05/size,bonferroni_low=float(bl),bonferroni_high=float(bh),interpretation='first_higher' if bl>0 else 'second_higher' if bh<0 else 'zero_included_not_equivalence'))
    write_csv(out/'model_summary.csv',pd.DataFrame(summary));write_csv(out/'paired_comparisons.csv',pd.DataFrame(comparisons))
    write_npz(out/'bootstrap_distribution.npz',model_names=np.array(b['names']),macro_f1=b['metrics'],contrast_names=np.array(b['contrast_names']),deltas=b['deltas'])
    write_json(out/'statistics.json',dict(draws=args.bootstrap,seed=2026100759,episodes=b['episodes'],endpoints=len(y),seed_resampling=False,estimand='Arithmetic mean of per-seed endpoint-level macro-F1; not ensemble or pooled-seed confusion',resampling='Whole episodes within each true class, same draw across all methods and fixed seeds',uncertainty='Conditional on these trained seeds, this synthetic realization and fixed protocol; assumes within-class episode exchangeability',equivalence_margin=None,primary_family='Two L10 Semantic22-minus-RawScoped contrasts (matched single-path scope); scope-effect (RawScoped-Raw), all-link (Semantic22-Raw), L1 and optional combined families reported separately',comparisons=comparisons))

def archive_run(out,manifest):
    excluded={'.run.lock','run_manifest.json'}
    files=[p for p in out.rglob('*') if p.is_file() and p.name not in excluded and not p.name.endswith('.tmp')]
    manifest.update(updated_at_utc=utc(),output_sha256={p.relative_to(out).as_posix():sha256(p) for p in files},review_excludes=['*.joblib model checkpoints'])
    write_json(out/'run_manifest.json',manifest);dest=out.with_name(out.name+('_FAILED_review.zip' if manifest['status']=='FAILED' else '_review.zip'));tmp=dest.with_name(dest.name+'.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[out/'run_manifest.json']:
            if p.suffix!='.joblib':z.write(p,p.relative_to(out).as_posix())
    os.replace(tmp,dest);return dest

def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--include-combined',action='store_true',help='Prespecified secondary Raw+Semantic22 arm; changes run identity')
    p.add_argument('--run-name',default='run_raw_semantic_v2')
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--memory-budget-gib',type=float,default=8.)
    p.add_argument('--bootstrap',type=int,default=20000)
    p.add_argument('--run46',type=Path,help='Optional existing completed Step46 run directory; default formal five-seed run')
    p.add_argument('--rule-run',type=Path,help='Optional existing completed Step16 run directory; otherwise unique matching run')
    return p

def main(argv=None):
    args=parser().parse_args(argv);root=args.project_root.expanduser().resolve()
    require(Path(args.run_name).name==args.run_name and args.run_name not in ('','.', '..'),'--run-name must be one directory name')
    require(args.threads>0 and np.isfinite(args.memory_budget_gib) and args.memory_budget_gib>0 and args.bootstrap>=1000,'Invalid CPU/memory/bootstrap setting')
    out=root/'outputs/shared_nominal_v2/59_raw_semantic'/args.run_name;out.mkdir(parents=True,exist_ok=True)
    with run_lock(out/'.run.lock'):
        manifest=dict(status='RUNNING',version=VERSION,started_at_utc=utc())
        bound=False;stage=Path(tempfile.mkdtemp(prefix='.step59_audit_',dir=out.parent))
        try:
            import sklearn,joblib
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=args.threads):
                print('Step59 '+VERSION+': matched CPU raw-monitor (all-link, single-path scoped) / Semantic22 comparison',flush=True)
                env=prepare_sources(root,stage)
                representations=['Raw','RawScoped','Semantic22']+(['RawPlusSemantic22'] if args.include_combined else [])
                resource_preflight(env,representations,stage,args)
                references=discover_references(env,stage,args)
                payload=dict(version=VERSION,script_sha256=sha256(Path(__file__)),dataset_id=env['build']['dataset_id'],inputs=env['inputs'],representations=representations,families=['HGB','MLP'],lengths=[1,10],seeds=list(SEEDS),training=DEFAULT_TRAINING,threads=args.threads,memory_budget_gib=args.memory_budget_gib,bootstrap=args.bootstrap,bootstrap_seed=2026100759,runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__,platform=platform.platform()),raw_fields=list(RAW_FIELDS),scoped_fields=list(SCOPED_NUMERIC),scope_policy=SCOPE_POLICY,link_of_path=LINK_OF_PATH,semantic_fields=list(FIELDS),raw_monitor_policy=env['report']['coalescing'],normalization_fit='Train only; test/val category vocabularies are not learned',temporal_features='Numeric and one-hot columns: mean,std(ddof0),min,max,last; identical operations for both representations',labels=list(LABELS),primary='Two paired L10 Semantic22-minus-RawScoped macro-F1 contrasts (HGB,MLP) under the single-path observation contract; RawScoped-minus-Raw (scope effect) and Semantic22-minus-Raw (all-link) are separate families; Bonferroni within family at 95%',selection='Fixed HGB budget, external existing validation for MLP; all main fits complete before main test scoring',audit='Sixteen fixed first-seed probes: Raw on semantic-/raw-no-change and RawScoped on semantic-/scoped-no-change cohorts x 2 lengths x 2 families; no test-based retry/pass threshold',limitations=['Synthetic measurements and monitor outputs, not raw RF or an external detector validation.','Deterministic semantics cannot add observation information; improvements indicate learner/feature interaction under this protocol.','L10 statistics lose temporal order; raw statistics are not an information upper bound.','No-change semantic and raw audit strata differ; above chance is not automatically leakage.','No equivalence/portability/operational validity claim follows from these accuracy comparisons.','Previously seen benchmark A test; this addition is a revision analysis with settings frozen before new test predictions.','The scope policy mirrors the pinned mapper; it is not a physical claim that other links are unobservable. The all-link Raw arm is retained and reported.'])
                payload['references']=references
                protocol=freeze_protocol(out,payload);bound=True
                manifest.update(protocol_identity=protocol['identity'],dataset_id=payload['dataset_id'])
                if args.preflight_only and (out/'run_manifest.json').is_file() and read_json(out/'run_manifest.json')['status']=='RAW_SEMANTIC_BASELINE_COMPLETE':
                    print('Preflight passed; existing completed run and its manifest are preserved.',flush=True);return 0
                for p in stage.iterdir():os.replace(p,out/p.name)
                (out/'error.txt').unlink(missing_ok=True)
                source=out/'source';source.mkdir(exist_ok=True)
                for p in [Path(__file__),root/'shared_nominal.py',root/'shared_context_v4.py']:
                    (source/p.name).write_bytes(p.read_bytes())
                if args.preflight_only:
                    manifest['status']='PREFLIGHT_COMPLETE';archive=archive_run(out,manifest)
                    print('Preflight passed; no classifiers fitted. Review ZIP: '+str(archive),flush=True);return 0
                reports=main_training(env,out,protocol,args)
                predictions,metrics=evaluate_main(env,out,protocol,args)
                probes=fingerprint_audit(env,out,protocol,args)
                predictions.update(evaluate_references(env,out,protocol))
                statistical_summary(env,out,protocol,predictions,args)
                for p,h in env['inputs'].items():require(sha256(p)==h,'Consumed source changed during run: '+p)
                require(sha256(Path(__file__))==payload['script_sha256'],'Step59 code changed during run')
                manifest.update(status='RAW_SEMANTIC_BASELINE_COMPLETE',completed_jobs=len(reports),test_endpoints=len(targets(env,'test',10)),probe_statuses=[p['status'] for p in probes],mlp_budget_exhausted=sum(bool(r['budget_exhausted']) for r in reports),training=True,test_used_for_selection=False,finished_at_utc=utc())
                archive=archive_run(out,manifest)
                print('Completed. Summary: '+str(out/'model_summary.csv')+'\nReview ZIP: '+str(archive),flush=True);return 0
        except (Exception,KeyboardInterrupt) as exc:
            diagnostic=out if bound else out.parent/'diagnostics'/(out.name+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
            diagnostic.mkdir(parents=True,exist_ok=True)
            if not bound:
                for p in stage.iterdir():shutil.copy2(p,diagnostic/p.name)
            (diagnostic/'error.txt').write_text(traceback.format_exc(),encoding='utf-8');manifest.update(status='FAILED',error_type=type(exc).__name__,error=str(exc),original_run_preserved=not bound)
            archive=archive_run(diagnostic,manifest)
            print('\nSTOPPED: '+str(exc)+'\nDiagnostic ZIP: '+str(archive)+'\nCompleted training jobs are retained. Correct the reported issue before resuming.',file=sys.stderr,flush=True);return 1
        finally:shutil.rmtree(stage,ignore_errors=True)

if __name__=='__main__':sys.exit(main())
