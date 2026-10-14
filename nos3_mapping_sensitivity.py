#!/usr/bin/env python3
"""NOS3 mapping sensitivity on the Windows research desktop, CPU only.

Uses the original Step09 campaign and the completed 1.3 detector result ZIP.
No training, live commands, downloads, threshold changes, or earlier-file edits.
--audit-only verifies archives and mappings without importing neural libraries.
"""
from __future__ import annotations
import argparse
import collections
import copy
import csv
import datetime as dt
import hashlib
import io
import json
import math
import shutil
import sys
import traceback
import types
import zipfile
from pathlib import Path, PurePosixPath

VERSION = '1.0-nos3-evidence-mapping-sensitivity'
MAPPING_VERSION = '08.2-restricted-evidence-mapping'
REFERENCE_NAME = 'run_20261005T144929_212195Z.zip'
REFERENCE_SHA = '68d3cbd8c407ca8366e096b757a34848767512aabe9482bc827e29773bc821db'
EVALUATOR_SHA = 'c7401887ce703831187b7a6d47244f3686f2cb8608dd6ecb686aec16d775d069'
FIELDS = ('observation_stage path_type direction source_role destination_role protocol_context '
          'operational_phase message_type service_context command_intent payload_or_parameter_context '
          'sequence_freshness_status control_security_state signal_quality_state tracking_stability '
          'navigation_consistency timing_consistency decoder_status command_execution_result '
          'retry_timeout_state session_state state_transition_consistency').split()
FORBIDDEN_IMPUTATION = ('operational_phase', 'payload_or_parameter_context', 'signal_quality_state',
                       'tracking_stability', 'navigation_consistency', 'timing_consistency',
                       'decoder_status', 'retry_timeout_state', 'session_state', 'state_transition_consistency')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def load_reference(path, out):
    """Only execute the exact previously reviewed evaluator, not arbitrary ZIP code."""
    require(Path(path).is_file(), f'Reviewed evaluator result ZIP not found: {path}')
    require(digest(path) == REFERENCE_SHA, 'Expected the completed 2026-10-05 14:49 UTC detector result ZIP')
    with zipfile.ZipFile(path) as archive:
        names = [i.filename for i in archive.infolist() if not i.is_dir()]
        require(len(names) == len(set(names)), 'Duplicate reference archive members')
        prefix = REFERENCE_NAME[:-4]+'/'
        manifest = json.loads(archive.read(prefix+'result_manifest.json'))
        require(set(names) == {prefix+n for n in manifest} | {prefix+'result_manifest.json'},
                'Reference archive inventory differs')
        for name, expected in manifest.items():
            require(hashlib.sha256(archive.read(prefix+name)).hexdigest() == expected, 'Reference member changed: '+name)
        summary = json.loads(archive.read(prefix+'summary.json'))
        require(summary.get('status') == 'EXTENDED_COMPLETE' and summary.get('code_version') == '1.3-nos3-boundary-audit',
                'The original extended detector evaluation did not complete')
        source = archive.read(prefix+'nos3_detector_evaluation.py')
        require(hashlib.sha256(source).hexdigest() == EVALUATOR_SHA, 'Unreviewed evaluator implementation')
        previous = {k: json.loads(archive.read(prefix+f)) for k, f in {
            'provenance':'model_provenance.json', 'input_sha256':'input_sha256.json'}.items()}
        for name in ('observable_windows', 'trajectory_predictions'):
            previous[name] = list(csv.DictReader(io.StringIO(archive.read(prefix+name+'.csv').decode('utf-8-sig'))))
    source_path = out/'verified_evaluator_1_3.py'
    source_path.write_bytes(source)
    module = types.ModuleType('nos3_reviewed_evaluator_1_3')
    module.__file__ = str(source_path)
    exec(compile(source, str(source_path), 'exec'), module.__dict__)
    require(module.FIELDS == FIELDS, 'Observable schema differs')
    module.pin(path, REFERENCE_SHA)
    module.pin(source_path, EVALUATOR_SHA)
    write_json(out/'historical_reference.json', dict(path=str(Path(path).resolve()), sha256=REFERENCE_SHA,
        evaluator_sha256=EVALUATOR_SHA, purpose='V1 reproduction and original checkpoint identity; no model selection'))
    return module, previous


def compare_csv_records(old, new):
    require(len(old) == len(new) and old == new, 'Regenerated V1 observable records differ from the reviewed result')
    return dict(status='PASS', records_checked=len(old))


def project_window(original, index, events, packets, ips):
    """Values depend only on within-window observations and declared logical roles.

    TC wins the representative-path choice when TC and TLM coexist. GATEWAY_LEO
    and the two roles are logical SIL endpoint abstractions, not sender identity,
    authentication, radio geometry, or a claim that replay followed RADIO.
    """
    require(set(original) == set(FIELDS), 'Expected exactly 22 observable fields')
    start, end = index['window_start'], index['window_end']
    require(end == start+1, 'Expected a complete one-second window')
    tc = int(index['tc_frames'])
    require(tc in (0, 1), 'Ambiguous TC multiplicity')
    event_count = sum(start <= e['wall'] < end for e in events)
    tlm, apids, hashes = [], set(), []
    for packet in packets:
        if not (packet.get('kind') == 'packet' and packet.get('protocol') == 'UDP'
                and packet.get('src') == ips['fsw'] and packet.get('dst') == ips['cosmos']
                and packet.get('dport') == 5013 and packet.get('packet_type') == 4
                and start <= packet['wall'] < end):
            continue
        payload = bytes.fromhex(packet['payload_hex'])
        require(not packet.get('payload_truncated') and len(payload) == packet['payload_bytes']
                and len(payload) >= 7, 'Incomplete candidate telemetry packet')
        header = int.from_bytes(payload[:2], 'big')
        require(header >> 13 == 0 and header & 0x1000 == 0
                and int.from_bytes(payload[4:6], 'big')+7 == len(payload), 'Invalid candidate telemetry header/length')
        tlm.append(packet)
        apids.add(header & 0x7ff)
        hashes.append(hashlib.sha256(payload).hexdigest())
    row = dict(original)
    if tc:
        kind = 'TC'
        row.update(path_type='GATEWAY_LEO', direction='GROUND_TO_SPACE',
                   source_role='GROUND_GATEWAY', destination_role='MONITORED_LEO')
    elif tlm:
        kind = 'TLM_WITH_COMMAND_EVENT' if event_count else 'TLM'
        row.update(path_type='GATEWAY_LEO', direction='SPACE_TO_GROUND', source_role='MONITORED_LEO',
                   destination_role='GROUND_GATEWAY', message_type='TELEMETRY', service_context='UNK')
        # Capture/NOOP-event completeness was authenticated by the original 08 verifier.
        # This absence applies only to that recorded TC ingress and native-event scope.
        if not event_count:
            row.update(command_intent='NO_COMMAND', sequence_freshness_status='NOT_APPLICABLE',
                       control_security_state='NOT_APPLICABLE', command_execution_result='NO_COMMAND')
    else:
        kind = 'EVENT_ONLY' if event_count else 'UNOBSERVED'
        # Without an evidence-bearing packet there is no representative direction.
        row.update(path_type='GATEWAY_LEO', direction='UNK', source_role='UNK', destination_role='UNK')
    for field in FORBIDDEN_IMPUTATION:
        require(row[field] == original[field], 'An unobserved state was imputed: '+field)
    evidence = dict(window_start=start, window_end=end, window_kind=kind, tc_frames=tc,
                    native_events=event_count, validated_tlm_packets=len(tlm),
                    tlm_apids=';'.join(str(i) for i in sorted(apids)),
                    tlm_payloads_sha256=hashlib.sha256('\n'.join(hashes).encode()).hexdigest() if hashes else None,
                    missing_fields=sum(v == 'UNK' for v in row.values()),
                    absence_scope='Captured TC ingress and reviewed native command-event stream only',
                    changed_fields=';'.join(f for f in FIELDS if row[f] != original[f]))
    return row, evidence


def build_v2(runs, schema):
    result, evidence, changes = [], [], []
    for run in runs:
        raw = run['raw']
        ips = json.loads(raw['files']['network_endpoints.json'])['endpoints']
        records = [json.loads(line) for line in raw['files']['sc01-nos-fsw_transport.jsonl'].splitlines() if line.strip()]
        view = copy.deepcopy(run['view'])
        for i, original in enumerate(run['view']['features']):
            row, info = project_window(original, view['index'][i], raw['events'], records, ips)
            require(all(v == 'UNK' or v in schema['vocabulary'][k] for k,v in row.items()), 'V2 value outside frozen train vocabulary')
            view['features'][i] = row
            view['without_history'][i] = dict(row, sequence_freshness_status='UNK')
            view['index'][i]['missing_fields'] = info['missing_fields']
            evidence.append(dict(run=run['name'], row_index=i, **info))
            for field in FIELDS:
                if original[field] != row[field]:
                    changes.append(dict(run=run['name'], row_index=i, window_start=info['window_start'],
                                        field=field, v1=original[field], v2=row[field], window_kind=info['window_kind']))
        item = dict(run, view=view)
        result.append(item)
    counts = collections.Counter(r['window_kind'] for r in evidence)
    missing = {kind: dict(sorted(collections.Counter(str(r['missing_fields']) for r in evidence if r['window_kind']==kind).items()))
               for kind in sorted(counts)}
    cross_second = 0
    for run in runs:
        noops = [e for e in run['raw']['events'] if e['kind']=='noop']
        require(len(noops) == len(run['view']['frames']), 'Command/native-event count differs')
        cross_second += sum(math.floor(f['receive_wall']) != math.floor(e['wall']) for f,e in zip(run['view']['frames'], noops))
    audit = dict(mapping_version=MAPPING_VERSION, total_windows=len(evidence), window_counts=dict(sorted(counts.items())),
                 unk_counts_by_kind=missing, first_frame_rule='UNK',
                 cold_start_unk_count=sum(r['view']['frames'][0]['sequence_freshness_status']=='UNK' for r in result),
                 command_receive_event_cross_second_count=cross_second,
                 validated_tlm_packets_in_complete_windows=sum(r['validated_tlm_packets'] for r in evidence),
                 unobserved_window_count=counts.get('UNOBSERVED', 0), evidence=evidence, changes=changes)
    return result, audit


def compare_predictions(old, new):
    def key(row):
        return tuple(str(row[k]) for k in ('run','endpoint_row','model','view','serialization'))
    left, right = {key(r):r for r in old}, {key(r):r for r in new}
    require(len(left)==len(old) and len(right)==len(new) and left.keys()==right.keys(), 'V1 prediction endpoint inventory changed')
    maximum = 0.0
    for k, a in left.items():
        b = right[k]
        require(a['prediction']==b['prediction'], 'V1 decision differs from the reviewed evaluation: '+str(k))
        for col in a:
            if not col.startswith(('prob_', 'rule_score_')) or a[col] in ('', None):
                continue
            x,y = float(a[col]),float(b[col]); delta=abs(x-y)
            require(math.isfinite(x) and math.isfinite(y), 'Nonfinite reproduction value')
            tolerance = 1e-12*(1+abs(x)) if col.startswith('rule_score_') else 1e-4*(1+abs(x))
            require(delta <= tolerance, 'V1 probability/score differs from reviewed evaluation: '+str(k))
            maximum=max(maximum,delta)
    return dict(status='PASS', prediction_rows=len(old), identical_decisions=True, max_abs_difference=maximum)


def verify_checkpoint_identity(module, context, previous):
    old = previous['provenance']
    require(old['source_contract']==context['contract'] and old['step11_run_id']==context['r11']['run_id']
            and old['tokenizer_sha256']==context['tokenizer_sha'], 'Original model/data identity changed')
    for length in (1,10):
        rel=f'L{length}/checkpoints/best_tst_model_L{length}.pt'
        hashes={h for p,h in previous['input_sha256'].items() if p.replace('\\','/').endswith('/'+rel)}
        require(len(hashes)==1, 'Original TST checkpoint identity is missing/ambiguous')
        module.pin(context['tst_root']/rel, next(iter(hashes)))


def write_protocol(module, out):
    protocol=dict(version=VERSION, mapping_version=MAPPING_VERSION, analysis='Post-hoc mapping sensitivity on previously evaluated NOS3 records',
        conditions=dict(mapping=['V1','V2'], freshness=['observed_partial','freshness_unk'],
                        serialization=['fixed_unk','omit_unknown_clauses']),
        models=['TST_L1','TST_L10','Rule_L10'], main_seed=20260610,
        rules='Structured inputs; evaluated once per mapping/freshness pair, not duplicated for textual serialization',
        roles='Declared logical ground/monitored-spacecraft abstraction at the captured SIL boundary; not authenticated sender identity or physical RF path',
        tc_path_priority='A captured TC frame makes TC the representative path, even when downlink packets coexist',
        telemetry='FSW source to recorded COSMOS peer, outbound UDP 5013, complete version-0 telemetry packet header and exact length',
        absence='No captured TC frame AND no reviewed native command event in the complete second; scoped absence only',
        cold_start='UNK retained; no prior sequence inferred', events='Native event stays in its observed second; no backward attribution',
        retained_unknown=list(FORBIDDEN_IMPUTATION),
        no_command_semantics='NO_COMMAND/NOT_APPLICABLE do not assert healthy authentication, decoder, session or state transitions',
        freshness_ablation='Remove the whole field, including TLM NOT_APPLICABLE; effect is not an isolated REPLAYED-token intervention',
        serialization='Omission changes field names, text length and positions as well as UNK tokens',
        observations='899 unique seconds; 30 command endpoints; all 809 common L10 endpoints; no independent-window statistical inference',
        truth='25 ordinary command endpoints and 5 controlled replay endpoints; other endpoints have no class truth',
        comparison='Same original fixed checkpoints and six-class argmax; V1 predictions must reproduce before V2 inference',
        scope='No new benchmark mask evaluation, model selection, training, live commands or operational-performance claim')
    module.write_json(out/'evaluation_protocol.json',protocol)
    rows=[]
    for field in FIELDS:
        rule='Retain V1 evidence or UNK; do not impute a normal state'
        if field in ('path_type','direction','source_role','destination_role'):
            rule='Configured logical endpoint roles and representative TC/TLM packet direction'
        elif field=='message_type':rule='Retain TELECOMMAND; set TELEMETRY only for a validated downlink packet'
        elif field=='service_context':rule='Retain TTC_CONTROL for TC; telemetry service remains UNK (HK subtype not certified)'
        elif field in ('command_intent','sequence_freshness_status','control_security_state','command_execution_result'):
            rule='Retain causal V1 TC/event evidence; scoped non-command TLM uses NO_COMMAND or NOT_APPLICABLE'
        rows.append(dict(field=field,v2_rule=rule))
    module.write_csv(out/'mapping_rules.csv',rows)
    return hashlib.sha256((out/'evaluation_protocol.json').read_bytes()).hexdigest()


def combine_results(module, out):
    all_rows=[]
    for mapping in ('V1','V2'):
        all_rows += [dict(mapping_version=mapping,**r) for r in read_csv(out/mapping/'trajectory_predictions.csv')]
    module.write_csv(out/'trajectory_predictions.csv',all_rows)
    events=[r for r in all_rows if r['activity_group']=='current_command']
    module.write_csv(out/'event_predictions.csv',events)
    summaries=[];confusions=[]
    for mapping in ('V1','V2'):
        grouped=collections.defaultdict(list)
        for row in events:
            if row['mapping_version']!=mapping:continue
            for population in (row['population'],'all_commands'):
                grouped[(row['model'],row['view'],row['serialization'],population)].append(row)
        for (model,view,style,population),rows in grouped.items():
            normal=[r for r in rows if r['intervention_truth']=='NORMAL']
            replay=[r for r in rows if r['intervention_truth']=='TELECOMMAND_REPLAY']
            item=dict(mapping_version=mapping,model=model,view=view,serialization=style,population=population,
                endpoint_count=len(rows),normal_count=len(normal),normal_non_normal_predictions=sum(r['prediction']!='NORMAL' for r in normal),
                replay_count=len(replay),replay_predicted_as_replay=sum(r['prediction']=='TELECOMMAND_REPLAY' for r in replay),
                replay_predicted_as_normal=sum(r['prediction']=='NORMAL' for r in replay),
                replay_predicted_as_other_attack=sum(r['prediction'] not in ('NORMAL','TELECOMMAND_REPLAY') for r in replay))
            item.update({'normal_to_'+label.lower():sum(r['prediction']==label for r in normal) for label in module.LABELS})
            summaries.append(item)
            for (truth,prediction),count in collections.Counter((r['intervention_truth'],r['prediction']) for r in rows).items():
                confusions.append(dict(mapping_version=mapping,model=model,view=view,serialization=style,population=population,
                                       truth=truth,prediction=prediction,count=count))
    module.write_csv(out/'endpoint_summary.csv',summaries)
    module.write_csv(out/'event_confusion_counts.csv',confusions)
    module.write_csv(out/'trajectory_class_distribution.csv',
        [dict(mapping_version=v,**r) for v in ('V1','V2') for r in read_csv(out/v/'trajectory_class_distribution.csv')])
    by_key={tuple(r[k] for k in ('mapping_version','run','endpoint_row','model','view','serialization')):r for r in all_rows}
    paired=[]
    for row in all_rows:
        if row['mapping_version']!='V1':continue
        key=tuple(row[k] for k in ('run','endpoint_row','model','view','serialization'))
        other=by_key[('V2',)+key]
        paired.append(dict(zip(('run','endpoint_row','model','view','serialization'),key),
                          activity_group=row['activity_group'],intervention_truth=row['intervention_truth'],
                          v1_prediction=row['prediction'],v2_prediction=other['prediction'],
                          decision_changed=row['prediction']!=other['prediction']))
    module.write_csv(out/'mapping_paired_changes.csv',paired)
    return dict(neural_conditions=8, common_endpoints=809, unique_command_endpoints=30,
                prediction_rows=len(all_rows),command_prediction_rows=len(events),rule_serialization_duplicated=False)


def seal(out):
    files=[p for p in out.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='result_manifest.json']
    write_json(out/'result_manifest.json',{p.relative_to(out).as_posix():digest(p) for p in sorted(files)})
    target=out.with_suffix('.zip')
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as archive:
        for path in [*sorted(files),out/'result_manifest.json']:
            archive.write(path, out.name+'/'+path.relative_to(out).as_posix())
    with zipfile.ZipFile(target) as archive:
        require(archive.testzip() is None, 'Result ZIP integrity check failed')
    return target


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nos3-zip',type=Path,required=True)
    parser.add_argument('--previous-result',type=Path)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--tst-root',type=Path)
    parser.add_argument('--device',choices=['cpu'],default='cpu')
    parser.add_argument('--threads',type=int,choices=[1,2,3,4],default=2)
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args(argv)
    root=args.project_root.resolve()
    def resolve(path):return path.resolve() if path.is_absolute() else (root/path).resolve()
    campaign=resolve(args.nos3_zip)
    previous=resolve(args.previous_result) if args.previous_result else root/'outputs/shared_nominal_v2/nos3_detector_evaluation'/REFERENCE_NAME
    output_root=resolve(args.output_root) if args.output_root else root/'outputs/shared_nominal_v2/nos3_mapping_sensitivity'
    out=output_root/('run_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(Path(__file__),out/'nos3_mapping_sensitivity.py')
    report=dict(code_version=VERSION,mapping_version=MAPPING_VERSION,status='STARTED',errors=[],
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),audit_only=args.audit_only,
        detector_evaluation_performed=False,formal_external_validation=False,post_hoc_analysis=True,
        additional_live_commands_sent=0,trained=False,stages=dict(mapping='PENDING',controls='PENDING',V1='PENDING',V2='PENDING'))
    module=None;stage='mapping';code=1
    try:
        print('NOS3 mapping sensitivity: '+VERSION,flush=True)
        print('CPU only | Original frozen TST L1/L10 and Rule L10 | No training | No live commands',flush=True)
        print('[1/5] Authenticate the previous evaluation and reproduce Step09/V1.',flush=True)
        module,reference=load_reference(previous,out)
        module.pin(Path(__file__))
        runs,schema=module.verify_campaign(campaign,out/'verified_inputs')
        for name in ('V1','V2'):(out/name).mkdir()
        module.prepare_inputs(runs,out/'V1')
        write_json(out/'v1_mapping_reproduction.json',compare_csv_records(reference['observable_windows'],read_csv(out/'V1/observable_windows.csv')))
        print('[2/5] Build the evidence-limited V2 mapping before model loading.',flush=True)
        v2,audit=build_v2(runs,schema)
        require(audit['total_windows']==899 and audit['window_counts']=={'TC':30,'TLM':869}
                and audit['cold_start_unk_count']==10, 'Reviewed mapping population differs')
        module.prepare_inputs(v2,out/'V2')
        module.write_csv(out/'mapping_evidence.csv',audit.pop('evidence'))
        module.write_csv(out/'mapping_changes.csv',audit.pop('changes'))
        module.write_json(out/'mapping_audit.json',audit)
        module.write_json(out/'schema_contract.json',schema)
        protocol_sha=write_protocol(module,out)
        frozen_mapping_hashes={str(p):digest(p) for p in [out/'evaluation_protocol.json',out/'mapping_rules.csv',
            out/'mapping_evidence.csv',out/'mapping_changes.csv',out/'V1/observable_windows.csv',out/'V2/observable_windows.csv']}
        report.update(mapping_audit=audit,protocol_sha256=protocol_sha)
        report['stages']['mapping']='COMPLETE'
        print('Mapping verified: 30 TC + 869 TLM windows; first-frame UNK preserved.',flush=True)
        if args.audit_only:
            report['status']='MAPPING_AUDIT_COMPLETE'
            for k in ('controls','V1','V2'):report['stages'][k]='NOT_REQUESTED'
            code=0
        else:
            stage='controls'
            print('[3/5] Verify original checkpoints and reproduce clean controls.',flush=True)
            context=module.model_context(root,args.tst_root)
            verify_checkpoint_identity(module,context,reference)
            module.write_json(out/'model_provenance.json',dict(source_contract=context['contract'],
                step11_run_id=context['r11']['run_id'],tokenizer_sha256=context['tokenizer_sha'],settings=context['r11']['settings']))
            engine=module.FrozenEngine(context,args.threads)
            report['runtime']=engine.versions
            clean=module.clean_control(context,engine,out)
            rule=module.load_frozen_rule(context,None,clean,out)
            report['stages']['controls']='COMPLETE'
            stage='V1'
            print('[4/5] Reproduce all 809 V1 endpoints and both serialization styles.',flush=True)
            module.evaluate_trajectory(runs,engine,rule,out/'V1')
            report['detector_evaluation_performed']=True
            reproduction=compare_predictions(reference['trajectory_predictions'],read_csv(out/'V1/trajectory_predictions.csv'))
            module.write_json(out/'v1_prediction_reproduction.json',reproduction)
            report['stages']['V1']='COMPLETE'
            stage='V2'
            print('[5/5] Evaluate the same endpoints using the frozen V2 mapping.',flush=True)
            module.evaluate_trajectory(v2,engine,rule,out/'V2')
            report['results']=combine_results(module,out)
            report['stages']['V2']='COMPLETE'
            report['status']='MAPPING_SENSITIVITY_COMPLETE'
            code=0
        for path,expected in {**module.PINS,**frozen_mapping_hashes}.items():
            require(digest(path)==expected,'Input or frozen mapping changed: '+path)
    except (Exception,KeyboardInterrupt) as error:
        report['status']='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAILED'
        report['stages'][stage]='FAILED'
        report['errors'].append(f'{type(error).__name__}: {error}')
        (out/'failure_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print('Evaluation stopped: '+str(error),file=sys.stderr,flush=True)
        code=130 if isinstance(error,KeyboardInterrupt) else 1
    finally:
        report['finished_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(out/'summary.json',report)
        write_json(out/'input_sha256.json',module.PINS if module else {})
        if (out/'verified_inputs').exists():shutil.rmtree(out/'verified_inputs')
        target=seal(out)
        print(json.dumps({k:report[k] for k in ('status','detector_evaluation_performed','errors')},indent=2))
        print('ZIP_SHA256='+digest(target),flush=True)
        print('UPLOAD_THIS='+str(target),flush=True)
    return code


if __name__=='__main__':
    raise SystemExit(main())
