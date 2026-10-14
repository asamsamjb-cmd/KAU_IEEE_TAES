#!/usr/bin/env python3
"""Offline NOS3 partial-observation detector evaluation, adapted from Step 19.

Run on the Windows research desktop in its existing virtual environment.
No training, online commands, model downloads, or writes to previous experiments.
The result ZIP contains every outcome, including diagnostic failures.
"""
from __future__ import annotations
import argparse
import ast
import collections
import csv
import datetime as dt
import hashlib
import io
import importlib.util
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
import types
import zipfile
from pathlib import Path, PurePosixPath

VERSION = "1.3-nos3-boundary-audit"
SEED = 20260610
DATASET_ID = "478f782ceb970d7a2de73319ccc4e929be06cfea684a054aa72d932ed65ae83b"
SOURCE08_SHA = "c7d4ab5285ea0aa93dc0b9c962b840702e4f9fce5c89b150e7c44235600441bb"
SOURCE09_SHA = "5568b681bc71276c46715e990fe5e09996d921343f4d8f8235d6e84c7fd35026"
REVIEWED_CAMPAIGN_SHA = "49935f774ad4f16d2732effa363fb9d17e61abcc94ccddc4ef88325c5c69598d"
RULE_RUN_NAME = "run_20261004T102431_766880Z"
RULE_MANIFEST_SHA = "e82ae8bb6ffe19029b48fa4465b1f9b0f74544cc8d4d4a94452a03a4dcf1d651"
RULE_CONFIG_SHA = "02b895a7b6b208e605d6b082263759f975c52371f062ce83f65f0b2dc44a1893"
RULE_PREDICTIONS_SHA = "24c02d92d9763ccc2c47ce44069fdfe18489026fc81436dccac09b7dc137cf45"
LABELS = ["NORMAL", "RF_LINK_JAMMING", "GNSS_SPOOFING", "UNAUTHORIZED_TELECOMMAND_INJECTION",
          "TELECOMMAND_REPLAY", "GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE"]
FIELDS = ["observation_stage", "path_type", "direction", "source_role", "destination_role",
          "protocol_context", "operational_phase", "message_type", "service_context", "command_intent",
          "payload_or_parameter_context", "sequence_freshness_status", "control_security_state",
          "signal_quality_state", "tracking_stability", "navigation_consistency", "timing_consistency",
          "decoder_status", "command_execution_result", "retry_timeout_state", "session_state",
          "state_transition_consistency"]
META = ["scenario_id", "episode_id", "episode_type", "episode_step", "window_id", "timestamp", "label"]
IDENTITY_KEYS = ("dataset_id", "step8_run_id", "step9_run_id", "step10_run_id")
KEYS = ["episode_id", "episode_step", "window_id"]
VIEWS = ("observed_partial", "freshness_unk")
SERIALIZATIONS = ("fixed_unk", "omit_unknown_clauses")
AVAILABLE_FIELDS = ("observation_stage", "direction", "destination_role", "protocol_context",
                    "message_type", "service_context", "sequence_freshness_status", "command_execution_result")
UNAVAILABLE_FIELDS = tuple(f for f in FIELDS if f not in AVAILABLE_FIELDS)
PREVIOUS_NAME = "run_20261005T132510_678888Z.zip"
PREVIOUS_SHA = "96fdb7d8abd5bbfa68425314dafd75fc58d2cd1d88f6d9867445c753f57da0d5"
PINS = {}
ATOL = RTOL = 1e-4


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pin(path, expected=None):
    path = Path(path).resolve()
    require(path.is_file(), f"Required file is missing: {path}")
    digest = sha_file(path)
    require(expected is None or digest == expected, f"SHA256 mismatch: {path}")
    require(str(path) not in PINS or PINS[str(path)] == digest, f"Input changed: {path}")
    PINS[str(path)] = digest
    return digest


def read_json(path, expected=None):
    pin(path, expected)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    require(rows, f"Cannot write an empty table: {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def serial(row):
    require(set(row) == set(FIELDS), "Model input must contain exactly the 22 observable fields")
    values = [row[f] for f in FIELDS]
    require(all(isinstance(v, str) and v and v == v.strip() for v in values), "Invalid observable value")
    return "; ".join(f"{f} is {v}" for f, v in zip(FIELDS, values)) + "."


def serialize_diagnostic(row, style):
    original = serial(row)
    require(style in SERIALIZATIONS, "Unknown diagnostic serialization")
    if style == "fixed_unk":
        return original
    clauses = [f"{f} is {row[f]}" for f in FIELDS if row[f] != "UNK"]
    require(clauses, "Omission would produce an empty input")
    return "; ".join(clauses) + "."


def mask14(frame):
    """Fixed field-name mask; no class, intervention or prediction is consulted."""
    require(set(FIELDS) <= set(frame.columns), "Missing benchmark fields")
    result = frame[FIELDS].copy()
    result.loc[:, list(UNAVAILABLE_FIELDS)] = "UNK"
    return result


def rule_identity(record, contract):
    sources = [(key, record[key]) for key in ("dataset_contract", "source_contract") if key in record]
    require(sources, "Rule manifest has no recognized input contract")
    for name, source in sources:
        if name == "dataset_contract":
            require(source.get("dataset_version") == "shared_nominal_v2", "Rule dataset version differs")
        for key in IDENTITY_KEYS:
            require(source.get(key) == contract[key],
                    f"Rule {name}.{key} differs: {source.get(key)!r} != {contract[key]!r}")
    return sources[0][1]


def trajectory_endpoints(runs):
    rows = []
    for run in runs:
        starts = [r["window_start"] for r in run["view"]["index"]]
        commands = {r["endpoint_row"]: r for r in run["endpoint_records"]}
        for end in range(9, len(starts)):
            seq_indices([end], 10, starts)
            count = sum(i in commands for i in range(end - 9, end + 1))
            group = "current_command" if end in commands else "recent_command" if count else "quiet_history"
            row = dict(run=run["name"], pair=run["pair"], condition=run["mode"], endpoint_row=end,
                       window_start=starts[end], window_end=starts[end]+1, activity_group=group,
                       commands_in_L10=count, intervention_truth=None, stage=None, population="unlabelled_timeline")
            if end in commands:
                row.update(commands[end])
            row["missing_fields"] = sum(run["view"]["features"][end][f] == "UNK" for f in FIELDS)
            rows.append(row)
    return rows


def classification_metrics(y, pred):
    import numpy as np
    y, pred = np.asarray(y, dtype=int), np.asarray(pred, dtype=int)
    require(y.ndim == 1 and y.shape == pred.shape and len(y) and
            np.isin(y, range(6)).all() and np.isin(pred, range(6)).all(), "Invalid metric population")
    cm = np.bincount(6*y+pred, minlength=36).reshape(6,6)
    rows = []
    for k, label in enumerate(LABELS):
        tp, support, predicted = int(cm[k,k]), int(cm[k].sum()), int(cm[:,k].sum())
        rows.append(dict(label=label, support=support, predicted_count=predicted, true_positive=tp,
                         precision=tp/predicted if predicted else 0., recall=tp/support if support else 0.,
                         f1=2*tp/(support+predicted) if support+predicted else 0.))
    return rows, dict(endpoint_count=len(y), correct=int(np.trace(cm)), accuracy=float(np.trace(cm)/len(y)),
                      macro_f1=sum(r["f1"] for r in rows)/6, normal_count=int(cm[0].sum()),
                      normal_false_alarms=int(cm[0].sum()-cm[0,0]),
                      normal_false_alarm_fraction=float((cm[0].sum()-cm[0,0])/cm[0].sum()) if cm[0].sum() else None)


def verify_previous_result(path, out):
    require(pin(path) == PREVIOUS_SHA, "Expected the reviewed 2026-10-05 13:25 UTC result ZIP")
    with zipfile.ZipFile(path) as z:
        require(len(z.namelist()) == len(set(z.namelist())), "Duplicate prior ZIP members")
        prefix = PREVIOUS_NAME[:-4] + "/"
        manifest = json.loads(z.read(prefix + "result_manifest.json"))
        for name, digest in manifest.items():
            require(sha_bytes(z.read(prefix+name)) == digest, f"Previous artifact changed: {name}")
        summary = json.loads(z.read(prefix+"summary.json"))
        require(summary.get("detector_evaluation_performed") is True and summary.get("code_version") == "1.1-nos3-timestamp-unit-fix",
                "Previous detector run was not completed")
        predictions = list(csv.DictReader(io.StringIO(z.read(prefix+"event_predictions.csv").decode("utf-8-sig"))))
        provenance = json.loads(z.read(prefix+"model_provenance.json"))
    report = dict(path=str(path), sha256=PREVIOUS_SHA, predictions=predictions, provenance=provenance)
    write_json(out/"previous_result_reference.json", dict(path=str(path), sha256=PREVIOUS_SHA,
               purpose="Previously observed results are retained; new diagnostics are post hoc and do not select a model."))
    return report


def compare_previous_predictions(previous, current):
    def key(r): return (r["run"], str(r["endpoint_row"]), r["model"], r["view"])
    lookup = {key(r): r for r in current}
    require(len(lookup) == len(current), "Duplicate current predictions")
    maximum = 0.
    for old in previous["predictions"]:
        require(key(old) in lookup, "A previously evaluated endpoint disappeared")
        new = lookup[key(old)]
        require(new["prediction"] == old["prediction"], "A previous fixed-input decision changed")
        for label in LABELS:
            col = "prob_"+label.lower()
            a, b = float(old[col]), float(new[col])
            delta = abs(a-b); maximum = max(maximum, delta)
            require(delta <= ATOL+RTOL*abs(a), "A previous fixed-input probability changed")
    return dict(status="PASS", predictions_checked=len(previous["predictions"]),
                identical_decisions=True, max_abs_probability_difference=maximum, atol=ATOL, rtol=RTOL)


def seq_indices(endpoints, length, starts):
    # A sequence may never cross a run boundary or a missing second.
    result = []
    for end in endpoints:
        require(type(end) is int and end >= length - 1, "Endpoint lacks the required history")
        ids = list(range(end - length + 1, end + 1))
        require(all(starts[i] == starts[end] - (end - i) for i in ids), "Nonconsecutive sequence")
        result.append(ids)
    return result


def verify_campaign(path, work):
    require(pin(path) == REVIEWED_CAMPAIGN_SHA,
            "This evaluator is frozen for the reviewed 2026-10-05 Step 09 ZIP. Supply that original ZIP.")
    work.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(path) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        require(len(infos) <= 1500 and sum(i.file_size for i in infos) <= 128 * 1024 ** 2,
                "Campaign archive exceeds limits")
        names = [i.filename for i in infos]
        require(len(names) == len(set(names)), "Duplicate archive members")
        roots = set()
        for info in infos:
            p = PurePosixPath(info.filename)
            require(not p.is_absolute() and not any(c in ("", ".", "..") for c in p.parts)
                    and "\\" not in info.filename and ":" not in info.filename
                    and ((info.external_attr >> 16) & 0o170000) != 0o120000, "Unsafe archive member")
            roots.add(p.parts[0])
        require(len(roots) == 1, "Expected one campaign root")
        prefix = next(iter(roots)) + "/"
        manifest = json.loads(z.read(prefix + "campaign_manifest.json"))
        require(set(manifest) == {n[len(prefix):] for n in names if n != prefix + "campaign_manifest.json"},
                "Campaign member inventory differs from manifest")
        for rel, digest in manifest.items():
            require(sha_bytes(z.read(prefix + rel)) == digest, f"Corrupt campaign member: {rel}")
        source = z.read(prefix + "baseline/08_nos3_mapping_audit.py")
        require(sha_bytes(source) == SOURCE08_SHA, "Unsupported mapping implementation")
        require(sha_bytes(z.read(prefix + "09_nos3_repeatability.py")) == SOURCE09_SHA,
                "Unsupported campaign implementation")
        source_path = work / "08_nos3_mapping_audit.py"
        source_path.write_bytes(source)
        spec = importlib.util.spec_from_file_location("nos3_verified_mapping08", source_path)
        m08 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m08)
        require(m08.FIELDS == FIELDS and m08.CONTRACT["dataset_id"] == DATASET_ID, "NOS3 schema mismatch")
        journal = json.loads(z.read(prefix + "journal.json"))
        require(len(journal["runs"]) == 10 and len(journal["pair_audits"]) == 5, "Incomplete campaign")
        runs = []
        last_end = None
        baseline_hashes = baseline_build = baseline_runtime = None
        for pair in range(1, 6):
            paths = []
            for mode in ("NORMAL", "REPLAY"):
                target = work / f"P{pair:02d}_{mode}.zip"
                target.write_bytes(z.read(prefix + "runs/" + target.name))
                paths.append(target)
            a, b, runtime = m08.verify_pair(*paths)
            views = []
            for run in (a, b):
                mode = run["summary"]["mode"]
                item = next(r for r in journal["runs"] if r["pair"] == pair and r["mode"] == mode)
                require(item["state"] == "VALID" and item["result"]["summary"] == run["summary"]
                        and item["result"]["archive_sha256"] == run["sha256"], "Run verification disagrees with journal")
                require(last_end is None or run["start"] > last_end, "Campaign chronology mismatch")
                last_end = run["end"]
                if baseline_hashes is None:
                    baseline_hashes, baseline_build, baseline_runtime = run["hashes"], run["build"], run["before"]
                require(run["hashes"] == baseline_hashes and run["build"] == baseline_build, "Build changed across campaign")
                pure07 = m08.offline07(run["files"]["07_nos3_replay_case_study.py"])
                for boundary in ("before", "after"):
                    comparison = pure07["compare_runtime"](baseline_runtime, run[boundary])
                    require(all(v is True for row in comparison.values() for v in row.values()), "Runtime session changed")
                view = m08.build_views(run["packets"], run["events"], run["start"], run["end"])
                views.append(view)
                runs.append(dict(pair=pair, mode=mode, name=f"P{pair:02d}_{mode.upper()}", raw=run, view=view))
            expected = next(x["summary"] for x in journal["pair_audits"] if x["pair"] == pair)
            require(m08.audit_summary(a, b, views, runtime) == expected, "Regenerated mapping audit differs")
    return runs, m08.CONTRACT


def prepare_inputs(runs, out):
    feature_rows, endpoints, frame_rows = [], [], []
    for run in runs:
        view = run["view"]
        starts = [r["window_start"] for r in view["index"]]
        require(len(view["frames"]) == 3, "Expected three captured command frames per run")
        end_ids = []
        for i, frame in enumerate(view["frames"]):
            start = math.floor(frame["receive_wall"])
            require(start in starts, "A command is outside complete observation bins")
            end = starts.index(start)
            index = view["index"][end]
            require(index["tc_frames"] == 1 and not index["issues"], "Ambiguous command window")
            seq_indices([end], 10, starts)
            stage = run["raw"]["summary"]["case_rows"][i]["stage"]
            replay = run["mode"] == "replay" and i == 1
            require(frame["exact_repeat"] == replay, "Intervention and independent byte comparison disagree")
            row = dict(run=run["name"], pair=run["pair"], condition=run["mode"], stage=stage,
                       endpoint_row=end, window_start=start, window_end=start + 1,
                       population="primary_middle_command" if i == 1 else "secondary_command_controls",
                       intervention_truth="TELECOMMAND_REPLAY" if replay else "NORMAL",
                       exact_frame_repeat=bool(frame["exact_repeat"]),
                       original_age_seconds=frame.get("previous_identical_age_seconds"),
                       original_inside_L10=frame["previous_identical_in_l10"],
                       missing_fields=index["missing_fields"])
            endpoints.append(row)
            end_ids.append(end)
            frame_rows.append(dict(run=run["name"], **frame))
        run["endpoints"] = end_ids
        run["endpoint_records"] = endpoints[-3:]
        for i, (a, b) in enumerate(zip(view["features"], view["without_history"])):
            require(b == dict(a, sequence_freshness_status="UNK"), "History ablation changed another field")
            for name, values in zip(VIEWS, (a, b)):
                serial(values)
                feature_rows.append(dict(run=run["name"], row_index=i, view=name,
                                         window_start=starts[i], **values))
    require(len(endpoints) == 30 and sum(r["population"] == "primary_middle_command" for r in endpoints) == 10,
            "Unexpected endpoint count")
    write_csv(out / "observable_windows.csv", feature_rows)
    write_csv(out / "event_endpoints.csv", endpoints)
    write_csv(out / "frame_observer_AUDIT_ONLY.csv", frame_rows)
    collisions = []
    for pair in range(1, 6):
        a, b = [r for r in runs if r["pair"] == pair]
        for length in (1, 10):
            for name, key in zip(VIEWS, ("features", "without_history")):
                def window_texts(run):
                    end = run["endpoints"][1]
                    return [serial(row) for row in run["view"][key][end - length + 1:end + 1]]
                left, right = window_texts(a), window_texts(b)
                collisions.append(dict(pair=pair, length=length, view=name, identical_normal_replay_input=left == right,
                                       normal_input_sha256=sha_bytes(json.dumps(left).encode()),
                                       replay_input_sha256=sha_bytes(json.dumps(right).encode())))
    write_csv(out / "primary_input_collisions.csv", collisions)
    return endpoints


def resolve_root(root, value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def model_context(root, explicit_tst=None):
    base = root / "outputs/shared_nominal_v2"
    roots = {"step8": base / "distilbert_window_baseline",
             "step9": base / "tst_embeddings_finetuned_main",
             "step10": base / "tst_sequences_finetuned_main"}
    records = {k: read_json(p / "run_manifest.json") for k, p in roots.items()}
    for k, r in records.items():
        require(r.get("status") == "COMPLETE" and r.get("dataset_id") == DATASET_ID and r.get("run_id"),
                f"{k}: expected the completed shared_nominal_v2 run")
    tst_root = resolve_root(root, explicit_tst) if explicit_tst else base / "tst_training_finetuned_main"
    if not explicit_tst and not (tst_root / "run_manifest.json").is_file():
        multi = read_json(base / "tst_training_finetuned_multiseed/multiseed_run_manifest.json")
        require(multi.get("status") == "COMPLETE", "Multiseed manifest is incomplete; provide --tst-root")
        candidates = [r for r in multi["runs"] if r["seed"] == SEED]
        require(len(candidates) == 1, "Cannot identify the prescribed TST seed")
        tst_root = resolve_root(root, candidates[0]["output_root"])
        pin(tst_root / "run_manifest.json", candidates[0]["manifest_sha256"])
    r11 = read_json(tst_root / "run_manifest.json")
    require(r11.get("status") == "COMPLETE" and r11.get("dataset_id") == DATASET_ID, "TST run is incomplete or from different data")
    contract = r11["source_contract"]
    require(contract.get("dataset_id") == DATASET_ID, "TST source dataset mismatch")
    for key, r in records.items():
        require(contract.get(key + "_run_id") == r["run_id"], f"{key} run identity mismatch")
        pin(roots[key] / "run_manifest.json", contract["manifest_sha256"][key])
    settings = r11["settings"]
    require(settings.get("DEFAULT_RANDOM_SEED") == SEED and settings.get("USE_MISSING_AWARE_TRAINING") is False
            and settings.get("SEQUENCE_FIT_SUBSET") == "full"
            and settings.get("PRIMARY_SELECTION_METRIC") == "val_macro_f1"
            and settings.get("USE_ROBUST_EVAL_AS_PRIMARY_FOR_LONG_CONTEXTS") is False,
            "Expected the prescribed main clean-trained TST, seed 20260610")
    require({1, 10} <= set(r11["selected_sequence_lengths"]), "Both L1 and L10 checkpoints are required")
    r8, r9, r10 = (records[k] for k in ("step8", "step9", "step10"))
    model_dir = roots["step8"] / "trained_distilbert_model"
    expected = {k: v for k, v in r8["artifacts"].items() if k.startswith("trained_distilbert_model/")}
    require(expected == contract["step8_model_sha256"] and expected, "Encoder inventory mismatch")
    require({p.relative_to(roots["step8"]).as_posix() for p in model_dir.rglob("*") if p.is_file()} == set(expected),
            "Encoder directory differs from its completed inventory")
    for name, digest in expected.items():
        pin(roots["step8"] / name, digest)
    extraction = read_json(roots["step9"] / "embedding_extraction_report.json",
                           contract["step9_artifacts"]["embedding_extraction_report.json"])
    index = read_json(roots["step10"] / "sequence_build_report_index.json",
                     contract["step10_artifacts"]["sequence_build_report_index.json"])
    require(extraction.get("status") == index.get("status") == "COMPLETE"
            and extraction.get("run_id") == r9["run_id"] and index.get("run_id") == r10["run_id"],
            "Embedding/sequence report identity mismatch")
    require(extraction.get("input_sha256") == r8["input_sha256"] == contract["input_sha256"]
            and extraction.get("step8_model_sha256") == expected, "Encoder/extraction input mismatch")
    source9 = r9["source_contract"]
    require(source9.get("step8_manifest_sha256") == contract["manifest_sha256"]["step8"]
            and source9.get("step8_run_id") == r8["run_id"]
            and {k: v for k, v in source9.get("step8_artifacts", {}).items() if k.startswith("trained_distilbert_model/")} == expected,
            "Embedding run uses a different encoder")
    enc = extraction["encoder_source_summary"]
    require(enc.get("representation_type") == "supervised_finetuned_distilbert_cls"
            and enc.get("pooling") == "last_hidden_state[:, 0, :]" and enc.get("normalization") == "none"
            and extraction.get("embedding_dtype") == "float32" and extraction.get("use_fp16_inference") is False,
            "Expected unnormalized FP32 fine-tuned CLS embeddings")
    mapping = read_json(model_dir / "label_mapping.json")
    require(mapping.get("label_order") == LABELS and mapping.get("label_to_id") == dict(zip(LABELS, range(6)))
            and mapping.get("observable_fields") == FIELDS and mapping.get("metadata_fields") == META
            and mapping.get("dataset_id") == DATASET_ID and mapping.get("encoder_source") == "finetuned"
            and mapping.get("serializer_version") == "observable22_fixed_order_is_v1", "Model serialization/schema mismatch")
    require(type(mapping.get("max_length")) is int and mapping["max_length"] > 0
            and mapping["max_length"] == extraction.get("max_length") == r8["protocol"].get("max_length"),
            "Token length contract mismatch")
    tokenizer_sha = r8["tokenizer_sha256"]
    require(tokenizer_sha == source9.get("tokenizer_sha256") == enc.get("tokenizer_sha256"), "Tokenizer identity mismatch")
    return dict(root=root, base=base, roots=roots, records=records, r11=r11, contract=contract,
                tst_root=tst_root, model_dir=model_dir, mapping=mapping, tokenizer_sha=tokenizer_sha)


def make_tst(config, torch):
    nn = torch.nn
    ModelConfig = types.SimpleNamespace
    class TimeSeriesTransformerClassifier(nn.Module):
        def __init__(self, config: ModelConfig):
            super().__init__()
            if config.pooling != "last":
                raise ValueError("Only pooling='last' is supported for last-window state classification.")
            if config.d_model % config.num_heads != 0:
                raise ValueError(
                    f"d_model={config.d_model} must be divisible by num_heads={config.num_heads}"
                )
    
            self.config = config
            self.input_norm = nn.LayerNorm(config.input_dim)
            self.input_projection = nn.Linear(config.input_dim, config.d_model)
            self.position_embedding = nn.Parameter(torch.zeros(1, config.sequence_length, config.d_model))
            self.dropout = nn.Dropout(config.dropout)
    
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=config.num_encoder_layers,
            )
            self.output_norm = nn.LayerNorm(config.d_model)
            self.classifier = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model, config.num_classes),
            )
            self.reset_parameters()
    
        def reset_parameters(self) -> None:
            nn.init.trunc_normal_(self.position_embedding, std=0.02)
            nn.init.xavier_uniform_(self.input_projection.weight)
            nn.init.zeros_(self.input_projection.bias)
            final_linear = self.classifier[-1]
            if isinstance(final_linear, nn.Linear):
                nn.init.xavier_uniform_(final_linear.weight)
                nn.init.zeros_(final_linear.bias)
    
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [batch, L, input_dim]
            if x.ndim != 3:
                raise ValueError(f"TST input must be 3-D [B,L,D], got {tuple(x.shape)}")
            if x.shape[1] != self.config.sequence_length:
                raise ValueError(
                    f"Sequence length mismatch: model L={self.config.sequence_length}, input L={x.shape[1]}"
                )
            h = self.input_norm(x)
            h = self.input_projection(h)
            h = h + self.position_embedding[:, : h.shape[1], :]
            h = self.dropout(h)
            h = self.encoder(h)
            h_last = h[:, -1, :]
            h_last = self.output_norm(h_last)
            logits = self.classifier(h_last)
            return logits
    return TimeSeriesTransformerClassifier(types.SimpleNamespace(**config))


class FrozenEngine:
    def __init__(self, context, threads):
        # Import heavy dependencies only after the ZIP and file contracts pass.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import numpy as np
        import torch
        from transformers import AutoTokenizer, DistilBertForSequenceClassification
        self.np, self.torch, self.context = np, torch, context
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(SEED)
        self.tokenizer = AutoTokenizer.from_pretrained(context["model_dir"], use_fast=True, local_files_only=True)
        backend = json.loads(self.tokenizer.backend_tokenizer.to_str())
        backend.pop("padding", None)
        backend.pop("truncation", None)
        identity = dict(backend=backend, special_tokens_map=self.tokenizer.special_tokens_map,
                        model_input_names=self.tokenizer.model_input_names)
        digest = sha_bytes(json.dumps(identity, sort_keys=True, ensure_ascii=False, allow_nan=False).encode())
        require(digest == context["tokenizer_sha"], "Loaded tokenizer fingerprint differs from training")
        self.encoder, info = DistilBertForSequenceClassification.from_pretrained(
            context["model_dir"], local_files_only=True, output_loading_info=True)
        require(not any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")),
                "Encoder weights did not load exactly")
        require(self.encoder.config.dim == 768 and self.encoder.config.num_labels == 6
                and self.encoder.config.label2id == dict(zip(LABELS, range(6)))
                and {int(k): v for k, v in self.encoder.config.id2label.items()} == dict(enumerate(LABELS)),
                "Encoder class order/dimensions differ")
        self.encoder.float().cpu().eval()
        self.heads = {}
        for length in (1, 10):
            rel = f"L{length}/checkpoints/best_tst_model_L{length}.pt"
            path = context["tst_root"] / rel
            pin(path, context["r11"]["artifacts"][rel])
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            pin(path, context["r11"]["artifacts"][rel])
            provenance = {k: context["contract"][k] for k in IDENTITY_KEYS}
            provenance["step11_run_id"] = context["r11"]["run_id"]
            require(checkpoint.get("provenance") == provenance and checkpoint.get("source_contract") == context["contract"],
                    f"L{length}: checkpoint provenance mismatch")
            require(checkpoint.get("label_order") == LABELS and checkpoint.get("label_to_id") == dict(zip(LABELS, range(6))),
                    f"L{length}: checkpoint class order mismatch")
            train = checkpoint["train_config"]
            require(train.get("random_seed") == SEED and train.get("use_missing_aware_training") is False
                    and train.get("use_amp") == context["r11"]["settings"]["USE_AMP"], "Checkpoint training identity mismatch")
            config = checkpoint["model_config"]
            require(config["input_dim"] == 768 and config["num_classes"] == 6 and config["sequence_length"] == length
                    and config["temporal_context_seconds"] == length, "Checkpoint input dimensions mismatch")
            settings = context["r11"]["settings"]
            for key in ("d_model", "num_heads", "num_encoder_layers", "dim_feedforward", "dropout", "pooling"):
                require(config[key] == settings[key.upper()], f"Architecture mismatch: {key}")
            head = make_tst(config, torch)
            head.load_state_dict(checkpoint["model_state_dict"], strict=True)
            self.heads[length] = head.float().cpu().eval()
        self.cache = {}
        self.versions = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                             transformers=__import__("transformers").__version__, device="cpu", threads=threads,
                             pandas=__import__("pandas").__version__, precision="float32", autocast=False)

    def embed(self, texts):
        np, torch = self.np, self.torch
        unseen = list(dict.fromkeys(t for t in texts if t not in self.cache))
        for lo in range(0, len(unseen), 4):
            batch = unseen[lo:lo + 4]
            encoded = self.tokenizer(batch, truncation=False, padding=True, return_tensors="pt")
            require(not (encoded["attention_mask"].sum(1) > self.context["mapping"]["max_length"]).any(),
                    "NOS3 input exceeds trained token length; truncation is forbidden")
            require(self.tokenizer.cls_token_id is not None
                    and (encoded["input_ids"][:, 0] == self.tokenizer.cls_token_id).all(), "First token is not CLS")
            require(self.tokenizer.unk_token_id is None or not (encoded["input_ids"] == self.tokenizer.unk_token_id).any(),
                    "Tokenizer produced [UNK]; field value UNK must use ordinary saved vocabulary")
            with torch.inference_mode():
                output = self.encoder.distilbert(**{k: encoded[k] for k in ("input_ids", "attention_mask")})
                vectors = output.last_hidden_state[:, 0, :].float().cpu().numpy()
            require(vectors.shape == (len(batch), 768) and np.isfinite(vectors).all(), "Invalid CLS embeddings")
            self.cache.update(zip(batch, vectors))
        return np.stack([self.cache[t] for t in texts])

    def predict(self, length, embeddings, indices):
        np, torch = self.np, self.torch
        ids = np.asarray(indices, dtype=np.int64)
        require(ids.ndim == 2 and ids.shape[1] == length and (ids >= 0).all()
                and (ids < len(embeddings)).all(), "Invalid TST sequence indices")
        result = []
        with torch.inference_mode():
            for lo in range(0, len(ids), 32):
                x = torch.from_numpy(np.asarray(embeddings[ids[lo:lo + 32]], dtype=np.float32))
                result.append(torch.softmax(self.heads[length](x).float(), dim=1).cpu().numpy())
        probs = np.concatenate(result)
        validate_probs(probs, len(ids))
        return probs


def validate_probs(probs, count):
    import numpy as np
    require(probs.shape == (count, 6) and np.isfinite(probs).all()
            and (probs >= 0).all() and (probs <= 1).all()
            and np.allclose(probs.sum(axis=1), 1, atol=1e-5, rtol=0), "Invalid six-class probabilities")


def validate_one_second_timestamps(values, context_name):
    """Compare timedeltas exactly without assuming integer datetime storage units."""
    import pandas as pd
    times = pd.to_datetime(pd.Series(values).reset_index(drop=True), errors="raise", utc=True)
    require(len(times) >= 2 and times.notna().all(), f"{context_name}: missing/insufficient timestamps")
    deltas = times.diff().iloc[1:]
    valid = deltas.eq(pd.Timedelta(seconds=1))
    if not valid.all():
        bad = [{"row_offset": int(i), "interval": str(deltas.loc[i]),
                "previous": str(times.iloc[i - 1]), "current": str(times.iloc[i])}
               for i in valid.index[~valid][:5]]
        raise ValueError(f"{context_name}: timestamps are not consecutive seconds; "
                         f"parsed_dtype={times.dtype}; mismatches={bad}")
    return dict(context=context_name, parsed_dtype=str(times.dtype), timestamp_count=len(times),
                first_timestamp=str(times.iloc[0]), last_timestamp=str(times.iloc[-1]),
                interval="Exactly one second", status="PASS")


def clean_control(context, engine, out):
    """Fixed first 16 common endpoints: technical reproduction, no model selection."""
    import numpy as np
    import pandas as pd
    c = context["contract"]
    root9, root10 = context["roots"]["step9"], context["roots"]["step10"]
    raw_path = context["root"] / "dataset/shared_nominal_v2/splits/test_observable.csv"
    pin(raw_path, c["input_sha256"]["splits/test_observable.csv"])
    metadata_path = root9 / "test_window_metadata.csv"
    pin(metadata_path, c["step9_artifacts"][metadata_path.name])
    original = pd.read_csv(raw_path, keep_default_na=False)
    windows = pd.read_csv(metadata_path, keep_default_na=False)
    require(list(original.columns) == META + FIELDS and not original.duplicated(KEYS).any()
            and not windows.duplicated(KEYS).any(), "Clean control schema/identity mismatch")
    require(np.array_equal(windows.embedding_row_index, np.arange(len(windows))), "Embedding row order mismatch")
    positions = pd.MultiIndex.from_frame(original[KEYS]).get_indexer(pd.MultiIndex.from_frame(windows[KEYS]))
    require(len(original) == len(windows) and (positions >= 0).all(), "Clean window identity mismatch")
    aligned = original.iloc[positions].reset_index(drop=True)
    for k in META:
        require(aligned[k].astype(str).equals(windows[k].astype(str)), f"Clean metadata mismatch: {k}")
    meta_path = root10 / "test_tst_sequence_metadata_L10.csv"
    pin(meta_path, c["step10_artifacts"][meta_path.name])
    seq_meta = pd.read_csv(meta_path, keep_default_na=False)
    selected = seq_meta.sort_values("sequence_id").head(16).copy()
    require(len(selected) == 16, "Fewer than 16 technical-control endpoints")
    ends = selected.sequence_end_row.to_numpy(dtype=np.int64)
    starts = selected.sequence_start_row.to_numpy(dtype=np.int64)
    require(np.all(ends - starts == 9) and len(set(ends)) == 16, "Invalid clean L10 span")
    ids10 = starts[:, None] + np.arange(10)[None, :]
    require((ids10 >= 0).all() and (ids10 < len(windows)).all(), "Clean sequence is out of bounds")
    time_checks = []
    for seq, (_, row) in zip(ids10, selected.iterrows()):
        w = windows.iloc[seq]
        require((w.episode_id == row.episode_id).all() and np.all(np.diff(w.episode_step.astype(int)) == 1),
                "Clean sequence crosses an episode")
        time_checks.append(validate_one_second_timestamps(
            w.timestamp, f"Clean L10 sequence {row.sequence_id}, episode {row.episode_id}"))
    write_json(out / "clean_time_alignment.json", dict(pandas_version=pd.__version__, checks=time_checks))
    unique = np.unique(ids10)
    texts = [serial({f: str(aligned.iloc[i][f]) for f in FIELDS}) for i in unique]
    live = engine.embed(texts)
    npz_path = root9 / "test_window_embeddings.npz"
    pin(npz_path, c["step9_artifacts"][npz_path.name])
    with np.load(npz_path, allow_pickle=False) as saved:
        cached = saved["embeddings"][unique]
        require(np.array_equal(saved["row_indices"], np.arange(len(windows))), "Cached embedding order mismatch")
    report = dict(selection="First 16 Step 10 common endpoints by sequence_id; no label/performance selection",
                  endpoint_rows=ends.tolist(), window_count=len(unique), atol=ATOL, rtol=RTOL,
                  embedding_max_abs_difference=float(np.max(np.abs(live - cached))), tst={})
    write_json(out / "clean_control.json", report)
    require(np.allclose(live, cached, atol=ATOL, rtol=RTOL),
            "Clean encoder reproduction failed; see clean_control.json. No NOS3 prediction was produced.")
    index_map = {int(v): i for i, v in enumerate(unique)}
    for length in (1, 10):
        ids = ends[:, None] if length == 1 else ids10
        local = np.array([[index_map[int(v)] for v in row] for row in ids])
        probabilities = engine.predict(length, live, local)
        rel = f"L{length}/predictions/test_predictions_L{length}.csv"
        path = context["tst_root"] / rel
        pin(path, context["r11"]["artifacts"][rel])
        saved = pd.read_csv(path, keep_default_na=False)
        require(not saved.sequence_end_row.duplicated().any(), "Duplicate saved prediction endpoint")
        matched = saved.set_index("sequence_end_row").loc[ends]
        require(matched.episode_id.tolist() == selected.episode_id.tolist(), "Clean prediction episode mismatch")
        expected = matched[["prob_" + label.lower() for label in LABELS]].to_numpy(dtype=float)
        delta = float(np.max(np.abs(probabilities - expected)))
        same_decisions = bool(np.array_equal(probabilities.argmax(1), matched.pred_label_id.to_numpy(dtype=int)))
        passed = bool(np.allclose(probabilities, expected, atol=ATOL, rtol=RTOL) and same_decisions)
        report["tst"][str(length)] = dict(max_abs_probability_difference=delta, identical_decisions=same_decisions, passed=passed)
        write_json(out / "clean_control.json", report)
        require(passed, f"Clean TST L{length} reproduction failed; see clean_control.json")
    report["status"] = "PASS"
    write_json(out / "clean_control.json", report)
    return dict(frame=aligned, indices=ids10, metadata=selected)


def object_digest(value):
    return sha_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8"))


def benchmark_population(context, frame):
    """Validate all common L10 endpoints against the completed sequence artifacts."""
    import numpy as np
    import pandas as pd
    root = context["roots"]["step10"]
    c = context["contract"]
    names = ("test_tst_sequence_metadata_L10.csv", "test_tst_sequences_L10.npz")
    for name in names:
        pin(root/name, c["step10_artifacts"][name])
    meta = pd.read_csv(root/names[0], keep_default_na=False).sort_values("sequence_id").reset_index(drop=True)
    with np.load(root/names[1], allow_pickle=False) as z:
        ids, y = z["sequence_indices"].copy(), z["y"].copy()
    require(ids.dtype.kind in "iu" and y.dtype.kind in "iu" and ids.shape == (len(meta),10)
            and y.shape == (len(meta),), "Invalid full benchmark sequence arrays")
    expected = c["common_endpoint_report"]["splits"]["test"]
    require(len(meta) == expected["common_endpoint_count"] and len(meta) > 0
            and np.array_equal(meta.sequence_id, np.arange(len(meta))), "Incomplete common benchmark population")
    require((ids >= 0).all() and (ids < len(frame)).all() and (np.diff(ids,axis=1)==1).all(),
            "Invalid benchmark sequence row order")
    ends = ids[:,-1]
    require(np.array_equal(ids[:,0],meta.sequence_start_row) and np.array_equal(ends,meta.sequence_end_row)
            and len(np.unique(ends)) == len(ends), "Benchmark endpoint identity differs")
    require(sha_bytes(np.sort(ends).astype('<i8',copy=False).tobytes()) == expected["sorted_end_row_indices_sha256"],
            "Benchmark common endpoint digest differs")
    episode = frame.episode_id.to_numpy()
    require((episode[ids] == episode[ends,None]).all() and np.array_equal(episode[ends],meta.episode_id),
            "Benchmark sequence crosses an episode")
    complete = np.flatnonzero(frame.groupby("episode_id",sort=False).cumcount().to_numpy() >= 9)
    require(np.array_equal(np.sort(ends),complete), "Benchmark omits or adds eligible endpoints")
    steps = frame.episode_step.to_numpy()
    require((np.diff(steps[ids],axis=1)==1).all(), "Benchmark sequence steps are not consecutive")
    times = pd.to_datetime(frame.timestamp, errors="raise", utc=True)
    require(not times.isna().any(), "Missing benchmark timestamps")
    adjacent = times.diff().eq(pd.Timedelta(seconds=1)).to_numpy()
    require(adjacent[ids[:,1:]].all(), "Benchmark timestamps are not consecutive seconds")
    true = frame.label.map(dict(zip(LABELS,range(6)))).to_numpy()
    require(np.isin(true,range(6)).all() and np.array_equal(y,true[ends])
            and np.array_equal(meta.label_id,y) and np.array_equal(meta.label,frame.label.iloc[ends]),
            "Benchmark terminal labels differ")
    require(set(y.tolist()) == set(range(6)), "Benchmark does not cover the six prescribed classes")
    return dict(indices=ids.astype(np.int64,copy=False), metadata=meta, y=y.astype(np.int64,copy=False))


def rule_environment(spec):
    """Compile only the three inference definitions stored in the sealed rule specification."""
    import numpy as np
    import pandas as pd
    require(spec.get("available_fields") == FIELDS and spec.get("canonical_labels") == LABELS,
            "Saved rule schema differs")
    tokens = spec.get("membership_sets")
    require(isinstance(tokens,dict) and tokens, "Saved rule membership sets are missing")
    env = dict(np=np, pd=pd, require=require, FIELDS=tuple(FIELDS), LABELS=tuple(LABELS),
               ATTACKS=tuple(LABELS[1:]), LENGTH=10, TOKENS=tokens, __builtins__=__builtins__)
    nodes = []
    for name in ("window_evidence","aggregate_evidence","classify"):
        text = spec.get("implementation",{}).get(name)
        require(isinstance(text,str), f"Saved rule implementation is missing: {name}")
        tree = ast.parse(text)
        require(len(tree.body)==1 and isinstance(tree.body[0],ast.FunctionDef) and tree.body[0].name==name
                and not tree.body[0].decorator_list and not tree.body[0].args.defaults
                and not any(tree.body[0].args.kw_defaults), "Unexpected saved rule inference definition")
        nodes.append(tree.body[0])
    exec(compile(ast.Module(body=nodes,type_ignores=[]),"<verified saved rule_spec>","exec"),env)
    return env


def load_frozen_rule(context, explicit_source, clean, out):
    """Read the existing v1 dataset_contract and its hash-bound inference implementation."""
    import numpy as np
    import pandas as pd
    folder = context["base"] / "16_observable_rule_baseline" / RULE_RUN_NAME
    run = read_json(folder / "run_manifest.json", RULE_MANIFEST_SHA)
    shutil.copy2(folder/"run_manifest.json", out/"rule_original_run_manifest.json")
    config = read_json(folder / "frozen_rule_config.json", RULE_CONFIG_SHA)
    shutil.copy2(folder/"frozen_rule_config.json", out/"frozen_rule_config.json")
    require(run.get("status") == "COMPLETE" and config.get("protocol_version") == "shared_nominal_v2_semantic_rules_v1",
            "Expected the previously evaluated frozen Rule v1 run")
    require(config.get("label_to_id") == dict(zip(LABELS, range(6))), "Rule class order mismatch")
    source = rule_identity(run,context["contract"])
    c = context["contract"]
    build_path = context["root"]/"dataset/shared_nominal_v2/build_manifest.json"
    build = read_json(build_path,c["observability_audit"]["step7_build_manifest_sha256"])
    require(build.get("dataset_id") == c["dataset_id"], "Rule dataset build identity differs")
    needed = {build_path:sha_file(build_path)}
    dataset = context["root"]/"dataset/shared_nominal_v2"
    for split in ("train","val","test"):
        counts = build["counts"][split]
        expected_count = [counts["rows"],counts["episodes"],c["common_endpoint_report"]["splits"][split]["common_endpoint_count"]]
        require(source.get("expected_counts",{}).get(split) == expected_count,
                f"Rule {split} population differs")
        name = f"splits/{split}_observable.csv"
        needed[dataset/name] = c["input_sha256"][name]
    needed[dataset/"splits/label_mapping.json"] = c["input_sha256"]["splits/label_mapping.json"]
    name = "sequence_build_report_index.json"
    needed[context["roots"]["step10"]/name] = c["step10_artifacts"][name]
    for split in ("val","test"):
        name=f"{split}_window_metadata.csv"
        needed[context["roots"]["step9"]/name] = c["step9_artifacts"][name]
        for suffix in ("sequence_metadata_L10.csv","sequences_L10.npz"):
            name=f"{split}_tst_{suffix}"
            needed[context["roots"]["step10"]/name] = c["step10_artifacts"][name]
    for path,digest in needed.items():
        relative=path.relative_to(context["root"]).as_posix()
        matches=[x for x in run.get("inputs",[]) if str(x.get("path","")).replace("\\","/").endswith("/"+relative)]
        require(len(matches)==1 and matches[0].get("sha256")==digest,
                f"Rule v1 recorded input differs: {relative}")
        pin(path,digest)
    outputs=run.get("outputs",[])
    require(len({x["file"] for x in outputs})==len(outputs), "Duplicate Rule output inventory")
    hashes={x["file"]:x["sha256"] for x in outputs}
    require(hashes.get("frozen_rule_config.json")==RULE_CONFIG_SHA
            and run.get("frozen_config_sha256")==RULE_CONFIG_SHA, "Rule config is not bound to the completed run")
    spec=read_json(folder/"rule_spec.json",hashes["rule_spec.json"])
    protocol=read_json(folder/"protocol.json",hashes["protocol.json"])
    shutil.copy2(folder/"rule_spec.json",out/"frozen_rule_spec.json")
    shutil.copy2(folder/"protocol.json",out/"rule_original_protocol.json")
    require(object_digest(spec)==config.get("rule_spec_sha256")==protocol.get("rule_spec_sha256"),
            "Rule specification identity differs")
    require(object_digest(protocol)==config.get("protocol_sha256")==run.get("protocol_sha256")
            and protocol.get("version")==config["protocol_version"]==spec.get("version"),
            "Rule protocol identity differs")
    require(protocol.get("dataset_contract")==run.get("dataset_contract")
            and protocol.get("metadata_feature_use") is False
            and protocol.get("available_observable_fields")==FIELDS, "Rule input contract differs")
    require(config.get("script_sha256")==run.get("script_sha256"), "Rule producer source identity differs")
    if explicit_source:
        original=resolve_root(context["root"],explicit_source)
        pin(original,run["script_sha256"])
        shutil.copy2(original,out/"frozen_rule_source.py")
    env=rule_environment(spec)
    thresholds = config["models"]["Rule_L10"]["thresholds"]
    population=benchmark_population(context,clean["frame"])
    scores = env["aggregate_evidence"](clean["frame"][FIELDS], population["indices"], 10)
    pred, diag = env["classify"](scores, thresholds)
    path = folder / "test_predictions_Rule_L10.csv"
    expected_hash = RULE_PREDICTIONS_SHA
    require(hashes.get(path.name)==expected_hash,"Rule saved prediction inventory differs")
    pin(path, expected_hash)
    saved = pd.read_csv(path, keep_default_na=False)
    require(len(saved)==len(population["metadata"]) and not saved.sequence_end_row.duplicated().any(),
            "Duplicate or missing rule endpoint")
    matched = saved.set_index("sequence_end_row").loc[population["metadata"].sequence_end_row]
    require(matched.episode_id.tolist()==population["metadata"].episode_id.tolist(),"Rule episode alignment differs")
    pred_col = "pred_label_id" if "pred_label_id" in matched else "y_pred"
    require(pred_col in matched and np.array_equal(pred, matched[pred_col].to_numpy(dtype=int)),
            "Frozen rule predictions failed clean reproduction")
    score_cols=["evidence_"+label for label in LABELS[1:]]
    require(set(score_cols)<=set(matched) and np.allclose(scores,matched[score_cols].to_numpy(float),atol=1e-12,rtol=1e-12),
            "Frozen rule scores failed clean reproduction")
    report = dict(status="PASS", source="Hash-bound inference functions and membership sets in saved rule_spec.json",
                  source_sha256=run["script_sha256"], frozen_config_sha256=RULE_CONFIG_SHA,
                  original_run=RULE_RUN_NAME, protocol_version=config["protocol_version"], thresholds=thresholds,
                  contract_adapter="dataset_contract_v1", clean_control_endpoints=len(pred), recalibrated=False)
    write_json(out / "rule_provenance.json", report)
    return dict(env=env, thresholds=thresholds)


def evaluate(runs, engine, rule, out):
    import numpy as np
    import pandas as pd
    predictions = []
    all_texts = [serial(row) for run in runs for key in ("features", "without_history") for row in run["view"][key]]
    engine.embed(list(dict.fromkeys(all_texts)))
    for run in runs:
        starts = [r["window_start"] for r in run["view"]["index"]]
        for view_name, key in zip(VIEWS, ("features", "without_history")):
            features = run["view"][key]
            embeddings = engine.embed([serial(row) for row in features])
            for length in (1, 10):
                ids = seq_indices(run["endpoints"], length, starts)
                probs = engine.predict(length, embeddings, ids)
                validate_probs(probs, 3)
                for endpoint, p in zip(run["endpoint_records"], probs):
                    pred = int(np.argmax(p))
                    row = dict(endpoint, view=view_name, model=f"TST_L{length}", prediction=LABELS[pred],
                               predicted_label_id=pred, confidence=float(p[pred]))
                    row.update({"prob_" + label.lower(): float(p[j]) for j, label in enumerate(LABELS)})
                    predictions.append(row)
            if rule is not None:
                ids = np.asarray(seq_indices(run["endpoints"], 10, starts), dtype=np.int64)
                scores = rule["env"]["aggregate_evidence"](pd.DataFrame(features, columns=FIELDS), ids, 10)
                pred, diag = rule["env"]["classify"](scores, rule["thresholds"])
                for i, endpoint in enumerate(run["endpoint_records"]):
                    row = dict(endpoint, view=view_name, model="Rule_L10", prediction=LABELS[int(pred[i])],
                               predicted_label_id=int(pred[i]))
                    row.update({"rule_score_" + label.lower(): float(scores[i, j]) for j, label in enumerate(LABELS[1:])})
                    predictions.append(row)
    write_csv(out / "event_predictions.csv", predictions)
    return predictions


def evaluate_trajectory(runs, engine, rule, out, previous=None):
    import numpy as np
    import pandas as pd
    endpoints=trajectory_endpoints(runs)
    require(len(endpoints)==809,"Unexpected reviewed campaign trajectory population")
    write_csv(out/"trajectory_endpoints.csv",endpoints)
    predictions=[]
    inventory={}
    for run in runs:
        selected=[r for r in endpoints if r["run"]==run["name"]]
        ends=[r["endpoint_row"] for r in selected]
        starts=[r["window_start"] for r in run["view"]["index"]]
        for view,key in zip(VIEWS,("features","without_history")):
            features=run["view"][key]
            for style in SERIALIZATIONS:
                texts=[serialize_diagnostic(r,style) for r in features]
                for text,row in zip(texts,features):
                    digest=sha_bytes(text.encode())
                    item=inventory.setdefault((view,style,digest),dict(view=view,serialization=style,text_sha256=digest,
                        text=text,missing_fields=sum(row[f]=='UNK' for f in FIELDS),occurrences=0))
                    item["occurrences"]+=1
                embeddings=engine.embed(texts)
                for length in (1,10):
                    probs=engine.predict(length,embeddings,seq_indices(ends,length,starts))
                    validate_probs(probs,len(ends))
                    for endpoint,p in zip(selected,probs):
                        pred=int(np.argmax(p))
                        record=dict(endpoint,view=view,serialization=style,model=f"TST_L{length}",
                            prediction=LABELS[pred],predicted_label_id=pred,confidence=float(p[pred]))
                        record.update({"prob_"+label.lower():float(p[j]) for j,label in enumerate(LABELS)})
                        predictions.append(record)
            if rule is not None:
                ids=np.asarray(seq_indices(ends,10,starts),dtype=np.int64)
                scores=rule["env"]["aggregate_evidence"](pd.DataFrame(features,columns=FIELDS),ids,10)
                labels,_=rule["env"]["classify"](scores,rule["thresholds"])
                for endpoint,label,score in zip(selected,labels,scores):
                    record=dict(endpoint,view=view,serialization="structured",model="Rule_L10",
                        prediction=LABELS[int(label)],predicted_label_id=int(label),confidence=None)
                    record.update({"rule_score_"+name.lower():float(score[j]) for j,name in enumerate(LABELS[1:])})
                    predictions.append(record)
    write_csv(out/"trajectory_predictions.csv",predictions)
    write_csv(out/"serialization_input_inventory.csv",list(inventory.values()))
    grouped=collections.defaultdict(list)
    for row in predictions:
        for scope in (row["run"],"ALL_RUNS_DESCRIPTIVE"):
            for activity in (row["activity_group"],"ALL_ENDPOINTS"):
                grouped[(scope,row["model"],row["view"],row["serialization"],activity)].append(row)
    summary=[]
    for (scope,model,view,style,activity),rs in grouped.items():
        counts=collections.Counter(r["prediction"] for r in rs)
        item=dict(run=scope,model=model,view=view,serialization=style,activity_group=activity,endpoint_count=len(rs),
                  labelled_command_endpoints=sum(r["intervention_truth"] is not None for r in rs),
                  statistic="Descriptive output distribution; unlabelled quiet endpoints are not false-alarm trials")
        for label in LABELS:
            item["count_"+label.lower()]=counts[label]
            item["fraction_"+label.lower()]=counts[label]/len(rs)
        summary.append(item)
    write_csv(out/"trajectory_class_distribution.csv",summary)
    lookup={(r["run"],r["endpoint_row"],r["model"],r["view"],r["serialization"]):r for r in predictions}
    paired=[]
    for r in predictions:
        if r["serialization"]!="fixed_unk":continue
        other=lookup[(r["run"],r["endpoint_row"],r["model"],r["view"],"omit_unknown_clauses")]
        item={k:r[k] for k in ("run","pair","endpoint_row","window_start","activity_group","intervention_truth","model","view")}
        item.update(fixed_prediction=r["prediction"],omission_prediction=other["prediction"],
                    decision_changed=r["prediction"]!=other["prediction"],
                    probability_total_variation=0.5*sum(abs(r["prob_"+x.lower()]-other["prob_"+x.lower()]) for x in LABELS))
        paired.append(item)
    write_csv(out/"serialization_paired.csv",paired)
    command_predictions=[r for r in predictions if r["activity_group"]=="current_command"]
    write_csv(out/"serialization_event_predictions.csv",command_predictions)
    if previous is not None:
        fixed=[r for r in command_predictions if r["serialization"]=="fixed_unk"]
        write_json(out/"trajectory_fixed_reproduction.json",compare_previous_predictions(previous,fixed))
    # Exact input identity, including the nine preceding rows, determines grouping.
    model_inputs=collections.defaultdict(list)
    runs_by_name={r["name"]:r for r in runs}
    for row in command_predictions:
        if row["serialization"]=="structured":continue
        run=runs_by_name[row["run"]];key="features" if row["view"]=="observed_partial" else "without_history"
        length=int(row["model"].split("L")[-1]);end=row["endpoint_row"]
        texts=[serialize_diagnostic(x,row["serialization"]) for x in run["view"][key][end-length+1:end+1]]
        digest=object_digest(texts)
        model_inputs[(row["model"],row["view"],row["serialization"],digest)].append(row)
    consistency=[]
    for (model,view,style,digest),rs in model_inputs.items():
        probabilities=np.asarray([[r["prob_"+x.lower()] for x in LABELS] for r in rs])
        delta=float(np.max(np.abs(probabilities-probabilities[0])))
        require(len({r["prediction"] for r in rs})==1 and delta<=ATOL,"Identical model inputs produced inconsistent outputs")
        consistency.append(dict(model=model,view=view,serialization=style,input_sha256=digest,occurrences=len(rs),
                                prediction=rs[0]["prediction"],max_abs_probability_difference=delta))
    write_csv(out/"identical_input_predictions.csv",consistency)
    report=dict(status="COMPLETE",common_endpoints=len(endpoints),
        activity_counts=dict(collections.Counter(r["activity_group"] for r in endpoints)),
        tst_prediction_rows=sum(r["model"]!="Rule_L10" for r in predictions),
        rule_prediction_rows=sum(r["model"]=="Rule_L10" for r in predictions),
        silent_truth_assigned=False,rule_serialization_repeats=False,
        interpretation="Text omission changes field names, sequence length and token positions as well as UNK text. It is not an isolated UNK-token causal effect.")
    write_json(out/"trajectory_summary.json",report)
    return report


def audit_benchmark_reproduction(model, probabilities, matched, meta, y, out):
    """Audit numerical agreement separately from discontinuous argmax decisions.

    Keep the pre-existing probability tolerance. A changed class is acceptable
    for this paired CPU diagnostic only when the saved class is a maximum of
    its saved probability vector and the change is bounded by the measured
    probability perturbation. Never replace a CPU prediction with a saved one.
    This audit does not establish the device/kernel cause of numerical drift.
    """
    import numpy as np
    import pandas as pd
    count=len(meta)
    validate_probs(probabilities,count)
    expected=matched[["prob_"+label.lower() for label in LABELS]].to_numpy(dtype=float)
    validate_probs(expected,count)
    values=pd.to_numeric(matched.pred_label_id,errors="raise").to_numpy(dtype=float)
    require(values.shape==(count,) and np.isfinite(values).all()
            and (values==np.floor(values)).all() and np.isin(values,range(6)).all(),
            "Invalid historical clean class IDs")
    historical=values.astype(np.int64)
    current=probabilities.argmax(1)
    row=np.arange(count)
    error=np.abs(np.asarray(probabilities,dtype=float)-expected)
    row_error=error.max(1)
    close=np.isclose(probabilities,expected,atol=ATOL,rtol=RTOL).all(1)
    # A logits argmax may select a different co-maximum after softmax rounding;
    # accept exact probability ties, but not a non-maximal historical class.
    historical_is_maximum=expected[row,historical]==expected.max(1)
    changed=historical!=current
    historical_gap=expected[row,historical]-expected[row,current]
    current_gap=np.asarray(probabilities,dtype=float)[row,current]-np.asarray(probabilities,dtype=float)[row,historical]
    pair_bound=error[row,historical]+error[row,current]
    arithmetic_slack=16*np.finfo(np.float64).eps
    boundary=(historical_is_maximum & close & (historical_gap>=0)
              & (historical_gap<=pair_bound+arithmetic_slack)
              & (current_gap<=pair_bound+arithmetic_slack))
    ordered_historical=np.sort(expected,axis=1)
    ordered_current=np.sort(np.asarray(probabilities,dtype=float),axis=1)
    details=pd.DataFrame(dict(sequence_id=meta.sequence_id.to_numpy(),episode_id=meta.episode_id.to_numpy(),
        sequence_end_row=meta.sequence_end_row.to_numpy(),model=model,
        historical_prediction=np.asarray(LABELS)[historical],current_prediction=np.asarray(LABELS)[current],
        historical_predicted_label_id=historical,current_predicted_label_id=current,
        decision_changed=changed,probabilities_within_tolerance=close,
        historical_label_is_probability_maximum=historical_is_maximum,
        row_max_abs_probability_difference=row_error,
        historical_top_two_margin=ordered_historical[:,-1]-ordered_historical[:,-2],
        current_top_two_margin=ordered_current[:,-1]-ordered_current[:,-2],
        historical_old_minus_new_class_probability=historical_gap,
        current_new_minus_old_class_probability=current_gap,
        measured_pair_perturbation_bound=pair_bound,boundary_consistent_change=changed & boundary))
    for j,label in enumerate(LABELS):
        details["historical_prob_"+label.lower()]=expected[:,j]
        details["current_prob_"+label.lower()]=probabilities[:,j]
    # Preserve full evidence before deciding whether downstream inference may run.
    details.to_csv(out/f"benchmark_clean_agreement_{model}.csv",index=False,encoding="utf-8-sig")
    details.loc[changed].to_csv(out/f"benchmark_clean_decision_changes_{model}.csv",index=False,encoding="utf-8-sig")
    passed=bool(close.all() and historical_is_maximum.all() and boundary[changed].all())
    _,historical_metrics=classification_metrics(y,historical)
    _,current_metrics=classification_metrics(y,current)
    return dict(model=model,endpoint_count=count,passed=passed,
        status=("NUMERICAL_AGREEMENT_WITH_BOUNDARY_CHANGES" if changed.any()
                else "NUMERICAL_AND_DECISION_AGREEMENT") if passed else "REPRODUCTION_MISMATCH",
        probability_comparison_passed=bool(close.all()),probability_mismatch_rows=int((~close).sum()),
        atol=ATOL,rtol=RTOL,max_abs_probability_difference=float(row_error.max()),
        identical_decisions=bool(not changed.any()),decision_change_count=int(changed.sum()),
        decision_change_fraction=float(changed.mean()),boundary_consistent_change_count=int((changed & boundary).sum()),
        unexplained_decision_change_count=int((changed & ~boundary).sum()),
        historical_label_inconsistency_count=int((~historical_is_maximum).sum()),
        historical_metrics=historical_metrics,current_cpu_metrics=current_metrics,
        delta_cpu_minus_historical_macro_f1=current_metrics["macro_f1"]-historical_metrics["macro_f1"],
        delta_cpu_minus_historical_accuracy=current_metrics["accuracy"]-historical_metrics["accuracy"],
        paired_benchmark_baseline="Current CPU FP32 clean predictions; the mask14 condition uses the same engine",
        interpretation="Probability agreement within the original tolerance does not imply identical decisions or metrics. All decision changes are disclosed; no historical prediction is substituted. The specific numerical/backend cause is not established by this audit.")


def evaluate_benchmark_mask(context, engine, rule, clean, out, work):
    """Paired diagnostic on the entire original common test population, without adaptation."""
    import numpy as np
    import pandas as pd
    frame=clean["frame"]
    population=benchmark_population(context,frame)
    ids,meta,y=population["indices"],population["metadata"],population["y"]
    protocol=dict(status="STARTED",windows=len(frame),common_endpoints=len(meta),episodes=int(meta.episode_id.nunique()),
        conditions=["clean","fixed_mask14"],available_fields=list(AVAILABLE_FIELDS),masked_fields=list(UNAVAILABLE_FIELDS),
        mask_selection="Fixed field names from the reviewed NOS3 mapping, identical for every benchmark row and class",
        population="All original common L10 test endpoints; L1 evaluated on the same endpoints",
        serializer="observable22_fixed_order_is_v1",missing_placeholder="UNK",seed=SEED,
        model_selection=False,threshold_tuning=False,labels_used_for_mask=False,
        clean_embedding_source="Hash-verified existing test embeddings, with the independent clean-control reproduction gate",
        paired_baseline="Current CPU FP32 clean predictions, not the historical saved predictions",
        clean_reproduction_policy=dict(probability_atol=ATOL,probability_rtol=RTOL,
            exact_decisions_required=False,
            changed_decisions="Permitted only with numerical probability agreement, a valid historical maximum and measured perturbation bounds; export every change",
            historical_results_overwritten=False,probability_tolerance_changed=False),
        inference_precision="float32",formal_external_validation=False,
        limitations=["Post-hoc diagnostic proposed after NOS3 results; no model or threshold selected from these outcomes.",
                     "The fixed 14-field mask does not reproduce NOS3's conditional 14/15/18-field missingness or its field-value distribution.",
                     "This comparison estimates the paired effect of this fixed mask on this benchmark, not a decomposition of domain shift.",
                     "All six class outputs are retained. No statistical significance or independent-window confidence interval is claimed."])
    write_json(out/"benchmark_protocol.json",protocol)
    summary=[];perclass=[];confusions=[];episode_rows=[];decisions={};reproduction=[]
    for condition in ("clean","fixed_mask14"):
        observed=frame[FIELDS].copy() if condition=="clean" else mask14(frame)
        if condition=="clean":
            path=context["roots"]["step9"]/"test_window_embeddings.npz"
            pin(path,context["contract"]["step9_artifacts"][path.name])
            with np.load(path,allow_pickle=False) as z:
                require(np.array_equal(z["row_indices"],np.arange(len(frame))),"Full cached embedding row order differs")
                embeddings=z["embeddings"].copy()
            require(embeddings.dtype==np.float32 and embeddings.shape==(len(frame),768)
                    and np.isfinite(embeddings).all(),"Full cached embedding array differs")
        else:
            groups=collections.defaultdict(list)
            for row_id,values in enumerate(observed.itertuples(index=False,name=None)):
                groups[serial(dict(zip(FIELDS,values)))].append(row_id)
            unique_texts=list(groups)
            protocol.update(masked_unique_texts=len(unique_texts),
                embedding_reuse="Identical serialized inputs are encoded once; every original window and endpoint remains in evaluation")
            write_json(out/"benchmark_protocol.json",protocol)
            print(f"Benchmark mask14: encoding {len(unique_texts):,} unique inputs for {len(frame):,} windows on CPU; no training.",flush=True)
            embeddings=np.empty((len(frame),768),dtype=np.float32)
            last_progress=time.monotonic()
            engine.cache.clear()
            for lo in range(0,len(unique_texts),128):
                hi=min(lo+128,len(unique_texts))
                texts=unique_texts[lo:hi]
                vectors=engine.embed(texts)
                for text,vector in zip(texts,vectors):
                    embeddings[groups[text]]=vector
                engine.cache.clear()
                if time.monotonic()-last_progress>=20 or hi==len(unique_texts):
                    print(f"  Mask14 unique embeddings: {hi:,}/{len(unique_texts):,}",flush=True)
                    last_progress=time.monotonic()
            del groups,unique_texts
        for model in (["TST_L1","TST_L10"]+(["Rule_L10"] if rule is not None else [])):
            probabilities=None;scores=None
            if model.startswith("TST"):
                length=int(model.split("L")[-1]);indices=ids[:,-1:] if length==1 else ids
                probabilities=engine.predict(length,embeddings,indices)
                validate_probs(probabilities,len(meta));pred=probabilities.argmax(1)
                if condition=="clean":
                    rel=f"L{length}/predictions/test_predictions_L{length}.csv"
                    path=context["tst_root"]/rel;pin(path,context["r11"]["artifacts"][rel])
                    saved=pd.read_csv(path,keep_default_na=False)
                    require(not saved.sequence_end_row.duplicated().any(),"Duplicate full clean prediction endpoints")
                    match=saved.set_index("sequence_end_row").loc[ids[:,-1]]
                    require(match.episode_id.tolist()==meta.episode_id.tolist(),"Full clean prediction episode mismatch")
                    audit=audit_benchmark_reproduction(model,probabilities,match,meta,y,out)
                    reproduction.append(audit)
                    write_json(out/"benchmark_clean_reproduction.json",reproduction)
                    require(audit["passed"],f"Full clean numerical/decision audit failed: {model}; see benchmark_clean_reproduction.json and benchmark_clean_agreement_{model}.csv")
                    print(f"  {model} clean audit: max probability difference {audit['max_abs_probability_difference']:.8g}; "
                          f"{audit['decision_change_count']:,} boundary decision changes retained and recorded.",flush=True)
            else:
                scores=rule["env"]["aggregate_evidence"](observed,ids,10)
                pred,_=rule["env"]["classify"](scores,rule["thresholds"])
            rows,metrics=classification_metrics(y,pred)
            summary.append(dict(condition=condition,model=model,**metrics))
            perclass.extend(dict(condition=condition,model=model,**row) for row in rows)
            cm=np.bincount(6*y+pred,minlength=36).reshape(6,6)
            confusions.extend(dict(condition=condition,model=model,true_label=a,predicted_label=b,count=int(cm[i,j]))
                for i,a in enumerate(LABELS) for j,b in enumerate(LABELS))
            record=dict(sequence_id=meta.sequence_id.to_numpy(),episode_id=meta.episode_id.to_numpy(),
                        sequence_end_row=ids[:,-1],condition=condition,model=model,
                        true_label=np.asarray(LABELS)[y],true_label_id=y,
                        prediction=np.asarray(LABELS)[pred],predicted_label_id=pred)
            if probabilities is not None:
                record.update({"prob_"+name.lower():probabilities[:,j] for j,name in enumerate(LABELS)})
            else:
                record.update({"rule_score_"+name.lower():scores[:,j] for j,name in enumerate(LABELS[1:])})
            pd.DataFrame(record).to_csv(out/f"benchmark_{condition}_{model}_predictions.csv",index=False,encoding="utf-8-sig")
            decisions[(condition,model)]=pred.copy()
            for episode,group in meta.groupby("episode_id",sort=False):
                indices=group.index.to_numpy();local=np.bincount(6*y[indices]+pred[indices],minlength=36)
                item=dict(condition=condition,model=model,episode_id=episode,endpoint_count=len(indices))
                item.update({f"cm_{i}_{j}":int(local[6*i+j]) for i in range(6) for j in range(6)})
                episode_rows.append(item)
            # Keep completed stages reviewable even if a later condition fails.
            write_csv(out/"benchmark_summary.csv",summary)
            write_csv(out/"benchmark_per_class.csv",perclass)
            write_csv(out/"benchmark_confusion_counts.csv",confusions)
            print(f"  {condition} / {model}: {len(meta):,} common endpoints evaluated.",flush=True)
        del embeddings
    contrasts=[];changes=[]
    for model in sorted({r["model"] for r in summary}):
        a,b=[next(r for r in summary if r["model"]==model and r["condition"]==condition)
             for condition in ("clean","fixed_mask14")]
        contrasts.append(dict(model=model,direction="fixed_mask14 minus clean",endpoint_count=len(meta),
            delta_macro_f1=b["macro_f1"]-a["macro_f1"],delta_accuracy=b["accuracy"]-a["accuracy"],
            delta_normal_false_alarm_fraction=b["normal_false_alarm_fraction"]-a["normal_false_alarm_fraction"]))
        before,after=decisions[("clean",model)],decisions[("fixed_mask14",model)]
        changes.extend(dict(model=model,sequence_id=int(meta.iloc[i].sequence_id),episode_id=str(meta.iloc[i].episode_id),
            true_label=LABELS[int(y[i])],clean_prediction=LABELS[int(before[i])],masked_prediction=LABELS[int(after[i])],
            decision_changed=bool(before[i]!=after[i])) for i in range(len(meta)))
    write_csv(out/"benchmark_paired_differences.csv",contrasts)
    write_csv(out/"benchmark_endpoint_changes.csv",changes)
    write_csv(out/"benchmark_episode_confusions.csv",episode_rows)
    protocol.update(status="COMPLETE",completed_models=sorted({r["model"] for r in summary}),summary=summary,
        historical_clean_audit=reproduction)
    write_json(out/"benchmark_protocol.json",protocol)
    return protocol


def summarize_predictions(predictions, endpoints, out):
    summary = []
    confusion = []
    models = sorted({row["model"] for row in predictions})
    for model in models:
        for view in VIEWS:
            for population in ("primary_middle_command", "secondary_command_controls"):
                rows = [r for r in predictions if (r["model"], r["view"], r["population"]) == (model, view, population)]
                require(len(rows) == (10 if population == "primary_middle_command" else 20), "Prediction population is incomplete")
                normal = [r for r in rows if r["intervention_truth"] == "NORMAL"]
                replay = [r for r in rows if r["intervention_truth"] == "TELECOMMAND_REPLAY"]
                summary.append(dict(model=model, view=view, population=population, endpoint_count=len(rows),
                                    normal_count=len(normal), normal_non_normal_predictions=sum(r["prediction"] != "NORMAL" for r in normal),
                                    replay_count=len(replay), replay_predicted_as_replay=sum(r["prediction"] == "TELECOMMAND_REPLAY" for r in replay),
                                    replay_predicted_as_normal=sum(r["prediction"] == "NORMAL" for r in replay),
                                    replay_predicted_as_other_attack=sum(r["prediction"] not in ("NORMAL", "TELECOMMAND_REPLAY") for r in replay)))
                for truth in ("NORMAL", "TELECOMMAND_REPLAY"):
                    for pred in LABELS:
                        confusion.append(dict(model=model, view=view, population=population, intervention_truth=truth,
                                              prediction=pred, count=sum(r["intervention_truth"] == truth and r["prediction"] == pred for r in rows)))
    paired = []
    for model in models:
        for endpoint in endpoints:
            lookup = {r["view"]: r for r in predictions if r["model"] == model and r["run"] == endpoint["run"]
                      and r["stage"] == endpoint["stage"]}
            a, b = lookup[VIEWS[0]], lookup[VIEWS[1]]
            row = dict(endpoint, model=model, observed_prediction=a["prediction"], freshness_unk_prediction=b["prediction"],
                       decision_changed=a["prediction"] != b["prediction"])
            if model.startswith("TST"):
                row["replay_probability_observed"] = a["prob_telecommand_replay"]
                row["replay_probability_freshness_unk"] = b["prob_telecommand_replay"]
                row["replay_probability_difference"] = a["prob_telecommand_replay"] - b["prob_telecommand_replay"]
            paired.append(row)
    write_csv(out / "endpoint_summary.csv", summary)
    write_csv(out / "confusion_counts.csv", confusion)
    write_csv(out / "history_ablation_paired.csv", paired)
    refs = [dict(r, reference="Exact prior frame bytes within the same run",
                 repeated_frame_flag=r["exact_frame_repeat"]) for r in endpoints]
    write_csv(out / "exact_frame_reference_AUDIT_ONLY.csv", refs)
    # This reference sees raw history, so it is not repeated as a no-history baseline.
    return summary


def seal_result(out):
    manifest = {p.relative_to(out).as_posix(): sha_file(p) for p in sorted(out.rglob("*"))
                if p.is_file() and p.name != "result_manifest.json"}
    write_json(out / "result_manifest.json", manifest)
    target = out.with_suffix(".zip")
    require(not target.exists(), "Result ZIP already exists; refusing to overwrite")
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                z.write(p, f"{out.name}/{p.relative_to(out).as_posix()}")
    with zipfile.ZipFile(target) as z:
        require(z.testzip() is None, "Result ZIP integrity check failed")
        for name, digest in manifest.items():
            require(sha_bytes(z.read(out.name + "/" + name)) == digest, "Result ZIP member hash mismatch")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nos3-zip", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--tst-root", type=Path)
    parser.add_argument("--rule-source", type=Path)
    parser.add_argument("--extended", action="store_true", help="Add all common NOS3 endpoints, clause omission and the full paired benchmark mask14 diagnostic")
    parser.add_argument("--previous-result", type=Path, help="Reviewed 1.1 result ZIP; defaults to the original result under outputs/shared_nominal_v2/nos3_detector_evaluation")
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--threads", type=int, choices=(1, 2, 3, 4), default=2)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    output_root = args.output_root.resolve() if args.output_root else root / "outputs/shared_nominal_v2/nos3_detector_evaluation"
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = output_root / ("run_" + stamp)
    out.mkdir(parents=True, exist_ok=False)
    PINS.clear()
    script_hash = pin(Path(__file__))
    shutil.copy2(Path(__file__), out / "nos3_detector_evaluation.py")
    report = dict(code_version=VERSION, status="STARTED", started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  project_root=str(root), input_zip=str(args.nos3_zip.resolve()), script_sha256=script_hash,
                  analysis_kind="Exploratory frozen-detector inference on partial NOS3 observations",
                  formal_evaluation=False, model_input_ready_for_formal_evaluation=False,
                  detector_evaluation_performed=False, trained=False, additional_live_commands_sent=0,
                  seed=SEED, device="cpu", threads=args.threads, errors=[], rule_errors=[],
                  extended_requested=args.extended,
                  stages=dict(events="PENDING",trajectory="PENDING" if args.extended else "NOT_REQUESTED",
                              benchmark="PENDING" if args.extended else "NOT_REQUESTED"),
                  limitations=["Five paired repeats within one continuous NOS3 session, not five independent sessions.",
                               "Event-level intervention labels; silent windows have no assigned class truth.",
                               "Fourteen fields were never observed; the trained missingness range does not establish validity here.",
                               "Exact frame history is computed before the model; it is not TST temporal reasoning.",
                               "Original replay frame is outside the preceding ten-second model input.",
                               "Offline evidence-time bins do not measure online detection latency.",
                               "No claim of six-class external validation, RF attack coverage, authentication or anti-replay bypass."])
    write_json(out / "summary.json", report)
    work = out / "verified_inputs"
    exit_code = 1
    active_stage="events"
    try:
        print(f"NOS3 detector evaluation: {VERSION}", flush=True)
        print("CPU inference only | No training | No live commands | Frozen seed 20260610", flush=True)
        print("[1/5] Verify the completed NOS3 campaign and regenerate causal partial inputs.", flush=True)
        runs, mapping = verify_campaign(args.nos3_zip.resolve(), work)
        endpoints = prepare_inputs(runs, out)
        write_json(out / "schema_contract.json", mapping)
        previous=None
        if args.extended:
            previous_path=(resolve_root(root,args.previous_result) if args.previous_result else
                           root/"outputs/shared_nominal_v2/nos3_detector_evaluation"/PREVIOUS_NAME)
            previous=verify_previous_result(previous_path,out)
        write_json(out / "evaluation_protocol.json", dict(
            models=["TST_L1", "TST_L10", "Frozen_Rule_L10_if_saved_specification_is_verified"], views=list(VIEWS),
            primary="Second command in each run; 5 normal and 5 replay endpoints, paired by campaign pair",
            secondary="First and third command controls; 20 ordinary command endpoints",
            label_order=LABELS, bin_seconds=1, decision="Unmodified six-class argmax; no thresholds tuned on NOS3",
            rule="Previously fixed Rule v1 thresholds; unavailable rule is reported separately, never substituted",
            history_ablation="Only sequence_freshness_status becomes UNK; all other 21 fields and endpoints unchanged",
            technical_control="First 16 saved common endpoints reproduce embeddings and L1/L10 probabilities",
            exact_frame_reference="Raw within-run byte history diagnostic, not a learned detector or no-history baseline"))
        if args.extended:
            write_json(out/"extended_protocol.json",dict(
                timing="Post-hoc diagnostics after the reviewed v1.1 NOS3 result; no subsequent tuning or model selection",
                primary_unchanged=True,main_seed=SEED,serialization_conditions=list(SERIALIZATIONS),
                trajectory="All 809 within-run L10 endpoints; L1 and Rule use these same endpoints",
                quiet_windows="No assigned class truth; report output distributions only",
                rule="Same structured values for both text serializations; do not duplicate Rule trials",
                benchmark="Full original common test endpoints, clean versus fixed label-independent mask14",
                unavailable_fields=list(UNAVAILABLE_FIELDS),additional_live_commands=0,training=False))
        print("[2/5] Verify the existing encoder/tokenizer and prescribed L1/L10 checkpoints.", flush=True)
        context = model_context(root, args.tst_root)
        if previous is not None:
            old=previous["provenance"]
            require(old["source_contract"]==context["contract"] and old["step11_run_id"]==context["r11"]["run_id"]
                    and old["tokenizer_sha256"]==context["tokenizer_sha"],"The previously evaluated model/input lineage changed")
        write_json(out / "model_provenance.json", dict(tst_root=str(context["tst_root"]),
                   source_contract=context["contract"], step11_run_id=context["r11"]["run_id"],
                   settings=context["r11"]["settings"], tokenizer_sha256=context["tokenizer_sha"]))
        engine = FrozenEngine(context, args.threads)
        report["runtime"] = engine.versions
        print("[3/5] Reproduce 16 fixed clean-control endpoints before any NOS3 prediction.", flush=True)
        clean = clean_control(context, engine, out)
        try:
            rule = load_frozen_rule(context, args.rule_source, clean, out)
            report["rule_status"] = "VERIFIED_FROZEN_RULE"
        except Exception as error:
            rule = None
            report["rule_status"] = "UNAVAILABLE_NO_SUBSTITUTE"
            report["rule_errors"].append(str(error))
            write_json(out / "rule_provenance.json", dict(status=report["rule_status"], error=str(error)))
            (out / "rule_failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"Rule comparison unavailable: {error}", flush=True)
        print("[4/5] Predict the same command endpoints with L1/L10 and both input views.", flush=True)
        predictions = evaluate(runs, engine, rule, out)
        endpoint_summary = summarize_predictions(predictions, endpoints, out)
        report.update(detector_evaluation_performed=True, endpoint_count=30, primary_endpoint_count=10,
                      command_predictions=len(predictions), endpoint_summary=endpoint_summary,
                      status="COMPLETE" if rule else "TST_COMPLETE_RULE_UNAVAILABLE")
        report["stages"]["events"]="COMPLETE"
        write_json(out/"summary.json",report)
        if previous is not None:
            write_json(out/"previous_prediction_reproduction.json",compare_previous_predictions(previous,predictions))
            active_stage="trajectory";report["stages"][active_stage]="RUNNING"
            print("[EXT 1/2] All 809 common NOS3 endpoints and both missing-value serializations.",flush=True)
            report["trajectory"]=evaluate_trajectory(runs,engine,rule,out,previous)
            report["stages"][active_stage]="COMPLETE";write_json(out/"summary.json",report)
            active_stage="benchmark";report["stages"][active_stage]="RUNNING"
            print("[EXT 2/2] Full paired clean/mask14 benchmark at the original common endpoints.",flush=True)
            report["benchmark"]=evaluate_benchmark_mask(context,engine,rule,clean,out,work)
            report["stages"][active_stage]="COMPLETE"
            report["status"]="EXTENDED_COMPLETE" if rule else "EXTENDED_COMPLETE_RULE_UNAVAILABLE"
        print("[5/5] Verify unchanged inputs and preserve all predictions in one ZIP.", flush=True)
        for path, digest in list(PINS.items()):
            require(sha_file(path) == digest, f"Input changed during evaluation: {path}")
        exit_code = 0 if rule else 2
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "PARTIAL_DIAGNOSTICS" if report["detector_evaluation_performed"] else "FAILED"
        report["stages"][active_stage]="FAILED"
        exit_code=3 if report["detector_evaluation_performed"] else 1
        if isinstance(error,KeyboardInterrupt):
            report["status"]="INTERRUPTED";exit_code=130
        report["errors"].append(f"{type(error).__name__}: {error}")
        (out / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"Evaluation stopped: {error}", file=sys.stderr, flush=True)
    finally:
        report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(out / "summary.json", report)
        write_json(out / "input_sha256.json", PINS)
        # Original NOS3 evidence is already in the supplied campaign ZIP; no need to duplicate it.
        if work.exists():
            shutil.rmtree(work)
        target = seal_result(out)
        print(json.dumps({k: report[k] for k in ("status", "detector_evaluation_performed", "errors", "rule_errors")}, indent=2))
        print(f"ZIP_SHA256={sha_file(target)}", flush=True)
        print(f"UPLOAD_THIS={target}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
