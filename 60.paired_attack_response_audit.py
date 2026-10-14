#!/usr/bin/env python3
"""Step60: post-hoc matched nominal/attack response audit of revised Step49.

Run from the project code directory (CPU only; numpy + pandas):
    python "60.paired_attack_response_audit.py"
Optional explicit source (a completed run directory, including extracted review ZIP):
    python "60.paired_attack_response_audit.py" --step49-run "PATH"

No model training, inference, raw-feature comparison, threshold changes, or input edits.
Every attack L10 endpoint is paired with its exact nominal L10 counterfactual.
All endpoints, including unchanged semantic inputs, remain in ALL results.
Main descriptive metric: paired_correct = nominal prediction NORMAL AND attack
prediction its scenario label. Rates are endpoint-weighted, then averaged over
all fixed head seeds. NORMAL false alarms use unique NORMAL endpoints only.
Bootstrap resamples whole shared nominal episodes jointly across attacks,
models and regimes. It does NOT resample heads/encoder/generator seeds; intervals
are conditional on this realization and these frozen models, not external validity.
The audit is retrospective. A changed 22-field input is an observable projection
change, not proof of attack-specific semantics or absence of simulator fingerprints.

Output: outputs/shared_nominal_v2/60_paired_attack_response/run_<UTC>/
and a sibling *_review.zip. Existing Step49 outputs are read-only. A new run is
created each time. Preflight validates sources/pairs but does not compute statistics.
Consumed files alone are hash-verified; omitted upstream caches are not required.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import platform
import shutil
import sys
import time
import traceback
import zipfile

# Bound BLAS parallelism for this small CPU audit, respecting explicit user settings.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
import numpy as np
import pandas as pd

VERSION = "paired_attack_response_audit_v1"
SOURCE_VERSION = "shared_nominal_A_parameter_stress_v1"
CONDITIONS = ("P_reference", "P_mild", "P_moderate", "P_severe")
LABELS = ("NORMAL", "RF_LINK_JAMMING", "GNSS_SPOOFING",
          "UNAUTHORIZED_TELECOMMAND_INJECTION", "TELECOMMAND_REPLAY",
          "GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE")
FIELDS = ("observation_stage", "path_type", "direction", "source_role",
          "destination_role", "protocol_context", "operational_phase", "message_type",
          "service_context", "command_intent", "payload_or_parameter_context",
          "sequence_freshness_status", "control_security_state", "signal_quality_state",
          "tracking_stability", "navigation_consistency", "timing_consistency",
          "decoder_status", "command_execution_result", "retry_timeout_state",
          "session_state", "state_transition_consistency")
META = ("scenario_id", "episode_id", "episode_type", "episode_step", "window_id", "timestamp", "label")
FAMILIES = ("TST", "Mean_MLP", "Max_MLP", "MeanMax_MLP", "LSTM",
            "ProbabilityMean", "MajorityVote", "EWMA", "Rule_L10")
RUNS = tuple((f, s) for f in FAMILIES[:5] for s in range(20260610, 20260620)) + tuple((f, -1) for f in FAMILIES[5:])
METRICS = ("paired_correct", "target_recall", "target_gain", "attack_alarm",
           "normal_to_any_attack", "normal_to_wrong_attack", "label_change")
STRATA = ("ALL", "CHANGED", "UNCHANGED")
DEFAULT_SOURCE = Path("outputs/shared_nominal_v2/49_parameter_stress/run_parameter_stress_v1")
DEFAULT_OUTPUT = Path("outputs/shared_nominal_v2/60_paired_attack_response")


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def object_hash(value):
    # Exactly the canonical JSON digest used by Step47/49.
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def check_identity(value, description):
    require(isinstance(value, dict) and value.get("identity") == object_hash(
        {k: v for k, v in value.items() if k != "identity"}), description + " identity changed")


def safe_path(root, name):
    require(isinstance(name, str) and name and "\\" not in name, "Unsafe artifact path")
    relative = Path(name)
    require(not relative.is_absolute() and not PureWindowsPath(name).drive and ".." not in relative.parts,
            "Unsafe artifact path: " + name)
    root = Path(root).resolve(); path = (root / relative).resolve()
    require(root in path.parents, "Artifact escaped source folder: " + name)
    return path


def atomic_replace(temp, target):
    for attempt in range(6):
        try:
            Path(temp).replace(target)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.2 * (attempt + 1))


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    atomic_replace(temp, path)


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    compression = {"method": "gzip", "mtime": 0} if path.name.endswith(".gz") else None
    pd.DataFrame(rows).to_csv(temp, index=False, encoding="utf-8-sig", compression=compression)
    atomic_replace(temp, path)


def write_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    atomic_replace(temp, path)


class BoundReader:
    """Hash-bind every consumed artifact to the completed upstream manifest."""
    def __init__(self, root, hashes):
        self.root = Path(root).resolve()
        self.hashes = hashes
        self.observed = {}

    def path(self, name):
        expected = self.hashes.get(name)
        require(isinstance(expected, str) and len(expected) == 64, "Unbound source artifact: " + name)
        path = safe_path(self.root, name)
        require(path.is_file(), "Missing source artifact: " + str(path))
        actual = digest(path)
        require(actual == expected, "Source hash changed: " + str(path))
        self.observed[name] = actual
        return path

    def json(self, name):
        return read_json(self.path(name))

    def csv(self, name):
        frame = pd.read_csv(self.path(name), keep_default_na=False, encoding="utf-8-sig")
        require(not frame.isna().any().any(), "Missing values: " + name)
        return frame

    def arrays(self, name, keys):
        with np.load(self.path(name), allow_pickle=False) as z:
            require(set(keys) <= set(z.files), "Missing arrays: " + name)
            return {k: z[k] for k in keys}

    def recheck(self):
        for name in list(self.observed):
            self.path(name)


def locate_source(root, explicit):
    root = Path(root).resolve()
    if explicit is not None:
        p = Path(explicit).expanduser()
        p = p if p.is_absolute() else root / p
        require(p.is_dir(), "--step49-run must be an existing completed run directory: " + str(p))
        return p.resolve()
    base = root / "outputs/shared_nominal_v2/49_parameter_stress"
    candidates = []
    for p in sorted(base.glob("run_*/run_manifest.json")):
        try:
            if read_json(p).get("status") == "PARAMETER_STRESS_COMPLETE":
                candidates.append(p.parent.resolve())
        except (OSError, ValueError):
            continue
    require(len(candidates) != 0, "No completed revised Step49 run. Expected " + str(root / DEFAULT_SOURCE)
            + "; or use --step49-run PATH. The old generator_b run is not accepted.")
    require(len(candidates) == 1, "Multiple completed Step49 runs; use --step49-run explicitly:\n"
            + "\n".join(map(str, candidates)))
    return candidates[0]


def open_source(folder):
    manifest_path = folder / "run_manifest.json"
    require(manifest_path.is_file(), "Missing Step49 run_manifest.json")
    manifest = read_json(manifest_path)
    require(manifest.get("status") == "PARAMETER_STRESS_COMPLETE", "Need completed revised Step49 parameter-stress run")
    check_identity(manifest, "Step49 manifest")
    require(manifest.get("training") is False and manifest.get("stress_used_for_selection") is False,
            "Step49 training/selection contract differs")
    reader = BoundReader(folder, dict(manifest["output_sha256"]))
    reader.hashes["run_manifest.json"] = digest(manifest_path)
    reader.path("run_manifest.json")
    protocol = reader.json("protocol.json"); check_identity(protocol, "Step49 protocol")
    require(protocol.get("version") == SOURCE_VERSION and protocol["identity"] == manifest["protocol_identity"],
            "Unsupported Step49 protocol; use revised original-maker parameter stress")
    require(protocol.get("training") is False and protocol.get("stress_used_for_selection") is False,
            "Source protocol allows training/selection on stress")
    require(tuple(protocol["fields"]) == FIELDS and tuple(protocol["generator"]["classes"]) == LABELS,
            "Source observable/label schema differs")
    require(set(protocol["generator"]["conditions"]) == set(CONDITIONS), "Source condition inventory differs")
    require(protocol["generator"]["version"] == "original_maker_parameter_interventions_v1", "Source generator differs")
    runs = tuple((f, -1 if s is None else s) for f, s in protocol["runs"])
    require(runs == RUNS and tuple(protocol["families"]) == FAMILIES, "Source model/seed inventory differs")
    selection = reader.json("validation_selection.json")
    require(selection == protocol["frozen_selection"] and selection["selected_pool"] in FAMILIES[1:4]
            and selection.get("all_ten_head_seeds_retained") is True, "Frozen validation selection differs")
    source_code = reader.path("source/49.generator_b_transfer_evaluation.py")
    require(digest(source_code) == protocol["script_sha256"], "Saved Step49 source differs from producer")
    return reader, manifest, protocol


def check_stage(reader, base, filename, stage, condition, names, protocol):
    done = reader.json(base + "/" + filename); check_identity(done, base + " completion")
    require(done["contract"] == object_hash(dict(protocol=protocol["identity"], stage=stage, condition=condition)),
            "Stage contract differs: " + base)
    for name in names:
        require(done["artifact_sha256"].get(name) == reader.hashes.get(base + "/" + name),
                "Stage/manifest hash mismatch: " + base + "/" + name)
    return done["result"]


def integer_array(values, name):
    values = np.asarray(values)
    require(np.issubdtype(values.dtype, np.integer), "Expected integer values: " + name)
    return values.astype(np.int64, copy=False)


def make_pairs(meta, block_ids):
    """Return attack and matching nominal positions without positional assumptions."""
    require(len(meta) == len(block_ids), "Pairing block length mismatch")
    keys = pd.MultiIndex.from_arrays([block_ids, meta.sequence_start_step, meta.sequence_end_step, meta.sequence_length])
    y = meta.label_id.to_numpy()
    nominal = np.flatnonzero(y == 0); attack = np.flatnonzero(y != 0)
    require(len(nominal) > 0 and len(attack) > 0, "No nominal/attack pairs")
    lookup = pd.Series(nominal, index=keys[nominal])
    require(lookup.index.is_unique, "Duplicate nominal pairing key")
    for label in np.unique(y[attack]):
        idx = np.flatnonzero(y == label); ak = keys[idx]
        require(ak.is_unique, "Duplicate attack pairing key")
        require(len(idx) == len(nominal) and set(ak) == set(lookup.index),
                "Incomplete one-to-one nominal counterpart pairing for label " + str(label))
    paired = lookup.reindex(keys[attack]).to_numpy()
    require(not pd.isna(paired).any(), "Missing nominal counterpart")
    return attack, paired.astype(np.int64)


def validate_cohort(frame, meta, blocks, seq, raw, predictions, protocol):
    require(tuple(frame.columns) == META + FIELDS, "Records schema/order differs")
    require(not frame.eq("").any().any(), "Empty observable or metadata cell")
    n = len(frame); e = len(meta)
    require(n == 6 * protocol["generator"]["rows_per_class"], "Incomplete record population")
    require(meta.sequence_id.is_unique and meta.episode_id.ne("").all(), "Duplicate endpoint ID or empty episode")
    for col in ("sequence_id", "sequence_start_row", "sequence_end_row", "sequence_start_step",
                "sequence_end_step", "sequence_length", "label_id"):
        integer_array(meta[col], col)
    for col in ("sequence_id", "block_id"):
        integer_array(blocks[col], col)
    require(np.array_equal(meta.sequence_id, blocks.sequence_id) and np.array_equal(meta.episode_id, blocks.episode_id),
            "Endpoint block metadata is misaligned")
    ix = integer_array(seq["indices"], "sequence indices"); y = integer_array(seq["y"], "sequence y")
    require(ix.shape == (e, 10) and y.shape == (e,) and np.all((ix >= 0) & (ix < n)), "Invalid sequence dimensions/ranges")
    require(np.all(np.diff(ix, axis=1) == 1), "Nonconsecutive sequence rows")
    require(np.array_equal(ix[:, 0], meta.sequence_start_row) and np.array_equal(ix[:, -1], meta.sequence_end_row),
            "Sequence row bounds differ")
    require(np.all(meta.sequence_length == 10) and np.all(meta.source_split == "test"), "Need complete test L10 endpoints")
    require(np.array_equal(y, meta.label_id) and np.all((y >= 0) & (y < 6)), "Endpoint label IDs differ")
    require(np.array_equal(np.asarray(LABELS)[y], meta.label), "Endpoint label mapping differs")
    row_steps = integer_array(frame.episode_step, "episode_step")
    row_blocks = integer_array(raw["block_id"], "raw block_id")
    baseline = raw["baseline_observables"]
    benign = integer_array(raw["nominal_benign"], "nominal_benign")
    require(row_blocks.shape == (n,) and benign.shape == (n,) and baseline.shape == (n, 22), "Raw audit dimensions differ")
    require(np.isin(benign, [0, 1]).all() and baseline.dtype.kind in "US", "Invalid nominal audit arrays")
    require(np.array_equal(row_steps[ix[:, 0]], meta.sequence_start_step)
            and np.array_equal(row_steps[ix[:, -1]], meta.sequence_end_step), "Sequence step bounds differ")
    require(np.array_equal(row_blocks[ix[:, -1]], blocks.block_id)
            and np.all(row_blocks[ix] == blocks.block_id.to_numpy()[:, None]), "Sequence crosses paired nominal episode")
    episodes = frame.episode_id.to_numpy(); row_labels = frame.label.to_numpy()
    require(np.all(episodes[ix] == meta.episode_id.to_numpy()[:, None]), "Sequence crosses source episode")
    require(np.all(row_labels[ix] == np.asarray(LABELS)[y, None]), "Sequence crosses scenario label")
    # Reconstruct the full eligible L10 population, preventing quiet-row/endpoint omission.
    expected_ends = []
    episode_blocks = []
    for episode, pos in frame.groupby("episode_id", sort=False).indices.items():
        pos = np.asarray(pos)
        require(np.all(np.diff(pos) == 1) and np.array_equal(row_steps[pos], np.arange(len(pos))),
                "Source episode discontinuity: " + str(episode))
        require(len(np.unique(row_labels[pos])) == 1 and len(np.unique(row_blocks[pos])) == 1, "Episode labels/blocks differ")
        episode_blocks.append((row_labels[pos[0]], int(row_blocks[pos[0]])))
        expected_ends.extend(pos[9:].tolist())
    require(len(episode_blocks) == len(set(episode_blocks)), "One block maps to multiple episodes within a label")
    require(np.array_equal(np.sort(ix[:, -1]), np.sort(expected_ends)), "Incomplete or duplicate eligible endpoint population")
    require(set(row_labels) == set(LABELS), "Missing scenario class")
    normal_rows = np.flatnonzero(row_labels == "NORMAL")
    obs = frame[list(FIELDS)].to_numpy(dtype=str)
    require(np.array_equal(obs[normal_rows], baseline[normal_rows]), "NORMAL differs from its nominal projection")
    for label in LABELS:
        pos = np.flatnonzero(row_labels == label)
        require(len(pos) == len(normal_rows), "Unequal class record counts")
        require(np.array_equal(row_blocks[pos], row_blocks[normal_rows])
                and np.array_equal(row_steps[pos], row_steps[normal_rows])
                and np.array_equal(frame.timestamp.to_numpy()[pos], frame.timestamp.to_numpy()[normal_rows]),
                "Class-specific nominal episode timeline")
        require(np.array_equal(baseline[pos], obs[normal_rows])
                and np.array_equal(benign[pos], benign[normal_rows]), "Class-specific nominal counterfactual")
        require(np.array_equal(obs[pos, :7], obs[normal_rows, :7]), "Class-specific context fields")
    attack, nominal_pair = make_pairs(meta, blocks.block_id.to_numpy())
    require(len(attack) == 5 * int((y == 0).sum()), "Incomplete five-attack pair population")
    pred = integer_array(predictions["predictions"], "predictions")
    require(pred.shape == (len(RUNS), e) and np.isin(pred, np.arange(6)).all(), "Prediction dimensions/labels differ")
    require(np.array_equal(predictions["y"], y) and np.array_equal(predictions["sequence_id"], meta.sequence_id),
            "Predictions/endpoint IDs or labels are misaligned")
    actual_runs = tuple(zip(predictions["families"].tolist(), integer_array(predictions["seeds"], "seed").tolist()))
    require(actual_runs == RUNS, "Prediction family/seed inventory/order differs")
    delta_windows = np.zeros(len(attack), dtype=np.int16)
    delta_cells = np.zeros(len(attack), dtype=np.int16)
    current_changed = np.zeros(len(attack), dtype=bool)
    for start in range(0, len(attack), 256):
        end = min(start + 256, len(attack)); ai = ix[attack[start:end]]; ni = ix[nominal_pair[start:end]]
        require(np.array_equal(baseline[ai], obs[ni]), "Full L10 nominal counterpart differs")
        require(np.array_equal(row_steps[ai], row_steps[ni]), "Full L10 step alignment differs")
        delta = obs[ai] != obs[ni]
        delta_windows[start:end] = delta.any(2).sum(1)
        delta_cells[start:end] = delta.sum(axis=(1, 2))
        current_changed[start:end] = delta[:, -1, :].any(1)
    block_ids, block_codes = np.unique(blocks.block_id.to_numpy(), return_inverse=True)
    normal = np.flatnonzero(y == 0)
    require(len(block_ids) >= 2, "Need at least two nominal episodes for cluster bootstrap")
    require(set(blocks.block_id.iloc[normal]) == set(block_ids), "Missing NORMAL episode cluster")
    pairs = pd.DataFrame(dict(attack_sequence_id=meta.sequence_id.to_numpy()[attack],
        nominal_sequence_id=meta.sequence_id.to_numpy()[nominal_pair],
        attack_episode_id=meta.episode_id.to_numpy()[attack], nominal_episode_id=meta.episode_id.to_numpy()[nominal_pair],
        block_id=blocks.block_id.to_numpy()[attack], sequence_start_step=meta.sequence_start_step.to_numpy()[attack],
        sequence_end_step=meta.sequence_end_step.to_numpy()[attack], label=np.asarray(LABELS)[y[attack]],
        label_id=y[attack], changed_windows_L10=delta_windows, changed_cells_L10=delta_cells,
        current_changed=current_changed, history_only_changed=(delta_windows > 0) & ~current_changed,
        stratum=np.where(delta_windows > 0, "CHANGED", "UNCHANGED")))
    return dict(meta=meta, pairs=pairs, predictions=pred.astype(np.int8), y=y, attack=attack,
        nominal_pair=nominal_pair, normal=normal, block_ids=block_ids, block_codes=block_codes,
        changed=delta_windows > 0, nominal_benign=benign[ix[normal]].any(1),
        endpoint_signature=object_hash(dict(meta=meta.to_dict(orient="list"), blocks=blocks.to_dict(orient="list"))))


def load_condition(reader, protocol, condition):
    g = "generation/" + condition; e = "evaluation/" + condition
    names = ("evaluation_records.csv", "evaluation_endpoints_L10.csv", "endpoint_blocks.csv", "sequences_L10.npz", "measurement_audit.npz")
    gd = check_stage(reader, g, "complete.json", "generation", condition, names, protocol)
    ed = check_stage(reader, e, "transfer_complete.json", "combined_rule_and_neural", condition,
                     ("combined_predictions.npz",), protocol)
    data = validate_cohort(reader.csv(g + "/evaluation_records.csv"), reader.csv(g + "/evaluation_endpoints_L10.csv"),
        reader.csv(g + "/endpoint_blocks.csv"), reader.arrays(g + "/sequences_L10.npz", ("indices", "y")),
        reader.arrays(g + "/measurement_audit.npz", ("block_id", "baseline_observables", "nominal_benign")),
        reader.arrays(e + "/combined_predictions.npz", ("predictions", "y", "sequence_id", "families", "seeds")), protocol)
    require(gd["endpoints"] == ed["endpoints"] == len(data["y"]) and ed["runs"] == len(RUNS)
            and gd["paired_episode_blocks"] == len(data["block_ids"]), "Completion population counts differ")
    return data


def pair_outcomes(normal, attack, target):
    normal = np.asarray(normal); attack = np.asarray(attack); target = np.asarray(target)
    require(normal.shape == attack.shape and normal.ndim == 2 and target.shape == (normal.shape[1],), "Pair outcome shape mismatch")
    correct = attack == target[None, :]; nominal_ok = normal == 0; alarm = attack != 0
    return dict(paired_correct=nominal_ok & correct, target_recall=correct,
        target_gain=correct.astype(np.int16) - (normal == target[None, :]).astype(np.int16),
        attack_alarm=alarm, normal_to_any_attack=nominal_ok & alarm,
        normal_to_wrong_attack=nominal_ok & alarm & ~correct, label_change=normal != attack)


def cluster_sums(values, codes, clusters):
    values = np.asarray(values); codes = np.asarray(codes)
    require(values.ndim == 2 and values.shape[1] == len(codes), "Cluster value shape mismatch")
    return np.asarray([np.bincount(codes, weights=row, minlength=clusters) for row in values], dtype=np.float64)


def nominal_counts(predictions, normal, block_codes, mask, clusters):
    idx = np.asarray(normal)[mask]
    den = np.bincount(np.asarray(block_codes)[idx], minlength=clusters)
    return cluster_sums(np.asarray(predictions)[:, idx] != 0, np.asarray(block_codes)[idx], clusters), den


def bootstrap_weights(clusters, draws, seed):
    require(clusters >= 2 and draws > 0 and seed >= 0, "Invalid bootstrap settings")
    rng = np.random.default_rng(seed)
    weights = np.zeros((draws, clusters), dtype=np.int32)
    for start in range(0, draws, 512):
        stop = min(start + 512, draws)
        choices = rng.integers(0, clusters, size=(stop - start, clusters))
        for j, chosen in enumerate(choices, start):
            weights[j] = np.bincount(chosen, minlength=clusters)
    return weights


def ratio_bootstrap(numerators, denominator, weights):
    """Endpoint-weighted ratios; samples jointly resample complete episode clusters."""
    nums = np.asarray(numerators, dtype=np.float64); den = np.asarray(denominator, dtype=np.float64)
    require(nums.ndim == 2 and nums.shape[1] == len(den) and weights.shape[1] == len(den), "Bootstrap shape mismatch")
    point = np.full(nums.shape[0], np.nan)
    if den.sum() > 0:
        point = nums.sum(1) / den.sum()
    boot_den = weights @ den
    sample = np.full((len(weights), nums.shape[0]), np.nan)
    np.divide(weights @ nums.T, boot_den[:, None], out=sample, where=boot_den[:, None] > 0)
    return point, sample


def interval(samples, alpha=.05):
    values = np.asarray(samples); values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), 0
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2], method="linear")
    return float(lo), float(hi), int(len(values))


def audit_pair_predictions(data, condition, out):
    pred = data["predictions"]; a = data["attack"]; n = data["nominal_pair"]
    mismatch = (pred[:, a] != pred[:, n]) & ~data["changed"][None, :]
    rr, cc = np.nonzero(mismatch)
    rows = [dict(condition=condition, family=RUNS[r][0], seed=RUNS[r][1],
        attack_sequence_id=int(data["pairs"].attack_sequence_id.iloc[c]),
        nominal_sequence_id=int(data["pairs"].nominal_sequence_id.iloc[c]),
        nominal_prediction=int(pred[r, n[c]]), attack_prediction=int(pred[r, a[c]])) for r, c in zip(rr, cc)]
    write_csv(out / condition / "identical_input_disagreements.csv", pd.DataFrame(rows, columns=[
        "condition", "family", "seed", "attack_sequence_id", "nominal_sequence_id", "nominal_prediction", "attack_prediction"]))
    report = dict(condition=condition, endpoints=int(len(data["y"])), normal_endpoints=int(len(data["normal"])),
        attack_pairs=int(len(a)), nominal_episode_clusters=int(len(data["block_ids"])), models=len(RUNS),
        changed_pairs=int(data["changed"].sum()), unchanged_pairs=int((~data["changed"]).sum()),
        identical_input_prediction_disagreements=int(mismatch.sum()),
        pairing="Exact full-L10 nominal projection, episode/step alignment and full endpoint population verified",
        prediction_consistency_status="CONSISTENT" if not len(rr) else "REVIEW_REQUIRED",
        consistency_scope="Equality is a pipeline consistency check, not proof of semantic learning or absence of shortcuts. "
                          "Disagreements can also reflect batch/numerical inference effects; no labels are corrected.")
    write_json(out / condition / "pairing_audit.json", report)
    return report


def summarize_condition(data, condition, weights, out):
    folder = out / condition; folder.mkdir(parents=True, exist_ok=True)
    pred = data["predictions"]; a = data["attack"]; n = data["nominal_pair"]
    target = data["y"][a]; codes = data["block_codes"][a]; clusters = len(data["block_ids"])
    outcomes = pair_outcomes(pred[:, n], pred[:, a], target)
    pair_rows, summaries, nominal_rows, transitions = [], [], [], []
    group_names, cluster_num, cluster_den = [], [], []
    nominal_num, nominal_den = [], []
    family_idx = [np.asarray([i for i, (f, _) in enumerate(RUNS) if f == family]) for family in FAMILIES]
    main_boot = main_point = None

    def summarize_group(nums, den, metrics, stratum, label, run_rows, summary_rows):
        # Denominator identical for all fixed seeds -> averaging cluster numerators
        # is exactly mean per-seed rate, NOT ensemble predictions/confusion matrices.
        total = int(den.sum()); support = int(np.count_nonzero(den))
        means = np.stack([nums[:, ids, :].mean(axis=1) for ids in family_idx])
        point, boot = ratio_bootstrap(means.reshape(-1, clusters), den, weights)
        point = point.reshape(len(FAMILIES), len(metrics))
        boot = boot.reshape(len(weights), len(FAMILIES), len(metrics))
        run_point = nums.sum(2) / total if total else np.full(nums.shape[:2], np.nan)
        for mi, metric in enumerate(metrics):
            for r, (family, seed) in enumerate(RUNS):
                run_rows.append(dict(condition=condition, label=label, stratum=stratum, metric=metric,
                    family=family, seed=seed, numerator=int(nums[mi, r].sum()), denominator=total,
                    rate=float(run_point[mi, r]), nonempty_clusters=support))
            for fi, family in enumerate(FAMILIES):
                lo, hi, valid = interval(boot[:, fi, mi])
                if support < 2:
                    lo = hi = float("nan")
                seeds = run_point[mi, family_idx[fi]]
                sd = float(np.std(seeds, ddof=1)) if len(seeds) > 1 and total else float("nan")
                summary_rows.append(dict(condition=condition, label=label, stratum=stratum, metric=metric,
                    family=family, fixed_runs=len(seeds), denominator=total, nonempty_clusters=support,
                    mean_rate=float(point[fi, mi]), head_seed_sd=sd, ci95_low=lo, ci95_high=hi,
                    bootstrap_valid=valid, bootstrap_invalid=len(weights)-valid,
                    ci_status="UNAVAILABLE" if support < 2 else ("CONDITIONAL_NONEMPTY_DRAWS" if valid < len(weights) else "OK")))
        return point, boot

    for class_id in (None, 1, 2, 3, 4, 5):
        label = "ALL_ATTACKS" if class_id is None else LABELS[class_id]
        cm = np.ones(len(a), dtype=bool) if class_id is None else target == class_id
        for stratum in STRATA:
            sm = np.ones(len(a), dtype=bool) if stratum == "ALL" else (data["changed"] if stratum == "CHANGED" else ~data["changed"])
            mask = cm & sm
            den = np.bincount(codes[mask], minlength=clusters)
            nums = np.stack([cluster_sums(outcomes[k][:, mask], codes[mask], clusters) for k in METRICS])
            group_names.append(label + "/" + stratum); cluster_num.append(nums.astype(np.int32)); cluster_den.append(den)
            point, boot = summarize_group(nums, den, METRICS, stratum, label, pair_rows, summaries)
            if class_id is None and stratum == "ALL":
                main_point = point[:, 0].copy(); main_boot = boot[:, :, 0].copy()
            if class_id is not None:
                for r, (family, seed) in enumerate(RUNS):
                    cells = np.bincount(6 * pred[r, n[mask]].astype(np.int64) + pred[r, a[mask]], minlength=36).reshape(6, 6)
                    for before in range(6):
                        for after in range(6):
                            transitions.append(dict(condition=condition, label=label, stratum=stratum,
                                family=family, seed=seed, nominal_prediction=LABELS[before],
                                attack_prediction=LABELS[after], count=int(cells[before, after])))
    normal_summary = []
    for stratum, mask in (("ALL", np.ones(len(data["normal"]), bool)),
                          ("BENIGN_PRESENT_L10", data["nominal_benign"]),
                          ("NO_BENIGN_L10", ~data["nominal_benign"])):
        nums, den = nominal_counts(pred, data["normal"], data["block_codes"], mask, clusters)
        nominal_num.append(nums.astype(np.int32)); nominal_den.append(den)
        summarize_group(nums[None, :, :], den, ("nominal_false_alarm",), stratum, "NORMAL", nominal_rows, normal_summary)
    write_csv(folder / "paired_endpoint_index.csv.gz", data["pairs"])
    write_npz(folder / "paired_predictions.npz", families=np.asarray([f for f, s in RUNS]), seeds=np.asarray([s for f, s in RUNS]),
        attack_prediction=pred[:, a], nominal_prediction_for_pair=pred[:, n],
        attack_sequence_id=data["pairs"].attack_sequence_id.to_numpy(),
        nominal_sequence_id_for_pair=data["pairs"].nominal_sequence_id.to_numpy(),
        unique_nominal_sequence_id=data["meta"].sequence_id.to_numpy()[data["normal"]],
        unique_nominal_prediction=pred[:, data["normal"]],
        unique_nominal_block_id=data["block_ids"][data["block_codes"][data["normal"]]],
        unique_nominal_benign_L10=data["nominal_benign"], target=target)
    write_npz(folder / "cluster_counts.npz", block_ids=data["block_ids"], groups=np.asarray(group_names),
        metrics=np.asarray(METRICS), families=np.asarray([f for f, s in RUNS]), seeds=np.asarray([s for f, s in RUNS]),
        numerator=np.stack(cluster_num), denominator=np.stack(cluster_den),
        nominal_strata=np.asarray(["ALL", "BENIGN_PRESENT_L10", "NO_BENIGN_L10"]),
        nominal_numerator=np.stack(nominal_num), nominal_denominator=np.stack(nominal_den))
    write_csv(folder / "pair_metrics_by_run.csv", pair_rows)
    write_csv(folder / "nominal_metrics_by_run.csv", nominal_rows)
    write_csv(folder / "transition_counts.csv.gz", transitions)
    return summaries, normal_summary, main_point, main_boot


def analysis_protocol(source, upstream, args):
    return dict(version=VERSION, created_utc=utc(), script_sha256=digest(Path(__file__)), source_folder=str(source),
        source_protocol_identity=upstream["identity"], source_dataset_id=upstream["source_contract"]["dataset_id"],
        source_encoder_run_id=upstream["source_contract"]["step8_run_id"], retrospective=True,
        training=False, inference=False, selection_on_stress=False, source_version=SOURCE_VERSION,
        labels=list(LABELS), fields=list(FIELDS), conditions=list(CONDITIONS), families=list(FAMILIES), runs=list(RUNS),
        pairing_key=["shared_nominal_block_id", "sequence_start_step", "sequence_end_step", "sequence_length"],
        retained_population="Every eligible L10 attack endpoint, including unchanged semantic inputs; no relabeling",
        headline_metric="paired_correct on ALL_ATTACKS/ALL; endpoint-weighted, mean over fixed trained heads",
        outcome_definitions={
            "paired_correct": "I(nominal prediction == NORMAL and attack prediction == target)",
            "target_recall": "I(attack prediction == target), includes attacks with unchanged semantic inputs",
            "target_gain": "I(attack prediction == target) - I(nominal prediction == target); signed [-1,1]",
            "attack_alarm": "I(attack prediction != NORMAL), including wrong attack classes",
            "normal_to_any_attack": "I(nominal prediction == NORMAL and attack prediction != NORMAL)",
            "normal_to_wrong_attack": "I(nominal prediction == NORMAL and attack prediction neither NORMAL nor target)",
            "label_change": "I(nominal prediction != attack prediction); not itself a success measure",
            "nominal_false_alarm": "I(unique nominal endpoint prediction != NORMAL); NORMAL endpoints counted once"},
        denominators="Unconditional on nominal correctness; no conditional-detection denominator. ALL_ATTACKS pools endpoints. "
                     "Classes are balanced in ALL but may not be balanced within CHANGED/UNCHANGED strata.",
        strata="ALL includes every pair. CHANGED and UNCHANGED partition ALL by exact full-L10 22-field differences. "
               "Benign nominal strata use any recorded baseline impairment flag in L10, as audit metadata only.",
        bootstrap=dict(draws=args.bootstrap_draws, seed=args.bootstrap_seed, unit="whole shared nominal episode",
            pairing="same multiplicity vector across all six class replicas, all models, and all four regimes",
            estimator="ratio of endpoint sums after episode resampling; fixed-head mean; not equal-weight episode-rate mean",
            head_seed_resampling=False, generator_seed_resampling=False, encoder_seed_resampling=False,
            intervals="linear percentile95, conditional on fixed source realization/encoder/heads; no equivalence tests",
            empty_strata="NA, not zero. Omit zero-denominator draws and report their count; CI suppressed if <2 contributing clusters",
            multiplicity="32 TST-minus-comparator contrasts for ALL_ATTACKS/ALL paired_correct: four conditions x eight comparators. "
                         "Provide unadjusted95 and Bonferroni95 across those32 only; all analyses retrospective/descriptive"),
        focus_comparator=upstream["frozen_selection"]["selected_pool"], focus_condition="P_moderate",
        focus_scope="Inherited original validation-selected pooling comparator; not newly selected or preregistered",
        verification_scope="Consumed Step49 artifacts and saved producer source only. No re-execution of generator/mapper/inference "
                           "and no verification of omitted upstream caches/checkpoints. Hash identity is internal integrity, not independent authenticity.",
        interpretation="Within-generator matched prediction response, not external validity, raw-vs-semantic superiority, "
                       "absence of all simulator fingerprints, or isolated-factor causal identification.",
        runtime=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, system=platform.platform()))


def package(out, status, extra):
    paths = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "run_manifest.json" and not p.name.endswith(".tmp"))
    manifest = dict(status=status, version=VERSION, completed_utc=utc(), **extra,
        output_sha256={p.relative_to(out).as_posix(): digest(p) for p in paths})
    manifest["identity"] = object_hash(manifest)
    write_json(out / "run_manifest.json", manifest)
    archive = out.with_name(out.name + ("_FAILED_review.zip" if status == "FAILED" else "_review.zip"))
    temp = archive.with_name(archive.name + ".tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file() and not p.name.endswith(".tmp"):
                z.write(p, p.relative_to(out).as_posix())
    atomic_replace(temp, archive)
    return archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--step49-run", type=Path, help="Explicit completed revised Step49 run directory")
    parser.add_argument("--output-root", type=Path, help="Parent for a NEW timestamped run; default project outputs/.../60_paired_attack_response")
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026100701)
    parser.add_argument("--preflight-only", action="store_true", help="Validate hashes/pairs/IDs only; do not compute statistical results")
    args = parser.parse_args(argv)
    out = None; out_created = False; started = time.perf_counter()
    try:
        require(args.bootstrap_draws >= 1000 and args.bootstrap_seed >= 0, "Use >=1000 bootstrap draws and a nonnegative seed")
        root = args.project_root.expanduser().resolve(); source = locate_source(root, args.step49_run)
        parent = args.output_root.expanduser() if args.output_root else root / DEFAULT_OUTPUT
        parent = parent if parent.is_absolute() else root / parent
        out = (parent / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")).resolve()
        require(out != source and source not in out.parents and out not in source.parents,
                "Output must not overlap the source Step49 run")
        out.mkdir(parents=True, exist_ok=False)
        out_created = True
        print("Step60: verify completed Step49 source and exact nominal/attack L10 pairs ...", flush=True)
        reader, manifest, upstream = open_source(source)
        protocol = analysis_protocol(source, upstream, args)
        protocol["identity"] = object_hash(protocol)
        write_json(out / "protocol.json", protocol)
        source_meta = out / "source_metadata"; source_meta.mkdir()
        for name in ("run_manifest.json", "protocol.json", "validation_selection.json"):
            shutil.copyfile(reader.path(name), source_meta / name)
        (out / "source").mkdir(); shutil.copyfile(__file__, out / "source" / Path(__file__).name)
        audits, all_summary, all_nominal = [], [], []
        points, bootstraps = [], []
        weights = common_blocks = reference_signature = None
        for condition in CONDITIONS:
            print("  " + condition + ": verifying hashes, endpoint completeness and paired inputs ...", flush=True)
            data = load_condition(reader, upstream, condition)
            if common_blocks is None:
                common_blocks = data["block_ids"].copy(); reference_signature = data["endpoint_signature"]
                if not args.preflight_only:
                    weights = bootstrap_weights(len(common_blocks), args.bootstrap_draws, args.bootstrap_seed)
                    write_npz(out / "bootstrap_design.npz", block_ids=common_blocks, weights=weights)
            else:
                require(np.array_equal(common_blocks, data["block_ids"]) and reference_signature == data["endpoint_signature"],
                        "Conditions are not aligned to the SAME endpoint/episode population")
            audits.append(audit_pair_predictions(data, condition, out))
            print(f"    {len(data['attack']):,} matched pairs; {len(data['normal']):,} unique NORMAL endpoints; "
                  f"{len(common_blocks)} shared nominal episodes.", flush=True)
            if not args.preflight_only:
                print(f"    Computing fixed-seed means and {args.bootstrap_draws:,} paired episode bootstrap draws ...", flush=True)
                summary, nominal, point, boot = summarize_condition(data, condition, weights, out)
                all_summary.extend(summary); all_nominal.extend(nominal); points.append(point); bootstraps.append(boot)
            del data
        write_csv(out / "pairing_summary.csv", audits)
        warnings = [a["condition"] + ": identical semantic inputs gave different hard labels; inspect diagnostic CSV (numerical/batch effects possible)."
                    for a in audits if a["identical_input_prediction_disagreements"]]
        if not args.preflight_only:
            write_csv(out / "paired_model_summary.csv", all_summary)
            write_csv(out / "nominal_false_alarm_summary.csv", all_nominal)
            point_array = np.stack(points); boot_array = np.stack(bootstraps, axis=1)
            write_npz(out / "paired_success_bootstrap.npz", conditions=np.asarray(CONDITIONS), families=np.asarray(FAMILIES),
                point=point_array, samples=boot_array)
            contrasts = []
            for ci, condition in enumerate(CONDITIONS):
                for fi, comparator in enumerate(FAMILIES[1:], 1):
                    delta = boot_array[:, ci, 0] - boot_array[:, ci, fi]
                    lo, hi, valid = interval(delta); blo, bhi, _ = interval(delta, alpha=.05/32)
                    contrasts.append(dict(condition=condition, comparator=comparator, metric="paired_correct", label="ALL_ATTACKS", stratum="ALL",
                        tst_minus_comparator=float(point_array[ci, 0] - point_array[ci, fi]), ci95_low=lo, ci95_high=hi,
                        bonferroni32_low=blo, bonferroni32_high=bhi, bootstrap_valid=valid,
                        inherited_focus=condition == "P_moderate" and comparator == protocol["focus_comparator"],
                        interpretation="Retrospective conditional interval; no confirmatory/equivalence claim"))
            write_csv(out / "TST_paired_comparisons.csv", contrasts)
            main_rows = [r for r in all_summary if r["label"] == "ALL_ATTACKS" and r["stratum"] == "ALL" and r["metric"] == "paired_correct"]
            write_csv(out / "main_paired_success.csv", main_rows)
        # Re-read only consumed hashes to catch a source being changed during this audit.
        reader.recheck()
        require(digest(Path(__file__)) == protocol["script_sha256"], "Step60 code changed during execution")
        write_json(out / "input_audit.json", dict(source_folder=str(source), source_manifest_identity=manifest["identity"],
            source_protocol_identity=upstream["identity"], consumed_sha256=reader.observed, rechecked_at_end=True,
            verification_scope=protocol["verification_scope"]))
        write_json(out / "interpretation.json", dict(retrospective=True, warnings=warnings,
            main_metric=protocol["headline_metric"], denominators=protocol["denominators"], strata=protocol["strata"],
            intervals=protocol["bootstrap"], limitations=protocol["interpretation"],
            identical_input_check="Pipeline consistency only; deterministic incorrect models can also pass.",
            elapsed_seconds=time.perf_counter()-started))
        status = ("PREFLIGHT_COMPLETE_WITH_WARNINGS" if warnings else "PREFLIGHT_COMPLETE") if args.preflight_only else (
            "PAIRED_RESPONSE_COMPLETE_WITH_WARNINGS" if warnings else "PAIRED_RESPONSE_COMPLETE")
        archive = package(out, status, dict(protocol_identity=protocol["identity"], retrospective=True,
            training=False, inference=False, warnings=warnings, elapsed_seconds=time.perf_counter()-started))
        print("\n" + status, flush=True)
        print("Results: " + str(out), flush=True)
        print("Review ZIP: " + str(archive), flush=True)
        for message in warnings:
            print("REVIEW REQUIRED: " + message, flush=True)
        return 0
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        print("\nSTOPPED: " + message, file=sys.stderr, flush=True)
        if out_created and out is not None and out.is_dir():
            try:
                write_json(out / "failure.json", dict(status="FAILED", error=message, traceback=traceback.format_exc(),
                    elapsed_seconds=time.perf_counter()-started, original_step49_modified=False))
                archive = package(out, "FAILED", dict(error=message))
                print("Diagnostic ZIP: " + str(archive), file=sys.stderr, flush=True)
            except Exception as packaging_error:
                print("Could not package diagnostics: " + str(packaging_error), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
