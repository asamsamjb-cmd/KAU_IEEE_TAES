#!/usr/bin/env python3
"""Step64: CPU-only, fixed-head alarm/delay analysis of the completed Step62 run.

Put this file beside the matching 63.temporal_evidence_analysis.py. No model is
loaded, trained, or evaluated. The validation-only threshold lock precedes any
test-prediction read. See protocol.json in each output for the estimand contract.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import math
import platform
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
import pandas as pd

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

VERSION = "step64_fixed_horizon_alarm_delay_v1"
EXPECTED_READER_API = "step62_followup_reader_v1"
EXPECTED_STEP63_SHA256 = "29a8d528634f30b355584e31d82dce0fc522b227862f0b6051bfeb0d0a4d7681"
LABELS = ("NORMAL", "RF_LINK_JAMMING", "GNSS_SPOOFING",
          "UNAUTHORIZED_TELECOMMAND_INJECTION", "TELECOMMAND_REPLAY",
          "GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE")
CONDITIONS = ("actual_history", "repeat_current")
FPR_TARGETS = (0.01, 0.05)
ANCHORS = ("active", "observable")
HORIZONS = (1, 3, 5, 10)
PRIMARY_HORIZON = 10
ALARM_TYPES = ("any_attack", "correct_type")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def time_seconds(series):
    """Avoid assumptions about pandas' datetime64 storage unit (ns versus us)."""
    parsed = pd.to_datetime(series, utc=True, errors="raise")
    require(not parsed.isna().any(), "Missing timestamp")
    return (parsed - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()


def select_threshold(normal_scores, target):
    """Most permissive observed boundary with strict > and empirical FPR <= target.

    With n scores and k=floor(n*target), score_(n-k) is the boundary, in
    one-based ascending indexing. Ties at this boundary are NOT alarms; therefore
    fewer than k alarms are possible. Zero budget selects the maximum score.
    """
    scores = np.asarray(normal_scores, dtype=np.float64)
    require(scores.ndim == 1 and len(scores) > 0, "Need nonempty validation NORMAL scores")
    require(np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all(),
            "Invalid validation NORMAL scores")
    require(0 <= target < 1, "FPR target must be in [0,1)")
    n = len(scores)
    budget = math.floor(n * float(target) + 1e-12)
    threshold = float(np.partition(scores, n - budget - 1)[n - budget - 1])
    count = int(np.sum(scores > threshold))
    require(count <= budget, "Threshold violated validation false-alarm budget")
    return dict(target_normal_endpoint_fpr=float(target), threshold=threshold,
                validation_normal_endpoints=n, validation_allowed_alarm_count=budget,
                validation_alarm_count=count, validation_achieved_fpr=count / n,
                validation_ties_at_threshold=int(np.sum(scores == threshold)),
                comparison="score > threshold", score="1 - prob_0")


def build_event_catalog(windows, endpoints):
    """First maker-active / first counterfactual-change events, never label onsets.

    The full Step7 timeline preserves onsets inside the unavailable first nine
    rows. Full follow-up is determined from timestamps, not number of endpoints.
    A warm-up interval is known detector unavailability, not a fabricated score.
    """
    required = {"episode_id", "episode_step", "timestamp", "label",
                "audit_attack_active", "audit_observable_effect_present"}
    require(required.issubset(windows), "Step7 audit columns missing")
    endpoint_groups = {str(k): v for k, v in endpoints.groupby("episode_id", sort=False)}
    rows = []
    for episode_id, group in windows.groupby("episode_id", sort=False):
        group = group.sort_values("episode_step", kind="stable")
        require(group.label.nunique() == 1, "Mixed labels within episode")
        label = str(group.label.iloc[0])
        require(label in LABELS, "Unknown episode label")
        if label == "NORMAL":
            continue
        times = time_seconds(group.timestamp)
        steps = group.episode_step.to_numpy(dtype=int)
        require(np.all(np.diff(times) == 1) and np.all(np.diff(steps) == 1),
                "Event timeline must have one-second cadence")
        endpoint_group = endpoint_groups.get(str(episode_id))
        endpoint_times = (time_seconds(endpoint_group.timestamp) if endpoint_group is not None
                          else np.empty(0))
        first_endpoint = float(endpoint_times.min()) if len(endpoint_times) else np.nan
        last_endpoint = float(endpoint_times.max()) if len(endpoint_times) else np.nan
        for anchor, column in (("active", "audit_attack_active"),
                               ("observable", "audit_observable_effect_present")):
            flags = group[column].to_numpy()
            require(np.isin(flags, [0, 1, False, True]).all(), "Nonbinary onset audit flag")
            positive = np.flatnonzero(flags)
            missing = len(positive) == 0
            index = int(positive[0]) if not missing else None
            onset = float(times[index]) if not missing else np.nan
            row = dict(episode_id=str(episode_id), label=label, label_id=LABELS.index(label),
                       anchor=anchor, anchor_missing=missing,
                       onset_step=int(steps[index]) if not missing else np.nan,
                       onset_timestamp=str(group.timestamp.iloc[index]) if not missing else "",
                       onset_epoch_seconds=onset, episode_start_timestamp=str(group.timestamp.iloc[0]),
                       episode_end_timestamp=str(group.timestamp.iloc[-1]),
                       episode_end_epoch_seconds=float(times[-1]),
                       episode_windows=len(group), eligible_endpoints=len(endpoint_times),
                       no_eligible_endpoints=len(endpoint_times) == 0,
                       first_eligible_endpoint_epoch_seconds=first_endpoint,
                       last_eligible_endpoint_epoch_seconds=last_endpoint,
                       onset_during_first_nine_windows=bool(not missing and steps[index] < 9),
                       warmup_unavailable_seconds=max(0., first_endpoint - onset)
                       if not missing and len(endpoint_times) else np.nan,
                       available_followup_seconds=last_endpoint - onset
                       if not missing and len(endpoint_times) else np.nan,
                       recorded_followup_seconds=float(times[-1]) - onset if not missing else np.nan,
                       positive_audit_windows=len(positive))
            for horizon in HORIZONS:
                # Right follow-up must reach the deadline. Warm-up unavailability
                # counts as detector unavailability rather than shifting onset.
                row[f"fully_followed_{horizon}s"] = bool(
                    not missing and len(endpoint_times) and last_endpoint >= onset + horizon)
            rows.append(row)
    columns = ["episode_id", "label", "label_id", "anchor", "anchor_missing",
               "onset_epoch_seconds", "fully_followed_10s"]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)


def evaluate_alarm_events(catalog, endpoints, alarms, predicted_attack_types):
    """One row per attack episode/anchor for one seed, condition, and threshold."""
    alarms = np.asarray(alarms, dtype=bool)
    types = np.asarray(predicted_attack_types, dtype=int)
    require(alarms.shape == types.shape == (len(endpoints),), "Alarm endpoint shape mismatch")
    times = time_seconds(endpoints.timestamp)
    groups = endpoints.groupby("episode_id", sort=False).indices
    rows = []
    for event in catalog.to_dict("records"):
        row = dict(event)
        ids = np.asarray(groups.get(event["episode_id"], []), dtype=int)
        onset = event["onset_epoch_seconds"]
        present = not event["anchor_missing"]
        post = ids[times[ids] >= onset] if present else np.empty(0, dtype=int)
        alarm_ids = post[alarms[post]]
        correct_ids = alarm_ids[types[alarm_ids] == event["label_id"]]
        any_delay = float(times[alarm_ids[0]] - onset) if len(alarm_ids) else np.nan
        correct_delay = float(times[correct_ids[0]] - onset) if len(correct_ids) else np.nan
        row.update(first_any_alarm_delay_seconds=any_delay,
                   first_correct_type_alarm_delay_seconds=correct_delay,
                   first_any_alarm_correct_type=bool(types[alarm_ids[0]] == event["label_id"])
                   if len(alarm_ids) else np.nan,
                   pre_onset_alarm_count=int(np.sum(alarms[ids[times[ids] < onset]]))
                   if present else np.nan)
        for alarm_type, delay in zip(ALARM_TYPES, (any_delay, correct_delay)):
            for horizon in HORIZONS:
                full = bool(event[f"fully_followed_{horizon}s"])
                detected = bool(np.isfinite(delay) and delay <= horizon)
                row[f"{alarm_type}_detected_{horizon}s"] = detected
                row[f"{alarm_type}_capped_delay_{horizon}s"] = (
                    float(delay) if detected else float(horizon)) if full else np.nan
                row[f"{alarm_type}_lower_bound_{horizon}s"] = float(detected) if present else np.nan
                row[f"{alarm_type}_upper_bound_{horizon}s"] = (
                    float(detected or not full) if present else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def normal_episode_counts(endpoints, alarms):
    """Alarm runs restart at each NORMAL episode; do not bridge a gap/boundary."""
    alarms = np.asarray(alarms, dtype=bool)
    require(alarms.shape == (len(endpoints),), "Normal alarm shape mismatch")
    times = time_seconds(endpoints.timestamp)
    rows = []
    for episode_id, ids in endpoints.groupby("episode_id", sort=False).indices.items():
        ids = np.asarray(ids, dtype=int)
        labels = endpoints.label.iloc[ids]
        require(labels.nunique() == 1, "Mixed labels within NORMAL episode")
        if labels.iloc[0] != "NORMAL":
            continue
        alarm = alarms[ids]
        previous = np.r_[False, alarm[:-1] & (np.diff(times[ids]) == 1)]
        rows.append(dict(episode_id=str(episode_id), eligible_normal_endpoints=len(ids),
                         alarm_count=int(alarm.sum()), episode_any_alarm=int(alarm.any()),
                         alarm_run_starts=int(np.sum(alarm & ~previous))))
    return pd.DataFrame(rows, columns=["episode_id", "eligible_normal_endpoints", "alarm_count",
                                       "episode_any_alarm", "alarm_run_starts"])


def divide(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(np.asarray(numerator, dtype=float),
                                                  np.asarray(denominator, dtype=float))
    return np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan),
                     where=denominator > 0)


def paired_episode_bootstrap(labels, numerators, denominators, draws, seed,
                             class_ids=(1, 2, 3, 4, 5), chunk_size=256):
    """Class-stratified paired whole-episode ratio bootstrap, with fixed heads.

    numerators: [metric, episode, condition], already averaged over fixed heads
    when the denominator is shared by heads (detection/capped delay/normal FPR).
    denominators: [metric, episode]. Each class is resampled at its original
    episode count, retaining missing/censored episodes and all seed pairs.
    The estimator is the equally weighted mean of the within-class ratios.
    This is NOT a bootstrap of endpoint rows, seeds, or an ensemble prediction.
    """
    labels = np.asarray(labels)
    num = np.asarray(numerators, dtype=np.float64)
    den = np.asarray(denominators, dtype=np.float64)
    require(num.ndim == 3 and num.shape[-1] == 2, "Expected [metric,episode,2] numerators")
    require(den.shape == num.shape[:2] and len(labels) == num.shape[1], "Bootstrap shape mismatch")
    require(np.isfinite(num).all() and np.isfinite(den).all() and (den >= 0).all(),
            "Bootstrap components must be finite; use zero denominator for missing events")
    require(draws > 0 and len(class_ids) > 0, "Invalid bootstrap configuration")
    metric_count = num.shape[0]
    groups = [np.flatnonzero(labels == label) for label in class_ids]
    point_by_class = [divide(num[:, ids].sum(axis=1), den[:, ids].sum(axis=1)[:, None])
                      for ids in groups]
    point_conditions = np.mean(point_by_class, axis=0)
    point = np.column_stack([point_conditions, point_conditions[:, 0] - point_conditions[:, 1]])
    samples = np.full((draws, metric_count, 3), np.nan)
    if any(len(ids) == 0 for ids in groups):
        return point, samples
    rng = np.random.default_rng(seed)
    for start in range(0, draws, chunk_size):
        batch = min(chunk_size, draws - start)
        class_results = []
        for ids in groups:
            n = len(ids)
            # One episode-count draw is reused for all metrics/conditions/heads.
            counts = rng.multinomial(n, np.full(n, 1. / n), size=batch)
            class_num = num[:, ids, :].transpose(1, 0, 2).reshape(n, metric_count * 2)
            summed_num = (counts @ class_num).reshape(batch, metric_count, 2)
            summed_den = counts @ den[:, ids].T
            class_results.append(divide(summed_num, summed_den[:, :, None]))
        conditions = np.mean(class_results, axis=0)
        samples[start:start + batch, :, :2] = conditions
        samples[start:start + batch, :, 2] = conditions[:, :, 0] - conditions[:, :, 1]
    return point, samples


def summarize_bootstrap(specifications, points, samples):
    rows = []
    for index, spec in enumerate(specifications):
        for c, condition in enumerate((*CONDITIONS, "actual_history_minus_repeat_current")):
            values = samples[:, index, c]
            valid = values[np.isfinite(values)]
            # Do not silently discard undefined resamples. Suppress the interval
            # unless every draw defines all required class-specific denominators.
            complete = len(valid) == len(values)
            row = dict(spec, condition=condition, estimate=float(points[index, c]),
                       bootstrap_draws=len(values), defined_bootstrap_draws=len(valid),
                       interval_status="defined" if complete else "undefined_denominator_in_resamples")
            for name, probs in (("pointwise_95", (.025, .975)),
                                ("bonferroni_two_targets_97_5", (.0125, .9875))):
                # The two-comparison family is ONLY active/any-attack/10-s
                # macro detection improvement at the 1% and 5% targets.
                allowed = complete and (name == "pointwise_95" or (spec.get("primary", False) and c == 2))
                bounds = np.quantile(valid, probs) if allowed else (np.nan, np.nan)
                row[f"{name}_lower"] = float(bounds[0])
                row[f"{name}_upper"] = float(bounds[1])
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_events_by_head(events):
    """Per-head class/macro summaries, with all exclusions and descriptive delays."""
    rows = []
    keys = ["seed", "condition", "target_normal_endpoint_fpr", "anchor"]
    for key, group in events.groupby(keys, sort=False):
        meta = dict(zip(keys, key))
        for horizon in HORIZONS:
            class_rows = []
            for label in LABELS[1:]:
                part = group.loc[group.label == label]
                present = ~part.anchor_missing
                full = part[f"fully_followed_{horizon}s"].astype(bool)
                for alarm_type in ALARM_TYPES:
                    detected = part[f"{alarm_type}_detected_{horizon}s"].astype(bool)
                    delay_column = ("first_any_alarm_delay_seconds" if alarm_type == "any_attack"
                                    else "first_correct_type_alarm_delay_seconds")
                    row = dict(meta, label=label, horizon_seconds=horizon, alarm_type=alarm_type,
                        total_attack_episodes=len(part), anchor_missing=int((~present).sum()),
                        anchor_present=int(present.sum()), fully_followed_events=int(full.sum()),
                        right_censored_events=int((present & ~full).sum()),
                        warmup_onsets=int((present & part.onset_during_first_nine_windows).sum()),
                        no_eligible_endpoint_events=int(part.no_eligible_endpoints.sum()),
                        fixed_horizon_detections=int((full & detected).sum()),
                        fixed_horizon_misses=int((full & ~detected).sum()),
                        censored_observed_detections=int((present & ~full & detected).sum()),
                        detection_probability=float(divide((full & detected).sum(), full.sum())),
                        capped_delay_seconds=float(part.loc[full, f"{alarm_type}_capped_delay_{horizon}s"].mean()),
                        conditional_mean_delay_fully_followed_detected_seconds=float(part.loc[full & detected, delay_column].mean()),
                        conditional_mean_delay_all_observed_detected_seconds=float(part.loc[present & detected, delay_column].mean()),
                        all_anchor_present_detection_lower_bound=float(part.loc[present, f"{alarm_type}_lower_bound_{horizon}s"].mean()),
                        all_anchor_present_detection_upper_bound=float(part.loc[present, f"{alarm_type}_upper_bound_{horizon}s"].mean()))
                    rows.append(row)
                    class_rows.append(row)
            for alarm_type in ALARM_TYPES:
                chosen = [r for r in class_rows if r["alarm_type"] == alarm_type]
                macro = dict(meta, label="MACRO_ATTACK", horizon_seconds=horizon, alarm_type=alarm_type)
                for field in chosen[0]:
                    if field in macro:
                        continue
                    values = [r[field] for r in chosen]
                    if field in ("total_attack_episodes", "anchor_missing", "anchor_present", "fully_followed_events",
                                 "right_censored_events", "warmup_onsets", "no_eligible_endpoint_events",
                                 "fixed_horizon_detections", "fixed_horizon_misses", "censored_observed_detections"):
                        macro[field] = int(sum(values))
                    else:
                        # np.mean propagates undefined classes. No implicit
                        # change from five-class macro to a smaller class set.
                        macro[field] = float(np.mean(values))
                rows.append(macro)
    return pd.DataFrame(rows)


def mean_head_summary(per_head):
    keys = ["condition", "target_normal_endpoint_fpr", "anchor", "label", "horizon_seconds", "alarm_type"]
    numeric = [column for column in per_head if column not in keys + ["seed"]]
    rows = []
    for key, group in per_head.groupby(keys, sort=False):
        row = dict(zip(keys, key), fixed_head_count=len(group))
        for column in numeric:
            row[column] = float(np.mean(group[column].to_numpy(dtype=float)))
        rows.append(row)
    return pd.DataFrame(rows)


def event_bootstrap_inputs(events, seeds):
    """Average paired head numerators only after applying each head's alarm rule."""
    episodes = events.loc[:, ["episode_id", "label_id"]].drop_duplicates().sort_values("episode_id")
    ids = episodes.episode_id.tolist()
    specs, numerators, denominators = [], [], []
    for target in FPR_TARGETS:
        for anchor in ANCHORS:
            group = events.loc[(events.target_normal_endpoint_fpr == target) & (events.anchor == anchor)]
            for alarm_type in ALARM_TYPES:
                for metric in ("detection_probability", "capped_delay_seconds"):
                    condition_num = []
                    common_den = None
                    for condition in CONDITIONS:
                        per_head = []
                        for seed in seeds:
                            part = group.loc[(group.condition == condition) & (group.seed == seed)].set_index("episode_id").reindex(ids)
                            require(not part.label_id.isna().any(), "Missing seed/condition event pairing")
                            den = part.fully_followed_10s.to_numpy(dtype=float)
                            if common_den is None:
                                common_den = den
                            require(np.array_equal(common_den, den), "Condition-dependent event denominator")
                            column = (f"{alarm_type}_detected_10s" if metric == "detection_probability"
                                      else f"{alarm_type}_capped_delay_10s")
                            values = part[column].to_numpy(dtype=float)
                            per_head.append(np.where(den > 0, values, 0.))
                        condition_num.append(np.mean(per_head, axis=0))
                    numerators.append(np.stack(condition_num, axis=-1))
                    denominators.append(common_den)
                    specs.append(dict(target_normal_endpoint_fpr=target, anchor=anchor,
                                      alarm_type=alarm_type, horizon_seconds=10, metric=metric,
                                      aggregation="five_attack_class_macro_mean_of_fixed_head_metrics",
                                      primary=anchor == "active" and alarm_type == "any_attack" and metric == "detection_probability"))
    return specs, episodes.label_id.to_numpy(), np.asarray(numerators), np.asarray(denominators)


def normal_summaries(normal_counts, seeds):
    rows, specs, numerators, denominators = [], [], [], []
    for (seed, condition, target), group in normal_counts.groupby(
            ["seed", "condition", "target_normal_endpoint_fpr"], sort=False):
        n = int(group.eligible_normal_endpoints.sum())
        rows.append(dict(seed=seed, condition=condition, target_normal_endpoint_fpr=target,
                         normal_episodes=len(group), eligible_normal_endpoints=n,
                         normal_endpoint_fpr=float(divide(group.alarm_count.sum(), n)),
                         normal_episode_any_alarm_probability=float(group.episode_any_alarm.mean()),
                         alarm_run_starts_per_1000_eligible_normal_endpoints=float(divide(1000 * group.alarm_run_starts.sum(), n))))
    episode_ids = sorted(normal_counts.episode_id.unique())
    for target in FPR_TARGETS:
        for metric, column, scale, endpoint_den in (
                ("normal_endpoint_fpr", "alarm_count", 1, True),
                ("normal_episode_any_alarm_probability", "episode_any_alarm", 1, False),
                ("alarm_run_starts_per_1000_eligible_normal_endpoints", "alarm_run_starts", 1000, True)):
            nums, common_den = [], None
            for condition in CONDITIONS:
                per_head = []
                for seed in seeds:
                    part = normal_counts.loc[(normal_counts.target_normal_endpoint_fpr == target) &
                        (normal_counts.condition == condition) & (normal_counts.seed == seed)].set_index("episode_id").reindex(episode_ids)
                    require(not part.eligible_normal_endpoints.isna().any(), "NORMAL episode pairing mismatch")
                    den = part.eligible_normal_endpoints.to_numpy(dtype=float) if endpoint_den else np.ones(len(part))
                    if common_den is None:
                        common_den = den
                    require(np.array_equal(den, common_den), "NORMAL denominator pairing mismatch")
                    per_head.append(scale * part[column].to_numpy(dtype=float))
                nums.append(np.mean(per_head, axis=0))
            numerators.append(np.stack(nums, axis=-1))
            denominators.append(common_den)
            specs.append(dict(target_normal_endpoint_fpr=target, metric=metric,
                              aggregation="mean_of_fixed_head_metrics", primary=False))
    return pd.DataFrame(rows), specs, np.zeros(len(episode_ids), dtype=int), np.asarray(numerators), np.asarray(denominators)


def load_reader():
    path = Path(__file__).resolve().with_name("63.temporal_evidence_analysis.py")
    require(path.is_file(), "Place the matching 63.temporal_evidence_analysis.py beside Step64")
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    require(actual_hash == EXPECTED_STEP63_SHA256,
            "Step63 reader SHA256 mismatch; use the exact paired delivery, never bypass the pin")
    spec = importlib.util.spec_from_file_location("step63_followup_reader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.READER_API == EXPECTED_READER_API, "Step63 reader API mismatch")
    require(tuple(module.LABELS) == LABELS and tuple(module.CONDITIONS) == CONDITIONS, "Reader class/condition mismatch")
    return module


def protocol_document(reader, ctx, args):
    return dict(version=VERSION, created_at_utc=reader.utc_now(), training=False, new_inference=False,
        script_sha256=reader.digest(__file__), step63_reader_sha256=reader.digest(reader.__file__),
        reader_api=EXPECTED_READER_API, source_step62_run=str(ctx["run"]),
        source_step62_protocol_sha256=reader.digest(ctx["run"] / "protocol.json"),
        source_step62_identity=ctx["protocol"].get("identity"), source_audit=ctx["source_audit"],
        runtime=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                     platform=platform.platform(), compute="CPU NumPy/pandas; no torch/model loading",
                     bootstrap_blas_thread_limit=4 if threadpool_limits is not None else None),
        fixed_head_seeds=ctx["seeds"], conditions=CONDITIONS,
        binary_alarm=dict(score="1 - prob_0", rule="strict score > threshold", smoothing=False,
                          persistence_requirement=False, refractory_period=False),
        calibration=dict(split="val", eligible_population="Step62 common L10 endpoints labeled NORMAL",
            target_endpoint_fpr=FPR_TARGETS, by="separately for every matched seed and condition",
            rule="threshold = ascending score[n-floor(n*target)-1]; ties at threshold do not alarm",
            zero_budget="threshold equals maximum validation NORMAL score; zero validation alarms",
            selection_uses_test=False, lock="thresholds.json and thresholds.csv saved before test prediction loading",
            validation_reuse="The same validation split selected the original Step62 checkpoints; this is not a fresh held-out calibration cohort",
            interpretation="empirical validation budgets; test FPR is measured, not guaranteed"),
        event=dict(unit="first event per attack episode; repeated bursts not additional events",
            anchors=dict(active="first audit_attack_active=1 (maker-recorded intervention, NOT physical onset)",
                         observable="first audit_observable_effect_present=1 (counterfactual observable change)"),
            labels="Episode attack labels identify attack type only; never infer onset from a label",
            persistent_effects="Observable/persistent evidence may continue after the active flag stops; later eligible alarms remain eligible",
            alarm_type="Any-attack first post-onset threshold exceedance is primary; first threshold exceedance with argmax(prob_1..prob_5) equal to episode class is separate correct-type analysis",
            type_ties="numpy argmax chooses the first attack class in LABELS order",
            pre_onset="Alarms strictly before the anchor are counted separately and never detections",
            horizons_seconds=HORIZONS, primary_horizon_seconds=10, interval="[onset, onset+horizon], both endpoints inclusive",
            delay="Recorded window timestamp difference in seconds, never endpoint index distance or measured inference wall-clock latency",
            warmup="First nine episode windows have no L10 score. Preserve their onsets; earliest available alarm includes real elapsed warm-up delay. No imputed scores or shifted onsets.",
            fully_followed="Anchor present and last eligible endpoint timestamp >= onset+horizon. This is right follow-up; known initial detector warm-up remains part of operational behavior.",
            censoring="Anchor-present events ending before deadline are excluded from fixed-horizon denominator regardless of early detection; early detections retained in descriptive/all-event bounds",
            missing_anchor="Reported separately and excluded from detection estimands; never silently treated as a miss",
            capped_delay="Fully followed events: min(first detection delay,horizon), with misses assigned horizon; report alongside detection since detection exactly at horizon also has capped delay=horizon",
            conditional_mean_delay="Descriptive only; report conditional on detected events, separately fully followed and all observed detections",
            bounds="Among all anchor-present events: lower=observed detected by deadline; upper=observed detected OR right-censored. These are identification bounds, not confidence intervals."),
        primary=dict(metric="five-class macro any-attack detection probability within 10 s from active anchor",
                     contrast="actual_history minus repeat_current", two_targets=FPR_TARGETS,
                     estimand="mean per-head metric across the fixed ten paired head seeds; NOT ensemble probabilities"),
        statistics=dict(draws=args.bootstrap_draws, seed=args.bootstrap_seed,
            unit="class-stratified paired whole episode; same episode multiplicities across all metrics/conditions/heads",
            head_seed_resampling=False, class_weights="equal 1/5 for five attack classes; NORMAL its own class",
            missing_and_censored="Retained as episodes during resampling with zero fixed-horizon numerator/denominator",
            pointwise_interval="percentile 95%: quantiles 0.025, 0.975",
            primary_two_target_family="Bonferroni: individual percentile 97.5% intervals, quantiles 0.0125, 0.9875, for two active/any-attack/10-s detection contrasts only; approximate nominal family coverage",
            exploratory="Other anchors, correct-type results, capped delay, NORMAL metrics and shorter horizons are secondary/descriptive; no additional multiplicity control",
            thresholds="Held fixed in bootstrap; uncertainty conditional on realized validation calibration",
            undefined="Any absent class denominator makes the macro undefined; intervals suppressed if any bootstrap draw has undefined denominator"),
        false_alarms=dict(population="NORMAL-labeled common L10 endpoints only; not quiet attack-labeled windows",
            metrics=["NORMAL endpoint FPR", "NORMAL episode any-alarm probability",
                     "alarm run starts per 1000 eligible NORMAL endpoints"],
            run_definition="Consecutive one-second eligible endpoint alarms within the same NORMAL episode; reset at gap or episode boundary",
            episode_population="NORMAL episodes having at least one eligible L10 endpoint; no-endpoint episodes reported separately"),
        scope="Post-hoc follow-up after inspection of the original test results; not independent confirmatory evidence. Alarm scores come from Step62 episode/scenario-label classifiers, not newly trained instantaneous-active detectors. Conditional fixed-encoder/fixed-trained-head within-generator information comparison. No physical-onset, causal intervention, external-generalization, temporal-order, or TST-superiority claim.")


def main(argv=None):
    try:
        reader = load_reader()
    except Exception as exc:
        print(f"Step64 STOPPED before prediction access: {exc}", file=sys.stderr, flush=True)
        return 1
    parser = argparse.ArgumentParser(description=__doc__)
    reader.add_common_arguments(parser)
    parser.set_defaults(project_root=Path(__file__).resolve().parent, bootstrap_seed=2026100764)
    args = parser.parse_args(argv)
    out = reader.create_output(args.project_root, 64, "alarm_delay_evaluation")
    try:
        require(args.bootstrap_draws >= 2000, "Use at least 2000 bootstrap draws (default 20000)")
        ctx = reader.load_context(args)
        require(len(ctx["seeds"]) == 10, "Step64 requires exactly the ten fixed Step62 heads")
        source = out / "source"
        source.mkdir()
        shutil.copy2(__file__, source / Path(__file__).name)
        shutil.copy2(reader.__file__, source / Path(reader.__file__).name)
        reader.write_json(out / "protocol.json", protocol_document(reader, ctx, args))
        reader.write_json(out / "input_audit.json", ctx["source_audit"])
        catalogs = {split: build_event_catalog(ctx["windows"][split], ctx["endpoints"][split])
                    for split in ("val", "test")}
        population_rows = []
        for split, catalog in catalogs.items():
            reader.write_csv(out / f"event_catalog_{split}.csv", catalog)
            windows, endpoints = ctx["windows"][split], ctx["endpoints"][split]
            for label in LABELS:
                all_ids = set(windows.loc[windows.label == label, "episode_id"].astype(str))
                eligible_ids = set(endpoints.loc[endpoints.label == label, "episode_id"].astype(str))
                population_rows.append(dict(split=split, label=label, full_timeline_episodes=len(all_ids),
                    eligible_endpoint_episodes=len(eligible_ids), no_eligible_endpoint_episodes=len(all_ids - eligible_ids),
                    unavailable_first_nine_windows=int((windows.loc[windows.label == label, "episode_step"] < 9).sum()),
                    eligible_endpoints=int((endpoints.label == label).sum())))
        reader.write_csv(out / "population_audit.csv", pd.DataFrame(population_rows))
        if args.preflight_only:
            reader.write_json(out / "summary.json", dict(status="PREFLIGHT_COMPLETE", predictions_loaded=False,
                thresholds_selected=False, note="Source hashes, alignment, audit flags, cadence and populations verified; no analysis scores read."))
            archive = reader.finish_output(out, "PREFLIGHT_COMPLETE")
            print(f"Preflight complete: {archive}", flush=True)
            return 0
        print("Source preflight passed. Calibrating validation NORMAL thresholds.", flush=True)
        val_predictions = reader.load_predictions(ctx, "val")
        val_normal = ctx["endpoints"]["val"].label.to_numpy() == "NORMAL"
        thresholds, threshold_rows = {}, []
        for s, seed in enumerate(ctx["seeds"]):
            for c, condition in enumerate(CONDITIONS):
                for target in FPR_TARGETS:
                    record = dict(seed=seed, condition=condition,
                                  **select_threshold(1. - val_predictions[s, c, val_normal, 0], target))
                    thresholds[(seed, condition, target)] = record["threshold"]
                    threshold_rows.append(record)
        threshold_lock = dict(created_at_utc=reader.utc_now(), selection_split="val", records=threshold_rows,
                              test_predictions_loaded=False, score="1 - prob_0", rule="score > threshold")
        reader.write_json(out / "thresholds.json", threshold_lock)
        reader.write_csv(out / "thresholds.csv", pd.DataFrame(threshold_rows))
        # Test prediction access happens only after both threshold artifacts have
        # been closed and the immutable content hash has been recorded.
        lock_hash = reader.digest(out / "thresholds.json")
        reader.write_json(out / "threshold_lock.json", dict(locked_at_utc=reader.utc_now(),
            thresholds_json_sha256=lock_hash, thresholds_csv_sha256=reader.digest(out / "thresholds.csv"),
            test_predictions_loaded=False))
        del val_predictions
        print("Validation thresholds saved and locked. Reading fixed test predictions.", flush=True)
        predictions = reader.load_predictions(ctx, "test")
        require(reader.digest(out / "thresholds.json") == lock_hash, "Threshold lock changed")
        events_list, normal_list = [], []
        for s, seed in enumerate(ctx["seeds"]):
            for c, condition in enumerate(CONDITIONS):
                scores = 1. - predictions[s, c, :, 0]
                attack_types = np.argmax(predictions[s, c, :, 1:], axis=-1) + 1
                for target in FPR_TARGETS:
                    alarms = scores > thresholds[(seed, condition, target)]
                    events = evaluate_alarm_events(catalogs["test"], ctx["endpoints"]["test"], alarms, attack_types)
                    normal = normal_episode_counts(ctx["endpoints"]["test"], alarms)
                    for frame in (events, normal):
                        frame["seed"] = seed
                        frame["condition"] = condition
                        frame["target_normal_endpoint_fpr"] = target
                    events_list.append(events)
                    normal_list.append(normal)
            print(f"Evaluated head seed {seed}.", flush=True)
        events = pd.concat(events_list, ignore_index=True)
        normal_counts = pd.concat(normal_list, ignore_index=True)
        reader.write_csv(out / "event_results_by_seed.csv", events)
        reader.write_csv(out / "normal_episode_results_by_seed.csv", normal_counts)
        per_head = summarize_events_by_head(events)
        reader.write_csv(out / "event_metrics_by_seed.csv", per_head)
        reader.write_csv(out / "event_metrics_mean_heads.csv", mean_head_summary(per_head))
        specs, labels, num, den = event_bootstrap_inputs(events, ctx["seeds"])
        print(f"Bootstrapping {args.bootstrap_draws} paired whole-episode draws (fixed heads).", flush=True)
        with threadpool_limits(limits=4) if threadpool_limits is not None else nullcontext():
            points, boot = paired_episode_bootstrap(labels, num, den, args.bootstrap_draws, args.bootstrap_seed)
        event_intervals = summarize_bootstrap(specs, points, boot)
        reader.write_csv(out / "event_intervals_10s.csv", event_intervals)
        primary = event_intervals.loc[event_intervals.primary].copy()
        reader.write_csv(out / "primary_detection_10s.csv", primary)
        primary_draws = {"draw": np.arange(1, args.bootstrap_draws + 1)}
        for index, spec in enumerate(specs):
            if spec["primary"]:
                for c, name in enumerate((*CONDITIONS, "delta")):
                    primary_draws[f"target_{int(100 * spec['target_normal_endpoint_fpr'])}pct_{name}"] = boot[:, index, c]
        reader.write_csv(out / "primary_episode_bootstrap.csv", pd.DataFrame(primary_draws))
        normal_head, normal_specs, normal_labels, normal_num, normal_den = normal_summaries(normal_counts, ctx["seeds"])
        with threadpool_limits(limits=4) if threadpool_limits is not None else nullcontext():
            normal_points, normal_boot = paired_episode_bootstrap(normal_labels, normal_num, normal_den,
                args.bootstrap_draws, args.bootstrap_seed + 1, class_ids=(0,))
        reader.write_csv(out / "normal_metrics_by_seed.csv", normal_head)
        normal_intervals = summarize_bootstrap(normal_specs, normal_points, normal_boot)
        reader.write_csv(out / "normal_metric_intervals.csv", normal_intervals)
        require(reader.digest(out / "thresholds.json") == lock_hash, "Threshold lock changed during analysis")
        summary = dict(status="COMPLETE", source_step62_identity=ctx["protocol"].get("identity"),
            threshold_lock_sha256=lock_hash, fixed_head_count=len(ctx["seeds"]),
            primary_comparisons=primary.loc[primary.condition == "actual_history_minus_repeat_current"].to_dict("records"),
            normal_metrics=normal_intervals.to_dict("records"),
            interpretation="Positive primary delta favors actual_history at separately validation-calibrated endpoint FPR targets. Test NORMAL FPR may differ between conditions; inspect normal_metric_intervals.csv.",
            estimand="Mean fixed-head class-macro event metric; no ensemble and no head-seed resampling",
            limitations=["Maker active flags are intervention anchors, not physical onset ground truth.",
                "This post-hoc follow-up is not independent confirmatory evidence; the original test results were already inspected.",
                "Step62 learned episode/scenario labels, including pre-active/recovery windows; these alarm scores are not a newly trained instantaneous-active detector.",
                "The validation split also selected the original checkpoints; it is not a fresh held-out calibration cohort.",
                "Counterfactual-observable onset is a separate sensitivity anchor, with missing anchors reported.",
                "Fixed-horizon denominators exclude right-censored events even if early detection is observed; bounds retain them.",
                "Known initial nine-window L10 warm-up remains part of operational delay.",
                "Intervals condition on fixed encoder, heads and validation-calibrated thresholds.",
                "Only the two primary FPR-target contrasts have Bonferroni intervals; other analyses are exploratory.",
                "Results do not establish causal mechanism, physical onset timing, external validity or architecture superiority."])
        reader.write_json(out / "summary.json", summary)
        archive = reader.finish_output(out, "COMPLETE")
        print(f"Step64 complete: {archive}", flush=True)
        return 0
    except Exception as exc:
        reader.write_json(out / "failure.json", dict(error=str(exc), traceback=traceback.format_exc()))
        archive = reader.finish_output(out, "FAILED", dict(error=str(exc)))
        print(f"Step64 FAILED: {exc}\nReview archive: {archive}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
