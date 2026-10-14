# Revision: common nominal background + shared observation mapping.
# Research objective: reviewer revision on synthetic-rule dependence and temporal evidence.
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import pandas as pd
import shared_nominal as nominal
from shared_nominal import METADATA_FIELDS, OBSERVABLE_FIELDS, normalize_raw_to_observable
DEFAULT_NUM_ROWS = 50000
DEFAULT_WINDOW_SECONDS = 1
DEFAULT_OUTPUT_PATH = "dataset/normal_observable_dataset.csv"
DEFAULT_RAW_AUDIT_OUTPUT_PATH = "dataset/normal_observable_dataset_with_raw.csv"
DEFAULT_SAMPLE_TIMELINE_OUTPUT_PATH = "dataset/normal_sample_episode_timeline.csv"
DEFAULT_RANDOM_SEED = 42
LABEL_NORMAL = "NORMAL"

MAKER_NUMBER = 1
CLASS_LABEL = LABEL_NORMAL
CODE_VERSION = nominal.CODE_VERSION
TIMELINE_BUILDER = 'build_normal_episode_timeline'
RAW_GENERATOR = 'generate_normal_raw_window'
OBSERVABLE_DOMAINS = nominal.OBSERVABLE_DOMAINS

def build_normal_dataset(num_rows: int = DEFAULT_NUM_ROWS, window_seconds: int = DEFAULT_WINDOW_SECONDS,
    start_time: datetime | None = None, seed: int = DEFAULT_RANDOM_SEED, return_raw: bool = False,
    *, background_seed: int | None = None, attacks_enabled: bool = True) -> pd.DataFrame:
    """Build from the common nominal substrate; explicit background_seed is for paired audits.

    Default background streams are disjoint across makers, including equal seed values.
    Do not pool replicated paired-audit backgrounds into independent train/test splits.
    """
    return nominal.build_dataset(globals(), num_rows, window_seconds, start_time, seed,
                                 return_raw, background_seed, attacks_enabled)

def validate_dataset(df, expected_rows=None, expected_window_seconds=1):
    nominal.validate_learning_dataset(df, CLASS_LABEL, expected_rows, expected_window_seconds)

def validate_learning_ready_schema(df_learning, df_full=None, expected_rows=None, expected_window_seconds=1):
    validate_dataset(df_learning, expected_rows, expected_window_seconds)
    if df_full is not None and not df_learning.reset_index(drop=True).equals(
            df_full[METADATA_FIELDS + OBSERVABLE_FIELDS].reset_index(drop=True)):
        raise ValueError("Learning and audit rows differ.")

def validate_raw_audit_dataset(df, expected_rows=None, expected_window_seconds=1):
    nominal.validate_raw_dataset(df, globals(), expected_rows, expected_window_seconds)

if __name__ == "__main__":
    nominal.run_cli(globals(), 'build_normal_dataset')
