"""Load the competition CSVs and the Kaggle notebook outputs, and line them up.

The Kaggle notebook (kaggle_transcribe.ipynb) writes every output in the same
row order as outputs/transcripts.csv, keyed by (split, file_id). Everything
here checks that order instead of trusting it.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


def load_transcripts(output_dir=OUTPUT_DIR):
    """Transcripts plus audio timing stats, one row per clip (train rows first, then test)."""
    df = pd.read_csv(output_dir / "transcripts.csv", dtype={"file_id": str})
    df["transcript"] = df["transcript"].fillna("")  # silent clips give an empty transcript
    return df


def check_alignment(df, transcripts, source):
    """Fail loudly if a saved output is not in the same row order as the transcripts."""
    same = np.array_equal(np.asarray(df["file_id"], dtype=str), transcripts["file_id"].to_numpy(dtype=str)) and \
        np.array_equal(np.asarray(df["split"], dtype=str), transcripts["split"].to_numpy(dtype=str))
    if not same:
        raise ValueError(f"{source} rows do not line up with transcripts.csv")


def load_embedding(name, transcripts, output_dir=OUTPUT_DIR):
    """Load one embedding matrix saved by the Kaggle notebook, e.g. name='text_mpnet'."""
    saved = np.load(output_dir / f"emb_{name}.npz")
    check_alignment(saved, transcripts, f"emb_{name}.npz")
    return saved["emb"].astype(np.float32)


def load_row_features(filename, transcripts, output_dir=OUTPUT_DIR):
    """Load a per-clip feature CSV (e.g. cola_features.csv) and drop its key columns."""
    df = pd.read_csv(output_dir / filename, dtype={"file_id": str})
    check_alignment(df, transcripts, filename)
    return df.drop(columns=["file_id", "split"])


def detect_columns(df, file_ids):
    """Find the file-name column (its values match the transcript ids) and the score column.

    Column names are not hard-coded because they come from the competition files.
    The score column is the only numeric column besides the file column.
    """
    file_ids = set(file_ids)
    match_rate = {c: df[c].astype(str).isin(file_ids).mean() for c in df.columns}
    file_col = max(match_rate, key=match_rate.get)
    if match_rate[file_col] < 0.99:
        raise ValueError(f"no column of {list(df.columns)} matches the transcript file ids")
    numeric = [c for c in df.columns if c != file_col and pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric) != 1:
        raise ValueError(f"cannot tell which of {numeric} is the score column")
    return file_col, numeric[0]


def load_dataset(data_dir=DATA_DIR, output_dir=OUTPUT_DIR):
    """Return transcripts with a `label` column (NaN for test) and the (file, score) column names.

    Test rows come in test.csv order. sample_submission.csv is only used for its
    column names: its ids are a mix of train, test and unknown files.
    """
    transcripts = load_transcripts(output_dir)
    train_csv = pd.read_csv(data_dir / "train.csv")

    is_train = transcripts["split"] == "train"
    file_col, label_col = detect_columns(train_csv, transcripts.loc[is_train, "file_id"])
    labels = train_csv.set_index(train_csv[file_col].astype(str))[label_col]
    transcripts["label"] = np.where(is_train, transcripts["file_id"].map(labels), np.nan)
    if transcripts.loc[is_train, "label"].isna().any():
        raise ValueError("some training clips have no label")

    sample_cols = list(pd.read_csv(data_dir / "sample_submission.csv", nrows=0).columns)
    if sample_cols != [file_col, label_col]:
        raise ValueError(f"sample_submission columns {sample_cols} differ from train.csv")
    print(f"file column '{file_col}', score column '{label_col}'")
    return transcripts, (file_col, label_col)


def write_submission(test_ids, test_pred, columns, path):
    """Write one row per test clip, with the same columns as sample_submission.csv."""
    file_col, label_col = columns
    sub = pd.DataFrame({file_col: list(test_ids), label_col: test_pred})
    sub.to_csv(path, index=False)
    return sub
