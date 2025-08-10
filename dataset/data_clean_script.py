import json
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from utils import (
    COMMON_DROP, NUMERIC_COLS, ORDERED_ATTRS, TASKS_ATTRS
)

def _label_encode_series(s: pd.Series) -> tuple[pd.Series, dict]:
    s_norm = s.astype("string").str.strip().str.lower()
    uniq = sorted([u for u in s_norm.dropna().unique()])
    mapping = {u: i for i, u in enumerate(uniq)}
    enc = s_norm.map(mapping).astype("Int64")
    return enc, mapping

def compute_split_stats(splits: dict, out_dir: Path):
    """Compute simple stats per split, print to terminal, and save to split_stats.json."""
    stats = {}
    for name, sdf in splits.items():
        info = {"n_rows": int(len(sdf))}
        print(f"\n=== Split: {name} ===")
        print(f"Rows: {info['n_rows']}")

        # Class distributions (0..K-1), include -1 if any slipped through
        for col in ("Attrition", "JobLevel", "JobSatisfaction"):
            if col in sdf.columns:
                vc = sdf[col].value_counts(dropna=False).sort_index()
                total = int(vc.sum())
                if total > 0:
                    print(f"\n{col} distribution:")
                    for k, v in vc.items():
                        try:
                            key = int(k)
                        except Exception:
                            key = str(k)
                        pct = (v / total) * 100
                        print(f"  {key}: {v} ({pct:.2f}%)")
                info[col] = {
                    int(k) if isinstance(k, (int, np.integer)) else str(k):
                    {"count": int(v), "pct": float(v) / total}
                    for k, v in vc.items()
                }

        # MonthlyIncome descriptives
        if "MonthlyIncome" in sdf.columns:
            s = pd.to_numeric(sdf["MonthlyIncome"], errors="coerce")
            desc = {
                "min": float(s.min()),
                "p25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "p75": float(s.quantile(0.75)),
                "max": float(s.max()),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)),
            }
            info["MonthlyIncome"] = desc
            print("\nMonthlyIncome stats:")
            for stat, val in desc.items():
                print(f"  {stat}: {val:.2f}")

        stats[name] = info

    # Save JSON summary
    with open(out_dir / "split_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\nSaved stats to: {out_dir/'split_stats.json'}")


def clean_and_split(raw_csv, out_dir, seed):
    # Load
    raw_csv = Path(raw_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw_csv)

    # Trim + lowercase on text columns
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype("string").str.strip().str.lower()

    # Drop useless + keep ordered attrs
    keep = [c for c in ORDERED_ATTRS if c in df.columns]
    df = df.drop(columns=[c for c in COMMON_DROP if c in df.columns], errors="ignore")
    df = df[keep].copy()

    # Force known numeric columns to numeric
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Encode ONLY task attributes and force 0..K-1 for classes
    enc_mappings = {}
    for col in TASKS_ATTRS:
        if col not in df.columns:
            continue

        # Binary attrition: no->0, yes->1
        if col == "Attrition":
            s = df[col].astype("string").str.strip().str.lower()
            mapping = {"no": 0, "yes": 1}
            df[col] = s.map(mapping).fillna(-1).astype("int32")
            enc_mappings[col] = mapping
            continue

        # Ordinal classifiers: shift 1-based -> 0-based
        if col in ("JobLevel", "JobSatisfaction"):
            s = pd.to_numeric(df[col], errors="coerce")                     # 1..K
            df[col] = (s - 1).astype("Int64").fillna(-1).astype("int32")    # 0..K-1
            continue

        # Other categoricals (if any): label-encode deterministically
        if not is_numeric_dtype(df[col]):
            enc, mapping = _label_encode_series(df[col])
            df[col] = enc.fillna(-1).astype("int32")
            enc_mappings[col] = mapping

    # Shuffle & split 80/10/10
    n = len(df)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    idx_train = idx[:n_train]
    idx_val = idx[n_train:n_train + n_val]
    idx_test = idx[n_train + n_val:]

    splits = {
        "train": df.iloc[idx_train].reset_index(drop=True),
        "val":   df.iloc[idx_val].reset_index(drop=True),
        "test":  df.iloc[idx_test].reset_index(drop=True),
    }

    # Write JSON
    for name, sdf in splits.items():
        with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(sdf.to_dict(orient="records"), f, ensure_ascii=False)

    # Save mappings for encoded task targets
    with open(out_dir / "categorical_mappings.json", "w", encoding="utf-8") as f:
        json.dump(enc_mappings, f, ensure_ascii=False, indent=2)

    # Compute and save split stats
    compute_split_stats(splits, out_dir)

    print(f"Saved: {out_dir/'train.json'} ({len(splits['train'])} rows)")
    print(f"Saved: {out_dir/'val.json'}   ({len(splits['val'])} rows)")
    print(f"Saved: {out_dir/'test.json'}  ({len(splits['test'])} rows)")

if __name__ == "__main__":
    clean_and_split(
        raw_csv="raw_data/hr-employee-attrition.csv",
        out_dir="toy_data",
        seed=1337,
    )