import argparse
import json
import os
from glob import glob
from typing import Dict, List, Optional, Tuple

import pandas as pd


FEATURE_COLS = ["open", "high", "low", "close", "vol", "amount"]


def _resolve_index_id(df: pd.DataFrame) -> str:
    if "ts_code" in df.columns:
        ts_code = df["ts_code"].dropna().astype(str)
        if not ts_code.empty and ts_code.iloc[0].strip():
            return ts_code.iloc[0].strip()
    if "name" in df.columns:
        names = df["name"].dropna().astype(str)
        if not names.empty and names.iloc[0].strip():
            return names.iloc[0].strip()
    raise ValueError("Unable to resolve index_id from ts_code or name.")


def _parse_date_series(date_series: pd.Series) -> pd.Series:
    series = date_series.copy()

    if pd.api.types.is_numeric_dtype(series):
        series = series.astype("Int64").astype(str)
    else:
        series = series.astype(str).str.strip()

    non_null = series[series.notna()]
    is_yyyymmdd = not non_null.empty and non_null.str.fullmatch(r"\d{8}").all()

    if is_yyyymmdd:
        parsed = pd.to_datetime(series, format="%Y%m%d", errors="raise")
    else:
        parsed = pd.to_datetime(series, errors="raise")

    return parsed.dt.normalize()


def _load_single_index_csv(csv_path: str, date_col: str = "date") -> Tuple[str, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    if date_col not in df.columns:
        raise ValueError(f"{csv_path} does not contain date column `{date_col}`.")

    index_id = _resolve_index_id(df)
    df = df.copy()
    df[date_col] = _parse_date_series(df[date_col])

    missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{csv_path} is missing required columns: {missing_cols}")

    loaded = df[[date_col] + FEATURE_COLS].rename(columns={date_col: "date"})
    loaded = loaded.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    loaded["index_id"] = index_id
    return index_id, loaded


def merge_indices_directory(
    indices_dir: str,
    output_csv: str,
    metadata_json: Optional[str] = None,
    date_col: str = "date",
) -> Dict[str, object]:
    csv_paths = sorted(glob(os.path.join(indices_dir, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No index csv files found under {indices_dir}")

    loaded_frames: List[pd.DataFrame] = []
    date_sets: List[set] = []
    index_ids: List[str] = []

    for csv_path in csv_paths:
        index_id, loaded = _load_single_index_csv(csv_path, date_col=date_col)
        loaded_frames.append(loaded)
        date_sets.append(set(loaded["date"]))
        index_ids.append(index_id)

    shared_dates = set.intersection(*date_sets) if date_sets else set()
    if not shared_dates:
        raise ValueError("No shared trading dates found across indices.")

    merged = []
    for frame in loaded_frames:
        filtered = frame[frame["date"].isin(shared_dates)].copy()
        merged.append(filtered[["date", "index_id"] + FEATURE_COLS])

    merged_df = pd.concat(merged, axis=0, ignore_index=True)
    merged_df = merged_df.sort_values(["date", "index_id"]).reset_index(drop=True)
    merged_df["date"] = merged_df["date"].dt.strftime("%Y-%m-%d")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    merged_df.to_csv(output_csv, index=False)

    metadata = {
        "index_ids": sorted(index_ids),
        "num_indices": len(index_ids),
        "shared_dates": len(shared_dates),
        "date_min": min(shared_dates).strftime("%Y-%m-%d"),
        "date_max": max(shared_dates).strftime("%Y-%m-%d"),
        "output_csv": output_csv,
    }

    if metadata_json is not None:
        os.makedirs(os.path.dirname(metadata_json) or ".", exist_ok=True)
        with open(metadata_json, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=True, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Merge multiple index daily csv files into a long-format file.")
    parser.add_argument("--indices-dir", type=str, default="data/indices", help="Directory containing index csv files.")
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/indices/merged_indices.csv",
        help="Path to the merged long-format output csv.",
    )
    parser.add_argument(
        "--metadata-json",
        type=str,
        default="data/indices/index_ids.json",
        help="Path to the metadata json containing ordered index ids.",
    )
    parser.add_argument("--date-col", type=str, default="date", help="Date column name used in the index csv files.")
    args = parser.parse_args()

    metadata = merge_indices_directory(
        indices_dir=args.indices_dir,
        output_csv=args.output_csv,
        metadata_json=args.metadata_json,
        date_col=args.date_col,
    )
    print(json.dumps(metadata, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
