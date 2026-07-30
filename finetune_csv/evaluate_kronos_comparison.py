import argparse
import json
import os
import re
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PRICE_COLS = ["open", "high", "low", "close"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate existing Craft / Kronos prediction csv folders with the Kronos-paper metric definitions. "
            "Files are matched by 6-digit stock code and aligned by row order rather than by date."
        )
    )
    parser.add_argument("--truth-dir", type=str, required=True, help="Directory containing ground-truth stock csv files.")
    parser.add_argument("--craft-pred-dir", type=str, default=None, help="Directory containing Craft prediction csv files.")
    parser.add_argument("--kronos-pred-dir", type=str, default=None, help="Directory containing Kronos prediction csv files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save evaluation results.")
    parser.add_argument(
        "--vol-price-col",
        type=str,
        default="close",
        choices=PRICE_COLS,
        help="Price channel used to compute realized variance for volatility evaluation.",
    )
    parser.add_argument(
        "--allow-length-trim",
        action="store_true",
        help="If set, when a prediction file and truth file have different row counts, both are truncated to the shorter length.",
    )
    parser.add_argument(
        "--skip-length-mismatch",
        action="store_true",
        help="If set, skip stocks whose prediction/truth row counts differ instead of raising an error.",
    )
    return parser.parse_args()


def extract_stock_code(path: str) -> str:
    filename = os.path.basename(path)

    kronos_match = re.search(r"ped_(\d{6})_data\.csv$", filename, flags=re.IGNORECASE)
    if kronos_match:
        return kronos_match.group(1)

    craft_match = re.search(r"(\d{6})\.(?:SZ|SH)_qfq_daily\.csv$", filename, flags=re.IGNORECASE)
    if craft_match:
        return craft_match.group(1)

    generic_match = re.search(r"(\d{6})", filename)
    if generic_match:
        return generic_match.group(1)

    raise ValueError(f"Could not extract 6-digit stock code from filename: {filename}")


def build_file_map(folder: str, label: str) -> Dict[str, str]:
    if folder is None:
        return {}
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"{label} directory not found: {folder}")

    files = sorted(glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No csv files found under {label} directory: {folder}")

    mapping: Dict[str, str] = {}
    duplicates: Dict[str, List[str]] = {}
    for path in files:
        code = extract_stock_code(path)
        if code in mapping:
            duplicates.setdefault(code, [mapping[code]]).append(path)
        else:
            mapping[code] = path

    if duplicates:
        raise ValueError(f"Duplicate stock codes found in {label} directory: {duplicates}")
    return mapping


def ensure_columns(df: pd.DataFrame, columns: List[str], label: str):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def normalize_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    if "vol" not in renamed.columns and "volume" in renamed.columns:
        renamed = renamed.rename(columns={"volume": "vol"})
    return renamed


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    rank_x = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    rank_y = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    return pearson_corr(rank_x, rank_y)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    ss_res = float(np.sum((y_pred - y_true) ** 2))
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def mse(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    return float(np.mean((x - y) ** 2))


def mae(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    return float(np.mean(np.abs(x - y)))


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    value = mse(x, y)
    if np.isnan(value):
        return float("nan")
    return float(np.sqrt(value))


def mape(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    safe_y = np.where(np.abs(y) < 1e-8, np.nan, y)
    ratio = np.abs((x - y) / safe_y)
    if np.all(np.isnan(ratio)):
        return float("nan")
    return float(np.nanmean(ratio) * 100.0)


def direction_accuracy(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    pred_diff = np.sign(np.diff(x))
    true_diff = np.sign(np.diff(y))
    return float(np.mean(pred_diff == true_diff))


def realized_variance_from_prices(price_path: np.ndarray) -> float:
    if price_path.size < 2:
        return float("nan")
    safe_prices = np.clip(price_path.astype(np.float64), 1e-8, None)
    log_returns = np.diff(np.log(safe_prices))
    return float(np.sum(log_returns ** 2))


def align_by_row_order(
    pred_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    pred_path: str,
    truth_path: str,
    allow_length_trim: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pred_df = pred_df.reset_index(drop=True)
    truth_df = truth_df.reset_index(drop=True)

    if len(pred_df) != len(truth_df):
        if not allow_length_trim:
            raise ValueError(
                f"Row count mismatch between prediction and truth files: {pred_path} ({len(pred_df)}) vs "
                f"{truth_path} ({len(truth_df)}). Use --allow-length-trim if you want to truncate both sides to the shorter length."
            )
        keep_len = min(len(pred_df), len(truth_df))
        pred_df = pred_df.iloc[:keep_len].reset_index(drop=True)
        truth_df = truth_df.iloc[:keep_len].reset_index(drop=True)

    if len(pred_df) < 2:
        raise ValueError(f"Aligned sequence is too short for correlation / volatility evaluation: {pred_path}")
    return pred_df, truth_df


def evaluate_single_stock(
    model_name: str,
    code: str,
    pred_path: str,
    truth_path: str,
    vol_price_col: str,
    allow_length_trim: bool,
) -> Dict[str, object]:
    pred_df = normalize_prediction_columns(pd.read_csv(pred_path))
    truth_df = pd.read_csv(truth_path)

    ensure_columns(pred_df, ["date", "open", "high", "low", "close", "vol", "amount"], f"{model_name} prediction csv")
    ensure_columns(
        truth_df,
        ["ts_code", "date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"],
        "truth csv",
    )

    pred_df, truth_df = align_by_row_order(pred_df, truth_df, pred_path, truth_path, allow_length_trim=allow_length_trim)

    row: Dict[str, object] = {
        "model": model_name,
        "stock_code": code,
        "pred_path": pred_path,
        "truth_path": truth_path,
        "pred_rows": int(len(pred_df)),
        "truth_rows": int(len(truth_df)),
        "pred_start_date": str(pred_df["date"].iloc[0]),
        "pred_end_date": str(pred_df["date"].iloc[-1]),
        "truth_start_date": str(truth_df["date"].iloc[0]),
        "truth_end_date": str(truth_df["date"].iloc[-1]),
    }

    channel_ics: List[float] = []
    channel_rankics: List[float] = []
    channel_mses: List[float] = []
    channel_maes: List[float] = []
    channel_rmses: List[float] = []
    channel_mapes: List[float] = []
    channel_direction_accuracies: List[float] = []
    for channel in PRICE_COLS:
        pred_seq = pred_df[channel].to_numpy(dtype=np.float64)
        true_seq = truth_df[channel].to_numpy(dtype=np.float64)
        ic = pearson_corr(pred_seq, true_seq)
        rank_ic = spearman_corr(pred_seq, true_seq)
        channel_mse = mse(pred_seq, true_seq)
        channel_mae = mae(pred_seq, true_seq)
        channel_rmse = rmse(pred_seq, true_seq)
        channel_mape = mape(pred_seq, true_seq)
        channel_direction_accuracy = direction_accuracy(pred_seq, true_seq)
        row[f"{channel}_ic"] = ic
        row[f"{channel}_rankic"] = rank_ic
        row[f"{channel}_mse"] = channel_mse
        row[f"{channel}_mae"] = channel_mae
        row[f"{channel}_rmse"] = channel_rmse
        row[f"{channel}_mape"] = channel_mape
        row[f"{channel}_direction_accuracy"] = channel_direction_accuracy
        channel_ics.append(ic)
        channel_rankics.append(rank_ic)
        channel_mses.append(channel_mse)
        channel_maes.append(channel_mae)
        channel_rmses.append(channel_rmse)
        channel_mapes.append(channel_mape)
        channel_direction_accuracies.append(channel_direction_accuracy)

    row["price_ic"] = float(np.nanmean(channel_ics))
    row["price_rankic"] = float(np.nanmean(channel_rankics))
    row["price_mse"] = float(np.nanmean(channel_mses))
    row["price_mae"] = float(np.nanmean(channel_maes))
    row["price_rmse"] = float(np.nanmean(channel_rmses))
    row["price_mape"] = float(np.nanmean(channel_mapes))
    row["price_direction_accuracy"] = float(np.nanmean(channel_direction_accuracies))

    context_close = float(truth_df["pre_close"].iloc[0])
    pred_terminal_close = float(pred_df["close"].iloc[-1])
    true_terminal_close = float(truth_df["close"].iloc[-1])
    row["context_close"] = context_close
    row["pred_terminal_close"] = pred_terminal_close
    row["true_terminal_close"] = true_terminal_close
    row["pred_return"] = pred_terminal_close / max(context_close, 1e-8) - 1.0
    row["actual_return"] = true_terminal_close / max(context_close, 1e-8) - 1.0

    pred_vol_path = pred_df[vol_price_col].to_numpy(dtype=np.float64)
    true_vol_path = truth_df[vol_price_col].to_numpy(dtype=np.float64)
    row["pred_realized_variance"] = realized_variance_from_prices(pred_vol_path)
    row["actual_realized_variance"] = realized_variance_from_prices(true_vol_path)
    return row


def evaluate_model_folder(
    model_name: str,
    pred_dir: str,
    truth_map: Dict[str, str],
    vol_price_col: str,
    allow_length_trim: bool,
    skip_length_mismatch: bool,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    pred_map = build_file_map(pred_dir, f"{model_name} prediction")
    common_codes = sorted(set(pred_map.keys()) & set(truth_map.keys()))
    missing_truth_codes = sorted(set(pred_map.keys()) - set(truth_map.keys()))
    missing_prediction_codes = sorted(set(truth_map.keys()) - set(pred_map.keys()))

    if not common_codes:
        raise ValueError(f"No overlapping stock codes found between {model_name} predictions and truth csv files.")

    rows: List[Dict[str, object]] = []
    skipped_length_mismatch: List[Dict[str, object]] = []
    for code in common_codes:
        try:
            rows.append(
                evaluate_single_stock(
                    model_name=model_name,
                    code=code,
                    pred_path=pred_map[code],
                    truth_path=truth_map[code],
                    vol_price_col=vol_price_col,
                    allow_length_trim=allow_length_trim,
                )
            )
        except ValueError as exc:
            if skip_length_mismatch and "Row count mismatch between prediction and truth files" in str(exc):
                pred_rows = len(pd.read_csv(pred_map[code]))
                truth_rows = len(pd.read_csv(truth_map[code]))
                skipped_length_mismatch.append(
                    {
                        "stock_code": code,
                        "pred_rows": int(pred_rows),
                        "truth_rows": int(truth_rows),
                        "pred_path": pred_map[code],
                        "truth_path": truth_map[code],
                    }
                )
                continue
            raise

    if not rows:
        raise ValueError(
            f"All matched stocks were skipped for {model_name}. "
            "Please check whether the truth files really contain full 12-step future windows, "
            "or rerun with --allow-length-trim if you intentionally want to evaluate on truncated horizons."
        )

    per_stock_df = pd.DataFrame(rows)
    valid_returns = per_stock_df[["pred_return", "actual_return"]].dropna()
    valid_volatility = per_stock_df[["pred_realized_variance", "actual_realized_variance"]].dropna()

    metadata = {
        "model": model_name,
        "prediction_dir": os.path.abspath(pred_dir),
        "matched_stock_count": int(len(common_codes)),
        "missing_truth_count": int(len(missing_truth_codes)),
        "missing_prediction_count": int(len(missing_prediction_codes)),
        "skipped_length_mismatch_count": int(len(skipped_length_mismatch)),
        "missing_truth_codes": missing_truth_codes,
        "missing_prediction_codes": missing_prediction_codes,
        "skipped_length_mismatch": skipped_length_mismatch,
        "price_ic": float(per_stock_df["price_ic"].mean()),
        "price_rankic": float(per_stock_df["price_rankic"].mean()),
        "price_mse": float(per_stock_df["price_mse"].mean()),
        "price_mae": float(per_stock_df["price_mae"].mean()),
        "price_rmse": float(per_stock_df["price_rmse"].mean()),
        "price_mape": float(per_stock_df["price_mape"].mean()),
        "price_direction_accuracy": float(per_stock_df["price_direction_accuracy"].mean()),
        "return_ic": pearson_corr(
            valid_returns["pred_return"].to_numpy(dtype=np.float64),
            valid_returns["actual_return"].to_numpy(dtype=np.float64),
        ),
        "return_rankic": spearman_corr(
            valid_returns["pred_return"].to_numpy(dtype=np.float64),
            valid_returns["actual_return"].to_numpy(dtype=np.float64),
        ),
        "vol_mae": float(
            np.mean(
                np.abs(
                    valid_volatility["pred_realized_variance"].to_numpy(dtype=np.float64)
                    - valid_volatility["actual_realized_variance"].to_numpy(dtype=np.float64)
                )
            )
        )
        if not valid_volatility.empty
        else float("nan"),
        "vol_r2": r2_score(
            valid_volatility["actual_realized_variance"].to_numpy(dtype=np.float64),
            valid_volatility["pred_realized_variance"].to_numpy(dtype=np.float64),
        ),
    }
    return per_stock_df, metadata


def build_summary_rows(model_metadata: Dict[str, object]) -> List[Dict[str, object]]:
    sample_count = int(model_metadata["matched_stock_count"])
    return [
        {
            "task": "price",
            "model": model_metadata["model"],
            "metric": "IC",
            "value": model_metadata["price_ic"],
            "sample_count": sample_count,
        },
        {
            "task": "price",
            "model": model_metadata["model"],
            "metric": "RankIC",
            "value": model_metadata["price_rankic"],
            "sample_count": sample_count,
        },
        {
            "task": "price",
            "model": model_metadata["model"],
            "metric": "MSE",
            "value": model_metadata["price_mse"],
            "sample_count": sample_count,
        },
        {
            "task": "price",
            "model": model_metadata["model"],
            "metric": "MAE",
            "value": model_metadata["price_mae"],
            "sample_count": sample_count,
        },
        {
            "task": "price",
            "model": model_metadata["model"],
            "metric": "RMSE",
            "value": model_metadata["price_rmse"],
            "sample_count": sample_count,
        },
        {
            "task": "price",
            "model": model_metadata["model"],
            "metric": "MAPE",
            "value": model_metadata["price_mape"],
            "sample_count": sample_count,
        },
        {
            "task": "price",
            "model": model_metadata["model"],
            "metric": "Direction_Accuracy",
            "value": model_metadata["price_direction_accuracy"],
            "sample_count": sample_count,
        },
        {
            "task": "return",
            "model": model_metadata["model"],
            "metric": "IC",
            "value": model_metadata["return_ic"],
            "sample_count": sample_count,
        },
        {
            "task": "return",
            "model": model_metadata["model"],
            "metric": "RankIC",
            "value": model_metadata["return_rankic"],
            "sample_count": sample_count,
        },
        {
            "task": "volatility",
            "model": model_metadata["model"],
            "metric": "MAE",
            "value": model_metadata["vol_mae"],
            "sample_count": sample_count,
        },
        {
            "task": "volatility",
            "model": model_metadata["model"],
            "metric": "R2",
            "value": model_metadata["vol_r2"],
            "sample_count": sample_count,
        },
    ]


def main():
    args = parse_args()
    if not args.craft_pred_dir and not args.kronos_pred_dir:
        raise ValueError("Please provide at least one of --craft-pred-dir or --kronos-pred-dir.")

    os.makedirs(args.output_dir, exist_ok=True)
    truth_map = build_file_map(args.truth_dir, "truth")

    summary_rows: List[Dict[str, object]] = []
    metadata: Dict[str, object] = {
        "truth_dir": os.path.abspath(args.truth_dir),
        "truth_stock_count": int(len(truth_map)),
        "vol_price_col": args.vol_price_col,
        "row_alignment": "line_order",
        "notes": [
            "Prediction and truth csv files are matched by 6-digit stock code extracted from filenames.",
            "Date values are ignored during alignment; rows are aligned strictly by file order.",
            "p_t is taken from the first row of truth csv `pre_close`, because only the future 12-day window is visible.",
            "Price IC / RankIC follow the paper-style sample-level definition: compute sequence correlation over H rows for O/H/L/C separately, average over channels, then average over samples.",
            "Price MSE / MAE / RMSE / MAPE are computed per sample and per O/H/L/C channel over the aligned future price path, then averaged over channels and finally averaged over samples.",
            "Price Direction_Accuracy is computed on the step-to-step direction within the forecast window: compare sign(diff(pred_path)) with sign(diff(true_path)) for each O/H/L/C channel, then average over channels and samples.",
            "MAPE is reported in percentage units.",
            "Return IC / RankIC follow the paper-style terminal-return definition using the final predicted / true close and the first-row `pre_close` as p_t.",
            "Volatility uses realized variance style definition: sum((log p_{i+1} - log p_i)^2) on the chosen price channel, then evaluates MAE and R2 across samples.",
        ],
    }

    if args.craft_pred_dir:
        craft_df, craft_meta = evaluate_model_folder(
            model_name="craft",
            pred_dir=args.craft_pred_dir,
            truth_map=truth_map,
            vol_price_col=args.vol_price_col,
            allow_length_trim=args.allow_length_trim,
            skip_length_mismatch=args.skip_length_mismatch,
        )
        craft_df.to_csv(os.path.join(args.output_dir, "craft_per_stock_metrics.csv"), index=False)
        summary_rows.extend(build_summary_rows(craft_meta))
        metadata["craft"] = craft_meta

    if args.kronos_pred_dir:
        kronos_df, kronos_meta = evaluate_model_folder(
            model_name="kronos",
            pred_dir=args.kronos_pred_dir,
            truth_map=truth_map,
            vol_price_col=args.vol_price_col,
            allow_length_trim=args.allow_length_trim,
            skip_length_mismatch=args.skip_length_mismatch,
        )
        kronos_df.to_csv(os.path.join(args.output_dir, "kronos_per_stock_metrics.csv"), index=False)
        summary_rows.extend(build_summary_rows(kronos_meta))
        metadata["kronos"] = kronos_meta

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(args.output_dir, "metrics_summary.csv"), index=False)
    with open(os.path.join(args.output_dir, "evaluation_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n=== Metrics Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved evaluation files to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
