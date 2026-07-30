import argparse
import os
import time
from typing import List, Optional, Tuple

import pandas as pd

try:
    import tushare as ts
except ImportError as exc:
    raise ImportError(
        "tushare is required for this script. Please install it with `pip install tushare`."
    ) from exc

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DEFAULT_START_DATE = "20251227"
DEFAULT_END_DATE = "20260407"
HARDCODED_TUSHARE_TOKEN = "f0c007a1d6183b0bd9c70ec4147769dca3594da7592e5d67b9538ec4"
HARDCODED_OUTPUT_DIR = "./data/stocks_test"
DEFAULT_FIELDS = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download A-share historical daily qfq data from Tushare and save one csv per stock."
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help=(
            "Tushare token. If omitted, the script will try the hardcoded token first, "
            "then TUSHARE_TOKEN / TUSHARE_API_TOKEN from the environment."
        ),
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=DEFAULT_START_DATE,
        help=f"Download start date in YYYYMMDD format. Default: {DEFAULT_START_DATE}",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=DEFAULT_END_DATE,
        help=f"Download end date in YYYYMMDD format. Default: {DEFAULT_END_DATE}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=HARDCODED_OUTPUT_DIR,
        help=f"Directory where per-stock csv files will be written. Default: {HARDCODED_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--list-status",
        type=str,
        default="L",
        choices=["L", "D", "P"],
        help="Tushare stock list status filter. L=listed, D=delisted, P=paused. Default: L",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.12,
        help="Sleep interval between requests to reduce rate-limit risk.",
    )
    parser.add_argument(
        "--retry-times",
        type=int,
        default=3,
        help="Retry times for each stock request.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing csv files in the output directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of stocks to download, useful for smoke tests.",
    )
    return parser.parse_args()


def resolve_token(explicit_token: Optional[str]) -> str:
    token = explicit_token or HARDCODED_TUSHARE_TOKEN or os.getenv("TUSHARE_TOKEN") or os.getenv("TUSHARE_API_TOKEN")
    if not token or token == "REPLACE_WITH_YOUR_TUSHARE_TOKEN":
        raise ValueError(
            "Missing Tushare token. Please edit HARDCODED_TUSHARE_TOKEN in the script, "
            "or pass --token, or set TUSHARE_TOKEN / TUSHARE_API_TOKEN."
        )
    return token.strip()


def normalize_date(date_str: str) -> str:
    value = str(date_str).strip()
    if len(value) != 8 or not value.isdigit():
        raise ValueError(f"Invalid date `{date_str}`. Expected YYYYMMDD.")
    return value


def is_a_share(ts_code: str) -> bool:
    code = str(ts_code).strip().upper()
    if not code.endswith((".SH", ".SZ", ".BJ")):
        return False

    symbol = code.split(".")[0]
    if symbol.startswith("200") or symbol.startswith("900"):
        return False
    return True


def fetch_a_share_stock_list(pro, list_status: str, end_date: str) -> pd.DataFrame:
    stock_df = pro.stock_basic(
        exchange="",
        list_status=list_status,
        fields="ts_code,symbol,name,area,industry,market,list_date",
    )
    if stock_df is None or stock_df.empty:
        raise ValueError("Tushare stock_basic returned no rows.")

    stock_df = stock_df.copy()
    stock_df["ts_code"] = stock_df["ts_code"].astype(str)
    stock_df["list_date"] = stock_df["list_date"].fillna("").astype(str)
    stock_df = stock_df[stock_df["ts_code"].map(is_a_share)].copy()
    stock_df = stock_df[(stock_df["list_date"] == "") | (stock_df["list_date"] <= end_date)].copy()
    stock_df = stock_df.sort_values("ts_code").reset_index(drop=True)
    return stock_df


def fetch_qfq_daily(pro, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = ts.pro_bar(
        api=pro,
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        asset="E",
        adj="qfq",
        freq="D",
    )
    if df is None:
        return pd.DataFrame()
    return df


def ensure_expected_columns(df: pd.DataFrame, ts_code: str) -> pd.DataFrame:
    working = df.copy()
    working["ts_code"] = ts_code

    for required_col in ["pre_close", "change", "pct_chg"]:
        if required_col not in working.columns:
            working[required_col] = pd.NA

    missing_cols = [col for col in DEFAULT_FIELDS if col not in working.columns]
    if missing_cols:
        raise ValueError(f"{ts_code} response is missing required columns: {missing_cols}")

    working = working[DEFAULT_FIELDS].copy()
    working["trade_date"] = working["trade_date"].astype(str)
    working = working.sort_values("trade_date").reset_index(drop=True)
    return working


def download_one_stock(
    pro,
    ts_code: str,
    start_date: str,
    end_date: str,
    retry_times: int,
    sleep_seconds: float,
) -> pd.DataFrame:
    last_error = None
    for attempt in range(1, retry_times + 1):
        try:
            df = fetch_qfq_daily(pro, ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df is None or df.empty:
                return pd.DataFrame()
            return ensure_expected_columns(df, ts_code=ts_code)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retry_times:
                time.sleep(max(0.0, sleep_seconds) * attempt)
            else:
                raise RuntimeError(f"Failed to download {ts_code} after {retry_times} attempts: {exc}") from exc
    raise RuntimeError(f"Failed to download {ts_code}: {last_error}")


def write_summary(output_dir: str, success_rows: List[dict], failed_rows: List[dict]):
    summary_path = os.path.join(output_dir, "_download_summary.csv")
    failed_path = os.path.join(output_dir, "_download_failed.csv")

    pd.DataFrame(success_rows).to_csv(summary_path, index=False)
    pd.DataFrame(failed_rows).to_csv(failed_path, index=False)
    print(f"Saved download summary: {summary_path}")
    print(f"Saved failed list: {failed_path}")


def iter_with_progress(items: List[Tuple[str, str]]):
    if tqdm is None:
        return items
    return tqdm(items, desc="Downloading A-share qfq daily data", unit="stock")


def main():
    args = parse_args()
    token = resolve_token(args.token)
    start_date = normalize_date(args.start_date)
    end_date = normalize_date(args.end_date)
    if start_date > end_date:
        raise ValueError(f"start_date {start_date} must be <= end_date {end_date}.")

    os.makedirs(args.output_dir, exist_ok=True)

    ts.set_token(token)
    pro = ts.pro_api(token)

    stock_df = fetch_a_share_stock_list(pro=pro, list_status=args.list_status, end_date=end_date)
    if args.limit is not None:
        stock_df = stock_df.head(args.limit).copy()

    print(
        f"Will download qfq daily data for {len(stock_df)} A-share stocks "
        f"from {start_date} to {end_date} into {args.output_dir}"
    )

    success_rows = []
    failed_rows = []
    tasks = list(stock_df[["ts_code", "name"]].itertuples(index=False, name=None))

    for ts_code, name in iter_with_progress(tasks):
        out_csv = os.path.join(args.output_dir, f"{ts_code}.csv")
        if os.path.exists(out_csv) and not args.overwrite:
            success_rows.append(
                {
                    "ts_code": ts_code,
                    "name": name,
                    "status": "skipped_existing",
                    "rows": pd.NA,
                    "output_csv": out_csv,
                }
            )
            continue

        try:
            df = download_one_stock(
                pro=pro,
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                retry_times=max(1, args.retry_times),
                sleep_seconds=max(0.0, args.sleep_seconds),
            )
            if df.empty:
                success_rows.append(
                    {
                        "ts_code": ts_code,
                        "name": name,
                        "status": "empty",
                        "rows": 0,
                        "output_csv": out_csv,
                    }
                )
            else:
                df.to_csv(out_csv, index=False)
                success_rows.append(
                    {
                        "ts_code": ts_code,
                        "name": name,
                        "status": "ok",
                        "rows": len(df),
                        "output_csv": out_csv,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            failed_rows.append(
                {
                    "ts_code": ts_code,
                    "name": name,
                    "error": str(exc),
                }
            )

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    write_summary(args.output_dir, success_rows=success_rows, failed_rows=failed_rows)
    ok_count = sum(1 for row in success_rows if row["status"] == "ok")
    skipped_count = sum(1 for row in success_rows if row["status"] == "skipped_existing")
    empty_count = sum(1 for row in success_rows if row["status"] == "empty")
    print(
        f"Download finished. ok={ok_count}, skipped_existing={skipped_count}, "
        f"empty={empty_count}, failed={len(failed_rows)}"
    )


if __name__ == "__main__":
    main()
