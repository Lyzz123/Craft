import json
import math
import os
from glob import glob
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


TIME_FEATURES = ["minute", "hour", "weekday", "day", "month"]
DEFAULT_FEATURE_COLS = ["open", "high", "low", "close", "vol", "amount"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = []
    for col in df.columns:
        cleaned = str(col).replace("\ufeff", "").strip()
        lowered = cleaned.lower()
        if lowered == "volume":
            lowered = "vol"
        normalized.append(lowered)
    df = df.copy()
    df.columns = normalized
    return df


def _parse_date_series(date_series: pd.Series) -> pd.Series:
    series = date_series.copy()
    null_mask = series.isna()

    if pd.api.types.is_numeric_dtype(series):
        series = series.astype("Int64").astype(str)
    else:
        series = series.astype(str).str.strip()

    series[null_mask] = pd.NA
    non_null = series[series.notna()]
    is_yyyymmdd = not non_null.empty and non_null.str.fullmatch(r"\d{8}").all()

    if is_yyyymmdd:
        parsed = pd.to_datetime(series, format="%Y%m%d", errors="raise")
    else:
        parsed = pd.to_datetime(series, errors="raise")
    return parsed.dt.normalize()


def _datetime_index_to_int64_array(index_like: Sequence[pd.Timestamp]) -> np.ndarray:
    return pd.DatetimeIndex(index_like).to_numpy(dtype="datetime64[ns]").astype(np.int64)


def _timestamp_to_int64(ts: pd.Timestamp) -> np.int64:
    return np.int64(pd.Timestamp(ts).to_datetime64().astype("datetime64[ns]").astype(np.int64))


def _canonicalize_feature_frame(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    df = _normalize_columns(df)

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return df


def build_time_features(dates: Sequence[pd.Timestamp]) -> np.ndarray:
    date_series = pd.Series(pd.to_datetime(dates))
    time_df = pd.DataFrame(
        {
            "minute": date_series.dt.minute,
            "hour": date_series.dt.hour,
            "weekday": date_series.dt.weekday,
            "day": date_series.dt.day,
            "month": date_series.dt.month,
        }
    )
    return time_df[TIME_FEATURES].values.astype(np.float32)


def normalize_stock_window(stock_window: np.ndarray, lookback: int, clip: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    context = stock_window[:lookback]
    mean = context.mean(axis=0)
    std = context.std(axis=0)
    normalized = (stock_window - mean) / (std + 1e-5)
    normalized = np.clip(normalized, -clip, clip)
    return normalized.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def normalize_index_window(index_window: np.ndarray, lookback: int, clip: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    context = index_window[:lookback]
    mean = context.mean(axis=0)
    std = context.std(axis=0)
    normalized = (index_window - mean[None, :, :]) / (std[None, :, :] + 1e-5)
    normalized = np.clip(normalized, -clip, clip)
    return normalized.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def load_index_metadata(metadata_json: Optional[str]) -> Optional[List[str]]:
    if not metadata_json or not os.path.exists(metadata_json):
        return None
    with open(metadata_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    index_ids = payload.get("index_ids")
    if not index_ids:
        return None
    return list(index_ids)


def load_merged_indices_panel(
    merged_indices_path: str,
    feature_cols: Sequence[str],
    date_col: str = "date",
    index_ids: Optional[Sequence[str]] = None,
) -> Tuple[pd.DatetimeIndex, np.ndarray, List[str]]:
    if not os.path.exists(merged_indices_path):
        raise FileNotFoundError(f"Merged indices file not found: {merged_indices_path}")

    df = pd.read_csv(merged_indices_path)
    if date_col not in df.columns:
        raise ValueError(f"Merged indices file is missing date column `{date_col}`.")
    if "index_id" not in df.columns:
        raise ValueError("Merged indices file is missing `index_id` column.")

    df = _canonicalize_feature_frame(df, feature_cols)
    df = df.copy()
    df[date_col] = _parse_date_series(df[date_col])
    if index_ids is None:
        index_ids = sorted(df["index_id"].astype(str).unique().tolist())
    else:
        index_ids = list(index_ids)

    df = df[df["index_id"].astype(str).isin(index_ids)].copy()
    df["index_id"] = df["index_id"].astype(str)

    counts = df.groupby(date_col)["index_id"].nunique()
    valid_dates = counts[counts == len(index_ids)].index
    df = df[df[date_col].isin(valid_dates)].copy()
    if df.empty:
        raise ValueError("Merged indices file does not contain any fully aligned dates for the selected indices.")

    all_features = []
    for feature in feature_cols:
        pivot = (
            df.pivot_table(index=date_col, columns="index_id", values=feature, aggfunc="last")
            .reindex(columns=index_ids)
            .sort_index()
        )
        all_features.append(pivot)

    valid_mask = ~np.logical_or.reduce([feature_df.isna().values for feature_df in all_features])
    valid_dates = all_features[0].index[valid_mask.all(axis=1)]
    if len(valid_dates) == 0:
        raise ValueError("Merged indices panel has no fully populated dates after feature alignment.")

    panel_slices = []
    for feature_df in all_features:
        feature_df = feature_df.loc[valid_dates]
        panel_slices.append(feature_df.values[:, :, None])

    panel = np.concatenate(panel_slices, axis=-1).astype(np.float32)
    return pd.DatetimeIndex(valid_dates), panel, index_ids


def _compute_split_bounds(length: int, train_ratio: float, val_ratio: float) -> Dict[str, Tuple[int, int]]:
    train_end = int(length * train_ratio)
    val_end = int(length * (train_ratio + val_ratio))
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, length),
    }


class MultiStreamStockDataset(Dataset):

    def __init__(
        self,
        stock_dir: str,
        merged_indices_path: str,
        split: str,
        lookback: int,
        horizon: int,
        clip: float = 5.0,
        stock_date_col: str = "trade_date",
        index_date_col: str = "date",
        feature_cols: Optional[Sequence[str]] = None,
        index_ids: Optional[Sequence[str]] = None,
        train_ratio: float = 0.9,
        val_ratio: float = 0.1,
        test_ratio: float = 0.0,
        sample_stride: int = 1,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of {'train', 'val', 'test'}")
        self.stock_dir = stock_dir
        self.merged_indices_path = merged_indices_path
        self.split = split
        self.lookback = lookback
        self.horizon = horizon
        self.window = lookback + horizon + 1
        self.sample_stride = int(sample_stride)
        if self.sample_stride < 1:
            raise ValueError("sample_stride must be at least 1")
        self.clip = clip
        self.stock_date_col = stock_date_col
        self.index_date_col = index_date_col
        self.feature_cols = list(feature_cols or DEFAULT_FEATURE_COLS)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        index_dates, index_panel, index_labels = load_merged_indices_panel(
            merged_indices_path=self.merged_indices_path,
            feature_cols=self.feature_cols,
            date_col=self.index_date_col,
            index_ids=index_ids,
        )
        self.index_dates = index_dates
        self.index_panel = index_panel
        self.index_labels = index_labels
        self.index_id_tensor = torch.arange(len(index_labels), dtype=torch.long)
        self.num_indices = len(index_labels)

        self.series: List[Dict[str, object]] = []
        self.samples: List[Tuple[int, int]] = []
        self.diagnostics: Dict[str, object] = {
            "stock_files": 0,
            "aligned_series": 0,
            "contributing_series": 0,
            "skipped_short_common_dates": [],
            "skipped_short_split": [],
        }
        self._build_index()

    def _resolve_stock_id(self, stock_df: pd.DataFrame, stock_path: str) -> str:
        if "ts_code" in stock_df.columns:
            series = stock_df["ts_code"].dropna().astype(str)
            if not series.empty:
                return series.iloc[0]
        return os.path.splitext(os.path.basename(stock_path))[0]

    def _build_index(self):
        stock_paths = sorted(glob(os.path.join(self.stock_dir, "*.csv")))
        if not stock_paths:
            raise FileNotFoundError(f"No stock csv files found under {self.stock_dir}")
        self.diagnostics["stock_files"] = len(stock_paths)

        index_pos = {date: pos for pos, date in enumerate(self.index_dates)}

        for stock_path in stock_paths:
            raw_stock_df = pd.read_csv(stock_path)
            raw_stock_df = _normalize_columns(raw_stock_df)
            if self.stock_date_col not in raw_stock_df.columns:
                raise ValueError(f"{stock_path} does not contain stock date column `{self.stock_date_col}`.")

            stock_id = self._resolve_stock_id(raw_stock_df, stock_path)
            stock_df = _canonicalize_feature_frame(raw_stock_df, self.feature_cols)
            stock_df = stock_df.copy()
            stock_df[self.stock_date_col] = _parse_date_series(stock_df[self.stock_date_col])
            stock_df = stock_df.sort_values(self.stock_date_col).drop_duplicates(subset=[self.stock_date_col], keep="last")
            stock_df = stock_df[[self.stock_date_col] + self.feature_cols]
            stock_df = stock_df.rename(columns={self.stock_date_col: "date"})

            common_dates = pd.Index(stock_df["date"]).intersection(self.index_dates).sort_values()
            if len(common_dates) < self.window:
                self.diagnostics["skipped_short_common_dates"].append(
                    {
                        "stock_id": stock_id,
                        "aligned_length": int(len(common_dates)),
                    }
                )
                continue

            stock_aligned = stock_df.set_index("date").loc[common_dates, self.feature_cols].values.astype(np.float32)
            index_positions = [index_pos[date] for date in common_dates]
            indices_aligned = self.index_panel[index_positions].astype(np.float32)
            time_aligned = build_time_features(common_dates)
            self.diagnostics["aligned_series"] += 1

            series_payload = {
                "stock_id": stock_id,
                "stock_path": stock_path,
                "dates": pd.DatetimeIndex(common_dates),
                "stock_values": stock_aligned,
                "index_values": indices_aligned,
                "time_values": time_aligned,
            }
            series_idx = len(self.series)
            self.series.append(series_payload)

            split_bounds = _compute_split_bounds(len(common_dates), self.train_ratio, self.val_ratio)
            split_start, split_end = split_bounds[self.split]
            segment_len = split_end - split_start
            if segment_len < self.window:
                self.diagnostics["skipped_short_split"].append(
                    {
                        "stock_id": stock_id,
                        "aligned_length": int(len(common_dates)),
                        "split_length": int(segment_len),
                    }
                )
                continue

            local_samples = 0
            for local_start in range(0, segment_len - self.window + 1, self.sample_stride):
                self.samples.append((series_idx, split_start + local_start))
                local_samples += 1
            if local_samples > 0:
                self.diagnostics["contributing_series"] += 1

    def __len__(self) -> int:
        return len(self.samples)

    def diagnostics_summary(self, max_examples: int = 5) -> str:
        def _format_examples(records: List[Dict[str, int]]) -> str:
            if not records:
                return "none"
            preview = records[:max_examples]
            return "; ".join(
                [
                    ", ".join([f"{key}={value}" for key, value in record.items()])
                    for record in preview
                ]
            )

        return (
            f"split={self.split}, samples={len(self.samples)}, stock_files={self.diagnostics['stock_files']}, "
            f"aligned_series={self.diagnostics['aligned_series']}, "
            f"contributing_series={self.diagnostics['contributing_series']}, "
            f"window={self.window}, "
            f"sample_stride={self.sample_stride}, "
            f"skipped_short_common_dates={len(self.diagnostics['skipped_short_common_dates'])} "
            f"({ _format_examples(self.diagnostics['skipped_short_common_dates']) }), "
            f"skipped_short_split={len(self.diagnostics['skipped_short_split'])} "
            f"({ _format_examples(self.diagnostics['skipped_short_split']) })"
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        series_idx, start = self.samples[idx]
        payload = self.series[series_idx]
        end = start + self.window

        stock_window = payload["stock_values"][start:end]
        index_window = payload["index_values"][start:end]
        time_window = payload["time_values"][start:end]

        stock_seq, stock_mean, stock_std = normalize_stock_window(stock_window, self.lookback, self.clip)
        index_seq, index_mean, index_std = normalize_index_window(index_window, self.lookback, self.clip)

        return {
            "stock_seq": torch.from_numpy(stock_seq),
            "index_seq": torch.from_numpy(index_seq),
            "time_seq": torch.from_numpy(time_window.astype(np.float32)),
            "index_ids": self.index_id_tensor.clone(),
            "stock_mean": torch.from_numpy(stock_mean),
            "stock_std": torch.from_numpy(stock_std),
            "index_mean": torch.from_numpy(index_mean),
            "index_std": torch.from_numpy(index_std),
        }


class MultiStreamInferenceDataset(Dataset):

    def __init__(
        self,
        stock_paths: Sequence[str],
        merged_indices_path: str,
        lookback: int,
        horizon: int,
        clip: float = 5.0,
        stock_date_col: str = "trade_date",
        index_date_col: str = "date",
        feature_cols: Optional[Sequence[str]] = None,
        index_ids: Optional[Sequence[str]] = None,
        horizon_start_date: Optional[str] = None,
    ):
        if not stock_paths:
            raise ValueError("stock_paths must not be empty for inference.")
        self.stock_paths = list(stock_paths)
        self.lookback = lookback
        self.horizon = horizon
        self.clip = clip
        self.stock_date_col = stock_date_col
        self.index_date_col = index_date_col
        self.feature_cols = list(feature_cols or DEFAULT_FEATURE_COLS)
        self.horizon_start_date = (
            pd.Timestamp(horizon_start_date).normalize() if horizon_start_date is not None else None
        )

        index_dates, index_panel, index_labels = load_merged_indices_panel(
            merged_indices_path=merged_indices_path,
            feature_cols=self.feature_cols,
            date_col=self.index_date_col,
            index_ids=index_ids,
        )
        self.index_dates = index_dates
        self.index_panel = index_panel
        self.index_labels = index_labels
        self.index_id_tensor = torch.arange(len(index_labels), dtype=torch.long)
        self.samples: List[Dict[str, object]] = []
        self._build_samples()

    def _resolve_stock_id(self, stock_df: pd.DataFrame, stock_path: str) -> str:
        if "ts_code" in stock_df.columns:
            series = stock_df["ts_code"].dropna().astype(str)
            if not series.empty:
                return series.iloc[0]
        return os.path.splitext(os.path.basename(stock_path))[0]

    def _resolve_lookback_bounds(self, common_dates: pd.DatetimeIndex) -> Optional[Tuple[int, int]]:
        if len(common_dates) < self.lookback:
            return None

        if self.horizon_start_date is None:
            end_idx = len(common_dates)
        else:
            # Use the aligned rows strictly before the requested prediction horizon start.
            end_idx = int(common_dates.searchsorted(self.horizon_start_date, side="left"))

        start_idx = end_idx - self.lookback
        if end_idx <= 0 or start_idx < 0:
            return None
        return start_idx, end_idx

    def _build_samples(self):
        index_pos = {date: pos for pos, date in enumerate(self.index_dates)}
        for stock_path in self.stock_paths:
            raw_stock_df = pd.read_csv(stock_path)
            raw_stock_df = _normalize_columns(raw_stock_df)
            if self.stock_date_col not in raw_stock_df.columns:
                raise ValueError(f"{stock_path} does not contain stock date column `{self.stock_date_col}`.")
            stock_id = self._resolve_stock_id(raw_stock_df, stock_path)
            stock_df = _canonicalize_feature_frame(raw_stock_df, self.feature_cols)
            stock_df = stock_df.copy()
            stock_df[self.stock_date_col] = _parse_date_series(stock_df[self.stock_date_col])
            stock_df = stock_df.sort_values(self.stock_date_col).drop_duplicates(subset=[self.stock_date_col], keep="last")
            stock_df = stock_df[[self.stock_date_col] + self.feature_cols].rename(columns={self.stock_date_col: "date"})

            common_dates = pd.Index(stock_df["date"]).intersection(self.index_dates).sort_values()
            lookback_bounds = self._resolve_lookback_bounds(pd.DatetimeIndex(common_dates))
            if lookback_bounds is None:
                continue

            stock_aligned = stock_df.set_index("date").loc[common_dates, self.feature_cols].values.astype(np.float32)
            index_positions = [index_pos[date] for date in common_dates]
            indices_aligned = self.index_panel[index_positions].astype(np.float32)

            start_idx, end_idx = lookback_bounds
            stock_context = stock_aligned[start_idx:end_idx]
            index_context = indices_aligned[start_idx:end_idx]
            x_dates = pd.DatetimeIndex(common_dates[start_idx:end_idx])
            if self.horizon_start_date is None:
                y_dates = pd.bdate_range(start=x_dates[-1] + pd.Timedelta(days=1), periods=self.horizon)
            else:
                y_dates = pd.bdate_range(start=self.horizon_start_date, periods=self.horizon)

            stock_norm, stock_mean, stock_std = normalize_stock_window(stock_context, self.lookback, self.clip)
            index_norm, index_mean, index_std = normalize_index_window(index_context, self.lookback, self.clip)

            self.samples.append(
                {
                    "stock_id": stock_id,
                    "stock_path": stock_path,
                    "stock_seq": stock_norm,
                    "index_seq": index_norm,
                    "x_time_seq": build_time_features(x_dates),
                    "y_time_seq": build_time_features(y_dates),
                    "lookback_start_date": _timestamp_to_int64(x_dates[0]),
                    "lookback_end_date": _timestamp_to_int64(x_dates[-1]),
                    "horizon_start_date": _timestamp_to_int64(y_dates[0]),
                    "horizon_end_date": _timestamp_to_int64(y_dates[-1]),
                    "future_dates": _datetime_index_to_int64_array(y_dates),
                    "stock_mean": stock_mean,
                    "stock_std": stock_std,
                    "index_mean": index_mean,
                    "index_std": index_std,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        payload = self.samples[idx]
        return {
            "stock_id": payload["stock_id"],
            "stock_path": payload["stock_path"],
            "stock_seq": torch.from_numpy(payload["stock_seq"]),
            "index_seq": torch.from_numpy(payload["index_seq"]),
            "x_time_seq": torch.from_numpy(payload["x_time_seq"]),
            "y_time_seq": torch.from_numpy(payload["y_time_seq"]),
            "lookback_start_date": torch.tensor(payload["lookback_start_date"], dtype=torch.long),
            "lookback_end_date": torch.tensor(payload["lookback_end_date"], dtype=torch.long),
            "horizon_start_date": torch.tensor(payload["horizon_start_date"], dtype=torch.long),
            "horizon_end_date": torch.tensor(payload["horizon_end_date"], dtype=torch.long),
            "future_dates": torch.from_numpy(payload["future_dates"]),
            "index_ids": self.index_id_tensor.clone(),
            "stock_mean": torch.from_numpy(payload["stock_mean"]),
            "stock_std": torch.from_numpy(payload["stock_std"]),
            "index_mean": torch.from_numpy(payload["index_mean"]),
            "index_std": torch.from_numpy(payload["index_std"]),
        }


def multistream_collate_fn(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    collated: Dict[str, object] = {}
    tensor_keys = [key for key, value in batch[0].items() if torch.is_tensor(value)]
    for key in tensor_keys:
        collated[key] = torch.stack([item[key] for item in batch], dim=0)

    non_tensor_keys = [key for key in batch[0].keys() if key not in tensor_keys]
    for key in non_tensor_keys:
        collated[key] = [item[key] for item in batch]
    return collated


def create_multistream_dataloaders(config) -> Tuple[DataLoader, DataLoader, MultiStreamStockDataset, MultiStreamStockDataset]:
    train_dataset = MultiStreamStockDataset(
        stock_dir=config.stock_dir,
        merged_indices_path=config.merged_indices_path,
        split="train",
        lookback=config.lookback,
        horizon=config.horizon,
        sample_stride=config.sample_stride,
        clip=config.clip,
        stock_date_col=config.stock_date_col,
        index_date_col=config.index_date_col,
        feature_cols=config.feature_cols,
        index_ids=config.index_ids,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )
    val_dataset = MultiStreamStockDataset(
        stock_dir=config.stock_dir,
        merged_indices_path=config.merged_indices_path,
        split="val",
        lookback=config.lookback,
        horizon=config.horizon,
        sample_stride=config.sample_stride,
        clip=config.clip,
        stock_date_col=config.stock_date_col,
        index_date_col=config.index_date_col,
        feature_cols=config.feature_cols,
        index_ids=train_dataset.index_labels,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )

    if len(train_dataset) == 0:
        min_total_length = math.ceil(train_dataset.window / max(config.train_ratio, 1e-8))
        raise ValueError(
            "No training samples were created for the multistream dataset. "
            f"Current window length is lookback + horizon + 1 = {config.lookback} + {config.horizon} + 1 = {train_dataset.window}. "
            f"With train_ratio={config.train_ratio}, each stock needs at least about {min_total_length} aligned dates "
            "after stock/index date intersection to contribute one training sample. "
            f"Diagnostics: {train_dataset.diagnostics_summary()}. "
            "Try reducing lookback or horizon, increasing train_ratio, or checking whether stock dates and merged index dates overlap enough."
        )

    if config.val_ratio > 0 and len(val_dataset) == 0:
        min_total_length = math.ceil(val_dataset.window / max(config.val_ratio, 1e-8))
        raise ValueError(
            "No validation samples were created for the multistream dataset. "
            f"Current window length is {val_dataset.window}, and with val_ratio={config.val_ratio}, "
            f"each stock needs at least about {min_total_length} aligned dates to contribute one validation sample. "
            f"Diagnostics: {val_dataset.diagnostics_summary()}. "
            "Try increasing val_ratio, reducing lookback or horizon, or using a larger aligned history window."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=len(train_dataset) >= config.batch_size,
        collate_fn=multistream_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=multistream_collate_fn,
    )
    return train_loader, val_loader, train_dataset, val_dataset
