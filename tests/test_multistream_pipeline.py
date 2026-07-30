from pathlib import Path

import pandas as pd
import torch

from finetune_csv.multistream_dataset import MultiStreamStockDataset
from finetune_csv.preprocess_indices import merge_indices_directory
from finetune_csv.token_cache import build_cached_tokenized_dataset
from model.craft import CraftBackbone
from model.multistream_craft import MultiIndexMemoryBuilder, MultiStreamCraftModel


FEATURE_COLS = ["open", "high", "low", "close", "vol", "amount"]


def _write_index_csv(path: Path, index_id: str = None, name: str = None, dates=None, base_value: float = 1.0):
    rows = []
    for idx, date in enumerate(dates):
        rows.append(
            {
                "name": name,
                "ts_code": index_id,
                "date": date,
                "open": base_value + idx,
                "high": base_value + idx + 0.5,
                "low": base_value + idx - 0.5,
                "close": base_value + idx + 0.1,
                "pre_close": base_value + idx - 0.1,
                "change": 0.1,
                "pct_chg": 0.2,
                "vol": 1000 + idx,
                "amount": 2000 + idx,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_stock_csv(path: Path, dates, base_value: float = 10.0):
    rows = []
    for idx, date in enumerate(dates):
        rows.append(
            {
                "ts_code": "000001.SZ",
                "trade_date": date,
                "open": base_value + idx,
                "high": base_value + idx + 1.0,
                "low": base_value + idx - 1.0,
                "close": base_value + idx + 0.5,
                "pre_close": base_value + idx - 0.5,
                "change": 0.1,
                "pct_chg": 0.2,
                "vol": 5000 + idx,
                "amount": 8000 + idx,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_tiny_craft() -> CraftBackbone:
    return CraftBackbone(
        s1_bits=2,
        s2_bits=2,
        n_layers=1,
        d_model=8,
        n_heads=2,
        ff_dim=16,
        ffn_dropout_p=0.0,
        attn_dropout_p=0.0,
        resid_dropout_p=0.0,
        token_dropout_p=0.0,
        learn_te=True,
    )


class DummyTokenizer:

    def __init__(self):
        self.training = True

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def eval(self):
        self.training = False
        return self

    def encode(self, x, half=True):
        s1 = torch.round(x[..., 0]).long().abs() % 17
        s2 = torch.round(x[..., 1]).long().abs() % 19
        return s1, s2


def test_merge_indices_intersection_and_fallback_name(tmp_path: Path):
    indices_dir = tmp_path / "indices"
    indices_dir.mkdir()

    dates_1 = pd.date_range("2024-01-01", periods=5, freq="D")
    dates_2 = pd.date_range("2024-01-03", periods=5, freq="D")
    _write_index_csv(indices_dir / "idx_1.csv", index_id="000300.SH", name="CSI300", dates=dates_1, base_value=1.0)
    _write_index_csv(indices_dir / "idx_2.csv", index_id=None, name="ZZ500", dates=dates_2, base_value=2.0)

    merged_csv = tmp_path / "merged_indices.csv"
    metadata_json = tmp_path / "index_ids.json"
    metadata = merge_indices_directory(str(indices_dir), str(merged_csv), str(metadata_json))

    merged_df = pd.read_csv(merged_csv, parse_dates=["date"])
    expected_dates = pd.date_range("2024-01-03", periods=3, freq="D")
    assert sorted(merged_df["date"].dt.strftime("%Y-%m-%d").unique().tolist()) == [d.strftime("%Y-%m-%d") for d in expected_dates]
    assert sorted(merged_df["index_id"].unique().tolist()) == ["000300.SH", "ZZ500"]
    assert metadata["num_indices"] == 2


def test_multistream_dataset_shapes_and_context_only_normalization(tmp_path: Path):
    indices_dir = tmp_path / "indices"
    stocks_dir = tmp_path / "stocks"
    indices_dir.mkdir()
    stocks_dir.mkdir()

    dates = pd.date_range("2024-02-01", periods=8, freq="D")
    _write_index_csv(indices_dir / "idx_1.csv", index_id="000300.SH", name="CSI300", dates=dates, base_value=1.0)
    _write_index_csv(indices_dir / "idx_2.csv", index_id="000905.SH", name="CSI500", dates=dates, base_value=5.0)
    merged_csv = tmp_path / "merged_indices.csv"
    merge_indices_directory(str(indices_dir), str(merged_csv), str(tmp_path / "index_ids.json"))

    _write_stock_csv(stocks_dir / "000001.csv", dates=dates, base_value=10.0)

    dataset = MultiStreamStockDataset(
        stock_dir=str(stocks_dir),
        merged_indices_path=str(merged_csv),
        split="train",
        lookback=3,
        horizon=2,
        clip=5.0,
        stock_date_col="trade_date",
        index_date_col="date",
        feature_cols=FEATURE_COLS,
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
    )

    sample = dataset[0]
    assert sample["stock_seq"].shape == (6, 6)
    assert sample["index_seq"].shape == (6, 2, 6)
    assert sample["time_seq"].shape == (6, 5)
    assert sample["index_ids"].tolist() == [0, 1]

    raw_stock = pd.read_csv(stocks_dir / "000001.csv")
    raw_window = raw_stock[FEATURE_COLS].iloc[:6].to_numpy(dtype="float32")
    expected_mean = raw_window[:3].mean(axis=0)
    expected_std = raw_window[:3].std(axis=0)
    expected_norm = (raw_window - expected_mean) / (expected_std + 1e-5)
    assert torch.allclose(sample["stock_mean"], torch.tensor(expected_mean), atol=1e-5)
    assert torch.allclose(sample["stock_seq"], torch.tensor(expected_norm), atol=1e-4)


def test_cached_tokenized_dataset_matches_raw_encoding(tmp_path: Path):
    indices_dir = tmp_path / "indices"
    stocks_dir = tmp_path / "stocks"
    indices_dir.mkdir()
    stocks_dir.mkdir()

    dates = pd.date_range("2024-03-01", periods=9, freq="D")
    _write_index_csv(indices_dir / "idx_1.csv", index_id="000300.SH", name="CSI300", dates=dates, base_value=1.0)
    _write_index_csv(indices_dir / "idx_2.csv", index_id="000905.SH", name="CSI500", dates=dates, base_value=5.0)
    merged_csv = tmp_path / "merged_indices.csv"
    merge_indices_directory(str(indices_dir), str(merged_csv), str(tmp_path / "index_ids.json"))

    _write_stock_csv(stocks_dir / "000001.csv", dates=dates, base_value=10.0)
    dataset = MultiStreamStockDataset(
        stock_dir=str(stocks_dir),
        merged_indices_path=str(merged_csv),
        split="train",
        lookback=3,
        horizon=2,
        clip=5.0,
        stock_date_col="trade_date",
        index_date_col="date",
        feature_cols=FEATURE_COLS,
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
    )

    cached_dataset = build_cached_tokenized_dataset(
        dataset=dataset,
        stock_tokenizer=DummyTokenizer(),
        index_tokenizer=DummyTokenizer(),
        device=torch.device("cpu"),
        batch_size=2,
        num_workers=0,
        logger=None,
        split_name="train",
    )

    raw_sample = dataset[0]
    cached_sample = cached_dataset[0]
    expected_stock_s1 = torch.round(raw_sample["stock_seq"][..., 0]).long().abs() % 17
    expected_stock_s2 = torch.round(raw_sample["stock_seq"][..., 1]).long().abs() % 19
    expected_index_s1 = torch.round(raw_sample["index_seq"][..., 0]).long().abs() % 17
    expected_index_s2 = torch.round(raw_sample["index_seq"][..., 1]).long().abs() % 19

    assert cached_sample["stock_s1"].shape == (6,)
    assert cached_sample["index_s1"].shape == (6, 2)
    assert torch.equal(cached_sample["stock_s1"], expected_stock_s1)
    assert torch.equal(cached_sample["stock_s2"], expected_stock_s2)
    assert torch.equal(cached_sample["index_s1"], expected_index_s1)
    assert torch.equal(cached_sample["index_s2"], expected_index_s2)
    assert torch.equal(cached_sample["time_seq"], raw_sample["time_seq"])
    assert cached_sample["index_ids"].tolist() == [0, 1]


def test_memory_builder_is_past_only():
    hidden = torch.arange(1 * 4 * 2 * 3, dtype=torch.float32).view(1, 4, 2, 3)
    builder = MultiIndexMemoryBuilder(lag_window=2)
    memory, lag_ids, valid_mask = builder(hidden)

    assert memory.shape == (1, 4, 6, 3)
    assert lag_ids.shape == (4, 6)
    assert valid_mask[0, 0].tolist() == [True, True, False, False, False, False]
    assert torch.equal(memory[0, 1, 2:4], hidden[0, 0])
    assert torch.equal(memory[0, 3, 4:6], hidden[0, 1])


def test_multistream_model_forward_shapes():
    model = MultiStreamCraftModel(
        stock_predictor=_build_tiny_craft(),
        index_predictor=_build_tiny_craft(),
        num_indices=2,
        lag_window=2,
    )

    batch_size, seq_len, num_indices = 2, 5, 2
    stock_s1 = torch.randint(0, 4, (batch_size, seq_len))
    stock_s2 = torch.randint(0, 4, (batch_size, seq_len))
    index_s1 = torch.randint(0, 4, (batch_size, seq_len, num_indices))
    index_s2 = torch.randint(0, 4, (batch_size, seq_len, num_indices))
    stamp = torch.randint(0, 5, (batch_size, seq_len, 5))
    index_ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)

    outputs = model(
        stock_s1_ids=stock_s1,
        stock_s2_ids=stock_s2,
        index_s1_ids=index_s1,
        index_s2_ids=index_s2,
        stamp=stamp,
        index_ids=index_ids,
        stock_s1_targets=stock_s1,
        index_s1_targets=index_s1,
    )
    losses = model.compute_losses(
        outputs,
        stock_targets=(stock_s1, stock_s2),
        index_targets=(index_s1, index_s2),
    )

    assert outputs["stock_s1_logits"].shape[:2] == (batch_size, seq_len)
    assert outputs["index_s1_logits"].shape[:3] == (batch_size, seq_len, num_indices)
    assert outputs["fused_stock_context"].shape == (batch_size, seq_len, model.d_model)
    assert losses["stock_main_loss"].ndim == 0
    assert losses["index_aux_loss"].ndim == 0
