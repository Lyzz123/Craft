import argparse
import os
import sys
from glob import glob
from typing import List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config_loader import CustomFinetuneConfig
from multistream_dataset import MultiStreamInferenceDataset, multistream_collate_fn
from predict_multistream import autoregressive_multistream_inference, load_checkpoint


def resolve_checkpoint_dir(checkpoint_dir: Optional[str], config: CustomFinetuneConfig) -> str:
    candidates = []
    if checkpoint_dir:
        candidates.append(checkpoint_dir)
    if config.basemodel_best_model_path:
        candidates.append(config.basemodel_best_model_path)
    if config.basemodel_save_path:
        candidates.append(config.basemodel_save_path)

    checked = []
    for candidate in candidates:
        if not candidate:
            continue
        normalized = os.path.abspath(candidate)
        if normalized in checked:
            continue
        checked.append(normalized)
        if os.path.isdir(normalized):
            return normalized

    searched = ", ".join(checked) if checked else "none"
    raise FileNotFoundError(
        "Could not locate a valid checkpoint directory. "
        f"Checked: {searched}. "
        "Please pass --checkpoint-dir explicitly or make sure the training config points to a saved best model."
    )


def collect_stock_paths(input_dir: str, pattern: str = "*.csv") -> List[str]:
    stock_paths = sorted(glob(os.path.join(input_dir, pattern)))
    if not stock_paths:
        raise FileNotFoundError(f"No stock csv files found under {input_dir} with pattern `{pattern}`.")
    return stock_paths


def build_prediction_frames(
    batch,
    stock_preds,
    index_preds,
    feature_cols: List[str],
    index_labels: List[str],
    output_index_aux: bool,
) -> List[Tuple[str, pd.DataFrame, Optional[pd.DataFrame]]]:
    results = []
    stock_mean = batch["stock_mean"].cpu().numpy()
    stock_std = batch["stock_std"].cpu().numpy()
    index_mean = batch["index_mean"].cpu().numpy()
    index_std = batch["index_std"].cpu().numpy()

    for idx, stock_path in enumerate(batch["stock_path"]):
        future_dates = pd.to_datetime(batch["future_dates"][idx].cpu().numpy())
        stock_denorm = stock_preds[idx] * (stock_std[idx][None, :] + 1e-5) + stock_mean[idx][None, :]
        stock_df = pd.DataFrame(stock_denorm, columns=feature_cols)
        stock_df.insert(0, "date", future_dates)

        index_df = None
        if output_index_aux and index_preds is not None:
            index_denorm = index_preds[idx] * (index_std[idx][None, :, :] + 1e-5) + index_mean[idx][None, :, :]
            rows = []
            for t, date in enumerate(future_dates):
                for index_pos, index_id in enumerate(index_labels):
                    row = {"date": date, "index_id": index_id}
                    for feature_idx, feature_name in enumerate(feature_cols):
                        row[feature_name] = index_denorm[t, index_pos, feature_idx]
                    rows.append(row)
            index_df = pd.DataFrame(rows)

        results.append((stock_path, stock_df, index_df))
    return results


def resolve_output_paths(output_dir: str, stock_path: str) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    source_name = os.path.basename(stock_path)
    stem, ext = os.path.splitext(source_name)
    ext = ext if ext else ".csv"

    stock_out = os.path.join(output_dir, source_name)
    if os.path.abspath(stock_out) == os.path.abspath(stock_path):
        stock_out = os.path.join(output_dir, f"{stem}_prediction{ext}")

    index_out = os.path.join(output_dir, f"{stem}_index_aux_prediction{ext}")
    return stock_out, index_out


def main():
    parser = argparse.ArgumentParser(
        description="Batch multistream inference for a folder of stock csv files using the training config."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to the multistream yaml config.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing stock csv files. Defaults to data.stock_dir from the config.",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Directory where prediction csv files are written.")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Optional checkpoint directory. Defaults to the best model path derived from the training config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Inference batch size. Defaults to training.batch_size from the config.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern for stock files inside the input directory.",
    )
    parser.add_argument(
        "--output-index-aux",
        action="store_true",
        help="Also save index auxiliary predictions, overriding the config if needed.",
    )
    parser.add_argument(
        "--horizon-start-date",
        type=str,
        default=None,
        help=(
            "Optional prediction horizon start date (YYYY-MM-DD). "
            "The dataset will automatically backtrack the previous lookback aligned rows before this date."
        ),
    )
    parser.add_argument("--lookback-start-date", dest="horizon_start_date", help=argparse.SUPPRESS)
    args = parser.parse_args()

    config = CustomFinetuneConfig(args.config)
    input_dir = args.input_dir or config.stock_dir
    if not input_dir:
        raise ValueError("No input directory was provided, and `data.stock_dir` is empty in the config.")

    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_dir, config)
    batch_size = args.batch_size or config.batch_size
    output_index_aux = bool(args.output_index_aux or config.output_index_aux)

    device = torch.device("cuda" if config.use_cuda and torch.cuda.is_available() else "cpu")
    stock_paths = collect_stock_paths(input_dir, pattern=args.pattern)
    model, stock_tokenizer, index_tokenizer, index_labels = load_checkpoint(checkpoint_dir, config, device)

    print(f"Using device: {device}")
    print(f"Using checkpoint: {checkpoint_dir}")
    print(
        f"Config-aligned inference parameters: lookback={config.lookback}, "
        f"horizon={config.horizon}, max_context={config.max_context}, batch_size={batch_size}"
    )
    if args.horizon_start_date is not None:
        print(
            f"Requested horizon start date: {args.horizon_start_date} "
            "(the dataset will backtrack the previous aligned lookback rows before this date)"
        )
    print(f"Discovered {len(stock_paths)} stock csv files under {input_dir}")

    dataset = MultiStreamInferenceDataset(
        stock_paths=stock_paths,
        merged_indices_path=config.merged_indices_path,
        lookback=config.lookback,
        horizon=config.horizon,
        clip=config.clip,
        stock_date_col=config.stock_date_col,
        index_date_col=config.index_date_col,
        feature_cols=config.feature_cols,
        index_ids=index_labels,
        horizon_start_date=args.horizon_start_date,
    )
    if len(dataset) == 0:
        if args.horizon_start_date is not None:
            raise ValueError(
                "No inference samples were constructed after aligning stocks with merged indices. "
                "The requested horizon start date may be too early, or some stocks may not have enough aligned history before it."
            )
        raise ValueError("No inference samples were constructed after aligning stocks with merged indices.")
    print(
        f"Final usable inference samples after stock/index alignment: {len(dataset)} "
        f"(skipped {len(stock_paths) - len(dataset)} files)"
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=multistream_collate_fn,
    )

    processed = 0
    for batch_idx, batch in enumerate(loader, start=1):
        print(f"Inference batch {batch_idx}/{len(loader)}")
        stock_seq = batch["stock_seq"].to(device, non_blocking=True)
        index_seq = batch["index_seq"].to(device, non_blocking=True)
        x_time_seq = batch["x_time_seq"].to(device, non_blocking=True)
        y_time_seq = batch["y_time_seq"].to(device, non_blocking=True)
        index_ids = batch["index_ids"].to(device, non_blocking=True)

        try:
            stock_preds, index_preds = autoregressive_multistream_inference(
                stock_tokenizer=stock_tokenizer,
                index_tokenizer=index_tokenizer,
                model=model,
                stock_seq=stock_seq,
                index_seq=index_seq,
                x_time_seq=x_time_seq,
                y_time_seq=y_time_seq,
                index_ids=index_ids,
                max_context=config.max_context,
                temperature=config.inference_temperature,
                top_k=config.inference_top_k,
                top_p=config.inference_top_p,
                sample_count=config.inference_sample_count,
            )
        except Exception as exc:
            batch_preview = batch["stock_path"][:5]
            raise RuntimeError(
                "Inference failed while processing a batch. "
                f"batch_idx={batch_idx}, batch_size={len(batch['stock_path'])}, "
                f"sample_preview={batch_preview}"
            ) from exc

        output_frames = build_prediction_frames(
            batch=batch,
            stock_preds=stock_preds,
            index_preds=index_preds,
            feature_cols=config.feature_cols,
            index_labels=index_labels,
            output_index_aux=output_index_aux,
        )

        for item_idx, (stock_path, stock_df, index_df) in enumerate(output_frames):
            stock_out, index_out = resolve_output_paths(args.output_dir, stock_path)
            stock_df.to_csv(stock_out, index=False)
            actual_start = pd.to_datetime(batch["lookback_start_date"][item_idx].cpu().numpy())
            actual_end = pd.to_datetime(batch["lookback_end_date"][item_idx].cpu().numpy())
            horizon_start = pd.to_datetime(batch["horizon_start_date"][item_idx].cpu().numpy())
            horizon_end = pd.to_datetime(batch["horizon_end_date"][item_idx].cpu().numpy())
            print(
                f"Saved stock prediction: {stock_out} | "
                f"lookback window: {actual_start.date()} -> {actual_end.date()} | "
                f"horizon window: {horizon_start.date()} -> {horizon_end.date()}"
            )

            if index_df is not None:
                index_df.to_csv(index_out, index=False)
                print(f"Saved index auxiliary prediction: {index_out}")

            processed += 1

    print(f"Completed inference for {processed} stock csv files.")


if __name__ == "__main__":
    main()
