import argparse
import json
import os
import sys
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import CraftTokenizer, MultiStreamCraftModel
from model.craft import sample_from_craft_logits

from config_loader import CustomFinetuneConfig
from multistream_dataset import MultiStreamInferenceDataset, multistream_collate_fn


def load_checkpoint(checkpoint_dir: str, config: CustomFinetuneConfig, device: torch.device):
    if not os.path.exists(os.path.join(checkpoint_dir, "multistream_model.pt")):
        candidate_dir = os.path.join(checkpoint_dir, "best_model")
        if os.path.exists(os.path.join(candidate_dir, "multistream_model.pt")):
            checkpoint_dir = candidate_dir

    model_ckpt_path = os.path.join(checkpoint_dir, "multistream_model.pt")
    if not os.path.exists(model_ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {model_ckpt_path}")

    try:
        payload = torch.load(model_ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(model_ckpt_path, map_location=device)
    model = MultiStreamCraftModel.from_serializable_config(payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    stock_tokenizer_dir = os.path.join(checkpoint_dir, "stock_tokenizer")
    index_tokenizer_dir = os.path.join(checkpoint_dir, "index_tokenizer")
    stock_tokenizer_path = stock_tokenizer_dir if os.path.exists(stock_tokenizer_dir) else config.pretrained_tokenizer_path
    index_tokenizer_path = index_tokenizer_dir if os.path.exists(index_tokenizer_dir) else config.pretrained_tokenizer_path

    stock_tokenizer = CraftTokenizer.from_pretrained(stock_tokenizer_path).to(device).eval()
    index_tokenizer = CraftTokenizer.from_pretrained(index_tokenizer_path).to(device).eval()

    if stock_tokenizer.s1_bits != model.stock_predictor.s1_bits or stock_tokenizer.s2_bits != model.stock_predictor.s2_bits:
        raise ValueError(
            "Stock tokenizer/predictor mismatch: "
            f"tokenizer bits=({stock_tokenizer.s1_bits}, {stock_tokenizer.s2_bits}), "
            f"predictor bits=({model.stock_predictor.s1_bits}, {model.stock_predictor.s2_bits}). "
            f"Loaded stock tokenizer from: {stock_tokenizer_path}"
        )
    if index_tokenizer.s1_bits != model.index_predictor.s1_bits or index_tokenizer.s2_bits != model.index_predictor.s2_bits:
        raise ValueError(
            "Index tokenizer/predictor mismatch: "
            f"tokenizer bits=({index_tokenizer.s1_bits}, {index_tokenizer.s2_bits}), "
            f"predictor bits=({model.index_predictor.s1_bits}, {model.index_predictor.s2_bits}). "
            f"Loaded index tokenizer from: {index_tokenizer_path}"
        )

    index_ids_json = os.path.join(checkpoint_dir, "index_ids.json")
    if os.path.exists(index_ids_json):
        with open(index_ids_json, "r", encoding="utf-8") as f:
            index_labels = json.load(f)["index_ids"]
    else:
        index_labels = config.index_ids

    if index_labels is None:
        index_labels = []

    if hasattr(model, "num_indices") and model.num_indices != len(index_labels):
        raise ValueError(
            "Checkpoint index dimension mismatch: "
            f"model.num_indices={model.num_indices}, but loaded {len(index_labels)} index labels. "
            "Please make sure the checkpoint directory contains the matching `index_ids.json`, "
            "and that you are not mixing a checkpoint with a different merged indices definition."
        )

    print(
        "Loaded tokenizer/checkpoint pair: "
        f"stock_tokenizer={stock_tokenizer_path}, "
        f"index_tokenizer={index_tokenizer_path}, "
        f"model_num_indices={model.num_indices}, "
        f"loaded_index_labels={len(index_labels)}"
    )

    return model, stock_tokenizer, index_tokenizer, index_labels


def _validate_time_features(x_time_seq: torch.Tensor, y_time_seq: torch.Tensor):
    all_time = torch.cat([x_time_seq, y_time_seq], dim=1)
    minute = all_time[..., 0]
    hour = all_time[..., 1]
    weekday = all_time[..., 2]
    day = all_time[..., 3]
    month = all_time[..., 4]

    checks = [
        ("minute", minute, 0, 59),
        ("hour", hour, 0, 23),
        ("weekday", weekday, 0, 6),
        ("day", day, 1, 31),
        ("month", month, 1, 12),
    ]
    for name, tensor, lower, upper in checks:
        observed_min = int(tensor.min().item())
        observed_max = int(tensor.max().item())
        if observed_min < lower or observed_max > upper:
            raise ValueError(
                f"Temporal feature `{name}` is out of range: observed [{observed_min}, {observed_max}], "
                f"expected within [{lower}, {upper}]."
            )


def _validate_index_ids(index_ids: torch.Tensor, num_indices: int):
    observed_min = int(index_ids.min().item())
    observed_max = int(index_ids.max().item())
    if observed_min < 0 or observed_max >= num_indices:
        raise ValueError(
            "Index id tensor is out of range for the loaded checkpoint: "
            f"observed [{observed_min}, {observed_max}], allowed [0, {num_indices - 1}]."
        )


def _validate_token_range(name: str, token_tensor: torch.Tensor, vocab_size: int):
    observed_min = int(token_tensor.min().item())
    observed_max = int(token_tensor.max().item())
    if observed_min < 0 or observed_max >= vocab_size:
        raise ValueError(
            f"Tokenizer produced `{name}` ids out of range: observed [{observed_min}, {observed_max}], "
            f"allowed [0, {vocab_size - 1}]. "
            "This usually means the tokenizer/checkpoint pair is inconsistent."
        )


def _format_token_range(name: str, token_tensor: torch.Tensor) -> str:
    observed_min = int(token_tensor.min().item())
    observed_max = int(token_tensor.max().item())
    return f"{name}=[{observed_min}, {observed_max}]"


def _init_stock_buffers(stock_tokens, max_context: int):
    context_len = stock_tokens[0].size(1)
    batch_size = stock_tokens[0].size(0)
    pre_buffer = stock_tokens[0].new_zeros(batch_size, max_context)
    post_buffer = stock_tokens[1].new_zeros(batch_size, max_context)
    buffer_len = min(context_len, max_context)
    start_idx = max(0, context_len - max_context)
    pre_buffer[:, :buffer_len] = stock_tokens[0][:, start_idx : start_idx + buffer_len]
    post_buffer[:, :buffer_len] = stock_tokens[1][:, start_idx : start_idx + buffer_len]
    return pre_buffer, post_buffer


def _init_index_buffers(index_tokens, max_context: int):
    context_len = index_tokens[0].size(1)
    batch_size, _, num_indices = index_tokens[0].shape
    pre_buffer = index_tokens[0].new_zeros(batch_size, num_indices, max_context)
    post_buffer = index_tokens[1].new_zeros(batch_size, num_indices, max_context)
    buffer_len = min(context_len, max_context)
    start_idx = max(0, context_len - max_context)
    pre_buffer[:, :, :buffer_len] = index_tokens[0][:, start_idx : start_idx + buffer_len, :].permute(0, 2, 1)
    post_buffer[:, :, :buffer_len] = index_tokens[1][:, start_idx : start_idx + buffer_len, :].permute(0, 2, 1)
    return pre_buffer, post_buffer


def _append_stock_token(pre_buffer, post_buffer, current_seq_len: int, max_context: int, next_pre, next_post):
    if current_seq_len < max_context:
        pre_buffer[:, current_seq_len] = next_pre
        post_buffer[:, current_seq_len] = next_post
        return
    pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
    post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
    pre_buffer[:, -1] = next_pre
    post_buffer[:, -1] = next_post


def _append_index_token(pre_buffer, post_buffer, current_seq_len: int, max_context: int, next_pre, next_post):
    if current_seq_len < max_context:
        pre_buffer[:, :, current_seq_len] = next_pre
        post_buffer[:, :, current_seq_len] = next_post
        return
    pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=2))
    post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=2))
    pre_buffer[:, :, -1] = next_pre
    post_buffer[:, :, -1] = next_post


def autoregressive_multistream_inference(
    stock_tokenizer: CraftTokenizer,
    index_tokenizer: CraftTokenizer,
    model: MultiStreamCraftModel,
    stock_seq: torch.Tensor,
    index_seq: torch.Tensor,
    x_time_seq: torch.Tensor,
    y_time_seq: torch.Tensor,
    index_ids: torch.Tensor,
    max_context: int,
    temperature: float,
    top_k: int,
    top_p: float,
    sample_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    batch_size, context_len, feature_dim = stock_seq.shape
    _, _, num_indices, _ = index_seq.shape
    pred_len = y_time_seq.size(1)
    full_stamp = torch.cat([x_time_seq, y_time_seq], dim=1)

    _validate_time_features(x_time_seq, y_time_seq)
    _validate_index_ids(index_ids, model.num_indices)

    stock_samples = []
    index_samples = []

    with torch.no_grad():
        for _ in range(sample_count):
            stock_tokens = stock_tokenizer.encode(stock_seq, half=True)
            index_flat = index_seq.permute(0, 2, 1, 3).reshape(batch_size * num_indices, context_len, feature_dim)
            index_token_flat = index_tokenizer.encode(index_flat, half=True)
            index_tokens = (
                index_token_flat[0].reshape(batch_size, num_indices, context_len).permute(0, 2, 1).contiguous(),
                index_token_flat[1].reshape(batch_size, num_indices, context_len).permute(0, 2, 1).contiguous(),
            )

            _validate_token_range("stock_s1", stock_tokens[0], model.stock_predictor.s1_vocab_size)
            _validate_token_range("stock_s2", stock_tokens[1], model.stock_predictor.head.vocab_s2)
            _validate_token_range("index_s1", index_tokens[0], model.index_predictor.s1_vocab_size)
            _validate_token_range("index_s2", index_tokens[1], model.index_predictor.head.vocab_s2)

            generated_stock_pre = stock_tokens[0].new_zeros(batch_size, pred_len)
            generated_stock_post = stock_tokens[1].new_zeros(batch_size, pred_len)
            generated_index_pre = index_tokens[0].new_zeros(batch_size, pred_len, num_indices)
            generated_index_post = index_tokens[1].new_zeros(batch_size, pred_len, num_indices)

            stock_pre_buffer, stock_post_buffer = _init_stock_buffers(stock_tokens, max_context=max_context)
            index_pre_buffer, index_post_buffer = _init_index_buffers(index_tokens, max_context=max_context)

            for step in range(pred_len):
                current_seq_len = context_len + step
                window_len = min(current_seq_len, max_context)

                stock_input_pre = stock_pre_buffer[:, :window_len] if current_seq_len <= max_context else stock_pre_buffer
                stock_input_post = stock_post_buffer[:, :window_len] if current_seq_len <= max_context else stock_post_buffer
                index_input_pre = (
                    index_pre_buffer[:, :, :window_len] if current_seq_len <= max_context else index_pre_buffer
                )
                index_input_post = (
                    index_post_buffer[:, :, :window_len] if current_seq_len <= max_context else index_post_buffer
                )

                context_end = current_seq_len
                context_start = max(0, context_end - max_context)
                current_stamp = full_stamp[:, context_start:context_end, :].contiguous()

                try:
                    _validate_token_range(
                        f"loop_stock_s1_input_step_{step}",
                        stock_input_pre,
                        model.stock_predictor.s1_vocab_size,
                    )
                    _validate_token_range(
                        f"loop_stock_s2_input_step_{step}",
                        stock_input_post,
                        model.stock_predictor.head.vocab_s2,
                    )
                    _validate_token_range(
                        f"loop_index_s1_input_step_{step}",
                        index_input_pre,
                        model.index_predictor.s1_vocab_size,
                    )
                    _validate_token_range(
                        f"loop_index_s2_input_step_{step}",
                        index_input_post,
                        model.index_predictor.head.vocab_s2,
                    )
                except Exception as exc:
                    raise ValueError(
                        "Autoregressive token-range validation failed before model call: "
                        f"step={step}, window_len={window_len}, "
                        f"{_format_token_range('stock_s1', stock_input_pre)}, "
                        f"{_format_token_range('stock_s2', stock_input_post)}, "
                        f"{_format_token_range('index_s1', index_input_pre)}, "
                        f"{_format_token_range('index_s2', index_input_post)}"
                    ) from exc

                index_s1_logits, index_context = model.encode_index_context(
                    index_s1_ids=index_input_pre.permute(0, 2, 1).contiguous(),
                    index_s2_ids=index_input_post.permute(0, 2, 1).contiguous(),
                    stamp=current_stamp,
                    index_ids=index_ids,
                )
                stock_s1_logits, fused_stock_context, _ = model.decode_stock_s1_with_indices(
                    stock_s1_ids=stock_input_pre,
                    stock_s2_ids=stock_input_post,
                    stamp=current_stamp,
                    index_context=index_context,
                )

                next_stock_pre = sample_from_craft_logits(
                    stock_s1_logits[:, -1, :],
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    sample_logits=True,
                ).squeeze(-1)

                index_s1_last = index_s1_logits[:, -1, :, :].reshape(batch_size * num_indices, -1)
                next_index_pre = sample_from_craft_logits(
                    index_s1_last,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    sample_logits=True,
                ).view(batch_size, num_indices)

                stock_s2_logits = model.decode_stock_s2(fused_stock_context, next_stock_pre.unsqueeze(1))
                next_stock_post = sample_from_craft_logits(
                    stock_s2_logits[:, -1, :],
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    sample_logits=True,
                ).squeeze(-1)

                index_s2_logits = model.decode_index_s2(index_context, next_index_pre.unsqueeze(1))
                index_s2_last = index_s2_logits[:, -1, :, :].reshape(batch_size * num_indices, -1)
                next_index_post = sample_from_craft_logits(
                    index_s2_last,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    sample_logits=True,
                ).view(batch_size, num_indices)

                try:
                    _validate_token_range(
                        f"sampled_stock_s1_step_{step}",
                        next_stock_pre,
                        model.stock_predictor.s1_vocab_size,
                    )
                    _validate_token_range(
                        f"sampled_stock_s2_step_{step}",
                        next_stock_post,
                        model.stock_predictor.head.vocab_s2,
                    )
                    _validate_token_range(
                        f"sampled_index_s1_step_{step}",
                        next_index_pre,
                        model.index_predictor.s1_vocab_size,
                    )
                    _validate_token_range(
                        f"sampled_index_s2_step_{step}",
                        next_index_post,
                        model.index_predictor.head.vocab_s2,
                    )
                except Exception as exc:
                    raise ValueError(
                        "Autoregressive token-range validation failed after sampling: "
                        f"step={step}, "
                        f"{_format_token_range('next_stock_s1', next_stock_pre)}, "
                        f"{_format_token_range('next_stock_s2', next_stock_post)}, "
                        f"{_format_token_range('next_index_s1', next_index_pre)}, "
                        f"{_format_token_range('next_index_s2', next_index_post)}"
                    ) from exc

                generated_stock_pre[:, step] = next_stock_pre
                generated_stock_post[:, step] = next_stock_post
                generated_index_pre[:, step, :] = next_index_pre
                generated_index_post[:, step, :] = next_index_post

                _append_stock_token(stock_pre_buffer, stock_post_buffer, current_seq_len, max_context, next_stock_pre, next_stock_post)
                _append_index_token(index_pre_buffer, index_post_buffer, current_seq_len, max_context, next_index_pre, next_index_post)

            total_seq_len = context_len + pred_len
            decode_start = max(0, total_seq_len - max_context)

            full_stock_pre = torch.cat([stock_tokens[0], generated_stock_pre], dim=1)
            full_stock_post = torch.cat([stock_tokens[1], generated_stock_post], dim=1)
            decoded_stock = stock_tokenizer.decode(
                [full_stock_pre[:, decode_start:total_seq_len], full_stock_post[:, decode_start:total_seq_len]],
                half=True,
            )[:, -pred_len:, :]

            full_index_pre = torch.cat([index_tokens[0], generated_index_pre], dim=1)
            full_index_post = torch.cat([index_tokens[1], generated_index_post], dim=1)
            decode_index_pre = (
                full_index_pre[:, decode_start:total_seq_len, :].permute(0, 2, 1).reshape(batch_size * num_indices, -1)
            )
            decode_index_post = (
                full_index_post[:, decode_start:total_seq_len, :].permute(0, 2, 1).reshape(batch_size * num_indices, -1)
            )
            decoded_index = index_tokenizer.decode([decode_index_pre, decode_index_post], half=True)
            decoded_index = decoded_index.view(batch_size, num_indices, -1, feature_dim).permute(0, 2, 1, 3)
            decoded_index = decoded_index[:, -pred_len:, :, :]

            stock_samples.append(decoded_stock.cpu().numpy())
            index_samples.append(decoded_index.cpu().numpy())

    stock_preds = np.mean(stock_samples, axis=0)
    index_preds = np.mean(index_samples, axis=0)
    return stock_preds, index_preds


def _build_output_frames(
    stock_preds: np.ndarray,
    index_preds: Optional[np.ndarray],
    batch,
    feature_cols: List[str],
    index_labels: List[str],
    output_index_aux: bool,
) -> List[Tuple[str, pd.DataFrame, Optional[pd.DataFrame]]]:
    results = []
    stock_mean = batch["stock_mean"].cpu().numpy()
    stock_std = batch["stock_std"].cpu().numpy()
    index_mean = batch["index_mean"].cpu().numpy()
    index_std = batch["index_std"].cpu().numpy()

    for idx, stock_id in enumerate(batch["stock_id"]):
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

        results.append((stock_id, stock_df, index_df))
    return results


def main():
    parser = argparse.ArgumentParser(description="Run multistream Craft inference for one or more A-share stock csv files.")
    parser.add_argument("--config", type=str, required=True, help="Path to the multistream yaml config.")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Checkpoint directory. You can pass either the best_model directory or its parent save directory.",
    )
    parser.add_argument("--stock-file", type=str, default=None, help="Single stock csv file to predict.")
    parser.add_argument("--stock-dir", type=str, default=None, help="Directory of stock csv files for batch prediction.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory where prediction files will be written.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size used during batch inference.")
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

    if bool(args.stock_file) == bool(args.stock_dir):
        raise ValueError("Please provide exactly one of `--stock-file` or `--stock-dir`.")

    config = CustomFinetuneConfig(args.config)
    device = torch.device("cuda" if config.use_cuda and torch.cuda.is_available() else "cpu")
    model, stock_tokenizer, index_tokenizer, index_labels = load_checkpoint(args.checkpoint_dir, config, device)

    if args.stock_file is not None:
        stock_paths = [args.stock_file]
    else:
        stock_paths = sorted(glob(os.path.join(args.stock_dir, "*.csv")))
        if not stock_paths:
            raise FileNotFoundError(f"No stock csv files found under {args.stock_dir}")

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
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=multistream_collate_fn,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for batch in loader:
        stock_seq = batch["stock_seq"].to(device, non_blocking=True)
        index_seq = batch["index_seq"].to(device, non_blocking=True)
        x_time_seq = batch["x_time_seq"].to(device, non_blocking=True)
        y_time_seq = batch["y_time_seq"].to(device, non_blocking=True)
        index_ids = batch["index_ids"].to(device, non_blocking=True)

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

        output_frames = _build_output_frames(
            stock_preds=stock_preds,
            index_preds=index_preds,
            batch=batch,
            feature_cols=config.feature_cols,
            index_labels=index_labels,
            output_index_aux=config.output_index_aux,
        )

        for item_idx, (stock_id, stock_df, index_df) in enumerate(output_frames):
            stock_out = os.path.join(args.output_dir, f"{stock_id}_stock_prediction.csv")
            stock_df.to_csv(stock_out, index=False)
            lookback_start = pd.to_datetime(batch["lookback_start_date"][item_idx].cpu().numpy())
            lookback_end = pd.to_datetime(batch["lookback_end_date"][item_idx].cpu().numpy())
            horizon_start = pd.to_datetime(batch["horizon_start_date"][item_idx].cpu().numpy())
            horizon_end = pd.to_datetime(batch["horizon_end_date"][item_idx].cpu().numpy())
            print(
                f"Saved stock prediction: {stock_out} | "
                f"lookback: {lookback_start.date()} -> {lookback_end.date()} | "
                f"horizon: {horizon_start.date()} -> {horizon_end.date()}"
            )

            if index_df is not None:
                index_out = os.path.join(args.output_dir, f"{stock_id}_index_aux_prediction.csv")
                index_df.to_csv(index_out, index=False)
                print(f"Saved index auxiliary prediction: {index_out}")


if __name__ == "__main__":
    main()
