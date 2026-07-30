import argparse
import json
import logging
import math
import os
import random
import re
import sys
import time
from contextlib import nullcontext
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import CraftTokenizer, MultiStreamCraftModel, build_craft_backbone

from config_loader import CustomFinetuneConfig
from finetune_base_model import setup_logging
from multistream_dataset import create_multistream_dataloaders, load_index_metadata, load_merged_indices_panel
from token_cache import build_cached_tokenized_dataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_progress_percent(progress: float, total_steps: int) -> str:
    if total_steps >= 10000:
        return f"{progress * 100:7.3f}%"
    return f"{progress * 100:6.2f}%"


def build_progress_bar(progress: float, width: int = 24) -> str:
    progress = min(1.0, max(0.0, progress))
    if width <= 0:
        return "[]"
    if progress >= 1.0:
        return "[" + "=" * width + "]"

    filled = int(progress * width)
    head = "=" if filled == width else ">"
    if filled <= 0:
        bar = head + "." * (width - 1)
    else:
        remaining = max(0, width - filled - 1)
        bar = "=" * filled + head + "." * remaining
    return "[" + bar[:width] + "]"


def write_progress_line(message: str, previous_width: int = 0) -> int:
    padded = message.ljust(previous_width)
    sys.stdout.write("\r" + padded)
    sys.stdout.flush()
    return max(previous_width, len(message))


def clear_progress_line(previous_width: int):
    if previous_width <= 0:
        return
    sys.stdout.write("\r" + (" " * previous_width) + "\r")
    sys.stdout.flush()


def count_training_batches(dataset_len: int, batch_size: int, drop_last: bool) -> int:
    if dataset_len <= 0:
        return 0
    if drop_last:
        return dataset_len // max(1, batch_size)
    return math.ceil(dataset_len / max(1, batch_size))


def find_latest_epoch_checkpoint(save_dir: str) -> Optional[str]:
    if not os.path.isdir(save_dir):
        return None
    latest_epoch = -1
    latest_path = None
    for entry in os.listdir(save_dir):
        match = re.fullmatch(r"epoch_(\d+)", entry)
        if not match:
            continue
        epoch_num = int(match.group(1))
        entry_path = os.path.join(save_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        model_ckpt_path = os.path.join(entry_path, "multistream_model.pt")
        if not os.path.exists(model_ckpt_path):
            continue
        if epoch_num > latest_epoch:
            latest_epoch = epoch_num
            latest_path = entry_path
    return latest_path


def build_epoch_sample_indices(
    dataset_len: int,
    batch_size: int,
    drop_last: bool,
    seed: int,
    epoch: int,
) -> Sequence[int]:
    if dataset_len <= 0:
        return []

    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(epoch))
    indices = torch.randperm(dataset_len, generator=generator).tolist()

    if drop_last:
        effective_len = (len(indices) // max(1, batch_size)) * max(1, batch_size)
        indices = indices[:effective_len]
    return indices


def build_epoch_train_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    collate_fn,
    seed: int,
    epoch: int,
    start_batch_idx: int,
    drop_last: bool,
) -> Tuple[DataLoader, int]:
    epoch_indices = build_epoch_sample_indices(
        dataset_len=len(dataset),
        batch_size=batch_size,
        drop_last=drop_last,
        seed=seed,
        epoch=epoch,
    )
    total_batches = count_training_batches(len(epoch_indices), batch_size, drop_last=False)
    start_sample_idx = start_batch_idx * max(1, batch_size)
    remaining_indices = epoch_indices[start_sample_idx:]
    epoch_subset = Subset(dataset, remaining_indices)
    loader = DataLoader(
        epoch_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return loader, total_batches


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[Dict[str, Any]]):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _named_trainable_parameters(
    model: MultiStreamCraftModel,
    stock_tokenizer: CraftTokenizer,
    index_tokenizer: CraftTokenizer,
) -> Dict[str, torch.nn.Parameter]:
    named_params: Dict[str, torch.nn.Parameter] = {}
    for prefix, module in (
        ("model", model),
        ("stock_tokenizer", stock_tokenizer),
        ("index_tokenizer", index_tokenizer),
    ):
        for name, param in module.named_parameters():
            named_params[f"{prefix}.{name}"] = param
    return named_params


def capture_gradient_state(
    model: MultiStreamCraftModel,
    stock_tokenizer: CraftTokenizer,
    index_tokenizer: CraftTokenizer,
) -> Dict[str, torch.Tensor]:
    gradient_state: Dict[str, torch.Tensor] = {}
    for name, param in _named_trainable_parameters(model, stock_tokenizer, index_tokenizer).items():
        if param.grad is not None:
            gradient_state[name] = param.grad.detach().cpu()
    return gradient_state


def restore_gradient_state(
    gradient_state: Optional[Dict[str, torch.Tensor]],
    model: MultiStreamCraftModel,
    stock_tokenizer: CraftTokenizer,
    index_tokenizer: CraftTokenizer,
):
    named_params = _named_trainable_parameters(model, stock_tokenizer, index_tokenizer)
    for name, param in named_params.items():
        if gradient_state is not None and name in gradient_state:
            param.grad = gradient_state[name].to(device=param.device, dtype=param.dtype)
        else:
            param.grad = None


def log_file_only(logger, message: str):
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        fn=__file__,
        lno=0,
        msg=message,
        args=(),
        exc_info=None,
    )
    for handler in logger.handlers:
        if getattr(handler, "baseFilename", None):
            handler.handle(record)


def format_progress_message(
    stage: str,
    epoch: int,
    total_epochs: int,
    step: int,
    total_steps: int,
    stock_loss: float,
    index_loss: float,
    total_loss: float,
    elapsed: float,
    remaining: float,
    extra: str = "",
) -> str:
    progress = step / max(1, total_steps)
    bar = build_progress_bar(progress)
    base = (
        f"{stage} E{epoch}/{total_epochs} {bar} {format_progress_percent(progress, total_steps)} "
        f"({step}/{total_steps}) | Stock {stock_loss:.4f} | Index {index_loss:.4f} | "
        f"Total {total_loss:.4f} | Elapsed {format_duration(elapsed)} | ETA {format_duration(remaining)}"
    )
    if extra:
        base += f" | {extra}"
    return base


def should_refresh_progress(
    step: int,
    total_steps: int,
    refresh_interval: int,
    last_refresh_time: float,
    now: float,
    min_refresh_seconds: float = 2.0,
) -> bool:
    if total_steps <= 0:
        return False
    if step <= 1 or step >= total_steps:
        return True
    if step % max(1, refresh_interval) == 0:
        return True
    return (now - last_refresh_time) >= min_refresh_seconds


def build_optimizer(config: CustomFinetuneConfig, model: MultiStreamCraftModel, stock_tokenizer, index_tokenizer):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_params.extend([p for p in stock_tokenizer.parameters() if p.requires_grad])
    trainable_params.extend([p for p in index_tokenizer.parameters() if p.requires_grad])

    optimizer_name = config.optimizer_name.lower()
    if optimizer_name != "adamw":
        raise NotImplementedError(f"Unsupported optimizer `{config.optimizer_name}`. Only AdamW is implemented.")

    return torch.optim.AdamW(
        trainable_params,
        lr=config.predictor_learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
    )


def resolve_mixed_precision_mode(config: CustomFinetuneConfig, device: torch.device, logger=None) -> str:
    aliases = {
        "none": "none",
        "off": "none",
        "false": "none",
        "fp32": "none",
        "float32": "none",
        "bf16": "bf16",
        "bfloat16": "bf16",
        "fp16": "fp16",
        "float16": "fp16",
    }
    raw_mode = str(getattr(config, "mixed_precision", "none")).strip().lower()
    if raw_mode not in aliases:
        raise ValueError(
            f"Unsupported mixed_precision `{config.mixed_precision}`. "
            "Please choose one of: none, bf16, fp16."
        )

    mode = aliases[raw_mode]
    if device.type != "cuda":
        if mode != "none":
            message = f"mixed_precision={mode} requested, but device is `{device.type}`. Falling back to `none`."
            if logger is not None:
                logger.warning(message)
            print(message)
        return "none"

    if mode == "bf16":
        bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        if not bf16_supported:
            message = "BF16 mixed precision is not supported on this CUDA setup. Falling back to `none`."
            if logger is not None:
                logger.warning(message)
            print(message)
            return "none"
    return mode


def get_autocast_context(mixed_precision_mode: str, device: torch.device):
    if mixed_precision_mode == "none":
        return nullcontext()
    if mixed_precision_mode == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    if mixed_precision_mode == "fp16":
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    raise ValueError(f"Unsupported mixed precision mode `{mixed_precision_mode}`.")


def build_scheduler(config: CustomFinetuneConfig, optimizer, steps_per_epoch: int):
    scheduler_name = config.scheduler_name.lower()
    if scheduler_name == "none":
        return None
    if scheduler_name == "onecycle":
        total_steps_per_epoch = max(1, steps_per_epoch)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.predictor_learning_rate,
            steps_per_epoch=total_steps_per_epoch,
            epochs=config.basemodel_epochs,
            pct_start=config.scheduler_config.get("pct_start", 0.03),
            div_factor=config.scheduler_config.get("div_factor", 10.0),
        )
    if scheduler_name == "cosine":
        total_steps = max(1, steps_per_epoch * config.basemodel_epochs)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    raise NotImplementedError(f"Unsupported scheduler `{config.scheduler_name}`.")


def set_tokenizer_trainable(tokenizer: CraftTokenizer, trainable: bool):
    tokenizer.train(mode=trainable)
    for param in tokenizer.parameters():
        param.requires_grad = trainable
    if not trainable:
        tokenizer.eval()


def encode_batch_tokens(
    stock_tokenizer: CraftTokenizer,
    index_tokenizer: CraftTokenizer,
    stock_seq: torch.Tensor,
    index_seq: torch.Tensor,
    freeze_stock_tokenizer: bool,
    freeze_index_tokenizer: bool,
) -> Dict[str, torch.Tensor]:
    stock_ctx = torch.no_grad() if freeze_stock_tokenizer else nullcontext()
    index_ctx = torch.no_grad() if freeze_index_tokenizer else nullcontext()

    with stock_ctx:
        stock_s1, stock_s2 = stock_tokenizer.encode(stock_seq, half=True)

    batch_size, seq_len, num_indices, feature_dim = index_seq.shape
    index_flat = index_seq.permute(0, 2, 1, 3).reshape(batch_size * num_indices, seq_len, feature_dim)
    with index_ctx:
        index_s1, index_s2 = index_tokenizer.encode(index_flat, half=True)

    index_s1 = index_s1.reshape(batch_size, num_indices, seq_len).permute(0, 2, 1).contiguous()
    index_s2 = index_s2.reshape(batch_size, num_indices, seq_len).permute(0, 2, 1).contiguous()

    return {
        "stock_s1": stock_s1,
        "stock_s2": stock_s2,
        "index_s1": index_s1,
        "index_s2": index_s2,
    }


def get_batch_tokens(
    batch: Dict[str, torch.Tensor],
    stock_tokenizer: CraftTokenizer,
    index_tokenizer: CraftTokenizer,
    device: torch.device,
    freeze_stock_tokenizer: bool,
    freeze_index_tokenizer: bool,
) -> Dict[str, torch.Tensor]:
    cached_keys = {"stock_s1", "stock_s2", "index_s1", "index_s2"}
    if cached_keys.issubset(batch.keys()):
        return {
            "stock_s1": batch["stock_s1"].to(device, non_blocking=True),
            "stock_s2": batch["stock_s2"].to(device, non_blocking=True),
            "index_s1": batch["index_s1"].to(device, non_blocking=True),
            "index_s2": batch["index_s2"].to(device, non_blocking=True),
        }

    stock_seq = batch["stock_seq"].to(device, non_blocking=True)
    index_seq = batch["index_seq"].to(device, non_blocking=True)
    return encode_batch_tokens(
        stock_tokenizer=stock_tokenizer,
        index_tokenizer=index_tokenizer,
        stock_seq=stock_seq,
        index_seq=index_seq,
        freeze_stock_tokenizer=freeze_stock_tokenizer,
        freeze_index_tokenizer=freeze_index_tokenizer,
    )


def maybe_build_cached_loaders(
    config: CustomFinetuneConfig,
    train_loader,
    val_loader,
    train_dataset,
    val_dataset,
    stock_tokenizer: CraftTokenizer,
    index_tokenizer: CraftTokenizer,
    device: torch.device,
    logger,
):
    if not config.cache_tokenized_sequences:
        return train_loader, val_loader
    if not config.freeze_stock_tokenizer or not config.freeze_index_tokenizer:
        raise ValueError("Token caching requires both stock and index tokenizers to be frozen.")

    train_cached_dataset = build_cached_tokenized_dataset(
        dataset=train_dataset,
        stock_tokenizer=stock_tokenizer,
        index_tokenizer=index_tokenizer,
        device=device,
        batch_size=config.token_cache_batch_size,
        num_workers=config.token_cache_num_workers,
        logger=logger,
        split_name="train",
    )
    val_cached_dataset = build_cached_tokenized_dataset(
        dataset=val_dataset,
        stock_tokenizer=stock_tokenizer,
        index_tokenizer=index_tokenizer,
        device=device,
        batch_size=config.token_cache_batch_size,
        num_workers=config.token_cache_num_workers,
        logger=logger,
        split_name="val",
    )

    train_loader = torch.utils.data.DataLoader(
        train_cached_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=len(train_cached_dataset) >= config.batch_size,
        collate_fn=None,
    )
    val_loader = torch.utils.data.DataLoader(
        val_cached_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=None,
    )

    stock_tokenizer.cpu()
    index_tokenizer.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    cache_msg = (
        f"Tokenizer outputs cached in memory for training and validation. "
        f"Train samples: {len(train_cached_dataset)}, Val samples: {len(val_cached_dataset)}. "
        "Frozen tokenizers moved to CPU to free GPU memory."
    )
    logger.info(cache_msg)
    print(cache_msg)
    return train_loader, val_loader


def prepare_shifted_tokens(tokens: Dict[str, torch.Tensor]) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    stock_inputs = (tokens["stock_s1"][:, :-1], tokens["stock_s2"][:, :-1])
    stock_targets = (tokens["stock_s1"][:, 1:], tokens["stock_s2"][:, 1:])
    index_inputs = (tokens["index_s1"][:, :-1], tokens["index_s2"][:, :-1])
    index_targets = (tokens["index_s1"][:, 1:], tokens["index_s2"][:, 1:])
    return (stock_inputs, stock_targets), (index_inputs, index_targets)


def compute_batch_losses(
    model: MultiStreamCraftModel,
    stock_inputs,
    stock_targets,
    index_inputs,
    index_targets,
    time_seq,
    index_ids,
    alpha: float,
    use_sampled_s1_for_s2: bool,
) -> Dict[str, torch.Tensor]:
    outputs = model(
        stock_s1_ids=stock_inputs[0],
        stock_s2_ids=stock_inputs[1],
        index_s1_ids=index_inputs[0],
        index_s2_ids=index_inputs[1],
        stamp=time_seq[:, :-1, :],
        index_ids=index_ids,
        stock_s1_targets=stock_targets[0],
        index_s1_targets=index_targets[0],
        use_sampled_s1_for_s2=use_sampled_s1_for_s2,
    )
    losses = model.compute_losses(outputs, stock_targets=stock_targets, index_targets=index_targets)
    losses["total_loss"] = losses["stock_main_loss"] + alpha * losses["index_aux_loss"]
    return losses


def save_checkpoint_bundle(
    checkpoint_dir: str,
    model: MultiStreamCraftModel,
    stock_tokenizer: CraftTokenizer,
    index_tokenizer: CraftTokenizer,
    config: CustomFinetuneConfig,
    index_labels,
    best_val_loss: float,
    epoch: int = None,
    val_total_loss: float = None,
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.get_serializable_config(),
            "best_val_loss": best_val_loss,
            "epoch": epoch,
            "epoch_val_total_loss": val_total_loss,
        },
        os.path.join(checkpoint_dir, "multistream_model.pt"),
    )

    stock_tokenizer.save_pretrained(os.path.join(checkpoint_dir, "stock_tokenizer"))
    index_tokenizer.save_pretrained(os.path.join(checkpoint_dir, "index_tokenizer"))

    with open(os.path.join(checkpoint_dir, "index_ids.json"), "w", encoding="utf-8") as f:
        json.dump({"index_ids": list(index_labels)}, f, ensure_ascii=True, indent=2)

    with open(os.path.join(checkpoint_dir, "source_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "pretrained_tokenizer_path": config.pretrained_tokenizer_path,
                "pretrained_predictor_path": config.pretrained_predictor_path,
                "pre_trained_predictor": config.pre_trained_predictor,
                "freeze_stock_tokenizer": config.freeze_stock_tokenizer,
                "freeze_index_tokenizer": config.freeze_index_tokenizer,
                "mixed_precision": config.mixed_precision,
                "cache_tokenized_sequences": config.cache_tokenized_sequences,
                "token_cache_batch_size": config.token_cache_batch_size,
                "feature_cols": config.feature_cols,
                "lookback": config.lookback,
                "horizon": config.horizon,
                "lag_window": config.lag_window,
                "alpha": config.alpha,
                "use_sampled_s1_for_s2": config.use_sampled_s1_for_s2,
                "epoch": epoch,
                "epoch_val_total_loss": val_total_loss,
            },
            f,
            ensure_ascii=True,
            indent=2,
        )


def save_best_checkpoint(
    save_dir: str,
    model: MultiStreamCraftModel,
    stock_tokenizer: CraftTokenizer,
    index_tokenizer: CraftTokenizer,
    config: CustomFinetuneConfig,
    index_labels,
    best_val_loss: float,
    epoch: int = None,
):
    best_dir = os.path.join(save_dir, "best_model")
    save_checkpoint_bundle(
        checkpoint_dir=best_dir,
        model=model,
        stock_tokenizer=stock_tokenizer,
        index_tokenizer=index_tokenizer,
        config=config,
        index_labels=index_labels,
        best_val_loss=best_val_loss,
        epoch=epoch,
        val_total_loss=best_val_loss,
    )


def save_epoch_checkpoint(
    save_dir: str,
    epoch: int,
    model: MultiStreamCraftModel,
    stock_tokenizer: CraftTokenizer,
    index_tokenizer: CraftTokenizer,
    config: CustomFinetuneConfig,
    index_labels,
    best_val_loss: float,
    val_total_loss: float,
    optimizer=None,
    scheduler=None,
    scaler: Optional[GradScaler] = None,
    global_step: Optional[int] = None,
    training_elapsed_time: Optional[float] = None,
):
    epoch_dir = os.path.join(save_dir, f"epoch_{epoch:03d}")
    save_checkpoint_bundle(
        checkpoint_dir=epoch_dir,
        model=model,
        stock_tokenizer=stock_tokenizer,
        index_tokenizer=index_tokenizer,
        config=config,
        index_labels=index_labels,
        best_val_loss=best_val_loss,
        epoch=epoch,
        val_total_loss=val_total_loss,
    )
    if optimizer is not None and scaler is not None and global_step is not None and training_elapsed_time is not None:
        save_training_state_checkpoint(
            checkpoint_dir=epoch_dir,
            model=model,
            stock_tokenizer=stock_tokenizer,
            index_tokenizer=index_tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            index_labels=index_labels,
            best_val_loss=best_val_loss,
            resume_epoch=epoch,
            next_batch_idx=0,
            global_step=global_step,
            training_elapsed_time=training_elapsed_time,
            current_val_total_loss=val_total_loss,
            checkpoint_epoch=epoch,
        )


def save_training_state_checkpoint(
    checkpoint_dir: str,
    model: MultiStreamCraftModel,
    stock_tokenizer: CraftTokenizer,
    index_tokenizer: CraftTokenizer,
    optimizer,
    scheduler,
    scaler: GradScaler,
    config: CustomFinetuneConfig,
    index_labels,
    best_val_loss: float,
    resume_epoch: int,
    next_batch_idx: int,
    global_step: int,
    training_elapsed_time: float,
    current_val_total_loss: Optional[float] = None,
    checkpoint_epoch: Optional[int] = None,
):
    if checkpoint_epoch is None:
        checkpoint_epoch = resume_epoch
    save_checkpoint_bundle(
        checkpoint_dir=checkpoint_dir,
        model=model,
        stock_tokenizer=stock_tokenizer,
        index_tokenizer=index_tokenizer,
        config=config,
        index_labels=index_labels,
        best_val_loss=best_val_loss,
        epoch=checkpoint_epoch,
        val_total_loss=current_val_total_loss,
    )

    training_state = {
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
        "rng_state": capture_rng_state(),
        "gradient_state": capture_gradient_state(model, stock_tokenizer, index_tokenizer),
        "epoch": resume_epoch,
        "next_batch_idx": next_batch_idx,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "training_elapsed_time": training_elapsed_time,
        "current_val_total_loss": current_val_total_loss,
    }
    torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))


def load_training_state_checkpoint(
    checkpoint_dir: str,
    optimizer,
    scheduler,
    scaler: GradScaler,
    model: MultiStreamCraftModel,
    stock_tokenizer: CraftTokenizer,
    index_tokenizer: CraftTokenizer,
    device: torch.device,
) -> Dict[str, Any]:
    training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
    if not os.path.exists(training_state_path):
        raise FileNotFoundError(
            f"Training state file not found: {training_state_path}. "
            "Please resume from a full training-state checkpoint such as `latest_resume`."
        )

    try:
        payload = torch.load(training_state_path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(training_state_path, map_location=device)

    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None:
        scheduler_state = payload.get("scheduler_state_dict")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
    if scaler.is_enabled():
        scaler_state = payload.get("scaler_state_dict")
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)

    restore_rng_state(payload.get("rng_state"))
    restore_gradient_state(
        gradient_state=payload.get("gradient_state"),
        model=model,
        stock_tokenizer=stock_tokenizer,
        index_tokenizer=index_tokenizer,
    )
    return payload


def load_model_bundle_for_training(
    checkpoint_dir: str,
    config: CustomFinetuneConfig,
    device: torch.device,
) -> Tuple[MultiStreamCraftModel, CraftTokenizer, CraftTokenizer, Sequence[str], Dict[str, Any]]:
    model_ckpt_path = os.path.join(checkpoint_dir, "multistream_model.pt")
    if not os.path.exists(model_ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {model_ckpt_path}")

    try:
        payload = torch.load(model_ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(model_ckpt_path, map_location=device)

    model = MultiStreamCraftModel.from_serializable_config(payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state_dict"])

    stock_tokenizer_dir = os.path.join(checkpoint_dir, "stock_tokenizer")
    index_tokenizer_dir = os.path.join(checkpoint_dir, "index_tokenizer")
    stock_tokenizer_path = stock_tokenizer_dir if os.path.exists(stock_tokenizer_dir) else config.pretrained_tokenizer_path
    index_tokenizer_path = index_tokenizer_dir if os.path.exists(index_tokenizer_dir) else config.pretrained_tokenizer_path
    stock_tokenizer = CraftTokenizer.from_pretrained(stock_tokenizer_path).to(device)
    index_tokenizer = CraftTokenizer.from_pretrained(index_tokenizer_path).to(device)

    index_ids_path = os.path.join(checkpoint_dir, "index_ids.json")
    index_labels = []
    if os.path.exists(index_ids_path):
        with open(index_ids_path, "r", encoding="utf-8") as f:
            index_labels = json.load(f)["index_ids"]

    return model, stock_tokenizer, index_tokenizer, index_labels, payload


def evaluate_validation(
    model: MultiStreamCraftModel,
    stock_tokenizer: CraftTokenizer,
    index_tokenizer: CraftTokenizer,
    val_loader: DataLoader,
    device: torch.device,
    config: CustomFinetuneConfig,
    mixed_precision_mode: str,
    logger,
    epoch_label: str,
    total_epochs: int,
) -> Dict[str, float]:
    model.eval()
    stock_tokenizer.eval()
    index_tokenizer.eval()

    running_val = {"stock": 0.0, "index": 0.0, "total": 0.0, "batches": 0}
    progress_width = 0
    with torch.no_grad():
        val_start_time = time.time()
        val_progress_interval = max(1, len(val_loader) // 20) if len(val_loader) > 0 else 1
        last_val_progress_time = 0.0
        for val_batch_idx, batch in enumerate(val_loader):
            time_seq = batch["time_seq"].to(device, non_blocking=True)
            index_ids = batch["index_ids"].to(device, non_blocking=True)

            tokens = get_batch_tokens(
                batch=batch,
                stock_tokenizer=stock_tokenizer,
                index_tokenizer=index_tokenizer,
                device=device,
                freeze_stock_tokenizer=config.freeze_stock_tokenizer,
                freeze_index_tokenizer=config.freeze_index_tokenizer,
            )
            (stock_inputs, stock_targets), (index_inputs, index_targets) = prepare_shifted_tokens(tokens)
            with get_autocast_context(mixed_precision_mode, device):
                losses = compute_batch_losses(
                    model=model,
                    stock_inputs=stock_inputs,
                    stock_targets=stock_targets,
                    index_inputs=index_inputs,
                    index_targets=index_targets,
                    time_seq=time_seq,
                    index_ids=index_ids,
                    alpha=config.alpha,
                    use_sampled_s1_for_s2=config.use_sampled_s1_for_s2,
                )

            running_val["stock"] += losses["stock_main_loss"].item()
            running_val["index"] += losses["index_aux_loss"].item()
            running_val["total"] += losses["total_loss"].item()
            running_val["batches"] += 1
            val_elapsed = time.time() - val_start_time
            val_progress = (val_batch_idx + 1) / max(1, len(val_loader))
            val_total_estimate = val_elapsed / max(val_progress, 1e-8)
            val_remaining = max(0.0, val_total_estimate - val_elapsed)
            avg_val_stock_so_far = running_val["stock"] / max(1, running_val["batches"])
            avg_val_index_so_far = running_val["index"] / max(1, running_val["batches"])
            avg_val_total_so_far = running_val["total"] / max(1, running_val["batches"])
            now = time.time()
            if should_refresh_progress(
                step=val_batch_idx + 1,
                total_steps=len(val_loader),
                refresh_interval=val_progress_interval,
                last_refresh_time=last_val_progress_time,
                now=now,
            ):
                progress_width = write_progress_line(
                    format_progress_message(
                        stage="Val  ",
                        epoch=epoch_label,
                        total_epochs=total_epochs,
                        step=val_batch_idx + 1,
                        total_steps=len(val_loader),
                        stock_loss=avg_val_stock_so_far,
                        index_loss=avg_val_index_so_far,
                        total_loss=avg_val_total_so_far,
                        elapsed=val_elapsed,
                        remaining=val_remaining,
                    ),
                    progress_width,
                )
                last_val_progress_time = now

    clear_progress_line(progress_width)
    avg_val_stock = running_val["stock"] / max(1, running_val["batches"])
    avg_val_index = running_val["index"] / max(1, running_val["batches"])
    avg_val_total = running_val["total"] / max(1, running_val["batches"])
    return {
        "stock": avg_val_stock,
        "index": avg_val_index,
        "total": avg_val_total,
        "batches": running_val["batches"],
    }


def train_model(
    model: MultiStreamCraftModel,
    stock_tokenizer: CraftTokenizer,
    index_tokenizer: CraftTokenizer,
    device: torch.device,
    config: CustomFinetuneConfig,
    save_dir: str,
    logger,
    resume_from: Optional[str] = None,
):
    train_loader, val_loader, train_dataset, val_dataset = create_multistream_dataloaders(config)
    train_loader, val_loader = maybe_build_cached_loaders(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        stock_tokenizer=stock_tokenizer,
        index_tokenizer=index_tokenizer,
        device=device,
        logger=logger,
    )
    mixed_precision_mode = resolve_mixed_precision_mode(config, device, logger=logger)
    train_data_source = train_loader.dataset
    train_collate_fn = train_loader.collate_fn
    train_drop_last = train_loader.drop_last
    train_num_workers = train_loader.num_workers
    train_pin_memory = train_loader.pin_memory

    optimizer = build_optimizer(config, model, stock_tokenizer, index_tokenizer)
    base_total_train_batches = count_training_batches(
        dataset_len=len(train_data_source),
        batch_size=config.batch_size,
        drop_last=train_drop_last,
    )
    effective_steps_per_epoch = math.ceil(max(1, base_total_train_batches) / max(1, config.accumulation_steps))
    scheduler = build_scheduler(config, optimizer, effective_steps_per_epoch)
    scaler = GradScaler(enabled=(mixed_precision_mode == "fp16" and device.type == "cuda"))

    best_val_loss = float("inf")
    global_step = 0
    training_start_time = time.time()
    progress_width = 0
    train_progress_interval = max(1, config.log_interval)
    last_train_progress_time = 0.0
    start_epoch = 0
    start_batch_idx = 0
    latest_val_total_loss = None

    if resume_from is not None:
        training_state_path = os.path.join(resume_from, "training_state.pt")
        if os.path.exists(training_state_path):
            resume_payload = load_training_state_checkpoint(
                checkpoint_dir=resume_from,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                model=model,
                stock_tokenizer=stock_tokenizer,
                index_tokenizer=index_tokenizer,
                device=device,
            )
            start_epoch = int(resume_payload.get("epoch", 0))
            start_batch_idx = int(resume_payload.get("next_batch_idx", 0))
            global_step = int(resume_payload.get("global_step", 0))
            best_val_loss = float(resume_payload.get("best_val_loss", float("inf")))
            latest_val_total_loss = resume_payload.get("current_val_total_loss")
            training_start_time = time.time() - float(resume_payload.get("training_elapsed_time", 0.0))
            resume_msg = (
                f"Resuming training from {resume_from}. "
                f"Epoch={start_epoch + 1}, next_batch_idx={start_batch_idx}, global_step={global_step}, "
                f"best_val_total={best_val_loss:.4f}"
            )
        else:
            model_ckpt_path = os.path.join(resume_from, "multistream_model.pt")
            try:
                resume_payload = torch.load(model_ckpt_path, map_location=device, weights_only=True)
            except TypeError:
                resume_payload = torch.load(model_ckpt_path, map_location=device)
            completed_epoch = int(resume_payload.get("epoch", 0))
            start_epoch = completed_epoch
            start_batch_idx = 0
            best_val_loss = float(resume_payload.get("best_val_loss", float("inf")))
            latest_val_total_loss = resume_payload.get("epoch_val_total_loss")
            resume_msg = (
                f"Resuming from epoch checkpoint {resume_from}. "
                f"Continuing at epoch={start_epoch + 1} with restored model/tokenizer weights. "
                "Optimizer, scheduler, scaler, and elapsed-time state were not available in this checkpoint, "
                "so they were reinitialized."
            )
        logger.info(resume_msg)
        print(resume_msg)
    else:
        init_metrics = evaluate_validation(
            model=model,
            stock_tokenizer=stock_tokenizer,
            index_tokenizer=index_tokenizer,
            val_loader=val_loader,
            device=device,
            config=config,
            mixed_precision_mode=mixed_precision_mode,
            logger=logger,
            epoch_label="0",
            total_epochs=config.basemodel_epochs,
        )
        latest_val_total_loss = init_metrics["total"]
        init_summary = (
            "\n--- Initialization Summary ---\n"
            f"Init Val Stock Main Loss: {init_metrics['stock']:.4f}\n"
            f"Init Val Index Aux Loss: {init_metrics['index']:.4f}\n"
            f"Init Val Total Loss: {init_metrics['total']:.4f}\n"
        )
        logger.info(init_summary)
        print(init_summary)
        init_dir = os.path.join(save_dir, "init_model")
        save_checkpoint_bundle(
            checkpoint_dir=init_dir,
            model=model,
            stock_tokenizer=stock_tokenizer,
            index_tokenizer=index_tokenizer,
            config=config,
            index_labels=train_dataset.index_labels,
            best_val_loss=init_metrics["total"],
            epoch=0,
            val_total_loss=init_metrics["total"],
        )
        init_msg = f"Initialization checkpoint saved to {init_dir}"
        logger.info(init_msg)
        print(init_msg)

    loop_setup_msg = (
        f"Training loop setup: start_epoch={start_epoch + 1}, total_epochs={config.basemodel_epochs}, "
        f"start_batch_idx={start_batch_idx}, train_samples={len(train_data_source)}, "
        f"base_total_train_batches={base_total_train_batches}, val_batches={len(val_loader)}"
    )
    logger.info(loop_setup_msg)
    print(loop_setup_msg)

    if base_total_train_batches <= 0:
        empty_train_msg = (
            "Training dataset resolved to 0 batches. "
            "This usually means the train split has no usable windows after stock/index alignment and windowing."
        )
        logger.warning(empty_train_msg)
        print(empty_train_msg)

    if start_epoch >= config.basemodel_epochs:
        no_epoch_msg = (
            f"No remaining epochs to run: start_epoch={start_epoch + 1}, "
            f"total_epochs={config.basemodel_epochs}. "
            "If you meant to continue training, please increase `training.basemodel_epochs` "
            "or resume from an earlier checkpoint."
        )
        logger.warning(no_epoch_msg)
        print(no_epoch_msg)
        return best_val_loss, False

    latest_resume_dir = os.path.join(save_dir, "latest_resume")
    current_epoch = start_epoch
    current_next_batch_idx = start_batch_idx

    try:
        for epoch in range(start_epoch, config.basemodel_epochs):
            epoch_start_time = time.time()
            epoch_resume_batch_idx = start_batch_idx if epoch == start_epoch else 0
            current_epoch = epoch
            current_next_batch_idx = epoch_resume_batch_idx

            epoch_train_loader, total_train_batches = build_epoch_train_loader(
                dataset=train_data_source,
                batch_size=config.batch_size,
                num_workers=train_num_workers,
                pin_memory=train_pin_memory,
                collate_fn=train_collate_fn,
                seed=config.seed,
                epoch=epoch,
                start_batch_idx=epoch_resume_batch_idx,
                drop_last=train_drop_last,
            )

            model.train()
            if not config.freeze_stock_tokenizer:
                stock_tokenizer.train()
            if not config.freeze_index_tokenizer:
                index_tokenizer.train()

            if epoch_resume_batch_idx == 0:
                optimizer.zero_grad(set_to_none=True)
            running_train = {"stock": 0.0, "index": 0.0, "total": 0.0, "batches": 0}
            last_train_progress_time = 0.0

            for batch_idx, batch in enumerate(epoch_train_loader, start=epoch_resume_batch_idx):
                current_next_batch_idx = batch_idx
                time_seq = batch["time_seq"].to(device, non_blocking=True)
                index_ids = batch["index_ids"].to(device, non_blocking=True)

                tokens = get_batch_tokens(
                    batch=batch,
                    stock_tokenizer=stock_tokenizer,
                    index_tokenizer=index_tokenizer,
                    device=device,
                    freeze_stock_tokenizer=config.freeze_stock_tokenizer,
                    freeze_index_tokenizer=config.freeze_index_tokenizer,
                )
                (stock_inputs, stock_targets), (index_inputs, index_targets) = prepare_shifted_tokens(tokens)

                with get_autocast_context(mixed_precision_mode, device):
                    losses = compute_batch_losses(
                        model=model,
                        stock_inputs=stock_inputs,
                        stock_targets=stock_targets,
                        index_inputs=index_inputs,
                        index_targets=index_targets,
                        time_seq=time_seq,
                        index_ids=index_ids,
                        alpha=config.alpha,
                        use_sampled_s1_for_s2=config.use_sampled_s1_for_s2,
                    )

                scaled_total_loss = losses["total_loss"] / max(1, config.accumulation_steps)
                if scaler.is_enabled():
                    scaler.scale(scaled_total_loss).backward()
                else:
                    scaled_total_loss.backward()

                should_step = (
                    ((batch_idx + 1) % max(1, config.accumulation_steps) == 0)
                    or (batch_idx == total_train_batches - 1)
                )
                if should_step:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                    if not config.freeze_stock_tokenizer:
                        torch.nn.utils.clip_grad_norm_(stock_tokenizer.parameters(), max_norm=3.0)
                    if not config.freeze_index_tokenizer:
                        torch.nn.utils.clip_grad_norm_(index_tokenizer.parameters(), max_norm=3.0)
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                running_train["stock"] += losses["stock_main_loss"].item()
                running_train["index"] += losses["index_aux_loss"].item()
                running_train["total"] += losses["total_loss"].item()
                running_train["batches"] += 1

                current_next_batch_idx = batch_idx + 1
                now = time.time()
                lr = optimizer.param_groups[0]["lr"]
                epoch_elapsed = now - epoch_start_time
                epoch_progress = (batch_idx + 1) / max(1, total_train_batches)
                epoch_total_estimate = epoch_elapsed / max(epoch_progress, 1e-8)
                epoch_remaining = max(0.0, epoch_total_estimate - epoch_elapsed)
                avg_stock_so_far = running_train["stock"] / max(1, running_train["batches"])
                avg_index_so_far = running_train["index"] / max(1, running_train["batches"])
                avg_total_so_far = running_train["total"] / max(1, running_train["batches"])
                if should_refresh_progress(
                    step=batch_idx + 1,
                    total_steps=total_train_batches,
                    refresh_interval=train_progress_interval,
                    last_refresh_time=last_train_progress_time,
                    now=now,
                ):
                    progress_width = write_progress_line(
                        format_progress_message(
                            stage="Train",
                            epoch=epoch + 1,
                            total_epochs=config.basemodel_epochs,
                            step=batch_idx + 1,
                            total_steps=total_train_batches,
                            stock_loss=avg_stock_so_far,
                            index_loss=avg_index_so_far,
                            total_loss=avg_total_so_far,
                            elapsed=epoch_elapsed,
                            remaining=epoch_remaining,
                            extra=f"LR {lr:.6f}",
                        ),
                        progress_width,
                    )
                    last_train_progress_time = now

                if (global_step + 1) % config.log_interval == 0:
                    total_elapsed = now - training_start_time
                    total_progress = (epoch + epoch_progress) / max(1, config.basemodel_epochs)
                    total_estimate = total_elapsed / max(total_progress, 1e-8)
                    total_remaining = max(0.0, total_estimate - total_elapsed)

                    log_msg = (
                        f"[Epoch {epoch + 1}/{config.basemodel_epochs}, Step {batch_idx + 1}/{total_train_batches}] "
                        f"LR: {lr:.6f}, Stock Main: {avg_stock_so_far:.4f}, "
                        f"Index Aux: {avg_index_so_far:.4f}, Total: {avg_total_so_far:.4f}, "
                        f"Epoch ETA: {format_duration(epoch_remaining)}, Total ETA: {format_duration(total_remaining)}"
                    )
                    log_file_only(logger, log_msg)
                global_step += 1

            clear_progress_line(progress_width)
            progress_width = 0
            current_next_batch_idx = total_train_batches

            val_metrics = evaluate_validation(
                model=model,
                stock_tokenizer=stock_tokenizer,
                index_tokenizer=index_tokenizer,
                val_loader=val_loader,
                device=device,
                config=config,
                mixed_precision_mode=mixed_precision_mode,
                logger=logger,
                epoch_label=str(epoch + 1),
                total_epochs=config.basemodel_epochs,
            )

            avg_train_stock = running_train["stock"] / max(1, running_train["batches"])
            avg_train_index = running_train["index"] / max(1, running_train["batches"])
            avg_train_total = running_train["total"] / max(1, running_train["batches"])
            avg_val_stock = val_metrics["stock"]
            avg_val_index = val_metrics["index"]
            avg_val_total = val_metrics["total"]
            latest_val_total_loss = avg_val_total
            epoch_elapsed = time.time() - epoch_start_time
            total_elapsed = time.time() - training_start_time
            average_epoch_time = total_elapsed / max(1, epoch + 1)
            remaining_epochs = max(0, config.basemodel_epochs - epoch - 1)
            total_remaining = average_epoch_time * remaining_epochs

            epoch_summary = (
                f"\n--- Epoch {epoch + 1}/{config.basemodel_epochs} Summary ---\n"
                f"Train Stock Main Loss: {avg_train_stock:.4f}\n"
                f"Train Index Aux Loss: {avg_train_index:.4f}\n"
                f"Train Total Loss: {avg_train_total:.4f}\n"
                f"Val Stock Main Loss: {avg_val_stock:.4f}\n"
                f"Val Index Aux Loss: {avg_val_index:.4f}\n"
                f"Val Total Loss: {avg_val_total:.4f}\n"
                f"Epoch Time: {format_duration(epoch_elapsed)}\n"
                f"Elapsed: {format_duration(total_elapsed)}\n"
                f"Estimated Remaining: {format_duration(total_remaining)}\n"
            )
            logger.info(epoch_summary)
            print(epoch_summary)

            is_new_best = avg_val_total < best_val_loss
            if is_new_best:
                best_val_loss = avg_val_total
                save_best_checkpoint(
                    save_dir=save_dir,
                    model=model,
                    stock_tokenizer=stock_tokenizer,
                    index_tokenizer=index_tokenizer,
                    config=config,
                    index_labels=train_dataset.index_labels,
                    best_val_loss=best_val_loss,
                    epoch=epoch + 1,
                )
                best_msg = f"Best checkpoint saved to {os.path.join(save_dir, 'best_model')} (val total loss: {best_val_loss:.4f})"
                logger.info(best_msg)
                print(best_msg)

            save_epoch_checkpoint(
                save_dir=save_dir,
                epoch=epoch + 1,
                model=model,
                stock_tokenizer=stock_tokenizer,
                index_tokenizer=index_tokenizer,
                config=config,
                index_labels=train_dataset.index_labels,
                best_val_loss=best_val_loss,
                val_total_loss=avg_val_total,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                global_step=global_step,
                training_elapsed_time=time.time() - training_start_time,
            )
            epoch_ckpt_msg = f"Epoch checkpoint saved to {os.path.join(save_dir, f'epoch_{epoch + 1:03d}')}"
            logger.info(epoch_ckpt_msg)
            print(epoch_ckpt_msg)
            save_training_state_checkpoint(
                checkpoint_dir=latest_resume_dir,
                model=model,
                stock_tokenizer=stock_tokenizer,
                index_tokenizer=index_tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                index_labels=train_dataset.index_labels,
                best_val_loss=best_val_loss,
                resume_epoch=epoch + 1,
                next_batch_idx=0,
                global_step=global_step,
                training_elapsed_time=time.time() - training_start_time,
                current_val_total_loss=avg_val_total,
                checkpoint_epoch=epoch + 1,
            )
            start_batch_idx = 0

    except KeyboardInterrupt:
        clear_progress_line(progress_width)
        progress_width = 0
        interrupted_msg = (
            f"Training interrupted. Saving resumable state to {latest_resume_dir} "
            f"(epoch={current_epoch + 1}, next_batch_idx={current_next_batch_idx}, global_step={global_step})."
        )
        logger.warning(interrupted_msg)
        print(interrupted_msg)
        save_training_state_checkpoint(
            checkpoint_dir=latest_resume_dir,
            model=model,
            stock_tokenizer=stock_tokenizer,
            index_tokenizer=index_tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            index_labels=train_dataset.index_labels,
            best_val_loss=best_val_loss,
            resume_epoch=current_epoch,
            next_batch_idx=current_next_batch_idx,
            global_step=global_step,
            training_elapsed_time=time.time() - training_start_time,
            current_val_total_loss=latest_val_total_loss,
            checkpoint_epoch=current_epoch + 1,
        )
        return best_val_loss, True

    return best_val_loss, False


def main():
    parser = argparse.ArgumentParser(description="Train the multistream A-share Craft v1 model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the multistream yaml config.")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a resumable training checkpoint directory such as `latest_resume`.",
    )
    parser.add_argument(
        "--resume-last-epoch",
        action="store_true",
        help="Automatically resume from the latest saved `epoch_XXX` checkpoint under the experiment save directory.",
    )
    parser.add_argument(
        "--use-sampled-s1-for-s2",
        dest="use_sampled_s1_for_s2",
        action="store_true",
        help="Train s2 with sampled s1 conditioning, matching the original Kronos-style path.",
    )
    parser.add_argument(
        "--use-teacher-forced-s1-for-s2",
        dest="use_sampled_s1_for_s2",
        action="store_false",
        help="Train s2 with teacher-forced true s1 conditioning.",
    )
    parser.set_defaults(use_sampled_s1_for_s2=None)
    args = parser.parse_args()

    config = CustomFinetuneConfig(args.config)
    if args.use_sampled_s1_for_s2 is not None:
        config.use_sampled_s1_for_s2 = args.use_sampled_s1_for_s2
    if config.stock_dir is None or config.merged_indices_path is None:
        raise ValueError("Multistream training requires `data.stock_dir` and `data.merged_indices_path` in the yaml config.")
    if not config.freeze_stock_tokenizer or not config.freeze_index_tokenizer:
        raise NotImplementedError(
            "Tokenizer unfreezing is reserved for a later extension. "
            "This v1 implementation keeps the original discrete tokenizer path frozen."
        )

    device = torch.device("cuda" if config.use_cuda and torch.cuda.is_available() else "cpu")
    set_seed(config.seed)

    os.makedirs(config.basemodel_save_path, exist_ok=True)
    log_dir = os.path.join(config.base_save_path, "logs")
    logger = setup_logging(config.exp_name, log_dir, 0)

    resume_from = args.resume_from
    if resume_from is None and args.resume_last_epoch:
        resume_from = find_latest_epoch_checkpoint(config.basemodel_save_path)
        if resume_from is None:
            raise FileNotFoundError(
                f"No epoch checkpoint was found under {config.basemodel_save_path}. "
                "Please make sure training has completed at least one epoch."
            )

    if resume_from is not None:
        model, stock_tokenizer, index_tokenizer, resume_index_labels, _ = load_model_bundle_for_training(
            checkpoint_dir=resume_from,
            config=config,
            device=device,
        )
        config.index_ids = list(resume_index_labels) if resume_index_labels else config.index_ids
    else:
        stock_tokenizer = CraftTokenizer.from_pretrained(config.pretrained_tokenizer_path).to(device)
        index_tokenizer = CraftTokenizer.from_pretrained(config.pretrained_tokenizer_path).to(device)
        stock_predictor = build_craft_backbone(
            config.pretrained_predictor_path,
            load_pretrained_weights=config.pre_trained_predictor,
        ).to(device)
        index_predictor = build_craft_backbone(
            config.pretrained_predictor_path,
            load_pretrained_weights=config.pre_trained_predictor,
        ).to(device)

        metadata_index_ids = load_index_metadata(
            os.path.join(os.path.dirname(config.merged_indices_path), "index_ids.json") if config.merged_indices_path else None
        )
        effective_index_ids = config.index_ids or metadata_index_ids
        if not effective_index_ids:
            _, _, effective_index_ids = load_merged_indices_panel(
                merged_indices_path=config.merged_indices_path,
                feature_cols=config.feature_cols,
                date_col=config.index_date_col,
                index_ids=None,
            )
        config.index_ids = effective_index_ids
        num_indices = config.num_indices if config.num_indices not in (None, 0) else len(effective_index_ids)
        if num_indices <= 0:
            raise ValueError("Please set `model.num_indices` or `model.index_ids` in the yaml config.")

        model = MultiStreamCraftModel(
            stock_predictor=stock_predictor,
            index_predictor=index_predictor,
            num_indices=num_indices,
            lag_window=config.lag_window,
            no_early_pooling=config.no_early_pooling,
        ).to(device)

    set_tokenizer_trainable(stock_tokenizer, trainable=not config.freeze_stock_tokenizer)
    set_tokenizer_trainable(index_tokenizer, trainable=not config.freeze_index_tokenizer)

    logger.info("=== Multistream Training Configuration ===")
    logger.info(f"Stock dir: {config.stock_dir}")
    logger.info(f"Merged indices path: {config.merged_indices_path}")
    logger.info(f"Lookback: {config.lookback}")
    logger.info(f"Horizon: {config.horizon}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Predictor LR: {config.predictor_learning_rate}")
    logger.info(f"Mixed precision: {config.mixed_precision}")
    logger.info(f"Cache tokenized sequences: {config.cache_tokenized_sequences}")
    logger.info(f"Token cache batch size: {config.token_cache_batch_size}")
    logger.info(f"Alpha: {config.alpha}")
    logger.info(f"Lag window: {config.lag_window}")
    logger.info(f"Use sampled s1 for s2: {config.use_sampled_s1_for_s2}")
    logger.info(f"Use pre-trained predictor weights: {config.pre_trained_predictor}")
    logger.info(f"Freeze stock tokenizer: {config.freeze_stock_tokenizer}")
    logger.info(f"Freeze index tokenizer: {config.freeze_index_tokenizer}")
    logger.info(f"Resume from: {resume_from}")

    best_val_loss, interrupted = train_model(
        model=model,
        stock_tokenizer=stock_tokenizer,
        index_tokenizer=index_tokenizer,
        device=device,
        config=config,
        save_dir=config.basemodel_save_path,
        logger=logger,
        resume_from=resume_from,
    )
    if interrupted:
        final_msg = (
            f"Multistream training interrupted after saving resumable state. "
            f"Best validation total loss so far: {best_val_loss:.4f}"
        )
    else:
        final_msg = f"Multistream training completed. Best validation total loss: {best_val_loss:.4f}"
    logger.info(final_msg)
    print(final_msg)


if __name__ == "__main__":
    main()
