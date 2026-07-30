import time
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from multistream_dataset import multistream_collate_fn
except ImportError:
    from .multistream_dataset import multistream_collate_fn


class CachedTokenizedMultiStreamDataset(Dataset):

    def __init__(
        self,
        stock_s1: torch.Tensor,
        stock_s2: torch.Tensor,
        index_s1: torch.Tensor,
        index_s2: torch.Tensor,
        time_seq: torch.Tensor,
        index_ids: torch.Tensor,
    ):
        if not (len(stock_s1) == len(stock_s2) == len(index_s1) == len(index_s2) == len(time_seq)):
            raise ValueError("All cached tensors must have the same leading dimension.")
        self.stock_s1 = stock_s1.contiguous()
        self.stock_s2 = stock_s2.contiguous()
        self.index_s1 = index_s1.contiguous()
        self.index_s2 = index_s2.contiguous()
        self.time_seq = time_seq.contiguous()
        self.index_ids = index_ids.clone().long()

    def __len__(self) -> int:
        return int(self.stock_s1.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "stock_s1": self.stock_s1[idx],
            "stock_s2": self.stock_s2[idx],
            "index_s1": self.index_s1[idx],
            "index_s2": self.index_s2[idx],
            "time_seq": self.time_seq[idx],
            "index_ids": self.index_ids.clone(),
        }


def encode_multistream_batch(
    stock_tokenizer,
    index_tokenizer,
    stock_seq: torch.Tensor,
    index_seq: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    stock_s1, stock_s2 = stock_tokenizer.encode(stock_seq, half=True)

    batch_size, seq_len, num_indices, feature_dim = index_seq.shape
    index_flat = index_seq.permute(0, 2, 1, 3).reshape(batch_size * num_indices, seq_len, feature_dim)
    index_s1, index_s2 = index_tokenizer.encode(index_flat, half=True)

    index_s1 = index_s1.reshape(batch_size, num_indices, seq_len).permute(0, 2, 1).contiguous()
    index_s2 = index_s2.reshape(batch_size, num_indices, seq_len).permute(0, 2, 1).contiguous()

    return {
        "stock_s1": stock_s1,
        "stock_s2": stock_s2,
        "index_s1": index_s1,
        "index_s2": index_s2,
    }


def build_cached_tokenized_dataset(
    dataset: Dataset,
    stock_tokenizer,
    index_tokenizer,
    device: torch.device,
    batch_size: int,
    num_workers: int = 0,
    logger: Optional[object] = None,
    split_name: str = "train",
) -> CachedTokenizedMultiStreamDataset:
    if len(dataset) == 0:
        raise ValueError(f"Cannot cache tokenized samples for empty split `{split_name}`.")

    if not hasattr(dataset, "window") or not hasattr(dataset, "num_indices") or not hasattr(dataset, "index_id_tensor"):
        raise ValueError("The dataset does not expose the attributes required for token caching.")

    total_samples = len(dataset)
    seq_len = int(dataset.window)
    num_indices = int(dataset.num_indices)
    time_dim = 5

    stock_s1 = torch.empty((total_samples, seq_len), dtype=torch.long)
    stock_s2 = torch.empty((total_samples, seq_len), dtype=torch.long)
    index_s1 = torch.empty((total_samples, seq_len, num_indices), dtype=torch.long)
    index_s2 = torch.empty((total_samples, seq_len, num_indices), dtype=torch.long)
    time_seq = torch.empty((total_samples, seq_len, time_dim), dtype=torch.float32)

    cache_loader = DataLoader(
        dataset,
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=multistream_collate_fn,
    )

    previous_stock_mode = stock_tokenizer.training
    previous_index_mode = index_tokenizer.training
    stock_tokenizer.eval()
    index_tokenizer.eval()

    start_time = time.time()
    last_log_time = start_time
    offset = 0
    total_batches = max(1, len(cache_loader))

    if logger is not None:
        logger.info(
            f"Building cached tokenizer outputs for split `{split_name}` "
            f"({total_samples} samples, cache_batch_size={max(1, batch_size)})."
        )
    print(
        f"Building cached tokenizer outputs for split `{split_name}` "
        f"({total_samples} samples, cache_batch_size={max(1, batch_size)})..."
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(cache_loader):
            stock_batch = batch["stock_seq"].to(device, non_blocking=True)
            index_batch = batch["index_seq"].to(device, non_blocking=True)

            encoded = encode_multistream_batch(
                stock_tokenizer=stock_tokenizer,
                index_tokenizer=index_tokenizer,
                stock_seq=stock_batch,
                index_seq=index_batch,
            )

            batch_size_actual = stock_batch.shape[0]
            next_offset = offset + batch_size_actual

            stock_s1[offset:next_offset] = encoded["stock_s1"].detach().cpu().long()
            stock_s2[offset:next_offset] = encoded["stock_s2"].detach().cpu().long()
            index_s1[offset:next_offset] = encoded["index_s1"].detach().cpu().long()
            index_s2[offset:next_offset] = encoded["index_s2"].detach().cpu().long()
            time_seq[offset:next_offset] = batch["time_seq"].detach().cpu().float()
            offset = next_offset

            now = time.time()
            should_log = (batch_idx + 1 == total_batches) or (now - last_log_time >= 5.0)
            if should_log:
                progress = (batch_idx + 1) / total_batches
                elapsed = now - start_time
                samples_per_second = offset / max(elapsed, 1e-8)
                message = (
                    f"Token cache [{split_name}] {progress * 100:6.2f}% "
                    f"({batch_idx + 1}/{total_batches}) | cached {offset}/{total_samples} samples | "
                    f"{samples_per_second:.1f} samples/s"
                )
                if logger is not None:
                    logger.info(message)
                print(message)
                last_log_time = now

    stock_tokenizer.train(previous_stock_mode)
    index_tokenizer.train(previous_index_mode)

    return CachedTokenizedMultiStreamDataset(
        stock_s1=stock_s1,
        stock_s2=stock_s2,
        index_s1=index_s1,
        index_s2=index_s2,
        time_seq=time_seq,
        index_ids=dataset.index_id_tensor,
    )

