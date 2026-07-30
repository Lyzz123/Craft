import json
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .craft import CraftBackbone


def load_craft_init_config(pretrained_predictor_path: str) -> Dict[str, int]:
    cfg_path = os.path.join(pretrained_predictor_path, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Could not find Craft predictor config at {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw_cfg = json.load(f)

    return {
        "s1_bits": raw_cfg.get("s1_bits", 10),
        "s2_bits": raw_cfg.get("s2_bits", 10),
        "n_layers": raw_cfg.get("n_layers", 12),
        "d_model": raw_cfg.get("d_model", 832),
        "n_heads": raw_cfg.get("n_heads", 16),
        "ff_dim": raw_cfg.get("ff_dim", 2048),
        "ffn_dropout_p": raw_cfg.get("ffn_dropout_p", 0.2),
        "attn_dropout_p": raw_cfg.get("attn_dropout_p", 0.0),
        "resid_dropout_p": raw_cfg.get("resid_dropout_p", 0.2),
        "token_dropout_p": raw_cfg.get("token_dropout_p", 0.0),
        "learn_te": raw_cfg.get("learn_te", True),
    }


def build_craft_from_config(pretrained_predictor_path: str) -> CraftBackbone:
    return CraftBackbone(**load_craft_init_config(pretrained_predictor_path))


def build_craft_backbone(pretrained_predictor_path: str, load_pretrained_weights: bool = True) -> CraftBackbone:
    if load_pretrained_weights:
        return CraftBackbone.from_pretrained(pretrained_predictor_path)
    return build_craft_from_config(pretrained_predictor_path)


def extract_craft_init_config(model: CraftBackbone) -> Dict[str, int]:
    return {
        "s1_bits": model.s1_bits,
        "s2_bits": model.s2_bits,
        "n_layers": model.n_layers,
        "d_model": model.d_model,
        "n_heads": model.n_heads,
        "ff_dim": model.ff_dim,
        "ffn_dropout_p": model.ffn_dropout_p,
        "attn_dropout_p": model.attn_dropout_p,
        "resid_dropout_p": model.resid_dropout_p,
        "token_dropout_p": model.token_dropout_p,
        "learn_te": model.learn_te,
    }


class IndexIdEmbedding(nn.Module):
    def __init__(self, num_indices: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(num_indices, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=d_model ** -0.5)

    def forward(self, index_hidden: torch.Tensor, index_ids: torch.Tensor) -> torch.Tensor:
        if index_ids.dim() == 1:
            index_ids = index_ids.unsqueeze(0).expand(index_hidden.size(0), -1)
        emb = self.embedding(index_ids).unsqueeze(1)
        return index_hidden + emb


class MultiIndexMemoryBuilder(nn.Module):
    def __init__(self, lag_window: int):
        super().__init__()
        self.lag_window = lag_window

    def forward(self, index_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, num_indices, hidden_dim = index_hidden.shape
        memories = []
        masks = []
        lag_id_slices = []

        for delta in range(self.lag_window + 1):
            shifted = index_hidden.new_zeros(batch_size, seq_len, num_indices, hidden_dim)
            valid_mask = torch.zeros(batch_size, seq_len, num_indices, dtype=torch.bool, device=index_hidden.device)
            if delta == 0:
                shifted = index_hidden
                valid_mask[:] = True
            else:
                shifted[:, delta:] = index_hidden[:, :-delta]
                valid_mask[:, delta:] = True

            memories.append(shifted)
            masks.append(valid_mask)
            lag_id_slices.append(
                torch.full((seq_len, num_indices), delta, dtype=torch.long, device=index_hidden.device)
            )

        memory = torch.cat(memories, dim=2)
        valid_mask = torch.cat(masks, dim=2)
        lag_ids = torch.cat(lag_id_slices, dim=1)
        return memory, lag_ids, valid_mask


class LaggedCrossAttentionTopAdapter(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        lag_window: int,
        attn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.attn_dropout = nn.Dropout(attn_dropout_p)
        self.resid_dropout = nn.Dropout(resid_dropout_p)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        self.lag_bias = nn.Embedding(lag_window + 1, n_heads)

    def forward(
        self,
        stock_hidden: torch.Tensor,
        memory: torch.Tensor,
        lag_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, memory_slots, _ = memory.shape

        q = self.q_proj(stock_hidden).view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(memory).view(batch_size, seq_len, memory_slots, self.n_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        v = self.v_proj(memory).view(batch_size, seq_len, memory_slots, self.n_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        scores = torch.einsum("bhtd,bhtmd->bhtm", q, k) * self.scale
        lag_bias = self.lag_bias(lag_ids).permute(2, 0, 1).unsqueeze(0)
        scores = scores + lag_bias
        scores = scores.masked_fill(~valid_mask.unsqueeze(1), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_out = torch.einsum("bhtm,bhtmd->bhtd", attn_weights, v)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)
        attn_out = self.resid_dropout(self.out_proj(attn_out))

        gate = torch.sigmoid(self.gate_proj(torch.cat([stock_hidden, attn_out], dim=-1)))
        fused = stock_hidden + gate * attn_out
        return fused, attn_weights


class MultiStreamCraftModel(nn.Module):
    def __init__(
        self,
        stock_predictor: CraftBackbone,
        index_predictor: CraftBackbone,
        num_indices: int,
        lag_window: int = 5,
        no_early_pooling: bool = True,
    ):
        super().__init__()
        if stock_predictor.d_model != index_predictor.d_model:
            raise ValueError("Stock and index predictors must share the same hidden dimension.")

        self.stock_predictor = stock_predictor
        self.index_predictor = index_predictor
        self.num_indices = num_indices
        self.lag_window = lag_window
        self.no_early_pooling = no_early_pooling

        self.index_id_embedding = IndexIdEmbedding(num_indices, stock_predictor.d_model)
        self.memory_builder = MultiIndexMemoryBuilder(lag_window)
        self.top_adapter = LaggedCrossAttentionTopAdapter(
            d_model=stock_predictor.d_model,
            n_heads=stock_predictor.n_heads,
            lag_window=lag_window,
            attn_dropout_p=stock_predictor.attn_dropout_p,
            resid_dropout_p=stock_predictor.resid_dropout_p,
        )

    @property
    def d_model(self) -> int:
        return self.stock_predictor.d_model

    def get_serializable_config(self) -> Dict[str, object]:
        return {
            "num_indices": self.num_indices,
            "lag_window": self.lag_window,
            "no_early_pooling": self.no_early_pooling,
            "stock_predictor": extract_craft_init_config(self.stock_predictor),
            "index_predictor": extract_craft_init_config(self.index_predictor),
        }

    @classmethod
    def from_serializable_config(cls, payload: Dict[str, object]) -> "MultiStreamCraftModel":
        stock_predictor = CraftBackbone(**payload["stock_predictor"])
        index_predictor = CraftBackbone(**payload["index_predictor"])
        return cls(
            stock_predictor=stock_predictor,
            index_predictor=index_predictor,
            num_indices=payload["num_indices"],
            lag_window=payload.get("lag_window", 5),
            no_early_pooling=payload.get("no_early_pooling", True),
        )

    def _sample_s1_ids(self, logits: torch.Tensor, vocab_size: int) -> torch.Tensor:
        probs = F.softmax(logits.detach(), dim=-1)
        return torch.multinomial(probs.view(-1, vocab_size), 1).view(logits.shape[:-1])

    @staticmethod
    def _validate_decode_s2_output(
        branch_name: str,
        logits: torch.Tensor,
        expected_batch: int,
        expected_seq_len: int,
    ) -> None:
        if logits.dim() != 3:
            raise ValueError(
                f"{branch_name} decode_s2 expected a 3D tensor [batch, seq_len, vocab], "
                f"but received shape={tuple(logits.shape)}."
            )
        actual_batch, actual_seq_len, _ = logits.shape
        if actual_batch != expected_batch or actual_seq_len != expected_seq_len:
            raise ValueError(
                f"{branch_name} decode_s2 returned an unexpected shape: "
                f"expected batch={expected_batch}, seq_len={expected_seq_len}, "
                f"observed shape={tuple(logits.shape)}. "
                "This usually indicates a sequence-length broadcast mismatch in the conditional s2 path."
            )

    def encode_index_context(
        self,
        index_s1_ids: torch.Tensor,
        index_s2_ids: torch.Tensor,
        stamp: torch.Tensor,
        index_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, num_indices = index_s1_ids.shape
        flat_s1 = index_s1_ids.permute(0, 2, 1).reshape(batch_size * num_indices, seq_len)
        flat_s2 = index_s2_ids.permute(0, 2, 1).reshape(batch_size * num_indices, seq_len)
        flat_stamp = stamp.unsqueeze(1).expand(batch_size, num_indices, seq_len, stamp.size(-1)).reshape(
            batch_size * num_indices, seq_len, stamp.size(-1)
        )

        flat_s1_logits, flat_context = self.index_predictor.decode_s1(flat_s1, flat_s2, flat_stamp)
        s1_logits = flat_s1_logits.view(batch_size, num_indices, seq_len, -1).permute(0, 2, 1, 3).contiguous()
        context = flat_context.view(batch_size, num_indices, seq_len, -1).permute(0, 2, 1, 3).contiguous()
        context = self.index_id_embedding(context, index_ids)
        return s1_logits, context

    def decode_index_s2(self, index_context: torch.Tensor, index_s1_ids: torch.Tensor) -> torch.Tensor:
        batch_size, ctx_seq_len, num_indices, hidden_dim = index_context.shape
        query_seq_len = index_s1_ids.shape[1]
        flat_context = index_context.permute(0, 2, 1, 3).reshape(batch_size * num_indices, ctx_seq_len, hidden_dim)
        flat_s1 = index_s1_ids.permute(0, 2, 1).reshape(batch_size * num_indices, query_seq_len)
        flat_logits = self.index_predictor.decode_s2(flat_context, flat_s1)
        self._validate_decode_s2_output(
            branch_name="index",
            logits=flat_logits,
            expected_batch=batch_size * num_indices,
            expected_seq_len=query_seq_len,
        )
        return flat_logits.view(batch_size, num_indices, query_seq_len, -1).permute(0, 2, 1, 3).contiguous()

    def decode_stock_s1_with_indices(
        self,
        stock_s1_ids: torch.Tensor,
        stock_s2_ids: torch.Tensor,
        stamp: torch.Tensor,
        index_context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, stock_context = self.stock_predictor.decode_s1(stock_s1_ids, stock_s2_ids, stamp)
        memory, lag_ids, valid_mask = self.memory_builder(index_context)
        fused_context, attn_weights = self.top_adapter(stock_context, memory, lag_ids, valid_mask)
        stock_s1_logits = self.stock_predictor.head(fused_context)
        return stock_s1_logits, fused_context, attn_weights

    def decode_stock_s2(self, fused_stock_context: torch.Tensor, stock_s1_ids: torch.Tensor) -> torch.Tensor:
        stock_logits = self.stock_predictor.decode_s2(fused_stock_context, stock_s1_ids)
        self._validate_decode_s2_output(
            branch_name="stock",
            logits=stock_logits,
            expected_batch=fused_stock_context.size(0),
            expected_seq_len=stock_s1_ids.size(1),
        )
        return stock_logits

    def forward(
        self,
        stock_s1_ids: torch.Tensor,
        stock_s2_ids: torch.Tensor,
        index_s1_ids: torch.Tensor,
        index_s2_ids: torch.Tensor,
        stamp: torch.Tensor,
        index_ids: torch.Tensor,
        stock_s1_targets: Optional[torch.Tensor] = None,
        index_s1_targets: Optional[torch.Tensor] = None,
        use_sampled_s1_for_s2: bool = False,
    ) -> Dict[str, torch.Tensor]:
        index_s1_logits, index_context = self.encode_index_context(index_s1_ids, index_s2_ids, stamp, index_ids)

        if use_sampled_s1_for_s2 or index_s1_targets is None:
            sampled_index_s1 = self._sample_s1_ids(index_s1_logits, self.index_predictor.s1_vocab_size)
        else:
            sampled_index_s1 = index_s1_targets
        index_s2_logits = self.decode_index_s2(index_context, sampled_index_s1)

        stock_s1_logits, fused_stock_context, attn_weights = self.decode_stock_s1_with_indices(
            stock_s1_ids=stock_s1_ids,
            stock_s2_ids=stock_s2_ids,
            stamp=stamp,
            index_context=index_context,
        )

        if use_sampled_s1_for_s2 or stock_s1_targets is None:
            sampled_stock_s1 = self._sample_s1_ids(stock_s1_logits, self.stock_predictor.s1_vocab_size)
        else:
            sampled_stock_s1 = stock_s1_targets
        stock_s2_logits = self.decode_stock_s2(fused_stock_context, sampled_stock_s1)

        return {
            "stock_s1_logits": stock_s1_logits,
            "stock_s2_logits": stock_s2_logits,
            "index_s1_logits": index_s1_logits,
            "index_s2_logits": index_s2_logits,
            "fused_stock_context": fused_stock_context,
            "index_context": index_context,
            "attn_weights": attn_weights,
        }

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        stock_targets: Tuple[torch.Tensor, torch.Tensor],
        index_targets: Tuple[torch.Tensor, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        stock_loss, stock_s1_loss, stock_s2_loss = self.stock_predictor.head.compute_loss(
            outputs["stock_s1_logits"],
            outputs["stock_s2_logits"],
            stock_targets[0],
            stock_targets[1],
        )

        index_s1_logits = outputs["index_s1_logits"].permute(0, 2, 1, 3).reshape(
            -1, outputs["index_s1_logits"].size(1), outputs["index_s1_logits"].size(-1)
        )
        index_s2_logits = outputs["index_s2_logits"].permute(0, 2, 1, 3).reshape(
            -1, outputs["index_s2_logits"].size(1), outputs["index_s2_logits"].size(-1)
        )
        index_s1_targets = index_targets[0].permute(0, 2, 1).reshape(-1, index_targets[0].size(1))
        index_s2_targets = index_targets[1].permute(0, 2, 1).reshape(-1, index_targets[1].size(1))

        index_loss, index_s1_loss, index_s2_loss = self.index_predictor.head.compute_loss(
            index_s1_logits,
            index_s2_logits,
            index_s1_targets,
            index_s2_targets,
        )

        return {
            "stock_main_loss": stock_loss,
            "stock_s1_loss": stock_s1_loss,
            "stock_s2_loss": stock_s2_loss,
            "index_aux_loss": index_loss,
            "index_s1_loss": index_s1_loss,
            "index_s2_loss": index_s2_loss,
        }
