import json
import os
from typing import Any, Dict, List, Optional

import yaml


DEFAULT_FEATURE_COLS = ["open", "high", "low", "close", "vol", "amount"]


class ConfigLoader:

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        config = self._resolve_dynamic_paths(config)
        return config

    def _resolve_dynamic_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        model_paths = config.get("model_paths", {})
        exp_name = model_paths.get("exp_name", "")
        if not exp_name:
            return config

        base_path = model_paths.get("base_path", "")
        path_templates = {
            "base_save_path": f"{base_path}/{exp_name}",
            "finetuned_tokenizer": f"{base_path}/{exp_name}/tokenizer/best_model",
        }

        for key, template in path_templates.items():
            if key not in model_paths:
                continue
            current_value = model_paths[key]
            if current_value == "" or current_value is None:
                model_paths[key] = template
            elif isinstance(current_value, str) and "{exp_name}" in current_value:
                model_paths[key] = current_value.format(exp_name=exp_name)
        return config

    def get(self, key: str, default=None):
        keys = key.split(".")
        value: Any = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_data_config(self) -> Dict[str, Any]:
        return self.config.get("data", {})

    def get_training_config(self) -> Dict[str, Any]:
        return self.config.get("training", {})

    def get_model_paths(self) -> Dict[str, Any]:
        return self.config.get("model_paths", {})

    def get_model_config(self) -> Dict[str, Any]:
        return self.config.get("model", {})

    def get_experiment_config(self) -> Dict[str, Any]:
        return self.config.get("experiment", {})

    def get_device_config(self) -> Dict[str, Any]:
        return self.config.get("device", {})

    def get_distributed_config(self) -> Dict[str, Any]:
        return self.config.get("distributed", {})

    def get_inference_config(self) -> Dict[str, Any]:
        return self.config.get("inference", {})

    def update_config(self, updates: Dict[str, Any]):

        def update_nested_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_nested_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        self.config = update_nested_dict(self.config, updates)

    def save_config(self, save_path: str = None):
        if save_path is None:
            save_path = self.config_path

        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True, indent=2)

    def print_config(self):
        print("=" * 50)
        print("Current configuration:")
        print("=" * 50)
        print(yaml.dump(self.config, default_flow_style=False, allow_unicode=True, indent=2))
        print("=" * 50)


class CustomFinetuneConfig:

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

        self.loader = ConfigLoader(config_path)
        self._load_all_configs()

    @staticmethod
    def _load_json_config_from_path(path: Optional[str]) -> Dict[str, Any]:
        if not path:
            return {}
        cfg_path = os.path.join(path, "config.json")
        if not os.path.exists(cfg_path):
            return {}
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _normalize_feature_cols(feature_cols: Optional[List[str]]) -> List[str]:
        if not feature_cols:
            return DEFAULT_FEATURE_COLS.copy()
        normalized = []
        for col in feature_cols:
            if col == "volume":
                normalized.append("vol")
            else:
                normalized.append(col)
        return normalized

    def _infer_hidden_dim(self, requested_hidden_dim: Optional[int]) -> Optional[int]:
        predictor_cfg = self._load_json_config_from_path(self.pretrained_predictor_path)
        inferred_hidden_dim = predictor_cfg.get("d_model")
        if requested_hidden_dim is None:
            return inferred_hidden_dim
        if inferred_hidden_dim is not None and requested_hidden_dim != inferred_hidden_dim:
            raise ValueError(
                f"Configured hidden_dim={requested_hidden_dim} does not match "
                f"predictor config d_model={inferred_hidden_dim}."
            )
        return requested_hidden_dim

    def _load_all_configs(self):
        data_config = self.loader.get_data_config()
        self.data_path = data_config.get("data_path")
        self.stock_dir = data_config.get("stock_dir", self.data_path)
        self.indices_dir = data_config.get("indices_dir")
        self.merged_indices_path = data_config.get("merged_indices_path")
        self.lookback_window = data_config.get("lookback_window", data_config.get("lookback", 512))
        self.predict_window = data_config.get("predict_window", data_config.get("horizon", 48))
        self.lookback = self.lookback_window
        self.horizon = self.predict_window
        self.sample_stride = int(data_config.get("sample_stride", 5))
        self.max_context = data_config.get("max_context", self.lookback_window)
        self.clip = data_config.get("clip", 5.0)
        self.train_ratio = data_config.get("train_ratio", 0.9)
        self.val_ratio = data_config.get("val_ratio", 0.1)
        self.test_ratio = data_config.get("test_ratio", 0.0)

        self.stock_date_col = data_config.get("stock_date_col", data_config.get("date_col", "trade_date"))
        self.index_date_col = data_config.get("index_date_col", "date")
        self.feature_cols = self._normalize_feature_cols(data_config.get("feature_cols"))
        self.time_feature_list = ["minute", "hour", "weekday", "day", "month"]

        training_config = self.loader.get_training_config()
        optimizer_config = training_config.get("optimizer", {})
        scheduler_config = training_config.get("scheduler", {})

        self.tokenizer_epochs = training_config.get("tokenizer_epochs", training_config.get("epochs", 30))
        self.basemodel_epochs = training_config.get("basemodel_epochs", training_config.get("epochs", 30))
        self.batch_size = training_config.get("batch_size", 160)
        self.log_interval = training_config.get("log_interval", 50)
        self.num_workers = training_config.get("num_workers", 6)
        self.seed = training_config.get("seed", 100)
        self.tokenizer_learning_rate = training_config.get(
            "tokenizer_learning_rate",
            optimizer_config.get("tokenizer_lr", 2e-4),
        )
        self.predictor_learning_rate = training_config.get(
            "predictor_learning_rate",
            optimizer_config.get("lr", 4e-5),
        )
        self.mixed_precision = str(training_config.get("mixed_precision", "none")).strip().lower()
        self.adam_beta1 = training_config.get("adam_beta1", optimizer_config.get("beta1", 0.9))
        self.adam_beta2 = training_config.get("adam_beta2", optimizer_config.get("beta2", 0.95))
        self.adam_weight_decay = training_config.get("adam_weight_decay", optimizer_config.get("weight_decay", 0.1))
        self.accumulation_steps = training_config.get("accumulation_steps", 1)
        self.use_sampled_s1_for_s2 = training_config.get("use_sampled_s1_for_s2", False)
        self.token_cache_batch_size = training_config.get("token_cache_batch_size", self.batch_size)
        self.token_cache_num_workers = training_config.get("token_cache_num_workers", 0)
        self.optimizer_name = optimizer_config.get("name", "adamw")
        self.optimizer_config = optimizer_config
        self.scheduler_name = scheduler_config.get("name", "onecycle")
        self.scheduler_config = scheduler_config

        model_paths = self.loader.get_model_paths()
        model_config = self.loader.get_model_config()

        self.exp_name = model_paths.get("exp_name", model_config.get("exp_name", "default_experiment"))
        self.pretrained_tokenizer_path = model_config.get(
            "pretrained_tokenizer_path",
            model_paths.get("pretrained_tokenizer"),
        )
        self.pretrained_predictor_path = model_config.get(
            "pretrained_predictor_path",
            model_paths.get("pretrained_predictor"),
        )
        self.base_save_path = model_paths.get("base_save_path", model_config.get("base_save_path"))
        self.tokenizer_save_name = model_paths.get("tokenizer_save_name", "tokenizer")
        self.basemodel_save_name = model_paths.get("basemodel_save_name", "basemodel")
        self.finetuned_tokenizer_path = model_paths.get("finetuned_tokenizer")

        self.hidden_dim = self._infer_hidden_dim(model_config.get("hidden_dim"))
        self.lag_window = model_config.get("lag_window", 5)
        self.alpha = model_config.get("alpha", 0.2)
        self.freeze_stock_tokenizer = model_config.get("freeze_stock_tokenizer", True)
        self.freeze_index_tokenizer = model_config.get("freeze_index_tokenizer", True)
        self.index_ids = model_config.get("index_ids")
        self.num_indices = model_config.get("num_indices")
        self.no_early_pooling = model_config.get("no_early_pooling", True)
        self.cache_tokenized_sequences = training_config.get(
            "cache_tokenized_sequences",
            self.freeze_stock_tokenizer and self.freeze_index_tokenizer,
        )

        experiment_config = self.loader.get_experiment_config()
        self.experiment_name = experiment_config.get("name", "kronos_custom_finetune")
        self.experiment_description = experiment_config.get("description", "")
        self.use_comet = experiment_config.get("use_comet", False)
        self.train_tokenizer = experiment_config.get("train_tokenizer", True)
        self.train_basemodel = experiment_config.get("train_basemodel", True)
        self.skip_existing = experiment_config.get("skip_existing", False)

        unified_pretrained = experiment_config.get("pre_trained", None)
        self.pre_trained_tokenizer = experiment_config.get(
            "pre_trained_tokenizer",
            unified_pretrained if unified_pretrained is not None else True,
        )
        self.pre_trained_predictor = experiment_config.get(
            "pre_trained_predictor",
            unified_pretrained if unified_pretrained is not None else True,
        )

        device_config = self.loader.get_device_config()
        self.use_cuda = device_config.get("use_cuda", True)
        self.device_id = device_config.get("device_id", 0)

        distributed_config = self.loader.get_distributed_config()
        self.use_ddp = distributed_config.get("use_ddp", False)
        self.ddp_backend = distributed_config.get("backend", "nccl")

        inference_config = self.loader.get_inference_config()
        self.inference_temperature = inference_config.get("T", 1.0)
        self.inference_top_k = inference_config.get("top_k", 0)
        self.inference_top_p = inference_config.get("top_p", 0.9)
        self.inference_sample_count = inference_config.get("sample_count", 1)
        self.output_index_aux = inference_config.get("output_index_aux", False)

        self._compute_full_paths()

    def _compute_full_paths(self):
        self.tokenizer_save_path = os.path.join(self.base_save_path, self.tokenizer_save_name) if self.base_save_path else None
        self.tokenizer_best_model_path = os.path.join(self.tokenizer_save_path, "best_model") if self.tokenizer_save_path else None

        self.basemodel_save_path = os.path.join(self.base_save_path, self.basemodel_save_name) if self.base_save_path else None
        self.basemodel_best_model_path = os.path.join(self.basemodel_save_path, "best_model") if self.basemodel_save_path else None

    def get_tokenizer_config(self):
        return {
            "data_path": self.data_path,
            "lookback_window": self.lookback_window,
            "predict_window": self.predict_window,
            "max_context": self.max_context,
            "clip": self.clip,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "epochs": self.tokenizer_epochs,
            "batch_size": self.batch_size,
            "log_interval": self.log_interval,
            "num_workers": self.num_workers,
            "seed": self.seed,
            "learning_rate": self.tokenizer_learning_rate,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_weight_decay": self.adam_weight_decay,
            "accumulation_steps": self.accumulation_steps,
            "use_sampled_s1_for_s2": self.use_sampled_s1_for_s2,
            "pretrained_model_path": self.pretrained_tokenizer_path,
            "save_path": self.tokenizer_save_path,
            "use_comet": self.use_comet,
        }

    def get_basemodel_config(self):
        return {
            "data_path": self.data_path,
            "lookback_window": self.lookback_window,
            "predict_window": self.predict_window,
            "max_context": self.max_context,
            "clip": self.clip,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "epochs": self.basemodel_epochs,
            "batch_size": self.batch_size,
            "log_interval": self.log_interval,
            "num_workers": self.num_workers,
            "seed": self.seed,
            "predictor_learning_rate": self.predictor_learning_rate,
            "tokenizer_learning_rate": self.tokenizer_learning_rate,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_weight_decay": self.adam_weight_decay,
            "use_sampled_s1_for_s2": self.use_sampled_s1_for_s2,
            "pretrained_tokenizer_path": self.finetuned_tokenizer_path,
            "pretrained_predictor_path": self.pretrained_predictor_path,
            "pre_trained_predictor": self.pre_trained_predictor,
            "save_path": self.basemodel_save_path,
            "use_comet": self.use_comet,
        }

    def print_config_summary(self):
        print("=" * 60)
        print("Kronos finetuning configuration summary")
        print("=" * 60)
        print(f"Experiment name: {self.exp_name}")
        print(f"Data path: {self.data_path}")
        print(f"Stock dir: {self.stock_dir}")
        print(f"Indices dir: {self.indices_dir}")
        print(f"Merged indices path: {self.merged_indices_path}")
        print(f"Lookback window: {self.lookback_window}")
        print(f"Predict window: {self.predict_window}")
        print(f"Tokenizer training epochs: {self.tokenizer_epochs}")
        print(f"Basemodel training epochs: {self.basemodel_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Tokenizer learning rate: {self.tokenizer_learning_rate}")
        print(f"Predictor learning rate: {self.predictor_learning_rate}")
        print(f"Mixed precision: {self.mixed_precision}")
        print(f"Cache tokenized sequences: {self.cache_tokenized_sequences}")
        print(f"Token cache batch size: {self.token_cache_batch_size}")
        print(f"Use sampled s1 for s2 training: {self.use_sampled_s1_for_s2}")
        print(f"Train tokenizer: {self.train_tokenizer}")
        print(f"Train basemodel: {self.train_basemodel}")
        print(f"Skip existing: {self.skip_existing}")
        print(f"Use pre-trained tokenizer: {self.pre_trained_tokenizer}")
        print(f"Use pre-trained predictor: {self.pre_trained_predictor}")
        print(f"Base save path: {self.base_save_path}")
        print(f"Tokenizer save path: {self.tokenizer_save_path}")
        print(f"Basemodel save path: {self.basemodel_save_path}")
        print(f"Hidden dim: {self.hidden_dim}")
        print(f"Lag window: {self.lag_window}")
        print(f"Alpha: {self.alpha}")
        print(f"Freeze stock tokenizer: {self.freeze_stock_tokenizer}")
        print(f"Freeze index tokenizer: {self.freeze_index_tokenizer}")
        print(f"Number of indices: {self.num_indices}")
        print(f"No early pooling: {self.no_early_pooling}")
        print("=" * 60)
