from .kronos import Kronos, KronosPredictor, KronosTokenizer
from .craft import Craft, CraftBackbone, CraftPredictor, CraftTokenizer
from .multistream_craft import (
    IndexIdEmbedding,
    LaggedCrossAttentionTopAdapter,
    MultiIndexMemoryBuilder,
    MultiStreamCraftModel,
    build_craft_backbone,
    build_craft_from_config,
    extract_craft_init_config,
    load_craft_init_config,
)

model_dict = {
    "kronos_tokenizer": KronosTokenizer,
    "kronos": Kronos,
    "kronos_predictor": KronosPredictor,
    "craft": Craft,
    "craft_tokenizer": CraftTokenizer,
    "craft_predictor": CraftPredictor,
    "multistream_craft": MultiStreamCraftModel,
}


def get_model_class(model_name):
    if model_name in model_dict:
        return model_dict[model_name]
    print(f"Model {model_name} not found in model_dict")
    raise NotImplementedError
