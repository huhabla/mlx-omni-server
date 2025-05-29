from typing import Type

from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import get_model_path, load, load_config

from ..text_models import BaseTextModel
from .mlx_model import MLXModel
from .tools.chat_tokenizer import ChatTokenizer
from .tools.hugging_face import HuggingFaceChatTokenizer
from .tools.llama3 import Llama3ChatTokenizer
from .tools.mistral import MistralChatTokenizer


def load_tools_handler(model_type: str, tokenizer: TokenizerWrapper) -> ChatTokenizer:
    """Factory function to load appropriate tools handler based on model ID."""
    handlers: dict[str, Type[ChatTokenizer]] = {
        # Llama models
        "llama": Llama3ChatTokenizer,
        "mistral": MistralChatTokenizer,
        "qwen2": HuggingFaceChatTokenizer,
    }

    # Get handler class based on model ID or use Llama handler as default
    handler_class = handlers.get(model_type, HuggingFaceChatTokenizer)
    return handler_class(tokenizer)


def load_model(model_id: str, adapter_path: str = None) -> BaseTextModel:
    """Load a model and tokenizer from the given model ID."""
    # Parse cache path from model_id if present
    cache_path = None
    if "@" in model_id:
        model_id, cache_path = model_id.split("@", 1)

    model, tokenizer = load(
        model_id,
        tokenizer_config={"trust_remote_code": True},
        adapter_path=adapter_path,
    )

    model_path = get_model_path(model_id)
    config = load_config(model_path)

    chat_tokenizer = load_tools_handler(config["model_type"], tokenizer)

    mlx_model = MLXModel(model_id=model_id, model=model, tokenizer=chat_tokenizer)

    # Load pre-computed cache if path provided
    if cache_path:
        mlx_model.load_precomputed_cache(cache_path)

    return mlx_model

