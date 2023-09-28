"""A module for managing local vLLM models."""

from dataclasses import dataclass

from sciphi.core import ProviderName
from sciphi.llm.base import LLM, LLMConfig
from sciphi.llm.config_manager import model_config


@model_config
@dataclass
class LlamaCPPConfig(LLMConfig):
    """Configuration for local llama_cpp models."""

    # Base
    provider_name: ProviderName = ProviderName.LLAMA_CPP
    model_name: str = "llama-v2-7b"
    model_path: str = ""
    temperature: float = 0.8
    top_p: float = 0.95

    # Generation parameters
    top_k: int = 40
    max_tokens_to_sample: int = 128


class LlamaCPP(LLM):
    """Configuration for local llama_cpp models."""

    def __init__(
        self,
        config: LlamaCPPConfig,
    ) -> None:
        super().__init__(config)
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "Please install the llama_cpp_python package before attempting to run with an vLLM model. This can be accomplished via `poetry install -E vllm_support, ...OTHER_DEPENDENCIES_HERE`."
            )
        assert config.model_path is not None
        self.model = Llama(model_path=config.model_path)
        self.sampling_params = dict(
            temperature=config.temperature,
            max_tokens=config.max_tokens_to_sample,
            top_p=config.top_p,
            top_k=config.top_k,
        )

    def get_chat_completion(self, messages: list[dict[str, str]]) -> str:
        """Get a completion from the OpenAI API based on the provided messages."""
        raise NotImplementedError(
            "Chat completion not yet implemented for vLLM."
        )

    def get_instruct_completion(self, prompt: str) -> str:
        """Get an instruction completion from local vLLM API."""
        outputs = self.model(prompt, **self.sampling_params)
        return outputs['choices'][0]['text']

    def get_batch_instruct_completion(self, prompts: list[str]) -> list[str]:
        """Get batch instruction completion from local llama_cpp.
        It doesn't work with a batch, so output the same as before"""
        assert len(prompts) == 1, 'Batch size must be one'
        return [self.get_instruct_completion(prompts[0])]
