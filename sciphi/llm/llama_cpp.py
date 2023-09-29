"""A module for managing local vLLM models."""
import json
import logging
import urllib.parse

import requests
from time import sleep
from dataclasses import dataclass

from sciphi.core import ProviderName
from sciphi.llm.base import LLM, LLMConfig
from sciphi.llm.config_manager import model_config

logger = logging.getLogger(__name__)

# Llama-2 has context length of 4096 token
# 1 token = ~4 characters, so 3500 * 4 provides plenty of room.
MAX_TOKENS = 3300 * 4
# Empirically found to be the cutoff of specific questions vs. generic comments about previous answer
SEARCH_THRESHOLD = 10
NUM_HOLOSCAN_DOCS = 7
LLAMA_SERVER = "http://localhost:8080"


@model_config
@dataclass
class LlamaCPPConfig(LLMConfig):
    """Configuration for local llama_cpp models."""

    # Base
    provider_name: ProviderName = ProviderName.LLAMA_CPP
    model_name: str = "llama-v2-7b"
    model_path: str = ""
    temperature: float = 0.8
    top_p: float = 0.75
    n_gpu_layers: int = 0

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
        assert config.model_path is not None
        self._wait_for_server()
        self.config = config

    def _stream_ai_response(self, llama_prompt):
        request_data = {
            "prompt": llama_prompt,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "n_predict": self.config.max_tokens_to_sample,
            "stop": ["</s>"],
            "n_keep": -1,
        }
        resData = requests.request(
            "POST",
            urllib.parse.urljoin(LLAMA_SERVER, "/completion"),
            data=json.dumps(request_data),
            stream=False,
        )
        return resData.json().get("content")

    def get_chat_completion(self, messages: list[dict[str, str]]) -> str:
        """Get a completion from the OpenAI API based on the provided messages."""
        raise NotImplementedError(
            "Chat completion not yet implemented for vLLM."
        )

    def get_instruct_completion(self, prompt: str) -> str:
        """Get an instruction completion from local LlamaCPP model."""
        return self._stream_ai_response(prompt)

    def get_batch_instruct_completion(self, prompts: list[str]) -> list[str]:
        """Get batch instruction completion from local llama_cpp.
        It doesn't work with a batch, so output the same as before"""
        assert len(prompts) == 1, 'Batch size must be one'
        return [self.get_instruct_completion(prompts[0])]

    @staticmethod
    def _wait_for_server():
        attempts = 0
        while attempts < SEARCH_THRESHOLD:
            try:
                response = requests.get(LLAMA_SERVER)
                # Check for a successful response status code (e.g., 200 OK)
                if response.status_code == 200:
                    logger.debug(
                        f"Connected to Llama.cpp server on {LLAMA_SERVER}"
                    )
                    return
            except requests.ConnectionError:
                logger.debug("Connection failed. Attempting to connect in 20 seconds")
                sleep(20)
            attempts += 1
        raise requests.ConnectionError("Unable to connect to Llama.cpp server")
