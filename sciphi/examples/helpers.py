"""Helper functions to be used across example scripts"""
import argparse
import json

from sciphi.interface import ProviderName


def build_llm_config(args: argparse.Namespace) -> dict:
    """Constructs the LLM config based on provided arguments."""

    config_args = {
        "provider_name": ProviderName(args.provider_name),
        "model_name": args.model_name,
        "model_path": args.model_path,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens_to_sample": args.max_tokens_to_sample,
        "do_sample": args.do_sample,
        "num_beams": args.num_beams,
        "do_stream": args.do_stream,
        "device": args.device,
        "version": args.version,
        "llama_data_dir": args.llama_data_dir,
        "llama_index_name": args.llama_index_name,
        "llama_out_store": args.llama_out_store,
        "llama_load_from_avail_store": args.llama_load_from_avail_store,
        "llama_chunk_size": args.llama_chunk_size,
        "llama_top_k_similarity": args.llama_top_k_similarity,
        "add_model_kwargs": json.loads(args.add_model_kwargs)
        if args.add_model_kwargs
        else None,
        "add_generation_kwargs": json.loads(args.add_generation_kwargs)
        if args.add_generation_kwargs
        else None,
        "add_tokenizer_kwargs": json.loads(args.add_tokenizer_kwargs)
        if args.add_tokenizer_kwargs
        else None,
        "functions": json.loads(args.functions) if args.functions else None,
    }

    # Filter out None values
    return {k: v for k, v in config_args.items() if v is not None}


def prep_for_file_path(in_path: str) -> str:
    """Prepare a string to be used in a file path."""
    return in_path.replace("-", "_").replace(".", "p").replace("/", "_")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Parse SciPhi running commands"
    )
    # Run arguments
    parser.add_argument(
        "--run_name",
        type=str,
        default="run_0",
        help="The name of the run.",
    )
    parser.add_argument(
        "--provider_name",
        type=str,
        default="openai",
        help="Which provider to use for zero-shot completions?",
    )
    # Base LLM arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-3.5-turbo",
        help="Model name to load from the provider.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Path to the model",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature parameter for provided model.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1,
        help="Top-p parameter for provided model.",
    )
    # Model dependent arguments (each defaults to None)
    parser.add_argument(
        "--top_k",
        type=float,
        default=None,
        help="Top-k parameter for provided model.",
    )
    parser.add_argument(
        "--max_tokens_to_sample",
        type=int,
        default=1_024,
        help="Max tokens to sample for each completion from the provided model.",
    )
    parser.add_argument(
        "--do_sample",
        type=bool,
        default=None,
        help="Should the model run with sampling?",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="The number of beams to use for beam search.",
    )
    parser.add_argument(
        "--do_stream",
        type=bool,
        default=None,
        help="Should the model run with streaming?",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Which device should the model run on (relevant for local models, like HuggingFace)?",
    )
    parser.add_argument(
        "--add_model_kwargs",
        type=str,
        default=None,
        help="Additional model kwargs for local model generation, in JSON format.",
    )
    parser.add_argument(
        "--add_tokenizer_kwargs",
        type=str,
        default=None,
        help="Additional tokenizer kwargs for local model generation, in JSON format.",
    )
    parser.add_argument(
        "--add_generation_kwargs",
        type=str,
        default=None,
        help="Additional generation kwargs for local model generation, in JSON format.",
    )
    parser.add_argument(
        "--functions",
        type=str,
        default=None,
        help="List of functions which conform to for OpenAI function calling format, in JSON format.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="0.1.0",
        help="Version of this run.",
    )
    # Llama data arguments
    parser.add_argument(
        "--llama_data_dir",
        type=str,
        default=None,
        help="Which directory to load the LlamaIndex data from?",
    )
    parser.add_argument(
        "--llama_index_name",
        type=str,
        default=None,
        help="Which directory to load the LlamaIndex data from?",
    )
    parser.add_argument(
        "--llama_out_store",
        type=str,
        default=None,
        help="Which directory save the LlamaIndex store to?",
    )
    parser.add_argument(
        "--llama_load_from_avail_store",
        type=bool,
        default=True,
        help="Should the LlamaIndex load from the available store?",
    )
    parser.add_argument(
        "--llama_chunk_size",
        type=int,
        default=None,
        help="What size should the LlamaIndex chunks be?",
    )
    parser.add_argument(
        "--llama_top_k_similarity",
        type=int,
        default=None,
        help="What top k should the LlamaIndex use for similarity?",
    )
    # Prompt arguments
    parser.add_argument(
        "--prompt_override",
        type=str,
        default="",
        help="Override the prompt type for the prompt with a comma separated list. The first entry is the prompt, followed by expected prompt inputs. E.g. `my_prompt requires {input},input`",
    )
    # Data generation arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of examples to generate per batch.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Optional path to the configuration path, if specified the example config is overridden.",
    )
    parser.add_argument(
        "--example_config",
        type=str,
        default="textbooks_are_all_you_need_basic_split",
        help="Which configuration to use for data generation?",
    )
    parser.add_argument(
        "--extra_output_file_text",
        default="",
        help="Extra text to append to the end of the string.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging verbosity level.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1_024,
        help="Number of samples to generate.",
    )
    parser.add_argument(
        "--output_file_name",
        default=None,
        help="Filename to override the default output file name with.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Directory to write generated output to.",
    )
    return parser.parse_args()
