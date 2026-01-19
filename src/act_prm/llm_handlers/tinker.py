"""
Our version of Tinker completers

Copied and modified from tinker_cookbook/completers.py (v0.1.0)

Maybe this permalink?
https://github.com/thinking-machines-lab/tinker-cookbook/blob/22483a6b04400f79da13557a8229bc98b309b026/tinker_cookbook/completers.py
"""

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Any, TypeAlias

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.renderers import Message, Renderer
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .action_utils import get_actions, get_messages_from_text
from .base import LLM
from .types import ActionFromLLM

# Interfaces
StopCondition: TypeAlias = list[str] | list[int]

logger = logging.getLogger(__name__)


@dataclass
class TokensWithLogprobs:
    """
    Object to store tokens and logprobs for single model generation
    """
    tokens: list[int]
    maybe_logprobs: list[float] | None

    @property
    def logprobs(self) -> list[float]:
        """
        Retrieve logprobs from object
        """
        if self.maybe_logprobs is None:
            raise ValueError("Logprobs are not available")
        return self.maybe_logprobs


@dataclass
class TokensWithLogprobsAndText(TokensWithLogprobs):
    """
    Object to store tokens, logprobs, and text content from single model generation
    """
    text: str
    is_complete: bool


class TinkerCompleter:
    """
    Base class for async Tinker completers (model generation)
    """
    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        renderer: Renderer,
        max_tokens: int,
        temperature: float = 1.0,
        stop_condition: StopCondition | None = None,
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
        tool_call_bos: str = "<tool_call>",
        tool_call_eos: str = "</tool_call>",
        tool_call_argname: str = "arguments",
    ) -> None:
        self.sampling_client = sampling_client
        self.renderer = renderer
        self.hf_tokenizer = hf_tokenizer

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stop_condition = stop_condition or self.renderer.get_stop_sequences()

        self.tool_call_parse_kwargs: dict[str, str] = {
            "tool_call_bos": tool_call_bos,
            "tool_call_eos": tool_call_eos,
            "tool_call_argname": tool_call_argname,
        }

    async def __call__(
        self,
        **kwargs: Any,
    ) -> TokensWithLogprobs | None:
        """
        Generate tokens (see self.generate for class-specific implementation)
        """
        return await self.generate(**kwargs)

    async def generate(
        self,
        model_input: tinker.ModelInput,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: StopCondition | None = None,
    ) -> TokensWithLogprobs | None:
        """
        Generate tokens from model input
        """
        # Sample from model
        sampling_params = tinker.SamplingParams(
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
            stop=stop or self.stop_condition,
        )
        try:
            response = await self.sampling_client.sample_async(
                model_input,
                num_samples=1,
                sampling_params=sampling_params,
            )
            # Extract tokens and logprobs from the first (and only) sample
            sampled_tokens = response.sequences[0].tokens
            sampled_logprobs = response.sequences[0].logprobs
            assert sampled_logprobs is not None
            # Decode the response
            parsed_message, is_complete = self.renderer.parse_response(sampled_tokens)
            text_content = get_text_content(parsed_message)
            
            return TokensWithLogprobsAndText(
                tokens=sampled_tokens,
                maybe_logprobs=sampled_logprobs,
                text=text_content,
                is_complete=is_complete
            )
        # except tinker.BadRequestError as e:
        except Exception as e:
            _exception_class = type(e).__name__
            logger.error(f"[red]Tinker {_exception_class}: {e}[/red]")
            return None
        

    async def get_actions(self, response: list[dict[str, Any]]) -> list[ActionFromLLM]:
        """
        Parse response into list of actions
        """
        return get_actions(response, **self.tool_call_parse_kwargs)

    async def get_messages_from_text(self, text: str) -> list[dict[str, Any]]:
        """
        Parse text into list of messages
        """
        return get_messages_from_text(text, **self.tool_call_parse_kwargs)

    async def compute_logprobs_async(self, model_input: tinker.ModelInput) -> list[float]:
        """
        Compute logprobs for a model input
        """
        return await self.sampling_client.compute_logprobs_async(model_input)


def get_text_content(message: Message, remove_thinking: bool = False) -> str:
    """
    Extract text content from message, optionally stripping thinking parts.
    """
    content = message["content"]
    if isinstance(content, str):
        return content
    if remove_thinking:
        return "".join(p["text"] for p in content if p["type"] == "text")
    return "\n".join(p["text"] for p in content)


def _load_tokenizer(
    pretrained_model_name_or_path: str,
    chat_template_path: str | None = None,
    **kwargs: Any,
) -> PreTrainedTokenizerBase:
    """
    Load tokenizer with optional chat template override.
    """
    if pretrained_model_name_or_path == "Qwen/Qwen3-8B":  # hack but get Qwen2.5 tokenizer
        pretrained_model_name_or_path = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
    if chat_template_path is not None:
        with open(chat_template_path, "r", encoding="utf-8") as f:
            tokenizer.chat_template = f.read()
            logger.info("-> Overriding chat template with %s", chat_template_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _run_coroutine_blocking(coro: Any) -> Any:
    """
    Run an async coroutine from sync code without deadlocking a running event loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_container: dict[str, Any] = {}
    error_container: dict[str, BaseException] = {}

    def _runner() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_container["result"] = loop.run_until_complete(coro)
        except BaseException as e:
            error_container["error"] = e
        finally:
            loop.close()

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if error_container:
        raise error_container["error"]
    return result_container.get("result")


class TinkerLLM(LLM):
    """
    Tinker-backed LLM wrapper that conforms to the local LLM interface.
    """
    def __init__(
        self,
        model: str | None = None,
        model_config: dict[str, Any] | None = None,
        generation_config: dict[str, Any] | None = None,
        sampling_client: tinker.SamplingClient | None = None,
        base_url: str | None = None,
        renderer_name: str | None = None,
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
        tool_call_bos: str = "<tool_call>",
        tool_call_eos: str = "</tool_call>",
        tool_call_argname: str = "arguments",
        **kwargs: Any,
    ) -> None:
        model_config = model_config or {}
        super().__init__(model=model or model_config.get("model_name", ""), generation_config=generation_config, **kwargs)
        self.model_name = (
            model_config.get("model_name")
            or model_config.get("base_model")
            or model
            or model_config.get("pretrained_model_name_or_path", "")
        )
        self.base_url = base_url or model_config.get("base_url")

        tokenizer_name_or_path = (
            model_config.get("tokenizer_name_or_path")
            or model_config.get("pretrained_model_name_or_path")
            or self.model_name
        )
        if hf_tokenizer is None and tokenizer_name_or_path:
            _chat_template_path = model_config.get("chat_template_path")
            _tokenizer_kwargs = model_config.get("tokenizer_kwargs", {})
            hf_tokenizer = _load_tokenizer(
                tokenizer_name_or_path,
                chat_template_path=_chat_template_path,
                **_tokenizer_kwargs,
            )
        self.hf_tokenizer = hf_tokenizer

        if renderer_name is None:
            renderer_name = model_info.get_recommended_renderer_name(self.model_name)
        self.renderer = renderers.get_renderer(renderer_name, self.hf_tokenizer)
        self.stop_condition = model_config.get("stop_condition", self.renderer.get_stop_sequences())

        if sampling_client is None:
            if self.base_url is None:
                raise ValueError("base_url must be provided when sampling_client is not set")
            service_client = tinker.ServiceClient(base_url=self.base_url)
            sampler_path = model_config.get("sampler_path")
            if sampler_path:
                sampling_client = service_client.create_sampling_client(model_path=sampler_path)
            else:
                sampling_client = service_client.create_sampling_client(base_model=self.model_name)
        self.sampling_client = sampling_client

        self.tool_call_parse_kwargs: dict[str, str] = {
            "tool_call_bos": tool_call_bos,
            "tool_call_eos": tool_call_eos,
            "tool_call_argname": tool_call_argname,
        }

    def sample(
        self,
        system_prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_new_tokens: int = 1024,
        num_return_sequences: int = 1,
        **generation_kwargs: Any,
    ) -> list[list[dict[str, Any]]]:
        """
        Generate text from a prompt using a Tinker sampling client.
        """
        if messages is None:
            messages = [{"role": "user", "content": ""}]
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        if self.hf_tokenizer is None:
            raise ValueError("hf_tokenizer must be set to build model inputs")

        input_ids = self.hf_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            tools=tools,
        )
        model_input = tinker.ModelInput.from_ints(input_ids)
        sampling_params = tinker.SamplingParams(
            max_tokens=max_new_tokens,
            temperature=generation_kwargs.get("temperature", self.generation_config.get("temperature", 1.0)),
            stop=generation_kwargs.get("stop", self.stop_condition),
        )

        response = _run_coroutine_blocking(
            self.sampling_client.sample_async(
                model_input,
                num_samples=num_return_sequences,
                sampling_params=sampling_params,
            )
        )
        outputs: list[list[dict[str, Any]]] = []
        for sequence in response.sequences:
            parsed_message, _ = self.renderer.parse_response(sequence.tokens)
            text_content = get_text_content(parsed_message)
            outputs.append([{"role": "assistant", "content": text_content}])
        return outputs

    def get_actions(self, response: list[dict[str, Any]]) -> list[ActionFromLLM]:
        """
        Process response into list of actions.
        """
        return get_actions(response, **self.tool_call_parse_kwargs)

    def get_messages_from_text(self, text: str) -> list[dict[str, Any]]:
        """
        Parse text into list of messages.
        """
        return get_messages_from_text(text, **self.tool_call_parse_kwargs)
