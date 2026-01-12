"""
HuggingFace LLM class
"""

from copy import copy, deepcopy
from typing import Any

# import ast
import json
from json import JSONDecodeError
from rich import print as rich_print

from transformers import AutoTokenizer, TextStreamer

from .base import LLM
from .types import ActionFromLLM


class HuggingFaceLLM(LLM):
    """
    HuggingFace LLM class
    """

    def __init__(
        self,
        model: Any | None = None,
        model_config: dict[str, Any] | None = None,
        tokenizer: AutoTokenizer | None = None,
        stream_generation: bool = False,
        **kwargs: Any,
    ) -> None:
        assert model is not None and model_config is not None, (
            "model and model_config must be provided"
        )
        model_name = model_config["pretrained_model_name_or_path"]
        super().__init__(model=model_name, model_config=model_config, **kwargs)
        self.model = model

        # Get tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            _model_name_or_path = model_config["pretrained_model_name_or_path"]
            _kwargs = {
                k: v
                for k, v in model_config.items()
                if k != "pretrained_model_name_or_path"
            }
            if _model_name_or_path == "Qwen/Qwen3-8B":  # hack but get Qwen2.5 tokenizer
                _model_name_or_path = "Qwen/Qwen2.5-3B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(_model_name_or_path, **_kwargs)
        
        # Maybe unnecessary, but specify how tool calls appear
        if "llama" in model_name:
            self.func_call_argname = "parameters"  # llama3 uses parameters
        else:
            self.func_call_argname = "arguments"  # qwen uses arguments

        # Stream tokens as they generate
        if stream_generation:
            self.streamer = TextStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
        else:
            self.streamer = None

    def sample(
        self,
        system_prompt: str | None = None,
        messages: list[str] | list[list[dict[str, Any]]] | None = None,
        tools: list[dict[str, Any]] | list[list[dict[str, Any]]] | None = None,
        max_new_tokens: int = 1024,
        num_return_sequences: int = 1,
        verbose: bool = False,
        **generation_kwargs: Any,
    ) -> list[list[dict[str, Any]]]:
        """
        Generate text from a prompt
        """
        if isinstance(messages, list) and not isinstance(messages[0], list):
            messages = [messages]
        elif messages is None:
            messages = [[{"role": "user", "content": ""}]]

        if system_prompt is not None:
            messages = [
                [{"role": "system", "content": system_prompt}] + single_chat
                for single_chat in messages
            ]
        # Get model inputs
        if isinstance(tools, list) and not isinstance(tools[0], list):
            model_inputs = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                # enable_thinking=True,
                enable_thinking=False,
            )
        else:
            og_padding_side = copy(self.tokenizer.padding_side)
            self.tokenizer.padding_side = "left"
            # Apply chat template to each sample in batch, as tools may be different per sample
            input_texts = [
                self.tokenizer.apply_chat_template(
                    messages[_idx],
                    tools=_tools,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                )
                for _idx, _tools in enumerate(tools)
            ]
            # for _idx, _input_text in enumerate(input_texts):
            #     rich_print(f"[{_idx}]\n---\n{_input_text}\n============================\n")
            model_inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True)
            self.tokenizer.padding_side = og_padding_side
        
        # Get input lengths
        # input_lens = model_inputs["attention_mask"].sum(dim=1)  # (batch_size,)
        input_len = model_inputs["input_ids"].shape[1]
        # Get generation config
        generation_config = (
            generation_kwargs
            if generation_kwargs is not None
            else self.generation_config
        )
        if generation_config.get("pad_token_id", None) is None:
            generation_config["pad_token_id"] = self.tokenizer.pad_token_id

        # Generate and decode
        outputs = self.model.generate(
            **model_inputs.to(self.model.device),
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            streamer=self.streamer,  # For silly visuals
            **generation_config,
        )
        # Track tokens hack
        # -> Also not correct if messages is a batch of messages
        # outputs.prompt_tokens = input_lens.sum()
        # outputs.completion_tokens = outputs.shape[0] * outputs.shape[1] - input_lens.sum()
        outputs.prompt_tokens = input_len
        outputs.completion_tokens = outputs.shape[1] - input_len
        self._track_tokens(outputs)

        # Decode and convert tokens to messages
        decoded_texts = self.tokenizer.batch_decode(
            outputs[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        # decoded_texts = [
        #     self.tokenizer.decode(
        #         output[input_lens[i]:], skip_special_tokens=True, clean_up_tokenization_spaces=True,
        #     ) for i, output in enumerate(outputs)
        # ]
        # breakpoint()
        # MZ Hack 10/31/2025, only allow single tool call
        decoded_texts = [
            text.split("</tool_call>")[0] + "</tool_call>"
            if "<tool_call>" in text
            else text
            for text in decoded_texts
        ]
        if verbose:
            for _text in decoded_texts:
                rich_print(f"{_text}\n{"-" * 100}")
        return [
            [{"role": "assistant", "content": message}] for message in decoded_texts
        ]

    def update_messages(
        self,
        messages: list[dict[str, Any]],
        model_response: list[dict[str, Any]] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        interleave: bool = False,
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Update the messages with the model response
        """
        if system_prompt is None:
            system_messages: list[dict[str, Any]] = []
        else:
            system_messages = [{"role": "system", "content": system_prompt}]

        if prior_messages is None:
            prior_messages: list[dict[str, Any]] = []

        # We build up new messages, then prepend
        # system prompt (optional) and prior messages
        if interleave or model_response is None:
            new_messages: list[dict[str, Any]] = []
        else:
            new_messages = deepcopy(model_response)

        for idx, message in enumerate(messages):
            if interleave and model_response is not None:
                new_messages.append(model_response[idx])
            if message.get("type", None) == "function_call_output":
                new_messages.append({"role": "tool", "content": message["output"]})
            else:
                new_messages.append(message)
        return system_messages + prior_messages + new_messages

    def get_actions(self, response: list[dict[str, Any]]) -> list[ActionFromLLM]:
        """
        Process response from HuggingFace LLM

        `response` is a list of standard chat messages, i.e.,
        [
            {"role": "assistant", "content": content},
            {"role": "assistant", "tool_calls": [{"type": "function", "function": {}}]}
        ]
        """
        action_list = []
        # HF Transformers output thoughts and tool calls in the same message
        # -> Split into separate messages
        try:
            response = self.get_messages_from_text(response[0]["content"])
        except Exception as e:
            rich_print(f"[red]Error[/red] in get_messages_from_text: [red]{e}[/red]")
            breakpoint()
        for message in response:
            if message.get("tool_calls", None) is not None:
                for tool_call in message["tool_calls"]:
                    output = tool_call["function"]
                    try:
                        name = output["name"]
                    except KeyError as e:
                        rich_print(f"[red]KeyError[/red]: [red]{e}[/red]")
                        name = "invalid_tool_call"
                    try:
                        arguments = output[self.func_call_argname]
                        if not isinstance(arguments, dict):
                            arguments = {"arguments": json.dumps(arguments)}
                    except KeyError as e:
                        rich_print(f"[red]KeyError[/red]: [red]{e}[/red]")
                        arguments = {"arguments": json.dumps({})}
                    text_repr = json.dumps(output)
                    try:
                        action_list.append(
                            ActionFromLLM(
                                role="assistant",
                                type="function_call",
                                text=text_repr,
                                call_id=None,  # none by default
                                name=name,
                                arguments=arguments,
                            )
                        )
                    except Exception as e:
                        rich_print(f"[red]Exception[/red]: [red]{e}[/red]")
                        breakpoint()
            else:  # regular message
                action_list.append(
                    ActionFromLLM(
                        role="assistant",
                        type="message",
                        text=message["content"],
                        call_id=None,
                        name=None,
                        arguments=None,
                    )
                )
        return action_list

    def get_messages_from_text(
        self,
        text: str,
        tool_call_bos: str = "<tool_call>",
        tool_call_eos: str = "</tool_call>",
        tool_call_argname: str = "arguments",
    ) -> list[dict[str, Any]]:
        """
        Convert text to LLM chat messages
        """
        messages = []
        try:
            tool_call_str = text.split(tool_call_bos)[-1].split(tool_call_eos)[0]
            tool_call = json.loads(tool_call_str)
            valid_tool_call = True
        except JSONDecodeError:
            valid_tool_call = False

        if valid_tool_call:
            if isinstance(tool_call, str):
                valid_tool_call = False
            else:
                try:
                    assert tool_call.get("name", None) is not None
                    assert tool_call.get(self.func_call_argname, None) is not None
                except AssertionError:
                    if tool_call.get("arguments", None) is not None:
                        self.func_call_argname = "arguments"
                    else:
                        valid_tool_call = False

        # Convert any text before to regular message
        message = text.split(tool_call_bos)[0].strip()
        if len(message) > 0:
            messages.append({"role": "assistant", "content": message})

        if valid_tool_call:
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [{"type": "function", "function": tool_call}],
                }
            )
        elif tool_call_bos in text:  # Invalid tool call
            try:
                _invalid_tool_call_text = text.split(tool_call_bos)[-1].strip()
                assert _invalid_tool_call_text != "", "Invalid tool call"
                try:
                    _invalid_tool_call_text = _invalid_tool_call_text.split(tool_call_eos)[0].strip()
                    assert _invalid_tool_call_text != "", "Invalid tool call"
                except AssertionError:
                    pass
            except AssertionError:
                _invalid_tool_call_text = text.strip()

            _invalid_tool_call = {
                "name": "invalid_tool_call",
                "arguments": _invalid_tool_call_text,
            }
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [{"type": "function", "function": _invalid_tool_call}],
                }
            )
        return messages
