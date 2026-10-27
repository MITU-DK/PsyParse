import os
import re
import json
import time
import copy
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


# histories are stored from the patient's side: "user" = therapist, "assistant" = patient
def clean_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    res = []
    for m in history:
        txt = m["content"].replace("[SESSION_END]", "").strip()
        if not txt or txt.startswith("[Internal Instruction") or txt.startswith("[Instruction"):
            continue
        res.append({"speaker": "Therapist" if m["role"] == "user" else "Patient", "text": txt})
    return res


def format_dialogue(history: List[Dict[str, str]]) -> str:
    return "\n".join(f"{m['speaker']}: {m['text']}" for m in clean_history(history))


def flip_roles(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    # patient-side history -> therapist-side history (therapist = assistant)
    res = []
    for m in clean_history(history):
        role = "assistant" if m["speaker"] == "Therapist" else "user"
        res.append({"role": role, "content": m["text"]})
    return res


def has_cjk(txt: str) -> bool:
    # qwen sometimes drifts into chinese in long chats
    return bool(re.search(r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]", txt or ""))


_warned = set()


def _check_limits(completion, ctx_limit: int, max_tokens: int) -> None:
    # ollama silently drops the start of prompts longer than its context, so at least shout about it
    try:
        used = completion.usage.prompt_tokens + completion.usage.completion_tokens
        if used > 0.9 * ctx_limit and "ctx" not in _warned:
            _warned.add("ctx")
            print(f"  [warn] a call used {used} tokens, close to CTX_LIMIT={ctx_limit}. "
                  "start ollama with a bigger OLLAMA_CONTEXT_LENGTH and set CTX_LIMIT to match")
        if completion.choices[0].finish_reason == "length" and "len" not in _warned:
            _warned.add("len")
            print(f"  [warn] a reply hit max_tokens={max_tokens} and was cut, raise MAX_TOKENS if this keeps happening")
    except Exception:
        pass


class BaseAgent:
    """
    Core LLM Agent with state management, exponential backoff retry logic,
    stateless branch generation, and chat-template normalization for local Ollama / DeepSeek.
    """
    def __init__(
        self,
        system_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ):
        import config
        self.system_prompt = system_prompt
        self.model = model or config.DEEPSEEK_MODEL
        self.temperature = temperature
        self.max_tokens = config.MAX_TOKENS
        self.ctx_limit = config.CTX_LIMIT
        self.reasoning_effort = config.REASONING_EFFORT if reasoning_effort is None else reasoning_effort
        
        resolved_base_url = base_url or config.DEEPSEEK_BASE_URL
        resolved_api_key = api_key or config.DEEPSEEK_API_KEY
        
        self.client = OpenAI(base_url=resolved_base_url, api_key=resolved_api_key)
        self.history: List[Dict[str, str]] = []

    def get_history(self) -> List[Dict[str, str]]:
        return copy.deepcopy(self.history)

    def set_history(self, history: List[Dict[str, str]]) -> None:
        self.history = copy.deepcopy(history)

    def clear_history(self) -> None:
        self.history = []

    def snapshot(self) -> List[Dict[str, str]]:
        return copy.deepcopy(self.history)

    def restore(self, snapshot: List[Dict[str, str]]) -> None:
        self.history = copy.deepcopy(snapshot)

    def swap_system_prompt(self, new_prompt: str) -> None:
        self.system_prompt = new_prompt

    def add_message(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def _build_payload(
        self,
        history: List[Dict[str, str]],
        ephemeral_system: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        sys_content = ephemeral_system or self.system_prompt
        payload = []
        is_gemma = "gemma" in self.model.lower()

        if is_gemma:
            if history:
                first_turn = copy.deepcopy(history[0])
                first_turn["content"] = f"[INSTRUCTION: {sys_content}]\n\n{first_turn['content']}"
                payload = [first_turn] + history[1:]
            else:
                payload = [{"role": "user", "content": f"[INSTRUCTION: {sys_content}]"}]
        else:
            payload = [{"role": "system", "content": sys_content}] + history

        return payload

    def send(
        self,
        user_message: str,
        temperature: Optional[float] = None,
        max_retries: int = 15,
        ephemeral_system: Optional[str] = None,
    ) -> str:
        self.history.append({"role": "user", "content": user_message})
        messages = self._build_payload(self.history, ephemeral_system)
        temp = self.temperature if temperature is None else temperature

        response = self._call_api_with_retry(messages, temp, max_retries)
        self.history.append({"role": "assistant", "content": response})
        return response

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_retries: int = 15,
        ephemeral_system: Optional[str] = None,
        response_format: Optional[Dict[str, str]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        payload = self._build_payload(messages, ephemeral_system)
        temp = self.temperature if temperature is None else temperature
        return self._call_api_with_retry(payload, temp, max_retries, response_format, max_tokens)

    def _call_api_with_retry(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_retries: int = 15,
        response_format: Optional[Dict[str, str]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        delay = 4.0
        last_exception = None

        for attempt in range(max_retries):
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": max(0.0, min(temperature, 1.0)),
                    # cap output so a rambling model can't eat minutes on one reply
                    "max_tokens": max_tokens or self.max_tokens,
                }
                if response_format:
                    kwargs["response_format"] = response_format
                if self.reasoning_effort:
                    kwargs["reasoning_effort"] = self.reasoning_effort

                completion = self.client.chat.completions.create(**kwargs)
                content = completion.choices[0].message.content or ""
                _check_limits(completion, self.ctx_limit, kwargs["max_tokens"])
                return content.strip()
            except Exception as e:
                last_exception = e
                logger.warning("LLM API attempt %d failed: %s. Retrying in %.1fs...", attempt + 1, e, delay)
                time.sleep(delay)
                delay = min(delay * 2.0, 60.0)

        raise RuntimeError(f"API call failed after {max_retries} attempts: {last_exception}")
