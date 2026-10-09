import os
import re
import json
import time
import copy
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

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
    ):
        from psyparse import config
        self.system_prompt = system_prompt
        self.model = model or config.DEEPSEEK_MODEL
        self.temperature = temperature
        
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
        max_retries: int = 3,
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
        max_retries: int = 3,
        ephemeral_system: Optional[str] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        payload = self._build_payload(messages, ephemeral_system)
        temp = self.temperature if temperature is None else temperature
        return self._call_api_with_retry(payload, temp, max_retries, response_format)

    def _call_api_with_retry(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_retries: int = 3,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        delay = 1.0
        last_exception = None

        for attempt in range(max_retries):
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": max(0.0, min(temperature, 1.0)),
                }
                if response_format:
                    kwargs["response_format"] = response_format

                completion = self.client.chat.completions.create(**kwargs)
                content = completion.choices[0].message.content or ""
                return content.strip()
            except Exception as e:
                last_exception = e
                logger.warning("LLM API attempt %d failed: %s. Retrying in %.1fs...", attempt + 1, e, delay)
                time.sleep(delay)
                delay *= 2.0

        raise RuntimeError(f"API call failed after {max_retries} attempts: {last_exception}")
