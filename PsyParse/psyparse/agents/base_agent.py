import copy
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import DEEPSEEK_API_KEY as _API_KEY, DEEPSEEK_BASE_URL as _BASE_URL, DEEPSEEK_MODEL as _MODEL

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=_API_KEY, base_url=_BASE_URL)
    return _client


class BaseAgent:
    def __init__(self, sys_prompt, model=_MODEL, temp=0.7):
        self.sys_prompt = sys_prompt
        self.model = model
        self.temp = temp
        # conversation state: list of {role, content}
        self.history = []

    # ----- state management -----

    def snapshot(self):
        return {
            "sys_prompt": self.sys_prompt,
            "history": copy.deepcopy(self.history),
            "temp": self.temp,
        }

    def restore(self, snap):
        self.sys_prompt = snap["sys_prompt"]
        self.history = copy.deepcopy(snap["history"])
        self.temp = snap["temp"]

    def clone(self):
        ag = self.__class__.__new__(self.__class__)
        ag.sys_prompt = self.sys_prompt
        ag.model = self.model
        ag.temp = self.temp
        ag.history = copy.deepcopy(self.history)
        return ag

    def swap_prompt(self, new_prompt):
        self.sys_prompt = new_prompt

    def get_history(self):
        return list(self.history)

    # ----- core API call -----

    def _call_api(self, messages, temp, max_retries=3, max_tokens=800):
        client = _get_client()
        delay = 10.0
        last_err = None
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tokens,  # ponytail: caller controls budget; default 800 fits Groq 8k TPM
                )
                tok = resp.usage
                if tok:
                    print(
                        f"[tokens] prompt={tok.prompt_tokens} "
                        f"completion={tok.completion_tokens} "
                        f"total={tok.total_tokens}"
                    )
                if not resp.choices:
                    raise ValueError("API returned empty/None choices")
                return resp.choices[0].message.content
            except Exception as e:
                last_err = e
                # safety filter check - log and let caller handle it
                if "content_filter" in str(e).lower() or "safety" in str(e).lower():
                    raise SafetyFilterError(str(e)) from e
                print(f"[retry {attempt+1}/{max_retries}] {e}, sleeping {delay}s")
                time.sleep(delay)
                delay *= 2
        raise RuntimeError(f"API failed after {max_retries} retries: {last_err}")

    def _build_messages(self, user_msg=None):
        msgs = [{"role": "system", "content": self.sys_prompt}]
        msgs += self.history
        if user_msg is not None:
            msgs.append({"role": "user", "content": user_msg})
        return msgs

    # ----- send (stateful, appends to history) -----

    def send(self, user_msg):
        msgs = self._build_messages(user_msg)
        reply = self._call_api(msgs, self.temp)
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": reply})
        return reply

    # ----- stateless generate (does NOT touch self.history) -----
    # parallel branches via threadpool, default n=4

    def generate(self, prompt, n=1, temp=None):
        t = temp if temp is not None else self.temp
        msgs = self._build_messages(prompt)

        if n == 1:
            return [self._call_api(msgs, t)]

        results = [None] * n

        def _worker(idx):
            return idx, self._call_api(msgs, t)

        with ThreadPoolExecutor(max_workers=n) as ex:
            futures = {ex.submit(_worker, i): i for i in range(n)}
            for fut in as_completed(futures):
                idx, txt = fut.result()
                results[idx] = txt

        assert len(results) == n, f"expected {n} branches, got {len(results)}"
        return results

    # ----- stateless single call with a one-off messages list -----

    def generate_from(self, messages, temp=None, max_tokens=800):
        t = temp if temp is not None else self.temp
        return self._call_api(messages, t, max_tokens=max_tokens)


class SafetyFilterError(Exception):
    # caller catches this, drops A/B pair, logs it, never falls back to ollama
    pass


# ----- smoke test -----
# 5 calls at T=0.9 should differ; 5 at T=0.0 should be near-identical
# run: python -m psyparse.agents.base_agent

if __name__ == "__main__":
    prompt = "Say a random single word."
    ag = BaseAgent("You are a test assistant.", temp=0.9)
    print("=== T=0.9 (expect variety) ===")
    outs_high = ag.generate(prompt, n=5)
    for o in outs_high:
        print(" ", o.strip())

    ag.temp = 0.0
    print("=== T=0.0 (expect near-identical) ===")
    outs_low = ag.generate(prompt, n=5)
    for o in outs_low:
        print(" ", o.strip())
