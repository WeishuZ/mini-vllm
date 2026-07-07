"""Minimal OpenAI-compatible client for a local vLLM server.

Start a server first, for example:

    vllm serve Qwen/Qwen2.5-1.5B-Instruct
"""
from __future__ import annotations

from openai import OpenAI


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def main() -> None:
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "用三句话解释 vLLM 为什么快。"}],
        temperature=0.2,
        max_tokens=256,
    )
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()

