from __future__ import annotations

import sys
import types

from trialmatchai.models.llm.mlx_loader import MLXTextGenerator


def test_mlx_loader_passes_revision_and_generation_options(monkeypatch):
    calls = {}

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return f"formatted:{messages[0]['content']}"

    def load(path, **kwargs):
        calls["load"] = (path, kwargs)
        return object(), Tokenizer()

    def make_sampler(temperature):
        calls["temperature"] = temperature
        return "sampler"

    def generate(model, tokenizer, **kwargs):
        calls["generate"] = kwargs
        return "answer"

    monkeypatch.setitem(
        sys.modules,
        "mlx_lm",
        types.SimpleNamespace(load=load, generate=generate),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.sample_utils",
        types.SimpleNamespace(make_sampler=make_sampler),
    )

    generator = MLXTextGenerator("mlx-community/example", revision="abc123")
    assert calls["load"] == (
        "mlx-community/example",
        {"tokenizer_config": None, "revision": "abc123"},
    )
    assert generator.format_messages([{"role": "user", "content": "hello"}]) == "formatted:hello"
    assert generator.generate_text("prompt", max_tokens=7, temperature=0.2) == "answer"
    assert calls["temperature"] == 0.2
    assert calls["generate"] == {
        "prompt": "prompt",
        "max_tokens": 7,
        "sampler": "sampler",
        "verbose": False,
    }
