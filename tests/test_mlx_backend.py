from __future__ import annotations

import sys
import types

import pytest

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

    def stream_generate(model, tokenizer, **kwargs):
        calls["stream_generate"] = kwargs
        yield types.SimpleNamespace(logprobs={7: -0.25, 8: -1.25})

    monkeypatch.setitem(
        sys.modules,
        "mlx_lm",
        types.SimpleNamespace(
            load=load,
            generate=generate,
            stream_generate=stream_generate,
        ),
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
    assert generator.first_token_logprobs("prompt", (7, 8), temperature=0.0) == {
        7: -0.25,
        8: -1.25,
    }
    assert calls["stream_generate"] == {
        "prompt": "prompt",
        "max_tokens": 1,
        "sampler": "sampler",
    }


def test_mlx_reranker_uses_binary_yes_probability(monkeypatch):
    import trialmatchai.models.llm.mlx_reranker as reranker_module

    class Generator:
        def __init__(self, *args, **kwargs):
            pass

        def first_token_id(self, text):
            return {"Yes": 7, "No": 8}[text]

        def format_messages(self, messages):
            return "formatted prompt"

        def first_token_logprobs(self, prompt, token_ids, *, temperature):
            assert prompt == "formatted prompt"
            assert token_ids == (7, 8)
            assert temperature == 0.0
            return {7: -0.25, 8: -1.25}

    monkeypatch.setattr(reranker_module, "MLXTextGenerator", Generator)
    reranker = reranker_module.MLXReranker("example")
    result = reranker.rank_pairs([("patient", "criterion")])[0]

    assert result["score_type"] == "binary_yes_probability"
    assert result["answer"] == "Yes"
    assert result["llm_score"] == pytest.approx(0.7310586)


def test_mlx_reranker_marks_missing_logprobs_unknown(monkeypatch):
    import trialmatchai.models.llm.mlx_reranker as reranker_module

    class Generator:
        def __init__(self, *args, **kwargs):
            pass

        def first_token_id(self, text):
            return {"Yes": 7, "No": 8}[text]

        def format_messages(self, messages):
            return "formatted prompt"

        def first_token_logprobs(self, prompt, token_ids, *, temperature):
            return {}

    monkeypatch.setattr(reranker_module, "MLXTextGenerator", Generator)
    reranker = reranker_module.MLXReranker("example")
    result = reranker.rank_pairs([("patient", "criterion")])[0]

    assert result == {
        "llm_score": 0.5,
        "answer": "Unknown",
        "score_type": "binary_yes_probability",
    }
