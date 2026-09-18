"""P2: pure, GPU-free helpers in the model layer that had zero coverage — the
vLLM engine factory's type guards and the reranker's Yes/No probability math
(test/codebase audit)."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest


def test_vllm_type_guards():
    from trialmatchai.models.llm.vllm_loader import _as_float, _as_int, _as_str

    assert _as_str("model", "name") == "model"
    assert _as_str(None, "name") is None
    assert _as_int("5", "name") == 5
    assert _as_int(None, "name") is None
    assert _as_float("1.5", "name") == 1.5
    # the guard that prevents a float being used as a path/repo id
    with pytest.raises(TypeError):
        _as_str(1.0, "model_path")


def test_vllm_engine_disables_guided_json_whitespace(monkeypatch):
    """The whitespace bound must reach the engine build: in vLLM 0.23.0 only
    vllm_config.structured_outputs_config.disable_any_whitespace reaches xgrammar
    (the per-request StructuredOutputsParams field is read by nothing)."""
    from trialmatchai.models.llm import vllm_loader

    constructions = []
    vllm_stub = ModuleType("vllm")

    class _Engine:
        def __init__(self, **kwargs):
            constructions.append(kwargs)

        def get_tokenizer(self):
            return object()

    vllm_stub.LLM = _Engine
    monkeypatch.setitem(sys.modules, "vllm", vllm_stub)
    vllm_loader._ENGINE_CACHE.clear()
    try:
        vllm_loader.load_vllm_engine({"base_model": "fixture-model"}, {})
    finally:
        vllm_loader._ENGINE_CACHE.clear()

    assert len(constructions) == 1
    assert constructions[0]["structured_outputs_config"] == {
        "backend": "xgrammar",
        "disable_any_whitespace": True,
    }


def test_reranker_yes_probability_softmax():
    from trialmatchai.models.llm.llm_reranker import LLMReranker

    r = LLMReranker.__new__(LLMReranker)
    r.applicable_token_id = 1
    r.not_applicable_token_id = 2

    def out(yes=None, no=None):
        m = {}
        if yes is not None:
            m[1] = SimpleNamespace(logprob=yes)
        if no is not None:
            m[2] = SimpleNamespace(logprob=no)
        return SimpleNamespace(outputs=[SimpleNamespace(logprobs=[m])])

    assert r._yes_probability(out(yes=0.0, no=-10.0)) > 0.99  # yes dominates
    assert r._yes_probability(out(yes=-10.0, no=0.0)) < 0.01  # no dominates
    assert abs(r._yes_probability(out(yes=-1.0, no=-1.0)) - 0.5) < 1e-9  # tie
    assert r._yes_probability(out(yes=0.0, no=None)) == 1.0  # only yes present
    assert r._yes_probability(out()) == 0.0  # both missing -> 0
    assert r._yes_probability(SimpleNamespace(outputs=[])) == 0.0  # malformed -> 0
