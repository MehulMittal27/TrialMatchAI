

def test_expansion_shares_the_eligibility_engine_when_running_base_weights():
    """One phi-4 engine, not two, when expansion runs the unadapted model.

    The engine cache is keyed on the resolved adapter path. Overwriting the model section's
    ``cot_adapter_path`` with None - which is what ``query_expansion.adapter: null`` used to
    do - gave expansion a different key and built a SECOND copy of the same multi-GB base
    model. Summed engine budgets then exceeded the card: reranker 0.22 + expansion 0.60 +
    eligibility 0.60 = 1.42 of total GPU memory, which cannot fit any single GPU because
    vLLM's gpu_memory_utilization is a fraction of the card, not an absolute size.

    Base weights are selected per request instead, by omitting the LoRA at generate time on
    an engine already built with ``enable_lora=True``.
    """
    from trialmatchai.matching.query_expansion import QueryExpander

    captured = {}

    def fake_loader(*, model_config, vllm_cfg):
        captured["model_config"] = dict(model_config)
        return ("engine", "tokenizer", "LORA_REQUEST")

    config = {
        "model": {
            "base_model": "microsoft/phi-4",
            "cot_adapter_path": "models/trialmatchai-phi4-reasoning-lora",
        },
        "rag": {"backend": "vllm"},
        "query_expansion": {"enabled": True, "backend": "vllm", "adapter": None},
        "vllm": {},
    }

    import trialmatchai.models.llm.vllm_loader as loader_module

    original = loader_module.load_vllm_engine
    loader_module.load_vllm_engine = fake_loader
    try:
        expander = QueryExpander.__new__(QueryExpander)
        expander.config = config
        from trialmatchai.matching.query_expansion import _resolve_settings

        expander.settings = _resolve_settings(config)
        expander._init_vllm()
    finally:
        loader_module.load_vllm_engine = original

    # The cache key is built from cot_adapter_path, so it must still be the eligibility
    # adapter for the two stages to land on one engine.
    assert captured["model_config"]["cot_adapter_path"] == (
        "models/trialmatchai-phi4-reasoning-lora"
    )
    # ...while expansion itself generates on the BASE model.
    assert expander.lora_request is None


def test_expansion_keeps_its_own_engine_when_given_an_explicit_adapter():
    """An explicitly configured expansion adapter is honoured, and keys to its own engine."""
    from trialmatchai.matching.query_expansion import QueryExpander, _resolve_settings

    captured = {}

    def fake_loader(*, model_config, vllm_cfg):
        captured["model_config"] = dict(model_config)
        return ("engine", "tokenizer", "LORA_REQUEST")

    config = {
        "model": {
            "base_model": "microsoft/phi-4",
            "cot_adapter_path": "models/trialmatchai-phi4-reasoning-lora",
        },
        "rag": {"backend": "vllm"},
        "query_expansion": {"enabled": True, "backend": "vllm", "adapter": "models/other"},
        "vllm": {},
    }

    import trialmatchai.models.llm.vllm_loader as loader_module

    original = loader_module.load_vllm_engine
    loader_module.load_vllm_engine = fake_loader
    try:
        expander = QueryExpander.__new__(QueryExpander)
        expander.config = config
        expander.settings = _resolve_settings(config)
        expander._init_vllm()
    finally:
        loader_module.load_vllm_engine = original

    assert captured["model_config"]["cot_adapter_path"] == "models/other"
    assert expander.lora_request == "LORA_REQUEST"
