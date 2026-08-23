"""The adapter observation must be able to FAIL, or it is not a check."""

import pytest

from trialmatchai.models.llm.lora_observation import (
    AdapterAttachmentError,
    assert_attached_adapter,
)

ADAPTER = "models/trialmatchai-phi4-reasoning-lora"


class _Req:
    def __init__(self, path):
        self.lora_path = path
        self.lora_name = "cot"


class _Out:
    """Stands in for a vLLM RequestOutput, which records the adapter actually used."""

    def __init__(self, request):
        self.lora_request = request


class _NoAttr:
    """A RequestOutput from a build that does not report the adapter at all."""


def test_reasoning_without_its_adapter_is_a_loud_failure():
    # THE failure this exists to catch: the fine-tuned model is the purpose of the stage,
    # and losing it would look like a modest score drop with nothing in the artifacts.
    with pytest.raises(AdapterAttachmentError, match="attached\\s+NO adapter"):
        assert_attached_adapter(
            [_Out(None)], expected_path=ADAPTER, caller="eligibility reasoning"
        )


def test_reasoning_with_the_wrong_adapter_is_a_loud_failure():
    with pytest.raises(AdapterAttachmentError, match="attached"):
        assert_attached_adapter(
            [_Out(_Req("models/some-other-lora"))],
            expected_path=ADAPTER,
            caller="eligibility reasoning",
        )


def test_expansion_silently_gaining_an_adapter_is_a_loud_failure():
    # The mirror case: sharing one engine must not attach the CoT adapter to expansion.
    with pytest.raises(AdapterAttachmentError, match="unadapted base model"):
        assert_attached_adapter(
            [_Out(_Req(ADAPTER))], expected_path=None, caller="query expansion"
        )


def test_a_batch_with_mixed_adapters_is_a_loud_failure():
    with pytest.raises(AdapterAttachmentError, match="more than one adapter"):
        assert_attached_adapter(
            [_Out(_Req(ADAPTER)), _Out(None)],
            expected_path=ADAPTER,
            caller="eligibility reasoning",
        )


def test_an_unobservable_build_is_a_loud_failure_not_a_pass():
    # If the engine cannot report what it attached, the honest outcome is refusal. Passing
    # here would make the check incapable of failing, which is the same as no check.
    with pytest.raises(AdapterAttachmentError, match="cannot be observed"):
        assert_attached_adapter(
            [_NoAttr()], expected_path=ADAPTER, caller="eligibility reasoning"
        )


def test_the_two_correct_configurations_pass():
    assert (
        assert_attached_adapter(
            [_Out(_Req(ADAPTER))], expected_path=ADAPTER, caller="eligibility reasoning"
        )
        == ADAPTER
    )
    assert (
        assert_attached_adapter([_Out(None)], expected_path=None, caller="query expansion")
        is None
    )
