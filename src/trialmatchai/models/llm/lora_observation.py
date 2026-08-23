"""Read back from the engine which adapter it actually attached to a request.

Sharing one engine between query expansion and eligibility reasoning is meant to be an
efficiency change and nothing else: expansion must keep running the unadapted base model
and reasoning must keep running with the fine-tuned CoT adapter. Reading the configuration
back proves nothing about that, because the configuration is the thing whose effect is in
question. vLLM records the adapter it used on each ``RequestOutput`` (``lora_request``,
documented as "the LoRA request that was used to generate the output"), so this asserts
against the engine's own account of what it did.

This is deliberately a check that can fail. The failure it exists to catch is the
reasoning stage silently losing its adapter: the fine-tuned model is the entire purpose of
that stage, and losing it would look like a modest score drop and leave no trace in the
artifacts.
"""

from __future__ import annotations

from typing import Any, Sequence

from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)

_UNOBSERVED = object()


class AdapterAttachmentError(RuntimeError):
    """The engine attached a different adapter than the stage requires."""


def _attached_path(output: Any) -> str | None:
    """The adapter path the engine reports for a completed request."""

    request = getattr(output, "lora_request", _UNOBSERVED)
    if request is _UNOBSERVED:
        # Refusing here is the point. Passing silently would make this a check that cannot
        # fail, which is indistinguishable from no check at all.
        raise AdapterAttachmentError(
            "vLLM RequestOutput has no lora_request attribute, so the attached adapter "
            "cannot be observed. Adapter attachment is unverifiable on this vLLM build; "
            "do not treat the run as adapter-verified."
        )
    if request is None:
        return None
    for attribute in ("lora_path", "lora_local_path", "path"):
        value = getattr(request, attribute, None)
        if value:
            return str(value)
    name = getattr(request, "lora_name", None)
    return str(name) if name else "<unnamed-adapter>"


def assert_attached_adapter(
    outputs: Sequence[Any],
    *,
    expected_path: str | None,
    caller: str,
) -> str | None:
    """Assert the engine attached ``expected_path`` (or nothing) to every output.

    ``expected_path=None`` asserts the base model ran with no adapter, which is what query
    expansion requires. A non-None value asserts that exact adapter ran, which is what the
    eligibility stage requires.
    """

    if not outputs:
        return None
    observed: set[str | None] = set()
    for output in outputs:
        observed.add(_attached_path(output))
    if len(observed) != 1:
        raise AdapterAttachmentError(
            f"{caller}: the engine attached more than one adapter across a single batch: "
            f"{sorted(str(item) for item in observed)}"
        )
    attached = observed.pop()
    if expected_path is None:
        if attached is not None:
            raise AdapterAttachmentError(
                f"{caller} must run the unadapted base model, but the engine attached "
                f"{attached!r}. Sharing one engine has changed what this stage computes."
            )
        logger.info("[lora] %s ran on the base model, engine-confirmed.", caller)
        return None
    if attached is None:
        raise AdapterAttachmentError(
            f"{caller} must run with adapter {expected_path!r}, but the engine attached "
            "NO adapter. The fine-tuned model is the purpose of this stage; a run in this "
            "state is not the system we intend to measure."
        )
    if str(attached) != str(expected_path):
        raise AdapterAttachmentError(
            f"{caller} must run with adapter {expected_path!r}, but the engine attached "
            f"{attached!r}."
        )
    logger.info("[lora] %s ran with adapter %s, engine-confirmed.", caller, attached)
    return attached
