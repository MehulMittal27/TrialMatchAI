"""Small optional MLX-LM loader for Apple Silicon text generation and scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MLXTextGenerator:
    """Wrapper around MLX-LM text generation and first-token log probabilities."""

    model_path: str
    revision: str | None = None
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        try:
            from mlx_lm import generate, load, stream_generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "MLX generation requires the mlx extra (`uv sync --extra mlx`)."
            ) from exc

        tokenizer_config: dict[str, Any] = {}
        if self.trust_remote_code:
            tokenizer_config["trust_remote_code"] = True
        self.model, self.tokenizer = load(
            self.model_path,
            tokenizer_config=tokenizer_config or None,
            revision=self.revision,
        )
        self._generate = generate
        self._stream_generate = stream_generate
        self._make_sampler = make_sampler

    def format_messages(self, messages: list[dict[str, str]]) -> str:
        template = getattr(self.tokenizer, "apply_chat_template", None)
        if template is not None:
            try:
                return template(messages, tokenize=False, add_generation_prompt=True)
            except (TypeError, ValueError):
                pass
        return "\n\n".join(
            f"{message.get('role', 'user').title()}: {message.get('content', '')}"
            for message in messages
        ) + "\nAssistant:"

    def generate_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        return str(
            self._generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=self._make_sampler(temperature),
                verbose=False,
            )
        ).strip()

    def first_token_id(self, text: str) -> int:
        """Return the first tokenizer ID for a label, matching upstream reranker behavior."""
        try:
            encoded = self.tokenizer(text, add_special_tokens=False)
            token_ids = encoded["input_ids"]
        except (TypeError, AttributeError):
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        if not token_ids:
            raise ValueError(f"Tokenizer could not encode {text!r}.")
        return int(token_ids[0])

    def first_token_logprobs(
        self,
        prompt: str,
        token_ids: tuple[int, ...],
        *,
        temperature: float = 0.0,
    ) -> dict[int, float]:
        """Return selected first-token log probabilities for one prompt.

        ``mlx_lm.generate`` returns decoded text only. ``stream_generate`` exposes the
        log-probability vector on its first ``GenerationResponse``, which lets callers compute
        the same two-class Yes/No probability used by the upstream reranker.
        """
        responses = self._stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=1,
            sampler=self._make_sampler(temperature),
        )
        response = next(iter(responses), None)
        raw_logprobs = getattr(response, "logprobs", None)
        if raw_logprobs is None:
            return {}

        scores: dict[int, float] = {}
        for token_id in token_ids:
            try:
                value = raw_logprobs[int(token_id)]
                value = value.item() if hasattr(value, "item") else value
                scores[int(token_id)] = float(value)
            except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                continue
        return scores
