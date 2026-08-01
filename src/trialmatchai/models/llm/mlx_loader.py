"""Small optional MLX-LM loader for Apple Silicon text generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MLXTextGenerator:
    """Lazy-free wrapper around the public ``mlx_lm`` load/generate API.

    MLX-LM exposes generated text rather than vLLM-style token log-probabilities.
    Callers that need calibrated probabilities must record that they are using
    a generated-label score instead.
    """

    model_path: str
    revision: str | None = None
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        try:
            from mlx_lm import generate, load
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
