"""Optional MLX-VLM loader for multimodal Qwen3.5 checkpoints."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MLXVLMTextGenerator:
    """Generate text with an MLX-VLM model while using text-only prompts.

    Qwen3.5 checkpoints are multimodal, but TrialMatchAI currently supplies text.
    The VLM runtime still applies the model-specific chat template and returns the
    generated text needed by the existing eligibility parser.
    """

    model_path: str
    revision: str | None = None

    def __post_init__(self) -> None:
        try:
            from mlx_vlm import generate, load
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_config
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "Qwen3.5 generation requires mlx-vlm (`uv sync --extra mlx-vlm`)."
            ) from exc

        self.model, self.processor = load(self.model_path, revision=self.revision)
        self.config = load_config(self.model_path, revision=self.revision)
        self._generate = generate
        self._apply_chat_template = apply_chat_template

    def format_prompt(self, prompt: str, *, thinking: bool = False) -> str:
        return str(
            self._apply_chat_template(
                self.processor,
                self.config,
                prompt,
                num_images=0,
                enable_thinking=thinking,
            )
        )

    def generate_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        thinking: bool = False,
    ) -> str:
        formatted = self.format_prompt(prompt, thinking=thinking)
        result = self._generate(
            self.model,
            self.processor,
            formatted,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False,
        )
        return str(getattr(result, "text", result)).strip()
