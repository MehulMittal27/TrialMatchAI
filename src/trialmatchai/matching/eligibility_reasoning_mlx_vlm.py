from __future__ import annotations

import os
from typing import Any

from trialmatchai.matching.eligibility_base import BaseTrialProcessor
from trialmatchai.models.llm.mlx_vlm_loader import MLXVLMTextGenerator
from trialmatchai.utils.file_utils import write_json_file
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class BatchTrialProcessorMLXVLM(BaseTrialProcessor):
    """Eligibility processor for multimodal MLX-VLM models used with text only."""

    def __init__(
        self,
        model_path: str,
        *,
        batch_size: int = 1,
        use_cot: bool = True,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        revision: str | None = None,
        no_think: bool = False,
    ):
        self.generator = MLXVLMTextGenerator(model_path, revision=revision)
        # Keep prompts plain in BaseTrialProcessor. MLX-VLM applies the template once
        # immediately before generation, which avoids double-formatting.
        self.tokenizer = None
        self.batch_size = max(1, int(batch_size))
        self.use_cot = use_cot
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.temperature = max(0.0, float(temperature))
        self.no_think = no_think
        logger.info("Loaded MLX-VLM eligibility model %s.", model_path)

    def _progress_desc(self) -> str:
        return "MLX-VLM Processing Trials"

    def _process_batch(self, batch: list[dict[str, Any]], output_folder: str):
        for item in batch:
            try:
                response = self.generator.generate_text(
                    item["prompt"],
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    thinking=not self.no_think,
                )
                self._save_outputs(item["nct_id"], response, output_folder)
            except Exception as exc:
                logger.error("MLX-VLM processing failed for %s: %s", item["nct_id"], exc)
                os.makedirs(output_folder, exist_ok=True)
                write_json_file(
                    {"error": "processing_failed", "detail": str(exc)},
                    os.path.join(output_folder, f"{item['nct_id']}.json"),
                )
