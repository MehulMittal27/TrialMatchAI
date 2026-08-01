from __future__ import annotations

import os
from typing import Any

from trialmatchai.matching.eligibility_base import BaseTrialProcessor
from trialmatchai.models.llm.mlx_loader import MLXTextGenerator
from trialmatchai.utils.file_utils import write_json_file
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class BatchTrialProcessorMLX(BaseTrialProcessor):
    """Apple-Silicon eligibility processor backed by MLX-LM text generation."""

    def __init__(
        self,
        model_path: str,
        *,
        batch_size: int = 1,
        use_cot: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        revision: str | None = None,
        trust_remote_code: bool = False,
        no_think: bool = False,
    ):
        self.generator = MLXTextGenerator(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer = self.generator.tokenizer
        self.batch_size = max(1, int(batch_size))
        self.use_cot = use_cot
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.temperature = max(0.0, float(temperature))
        self.no_think = no_think
        logger.info("Loaded MLX eligibility model %s.", model_path)

    def _progress_desc(self) -> str:
        return "MLX Processing Trials"

    def _process_batch(self, batch: list[dict[str, Any]], output_folder: str):
        # Keep the first implementation deliberately sequential. This is slower than a
        # server-backed batch, but it is deterministic and keeps Apple unified memory bounded.
        for item in batch:
            try:
                # BaseTrialProcessor already applies the tokenizer's chat template in
                # _format_prompt. Applying it again here would wrap an already formatted
                # conversation inside a second user message and degrade model output.
                prompt = item["prompt"]
                response = self.generator.generate_text(
                    prompt,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
                self._save_outputs(item["nct_id"], response, output_folder)
            except Exception as exc:
                logger.error("MLX processing failed for %s: %s", item["nct_id"], exc)
                os.makedirs(output_folder, exist_ok=True)
                write_json_file(
                    {"error": "processing_failed", "detail": str(exc)},
                    os.path.join(output_folder, f"{item['nct_id']}.json"),
                )
