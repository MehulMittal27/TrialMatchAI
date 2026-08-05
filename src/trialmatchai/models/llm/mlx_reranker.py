from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from tqdm import tqdm

from trialmatchai.models.llm.llm_reranker import LLMReranker
from trialmatchai.models.llm.mlx_loader import MLXTextGenerator
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class MLXReranker:
    """Apple-Silicon Yes/No reranker using first-token log probabilities from MLX-LM."""

    score_type = "binary_yes_probability"

    def __init__(
        self,
        model_path: str,
        *,
        batch_size: int = 1,
        revision: str | None = None,
        trust_remote_code: bool = False,
    ):
        self.generator = MLXTextGenerator(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        self.batch_size = max(1, int(batch_size))
        self.yes_token_id = self.generator.first_token_id("Yes")
        self.no_token_id = self.generator.first_token_id("No")
        if self.yes_token_id == self.no_token_id:
            raise ValueError("Yes and No must map to different tokenizer IDs.")
        logger.info("Loaded MLX reranker %s.", model_path)

    @staticmethod
    def binary_yes_probability(
        logprobs: Mapping[int, float], yes_token_id: int, no_token_id: int
    ) -> tuple[float, bool]:
        """Normalize the available Yes/No log probabilities and report availability."""
        yes_lp = float(logprobs.get(yes_token_id, float("-inf")))
        no_lp = float(logprobs.get(no_token_id, float("-inf")))
        highest = max(yes_lp, no_lp)
        if not math.isfinite(highest):
            return 0.5, False
        yes = math.exp(yes_lp - highest) if math.isfinite(yes_lp) else 0.0
        no = math.exp(no_lp - highest) if math.isfinite(no_lp) else 0.0
        total = yes + no
        if total <= 0.0:
            return 0.5, False
        return yes / total, True

    def rank_pairs(self, patient_trial_pairs: list[tuple]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for start in tqdm(
            range(0, len(patient_trial_pairs), self.batch_size),
            desc="MLX reranking batches",
        ):
            batch = patient_trial_pairs[start : start + self.batch_size]
            for patient_text, trial_text in batch:
                prompt = self.generator.format_messages(
                    LLMReranker.create_messages(patient_text, trial_text)
                )
                logprobs = self.generator.first_token_logprobs(
                    prompt,
                    (self.yes_token_id, self.no_token_id),
                    temperature=0.0,
                )
                score, available = self.binary_yes_probability(
                    logprobs, self.yes_token_id, self.no_token_id
                )
                if not available:
                    label = "Unknown"
                else:
                    label = "Yes" if score > 0.5 else "No"
                results.append(
                    {
                        "llm_score": score,
                        "answer": label,
                        "score_type": self.score_type,
                    }
                )
        return results
