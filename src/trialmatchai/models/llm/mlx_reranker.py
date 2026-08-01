from __future__ import annotations

from typing import Any

from tqdm import tqdm

from trialmatchai.models.llm.llm_reranker import LLMReranker
from trialmatchai.models.llm.mlx_loader import MLXTextGenerator
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class MLXReranker:
    """Apple-Silicon Yes/No reranker using deterministic MLX text generation.

    Unlike vLLM and Transformers, the public MLX-LM generation API returns text rather than
    next-token log-probabilities. The score is therefore a generated-label score (1.0 for Yes,
    0.0 for No, 0.5 for an unrecognized answer) and must be recorded as such in provenance.
    """

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
        logger.info("Loaded MLX reranker %s.", model_path)

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
                answer = self.generator.generate_text(
                    prompt,
                    max_tokens=1,
                    temperature=0.0,
                )
                normalized = answer.strip().casefold()
                if normalized.startswith("yes"):
                    score, label = 1.0, "Yes"
                elif normalized.startswith("no"):
                    score, label = 0.0, "No"
                else:
                    score, label = 0.5, "Unknown"
                results.append({"llm_score": score, "answer": label})
        return results
