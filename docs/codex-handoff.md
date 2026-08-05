# Codex Handoff: TAIM and TrialMatchAI

## Project and goal

- Main repository: /Users/mehulmittal/TAIM
- TrialMatchAI checkout: /Users/mehulmittal/TAIM/external/trialmatchai-current
- TrialMatchAI checkout: v0.7.0, detached at commit 071a76e.
- Owner/collaborator: Hannes Ill.
- Machine: Apple MacBook Pro with M5 Pro, 24 GB RAM.
- Python: CPython 3.11.15 installed and run through uv.
- No Docker or NVIDIA CUDA is available.
- Goal: adapt TrialMatchAI to the TAIM benchmark contract, preserve reproducibility, and evaluate a modern local model stack.
- Do not inspect or use protected TREC 2022/2023 topics or judgments.

## What TrialMatchAI does

TrialMatchAI matches a patient description to clinical trials:

1. Prepare trial and eligibility-criterion records.
2. Embed trial and criterion text.
3. Store vectors in LanceDB.
4. Embed the patient query.
5. Retrieve candidate trials with hybrid search.
6. Rerank candidates with a language model.
7. Check eligibility with a reasoning model.
8. Produce ranked results and reports.

The end-to-end order is: index -> ingest -> expand -> match.

Ranking scores are internal ranking values, not probabilities or medical recommendations.

## Completed setup

The following worked:

    uv python install 3.11
    uv lock --check
    uv sync --group dev
    uv run pytest

The full test suite passed with 414 tests after the MLX work.

Data is present:

- data/processed_trials: 109,281 trial JSON files.
- data/processed_criteria: approximately 2.1 million criterion JSON records.
- Criteria are organized by NCT identifier.
- Records include text and precomputed vector fields.

The original LanceDB index was built and verified with:

    uv run trialmatchai index \
      --processed-trials-folder data/processed_trials \
      --processed-criteria-folder data/processed_criteria

    uv run trialmatchai healthcheck --require-tables

A directory-marker/download layout problem was fixed and documented in docs/bootstrap-data-directory-normalization.md.

## BGE-M3 status

The embedding model is BAAI/bge-m3.

Previously, the prepared trial and criterion files contained precomputed BGE-M3 vectors. New patient/query vectors were computed at runtime. Logs showed:

    Loading embedder model BAAI/bge-m3 on device cpu

Those corpus vectors came from the older text/Snapshot representation. Since Snapshot v2 changes canonical text and structured inputs, the corpus vectors should be recomputed for v2.

The current embedder device logic recognizes CUDA or CPU, not Apple MPS. Setting use_gpu=true does not automatically use the M5 GPU. MPS support would require a deliberate code change and testing.

### Important re-embedding memory fix

An initial CPU re-embedding run was stopped after about 1.5 hours because it consumed almost all physical memory and swap without writing an index. The cause was a materialization bug in `src/trialmatchai/orchestration.py`: when `search_backend.reembed_index` was true, the code converted the entire criteria generator into one list before embedding.

The re-embedding path now streams criteria in bounded batches of 8,192 records. Trial records and the first criteria batch are embedded, then each criteria batch is written to LanceDB incrementally. Only the final batch creates the vector indexes. The full criteria corpus is never held in memory at once.

The process was stopped safely before it produced a partial replacement index. A regression test verifies that 8,193 criteria are handled as batches of 8,192 and 1. The external TrialMatchAI suite passes after the fix.

The follow-up CPU run was later stopped after confirming that it was progressing but would take multiple days. Its empty `data/search_bge_m3_cpu_v2` directory and temporary config were removed. The original `data/search` index and processed data were left untouched. Hannes will provide embeddings generated on GCP instead.

## Current CPU recomputation

A separate CPU-only index is being built with:

    jq '
      .embedder.use_gpu = false
      | .search_backend.reembed_index = true
      | .search_backend.db_path = "data/search_bge_m3_cpu_v2"
    ' src/trialmatchai/config/config.json > /tmp/trialmatchai-cpu-reembed.json

    uv run trialmatchai index \
      --config /tmp/trialmatchai-cpu-reembed.json \
      --reindex \
      --processed-trials-folder data/processed_trials \
      --processed-criteria-folder data/processed_criteria

This recomputes BGE-M3 vectors for more than 2.2 million texts. It may take several hours on CPU. It writes to data/search_bge_m3_cpu_v2 and leaves data/search untouched.

After completion:

1. Record the total runtime.
2. Check the final trial and criteria counts.
3. Run a healthcheck using the new config.
4. Run a small e2e smoke test using the new config.
5. Keep the old index until the new one is verified.

## Modern MLX model stack

The original config referenced microsoft/phi-4 and google/gemma-2-2b-it.

The modern local setup is:

- Eligibility reasoning: mlx-community/Qwen3.5-9B-4bit
- Retrieval reranking: mlx-community/gemma-4-e4b-it-4bit
- Embeddings/search: BAAI/bge-m3 for now

Standalone tests succeeded for Qwen3.5 and Gemma 4. MLX-VLM 0.6.8 is in the lockfile. Dependency pins were widened to support MLX-VLM:

- transformers >=5.14,<6
- safetensors >=0.8,<1

MLX support was added in:

- src/trialmatchai/models/llm/mlx_loader.py
- src/trialmatchai/models/llm/mlx_reranker.py
- src/trialmatchai/models/llm/mlx_vlm_loader.py
- src/trialmatchai/matching/eligibility_reasoning_mlx.py
- src/trialmatchai/matching/eligibility_reasoning_mlx_vlm.py
- tests/test_mlx_backend.py
- docs/mac-mlx-implementation.md

The MLX-LM API required make_sampler(temperature), not the old temp argument. That compatibility issue was fixed.

Qwen3.5 eligibility output initially stopped with invalid JSON at a low token limit. Using TRIALMATCHAI_MLX_MAX_NEW_TOKENS=8192 allowed a complete smoke run.

## Modern smoke tests

Synthetic files were used only for wiring/runtime checks, not clinical claims:

- data/patients/mlx_smoke_patient.txt
- data/patients/mlx_modern_smoke_patient.txt

The modern e2e run completed retrieval, reranking, Qwen3.5 eligibility reasoning, final ranking, and report creation. One result was produced for NCT05752136.

Expected smoke-test warnings:

- Hugging Face unauthenticated request: rate-limit notice.
- BGE-M3 optional file 404s: normal repository probing.
- No annotations found: expected because entity extraction was disabled.
- Existing outputs may be skipped unless --rematch is supplied.

## Snapshot v2 contract

The owner proposed TAIM Issue #8:
https://github.com/hannesill/TAIM/issues/8

The specification is appropriate and aligns with the adapter plan. Its key rules are:

- Preserve structured patient and trial fields.
- Also provide deterministic canonical text for text-only systems.
- Preserve field-level provenance.
- Preserve missingness and conflicts; never guess defaults.
- Keep derived views separately named, versioned, and hashed.
- Keep judgments/qrels outside the System request.
- Use only frozen TREC 2021 source artifacts for the historical benchmark.
- Keep TREC 2022/2023 protected.
- Do not relabel v1 indexes, embeddings, or results as v2.
- Keep TrialMatchAI prompts, ranking, filtering, and eligibility decisions outside the shared Snapshot contract.

## Adapter design

The intended flow is:

    TAIM Snapshot v2
      -> adapter reads permitted structured fields
      -> deterministic renderer creates TrialMatchAI prompts
      -> TrialMatchAI retrieval/reranking/eligibility
      -> adapter emits standard TAIM Candidate records

The adapter must not use evaluation judgments or hidden evaluator information. It should preserve retrieval, reranking, eligibility decisions, explanations, and raw outputs as diagnostics.

The owner still needs to confirm:

1. Which Snapshot capabilities the adapter may consume.
2. The exact deterministic rendering format.
3. Whether final post-eligibility ranking is the Primary Ranking.
4. Which intermediate outputs must be stored as Stage Rankings/artifacts.

Recommended Primary Ranking: final post-eligibility ranking. Retrieval and reranking outputs remain diagnostic.

## Adapter work completed so far

The TAIM repository now contains the first adapter boundary slice:

- `src/taim/adapters/trialmatchai.py`
- `src/taim/adapters/__init__.py`
- `tests/test_trialmatchai_adapter.py`
- `docs/trialmatchai-adapter.md`

These helpers do not import or patch TrialMatchAI. They:

1. Render deterministic patient input from either the v1 `patient_text` field or the v2 `canonical_text` field.
2. Append Snapshot v2 typed patient facts in stable field order when explicitly available.
3. Write newline-stable patient files for the upstream command.
4. Run a pinned TrialMatchAI command as an isolated subprocess with timeout and failure handling.
5. Parse TrialMatchAI `ranked_trials.json` output.
6. Convert final results into TAIM `Candidate` records.
7. Apply deterministic score-descending/trial-ID-ascending tie breaking.
8. Keep optional first-level retrieval scores as diagnostic rankings rather than silently treating them as final output.

The full TAIM suite passes with 266 tests, and Ruff checks pass. This is the adapter foundation, not yet the complete benchmark runner.

## Current plan

Completed:

1. Understand repository, vocabulary, and pipeline.
2. Set up uv and Python.
3. Study CLI and file structure.
4. Download and inspect data.
5. Build and verify LanceDB.
6. Add and test MLX support.
7. Validate Qwen3.5 and Gemma 4 locally.
8. Run modern TrialMatchAI smoke tests.

In progress:

9. Recompute BGE-M3 corpus embeddings on CPU for Snapshot v2 compatibility.
10. Implement and test the adapter input/output boundary.

Next:

11. Verify the new CPU index.
12. Obtain a concrete Snapshot v2 schema/sample in the working branch after PR #25 lands.
13. Agree on permitted Snapshot capabilities.
14. Confirm Primary Ranking semantics.
15. Build the full topic-run orchestrator around the adapter boundary.
16. Stage complete Snapshot topics/trials for the pinned TrialMatchAI checkout.
17. Record Snapshot identity, model revisions, embedding identity, timings, manifests, raw-output hashes, and artifacts.
18. Run the original setup on a small fixed calibration subset if required.
19. Run the modern model configuration on the agreed benchmark input.

## Safety and reproducibility rules

- Do not use protected TREC 2022/2023 data.
- Do not call synthetic smoke-test results clinical performance.
- Do not treat ranking scores as probabilities.
- Do not enrich frozen TREC 2021 data from live ClinicalTrials.gov.
- Do not silently reuse old vectors for Snapshot v2.
- Do not replace structured source fields with model-inferred facts.
- Do not change ranking or eligibility semantics while building the shared adapter unless explicitly agreed.

Useful checks:

    uv run pytest
    uv run trialmatchai healthcheck --require-tables
    uv run trialmatchai build --status
    git status --short --branch

When resuming, first check whether the CPU embedding command is still running or has completed. Inspect its final output before starting another expensive job.
