# TrialMatchAI on Apple Silicon: MLX Implementation Notes

This document records the Apple-Silicon work completed for the TrialMatchAI v0.7.0
checkout. It is written as both a review record and a beginner-friendly explanation.

## 1. The original problem

TrialMatchAI's original language-model path is designed around vLLM. vLLM is a fast
model-serving engine that is mainly used with NVIDIA CUDA GPUs and Linux. The M5
MacBook Pro has an Apple GPU, not an NVIDIA CUDA GPU, so the original vLLM path is
not the natural local runtime for this machine.

The goal was therefore:

1. Keep the original vLLM and Phi-4 configuration intact for reproducibility.
2. Add a separate MLX path for local Apple-Silicon experiments.
3. Use a smaller modern model locally before attempting a full pipeline run.
4. Keep the change optional, so users who do not need MLX do not install it.

## 2. Simple vocabulary

### Runtime backend

A runtime backend is the engine that loads and runs a language model.

Analogy: the model is a car, while vLLM, Transformers, and MLX are different roads
and driving systems. The same car cannot use every road in exactly the same way.

### vLLM

vLLM is a high-throughput model engine. It is excellent for NVIDIA GPU servers and
supports features such as batching and token-level scores used by the existing
TrialMatchAI implementation.

### Transformers

Transformers is the general Hugging Face model interface. It can run models directly,
including CPU smoke tests, but is usually slower for large-scale serving.

### MLX

MLX is Apple's machine-learning framework for Apple Silicon. MLX-LM adds convenient
loading and text generation for language models on the Mac GPU.

### Quantized model

Quantization stores model numbers using fewer bits. This makes the model smaller and
faster, with a possible quality trade-off.

Analogy: instead of storing every measurement with ten decimal places, we store only
the precision needed for the task.

### Smoke test

A smoke test is a tiny test that answers: "Does the basic path start and produce an
answer?" It is not a scientific benchmark and does not prove final quality.

## 3. Why Qwen3-8B-4bit was selected first

The first local candidate is:

```text
mlx-community/Qwen3-8B-4bit
```

Reasons:

- It has an MLX-ready model package.
- It is text-only, so it matches TrialMatchAI's language workload.
- The 4-bit package is approximately 4.6 GB, which is practical on a 24 GB Mac.
- It is much smaller than Phi-4 14B, leaving memory for the operating system,
  embeddings, LanceDB, and application code.
- Its chat template can be used through the standard MLX-LM API.

This is an engineering starting point, not a claim that Qwen3 is automatically better
than every Gemma or Phi model. Quality must be measured later on the agreed benchmark.

## 4. What was changed

### Optional dependency

In `pyproject.toml`, an optional dependency group was added:

```toml
mlx = [
  "mlx-lm>=0.26; sys_platform == 'darwin'",
]
```

The Darwin condition means this dependency is selected on macOS. It is not forced on
Linux installations that use vLLM.

Install it with:

```bash
uv sync --extra mlx
```

This installed MLX-LM, MLX, and the Apple Metal support package in the local virtual
environment. `uv.lock` records the exact resolved versions for reproducibility.

### MLX model loader

File: `src/trialmatchai/models/llm/mlx_loader.py`

This wrapper does four jobs:

1. Imports MLX-LM only when the MLX backend is actually used.
2. Loads a local directory or Hugging Face model identifier.
3. Applies the model's chat template when one exists.
4. Calls MLX-LM generation with a maximum token count and temperature.

Lazy importing is important. A normal TrialMatchAI installation should still be able
to use the original backends without requiring MLX.

### MLX eligibility reasoning

File: `src/trialmatchai/matching/eligibility_reasoning_mlx.py`

This adds `BatchTrialProcessorMLX`, which follows the existing
`BaseTrialProcessor` contract:

1. Load a trial's eligibility criteria.
2. Combine the criteria with the patient narrative.
3. Build the existing TrialMatchAI prompt.
4. Send the prompt to MLX-LM.
5. Save the raw text and parsed JSON result in the same output structure.

The first version intentionally processes items sequentially. This is slower than a
server batch, but it keeps unified Apple memory bounded and makes debugging easier.

Important correction: `BaseTrialProcessor` already applies the tokenizer chat
template. The MLX processor now sends that formatted prompt directly. Applying the
template a second time would incorrectly put one formatted conversation inside another
user message.

### MLX reranker

File: `src/trialmatchai/models/llm/mlx_reranker.py`

The existing reranker asks a language model whether a patient-trial pair is a match.
The MLX implementation generates a short answer and maps it as follows:

```text
Yes       -> 1.0
No        -> 0.0
Unclear   -> 0.5
```

This is a generated-label score. It is not the same as a calibrated next-token
probability from vLLM. Therefore, when we benchmark this path, the provenance must
record that the MLX reranker uses generated labels.

### Query expansion

File: `src/trialmatchai/matching/query_expansion.py`

The optional query-expansion component can now use MLX as well. It remains disabled by
default, so adding MLX support does not silently change the normal pipeline.

### Configuration and preflight

Files:

- `src/trialmatchai/config/settings.py`
- `src/trialmatchai/main.py`
- `src/trialmatchai/services/preflight.py`

The accepted backend names now include `mlx` for retrieval reasoning, reranking, and
query expansion. When the configured backend is MLX, the program checks that `mlx_lm`
is installed and gives this actionable message if it is missing:

```text
MLX backend requires mlx-lm (`uv sync --extra mlx`).
```

The default configuration still points to Phi-4, Gemma 2, and vLLM. A separate Mac
configuration should be created for experiments rather than overwriting the canonical
configuration.

## 5. What was not changed

- The original Phi-4 model configuration was not replaced.
- The original vLLM backend was not removed.
- The LanceDB data and index format was not changed.
- No claim was made that the MLX model reproduces vLLM scores exactly.
- No full benchmark result has been produced yet.

This separation lets us compare an original calibration run with a modern local-model
run later.

## 6. Verification performed

The following checks passed after the implementation:

```text
414 tests passed
ruff lint passed for the new MLX files
```

A focused unit test also verified that:

- the model revision is passed to MLX-LM's `load()` function;
- the prompt is passed once to generation;
- `max_tokens`, temperature, and quiet output are forwarded correctly.

The current execution environment cannot access a physical Metal device, so actual
GPU inference must be run on the user's M5 Mac. The code and API contract were checked
against the installed MLX-LM package.

## 7. First real M5 smoke test

Run this from the TrialMatchAI checkout:

```bash
cd /Users/mehulmittal/TAIM/external/trialmatchai-current

uv sync --extra mlx

uv run mlx_lm.generate \
  --model mlx-community/Qwen3-8B-4bit \
  --prompt "Reply with exactly: MLX works" \
  --max-tokens 16 \
  --temp 0
```

The first run downloads the model. Success means the command returns the requested
answer without a Metal, model-loading, or tokenizer error.

## 8. Selecting MLX without changing the original config

The repository's standard configuration intentionally still names the original
backends. For a Mac experiment, backend and model values can be supplied through
environment variables instead of editing `config.json`:

```bash
export TRIALMATCHAI_RAG_BACKEND=mlx
export TRIALMATCHAI_RERANKER_BACKEND=mlx
export TRIALMATCHAI_MODEL_BASE_MODEL=mlx-community/Qwen3-8B-4bit
export TRIALMATCHAI_MODEL_RERANKER_MODEL_PATH=mlx-community/Qwen3-8B-4bit
export TRIALMATCHAI_EMBEDDER_USE_GPU=false
export TRIALMATCHAI_ENTITY_BACKEND=disabled
export TRIALMATCHAI_CONCEPT_LINKER_ENABLED=false
```

The first two variables select the MLX engine. The next two choose the Qwen3 model.
The embedder flag prevents a CUDA check. Entity extraction and concept linking are
disabled for the first smoke run because they are optional and would introduce extra
models before the core path is proven.

The configuration can be checked with:

```bash
uv run trialmatchai healthcheck --require-tables --models
```

The healthcheck may warn that it cannot contact Hugging Face when the model is already
cached or the network is unavailable. That warning is separate from the LanceDB and
MLX dependency checks.

The full matching path also needs PyTorch because TrialMatchAI's BGE-M3 embedder uses
PyTorch to turn the patient narrative into a search vector. On this Mac it can run on
CPU; it does not require CUDA:

```bash
uv sync --extra llm
```

## 9. What this smoke test does not prove

It does not prove that TrialMatchAI's clinical matching quality is good. It only proves
that:

- the model can be downloaded;
- MLX can use the Mac GPU;
- the tokenizer and generation API work;
- a small prompt can complete.

The next test is an isolated TrialMatchAI eligibility run on one patient and a very
small fixed trial list. Only after that should we run a larger comparison.

## 10. Synthetic patient smoke test

The repository did not include a patient profile. A small fictional input was added at:

```text
data/patients/mlx_smoke_patient.txt
```

This file is ignored by Git because the repository ignores the whole `data/` directory.
It is not real clinical data and must only be used to test wiring. It describes a
fictional 55-year-old woman with rectal adenocarcinoma.

The first end-to-end run uses a deliberately small retrieval budget. This prevents a
beginner smoke test from invoking the model for hundreds of trials:

```bash
export TRIALMATCHAI_RAG_BACKEND=mlx
export TRIALMATCHAI_RERANKER_BACKEND=mlx
export TRIALMATCHAI_MODEL_BASE_MODEL=mlx-community/Qwen3-8B-4bit
export TRIALMATCHAI_MODEL_RERANKER_MODEL_PATH=mlx-community/Qwen3-8B-4bit
export TRIALMATCHAI_EMBEDDER_USE_GPU=false
export TRIALMATCHAI_ENTITY_BACKEND=disabled
export TRIALMATCHAI_CONCEPT_LINKER_ENABLED=false
export TRIALMATCHAI_SEARCH_MAX_TRIALS_FIRST_LEVEL=5
export TRIALMATCHAI_FIRST_LEVEL_MAX_TRIALS=5
export TRIALMATCHAI_FIRST_LEVEL_PER_CHANNEL_SIZE=5
export TRIALMATCHAI_SEARCH_CANDIDATE_LIMIT=20
export TRIALMATCHAI_RAG_NO_THINK=true
export TRIALMATCHAI_MLX_MAX_NEW_TOKENS=2048

uv run trialmatchai e2e \
  --input data/patients/mlx_smoke_patient.txt \
  --format text \
  --processed-trials-folder data/processed_trials \
  --processed-criteria-folder data/processed_criteria \
  --no-entities \
  --reingest \
  --rematch
```

What the switches mean:

- `RAG_BACKEND` selects MLX for eligibility reasoning.
- `RERANKER_BACKEND` selects MLX for the reranking step.
- The two model variables select Qwen3 for both temporary smoke-test roles.
- `EMBEDDER_USE_GPU=false` avoids a CUDA check for the BGE-M3 search embedder.
- `ENTITY_BACKEND=disabled` and `CONCEPT_LINKER_ENABLED=false` remove optional stages
  from this first wiring test.
- The search limits reduce the candidate set to approximately five trials.
- `RAG_NO_THINK=true` tells Qwen3 not to spend the output budget on a private thinking
  preamble before the required JSON.
- `MLX_MAX_NEW_TOKENS=2048` gives the structured JSON response enough room to finish.
- `--reingest` imports the input again, and `--rematch` forces a fresh match.

This run is a wiring test, not a medical recommendation and not a benchmark result.
Inspect the generated files under the configured `results/` directory, especially the
ranked trial list, per-trial eligibility JSON, and raw model text.

### MLX-LM sampling compatibility note

MLX-LM's command-line interface accepts `--temp`, but its Python generation function
passes sampling through a sampler callback. The adapter therefore uses
`mlx_lm.sample_utils.make_sampler(temperature)` and passes the resulting sampler to
generation. Passing `temp` directly caused this runtime error with MLX-LM 0.31.3:

```text
TypeError: generate_step() got an unexpected keyword argument 'temp'
```

The fix is covered by `tests/test_mlx_backend.py` and the complete test suite passes.

### Structured-output note

The first end-to-end run completed retrieval and ranking, but the eligibility response
for one trial was cut off inside `<think>` after the default 256-token limit. TrialMatchAI
correctly saved an `invalid_json_response` artifact and continued, but that is not a
usable eligibility explanation. Disabling thinking and increasing the MLX output budget
to 2048 tokens addresses this smoke-test failure. This setting is specific to the local
Qwen3 experiment and should be recorded in any later benchmark manifest.

## 11. Planned next steps

1. Run the Qwen3 MLX smoke test on the physical M5 Mac.
2. Run the Qwen3.5 plus Gemma4 smoke test using the integrated `mlx_vlm` and `mlx`
   backends.
3. Run one patient against a tiny, fixed trial subset.
4. Inspect the saved JSON and raw text outputs.
5. Compare the original setup and modern MLX setup on a reduced calibration dataset.
6. Build the TAIM adapter using the shared ranking and provenance contracts.
7. Run the agreed benchmark using graded nDCG@10 as the primary metric.

## References

- [MLX-LM official repository](https://github.com/ml-explore/mlx-lm)
- [Qwen3-8B MLX 4-bit model card](https://huggingface.co/mlx-community/Qwen3-8B-4bit)
- [Gemma 4 MLX 12B model card](https://huggingface.co/mlx-community/gemma-4-12B-4bit)

## Qwen3.5 investigation note

Qwen3.5-9B-4bit loaded on the M5 and used about 6.7 GB peak memory, but the short
text-only smoke prompt produced corrupted-looking output. The model card identifies it
as a multimodal checkpoint converted with an older MLX-VLM release. Current MLX-VLM
release notes mention Qwen3.5 fixes and text-only wrapper fixes, so this may be a runtime
compatibility problem rather than a hardware problem.

The newest MLX-VLM release requires a newer Transformers version than TrialMatchAI's
entity/LLM extras currently pin. We therefore keep the project lock compatible and test
the newer MLX-VLM in an isolated environment:

```bash
uv run --isolated --with "mlx-vlm==0.6.6" \
  python -m mlx_vlm.generate \
  --model mlx-community/Qwen3.5-9B-4bit \
  --prompt "Reply with exactly: Qwen 3.5 works" \
  --max-tokens 32 \
  --temperature 0.0 \
  --thinking-mode disabled
```

Only if this produces clean text should Qwen3.5 be integrated into the TrialMatchAI
adapter. Otherwise, Qwen3 remains the proven local reasoning fallback.

## Modern model-stack smoke run

After installing the resolved optional dependencies:

```bash
uv sync --extra llm --extra mlx --extra mlx-vlm
```

select the modern model pair with environment variables:

```bash
export TRIALMATCHAI_RAG_BACKEND=mlx_vlm
export TRIALMATCHAI_RERANKER_BACKEND=mlx
export TRIALMATCHAI_MODEL_BASE_MODEL=mlx-community/Qwen3.5-9B-4bit
export TRIALMATCHAI_MODEL_RERANKER_MODEL_PATH=mlx-community/gemma-4-e4b-it-4bit
export TRIALMATCHAI_EMBEDDER_USE_GPU=false
export TRIALMATCHAI_ENTITY_BACKEND=disabled
export TRIALMATCHAI_CONCEPT_LINKER_ENABLED=false
export TRIALMATCHAI_TRIALS_JSON_FOLDER=data/processed_trials
export TRIALMATCHAI_SEARCH_MAX_TRIALS_FIRST_LEVEL=5
export TRIALMATCHAI_FIRST_LEVEL_MAX_TRIALS=5
export TRIALMATCHAI_FIRST_LEVEL_PER_CHANNEL_SIZE=5
export TRIALMATCHAI_SEARCH_CANDIDATE_LIMIT=20
export TRIALMATCHAI_RAG_NO_THINK=true
export TRIALMATCHAI_MLX_MAX_NEW_TOKENS=2048
```

Then run the same synthetic patient:

```bash
uv run trialmatchai e2e \
  --input data/patients/mlx_smoke_patient.txt \
  --format text \
  --processed-trials-folder data/processed_trials \
  --processed-criteria-folder data/processed_criteria \
  --no-entities \
  --reingest \
  --rematch
```

This run tests the full modern pair: Qwen3.5 performs the long eligibility response,
while Gemma4 performs criterion reranking. It is still a wiring test, not a clinical
quality result.

The first successful modern run produced:

```text
Qwen3.5 MLX-VLM eligibility output: valid JSON
Gemma4 MLX reranker: loaded and completed
TrialMatchAI pipeline: complete
```

For the synthetic modern patient, the selected trial was `NCT05752136` and the model
returned `Likely Ineligible (leaning toward exclusion)`. The result contained 11
inclusion evaluations and 16 exclusion evaluations. The ranking score is an internal
pipeline score, not a probability that the patient is eligible.
