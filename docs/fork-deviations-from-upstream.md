# How this fork differs from upstream TrialMatchAI

**This repository is a fork. It is not the system the TrialMatchAI paper describes, and it is
not upstream `cbib/TrialMatchAI` either.** Its concept-linking stage retrieves and filters
candidate concepts differently from every upstream release, so it links different concepts to
the same text. That propagates into patient facts, first-level query expansion, criteria
linking, and every downstream score.

If you are about to run this code and quote a number from it, read [What is different that
changes results](#what-is-different-that-changes-results) first, and paste the sentence in
[Quoting a number from this fork](#quoting-a-number-from-this-fork) next to the number.

This is a statement about **comparability, not correctness**. Two of the three result-altering
changes are arguably bug fixes, and this fork's linking is plausibly *better* than what it
replaced. It is simply not the same measurement.

## Provenance: what you are running

| role | commit | what it is |
| --- | --- | --- |
| **this fork** | `051ef94fba260675fca59b0a6369b654604df8a0` (tree `9fc5e771a563771fb5c4b6457bd46be1b77ccb1f`) | `MehulMittal27/TrialMatchAI` `main` |
| **fork point** | `071a76ee6d82dcf96a59ed793767f9aee11ec0ea` | `git merge-base` with upstream, and exactly upstream tag `v0.7.0` |
| upstream now | `ee221e3106cc81dd633a25f07e97c55a36feac73` | `cbib/TrialMatchAI` `main`, version 0.8.1 |
| paper release | `395a486cabd3769c8a8da49d629f799d4c31a7b4` | upstream tag `v0.01` |

The paper release sits **184 commits before the fork point** and is a strict ancestor of it. The
original authors had already moved far past the paper before this fork existed, so *"this fork
versus the paper"* and *"this fork versus upstream"* are two different questions with two
different answers. **Everything below is measured against `v0.7.0`, the fork point** - it is not
a diff against the paper.

Scale of the divergence:

- **This fork since the fork point:** 14 commits, 41 files, +3822 / -756.
- **Upstream since the fork point:** 5 commits, 17 files, +522 / -30. That is what a future
  upstream sync would bring in; it is not part of this fork.

Of the 41 changed files, **21 are under `src/`** - 19 Python modules and two config files
(`config/config.json` and the added `config/taim_l4_cuda.json`). The rest are 15 test files,
three `docs/` pages, `pyproject.toml` and `uv.lock`. Reproduce with
`git diff --name-only 071a76e HEAD -- src/`. (Earlier notes quoted 19 or 20 here; 19 counts only
`.py` files and 20 is simply wrong. 21 is the number of changed files under `src/`.)

### The version string does not identify this code

`src/trialmatchai/__init__.py` still reports `__version__ = "0.7.0"`, unchanged since the fork
point. **This fork and upstream `v0.7.0` report the same version while behaving measurably
differently in retrieval.** Any artifact that records "TrialMatchAI 0.7.0" is ambiguous between
the two. **The commit id `051ef94fba260675fca59b0a6369b654604df8a0` is the only unambiguous
identifier** for this code; record it, not the version string.

## What is different that changes results

All three of these are in **concept linking** - `src/trialmatchai/entities/linker.py` and its
callers. They are described here as what the system now *does* differently, not as line changes.

### 1. The candidate pool is five times deeper before ranking is decided

*Commit `f60f227`, "fix: preserve concept candidates until lexical reranking".*

Upstream fetched 10 candidate concepts for a mention and reranked those 10. **A concept that
the store's fused ranking placed at position 11 or below was unreachable, no matter how good the
reranker was** - it never entered the list the reranker saw.

This fork fetches 50 (`retrieval_limit`, now also the shipped default in
`config/config.json`), reranks all 50 lexically, and only then truncates to `search_limit` (10)
before the accept gate. A correct concept buried at fused rank 11-50 by a term-frequency effect
can now be promoted to the top and accepted.

`src/trialmatchai/linking.py` passes the same limit into the **corpus** linker, so trial
criteria are linked at the same depth as patients. That means **the indexed corpus itself
differs**, not only the query side. `entities/linker_eval.py` was moved to the same depth so the
offline gate-tuning harness measures the pool production actually uses.

### 2. Vocabulary restriction is applied during the search, not after it

*Commit `2657fee`, "Harden indexed concept and eligibility execution".*

Concept search is restricted to the vocabularies and domains a mention's schema allows. Upstream
left LanceDB on its default **post**-filtering: the ANN search returns the `limit` global nearest
neighbours and *then* discards the ones failing the vocabulary predicate. **A restricted search
therefore routinely returned far fewer rows than requested** - sometimes two or three where ten
were asked for - because most of the global neighbours were out of vocabulary.

This fork passes `prefilter=True`, so the predicate is applied inside the search and the linker
receives a full `limit` of in-vocabulary candidates.

**This changes the candidate set for every mention in every run.** It is the single
highest-impact behavioural change in the fork. The pinned `config/taim_l4_cuda.json` also sets
`ann_nprobes: 64` and `ann_refine_factor: 4`, replacing whatever LanceDB's defaults happened to
be with explicit, recorded values.

### 3. A broken concept store aborts the run instead of quietly degrading it

*Commits `2657fee`, `097d761` ("Fail closed on TrialMatchAI ANN errors"), `051ef94` ("Require
pinned TrialMatchAI ANN execution").*

Upstream's design was to log a warning and continue with reduced capability. A missing concept
database, an unreadable path, or a LanceDB query error produced lexical-only linking, and the run
completed and emitted a ranking that looked like a normal TrialMatchAI result.

This fork raises `ConceptStoreSearchError` instead, in each place that previously swallowed the
failure: `entities/linker.py` (the ANN query itself), `entities/annotator.py` (store
construction, and the annotation thread pool, so a single failing worker cannot be absorbed by
the generic handler), `matching/retrieval/synonyms.py` (the one place the first-level planner
reaches into linking), and `cli/import_patient.py`. The switch is
`ann_retrieval_configured()`: **only a config that explicitly pins `ann_nprobes` or
`ann_refine_factor` opts in.** `taim_l4_cuda.json` does; the shipped `config.json` does not, and
keeps upstream's forgiving behaviour.

> **The asymmetry, stated explicitly:** this change alters behaviour **only when something is
> already broken**. A healthy run produces byte-identical results with or without it. Changes
> (1) and (2) change **healthy** runs. So fail-closed is not a comparability problem - it is the
> guarantee that a degraded run yields *no* number rather than a number that resembles a
> TrialMatchAI number but was produced by a different system.

## What is untouched - as important as what changed

Most of the pipeline is byte-identical to upstream `v0.7.0`. Verify any line below with
`git diff --name-only 071a76e HEAD -- <path>`.

- **`constraints/` entirely.** Eligibility constraint extraction and scoring are unchanged.
- **`matching/retrieval/trial_retrieval.py` and the rest of the retrieval package** -
  `criteria_retrieval.py`, `first_level_planner.py`, `location.py`, `__init__.py`. Second-level
  trial retrieval and its scoring are unchanged. Only `synonyms.py` differs, by the three-line
  fail-closed re-raise in change (3).
- **`entities/recognizers.py`.** Entity *recognition* (GLiNER2) is upstream. **Only entity
  *linking* changed** - the spans this system finds are the spans upstream finds.
- **`matching/eligibility_reasoning_vllm.py`** - the eligibility path that
  `config/taim_l4_cuda.json` actually runs - along with `eligibility_reasoning_transformers.py`
  and `matching/ranking.py`. *(Stale as of `7eba8f3`, which post-dates this page's last
  update: that commit adds an engine-confirmed adapter-attachment assertion to the vLLM
  eligibility path and the query expander. It observes and asserts which adapter each request
  actually ran with; it does not change what is generated on a healthy run - a wrong or missing
  adapter now raises instead of running silently.)*
- **`trec/` entirely.** No evaluation code and no TREC track support was added or changed.
- **`finetuning/`, `interop/`, `registry/`, `schemas/`, `utils/`** - all unchanged.

## Everything else the fork adds

None of this changes ranking under the pinned CUDA config, but it is what the other ~90% of the
diff is:

- **MLX backends for Apple Silicon** - `matching/eligibility_reasoning_mlx.py`,
  `matching/eligibility_reasoning_mlx_vlm.py`, `models/llm/mlx_loader.py`, `mlx_vlm_loader.py`
  and `mlx_reranker.py`, plus MLX branches in `main.py`, `query_expansion.py` and the preflight
  check. See [`mac-mlx-implementation.md`](mac-mlx-implementation.md).
- **An eligibility JSON comma repair** in `matching/eligibility_base.py`, which recovers verdicts
  upstream discarded as parse failures, plus an individual retry of still-invalid outputs and a
  fail-closed raise if any survive.
- **Retry-evidence preservation** - failed eligibility attempts are archived rather than
  overwritten, so a resumed run leaves an audit trail.
- **Batched, streaming index building** in `orchestration.py` and `search/lancedb_backend.py`,
  so a full-corpus build fits bounded memory. Index contents after a complete build are
  identical.
- **`config/taim_l4_cuda.json`** - a fully addressed publication config: every model and adapter
  pinned by commit hash, a fixed seed, explicit ANN controls and a GPU memory budget, all
  asserted by `tests/test_deployment_readiness.py` so a silent edit fails CI.
- **Archive-layout normalisation** in `cli/bootstrap_data.py`; see
  [`bootstrap-data-directory-normalization.md`](bootstrap-data-directory-normalization.md).

Two of these do move numbers **when exercised**: the JSON comma repair changes which trials
enter the ranking, and the MLX path is a backend upstream does not have at all, with a reranker
scoring function that exists nowhere upstream. `taim_l4_cuda.json` selects vLLM, so **which
backend produced a given number has to be checked per number.**

## The shared phi-4 engine: one unverified assumption (recorded 2026-08-28)

Commit `68eb842` (post-dating this page's pinned state, like the `7eba8f3` note above) makes
query expansion and eligibility reasoning share one phi-4 vLLM engine instead of loading the
model twice. The change is **intended to alter memory behaviour only, and that has not been
measured.** Every result produced from this fork at or after `68eb842` therefore carries an
**unverified assumption**: that sharing one LLM engine between two consumers does not change
generation. Sampling state, batching, KV cache reuse and dtype handling can all
cross-contaminate between two consumers of one engine, and this campaign has already been
burned by an "inert" assumption of exactly this class (float16 scores that were not
batch-invariant).

What exists today is a layered code-reading argument (TAIM's
`docs/evidence/trialmatchai-shared-engine-inertness-2026-08-23.md`: per-request settings
established, adapter attachment armed, retrieval bit-identity not establishable), which is not
a measurement. A direct generation-level measurement **was designed and submitted** (Delta job
`21539429`, 2026-08-28) and **cancelled before it started**: it needs an eight-hour A40 wall
and nothing is currently blocked on its answer. **It is deferred, not overlooked.** The full
design - including the controls without which its result would be unattributable - is recorded
in [`concept-linking-rationale-2026-08-28.md`](concept-linking-rationale-2026-08-28.md), and
runnable copies of the probe live in `scripts/equivalence_probe/`, so picking it up requires no
re-derivation.

## Where a future upstream sync will collide (measured 2026-08-28)

With an `upstream` remote now configured and fetched (no merge, no rebase), the divergence
this page could previously only quote is checkable: upstream `main` is still exactly
`ee221e3` (version 0.8.1) - the same five commits, 17 files, +522/-30 recorded above; upstream
has not moved since this page was written. Upstream touched **none** of the concept-linking
files changed by this fork.

Upstream commit `e4a6d5b` does, however, change guided-JSON **generation behaviour** in two
files this fork also modified. These are **the collision points for any future rebase onto
upstream**, named here so they do not have to be discovered mid-rebase:

- `src/trialmatchai/matching/query_expansion.py` - upstream relaxes the keyword schema
  (`expanded_sentences` loses its per-string length cap, `maxItems` 30 -> 15) and adds
  `disable_any_whitespace` to structured outputs; this fork changed the same file's engine
  construction (`68eb842`) and generation call (`7eba8f3`).
- `src/trialmatchai/matching/eligibility_reasoning_vllm.py` - upstream adds
  `disable_any_whitespace` to the eligibility schema constraint; this fork added the
  engine-confirmed adapter assertion (`7eba8f3`).

Upstream also pins the structured-outputs backend to xgrammar with whitespace disabled at
**engine build time** in `src/trialmatchai/models/llm/vllm_loader.py` (untouched by this
fork, so it merges cleanly - but it changes what both stages generate whenever guided JSON is
on, which the pinned L4 config enables for eligibility). The remaining both-sides files are
metadata and docs: `README.md`, `pyproject.toml`, `uv.lock`.

## Quoting a number from this fork

Copy this next to any published number produced from this repository:

> Produced on the TrialMatchAI fork `MehulMittal27/TrialMatchAI` at commit
> `051ef94fba260675fca59b0a6369b654604df8a0`, which changes concept-linking candidate retrieval
> relative to upstream `v0.7.0` and the paper release `v0.01`; not directly comparable to
> published TrialMatchAI results.

## No recorded rationale for the linking changes

This repository's own [`codex-handoff.md`](codex-handoff.md) carries the rule:

> Do not change ranking or eligibility semantics while building the shared adapter unless
> explicitly agreed.

All 14 commit messages in this fork are bare one-liners with empty bodies (the only non-empty
body is a merge commit echoing its branch's single commit subject). **There is therefore no
written record of the concept-linking changes being weighed against that rule.**

This is recorded as an observation about documentation, not as a criticism of the changes. They
are defensible on their merits and two of the three are arguably bug fixes. It is noted here
because the absence of that record is the reason this page had to be written after the fact
rather than read off the history - and it is why the page exists at all.

The rationale for each of these changes - what it does, why, and whether it is a bug fix or a
deliberate divergence - is now recorded after the fact in
[`concept-linking-rationale-2026-08-28.md`](concept-linking-rationale-2026-08-28.md). Writing it
down did not retroactively create the agreement the rule asks for; it makes the reasoning
inspectable instead of absent.

## Reproducing this comparison

```sh
git fetch https://github.com/cbib/TrialMatchAI.git v0.7.0 main
git merge-base HEAD FETCH_HEAD            # -> 071a76e…, which is tag v0.7.0
git diff --shortstat 071a76e HEAD         # -> 41 files, +3822 / -756
git diff --name-status 071a76e HEAD
git diff 071a76e HEAD -- src/trialmatchai/entities/linker.py
```

A fuller file-by-file analysis - including the diff upstream has accumulated since the fork
point, a future-merge overlap analysis, and a list of apparently unintentional changes such as a
transitive `gliner` downgrade on the entity-recognition path and a loosened `transformers` pin -
was produced by an external review and is not committed here.
