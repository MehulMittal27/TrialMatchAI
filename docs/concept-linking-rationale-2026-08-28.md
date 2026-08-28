# The concept-linking changes: rationale, recorded after the fact

**Recorded 2026-08-28.** This repository's own [`codex-handoff.md`](codex-handoff.md) carries
the rule:

> Do not change ranking or eligibility semantics while building the shared adapter unless
> explicitly agreed.

Fourteen commits in this fork - every commit from `39bb3a3` through `051ef94` - are bare
one-liners with empty bodies, and among them are changes that do alter ranking-relevant
behaviour. **The history therefore contains no evidence that the rule was considered when those
changes were made.** That is the gap this document closes, with one thing stated up front:

**Writing the rationale down now does not retroactively create agreement.** A record produced
after the fact cannot show that the rule was weighed at the time, and it does not claim to. What
it does is make the reasoning inspectable instead of absent, so the next reader can judge each
change on its merits rather than reconstructing intent from diffs.

The concern this document answers is **not** that the changes are wrong. The review found two of
the three result-altering changes arguably bug fixes, and
[`fork-deviations-from-upstream.md`](fork-deviations-from-upstream.md) already records *what*
changed and how it propagates. This page records *why* each change was made, and whether it is a
bug fix or a deliberate divergence. What "explicit agreement" would have looked like - a commit
body or issue reference naming the rule and the decision - is exactly what the commits lack, and
what every later commit in this fork now carries.

The five most recent commits (`0ee05c5` through `7eba8f3`) are outside this document's scope:
their bodies already record their rationale in detail.

## The three result-altering concept-linking changes

All three are in concept linking - `src/trialmatchai/entities/linker.py` and its callers - and
all three were authored by the project owner (`hill <hannes.ill@tum.de>`). Mechanism details are
in the deviations document; this page adds the why and the verdict.

### 1. The candidate pool is five times deeper before ranking is decided

*Commit `f60f227`, "fix: preserve concept candidates until lexical reranking".*

**What it does.** Upstream fetched 10 candidate concepts per mention and reranked those 10. This
fork fetches 50 (`retrieval_limit`), reranks all 50 lexically, and only then truncates to
`search_limit` (10) before the accept gate. The same depth is applied to corpus-side criteria
linking (`linking.py`) and to the offline gate-tuning harness (`entities/linker_eval.py`).

**Why.** A concept the store's fused ranking placed at position 11 or below was unreachable no
matter how good the reranker was - it never entered the list the reranker saw. Correct concepts
buried at fused rank 11-50 by term-frequency effects could never be promoted. The change removes
an arbitrary reachability ceiling, not a scoring decision: the accept gate and the reranker are
untouched, they simply see the candidates they were always supposed to judge.

**Bug fix or divergence.** Arguably a bug fix - the truncation happened before the component
whose job was to order the candidates. But it is a **result-moving** change on every healthy
run, which is precisely the class the handoff rule covers, and no agreement is recorded.

**The consequence that must not be underestimated.** This is a component-depth change, and this
campaign has since **measured** what component-depth changes do: they reorganise the candidate
pool wholesale rather than nudging it - roughly **40% membership churn**, with newly-retrieved
items arriving **mid-list** rather than at the boundary. Deepening a component does not "add a
few extra candidates at the bottom"; it produces a substantially different pool with a
substantially different order. **Any comparison between this fork and upstream v0.7.0 carries
that difference, whether or not it was intended**, and no recall-style aggregate metric can be
assumed to surface it - membership can churn while a gate still passes.

### 2. Vocabulary restriction is applied during the search, not after it

*Commit `2657fee`, "Harden indexed concept and eligibility execution".*

**What it does.** Concept search restricted to a mention's allowed vocabularies/domains now
passes `prefilter=True` to LanceDB, applying the predicate inside the ANN search instead of
discarding out-of-vocabulary rows from the global nearest neighbours afterwards.

**Why.** Under upstream's post-filtering, a restricted search routinely returned far fewer rows
than requested - sometimes two or three where ten were asked for - because most global
neighbours were out of vocabulary. The linker was silently working from starved candidate lists.

**Bug fix or divergence.** Arguably a bug fix - the search was not delivering what it was asked
for. It is also the **single highest-impact behavioural change in the fork**: it changes the
candidate set for every mention in every run. The handoff rule applies to it in full, and no
agreement is recorded.

### 3. A broken concept store aborts the run instead of quietly degrading it

*Commits `2657fee`, `097d761` ("Fail closed on TrialMatchAI ANN errors"), `051ef94` ("Require
pinned TrialMatchAI ANN execution").*

**What it does.** A missing concept database, unreadable path, or LanceDB query error raises
`ConceptStoreSearchError` in every place that previously logged a warning and continued with
lexical-only linking. Fail-closed is opt-in: only a config that explicitly pins `ann_nprobes` or
`ann_refine_factor` (the pinned `taim_l4_cuda.json` does) gets it; the shipped `config.json`
keeps upstream's forgiving behaviour.

**Why.** Upstream's degraded runs completed and emitted rankings that looked like normal
TrialMatchAI results, with nothing in the artifacts to show the concept store had been absent.
For benchmark evidence that is worse than no result: a number that resembles a TrialMatchAI
number but was produced by a different system.

**Bug fix or divergence.** A **deliberate policy divergence**, not a bug fix - upstream's
log-and-continue was a design choice, and this fork rejects it for pinned benchmark execution.
The asymmetry matters: this change alters behaviour **only when something is already broken**. A
healthy run produces byte-identical results with or without it, so it is not a comparability
problem; it is the guarantee that a degraded run yields no number rather than a wrong one.

## Other empty-body commits that touch ranking or eligibility when exercised

### Eligibility JSON comma repair - `21fdc94` (Mehul Mittal)

Recovers eligibility verdicts whose JSON was invalid only by a missing comma, retries
still-invalid outputs individually, and raises if any survive. Upstream discarded these as parse
failures, silently dropping the affected trials from the ranking. Arguably a bug fix, but **when
exercised it changes which trials enter the ranking** - eligibility semantics under the rule -
and no agreement is recorded. On a run where every generation parses cleanly it is inert.

### MLX reranker probability scoring - `c419482`, on the backend added by `39bb3a3` (Mehul Mittal)

The MLX (Apple Silicon) reranker scores candidates by token probability, a scoring function that
exists nowhere upstream, on a backend that exists nowhere upstream. A deliberate addition, not a
fix to upstream behaviour. It is **not exercised** under the pinned CUDA config
(`taim_l4_cuda.json` selects vLLM), so which backend produced a given number has to be checked
per number - the deviations document says the same.

### Shortlist-divisor environment override - `f5d6a88` (Mehul Mittal)

Adds the `TRIALMATCHAI_SEARCH_SECOND_LEVEL_KEEP_DIVISOR` environment override for
`search.second_level_keep_divisor`. The default is unchanged (upstream's `settings.py` value of
3), so the commit itself moves nothing. It deserves its entry for what happened next: this
override family was later used to pin first-level retrieval to 15 trials of 26,149 in the TAIM
contract - "frozen into a System contract without a recorded reason", as the L4 contract now
puts it - and all four such variables are today **forbidden** by that contract, their absence
enforced before every launch. A knob added without recorded rationale became a silent depth cap.
That is the failure mode this document exists to prevent, demonstrated inside this very fork.

## The remaining empty-body commits, for completeness

Reviewed and judged not to touch ranking or eligibility semantics: `db8e51d` (free the MLX
reranker before eligibility - memory lifecycle), `946174b` and `263a227` (L4 KV-cache reserve
and the pinned CUDA config - execution environment), `5163594` (reuse models across benchmark
topics - load-time behaviour), `bca1714` (archive failed eligibility attempts instead of
overwriting them - evidence preservation; it changes what is *kept*, not what is *decided*), and
`aafb6d4` (the merge of `c419482`, covered above). They are listed so it is visible the whole
set of fourteen was considered, not filtered.

## Addendum, 2026-08-28: the shared-engine inertness claim - measurement designed, deferred

Commit `68eb842` (outside the empty-body set; its body records its rationale) makes query
expansion and eligibility reasoning share one phi-4 vLLM engine. It is **intended to alter
memory behaviour only, and that has not been measured.** Until it is, every result from this
fork at or after `68eb842` carries an unverified assumption: that two consumers of one engine
do not change each other's generations through sampling state, batching, KV cache reuse, or
dtype handling.

A direct measurement was designed, staged, and submitted (Delta job `21539429`, A40,
2026-08-28), then **cancelled before it started: it needs an eight-hour wall on a shared
allocation, and nothing currently blocked on its answer justified spending that now. This is a
deferral, not an oversight** - the distinction is recorded here precisely so an unexplained
absence does not read as one forever.

### The probe design, so it is never re-derived

Runnable copies live in `scripts/equivalence_probe/` in this repository; a fully staged
instance lives on Delta at `/work/hdd/bbnv/mmittal/taim-equivalence-probe` (an **isolated**
clone at `7eba8f39` with its own venv - never the campaign checkout, whose clean-tree gate a
stray file would fail). To run it: `sbatch equivalence_probe_a40.sbatch` from the probe root.
The design points that must not be lost:

- **The shared engine must be genuinely warm.** The shared-engine arms run a **real expansion
  generation first** - the topic's actual one-request warm-up, exactly as the live `e2e` path
  issues it - before the match stage reuses the engine. Without that, the experiment compares
  a warm engine against a cold one and attributes the difference to sharing.
- **Five arms, one topic (TREC-CT 2021 topic 1), one GPU, one node, interleaved S1 T1 S2 T2
  X1:** S1/S2 shared engine (warm-up then match in one process); T1/T2 an
  identically-constructed **fresh** engine (match only - what eligibility saw in the
  two-engine world); X1 one expansion generation on a **pre-fix-construction** engine with no
  LoRA registered at all, testing `68eb842`'s specific claim that omitting the LoRA request at
  generate time equals running the unadapted model.
- **The repeatability controls are not optional.** Same-architecture runs of this pipeline are
  already known not to be bit-identical (vLLM continuous batching reorders reductions; greedy
  decoding flips on near-ties at temperature 0.0, seed 1234). Without S1-vs-S2 and T1-vs-T2,
  an S-vs-T divergence is **unattributable** - it could be ordinary run-to-run variance. If
  the controls themselves diverge, that is the finding: generation is not stable even within
  one configuration, and "identical generations" is not a property this pipeline has.
- **Frozen inputs.** Every arm reads the same topic-1 profile and already-query-expanded
  summary, copied from completed A40 run `l4-full75-v3-21376668` - the recipe from TAIM's
  shared-engine inertness evidence doc: freeze the nondeterministic stage, then vary only the
  engine sharing. The frozen copies are staged under the probe root's `reference/`.
- **Compare outputs, not metrics, byte for byte:** the raw expansion generation, every raw
  per-trial eligibility generation (`NCT*.txt`), parsed verdicts, retrieval artifacts, and
  final ranking order (`compare_arms.py` writes `COMPARISON.md` and `hashes.json`). Scores
  cannot distinguish a real change from ordinary variance; matching aggregates prove nothing.

The probe writes only under its own workspaces and `/tmp`; it must never touch the campaign
checkout, its staged inputs, or any existing run. Job `21539429` never began executing, so no
partial probe artifacts exist.
