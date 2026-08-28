#!/bin/bash
# SHARED-ENGINE EQUIVALENCE PROBE - NOT A QUALIFICATION RUN.
#
# One topic (TREC-CT 2021 topic 1), five sequential single-process runs on ONE
# A40, comparing GENERATIONS (not metrics) between the shared phi-4 engine
# (fork commit 68eb842) and a fresh identically-constructed engine:
#
#   S1, T1, S2, T2, X1   (interleaved so S-vs-T cannot be blamed on drift)
#
# Inputs are frozen: topic-1 profile + query-expanded summary copied from the
# completed A40 run l4-full75-v3-21376668, per the recipe in TAIM's
# docs/evidence/trialmatchai-shared-engine-inertness-2026-08-23.md ("freeze the
# nondeterministic stage, then vary only the engine sharing").
#
# ISOLATION: runs from its own clone at 7eba8f39 with its own venv under
# /work/hdd/bbnv/mmittal/taim-equivalence-probe. Reads the campaign's staged
# corpus/concept-store/model caches READ-ONLY (same stage-to-/tmp pattern as
# every existing job). Writes only under the probe root and /tmp. It does NOT
# touch the campaign checkout, its runs/, or any existing artifact.

#SBATCH --job-name=EQPROBE-a40
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=62g
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=/work/hdd/bbnv/mmittal/taim-equivalence-probe/logs/%x.%j.out
#SBATCH --error=/work/hdd/bbnv/mmittal/taim-equivalence-probe/logs/%x.%j.err

set -uo pipefail
module reset >/dev/null 2>&1 || true

R=/work/hdd/bbnv/mmittal/taim-trialmatchai-cuda
P=/work/hdd/bbnv/mmittal/taim-equivalence-probe
STORE_SRC=/work/hdd/bbnv/mmittal/taim-cs-shard/store/candidate-21352305
export TMPDIR=/tmp
export VLLM_LOGGING_LEVEL=INFO

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} started=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h /tmp | tail -1

# Canonical L4 wiring (HF caches staged to node-local /tmp, offline mode,
# pinned refs, forbidden depth vars cleared, spawn method). Its
# taim_l4_link_adapters call is a no-op here: the campaign checkout's links
# already exist and the function only creates missing ones.
# shellcheck disable=SC1091
source "$R/job/trialmatchai_l4_env.sh" --stage

# Probe-local vLLM compile cache: never write into the campaign's cache root.
export VLLM_CACHE_ROOT="$P/cache/vllm"
mkdir -p "$VLLM_CACHE_ROOT"

stage() {
  local src="$1" dst="$2"
  if [ ! -e "$dst" ]; then
    echo "[stage] $(basename "$src")"
    local t0=$SECONDS
    cp -a "$src" "$dst"
    echo "[stage] $(basename "$src") in $((SECONDS - t0))s"
  fi
}
stage "$STORE_SRC" /tmp/l4-concepts
stage "$R/corpus/common-trialgpt-75/search" /tmp/l4-search

export TRIALMATCHAI_CONCEPT_DB_PATH=/tmp/l4-concepts
export TRIALMATCHAI_SEARCH_DB_PATH=/tmp/l4-search
export TRIALMATCHAI_TRIALS_JSON_FOLDER="$R/corpus/common-trialgpt-75/processed_trials"

# The probe's own clone: same commit, same config digest, own venv.
CHECKOUT="$P/checkout"
mkdir -p "$CHECKOUT/models" "$P/logs"
for name in trialmatchai-phi4-reasoning-lora trialmatchai-gemma2-reranker-lora; do
  if [ ! -e "$CHECKOUT/models/$name" ]; then
    ln -s "$R/models/$name" "$CHECKOUT/models/$name"
  fi
done

cd "$CHECKOUT"
echo "=== probe checkout identity ==="
git rev-parse HEAD
echo "dirty: '$(git status --porcelain)'"
sha256sum src/trialmatchai/config/taim_l4_cuda.json
echo "=== forbidden caps present? (must be empty) ==="
env | grep -E "TRIALMATCHAI_(FIRST_LEVEL_MAX_TRIALS|FIRST_LEVEL_PER_CHANNEL_SIZE|SEARCH_MAX_TRIALS_FIRST_LEVEL|SEARCH_SECOND_LEVEL_KEEP_DIVISOR)=" || echo "  none"

UV=/work/hdd/bbnv/mmittal/demo/.local/bin/uv
overall_rc=0
for arm in S1 T1 S2 T2 X1; do
  echo "=== arm $arm started $(date -Is) ==="
  t0=$SECONDS
  "$UV" run --frozen --no-sync python "$P/probe_equivalence.py" \
    --arm "$arm" --probe-root "$P" > "$P/logs/arm-$arm.log" 2>&1
  rc=$?
  echo "=== arm $arm rc=$rc elapsed=$((SECONDS - t0))s ==="
  grep -h '\[lora\]' "$P/logs/arm-$arm.log" | tail -5 || true
  if [ "$rc" -ne 0 ]; then
    overall_rc=$rc
    tail -30 "$P/logs/arm-$arm.log"
  fi
done

echo "=== comparing arms ==="
python3 "$P/compare_arms.py" "$P" > "$P/logs/compare.log" 2>&1
cat "$P/COMPARISON.md" 2>/dev/null || tail -40 "$P/logs/compare.log"

if [ "$overall_rc" -eq 0 ]; then
  echo "PROBE_OK finished=$(date -Is)" > "$P/COMPLETED"
fi
echo "finished=$(date -Is) rc=$overall_rc"
exit "$overall_rc"
