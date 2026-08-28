#!/usr/bin/env python3
"""Shared-engine generation-equivalence probe. NOT A QUALIFICATION RUN.

Measures whether sharing one phi-4 vLLM engine between query expansion and
eligibility reasoning (fork commit 68eb842) changes GENERATIONS, versus giving
eligibility a fresh engine that never served expansion.

Design (one topic, TREC-CT 2021 topic 1, frozen expansion keywords):

  S1, S2  shared-engine arms: build the expander engine exactly as the live
          pipeline does, issue the topic's ONE real expansion generation on it
          (warm-up), leave the engine cached, then run the match stage in the
          same process. Eligibility resolves the same engine-cache key and
          reuses the engine that served expansion. Run twice: S1 vs S2 is the
          run-repeatability control.
  T1, T2  fresh-engine arms: identical process, identical config, but NO
          expansion generation. Eligibility builds an identically-constructed
          engine that has served nothing. T1 vs T2 is the control again.
  X1      expansion-construction arm: one expansion generation on an engine
          built the PRE-fix way (no LoRA registered at all), versus S1/S2's
          engine (LoRA registered via enable_lora but omitted per request).
          Tests 68eb842's claim that omitting the LoRA request at generate
          time equals running the unadapted model. No match stage.

Every arm reads the SAME frozen inputs: the topic-1 profile and the
query-expanded summary copied from completed A40 run l4-full75-v3-21376668
(evidence doc trialmatchai-shared-engine-inertness-2026-08-23.md, step 1:
freeze the nondeterministic stage, then vary only the engine sharing).
Nothing outside this probe's own workspaces is written.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path


def build_workspace(probe_root: Path, arm: str) -> Path:
    ws = probe_root / "workspaces" / arm
    if ws.exists():
        shutil.rmtree(ws)
    (ws / "profiles").mkdir(parents=True)
    (ws / "summaries").mkdir()
    (ws / "outputs").mkdir()
    (ws / "probe").mkdir()
    ref = probe_root / "reference"
    shutil.copy2(ref / "profiles/1.json", ws / "profiles/1.json")
    shutil.copy2(ref / "summaries/1.json", ws / "summaries/1.json")
    return ws


def load_probe_config(ws: Path) -> dict:
    from trialmatchai.config.config_loader import load_config

    cfg = load_config("src/trialmatchai/config/taim_l4_cuda.json")
    cfg.setdefault("patient_inputs", {})["profile_dir"] = str(ws / "profiles")
    cfg["patient_inputs"]["summary_dir"] = str(ws / "summaries")
    cfg.setdefault("paths", {})["output_dir"] = str(ws / "outputs")
    return cfg


def narrative_sentences(ws: Path) -> list[str]:
    # Exactly orchestration.expand_queries: profile notes first, summary fallback.
    profile = json.loads((ws / "profiles/1.json").read_text(encoding="utf-8"))
    narrative = [n.get("text", "") for n in profile.get("notes", []) if n.get("text")]
    if not narrative:
        summary = json.loads((ws / "summaries/1.json").read_text(encoding="utf-8"))
        narrative = list(summary.get("patient_narrative", []))
    return narrative


def one_expansion_generation(cfg: dict, ws: Path, label: str) -> None:
    """Issue exactly ONE expansion generation (what expand_queries does per
    patient) and record the raw bytes. Does NOT touch the frozen summary."""
    from trialmatchai.matching.eligibility_base import BaseTrialProcessor
    from trialmatchai.matching.query_expansion import build_query_expander
    from trialmatchai.utils.json_utils import extract_json_object

    expander = build_query_expander(cfg)
    if expander is None:
        raise RuntimeError("query_expansion.enabled is false; probe config is wrong")
    sentences = narrative_sentences(ws)
    joined = " ".join(s for s in sentences if s).strip()
    raw = expander._generate(joined)
    (ws / "probe" / f"expansion_raw_{label}.txt").write_text(raw, encoding="utf-8")
    try:
        parsed = extract_json_object(BaseTrialProcessor._strip_thinking_tags(raw))
    except Exception as exc:  # record parse failures, don't die on them
        parsed = {"_probe_parse_error": str(exc)}
    (ws / "probe" / f"expansion_parsed_{label}.json").write_text(
        json.dumps(parsed, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[probe] expansion generation done ({label}): {len(raw)} chars", flush=True)
    # As in expand_queries: drop the wrapper, keep the cached engine.
    del expander


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["S1", "S2", "T1", "T2", "X1"])
    ap.add_argument("--probe-root", required=True)
    args = ap.parse_args()
    probe_root = Path(args.probe_root)
    arm = args.arm

    ws = build_workspace(probe_root, arm)
    cfg = load_probe_config(ws)
    print(f"[probe] arm={arm} workspace={ws}", flush=True)

    if arm == "X1":
        # Pre-fix engine construction for expansion: strip the model section's
        # adapter so load_vllm_engine registers no LoRA at all (cache key ""),
        # which is byte-for-byte what the pre-68eb842 expander built.
        cfg_x = deepcopy(cfg)
        cfg_x.get("model", {}).pop("cot_adapter_path", None)
        cfg_x.get("model", {}).pop("cot_adapter_revision", None)
        cfg_x.setdefault("query_expansion", {})["adapter"] = None
        one_expansion_generation(cfg_x, ws, "noLoRA")
        return 0

    if arm in ("S1", "S2"):
        one_expansion_generation(cfg, ws, "shared")

    from trialmatchai.orchestration import run_matching

    rc = run_matching(cfg, resume=True, force=False)
    print(f"[probe] run_matching rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
