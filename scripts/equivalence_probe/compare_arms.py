#!/usr/bin/env python3
"""Compare the equivalence-probe arms byte-for-byte. Stdlib only.

Writes COMPARISON.md and hashes.json into the probe root. Compares OUTPUTS,
not metrics: raw expansion generations, raw per-trial eligibility generations
(NCT*.txt), parsed eligibility verdicts (NCT*.json), retrieval artifacts, and
final ranking order.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import sys
from pathlib import Path

RETRIEVAL_FILES = [
    "keywords.json",
    "first_level_query_plan.json",
    "first_level_candidates.json",
    "first_level_scores.json",
    "nct_ids.txt",
    "top_trials.txt",
]


def sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_divergence(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1


def ranked_order(topic_dir: Path) -> list[str] | None:
    p = topic_dir / "ranked_trials.json"
    if not p.is_file():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    rows = data.get("RankedTrials", data) if isinstance(data, dict) else data
    out = []
    for row in rows:
        if isinstance(row, dict):
            out.append(str(row.get("TrialID") or row.get("NCT Number") or row.get("nct_id") or row))
        else:
            out.append(str(row))
    return out


def compare_pair(root: Path, a: str, b: str, lines: list[str]) -> None:
    ta, tb = root / "workspaces" / a / "outputs" / "1", root / "workspaces" / b / "outputs" / "1"
    lines.append(f"\n## {a} vs {b}")

    lines.append("\n### Retrieval artifacts")
    for name in RETRIEVAL_FILES:
        ha, hb = sha(ta / name), sha(tb / name)
        mark = "IDENTICAL" if ha == hb and ha is not None else ("MISSING" if ha is None or hb is None else "*** DIFFERS ***")
        lines.append(f"- `{name}`: {mark}")

    lines.append("\n### Eligibility generations (raw NCT*.txt)")
    txts_a = sorted(p.name for p in ta.glob("NCT*.txt"))
    txts_b = sorted(p.name for p in tb.glob("NCT*.txt"))
    lines.append(f"- trial sets: {'IDENTICAL' if txts_a == txts_b else '*** DIFFER ***'} ({len(txts_a)} vs {len(txts_b)} files)")
    common = sorted(set(txts_a) & set(txts_b))
    n_same = 0
    for name in common:
        ca, cb = (ta / name).read_text(encoding="utf-8"), (tb / name).read_text(encoding="utf-8")
        if ca == cb:
            n_same += 1
        else:
            off = first_divergence(ca, cb)
            ratio = difflib.SequenceMatcher(None, ca, cb).ratio()
            lines.append(
                f"- `{name}`: *** DIFFERS *** (lengths {len(ca)}/{len(cb)}, "
                f"first divergence at char {off}, similarity {ratio:.4f})"
            )
            lines.append(f"    - {a} around divergence: `{ca[max(0, off - 60):off + 60]!r}`")
            lines.append(f"    - {b} around divergence: `{cb[max(0, off - 60):off + 60]!r}`")
    lines.append(f"- identical raw generations: {n_same}/{len(common)}")

    lines.append("\n### Parsed verdicts (NCT*.json)")
    n_same_json, n_diff_json = 0, []
    for name in sorted(set(p.name for p in ta.glob("NCT*.json")) & set(p.name for p in tb.glob("NCT*.json"))):
        if (ta / name).read_bytes() == (tb / name).read_bytes():
            n_same_json += 1
        else:
            da = json.loads((ta / name).read_text(encoding="utf-8"))
            db = json.loads((tb / name).read_text(encoding="utf-8"))
            va, vb = da.get("Final Decision"), db.get("Final Decision")
            n_diff_json.append(f"`{name}` ({'same' if va == vb else 'DIFFERENT'} verdict: {va!r} vs {vb!r})")
    lines.append(f"- byte-identical: {n_same_json}; differing: {len(n_diff_json)}")
    for d in n_diff_json:
        lines.append(f"    - {d}")

    ra, rb = ranked_order(ta), ranked_order(tb)
    lines.append("\n### Final ranking")
    if ra is None or rb is None:
        lines.append("- ranked_trials.json MISSING in at least one arm")
    elif ra == rb:
        lines.append(f"- order IDENTICAL ({len(ra)} trials)")
    else:
        moved = sum(1 for i, t in enumerate(ra) if i >= len(rb) or rb[i] != t)
        lines.append(f"- *** ORDER DIFFERS ***: {moved}/{len(ra)} rank positions changed")
        lines.append(f"    - {a}: {ra}")
        lines.append(f"    - {b}: {rb}")


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    lines = ["# Shared-engine equivalence probe: arm comparison", ""]
    lines.append("Arms: S1/S2 shared engine (expansion warm-up then match, one process);")
    lines.append("T1/T2 fresh engine (match only); X1 pre-fix no-LoRA expansion construction.")
    lines.append("All arms: same node, same GPU, same frozen topic-1 inputs.")

    lines.append("\n## Expansion generations")
    exp = {}
    for arm, label in [("S1", "shared"), ("S2", "shared"), ("X1", "noLoRA")]:
        p = root / "workspaces" / arm / "probe" / f"expansion_raw_{label}.txt"
        exp[arm] = p.read_text(encoding="utf-8") if p.is_file() else None
        lines.append(f"- {arm}: {'missing' if exp[arm] is None else f'{len(exp[arm])} chars, sha256 {sha(p)[:16]}'}")
    for a, b in [("S1", "S2"), ("S1", "X1")]:
        if exp.get(a) is not None and exp.get(b) is not None:
            if exp[a] == exp[b]:
                lines.append(f"- {a} vs {b}: IDENTICAL")
            else:
                off = first_divergence(exp[a], exp[b])
                ratio = difflib.SequenceMatcher(None, exp[a], exp[b]).ratio()
                lines.append(f"- {a} vs {b}: *** DIFFERS *** (first divergence at char {off}, similarity {ratio:.4f})")
                lines.append(f"    - {a}: `{exp[a][max(0, off - 60):off + 60]!r}`")
                lines.append(f"    - {b}: `{exp[b][max(0, off - 60):off + 60]!r}`")

    for a, b in [("S1", "S2"), ("T1", "T2"), ("S1", "T1"), ("S2", "T2")]:
        compare_pair(root, a, b, lines)

    hashes = {}
    for arm in ["S1", "S2", "T1", "T2", "X1"]:
        ws = root / "workspaces" / arm
        hashes[arm] = {
            str(p.relative_to(ws)): sha(p)
            for p in sorted(ws.rglob("*"))
            if p.is_file() and p.suffix in {".json", ".txt", ".html"}
        }
    (root / "hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True), encoding="utf-8")
    (root / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
