from __future__ import annotations

# "0.7.0+taim", not "0.7.0": this fork changes concept-linking behaviour relative to
# upstream v0.7.0 (see docs/fork-deviations-from-upstream.md), so reporting a bare
# release version would misidentify the code. The +taim local-version suffix says
# "derived from v0.7.0, not v0.7.0". The commit id remains the only unambiguous
# identifier for this code; record it, not this string.
__version__ = "0.7.0+taim"
