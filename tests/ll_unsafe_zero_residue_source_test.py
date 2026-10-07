#!/usr/bin/env python3
"""Source regression test: LL-UNSAFE (runPrpOrLlMarin, mode "ll") accepts both
representations of zero as a prime verdict, so it must also report the
canonical zero residue when the engine holds 0 as 2^p - 1, as LL-SAFE2 does."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    text = (ROOT / "src/modes/RunPrpOrLlMarin.cpp").read_text()
    start = text.index("int App::runPrpOrLlMarin()")
    body = text[start:]
    ok = re.search(r'options\.mode == "ll" && is_prp_prime && digit\.equal_to_Mp\(\)\)\s*\{\s*'
                   r'(//[^\n]*\n\s*)?std::fill\(words\.begin\(\), words\.end\(\), 0u\)', body)
    if not ok:
        print("FAIL: LL-UNSAFE does not normalise a 2^p - 1 residue to zero")
        return 1
    print("LL-UNSAFE zero residue source test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
