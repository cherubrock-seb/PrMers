#!/usr/bin/env python3
"""Source regression test: the Marin LL-SAFE driver must delete its checkpoint
only when the result was saved and, for a worktodo entry, only after the entry
was retired (run_llsafe_worktodo_retire_failure.sh checks the behaviour)."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    text = (ROOT / "src/modes/RunLlSafeMarin.cpp").read_text()
    start = text.index("int App::runLlSafeMarin()")
    body = text[start:]
    calls = list(re.finditer(r"delete_checkpoints\s*\(", body))
    if not calls:
        print("FAIL: LL-SAFE no longer deletes its checkpoint at all")
        return 1
    for m in calls:
        before = body[:m.start()]
        guard = before.rfind("if (")
        if guard == -1 or "resultSaved" not in before[guard:] or "}" in before[guard:]:
            print("FAIL: LL-SAFE calls delete_checkpoints without checking that the result was saved")
            return 1
        if "retired" not in before[guard:]:
            print("FAIL: LL-SAFE calls delete_checkpoints without checking that the worktodo entry was retired")
            return 1
        retire = before.rfind("removeProcessedLine(")
        if retire == -1:
            print("FAIL: LL-SAFE deletes its checkpoint before retiring the worktodo entry")
            return 1
    print("LL-SAFE checkpoint retention source test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
