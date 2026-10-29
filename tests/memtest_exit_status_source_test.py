#!/usr/bin/env python3
# -memtest must not exit 0 when it found memory errors, or when it could not allocate any memory to test: it used to
# print the error count (or a coverage of 0 MB) and exit 0, so a script or a burn-in loop saw a pass.
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def code_only(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


src = code_only((ROOT / "src/modes/RunMemTest.cpp").read_text())
fn = src[src.index("int core::App::runMemtestOpenCL()"):]
end = fn[fn.index("===== GPU Memtest Report ====="):]
end = end[:end.rindex("}")]

assert "const bool nothing_tested = bufs.empty();" in end, "a memtest that allocated nothing must be told apart"
assert re.search(r"if \(total_err\) rpt << \"Status: FAILED, ", end), "the report must say errors fail the test"
assert re.search(r"else if \(nothing_tested\) rpt << \"Status: FAILED, ", end), "the report must say nothing was tested"

# The last return: a failure exits like the memtest's other failures (-1), never 0 or the interrupted code 1.
fail = re.search(r"if \(total_err \|\| nothing_tested\) \{[^}]*return -1;\s*\}\s*return interrupted \? 1 : 0;", end)
assert fail, "memtest errors or an empty test must return -1 before the pass/interrupted return"

print("Memtest exit status source regression: PASS")
