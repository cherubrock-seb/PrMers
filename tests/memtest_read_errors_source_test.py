#!/usr/bin/env python3
# -memtest must stop and fail on a failed OpenCL read, write or wait: a failed error count read left the count at
# 0, and the test reported "Errors 0" and exited 0 for memory it never checked.
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def code_only(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


src = code_only((ROOT / "src/modes/RunMemTest.cpp").read_text())
fn = src[src.index("int core::App::runMemtestOpenCL()"):]
loop = fn[fn.index("bool interrupted = false;"):fn.index("===== GPU Memtest Report =====")]

# Every read is checked: the error counts and the error report.
reads = list(re.finditer(r"clEnqueueReadBuffer\(", loop))
assert len(reads) == 8, f"expected 4 error count reads and 4 report reads, found {len(reads)}"
for m in reads:
    before = loop[max(0, m.start() - 20):m.start()]
    assert before.endswith("fails("), "unchecked clEnqueueReadBuffer in memtest: " + loop[m.start():m.start() + 120]

# Every reset of the error count is checked.
for m in re.finditer(r"zero_err\(\)", loop):
    before = loop[max(0, m.start() - 20):m.start()]
    assert before.endswith("fails("), "unchecked zero_err() in memtest: " + loop[max(0, m.start() - 60):m.start() + 60]
assert "-> cl_int" in fn[fn.index("auto zero_err"):fn.index("auto zero_err") + 80], "zero_err must return its status"

# The status of every kernel and wait is kept (or-ing the codes made up a different code) and checked right after
# the wait, before anything is read.
assert "err|=" not in loop.replace(" ", ""), "or-ed OpenCL statuses in memtest"
finishes = list(re.finditer(r"clFinish\(", loop))
assert len(finishes) == 7, f"expected 7 clFinish calls, found {len(finishes)}"
for m in finishes:
    before = loop[max(0, m.start() - 10):m.start()]
    assert before.endswith("keep("), "clFinish status not kept in memtest: " + loop[m.start():m.start() + 80]
    after = loop[m.end():m.end() + 200]
    assert re.match(r"queue\)\);\s*if \(fails\(err, ", after), "clFinish not checked in memtest: " + after
for call in ("clSetKernelArg(", "clEnqueueNDRangeKernel("):
    for m in re.finditer(re.escape(call), loop):
        before = loop[max(0, m.start() - 10):m.start()]
        assert before.endswith("keep("), f"{call} status not kept in memtest: " + loop[m.start():m.start() + 80]

# A failed call fails the test: it says so in the report and does not exit 0 (or 1, the interrupted code).
end = fn[fn.index("===== GPU Memtest Report ====="):]
assert "if (failed) rpt << \"Status: FAILED" in end
assert re.search(r"if \(failed\) \{[^}]*return -1;", end), "a failed memtest must exit like its other failures"

# A sub-buffer that cannot be created fails the test instead of leaving the rest of the memory untested.
sub = loop[loop.index("clCreateSubBuffer("):]
sub = sub[:sub.index("auto t0")]
assert "failed = true;" in sub and "break;" in sub

print("Memtest read error source regression: PASS")
