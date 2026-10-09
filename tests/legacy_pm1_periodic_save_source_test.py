from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

src = (ROOT / "src/modes/RunPM1.cpp").read_text()
start = src.index("int App::runPM1() {")
end = src.index("\nint App::", start + 1)
body = src[start:end]

# saveState(x) records x + 1 and the loop resumes at the recorded value.  The
# interrupt save runs before bit i-1 is processed and stores i; the periodic
# save runs after it, so it must store i-1 (argument i-2) and be skipped when no
# bit is left.  Passing lastIter-1 would make a resume process bit i-1 twice.
interrupt = body.index("Interrupted signal received")
assert "saveState(buffers->input, lastIter-1, &E)" in body[interrupt:interrupt + 600]

periodic = body.index("seconds(180)")
block = body[periodic:periodic + 700]
assert "saveState(buffers->input, lastIter-2)" in block, block
assert "lastIter > 1" in block, block
assert "saveState(buffers->input, lastIter-1)" not in block, block

# The 10 s progress display resets lastDisplay, so the 180 s save must have its own clock or it
# never runs.
assert "now - lastDisplay >= seconds(180)" not in body
assert "now - lastBackup >= seconds(180)" in body
assert "lastBackup = now;" in block, block
print("legacy P-1 periodic save test passed")
