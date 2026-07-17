#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / "src/Gpu.cpp").read_text(encoding="utf-8")
start = source.index("vector<Word> Gpu::readChecked(Buffer<Word>& buf)")
end = source.index("Words Gpu::readAndCompress", start)
block = source[start:end]

assert "#if defined(__APPLE__)" in block
apple, non_apple = block.split("#else", 1)
assert apple.count("readOut(buf)") == 2
assert "vector<Word> first = readOut(buf);" in apple
assert "vector<Word> second = readOut(buf);" in apple
assert "if (first == second)" in apple
assert "GPU double-read mismatch at word" in apple
assert "sum64(" not in apple
assert "bufSumOut" not in apple
assert "queue.finish();" in apple

assert "sum64(bufSumOut, N, buf);" in non_apple
assert "bufSumOut.readAsync(expectedVect);" in non_apple
assert "GPU read failed:" in non_apple

# Model the intended acceptance rule: only two identical complete reads pass.
def accepted(first, second):
    return first == second

assert accepted([1, 2, 3], [1, 2, 3])
assert not accepted([1, 2, 3], [1, 2, 4])
assert not accepted([0, 2, 3], [1, 2, 3])

print("Aevum Apple readChecked deterministic double-sync test passed")
