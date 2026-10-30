#!/usr/bin/env python3
from pathlib import Path
import sys
path = Path(sys.argv[1])
src = path.read_text()
start = src.find("PRPResult Gpu::isPrimePRP(")
end = src.find("LLResult Gpu::isPrimeLL(", start)
body = src[start:end] if start >= 0 and end >= 0 else ""
tail_start = body.rfind("if (k >= kEndEnd)")
tail = body[tail_start:] if tail_start >= 0 else ""
terms = ["fs::path proofFile;", "if (proofSet) {", "background->waitEmpty();", "try {", "proofFile = saveProof(args, *proofSet);", "catch (...)", "if (Signal::stopRequested()) { throw; }", "Proof generation failed; reporting the result without a proof", "return {isPrime, finalRes64, nErrors, proofFile.string(), toHex(res2048)};"]
missing = [term for term in terms if term not in tail]
positions = [tail.find(term) for term in terms]
ordered = all(pos >= 0 for pos in positions) and positions == sorted(positions)
ok = start >= 0 and end >= 0 and tail_start >= 0 and not missing and ordered
print("proof_result_survival_source_test: ok" if ok else "proof_result_survival_source_test: FAIL")
print("missing=" + ",".join(missing)) if missing else None
print("positions=" + ",".join(str(pos) for pos in positions)) if not ordered else None
sys.exit(0 if ok else 1)
