import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def body(name):
    text = (ROOT / "src/modes" / name).read_text()
    start = text.rindex("bool resultSaved = wm.saveIndividualJson")
    return text[start:start + 3500]


for driver in ("RunPrpOrLlMarin.cpp", "RunPrpOrLl.cpp"):
    text = (ROOT / "src/modes" / driver).read_text()
    end = body(driver)

    # The proof outcome is recorded where the proof is made, before options.proof
    # is cleared on a failure.
    assert "const bool proofRequested = options.mode == \"prp\" && options.proof;" in text, driver
    assert "proofCompleted = true;" in text, driver

    # End of job: the result is saved, then the worktodo entry is retired, and
    # only then is anything removed; the removals depend on both.
    save = 0
    retire = end.index("worktodoParser_->removeProcessedLine(activeWorktodoRawLine_)")
    assert end.count("removeProcessedLine(") == 1, driver
    assert "const bool entryRetired = !hasWorktodoEntry_ || retired;" in end, driver
    if driver == "RunPrpOrLlMarin.cpp":
        gate = re.search(r"if \(resultSaved && entryRetired\)\s*\{", end)
        assert gate and gate.start() > retire > save, driver
        exact_remove = end.index("std::filesystem::remove(ckpt_file, ec);")
        clear_state = end.index("backupManager.clearState();")
        assert gate.start() < exact_remove < clear_state, driver
    else:
        gate = re.search(r"if \(resultSaved && entryRetired\)\s*\{?\s*backupManager\.clearState\(\);", end)
        assert gate and gate.start() > retire > save, driver
    assert "ProofSetMarin::residueAction(" in end, driver
    assert "options.mode == \"prp\", options.wagstaff, proofRequested," in end, driver
    assert "proofCompleted, resultSaved, entryRetired);" in end, driver
    assert end.index("ProofSetMarin::clearResidues") > end.index("residueAction(") > retire, driver
    # The "kept" notice does not depend on a worktodo entry.
    assert "if (!resultSaved || !entryRetired) {" in end, driver
    # No unconditional removal ahead of the result write.
    head = text[:text.index("bool resultSaved = wm.saveIndividualJson")]
    tail_of_head = head[-400:]
    assert "backupManager.clearState();" not in tail_of_head, driver

marin = body("RunPrpOrLlMarin.cpp")
assert re.search(
    r"if \(resultSaved && entryRetired\) \{.*?"
    r"std::filesystem::remove\(ckpt_file, ec\);.*?"
    r"backupManager\.clearState\(\);",
    marin,
    re.S,
)
assert "delete_checkpoints(p, options.wagstaff, false, false);" not in marin

# The kept checkpoint must be the finished one: a final checkpoint is saved after
# the loop and before the proof releases the engine, and not when the run resumed
# from a checkpoint already at the end. It is saved for every mode, LL included.
full = (ROOT / "src/modes/RunPrpOrLlMarin.cpp").read_text()
final_save = re.search(r"if \(!\(r == 0 && ri == totalIters\)\) \{[^}]*save_ckpt\(static_cast<uint32_t>\(totalIters\)", full)
assert final_save, "no final checkpoint after the last iteration"
assert final_save.start() > full.index("for (uint64_t iter = resumeIter")
assert final_save.start() < full.index("saveProofResidue(d, totalIters)")
assert final_save.start() < full.index("delete eng;", final_save.start())

print("Proof residue gating source regression: PASS")
