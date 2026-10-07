import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def body(name):
    text = (ROOT / "src/modes" / name).read_text()
    start = text.index("bool resultSaved = wm.saveIndividualJson")
    return text[start:start + 3500]


for driver in ("RunPrpOrLlMarin.cpp", "RunPrpOrLl.cpp"):
    text = (ROOT / "src/modes" / driver).read_text()
    end = body(driver)

    # The proof outcome is recorded where the proof is made, before options.proof
    # is cleared on a failure.
    assert "const bool proofRequested = options.mode == \"prp\" && options.proof;" in text, driver
    assert "proofCompleted = true;" in text, driver

    # Nothing is removed before the result write, and the removals depend on it.
    save = 0
    gate = re.search(r"if \(resultSaved\)\s*\{?\s*backupManager\.clearState\(\);", end)
    assert gate and gate.start() > save, driver
    assert "ProofSetMarin::residueAction(" in end, driver
    assert "options.mode == \"prp\", options.wagstaff, proofRequested," in end, driver
    assert "proofCompleted, resultSaved);" in end, driver
    assert end.index("ProofSetMarin::clearResidues") > end.index("residueAction(") > save, driver
    # The "could not be saved" notice does not depend on a worktodo entry.
    assert "if (!resultSaved) {" in end, driver
    # No unconditional removal ahead of the result write.
    head = text[:text.index("bool resultSaved = wm.saveIndividualJson")]
    tail_of_head = head[-400:]
    assert "backupManager.clearState();" not in tail_of_head, driver

marin = body("RunPrpOrLlMarin.cpp")
assert re.search(r"if \(resultSaved\) \{\s*backupManager\.clearState\(\);\s*delete_checkpoints\(", marin)

print("Proof residue gating source regression: PASS")
