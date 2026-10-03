#!/usr/bin/env python3
"""A Ctrl-C during RunEcm stage 1 must save the checkpoint and publish nothing."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = (root / 'src/modes/RunEcm.cpp').read_text().splitlines()

hits = [l for l in src if 'if (interrupted) {' in l and 'Interrupted at curve' in l]
assert len(hits) == 1, hits
line = hits[0]
assert 'save_ckpt(' in line
assert 'write_result' not in line and 'publish_json' not in line
assert 'curves_tested_for_found' not in line

te = (root / 'src/modes/RunEcmTwistedEdwards.cpp').read_text()
assert 'Interrupt received — exiting without publishing a final result.' in te
print('PrMers ECM interrupt-without-result test passed')
