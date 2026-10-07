#!/usr/bin/env python3
"""A restarted GMCHAIN must skip the families and phases it already finished."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
app = (root / 'src/core/App.cpp').read_text()

assert '#include "core/GmChainProgress.hpp"' in app
assert 'core::gm_chain_progress::Progress chain_progress(' in app
assert 'std::string chain_key = activeWorktodoRawLine_;' in app

start = app.index('auto run_family_pipeline = [&]')
end = app.index('if (requested_family == "BOTH")', start)
body = app[start:end]

# Finished families are skipped and return their recorded result.
assert 'chain_progress.family_rc(family)' in body
# Each phase is skipped when recorded, and recorded only after a clean finish.
for phase in ('pm1', 'ecm'):
    assert f'chain_progress.has(gcp::phase_token(family, "{phase}"))' in body, phase
    assert f'chain_progress.mark(gcp::phase_token(family, "{phase}"))' in body, phase
assert 'chain_progress.mark(gcp::done_token(family, family_rc))' in body
assert body.count('!interrupted && family_rc') >= 3

# The record is removed when the chain ends without being interrupted or failing.
assert 'if (!interrupted && rc != 2) chain_progress.clear();' in app
print('GM chain resume source test passed')
