#!/usr/bin/env python3
"""GUI "Append & Run" must only append to worktodo from the HTTP worker thread.

Restarting the process from the submit callback killed the running test without a checkpoint.
The main thread's idle loops start the appended entry instead, and a Stop always wins over that
restart (core::StopRestartGate, exercised by tests/stop_restart_gate_test.cpp).
"""
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
app = (root / 'src/core/App.cpp').read_text()
algo_hpp = (root / 'include/core/AlgoUtils.hpp').read_text()
algo_cpp = (root / 'src/core/AlgoUtils.cpp').read_text()

m = re.search(r'auto submitFn = \[this\]\(const std::string& line\)\{(.*?)\n        \};', app, re.S)
assert m, 'submitFn lambda not found'
body = m.group(1)
assert 'restart_self' not in body, 'submitFn must not restart the process from the HTTP thread'
assert 'stop()' not in body, 'submitFn must not stop the GUI server from the HTTP thread'
# Only mark the entry pending after the line was written.
assert body.index('appendLine') < body.index('g_stop_restart_gate.markAppendPending();')

# Both idle loops (empty worktodo, and after a run finished) must pick the new entry up on the main
# thread, and both end on the gate's stop flag.
assert app.count('gui_restart_for_appended_entry(guiServer_, argc_, argv_);') == 2
assert 'while (!stop_requested() && gui_alive) {' in app
assert 'while (!stop_requested()) {' in app
assert not re.search(r'\bg_stop\b(?!_restart_gate)', re.sub(r'//.*', '', app)), 'stale g_stop flag'
helper = re.search(r'static void gui_restart_for_appended_entry\(.*?\n}\n', app, re.S)
assert helper, 'helper not found'
h = helper.group(0)
assert 'claimAppendedEntry() != core::StopRestartGate::Claim::Restart) return;' in h
assert h.index('claimAppendedEntry') < h.index('restart_self(argc, argv)')

# Every stop source goes through the gate: the App signal handler (also called by the GUI's Stop)
# and the generic SIGINT handler the modes install.
sig = re.search(r'void handle_signal\(int\) noexcept \{(.*?)\n  \}', app, re.S)
assert sig and 'core::stop_or_exit();' in sig.group(1)
stopfn = re.search(r'auto stopFn = \[&\]\(\)\{(.*?)\};', app, re.S)
assert stopfn and 'handle_signal(SIGINT);' in stopfn.group(1)
assert 'core::stop_or_exit();' in algo_cpp

# restart_self commits before it execs and returns when Stop came first.
rs = re.search(r'inline void restart_self\(int argc, char\* argv\[\]\) \{(.*?)\n}\n', algo_hpp, re.S)
assert rs, 'restart_self not found'
r = rs.group(1)
assert 'if (!core::g_stop_restart_gate.commitRestart()) {' in r
assert r.index('commitRestart') < r.index('CreateProcessA') and r.index('commitRestart') < r.index('util::execSelf(args)')
# ... and holds the worktodo write lock from the commit through the exec, so a GUI append on another
# thread is never cut off half-written.
lock = r.index('io::WorktodoParser::lockFileWrites();')
assert r.index('commitRestart') < lock < r.index('CreateProcessA') and lock < r.index('util::execSelf(args)')

# The bench and memtest SIGINT handlers are stop sources too.
assert 'static void prmers_bench_sigint(int) { prmers_bench_stop = 1; core::stop_or_exit(); }' in app
memtest = (root / 'src/modes/RunMemTest.cpp').read_text()
assert 'auto onint = +[](int){ stop_flag = 1; core::stop_or_exit(); };' in memtest
print('GUI append-run source test passed')
