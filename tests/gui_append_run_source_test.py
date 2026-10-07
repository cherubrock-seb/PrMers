#!/usr/bin/env python3
"""GUI "Append & Run" must only append to worktodo from the HTTP worker thread.

Restarting the process from the submit callback killed the running test without a checkpoint.
The main thread's idle loops start the appended entry instead.
"""
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
app = (root / 'src/core/App.cpp').read_text()

m = re.search(r'auto submitFn = \[this\]\(const std::string& line\)\{(.*?)\n        \};', app, re.S)
assert m, 'submitFn lambda not found'
body = m.group(1)
assert 'restart_self' not in body, 'submitFn must not restart the process from the HTTP thread'
assert 'stop()' not in body, 'submitFn must not stop the GUI server from the HTTP thread'
assert 'g_gui_append_pending.store(true' in body

# Both idle loops (empty worktodo, and after a run finished) must pick the new entry up on the main thread.
assert app.count('gui_restart_for_appended_entry(guiServer_, argc_, argv_);') == 2
helper = re.search(r'static void gui_restart_for_appended_entry\(.*?\n}\n', app, re.S)
assert helper and 'restart_self(argc, argv)' in helper.group(0)
print('GUI append-run source test passed')
