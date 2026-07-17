#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
hdr = (root / "src/Queue.h").read_text()
text = (root / "src/Queue.cpp").read_text()
api = (root / "src/EngineApi.cpp").read_text()

assert "bool appleMarkerWait" in hdr
assert 'std::getenv("AEVUM_APPLE_QUEUE_MARKER_WAIT")' in text
assert "if (!appleMarkerWait)" in text

start = text.index("void Queue::queueMarkerEvent()")
end = text.index("void Queue::waitForMarkerEvent()", start)
body = text[start:end]

fast = body.index("if (!appleMarkerWait)")
flush = body.index("::flush(get());", fast)
reset = body.index("queueCount = 0;", flush)
ret = body.index("return;", reset)
marker = body.index("markerEvent = enqueueMarker(get());")
assert fast < flush < reset < ret < marker

# Default Apple pacing submits work but never waits or polls.
default_path = body[fast:ret + len("return;")]
assert "waitForMarkerEvent();" not in default_path
assert "finish();" not in default_path
assert "enqueueMarker" not in default_path

# The old marker path remains available after the default early return and is
# still Apple-only where the explicit submission before polling is required.
legacy = body[marker:]
assert "#if defined(__APPLE__)" in legacy
assert "::flush(get());" in legacy
assert "markerQueued = true;" in legacy

assert "nonblocking clFlush" in api
assert "AEVUM_APPLE_QUEUE_MARKER_WAIT=1" in api
print("Aevum Apple nonblocking queue pacing test passed")
