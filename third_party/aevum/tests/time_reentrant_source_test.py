#!/usr/bin/env python3
"""Host code must not call localtime(), gmtime(), ctime() or asctime().

Those return a pointer to one static buffer shared by every thread (localtime and gmtime share it in glibc), and each
worker thread timestamps its log lines, so two workers logging at once race on it.  Use the localtime_r/gmtime_r
(localtime_s/gmtime_s on Windows) forms that fill a caller-owned struct tm instead.
"""
import pathlib
import re
import sys

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
CALL = re.compile(r'(?<![\w.>])(?:std::)?(localtime|gmtime|ctime|asctime)\s*\(')


def strip_comments(text):
    text = re.sub(r'/\*.*?\*/', lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    return re.sub(r'//[^\n]*', '', text)


def main():
    bad = []
    files = [p for ext in ("*.cpp", "*.h", "*.hpp") for p in SRC.rglob(ext)]
    for path in sorted(files):
        for n, line in enumerate(strip_comments(path.read_text(errors="replace")).splitlines(), 1):
            for m in CALL.finditer(line):
                bad.append(f"{path.relative_to(SRC.parent)}:{n}: {m.group(1)}()")
    if bad:
        print("non-reentrant time calls:\n  " + "\n  ".join(bad))
        return 1
    print(f"ok: {len(files)} host files use only reentrant time conversions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
