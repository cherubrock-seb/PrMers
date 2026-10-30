import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

header = (ROOT / "include/core/ProofCheckpoint.hpp").read_text()

m = re.search(r"inline bool syncFileToDisk\(.*?\n\}\n", header, re.S)
assert m, "syncFileToDisk not found"
body = m.group(0)

posix, sep, windows = body.partition("#else")
assert sep, "syncFileToDisk has no Windows branch"

# The POSIX branch syncs the file; the Windows branch must flush it too and not be a stub.
assert "::fsync(fd)" in posix
assert "_commit(" in windows or "FlushFileBuffers(" in windows, "Windows branch does not flush the file"
assert "(void)path;" not in windows, "Windows branch still ignores the path"
# Like the POSIX branch, a file that cannot be opened is not an error; one that fails to flush is.
assert re.search(r"if \(fd < 0\) return true;", windows)
assert re.search(r"return ok;", windows)
# The file is opened for writing (a flush needs write access) without truncating it.
assert "_O_WRONLY" in windows or "GENERIC_WRITE" in windows
assert "_O_TRUNC" not in windows and "_O_CREAT" not in windows
# Windows has no directory sync; the header says so.
assert "no directory sync" in header

# The Windows headers must be included for the Windows branch.
win_includes = header.split("#else", 1)[1].split("#endif", 1)[0]
assert "<io.h>" in win_includes and "<fcntl.h>" in win_includes

# The BackupManager directory sync is a no-op on Windows, and says why.
bm = (ROOT / "src/core/BackupManager.cpp").read_text()
dirsync = bm[bm.index("void syncDirectory"):]
dirsync = dirsync[: dirsync.index("\n}\n")]
assert "cannot open a directory" in dirsync

# Compile the header for Windows when a cross compiler is installed.
cross = shutil.which("x86_64-w64-mingw32-g++")
if cross:
    with tempfile.TemporaryDirectory() as tmp:
        tu = Path(tmp) / "sync_tu.cpp"
        tu.write_text(
            '#include "core/ProofCheckpoint.hpp"\n'
            "int main() { return core::syncFileToDisk(\"x\") ? 0 : 1; }\n"
        )
        r = subprocess.run(
            [cross, "-std=c++20", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "include"),
             "-fsyntax-only", str(tu)],
            capture_output=True, text=True)
        if r.returncode != 0:
            sys.stderr.write(r.stdout + r.stderr)
            sys.exit(1)
    print("mingw syntax check: ok")
else:
    print("mingw cross compiler not installed: Windows compile check skipped")

print("Proof checkpoint sync source regression: PASS")
