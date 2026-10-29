#!/usr/bin/env python3
"""Prevent release identity drift without requiring .git in source distributions.

The checked-in Version.hpp is the canonical binary/source identity.  A release
workflow must use the matching v-prefixed tag, and every packaged host binary
must report that same identity via -v.  Works on Linux, Windows and macOS.
"""
import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile


def source_version(header_text):
    matches = re.findall(r'const\s+std::string\s+PRMERS_VERSION\s*=\s*"([^"]+)"\s*;', header_text)
    if len(matches) != 1:
        raise ValueError("Expected exactly one literal PRMERS_VERSION in Version.hpp")
    return matches[0]


def release_tag(workflow_text):
    matches = re.findall(r'^  TAG:\s*(v[^\s#]+)\s*$', workflow_text, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError("Expected exactly one release TAG in the unified workflow")
    return matches[0]


def verify_version(version, tag, binary_output=None):
    expected = "v" + version
    if tag != expected:
        raise ValueError("Release tag/source mismatch: %s != %s" % (tag, expected))
    if binary_output is not None and ("prmers Release " + expected) not in binary_output:
        raise ValueError("Binary -v does not report %s: %r" % (expected, binary_output))


def self_test():
    verify_version("4.20.97-alpha-v100.47-cuda-autobuild",
                   "v4.20.97-alpha-v100.47-cuda-autobuild",
                   "prmers Release v4.20.97-alpha-v100.47-cuda-autobuild\n")
    cases = [
        ("4.20.97-alpha-v100.44-old", "v4.20.97-alpha-v100.47-cuda-autobuild", None),
        ("4.20.97-alpha-v100.47-cuda-autobuild", "v4.20.97-alpha-v100.44-old", None),
        ("4.20.97-alpha-v100.47-cuda-autobuild",
         "v4.20.97-alpha-v100.47-cuda-autobuild", "prmers Release v4.20.97-alpha-v100.44-old"),
    ]
    for version, tag, output in cases:
        try:
            verify_version(version, tag, output)
        except ValueError:
            continue
        raise AssertionError("Negative release-identity regression did not fail")
    # These parsers require only source contents.  They do not invoke git.
    assert source_version('const std::string PRMERS_VERSION = "1.2.3";') == "1.2.3"
    assert release_tag("env:\n  TAG: v1.2.3\n") == "v1.2.3"


def test_gitless_package_version(root, expected):
    """Resolve a package version from a minimal exported source tree without .git."""
    make = shutil.which("make")
    if not make:
        raise ValueError("make is required for the source-archive version gate")
    with tempfile.TemporaryDirectory(prefix="prmers-no-git-version-") as scratch:
        work = pathlib.Path(scratch)
        (work / "src").mkdir()
        (work / "include/core").mkdir(parents=True)
        shutil.copyfile(root / "Makefile", work / "Makefile")
        shutil.copyfile(root / "include/core/Version.hpp", work / "include/core/Version.hpp")
        assert not (work / ".git").exists()
        # macOS ships GNU make 3.81, which supports -f but not --eval.
        # Read the real exported Makefile first, then add an isolated probe
        # target without changing the source tree's build or release rules.
        probe = work / "release-version-probe.mk"
        probe.write_text("print-release-version:\n\t@printf '%s\\n' '$(VERSION)'\n", encoding="utf-8")
        result = subprocess.run(
            [make, "-s", "-f", "Makefile", "-f", probe.name,
             "print-release-version"],
            cwd=work, text=True, capture_output=True, timeout=20, check=False)
        if result.returncode != 0 or result.stdout.strip() != expected:
            raise ValueError("Git-less source package VERSION failed: %r stderr=%r" %
                             (result.stdout.strip(), result.stderr))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", help="Built or staged executable to check with -v")
    parser.add_argument("--expected-tag", help="Also check against the release job's tag")
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    header = (root / "include/core/Version.hpp").read_text(encoding="utf-8")
    workflow = (root / ".github/workflows/unified-cuda-opencl-release.yml").read_text(encoding="utf-8")
    makefile = (root / "Makefile").read_text(encoding="utf-8")
    version = source_version(header)
    tag = release_tag(workflow)
    # Source archive extraction must not fall back to a historic version or
    # git describe: Makefile resolves VERSION from the checked-in header.
    make_version = re.findall(r'^VERSION\s*:=\s*(.*)$', makefile, re.MULTILINE)
    if len(make_version) != 1 or "Version.hpp" not in make_version[0] or "git describe" in make_version[0]:
        raise ValueError("Makefile VERSION must derive from Version.hpp, without git")
    self_test()
    verify_version(version, tag)
    test_gitless_package_version(root, tag)
    if args.expected_tag:
        verify_version(version, args.expected_tag)
    if args.binary:
        command = [str(pathlib.Path(args.binary).resolve()), "-v"]
        result = subprocess.run(command, text=True, capture_output=True, timeout=20, check=False)
        if result.returncode != 0:
            raise ValueError("%s -v exited %d: %s" % (command[0], result.returncode, result.stderr))
        verify_version(version, tag, result.stdout + result.stderr)
    print("RELEASE_VERSION_GATE=PASS version=%s tag=%s binary=%s" %
          (version, tag, args.binary or "NOT_RUN"))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, AssertionError, OSError, subprocess.TimeoutExpired) as exc:
        print("RELEASE_VERSION_GATE=FAIL: %s" % exc, file=sys.stderr)
        sys.exit(1)
