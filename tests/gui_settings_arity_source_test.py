#!/usr/bin/env python3
"""Every option the parsers read a value for must be in util/GuiSettings.hpp's value-option lists.

Settings-file tokens are spliced into the command line, and the loader drops an option left at the end
of the file without its value so that it cannot take the next command-line argument. That only works
if the header knows every option that takes a value.
"""
import pathlib
import re
import sys

root = pathlib.Path(__file__).resolve().parents[1]
header = (root / "include/util/GuiSettings.hpp").read_text()
block = header[header.index("// BEGIN VALUE OPTIONS"):header.index("// END VALUE OPTIONS")]
one = block[block.index("oneValueOptions()"):block.index("twoValueOptions()")]
two = block[block.index("twoValueOptions()"):block.index("optionalValueOptions()")]
opt = block[block.index("optionalValueOptions()"):]
one_set = set(re.findall(r'"(-[^"]+)"', one))
two_set = set(re.findall(r'"(-[^"]+)"', two))
opt_set = set(re.findall(r'"(-[^"]+)"', opt))


def conditions(src):
    for m in re.finditer(r"\bif\s*\(", src):
        i, depth = m.end(), 1
        while depth and i < len(src):
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        yield m.end(), src[m.end():i - 1]


failures = []
for rel in ("src/io/CliParser.cpp", "src/modes/RunGaussianTrialFactor.cpp", "src/main.cpp", "src/core/App.cpp"):
    src = (root / rel).read_text()
    for _, cond in conditions(src):
        if "i + 1 < argc" in cond or "i + 1 < args.size()" in cond:
            for name in re.findall(r'(?:strcmp\(argv\[i\],|==)\s*"(-[^"]+)"', cond):
                if name not in one_set and name not in two_set:
                    failures.append(f"{rel}: {name} takes a value but is not in oneValueOptions()")

gtf = (root / "src/modes/RunGaussianTrialFactor.cpp").read_text()
for name in re.findall(r'arg == "(-[^"]+)"[^\n]*\n\s*if \(i \+ 2 >= args\.size\(\)\)', gtf):
    if name not in two_set:
        failures.append(f"{name} takes two values but is not in twoValueOptions()")
if not {"-gm-tf", "--gm-tf"} <= two_set:
    failures.append("-gm-tf/--gm-tf missing from twoValueOptions()")

cli = (root / "src/io/CliParser.cpp").read_text()
if 'std::strcmp(argv[i], "-host") == 0' in cli and "-host" not in one_set:
    failures.append("-host (takes the next token when it does not start with '-') missing")
for name in ("-pfa", "-pfa-auto"):
    if f'std::strcmp(argv[i], "{name}") == 0' in cli and name not in opt_set:
        failures.append(f"{name} (optional 3/7/9 value) missing from optionalValueOptions()")

if failures:
    print("\n".join(failures))
    sys.exit(1)
print(f"GUI settings value-option lists cover the parsers ({len(one_set)} + {len(two_set)} + {len(opt_set)})")
