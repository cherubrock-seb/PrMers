#!/usr/bin/env python3
"""Every inline-asm template in src/cl must have balanced braces.

The template of an __asm(...) goes to ptxas (or the AMD assembler) verbatim, so a stray brace such as
"ld.global.nc.f32  %0}, [%1];" is a PTX parse error the first time an FP32 configuration emits that overload.  It stays latent until
then because unused device functions are stripped before ptxas sees them, so a plain build does not catch it.
"""
import pathlib
import re
import sys

CL_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "cl"
# Adjacent string literals, possibly separated by whitespace and comments, form one template.
STRING = re.compile(r'(?:\s|//[^\n]*\n|/\*.*?\*/)*"((?:[^"\\]|\\.)*)"', re.S)


def asm_templates(text):
    for m in re.finditer(r'__asm\s*\(', text):
        pos = m.end()
        parts = []
        while True:
            s = STRING.match(text, pos)
            if not s:
                break
            parts.append(s.group(1))
            pos = s.end()
        if parts:
            yield text[:m.start()].count("\n") + 1, "".join(parts)


def main():
    bad = []
    count = 0
    for path in sorted(CL_DIR.glob("*.cl")):
        for line, template in asm_templates(path.read_text()):
            count += 1
            depth = 0
            ok = True
            for ch in template:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth < 0:
                        ok = False
                        break
            if depth != 0 or not ok:
                bad.append(f"{path.name}:{line}: unbalanced braces in asm template {template!r}")
    if count < 50:
        print(f"FAIL: only {count} asm templates found, the scanner is broken")
        return 1
    for b in bad:
        print("FAIL:", b)
    if bad:
        return 1
    print(f"ok: {count} asm templates have balanced braces")
    return 0


if __name__ == "__main__":
    sys.exit(main())
