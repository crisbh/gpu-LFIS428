#!/usr/bin/env python3
"""Expand @include directives in a Marp markdown file into fenced code blocks.

This is the slides' equivalent of LaTeX's \\lstinputlisting: instead of
copy-pasting source into the .md, point at the real file and the build inlines
it. The expanded markdown is written to stdout; the Makefile/CI feed that to marp.

Directive (each on its own line):

    @include[<lang>]{<path>}            # whole file
    @include[<lang>]{<path>:<a>-<b>}    # lines a..b inclusive (1-based)

  - <lang>  is the code-fence language (e.g. cuda, c, python, sh). `cuda`/`cu`
            are aliased to the `cpp` highlighter, which is the right grammar for
            CUDA C/C++ (highlight.js has no `cuda` language). The same alias is
            applied to inline ```cuda fenced blocks.
  - <path>  is resolved relative to the current working directory (repo root),
            so use e.g. static/code/intro/hola_mundo.cu — the same file that is
            served for download.

Example:

    @include[cpp]{static/code/intro/hola_mundo.cu}
    @include[cpp]{static/code/intro/suma_vectores_gpu.cu:12-15}
"""
import re
import sys
from pathlib import Path

DIRECTIVE = re.compile(r"^@include\[([^\]]*)\]\{([^}]+)\}\s*$")
RANGE = re.compile(r"(\d+)-(\d+)")
FENCE_OPEN = re.compile(r"^(\s*)(`{3,}|~{3,})\s*([A-Za-z0-9_+-]+)\s*$")

# highlight.js has no `cuda` grammar, so author-friendly labels are mapped to the
# `cpp` highlighter (the correct grammar for CUDA C/C++). Applies to both
# @include directives and inline fenced blocks.
LANG_ALIAS = {"cuda": "cpp", "cu": "cpp"}


def alias(lang):
    return LANG_ALIAS.get(lang.lower(), lang)


def fence_for(lines):
    """Pick a backtick fence longer than any backtick run in the snippet."""
    longest = 0
    for line in lines:
        for run in re.findall(r"`+", line):
            longest = max(longest, len(run))
    return "`" * max(3, longest + 1)


def expand(md_path):
    out = []
    for lineno, line in enumerate(Path(md_path).read_text().splitlines(), 1):
        m = DIRECTIVE.match(line)
        if not m:
            fence = FENCE_OPEN.match(line)
            if fence and fence.group(3).lower() in LANG_ALIAS:
                line = f"{fence.group(1)}{fence.group(2)}{alias(fence.group(3))}"
            out.append(line)
            continue
        lang, spec = m.group(1), m.group(2)
        path, _, rng = spec.partition(":")
        src = Path(path)
        if not src.is_file():
            sys.exit(f"{md_path}:{lineno}: included file not found: {path}")
        code = src.read_text().splitlines()
        if rng:
            rm = RANGE.fullmatch(rng)
            if not rm:
                sys.exit(f"{md_path}:{lineno}: bad line range '{rng}' (use a-b)")
            a, b = int(rm.group(1)), int(rm.group(2))
            code = code[a - 1:b]
        fence = fence_for(code)
        out.append(f"{fence}{alias(lang)}")
        out.extend(code)
        out.append(fence)
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: expand-includes.py <input.md>")
    sys.stdout.write(expand(sys.argv[1]))
