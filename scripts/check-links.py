#!/usr/bin/env python3
"""Validate the internal links/assets used by the slides and content pages.

Run from the repo root (see `make check`). Reports every broken reference and
exits non-zero if any are found, so problems are caught before deploy. Checks:

  slides/*.md
    - figures   ![..](images/…)        -> static/slides/<ref>
    - downloads ](../code/…)           -> static/code/…
    - includes  @include[..]{path[:a-b]} -> <path> from repo root

  content/**/_index.md
    - downloads ](../code/…)           -> static/code/…
    - deck links ](….html[#frag])      -> source slides/<name>.md exists
                                          (checks the source, not built HTML,
                                           so it is build-order independent)

Generated HTML (static/slides/*.html) is intentionally NOT required to exist.
"""
import glob
import os
import re
import sys

ROOT = os.getcwd()
errors = []

IMG = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)]+)\)")
INCLUDE = re.compile(r"@include\[[^\]]*\]\{([^}:]+)(?::\d+-\d+)?\}")


def strip_frag(target):
    return target.split("#", 1)[0]


def check_exists(relpath, src, kind):
    if not os.path.isfile(os.path.join(ROOT, relpath)):
        errors.append(f"{src}: missing {kind}: {relpath}")


# --- slides ---------------------------------------------------------------
for md in sorted(glob.glob("slides/*.md")):
    text = open(md).read()
    for ref in IMG.findall(text):
        ref = strip_frag(ref)
        if ref.startswith(("http://", "https://", "data:")):
            continue
        check_exists(os.path.join("static/slides", ref), md, "figure")
    for target in LINK.findall(text):
        t = strip_frag(target)
        if t.startswith("../code/"):
            check_exists(os.path.join("static", t[len("../"):]), md, "code link")
    for inc in INCLUDE.findall(text):
        check_exists(inc.strip(), md, "@include source")

# --- content pages --------------------------------------------------------
for md in sorted(glob.glob("content/**/_index.md", recursive=True)):
    text = open(md).read()
    for target in LINK.findall(text):
        t = strip_frag(target)
        if t.startswith(("http://", "https://", "mailto:")):
            continue
        if "/code/" in t:                      # download link -> static/code/…
            idx = t.index("code/")
            check_exists(os.path.join("static", t[idx:]), md, "code link")
        elif t.endswith(".html"):              # deck link -> slides/<name>.md
            name = os.path.basename(t)[:-len(".html")]
            check_exists(os.path.join("slides", name + ".md"), md, "deck source")

# --- report ---------------------------------------------------------------
if errors:
    print(f"check-links: {len(errors)} broken reference(s):", file=sys.stderr)
    for e in errors:
        print("  - " + e, file=sys.stderr)
    sys.exit(1)
print("check-links: all figure/code/include/deck references resolve.")
