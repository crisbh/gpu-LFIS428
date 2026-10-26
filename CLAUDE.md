# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Course website for *Programación en GPUs* (LFIS428, IFA-UV), published at https://crisbh.github.io/gpu-LFIS428/. It is a Hugo site (theme `hugo-book`, a git submodule) plus Marp slide decks. All course content (slides, code comments, pages) is written in Spanish; keep it that way.

## Commands

- `make slides`: build every `slides/*.md` into `static/slides/*.html` (needs `marp`, i.e. `npm install -g @marp-team/marp-cli`).
- `make static/slides/threads.html`: rebuild a single deck.
- `make site`: build slides, check out the `hugo-book` theme submodule if missing, then `hugo server --minify --theme hugo-book`. The local URL is `http://localhost:1313/gpu-LFIS428/` because `baseURL` in `hugo.toml` has a path; `localhost:1313/` alone 404s.
- If every page is "page not found", check `git submodule status`: a leading `-` means the theme is not checked out (fix with `git submodule update --init --recursive themes/hugo-book`; `themes/relearn` is an unused submodule). Hugo then only warns `found no layout file` and builds nothing.
- `make check`: validate internal links (figures, code downloads, `@include` paths, deck links). CI runs this first and fails the deploy on any broken reference, so run it after editing slides or content pages.
- `make clean`: remove generated HTML, `.slides-build/` and `public/`.

There is no build or test harness for the CUDA examples; each file is standalone and compiled by hand, e.g. `nvcc -arch=sm_75 file.cu -o file.x` (the target GPU is a T4, Turing CC 7.5). Files that need `static/code/memoria/common.h` say so in their header.

## Architecture

Three source trees feed the site:

- `slides/*.md`: Marp decks (front matter `marp: true`, `theme: curso`, KaTeX math). Styling lives in `slides/themes/curso.css`. Images are referenced as `images/...` and resolve to `static/slides/images/...`.
- `static/code/<unit>/`: the course's example programs, served for download at `../code/<unit>/<file>` and also inlined into slides.
- `content/*/_index.md`: Hugo pages. `content/slides/_index.md` and `content/codigos/_index.md` are hand-maintained lists; a new deck or code file is not visible on the site until it is linked there. Units not yet taught are kept commented out (`<!-- ... -->`) and uncommented as the course progresses.

Slide pipeline: `scripts/expand-includes.py` replaces `@include[lang]{path}` or `@include[lang]{path:a-b}` lines (paths relative to repo root, 1-based inclusive line ranges) with fenced code from `static/code/`, writing to `.slides-build/`, which `marp` then renders. `cuda`/`cu` fences are aliased to `cpp` because highlight.js has no CUDA grammar. The Makefile rule and the CI loop in `.github/workflows/deploy.yml` duplicate this pipeline; change both together.

Line-range includes are fragile: editing a `.cu` file under `static/code/` shifts the lines that slides pull in. After changing any code file, `grep -n '@include.*<filename>' slides/*.md` and fix the ranges. `make check` only verifies that the file exists, not that the range still covers the intended code.

Speaker notes: HTML comments beginning `<!-- NOTA —` inside slides are Marp presenter notes for the instructor, not dead content.

Generated outputs (`static/slides/*.html`, `public/`, `.slides-build/`) are gitignored; do not commit them. `layouts/partials/docs/menu-hugo.html` overrides the theme only to stop internal menu links opening in a new tab.

## Deploy

Pushing to `main` triggers `.github/workflows/deploy.yml`: `make check`, Marp conversion, `hugo --minify` (Hugo extended 0.128.0), then GitHub Pages deploy.
