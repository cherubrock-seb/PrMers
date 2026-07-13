# Publishing Aevum on GitHub

## Recommended repository model

Create `cherubrock-seb/aevum-engine` as a GitHub fork of `gwoltman/gpuowl`, then rename the fork to `aevum-engine` if desired. Keeping the fork relationship is preferable to uploading a new repository because it preserves the visible history back to GPUOwl.

Upstream chain:

```text
preda/gpuowl
  -> gwoltman/gpuowl
      -> cherubrock-seb/aevum-engine
```

Aevum should remain GPLv3.

## Suggested workflow

```bash
git clone https://github.com/cherubrock-seb/aevum-engine.git
cd aevum-engine
git remote add upstream https://github.com/gwoltman/gpuowl.git
git fetch upstream
git checkout -b aevum-engine upstream/master
```

Copy or apply the Aevum changes, then commit them in logical groups:

```bash
git add src/EngineApi.cpp src/EngineApi.h src/Gpu.cpp src/Gpu.h
git commit -m "Add reusable Aevum register engine API"

git add examples tests Makefile
git commit -m "Add Aevum API examples and tests"

git add README.md README_GPUOWL.md UPSTREAM.md MODIFICATIONS.md NOTICE LICENSE
git commit -m "Document Aevum upstream and GPLv3 attribution"
```

Tag the release:

```bash
git tag -a v/aevum/0.3.3 -m "Aevum register engine 0.3.3"
git push origin aevum-engine --tags
```

## Files to retain

Do not remove:

- `LICENSE`
- original copyright headers
- `README_GPUOWL.md`
- `UPSTREAM.md`
- `MODIFICATIONS.md`
- `NOTICE`

The main `README.md` should state prominently that Aevum is a modified GPUOwl/PRPLL derivative and not an official upstream release.

## PrMers integration

Keep PrMers and Aevum as separate repositories:

```bash
cd PrMers
git submodule add https://github.com/cherubrock-seb/aevum-engine third_party/aevum
git commit -m "Add optional Aevum engine submodule"
```

PrMers-authored files can remain under MIT. The Aevum submodule remains GPLv3. A binary or bundle containing Aevum must be distributed with the corresponding source and in compliance with GPLv3.

Dynamic loading does not by itself guarantee that a combined distribution is outside GPL copyleft. Publishing complete source for both repositories and preserving all notices is the conservative approach. This guide is not legal advice.
