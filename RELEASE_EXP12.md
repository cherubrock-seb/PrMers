# exp12 publication commands

## Aevum standalone

```bash
cd /Users/fricadev/dev/aevum-engine
git switch main
git pull --ff-only origin main
git status --short

rsync -av --delete \
  --exclude='.git/' --exclude='.DS_Store' \
  --exclude='.aevum-kernel-cache/' \
  --exclude='build/' --exclude='build-engine/' --exclude='build-tests/' \
  --exclude='build-release/' --exclude='build-debug/' --exclude='build-cuda/' \
  --exclude='*.o' --exclude='*.d' --exclude='*.so' --exclude='*.dylib' \
  /Users/fricadev/Downloads/aevum-engine-v0.3.67-pow2-type4-lead-cache-exp12/ \
  /Users/fricadev/dev/aevum-engine/

git status --short
git diff --check
git diff --stat
make test-host
bash tests/native_pfa_opencl_syntax.sh
python3 tests/engine_lead_cache_test.py

git add -A
git diff --cached --check
git diff --cached --stat
git commit -m "Enable power-of-two FFT323161 lead caching"
git push origin main

AEVUM_TAG='v0.3.67-pow2-type4-lead-cache-exp12'
git tag -a "$AEVUM_TAG" -m "Aevum v0.3.67 power-of-two FFT323161 lead-cache release"
git push origin "$AEVUM_TAG"
```

## PrMers

```bash
cd /Users/fricadev/prmerscopy/PrMers
git switch main
git pull --ff-only origin main
git status --short

rsync -av --delete \
  --exclude='.git/' --exclude='.DS_Store' \
  --exclude='.aevum-kernel-cache/' --exclude='prmers' \
  --exclude='build/' --exclude='build-engine/' --exclude='build-tests/' \
  --exclude='build-release/' --exclude='package/' \
  --exclude='*.o' --exclude='*.d' --exclude='*.so' --exclude='*.dylib' \
  /Users/fricadev/Downloads/PrMers-v99.74-Aevum-Pow2-Type4-LeadCache-exp12/ \
  /Users/fricadev/prmerscopy/PrMers/

git status --short
git diff --check
git diff --stat
python3 tests/native_pfa_cli_source_test.py
python3 tests/aevum_pow2_type4_source_test.py
make test-aevum-host
make test-aevum-source
make test-aevum-auto
make test-aevum-default

git add -A
git diff --cached --check
git diff --cached --stat
git commit -m "Add power-of-two FFT323161 lead caching"
git push origin main

TAG='v4.20.68-alpha-v99.74-aevum-pow2-type4-lead-cache-exp12'
git fetch --tags --force
if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
  echo "ERREUR: le tag existe deja" >&2
  exit 1
fi
git tag -a "$TAG" -m "PrMers v99.74 power-of-two FFT323161 lead-cache release"
git push origin "$TAG"

gh run list --limit 10
gh run watch
```
