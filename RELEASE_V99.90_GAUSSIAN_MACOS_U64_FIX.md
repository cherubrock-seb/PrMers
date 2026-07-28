# PrMers v99.90 — Gaussian macOS uint64/GMP portability fix

PrMers version:

```text
4.20.79-alpha-v99.90-gaussian-macos-u64-fix
bash <<'BASH'
set -euo pipefail

REPO="$HOME/prmerscopy/PrMers"
BRANCH="release-prmers-v99.90-final"

OLD_BUILD="4.20.78-alpha-v99.89-gaussian-worktodo-progress-json"
NEW_BUILD="4.20.79-alpha-v99.90-gaussian-macos-u64-fix"

cd "$REPO"

# Sauvegarder proprement le collage incomplet précédent.
if test -n "$(git status --porcelain)"; then
  git stash push -u \
    -m "backup incomplete v99.90 correction $(date +%Y%m%d-%H%M%S)"
fi

git switch main
git pull --ff-only origin main
git fetch --prune origin

test -z "$(git status --porcelain)"

if git ls-remote --exit-code --heads origin \
  "refs/heads/$BRANCH" >/dev/null 2>&1
then
  echo "ERREUR: branche distante déjà présente : $BRANCH"
  exit 1
fi

git branch -D "$BRANCH" 2>/dev/null || true
git switch -c "$BRANCH"

python3 - "$OLD_BUILD" "$NEW_BUILD" <<'PY'
from pathlib import Path
import sys

old_build = sys.argv[1]
new_build = sys.argv[2]


def replace_once(filename: str, old: str, new: str) -> None:
    path = Path(filename)
    text = path.read_text(encoding="utf-8")
    count = text.count(old)

    if count != 1:
        raise SystemExit(
            f"ERREUR: {filename}: attendu exactement 1 occurrence "
            f"de {old!r}, trouvé {count}"
        )

    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# ---------------------------------------------------------------------------
# Correction portable std::uint64_t -> GMP.
# ---------------------------------------------------------------------------

replace_once(
    "src/modes/RunGaussianMersenneFactor.cpp",
    "    mpz_class sigma = sigma64;",
    """    mpz_class sigma;
    mpz_import(
        sigma.get_mpz_t(),
        1,
        1,
        sizeof(sigma64),
        0,
        0,
        &sigma64);""",
)

# ---------------------------------------------------------------------------
# Version complète.
# ---------------------------------------------------------------------------

for filename in (
    "include/core/Version.hpp",
    "tests/aevum_pow2_type4_source_test.py",
    "tests/gaussian_mersenne_factor_isolation_test.py",
):
    replace_once(filename, old_build, new_build)

# ---------------------------------------------------------------------------
# Version courte des JSON Gaussian.
# ---------------------------------------------------------------------------

for filename in (
    "src/modes/RunGaussianMersenne.cpp",
    "src/modes/RunGaussianMersenneFactor.cpp",
):
    replace_once(
        filename,
        'constexpr const char* GM_RELEASE = "v99.89";',
        'constexpr const char* GM_RELEASE = "v99.90";',
    )

# ---------------------------------------------------------------------------
# Audit empêchant le retour de la conversion ambiguë.
# ---------------------------------------------------------------------------

test_path = Path("tests/gaussian_mersenne_factor_isolation_test.py")
test = test_path.read_text(encoding="utf-8")

marker = f'assert "{new_build}" in version\n'
checks = """
# GMP C++ has no portable unsigned-long-long constructor on every ABI.
assert "mpz_class sigma = sigma64;" not in source
assert "mpz_import(" in source
assert "sizeof(sigma64)" in source
"""

if marker not in test:
    raise SystemExit("ERREUR: marqueur de version absent du test Gaussian")

if 'assert "mpz_class sigma = sigma64;" not in source' not in test:
    test = test.replace(marker, marker + checks, 1)

test_path.write_text(test, encoding="utf-8")

# ---------------------------------------------------------------------------
# Exécuter test-gm dans les CI Linux et macOS.
# ---------------------------------------------------------------------------

workflow_changes = (
    (
        Path(".github/workflows/build_linux.yml"),
        "          make test-aevum-host\n",
        "          make test-gm\n"
        "          make test-aevum-host\n",
    ),
    (
        Path(".github/workflows/build_mac_os.yml"),
        '          make test-aevum-host MACOSX_DEPLOYMENT_TARGET="$MACOSX_DEPLOYMENT_TARGET"\n',
        "          make test-gm\n"
        '          make test-aevum-host MACOSX_DEPLOYMENT_TARGET="$MACOSX_DEPLOYMENT_TARGET"\n',
    ),
)

for path, old, new in workflow_changes:
    data = path.read_text(encoding="utf-8")

    if "make test-gm" in data:
        continue

    if data.count(old) != 1:
        raise SystemExit(
            f"ERREUR: point d’insertion introuvable dans {path}"
        )

    path.write_text(data.replace(old, new, 1), encoding="utf-8")

# ---------------------------------------------------------------------------
# Documentation active.
# ---------------------------------------------------------------------------

readme = Path("README_GAUSSIAN_FACTORING.md")
data = readme.read_text(encoding="utf-8")

data = data.replace(
    "Gaussian-Mersenne P-1 and ECM — PrMers v99.89",
    "Gaussian-Mersenne P-1 and ECM — PrMers v99.90",
    1,
)

readme.write_text(data, encoding="utf-8")
PY

cat > RELEASE_V99.90_GAUSSIAN_MACOS_U64_FIX.md <<'EOF'
# PrMers v99.90 — Gaussian macOS uint64/GMP portability fix

PrMers version:

```text
4.20.79-alpha-v99.90-gaussian-macos-u64-fix
bash <<'BASH'
set -euo pipefail

REPO="$HOME/prmerscopy/PrMers"
BRANCH="release-prmers-v99.90-final"

OLD_BUILD="4.20.78-alpha-v99.89-gaussian-worktodo-progress-json"
NEW_BUILD="4.20.79-alpha-v99.90-gaussian-macos-u64-fix"

cd "$REPO"

# Sauvegarder un éventuel collage incomplet précédent.
if test -n "$(git status --porcelain)"; then
  git stash push -u \
    -m "backup incomplete v99.90 correction $(date +%Y%m%d-%H%M%S)"
fi

git switch main
git pull --ff-only origin main
git fetch --prune origin

test -z "$(git status --porcelain)"

if git ls-remote --exit-code --heads origin \
  "refs/heads/$BRANCH" >/dev/null 2>&1
then
  echo "ERREUR: branche distante déjà présente : $BRANCH"
  exit 1
fi

git branch -D "$BRANCH" 2>/dev/null || true
git switch -c "$BRANCH"

python3 - "$OLD_BUILD" "$NEW_BUILD" <<'PY'
from pathlib import Path
import sys

old_build = sys.argv[1]
new_build = sys.argv[2]


def replace_once(filename: str, old: str, new: str) -> None:
    path = Path(filename)
    text = path.read_text(encoding="utf-8")
    count = text.count(old)

    if count != 1:
        raise SystemExit(
            f"ERREUR: {filename}: attendu exactement 1 occurrence "
            f"de {old!r}, trouvé {count}"
        )

    path.write_text(
        text.replace(old, new, 1),
        encoding="utf-8",
    )


# Correction portable std::uint64_t vers GMP.
replace_once(
    "src/modes/RunGaussianMersenneFactor.cpp",
    "    mpz_class sigma = sigma64;",
    """    mpz_class sigma;
    mpz_import(
        sigma.get_mpz_t(),
        1,
        1,
        sizeof(sigma64),
        0,
        0,
        &sigma64);""",
)

# Mise à jour de la version complète.
for filename in (
    "include/core/Version.hpp",
    "tests/aevum_pow2_type4_source_test.py",
    "tests/gaussian_mersenne_factor_isolation_test.py",
):
    replace_once(filename, old_build, new_build)

# Mise à jour de la version courte dans les JSON Gaussian.
for filename in (
    "src/modes/RunGaussianMersenne.cpp",
    "src/modes/RunGaussianMersenneFactor.cpp",
):
    replace_once(
        filename,
        'constexpr const char* GM_RELEASE = "v99.89";',
        'constexpr const char* GM_RELEASE = "v99.90";',
    )

# Ajouter un test empêchant le retour de la conversion ambiguë.
test_path = Path("tests/gaussian_mersenne_factor_isolation_test.py")
test = test_path.read_text(encoding="utf-8")

marker = f'assert "{new_build}" in version\n'
checks = """
# GMP C++ has no portable unsigned-long-long constructor on every ABI.
assert "mpz_class sigma = sigma64;" not in source
assert "mpz_import(" in source
assert "sizeof(sigma64)" in source
"""

if marker not in test:
    raise SystemExit("ERREUR: marqueur de version absent du test Gaussian")

if 'assert "mpz_class sigma = sigma64;" not in source' not in test:
    test = test.replace(marker, marker + checks, 1)

test_path.write_text(test, encoding="utf-8")

# Exécuter les tests Gaussian dans les CI Linux et macOS.
workflow_changes = (
    (
        Path(".github/workflows/build_linux.yml"),
        "          make test-aevum-host\n",
        "          make test-gm\n"
        "          make test-aevum-host\n",
    ),
    (
        Path(".github/workflows/build_mac_os.yml"),
        '          make test-aevum-host MACOSX_DEPLOYMENT_TARGET="$MACOSX_DEPLOYMENT_TARGET"\n',
        "          make test-gm\n"
        '          make test-aevum-host MACOSX_DEPLOYMENT_TARGET="$MACOSX_DEPLOYMENT_TARGET"\n',
    ),
)

for path, old, new in workflow_changes:
    data = path.read_text(encoding="utf-8")

    if "make test-gm" in data:
        continue

    if data.count(old) != 1:
        raise SystemExit(
            f"ERREUR: point d’insertion introuvable dans {path}"
        )

    path.write_text(
        data.replace(old, new, 1),
        encoding="utf-8",
    )

# Mise à jour du titre de la documentation active.
readme = Path("README_GAUSSIAN_FACTORING.md")
data = readme.read_text(encoding="utf-8")

data = data.replace(
    "Gaussian-Mersenne P-1 and ECM — PrMers v99.89",
    "Gaussian-Mersenne P-1 and ECM — PrMers v99.90",
    1,
)

readme.write_text(data, encoding="utf-8")
PY

cat > RELEASE_V99.90_GAUSSIAN_MACOS_U64_FIX.md <<'EOF'
# PrMers v99.90 — Gaussian macOS uint64/GMP portability fix

PrMers version:

    4.20.79-alpha-v99.90-gaussian-macos-u64-fix

Aevum remains unchanged:

    v0.3.78-workload-plan-policy-audit-fix

## Correction

The Gaussian-Mersenne ECM Suyama setup previously initialized an mpz_class
directly from std::uint64_t.

On ABIs where std::uint64_t maps to unsigned long long, GMP C++ does not
provide an unambiguous constructor for that type. This caused the macOS Intel
release build to fail.

v99.90 imports the complete 64-bit value with mpz_import, without narrowing
to unsigned long.

## Isolation

No Aevum source, Aevum kernel, Marin kernel, ordinary Mersenne PRP/LL, P-1 or
ECM arithmetic has been modified.

## CI

The Gaussian mathematics, dispatch, JSON and worktodo tests are now explicitly
executed by the Linux and macOS release workflows.
