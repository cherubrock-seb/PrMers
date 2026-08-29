#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-v4.20.90-alpha-v100.02-gm-ecm-special4096}"
REPO="$HOME/PrMers"
BIN="$REPO/prmers"
ROOT="$HOME/gm-special4096-B8k-B400k-K12"

echo "======================================================"
echo "SPECIAL4096 CHERUBROCK2 DEPLOY"
echo "tag=$TAG"
echo "======================================================"

cd "$REPO"

if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: $REPO has local changes; refusing to overwrite."
    git status --short
    exit 2
fi

git fetch origin --tags
git checkout "$TAG"

echo
echo "BUILD..."
make -j"$(nproc)"

echo
echo "VERSION..."
"$BIN" -v || true

echo
echo "SPECIAL4096 REGRESSION..."
python3 tests/gaussian_mersenne_ecm_special4096_regression_test.py

echo
echo "Build/test OK. Only now stopping previous quickwin workers."
screen -S qw-vii  -X quit 2>/dev/null || true
screen -S qw-3080 -X quit 2>/dev/null || true

mkdir -p "$ROOT"

cat > "$ROOT/candidates.txt" <<'EOF'
15000017
15150013
15300001
15600001
15750019
15900023
16050007
16200029
16350001
16500007
16650047
16800023
16950013
17100047
17250011
17400049
17550007
17700031
17850013
18000041
18000137
18150019
18300083
18450011
18600073
18750001
18900053
19050011
19200011
19350077
19500043
19650019
19800083
19950013
20100053
20550007
20700037
20850019
EOF

# Split the 38 targets between the two GPUs.
head -n 19 "$ROOT/candidates.txt" > "$ROOT/vii.txt"
tail -n 19 "$ROOT/candidates.txt" | tac > "$ROOT/rtx3080.txt"

: > "$ROOT/FOUND.txt"

cat > "$ROOT/worker.sh" <<'EOF'
#!/usr/bin/env bash
set -uo pipefail

DEVICE="$1"
NAME="$2"
LIST="$3"

REPO="$HOME/PrMers"
BIN="$REPO/prmers"
ROOT="$HOME/gm-special4096-B8k-B400k-K12"

B1=8000
B2=400000
K=12

MASTER="$ROOT/${NAME}.master.log"
touch "$MASTER" "$ROOT/FOUND.txt"

while read -r P; do
    [ -n "$P" ] || continue

    OUT="$ROOT/p${P}-${NAME}"
    LOG="$OUT/run.log"
    mkdir -p "$OUT"

    echo "======================================================" >> "$MASTER"
    echo "Special4096 gpu=$NAME device=$DEVICE p=$P B1=$B1 B2=$B2 K=$K" >> "$MASTER"
    date -u +"UTC=%Y-%m-%dT%H:%M:%SZ" >> "$MASTER"

    set +e
    "$BIN" "$P" \
        -gm-ecm-special4096 \
        -gm-family GM \
        -gm-sieve 0 \
        -bsgs \
        -b1 "$B1" \
        -b2 "$B2" \
        -K "$K" \
        -aevum \
        -d "$DEVICE" \
        -r \
        -f "$OUT" \
        > "$LOG" 2>&1
    RC=$?
    set -e 2>/dev/null || true
    set +e

    CLEAN="$OUT/run.clean.log"
    tr '\r' '\n' < "$LOG" > "$CLEAN"

    FACTOR_LINE="$(
        grep -E \
          '>>> Gaussian pair ECM (setup|Stage 1|Stage 2) factor:' \
          "$CLEAN" |
        tail -1
    )"

    if [ -n "$FACTOR_LINE" ]; then
        {
            echo
            echo "************************************************"
            echo "*** SPECIAL4096 FACTOR FOUND ***"
            echo "p=$P"
            echo "gpu=$NAME"
            echo "device=$DEVICE"
            echo "B1=$B1"
            echo "B2=$B2"
            echo "K=$K"
            echo "$FACTOR_LINE"
            echo "************************************************"
        } | tee -a "$ROOT/FOUND.txt" "$MASTER"
    else
        LAST="$(
            grep -E \
              'Special4096|Stage 1 no factor|Stage 2 BSGS|no factor' \
              "$CLEAN" |
            tail -1
        )"
        echo "DONE p=$P rc=$RC ${LAST:-}" >> "$MASTER"
    fi
done < "$LIST"

echo "WORKER $NAME FINISHED" >> "$MASTER"
EOF

chmod +x "$ROOT/worker.sh"

screen -dmS s4096-vii \
    bash -lc "$ROOT/worker.sh 0 vii $ROOT/vii.txt"

screen -dmS s4096-3080 \
    bash -lc "$ROOT/worker.sh 1 rtx3080 $ROOT/rtx3080.txt"

sleep 2

echo
echo "======================================================"
echo "SPECIAL4096 LAUNCHED"
echo "======================================================"
echo "ROOT=$ROOT"
echo "B1=8000 B2=400000 K=12"
echo "VII: 19 exponents"
echo "RTX3080: 19 exponents (reverse order)"
echo
screen -ls | grep -E 's4096-vii|s4096-3080' || true

echo
echo "Monitor:"
cat <<'MONITOR'
ROOT="$HOME/gm-special4096-B8k-B400k-K12"
watch -n 10 '
clear
ROOT="$HOME/gm-special4096-B8k-B400k-K12"
echo "=============== SPECIAL4096 B8k/B400k K12 ==============="
for NAME in vii rtx3080; do
  echo
  echo "---------------- $NAME ----------------"
  M="$ROOT/${NAME}.master.log"
  [ -f "$M" ] || { echo idle; continue; }
  P=$(grep "Special4096 gpu=" "$M" | tail -1 | sed -n "s/.*p=\([0-9]*\).*/\1/p")
  echo "p=${P:-?}"
  if [ -n "$P" ]; then
    LOG="$ROOT/p${P}-${NAME}/run.log"
    if [ -f "$LOG" ]; then
      tr "\r" "\n" < "$LOG" |
      grep -E "Special4096|Stage 1.*%|Stage 1 no factor|Stage 2 BSGS.*%|Gaussian pair ECM.*factor:" |
      tail -10
    fi
  fi
done
echo
echo "================ FACTORS ================"
if [ -s "$ROOT/FOUND.txt" ]; then tail -50 "$ROOT/FOUND.txt"; else echo none; fi
'
MONITOR
