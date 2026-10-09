#!/usr/bin/env bash
# Runs prmers with -worktodo pointing into another directory and checks that the archive of finished
# entries (worktodo_save.txt) is written next to that worktodo file, not in the current directory.
#   other-dir: two PRP entries from a worktodo file in a different directory (absolute path)
#   gmtf:      a GMTF entry (archived by the trial-factor path)
#   relative:  the same with a relative path with a directory component
#   llsafe:    a DoubleCheck (LL-SAFE) entry
#   wagstaff:  a Wagstaff PRP entry on the legacy backend (-wagstaff -marin)
#   readonly:  the worktodo directory cannot be written: a clear error, the entry stays queued
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'chmod -R u+rwx "$WORK" 2>/dev/null || true; rm -rf "$WORK"' EXIT
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

setup() {
  local name="$1"; shift
  mkdir -p "$WORK/$name/cwd" "$WORK/$name/queue"
  printf '%s\n' "$@" > "$WORK/$name/queue/worktodo.txt"
  ln -s "$ROOT/kernels" "$WORK/$name/cwd/kernels"
}

run() {
  local name="$1" wt="$2"; shift 2
  ( cd "$WORK/$name/cwd" &&
    timeout --signal=INT --kill-after=10s 80 "$ROOT/prmers" -d "$DEVICE" -worktodo "$wt" --noask "$@" >run.log 2>&1 || true )
}

setup other-dir 'PRP=N/A,1,2,9941,-1,70,0' 'PRP=N/A,1,2,9973,-1,70,0'
run other-dir "$WORK/other-dir/queue/worktodo.txt"
[[ ! -s "$WORK/other-dir/queue/worktodo.txt" ]] || { echo "other-dir: worktodo not drained" >&2; cat "$WORK/other-dir/cwd/run.log" >&2; exit 1; }
[[ "$(head -1 "$WORK/other-dir/queue/worktodo_save.txt")" == PRP=* ]] || { echo "other-dir: PRP not archived beside the worktodo" >&2; exit 1; }
grep -qx 'PRP=N/A,1,2,9973,-1,70,0' "$WORK/other-dir/queue/worktodo_save.txt" || { echo "other-dir: second PRP not archived beside the worktodo" >&2; exit 1; }
[[ ! -e "$WORK/other-dir/cwd/worktodo_save.txt" ]] || { echo "other-dir: archive written in the cwd" >&2; exit 1; }
grep -q "saved to $WORK/other-dir/queue/worktodo_save.txt" "$WORK/other-dir/cwd/run.log" || { echo "other-dir: message does not name the archive path" >&2; exit 1; }

setup gmtf 'GMTF=1009,20,24'
run gmtf "$WORK/gmtf/queue/worktodo.txt"
grep -qx 'GMTF=1009,20,24' "$WORK/gmtf/queue/worktodo_save.txt" || { echo "gmtf: GMTF not archived beside the worktodo" >&2; cat "$WORK/gmtf/cwd/run.log" >&2; exit 1; }
[[ ! -e "$WORK/gmtf/cwd/worktodo_save.txt" ]] || { echo "gmtf: archive written in the cwd" >&2; exit 1; }
grep -q "saved to $WORK/gmtf/queue/worktodo_save.txt" "$WORK/gmtf/cwd/run.log" || { echo "gmtf: message does not name the archive path" >&2; exit 1; }

setup relative 'PRP=N/A,1,2,9941,-1,70,0'
run relative "../queue/worktodo.txt"
[[ ! -s "$WORK/relative/queue/worktodo.txt" ]] || { echo "relative: worktodo not drained" >&2; exit 1; }
grep -q '^PRP=' "$WORK/relative/queue/worktodo_save.txt" || { echo "relative: PRP not archived beside the worktodo" >&2; exit 1; }
[[ ! -e "$WORK/relative/cwd/worktodo_save.txt" ]] || { echo "relative: archive written in the cwd" >&2; exit 1; }

for c in 'llsafe|DoubleCheck=4423,70,1|' 'wagstaff|PRP=1,2,127,-1|-wagstaff -marin'; do
  IFS='|' read -r name entry flags <<<"$c"
  setup "$name" "$entry"
  # shellcheck disable=SC2086
  run "$name" "$WORK/$name/queue/worktodo.txt" $flags
  grep -qx "$entry" "$WORK/$name/queue/worktodo_save.txt" || { echo "$name: entry not archived beside the worktodo" >&2; cat "$WORK/$name/cwd/run.log" >&2; exit 1; }
  [[ ! -e "$WORK/$name/cwd/worktodo_save.txt" ]] || { echo "$name: archive written in the cwd" >&2; exit 1; }
  grep -q "saved to $WORK/$name/queue/worktodo_save.txt" "$WORK/$name/cwd/run.log" || { echo "$name: message does not name the archive path" >&2; exit 1; }
done

if [[ "$(id -u)" != 0 ]]; then
  setup readonly 'PRP=N/A,1,2,9941,-1,70,0'
  # The directory is made read-only after the run has started its work: simplest is to start read-only
  # and check the failure path. PrMers computes the result first, then cannot update the worktodo file.
  chmod 0555 "$WORK/readonly/queue"
  run readonly "$WORK/readonly/queue/worktodo.txt"
  chmod 0755 "$WORK/readonly/queue"
  grep -q 'PRP=N/A,1,2,9941,-1,70,0' "$WORK/readonly/queue/worktodo.txt" || { echo "readonly: entry lost" >&2; exit 1; }
  [[ ! -e "$WORK/readonly/cwd/worktodo_save.txt" ]] || { echo "readonly: silent fallback to the cwd" >&2; exit 1; }
  grep -q 'Cannot write .*worktodo.txt.tmp' "$WORK/readonly/cwd/run.log" || { echo "readonly: no clear error" >&2; cat "$WORK/readonly/cwd/run.log" >&2; exit 1; }
fi
echo "worktodo archive dir run test passed"
