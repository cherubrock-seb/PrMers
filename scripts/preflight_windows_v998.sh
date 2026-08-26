#!/usr/bin/env bash
set -euo pipefail

REPO_NAME="${PRMERS_GITHUB_REPO:-cherubrock-seb/PrMers}"
WORKFLOW="build_windows.yml"

command -v gh >/dev/null || { echo "gh CLI is required" >&2; exit 2; }

HEAD_SHA="$(git rev-parse HEAD)"
BRANCH="$(git branch --show-current)"
if [ "$BRANCH" != "main" ]; then
  echo "Expected main branch, got: $BRANCH" >&2
  exit 2
fi

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Tracked working tree is not clean; commit first." >&2
  git status --short
  exit 2
fi

REMOTE_SHA="$(git ls-remote origin refs/heads/main | awk '{print $1}')"
if [ "$HEAD_SHA" != "$REMOTE_SHA" ]; then
  echo "Local HEAD is not origin/main." >&2
  echo "local : $HEAD_SHA" >&2
  echo "remote: $REMOTE_SHA" >&2
  exit 2
fi

echo "Dispatching Windows/MSVC preflight for $HEAD_SHA"
gh workflow run "$WORKFLOW" --repo "$REPO_NAME" --ref main

RUN_ID=""
for _ in $(seq 1 30); do
  RUN_ID="$(
    gh run list --repo "$REPO_NAME" --workflow "$WORKFLOW" --branch main \
      --event workflow_dispatch --limit 20 \
      --json databaseId,headSha,createdAt \
      --jq '.[] | select(.headSha=="'"$HEAD_SHA"'") | .databaseId' | head -n1
  )"
  [ -n "$RUN_ID" ] && break
  sleep 2
done

if [ -z "$RUN_ID" ]; then
  echo "Could not locate the dispatched Windows run for $HEAD_SHA" >&2
  exit 3
fi

echo "Windows preflight run: $RUN_ID"
gh run watch "$RUN_ID" --repo "$REPO_NAME" --exit-status

echo
echo "WINDOWS/MSVC PREFLIGHT PASSED for $HEAD_SHA"
