#!/usr/bin/env bash
# push_to_GitHub.sh — push local repo to GitHub
# USAGE: ./push_to_GitHub.sh git@github.com:MartinTrappe/pykan_mit.git
set -euo pipefail

REMOTE_URL="${1:-}"
if [[ -z "$REMOTE_URL" ]]; then
  echo "Usage: $0 <remote-url>"
  echo "Example: $0 git@github.com:MartinTrappe/pykan_mit.git"
  exit 1
fi

# 1) ensure we're in a git repo
if [[ ! -d .git ]]; then
  git init
fi

# 2) make sure we can commit (set minimal local identity if missing)
git config --get user.name  >/dev/null 2>&1 || git config user.name  "local"
git config --get user.email >/dev/null 2>&1 || git config user.email "local@example.com"

# 3) stage & commit (create an initial commit if needed, otherwise sync changes)
git add -A
if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
  git commit -m "Initial commit"
else
  if [[ -n "$(git status --porcelain)" ]]; then
    git commit -m "Sync"
  fi
fi

# 4) wire up the remote (create or update 'origin')
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REMOTE_URL"
else
  git remote add origin "$REMOTE_URL"
fi

# 5) push HEAD to a new unique branch on the remote to avoid conflicts
SHORT_SHA="$(git rev-parse --short HEAD)"
STAMP="$(date +%Y%m%d-%H%M%S)"
TARGET_BRANCH="upload-${STAMP}-${SHORT_SHA}"

# If branch name somehow collides (extremely unlikely), add a nano‐timestamp suffix
if ! git push -u origin HEAD:"$TARGET_BRANCH"; then
  TARGET_BRANCH="${TARGET_BRANCH}-$(date +%s%N)"
  git push -u origin HEAD:"$TARGET_BRANCH"
fi

echo "✅ Done."
echo "   Pushed to: $REMOTE_URL"
echo "   Branch:    $TARGET_BRANCH"
echo "In GitHub → Settings → Branches, you can set $TARGET_BRANCH as default if you want."
