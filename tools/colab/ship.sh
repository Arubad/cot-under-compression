#!/usr/bin/env bash
# Ship the committed HEAD to a Colab session: tools/colab/ship.sh <session>
# Refuses a dirty tree so the commit recorded in every run log matches the code that ran.
set -euo pipefail
S="${1:?session name}"
cd "$(git rev-parse --show-toplevel)"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then echo "refusing: tracked files modified; commit first" >&2; exit 1; fi
TMP="$(mktemp -d)"; git rev-parse --short HEAD > "$TMP/GIT_COMMIT"
git archive --format=tar.gz --add-file="$TMP/GIT_COMMIT" -o "$TMP/repo.tgz" HEAD
colab upload -s "$S" "$TMP/repo.tgz" /content/repo.tgz </dev/null
echo "shipped $(cat "$TMP/GIT_COMMIT") to $S:/content/repo.tgz — now run: colab exec -s $S -f tools/colab/setup_vm.py </dev/null"
