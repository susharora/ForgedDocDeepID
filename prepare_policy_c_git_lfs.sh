#!/usr/bin/env bash
set -euo pipefail

# Prepare the two frozen Policy-C cache directories for GitHub using Git LFS.
# This script DOES NOT commit or push. It stages the files and verifies LFS tracking.
#
# Run from anywhere inside the ForgedDocDeepID repository:
#   bash prepare_policy_c_git_lfs.sh
#
# Preconditions:
#   - git-lfs installed
#   - data/processed/policy_c exists
#   - data/processed/policy_c_official_test exists

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$ROOT" ]]; then
  echo "ERROR: not inside a Git repository."
  exit 2
fi
cd "$ROOT"

PROJECT_DIR="data/processed/policy_c"
OFFICIAL_DIR="data/processed/policy_c_official_test"

for d in "$PROJECT_DIR" "$OFFICIAL_DIR"; do
  if [[ ! -d "$d" ]]; then
    echo "ERROR: required directory missing: $d"
    exit 3
  fi
done

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "ERROR: git-lfs is not installed."
  echo "On Ubuntu/WSL, install it with:"
  echo "  sudo apt update && sudo apt install git-lfs"
  exit 4
fi

echo "Repository: $ROOT"
echo "Git HEAD:   $(git rev-parse HEAD)"
echo
echo "Cache sizes:"
du -sh "$PROJECT_DIR" "$OFFICIAL_DIR"

echo
echo "Initialising Git LFS for this repository..."
git lfs install --local

append_once() {
  local line="$1"
  local file="$2"
  touch "$file"
  if ! grep -Fqx "$line" "$file"; then
    printf '%s\n' "$line" >> "$file"
  fi
}

echo
echo "Updating .gitignore so ONLY these frozen processed caches are unignored..."
append_once '!data/processed/policy_c/' .gitignore
append_once '!data/processed/policy_c/**' .gitignore
append_once '!data/processed/policy_c_official_test/' .gitignore
append_once '!data/processed/policy_c_official_test/**' .gitignore

echo
echo "Registering both cache trees with Git LFS..."
git lfs track "data/processed/policy_c/**"
git lfs track "data/processed/policy_c_official_test/**"

echo
echo "Staging metadata..."
git add .gitattributes .gitignore

echo
echo "Staging frozen cache files..."
git add "$PROJECT_DIR" "$OFFICIAL_DIR"

echo
echo "Verifying every staged cache file resolves to filter=lfs..."
bad=0
checked=0
while IFS= read -r -d '' f; do
  checked=$((checked + 1))
  attr="$(git check-attr filter -- "$f" | awk -F': ' '{print $3}')"
  if [[ "$attr" != "lfs" ]]; then
    echo "NOT LFS: $f  (filter=$attr)"
    bad=$((bad + 1))
  fi
done < <(git diff --cached --name-only -z -- "$PROJECT_DIR" "$OFFICIAL_DIR")

echo "Checked staged cache files: $checked"
if [[ "$checked" -eq 0 ]]; then
  echo "ERROR: no cache files were staged. Check .gitignore and directory contents."
  exit 5
fi
if [[ "$bad" -ne 0 ]]; then
  echo "ERROR: $bad staged cache files are not tracked by Git LFS."
  echo "Nothing has been committed or pushed."
  exit 6
fi
echo "All staged cache files are Git LFS-managed."

echo
echo "Git LFS status:"
git lfs status

echo
echo "Git status summary:"
git status --short

echo
echo "READY — nothing has been committed or pushed yet."
echo
echo "Recommended next commands:"
echo '  git commit -m "Track frozen Policy-C caches via Git LFS"'
echo "  git push origin main"
echo
echo "After pushing, on another machine:"
echo "  git pull"
echo "  git lfs pull"
