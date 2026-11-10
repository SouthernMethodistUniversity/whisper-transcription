#!/usr/bin/env bash
# Prefix every rule title and result message in a SARIF file with a label.
# Usage: tag-sarif.sh <sarif-file> <label>
#
# GitHub code scanning has no `category:` search qualifier, but alert titles are
# free-text searchable, so the label is what makes "[NOFIX]" filterable.
# The label must stay byte-identical between runs (GitHub identifies an alert
# partly by rule and message text). Rule IDs are deliberately left untouched.
set -euo pipefail

file="${1:?usage: tag-sarif.sh <sarif-file> <label>}"
label="${2:?usage: tag-sarif.sh <sarif-file> <label>}"

if [ ! -s "$file" ]; then
  echo "tag-sarif: no SARIF at '$file' (or empty) — nothing to tag"
  exit 0
fi

tmp="$(mktemp)"

# select(. != null) keeps this a valid path expression while skipping absent
# fields; without it jq aborts on any rule lacking a shortDescription.
jq --arg L "$label" '
    (.runs[]?.tool.driver.rules[]?.shortDescription.text | select(. != null)) |= ($L + " " + .)
  | (.runs[]?.results[]?.message.text                    | select(. != null)) |= ($L + " " + .)
' "$file" > "$tmp"

mv "$tmp" "$file"

echo "tag-sarif: labelled $(jq '[.runs[]?.results[]?] | length' "$file") result(s) in $file with '$label'"
