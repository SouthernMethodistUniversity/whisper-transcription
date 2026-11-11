#!/usr/bin/env bash
# Turn a Grype JSON report into a prioritized Markdown table (one row per CVE).
# Usage: grype-summary.sh <grype-json-file> <image-name> [max-rows]
#
# Report only: nothing here suppresses or filters findings. Rows are sorted by
# Grype's own risk score (which folds in CVSS, EPSS and KEV), highest first,
# with KEV hits always sorted above everything else.
#
# Columns:
#   Sev   - Grype's severity for the match (for Debian packages this is
#           Debian's rating; it is what appears in the GitHub alert title)
#   CVSS  - highest base score across the match and its related records
#           (often NVD's; this is what drives the GitHub alert badge)
#   EPSS  - probability of exploitation in the next 30 days (FIRST)
#   KEV   - listed in CISA's Known Exploited Vulnerabilities catalog
#   Fix   - Grype's fix state: fixed, not-fixed, wont-fix, unknown
set -euo pipefail

file="${1:?usage: grype-summary.sh <grype-json-file> <image-name> [max-rows]}"
image="${2:?usage: grype-summary.sh <grype-json-file> <image-name> [max-rows]}"
max="${3:-40}"

if [ ! -s "$file" ]; then
  echo "### Grype prioritized report: \`$image\`"
  echo
  echo "No Grype JSON at \`$file\` (or empty); nothing to summarize."
  exit 0
fi

jq -r --arg image "$image" --argjson max "$max" '
  def rel: [.vulnerability] + (.relatedVulnerabilities // []);
  [ .matches[]? | {
      id:   .vulnerability.id,
      sev:  (.vulnerability.severity // "Unknown"),
      risk: (.vulnerability.risk // 0),
      cvss: ([rel[] | (.cvss // [])[]?.metrics.baseScore] | max // null),
      epss: ([rel[] | (.epss  // [])[]?.epss]            | max // null),
      kev:  ([rel[] | (.knownExploited // [])[]?] | length > 0),
      fix:  (.vulnerability.fix.state // "unknown"),
      pkg:  .artifact.name
  } ]
  | group_by(.id)
  | map({
      id:   .[0].id,
      sev:  .[0].sev,
      risk: (map(.risk) | max),
      cvss: (map(.cvss) | map(select(. != null)) | max),
      epss: (map(.epss) | map(select(. != null)) | max),
      kev:  (map(.kev) | any),
      fix:  .[0].fix,
      pkgs: (map(.pkg) | unique)
    })
  | sort_by([(if .kev then 0 else 1 end), -.risk])
  | . as $rows
  | ("### Grype prioritized report: `" + $image + "`\n"),
    ("**" + ($rows | length | tostring) + "** distinct CVEs"
      + " | **" + ($rows | map(select(.kev)) | length | tostring) + "** in CISA KEV"
      + " | **" + ($rows | map(select((.epss // 0) >= 0.1)) | length | tostring) + "** with EPSS >= 10%"
      + " | **" + ($rows | map(select(.fix == "fixed")) | length | tostring) + "** with a fix available\n"),
    (if ($rows | length) == 0 then "_No findings._" else
      "| CVE | Sev | CVSS | EPSS | KEV | Fix | Packages |\n|---|---|---|---|---|---|---|",
      ($rows[:$max][] |
        "| " + .id
        + " | " + .sev
        + " | " + (if .cvss == null then "-" else (.cvss | tostring) end)
        + " | " + (if .epss == null then "-" else ((.epss * 1000 | round) / 10 | tostring) + "%" end)
        + " | " + (if .kev then "**YES**" else "no" end)
        + " | " + .fix
        + " | " + (.pkgs[:4] | join(", ")) + (if (.pkgs | length) > 4 then ", +" + ((.pkgs | length) - 4 | tostring) else "" end)
        + " |"),
      (if ($rows | length) > $max then "\n_Showing the top " + ($max | tostring) + " of " + ($rows | length | tostring) + "; the full JSON is attached to the run as an artifact._" else empty end)
    end)
' "$file"
