#!/usr/bin/env bash
# Turn a Grype JSON report into a prioritized Markdown table (one row per CVE).
# Usage: grype-summary.sh <grype-json-file> <image-name> [max-rows]
#
# Report only: nothing here suppresses or filters findings.
#
# Sort order: CISA KEV hits first (pinned so a row limit can never hide one),
# then CVSS descending, then EPSS, then Grype's risk score. CVEs with no CVSS
# score sort last.
#
# Colors (job summaries cannot color text, so colored circles are used):
#   CVSS band: 🔴 9.0-10.0 critical, 🟠 7.0-8.9 high, 🟡 4.0-6.9 medium,
#              🟢 0.1-3.9 low, ⚪ no score
#   Sev:       Grype's own severity (for Debian packages this is Debian's
#              rating and is what GitHub alert titles show), same colors;
#              ⚪ is Negligible/Unknown
#
# Columns:
#   CVSS  - highest base score across the match and its related records
#           (often NVD's; this is what drives the GitHub alert badge)
#   Sev   - Grype's severity for the match
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
  def band: if . == null then "⚪"
            elif . >= 9 then "🔴"
            elif . >= 7 then "🟠"
            elif . >= 4 then "🟡"
            elif . > 0  then "🟢"
            else "⚪" end;
  def sevdot: ascii_downcase
              | if . == "critical" then "🔴"
                elif . == "high" then "🟠"
                elif . == "medium" then "🟡"
                elif . == "low" then "🟢"
                else "⚪" end;
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
  | sort_by([(if .kev then 0 else 1 end), -(.cvss // -1), -(.epss // 0), -.risk])
  | . as $rows
  | def n(f): ($rows | map(select(f)) | length | tostring);
    ("### Grype prioritized report: `" + $image + "`\n"),
    ("**" + ($rows | length | tostring) + "** distinct CVEs, by CVSS: "
      + "🔴 **" + n(.cvss != null and .cvss >= 9) + "** critical · "
      + "🟠 **" + n(.cvss != null and .cvss >= 7 and .cvss < 9) + "** high · "
      + "🟡 **" + n(.cvss != null and .cvss >= 4 and .cvss < 7) + "** medium · "
      + "🟢 **" + n(.cvss != null and .cvss > 0 and .cvss < 4) + "** low · "
      + "⚪ **" + n(.cvss == null or .cvss <= 0) + "** unscored\n"),
    ("**" + n(.kev) + "** in CISA KEV | **" + n((.epss // 0) >= 0.1) + "** with EPSS >= 10% | **"
      + n(.fix == "fixed") + "** with a fix available\n"),
    (if ($rows | length) == 0 then "_No findings._" else
      "| CVE | CVSS | Sev | EPSS | KEV | Fix | Packages |\n|---|---|---|---|---|---|---|",
      ($rows[:$max][] |
        "| " + .id
        + " | " + (.cvss | band) + " " + (if .cvss == null then "-" else (.cvss | tostring) end)
        + " | " + (.sev | sevdot) + " " + .sev
        + " | " + (if .epss == null then "-"
                    elif .epss > 0 and .epss < 0.001 then "<0.1%"
                    else ((.epss * 1000 | round) / 10 | tostring) + "%" end)
        + " | " + (if .kev then "🚨 **YES**" else "no" end)
        + " | " + .fix
        + " | " + (.pkgs[:4] | join(", ")) + (if (.pkgs | length) > 4 then ", +" + ((.pkgs | length) - 4 | tostring) else "" end)
        + " |"),
      (if ($rows | length) > $max then "\n_Showing the top " + ($max | tostring) + " of " + ($rows | length | tostring) + " by CVSS; the full JSON is attached to the run as an artifact._" else empty end),
      "\n<sub>CVSS 🔴 9.0+ · 🟠 7.0-8.9 · 🟡 4.0-6.9 · 🟢 below 4.0 · ⚪ none. Sev is Grype'"'"'s own rating (Debian'"'"'s, for Debian packages); ⚪ there means Negligible or Unknown.</sub>"
    end)
' "$file"
