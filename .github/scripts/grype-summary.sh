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
# CVSS column: the score from the scorer marked Primary (usually NVD), labeled
# with who scored it and the CVSS version. If no scorer is marked Primary, the
# highest score is used and labeled the same way. "Other scores" lists every
# remaining distinct scorer/score pair, so disagreements stay visible. (The
# GitHub alert badge instead uses the highest score of all, which is why it can
# read higher than this column.)
#
# Scorer labels: nvd@nist.gov -> NVD, cve@mitre.org -> MITRE,
# security-advisories@github.com -> GitHub, secalert@redhat.com -> Red Hat,
# 134c704f-9b21-4f2e-91b3-4a467353bcc0 -> CISA-ADP (identified from memory),
# any other UUID -> "CNA <first 8 chars>" (a CNA's registered ID; the name is
# not in Grype's data), any other email -> its domain without the TLD
# (disclosure@vulncheck.com -> vulncheck).
#
# Colors (job summaries cannot color text, so colored circles are used):
#   CVSS band: 🔴 9.0-10.0 critical, 🟠 7.0-8.9 high, 🟡 4.0-6.9 medium,
#              🟢 0.1-3.9 low, ⚪ no score
#   Sev:       Grype's own severity (for Debian packages this is Debian's
#              rating and is what GitHub alert titles show), same colors;
#              ⚪ is Negligible/Unknown
#
# Other columns:
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
  def scorer: (. // "unknown")
              | if . == "" then "unknown"
                elif . == "nvd@nist.gov" then "NVD"
                elif . == "cve@mitre.org" then "MITRE"
                elif . == "security-advisories@github.com" then "GitHub"
                elif . == "secalert@redhat.com" then "Red Hat"
                elif . == "134c704f-9b21-4f2e-91b3-4a467353bcc0" then "CISA-ADP"
                elif test("^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-") then "CNA " + .[0:8]
                elif test("@") then (split("@")[1] | sub("\\.(com|org|net|io|edu|gov)$"; ""))
                else . end;
  def scorelabel: .src + (if .ver != "" then " v" + .ver else "" end);

  [ .matches[]? | {
      id:     .vulnerability.id,
      sev:    (.vulnerability.severity // "Unknown"),
      risk:   (.vulnerability.risk // 0),
      scores: [ rel[] | (.cvss // [])[]?
                | select(.metrics.baseScore != null)
                | { src: (.source | scorer), type: (.type // ""),
                    ver: (.version // ""), score: .metrics.baseScore } ],
      epss:   ([rel[] | (.epss // [])[]?.epss] | max // null),
      kev:    ([rel[] | (.knownExploited // [])[]?] | length > 0),
      fix:    (.vulnerability.fix.state // "unknown"),
      pkg:    .artifact.name
  } ]
  | group_by(.id)
  | map(
      (map(.scores) | add // [] | unique_by([.src, .score, .ver])) as $all
      | ($all | map(select(.type == "Primary"))) as $prim
      | (if ($prim | length) > 0 then ($prim | max_by(.score))
         else ($all | max_by(.score)) end) as $chosen
      | {
          id:     .[0].id,
          sev:    .[0].sev,
          risk:   (map(.risk) | max),
          cvss:   ($chosen.score // null),
          scorer: (if $chosen == null then "" else ($chosen | scorelabel) end),
          others: ($all | map(select(. != $chosen)) | sort_by(-.score)
                   | map((.score | tostring) + " " + scorelabel)),
          epss:   (map(.epss) | map(select(. != null)) | max),
          kev:    (map(.kev) | any),
          fix:    .[0].fix,
          pkgs:   (map(.pkg) | unique)
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
      "| CVE | CVSS (scorer) | Sev | EPSS | KEV | Fix | Other scores | Packages |\n|---|---|---|---|---|---|---|---|",
      ($rows[:$max][] |
        "| " + .id
        + " | " + (.cvss | band) + " " + (if .cvss == null then "-" else ((.cvss | tostring) + " (" + .scorer + ")") end)
        + " | " + (.sev | sevdot) + " " + .sev
        + " | " + (if .epss == null then "-"
                    elif .epss > 0 and .epss < 0.001 then "<0.1%"
                    else ((.epss * 1000 | round) / 10 | tostring) + "%" end)
        + " | " + (if .kev then "🚨 **YES**" else "no" end)
        + " | " + .fix
        + " | " + (if (.others | length) == 0 then "-"
                    else (.others[:3] | join(", ")) + (if (.others | length) > 3 then ", +" + ((.others | length) - 3 | tostring) else "" end) end)
        + " | " + (.pkgs[:4] | join(", ")) + (if (.pkgs | length) > 4 then ", +" + ((.pkgs | length) - 4 | tostring) else "" end)
        + " |"),
      (if ($rows | length) > $max then "\n_Showing the top " + ($max | tostring) + " of " + ($rows | length | tostring) + " by CVSS; the full JSON is attached to the run as an artifact._" else empty end),
      "\n<sub>CVSS is the Primary scorer'"'"'s score (usually NVD), or the highest score if none is marked Primary; Other scores lists the remaining scorers. 🔴 9.0+ · 🟠 7.0-8.9 · 🟡 4.0-6.9 · 🟢 below 4.0 · ⚪ none. Sev is Grype'"'"'s own rating (Debian'"'"'s, for Debian packages); ⚪ there means Negligible or Unknown.</sub>"
    end)
' "$file"
