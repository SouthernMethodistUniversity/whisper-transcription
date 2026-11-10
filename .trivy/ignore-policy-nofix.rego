package trivy

import future.keywords.in

# Host kernel / kernel header packages. Containers don't ship or run a kernel;
# these CVEs are false positives for container-only risk.
# Keep this list in sync with .trivy/ignore-policy.rego, .grype.yaml and
# .grype/nofix.yaml.
kernel_packages := {
	"linux-libc-dev",
	"linux-headers-generic",
	"linux-headers",
	"linux-image",
	"linux-modules",
	"linux-modules-extra",
	"linux-base",
	"linux-tools-common",
	"linux-firmware",
}

default ignore := false

# 1. Host kernel packages
ignore {
	pkg := lower(input.PkgName)
	some k in kernel_packages
	startswith(pkg, k)
}

# 2. Anything that HAS a fix belongs to the fixable pass.
ignore {
	has_fix
}

# Trivy OMITS FixedVersion when no fix exists (it is not ""), so this must
# tolerate the field being absent: an undefined comparison leaves has_fix
# undefined and rule 2 does not fire.
has_fix {
	input.FixedVersion != ""
}
