package trivy

import future.keywords.in

# Host kernel / kernel header packages. Containers don't ship or run a kernel;
# these CVEs are false positives for container-only risk.
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

# Kernel packages are the ONLY thing this policy suppresses. No severity
# threshold; fixed and unfixed both report (the workflow's ignore-unfixed
# input handles the fixable/no-fix split, not this file).
# Keep this list in sync with .trivy/ignore-policy-nofix.rego, .grype.yaml
# and .grype/nofix.yaml.
ignore {
	pkg := lower(input.PkgName)
	some k in kernel_packages
	startswith(pkg, k)
}
