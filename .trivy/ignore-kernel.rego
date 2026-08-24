package trivy

import future.keywords.in

# Packages that represent the host kernel or kernel headers.
# These CVEs describe bugs in the kernel itself, which containers
# don't ship or run — they share whatever kernel the host provides.
# Flagging them inside an image scan is a false positive for
# container-only risk, so we filter them out here.
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

ignore {
	pkg := lower(input.PkgName)
	some k in kernel_packages
	startswith(pkg, k)
}
