#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST_ROOTS=("$@")
if [[ ${#DEST_ROOTS[@]} -eq 0 ]]; then
  DEST_ROOTS=("${REPO_ROOT}/build")
fi

TCL_PKG_LIB=""
for tcl_pkg in "${HOME}/.conan2/p/b"/tcl*/p/lib; do
  if [[ -f "${tcl_pkg}/tcl8.6/init.tcl" ]]; then
    TCL_PKG_LIB="${tcl_pkg}"
    break
  fi
done
if [[ -z "${TCL_PKG_LIB}" ]]; then
  echo "ERROR: Conan Tcl not found under ${HOME}/.conan2/p/b" >&2
  exit 1
fi

for dest_root in "${DEST_ROOTS[@]}"; do
  TCL_DEST="${dest_root}/lib"
  mkdir -p "${TCL_DEST}"
  cp -a "${TCL_PKG_LIB}/tcl8.6" "${TCL_DEST}/"
  cp -a "${TCL_PKG_LIB}/tcl8" "${TCL_DEST}/"
  echo "Staged Tcl runtime from ${TCL_PKG_LIB} to ${TCL_DEST}"
done
