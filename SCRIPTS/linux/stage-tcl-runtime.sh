#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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
TCL_DEST="${REPO_ROOT}/build/lib"
mkdir -p "${TCL_DEST}"
cp -a "${TCL_PKG_LIB}/tcl8.6" "${TCL_DEST}/"
cp -a "${TCL_PKG_LIB}/tcl8" "${TCL_DEST}/"
echo "Staged Tcl runtime from ${TCL_PKG_LIB} to ${TCL_DEST}"
