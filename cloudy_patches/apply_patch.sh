#!/bin/bash
# Apply the TODDLERS "unattenuated diffuse nebular continuum" patch to a Cloudy
# source tree and rebuild. This adds the `save diffuse continuum unattenuated`
# command (writing the .diffContUnatt file) that the TODDLERS noDust SEDs read.
#
# Usage:
#   ./apply_patch.sh /path/to/cloudy            # apply + build
#   MAKE_JOBS=8 ./apply_patch.sh /path/to/cloudy
#
# The patch is against Cloudy C22, commit 01c4cfc6 (master), and is verified to
# apply (git apply) and compile cleanly on C22.00, the 2025 master branch, and
# C25.00. For those it applies as a plain patch; for nearby commits the script
# falls back to a 3-way merge, then to `patch -p1`. If none apply it errors out
# with re-port guidance (rerun with FORCE_COPY=1 to overwrite the seven source
# files with the bundled C22 versions, only safe on a close base).
# The change is purely additive (a new save command + a new continuum array), so
# it is safe to apply to a working tree and trivially reverted with `git checkout`.
set -euo pipefail

CLOUDY_DIR="${1:?usage: apply_patch.sh /path/to/cloudy   (the dir containing source/)}"
PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH="$PATCH_DIR/cloudy_unattenuated_diffuse_continuum.patch"
FILES="rfield.h cont_createmesh.cpp cont_setintensity.cpp iter_startend.cpp rt_continuum.cpp parse_save.cpp save_do.cpp"

[ -f "$PATCH" ] || { echo "ERROR: patch not found at $PATCH"; exit 1; }
[ -d "$CLOUDY_DIR/source" ] || { echo "ERROR: $CLOUDY_DIR has no source/ subdir"; exit 1; }
cd "$CLOUDY_DIR"

# Skip if already patched (idempotent).
if grep -q "ConEmitOutUnatt" source/rfield.h 2>/dev/null; then
    echo "Cloudy source already carries the patch (ConEmitOutUnatt present)."
else
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1 && git apply --check "$PATCH" 2>/dev/null; then
        git apply "$PATCH"; echo "Applied via 'git apply'."
    elif git rev-parse --is-inside-work-tree >/dev/null 2>&1 && git apply --check --3way "$PATCH" 2>/dev/null; then
        git apply --3way "$PATCH"; echo "Applied via 'git apply --3way'."
    elif patch -p1 --dry-run < "$PATCH" >/dev/null 2>&1; then
        patch -p1 < "$PATCH"; echo "Applied via 'patch -p1'."
    elif [ "${FORCE_COPY:-0}" = "1" ]; then
        echo "FORCE_COPY=1: overwriting the 7 source files with the C22/01c4cfc6 versions."
        echo "  (Only safe if your Cloudy is close to that base; verify the build.)"
        for f in $FILES; do cp "$PATCH_DIR/$f" "source/$f"; done
    else
        echo "ERROR: the patch did not apply to your Cloudy source." >&2
        echo "It is verified to apply+compile on C22.00, master (2025), and C25.00; a version that" >&2
        echo "refactored the continuum/save code may need the ~6-line change re-ported by hand" >&2
        echo "(see this README: one parallel ConEmitOutUnatt accumulator in rt_continuum + a save handler)." >&2
        echo "To overwrite the 7 files with the C22 versions anyway, re-run with FORCE_COPY=1." >&2
        exit 1
    fi
fi

echo "Building Cloudy (make -j${MAKE_JOBS:-4}) ..."
cd source && make -j"${MAKE_JOBS:-4}"
echo
echo "Done. Verify with:  printf 'test\\nsave diffuse continuum unattenuated \".d\"\\n' | ./cloudy.exe"
echo "The TODDLERS noDust SEDs use this via 'save last diffuse continuum unattenuated'."
