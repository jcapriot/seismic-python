#!/bin/sh
# Build a pristine Seismic Unix (upstream, not the pyimager fork) to validate seispy against the real programs.
#
# Nothing is exported to the calling shell: CWPROOT and PATH are only set inside this script.
# The source is, in order of preference:
#   1. an existing worktree/checkout at SU_REFERENCE_ROOT,
#   2. a git worktree of the clone SU_REFERENCE_CLONE (so the branch checked out there is left alone),
#   3. a shallow clone of SU_REFERENCE_URL (what CI does).
#
#   SU_REFERENCE_REF    commit to build        (default: pinned below, upstream master)
#   SU_REFERENCE_ROOT   source + install dir   (default: ~/codes/SeisUnix-master)
#   SU_REFERENCE_CLONE  existing SeisUnix clone (default: ~/codes/SeisUnix)
#   SU_REFERENCE_URL    where to clone from    (default: upstream on GitHub)
#   SU_REFERENCE_XDR    1 = big endian data (SU's -DSUXDR), 0 = native (default: 0)
#   JOBS                parallel make jobs

set -eu

REF=${SU_REFERENCE_REF:-4db4181f654e474507721412bd333708b62f81a1}
ROOT=${SU_REFERENCE_ROOT:-$HOME/codes/SeisUnix-master}
CLONE=${SU_REFERENCE_CLONE:-$HOME/codes/SeisUnix}
URL=${SU_REFERENCE_URL:-https://github.com/JohnWStockwellJr/SeisUnix.git}
XDR=${SU_REFERENCE_XDR:-0}
JOBS=${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 2)}

if [ ! -d "$ROOT/src" ]; then
    if [ -d "$CLONE/.git" ] || [ -f "$CLONE/.git" ]; then
        git -C "$CLONE" worktree add --detach "$ROOT" "$REF"
    else
        mkdir -p "$ROOT"
        git -C "$ROOT" init -q
        git -C "$ROOT" fetch --depth 1 "$URL" "$REF"
        git -C "$ROOT" checkout -q FETCH_HEAD
    fi
fi

export CWPROOT="$ROOT"
export PATH="$ROOT/bin:$PATH"

if [ "$XDR" = 1 ]; then XDRFLAG=-DSUXDR; else XDRFLAG=; fi

# SU's own configuration for the platform (the default src/Makefile.config is a macOS one). The XDR and
# TIRPC settings of the Linux one are overridden on the make command line.
if [ "$(uname -s)" = Linux ] && [ ! -f "$ROOT/src/Makefile.config.orig" ]; then
    cp "$ROOT/src/Makefile.config" "$ROOT/src/Makefile.config.orig"
    cp "$ROOT/src/configs/Makefile.config_Linux_Ubuntu_22.04" "$ROOT/src/Makefile.config"
fi

# Modern compilers treat the K&R-style code that SU is full of as errors.
SU_CFLAGS_EXTRA="-Wno-implicit-function-declaration -Wno-implicit-int -Wno-int-conversion \
-Wno-incompatible-pointer-types -Wno-deprecated-non-prototype -Wno-return-type -Wno-error"

cd "$ROOT/src"
sh ./mkdirectories.sh 2>/dev/null || true   # fails harmlessly if the directories exist

# only the command line programs: no X11/plotting/Mesa.
# The libraries are archive-based and race when built in parallel, so they are built serially;
# only the (independent) main programs use -j.
SU_MAKE="make XDRFLAG=$XDRFLAG XDRIFLAGS= XDRLFLAGS= OPTC=\"-g -std=gnu99 -Wall -Wno-long-long $SU_CFLAGS_EXTRA\""
for d in cwp par su; do
    echo "== building $d/lib =="
    eval "$SU_MAKE -C $d/lib" >/dev/null
    echo "== building $d/main =="
    eval "$SU_MAKE -j$JOBS -k -C $d/main" >/dev/null || echo "   (some $d/main programs failed to build)"
done

# fail loudly if the programs that the tests use were not made
for prog in suplane suspike susynlv sugain sushw sugethw; do
    [ -x "$ROOT/bin/$prog" ] || { echo "SU build failed: $prog is missing" >&2; exit 1; }
done

echo "SU built in $ROOT/bin (data is $([ "$XDR" = 1 ] && echo big-endian XDR || echo native-endian))"
