#!/bin/bash
# Move big idle directories from the NVMe (root filesystem) to the emptier hard drive and leave a symlink at the old path, so nothing that refers to the old path breaks.
#
#   offload.sh move <dir>...    copy <dir> to $DEST_ROOT<dir>, verify the copy file by file (checksums), replace <dir> by a symlink, then free the original
#   offload.sh restore <dir>... undo: copy the data back to <dir> and remove the copy on the hard drive
#   offload.sh list             what has been moved (the manifest)
#
# Nothing is removed before the copy has been verified identical. Runs as the owner of the directories; use sudo for root-owned ones.
set -u
DEST_ROOT=${DEST_ROOT:-/mnt/data/nvme-offload}
MANIFEST=$DEST_ROOT/MANIFEST.txt
LOW="nice -n 19 ionice -c3"

say() { echo "[offload $(date +%H:%M:%S)] $*"; }

move_one() {
    local src dest bytes
    src=$(realpath -s "$1")
    if [ -L "$src" ]; then say "skip $src: already a symlink -> $(readlink "$src")"; return 0; fi
    if [ ! -d "$src" ]; then say "skip $src: not a directory"; return 1; fi
    case "$src" in
        /|/home|/home/user|/opt|/opt/stacks|/var|/var/lib|/mnt|/mnt/data|/mnt/data) say "refuse $src: too broad"; return 1;;
        /usr*|/etc*|/boot*|/proc*|/sys*|/dev*|/run*|/var/lib/docker*|/var/lib/containerd*|/var/lib/libvirt*) say "refuse $src: not for this script"; return 1;;
    esac
    dest="$DEST_ROOT$src"
    bytes=$(du -sb "$src" | cut -f1)
    say "move $src ($((bytes / 1048576)) MiB) -> $dest"
    mkdir -p "$dest" || return 1
    # 1. copy
    if ! $LOW rsync -aH "$src"/ "$dest"/; then say "FAILED: rsync of $src, original untouched"; return 1; fi
    # 2. verify: every file compared by checksum, nothing missing or extra
    local diffs
    diffs=$($LOW rsync -aHcn --delete --itemize-changes "$src"/ "$dest"/ | head -20)
    if [ -n "$diffs" ]; then say "FAILED: the copy differs from the original, original untouched:"; echo "$diffs"; return 1; fi
    say "verified $src"
    # 3. swap in the symlink (the original is renamed first, so a failure leaves it in place)
    if ! mv "$src" "$src.offloading"; then say "FAILED: cannot rename $src"; return 1; fi
    if ! ln -s "$dest" "$src" || [ ! -d "$src" ]; then
        rm "$src" 2>/dev/null
        mv "$src.offloading" "$src"
        say "FAILED: symlink did not work, original restored"
        return 1
    fi
    # 4. free the original
    rm -r "$src.offloading"
    echo "$(date '+%F %T')  $src -> $dest  $((bytes / 1048576)) MiB" >> "$MANIFEST"
    say "done $src"
}

restore_one() {
    local src dest
    src=$(realpath -s "$1")
    dest="$DEST_ROOT$src"
    if [ ! -L "$src" ] || [ ! -d "$dest" ]; then say "skip $src: not an offloaded directory"; return 1; fi
    say "restore $src from $dest"
    local tmp="$src.restoring"
    mkdir -p "$tmp" || return 1
    $LOW rsync -aH "$dest"/ "$tmp"/ || { say "FAILED: copy back"; return 1; }
    [ -z "$($LOW rsync -aHcn --delete --itemize-changes "$dest"/ "$tmp"/ | head -5)" ] || { say "FAILED: verification"; return 1; }
    rm "$src" && mv "$tmp" "$src" && rm -r "$dest"
    echo "$(date '+%F %T')  RESTORED $src" >> "$MANIFEST"
    say "restored $src"
}

cmd=${1:-}
shift || true
mkdir -p "$DEST_ROOT"
case "$cmd" in
    move) for d in "$@"; do move_one "$d"; done; df -h / "$DEST_ROOT" | cat ;;
    restore) for d in "$@"; do restore_one "$d"; done ;;
    list) cat "$MANIFEST" 2>/dev/null || echo "nothing moved yet" ;;
    *) sed -n '2,10p' "$0"; exit 2 ;;
esac
