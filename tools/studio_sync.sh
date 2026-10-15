#!/bin/bash
# Copy Forge Studio (the studio/ project and the crates it depends on: forge-gguf, forge-core) to the homelab, so that it can be built and tested there when the PC must not be loaded (the user is gaming, a GPU job runs).
#   tools/studio_sync.sh            ->  ~/forgeinfer/studio_work/{studio,crates/forge-gguf,crates/forge-core,Cargo.toml}  (no target folders, no test homes, no screenshots)
# Then:  tools/hl.sh -n "cd ~/forgeinfer/studio_work/studio && . ~/.cargo/env && nice -n 15 cargo test --lib 2>&1 | tail -5"
# (Linux: the Windows-only text recognition is not built there, everything else is; files you deleted stay on the homelab, the build cache in studio_work/studio/target is kept.)
# The path dependencies inherit version and edition from the workspace root of the repository: a small root manifest with just those two crates is written next to them.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KEY="$HOME/.ssh/id_ed25519"
SSH="ssh -i $KEY -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=10 user@homelab"
VERSION=$(grep -m1 '^version' "$ROOT/Cargo.toml" | sed 's/.*"\(.*\)".*/\1/')
tar -C "$ROOT" --exclude=target --exclude='studio/home*' --exclude=studio/shots --exclude=studio/retry_test --exclude=__pycache__ -cf - studio crates/forge-gguf crates/forge-core | $SSH 'mkdir -p ~/forgeinfer/studio_work && tar -C ~/forgeinfer/studio_work -xf -'
$SSH 'cat > ~/forgeinfer/studio_work/Cargo.toml' <<EOF
[workspace]
resolver = "2"
members = ["crates/forge-core", "crates/forge-gguf"]

[workspace.package]
version = "$VERSION"
edition = "2021"
publish = false
EOF
echo "synced studio, forge-gguf and forge-core to ~/forgeinfer/studio_work (version $VERSION)"
