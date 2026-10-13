#!/bin/bash
# Push the sources to the homelab and run the Rust tests there (release mode, nice 15). Extra arguments go to `cargo test` (a crate, a test name ...).
#   tools/test.sh                       whole workspace
#   tools/test.sh -p forge-tok          one crate
#   tools/test.sh -p forge-tok bpe      tests whose name contains "bpe"
# The goldens (tests/golden) are read by the tests from the repository checkout on the homelab.
cd "$(dirname "$0")/.." || exit 1
bash tools/push.sh "${COMMIT_MESSAGE:-wip}" || exit 1
exec bash tools/hl.sh "cargo test --release $* 2>&1 | tail -n 60"
