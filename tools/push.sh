#!/bin/bash
# Commit the work tree on the editor machine and push it to the homelab (bare repo ~/forgeinfer/repo.git; its post-receive hook checks the files out into ~/forgeinfer/src).
#   tools/push.sh "message"
cd "$(dirname "$0")/.." || exit 1
export GIT_SSH_COMMAND='ssh -i $HOME/.ssh/id_ed25519 -o BatchMode=yes -o IdentitiesOnly=yes'
git add -A
git -c user.name="AizenKai1001" -c user.email="174204465+AizenKai1001@users.noreply.github.com" commit -q -m "${1:-update}" 2>/dev/null || true
git push -q homelab HEAD:main 2>&1 | tail -n 3
