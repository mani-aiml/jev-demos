#!/bin/bash
# Make each missing base photo with Codex gpt-6-luna (low effort). Mug variants take raw/mug_white.png as the reference.
# The PNG lands in ~/.codex/generated_images/<session id>/; the session id is in the log.
cd "$(dirname "$0")"
CX=~/.codex/plugins/.plugin-appserver/codex-cli/bin/codex
one() {
  name=$1; [ -f raw/$name.png ] && return
  ref=(); grep -q "attached image" prompts/$name.txt && ref=(-i raw/mug_white.png)
  $CX exec -m gpt-6-luna -c model_reasoning_effort="low" --skip-git-repo-check "$(cat prompts/$name.txt)" "${ref[@]}" \
    < /dev/null > prompts/$name.log 2>&1
  sid=$(sed -n 's/^session id: //p' prompts/$name.log | head -1)
  out=$(ls -t ~/.codex/generated_images/$sid/*.png 2>/dev/null | head -1)
  [ -f "$out" ] && cp "$out" raw/$name.png && echo "$name ok" || echo "$name FAILED"
}
for name in "$@"; do one "$name" & while [ "$(jobs -r | wc -l)" -ge 4 ]; do sleep 1; done; done
wait
