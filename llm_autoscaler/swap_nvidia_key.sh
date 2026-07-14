#!/bin/bash
# Swap the NVIDIA API key in api_keys.conf from a key file, WITHOUT the key
# ever being echoed to the terminal / transcript.
#
# Steps:
#   1. Put the new (verified-account) key on a single line in .nvkey_new:
#        - in YOUR terminal: printf %s 'PASTE_KEY_HERE' > llm_autoscaler/.nvkey_new
#        - or open an editor and save just the key to that path
#   2. Run: bash swap_nvidia_key.sh
# It backs up api_keys.conf, replaces the nvidia line, shreds .nvkey_new,
# and re-validates the new key against NVIDIA (prints only status, never the key).
set -euo pipefail
cd "$(dirname "$0")"

KEYFILE=".nvkey_new"
CONF="api_keys.conf"
[ -s "$KEYFILE" ] || { echo "ERROR: $KEYFILE missing/empty. Put the new key there first."; exit 1; }

cp "$CONF" "${CONF}.bak.$(date +%Y%m%d_%H%M%S)"

python3 - "$CONF" "$KEYFILE" <<'PY'
import re, sys, pathlib
conf, keyfile = sys.argv[1], sys.argv[2]
key = pathlib.Path(keyfile).read_text().strip()
assert key, "empty key"
lines = pathlib.Path(conf).read_text().splitlines()
found = False
for i, l in enumerate(lines):
    if re.match(r'^\s*nvidia\s*:', l):
        lines[i] = f"nvidia: {key}"
        found = True
if not found:
    lines.append(f"nvidia: {key}")
pathlib.Path(conf).write_text("\n".join(lines) + "\n")
print(f"nvidia line updated (key length {len(key)}, not shown)")
PY

# shred the temp key file
command -v shred >/dev/null && shred -u "$KEYFILE" || rm -f "$KEYFILE"
echo "temp key file removed"

echo "=== re-validating new key (status only, key not shown) ==="
KEY=$(awk -F: '/^[[:space:]]*nvidia/{k=$2; gsub(/^[ \t]+|[ \t]+$/,"",k); print k; exit}' "$CONF")
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 \
  https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"meta/llama-3.1-8b-instruct","messages":[{"role":"user","content":"reply with only the number 7"}],"max_tokens":5}')
echo "NVIDIA inference check: HTTP $code $( [ "$code" = 200 ] && echo '(OK — new key works)' || echo '(FAILED — check the key / restore api_keys.conf.bak.*)' )"
