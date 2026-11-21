#!/usr/bin/env bash
# keeper.sh <session> <queue_id> [interval_s]
# Free-tier Colab can reclaim the VM without warning when the GPU allowance runs out, and anything
# not downloaded is lost.  Every <interval_s> this packs the *live* results directory on the VM
# (finished cells inside a long command included) and downloads it to
# results/archive/<queue_id>_live.tgz (the previous good copy is kept as .prev).
# Every colab call has a hard `timeout`: `colab exec` can stall indefinitely on a lost connection.
# Exits when the queue finishes, a command fails, or the session is gone.
# Resume after a reclaim: new session -> ship -> setup -> upload the live archive -> restore_results.py
# (ARCHIVE=...) -> relaunch the same queue; --skip-existing skips finished cells.
cd "$(git rev-parse --show-toplevel)"
S="${1:?session}"; Q="${2:?queue id}"; INT="${3:-1200}"
DST="results/archive/${Q}_live.tgz"
while true; do
  OUT=$(timeout 240 colab exec -s "$S" --timeout 120 --env QUEUE_ID="$Q" --env N=6 -f tools/colab/qstatus.py </dev/null 2>&1)
  if echo "$OUT" | grep -qiE "session .*not found|no active session|does not exist|404"; then echo "[keeper $(date -u +%H:%M)] SESSION GONE"; break; fi
  STATE=$(echo "$OUT" | python3 -c 'import sys,json,re
t=sys.stdin.read(); m=re.search(r"^\{.*\}$", t, re.M)
if not m: print("unknown"); sys.exit()
d=json.loads(m.group(0))
if any(x["rc"]!=0 for x in d.get("done",[])): print("failed")
elif d.get("finished"): print("finished")
else: print("running:%s/%s:%s" % (d.get("index"), d.get("n"), d.get("running")))' 2>/dev/null)
  timeout 300 colab exec -s "$S" --timeout 200 --env JOB_ID="${Q}live" -f tools/colab/pack.py </dev/null >/dev/null 2>&1
  if timeout 400 colab download -s "$S" "/content/${Q}live_results.tgz" "${DST}.new" </dev/null >/dev/null 2>&1 && [ -s "${DST}.new" ]; then
    [ -f "$DST" ] && mv "$DST" "${DST}.prev"; mv "${DST}.new" "$DST"
    echo "[keeper $(date -u +%H:%M)] $STATE  saved $(du -h "$DST" | cut -f1)"
  else
    echo "[keeper $(date -u +%H:%M)] $STATE  download FAILED"
  fi
  case "$STATE" in finished|failed) echo "$OUT" | tail -20; break;; esac
  sleep "$INT"
done
