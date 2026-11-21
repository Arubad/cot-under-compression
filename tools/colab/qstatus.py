"""Queue progress: status JSON, last log lines, GPU state.  Env QUEUE_ID (default queue)."""
import os
import subprocess

qid = os.environ.get("QUEUE_ID", "queue")
st = f"/content/logs/{qid}.status"
print(open(st).read() if os.path.exists(st) else "no status yet")
log = f"/content/logs/{qid}.log"
if os.path.exists(log):
    out = subprocess.run(f"tail -c 6000 {log} | tr '\\r' '\\n' | grep -v '^\\s*$' | tail -n {os.environ.get('N', '25')}", shell=True, capture_output=True, text=True).stdout
    print(out)
print(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm", "--format=csv"], capture_output=True, text=True).stdout)
