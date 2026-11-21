"""Show the end of a job log, dropping tqdm noise: `colab exec -s cot-t4 --env JOB_ID=<id> -f tools/colab/tail.py </dev/null`."""
import os, re, subprocess

job = os.environ.get("JOB_ID", "")
log = f"/content/logs/{job}.log" if job else "/content/smoke.log"
txt = open(log, errors="replace").read() if os.path.exists(log) else f"(no log at {log})"
lines = [l for l in re.split(r"[\r\n]", txt) if l.strip() and not re.search(r"\d+/\d+ \[.*it/s\]$", l)]
print("\n".join(lines[-int(os.environ.get("N", 25)):]))
print("---", subprocess.run("pgrep -af 'src.run' | grep -v pgrep || echo 'no src.run process'", shell=True, capture_output=True, text=True).stdout.strip())
