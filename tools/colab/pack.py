"""Pack /content/cot/results (+ job log) into /content/<JOB_ID>_results.tgz for colab download."""
import os, subprocess, tarfile
job = os.environ.get("JOB_ID", "job")
out = f"/content/{job}_results.tgz"
with tarfile.open(out, "w:gz") as t:
    t.add("/content/cot/results", arcname="results")
    log = f"/content/logs/{job}.log"
    if os.path.exists(log):
        t.add(log, arcname=f"logs/{job}.log")
print(subprocess.run(["du", "-sh", out], capture_output=True, text=True).stdout)
print(subprocess.run("find /content/cot/results -type f | head -60", shell=True, capture_output=True, text=True).stdout)
