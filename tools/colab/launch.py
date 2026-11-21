"""Start a detached job on the Colab VM.

  colab exec -s cot-t4 --env JOB_ID=I1_baseline --env "JOB_ARGS=baseline --config configs/baseline.yaml" -f tools/colab/launch.py </dev/null

Output goes to /content/logs/<JOB_ID>.log; the job survives the exec returning.
"""
import os, shlex, subprocess

job, args = os.environ["JOB_ID"], os.environ["JOB_ARGS"]
os.makedirs("/content/logs", exist_ok=True)
log = f"/content/logs/{job}.log"
argv = ["python", "-u", "-m", "src.run", *shlex.split(args)]
with open(log, "ab") as f:
    p = subprocess.Popen(argv, cwd="/content/cot", stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
print(f"launched {job} pid={p.pid} -> {log}: {' '.join(argv)}")
