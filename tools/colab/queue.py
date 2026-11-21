"""Run a queue of `python -m src.run` commands as ONE detached process on the Colab VM.

  colab exec -s cot-t4 --env QUEUE_ID=q1 --env "QUEUE=control --config configs/baseline.yaml;;sweep --config ..." -f tools/colab/queue.py </dev/null

Commands are separated by ';;'.  The first command is a gate: if it fails (the zero-change control),
the queue stops.  Later failures are logged and the queue moves on.  After every command the results
are repacked into /content/<QUEUE_ID>_results.tgz, so `colab download` always has the latest finished
cells even if the free-tier VM is reclaimed mid-queue.  Status: /content/logs/<QUEUE_ID>.status.
"""
import os
import subprocess
import sys

qid = os.environ.get("QUEUE_ID", "queue")
cmds = [c.strip() for c in os.environ["QUEUE"].split(";;") if c.strip()]
os.makedirs("/content/logs", exist_ok=True)
runner = f"/content/logs/{qid}_runner.py"
with open(runner, "w") as f:
    f.write(f'''
import os, shlex, subprocess, tarfile, time, json
cmds = {cmds!r}
qid = {qid!r}
log = f"/content/logs/{{qid}}.log"
status = f"/content/logs/{{qid}}.status"
def pack():
    tmp = f"/content/{{qid}}_results.tgz.tmp"
    with tarfile.open(tmp, "w:gz") as t:
        t.add("/content/cot/results", arcname="results")
        t.add(log, arcname=f"logs/{{qid}}.log")
    os.replace(tmp, f"/content/{{qid}}_results.tgz")
state = []
for i, c in enumerate(cmds):
    t0 = time.time()
    with open(status, "w") as s:
        json.dump({{"running": c, "index": i, "n": len(cmds), "done": state}}, s)
    with open(log, "ab") as f:
        f.write(f"\\n##### [{{time.strftime('%H:%M:%S')}}] START {{i+1}}/{{len(cmds)}}: {{c}}\\n".encode())
        f.flush()
        rc = subprocess.call(["python", "-u", "-m", "src.run", *shlex.split(c)], cwd="/content/cot", stdout=f, stderr=subprocess.STDOUT)
        f.write(f"\\n##### END rc={{rc}} after {{time.time()-t0:.0f}} s: {{c}}\\n".encode())
    state.append({{"cmd": c, "rc": rc, "s": round(time.time() - t0)}})
    pack()
    if rc != 0 and i == 0:
        break
with open(status, "w") as s:
    json.dump({{"running": None, "finished": True, "done": state}}, s)
''')
p = subprocess.Popen([sys.executable, "-u", runner], cwd="/content/cot", stdout=open(f"/content/logs/{qid}_runner.out", "ab"), stderr=subprocess.STDOUT, start_new_session=True)
print(f"queue {qid} pid={p.pid}: {len(cmds)} commands")
for c in cmds:
    print("  ", c)
