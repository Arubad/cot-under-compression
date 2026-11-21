"""Show the end of a job log: `colab exec -s cot-t4 --env JOB_ID=<id> [--env N=25] -f tools/colab/tail.py </dev/null`.

tqdm progress lines are collapsed (only the latest is shown) so real messages are not drowned out.
"""
import os, re, subprocess

job = os.environ.get("JOB_ID", "")
log = f"/content/logs/{job}.log" if job else "/content/smoke.log"
txt = open(log, errors="replace").read() if os.path.exists(log) else f"(no log at {log})"
lines = [l for l in re.split(r"[\r\n]", txt) if l.strip()]
is_bar = lambda l: re.search(r"\|\s*\d+/\d+ \[", l) is not None
msgs = [l for l in lines if not is_bar(l)]
bars = [l for l in lines if is_bar(l)]
print("\n".join(msgs[-int(os.environ.get("N", 25)):]))
if bars:
    print("progress:", bars[-1].strip())
print("---", subprocess.run("pgrep -af 'src.run' | grep -v pgrep || echo 'no src.run process'", shell=True, capture_output=True, text=True).stdout.strip())
