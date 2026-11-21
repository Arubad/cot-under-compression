"""Run on the Colab VM via `colab exec -s <session> -f tools/colab/setup_vm.py </dev/null`.

Expects /content/repo.tgz (from `git archive HEAD`). Extracts to /content/cot, installs the
packages Colab lacks, runs the CPU unit tests and records the software stack.
"""
import subprocess, sys

def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    print(f"$ {c}\n{(r.stdout + r.stderr)[-3000:]}", flush=True)
    return r.returncode

sh("rm -rf /content/cot && mkdir -p /content/cot && tar xzf /content/repo.tgz -C /content/cot")
sh(f"{sys.executable} -m pip install -q bitsandbytes nvidia-ml-py pytest 2>&1 | tail -3")
sh("mkdir -p /content/cot/results/env && pip freeze > /content/cot/results/env/pip_freeze.txt && nvidia-smi > /content/cot/results/env/nvidia_smi.txt")
rc = sh("cd /content/cot && timeout 900 python -m pytest -q tests 2>&1 | tail -15")
print("SETUP", "OK" if rc == 0 else f"TESTS FAILED rc={rc}")
