"""Unpack a results archive uploaded to /content/<ARCHIVE> into /content/cot (gives /content/cot/results/...)."""
import os
import subprocess
import tarfile

arc = os.environ.get("ARCHIVE", "/content/I1_baseline_results.tgz")
with tarfile.open(arc) as t:
    t.extractall("/content/cot", filter="data")
print(subprocess.run("find /content/cot/results -maxdepth 3 | head -30", shell=True, capture_output=True, text=True).stdout)
