"""Generate the Kaggle notebooks in notebooks/ from one spec (run: python tools/make_notebooks.py).

Each notebook: clone the repo, install requirements, optionally import a previous
notebook's output as the cache, run one experiment through ``python -m src.run``,
then leave results/ in /kaggle/working so it can be attached as a dataset input
to the next notebook.  Kaggle settings to use: Accelerator = GPU T4 x2 (or P100),
Internet = On, Persistence = Files only.
"""
from __future__ import annotations

import os

import json

REPO = "https://github.com/Arubad/cot-under-compression.git"
NAME = "cot-under-compression"

SETUP = f"""# --- setup (identical in every notebook) -------------------------------------------------
import os, subprocess, sys
WORK = '/kaggle/working' if os.path.exists('/kaggle/working') else os.getcwd()
os.chdir(WORK)
if not os.path.exists('{NAME}'):
    subprocess.run(['git', 'clone', '--depth', '1', '{REPO}'], check=True)
os.chdir('{NAME}')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
os.environ['RESULTS_ROOT'] = os.path.join(WORK, 'results')
# To reuse caches from an earlier notebook: Add Input -> that notebook's output, then set:
PREV = ''   # e.g. '/kaggle/input/cot-i1-baseline/results'
if PREV:
    os.environ['PREV_RESULTS'] = PREV
import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
"""

NOTEBOOKS = [
    ("00_smoke_test", "Smoke test", "Every code path on 30 problems with 64-token answers (fp16 cache, zero-change control, int4 KV per-token and per-channel, top-k 50%, single-head leverage, int8 weights, HF quantized-cache backends). **Run this first.** Prints which backends work on this GPU. Nothing it produces is a result.",
     ["!python -m src.run smoke"]),
    ("01_I1_baseline_control", "I1: FP16 baseline, zero-change control, difficulty labels, head statistics", "Builds the 300-problem stratified subset (commit `configs/gsm8k_300_stratified.json` afterwards), runs the bit-exact control, caches fp16 greedy generations with top-16 logprobs, labels difficulty by pass@4, records per-head attention entropy, fits the latency model. ~3 T4-hours. Save the output as a Kaggle dataset; every later notebook imports it.",
     ["!python -m src.run make_subset --config configs/baseline.yaml", "!python -m src.run baseline --config configs/baseline.yaml",
      "import pandas as pd, json\nprint(json.load(open(os.path.join(os.environ['RESULTS_ROOT'],'cache','control.json'))))\nprint(pd.read_csv(os.path.join(os.environ['RESULTS_ROOT'],'cache','head_stats.csv')).describe())\nprint(pd.read_parquet(os.path.join(os.environ['RESULTS_ROOT'],'cache','difficulty_gsm8k.parquet'))['stratum'].value_counts())"]),
    ("02_I2_weights", "I2: weight quantisation (int8, nf4)", "bitsandbytes int8 / nf4 weights; measured energy. Minimum: nf4 only.",
     ["!python -m src.run control --config configs/baseline.yaml", "!python -m src.run sweep --config configs/weights_sweep.yaml --minimum --skip-existing", "!python -m src.run analyze"]),
    ("03_I3_kv", "I3: KV-cache quantisation (H1)", "FakeQuantCache at 8/4/2 bits, per-token and per-channel; prefill at fp16. Energy for these cells is simulated from the latency model (labelled). Minimum: int8 and int4 per-token.",
     ["!python -m src.run control --config configs/baseline.yaml", "!python -m src.run sweep --config configs/kv_sweep.yaml --minimum --skip-existing", "!python -m src.run analyze"]),
    ("04_I4_sparse", "I4: attention sparsification (all layers)", "sliding_window, sink_window, topk_heavy_hitter, leverage at 50% / 25% keys kept. Speed and energy SIMULATED. Minimum: sink_window and top-k at 50%.",
     ["!python -m src.run control --config configs/baseline.yaml", "!python -m src.run sweep --config configs/sparse_sweep.yaml --minimum --skip-existing", "!python -m src.run analyze"]),
    ("05_I5_perhead", "I5: per-head sweep on a 6-layer subset (H2)", "Single-head 50% top-k sparsification on every head of 6 layers chosen from the recorded entropies. ~70 cells x 300 problems (~8 T4-hours); `--minimum` runs 4 heads per layer.",
     ["!python -m src.run perhead --config configs/perhead_sweep.yaml --minimum --skip-existing", "!python -m src.run analyze"]),
    ("06_I6_I7_frontier_selector", "I6 controls + I7 frontier and selector (H3)", "ARC short-reasoning control, selector table from every finished cell, energy repeats on the measured headline configs, Pareto frontier on energy per correct answer.",
     ["!python -m src.run sweep --config configs/controls_math_arc.yaml --minimum --skip-existing", "!python -m src.run frontier --config configs/frontier.yaml", "!python -m src.run analyze"]),
    ("07_analysis_and_figures", "Analysis and figures", "Collects every `metrics.json`, applies the pre-registered tests mechanically and draws Figures 1-4 into `results/figures/`.",
     ["!python -m src.run analyze", "import json; print(json.dumps(json.load(open(os.path.join(os.environ['RESULTS_ROOT'],'figures','hypothesis_report.json'))), indent=2)[:4000])",
      "from IPython.display import Image, display\nimport glob\nfor f in sorted(glob.glob(os.path.join(os.environ['RESULTS_ROOT'],'figures','*.png'))): display(Image(f))"]),
]


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def _code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": text, "outputs": [], "execution_count": None}


def build(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    for fname, title, desc, cells in NOTEBOOKS:
        nb_cells = [_md(f"# {title}\n\n{desc}\n\nRepo: {REPO}  \nKaggle: Accelerator GPU T4 x2, Internet On. Results land in `/kaggle/working/results`."), _code(SETUP)]
        nb_cells += [_code(c) for c in cells]
        nb_cells.append(_code("# keep the results as this notebook's output (attach as a dataset to the next notebook)\n!du -sh $RESULTS_ROOT && ls $RESULTS_ROOT"))
        nb = {"cells": nb_cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
        with open(os.path.join(outdir, f"{fname}.ipynb"), "w") as f:
            json.dump(nb, f, indent=1)
        print("wrote", fname)


if __name__ == "__main__":
    build(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks"))
