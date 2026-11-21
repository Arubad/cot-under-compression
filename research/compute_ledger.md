# Compute ledger

Budget: minimum version ~18 T4-hours, full version ~100 (README §5). Wall-clock is measured from launch to results downloaded. Sessions left idle count too.

| Date | Run ID | Platform | GPU | Wall-clock (h) | Cumulative (h) | Notes |
|---|---|---|---|---|---|---|
| 2026-11-09 | SMOKE-01 | Kaggle | T4 ×2 | < 0.1 | 0.1 | fp16 NaNs |
| 2026-11-09 | SMOKE-02 | Kaggle | T4 ×2 | < 0.1 | 0.2 | sparse mask bug |
| 2026-11-21 | SMOKE-03 | Colab | T4 | 0.15 | 0.35 | first Colab session `cot-t4`; control passed, crashed in power summary (`np.trapz`) |
| 2026-11-21 | SMOKE-04 | Colab | T4 | 0.25 | 0.6 | rerun after NumPy-2 fix; exposed D-004/D-005 |
| 2026-11-21 | I1_baseline | Colab | T4 | 1.85 | 2.45 | commit `30249b4`; 08:16–10:03 UTC plus download |
| 2026-11-21 | LAT-01 | Colab | T4 | _running_ | | latency micro-benchmark at `be82d77` |
| 2026-11-21 | (debug) | Colab | CPU | 0.05 | — | `cot-cpu` session, created then stopped; tests ran locally in `.venv` instead |
