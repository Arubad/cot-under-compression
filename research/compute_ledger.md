# Compute ledger

Budget: minimum version ~18 T4-hours, full version ~100 (README §5). Wall-clock is measured from launch to results downloaded. Sessions left idle count too.

| Date | Run ID | Platform | GPU | Wall-clock (h) | Cumulative (h) | Notes |
|---|---|---|---|---|---|---|
| 2026-11-09 | SMOKE-01 | Kaggle | T4 ×2 | < 0.1 | 0.1 | fp16 NaNs |
| 2026-11-09 | SMOKE-02 | Kaggle | T4 ×2 | < 0.1 | 0.2 | sparse mask bug |
| 2026-11-21 | SMOKE-03 | Colab | T4 | _running_ | | first Colab session `cot-t4`; control passed, crashed in power summary (`np.trapz`) |
| 2026-11-21 | SMOKE-04 | Colab | T4 | _running_ | | rerun after NumPy-2 fix |
