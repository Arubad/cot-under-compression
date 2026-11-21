# Colab runbook

How experiments are run from this machine with the `colab` CLI (v0.8.0, authenticated). Verified 2026-11-21.

## Session lifecycle
```bash
colab new -s cot-t4 --gpu T4          # allocate (prints "Session READY")
colab status -s cot-t4                # hardware + IDLE/BUSY
colab usage                           # compute-unit balance and burn rate
colab stop -s cot-t4                  # ALWAYS release when done
```

## Ship code (the VM has no git credentials; the repo is public but we ship HEAD exactly)
```bash
tools/colab/ship.sh cot-t4                                     # git archive HEAD + GIT_COMMIT file -> /content/repo.tgz (refuses a dirty tree)
colab exec -s cot-t4 -f tools/colab/setup_vm.py </dev/null     # extract to /content/cot, pip install bitsandbytes nvidia-ml-py pytest, freeze env, run tests
```
Ship a committed HEAD, never a dirty tree. The run log records the commit hash (from the shipped `GIT_COMMIT` file, since the VM copy has no `.git`), and the code on the VM must match it.

## Launch a long job (detached, survives local disconnects)
```bash
colab exec -s cot-t4 -f tools/colab/launch.py </dev/null       # nohup python -u -m src.run <cmd> > /content/<id>.log
colab exec -s cot-t4 -f tools/colab/tail.py   </dev/null       # progress
```

## Bring results home (the VM is ephemeral)
```bash
# on VM: tar czf /content/<id>_results.tgz -C /content/cot results
colab download -s cot-t4 /content/<id>_results.tgz results/archive/
```
`results/` is gitignored. Raw artefacts live in `results/archive/` locally, and the numbers that matter are copied into the experiment page and `EXPERIMENT_LOG.md`.

## Gotchas observed
- **Always redirect stdin** (`</dev/null`) for `colab exec`, otherwise it can block.
- **Don't chain `colab upload … && colab exec …` in one shell command.** In this environment the `exec` hung both times. Run them as separate commands.
- No SSH key exists locally, so use `exec` with Python/`!` lines instead of `colab ssh`.
- Colab's stack is Python 3.13, torch 2.14, transformers 5.22 and **NumPy 2** (removed `np.trapz`). See D-003.
- CPU unit tests run locally in the repo's `.venv` (torch 2.15, transformers 5.22.2): `.venv/bin/python -m pytest -q tests`. Don't run tests on the GPU VM while a measured job is running, because CPU contention perturbs its timing and energy.
- Only one measured job per GPU at a time, for the same reason.
