"""Release GPU memory held by objects left in the persistent Colab kernel by earlier `colab exec -f` scripts."""
import gc
for _n in [n for n in list(globals()) if not n.startswith("_") and n not in ("gc",)]:
    del globals()[_n]
gc.collect()
import torch
torch.cuda.empty_cache()
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
