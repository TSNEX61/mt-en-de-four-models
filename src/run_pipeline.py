"""Run every step sequentially (so timings are not distorted by parallel jobs) and log each one."""
import ctypes
import subprocess
import sys
import time

from common import RESULTS

STEPS = [
    ["train_transformer.py", "--n_train", "12000", "--epochs", "5"],
    ["probe_transformer.py"],
    ["finetune_mt5.py", "--n_train", "1500", "--max_minutes", "20"],
    ["gemma_lora.py", "--n_train", "600", "--batch", "2", "--max_minutes", "20"],
    ["llama_prompting.py", "--shots", "3"],
    ["compare.py"],
]

if sys.platform == "win32":  # keep the laptop awake while the pipeline runs
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

only = sys.argv[1:]  # optional: run only the named scripts
for step in STEPS:
    if only and step[0] not in only:
        continue
    log = RESULTS / f"log_{step[0].removesuffix('.py')}.txt"
    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8") as f:
        code = subprocess.call([sys.executable, "-u", *step], stdout=f, stderr=subprocess.STDOUT)
    print(f"{step[0]}: exit {code} after {time.perf_counter() - t0:.0f}s", flush=True)
    if code != 0:
        sys.exit(code)
