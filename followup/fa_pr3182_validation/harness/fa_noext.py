# pytest plugin / import shim: register the flash_attn package from FA_SRC without running its __init__,
# which imports the compiled CUDA extension (not built here). The cross-entropy code under test is pure Triton.
import os, sys, types
src = os.environ["FA_SRC"]
pkg = types.ModuleType("flash_attn")
pkg.__path__ = [os.path.join(src, "flash_attn")]
pkg.__file__ = os.path.join(src, "flash_attn", "__init__.py")
sys.modules["flash_attn"] = pkg

# --- per-test peak GPU memory recorder (pytest hooks) ---
import csv, time
def pytest_runtest_setup(item):
    import torch
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    item._t0 = time.time()
def pytest_runtest_logreport(report):
    pass
def pytest_runtest_teardown(item):
    import torch
    torch.cuda.synchronize()
    out = os.environ.get("FA_MEMCSV")
    if out:
        with open(out, "a", newline="") as f:
            csv.writer(f).writerow([item.nodeid, round(time.time()-item._t0,3), torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()])
