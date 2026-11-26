"""Log-only tracing for the SGLang cliff study (Phase 2). No SGLang file is edited: this module is put on PYTHONPATH
(through sitecustomize.py) and wraps three methods when their modules are imported. It changes no argument, return value or
control flow; any error in the wrapper is swallowed. Output: JSON lines in $SGLANG_CLIFF_TRACE_DIR/trace_<pid>.jsonl.
Events (t = time.time()):
  arrive : request entered the waiting queue (rid, lora, resident = adapter already in the memory pool, running = adapter in
           the running batch, qlen = waiting queue length, nrun = running batch size)
  batch  : Scheduler.run_batch call (mode, n requests, distinct adapters, dur = CPU time in the call, gap = time since the
           previous call ended; rids for prefill batches)
  load   : LoRAManager.fetch_new_loras call that changed the pool (loads, evicts, dur = CPU time in the call)
  recv   : TokenizerManager._init_req_state, i.e. the HTTP front end has the request and has normalized it (rid)
  sent   : TokenizerManager._send_one_request entry, i.e. tokenized and about to be sent to the scheduler (rid)
(recv and sent are written by the front-end process, arrive/batch/load by the scheduler process; both use time.time().)
"""
import functools, importlib.abc, importlib.machinery, json, os, sys, time

_D = os.environ.get("SGLANG_CLIFF_TRACE_DIR")
_f = None
_last_end = [None]


def _w(ev):
    global _f
    try:
        if _f is None:
            os.makedirs(_D, exist_ok=True)
            _f = open(os.path.join(_D, f"trace_{os.getpid()}.jsonl"), "a", buffering=1)
        _f.write(json.dumps(ev, separators=(",", ":")) + "\n")
    except Exception:
        pass


def _lm(sched):
    try:
        return sched.tp_worker.model_runner.lora_manager
    except Exception:
        return None


def _wrap_scheduler(mod):
    S = mod.Scheduler
    o_add, o_run = S._add_request_to_queue, S.run_batch

    @functools.wraps(o_add)
    def add(self, req, *a, **k):
        try:
            if not (a and a[0]) and not k.get("is_retracted"):
                lm = _lm(self)
                lora = getattr(req, "lora_id", None)
                resident = running = None
                if lm is not None:
                    resident = lora in lm.memory_pool.uid_to_buffer_id
                    running = lora in {r.lora_id for r in self.running_batch.reqs}
                _w({"e": "arrive", "t": time.time(), "rid": req.rid, "lora": lora, "resident": resident, "running": running,
                    "qlen": len(self.waiting_queue), "nrun": len(self.running_batch.reqs)})
        except Exception:
            pass
        return o_add(self, req, *a, **k)

    @functools.wraps(o_run)
    def run(self, batch, *a, **k):
        t0 = time.time()
        ev = None
        try:
            reqs = batch.reqs
            mode = str(getattr(batch.forward_mode, "name", batch.forward_mode))
            ev = {"e": "batch", "t": t0, "mode": mode, "n": len(reqs), "adapters": len({r.lora_id for r in reqs}),
                  "gap": (t0 - _last_end[0]) if _last_end[0] else None}
            if "EXTEND" in mode.upper():
                ev["rids"] = [r.rid for r in reqs]
        except Exception:
            pass
        try:
            return o_run(self, batch, *a, **k)
        finally:
            t1 = time.time()
            _last_end[0] = t1
            if ev is not None:
                ev["dur"] = t1 - t0
                _w(ev)

    S._add_request_to_queue, S.run_batch = add, run


def _wrap_lora(mod):
    M = mod.LoRAManager
    o = M.fetch_new_loras

    @functools.wraps(o)
    def fetch(self, *a, **k):
        t0 = time.time()
        try:
            before = set(self.memory_pool.uid_to_buffer_id)
        except Exception:
            before = None
        try:
            return o(self, *a, **k)
        finally:
            try:
                if before is not None:
                    after = set(self.memory_pool.uid_to_buffer_id)
                    loads, ev_ = sorted(x for x in after - before if x), sorted(x for x in before - after if x)
                    if after != before:
                        _w({"e": "load", "t": t0, "loads": loads, "evicts": ev_, "dur": time.time() - t0})
            except Exception:
                pass

    M.fetch_new_loras = fetch


def _wrap_tokenizer_manager(mod):
    T = mod.TokenizerManager
    o_init, o_send = T._init_req_state, T._send_one_request

    @functools.wraps(o_init)
    def init(self, obj, *a, **k):
        try:
            if getattr(obj, "is_single", True):
                _w({"e": "recv", "t": time.time(), "rid": obj.rid})
        except Exception:
            pass
        return o_init(self, obj, *a, **k)

    @functools.wraps(o_send)
    async def send(self, tokenized_obj, *a, **k):
        try:
            _w({"e": "sent", "t": time.time(), "rid": tokenized_obj.rid})
        except Exception:
            pass
        return await o_send(self, tokenized_obj, *a, **k)

    T._init_req_state, T._send_one_request = init, send


_TARGETS = {"sglang.srt.managers.scheduler": _wrap_scheduler, "sglang.srt.lora.lora_manager": _wrap_lora,
            "sglang.srt.managers.tokenizer_manager": _wrap_tokenizer_manager}


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        loader, wrap = spec.loader, _TARGETS[name]
        orig = loader.exec_module

        def exec_module(module):
            orig(module)
            try:
                wrap(module)
            except Exception:
                pass

        loader.exec_module = exec_module
        return spec


def install():
    if _D:
        sys.meta_path.insert(0, _Finder())
