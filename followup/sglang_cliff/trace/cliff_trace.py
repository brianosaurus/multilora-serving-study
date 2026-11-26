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
    if os.environ.get("SGLANG_CLIFF_PHASES") == "1":
        _wrap_phases(S)


# Phase timing (only with SGLANG_CLIFF_PHASES=1; trigger step 4). Per scheduler-loop phase, per 1 s bucket: calls, wall time,
# thread CPU time (wall minus CPU = time the thread was off-CPU: GIL, preemption, I/O or GPU waits), and the scheduler thread's
# voluntary and involuntary context switches. Event "ph": t (bucket start), p = {phase: [calls, wall, cpu]}, vcs, ivcs, qlen.
_PH = {"t": None, "p": {}}


def _ctx():
    v = iv = None
    try:
        for l in open("/proc/thread-self/status"):
            if l.startswith("voluntary_ctxt_switches"): v = int(l.split()[1])
            elif l.startswith("nonvoluntary_ctxt_switches"): iv = int(l.split()[1])
    except Exception:
        pass
    return v, iv


def _ph_flush(sched, now):
    try:
        b = _PH
        if b["t"] is None:
            b["t"], b["ctx"] = int(now), _ctx()
            return
        if int(now) == b["t"]:
            return
        v, iv = _ctx(); v0, iv0 = b["ctx"]
        _w({"e": "ph", "t": b["t"], "p": {k: [c, round(w, 6), round(u, 6)] for k, (c, w, u) in b["p"].items()},
            "vcs": (v - v0) if v is not None and v0 is not None else None, "ivcs": (iv - iv0) if iv is not None and iv0 is not None else None,
            "qlen": len(sched.waiting_queue), "nrun": len(sched.running_batch.reqs)})
        b["t"], b["p"], b["ctx"] = int(now), {}, (v, iv)
    except Exception:
        pass


def _wrap_phases(S):
    def mk(name, o):
        @functools.wraps(o)
        def f(self, *a, **k):
            w0, c0 = time.time(), time.thread_time()
            if name == "ingest_requests":
                _ph_flush(self, w0)
            try:
                return o(self, *a, **k)
            finally:
                try:
                    r = _PH["p"].setdefault(name, [0, 0.0, 0.0])
                    r[0] += 1; r[1] += time.time() - w0; r[2] += time.thread_time() - c0
                except Exception:
                    pass
        return f
    names = ["ingest_requests", "get_next_batch_to_run", "run_batch", "process_batch_result"]
    if os.environ.get("SGLANG_CLIFF_SUBPHASES") == "1":  # trigger step 5: parts of get_next_batch_to_run
        names += ["get_new_batch_prefill", "update_running_batch", "can_schedule_lora_req", "process_pending_chunked_abort",
                  "_process_hicache_events"]
    for name in names:
        if hasattr(S, name):
            setattr(S, name, mk(name, getattr(S, name)))
    _PH["mk"] = mk


def _wrap_sub(cls_names, meth):
    def w(mod):
        if os.environ.get("SGLANG_CLIFF_PHASES") != "1" or os.environ.get("SGLANG_CLIFF_SUBPHASES") != "1":
            return
        for cn in cls_names:
            C = getattr(mod, cn, None)
            if C is None or not hasattr(C, meth):
                continue
            o = getattr(C, meth)

            def f(self, *a, _o=o, _n=f"{cn}.{meth}", **k):
                w0, c0 = time.time(), time.thread_time()
                try:
                    return _o(self, *a, **k)
                finally:
                    try:
                        r = _PH["p"].setdefault(_n, [0, 0.0, 0.0])
                        r[0] += 1; r[1] += time.time() - w0; r[2] += time.thread_time() - c0
                    except Exception:
                        pass
            setattr(C, meth, functools.wraps(o)(f))
    return w


def _multi(*fs):
    def w(mod):
        for f in fs:
            try:
                f(mod)
            except Exception:
                pass
    return w


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


def _wrap_split(mod):  # step 9: time the split's module-level helpers (only with SUBPHASES)
    if os.environ.get("SGLANG_CLIFF_PHASES") != "1" or os.environ.get("SGLANG_CLIFF_SUBPHASES") != "1":
        return
    for name in ("maybe_execute_split_decode", "_make_chunks", "_clone_rows", "_merge"):
        o = getattr(mod, name, None)
        if o is None:
            continue

        def f(*a, _o=o, _n=f"split.{name}", **k):
            w0, c0 = time.time(), time.thread_time()
            try:
                return _o(*a, **k)
            finally:
                try:
                    r = _PH["p"].setdefault(_n, [0, 0.0, 0.0])
                    r[0] += 1; r[1] += time.time() - w0; r[2] += time.thread_time() - c0
                except Exception:
                    pass
        setattr(mod, name, functools.wraps(o)(f))


_TARGETS = {"sglang.srt.managers.scheduler": _wrap_scheduler,
            "sglang.srt.lora.lora_manager": _multi(_wrap_lora, _wrap_sub(["LoRAManager"], "prepare_lora_batch")),
            "sglang.srt.lora.backend.chunked_backend": _wrap_sub(["ChunkedSgmvLoRABackend"], "prepare_lora_batch"),
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner": _multi(
                _wrap_sub(["DecodeCudaGraphRunner"], "execute"), _wrap_sub(["DecodeCudaGraphRunner"], "load_batch")),
            "sglang.srt.model_executor.runner.decode_graph_split": _wrap_split,
            "sglang.srt.managers.tokenizer_manager": _wrap_tokenizer_manager,
            "sglang.srt.managers.schedule_policy": _multi(_wrap_sub(["SchedulePolicy"], "calc_priority"), _wrap_sub(["PrefillAdder"], "add_one_req")),
            "sglang.srt.managers.scheduler_components.request_list": _wrap_sub(["IndexedReqList", "KeyedReqList", "ReqList"], "reorder")}


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
