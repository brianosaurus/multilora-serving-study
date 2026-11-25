import os
if os.environ.get("SGLANG_CLIFF_TRACE_DIR"):
    try:
        import cliff_trace
        cliff_trace.install()
    except Exception:
        pass
