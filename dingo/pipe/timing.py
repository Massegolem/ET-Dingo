import json
import os
import time
from contextlib import contextmanager

try:
    import torch
except ImportError:
    torch = None


def _sync():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


class StageTimer:
    def __init__(self, stage):
        self.stage = stage
        self.segments = {}

    @contextmanager
    def segment(self, name):
        _sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync()
            self.segments[name] = self.segments.get(name, 0.0) + (
                time.perf_counter() - t0
            )

    def write(self, outdir, label, **extra):
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f"{label}_timing.json")
        data = {
            "stage": self.stage,
            "label": label,
            "total_seconds": sum(self.segments.values()),
            "segments_seconds": self.segments,
            **extra,
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
        return path