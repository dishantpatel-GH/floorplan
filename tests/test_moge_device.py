"""MoGeRunner(device="cpu") must not touch the GPU: it used to call wait_for_gpu, which fails with "No CUDA GPUs are
available" when CUDA is hidden. The model and its weights are replaced by stubs, so nothing is downloaded or loaded.

Run: env -u PYTHONPATH .venv/bin/python -m pytest tests/test_moge_device.py -q
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _Model:
    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self


def test_cpu_device_does_not_wait_for_gpu(monkeypatch):
    moge_v2 = pytest.importorskip("moge.model.v2")
    import huggingface_hub
    import floorplan.photo.recon as recon
    from floorplan.photo.depth import MoGeRunner

    def no_gpu(*args, **kwargs):
        raise AssertionError("wait_for_gpu called for a cpu device")

    monkeypatch.setattr(recon, "wait_for_gpu", no_gpu)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *args, **kwargs: "model.pt")
    monkeypatch.setattr(moge_v2.MoGeModel, "from_pretrained", classmethod(lambda cls, path: _Model()))
    runner = MoGeRunner(device="cpu")
    assert runner.device == "cpu" and runner.model.device == "cpu"
