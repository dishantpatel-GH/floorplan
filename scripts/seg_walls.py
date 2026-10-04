#!/usr/bin/env python
"""Semantic class map per photo (ADE20K, SegFormer-B5) for the photo tier's wall selection (D-060).

Why: the photo tier boxes each room from MoGe-2 "wall points" (vertical surfaces in a 1-2 m band). Wardrobes, kitchen
cabinets, shower partitions and vanities are vertical surfaces too, so rooms came out 20-70% too small (sim 2 BHK,
exact GT). A semantic segmenter labels wall pixels directly; the layout fit then keeps only points on walls.

Runs in the separate env `envs/seg` (transformers must not enter the main .venv: I-002), with its own HF cache, and is
called by floorplan/photo/semantic.py as a subprocess. Output: one .npz with a uint8 ADE20K class map per photo at the
inference size (long side --long-side, the photo's own aspect), plus the class names; the caller resizes.

Usage: envs/seg/bin/python scripts/seg_walls.py --list images.txt --out sem.npz [--long-side 640] [--threads N]
       images.txt: one "name<TAB>path" per line
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

DEFAULT_MODEL = "nvidia/segformer-b5-finetuned-ade-640-640"
DEFAULT_REVISION = "739f5d4692954e4a185eac280dec1ba5a7d52f1d"   # pinned; scripts/fetch_weights.py fetches this one
MODEL = os.environ.get("SEG_MODEL", DEFAULT_MODEL)
REVISION = os.environ.get("SEG_REVISION", DEFAULT_REVISION if MODEL == DEFAULT_MODEL else "main")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--long-side", type=int, default=640, help="inference resolution (long side)")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    a = ap.parse_args()
    import torch
    from PIL import Image, ImageOps
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    t0 = time.time()
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = SegformerImageProcessor.from_pretrained(MODEL, revision=REVISION)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL, revision=REVISION).to(dev).eval()
    if dev == "cuda":
        model = model.half()
    names = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    out = {}
    items = [ln.rstrip("\n").split("\t") for ln in open(a.list) if ln.strip()]
    for name, path in items:
        im = Image.open(path)
        im.draft("RGB", (a.long_side * 2, a.long_side * 2))          # fast JPEG decode at reduced size
        im = ImageOps.exif_transpose(im).convert("RGB")
        s = a.long_side / max(im.size)
        im_s = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.BILINEAR)
        x = proc(images=im_s, return_tensors="pt", do_resize=False)["pixel_values"].to(dev)
        if dev == "cuda":
            x = x.half()
        with torch.no_grad():
            logits = model(pixel_values=x).logits                          # (1, C, h/4, w/4)
            logits = torch.nn.functional.interpolate(logits.float(), size=x.shape[-2:], mode="bilinear",
                                                     align_corners=False)
        out[name] = logits.argmax(1)[0].byte().cpu().numpy()
    np.savez_compressed(a.out, names=np.array(names), keys=np.array(list(out.keys())),
                        **{f"m{i}": v for i, v in enumerate(out.values())})
    print(f"[seg] {len(out)} photos, {MODEL}@{REVISION[:8]}, {dev}, {time.time() - t0:.1f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
