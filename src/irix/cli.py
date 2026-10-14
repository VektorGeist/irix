"""cli — Irix from the command line. Deterministic, model-free.

  irix scan      IMAGE                      injection screen
  irix elements  IMAGE                      parse UI elements
  irix scene     IMAGE [PREV]               describe a moving-world scene
  irix occupancy IMAGE [PREV]               BEV occupancy grid (+flow if PREV)
  irix world     IMAGE [IMAGE...]           persistent BEV world-model over frames
  irix ttc       PREV IMAGE                 time-to-contact / looming reflex
  irix enroll    IMAGE NAME GALLERY         remember a face
  irix recognize IMAGE GALLERY              recognize faces
  irix gallery   GALLERY                    gallery stats
  irix obj-enroll IMAGE NAME GALLERY        remember an object (salient region)
  irix identify  IMAGE GALLERY              name known objects + describe the rest
  irix play      [--ticks N]                closed-loop PLAY demo on the sim world
  irix bench     [--screens N] [--jsonl F]  reproducible GUI-grounding benchmark
  irix uia       [--hwnd H]                 read the accessibility tree (Windows)
  irix fuse      [IMAGE] [--hwnd H]         hybrid a11y-tree + pixel-CV element map
  irix watch     [--interval S]             remember screen changes to visual memory
  irix remember  IMAGE [--source S]         store a screenshot in visual memory
  irix recall    QUERY [-n N]               search visual memory by text
  irix list      [-n N]                     list recent memory episodes
  irix forget    ID                         delete a memory episode
"""
from __future__ import annotations

import argparse
import json
import sys


def _default_bev(image_path):
    """A sane uncalibrated ground-plane BEV from the image size, so `occupancy`
    works from the CLI without calibration flags (dash-cam defaults). For a real
    metric grid, calibrate `bev.BEV.from_camera` with the true camera geometry."""
    import cv2
    from . import bev
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    return bev.BEV.from_camera((w, h), height_m=1.5, pitch_deg=12.0,
                               hfov_deg=60.0, range_m=30, width_m=20, px_per_m=10)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="irix", description="Deterministic model-free eyes")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan"); s.add_argument("image")
    s = sub.add_parser("elements"); s.add_argument("image")
    s = sub.add_parser("scene"); s.add_argument("image"); s.add_argument("prev", nargs="?")
    s = sub.add_parser("occupancy")
    s.add_argument("image"); s.add_argument("prev", nargs="?")
    s.add_argument("--cell-m", type=float, default=0.5)
    s = sub.add_parser("world")
    s.add_argument("images", nargs="+")
    s.add_argument("--cell-m", type=float, default=0.5)
    s.add_argument("--save", default=None, help="persist the map (.npz)")
    s.add_argument("--load", default=None, help="continue a saved map (.npz)")
    s = sub.add_parser("ttc"); s.add_argument("prev"); s.add_argument("image")
    s.add_argument("--dt", type=float, default=None)
    s = sub.add_parser("enroll")
    s.add_argument("image"); s.add_argument("name"); s.add_argument("gallery")
    s = sub.add_parser("recognize"); s.add_argument("image"); s.add_argument("gallery")
    s = sub.add_parser("gallery"); s.add_argument("gallery")
    s = sub.add_parser("obj-enroll")
    s.add_argument("image"); s.add_argument("name"); s.add_argument("gallery")
    s = sub.add_parser("identify")
    s.add_argument("image"); s.add_argument("gallery")
    s.add_argument("--auto-enroll", default=None, metavar="PREFIX",
                   help="remember unknown objects on sight under PREFIX<n>")
    s = sub.add_parser("play")
    s.add_argument("--ticks", type=int, default=30)
    s = sub.add_parser("bench")
    s.add_argument("--screens", type=int, default=12)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--no-blur", action="store_true")
    s.add_argument("--jsonl", default=None,
                   help="score a ScreenSpot-style JSONL instead of synthetic")
    s = sub.add_parser("uia")
    s.add_argument("--hwnd", type=int, default=None)
    s.add_argument("--interactive", action="store_true")
    s = sub.add_parser("fuse")
    s.add_argument("image", nargs="?")
    s.add_argument("--hwnd", type=int, default=None)
    s.add_argument("--interactive", action="store_true")
    s = sub.add_parser("watch")
    s.add_argument("--interval", type=float, default=5.0)
    s.add_argument("--threshold", type=float, default=8.0)
    s.add_argument("--region", default=None)
    s.add_argument("--memory-dir", default="~/.irix/memory")
    s.add_argument("--frames", type=int, default=None)
    s.add_argument("--quiet", action="store_true")
    s = sub.add_parser("remember")
    s.add_argument("image")
    s.add_argument("--source", default="cli")
    s.add_argument("--caption", default=None)
    s.add_argument("--memory-dir", default="~/.irix/memory")
    s = sub.add_parser("recall")
    s.add_argument("query")
    s.add_argument("-n", type=int, default=5)
    s.add_argument("--memory-dir", default="~/.irix/memory")
    s = sub.add_parser("list")
    s.add_argument("-n", type=int, default=10)
    s.add_argument("--memory-dir", default="~/.irix/memory")
    s = sub.add_parser("forget")
    s.add_argument("id", type=int)
    s.add_argument("--memory-dir", default="~/.irix/memory")

    a = p.parse_args(argv)

    if a.cmd == "scan":
        from . import inject
        out = inject.scan(a.image)
    elif a.cmd == "elements":
        from . import elements
        out = elements.parse(a.image)
    elif a.cmd == "scene":
        from . import scene
        out = scene.describe_scene(a.image, a.prev)
    elif a.cmd == "occupancy":
        from . import bev, occupancy
        b = _default_bev(a.image)
        if b is None:
            out = {"error": "cannot read image"}
        elif a.prev:
            out = occupancy.occupancy_flow(a.prev, a.image, b, cell_m=a.cell_m)
        else:
            out = occupancy.occupancy_grid(a.image, b, cell_m=a.cell_m).summary()
    elif a.cmd == "world":
        from . import worldmodel
        b = _default_bev(a.images[0])
        if b is None:
            out = {"error": "cannot read image"}
        else:
            wm = worldmodel.WorldModel.load(a.load, b) if a.load \
                else worldmodel.WorldModel(b, cell_m=a.cell_m)
            for img in a.images:
                wm.observe(img)
            out = {**wm.summary(), "snapshot": wm.snapshot().summary()}
            if a.save:
                out["saved"] = wm.save(a.save)
    elif a.cmd == "ttc":
        from . import occupancy
        out = occupancy.ttc(a.prev, a.image, dt=a.dt)
    elif a.cmd == "enroll":
        from . import face
        out = face.Gallery(a.gallery).enroll(a.image, a.name)
    elif a.cmd == "recognize":
        from . import face
        out = face.Gallery(a.gallery).recognize(a.image)
    elif a.cmd == "gallery":
        from . import face
        out = face.Gallery(a.gallery).stats()
    elif a.cmd == "obj-enroll":
        from . import objects
        out = objects.ObjectGallery(a.gallery).enroll(a.image, a.name)
    elif a.cmd == "identify":
        from . import objects
        out = objects.ObjectGallery(a.gallery).identify(
            a.image, auto_enroll_unknown=a.auto_enroll)
    elif a.cmd == "play":
        from . import play
        out = play.run(ticks=a.ticks, verbose=True)
        out.pop("log", None)
    elif a.cmd == "bench":
        from . import bench
        if a.jsonl:
            out = bench.evaluate(bench.load_jsonl(a.jsonl))
            out["source"] = a.jsonl
        else:
            out = bench.run(n_screens=a.screens, seed=a.seed, blur=not a.no_blur)
    elif a.cmd == "uia":
        from . import uia
        out = uia.elements(hwnd=a.hwnd, interactive_only=a.interactive)
    elif a.cmd == "fuse":
        from . import fuse
        out = fuse.parse(a.image, hwnd=a.hwnd, interactive_only=a.interactive)
    elif a.cmd == "watch":
        from . import watch
        region = watch._parse_region(a.region) if a.region else None
        out = watch.run(interval=a.interval, threshold=a.threshold,
                        region=region, memory_dir=a.memory_dir,
                        max_frames=a.frames, quiet=a.quiet)
    elif a.cmd == "remember":
        from . import memory
        mem = memory.MemoryStore(a.memory_dir)
        out = mem.remember(a.image, source=a.source, caption=a.caption)
    elif a.cmd == "recall":
        from . import memory
        mem = memory.MemoryStore(a.memory_dir)
        out = {"query": a.query, "episodes": mem.recall(a.query, n=a.n)}
    elif a.cmd == "list":
        from . import memory
        mem = memory.MemoryStore(a.memory_dir)
        out = {"episodes": mem.list(n=a.n)}
    elif a.cmd == "forget":
        from . import memory
        mem = memory.MemoryStore(a.memory_dir)
        out = {"id": a.id, "forgotten": mem.forget(a.id)}
    else:
        return 2

    json.dump(out, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
