#!/usr/bin/env python
"""Apply a universal perturbation to one video and show what it does.

Renders a side-by-side clip -- original | protected | perturbation -- and, when the victim
checkpoint is available, reports how the prediction changes.

The perturbation is *not* computed here. It is read from `assets/perturbations/`, which ships
with the repository, or from a file you produced yourself with one of the adapters. That is the
point of a universal perturbation: applying it is one tensor addition.

Examples
--------
    # shipped C-DUP perturbation, visual output only
    PYTHONPATH=common python scripts/demo.py --video clip.avi --method cdup

    # both methods, with prediction change (needs checkpoints/)
    PYTHONPATH=common python scripts/demo.py --video clip.avi --method both

    # a perturbation you generated yourself
    PYTHONPATH=common python scripts/demo.py --video clip.avi \
        --uap results/u3d/u3d_perturbation.npy --method u3d
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "common"))

from dfbench.data import (load_video_clips, sample_frame_indices,          # noqa: E402
                          resize_and_crop)
from dfbench.paths import C3D_CKPT, VIDEO_DIR, load_class_index            # noqa: E402

SHIPPED = {
    "cdup": os.path.join(ROOT, "assets", "perturbations", "cdup_perturbation.npy"),
    "u3d":  os.path.join(ROOT, "assets", "perturbations", "u3d_perturbation.npy"),
}


def decode_clip(path):
    """-> (3,16,112,112) float32 RGB in [0,255], identical to what the victim is fed.

    Uses decord when available (the exact path the benchmark uses) and falls back to OpenCV,
    so the demo runs on a fresh checkout without extra dependencies.
    """
    try:
        return load_video_clips(path)[0]
    except ImportError:
        pass

    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"OpenCV could not open {path}")
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr[:, :, ::-1])                                     # BGR -> RGB
    cap.release()
    if not frames:
        raise SystemExit(f"no frames decoded from {path}")

    frames = np.stack(frames)
    inds = sample_frame_indices(len(frames))
    clip = resize_and_crop(frames[inds])                                   # (16,112,112,3)
    return np.ascontiguousarray(clip.transpose(3, 0, 1, 2), dtype=np.float32)


def load_uap(path):
    """-> (3,16,112,112) float32. U3D stores (16,112,112); broadcast it to three channels."""
    d = np.load(path).astype(np.float32)
    if d.ndim == 3:
        d = np.repeat(d[None], 3, axis=0)
    if d.shape != (3, 16, 112, 112):
        raise SystemExit(f"unexpected perturbation shape {d.shape}, want (3,16,112,112)")
    return d


def to_bgr_frames(clip):
    """(3,T,H,W) RGB float 0-255 -> list of T uint8 BGR frames for OpenCV."""
    x = np.clip(clip, 0, 255).astype(np.uint8).transpose(1, 2, 3, 0)       # (T,H,W,3) RGB
    return [f[:, :, ::-1] for f in x]


def render(panels, titles, out_path, upscale, fps, loops):
    import cv2
    h, w = panels[0][0].shape[:2]
    H, W = h * upscale, w * upscale
    label_h = max(28, int(0.14 * H))
    canvas_w, canvas_h = W * len(panels), H + label_h

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas_w, canvas_h))
    if not vw.isOpened():
        raise SystemExit(f"OpenCV could not open a writer for {out_path}")

    n = len(panels[0])
    for _ in range(loops):
        for t in range(n):
            canvas = np.full((canvas_h, canvas_w, 3), 255, np.uint8)
            for i, frames in enumerate(panels):
                big = cv2.resize(frames[t], (W, H), interpolation=cv2.INTER_NEAREST)
                canvas[label_h:, i * W:(i + 1) * W] = big
                cv2.putText(canvas, titles[i], (i * W + 10, int(label_h * 0.72)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6 * upscale / 4,
                            (20, 20, 20), max(1, upscale // 3), cv2.LINE_AA)
            vw.write(canvas)
    vw.release()
    return out_path


def predict(clip, device):
    """-> (class index, softmax confidence) or None when the checkpoint is absent."""
    if not os.path.exists(C3D_CKPT):
        return None
    import torch
    from dfbench.victim import C3DVictim
    victim = C3DVictim(device=device)
    logits = victim.predict(torch.as_tensor(clip[None], device=victim.device))
    prob = torch.softmax(logits, 1)[0]
    idx = int(prob.argmax())
    return idx, float(prob[idx])


def class_name(idx):
    try:
        _, idx2name = load_class_index()
        return idx2name.get(idx, str(idx))
    except Exception:
        return str(idx)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="any .avi/.mp4; defaults to a UCF-101 clip if present")
    ap.add_argument("--method", choices=["cdup", "u3d", "both"], default="both")
    ap.add_argument("--uap", help="perturbation .npy; defaults to the shipped one")
    ap.add_argument("--out", default=os.path.join(ROOT, "demo_out"))
    ap.add_argument("--amplify", type=int, default=8, help="gain on the perturbation panel")
    ap.add_argument("--upscale", type=int, default=4)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--loops", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    video = a.video
    if video is None:
        default = os.path.join(VIDEO_DIR, "ApplyEyeMakeup", "v_ApplyEyeMakeup_g01_c01.avi")
        if not os.path.exists(default):
            raise SystemExit("no --video given and no UCF-101 clip found; pass --video PATH")
        video = default
    if not os.path.exists(video):
        raise SystemExit(f"video not found: {video}")

    methods = ["cdup", "u3d"] if a.method == "both" else [a.method]
    if a.uap and len(methods) > 1:
        raise SystemExit("--uap applies to a single method; pass --method cdup or --method u3d")

    print(f"video   : {video}")
    clip = decode_clip(video)                                              # (3,16,112,112)
    base = predict(clip, a.device)
    if base is None:
        print("victim  : checkpoint not found -- rendering video only "
              "(run scripts/00_prepare_dataset.sh for predictions)")
    else:
        print(f"clean   : {class_name(base[0])}  ({base[1]*100:.1f} %)")

    for m in methods:
        path = a.uap or SHIPPED[m]
        if not os.path.exists(path):
            raise SystemExit(f"perturbation not found: {path}")
        delta = load_uap(path)
        adv = np.clip(clip + delta, 0, 255)
        applied = adv - clip                                               # after clipping

        print(f"\n[{m}] perturbation : {os.path.relpath(path, ROOT)}")
        print(f"[{m}] L_inf         : {np.abs(applied).max():.3f}")
        print(f"[{m}] mean |delta|  : {np.abs(applied).mean():.3f}")

        if base is not None:
            got = predict(adv, a.device)
            flipped = "ATTACK SUCCEEDS" if got[0] != base[0] else "prediction unchanged"
            print(f"[{m}] adversarial   : {class_name(got[0])}  ({got[1]*100:.1f} %)  -- {flipped}")

        vis = np.clip(applied * a.amplify + 128, 0, 255)
        out = os.path.join(a.out, f"demo_{m}.mp4")
        render([to_bgr_frames(clip), to_bgr_frames(adv), to_bgr_frames(vis)],
               ["original", f"protected ({m.upper()})", f"perturbation x{a.amplify}"],
               out, a.upscale, a.fps, a.loops)
        print(f"[{m}] wrote         : {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
