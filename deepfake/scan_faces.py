"""Step 1 - face scan of UCF-101 split 1 with SimSwap's own detector.

A deepfake (SimSwap) can only be made from a video in which its detector finds a face, so the
train and test lists are scanned first:
  * every video: K evenly spaced frames -> best face (score, bbox, landmarks, align matrix M)
  * train split: frames with a face are also stored (raw uint8) for UAP optimisation

    python -m deepfake.scan_faces --split test
    python -m deepfake.scan_faces --split train --save-frames

Outputs: cache/scan/<split>.jsonl, cache/frames/<split>_frames.u8 + <split>_frames.jsonl
"""

import argparse
import json
import time

import cv2
import numpy as np

from deepfake.detpool import spawn_pool
from deepfake.paths import CACHE, UCF_ROOT
from deepfake.ucf import split_videos

H, W = 240, 320
_aligner = None


def _init(threads: int):
    global _aligner
    cv2.setNumThreads(1)
    from deepfake.face import FaceAligner, limit_onnx_threads
    limit_onnx_threads(threads)
    _aligner = FaceAligner()


def _scan(args):
    rel, k, keep_frames = args
    cap = cv2.VideoCapture(str(UCF_ROOT / rel))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    want = sorted(set(np.linspace(0, max(n - 1, 0), k + 2)[1:-1].round().astype(int).tolist())) if n > 0 else []
    frames, idx = {}, 0
    while want and idx <= want[-1]:
        ok, f = cap.read()
        if not ok:
            break
        if idx in want:
            frames[idx] = f
        idx += 1
    cap.release()
    rec = {"video": rel, "n_frames": n, "fps": fps, "samples": []}
    kept = []
    for t in want:
        f = frames.get(t)
        if f is None:
            rec["samples"].append({"t": t, "face": None, "error": "decode"})
            continue
        rec["hw"] = list(f.shape[:2])
        r = _aligner.detect_full(f)
        if r is None:
            rec["samples"].append({"t": t, "face": None})
            continue
        s = {"t": t, "face": True, "score": r["score"], "bbox": r["bbox"], "kps": r["kps"],
             "M": r["M"].tolist()}
        rec["samples"].append(s)
        if keep_frames and f.shape[:2] == (H, W):
            kept.append((t, f, s))
    return rec, kept


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "test"], required=True)
    p.add_argument("--frames-per-video", type=int, default=4)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--save-frames", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    videos = split_videos(a.split)
    if a.limit:
        videos = videos[: a.limit]
    (CACHE / "scan").mkdir(parents=True, exist_ok=True)
    (CACHE / "frames").mkdir(parents=True, exist_ok=True)
    out = open(CACHE / "scan" / f"{a.split}.jsonl", "w")
    raw = open(CACHE / "frames" / f"{a.split}_frames.u8", "wb") if a.save_frames else None
    idx = open(CACHE / "frames" / f"{a.split}_frames.jsonl", "w") if a.save_frames else None
    n_saved, t0 = 0, time.time()
    jobs = [(v, a.frames_per_video, a.save_frames) for v in videos]
    with spawn_pool(a.workers, initializer=_init, initargs=(a.threads,)) as pool:
        for i, (rec, kept) in enumerate(pool.imap_unordered(_scan, jobs, chunksize=4)):
            out.write(json.dumps(rec) + "\n")
            for t, f, s in kept:
                raw.write(np.ascontiguousarray(f).tobytes())
                idx.write(json.dumps({"i": n_saved, "video": rec["video"], "t": t, "score": s["score"],
                                      "bbox": s["bbox"], "kps": s["kps"], "M": s["M"]}) + "\n")
                n_saved += 1
            if (i + 1) % 200 == 0:
                el = time.time() - t0
                print(f"[scan {a.split}] {i+1}/{len(jobs)} videos, {n_saved} frames saved, "
                      f"{el/60:.1f} min, ETA {(len(jobs)-i-1)*el/(i+1)/60:.1f} min", flush=True)
    out.close()
    if raw:
        raw.close()
        idx.close()
    print(f"[scan {a.split}] done: {len(jobs)} videos, {n_saved} frames saved in {(time.time()-t0)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
