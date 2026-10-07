"""Step 3 - fix the TEST videos used for the comparison (UCF-101 split-1 test list).

Candidates: a face in >= 3 of the 4 scanned frames and median face width >= 40 px (the same face
criterion as the training faces). From these, a seeded sample of N videos (the per-video
optimisation is expensive), always including v_ApplyEyeMakeup_g01_c01 (the earlier demo video).

For each video, SimSwap's detector runs on every frame of the evaluation window [0, 80) of the
CLEAN video - this is the defender's alignment used to place the face-anchored perturbation.
Each video gets an attacker partner face from another selected video of a different group
(different person) and a different class.

Output: results/eval_set.json, cache/dets/<video stem>.json, results/eval_set/partners/*.png
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from deepfake.common import MIN_FACE_W, dump_json
from deepfake.detpool import DetPool
from deepfake.face import crop_cv2
from deepfake.paths import CACHE, RESULTS
from deepfake.ucf import group_of, read_frames

WINDOW = 80
FORCE = ["ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi"]


def candidates():
    out = []
    for line in open(CACHE / "scan" / "test.jsonl"):
        r = json.loads(line)
        s = [x for x in r["samples"] if x.get("face")]
        if len(s) >= 3 and np.median([x["bbox"][2] - x["bbox"][0] for x in s]) >= MIN_FACE_W \
                and r.get("hw") == [240, 320]:
            out.append(r["video"])
    return sorted(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--extra", action="store_true",
                   help="all remaining candidates (not in eval_set.json) -> eval_set_extra.json (universal-only eval)")
    a = p.parse_args()

    cand = candidates()
    if a.extra:
        used = {e["video"] for e in json.load(open(RESULTS / "eval_set.json"))["videos"]}
        chosen = sorted(v for v in cand if v not in used)
    else:
        rng = random.Random(a.seed)
        rest = [v for v in cand if v not in FORCE]
        rng.shuffle(rest)
        chosen = sorted(FORCE + rest[: a.n - len(FORCE)])
    print(f"{len(cand)} candidate test videos; {len(chosen)} selected")

    (CACHE / "dets").mkdir(parents=True, exist_ok=True)
    pool = DetPool(a.workers)
    info = {}
    for k, v in enumerate(chosen):
        stem = Path(v).stem
        path = CACHE / "dets" / f"{stem}.json"
        if path.exists():
            info[v] = json.load(open(path))
            continue
        frames, fps = read_frames(v, WINDOW)
        dets = pool.detect(frames)
        rec = {"video": v, "fps": fps, "n_window": len(frames), "dets": dets}
        dump_json(path, rec)
        info[v] = rec
        if (k + 1) % 10 == 0:
            print(f"[dets] {k+1}/{len(chosen)}", flush=True)
    pool.close()

    # partner (attacker's own face): best clean face of another selected video, other group & class
    best_face = {}
    for v, rec in info.items():
        cands = [(d["score"], t) for t, d in enumerate(rec["dets"]) if d]
        best_face[v] = max(cands)[1] if cands else None
    ppath = RESULTS / "eval_set" / "partners"
    ppath.mkdir(parents=True, exist_ok=True)
    entries = []
    for v in chosen:
        rec = info[v]
        eval_t = [t for t in range(0, 80, 5) if t < rec["n_window"] and rec["dets"][t]]
        pool_v = [u for u in chosen if group_of(u) != group_of(v) and u.split("/")[0] != v.split("/")[0]
                  and best_face[u] is not None]
        partner = random.Random(f"{a.seed}-{v}").choice(pool_v)
        pt = best_face[partner]
        frames, _ = read_frames(partner, pt + 1)
        crop = crop_cv2(frames[pt], np.asarray(info[partner]["dets"][pt]["M"]))
        cv2.imwrite(str(ppath / f"{Path(v).stem}.png"), crop)
        entries.append({"video": v, "eval_frames": eval_t, "n_window": rec["n_window"],
                        "partner": {"video": partner, "t": pt, "png": str(ppath / f"{Path(v).stem}.png")}})
    name = "eval_set_extra.json" if a.extra else "eval_set.json"
    dump_json(RESULTS / name, {"window": WINDOW, "n_candidates": len(cand), "videos": entries})
    n_frames = sum(len(e["eval_frames"]) for e in entries)
    print(f"eval set: {len(entries)} videos, {n_frames} scored frames -> {RESULTS / name}")


if __name__ == "__main__":
    main()
