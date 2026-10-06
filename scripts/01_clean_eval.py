#!/usr/bin/env python
"""Stage 01 - measure the victim's clean accuracy on the full official split-1
test list (3783 videos) and freeze the evaluation subset every attack will use.

Two accuracies are recorded per video:
  * top1_1clip  - the single centre clip the attacks actually perturb
                  (SampleFrames num_clips=1, test_mode).  This defines V_clean.
  * top1_10clip - the official 10-clip protocol, reported only to validate the
                  checkpoint against the published MMAction2 number.

Resumable: appends one json line per video, skips whatever is already done.
"""
import argparse, collections, json, os, sys, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
from dfbench.paths import (VIDEO_DIR, RESULT_DIR, load_testlist, CLEAN_RESULTS,
                           EVAL_SUBSET)
from dfbench.data import load_video_clips
from dfbench.records import RecordStore, atomic_write_json
from dfbench.victim import C3DVictim


class ClipSet(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel, lab = self.items[i]
        p = os.path.join(VIDEO_DIR, rel)
        try:
            c1 = load_video_clips(p, num_clips=1)
            c10 = load_video_clips(p, num_clips=10)
            return rel, lab, c1, c10, ""
        except Exception as e:                       # corrupt/unreadable video
            z = np.zeros((1, 3, 16, 112, 112), np.float32)
            return rel, lab, z, np.zeros((10, 3, 16, 112, 112), np.float32), str(e)


def collate(b):
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--per-class", type=int, default=3,
                    help="videos per class in the frozen attack subset")
    a = ap.parse_args()

    items = load_testlist()
    store = RecordStore(CLEAN_RESULTS)
    done = store.done_videos()        # error records are retried on the next run
    todo = [it for it in items if it[0] not in done]
    print(f"[stage01] total={len(items)} done={len(done)} todo={len(todo)}", flush=True)

    if todo:
        victim = C3DVictim(max_batch=64)
        dl = DataLoader(ClipSet(todo), batch_size=8, num_workers=a.workers,
                        collate_fn=collate)
        t0 = time.time()
        n = 0
        for batch in dl:
            for rel, lab, c1, c10, err in batch:
                if err:
                    store.append({"video": rel, "label": int(lab), "error": err,
                                  "status": "error"})
                    n += 1
                    continue
                p1 = int(victim.predict(c1).argmax(1)[0])
                lg10 = victim.predict_video(c10, average="score")
                p10 = int(lg10.argmax())
                top5 = torch.topk(lg10, 5).indices.tolist()
                store.append({"video": rel, "label": int(lab), "status": "done",
                              "pred_1clip": p1, "pred_10clip": p10,
                              "correct_1clip": p1 == lab, "correct_10clip": p10 == lab,
                              "top5_10clip": top5})
                n += 1
            if n % 200 < 8:
                el = time.time() - t0
                rate = n / max(el, 1e-9)
                eta = (len(todo) - n) / max(rate, 1e-9)
                print(f"[stage01] {n}/{len(todo)}  {rate:.1f} vid/s  "
                      f"elapsed {el/60:.1f}m  ETA {eta/60:.1f}m", flush=True)
        store.close()

    # ---------------------------------------------------------------- summary
    recs = store.load()
    ok = [r for r in recs if r.get("status") == "done"]
    if not ok:
        errs = collections.Counter(r.get("error", "unknown")
                                   for r in recs if r.get("status") == "error")
        print(f"\n[stage01] FATAL: no video was evaluated successfully "
              f"({len(recs)} records written).", file=sys.stderr)
        for msg, n in errs.most_common(3):
            print(f"[stage01]   {n} x {msg}", file=sys.stderr)
        print("[stage01] Check the environment with  python scripts/check_env.py", file=sys.stderr)
        print("[stage01] Failed videos are retried automatically on the next run.",
              file=sys.stderr)
        sys.exit(1)
    acc1 = 100.0 * sum(r["correct_1clip"] for r in ok) / len(ok)
    acc10 = 100.0 * sum(r["correct_10clip"] for r in ok) / len(ok)
    top5 = 100.0 * sum(r["label"] in r["top5_10clip"] for r in ok) / len(ok)
    print(f"\n[stage01] videos evaluated : {len(ok)} / {len(items)}")
    print(f"[stage01] top-1 (1 clip)   : {acc1:.2f}%   <- defines V_clean")
    print(f"[stage01] top-1 (10 clips) : {acc10:.2f}%   (published: 83.27%)")
    print(f"[stage01] top-5 (10 clips) : {top5:.2f}%   (published: 95.90%)")

    # ------------------------------------------------- frozen attack subset
    # V_clean under the single-clip protocol the attacks operate on.
    clean = [r for r in ok if r["correct_1clip"]]
    by_cls = {}
    for r in clean:
        by_cls.setdefault(r["label"], []).append(r)
    subset, tiers = [], {}
    for k in range(a.per_class):
        tier = []
        for lab in sorted(by_cls):
            vids = sorted(by_cls[lab], key=lambda r: r["video"])
            if k < len(vids):
                tier.append({"video": vids[k]["video"], "label": lab,
                             "tier": k + 1})
        tiers[k + 1] = len(tier)
        subset.extend(tier)
    for i, s in enumerate(subset):
        s["index"] = i
    atomic_write_json(EVAL_SUBSET, {
        "protocol": "SampleFrames(clip_len=16, frame_interval=1, num_clips=1, "
                    "test_mode=True) + Resize(128,171) + CenterCrop(112) + "
                    "Normalize(mean=[104,117,128], std=[1,1,1], RGB)",
        "victim": "mmaction2 C3D sports1m -> ucf101 split1",
        "clean_top1_1clip": acc1, "clean_top1_10clip": acc10,
        "clean_top5_10clip": top5,
        "n_test_total": len(items), "n_clean_1clip": len(clean),
        "per_class": a.per_class, "tier_sizes": tiers,
        "n_subset": len(subset), "videos": subset,
    })
    print(f"[stage01] |V_clean| (1-clip) = {len(clean)}")
    print(f"[stage01] frozen attack subset = {len(subset)} videos, tiers {tiers}")
    print(f"[stage01] -> {EVAL_SUBSET}")


if __name__ == "__main__":
    main()
