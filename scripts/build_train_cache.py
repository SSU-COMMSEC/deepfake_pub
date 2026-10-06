#!/usr/bin/env python
"""Pre-decode the UCF-101 split-1 TRAIN list into a uint8 memmap.

Both attacks build their perturbation from training clips, and both revisit the
same clips many times:

  * C-DUP trains a generator for several epochs over the train split.
  * U3D evaluates its PSO objective on a fixed clip set once per candidate,
    hundreds to thousands of times.

Decoding the .avi files on the fly would dominate the runtime, so the 9537 train
videos are decoded once into a memmap that both adapters index into.

This is a caching layer only: the clips are byte-identical to what
`load_video_clips(..., num_clips=1)` returns, just stored as uint8.

Outputs (~5.8 GB total)
    dataset/train_clips_u8.npy          (9537, 3, 16, 112, 112) uint8 memmap
    dataset/train_clips_u8.npy.ok       per-video state: 0 pending, 1 ok, 2 undecodable
    dataset/train_clips_index.txt       index <-> relative path <-> label <-> state

Resumable - re-running fills in only what is missing.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
from dfbench.paths import VIDEO_DIR, SPLIT_DIR, DATASET_DIR, load_class_index
from dfbench.data import load_video_clips
from torch.utils.data import Dataset, DataLoader

CACHE = os.path.join(DATASET_DIR, "train_clips_u8.npy")
INDEX = os.path.join(DATASET_DIR, "train_clips_index.txt")
SHAPE = (3, 16, 112, 112)


def train_items():
    n2i, _ = load_class_index()
    out = []
    with open(os.path.join(SPLIT_DIR, "trainlist01.txt")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel = line.split()[0].replace("\\", "/")
            out.append((rel, n2i[rel.split("/")[0]]))
    return out


class DS(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        rel, lab = self.items[i]
        try:
            c = load_video_clips(os.path.join(VIDEO_DIR, rel), num_clips=1)[0]
            return i, np.clip(c, 0, 255).astype(np.uint8), lab, True
        except Exception:
            return i, np.zeros(SHAPE, np.uint8), lab, False


def main():
    items = train_items()
    n = len(items)
    print(f"[cache] train videos: {n}")
    mm = np.lib.format.open_memmap(CACHE, mode=("r+" if os.path.exists(CACHE) else "w+"),
                                   dtype=np.uint8, shape=(n,) + SHAPE)
    ok_path = CACHE + ".ok"
    okmask = (np.lib.format.open_memmap(ok_path, mode="r+", dtype=np.uint8, shape=(n,))
              if os.path.exists(ok_path)
              else np.lib.format.open_memmap(ok_path, mode="w+", dtype=np.uint8, shape=(n,)))
    todo = [i for i in range(n) if okmask[i] == 0]
    print(f"[cache] already cached: {n - len(todo)}   todo: {len(todo)}")
    if todo:
        dl = DataLoader(DS([items[i] for i in todo]), batch_size=16, num_workers=12,
                        collate_fn=lambda b: b)
        t0 = time.time(); c = 0
        for batch in dl:
            for j, clip, lab, good in batch:
                gi = todo[j]
                mm[gi] = clip
                okmask[gi] = 1 if good else 2
                c += 1
            if c % 500 < 16:
                r = c / max(time.time() - t0, 1e-9)
                print(f"[cache] {c}/{len(todo)}  {r:.1f}/s  ETA {(len(todo)-c)/max(r,1e-9)/60:.1f}m",
                      flush=True)
        mm.flush(); okmask.flush()
    with open(INDEX, "w") as f:
        for i, (rel, lab) in enumerate(items):
            f.write(f"{i}\t{rel}\t{lab}\t{int(okmask[i])}\n")
    bad = int((okmask == 2).sum())
    print(f"[cache] done. {n-bad} usable, {bad} undecodable -> {CACHE}")


if __name__ == "__main__":
    main()
