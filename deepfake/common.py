"""Data plumbing shared by optimisation and evaluation: face frames on the GPU, warp grids built
on the GPU, and the batched "place perturbation -> attacker crop -> PhantomSeal score" step."""

import json
import math
import random
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from deepfake.face import CROP
from deepfake.inject import T_CLIP, to_crop_space
from deepfake.paths import CACHE

H, W = 240, 320
EPS = 10 / 255          # C-DUP p_max = 10 (and the UCF-101 benchmark's eps); U3D's code default is 8
MIN_FACE_W = 40         # px; smaller faces are a few dozen pixels upsampled 4-6x by SimSwap's crop
MIN_SCORE = 0.6         # SimSwap's det_thresh


def face_width(rec) -> float:
    return rec["bbox"][2] - rec["bbox"][0]


def load_train_index(min_w=MIN_FACE_W, min_score=MIN_SCORE):
    raw = np.memmap(CACHE / "frames" / "train_frames.u8", dtype=np.uint8, mode="r").reshape(-1, H, W, 3)
    idx = []
    for line in open(CACHE / "frames" / "train_frames.jsonl"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:      # tolerate a partly written last line
            continue
        if r["i"] < raw.shape[0]:
            idx.append(r)
    keep = [r for r in idx if face_width(r) >= min_w and r["score"] >= min_score]
    return raw, keep


def one_frame_per_video(index, n, seed=0):
    by_video = {}
    for r in index:
        by_video.setdefault(r["video"], []).append(r)
    rng = random.Random(seed)
    videos = sorted(by_video)
    rng.shuffle(videos)
    return [rng.choice(by_video[v]) for v in videos[:n]]


# ---------------------------------------------------------------- GPU warp grids
def _inv_affine(M):
    A, b = M[:, :, :2], M[:, :, 2:]
    Ai = torch.linalg.inv(A)
    return torch.cat([Ai, -Ai @ b], dim=2)


def crop_grids(M: torch.Tensor, hw=(H, W), size=CROP) -> torch.Tensor:
    """[B,2,3] frame->crop matrices -> grid_sample grids equal to cv2.warpAffine(frame, M, (S,S))."""
    h, w = hw
    Mi = _inv_affine(M.double())
    v, u = torch.meshgrid(torch.arange(size, device=M.device, dtype=torch.float64),
                          torch.arange(size, device=M.device, dtype=torch.float64), indexing="ij")
    pts = torch.stack([u, v, torch.ones_like(u)], -1).reshape(1, -1, 3)
    xy = pts @ Mi.transpose(1, 2)
    g = torch.stack([2 * xy[..., 0] / (w - 1) - 1, 2 * xy[..., 1] / (h - 1) - 1], -1)
    return g.reshape(-1, size, size, 2).float()


def uncrop_grids(M: torch.Tensor, hw=(H, W), size=CROP) -> torch.Tensor:
    """[B,2,3] -> grids sending each frame pixel to its crop position (to place a crop-space pert)."""
    h, w = hw
    y, x = torch.meshgrid(torch.arange(h, device=M.device, dtype=torch.float64),
                          torch.arange(w, device=M.device, dtype=torch.float64), indexing="ij")
    pts = torch.stack([x, y, torch.ones_like(x)], -1).reshape(1, -1, 3)
    uv = pts @ M.double().transpose(1, 2)
    g = torch.stack([2 * uv[..., 0] / (size - 1) - 1, 2 * uv[..., 1] / (size - 1) - 1], -1)
    return g.reshape(-1, h, w, 2).float()


def warp(imgs, grid):
    return torch.nn.functional.grid_sample(imgs, grid, mode="bilinear", padding_mode="zeros",
                                           align_corners=True)


@dataclass
class FaceBatch:
    """Clean frames with their defender alignment, ready for placement + attacker crop."""
    frames: torch.Tensor       # [B,3,H,W] RGB in [0,1]
    M: torch.Tensor            # [B,2,3]
    cgrid: torch.Tensor
    ugrid: torch.Tensor
    clean: torch.Tensor        # attacker crops of the clean frames [B,3,224,224]
    ref: object = None         # ProtectionObjective.reference(clean)

    @staticmethod
    def build(frames_bgr, Ms, objective=None, device="cuda"):
        rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr])
        fr = torch.from_numpy(rgb).to(device).permute(0, 3, 1, 2).float().div(255)
        M = torch.as_tensor(np.stack(Ms), dtype=torch.float64, device=device)
        cg, ug = crop_grids(M, fr.shape[-2:]), uncrop_grids(M, fr.shape[-2:])
        clean = warp(fr, cg)
        fb = FaceBatch(fr, M, cg, ug, clean)
        if objective is not None:
            fb.ref = objective.reference(clean)
        return fb

    def subset(self, idx):
        ref = None
        if self.ref is not None:
            ref = type(self.ref)(self.ref.ident[idx], self.ref.ctx[idx])
        return FaceBatch(self.frames[idx], self.M[idx], self.cgrid[idx], self.ugrid[idx], self.clean[idx], ref)

    def __len__(self):
        return self.frames.shape[0]


def protected_crops(fb: FaceBatch, p_crop: torch.Tensor) -> torch.Tensor:
    """frame + M^-1(P_t) -> clamp -> attacker crop with the same M (training assumption)."""
    prot = torch.clamp(fb.frames + warp(p_crop, fb.ugrid), 0, 1)
    return warp(prot, fb.cgrid)


def clip_to_crop(clip: torch.Tensor) -> torch.Tensor:
    """[T,C,112,112] (C=1 for U3D, 3 for C-DUP) -> [T,3,224,224] crop-space perturbation."""
    return to_crop_space(clip)


def score_clip(objective, fb: FaceBatch, clip_crop: torch.Tensor, tau_idx: torch.Tensor, chunk=64):
    """Mean PhantomSeal score when sample i receives clip frame tau_idx[i] (no grad)."""
    total, n, parts = 0.0, 0, {"s_id": 0.0, "s_ctx": 0.0}
    with torch.no_grad():
        for i in range(0, len(fb), chunk):
            sub = fb.subset(slice(i, i + chunk))
            crops = protected_crops(sub, clip_crop[tau_idx[i:i + chunk]])
            s, info = objective.score(crops, sub.ref)
            total += float(s.sum())
            parts["s_id"] += float(info["s_id"].sum())
            parts["s_ctx"] += float(info["s_ctx"].sum())
            n += s.shape[0]
    return total / n, {k: v / n for k, v in parts.items()}


def eval_frame_indices(n_frames: int, stride: int = 5, count: int = 16) -> list[int]:
    """Frames scored per test video: 0, 5, ..., 75. gcd(5, 16) = 1, so the 16 frames receive the 16
    different clip frames of a T=16 perturbation exactly once (with tau = 0)."""
    return [t for t in range(0, stride * count, stride) if t < n_frames]


def clip_frame_for(t: int, tau: int = 0) -> int:
    return (t + tau) % T_CLIP


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dump_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    tmp.replace(path)


def finite(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else x
