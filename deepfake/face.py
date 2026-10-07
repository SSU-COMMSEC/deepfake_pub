"""SimSwap's own face pipeline (SCRFD detect -> estimate_norm align -> 224 crop -> paste back)
plus differentiable versions of the same warps.

The attacker in every experiment is SimSwap's video pipeline (`util/videoswap.py`): it detects
the face on whatever frame it is given, aligns it with `estimate_norm(mode='None')` and crops
224x224. The helpers below reproduce that pipeline exactly (checked in selftest.py).
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch import Tensor

from deepfake.paths import SIMSWAP, PHANTOMSEAL, setup_imports

setup_imports()
from src.common_utils import use_project, suppress_third_party_noise  # noqa: E402

CROP = 224


def limit_onnx_threads(n: int) -> None:
    """SCRFD runs on onnxruntime-CPU (no CUDA provider on this box). Caps its threads so that
    several worker processes can share the 28 cores."""
    import onnxruntime as ort

    original = ort.InferenceSession.__init__

    def patched(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kw):
        if sess_options is None:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = n
            sess_options.inter_op_num_threads = 1
        original(self, path_or_bytes, sess_options, providers or ["CPUExecutionProvider"],
                 provider_options, **kw)

    ort.InferenceSession.__init__ = patched


class FaceAligner:
    """SimSwap's `insightface_func.Face_detect_crop` (antelope SCRFD, det_thresh 0.6, det_size 640).

    SimSwap targets insightface 0.2, whose `detect()` took the threshold as an argument;
    insightface 0.7 keeps it as an attribute, so only `get()` is re-written. Detector, face
    choice (highest score) and alignment template are SimSwap's own.
    """

    def __init__(self, det_thresh: float = 0.6, det_size: int = 640, crop_size: int = CROP):
        with use_project([SIMSWAP]):
            from insightface_func.face_detect_crop_single import Face_detect_crop
            from insightface_func.utils import face_align_ffhqandnewarc as face_align
        self._face_align = face_align
        with suppress_third_party_noise():
            self.detector = Face_detect_crop(name="antelope", root=str(PHANTOMSEAL / "checkpoints" / "simswap"))
            self.detector.prepare(ctx_id=0, det_thresh=det_thresh, det_size=(det_size, det_size), mode="None")
        self.crop_size = crop_size

    def detect_full(self, frame_bgr: np.ndarray) -> dict | None:
        """Best face: score, bbox, 5 landmarks and the 2x3 frame->crop matrix M."""
        det = self.detector.det_model
        det.det_thresh = self.detector.det_thresh
        bboxes, kpss = det.detect(frame_bgr, max_num=0, metric="default")
        if bboxes.shape[0] == 0:
            return None
        best = int(np.argmax(bboxes[..., 4]))
        kps = kpss[best]
        M, _ = self._face_align.estimate_norm(kps, self.crop_size, mode="None")
        return {"score": float(bboxes[best, 4]), "bbox": bboxes[best, :4].astype(float).tolist(),
                "kps": kps.astype(float).tolist(), "M": np.asarray(M, dtype=np.float64)}

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        r = self.detect_full(frame_bgr)
        return None if r is None else r["M"]


def crop_cv2(frame_bgr: np.ndarray, M: np.ndarray, crop_size: int = CROP) -> np.ndarray:
    """The aligned crop exactly as SimSwap produces it (uint8 BGR)."""
    return cv2.warpAffine(frame_bgr, M, (crop_size, crop_size), borderValue=0.0)


def bgr_to_tensor(img_bgr: np.ndarray) -> Tensor:
    """SimSwap's `_totensor(cv2.cvtColor(img, BGR2RGB))`: [3,H,W] RGB float in [0,1]."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255)


def tensor_to_bgr(img: Tensor) -> np.ndarray:
    arr = (img.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def invert_affine(M: np.ndarray) -> np.ndarray:
    """The inverse SimSwap's `reverse2wholeimage` writes out element by element."""
    mat_rev = np.zeros([2, 3])
    div1 = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    mat_rev[0][0] = M[1][1] / div1
    mat_rev[0][1] = -M[0][1] / div1
    mat_rev[0][2] = -(M[0][2] * M[1][1] - M[0][1] * M[1][2]) / div1
    div2 = M[0][1] * M[1][0] - M[0][0] * M[1][1]
    mat_rev[1][0] = M[1][0] / div2
    mat_rev[1][1] = -M[0][0] / div2
    mat_rev[1][2] = -(M[0][2] * M[1][0] - M[0][0] * M[1][2]) / div2
    return mat_rev


def crop_grid(Ms, frame_hw, crop_size: int = CROP) -> Tensor:
    """grid_sample grid equal to `cv2.warpAffine(frame, M, (S,S))` (bilinear, zero border)."""
    h, w = frame_hw
    u, v = np.meshgrid(np.arange(crop_size), np.arange(crop_size))
    pts = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3).astype(np.float64)
    grids = []
    for M in Ms:
        xy = pts @ cv2.invertAffineTransform(np.asarray(M, np.float64)).T   # crop px -> frame px
        gx = 2 * xy[:, 0] / (w - 1) - 1
        gy = 2 * xy[:, 1] / (h - 1) - 1
        grids.append(np.stack([gx, gy], axis=-1).reshape(crop_size, crop_size, 2))
    return torch.from_numpy(np.stack(grids)).float()


def uncrop_grid(Ms, frame_hw, crop_size: int = CROP) -> Tensor:
    """grid_sample grid sending every frame pixel to its position in the aligned crop
    (used to place a crop-space perturbation onto the frame)."""
    h, w = frame_hw
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    pts = np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3).astype(np.float64)
    grids = []
    for M in Ms:
        uv = pts @ np.asarray(M, dtype=np.float64).T                        # frame px -> crop px
        gu = 2 * uv[:, 0] / (crop_size - 1) - 1
        gv = 2 * uv[:, 1] / (crop_size - 1) - 1
        grids.append(np.stack([gu, gv], axis=-1).reshape(h, w, 2))
    return torch.from_numpy(np.stack(grids)).float()


def warp_with_grid(imgs: Tensor, grid: Tensor) -> Tensor:
    return F.grid_sample(imgs, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def paste_back(swapped_rgb: np.ndarray, M: np.ndarray, frame_bgr: np.ndarray) -> np.ndarray:
    """Port of SimSwap `util/reverse2original.reverse2wholeimage` (use_mask=False, no logo);
    the original cannot run under NumPy 2 (`np.float`). Arithmetic unchanged."""
    crop_size = swapped_rgb.shape[0]
    mat_rev = invert_affine(M)
    orisize = (frame_bgr.shape[1], frame_bgr.shape[0])
    target_image = cv2.warpAffine(swapped_rgb, mat_rev, orisize)
    img_white = np.full((crop_size, crop_size), 255, dtype=float)
    img_white = cv2.warpAffine(img_white, mat_rev, orisize)
    img_white[img_white > 20] = 255
    img_mask = cv2.erode(img_white, np.ones((40, 40), np.uint8), iterations=1)
    img_mask = cv2.GaussianBlur(img_mask, (41, 41), 0)
    img_mask /= 255
    img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
    target_image = np.array(target_image, dtype=float)[..., ::-1] * 255
    img = np.array(frame_bgr, dtype=float)
    img = img_mask * target_image + (1 - img_mask) * img
    return img.astype(np.uint8)
