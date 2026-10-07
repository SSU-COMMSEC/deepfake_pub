"""How a perturbation clip gets into a video, and what the attacker then sees.

Both U3D (Perlin volume) and C-DUP (generator output) produce a clip P of T=16 frames at
112x112, the input resolution of the C3D models they were built for. Here the model under
attack is SimSwap, whose input is the 224x224 aligned face crop, so P lives in that crop
space ("face-anchored UAP"):

  1. P is bilinearly upsampled 112 -> 224 and tapered to zero over the last 16 px of the
     crop border (no hard square edge in the video).
  2. Video frame t receives clip frame (t + tau) mod T  -- U3D's Trans(xi, tau) / C-DUP's Roll.
  3. The defender finds the face with SimSwap's own detector on the clean frame (matrix M)
     and warps P_t onto the frame with M^-1 (bilinear). The L_inf budget holds in the frame.
  4. The attacker re-detects the face on the protected frame and crops 224x224 with its own
     matrix M'. Training uses M' = M (no spatial jitter); evaluation uses SimSwap's real
     re-detection.
"""

import numpy as np
import torch
import torch.nn.functional as F

from deepfake.face import CROP, crop_grid, uncrop_grid, warp_with_grid

T_CLIP = 16
NATIVE = 112
FEATHER = 16


def taper_mask(size: int = CROP, feather: int = FEATHER, device="cpu") -> torch.Tensor:
    """1 inside, raised-cosine ramp to 0 over `feather` px at the crop border."""
    r = torch.arange(size, dtype=torch.float32, device=device)
    d = torch.minimum(r, size - 1 - r)
    ramp = torch.where(d >= feather, torch.ones_like(d), 0.5 - 0.5 * torch.cos(np.pi * d / feather))
    return (ramp[:, None] * ramp[None, :])[None, None]


def to_crop_space(p_native: torch.Tensor) -> torch.Tensor:
    """[N,3,112,112] (or [N,1,..]) -> [N,3,224,224], bilinear, tapered."""
    p = F.interpolate(p_native, size=(CROP, CROP), mode="bilinear", align_corners=False)
    if p.shape[1] == 1:
        p = p.expand(-1, 3, -1, -1)
    return p * taper_mask(device=p.device)


class Placer:
    """Puts crop-space perturbations onto frames and crops frames like the attacker (batched)."""

    def __init__(self, frame_hw=(240, 320)):
        self.hw = frame_hw

    def place(self, frames, p_crop, Ms, uncrop=None):
        """frames [B,3,H,W] in [0,1]; p_crop [B,3,224,224]; Ms list of 2x3 -> protected frames."""
        g = uncrop if uncrop is not None else uncrop_grid(Ms, self.hw).to(frames.device)
        delta = warp_with_grid(p_crop, g)
        return torch.clamp(frames + delta, 0, 1)

    def crop(self, frames, Ms, grid=None):
        g = grid if grid is not None else crop_grid(Ms, self.hw).to(frames.device)
        return warp_with_grid(frames, g)


def clip_index(t: int, tau: int = 0, T: int = T_CLIP) -> int:
    return (t + tau) % T


def quantize_protect(clean_u8: np.ndarray, protected_float_rgb: np.ndarray, eps: float) -> np.ndarray:
    """Uploaded video = uint8. Rounds and keeps |change| <= eps*255 levels in every channel."""
    lv = np.floor(eps * 255 + 1e-6)
    c = clean_u8.astype(np.float64)
    q = np.rint(np.clip(protected_float_rgb, 0, 1) * 255)
    q = np.clip(np.clip(q, c - lv, c + lv), 0, 255)
    return q.astype(np.uint8)
