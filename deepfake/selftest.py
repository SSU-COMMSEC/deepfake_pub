"""Checks that the ported / re-written pieces do what the originals do.

    python -m deepfake.selftest   -> results/selftest.log
"""

import json
import sys

import cv2
import numpy as np
import torch

from deepfake.paths import CACHE, setup_imports

setup_imports()
RESULT = []


def check(name, ok, detail=""):
    RESULT.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def main():
    from dfbench import cdup_core, u3d_perlin
    from deepfake import u3d_core
    from deepfake.common import crop_grids, uncrop_grids, warp
    from deepfake.face import bgr_to_tensor, crop_cv2
    from deepfake.inject import quantize_protect, to_crop_space
    from deepfake.judges import DlibPool, FaceNetJudge, dlib_match, to_u8
    from deepfake.ucf import read_frames

    dl = DlibPool(4)

    # 1. Perlin port == upstream Rust module
    from perlin import perlin_noise
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(4):
        prm = [int(rng.integers(1, 6)), *rng.uniform(2, 180, 3), rng.uniform(1, 60)]
        ref = np.array([[[perlin_noise(float(x), float(y), float(t), int(prm[0]), *map(float, prm[1:]), 10.0)
                          for x in range(20)] for y in range(16)] for t in range(16)])
        ours = u3d_core.perlin_volume(16, 16, 20, prm, 10.0, noise="repo").cpu().numpy()
        worst = max(worst, float(np.abs(ref - ours).max()))
    check("U3D noise='repo' == compiled upstream perlin3d.rs", worst < 1e-10, f"max|diff|={worst:.1e}")

    # 1b. the paper noise (default) on the GPU == the numpy reference of Eq. 1-2
    worst = 0.0
    for _ in range(4):
        prm = [int(rng.integers(1, 6)), *rng.uniform(2, 180, 3), rng.uniform(1, 60)]
        ref = u3d_perlin.generate_noise(16, 48, 48, prm[0], *prm[1:], 10.0, mode="paper")
        ours = u3d_core.perlin_volume(16, 48, 48, prm, 10.0, noise="paper").cpu().numpy()
        worst = max(worst, float(np.abs(ref - ours).max()))
    check("U3D noise='paper' (GPU) == numpy reference of Eq. 1-2", worst < 1e-10, f"max|diff|={worst:.1e}")

    # 2. upstream Rust PSO with the corrected inertia converges inside the box. psolib is unseeded,
    #    so the median of 3 runs is tested. Over 20 runs each: omega -0.27 reached f <= 9e-3,
    #    omega 1.2 (the repository value) never got below 3e-2.
    target = np.array([3, 90, 90, 90, 30.0])
    span = np.array(u3d_core.UB) - np.array(u3d_core.LB)
    runs = [u3d_core.pso(lambda x: float((((np.asarray(x) - target) / span) ** 2).sum())) for _ in range(3)]
    best, f = sorted(runs, key=lambda r: r[1])[1]
    check("upstream Rust PSO, effective inertia 1+omega = 0.73, converges to an interior optimum",
          f < 2e-2, f"median of 3: best={np.round(best, 1)} f={f:.1e}")

    # 3. GPU crop grids == cv2.warpAffine crop; placement round trip
    frames, _ = read_frames("ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi", 1)
    from deepfake.face import FaceAligner
    M = FaceAligner().detect(frames[0])
    fr = bgr_to_tensor(frames[0]).cuda()[None]
    Mt = torch.as_tensor(M[None], dtype=torch.float64, device="cuda")
    c_gpu = warp(fr, crop_grids(Mt))[0]
    c_cv = bgr_to_tensor(crop_cv2(frames[0], M)).cuda()
    d = float((c_gpu - c_cv).abs().mean() * 255)
    check("GPU crop grid == SimSwap cv2.warpAffine crop", d < 0.5, f"mean|diff|={d:.3f}/255")
    P = to_crop_space(torch.randn(1, 3, 112, 112, device="cuda").clamp(-1, 1) * 10 / 255)
    placed = warp(P, uncrop_grids(Mt))
    back = warp(placed, crop_grids(Mt))
    inner = slice(40, 184)
    a_, b_ = P[0, :, inner, inner].flatten(), back[0, :, inner, inner].flatten()
    corr = float(torch.corrcoef(torch.stack([a_, b_]))[0, 1])
    check("crop-space perturbation survives place -> attacker crop (same M)", corr > 0.8, f"corr={corr:.3f}")
    check("placed perturbation within budget", float(placed.abs().max()) <= 10 / 255 + 1e-6,
          f"max={float(placed.abs().max()*255):.2f}/255")

    # 4. quantisation keeps |change| <= 10 levels
    clean = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
    prot = clean / 255 + np.random.uniform(-0.2, 0.2, clean.shape)
    q = quantize_protect(clean, prot, 10 / 255)
    check("uint8 upload respects 10/255 per channel", np.abs(q.astype(int) - clean).max() <= 10,
          f"max={np.abs(q.astype(int) - clean).max()}")

    # 5. C-DUP generators, Roll, Tile
    g3, g2 = cdup_core.Generator3D().cuda(), cdup_core.Generator2D().cuda()
    z = torch.randn(4, 100, device="cuda")
    o3, o2 = g3(z), g2(z)
    check("Generator3D output [B,3,16,112,112]", tuple(o3.shape) == (4, 3, 16, 112, 112), str(tuple(o3.shape)))
    check("Generator2D + Tile: 16 identical frames", tuple(o2.shape) == (4, 3, 16, 112, 112)
          and float((o2[:, :, 0] - o2[:, :, 7]).abs().max()) == 0)
    x = torch.arange(16.0).view(1, 1, 16, 1, 1)
    check("Roll == tf.manip.roll(axis=time)", np.array_equal(cdup_core.roll(x, 3).flatten().numpy(),
                                                              np.roll(np.arange(16.0), 3)))
    n_param = sum(p.numel() for p in g3.parameters())
    check("Generator3D parameter count (upstream architecture)", n_param > 2.4e6, f"{n_param:,}")

    # 6. judges == PhantomSeal's Effectiveness
    from deepfake.model import SimSwapTarget
    tg = SimSwapTarget()
    fn = FaceNetJudge(tg.effectiveness, float(tg.config.evaluate.facenet_512.threshold))
    recs = [json.loads(l) for l in open(CACHE / "scan" / "test.jsonl")][:60]
    crops = []
    for r in recs:
        s = [x for x in r["samples"] if x.get("face")]
        if s:
            fr_, _ = read_frames(r["video"], s[0]["t"] + 1)
            crops.append(bgr_to_tensor(crop_cv2(fr_[s[0]["t"]], np.asarray(s[0]["M"]))))
        if len(crops) == 16:
            break
    A = torch.stack(crops[:8]).cuda()
    B = torch.stack(crops[8:16]).cuda()
    swapped = tg.swap(A, B)
    ref_d = tg.effectiveness.get_images_distance(swapped, A)
    our_d = [fn.distance(fn.embed(s), fn.embed(x)) for s, x in zip(swapped, A)]
    same = all((np.isinf(r_) and np.isinf(o_)) or abs(r_ - o_) < 1e-4 for r_, o_ in zip(ref_d, our_d))
    check("FaceNet judge == PhantomSeal get_images_distance", same, f"{np.round(ref_d,3)}")
    m_ref, v_ref = tg.effectiveness._get_facerec_matching(swapped, A)
    enc_s, enc_a = dl.encode([to_u8(s) for s in swapped]), dl.encode([to_u8(x) for x in A])
    m_our = sum(dlib_match(e1, e2) for e1, e2 in zip(enc_a, enc_s))
    check("dlib judge == PhantomSeal _get_facerec_matching", int(m_ref) == m_our, f"{int(m_ref)} vs {m_our} of 8")

    # 7. per-frame PhantomSeal (frame space) == upstream Defense._perturb_imgs under an identity
    #    alignment, one step: pixels match except sign flips caused by ~1e-6 resampling error
    from src.simswap.defense import Defense
    from deepfake.phantomseal_frames import perturb_frames
    x4 = A[:4]
    cl = tg.base.cloak.find_best_cloaks(x4[:1]).cuda().expand(4, -1, -1, -1)
    dcfg = tg.config.third_party.defense
    epochs, dcfg.epochs = dcfg.epochs, 1
    torch.manual_seed(0)
    ref = Defense._perturb_imgs(tg.base, x4, cl)
    Mi = torch.tensor([[[1., 0, 0], [0, 1, 0]]] * 4, dtype=torch.float64, device="cuda")
    torch.manual_seed(0)
    ours = perturb_frames(tg.base, x4, Mi, cl, [dcfg.limit.R, dcfg.limit.G, dcfg.limit.B], epochs=1)
    dcfg.epochs = epochs
    same = float(((ref - ours).abs() < 1e-6).float().mean())
    check("per-frame PhantomSeal (frame space) == upstream _perturb_imgs, 1 step, identity alignment",
          same > 0.9, f"identical pixels {100 * same:.1f}%")
    dl.close()

    n_fail = sum(not ok for _, ok, _ in RESULT)
    print(f"{len(RESULT) - n_fail}/{len(RESULT)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
