"""Step 5 - evaluation: SimSwap's real video pipeline is the attacker, PhantomSeal's metrics score it.

For every test video of results/eval_set.json and every variant:
  1. the defender protects the video (variant clip frame t mod 16 placed on the face, uint8 upload,
     |change| <= 10/255 per channel); optional --codec xvid re-encodes the whole window first
  2. the attacker runs SimSwap's detector on the PROTECTED frame and crops 224x224 (if it finds
     no face it falls back to the clean alignment - conservative; the failure is recorded)
  3. identity theft: swap(source = protected face, target = partner face)
     context theft:  swap(source = partner face, target = protected face)
  4. PhantomSeal's judges (FaceNet-512, dlib) decide whether each attack succeeded:
       identity theft succeeds if the output matches the victim's clean face   (ASR_id)
       context theft  succeeds if the output matches the partner (attacker)    (ASR_ctx)
       x~ : protected face still recognised as the victim (utility of the upload)
  --oracle additionally scores the paper's protocol (attacker reuses the defender's crop).

    python -m deepfake.evaluate --variants clean,u3d_uap,cdup3d_uap --name main
Output: results/eval/<name>/records.jsonl (one line per video x variant; resumable)
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage import metrics as skm

from deepfake.common import EPS, clip_to_crop, uncrop_grids, warp
from deepfake.detpool import DetPool
from deepfake.face import bgr_to_tensor, crop_cv2
from deepfake.inject import T_CLIP, quantize_protect
from deepfake.judges import DlibPool, FaceNetJudge, dlib_match, to_u8
from deepfake.paths import CACHE, RESULTS
from deepfake.ucf import read_frames, write_video

VARIANTS = {
    "clean": None,
    "uniform_uap": ("universal", "uniform"),
    "u3d_uap": ("universal", "u3d"),
    "cdup3d_uap": ("universal", "cdup3d"),
    "cdup2d_uap": ("universal", "cdup2d"),
    "u3d_pv": ("per_video", "u3d"),
    "cdup3d_pv": ("per_video", "cdup3d"),
    # PhantomSeal run on every scored frame (phantomseal_frames.py): stored protected frames
    "phantomseal_pf": ("per_frame", "phantomseal"),
    # the same with PhantomSeal's own R/G/B bounds (--native-limits --tag _native); for reference
    "phantomseal_native_pf": ("per_frame", "phantomseal_native"),
    # the reference implementation's noise and diverging PSO (train with --u3d-noise repo
    # --u3d-omega 1.2 --tag _repo); kept to compare against runs made before the fix
    "u3d_repo_uap": ("universal", "u3d_repo"),
    "u3d_repo_pv": ("per_video", "u3d_repo"),
}


def load_clip(variant: str, stem: str, universal_dir: Path):
    kind, method = VARIANTS[variant]
    base = universal_dir if kind == "universal" else RESULTS / "uap" / "per_video"
    path = base / method / ("clip.npy" if kind == "universal" else f"{stem}.npy")
    return clip_to_crop(torch.from_numpy(np.load(path)).cuda())          # [16,3,224,224]


def load_per_frame(variant: str, stem: str, ts):
    """Protected uint8 BGR frames stored by a per-frame method, for the scored frames ts."""
    z = np.load(RESULTS / "uap" / "per_frame" / VARIANTS[variant][1] / f"{stem}.npz")
    if list(z["ts"]) != list(ts):
        raise RuntimeError(f"{variant}/{stem}: stored frames {list(z['ts'])} != eval frames {list(ts)}")
    return list(z["frames"])


def protect(frames_bgr, Ms, P, ts, eps=EPS):
    """Places clip frame (t mod 16) on each face; returns uint8 BGR frames (the upload)."""
    rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr])
    fr = torch.from_numpy(rgb).cuda().permute(0, 3, 1, 2).float().div(255)
    M = torch.as_tensor(np.stack(Ms), dtype=torch.float64, device="cuda")
    idx = torch.as_tensor([t % T_CLIP for t in ts], device="cuda")
    with torch.no_grad():
        prot = torch.clamp(fr + warp(P[idx], uncrop_grids(M, fr.shape[-2:])), 0, 1)
    q = quantize_protect(rgb, prot.permute(0, 2, 3, 1).cpu().numpy(), eps)
    return [cv2.cvtColor(x, cv2.COLOR_RGB2BGR) for x in q]


def corner_shift(M_att, M_def, size=224) -> float:
    """Mean displacement (crop px) of the crop corners/centre between attacker and defender crops."""
    q = np.array([[0, 0], [size - 1, 0], [0, size - 1], [size - 1, size - 1], [size / 2, size / 2]], float)
    inv = cv2.invertAffineTransform(np.asarray(M_att, np.float64))
    frame_pts = q @ inv[:, :2].T + inv[:, 2]
    q2 = frame_pts @ np.asarray(M_def)[:, :2].T + np.asarray(M_def)[:, 2]
    return float(np.linalg.norm(q2 - q, axis=1).mean())


def codec_roundtrip(frames_bgr, fps, fourcc="XVID", ext=".avi"):
    with tempfile.TemporaryDirectory(dir=CACHE) as d:
        path = Path(d) / f"rt{ext}"
        write_video(path, frames_bgr, fps, fourcc)
        out, _ = read_frames(path)
    if len(out) != len(frames_bgr):
        raise RuntimeError(f"{fourcc} round trip returned {len(out)} of {len(frames_bgr)} frames")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", default=",".join(v for v in VARIANTS if "_repo" not in v and "_native" not in v),
                   help="default: every variant except the u3d_repo_* / *_native_* comparison runs")
    p.add_argument("--name", default="main")
    p.add_argument("--oracle", action="store_true")
    p.add_argument("--codec", choices=["none", "xvid"], default="none")
    p.add_argument("--universal-dir", default=str(RESULTS / "uap" / "universal"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--eval-set", default="eval_set.json")
    p.add_argument("--det-workers", type=int, default=8)
    p.add_argument("--dlib-workers", type=int, default=12)
    a = p.parse_args()
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]

    det = DetPool(a.det_workers, 2)          # spawn CPU pools before CUDA starts
    dl = DlibPool(a.dlib_workers)

    from deepfake.model import ProtectionObjective, SimSwapTarget
    target = SimSwapTarget()
    obj = ProtectionObjective(target)
    fn = FaceNetJudge(target.effectiveness, float(target.config.evaluate.facenet_512.threshold))

    out_dir = RESULTS / "eval" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / "records.jsonl"
    done = set()
    if rec_path.exists():
        for line in open(rec_path):
            r = json.loads(line)
            done.add((r["video"], r["variant"]))
    entries = json.load(open(RESULTS / a.eval_set))["videos"]
    if a.limit:
        entries = entries[: a.limit]
    json.dump(vars(a), open(out_dir / "args.json", "w"), indent=1)
    t_start = time.time()

    for vi, e in enumerate(entries):
        todo = [v for v in variants if (e["video"], v) not in done]
        ts = e["eval_frames"]
        if not todo or not ts:
            continue
        stem = Path(e["video"]).stem
        rec = json.load(open(CACHE / "dets" / f"{stem}.json"))
        frames, fps = read_frames(e["video"], rec["n_window"])
        Mdef = [np.asarray(rec["dets"][t]["M"]) for t in ts]
        partner = bgr_to_tensor(cv2.imread(e["partner"]["png"])).cuda()
        n = len(ts)
        x_clean = torch.stack([bgr_to_tensor(crop_cv2(frames[t], M)) for t, M in zip(ts, Mdef)]).cuda()
        ref = obj.reference(x_clean)
        fn_ref, fn_par = fn.embed_many(x_clean), fn.embed(partner)
        ref_job = dl.submit([to_u8(x) for x in x_clean] + [to_u8(partner)])
        pending = []

        for v in todo:
            per_frame = VARIANTS[v] is not None and VARIANTS[v][0] == "per_frame"
            if per_frame and a.codec != "none":
                print(f"[eval] {v}: per-frame protection covers the scored frames only; "
                      f"skipped for --codec {a.codec}", flush=True)
                continue
            if a.codec == "none":
                if v == "clean":
                    prot = [frames[t] for t in ts]
                elif per_frame:
                    prot = load_per_frame(v, stem, ts)
                else:
                    prot = protect([frames[t] for t in ts], Mdef, load_clip(v, stem, Path(a.universal_dir)), ts)
            else:      # protect the whole window, then the platform re-encodes the upload
                all_t = [t for t in range(rec["n_window"]) if rec["dets"][t]]
                full = list(frames)
                if v != "clean":
                    pf = protect([frames[t] for t in all_t], [np.asarray(rec["dets"][t]["M"]) for t in all_t],
                                 load_clip(v, stem, Path(a.universal_dir)), all_t)
                    for t, f in zip(all_t, pf):
                        full[t] = f
                full = codec_roundtrip(full, fps)
                prot = [full[t] for t in ts]

            if v == "clean" and a.codec == "none":
                found = [True] * n
                Matt = Mdef
            else:
                dets = det.detect(prot)
                found = [d is not None for d in dets]
                Matt = [np.asarray(d["M"]) if d is not None else M for d, M in zip(dets, Mdef)]
            shift = [corner_shift(ma, md) for ma, md in zip(Matt, Mdef)]
            xt = torch.stack([bgr_to_tensor(crop_cv2(f, M)) for f, M in zip(prot, Matt)]).cuda()
            xt_or = torch.stack([bgr_to_tensor(crop_cv2(f, M)) for f, M in zip(prot, Mdef)]).cuda()
            par = partner.expand(n, -1, -1, -1).contiguous()
            outs = {"id": target.swap(xt, par), "ctx": target.swap(par, xt)}
            if a.oracle:
                outs["id_or"] = target.swap(xt_or, par)
                outs["ctx_or"] = target.swap(par, xt_or)

            fnd = {}
            for k, o in outs.items():
                embs = fn.embed_many(o)
                refs = fn_ref if k.startswith("id") else [fn_par] * n
                fnd[k] = [fn.distance(x, r) for x, r in zip(embs, refs)]
            fnd["xt"] = [fn.distance(x, r) for x, r in zip(fn.embed_many(xt), fn_ref)]
            jobs = {k: dl.submit([to_u8(o) for o in outs[k]]) for k in outs}
            jobs["xt"] = dl.submit([to_u8(x) for x in xt])

            with torch.no_grad():
                d_id, d_ctx = obj.deviations(xt, ref)
            util_crop = target.utility.calculate_utility(x_clean, xt_or)
            util_frame = {"psnr": [], "ssim": [], "mse": []}
            for t, f in zip(ts, prot):
                c = frames[t]
                mse = float(np.mean((c.astype(np.float64) - f.astype(np.float64)) ** 2))
                util_frame["mse"].append(mse)
                util_frame["psnr"].append(float(skm.peak_signal_noise_ratio(c, f, data_range=255)) if mse > 0 else None)
                util_frame["ssim"].append(float(skm.structural_similarity(c, f, channel_axis=2, data_range=255)))
            pending.append((v, {
                "video": e["video"], "variant": v, "frames": ts, "codec": a.codec,
                "found": found, "shift_px": shift, "fn": fnd,
                "dev": {"d_id": d_id.cpu().tolist(), "d_ctx": d_ctx.cpu().tolist()},
                "util_crop": {k: float(x) for k, x in util_crop.items()}, "util_frame": util_frame,
            }, jobs))

        refs_enc = ref_job.get()
        ref_enc, par_enc = refs_enc[:-1], refs_enc[-1]
        with open(rec_path, "a") as f:
            for v, payload, jobs in pending:
                enc = {k: j.get() for k, j in jobs.items()}
                dlr = {}
                for k, es in enc.items():
                    if k.startswith("id") or k == "xt":
                        dlr[k] = [dlib_match(x, r) for x, r in zip(es, ref_enc)]
                    else:
                        dlr[k] = [dlib_match(x, par_enc) for x in es]
                dlr["noface"] = {k: [x is None for x in es] for k, es in enc.items()}
                dlr["ref_noface"] = [x is None for x in ref_enc] + [par_enc is None]
                payload["dl"] = dlr
                f.write(json.dumps(payload) + "\n")
        el = time.time() - t_start
        print(f"[eval {a.name}] {vi+1}/{len(entries)} {stem} ({len(todo)} variants) {el/60:.1f} min", flush=True)
        torch.cuda.empty_cache()

    det.close()
    dl.close()
    print(f"[eval {a.name}] done in {(time.time()-t_start)/60:.1f} min -> {rec_path}", flush=True)


if __name__ == "__main__":
    main()
