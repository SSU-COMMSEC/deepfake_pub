"""Step 6 - turn records.jsonl into PhantomSeal-style tables.

    python -m deepfake.summarize --name main      -> results/eval/<name>/summary.{json,md}

Per variant (FaceNet-512 / dlib):
  ASR_id   identity-theft success rate  (output recognised as the victim)        lower = better
  ASR_ctx  context-theft success rate   (output recognised as the attacker)      lower = better
  x~       protected face still recognised as the victim (upload utility)        higher = better
  S        PhantomSeal score without tracing = (1 - ASR_id) + (1 - ASR_ctx)       max 2
  block_id / block_ctx: of the frames where the CLEAN attack succeeds, the share the
           protection turns into a failure (protection rate on attackable frames)
plus attacker re-detection failures, crop shift, utility (crop: PhantomSeal Utility; frame: uint8),
and optimisation cost.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from deepfake.paths import RESULTS

THR = 1.04
LABEL = {
    "clean": "no protection",
    "uniform_uap": "uniform noise (baseline)",
    "u3d_uap": "U3D-based, one universal UAP",
    "u3d_pv": "U3D-based, per-video",
    "u3d_repo_uap": "U3D-based (reference impl.), one universal UAP",
    "u3d_repo_pv": "U3D-based (reference impl.), per-video",
    "cdup3d_uap": "C-DUP-based, one universal UAP",
    "cdup3d_pv": "C-DUP-based, per-video",
    "cdup2d_uap": "C-DUP (2D-DUP), one universal UAP",
    "phantomseal_pf": "PhantomSeal, every frame optimised",
    "phantomseal_native_pf": "PhantomSeal, every frame optimised (its own R/G/B budget)",
}
ORDER = list(LABEL)


def rate(xs):
    xs = [x for x in xs if x is not None]
    return 100.0 * np.mean(xs) if xs else float("nan")


def load(name):
    recs = [json.loads(l) for l in open(RESULTS / "eval" / name / "records.jsonl")]
    by = defaultdict(dict)
    for r in recs:
        by[r["variant"]][r["video"]] = r
    return by


def cost_table():
    out = {}
    for m in ("u3d", "u3d_repo", "cdup3d", "cdup2d", "uniform"):
        p = RESULTS / "uap" / "universal" / m / "meta.json"
        if p.exists():
            out[f"{m}_uap"] = {"kind": "universal", "seconds_once": json.load(open(p)).get("seconds")}
    for m in ("u3d", "u3d_repo", "cdup3d"):
        p = RESULTS / "uap" / "per_video" / m / "meta.jsonl"
        if p.exists():
            secs = [json.loads(l)["seconds"] for l in open(p)]
            out[f"{m}_pv"] = {"kind": "per_video", "seconds_per_video": float(np.mean(secs)), "n": len(secs)}
    for m in ("phantomseal", "phantomseal_native"):
        p = RESULTS / "uap" / "per_frame" / m / "meta.jsonl"
        if p.exists():
            secs = [json.loads(l)["seconds"] for l in open(p)]
            out[f"{m}_pf"] = {"kind": "per_video", "seconds_per_video": float(np.mean(secs)), "n": len(secs)}
    return out


def summarize(name, oracle=False):
    by = load(name)
    common = set.intersection(*[set(v) for v in by.values()]) if by else set()
    clean = by.get("clean", {})
    rows = {}
    for v, vids in by.items():
        acc = defaultdict(list)
        for vid in sorted(common):
            r = vids[vid]
            c = clean.get(vid)
            for sfx in ([""] + (["_or"] if oracle else [])):
                for k in ("id", "ctx"):
                    key = k + sfx
                    if key not in r["fn"]:
                        continue
                    fnm = [d <= THR for d in r["fn"][key]]
                    dlm = r["dl"][key]
                    acc[f"fn_{key}"] += fnm
                    acc[f"dl_{key}"] += dlm
                    if c is not None and key in c["fn"]:
                        cf = [d <= THR for d in c["fn"][k]]
                        cd = c["dl"][k]
                        acc[f"fn_block_{key}"] += [not p for p, q in zip(fnm, cf) if q]
                        acc[f"dl_block_{key}"] += [not p for p, q in zip(dlm, cd) if q]
            if v != "clean":
                acc["fn_xt"] += [d <= THR for d in r["fn"]["xt"]]
                acc["dl_xt"] += r["dl"]["xt"]
            acc["found"] += r["found"]
            acc["shift"] += [s for s, f in zip(r["shift_px"], r["found"]) if f]
            acc["d_id"] += r["dev"]["d_id"]
            acc["d_ctx"] += r["dev"]["d_ctx"]
            for k, x in r["util_crop"].items():
                acc[f"crop_{k}"].append(x)
            acc["frame_psnr"] += [x for x in r["util_frame"]["psnr"] if x is not None]
            acc["frame_ssim"] += r["util_frame"]["ssim"]
        row = {"n_videos": len(common), "n_frames": len(acc["found"])}
        for j in ("fn", "dl"):
            for sfx in ([""] + (["_or"] if oracle else [])):
                row[f"{j}_ASR_id{sfx}"] = rate(acc[f"{j}_id{sfx}"])
                row[f"{j}_ASR_ctx{sfx}"] = rate(acc[f"{j}_ctx{sfx}"])
                row[f"{j}_S{sfx}"] = (200 - row[f"{j}_ASR_id{sfx}"] - row[f"{j}_ASR_ctx{sfx}"]) / 100
                row[f"{j}_block_id{sfx}"] = rate(acc[f"{j}_block_id{sfx}"])
                row[f"{j}_block_ctx{sfx}"] = rate(acc[f"{j}_block_ctx{sfx}"])
            row[f"{j}_xt"] = rate(acc[f"{j}_xt"]) if v != "clean" else None
        row["redetect_fail"] = 100 - rate(acc["found"])
        row["shift_px"] = float(np.mean(acc["shift"])) if acc["shift"] else 0.0
        row["d_id"], row["d_ctx"] = float(np.mean(acc["d_id"])), float(np.mean(acc["d_ctx"]))
        for k in ("mse", "psnr", "ssim", "lpips"):
            vals = [x for x in acc[f"crop_{k}"] if np.isfinite(x)]
            row[f"crop_{k}"] = float(np.mean(vals)) if vals else None
        row["frame_psnr"] = float(np.mean(acc["frame_psnr"])) if acc["frame_psnr"] else None
        row["frame_ssim"] = float(np.mean(acc["frame_ssim"]))
        rows[v] = row
    return rows


def fmt(x, d=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{d}f}"


def to_markdown(name, rows, costs, oracle):
    order = [v for v in ORDER if v in rows]
    n = rows[order[0]]
    L = [f"# Evaluation results ({name})", "",
         f"{n['n_videos']} test videos, {n['n_frames']} scored frames. Values are FaceNet-512 / dlib (Face Recognition), %.", "",
         "## Deepfake blocking (attacker = SimSwap video pipeline, re-detecting the face on the protected video)", "",
         "| Method | identity-theft ASR ↓ | context-theft ASR ↓ | block rate (identity) ↑ | block rate (context) ↑ | S ↑ (max 2) | x̃ ↑ |",
         "|---|---|---|---|---|---|---|"]
    for v in order:
        r = rows[v]
        L.append(f"| {LABEL.get(v, v)} | {fmt(r['fn_ASR_id'])} / {fmt(r['dl_ASR_id'])} | "
                 f"{fmt(r['fn_ASR_ctx'])} / {fmt(r['dl_ASR_ctx'])} | "
                 f"{fmt(r['fn_block_id'])} / {fmt(r['dl_block_id'])} | {fmt(r['fn_block_ctx'])} / {fmt(r['dl_block_ctx'])} | "
                 f"{fmt(r['fn_S'], 2)} / {fmt(r['dl_S'], 2)} | {fmt(r['fn_xt'])} / {fmt(r['dl_xt'])} |")
    if oracle:
        L += ["", "## Paper protocol (attacker reuses the defender's alignment, oracle)", "",
              "| Method | identity-theft ASR ↓ | context-theft ASR ↓ | S ↑ |", "|---|---|---|---|"]
        for v in order:
            r = rows[v]
            L.append(f"| {LABEL.get(v, v)} | {fmt(r['fn_ASR_id_or'])} / {fmt(r['dl_ASR_id_or'])} | "
                     f"{fmt(r['fn_ASR_ctx_or'])} / {fmt(r['dl_ASR_ctx_or'])} | {fmt(r['fn_S_or'], 2)} / {fmt(r['dl_S_or'], 2)} |")
    L += ["", "## Quality and cost", "",
          "| Method | frame PSNR ↑ | frame SSIM ↑ | face crop PSNR / SSIM / LPIPS | re-detection failure % | crop shift px | optimisation cost |",
          "|---|---|---|---|---|---|---|"]
    for v in order:
        r = rows[v]
        c = costs.get(v, {})
        cost = ("–" if v in ("clean",) else
                f"{c['seconds_per_video']:.0f} s per video" if c.get("kind") == "per_video" else
                f"{c.get('seconds_once', 0)/60:.1f} min once, then apply only" if c else "–")
        L.append(f"| {LABEL.get(v, v)} | {fmt(r['frame_psnr'], 2)} | {fmt(r['frame_ssim'], 3)} | "
                 f"{fmt(r['crop_psnr'], 2)} / {fmt(r['crop_ssim'], 3)} / {fmt(r['crop_lpips'], 3)} | "
                 f"{fmt(r['redetect_fail'])} | {fmt(r['shift_px'], 2)} | {cost} |")
    L += ["", "## Feature deviation (on the attacker's crop; PhantomSeal caps: identity 0.003, context 25)", "",
          "| Method | identity deviation D_id | context deviation D_ctx |", "|---|---|---|"]
    for v in order:
        r = rows[v]
        L.append(f"| {LABEL.get(v, v)} | {r['d_id']:.5f} | {r['d_ctx']:.3f} |")
    return "\n".join(L) + "\n"


def by_face_size(name, bins=((0, 60), (60, 100), (100, 1000))):
    """FaceNet ASR per face-width bin (defender's detection width on the clean frame)."""
    from deepfake.paths import CACHE
    by = load(name)
    width = {}
    for vid in next(iter(by.values())):
        rec = json.load(open(CACHE / "dets" / f"{Path(vid).stem}.json"))
        width[vid] = {t: d["bbox"][2] - d["bbox"][0] for t, d in enumerate(rec["dets"]) if d}
    out = {}
    for v, vids in by.items():
        for lo, hi in bins:
            idm, ctm, dli, dlc = [], [], [], []
            for vid, r in vids.items():
                for k, t in enumerate(r["frames"]):
                    if lo <= width[vid].get(t, 0) < hi:
                        idm.append(r["fn"]["id"][k] <= THR)
                        ctm.append(r["fn"]["ctx"][k] <= THR)
                        dli.append(r["dl"]["id"][k])
                        dlc.append(r["dl"]["ctx"][k])
            out.setdefault(v, {})[f"{lo}-{hi}"] = {"n": len(idm), "fn_id": rate(idm), "fn_ctx": rate(ctm),
                                                 "dl_id": rate(dli), "dl_ctx": rate(dlc)}
    return out


def size_markdown(sizes):
    order = [v for v in ORDER if v in sizes]
    keys = list(next(iter(sizes.values())).keys())
    L = ["", "## By face size (defender-detected face width, px) — identity-theft ASR / context-theft ASR (FaceNet, %)", "",
         "| Method | " + " | ".join(f"{k}px (n={sizes[order[0]][k]['n']})" for k in keys) + " |",
         "|---|" + "---|" * len(keys)]
    for v in order:
        L.append(f"| {LABEL.get(v, v)} | " + " | ".join(
            f"{fmt(sizes[v][k]['fn_id'])} / {fmt(sizes[v][k]['fn_ctx'])}" for k in keys) + " |")
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="main")
    p.add_argument("--oracle", action="store_true")
    a = p.parse_args()
    rows = summarize(a.name, a.oracle)
    costs = cost_table()
    sizes = by_face_size(a.name)
    out = RESULTS / "eval" / a.name
    json.dump({"rows": rows, "costs": costs, "by_face_size": sizes}, open(out / "summary.json", "w"), indent=1)
    md = to_markdown(a.name, rows, costs, a.oracle) + size_markdown(sizes)
    open(out / "summary.md", "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
