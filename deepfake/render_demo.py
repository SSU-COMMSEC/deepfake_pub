"""Step 7 - comparison videos: before/after protection and what SimSwap makes of each.

One grid video per demo test video:
  columns : original (no protection) | U3D per-video | U3D universal | C-DUP per-video | C-DUP universal | 2D-DUP universal
  rows    : 1 the uploaded video (after protection)
            2 the added noise (x8)
            3 context-theft deepfake - attacker's face swapped into this video (pasted back)
            4 identity-theft deepfake - this person's face taken from the video and put on the attacker's
  every deepfake cell carries PhantomSeal's FaceNet-512 verdict: red = deepfake succeeded,
  green = blocked. The attacker is SimSwap's video pipeline (re-detects the face on the upload).

    python -m deepfake.render_demo [--videos stem1,stem2] [--frames 100]
Output: results/demo/<stem>/compare_grid.{webm,mp4}, compare_frames.png
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from deepfake.common import EPS
from deepfake.detpool import DetPool
from deepfake.evaluate import load_clip, protect
from deepfake.face import bgr_to_tensor, crop_cv2, paste_back
from deepfake.paths import CACHE, RESULTS
from deepfake.ucf import read_frames, write_previews

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
COLS = [("clean", "original (no protection)"), ("u3d_pv", "U3D, per-video"), ("u3d_uap", "U3D, universal UAP"),
        ("cdup3d_pv", "C-DUP, per-video"), ("cdup3d_uap", "C-DUP, universal UAP"),
        ("cdup2d_uap", "2D-DUP, universal UAP")]
ROWS = ["uploaded video", "added noise x8", "context theft\nattacker's face\nswapped into this video",
        "identity theft\nuploader's face\nswapped onto the attacker"]
CW, CH, LW, HH, FH = 320, 240, 230, 44, 64
RED, GREEN, GRAY = (220, 40, 40), (30, 160, 60), (120, 120, 120)


def font(size):
    return ImageFont.truetype(FONT, size, index=1)


def label_box(img_rgb, text, color):
    im = Image.fromarray(img_rgb)
    d = ImageDraw.Draw(im)
    f = font(15)
    w = d.textlength(text, font=f)
    d.rectangle([0, CH - 24, w + 10, CH], fill=color)
    d.text((5, CH - 23), text, font=f, fill=(255, 255, 255))
    return np.asarray(im)


def fit_cell(img_rgb):
    h, w = img_rgb.shape[:2]
    s = min(CW / w, CH / h)
    r = cv2.resize(img_rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    cell = np.full((CH, CW, 3), 20, np.uint8)
    y, x = (CH - r.shape[0]) // 2, (CW - r.shape[1]) // 2
    cell[y:y + r.shape[0], x:x + r.shape[1]] = r
    return cell


def compose(cells, title, partner_rgb, t, victim_rgb=None, descs=("", "")):
    ncol = len(COLS)
    canvas = Image.new("RGB", (LW + ncol * CW, HH + len(ROWS) * CH + FH), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    d.text((8, 10), f"frame {t}", font=font(16), fill=(0, 0, 0))
    for j, (_, name) in enumerate(COLS):
        d.text((LW + j * CW + 8, 10), name, font=font(19), fill=(0, 0, 0))
    for i, rname in enumerate(ROWS):
        d.multiline_text((8, HH + i * CH + 8), rname, font=font(15), fill=(0, 0, 0), spacing=3)
        for j in range(ncol):
            canvas.paste(Image.fromarray(cells[i][j]), (LW + j * CW, HH + i * CH))
    th = 96
    # row 3: who should NOT appear in this video
    y3 = HH + 2 * CH + 84
    canvas.paste(Image.fromarray(cv2.resize(partner_rgb, (th, th))), (8, y3))
    d.multiline_text((8 + th + 6, y3), "attacker\n" + descs[1].replace("·", "\n"), font=font(13), fill=(0, 0, 0), spacing=2)
    # row 4: whose face should NOT be transplanted onto the attacker
    y4 = HH + 3 * CH + 84
    if victim_rgb is not None:
        canvas.paste(Image.fromarray(cv2.resize(victim_rgb, (th, th))), (8, y4))
        d.multiline_text((8 + th + 6, y4), "uploader\n" + descs[0], font=font(13), fill=(0, 0, 0), spacing=2)
    y = HH + len(ROWS) * CH + 8
    d.text((8, y), title, font=font(15), fill=(0, 0, 0))
    d.text((8, y + 24), "red = deepfake succeeded (defence failed) · green = blocked · judge: FaceNet-512 (PhantomSeal threshold L2 ≤ 1.04). "
           "Context theft succeeds if the output is recognised as the attacker, identity theft if as the uploader. "
           "Attacker = SimSwap video pipeline (re-detects the face on the protected video)", font=font(14), fill=(60, 60, 60))
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default="")
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--spec", default="", help="JSON list of {video, attacker, victim_desc, attacker_desc}")
    p.add_argument("--out-name", default="demo")
    a = p.parse_args()

    det = DetPool(8, 2)
    from deepfake.judges import FaceNetJudge
    from deepfake.model import SimSwapTarget
    target = SimSwapTarget()
    fn = FaceNetJudge(target.effectiveness, float(target.config.evaluate.facenet_512.threshold))

    entries = {}
    for name in ("eval_set.json", "eval_set_extra.json"):
        if (RESULTS / name).exists():
            entries.update({Path(e["video"]).stem: e for e in json.load(open(RESULTS / name))["videos"]})
    spec = {x["video"]: x for x in json.load(open(a.spec))} if a.spec else {}
    stems = [s for s in a.videos.split(",") if s] or list(spec) or json.load(open(RESULTS / "demo_videos.json"))
    for stem in stems:
        e = dict(entries[stem])
        sp = spec.get(stem, {})
        if sp:
            e["partner"] = {"png": str(Path(a.spec).parent / sp["attacker"])}
        frames, fps = read_frames(e["video"], a.frames)
        n = len(frames)
        cache = json.load(open(CACHE / "dets" / f"{stem}.json"))["dets"]
        extra = det.detect(frames[len(cache):]) if n > len(cache) else []
        dets = (cache + extra)[:n]
        ts = [t for t in range(n) if dets[t] is not None]
        Mdef = {t: np.asarray(dets[t]["M"]) for t in ts}
        partner_bgr = cv2.imread(e["partner"]["png"])
        partner = bgr_to_tensor(partner_bgr).cuda()
        ref = {t: fn.embed(bgr_to_tensor(crop_cv2(frames[t], Mdef[t])).cuda()) for t in ts}
        ref_par = fn.embed(partner)
        victim_rgb = cv2.cvtColor(crop_cv2(frames[ts[0]], Mdef[ts[0]]), cv2.COLOR_BGR2RGB)
        cols = {}
        for v, _ in COLS:
            if v == "clean":
                prot = {t: frames[t] for t in range(n)}
            else:
                P = load_clip(v, stem, RESULTS / "uap" / "universal")
                pf = protect([frames[t] for t in ts], [Mdef[t] for t in ts], P, ts, EPS)
                prot = {t: frames[t] for t in range(n)}
                prot.update(dict(zip(ts, pf)))
            found = det.detect([prot[t] for t in ts]) if v != "clean" else [{"M": Mdef[t].tolist()} for t in ts]
            Matt = {t: (np.asarray(d["M"]) if d is not None else Mdef[t]) for t, d in zip(ts, found)}
            xt = torch.stack([bgr_to_tensor(crop_cv2(prot[t], Matt[t])) for t in ts]).cuda()
            par = partner.expand(len(ts), -1, -1, -1).contiguous()
            ctx = target.swap(par, xt)
            idt = target.swap(xt, par)
            res = {}
            for k, t in enumerate(ts):
                d_ctx = fn.distance(fn.embed(ctx[k]), ref_par)
                d_id = fn.distance(fn.embed(idt[k]), ref[t])
                ctx_rgb = (ctx[k].permute(1, 2, 0).cpu().numpy())
                pasted = paste_back(ctx_rgb, Matt[t], prot[t])
                res[t] = (pasted, (idt[k].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), d_ctx, d_id)
            cols[v] = (prot, res)
            print(f"[demo {stem}] {v} done", flush=True)

        out = RESULTS / a.out_name / stem
        out.mkdir(parents=True, exist_ok=True)
        title = (f"{e['video']} (UCF-101 split-1 test) · noise budget L∞ ≤ 10/255 · universal UAP = made once from UCF-101 train faces, "
                 f"per-video = optimised on this video's own frames")
        grid_frames, stills = [], []
        for t in range(n):
            cells = [[None] * len(COLS) for _ in ROWS]
            for j, (v, _) in enumerate(COLS):
                prot, res = cols[v]
                rgb = cv2.cvtColor(prot[t], cv2.COLOR_BGR2RGB)
                cells[0][j] = fit_cell(rgb)
                diff = (prot[t].astype(np.int16) - frames[t].astype(np.int16)) * 8 + 128
                cells[1][j] = fit_cell(cv2.cvtColor(np.clip(diff, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
                if t in res:
                    pasted, idt, d_ctx, d_id = res[t]
                    ok_ctx, ok_id = d_ctx <= fn.thr, d_id <= fn.thr
                    c_cell = fit_cell(cv2.cvtColor(pasted, cv2.COLOR_BGR2RGB))
                    cells[2][j] = label_box(c_cell, ("looks like attacker → deepfake succeeded" if ok_ctx else "not the attacker → blocked")
                                            + f"  d={d_ctx:.2f}", RED if ok_ctx else GREEN)
                    cells[3][j] = label_box(fit_cell(idt), ("looks like this person → theft succeeded" if ok_id else "not this person → blocked")
                                            + f"  d={d_id:.2f}", RED if ok_id else GREEN)
                else:
                    blank = np.full((CH, CW, 3), 40, np.uint8)
                    cells[2][j] = label_box(blank, "no face detected", GRAY)
                    cells[3][j] = label_box(blank.copy(), "no face detected", GRAY)
            g = compose(cells, title, cv2.cvtColor(partner_bgr, cv2.COLOR_BGR2RGB), t, victim_rgb,
                        (sp.get("victim_desc", ""), sp.get("attacker_desc", "")))
            grid_frames.append(g)
            if t in (ts[0], ts[len(ts) // 2], ts[-1]):
                stills.append(g)
        write_previews(out / "compare_grid", grid_frames, fps)
        cv2.imwrite(str(out / "compare_frames.png"), np.concatenate(stills, axis=0))
        print(f"[demo] {stem}: {out}/compare_grid.webm ({len(grid_frames)} frames)", flush=True)
    det.close()


if __name__ == "__main__":
    main()
