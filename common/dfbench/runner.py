"""Uniform, resumable driver every attack in the benchmark runs under.

The point of this class is that *no attack gets to report its own numbers*.
Success, query count and distortion are all measured by this harness through
the shared C3DVictim, so a method cannot accidentally (or conveniently) define
success differently from its five competitors.

Unified success criterion, applied identically to all six methods:

    the adversarial clip is misclassified by the victim
    AND  ||x_adv - x_clean||_inf  <=  EPS      (pixel units, 0-255)

`queries_to_success` is the index of the FIRST query that satisfied both
conditions, recorded inside the victim as the attack runs.
"""
import json
import os
import time
import traceback

import numpy as np
import torch

from .paths import EVAL_SUBSET, VIDEO_DIR, RESULT_DIR

TARGETS_PATH = os.path.join(RESULT_DIR, "00_victim", "targets.json")
from .data import load_video_clips
from .records import RecordStore, new_attack_record, atomic_write_json
from .victim import C3DVictim, QueryBudgetExceeded
from . import metrics as M

# L_inf budget shared by every method, in 0-255 pixel units.
# Chosen as Geo-TRAP's own default (--max_p 10, applied in mean-subtracted space
# where std=1, hence pixel units).  V-BAD's default eps=0.05 in [0,1] == 12.75
# pixel units, so 10 sits inside the range the original papers use.
DEFAULT_EPS = 10.0   # L_inf budget in 0-255 pixel units (== Geo-TRAP max_p=10)
DEFAULT_BUDGET = 60000


def load_subset(tier=None):
    with open(EVAL_SUBSET) as f:
        d = json.load(f)
    vids = d["videos"]
    if tier is not None:
        vids = [v for v in vids if v["tier"] <= tier]
    return d, vids


class AttackRunner:
    def __init__(self, name, tier=1, eps=DEFAULT_EPS, budget=DEFAULT_BUDGET,
                 out_dir=None, max_batch=64, device="cuda", targeted=False,
                 save_adv=None):
        # save_adv: 채점에 쓴 적대적 클립을 uint8 로 이 디렉터리에 저장한다.
        # 영상 프레임이 uint8 이므로 실제 주입되는 영상이 바로 이 형태다.
        # 엣지 기기(Jetson)에서 같은 입력으로 재추론하기 위한 것.
        self.save_adv = save_adv
        if save_adv:
            os.makedirs(save_adv, exist_ok=True)
        self.name = name
        self.targeted = bool(targeted)
        # 표적 공격이면 영상마다 미리 배정된 목표 클래스를 읽는다.
        # 시드 고정 파일이라 모든 방법이 동일한 표적을 받는다 (scripts/make_targets.py).
        self.targets = None
        if self.targeted:
            with open(TARGETS_PATH) as f:
                self.targets = json.load(f)["targets"]
        self.eps = float(eps)
        self.budget = int(budget)
        self.out_dir = out_dir or os.path.join(RESULT_DIR, name)
        os.makedirs(self.out_dir, exist_ok=True)
        self.store = RecordStore(os.path.join(self.out_dir, "records.jsonl"))
        self.meta, self.videos = load_subset(tier)
        _n = os.environ.get("DF_CAL_N")      # eps 보정용 소규모 실행
        if _n:
            self.videos = self.videos[:int(_n)]
        self.tier = tier
        self.victim = C3DVictim(max_batch=max_batch, device=device,
                                query_budget=None)

    # ------------------------------------------------------------------ core
    def _success_test_factory(self, clean_clip, label, target=None):
        clean = torch.as_tensor(clean_clip, device=self.victim.device,
                                dtype=torch.float32)
        if clean.dim() == 5:
            clean = clean[0]
        eps = self.eps
        tgt = None if target is None else int(target)

        def test(logits_row, clip):
            pred = int(logits_row.argmax())
            if tgt is None:                       # 미표적: 정답만 아니면 된다
                if pred == int(label):
                    return False
            else:                                 # 표적: 지정 클래스여야 한다
                if pred != tgt:
                    return False
            return float((clip - clean).abs().max()) <= eps + 1e-4
        return test

    def run(self, attack_fn, note=""):
        """attack_fn(victim, clean_clip, label, eps, budget) -> adv_clip | None

        clean_clip is float32 (3,16,112,112) RGB in [0,255].
        The function may query `victim` freely; every query is counted.
        """
        done = self.store.done_videos()
        todo = [v for v in self.videos if v["video"] not in done]
        print(f"[{self.name}] tier<={self.tier}  eps={self.eps:.3f}  "
              f"budget={self.budget}  total={len(self.videos)}  "
              f"done={len(done)}  todo={len(todo)}", flush=True)
        t_start = time.time()

        for i, item in enumerate(todo):
            rel, label = item["video"], item["label"]
            rec = new_attack_record(rel, label, item["index"])
            rec["tier"] = item.get("tier")
            t0 = time.time()
            try:
                clean = load_video_clips(os.path.join(VIDEO_DIR, rel), num_clips=1)[0]
                self.victim.reset()
                self.victim.query_budget = self.budget
                tgt = (int(self.targets[rel]) if self.targets is not None else None)
                rec["target_label"] = tgt
                self.victim.attack_target = tgt      # 어댑터가 읽는다
                self.victim.arm(label, self._success_test_factory(clean, label, tgt))
                try:
                    adv = attack_fn(self.victim, clean, label, self.eps, self.budget)
                except QueryBudgetExceeded:
                    adv = None
                rec.update(self._score(clean, adv, label, tgt))
                if self.save_adv:
                    self._dump_adv(rel, clean, adv)
            except Exception as e:
                rec["status"] = "error"
                rec["error"] = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            rec["wall_seconds"] = round(time.time() - t0, 3)
            rec["queries_used"] = int(self.victim.n_queries)
            rec["queries_to_misclass"] = self.victim.first_misclass_query
            if rec.get("queries_to_success") is None:
                rec["queries_to_success"] = self.victim.first_success_query
            if rec["queries_to_success"] is not None:
                rec["success"] = True
            self.store.append(rec)
            self._progress(i + 1, len(todo), t_start)
        self.store.close()
        self.summarise(note)

    def _dump_adv(self, rel, clean, adv):
        """채점 대상 클립을 uint8 로 저장한다 (없으면 폴백 클립을 따른다)."""
        import numpy as _np
        if adv is None:
            adv = self.victim.first_success_clip
        if adv is None:
            adv = self.victim.first_misclass_clip
        if adv is None:
            return
        a = torch.as_tensor(adv, dtype=torch.float32)
        if a.dim() == 5:
            a = a[0]
        a = a.clamp(0, 255).round().to(torch.uint8).cpu().numpy()
        key = rel.replace("/", "__").replace(".avi", "")
        _np.save(os.path.join(self.save_adv, key + ".npy"), a)

    def _score(self, clean, adv, label, target=None):
        out = {}
        if adv is None:
            # The attack handed back nothing (budget exhausted mid-run, or it
            # only reports failure).  If it nonetheless passed through a valid
            # adversarial example, score THAT one, so a video counted as a
            # success always contributes a MAP.
            adv = self.victim.first_success_clip
            if adv is None:
                # eps 조건은 못 맞췄지만 오분류는 됐다면 그 클립이라도 채점한다.
                # 그래야 FR_0(무제약) 과 MAP_0 이 결측이 되지 않는다.
                adv = self.victim.first_misclass_clip
            if adv is None:
                out["success"] = self.victim.first_success_query is not None
                return out
        adv_t = torch.as_tensor(adv, device=self.victim.device, dtype=torch.float32)
        if adv_t.dim() == 4:
            adv_t = adv_t.unsqueeze(0)
        adv_t = adv_t.clamp(0, 255)
        clean_t = torch.as_tensor(clean, device=self.victim.device,
                                  dtype=torch.float32).unsqueeze(0)
        d = (adv_t - clean_t)
        logits = self.victim.query(adv_t, count=False)
        pred = int(logits.argmax(1)[0])
        linf = float(d.abs().max())
        hit = (pred == int(target)) if target is not None else (pred != int(label))
        out.update({
            "final_pred": pred,
            "success": hit and (linf <= self.eps + 1e-4),
            "map": float(d.abs().mean()),          # ||d||_1 / numel, 0-255 units
            "linf": linf,
            "l2": float(d.pow(2).sum().sqrt()),
            "n_elements": int(d.numel()),
        })
        return out

    def _progress(self, i, n, t_start):
        el = time.time() - t_start
        per = el / max(i, 1)
        eta = per * (n - i)
        recs = self.store.load()
        ok = [r for r in recs if r.get("status") == "done"]
        fr = 100.0 * sum(1 for r in ok if r.get("success")) / max(len(ok), 1)
        print(f"[{self.name}] {i}/{n}  FR={fr:5.1f}%  "
              f"{per:6.1f}s/video  elapsed {el/3600:5.2f}h  ETA {eta/3600:5.2f}h",
              flush=True)

    def summarise(self, note=""):
        recs = self.store.load()
        summary = {
            "method": self.name,
            "note": note,
            "tier": self.tier,
            "targeted": self.targeted,
            "eps_pixels_0_255": self.eps,
            "query_budget": self.budget,
            "metrics": M.compute(recs, budget=self.budget),
            "fr_vs_budget": M.fr_vs_budget(recs),
        }
        atomic_write_json(os.path.join(self.out_dir, "summary.json"), summary)
        print(json.dumps(summary, indent=2), flush=True)
        return summary
