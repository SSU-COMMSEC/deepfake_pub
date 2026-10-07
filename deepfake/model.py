"""PhantomSeal's SimSwap target and PhantomSeal's protection objective, loaded from the
unmodified PhantomSeal code base (~/deepfake/PhantomSeal).

What is taken from PhantomSeal (src/simswap/base.py, src/simswap/defense.py, config):
  * the deepfake model: SimSwap 224 ("people"), ArcFace identity extractor, netG encoder
  * identity extractor   id(x)  = Base._get_imgs_identity (ArcFace, nearest 224->112, L2-normalised)
  * context extractor    E(x)   = netG.encoder  (first_layer, down1..3)
  * the two deviations PhantomSeal maximises (Eq. 3/4 and _perturb_imgs):
        identity  D_id  = mean((id(x~) - id(x))^2), capped at delta_id  = 0.003
        context   D_ctx = mean((E(x~)  - E(x))^2),  capped at delta_ctx = 25
  * the evaluation code (Utility: MSE/PSNR/SSIM/LPIPS, Effectiveness: FaceNet-512 + dlib)

The universal setting changes one thing: the per-image utility term is replaced by a hard
L_inf budget (the UAP is a fixed-amplitude signal), and the two capped deviations are
normalised by their caps so neither dominates:  score = (min(D_id,d_id)/d_id + min(D_ctx,d_ctx)/d_ctx)/2
"""

import logging
from dataclasses import dataclass

import torch
from hydra import compose, initialize_config_dir

from deepfake.paths import PHANTOMSEAL, RESULTS, setup_imports

setup_imports()


def phantomseal_config(extra: list[str] | None = None):
    run_dir = RESULTS / "phantomseal_runtime"
    (run_dir / "image").mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(config_dir=str(PHANTOMSEAL / "config"), version_base=None):
        return compose(config_name="config", overrides=[
            "third_party=simswap", "evaluate=evaluate_local", f"root_dir={PHANTOMSEAL}",
            f"log_dir={run_dir}", f"image_dir={run_dir}/image", f"notes_path={run_dir}/notes.txt",
            *(extra or [])])


class SimSwapTarget:
    """Thin handle on PhantomSeal's `Base` (SimSwap + extractors + PhantomSeal evaluators)."""

    def __init__(self, logger: logging.Logger | None = None):
        from src.common_utils import cd
        from src.simswap.base import Base

        self.config = phantomseal_config()
        with cd(PHANTOMSEAL):      # PhantomSeal resolves data/ paths (decoy pool) from its own root
            self.base = Base(logger or logging.getLogger("deepfake"), self.config)
        for p in self.base.target.parameters():
            p.requires_grad_(False)
        d = self.config.third_party.defense
        self.delta_id = float(d.limit.identity)     # 0.003
        self.delta_ctx = float(d.limit.context)     # 25

    # -- the two extractors PhantomSeal attacks
    def identity(self, x):
        return self.base._get_imgs_identity(x)

    def context(self, x):
        return self.base.target.netG.encoder(x)

    # -- the attacker's generator
    @torch.no_grad()
    def swap(self, source, target, batch: int = 32):
        outs = []
        for i in range(0, source.shape[0], batch):
            outs.append(self.base.swap_face(source[i:i + batch], target[i:i + batch]))
        return torch.cat(outs)

    @property
    def utility(self):
        return self.base.utility

    @property
    def effectiveness(self):
        return self.base.effectiveness


@dataclass
class Reference:
    ident: torch.Tensor
    ctx: torch.Tensor


class ProtectionObjective:
    """PhantomSeal's identity + context deviation, evaluated on the crops the attacker sees."""

    def __init__(self, target: SimSwapTarget, w_id: float = 1.0, w_ctx: float = 1.0):
        self.t = target
        self.w_id, self.w_ctx = w_id, w_ctx

    @torch.no_grad()
    def reference(self, clean_crops, chunk: int = 64) -> Reference:
        ids, ctxs = [], []
        for i in range(0, clean_crops.shape[0], chunk):
            ids.append(self.t.identity(clean_crops[i:i + chunk]))
            ctxs.append(self.t.context(clean_crops[i:i + chunk]))
        return Reference(torch.cat(ids), torch.cat(ctxs))

    def deviations(self, prot_crops, ref: Reference):
        d_id = ((self.t.identity(prot_crops) - ref.ident) ** 2).flatten(1).mean(1)
        d_ctx = ((self.t.context(prot_crops) - ref.ctx) ** 2).flatten(1).mean(1)
        return d_id, d_ctx

    def score(self, prot_crops, ref: Reference):
        """Per-sample protection score in [0,1] (1 = both deviations reach PhantomSeal's caps)."""
        d_id, d_ctx = self.deviations(prot_crops, ref)
        s_id = torch.clamp(d_id, max=self.t.delta_id) / self.t.delta_id
        s_ctx = torch.clamp(d_ctx, max=self.t.delta_ctx) / self.t.delta_ctx
        s = (self.w_id * s_id + self.w_ctx * s_ctx) / (self.w_id + self.w_ctx)
        return s, {"d_id": d_id.detach(), "d_ctx": d_ctx.detach(), "s_id": s_id.detach(), "s_ctx": s_ctx.detach()}
