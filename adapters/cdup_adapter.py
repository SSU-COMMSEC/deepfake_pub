#!/usr/bin/env python
"""C-DUP (Li et al., NDSS'19 "Stealthy Adversarial Perturbations Against
Real-Time Video Classification Systems") against the shared C3D victim.

The upstream repo is TensorFlow 1.x, which has no sm_120 kernels and therefore
cannot execute a single op on this machine's RTX 5090.  The generator and its
training objective are consequently ported to PyTorch **layer for layer** from
`generator.py`, `model.py`, `para_model.py`, keeping every hyper-parameter:

    generator_3D  (para_model: z_size=100, k_size=3)
      z ~ N(0,I) (B,100) -> (B,100,1,1,1)
      deconv 1x7x7  100->512  stride 1 VALID  -> BN(eps=1e-5,decay=0.1) -> ReLU
      deconv 3x3x3  512->256  stride 2 SAME   -> BN -> ReLU
      deconv 3x3x3  256->128  stride 2 SAME   -> BN -> ReLU
      deconv 3x3x3  128->64   stride 2 SAME   -> BN -> ReLU
      deconv 3x3x3  64 ->3    stride 2 SAME   -> tanh
      -> (B,3,16,112,112) in [-1,1];  p = p_max * out,  p_max = 10
      then a random temporal roll, as tf.manip.roll(z_out, shift, axis=1)

    objective (model.py)
      loss = mean_b sum_c -onehot * log(clip(1 - softmax(C3D(x+p)), 1e-10, 1))
      Adam(lr = 0.002 * 0.95^(step//2000), beta1 = 0.3), train_epoch = 3

    evaluation (test.py)
      one perturbation is drawn from the trained generator (test_idx=0,
      test_shift=0) and applied to every video.

Two configuration choices are recorded in reports/DEVIATIONS.md:
  * `ratio` is set to 1.0 rather than the shipped 0.5.  With ratio=0.5 the repo
    trains the paper's *class-targeted* stealthy attack (misclassify one chosen
    class, keep every other class correct), whose fooling rate is not comparable
    with the five untargeted methods here.  ratio=1.0 selects the untargeted
    branch of the very same objective.
  * batch size 64 instead of 256; the original splits 256 across 16 GPUs
    (para_model.gpu_number = 16) and this machine has one.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn

# 프로젝트 루트: DF_ROOT 우선 (Jetson 은 ~/bench 로 경로가 다르다)
DF = os.environ.get("DF_ROOT",
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(DF, "common"))

from dfbench.runner import AttackRunner, DEFAULT_EPS                      # noqa: E402
from dfbench.paths import DATASET_DIR, RESULT_DIR                         # noqa: E402
from dfbench.victim import C3DVictim                                      # noqa: E402
from dfbench.records import atomic_write_json                             # noqa: E402

Z_SIZE, K_SIZE, P_MAX = 100, 3, 10          # para_model.py (P_MAX overridable via --eps)
NUM_CLASSES = 101
def ckpt_path(pmax, epochs=3):
    tag = "" if abs(pmax - 10) < 1e-6 else f"_p{pmax:g}"
    # epoch 수가 upstream 기본(3) 과 다르면 별도 파일로 — 기존 결과를 덮지 않는다
    if epochs != 3:
        tag += f"_ep{epochs}"
    return os.path.join(RESULT_DIR, "cdup", f"generator{tag}.pt")
def uap_path(pmax):
    tag = "" if abs(pmax - 10) < 1e-6 else f"_p{pmax:g}"
    return os.path.join(RESULT_DIR, "cdup", f"cdup_perturbation{tag}.npy")


class Generator3D(nn.Module):
    """PyTorch transcription of generator.py:generator_3D.

    TF 'SAME' with kernel 3 / stride 2 doubles each spatial dim; in PyTorch that
    is padding=1 with output_padding=1.  TF batch_norm(decay=0.1) corresponds to
    PyTorch momentum=1-decay=0.9.
    """
    def __init__(self, z_size=Z_SIZE, k=K_SIZE):
        super().__init__()
        bn = lambda c: nn.BatchNorm3d(c, eps=1e-5, momentum=0.9, affine=True)
        self.d1 = nn.ConvTranspose3d(z_size, 512, (1, 7, 7), stride=1, padding=0)
        self.b1 = bn(512)
        self.d2 = nn.ConvTranspose3d(512, 256, k, stride=2, padding=1, output_padding=1)
        self.b2 = bn(256)
        self.d3 = nn.ConvTranspose3d(256, 128, k, stride=2, padding=1, output_padding=1)
        self.b3 = bn(128)
        self.d4 = nn.ConvTranspose3d(128, 64, k, stride=2, padding=1, output_padding=1)
        self.b4 = bn(64)
        self.d5 = nn.ConvTranspose3d(64, 3, k, stride=2, padding=1, output_padding=1)
        for m in self.modules():                       # random_normal_initializer(0.02)
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        x = z.view(z.shape[0], -1, 1, 1, 1)
        x = torch.relu(self.b1(self.d1(x)))
        x = torch.relu(self.b2(self.d2(x)))
        x = torch.relu(self.b3(self.d3(x)))
        x = torch.relu(self.b4(self.d4(x)))
        return torch.tanh(self.d5(x))                  # (B,3,16,112,112) in [-1,1]


def train_clips(n=None, seed=0):
    mm = np.load(os.path.join(DATASET_DIR, "train_clips_u8.npy"), mmap_mode="r")
    idx = np.arange(mm.shape[0])
    if n:
        idx = np.random.RandomState(seed).choice(idx, n, replace=False)
    labels = [int(l.rstrip("\n").split("\t")[2])
              for l in open(os.path.join(DATASET_DIR, "train_clips_index.txt"))]
    return mm, idx, np.array(labels)


def train_generator(a, victim):
    CKPT = ckpt_path(a.eps, a.epochs)
    if os.path.exists(CKPT) and not a.force_train:
        g = Generator3D().cuda()
        g.load_state_dict(torch.load(CKPT))
        print(f"[cdup] reusing trained generator {CKPT}", flush=True)
        return g

    g = Generator3D().cuda().train()
    opt = torch.optim.Adam(g.parameters(), lr=a.start_lr, betas=(0.3, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: a.decay_rate ** (s // a.decay_steps))     # staircase decay

    mm, idx, labels = train_clips(a.n_train, a.seed)
    rng = np.random.RandomState(a.seed)
    step, losses, t0 = 0, [], time.time()
    total_steps = a.epochs * (len(idx) // a.batch)
    print(f"[cdup] training generator: {a.epochs} epochs x "
          f"{len(idx)//a.batch} steps = {total_steps} steps", flush=True)

    for ep in range(a.epochs):
        order = rng.permutation(idx)
        for i in range(0, len(order) - a.batch + 1, a.batch):
            sel = np.sort(order[i:i + a.batch])
            x = torch.from_numpy(np.array(mm[sel], dtype=np.float32)).cuda()
            c = torch.from_numpy(labels[sel]).cuda()

            z = torch.randn(a.batch, Z_SIZE, device="cuda")
            p = a.eps * g(z)
            shift = int(rng.randint(0, 16))
            p = torch.roll(p, shifts=shift, dims=2)          # tf.manip.roll(axis=1)

            logits = victim.model((torch.clamp(x + p, 0, 255) - victim._mean) / victim._std)
            sm = torch.softmax(logits, 1)
            onehot = torch.nn.functional.one_hot(c, NUM_CLASSES).float()
            # ratio=1.0 branch of model.py: minimise -log(1 - p_true)
            loss = (-(onehot * torch.log(torch.clamp(1 - sm, 1e-10, 1.0))).sum(1)).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(float(loss))
            step += 1
            if step % 25 == 0:
                el = time.time() - t0
                print(f"[cdup] step {step}/{total_steps} ep{ep+1} loss={np.mean(losses[-25:]):.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} elapsed {el/60:.1f}m "
                      f"ETA {(total_steps-step)*el/step/60:.1f}m", flush=True)

    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    torch.save(g.state_dict(), CKPT)
    atomic_write_json(os.path.join(RESULT_DIR, "cdup", "train_log.json"),
                      {"steps": step, "epochs": a.epochs, "batch": a.batch,
                       "start_lr": a.start_lr, "p_max": a.eps, "ratio": 1.0,
                       "loss_first50": losses[:50], "loss_last50": losses[-50:],
                       "train_seconds": round(time.time() - t0, 1)})
    print(f"[cdup] generator trained in {(time.time()-t0)/60:.1f} min", flush=True)
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier", type=int, default=1)
    p.add_argument("--eps", type=float, default=DEFAULT_EPS)
    p.add_argument("--epochs", type=int, default=3)          # para_model.train_epoch
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--start_lr", type=float, default=0.002)  # para_model.start_lr
    p.add_argument("--decay_steps", type=int, default=2000)
    p.add_argument("--decay_rate", type=float, default=0.95)
    p.add_argument("--n_train", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force_train", action="store_true")
    p.add_argument("--name", default="cdup")
    a = p.parse_args()
    a.n_train = a.n_train or None

    os.makedirs(os.path.join(RESULT_DIR, "cdup"), exist_ok=True)
    torch.manual_seed(a.seed)
    victim = C3DVictim(max_batch=64)
    g = train_generator(a, victim)

    # test.py: take perturbation index 0 of a generated batch, test_shift = 0
    # z 는 전용 generator 로 뽑는다. 전역 CUDA RNG 를 쓰면 학습을 막 끝낸 실행과
    # 체크포인트를 재사용한 실행의 RNG 상태가 달라져(학습 루프가 447 스텝만큼
    # 소비) 같은 생성기에서 서로 다른 UAP 가 나오고, 평가는 건너뛰므로
    # 디스크의 .npy 와 records.jsonl 이 다른 섭동을 가리키게 된다.
    # ConvTranspose3d 의 기본 cuDNN 알고리즘은 비결정적이어서 실행마다 1e-6 수준의
    # 차이가 난다. 추출은 순전파 1 회이므로 이 구간만 결정론 모드로 묶는다.
    g.eval()
    _prev_det, _prev_bench = (torch.backends.cudnn.deterministic,
                              torch.backends.cudnn.benchmark)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    try:
        with torch.no_grad():
            zgen = torch.Generator(device="cuda").manual_seed(a.seed)
            z = torch.randn(a.batch, Z_SIZE, device="cuda", generator=zgen)
            pert = (a.eps * g(z))[0]                           # (3,16,112,112)
    finally:
        torch.backends.cudnn.deterministic = _prev_det
        torch.backends.cudnn.benchmark = _prev_bench
    pert = pert.clamp(-a.eps, a.eps)
    np.save(uap_path(a.eps), pert.cpu().numpy())
    print(f"[cdup] UAP: Linf={float(pert.abs().max()):.3f} "
          f"mean|.|={float(pert.abs().mean()):.3f}", flush=True)

    def attack(victim, clean, label, eps, budget):
        adv = (torch.as_tensor(clean, device=victim.device) + pert).clamp(0, 255)
        victim.note_iterate(adv.unsqueeze(0))
        return adv

    r = AttackRunner(a.name, tier=a.tier, eps=a.eps, budget=1)
    r.run(attack, note=f"PyTorch port of C-DUP generator_3D (p_max={a.eps}, "
                       f"z=100, k=3, ratio=1.0 untargeted branch, "
                       f"Adam lr={a.start_lr} beta1=0.3, {a.epochs} epochs, "
                       f"batch {a.batch}); perturbation index 0 per test.py")


if __name__ == "__main__":
    main()
