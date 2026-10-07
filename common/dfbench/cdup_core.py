"""C-DUP core modules shared by both targets (adapters/cdup_adapter.py for C3D, deepfake/ for SimSwap).

Layer-for-layer PyTorch transcription of the authors' TensorFlow 1.x code
(sli057/Video-Perturbation @ 2488820), which has no kernels for sm_120:

  * generator_3D: z(100) -> 5 transposed 3D convs (BN, ReLU, tanh) -> 3x16x112x112 clip
                                         <- generator.py:generator_3D, para_model.py
  * generator_2D + Tile: one 112x112 frame repeated 16 times (2D-DUP)
                                         <- generator.py:generator_2D (tf.tile)
  * p = p_max * G(z): the L_inf budget holds through tanh
                                         <- model.py (tf.scalar_mul(p_max, .))
  * circular perturbation via a random temporal Roll of the clip in every step
                                         <- model.py: tf.manip.roll(z_out, shift, axis=1)
  * Adam(lr 0.002, beta1 0.3), staircase decay 0.95 every 2000 steps
                                         <- model.py, para_model.py
  * test time: one perturbation drawn from the trained generator, applied to every video
                                         <- generate_perturbations.py, test.py (p_batch[0])

The objective is not here: each target supplies its own (C3D cross-entropy, or PhantomSeal's
protection score).
"""
import torch
import torch.nn as nn

Z_SIZE, K_SIZE = 100, 3                                # para_model.py
START_LR, DECAY_STEPS, DECAY_RATE, BETA1 = 0.002, 2000, 0.95, 0.3


def _bn(c):
    # tf.contrib.layers.batch_norm(decay=0.1, epsilon=1e-5, scale=True) -> momentum = 1 - decay
    return nn.BatchNorm3d(c, eps=1e-5, momentum=0.9, affine=True)


def _init(m):
    for mod in m.modules():                            # random_normal_initializer(stddev=0.02), bias 0
        if isinstance(mod, nn.ConvTranspose3d):
            nn.init.normal_(mod.weight, 0.0, 0.02)
            nn.init.zeros_(mod.bias)


class Generator3D(nn.Module):
    """generator.py:generator_3D. TF 'SAME' with kernel 3 / stride 2 doubles each spatial dim; in
    PyTorch that is padding=1 with output_padding=1."""

    def __init__(self, z_size=Z_SIZE, k=K_SIZE):
        super().__init__()
        self.d1 = nn.ConvTranspose3d(z_size, 512, (1, 7, 7), stride=1, padding=0)
        self.b1 = _bn(512)
        self.d2 = nn.ConvTranspose3d(512, 256, k, stride=2, padding=1, output_padding=1)
        self.b2 = _bn(256)
        self.d3 = nn.ConvTranspose3d(256, 128, k, stride=2, padding=1, output_padding=1)
        self.b3 = _bn(128)
        self.d4 = nn.ConvTranspose3d(128, 64, k, stride=2, padding=1, output_padding=1)
        self.b4 = _bn(64)
        self.d5 = nn.ConvTranspose3d(64, 3, k, stride=2, padding=1, output_padding=1)
        _init(self)

    def forward(self, z):
        x = z.view(z.shape[0], -1, 1, 1, 1)
        x = torch.relu(self.b1(self.d1(x)))
        x = torch.relu(self.b2(self.d2(x)))
        x = torch.relu(self.b3(self.d3(x)))
        x = torch.relu(self.b4(self.d4(x)))
        return torch.tanh(self.d5(x))                  # (B,3,16,112,112) in [-1,1]


class Generator2D(nn.Module):
    """generator.py:generator_2D: temporal stride 1 everywhere (depth stays 1), then tf.tile x16."""

    def __init__(self, z_size=Z_SIZE, k=K_SIZE, T=16):
        super().__init__()
        self.T = T
        self.d1 = nn.ConvTranspose3d(z_size, 512, (1, 7, 7), stride=1, padding=0)
        self.b1 = _bn(512)
        s, pad, op = (1, 2, 2), (1, 1, 1), (0, 1, 1)
        self.d2 = nn.ConvTranspose3d(512, 256, k, stride=s, padding=pad, output_padding=op)
        self.b2 = _bn(256)
        self.d3 = nn.ConvTranspose3d(256, 128, k, stride=s, padding=pad, output_padding=op)
        self.b3 = _bn(128)
        self.d4 = nn.ConvTranspose3d(128, 64, k, stride=s, padding=pad, output_padding=op)
        self.b4 = _bn(64)
        self.d5 = nn.ConvTranspose3d(64, 3, k, stride=s, padding=pad, output_padding=op)
        _init(self)

    def forward(self, z):
        x = z.view(z.shape[0], -1, 1, 1, 1)
        x = torch.relu(self.b1(self.d1(x)))
        x = torch.relu(self.b2(self.d2(x)))
        x = torch.relu(self.b3(self.d3(x)))
        x = torch.relu(self.b4(self.d4(x)))
        out = torch.tanh(self.d5(x))                   # (B,3,1,112,112)
        return out.repeat(1, 1, self.T, 1, 1)          # tf.tile(out, [1,16,1,1,1])


def roll(p, shift: int):
    """model.py: tf.manip.roll(z_out, shift=shift, axis=1) — the time axis of (B,C,T,H,W)."""
    return torch.roll(p, shifts=int(shift), dims=2)


def make_optimizer(g, start_lr=START_LR, decay_steps=DECAY_STEPS, decay_rate=DECAY_RATE):
    opt = torch.optim.Adam(g.parameters(), lr=start_lr, betas=(BETA1, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: decay_rate ** (s // decay_steps))
    return opt, sched


@torch.no_grad()
def draw_perturbation(g, eps: float, seed: int = 0, batch: int = 16):
    """generate_perturbations.py + test.py: BN in inference mode, sample 0 of a batch -> (3,16,112,112).

    z comes from a dedicated generator, so the result does not depend on how much of the global CUDA
    RNG the training loop consumed; ConvTranspose3d's default cuDNN algorithm is non-deterministic,
    so this single forward pass runs in deterministic mode. Same generator + seed + batch -> same
    perturbation, bit for bit."""
    g.eval()
    prev = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    try:
        zgen = torch.Generator(device="cuda").manual_seed(seed)
        z = torch.randn(batch, Z_SIZE, device="cuda", generator=zgen)
        p = (eps * g(z))[0]
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = prev
    return p.clamp(-eps, eps)
