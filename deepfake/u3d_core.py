"""U3D core for the deepfake target: the paper's noise (Eq. 1-2) searched by a converging PSO.

Taken from U3D (alarst13/u3d @ 15541a0), unchanged in meaning:
  * the U3Dp perturbation: 3D Perlin noise with octaves, wavelengths (x, y, t), sine colour map,
    amplitude epsilon, the same value on R, G, B   <- perlin3d.rs, perlin_noise.py
  * the temporal transformation Trans(xi, tau) = roll along time, expectation over I = 5 shifts
                                                  <- u3d-attack-C3D.py
  * gradient-free PSO over the 5 parameters with the repository's bounds, swarm 20, 40 iterations
                                                  <- rust/pso (psolib), called as compiled
Replaced: the objective (PhantomSeal's protection score instead of a C3D feature distance).

Two defects of the reference implementation are corrected by default; both are reproducible with
noise="repo", omega=1.2 for comparison with runs made before the fix:
  * noise: the Rust colour map computes sin(v * 2*pi / color_period) (a division; Eq. 2 multiplies)
    on unipolar octaves, which turns most of the parameter space into a near-uniform brightness
    shift. noise="paper" follows Eq. 1-2 (dfbench.u3d_perlin, mode="paper").
  * PSO: psolib updates v <- v + omega*v + ..., an effective inertia of 1 + omega. The repository's
    omega = 1.2 (effective 2.2) diverges and pins every particle to the bounds. omega = -0.27 gives
    the standard inertia 0.73. minstep / minfunc are 0 so the stated 40 iterations actually run.
"""

import numpy as np

from dfbench import u3d_perlin

PARAM_NAMES = ["num_octaves", "wavelength_x", "wavelength_y", "wavelength_t", "color_period"]
LB = [1, 2.0, 2.0, 2.0, 1.0]           # u3d-attack-C3D.py --lb default (= paper Table XVIII)
UB = [5, 180.0, 180.0, 180.0, 60.0]    # u3d-attack-C3D.py --ub default
NOISE = "paper"                        # "repo" reproduces the Rust generator bit for bit
OMEGA = -0.27                          # psolib argument; effective inertia 1 + omega = 0.73
PSO_DEFAULTS = dict(swarmsize=20, omega=OMEGA, phip=2.0, phig=2.0, maxiter=40)
I_SHIFTS = 5                           # --iteration default


def perlin_volume(T: int, H: int, W: int, params, epsilon: float, noise: str = NOISE, device="cuda"):
    """[T,H,W] float64 tensor in [-eps, eps]. params = [num_octaves, wl_x, wl_y, wl_t, color_period].

    noise="paper": Eq. 1-2 (Lambda+1 bipolar octaves, sin(N * 2*pi*phi)).
    noise="repo" : the upstream Rust generator, bit-compatible (Lambda unipolar octaves, division)."""
    n_oct = int(round(float(params[0])))
    wx, wy, wt, cp = (float(params[i]) for i in range(1, 5))
    return u3d_perlin.generate_noise_torch(T, H, W, n_oct, wx, wy, wt, cp, epsilon, mode=noise,
                                           device=device)


def temporal_transformation(perturbation, tau_shift: int):
    """u3d-attack-C3D.py:temporal_transformation (roll along the T axis)."""
    import torch
    if isinstance(perturbation, torch.Tensor):
        return torch.roll(perturbation, int(tau_shift), dims=0)
    return np.roll(perturbation, int(tau_shift), axis=0)


def pso(func, lb=LB, ub=UB, **kw):
    """The upstream Rust PSO (psolib). See the module docstring for omega and minstep/minfunc."""
    from psolib import particle_swarm_optimization

    opts = dict(PSO_DEFAULTS)
    opts.update(kw)
    best, fbest = particle_swarm_optimization(func, lb=np.array(lb, dtype=np.float64),
                                              ub=np.array(ub, dtype=np.float64),
                                              minstep=0.0, minfunc=0.0, debug=False, **opts)
    best = [round(p) if i == 0 else float(p) for i, p in enumerate(list(best))]
    return best, float(fbest)
