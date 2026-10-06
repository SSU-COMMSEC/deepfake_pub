# Pre-generated universal perturbations

These are the perturbations produced by the two adapters under the benchmark's standard
settings. They let you run the demo and apply either attack **without training a generator or
running a parameter search.**

| File | Method | Shape | dtype | ‖δ‖∞ | mean &#124;δ&#124; |
|---|---|---|---|---|---|
| `cdup_perturbation.npy` | C-DUP | `(3, 16, 112, 112)` | float32 | 10.000 | 9.978 |
| `u3d_perturbation.npy` | U3D | `(16, 112, 112)` | float64 | 10.000 | 6.377 |
| `u3d_params.json` | U3D | — | — | — | the five parameters that generate the above |

Both respect the benchmark budget of `‖δ‖∞ ≤ 10` in 0–255 pixel units.

C-DUP's perturbation is near-saturated almost everywhere (mean magnitude 9.978 against a
bound of 10); U3D's is substantially sparser in magnitude. That difference is the visible one
in the demo.

## How they were produced

```bash
PYTHONPATH=common python adapters/cdup_adapter.py --tier 1   # -> results/cdup/cdup_perturbation.npy
PYTHONPATH=common python adapters/u3d_adapter.py  --tier 1   # -> results/u3d/u3d_perturbation.npy
```

Victim C3D (Sports-1M → UCF-101 split-1), budget L∞ ≤ 10/255. U3D used the repository's
default PSO settings; see [`../../docs/U3D.md` §4.2](../../docs/U3D.md) for what that implies.

`u3d_params.json` records the five noise parameters plus the full search configuration, so the
U3D perturbation can be regenerated at any resolution or clip length:

```python
from adapters.u3d_adapter import gen_noise   # requires the Rust extensions
```

## Shapes

C-DUP produces a per-channel tensor. U3D has no channel axis — the same value is added to all
three channels, matching `add_perlin_noise_to_frame` in the reference implementation.

```python
import numpy as np, torch

delta = torch.from_numpy(np.load("assets/perturbations/cdup_perturbation.npy"))  # (3,16,112,112)
adv   = (clip + delta).clamp(0, 255)

noise = torch.from_numpy(np.load("assets/perturbations/u3d_perturbation.npy")).float()
adv   = (clip + noise.unsqueeze(0)).clamp(0, 255)                                # (16,112,112)
```

`clip` is `(3, 16, 112, 112)` in pixel space, range 0–255.

## Integrity

```bash
sha256sum -c SHA256SUMS
```
