#!/usr/bin/env python
"""Report what is installed and what is missing, with the command that fixes each gap.

    python scripts/check_env.py

Exits non-zero if anything required is missing, so it can gate a CI job or a setup script.
Every import is guarded: this runs on a bare interpreter.
"""
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OK, WARN, BAD = "  ok  ", " note ", " MISS "
problems, notes = [], []


def row(status, name, detail=""):
    print(f"[{status}] {name:22} {detail}")


def version_of(mod_name):
    try:
        mod = importlib.import_module(mod_name)
        return getattr(mod, "__version__", "installed")
    except Exception:
        return None


def check(mod_name, label, fix, required=True, minimum=None):
    v = version_of(mod_name)
    if v is None:
        row(BAD if required else WARN, label, f"not installed  ->  {fix}")
        (problems if required else notes).append(fix)
        return None
    extra = ""
    if minimum:
        try:
            cur = tuple(int(x) for x in v.split("+")[0].split(".")[:2])
            if cur < minimum:
                extra = f"  (need >= {'.'.join(map(str, minimum))}  ->  {fix})"
                problems.append(fix)
        except ValueError:
            pass
    row(OK if not extra else BAD, label, v + extra)
    return v


def main():
    print(f"repository : {ROOT}")
    print(f"interpreter: {sys.executable}")
    print(f"DF_ROOT    : {os.environ.get('DF_ROOT', '(unset, using the repository root)')}\n")

    print("-- python packages " + "-" * 44)
    py = ".".join(map(str, sys.version_info[:3]))
    if sys.version_info[:2] < (3, 10):
        row(BAD, "python", f"{py}  (need >= 3.10)")
        problems.append("install Python 3.10 or newer")
    else:
        row(OK, "python", py)

    # 순서가 중요하다. pip torch 휠은 시스템 libstdc++ 를 로드하는데, 그것이
    # conda 의 opencv 가 요구하는 CXXABI 보다 낮으면 그 뒤의 `import cv2` 가 깨진다.
    # conda 의 numpy 를 먼저 올려 두면 conda libstdc++ 가 먼저 로드되어 둘 다 산다.
    # 모든 진입점이 numpy -> torch 순으로 import 하므로 여기서도 같은 순서를 쓴다.
    check("numpy", "numpy", "conda install -c conda-forge 'numpy>=1.24'", minimum=(1, 24))
    check("cv2", "opencv", "conda install -c conda-forge 'opencv>=4.8,<5'", minimum=(4, 8))
    torch_fix = ("pip install torch --index-url "
                 "https://download.pytorch.org/whl/cu128")
    check("torch", "torch", torch_fix, required=True, minimum=(2, 7))
    check("tqdm", "tqdm", "conda install -c conda-forge tqdm")
    check("decord", "decord", "pip install decord")   # required by the full pipeline
    # repos/u3d/python/video_classification/dataloaders/u3d_dataset.py imports
    # sklearn at module level, so the U3D adapter cannot even be imported without it.
    check("sklearn", "scikit-learn", "conda install -c conda-forge scikit-learn")

    print("\n-- gpu " + "-" * 56)
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            row(OK, "cuda", f"{torch.version.cuda}  ({torch.cuda.get_device_name(0)})")
            row(OK, "capability", f"sm_{cap[0]}{cap[1]}")
            if cap >= (12, 0) and tuple(int(x) for x in torch.__version__.split(".")[:2]) < (2, 7):
                row(BAD, "compatibility", "sm_120 needs torch >= 2.7")
                problems.append(torch_fix)
        else:
            row(WARN, "cuda", "no GPU visible -- demo works on CPU, training does not")
            notes.append("a CUDA GPU is needed to generate perturbations")
    except Exception as e:
        row(WARN, "cuda", f"could not query ({type(e).__name__})")

    print("\n-- u3d rust extensions " + "-" * 40)
    missing_rust = [m for m in ("perlin", "psolib") if version_of(m) is None]
    if missing_rust:
        row(WARN, "perlin / psolib", f"{', '.join(missing_rust)} missing  ->  "
                                     "bash scripts/setup_u3d.sh")
        notes.append("bash scripts/setup_u3d.sh  (required for U3D only)")
    else:
        row(OK, "perlin / psolib", "built")

    print("\n-- shipped perturbations " + "-" * 38)
    for f in ("cdup_perturbation.npy", "u3d_perturbation.npy"):
        p = os.path.join(ROOT, "assets", "perturbations", f)
        row(OK if os.path.exists(p) else BAD, f,
            f"{os.path.getsize(p) / 1e6:.1f} MB" if os.path.exists(p) else "not found")
        if not os.path.exists(p):
            problems.append(f"{f} is missing from assets/perturbations/")

    print("\n-- data and weights " + "-" * 43)
    try:
        sys.path.insert(0, os.path.join(ROOT, "common"))
        from dfbench.paths import VIDEO_DIR, SPLIT_DIR, C3D_CKPT, EVAL_SUBSET
        items = [("UCF-101 videos", VIDEO_DIR), ("split lists", SPLIT_DIR),
                 ("C3D checkpoint", C3D_CKPT), ("frozen eval subset", EVAL_SUBSET)]
        hints = {"UCF-101 videos": "bash scripts/00_prepare_dataset.sh",
                 "split lists": "bash scripts/00_prepare_dataset.sh",
                 "C3D checkpoint": "bash scripts/00_prepare_dataset.sh",
                 "frozen eval subset": "PYTHONPATH=common python scripts/01_clean_eval.py"}
        for label, path in items:
            there = os.path.exists(path)
            row(OK if there else WARN, label, path if there else f"not found  ->  {hints[label]}")
            if not there:
                notes.append(hints[label])
    except Exception as e:
        row(WARN, "dfbench", f"could not import ({type(e).__name__}: {e})")

    print()
    if problems:
        print("Required items are missing. Run:\n")
        for f in dict.fromkeys(problems):
            print(f"    {f}")
        return 1
    if notes:
        print("Ready for the demo. For the full pipeline you still need:\n")
        for f in dict.fromkeys(notes):
            print(f"    {f}")
        return 0
    print("Everything is in place.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
