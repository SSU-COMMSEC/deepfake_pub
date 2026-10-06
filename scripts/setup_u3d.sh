#!/usr/bin/env bash
set -euo pipefail

ROOT="${DF_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p "$ROOT/repos"

if [ ! -d "$ROOT/repos/u3d" ]; then
  git clone --depth 1 https://github.com/alarst13/u3d "$ROOT/repos/u3d"
fi

command -v cargo >/dev/null || {
  echo "Rust toolchain is neccesary!: https://rustup.rs"; exit 1; }
python -c "import maturin" 2>/dev/null || pip install maturin

cd "$ROOT/repos/u3d/rust/perlin" && maturin develop --release
cd "$ROOT/repos/u3d/rust/pso"    && maturin develop --release

python - <<'PY'
import perlin, psolib
print("OK: perlin / psolib import success!")
PY
