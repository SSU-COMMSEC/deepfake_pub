#!/usr/bin/env bash
# Deepfake (SimSwap) target, end to end. Every stage writes under results/deepfake/ and is
# resumable; a finished universal perturbation is not recomputed.
#
#   bash scripts/run_deepfake.sh            # all stages
#   STAGES="5 6" bash scripts/run_deepfake.sh
#
# Needs the unified environment and PhantomSeal (docs/DEEPFAKE.md).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
STAGES="${STAGES:-0 1 2 3 4 5 6 7}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/deepfake
run() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }
log() { echo "[deepfake $(date +%H:%M:%S)] $*"; }

if run 0; then log "0 self-test";            $PY -m deepfake.selftest; fi
if run 1; then
  log "1 face scan (test, then train with frame cache)"
  $PY -m deepfake.scan_faces --split test
  $PY -m deepfake.scan_faces --split train --save-frames
fi
if run 2; then
  for m in u3d cdup3d cdup2d uniform; do
    if [ -f "$R/uap/universal/$m/clip.npy" ]; then log "2 universal $m: done, skipped"; continue; fi
    log "2 universal $m"; $PY -m deepfake.train_universal --method "$m" --n-val 150
  done
fi
if run 3; then log "3 freeze the evaluation set"; [ -f "$R/eval_set.json" ] || $PY -m deepfake.prepare_eval --n 100; fi
if run 4; then
  for m in u3d cdup3d; do log "4 per-video $m"; $PY -m deepfake.per_video --method "$m"; done
  log "4 PhantomSeal on every scored frame (per-frame reference)"; $PY -m deepfake.phantomseal_frames
fi
if run 5; then log "5 evaluate";  $PY -m deepfake.evaluate --name main --oracle; fi
if run 6; then log "6 summarize"; $PY -m deepfake.summarize --name main --oracle; fi
if run 7; then log "7 demo videos"; $PY -m deepfake.render_demo --frames 100; fi
log "done"
