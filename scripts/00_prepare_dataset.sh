#!/usr/bin/env bash
# Stage 00 — fetch UCF-101, its official split lists, and the C3D victim checkpoint.
#
# Idempotent: every step is skipped if its output already exists, so the script is safe to
# re-run after an interrupted download.
#
# Produces
#   dataset/UCF-101/<class>/*.avi          13,320 videos in 101 classes
#   dataset/ucfTrainTestlist/{classInd,trainlist0*,testlist0*}.txt
#   checkpoints/c3d_sports1m_16x1x1_45e_ucf101_rgb_20201021-26655025.pth
set -uo pipefail

ROOT="${DF_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
DS="$ROOT/dataset"
CKPT_DIR="$ROOT/checkpoints"

UCF_RAR_URL="https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
UCF_SPLIT_URL="https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"
CKPT_URL="https://download.openmmlab.com/mmaction/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb/c3d_sports1m_16x1x1_45e_ucf101_rgb_20201021-26655025.pth"
CKPT="$CKPT_DIR/c3d_sports1m_16x1x1_45e_ucf101_rgb_20201021-26655025.pth"

# crcv.ucf.edu serves an incomplete TLS chain: the leaf certificate is sent without its
# intermediate, so verification fails with "unable to get local issuer certificate" on every
# client. The intermediate is published by the issuer, so we fetch it and build a bundle
# rather than disabling verification.
INTERMEDIATE_URL="http://crt.sectigo.com/InCommonRSAOVSSLCA3.crt"
CA_BUNDLE="$DS/.crcv-ca-bundle.pem"

log() { echo "[00_prepare_dataset] $*"; }
need() { command -v "$1" >/dev/null || { log "FATAL: '$1' is required but not installed"; exit 1; }; }

need curl
mkdir -p "$DS" "$CKPT_DIR"

# --- TLS chain repair for crcv.ucf.edu -----------------------------------------
build_ca_bundle() {
  [ -s "$CA_BUNDLE" ] && return 0
  command -v openssl >/dev/null || return 1

  local sys_ca=""
  for c in /etc/ssl/certs/ca-certificates.crt \
           /etc/pki/tls/certs/ca-bundle.crt \
           /etc/ssl/ca-bundle.pem \
           "${CONDA_PREFIX:-/nonexistent}/ssl/cacert.pem"; do
    [ -f "$c" ] && { sys_ca="$c"; break; }
  done
  [ -n "$sys_ca" ] || return 1

  local der="$DS/.incommon.der" pem="$DS/.incommon.pem"
  curl -sS --fail --max-time 60 -o "$der" "$INTERMEDIATE_URL" || return 1
  openssl x509 -inform DER -in "$der" -out "$pem" 2>/dev/null || return 1
  cat "$sys_ca" "$pem" > "$CA_BUNDLE" || return 1
  rm -f "$der" "$pem"
  return 0
}

# curl wrapper: adds the repaired bundle for CRCV hosts, honours an explicit opt-out.
fetch() {  # fetch <url> <output>
  local url="$1" out="$2" extra=()
  case "$url" in
    *crcv.ucf.edu*)
      if [ "${UAP_INSECURE_DOWNLOAD:-0}" = "1" ]; then
        log "WARNING: UAP_INSECURE_DOWNLOAD=1 -- TLS verification disabled for this host"
        extra=(--insecure)
      elif build_ca_bundle; then
        extra=(--cacert "$CA_BUNDLE")
      else
        log "WARNING: could not build the CA bundle; falling back to the system store"
      fi
      ;;
  esac
  curl -L --fail --retry 3 -C - "${extra[@]}" -o "$out" "$url"
}

# --- 1. videos ----------------------------------------------------------------
if [ -d "$DS/UCF-101" ]; then
  log "dataset/UCF-101 already present, skipping download"
else
  if [ ! -f "$DS/UCF101.rar" ]; then
    log "no UCF101.rar found at $DS/UCF101.rar"
    log "downloading UCF-101 videos (~6.5 GB) ..."
    fetch "$UCF_RAR_URL" "$DS/UCF101.rar" || {
      log "FATAL: download failed."
      log "  If the error mentions 'unable to get local issuer certificate', the CRCV server"
      log "  is sending an incomplete TLS chain and the repair above did not apply. Either"
      log "  install 'openssl' so the chain can be rebuilt, or re-run as"
      log "      UAP_INSECURE_DOWNLOAD=1 bash scripts/00_prepare_dataset.sh"
      log "  Otherwise fetch UCF101.rar manually from"
      log "      https://www.crcv.ucf.edu/data/UCF101.php"
      log "  and place it at $DS/UCF101.rar, then re-run this script."
      exit 1; }
  fi
  log "extracting UCF101.rar (several minutes) ..."
  if command -v bsdtar >/dev/null; then ( cd "$DS" && bsdtar -xf UCF101.rar )
  elif command -v unrar  >/dev/null; then ( cd "$DS" && unrar x -inul UCF101.rar )
  else log "FATAL: need 'bsdtar' (libarchive-tools) or 'unrar' to extract the archive"; exit 1
  fi
fi

# --- 2. official train/test splits --------------------------------------------
if [ -f "$DS/ucfTrainTestlist/testlist01.txt" ]; then
  log "dataset/ucfTrainTestlist already present, skipping"
else
  need unzip
  # use a manually downloaded archive if one is already there, under either name
  zip=""
  for cand in "$DS/UCF101TrainTestSplits-RecognitionTask.zip" "$DS/ucf101_splits.zip"; do
    [ -f "$cand" ] && { zip="$cand"; break; }
  done
  if [ -n "$zip" ]; then
    log "using existing archive $(basename "$zip")"
  else
    zip="$DS/ucf101_splits.zip"
    log "downloading official train/test splits ..."
    fetch "$UCF_SPLIT_URL" "$zip" || {
      log "FATAL: split list download failed (see the note above about TLS)"; exit 1; }
  fi
  unzip -q -o "$zip" -d "$DS/_splits_tmp"
  # the archive nests the lists one directory deep
  src=$(find "$DS/_splits_tmp" -type d -name ucfTrainTestlist | head -1)
  [ -n "$src" ] || { log "FATAL: ucfTrainTestlist not found inside the archive"; exit 1; }
  mv "$src" "$DS/ucfTrainTestlist"
  rm -rf "$DS/_splits_tmp"
  rm -f "$DS/ucf101_splits.zip"
fi

# --- 3. victim checkpoint ------------------------------------------------------
if [ -f "$CKPT" ]; then
  log "checkpoint already present, skipping"
else
  log "downloading C3D checkpoint (~313 MB) ..."
  fetch "$CKPT_URL" "$CKPT" || { log "FATAL: checkpoint download failed"; exit 1; }
fi

# --- 4. verify -----------------------------------------------------------------
n_class=$(find "$DS/UCF-101" -mindepth 1 -maxdepth 1 -type d | wc -l)
n_vid=$(find "$DS/UCF-101" -name '*.avi' | wc -l)
log "classes = $n_class (expected 101),  videos = $n_vid (expected 13320)"

missing=0
while read -r rel; do
  [ -z "$rel" ] && continue
  [ -f "$DS/UCF-101/$rel" ] || { missing=$((missing+1)); [ $missing -le 5 ] && log "  MISSING: $rel"; }
done < <(tr -d '\r' < "$DS/ucfTrainTestlist/testlist01.txt" | awk '{print $1}')
log "testlist01 entries missing: $missing / 3783"

if [ "$n_vid" -lt 13000 ] || [ "$missing" -gt 0 ] || [ ! -f "$CKPT" ]; then
  log "FATAL: verification failed"; exit 1
fi
log "dataset and checkpoint verified OK"
