#!/usr/bin/env bash
# start.sh — baked-image entrypoint wrapper.
# Keeps pod dockerArgs ultra-short (24 chars):  bash /opt/omr/start.sh
# All config comes from env vars (RunPod passes env fine; dockerArgs gets
# truncated when too long — even 153 chars failed on 2026-10-06).
set -euo pipefail

: "${OMR_PKG:?need OMR_PKG env (s3://bucket/path/training-pkg.tar.gz or https://...)}"
: "${OMR_S3_ENDPOINT_URL:?need OMR_S3_ENDPOINT_URL env}"
: "${AWS_ACCESS_KEY_ID:?need AWS_ACCESS_KEY_ID env}"
: "${AWS_SECRET_ACCESS_KEY:?need AWS_SECRET_ACCESS_KEY env}"

# v7 (2026-10-07): data tarballs on Hugging Face Hub — pod->RunPod-S3 is
# throttled to 0.03 MB/s (Elon speedtest, team-ops #26). Override via env
# if the HF repo name differs.
HF_DATA_REPO="${OMR_HF_DATA_REPO:-woodylks/omr-data}"
VEROVIO_URL="${OMR_VEROVIO_URL:-https://huggingface.co/datasets/${HF_DATA_REPO}/resolve/main/verovio-fixed.tar}"
LILYPOND_URL="${OMR_LILYPOND_URL:-https://huggingface.co/datasets/${HF_DATA_REPO}/resolve/main/lilypond-fixed.tar}"

exec /opt/omr/run_train.sh \
  --pkg "$OMR_PKG" \
  --s3-bucket "${OMR_S3_BUCKET:-7bpajmibp1}" \
  --data-tarball "$VEROVIO_URL" --expected-samples 99771 \
  --data-tarball "$LILYPOND_URL" --expected-samples 10000 \
  --s3-output-prefix "omr-runs/full-110k-$(date -u +%Y%m%d-%H%M%S)/" \
  --s3-endpoint-url "$OMR_S3_ENDPOINT_URL" \
  --s3-region "${OMR_S3_REGION:-us-ca-2}" \
  --aws-access-key-id "$AWS_ACCESS_KEY_ID" \
  --aws-secret-access-key "$AWS_SECRET_ACCESS_KEY" \
  --epochs "${OMR_EPOCHS:-30}" \
  ${OMR_NO_TERMINATE:+--no-terminate}
