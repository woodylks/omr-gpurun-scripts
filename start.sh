#!/usr/bin/env bash
# start.sh — baked-image entrypoint wrapper.
# Keeps pod dockerArgs ultra-short (24 chars):  bash /opt/omr/start.sh
# All config comes from env vars (RunPod passes env fine; dockerArgs gets
# truncated when too long — even 153 chars failed on 2026-10-06).
set -euo pipefail

: "${OMR_PKG:?need OMR_PKG env (s3://bucket/path/training-pkg.tar.gz)}"
: "${OMR_S3_ENDPOINT_URL:?need OMR_S3_ENDPOINT_URL env}"
: "${AWS_ACCESS_KEY_ID:?need AWS_ACCESS_KEY_ID env}"
: "${AWS_SECRET_ACCESS_KEY:?need AWS_SECRET_ACCESS_KEY env}"

exec /opt/omr/run_train.sh \
  --pkg "$OMR_PKG" \
  --s3-bucket "${OMR_S3_BUCKET:-7bpajmibp1}" \
  --s3-data-tarball s3://7bpajmibp1/omr-data-tarballs/verovio-fixed.tar --expected-samples 99771 \
  --s3-data-tarball s3://7bpajmibp1/omr-data-tarballs/lilypond-fixed.tar --expected-samples 10000 \
  --s3-output-prefix "omr-runs/full-110k-$(date -u +%Y%m%d-%H%M%S)/" \
  --s3-endpoint-url "$OMR_S3_ENDPOINT_URL" \
  --s3-region "${OMR_S3_REGION:-us-ca-2}" \
  --aws-access-key-id "$AWS_ACCESS_KEY_ID" \
  --aws-secret-access-key "$AWS_SECRET_ACCESS_KEY" \
  --epochs "${OMR_EPOCHS:-30}" \
  ${OMR_NO_TERMINATE:+--no-terminate}
