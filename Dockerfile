# Dockerfile — runpod/pytorch + BAKED OMR training scripts
#
# Background (2026-10-06): 13 RunPod pods died in one day. Root cause:
# dockerArgs gets TRUNCATED when too long (even 153 chars failed).
# New direction: BAKE the scripts INTO the image. Nothing to download
# at startup — the pod just runs them.
#
# Baked scripts (public repo woodylks/omr-gpurun-scripts):
#   /opt/omr/run_train.sh  — training runner (locates s3mini.py via
#                            BASH_SOURCE[0], needs nothing from PATH/cwd)
#   /opt/omr/s3mini.py     — stdlib-only S3 client (urllib + SigV4 header
#                            auth, zero dependencies, works on any image)
#   /opt/omr/bootstrap.py  — kept for reference (no longer needed at startup)
#
# Pod startup becomes trivial (Elon fills in <training-pkg> and <endpoint>):
#   bash /opt/omr/run_train.sh \
#     --pkg s3://7bpajmibp1/omr-code/<training-pkg>.tar.gz \
#     --s3-bucket 7bpajmibp1 \
#     --s3-data-tarball s3://7bpajmibp1/omr-data-tarballs/verovio-fixed.tar --expected-samples 99771 \
#     --s3-data-tarball s3://7bpajmibp1/omr-data-tarballs/lilypond-fixed.tar --expected-samples 10000 \
#     --s3-output-prefix omr-runs/full-110k-$(date -u +%Y%m%d-%H%M%S)/ \
#     --s3-endpoint-url <endpoint> \
#     --epochs 30 --no-terminate
#
# Data + training package are still pulled from S3 at RUNTIME via s3mini.py
# (stdlib, no aws CLI needed).

ARG BASE_TAG=2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
FROM runpod/pytorch:${BASE_TAG}

# Bake the training scripts in — nothing to download at pod startup.
COPY bootstrap.py s3mini.py run_train.sh /opt/omr/

# Fail the BUILD (not the pod) if a script is broken.
RUN chmod +x /opt/omr/run_train.sh \
    && ls -la /opt/omr/ \
    && bash -n /opt/omr/run_train.sh \
    && python3 -c "import py_compile; py_compile.compile('/opt/omr/s3mini.py', doraise=True); py_compile.compile('/opt/omr/bootstrap.py', doraise=True)" \
    && echo "baked scripts OK"
