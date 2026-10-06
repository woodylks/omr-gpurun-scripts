#!/usr/bin/env bash
#
# run_train.sh — bulletproof S3-native GPU training run for OMR CRNN-CTC.
#
# WHY THIS EXISTS (2026-10-05 incident): a 1-epoch smoke test ran fine on a
# RunPod pod, but the result could not be retrieved — outputs were written to
# the pod's ephemeral disk and lost when the pod went away. This script makes
# that impossible by construction.
#
#   IRON RULE 1 — S3 is the only source of truth for outputs. Checkpoints, logs,
#     metrics and run_meta.json are synced to S3 continuously (every
#     --sync-interval minutes). The run is NOT considered complete until every
#     expected file is verified present AND non-empty ON S3.
#   IRON RULE 2 — verify-before-terminate. The pod terminates itself via the
#     RunPod API only after verification passes. If verification fails (or
#     training itself fails), the pod is deliberately LEFT RUNNING for inspection.
#   IRON RULE 3 — fail fast. VOCAB_SIZE==425 assert (version-guard), data present
#     on S3, S3 writable (canary write) — all checked BEFORE any GPU time burns.
#
# S3-native design (2026-10-05): the old network volume was region-locked, which
# is why GPU retry could only target one region (NO_SUPPLY). S3 is global, so the
# retry cron can hunt GPUs in every region. The network volume is optional — used
# as a download cache when mounted, never required.
#
# TARBALL STAGING (2026-10-06): per-file `aws s3 sync` of 220k small files is
# unreasonably slow AND progress-blind (staging ran for hours with zero signal).
# Preferred mode is now one tarball per data source: single HTTP download,
# tar xf (compression auto-detect), count-verify. Legacy per-file sync is kept as fallback.
#
# Credentials are NEVER hardcoded. Pass via env vars (set them in the RunPod pod
# template, NOT on the command line) or CLI flags:
#   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY  (or --aws-access-key-id / --aws-secret-access-key)
#   RUNPOD_API_KEY                            (or --runpod-api-key; needed for self-terminate)
#
# One-line startup command (paste into RunPod "Start Command"):
#
#   bash run_train.sh --s3-bucket my-bucket \
#     --s3-data-tarball s3://my-bucket/omr-data/verovio-fixed.tar.gz --expected-samples 99771 \
#     --s3-data-tarball s3://my-bucket/omr-data/lilypond-fixed.tar.gz --expected-samples 10000 \
#     --s3-output-prefix omr-runs/full-20261006/ \
#     --pkg s3://my-bucket/omr-code/training-pkg-abc123.tar.gz \
#     --epochs 30
#
# run_smoke.sh is a thin wrapper: 1 epoch, smoke output prefix.
#
set -euo pipefail

# ---------------------------------------------------------------- defaults ---
EPOCHS=30
BATCH_SIZE=32
LR=1e-3
WORKERS=4
SEED=20260929
DEVICE="auto"
SYNC_INTERVAL_MIN=5
STAGING_TIMEOUT_MIN=30
VOLUME_DIR=""
S3_ENDPOINT_URL=""
S3_REGION=""
NO_TERMINATE=0
PKG=""
EXTRA_ARGS=""
POST_RUN_CMD=""

S3_BUCKET=""
S3_OUT_PREFIX=""
declare -a S3_DATA_PREFIXES=()
declare -a S3_DATA_TARBALLS=()
declare -a EXPECTED_SAMPLES=()

# ------------------------------------------------------------------ helpers --
TS() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log()  { echo "[$(TS)] $*" | tee -a "${LOGFILE:-/dev/null}"; }
die()  { log "FATAL: $*"; exit 1; }
free_kb() { df -k "$WORKDIR" | awk 'NR==2{print $4}'; }  # needs WORKDIR set

usage() {
  sed -n '2,/^# ----/p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Arguments:
  --s3-bucket BUCKET            (required) S3 bucket for data + outputs
  --s3-data-tarball s3://...    (repeatable) pre-packed data tarball(s), one
                                per data source. Single HTTP download + tar xf (auto-detect)
                                per tarball — PREFERRED over --s3-data-prefix
                                (220k small-file syncs are slow and blind).
                                Either --s3-data-tarball or --s3-data-prefix
                                is required.
  --expected-samples N          expected sample count for the most recent
                                --s3-data-tarball; mismatch fails fast
  --s3-data-prefix PREFIX       (repeatable, legacy fallback) per-file sync,
                                e.g. omr-data/verovio-fixed/
  --s3-output-prefix PREFIX     (required) run outputs go here, e.g.
                                omr-runs/full-20261006/
  --staging-timeout MIN         watchdog: self-terminate if staging makes no
                                heartbeat progress for MIN minutes
                                (default 30; RULES.md Rule 9)
  --pkg PATH| s3://...          (required) training package tarball
                                (from package_for_training.sh), local or on S3
  --epochs N                    default 30
  --batch-size N                default 32
  --lr F                        default 1e-3
  --workers N                   default 4
  --seed N                      default 20260929
  --device cuda|cpu|auto        default auto
  --volume-dir DIR              optional network-volume mount, used as cache
                                (auto-detected: /workspace if writable)
  --sync-interval MIN           S3 sync cadence, default 5
  --s3-endpoint-url URL         for S3-compatible stores (optional)
  --s3-region REGION            (optional)
  --aws-access-key-id KEY       (or env AWS_ACCESS_KEY_ID)
  --aws-secret-access-key KEY   (or env AWS_SECRET_ACCESS_KEY)
  --runpod-api-key KEY          (or env RUNPOD_API_KEY; for self-terminate)
  --pod-id ID                   default $RUNPOD_POD_ID
  --post-run-cmd CMD            optional hook run after training, before final
                                sync (env: RUN_OUT_DIR, RUN_S3_OUT). Failure
                                only warns, never blocks termination.
  --extra-args "STR"            passed through to train_crnn_ctc.py
  --no-terminate                leave pod running at the end (debugging)
  -h|--help
EOF
}

# ------------------------------------------------------------------ arg parse -
POD_ID="${RUNPOD_POD_ID:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --s3-bucket)            S3_BUCKET="$2"; shift 2 ;;
    --s3-data-prefix)       S3_DATA_PREFIXES+=("$2"); shift 2 ;;
    --s3-data-tarball)      S3_DATA_TARBALLS+=("$2"); EXPECTED_SAMPLES+=(""); shift 2 ;;
    --expected-samples)     [[ ${#S3_DATA_TARBALLS[@]} -gt 0 ]] \
                              || die "--expected-samples must follow a --s3-data-tarball"
                            EXPECTED_SAMPLES[$(( ${#S3_DATA_TARBALLS[@]} - 1 ))]="$2"; shift 2 ;;
    --staging-timeout)      STAGING_TIMEOUT_MIN="$2"; shift 2 ;;
    --s3-output-prefix)     S3_OUT_PREFIX="$2"; shift 2 ;;
    --pkg)                  PKG="$2"; shift 2 ;;
    --epochs)               EPOCHS="$2"; shift 2 ;;
    --batch-size)           BATCH_SIZE="$2"; shift 2 ;;
    --lr)                   LR="$2"; shift 2 ;;
    --workers)              WORKERS="$2"; shift 2 ;;
    --seed)                 SEED="$2"; shift 2 ;;
    --device)               DEVICE="$2"; shift 2 ;;
    --volume-dir)           VOLUME_DIR="$2"; shift 2 ;;
    --sync-interval)        SYNC_INTERVAL_MIN="$2"; shift 2 ;;
    --s3-endpoint-url)      S3_ENDPOINT_URL="$2"; shift 2 ;;
    --s3-region)            S3_REGION="$2"; shift 2 ;;
    --aws-access-key-id)     AWS_ACCESS_KEY_ID="$2"; shift 2 ;;
    --aws-secret-access-key) AWS_SECRET_ACCESS_KEY="$2"; shift 2 ;;
    --runpod-api-key)       RUNPOD_API_KEY="$2"; shift 2 ;;
    --pod-id)               POD_ID="$2"; shift 2 ;;
    --post-run-cmd)         POST_RUN_CMD="$2"; shift 2 ;;
    --extra-args)           EXTRA_ARGS="$2"; shift 2 ;;
    --no-terminate)         NO_TERMINATE=1; shift ;;
    -h|--help)              usage; exit 0 ;;
    *) die "unknown arg: $1 (see --help)" ;;
  esac
done

[[ -n "$S3_BUCKET" ]]      || die "--s3-bucket is required"
[[ -n "$S3_OUT_PREFIX" ]]  || die "--s3-output-prefix is required"
[[ ${#S3_DATA_PREFIXES[@]} -gt 0 || ${#S3_DATA_TARBALLS[@]} -gt 0 ]] \
  || die "at least one --s3-data-prefix or --s3-data-tarball is required"
[[ -n "$PKG" ]]             || die "--pkg is required"
# normalise prefixes: exactly one trailing slash (tarballs are full s3:// URLs, untouched)
norm() { local p="${1#/}"; p="${p%/}/"; printf '%s' "$p"; }
S3_OUT_PREFIX="$(norm "$S3_OUT_PREFIX")"
for i in "${!S3_DATA_PREFIXES[@]}"; do S3_DATA_PREFIXES[$i]="$(norm "${S3_DATA_PREFIXES[$i]}")"; done
# watchdog timeout: positive number of minutes -> seconds
STAGING_TIMEOUT_S=$(STAGING_TIMEOUT_MIN="$STAGING_TIMEOUT_MIN" python3 -c "
import os
t = float(os.environ['STAGING_TIMEOUT_MIN'])
assert t > 0, 'must be positive'
print(int(t * 60))
" 2>/dev/null) || die "--staging-timeout must be a positive number of minutes (got '$STAGING_TIMEOUT_MIN')"

export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-}"
[[ -n "$AWS_ACCESS_KEY_ID" && -n "$AWS_SECRET_ACCESS_KEY" ]] \
  || die "S3 credentials missing: set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars or flags"

AWS_EXTRA=()
[[ -n "$S3_ENDPOINT_URL" ]] && AWS_EXTRA+=(--endpoint-url "$S3_ENDPOINT_URL")
[[ -n "$S3_REGION" ]]       && AWS_EXTRA+=(--region "$S3_REGION")

# ---------------------------------------------------------------- work dirs --
# Prefer the network volume as a download cache when it exists; S3 remains the
# source of truth either way. Never assume the pod's disk is big or persistent.
if [[ -z "$VOLUME_DIR" && -d /workspace && -w /workspace ]]; then
  VOLUME_DIR="/workspace"
fi
if [[ -n "$VOLUME_DIR" ]]; then
  [[ -d "$VOLUME_DIR" && -w "$VOLUME_DIR" ]] || die "--volume-dir $VOLUME_DIR not writable"
  # v6: FIXED workdir (no timestamp). 2026-10-07 incident: the container
  # crash-looped ~every 5min during the 8.6GB download; each restart made a NEW
  # timestamped workdir, orphaning the 3.2GB partial download and restarting
  # from scratch (6.5hrs / $4.80 burned). A fixed path lets restarts reuse
  # completed downloads. The S3 *output* prefix keeps its timestamp (start.sh)
  # for result uniqueness; only the LOCAL workdir is fixed.
  WORKDIR="$VOLUME_DIR/omr-staging"
else
  WORKDIR="$HOME/omr-staging"
fi
mkdir -p "$WORKDIR"
STAGE="$WORKDIR/stage"; CODE="$WORKDIR/code"; OUT="$WORKDIR/out"
mkdir -p "$STAGE" "$CODE" "$OUT"
LOGFILE="$WORKDIR/run.log"
touch "$LOGFILE"

log "run_train.sh starting"
log "workdir=$WORKDIR (volume: ${VOLUME_DIR:-none})"
log "bucket=$S3_BUCKET out=s3://$S3_BUCKET/$S3_OUT_PREFIX"

# s3mini.py -- stdlib-only S3 client (urllib + SigV4 header auth).
# 2026-10-06: runpod/pytorch images ship no aws CLI, and pip install is too
# slow/flaky at container startup (8 dead pods on 2026-10-06). s3mini.py
# lives next to this script (baked into /opt/omr/ by the image, or
# downloaded next to it by bootstrap.py). Resolve via this script's own
# directory -- never PATH, never cwd, never pip. Works as
# /opt/omr/run_train.sh, /tmp/run_train.sh, or a relative invocation.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
S3MINI_PY="$SCRIPT_DIR/s3mini.py"  # kept for reference only; S3 ops use the
# official AWS CLI now (s3mini.py SigV4 proved broken 2026-10-06)
s3() { aws s3 "$@" --endpoint-url "$S3_ENDPOINT_URL" --region "${S3_REGION:-us-ca-2}"; }

# ------------------------------------------------------- self-terminate -----
# Defined early: the staging watchdog (below) may need it before training.
terminate_pod() {
  [[ "$NO_TERMINATE" == "1" ]] && { log "not terminating (--no-terminate)"; return 0; }
  local key="${RUNPOD_API_KEY:-}"
  [[ -n "$POD_ID" ]] || { log "WARN: no pod id (set --pod-id or RUNPOD_POD_ID); NOT terminating"; return 1; }
  [[ -n "$key" ]]    || { log "WARN: no RUNPOD_API_KEY; NOT terminating (do it manually)"; return 1; }
  log "terminating pod $POD_ID ..."
  curl -fsS -X POST "https://api.runpod.io/graphql" \
    -H "Authorization: Bearer $key" -H "Content-Type: application/json" \
    -d "{\"query\":\"mutation { podTerminate(input: { podId: \\\"$POD_ID\\\" }) { id } }\"}" \
    && log "terminate request sent" \
    || { log "ERROR: terminate API call failed — terminate manually"; return 1; }
}

# ------------------------------------------------------------------ cleanup --
# NOTE: the EXIT trap is installed HERE (early), not at the end of the script.
# 2026-10-06 bug: the trap used to be set after the staging section, so any
# die() during fail-fast/staging exited WITHOUT cleanup — no log push to S3,
# no self-terminate, and the staging monitor was orphaned (leaked forever).
# Never move this trap down again. All trap-referenced vars are initialised
# here so cleanup is safe no matter where die() fires (set -u is on).
SYNC_PID=""; STAGING_MON_PID=""; STAGE_OK=0; HB_STATE=""
cleanup() {
  local rc=$?
  if [[ -n "$SYNC_PID" ]]; then
    kill "$SYNC_PID" 2>/dev/null || true
    wait "$SYNC_PID" 2>/dev/null || true
  fi
  stop_staging_monitor 2>/dev/null || true
  if [[ "$rc" -eq 0 ]]; then
    if final_sync && verify_outputs; then
      log "ALL CHECKS PASSED — results are safe on S3"
      terminate_pod || rc=1
    else
      log "VERIFICATION FAILED — pod LEFT RUNNING for inspection. Results (partial) are on S3."
      rc=1
    fi
  else
    if [[ "$STAGE_OK" == "1" ]]; then
      log "RUN FAILED (rc=$rc) after staging — attempting final sync, pod LEFT RUNNING for inspection"
      final_sync || true
    else
      # RULES.md Rule 9: failed during startup/staging — training never began,
      # nothing on the pod worth inspecting. Push the log, then self-terminate
      # instead of idling on GPU money.
      log "RUN FAILED (rc=$rc) during startup/staging — pushing log to S3 and self-terminating per RULES.md Rule 9"
      s3 cp "$LOGFILE" "s3://${S3_BUCKET}/${S3_OUT_PREFIX}run.log" >/dev/null 2>&1 || true
      hb_push 2>/dev/null || true
      terminate_pod || true
    fi
  fi
  log "run_train.sh exiting rc=$rc"
  exit "$rc"
}
trap cleanup EXIT

# ------------------------------------------------------- fail fast: tools ---
[[ -f "$S3MINI_PY" ]] || die "s3mini.py not found next to run_train.sh ($S3MINI_PY) -- bootstrap.py should have downloaded it"
command -v python3 >/dev/null || die "python3 not found"
python3 -c "import torch" 2>/dev/null || log "WARN: torch not importable yet (training image should have it)"

# ------------------------------------------------- fail fast: S3 reachable ---
s3 ls "s3://${S3_BUCKET}/" >/dev/null \
  || die "cannot list s3://$S3_BUCKET/ — check bucket name, region and credentials"
for pfx in "${S3_DATA_PREFIXES[@]}"; do
  n=$(s3 ls "s3://${S3_BUCKET}/${pfx}" 2>/dev/null | wc -l)
  [[ "$n" -gt 0 ]] || die "data prefix s3://$S3_BUCKET/$pfx is empty or missing"
  log "data prefix OK: s3://$S3_BUCKET/$pfx"
done
declare -a S3_TARBALL_SIZES=()
for tb in "${S3_DATA_TARBALLS[@]}"; do
  [[ "$tb" == s3://* ]] || die "--s3-data-tarball must be a full s3:// URL (got '$tb')"
  bn="$(basename "$tb")"
  sz=$(s3 ls "$tb" 2>/dev/null | awk -v bn="$bn" '$4==bn {print $3}')
  if ! [[ "$sz" =~ ^[0-9]+$ ]] || [[ "$sz" -le 0 ]]; then
    die "data tarball $tb is missing on S3 (aws s3 ls returned nothing usable)"
  fi
  S3_TARBALL_SIZES+=("$sz")
  log "data tarball OK: $tb ($(( sz / 1024 / 1024 ))MB on S3)"
done
# dynamic disk check (2026-10-06: Verovio tarball alone is 8.5GB — the old
# static >3GB check would let the run die mid-download, burning GPU money).
# Peak disk per tarball = download copy + untarred copy; require
# 2x total tarball bytes + 2GB headroom, checked BEFORE any GPU time burns.
if [[ ${#S3_TARBALL_SIZES[@]} -gt 0 ]]; then
  TOTAL_TB_BYTES=0
  for sz in "${S3_TARBALL_SIZES[@]}"; do TOTAL_TB_BYTES=$(( TOTAL_TB_BYTES + sz )); done
  FREE_BYTES=$(( $(free_kb) * 1024 ))
  NEED_BYTES=$(( TOTAL_TB_BYTES * 2 + 2 * 1024 * 1024 * 1024 ))
  if [[ "$FREE_BYTES" -le "$NEED_BYTES" ]]; then
    die "DISK: need $(( NEED_BYTES / 1024 / 1024 / 1024 ))GB free (2x tarballs [$(( TOTAL_TB_BYTES / 1024 / 1024 ))MB] + 2GB headroom), have $(( FREE_BYTES / 1024 / 1024 / 1024 ))GB in $WORKDIR"
  fi
  log "disk OK: have $(( FREE_BYTES / 1024 / 1024 / 1024 ))GB free, need $(( NEED_BYTES / 1024 / 1024 / 1024 ))GB"
fi

# ------------------------------------------- fail fast: S3 writable (canary) -
CANARY="s3://${S3_BUCKET}/${S3_OUT_PREFIX}.write_test"
echo "write-test $(TS)" | s3 cp - "$CANARY" >/dev/null \
  || die "CANARY WRITE FAILED: cannot write to s3://$S3_BUCKET/$S3_OUT_PREFIX — fix permissions before burning GPU time"
s3 rm "$CANARY" >/dev/null 2>&1 || true
log "S3 output writable: s3://$S3_BUCKET/$S3_OUT_PREFIX"

# ------------------------------------------------------ fail fast: packaging -
if [[ "$PKG" == s3://* ]]; then
  log "downloading training package: $PKG"
  s3 cp "$PKG" "$WORKDIR/training-pkg.tar.gz" >/dev/null \
    || die "cannot download training package $PKG"
  PKG="$WORKDIR/training-pkg.tar.gz"
fi
[[ -f "$PKG" ]] || die "training package not found: $PKG"
tar xf "$PKG" -C "$CODE" || die "cannot unpack $PKG"
[[ -f "$CODE/m2/train_crnn_ctc.py" ]] || die "package has no m2/train_crnn_ctc.py — re-package with package_for_training.sh"
# version-guard, belt and suspenders (train_crnn_ctc.py also asserts at import):
VOCAB_SIZE_GOT=$(python3 -c "import sys; sys.path.insert(0,'$CODE/m2'); import vocab; print(vocab.VOCAB_SIZE)")
[[ "$VOCAB_SIZE_GOT" == "425" ]] \
  || die "VOCAB MISMATCH: package ships $VOCAB_SIZE_GOT tokens, want 425 — re-package with package_for_training.sh"
log "vocab OK: 425 tokens (package ${PKG##*/})"

# disk space: tarball mode already did its dynamic check above; legacy
# per-file sync mode keeps the old static check (~1GB data + headroom)
if [[ ${#S3_DATA_TARBALLS[@]} -eq 0 ]]; then
  FREE_KB=$(free_kb)
  [[ "$FREE_KB" -gt 3145728 ]] || die "only $((FREE_KB/1024))MB free in $WORKDIR, need >3GB for staging"
fi
log "fail-fast checks passed"

# ------------------------------------------------- staging heartbeat --------
# 2026-10-06 incident: staging 220k small files ran for hours with ZERO
# progress signal (no one could tell slow from stuck). Every staging phase now
# writes a progress JSON to S3:
#   s3://<bucket>/<out-prefix>.staging_progress.json
#   {phase, done_files, total_files, bytes, note, timestamp, pod}
# A background monitor uploads it regularly and enforces the watchdog:
# no heartbeat progress for --staging-timeout minutes -> self-terminate
# (RULES.md Rule 9: startup fail means go, don't idle-burn GPU money).
MAIN_PID=$$
HB_STATE="$WORKDIR/.staging_hb.json"
HB_S3_KEY="s3://${S3_BUCKET}/${S3_OUT_PREFIX}.staging_progress.json"

hb_write() {  # hb_write PHASE DONE TOTAL BYTES NOTE — full state, fresh timestamp
  HB_PHASE="$1" HB_DONE="$2" HB_TOTAL="$3" HB_BYTES="$4" HB_NOTE="$5" \
  python3 - "$HB_STATE" <<'EOF'
import json, os, sys, time
st = {
  "phase":      os.environ["HB_PHASE"],
  "done_files": int(os.environ["HB_DONE"]),
  "total_files": int(os.environ["HB_TOTAL"]),
  "bytes":      int(os.environ["HB_BYTES"]),
  "note":       os.environ["HB_NOTE"],
  "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "pod":        os.environ.get("POD_ID", ""),
}
open(sys.argv[1], "w").write(json.dumps(st))
EOF
}
hb_push() {  # upload current heartbeat state to S3 (best effort)
  if [[ -n "${HB_STATE:-}" && -f "$HB_STATE" ]]; then
    s3 cp "$HB_STATE" "$HB_S3_KEY" >/dev/null 2>&1 || true
  fi
}

start_staging_monitor() {
  # $1 = watchdog timeout in seconds
  local timeout_s=$1 tick_s
  tick_s=$(( timeout_s > 180 ? 60 : timeout_s / 3 ))
  [[ "$tick_s" -lt 2 ]] && tick_s=2
  # NOTE: reset the inherited EXIT trap — the monitor must die silently when
  # killed, never run cleanup re-entrantly (that orphaned monitors before).
  ( trap - EXIT; set +e
    while true; do
      sleep "$tick_s"
      if [[ -f "$HB_STATE" ]]; then
        s3 cp "$HB_STATE" "$HB_S3_KEY" >/dev/null 2>&1 || true
        # NOTE: pass $HB_STATE as argv, not via os.environ — the monitor is a
        # subshell (inherits shell vars by fork) but python3 is an external
        # process (sees only exported vars). os.environ['HB_STATE'] KeyErrors.
        ts=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['timestamp'])" "$HB_STATE" 2>/dev/null) || continue
        now_t=$(date -u +%s)
        ts_epoch=$(date -u -d "$ts" +%s 2>/dev/null || echo "$now_t")
        age=$(( now_t - ts_epoch ))
        if [[ "$age" -gt "$timeout_s" ]]; then
          echo "[$(TS)] STAGING WATCHDOG: no heartbeat for ${age}s (limit ${timeout_s}s) — self-terminating per RULES.md Rule 9" | tee -a "$LOGFILE"
          s3 cp "$HB_STATE" "$HB_S3_KEY" >/dev/null 2>&1 || true
          s3 cp "$LOGFILE" "s3://${S3_BUCKET}/${S3_OUT_PREFIX}run.log" >/dev/null 2>&1 || true
          terminate_pod || true
          kill -TERM "$MAIN_PID" 2>/dev/null || exit 99
        fi
      fi
    done ) &
  STAGING_MON_PID=$!
  log "staging monitor on: heartbeat -> S3 every ${tick_s}s, watchdog ${timeout_s}s (pid $STAGING_MON_PID)"
}
stop_staging_monitor() {
  if [[ -n "${STAGING_MON_PID:-}" ]]; then
    kill "$STAGING_MON_PID" 2>/dev/null || true
    wait "$STAGING_MON_PID" 2>/dev/null || true
    STAGING_MON_PID=""
  fi
}

# ------------------------------------------------------------- stage data ---
# Two modes:
#   TARBALL MODE (preferred): one HTTP download per --s3-data-tarball, tar xf,
#     verify extracted counts vs --expected-samples. Fast: single stream, no
#     220k round-trips. Tarball internals may nest arbitrarily (recursive find).
#   LEGACY MODE (fallback): per-file `aws s3 sync` per --s3-data-prefix.
declare -a STAGE_DIRS=()
declare -a STAGE_LABELS=()

hb_write "staging-start" 0 0 0 "staging begins"; hb_push
start_staging_monitor "$STAGING_TIMEOUT_S"

if [[ ${#S3_DATA_TARBALLS[@]} -gt 0 ]]; then
  log "TARBALL MODE: ${#S3_DATA_TARBALLS[@]} tarball(s)"
  for i in "${!S3_DATA_TARBALLS[@]}"; do
    tb="${S3_DATA_TARBALLS[$i]}"
    exp="${EXPECTED_SAMPLES[$i]}"
    sz="${S3_TARBALL_SIZES[$i]}"
    d="$STAGE/tar_$i"; mkdir -p "$d"
    fn="$WORKDIR/data_$i.tar.gz"
    # v6: skip entirely if already extracted in a previous container run.
    # (Fixed WORKDIR survives restarts; verified by sample count, and the
    # count check after extraction below guards against partial extracts.)
    if [[ -n "$exp" ]]; then
      already=$(find "$d" -name '*.tokens.txt' 2>/dev/null | wc -l)
      if [[ "$already" -eq "$exp" ]]; then
        log "tarball $i already extracted ($already samples == expected $exp), skipping download+extract"
        STAGE_DIRS+=("$d/")
        STAGE_LABELS+=("tarball:$tb")
        hb_write "tarball-done" "$(( i + 1 ))" "${#S3_DATA_TARBALLS[@]}" "$sz" "$already samples from $tb (cached from previous run)"; hb_push
        continue
      elif [[ "$already" -gt 0 ]]; then
        log "tarball $i partially extracted ($already of $exp samples), clearing for clean re-extract"
        rm -rf "${d:?}/"*
      fi
    fi
    # per-tarball disk recheck (free space may have shrunk since fail-fast)
    free_b=$(( $(free_kb) * 1024 ))
    need_b=$(( sz * 2 + 2 * 1024 * 1024 * 1024 ))
    [[ "$free_b" -gt "$need_b" ]] \
      || die "DISK: need $(( need_b / 1024 / 1024 / 1024 ))GB free for $tb, have $(( free_b / 1024 / 1024 / 1024 ))GB"
    # v6: skip-if-complete — a fully downloaded tarball from a previous
    # container run is reused (size must match S3). A partial file is
    # discarded for a clean retry (aws s3 cp has no resume; a corrupt
    # partial is worse than a fresh download).
    if [[ -f "$fn" ]]; then
      fbytes=$(stat -c%s "$fn")
      if [[ "$fbytes" -eq "$sz" ]]; then
        log "tarball $i already downloaded ($(( fbytes / 1024 / 1024 ))MB == S3 size), skipping download"
      else
        log "tarball $i partial download ($fbytes of $sz bytes), discarding for clean retry"
        rm -f "$fn"
      fi
    fi
    if [[ ! -f "$fn" ]]; then
      hb_write "download" "$i" "${#S3_DATA_TARBALLS[@]}" "$sz" "downloading $tb ($(( sz / 1024 / 1024 ))MB)"; hb_push
      log "downloading $tb ($(( sz / 1024 / 1024 ))MB) ..."
      # v6: retry with exponential backoff. 2026-10-07 incident: the 8.6GB
      # verovio tarball died ~5min in (rc=255) on EVERY attempt — likely a
      # transient S3-gateway/network drop on the long multipart download.
      # aws stderr is captured and logged (previously >/dev/null hid the real
      # error); the log is synced to S3 so the next failure is diagnosable.
      dl_ok=0
      for attempt in 1 2 3 4 5; do
        dl_err="$WORKDIR/dl_${i}_attempt${attempt}.err"
        if s3 cp "$tb" "$fn" --quiet 2>"$dl_err"; then
          dl_ok=1; rm -f "$dl_err"; break
        fi
        log "download attempt $attempt/5 FAILED for $tb: $(tail -c 500 "$dl_err" | tr '\n\r' '  ')"
        rm -f "$fn"  # discard partial before retry
        if [[ "$attempt" -lt 5 ]]; then
          nap=$(( 2 ** attempt ))
          log "retrying in ${nap}s ..."
          sleep "$nap"
        fi
      done
      [[ "$dl_ok" == "1" ]] \
        || die "download failed after 5 attempts: $tb (aws error logged above)"
    fi
    fbytes=$(stat -c%s "$fn")
    [[ "$fbytes" -eq "$sz" ]] \
      || die "SIZE MISMATCH: $tb downloaded $fbytes bytes, S3 says $sz — truncated download, refusing to untar"
    log "downloaded $(( fbytes / 1024 / 1024 ))MB, size verified"
    hb_write "untar" "$i" "${#S3_DATA_TARBALLS[@]}" "$fbytes" "extracting $tb"; hb_push
    log "extracting ..."
    # NOTE: plain `tar xf` — GNU tar auto-detects compression, so this handles
    # .tar, .tar.gz and .tgz. (Do NOT use `tar xzf`: it dies on uncompressed .tar.)
    tar xf "$fn" -C "$d" \
      || die "CORRUPT TARBALL: tar xf failed for $tb — re-create the tarball and re-upload"
    got=$(find "$d" -name '*.tokens.txt' | wc -l)
    if [[ -n "$exp" ]]; then
      [[ "$got" -eq "$exp" ]] \
        || die "COUNT MISMATCH: $tb extracted $got samples, expected $exp — tarball incomplete or wrong"
    fi
    [[ "$got" -gt 0 ]] || die "tarball $tb extracted 0 samples"
    log "tarball $i: $got samples (expected ${exp:-unspecified})"
    rm -f "$fn"  # free disk; the extracted tree is what training uses
    STAGE_DIRS+=("$d/")
    STAGE_LABELS+=("tarball:$tb")
    hb_write "tarball-done" "$(( i + 1 ))" "${#S3_DATA_TARBALLS[@]}" "$fbytes" "$got samples from $tb"; hb_push
  done
else
  log "LEGACY SYNC MODE: ${#S3_DATA_PREFIXES[@]} prefix(es)"
  for i in "${!S3_DATA_PREFIXES[@]}"; do
    pfx="${S3_DATA_PREFIXES[$i]}"
    hb_write "sync" "$i" "${#S3_DATA_PREFIXES[@]}" 0 "syncing s3://$S3_BUCKET/$pfx"; hb_push
    log "syncing s3://$S3_BUCKET/$pfx ..."
    s3 sync "s3://${S3_BUCKET}/${pfx}" "$STAGE/raw_$i/" --quiet \
      || die "sync failed for s3://$S3_BUCKET/$pfx"
    STAGE_DIRS+=("$STAGE/raw_$i/")
    STAGE_LABELS+=("prefix:s3://$S3_BUCKET/$pfx")
    hb_write "sync-done" "$(( i + 1 ))" "${#S3_DATA_PREFIXES[@]}" 0 "synced $pfx"; hb_push
  done
fi

# Merge staged sources into one tree via per-source subdirs with file symlinks.
# (Subdirs avoid train-XXXXXXXX.png name collisions across engravers;
#  discover() uses rglob and follows file symlinks.)
#
# NOTE (2026-10-06): S3 data is NESTED (e.g. verovio-fixed/shard0/…shard9/).
# Staging MUST be recursive — a flat glob "$d"*.tokens.txt matched nothing
# (TOTAL=0, "merged dataset is empty"). find -name '*.tokens.txt' handles
# arbitrary nesting.
# Each data source is staged in order and must contribute >0 samples; an empty
# source dies loudly so a run never trains silently on partial data.
hb_write "merge" 0 "${#STAGE_DIRS[@]}" 0 "building symlink farm"; hb_push
MERGED="$STAGE/merged"
TOTAL=0
for idx in "${!STAGE_DIRS[@]}"; do
  d="${STAGE_DIRS[$idx]}"
  label="${STAGE_LABELS[$idx]}"
  [[ -d "$d" ]] || die "staging dir $d missing ($label)"
  name="src_$idx"
  mkdir -p "$MERGED/$name"
  SRC_COUNT=0
  MISSING_PNG=0
  while IFS= read -r t; do
    # unique target name: keep the relative subdir path (e.g.
    # shard3__train-00001234) so duplicate basenames across shards can't collide
    rel="${t#$d}"
    rel="${rel%.tokens.txt}"
    target_base="${rel//\//__}"
    png="${t%.tokens.txt}.png"   # same directory as the tokens file
    if [[ -f "$png" ]]; then
      ln -sf "$t"   "$MERGED/$name/$target_base.tokens.txt"
      ln -sf "$png" "$MERGED/$name/$target_base.png"
      SRC_COUNT=$((SRC_COUNT + 1))
    else
      MISSING_PNG=$((MISSING_PNG + 1))
      [[ "$MISSING_PNG" -le 5 ]] && log "WARN: $rel has tokens but no png, skipped"
    fi
  done < <(find "$d" -name '*.tokens.txt' | LC_ALL=C sort)
  [[ "$MISSING_PNG" -gt 5 ]] && log "WARN: $name had $MISSING_PNG tokens files without matching png (first 5 shown)"
  [[ "$SRC_COUNT" -gt 0 ]] \
    || die "data source $label contributed 0 samples — refusing to train on partial data"
  log "staged $SRC_COUNT samples from $name ($label)"
  TOTAL=$((TOTAL + SRC_COUNT))
  hb_write "merge" "$(( idx + 1 ))" "${#STAGE_DIRS[@]}" 0 "merged $name: $SRC_COUNT samples"; hb_push
done
[[ "$TOTAL" -gt 0 ]] || die "merged dataset is empty — check data sources"
log "staged $TOTAL samples into $MERGED"
hb_write "staged" "$TOTAL" "$TOTAL" 0 "staging complete: $TOTAL samples"; hb_push
stop_staging_monitor
STAGE_OK=1

# ------------------------------------------------------- background S3 sync --
SYNC_PID=""
start_sync_loop() {
  # trap - EXIT: background loop must die silently, never run cleanup itself.
  ( trap - EXIT; while true; do
      sleep $(( SYNC_INTERVAL_MIN * 60 ))
      s3 sync "$OUT/" "s3://${S3_BUCKET}/${S3_OUT_PREFIX}" --quiet 2>/dev/null \
        || log "WARN: background S3 sync failed (will retry)"
    done ) &
  SYNC_PID=$!
  log "background S3 sync every ${SYNC_INTERVAL_MIN}min (pid $SYNC_PID)"
}
final_sync() {
  log "final S3 sync..."
  s3 sync "$OUT/" "s3://${S3_BUCKET}/${S3_OUT_PREFIX}" --quiet \
    && log "final sync OK" || { log "ERROR: final sync failed"; return 1; }
}

# ------------------------------------------------- verify outputs on S3 -----
# IRON RULE: nothing is "done" until these exist AND are non-empty ON S3.
REQUIRED_FILES=(run_meta.json log.jsonl last.pt)
verify_outputs() {
  local ok=1 f size
  for f in "${REQUIRED_FILES[@]}"; do
    size=$(s3 ls "s3://${S3_BUCKET}/${S3_OUT_PREFIX}${f}" 2>/dev/null | awk '{print $3}')
    if [[ -z "${size:-}" || "$size" == "0" ]]; then
      log "VERIFY FAIL: s3://$S3_BUCKET/${S3_OUT_PREFIX}${f} missing or empty"
      ok=0
    else
      log "VERIFY OK: $f (${size} bytes on S3)"
    fi
  done
  # run_meta.json must say vocab.size == 425 (catches stale-code runs)
  local tmp; tmp=$(mktemp)
  if s3 cp "s3://${S3_BUCKET}/${S3_OUT_PREFIX}run_meta.json" "$tmp" --quiet 2>/dev/null; then
    local vsz
    vsz=$(python3 -c "import json;print(json.load(open('$tmp'))['vocab']['size'])")
    if [[ "$vsz" == "425" ]]; then
      log "VERIFY OK: run_meta vocab.size=425"
    else
      log "VERIFY FAIL: run_meta vocab.size=$vsz (want 425)"
      ok=0
    fi
  else
    log "VERIFY FAIL: cannot download run_meta.json from S3"
    ok=0
  fi
  rm -f "$tmp"
  if s3 ls "s3://${S3_BUCKET}/${S3_OUT_PREFIX}best.pt" >/dev/null 2>&1; then
    log "VERIFY INFO: best.pt present"
  else
    log "VERIFY INFO: best.pt absent (ok if val loss never improved)"
  fi
  [[ "$ok" == "1" ]]
}

# (cleanup() + trap cleanup EXIT live near the top of this script — see the
#  big NOTE there. They must be installed before any die() can fire.)

# ------------------------------------------------------------------- train --
start_sync_loop
log "starting training: epochs=$EPOCHS batch=$BATCH_SIZE lr=$LR workers=$WORKERS seed=$SEED device=$DEVICE"
# shellcheck disable=SC2086
set +e
python3 "$CODE/m2/train_crnn_ctc.py" \
  --data "$MERGED" \
  --out "$OUT" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --workers "$WORKERS" \
  --seed "$SEED" \
  --device "$DEVICE" \
  $EXTRA_ARGS 2>&1 | tee -a "$LOGFILE"
TRAIN_RC=${PIPESTATUS[0]}
set -e
[[ "$TRAIN_RC" -eq 0 ]] || die "train_crnn_ctc.py exited $TRAIN_RC"
log "training finished"

# optional post-run hook (e.g. X6 eval) — failure warns, never blocks
if [[ -n "$POST_RUN_CMD" ]]; then
  log "running post-run hook..."
  RUN_OUT_DIR="$OUT" RUN_S3_OUT="s3://$S3_BUCKET/$S3_OUT_PREFIX" \
    bash -c "$POST_RUN_CMD" 2>&1 | tee -a "$LOGFILE" \
    || log "WARN: post-run hook failed (results still valid)"
fi

log "done — cleanup trap will final-sync, verify and terminate"
