# omr-gpurun-scripts

Public bootstrap scripts for OMR GPU training runs (Trumpet Tone Lab).

These scripts are intentionally **public**: GPU pods download them at startup
via `raw.githubusercontent.com` without authentication. They contain **no
secrets** — every credential, bucket and endpoint comes from environment
variables, never hardcoded.

## Files

| File | What |
|---|---|
| `bootstrap.py` | stdlib-only pod bootstrapper. Downloads `s3mini.py` + `run_train.sh` from S3 (SigV4 header auth), verifies the script, execs it. |
| `s3mini.py` | stdlib-only S3 client (`urllib` + SigV4). No boto3, no aws CLI. `cp`/`ls`/`sync`/`rm`. |
| `run_train.sh` | Training runner: tarball staging + heartbeat/watchdog, training, X6 eval hook, S3 sync, verify-before-terminate. |

## Pod start command

Keep it short — RunPod truncates long `dockerArgs` (1792 chars already fails):

```
python3 -c "import urllib.request;exec(urllib.request.urlopen('https://raw.githubusercontent.com/woodylks/omr-gpurun-scripts/main/bootstrap.py').read())"
```

153 chars. Zero shell operators. Everything else comes from env vars.

## Env vars

Pod template env vars (Elon fills these per run):

**Bootstrap (script download):**

| Var | Required | Example |
|---|---|---|
| `OMR_S3_BUCKET` | yes | `my-bucket` |
| `OMR_S3_ENDPOINT_URL` | yes | `https://s3.example.com` |
| `OMR_REGION` | no | default `us-east-1` |
| `OMR_S3MINI_KEY` | no | default `omr-code/s3mini.py` |
| `OMR_SCRIPT_KEY` | no | default `omr-code/run_train.sh` |
| `OMR_DEST_DIR` | no | default `/tmp` |

**Training (`run_train.sh` args, built from env):**

| Var | Required | Maps to |
|---|---|---|
| `OMR_PKG` | yes | `--pkg` (training package tarball `s3://…`) |
| `OMR_S3_OUTPUT_PREFIX` | yes | `--s3-output-prefix` |
| `OMR_DATA_TARBALLS` | yes | comma-separated `s3://url:expected_samples` pairs |
| `OMR_EPOCHS` | no | `--epochs` |
| `OMR_NO_TERMINATE` | no | `--no-terminate` if `1`/`true`/`yes` |
| `OMR_S3_REGION` | no | `--s3-region` |

(`OMR_S3_BUCKET` / `OMR_S3_ENDPOINT_URL` are reused for `--s3-bucket` /
`--s3-endpoint-url`.)

**AWS creds (both layers):** `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`.

## Test before GPU

Cheapest CPU pod first: check `/tmp/bootstrap.py` exists and the S3
download starts. Only then open the GPU pod.
