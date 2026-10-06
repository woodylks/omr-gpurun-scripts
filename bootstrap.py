#!/usr/bin/env python3
"""bootstrap.py -- stdlib-only pod bootstrapper (no aws CLI, no boto3, no pip).

Downloads s3mini.py + run_train.sh from S3 via SigV4 *header* auth,
verifies the script, then execs it.

RunPod start command (keep it SHORT -- dockerArgs gets truncated):

  python3 -c "import urllib.request;exec(urllib.request.urlopen('https://raw.githubusercontent.com/woodylks/omr-gpurun-scripts/main/bootstrap.py').read())"

Everything else comes from pod env vars (no shell operators, no quoting):

  Bootstrap (script download):
    OMR_S3_BUCKET            S3 bucket holding the scripts (required)
    OMR_S3_ENDPOINT_URL      S3-compatible endpoint (required)
    OMR_REGION               signing region, default us-east-1
    OMR_S3MINI_KEY           default omr-code/s3mini.py
    OMR_SCRIPT_KEY           default omr-code/run_train.sh
    OMR_DEST_DIR             default /tmp

  run_train.sh args (built from env):
    OMR_PKG                  -> --pkg (required)
    OMR_S3_OUTPUT_PREFIX     -> --s3-output-prefix (required)
    OMR_DATA_TARBALLS        -> comma-separated "s3://url:expected_samples"
                                (required, at least one pair)
    OMR_EPOCHS               -> --epochs (optional)
    OMR_NO_TERMINATE         -> --no-terminate if 1/true/yes (optional)
    OMR_S3_REGION            -> --s3-region (optional)
    (OMR_S3_BUCKET / OMR_S3_ENDPOINT_URL are reused for --s3-bucket /
     --s3-endpoint-url)

  AWS creds (both layers): AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
  (pod template env vars -- never on the command line).

Legacy form (argv flags override env; args after -- replace env-built args):
  python3 bootstrap.py --bucket B --endpoint-url E -- <run_train.sh args...>
"""
import hashlib
import hmac
import os
import sys
import time
import urllib.parse
import urllib.request


def _h(k, m):
    return hmac.new(k, m.encode(), hashlib.sha256).digest()


def s3_get(endpoint, bucket, key, ak, sk, region):
    """SigV4-signed GET, returns response bytes. Minimal, GET-only."""
    path = "/" + bucket + "/" + "/".join(
        urllib.parse.quote(s, safe="") for s in key.split("/"))
    url = endpoint.rstrip("/") + path
    host = urllib.parse.urlsplit(url).netloc
    amzdate = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    ds = amzdate[:8]
    ehash = hashlib.sha256(b"").hexdigest()
    creq = "\n".join(["GET", path, "",
                      "host:%s" % host, "x-amz-date:%s" % amzdate, "",
                      "host;x-amz-date", ehash])
    scope = "%s/%s/s3/aws4_request" % (ds, region)
    sts = "\n".join(["AWS4-HMAC-SHA256", amzdate, scope,
                     hashlib.sha256(creq.encode()).hexdigest()])
    k = _h(("AWS4" + sk).encode(), ds)
    for m in (region, "s3", "aws4_request"):
        k = _h(k, m)
    sig = hmac.new(k, sts.encode(), hashlib.sha256).hexdigest()
    req = urllib.request.Request(url, headers={
        "Authorization": ("AWS4-HMAC-SHA256 Credential=%s/%s, "
                          "SignedHeaders=host;x-amz-date, Signature=%s"
                          % (ak, scope, sig)),
        "x-amz-date": amzdate})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def _truthy(v):
    return v.strip().lower() in ("1", "true", "yes", "on")


def build_run_args():
    """Build run_train.sh argv from OMR_* env vars. List, or None on error."""
    missing = [ev for ev in ("OMR_S3_BUCKET", "OMR_S3_ENDPOINT_URL", "OMR_PKG",
                             "OMR_S3_OUTPUT_PREFIX", "OMR_DATA_TARBALLS")
               if not os.environ.get(ev)]
    if missing:
        sys.stderr.write("bootstrap: missing required env vars: %s\n"
                         % ", ".join(missing))
        return None
    args = ["--s3-bucket", os.environ["OMR_S3_BUCKET"],
            "--s3-endpoint-url", os.environ["OMR_S3_ENDPOINT_URL"],
            "--pkg", os.environ["OMR_PKG"],
            "--s3-output-prefix", os.environ["OMR_S3_OUTPUT_PREFIX"]]
    region = os.environ.get("OMR_S3_REGION", "")
    if region:
        args += ["--s3-region", region]
    epochs = os.environ.get("OMR_EPOCHS", "")
    if epochs:
        args += ["--epochs", epochs]
    for pair in os.environ["OMR_DATA_TARBALLS"].split(","):
        pair = pair.strip()
        if not pair:
            continue
        url, sep, n = pair.rpartition(":")
        if not sep or not url or not n.strip().isdigit():
            sys.stderr.write("bootstrap: bad OMR_DATA_TARBALLS entry %r "
                             "(want s3://url:expected_samples)\n" % pair)
            return None
        args += ["--s3-data-tarball", url, "--expected-samples", n.strip()]
    if _truthy(os.environ.get("OMR_NO_TERMINATE", "")):
        args += ["--no-terminate"]
    return args


def main(argv):
    cfg = {"bucket": "", "endpoint-url": "", "region": "us-east-1",
           "s3mini-key": "omr-code/s3mini.py",
           "script-key": "omr-code/run_train.sh", "dest-dir": "/tmp"}
    env_map = {"bucket": "OMR_S3_BUCKET", "endpoint-url": "OMR_S3_ENDPOINT_URL",
               "region": "OMR_REGION", "s3mini-key": "OMR_S3MINI_KEY",
               "script-key": "OMR_SCRIPT_KEY", "dest-dir": "OMR_DEST_DIR"}
    for k, ev in env_map.items():
        v = os.environ.get(ev, "")
        if v:
            cfg[k] = v
    rest, i, has_dd = None, 0, False
    while i < len(argv):
        a = argv[i]
        if a == "--":
            rest = argv[i + 1:]
            has_dd = True
            break
        if a.startswith("--") and a[2:] in cfg:
            if i + 1 >= len(argv):
                sys.stderr.write("bootstrap: --%s needs a value\n" % a[2:])
                return 2
            cfg[a[2:]] = argv[i + 1]
            i += 2
        else:
            sys.stderr.write("bootstrap: bad arg %r\n" % a)
            return 2
    if not has_dd:
        # Normal path (incl. the short dockerArgs one-liner): env drives.
        rest = build_run_args()
        if rest is None:
            return 2
    ak, sk = os.environ.get("AWS_ACCESS_KEY_ID", ""), os.environ.get(
        "AWS_SECRET_ACCESS_KEY", "")
    if not (cfg["bucket"] and cfg["endpoint-url"] and ak and sk):
        sys.stderr.write("bootstrap: need OMR_S3_BUCKET, OMR_S3_ENDPOINT_URL "
                         "and AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY env\n")
        return 2
    d = cfg["dest-dir"]
    os.makedirs(d, exist_ok=True)
    for key, name in ((cfg["s3mini-key"], "s3mini.py"),
                      (cfg["script-key"], "run_train.sh")):
        p = os.path.join(d, name)
        sys.stderr.write("bootstrap: downloading s3://%s/%s ...\n"
                         % (cfg["bucket"], key))
        data = s3_get(cfg["endpoint-url"], cfg["bucket"], key,
                      ak, sk, cfg["region"])
        with open(p, "wb") as f:
            f.write(data)
        sys.stderr.write("bootstrap: wrote %s (%d bytes)\n" % (p, len(data)))
    script = os.path.join(d, "run_train.sh")
    if not open(script, "rb").readline().startswith(b"#!"):
        sys.stderr.write("bootstrap: FATAL: %s is not a script "
                         "(bad download?)\n" % script)
        return 1
    sys.stderr.write("bootstrap: exec run_train.sh\n")
    sys.stderr.flush()
    os.execv("/bin/bash", ["bash", script] + rest)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
