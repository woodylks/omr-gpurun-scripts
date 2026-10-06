#!/usr/bin/env python3
"""
s3mini.py -- stdlib-only S3 client (urllib + hmac + hashlib + ElementTree).

Drop-in replacement for the `aws s3 cp|ls|sync|rm` subset used by
run_train.sh. No boto3, no pip, no aws CLI -- works on any image with
python3 (e.g. runpod/pytorch, where `aws` is missing and pip install is
too slow/flaky at container startup).

Auth: SigV4 **header** auth (NOT presigned URLs -- this endpoint 401s
presigned GETs). Path-style addressing: https://endpoint/bucket/key.

Interface (mirrors the aws CLI calls run_train.sh makes):
    s3mini.py [--endpoint-url URL] [--region R]
              [--aws-access-key-id K] [--aws-secret-access-key K]
              cp <src> <dst>        # file->s3, s3->file, stdin(-)->s3
              ls s3://bucket[/prefix]
              sync <src> <dst>       # s3->local or local->s3 (size-compare)
              rm s3://bucket/key
Global flags may appear anywhere in argv. --quiet is accepted and ignored.

Credentials: env AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (or flags).
Region: --region / AWS_DEFAULT_REGION / AWS_REGION, default us-east-1.
Endpoint: --endpoint-url / AWS_ENDPOINT_URL, default https://s3.amazonaws.com.
"""
import hashlib
import hmac
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
CHUNK = 8 * 1024 * 1024
MAX_RETRIES = 4


class S3Error(Exception):
    pass


# ------------------------------------------------------------- SigV4 ----
def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _signing_key(secret: str, datestamp: str, region: str, service: str = "s3") -> bytes:
    k = _hmac(("AWS4" + secret).encode("utf-8"), datestamp)
    k = _hmac(k, region)
    k = _hmac(k, service)
    k = _hmac(k, "aws4_request")
    return k


def _qenc(s: str) -> str:
    # RFC3986 unreserved set only (letters/digits/_.-~); quote() never
    # touches those, so safe='' gives exactly the AWS-required encoding.
    return urllib.parse.quote(s, safe="")


def canonical_request(method, canonical_uri, canonical_qs, headers, payload_hash):
    """headers: list of (lowercase-name, trimmed-value), sorted by name."""
    lines = [method, canonical_uri, canonical_qs]
    for n, v in headers:
        lines.append("%s:%s" % (n, v))
    lines.append("")
    lines.append(";".join(n for n, _ in headers))
    lines.append(payload_hash)
    return "\n".join(lines)


def string_to_sign(amzdate, datestamp, region, service, creq):
    scope = "%s/%s/%s/aws4_request" % (datestamp, region, service)
    return "\n".join(
        ["AWS4-HMAC-SHA256", amzdate, scope,
         hashlib.sha256(creq.encode("utf-8")).hexdigest()]
    )


def calc_signature(secret, datestamp, region, service, sts):
    return hmac.new(
        _signing_key(secret, datestamp, region, service),
        sts.encode("utf-8"), hashlib.sha256,
    ).hexdigest()


def auth_headers(method, url, query_params, ak, sk, region, service,
                 payload_hash, amzdate=None, datestamp=None):
    """Return dict of SigV4 Authorization headers for the request.

    Signed header set is exactly what botocore (the AWS reference SDK)
    uses for S3: host + x-amz-date only. The payload hash still goes in
    the canonical request's last line; the x-amz-content-sha256 header
    itself is neither sent nor signed (matches botocore byte-for-byte).
    """
    parts = urllib.parse.urlsplit(url)
    if amzdate is None:
        now = time.gmtime()
        amzdate = time.strftime("%Y%m%dT%H%M%SZ", now)
        datestamp = time.strftime("%Y%m%d", now)
    # canonical URI: each path segment encoded, '/' preserved.
    # NOTE: '/'.join over path.split('/') already yields the leading slash
    # (the first segment is ''), so do NOT prepend another one.
    canonical_uri = "/".join(_qenc(seg) for seg in parts.path.split("/"))
    cqs = "&".join(
        "%s=%s" % (_qenc(k), _qenc(v))
        for k, v in sorted(query_params, key=lambda kv: kv[0])
    )
    headers = [("host", parts.netloc),
               ("x-amz-date", amzdate)]
    headers.sort(key=lambda kv: kv[0])
    creq = canonical_request(method, canonical_uri, cqs, headers, payload_hash)
    sts = string_to_sign(amzdate, datestamp, region, service, creq)
    sig = calc_signature(sk, datestamp, region, service, sts)
    scope = "%s/%s/%s/aws4_request" % (datestamp, region, service)
    signed = ";".join(n for n, _ in headers)
    auth = ("AWS4-HMAC-SHA256 Credential=%s/%s, SignedHeaders=%s, Signature=%s"
            % (ak, scope, signed, sig))
    return {"Authorization": auth, "x-amz-date": amzdate}


# ------------------------------------------------------------- client ---
def parse_s3_url(url):
    if not url.startswith("s3://"):
        raise S3Error("not an s3:// URL: %r" % url)
    rest = url[5:]
    if "/" in rest:
        bucket, key = rest.split("/", 1)
    else:
        bucket, key = rest, ""
    if not bucket:
        raise S3Error("empty bucket in %r" % url)
    return bucket, key


class S3:
    def __init__(self, endpoint, region, ak, sk):
        self.endpoint = endpoint.rstrip("/")
        self.region = region
        self.ak = ak
        self.sk = sk

    def _url(self, bucket, key, query=()):
        path = "/" + bucket + ("/" + "/".join(
            _qenc(seg) for seg in key.split("/")) if key else "")
        qs = "&".join("%s=%s" % (_qenc(k), _qenc(v)) for k, v in query)
        return "%s%s%s" % (self.endpoint, path, ("?" + qs) if qs else "")

    def _send(self, method, bucket, key, query=(), body=None, length=None,
              payload_hash=None, ctype=None):
        """body: bytes, file object, or None. Returns urllib response."""
        if payload_hash is None:
            payload_hash = EMPTY_SHA256
        url = self._url(bucket, key, query)
        hdrs = auth_headers(method, url, list(query), self.ak, self.sk,
                            self.region, "s3", payload_hash)
        if length is not None:
            hdrs["Content-Length"] = str(length)
        if ctype:
            hdrs["Content-Type"] = ctype
        req = urllib.request.Request(url, data=body, headers=hdrs,
                                     method=method)
        last = None
        for attempt in range(MAX_RETRIES):
            try:
                return urllib.request.urlopen(req, timeout=600)
            except urllib.error.HTTPError as e:
                if e.code in (500, 502, 503, 504) and attempt < MAX_RETRIES - 1:
                    last = e
                    time.sleep(2 ** attempt)
                    continue
                body_txt = ""
                try:
                    body_txt = e.read().decode("utf-8", "replace")[:500]
                except Exception:
                    pass
                raise S3Error("%s %s -> HTTP %s %s" % (method, url, e.code,
                                                       body_txt))
            except urllib.error.URLError as e:
                if attempt < MAX_RETRIES - 1:
                    last = e
                    time.sleep(2 ** attempt)
                    continue
                raise S3Error("%s %s -> URL error: %s" % (method, url, e))
        raise S3Error("%s %s -> failed after retries: %s" % (method, url, last))

    # -- objects ---------------------------------------------------------
    def list_objects(self, bucket, prefix):
        """Yield dicts {key, size, lastmodified}. Handles pagination."""
        token = None
        while True:
            q = [("list-type", "2"), ("prefix", prefix), ("max-keys", "1000")]
            if token:
                q.append(("continuation-token", token))
            resp = self._send("GET", bucket, "", query=q)
            xml = resp.read()
            resp.close()
            root = ET.fromstring(xml)
            # strip namespaces (some S3-compatible servers vary them)
            for el in root.iter():
                if "}" in el.tag:
                    el.tag = el.tag.split("}", 1)[1]

            def txt(elem, name):
                c = elem.find(name)
                return c.text if c is not None and c.text else ""

            for c in root.findall("Contents"):
                yield {"key": txt(c, "Key"),
                       "size": int(txt(c, "Size") or 0),
                       "lastmodified": txt(c, "LastModified")}
            is_trunc = txt(root, "IsTruncated").lower() == "true"
            token = txt(root, "NextContinuationToken")
            if not (is_trunc and token):
                break

    def get_to_file(self, bucket, key, dest):
        """Streaming download (8.5GB tarball never sits fully in RAM)."""
        d = os.path.dirname(os.path.abspath(dest))
        if d:
            os.makedirs(d, exist_ok=True)
        resp = self._send("GET", bucket, key)
        try:
            with open(dest, "wb") as f:
                shutil.copyfileobj(resp, f, CHUNK)
        finally:
            resp.close()

    def put_bytes(self, bucket, key, data: bytes):
        h = hashlib.sha256(data).hexdigest()
        self._send("PUT", bucket, key, body=data, length=len(data),
                   payload_hash=h,
                   ctype="application/octet-stream").close()

    def put_file(self, bucket, key, path):
        """Streaming upload: hash first (one pass), then stream the body."""
        size = os.path.getsize(path)
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                blk = f.read(CHUNK)
                if not blk:
                    break
                h.update(blk)
        with open(path, "rb") as f:
            self._send("PUT", bucket, key, body=f, length=size,
                       payload_hash=h.hexdigest(),
                       ctype="application/octet-stream").close()

    def delete(self, bucket, key):
        try:
            self._send("DELETE", bucket, key).close()
        except S3Error as e:
            # deleting a missing key is not an error (matches aws s3 rm)
            if "HTTP 404" not in str(e):
                raise


# ------------------------------------------------------------- commands -
def _rel_key(key, prefix):
    return key[len(prefix):] if key.startswith(prefix) else key


def cmd_ls(s3, args):
    bucket, prefix = parse_s3_url(args[0])
    for obj in s3.list_objects(bucket, prefix):
        # display key: listing prefix stripped; for an exact-key ls the
        # strip yields '' so fall back to basename (run_train.sh awks $4==basename)
        rel = _rel_key(obj["key"], prefix) or obj["key"].rsplit("/", 1)[-1]
        lm = obj["lastmodified"][:19].replace("T", " ")  # YYYY-MM-DD HH:MM:SS
        # aws s3 ls field layout: date, time, size, key (script awks $3/$4)
        sys.stdout.write("%s %12d %s\n" % (lm, obj["size"], rel))


def cmd_cp(s3, args):
    src, dst = args[0], args[1]
    if src == "-":  # stdin -> s3
        data = sys.stdin.buffer.read()
        bucket, key = parse_s3_url(dst)
        s3.put_bytes(bucket, key, data)
    elif src.startswith("s3://"):  # s3 -> file
        bucket, key = parse_s3_url(src)
        s3.get_to_file(bucket, key, dst)
    elif dst.startswith("s3://"):  # file -> s3
        bucket, key = parse_s3_url(dst)
        s3.put_file(bucket, key, src)
    else:
        raise S3Error("cp needs an s3:// endpoint: cp %r %r" % (src, dst))


def _download_if_needed(s3, bucket, key, local, remote_size):
    if os.path.isfile(local) and os.path.getsize(local) == remote_size:
        return False
    s3.get_to_file(bucket, key, local)
    return True


def cmd_sync(s3, args):
    src, dst = args[0], args[1]
    if src.startswith("s3://"):  # s3 -> local
        bucket, prefix = parse_s3_url(src)
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        for obj in s3.list_objects(bucket, prefix):
            rel = _rel_key(obj["key"], prefix)
            if not rel or rel.endswith("/"):
                continue
            _download_if_needed(s3, bucket, obj["key"],
                                os.path.join(dst, rel), obj["size"])
    elif dst.startswith("s3://"):  # local -> s3
        bucket, prefix = parse_s3_url(dst)
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        remote = {o["key"]: o["size"]
                  for o in s3.list_objects(bucket, prefix)}
        for root, _, files in os.walk(src):
            for fn in files:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, src).replace(os.sep, "/")
                key = prefix + rel
                if remote.get(key) != os.path.getsize(full):
                    s3.put_file(bucket, key, full)
    else:
        raise S3Error("sync needs an s3:// endpoint: sync %r %r" % (src, dst))


def cmd_rm(s3, args):
    bucket, key = parse_s3_url(args[0])
    s3.delete(bucket, key)


COMMANDS = {"cp": (cmd_cp, 2), "ls": (cmd_ls, 1),
            "sync": (cmd_sync, 2), "rm": (cmd_rm, 1)}

GLOBAL_OPTS = {"--endpoint-url": "endpoint", "--region": "region",
               "--aws-access-key-id": "ak", "--aws-secret-access-key": "sk"}


def main(argv):
    cfg = {"endpoint": os.environ.get("AWS_ENDPOINT_URL", ""),
           "region": os.environ.get("AWS_DEFAULT_REGION",
                                    os.environ.get("AWS_REGION", "us-east-1")),
           "ak": os.environ.get("AWS_ACCESS_KEY_ID", ""),
           "sk": os.environ.get("AWS_SECRET_ACCESS_KEY", "")}
    rest = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in GLOBAL_OPTS:
            if i + 1 >= len(argv):
                sys.stderr.write("s3mini: %s needs a value\n" % a)
                return 2
            cfg[GLOBAL_OPTS[a]] = argv[i + 1]
            i += 2
        elif a == "--quiet":
            i += 1  # accepted, ignored (script passes it everywhere)
        else:
            rest.append(a)
            i += 1
    if not rest or rest[0] not in COMMANDS:
        sys.stderr.write(
            "s3mini: usage: s3mini.py [global opts] <cp|ls|sync|rm> ...\n")
        return 2
    fn, nargs = COMMANDS[rest[0]]
    args = rest[1:]
    if len(args) < nargs:
        sys.stderr.write("s3mini: %s needs %d args\n" % (rest[0], nargs))
        return 2
    if not cfg["endpoint"]:
        cfg["endpoint"] = "https://s3.amazonaws.com"
    if not cfg["ak"] or not cfg["sk"]:
        sys.stderr.write("s3mini: missing credentials "
                         "(AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)\n")
        return 2
    s3 = S3(cfg["endpoint"], cfg["region"], cfg["ak"], cfg["sk"])
    try:
        fn(s3, args)
    except S3Error as e:
        sys.stderr.write("s3mini: %s\n" % e)
        return 1
    except BrokenPipeError:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
