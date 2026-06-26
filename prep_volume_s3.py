#!/usr/bin/env python3
"""
Fill a RunPod network volume with the AgentNet dataset from OUR OWN S3 copy.

Sibling of prep_volume.py, but the source is our S3 mirror instead of HF. Use
this when the HF pull is the bottleneck (CPU-bound 7z extraction, or anonymous
429 rate limits): the S3 copy at agentnet-full is already extracted, so this is
a pure byte transfer -- list the prefix, fan out loose-file GETs onto the
mounted volume, done. No huggingface_hub, no hf_transfer, no 7z.

A RunPod network volume is exposed both as a mounted filesystem and via an
S3-compatible API over the same bytes, so writing to the mount == filling "your
volume's S3". We pull FROM our omnibrowse-s3 bucket and write TO the mount.

Layout is preserved one-to-one: each S3 key under the prefix maps to
DEST_DIR/<key-relative-to-prefix>, matching what dataset/agentnet expects (the
same mapping download.py --full-s3 uses for a local checkout). Keys ending in
.tar are extracted in place (so an agentnet-reduced prefix of tars works too);
everything else lands as a loose file.

Idempotent: a same-size local file is treated as complete and skipped, so an
interrupted/retried run no-ops what is already on the volume. For .tar keys a
per-set sentinel on DEST_DIR is the source of truth (the tar is deleted
post-extract).

Env vars:
  DEST_DIR     extracted data target = the mounted volume.
               default: /workspace/agentnet  (RunPod mounts net volumes at /workspace)
  S3_URI       source prefix in our bucket.
               default: s3://omnibrowse-s3/datasets/agentnet-full
  S3_WORKERS   parallel transfers. The full set is hundreds of thousands of tiny
               PNGs, so this is latency-bound -- a wide pool of single-threaded
               GETs beats a few multi-threaded ones. default: 64
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION
               boto3 credentials. Pass as pod env vars / RunPod secrets -- never
               bake them into the image.

  SELF_TERMINATE  set to "1" to terminate this RunPod pod when the job ends
               (success OR failure), so it stops billing instead of sitting
               "Exited" but allocated. A finished container does NOT stop
               billing on its own. Needs RUNPOD_API_KEY + RUNPOD_POD_ID.
  RUNPOD_API_KEY  RunPod API key (pass as a secret). RUNPOD_POD_ID is injected
               by RunPod automatically. The network volume survives termination,
               so resume state for completed files is preserved.
"""
import os
import sys
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEST_DIR = Path(os.environ.get("DEST_DIR", "/workspace/agentnet"))
S3_URI = os.environ.get("S3_URI", "s3://omnibrowse-s3/datasets/agentnet-full")
S3_WORKERS = int(os.environ.get("S3_WORKERS", "64"))


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {uri!r}")
    bucket, _, key = uri[len("s3://"):].partition("/")
    return bucket, key.rstrip("/")


def _sentinel(name: str) -> Path:
    return DEST_DIR / f".{name}.extracted"


def _loose_transfer_config():
    # Tiny objects: per-file threading buys nothing and would explode the thread
    # count under the outer pool. Transfer each file single-threaded and get
    # parallelism purely from the ThreadPoolExecutor.
    from boto3.s3.transfer import TransferConfig
    return TransferConfig(use_threads=False)


def _fetch_one(s3, bucket: str, key: str, size: int, rel: str) -> None:
    """Download a single key to DEST_DIR/<rel>, extracting if it's a .tar."""
    dest = DEST_DIR / rel

    if rel.endswith(".tar"):
        img_set = rel[:-len(".tar")]
        if _sentinel(img_set).exists():
            return
        from boto3.s3.transfer import TransferConfig
        tar_cfg = TransferConfig(
            multipart_threshold=64 * 1024 * 1024,
            multipart_chunksize=64 * 1024 * 1024,
            max_concurrency=16,
            use_threads=True,
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(dest), Config=tar_cfg)
        target = DEST_DIR / img_set
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(dest) as tar:
            tar.extractall(target, filter="data")
        dest.unlink()
        _sentinel(img_set).touch()
        return

    # Loose file. Resume/idempotency: a same-size local file is assumed complete.
    if dest.exists() and dest.stat().st_size == size:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(dest), Config=_loose_transfer_config())


def download_from_s3() -> None:
    import boto3

    bucket, prefix = _parse_s3_uri(S3_URI)
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[cfg] DEST={DEST_DIR}  SRC=s3://{bucket}/{prefix}  WORKERS={S3_WORKERS}",
          flush=True)

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    objects: list[tuple[str, int, str]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix) + 1:]
            if not rel:  # the prefix "directory" placeholder, if any
                continue
            objects.append((key, obj["Size"], rel))

    if not objects:
        raise RuntimeError(
            f"No objects under s3://{bucket}/{prefix}/. Upload the full "
            "dataset there first (done out-of-band, or via download.py)."
        )

    total = len(objects)
    print(f"[dl] {total} files from s3://{bucket}/{prefix}/", flush=True)
    done = 0
    lock = threading.Lock()

    def _task(key: str, size: int, rel: str) -> None:
        nonlocal done
        _fetch_one(s3, bucket, key, size, rel)
        with lock:
            done += 1
            if done % 2000 == 0 or done == total:
                print(f"[dl] {done}/{total}", flush=True)

    with ThreadPoolExecutor(max_workers=S3_WORKERS) as ex:
        futures = [ex.submit(_task, k, sz, rel) for k, sz, rel in objects]
        for fut in as_completed(futures):
            fut.result()

    print(f"[ready] AgentNet on the volume at {DEST_DIR}", flush=True)


def _self_terminate() -> None:
    """Terminate this pod via the RunPod API to stop compute billing. Opt-in.

    A finished container leaves the pod allocated-and-billed; only terminating
    the *pod* halts billing. Best-effort: if the API key is missing or the call
    fails, we log loudly and leave the pod up rather than risk a silent leak the
    other way (a stuck pod is recoverable by hand; a wrongly-killed one isn't).
    """
    if os.environ.get("SELF_TERMINATE") != "1":
        print("[shutdown] SELF_TERMINATE!=1; leaving pod up. Stop it manually to halt billing.",
              flush=True)
        return

    api_key = os.environ.get("RUNPOD_API_KEY")
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not (api_key and pod_id):
        print("[shutdown] RUNPOD_API_KEY or RUNPOD_POD_ID missing; cannot self-terminate. "
              "Stop the pod manually!", flush=True)
        return

    import json
    import urllib.request

    query = 'mutation { podTerminate(input: {podId: "%s"}) }' % pod_id
    req = urllib.request.Request(
        f"https://api.runpod.io/graphql?api_key={api_key}",
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    print(f"[shutdown] terminating pod {pod_id} to stop billing...", flush=True)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            print(f"[shutdown] runpod api -> {r.status} {r.read().decode()[:200]}", flush=True)
    except Exception as e:
        print(f"[shutdown] terminate request failed: {e!r}. Stop the pod manually!", flush=True)


def main() -> None:
    download_from_s3()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Honor "stop paying when it's over" even on failure -- completed files
        # are already on the persistent volume, so a retry resumes from there.
        print(f"[error] {e!r}", file=sys.stderr, flush=True)
        _self_terminate()
        raise
