"""
Standalone data-prep job: download AgentNet from HF, extract it, and upload the
full *uncompressed* dataset to S3 as loose files.

This is the producer side of download.py's `--full-s3` mode. It writes the exact
S3 key layout that download_full_s3() reads back — each object's key is
`<prefix>/<path-relative-to-dataset/agentnet/data>` — so a training box can later
`python download.py --full-s3` and skip the CPU-bound 7z extraction entirely.

Intended to run once on a cheap CPU-only box (see Dockerfile.dataprep). The
upload code lives here rather than in download.py on purpose, so the GPU training
image never carries it.

Required env (same as run_train.sh): AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
and optionally AWS_DEFAULT_REGION.
"""
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Reuse the download/extract pipeline and the S3 helpers so there's a single
# source of truth for the on-disk layout and key scheme.
from download import (
    DATA_DIR,
    DEFAULT_FULL_S3_URI,
    _loose_transfer_config,
    _parse_s3_uri,
    download_agentnet,
)


def _iter_upload_files():
    """Yield (local_path, rel_posix_key) for every dataset file worth uploading.

    Skips dotfiles (the `.<set>.extracted` sentinels) and any leftover split-zip
    archive parts (`images.zip`, `images.z01`, ...) — those are extraction
    scratch, not dataset content.
    """
    for path in sorted(DATA_DIR.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(DATA_DIR)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if path.name.startswith("images.z"):  # images.zip / images.z01 / ...
            continue
        yield path, rel.as_posix()


def upload_full_s3(s3_uri: str, workers: int) -> None:
    import boto3

    bucket, prefix = _parse_s3_uri(s3_uri)
    s3 = boto3.client("s3")
    cfg = _loose_transfer_config()

    files = list(_iter_upload_files())
    if not files:
        raise RuntimeError(
            f"Nothing to upload under {DATA_DIR}. Run the download/extract first "
            "(don't pass --skip-download)."
        )

    # One cheap paginated listing lets interrupted runs resume: an object already
    # present at the same byte size is assumed complete, so we skip it instead of
    # issuing a HEAD request per file.
    existing: dict[str, int] = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
        for obj in page.get("Contents", []):
            existing[obj["Key"]] = obj["Size"]

    total = len(files)
    print(
        f"[upload] {total} files -> s3://{bucket}/{prefix}/ "
        f"({len(existing)} already present)",
        flush=True,
    )
    done = 0
    lock = threading.Lock()

    def _put(local, rel: str) -> None:
        nonlocal done
        key = f"{prefix}/{rel}"
        if existing.get(key) != local.stat().st_size:
            s3.upload_file(str(local), bucket, key, Config=cfg)
        with lock:
            done += 1
            if done % 2000 == 0 or done == total:
                print(f"[upload] {done}/{total}", flush=True)

    # Many small files: throughput is latency-bound, so a wide pool of
    # single-threaded transfers beats a few multi-threaded ones.
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_put, lp, rel) for lp, rel in files]
        for fut in as_completed(futures):
            fut.result()

    print(f"[done] full uncompressed dataset uploaded to s3://{bucket}/{prefix}", flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--s3-uri",
        default=DEFAULT_FULL_S3_URI,
        help=f"S3 prefix to upload the loose dataset to (default: {DEFAULT_FULL_S3_URI}).",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Assume dataset/agentnet/data is already populated; only upload.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Concurrent S3 uploads (default: 64). Bump on fat network links.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not args.skip_download:
        download_agentnet()
    upload_full_s3(args.s3_uri, args.workers)
