"""
Download and extract the AgentNet dataset into dataset/agentnet/data/.

Default flow (no flags) — full dataset from HF, tuned to minimize wall-clock:
  * hf_transfer for multi-connection per-file downloads
  * metadata (jsonl) fetched first so it is ready while images are still arriving
  * each split-zip image set is downloaded and extracted in its own thread, so
    extraction of the smaller (ubuntu) set overlaps with download of the larger
    (win_mac) set
  * archive parts deleted post-extract to keep peak disk usage near the
    extracted size, not extracted + compressed
  * idempotent: sentinel files let interrupted/retried runs no-op on what is
    already complete; snapshot_download resumes partial files automatically

Test flow:
  python download.py --make-reduced [--n-trajs N] [--s3-uri ...]
      Subset a locally-extracted dataset to N trajectories per JSONL, tar up
      only the images those trajectories reference, push to S3.
  python download.py --reduced [--s3-uri ...]
      Pull and unpack that pre-built slice from S3 instead of doing the full HF
      download. Lands in the same dataset/agentnet/data/ layout so train.py
      doesn't need to know which mode produced it.

Full-from-S3 flow (uncompressed):
  python download.py --full-s3 [--s3-uri ...]
      Pull the full dataset straight from S3 as loose, already-extracted files.
      On machines where CPU-bound decompression is the bottleneck, this beats
      the HF zip download + 7z extract. Lands in the same
      dataset/agentnet/data/ layout. (The S3 copy is uploaded out-of-band.)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Must be set before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import snapshot_download  # noqa: E402

REPO_ID = "xlangai/AgentNet"
REPO_TYPE = "dataset"

DATA_DIR = Path(__file__).parent / "dataset" / "agentnet" / "data"

METADATA_PATTERNS = ["*.jsonl", "LICENSE.txt", "README.md", ".gitattributes"]
IMAGE_SETS = ["ubuntu_images", "win_mac_images"]
JSONL_FOR_SET = {
    "ubuntu_images": "agentnet_ubuntu_5k.jsonl",
    "win_mac_images": "agentnet_win_mac_18k.jsonl",
}

DEFAULT_REDUCED_S3_URI = "s3://omnibrowse-s3/datasets/agentnet-reduced"
DEFAULT_FULL_S3_URI = "s3://omnibrowse-s3/datasets/agentnet-full"


# ---------- full HF download ----------------------------------------------------


def _sentinel(name: str) -> Path:
    return DATA_DIR / f".{name}.extracted"


def _already_extracted(name: str) -> bool:
    return _sentinel(name).exists()


def _download_metadata() -> None:
    print("[download] metadata...", flush=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(DATA_DIR),
        allow_patterns=METADATA_PATTERNS,
        max_workers=4,
    )


def _extract(name: str) -> None:
    target = DATA_DIR / name
    zip_path = target / "images.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_path} missing — download incomplete")

    if shutil.which("7z") is None:
        raise RuntimeError(
            "7z not found. Install with `apt install p7zip-full` (handled in the Dockerfile)."
        )

    print(f"[extract] {name}...", flush=True)
    # 7z reads the .zip + all .z?? siblings natively. -bso0 silences stdout,
    # -bsp1 keeps progress on stderr.
    subprocess.run(
        ["7z", "x", "-y", "-bso0", "-bsp1", f"-o{target}", str(zip_path)],
        check=True,
    )

    # Free disk: drop archive parts now that PNGs are on disk.
    for part in sorted(target.glob("images.z*")):
        part.unlink()

    _sentinel(name).touch()
    print(f"[extract] {name} done", flush=True)


def _fetch_and_extract(name: str) -> None:
    if _already_extracted(name):
        print(f"[skip] {name} already extracted", flush=True)
        return

    print(f"[download] {name} archive parts...", flush=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(DATA_DIR),
        allow_patterns=[f"{name}/*"],
        max_workers=8,
    )
    _extract(name)


def download_agentnet() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _download_metadata()

    with ThreadPoolExecutor(max_workers=len(IMAGE_SETS)) as ex:
        futures = {ex.submit(_fetch_and_extract, n): n for n in IMAGE_SETS}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"[error] {name}: {e}", file=sys.stderr, flush=True)
                raise

    print(f"[done] dataset ready at {DATA_DIR}", flush=True)


# ---------- reduced slice (S3) --------------------------------------------------


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {uri!r}")
    bucket, _, key = uri[len("s3://"):].partition("/")
    return bucket, key.rstrip("/")


def _transfer_config():
    # Import lazily so the full-download path doesn't pay for boto3 startup.
    from boto3.s3.transfer import TransferConfig
    return TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=16,
        use_threads=True,
    )


def _loose_transfer_config():
    # The full uncompressed dataset is hundreds of thousands of small PNGs.
    # Per-file threading buys nothing for tiny objects and would explode the
    # thread count under the outer pool, so transfer each file single-threaded
    # and get parallelism purely from the outer ThreadPoolExecutor.
    from boto3.s3.transfer import TransferConfig
    return TransferConfig(use_threads=False)


def _subset_jsonl(src: Path, dst: Path, n_trajs: int) -> list[str]:
    """Copy the first n_trajs lines from src to dst. Return image filenames referenced."""
    images: list[str] = []
    with src.open() as fin, dst.open("w") as fout:
        for i, line in enumerate(fin):
            if i >= n_trajs:
                break
            traj = json.loads(line)
            for step in traj.get("traj", []):
                img = step.get("image")
                if img:
                    images.append(img)
            fout.write(line)
    return images


def make_reduced(s3_uri: str, n_trajs: int) -> None:
    import boto3

    bucket, prefix = _parse_s3_uri(s3_uri)
    s3 = boto3.client("s3")
    cfg = _transfer_config()

    # Sanity-check that the local full dataset is present.
    for img_set, jsonl in JSONL_FOR_SET.items():
        if not (DATA_DIR / jsonl).exists():
            raise FileNotFoundError(f"Missing {DATA_DIR / jsonl}; run a full download first.")
        if not (DATA_DIR / img_set).is_dir():
            raise FileNotFoundError(f"Missing {DATA_DIR / img_set}/; run a full download first.")

    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)

        # Subset each JSONL and tar only the PNGs those trajectories touch.
        for img_set, jsonl in JSONL_FOR_SET.items():
            referenced = _subset_jsonl(DATA_DIR / jsonl, tmp / jsonl, n_trajs)
            tar_path = tmp / f"{img_set}.tar"
            src_dir = DATA_DIR / img_set
            kept = 0
            with tarfile.open(tar_path, "w") as tar:
                for name in referenced:
                    p = src_dir / name
                    if p.exists():
                        tar.add(p, arcname=name)
                        kept += 1
            print(f"[reduce] {jsonl}: {n_trajs} trajs, {kept}/{len(referenced)} images -> {tar_path.name}",
                  flush=True)

        # meta_data_merged.jsonl is small and dataset code may want it later.
        meta = DATA_DIR / "meta_data_merged.jsonl"
        if meta.exists():
            shutil.copy(meta, tmp / meta.name)

        # Upload everything in tmp to the S3 prefix.
        for f in sorted(tmp.iterdir()):
            key = f"{prefix}/{f.name}"
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"[upload] s3://{bucket}/{key} ({size_mb:.1f} MB)", flush=True)
            s3.upload_file(str(f), bucket, key, Config=cfg)

    print(f"[done] reduced dataset uploaded to s3://{bucket}/{prefix}", flush=True)


def _download_one_s3_object(s3, bucket: str, key: str, cfg) -> None:
    name = key.rsplit("/", 1)[-1]
    if not name:
        return
    dest = DATA_DIR / name

    # Skip work that's already done. For tar files the sentinel is the source
    # of truth (tar is deleted post-extract). For loose metadata files we
    # compare size against S3 to catch partial downloads.
    if name.endswith(".tar"):
        img_set = name[:-len(".tar")]
        if _already_extracted(img_set):
            print(f"[skip] {img_set} already extracted", flush=True)
            return
    elif dest.exists():
        try:
            remote_size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
            if dest.stat().st_size == remote_size:
                print(f"[skip] {name} already present", flush=True)
                return
        except Exception:
            pass  # fall through and re-download

    print(f"[download] s3://{bucket}/{key}", flush=True)
    s3.download_file(bucket, key, str(dest), Config=cfg)

    if name.endswith(".tar"):
        img_set = name[:-len(".tar")]
        target = DATA_DIR / img_set
        target.mkdir(exist_ok=True)
        print(f"[extract] {name} -> {img_set}/", flush=True)
        with tarfile.open(dest) as tar:
            tar.extractall(target, filter="data")
        dest.unlink()
        _sentinel(img_set).touch()


def download_reduced(s3_uri: str) -> None:
    import boto3

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    bucket, prefix = _parse_s3_uri(s3_uri)
    s3 = boto3.client("s3")
    cfg = _transfer_config()

    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])

    if not keys:
        raise RuntimeError(
            f"No objects under s3://{bucket}/{prefix}/. "
            "Build it first with `python download.py --make-reduced`."
        )

    # Per-file boto3 transfer is already multi-part parallel; a small outer
    # pool lets the tar extractions overlap.
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(_download_one_s3_object, s3, bucket, k, cfg) for k in keys]
        for fut in as_completed(futures):
            fut.result()

    print(f"[done] reduced dataset ready at {DATA_DIR}", flush=True)


# ---------- full slice, uncompressed (S3) ---------------------------------------


def download_full_s3(s3_uri: str) -> None:
    """Pull the full, already-extracted dataset from S3 as loose files.

    Mirrors the on-disk layout one-to-one: each S3 key under the prefix maps to
    DATA_DIR/<key-relative-to-prefix>. No tar/zip, so there's nothing to
    decompress on arrival.
    """
    import boto3

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    bucket, prefix = _parse_s3_uri(s3_uri)
    s3 = boto3.client("s3")
    cfg = _loose_transfer_config()

    paginator = s3.get_paginator("list_objects_v2")
    objects: list[tuple[str, int]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
        for obj in page.get("Contents", []):
            objects.append((obj["Key"], obj["Size"]))

    if not objects:
        raise RuntimeError(
            f"No objects under s3://{bucket}/{prefix}/. Upload the full "
            "uncompressed dataset there first (done out-of-band)."
        )

    total = len(objects)
    print(f"[download] {total} files from s3://{bucket}/{prefix}/", flush=True)
    done = 0
    lock = threading.Lock()

    def _fetch(key: str, size: int) -> None:
        nonlocal done
        rel = key[len(prefix) + 1:]
        if not rel:  # the prefix "directory" placeholder, if any
            return
        dest = DATA_DIR / rel
        # Resume/idempotency: a same-size local file is assumed complete.
        if dest.exists() and dest.stat().st_size == size:
            pass
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(dest), Config=cfg)
        with lock:
            done += 1
            if done % 2000 == 0 or done == total:
                print(f"[download] {done}/{total}", flush=True)

    # Many small files: parallelism is latency-bound, so use a wide pool of
    # single-threaded transfers rather than a few multi-threaded ones.
    with ThreadPoolExecutor(max_workers=64) as ex:
        futures = [ex.submit(_fetch, k, sz) for k, sz in objects]
        for fut in as_completed(futures):
            fut.result()

    print(f"[done] full dataset ready at {DATA_DIR}", flush=True)


# ---------- entry point ---------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--reduced",
        action="store_true",
        help="Pull a small pre-built slice from S3 instead of the full HF dataset.",
    )
    mode.add_argument(
        "--make-reduced",
        action="store_true",
        help="Build a small slice from the local full dataset and upload it to S3.",
    )
    mode.add_argument(
        "--full-s3",
        action="store_true",
        help="Pull the full uncompressed dataset from S3 (loose files, no extract step).",
    )
    p.add_argument(
        "--s3-uri",
        default=None,
        help="S3 prefix override. Defaults to "
        f"{DEFAULT_REDUCED_S3_URI} for --reduced/--make-reduced and "
        f"{DEFAULT_FULL_S3_URI} for --full-s3.",
    )
    p.add_argument(
        "--n-trajs",
        type=int,
        default=20,
        help="Trajectories to keep per JSONL when building a reduced slice (default: 20).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.full_s3:
        download_full_s3(args.s3_uri or DEFAULT_FULL_S3_URI)
    elif args.make_reduced:
        make_reduced(args.s3_uri or DEFAULT_REDUCED_S3_URI, args.n_trajs)
    elif args.reduced:
        download_reduced(args.s3_uri or DEFAULT_REDUCED_S3_URI)
    else:
        download_agentnet()
