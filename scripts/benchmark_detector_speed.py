"""Time a TIFF stack and fingerprint every returned field.

Run once with each release build or thread count. --compare checks the input,
detection settings and exact output bytes against an earlier JSON report.
Image loading, warm-up and hashing are excluded from timings; model/residual
generation is included. Run builds sequentially to avoid CPU contention.
"""

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter

import numpy as np
import tifffile

from spotsolve import localize_stack


def fingerprint(results):
    hashes = []
    for result in results:
        digest = hashlib.sha256()
        for field in fields(result):
            value = getattr(result, field.name)
            digest.update(field.name.encode())
            if isinstance(value, np.ndarray):
                digest.update(repr((value.dtype.descr, value.shape)).encode())
                digest.update(value.tobytes())
            else:
                digest.update(json.dumps(value, sort_keys=True).encode())
        hashes.append(digest.hexdigest())
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--selection", choices=["fixed", "bic"], default="bic")
    parser.add_argument("--count-penalty", type=float, default=2.0)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if min(args.frames, args.threads, args.repeats) < 1:
        parser.error("frames, threads and repeats must be positive")
    stack = tifffile.imread(args.image)
    if stack.ndim == 2:
        stack = stack[None]
    if stack.ndim != 3 or len(stack) < args.frames:
        parser.error("expected a grayscale stack with at least --frames frames")
    stack = np.ascontiguousarray(stack[:args.frames])
    settings = dict(sigma=args.sigma, offset=args.offset,
                    selection=args.selection, count_penalty=args.count_penalty,
                    images=True)
    localize_stack(stack[:1], n_threads=args.threads, **settings)
    elapsed = []
    expected = None
    for _ in range(args.repeats):
        start = perf_counter()
        result = localize_stack(stack, n_threads=args.threads, **settings)
        elapsed.append(perf_counter() - start)
        current = fingerprint(result)
        if expected is not None and current != expected:
            raise RuntimeError("output changed between repeated runs")
        expected = current
    report = dict(
        image=str(args.image.resolve()), shape=list(stack.shape),
        input_dtype=str(stack.dtype), input_sha256=hashlib.sha256(stack.tobytes()).hexdigest(),
        settings=settings, threads=args.threads,
        machine=platform.machine(), platform=platform.platform(),
        seconds=elapsed, median_seconds=median(elapsed),
        counts=[len(frame) for frame in result], fingerprints=expected,
    )
    if args.compare:
        previous = json.loads(args.compare.read_text())
        for key in ("shape", "input_dtype", "input_sha256", "settings", "fingerprints"):
            if previous[key] != report[key]:
                raise RuntimeError(f"comparison differs: {key}")
        report["comparison"] = dict(
            report=str(args.compare), exact_outputs=True,
            speedup=previous["median_seconds"] / report["median_seconds"],
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("median_seconds", "counts")}))
    if "comparison" in report:
        print(json.dumps(report["comparison"]))


if __name__ == "__main__":
    main()
