import argparse
import functools
import multiprocessing
import os
import shutil
import time
import webbrowser
from pathlib import Path
import csv
import statistics

import requests
from PIL import Image
from dask.distributed import Client, LocalCluster, as_completed

DEFAULT_SOURCE_URL = "https://picsum.photos/2000/2000"
DEFAULT_INPUT_DIR = "downloads_fresh"
DEFAULT_OUTPUT_DIR = "processed"
DEFAULT_NUM_IMAGES = 160
DEFAULT_BLUR_RADIUS = 10
DEFAULT_TIMEOUT_S = 20
DEFAULT_RETRIES = 3
DEFAULT_BATCH_SIZE = 4
DEFAULT_TRIALS = 3
DEFAULT_WARMUP = 1
DEFAULT_REPORT_CSV = "benchmark_results.csv"
DEFAULT_PASSES = 1


def _now() -> float:
    return time.perf_counter()


def _ensure_dir(path: Path, clean: bool = False) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _generate_random_image(path: Path, width: int, height: int) -> None:
    """Fallback mechanism: to generate a real, non-hardcoded image."""
    raw = os.urandom(width * height * 3)
    img = Image.frombytes("RGB", (width, height), raw)
    img.save(path, format="JPEG", quality=92, optimize=True)


def _try_import_cv2():
    try:
        import cv2  

        return cv2
    except Exception:
        return None


def _chunked(items: list[Path], batch_size: int) -> list[list[Path]]:
    if batch_size <= 1:
        return [[p] for p in items]
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

def setup_environment(
    *,
    source_url: str,
    input_dir: Path,
    output_dir: Path,
    num_images: int,
    force_download: bool,
    timeout_s: int,
    retries: int,
    width: int,
    height: int,
) -> list[Path]:
    """Fetch images dynamically (web), with an offline fallback if downloads fail."""

    _ensure_dir(output_dir, clean=False)

    if input_dir.exists() and not force_download:
        cached = sorted(p for p in input_dir.iterdir() if p.suffix.lower() == ".jpg")
        if len(cached) >= num_images:
            print(f"✅ Using cached images in '{input_dir}'")
            return cached[:num_images]

    _ensure_dir(input_dir, clean=True)
    print(f"Downloading {num_images} images from live source...")

    session = requests.Session()
    paths: list[Path] = []
    for i in range(num_images):
        target = input_dir / f"image_{i}.jpg"
        tmp = input_dir / f"image_{i}.jpg.part"
        ok = False

        for attempt in range(1, retries + 1):
            try:
                with session.get(source_url, stream=True, timeout=timeout_s) as response:
                    response.raise_for_status()
                    with open(tmp, "wb") as f:
                        for chunk in response.iter_content(chunk_size=1024 * 256):
                            if chunk:
                                f.write(chunk)
                tmp.replace(target)
                ok = True
                break
            except Exception as e:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
                print(f" Download failed [{i+1}/{num_images}] attempt {attempt}/{retries}: {e}")

        if not ok:
            print(f"    Falling back to generated image for [{i+1}/{num_images}]")
            _generate_random_image(target, width, height)
            ok = True

        if ok:
            paths.append(target)
            print(f"   Saved {target}")

    return paths

def heavy_computation(image_path: str, output_dir: str, blur_radius: int) -> dict:
    """
    THE WORKLOAD:
    Applies a heavy Gaussian Blur.
    For 4MP images, this forces the CPU to work hard.
    """
    start_t = _now()
    try:
        os.makedirs(output_dir, exist_ok=True)
        out_name = os.path.basename(image_path)
        out_path = os.path.join(output_dir, out_name)

        t0 = _now()
        read_s = 0.0
        compute_s = 0.0
        write_s = 0.0

        cv2 = _try_import_cv2()
        if cv2 is None:
            raise RuntimeError("Missing dependency: opencv-python. Install with: python -m pip install opencv-python")

        # For Baseline Control, OpenCV internal multithreading disabled to ensure a true single-core comparison during sequential run
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("cv2.imread returned None (file unreadable)")
        t1 = _now()
        read_s = t1 - t0

        sigma = max(0.1, float(blur_radius))
        blurred = img
        # NOTE: passes are applied in the main loop via functools.partial args, here we default to 1 pass when called directly.
        passes = 1
        for _ in range(passes):
            blurred = cv2.GaussianBlur(blurred, (0, 0), sigmaX=sigma, sigmaY=sigma)
        t2 = _now()
        compute_s = t2 - t1

        ok = cv2.imwrite(out_path, blurred)
        if not ok:
            raise RuntimeError("cv2.imwrite failed")
        t3 = _now()
        write_s = t3 - t2

        return {
            "image": out_name,
            "ok": True,
            "duration_s": _now() - start_t,
            "read_s": read_s,
            "compute_s": compute_s,
            "write_s": write_s,
            "backend": "opencv",
        }
    except Exception as e:
        return {
            "image": os.path.basename(image_path),
            "ok": False,
            "duration_s": _now() - start_t,
            "read_s": 0.0,
            "compute_s": 0.0,
            "write_s": 0.0,
            "backend": "opencv",
            "error": str(e),
        }


def heavy_computation_batch(image_paths: list[str], output_dir: str, blur_radius: int, passes: int) -> dict:
    """Process a batch of images in a single Dask task to reduce scheduler overhead."""
    start_t = _now()
    fn = functools.partial(heavy_computation, output_dir=output_dir, blur_radius=blur_radius)
    results = []
    for p in image_paths:
        r = fn(p)
        results.append(r)
    ok = sum(1 for r in results if r.get("ok"))
    fail = len(results) - ok
    return {
        "batch_size": len(image_paths),
        "ok": fail == 0,
        "ok_count": ok,
        "fail_count": fail,
        "duration_s": _now() - start_t,
        "read_s": sum(float(r.get("read_s", 0.0)) for r in results),
        "compute_s": sum(float(r.get("compute_s", 0.0)) for r in results),
        "write_s": sum(float(r.get("write_s", 0.0)) for r in results),
        "backend": "opencv",
        "results": results,
    }


def _summarize(results: list[dict]) -> tuple[int, int, float]:
    ok = sum(1 for r in results if r.get("ok"))
    fail = len(results) - ok
    total_cpu = sum(float(r.get("duration_s", 0.0)) for r in results)
    return ok, fail, total_cpu



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential vs Dask parallel image processing benchmark")
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-images", type=int, default=DEFAULT_NUM_IMAGES)
    parser.add_argument("--blur-radius", type=int, default=DEFAULT_BLUR_RADIUS)
    parser.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--width", type=int, default=2000)
    parser.add_argument("--height", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=0, help="0 = auto")
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Dask mode: images per task. >1 reduces scheduling overhead.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=DEFAULT_PASSES,
        help="How many times to apply blur per image (increases CPU work fairly)",
    )
    parser.add_argument(
        "--open-dashboard",
        action="store_true",
        help="Open the Dask dashboard in your browser",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="Warmup runs (not counted)")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Measured runs")
    parser.add_argument(
        "--report-csv",
        default=DEFAULT_REPORT_CSV,
        help="Append benchmark rows to this CSV (set to empty string to disable)",
    )
    parser.add_argument(
        "--include-dask-startup",
        action="store_true",
        help="If set, parallel timing includes cluster startup. Default measures processing only.",
    )
    return parser.parse_args()


def _append_csv_row(path: str, row: dict) -> None:
    if not path:
        return
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _fmt_stats(name: str, values: list[float]) -> str:
    if not values:
        return f"{name}: n=0"
    med = _median(values)
    mn = min(values)
    mx = max(values)
    return f"{name}: median {med:.2f}s (min {mn:.2f}s, max {mx:.2f}s, n={len(values)})"

if __name__ == "__main__":
    multiprocessing.freeze_support()
    args = parse_args()

    if _try_import_cv2() is None:
        raise SystemExit("OpenCV is required for this project. Install with: python -m pip install opencv-python")

    input_dir = Path(args.input_dir)
    output_base = Path(args.output_dir)
    output_seq = output_base / "sequential"
    output_par = output_base / "dask"
    _ensure_dir(output_seq, clean=True)
    _ensure_dir(output_par, clean=True)

    image_files = setup_environment(
        source_url=args.source_url,
        input_dir=input_dir,
        output_dir=output_base,
        num_images=args.num_images,
        force_download=args.force_download,
        timeout_s=args.timeout_s,
        retries=args.retries,
        width=args.width,
        height=args.height,
    )

    if not image_files:
        raise SystemExit("No images available to process.")

    # Freeze dataset so trials are comparable
    image_files = sorted(image_files)
    
    print("-" * 60)
    
    # Benchmarks 
    total_runs = max(0, int(args.warmup)) + max(1, int(args.trials))
    measured_seq: list[float] = []
    measured_par: list[float] = []
    measured_speedup: list[float] = []
    measured_efficiency: list[float] = []
    dashboard_opened = False

    workers = args.workers if args.workers and args.workers > 0 else (os.cpu_count() or 2)

    print(f"\n Running {args.warmup} warmup + {args.trials} trials (dataset size={len(image_files)})")
    print(
        f"    Backend=opencv | Workers={workers} | Threads/worker={args.threads_per_worker} | "
        f"Batch size={args.batch_size} | Passes={args.passes}"
    )

    for run_idx in range(1, total_runs + 1):
        is_warmup = run_idx <= int(args.warmup)
        label = "WARMUP" if is_warmup else f"TRIAL {run_idx - int(args.warmup)}"
        print("-" * 60)
        print(f"🐢 [{label}] SEQUENTIAL")

        # Sequential (processing only)
        _ensure_dir(output_seq, clean=True)
        t_seq0 = _now()
        seq_results: list[dict] = []
        for img in image_files:
          
            r = heavy_computation(str(img), str(output_seq), args.blur_radius)
            if r.get("ok") and args.passes > 1:
                for _ in range(args.passes - 1):
                    r2 = heavy_computation(str(output_seq / r["image"]), str(output_seq), args.blur_radius)
                    # Aggregate timings into the original record
                    if r2.get("ok"):
                        r["duration_s"] += float(r2.get("duration_s", 0.0))
                        r["read_s"] += float(r2.get("read_s", 0.0))
                        r["compute_s"] += float(r2.get("compute_s", 0.0))
                        r["write_s"] += float(r2.get("write_s", 0.0))
                    else:
                        r["ok"] = False
                        r["error"] = r2.get("error")
                        break
            seq_results.append(r)
        t_seq = _now() - t_seq0

        # Parallel
        print(f" [{label}] DASK PARALLEL")
        if args.include_dask_startup:
            t_par0 = _now()

        cluster = LocalCluster(
            n_workers=workers,
            threads_per_worker=max(1, args.threads_per_worker),
            processes=True,
            silence_logs=True,
            host="127.0.0.1",
            dashboard_address=":0",
        )
        client = Client(cluster)
        dash = client.dashboard_link
        print(f"   --> Dashboard: {dash}")
        if args.open_dashboard and not dashboard_opened:
            try:
                webbrowser.open_new_tab(dash)
                dashboard_opened = True
            except Exception as e:
                print(f"    Could not auto-open dashboard: {e}")

        if not args.include_dask_startup:
            t_par0 = _now()

        par_results: list[dict] = []
        image_paths = [str(p) for p in image_files]
        if args.batch_size and args.batch_size > 1:
            batches = [list(map(str, batch)) for batch in _chunked(image_files, args.batch_size)]
            fn_batch = functools.partial(
                heavy_computation_batch,
                output_dir=str(output_par),
                blur_radius=args.blur_radius,
                passes=args.passes,
            )
            futures = client.map(fn_batch, batches)
            for fut in as_completed(futures):
                batch_r = fut.result()
                par_results.extend(list(batch_r.get("results", [])))
        else:
            fn = functools.partial(heavy_computation, output_dir=str(output_par), blur_radius=args.blur_radius)
            futures = client.map(fn, image_paths)
            for fut in as_completed(futures):
                par_results.append(fut.result())

            # If passes > 1, re-apply blur on the outputs in parallel, to keep sequential/parallel doing equivalent work.
            if args.passes > 1:
                for _ in range(args.passes - 1):
                    out_paths = [str(output_par / Path(r["image"])) for r in par_results if r.get("ok")]
                    futures2 = client.map(fn, out_paths)
                    par_results2 = [f.result() for f in as_completed(futures2)]
                    # Aggregate timings into the first-pass results (by filename)
                    by_name = {r["image"]: r for r in par_results if r.get("ok")}
                    for r2 in par_results2:
                        name = r2.get("image")
                        if name in by_name and r2.get("ok"):
                            by_name[name]["duration_s"] += float(r2.get("duration_s", 0.0))
                            by_name[name]["read_s"] += float(r2.get("read_s", 0.0))
                            by_name[name]["compute_s"] += float(r2.get("compute_s", 0.0))
                            by_name[name]["write_s"] += float(r2.get("write_s", 0.0))

        t_par = _now() - t_par0

        client.close()
        cluster.close()

        # Validate results
        seq_ok, seq_fail, _ = _summarize(seq_results)
        par_ok, par_fail, _ = _summarize(par_results)
        if seq_ok != len(image_files) or par_ok != len(image_files) or seq_fail or par_fail:
            print(f"  Correctness warning: seq OK/FAIL={seq_ok}/{seq_fail}, par OK/FAIL={par_ok}/{par_fail}")

        speedup = (t_seq / t_par) if t_par > 0 else 0.0
        print(f"[{label}] seq={t_seq:.2f}s | par={t_par:.2f}s | speedup={speedup:.2f}x")

        if not is_warmup:
            measured_seq.append(t_seq)
            measured_par.append(t_par)
            measured_speedup.append(speedup)
            efficiency = (speedup / float(workers)) if workers > 0 else 0.0
            measured_efficiency.append(efficiency)
            _append_csv_row(
                args.report_csv,
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "num_images": len(image_files),
                    "backend": "opencv",
                    "blur_radius": args.blur_radius,
                    "kernel_size": int(6 * args.blur_radius + 1),
                    "width": args.width,
                    "height": args.height,
                    "workers": workers,
                    "threads_per_worker": args.threads_per_worker,
                    "batch_size": args.batch_size,
                    "include_dask_startup": bool(args.include_dask_startup),
                    "seq_s": round(t_seq, 6),
                    "par_s": round(t_par, 6),
                    "speedup": round(speedup, 6),
                    "efficiency": round(efficiency, 6),
                    "seq_ok": seq_ok,
                    "par_ok": par_ok,
                },
            )

    print("\n" + "=" * 40)
    print("        BENCHMARK SUMMARY ")
    print("=" * 40)
    print(_fmt_stats("Sequential", measured_seq))
    print(_fmt_stats("Parallel", measured_par))
    if measured_speedup:
        print(f"Speedup: median {statistics.median(measured_speedup):.2f}x (n={len(measured_speedup)})")
    if measured_efficiency:
        med_eff = statistics.median(measured_efficiency)
        print(f"Efficiency: median {med_eff*100:.1f}% (Speedup/Cores)")
    print(f"CSV report: {args.report_csv if args.report_csv else 'disabled'}")
    print("=" * 40)