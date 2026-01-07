# ParaBlur

**ParaBlur** is a fault-tolerant, high-throughput image anonymization and benchmarking pipeline. It compares a Dask-based distributed parallel implementation against a sequential baseline for CPU-bound computer vision workloads (heavy 61x61 Gaussian blur on high-resolution images).

**Goals:** Measure parallel speedup, identify I/O and scheduling bottlenecks, and validate results against Amdahl's Law.

---

##  Key Features

- **Master-worker orchestration:** Uses `dask.distributed` with a `LocalCluster` to emulate a scalable worker pool.
- **Fair sequential baseline:** Disables OpenCV threading (`cv2.setNumThreads(0)`) to keep the sequential baseline strictly single-threaded.
- **Fault-tolerant ingestion:** Robust downloader with retries plus a **generated-image fallback** to guarantee a complete dataset even if URLs fail.
- **Task batching optimization:** Reduces IPC and scheduler overhead by grouping images per task.
- **Reproducible CSV output:** Results are automatically appended to `benchmark_results.csv` for offline analysis.

---

##  Summary of Key Results

These results are from the final project report, conducted on an 8-Core consumer machine:

- **Dataset:** 160 images (2000×2000 px)
- **Kernel:** Gaussian blur with sigma≈10 (kernel 61×61)
- **Median Speedup:** **4.52×**
- **Serial Overhead:** **11.0%** (Attributed to Disk I/O saturation)
- **Parallel Efficiency:** **~56.5%**

*Note: Variance in runtime was observed due to thermal throttling (CPU clock dropping from 3.8GHz to 2.5GHz at 86°C).*

---

##  Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/parablur.git](https://github.com/yourusername/parablur.git)
    cd parablur
    ```

2.  **Install dependencies:**
    ```bash
    pip install opencv-python dask[complete] requests pillow numpy
    ```

---

##  Quick Usage

The primary script `ParaBlur.py` handles both sequential and parallel execution.

### Standard Benchmark Run
Matching the settings used in the project report:
```bash
python ParaBlur.py --num-images 160 --batch-size 4 --trials 3
```

---

## Important CLI Flags
--num-images: Total images to process.

--batch-size: Images per Dask task (Higher = less scheduler overhead)

--workers: Number of Dask workers 

---

## Project Layout (after running ParaBlur.py)
downloads_fresh/ — Staging area for raw downloaded images.

processed/ — Output directory containing sequential/ and dask/ subfolders

ParaBlur.py — Primary benchmark script (Sequential + Dask Parallel)

benchmark_results.csv — Recorded benchmark metrics
