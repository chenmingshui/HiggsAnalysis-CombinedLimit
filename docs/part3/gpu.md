# GPU Acceleration

Combine can offload the evaluation of the negative log-likelihood (NLL) to an
NVIDIA GPU. This is useful for the CPU intensive parts of an analysis, in
particular:

* likelihood fits with large parametric models or many bins;
* toy based methods (`HybridNew`, `AsymptoticLimits`, `GoodnessOfFit`), where
  the same NLL is evaluated thousands of times;
* multi-dimensional likelihood scans (`MultiDimFit --algo grid`).

The GPU support is **opt-in** and fully backwards compatible: when no CUDA
device is present, or Combine was built without CUDA support, everything keeps
running on the CPU exactly as before.

## How it works

There are two complementary mechanisms.

1. **ROOT CUDA evaluation backend.** Combine creates the NLL with
   `RooFit::EvalBackend("cuda")`. ROOT's `RooBatchCompute` library then
   schedules the nodes of the computation graph on the GPU (and uses the CPU
   for the nodes it cannot run on the device). This works for any RooFit model
   whose PDFs are known to `RooBatchCompute`.

2. **Combine CUDA kernels.** For the operations that dominate toy based
   methods, Combine ships its own CUDA kernels (in `src/gpu_kernels.cu`):
   a binned Poisson NLL, the Gaussian constraint terms, and the Poisson
   constant `log(obs!)`. These are exposed through `GpuNllEngine` and are used
   for batch evaluations, for example when a large number of model predictions
   have already been computed and only the summation remains.

The GPU device is managed by the `GpuManager` singleton, which also handles
memory transfers and work splitting over multiple devices.

## Prerequisites

* An NVIDIA GPU with a recent driver.
* The CUDA toolkit, **and** a ROOT build that was compiled with CUDA support.
  The standard `conda-forge` ROOT does **not** include the CUDA backend; use a
  CUDA-enabled view such as the LCG `_cuda` stack:

  ```bash
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106b_cuda/x86_64-el8-gcc11-opt/setup.sh
  ```

* Combine built with `-DUSE_CUDA=TRUE` (see below).

## Building Combine with CUDA

The standalone CMake build supports CUDA directly:

```bash
git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git \
  HiggsAnalysis/CombinedLimit
cd HiggsAnalysis/CombinedLimit

mkdir build
cd build
cmake .. -DUSE_CUDA=TRUE -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build . -j$(nproc --ignore=2)
```

`CMAKE_CUDA_ARCHITECTURES` should match your device (e.g. `70` for V100, `80`
for A100, `90` for H100). If it is omitted, CMake compiles for a default set of
architectures.

The CMSSW/SCRAM build does not compile the `.cu` file automatically. The GPU
manager and the CPU fallbacks are still built, so a CMSSW build can be used on
a GPU machine only if it is linked against a CUDA-enabled ROOT and the kernels
are compiled by a CUDA-capable tool.

## Using the GPU

The simplest way to enable the GPU is the `--gpu` flag:

```bash
combine -M AsymptoticLimits datacard.txt --gpu
```

This selects the CUDA NLL backend and uses device 0. It is equivalent to
`--nllbackend cuda` plus device selection.

The device can be chosen explicitly:

```bash
combine -M MultiDimFit workspace.root --gpu --gpuDevice 1
```

For multi-GPU work (for example splitting a grid scan), a list of devices can
be given:

```bash
combine -M MultiDimFit workspace.root --gpu --gpuDevices 0,1,2,3
```

The backend can also be selected directly, which is useful for debugging:

```bash
combine -M MultiDimFit workspace.root --nllbackend cuda
```

The available NLL backends are:

| Backend   | Description                                                                 |
|-----------|-----------------------------------------------------------------------------|
| `combine` | Default. Combine's own `CachingSimNLL`, with analytic Barlow-Beeston, channel masking and fast constraints. |
| `cpu`     | ROOT's vectorized CPU backend.                                              |
| `legacy`  | ROOT's original evaluation backend.                                         |
| `codegen` | ROOT's code-generation backend (needs math functions declared).             |
| `cuda`    | ROOT's CUDA backend. Requires a CUDA-enabled ROOT build and `--gpu`.        |

### Behaviour when CUDA is unavailable

If `--gpu` or `--nllbackend cuda` is requested but no CUDA device is present, or
ROOT was built without CUDA support, Combine prints a warning and falls back to
the `cpu` backend. The analysis therefore still runs, just without GPU
acceleration.

## Tips

* **Use the binary workspace.** For toy based methods, convert the datacard to
  a workspace first (`text2workspace.py`) and use it as input. This makes the
  results reproducible and avoids rebuilding the model for every toy.
* **Profile first.** The GPU pays off when the model is large and the NLL is
  evaluated many times. For small counting experiments the host/device transfer
  overhead can dominate; in that case keep the default `combine` backend.
* **Fit the model on the GPU, generate toys on the CPU.** Toy generation is
  usually memory bound and does not benefit from the GPU as much as the
  likelihood evaluation.
* **Check that the model is supported.** Not every RooFit PDF is known to
  `RooBatchCompute`. If the fit fails or silently falls back, run with
  `--nllbackend cpu` and compare.

## Troubleshooting

| Symptom | Remedy |
|---------|--------|
| `--gpu` prints "no CUDA device is available" | Check `nvidia-smi`, the driver and that the CUDA toolkit is in `PATH`/`LD_LIBRARY_PATH`. |
| "ROOT was not built with the CUDA evaluation backend" | Use a CUDA-enabled ROOT build (e.g. the LCG `_cuda` view). |
| Results differ between `cpu` and `cuda` | Check the model for PDFs that are not numerically stable in single precision; contact the developers with a minimal reproducer. |
| Out of memory on the GPU | Reduce the batch size, or use `--gpuDevices` to spread the work over several devices. |

## Implementation details

* `interface/GpuManager.h`, `src/GpuManager.cc` — device discovery, memory
  management and work splitting. The CUDA calls are hidden behind a PImpl, so
  Combine still compiles without a CUDA toolchain.
* `interface/GpuNllEngine.h`, `src/GpuNllEngine.cc` — high level entry points
  used by the Combine algorithms.
* `src/GpuKernels.h`, `src/gpu_kernels.cu` — the CUDA kernels (Poisson NLL,
  Gaussian constraints, `log(obs!)`).
* `combineCreateNLL()` in `src/Combine.cc` — single place where the NLL is
  created; it validates the CUDA request and falls back to the CPU if needed.
* `FitterAlgoBase` — defines the `--gpu`, `--gpuDevice` and `--gpuDevices`
  options and the `cuda` entry in the `--nllbackend` whitelist.
