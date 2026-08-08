# MMA backends (`fem/integ/mma/mode/`)

Shared backends for MFEM partial-assembly MMA (CUDA DMMA / HIP MFMA / host blas+lapack).

Public package entry: [`../mma.hpp`](../mma.hpp) (`ForceMMA` / `Uses*` / simplex helpers).  
Integrator-agnostic engines: [`../form/`](../form/).  
Operator drivers (QFns + registration): [`../mass.hpp`](../mass.hpp), [`../diffusion.hpp`](../diffusion.hpp), [`../domain_lf.hpp`](../domain_lf.hpp).

## When it runs

| API | Role |
|-----|------|
| `ForceMMA` / `MMAForce` / env `MFEM_USE_MMA` | Opt-in switch (`GetForceMMA`) |
| `UsesSimplexMMA` | Fixed-order H1 / H1Pos triangle or tet |
| `UsesTensorMMA` | ForceMMA + H1 GLL quad/hex, **double**, **p ≥ 3** |

## Host vs device (apply)

```text
host tensor:   PreferTensorDense(D1D, NE) → blas sum-fact vs Emulate shell
               diffusion 2D → lapack fat GEMM if MFEM_USE_LAPACK
host simplex:  PreferMultiRhs(nq, ndof, NE) → lapack multi-RHS when large enough
               else dense / form simplex host path
device:        TensorMmaEnabled → dmma / mfma else blas Emulate
```

`PreferMultiRhs` is a pure size gate (no per-operator cost weight).

## Files in this directory

| File | Namespace / role |
|------|------------------|
| `common.hpp` | `mma::` maps, smem, launch, PreferTensorDense, TensorShell* |
| `dmma.hpp` | `mma::dmma` |
| `mfma.hpp` | `mma::mfma` |
| `blas.hpp` | `mma::blas` |
| `lapack.hpp` | `mma::lapack` PreferMultiRhs (ifdef LAPACK) |
| `dispatch.hpp` | `MMA_BACKEND_PICK`, public Gemm/Grad/Interp |
| `batch.hpp` | Multi-plane smem batch NB + Q-tile (`BatchNB*`) |

## Operator drivers (sibling headers under `mma/`)

| Header / TU | QFn / role |
|-------------|------------|
| `mass.hpp` + `mass.cpp` | `form::Mass` — MassIntegrator simplex + tensor |
| `mass.hpp` + `vecmass.cpp` | same QFn + `vdim` — VectorMassIntegrator |
| `diffusion.hpp` + `diffusion.cpp` | `DiffusionMetric` — DiffusionIntegrator |
| `diffusion.hpp` + `vecdiffusion.cpp` | SYM metric + `vdim` — VectorDiffusionIntegrator |
| `domain_lf.hpp` + `domain_lf.cpp` | `IdentityLoad` — DomainLFIntegrator simplex |

Custom forms: QFn + `qfn_traits<MyQ>` under `form/` only.

Design: `docs/design/mma-declarative-kernels.md`.  
Unit tests: `[MMA][Form]`, `[MMA][Form][Author]`, `[MMA][Form][Dump]`, `[MMA][GPU]`.

**Form dump:** `MFEM_MMA_FORM_DUMP=1` prints kinds + plan on Apply (host).

## Adding a specialization

Edit `Register*MmaKernels()` in the matching TU (`mass.cpp`, `diffusion.cpp`, `domain_lf.cpp`, `vecmass.cpp`, `vecdiffusion.cpp`).

Add `AddSimplexMmaSpecialization<DIM,D1D,QND>()` or `AddTensorsMmaSpecialization<DIM,D1D,Q1D>()`.  
Sort by DIM, D1D, QND. Unregistered sizes use **Fallback** (runtime shell).
