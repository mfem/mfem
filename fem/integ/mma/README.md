# MMA PA package (`fem/integ/mma/`)

Shared backends for MFEM partial-assembly MMA (CUDA DMMA / HIP MFMA / host blas+lapack).

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

## Files

| File | Namespace / role |
|------|------------------|
| `common.hpp` | `mma::` maps, smem, launch, PreferTensorDense, TensorShell* |
| `dmma.hpp` | `mma::dmma` |
| `mfma.hpp` | `mma::mfma` |
| `blas.hpp` | `mma::blas` |
| `lapack.hpp` | `mma::lapack` PreferMultiRhs (ifdef LAPACK) |
| `dispatch.hpp` | `MMA_BACKEND_PICK`, public Gemm/Grad/Interp |
| `mma.hpp` / `mma.cpp` | ForceMMA, Uses*, simplex helpers |
| `form/` | **Generic** Apply machinery (simplex + tensor-product) |
| `batch.hpp` | Multi-plane smem batch NB + Q-tile (`BatchNB*`) |
| `form/tensors.hpp` | `ApplyTensor` + tensor Eval/Grad engines |

**`mma/` is integrator-agnostic** (no Mass/Diffusion QFns or operator-named apply).

**Form layer is integrator-agnostic** — see **[`form/README.md`](form/README.md)**.

Physics QFns live next to drivers:

| Driver header | QFn / dispatch |
|---------------|----------------|
| `bilininteg_mass_pa_simplices_mma.hpp` | `MassScale` → `Apply<MassScale,…>` |
| `lininteg_domain_simplices_mma.hpp` | `IdentityLoad` → `ApplyLF<…>` |
| `bilininteg_diffusion_pa_simplices_mma.hpp` | `DiffusionMetric` + `ApplyDiffusionDispatch` |
| `bilininteg_mass_pa_tensors_mma.hpp` | `MassScale` → `ApplyTensor<MassScale,…>` |
| `bilininteg_diffusion_pa_tensors_mma.hpp` | `DiffusionMetric` → `ApplyTensor<…>` |

Custom forms: QFn + `qfn_traits<MyQ> : EvalEvalQFnTraits` (etc.) under `form/` only.

Design: `docs/design/mma-declarative-kernels.md`.  
Unit tests: `[MMA][Form]`, `[MMA][Form][Author]`, `[MMA][Form][Dump]`.

**Form dump:** `MFEM_MMA_FORM_DUMP=1` prints kinds + plan on Apply (host).

## Adding a specialization

Lists are **separate** (Option B):

1. Mass simplex — `bilininteg_mass_pa_simplices_mma.cpp` → `RegisterSimplexMmaKernels`
2. Diffusion simplex — `bilininteg_diffusion_pa_simplices_mma.cpp` → `RegisterSimplexMmaKernels`
3. Domain LF simplex — `lininteg_domain_simplices_mma.cpp` → `RegisterSimplexMmaKernels`
4. Tensor mass/diff — `bilininteg_*_tensors_mma.cpp` → `RegisterTensorsMmaKernels`

Add `AddSimplexMmaSpecialization<DIM,D1D,QND>()` or `AddTensorsMmaSpecialization<DIM,D1D,Q1D>()`.  
Sort by DIM, D1D, QND. Unregistered sizes use **Fallback** (runtime shell).
