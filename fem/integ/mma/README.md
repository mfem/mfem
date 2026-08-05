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
host simplex:  PreferMultiRhs(nq, ndof, NE, cost) → lapack multi-RHS
               cost = kMultiRhsCostLight (mass, DomainLF)
                     or kMultiRhsCostHeavy (diffusion)
               else → blas dense
device:        TensorMmaEnabled → dmma / mfma else blas Emulate
```

One simplex probe (`PreferMultiRhs`); `cost` is relative apply cost, not an operator name.

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

Drivers: `fem/integ/bilininteg_*_mma.hpp`, `lininteg_domain_simplices_mma.hpp`.  
Kernel entry points stay `internal::Mma*Apply*` (not under `mma::`).

## Adding a specialization

Lists are **separate** (Option B):

1. Mass simplex — `bilininteg_mass_pa_simplices_mma.cpp` → `RegisterSimplexMmaKernels`
2. Diffusion simplex — `bilininteg_diffusion_pa_simplices_mma.cpp` → `RegisterSimplexMmaKernels`
3. Domain LF simplex — `lininteg_domain_simplices_mma.cpp` → `RegisterSimplexMmaKernels`
4. Tensor mass/diff — `bilininteg_*_tensors_mma.cpp` → `RegisterTensorsMmaKernels`

Add `AddSimplexMmaSpecialization<DIM,D1D,QND>()` or `AddTensorsMmaSpecialization<DIM,D1D,Q1D>()`.  
Sort by DIM, D1D, QND. Unregistered sizes use **Fallback** (runtime shell).
