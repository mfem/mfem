# MMA form layer — generic Apply machinery

**This directory is integrator-agnostic.** It provides field types, trait helpers,
smem plans, and apply engines. Physics QFns live in sibling operator headers under
`fem/integ/mma/` (`mass.hpp`, `diffusion.hpp`, `domain_lf.hpp`).

| Location | Responsibility |
|----------|----------------|
| `mma/form/` | Generic: `eval_t`/`grad_t`, `Apply`/`ApplyLF`/`ApplyTensor`, plans |
| `mma/mass.hpp` | `form::Mass` QFn + Mass/VectorMass Kernel decls |
| `mma/diffusion.hpp` | `DiffusionMetric` + Diffusion/VectorDiffusion Kernel decls |
| `mma/domain_lf.hpp` | `IdentityLoad` QFn + DomainLF Kernel decls |
| `mma/mode/` | Backends (dmma/mfma/blas/lapack/batch) |

Design: [`docs/design/mma-declarative-kernels.md`](../../../../../docs/design/mma-declarative-kernels.md)

Umbrella: `#include "fem/integ/mma/form/form.hpp"`  
Namespace: `mfem::internal::mma::form`

---

## Contract

| Role | C++ | Meaning |
|------|-----|---------|
| **Trial** | `const eval_t &` / `const grad_t<DIM> &` | Input field |
| **Test** | `eval_t &` / `grad_t<DIM> &` | Output (`y = …`) |
| **No trial** | omit const field | Linear form |
| **Coeff** | `real_t` or `const tensor<…> &` | Point-local PA |

- No `q` / `e` in the QFn.
- `const` marks trial.
- Prefer tensor algebra: `y = d * u`, `y = A * u`.
- **Apply only** — PA assemble stays in the integrator / driver `.cpp`.

---

## Built-in integrators (not under `form/`)

```cpp
// Mass — mma/mass.hpp
struct Mass {
  void operator()(const eval_t &u, eval_t &y, real_t d) const { y = d * u; }
};
// Kernel → Apply<Mass, DIM, D1D, QND>
//          ApplyTensor<Mass, DIM, D1D, Q1D>(…)
// VectorMass: same QFn with vdim

// DomainLF — mma/domain_lf.hpp
struct IdentityLoad {
  void operator()(eval_t &y, real_t d) const { y = d; }
};
// Kernel → ApplyLF<IdentityLoad, …>

// Diffusion — mma/diffusion.hpp
template <int DIM, bool SYM>
struct DiffusionMetric {
  void operator()(const grad_t<DIM> &u, grad_t<DIM> &y,
                  const tensor<real_t, DIM, DIM> &A) const { y = A * u; }
};
// Kernel → ApplyDiffusionDispatch<DIM, D1D, QND>(…)
//          ApplyTensor<DiffusionMetric<DIM,SYM>, …>(…)
// VectorDiffusion: DiffusionMetric<DIM,true> + vdim
```

---

## Custom QFn (generic path)

```cpp
namespace mfem::internal::mma::form {

struct DensitySquaredMass {
  MFEM_HOST_DEVICE void operator()(const eval_t &u, eval_t &y, real_t d) const {
    y = (d * d) * u;
  }
};
template <>
struct qfn_traits<DensitySquaredMass> : EvalEvalQFnTraits {};

} // namespace

// Simplex dense path — no mass/diffusion headers required:
form::Apply<DensitySquaredMass, 2>(NE, P, D, x, y);

// Tensor-product path (Eval×Eval QFn):
form::ApplyTensor<DensitySquaredMass, 2, D1D, Q1D>(NE, B, Bt, D, x, y, d1d, q1d);
```

Trait helpers in `fields.hpp`:

| Helper | Trial × test |
|--------|----------------|
| `EvalEvalQFnTraits` | Eval × Eval |
| `NoneEvalQFnTraits` | None × Eval |
| `GradGradQFnTraits<DIM,SYM>` | Grad × Grad |

**Invocation:** engines call `InvokeQFn(qfn, …)` so arity follows traits
(`has_trial` → trial+test+coeff vs test+coeff). Call sites do not hard-code
`operator()` shape. Grad metric pack and PA layout use `spatial_dim` /
`symmetric_pa` from traits — `ApplyTensor` Grad has no runtime `symmetric`
argument. Integrator registration may still branch on a runtime flag only to
select the QFn type (e.g. `DiffusionMetric<DIM,true>` vs `…false>`).

---

## What `form/` owns

```text
form.hpp       umbrella include
fields.hpp     eval_t / grad_t / none_t + qfn_traits helpers
plan.hpp       MakeEvalPlan / MakeGradPlan + MFEM_MMA_FORM_DUMP
simplex.hpp    Apply / ApplyLF — simplex dense
tensors.hpp    ApplyTensor + sum-fact Eval/Grad engines (QFn templates)
```

Host multi-RHS preference is `mma::lapack::PreferMultiRhs` in `mma/mode/lapack.hpp`
(size-only gate; no form wrapper). Multi-plane smem batch/Q-tile tables live in
`mma/mode/batch.hpp` (not a QFn).

---

## Debug dump

```bash
MFEM_MMA_FORM_DUMP=1 ./app
```

---

## Unit tests

| Tag | Content |
|-----|---------|
| `[MMA][Form]` | Generic machinery + preset smoke via integ headers |
| `[MMA][Form][Author]` | Custom QFn only (no preset) |
| `[MMA][Form][Dump]` | Dump helpers |
