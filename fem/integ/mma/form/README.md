# MMA form layer — generic Apply machinery

**This directory is integrator-agnostic.** It provides field types, trait helpers,
smem plans, and apply engines. Physics QFns (mass, diffusion, linear form) live next
to their drivers under `fem/integ/`.

| Location | Responsibility |
|----------|----------------|
| `mma/form/` | Generic: `eval_t`/`grad_t`, `Apply`/`ApplyLF`/`ApplyTensor`, plans |
| `bilininteg_mass_pa_simplices_mma.hpp` | `MassScale` QFn + kernel registration |
| `bilininteg_diffusion_pa_simplices_mma.hpp` | `DiffusionMetric` + `ApplyDiffusionDispatch` |
| `lininteg_domain_simplices_mma.hpp` | `IdentityLoad` QFn + kernel registration |
| `bilininteg_*_tensors_mma.hpp` | Same QFns + `ApplyTensor<…>` |

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
- **Apply only** — PA assemble stays in the integrator `.cpp`.

---

## Built-in integrators (not under `form/`)

```cpp
// Mass — bilininteg_mass_pa_simplices_mma.hpp
struct MassScale {
  void operator()(const eval_t &u, eval_t &y, real_t d) const { y = d * u; }
};
// Kernel → Apply<MassScale, DIM, D1D, QND>

// DomainLF — lininteg_domain_simplices_mma.hpp
struct IdentityLoad {
  void operator()(eval_t &y, real_t d) const { y = d; }
};
// Kernel → ApplyLF<IdentityLoad, …>

// Diffusion — bilininteg_diffusion_pa_simplices_mma.hpp
template <int DIM, bool SYM>
struct DiffusionMetric {
  void operator()(const grad_t<DIM> &u, grad_t<DIM> &y,
                  const tensor<real_t, DIM, DIM> &A) const { y = A * u; }
};
// Kernel → ApplyDiffusionDispatch<DIM, D1D, QND>(…)

// Tensor mass — bilininteg_mass_pa_tensors_mma.hpp
// Kernel → ApplyTensor<MassScale, DIM, D1D, Q1D>(…)

// Tensor diffusion — bilininteg_diffusion_pa_tensors_mma.hpp
// Kernel → ApplyTensor<DiffusionMetric<DIM,SYM>, …>(…)
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
simplex.hpp   Apply / ApplyLF — simplex dense
tensors.hpp    ApplyTensor + sum-fact Eval/Grad engines (QFn templates)
```

Host multi-RHS preference is `mma::lapack::PreferMultiRhs` in `mma/lapack.hpp`
(size-only gate; no form wrapper). Multi-plane smem batch/Q-tile tables live in
`mma/batch.hpp` (not a QFn).

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
