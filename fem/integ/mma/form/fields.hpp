// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#pragma once

/** @file fields.hpp
    Field value types (eval_t / grad_t / curl_t / div_t / none_t) and QFn traits.
*/

#include "../../../../linalg/tensor.hpp"
#include <type_traits>

/// \cond DO_NOT_DOCUMENT

namespace mfem::internal::mma::form
{

using mfem::future::tensor;

enum class field_kind { Eval, Grad, Curl, Div, None };

/** Scalar value at a quadrature point.
    Storage is tensor<real_t,1> so scalar*tensor ops apply uniformly. */
struct eval_t : tensor<real_t, 1>
{
   static constexpr field_kind kind = field_kind::Eval;
   static constexpr bool needs_basis = true;
   static constexpr bool is_grad = false;
   static constexpr bool is_curl = false;
   static constexpr bool is_div = false;

   static constexpr int planes(int /*dim*/, int vdim = 1) { return vdim; }

   using base = tensor<real_t, 1>;
   using base::base;

   MFEM_HOST_DEVICE eval_t() : base{} { (*this)[0] = real_t(0); }

   MFEM_HOST_DEVICE eval_t(real_t s) : base{} { (*this)[0] = s; }

   MFEM_HOST_DEVICE eval_t(const base &t) : base(t) {}

   MFEM_HOST_DEVICE eval_t &operator=(const base &t)
   {
      (*this)[0] = t[0];
      return *this;
   }

   MFEM_HOST_DEVICE eval_t &operator=(real_t s)
   {
      (*this)[0] = s;
      return *this;
   }

   MFEM_HOST_DEVICE explicit operator real_t() const { return (*this)[0]; }
};

/** Gradient vector of length Dim at a quadrature point. */
template <int DIM>
struct grad_t : tensor<real_t, DIM>
{
   static constexpr field_kind kind = field_kind::Grad;
   static constexpr bool needs_basis = true;
   static constexpr bool is_grad = true;
   static constexpr bool is_curl = false;
   static constexpr bool is_div = false;
   static constexpr int spatial_dim = DIM;

   static constexpr int planes(int /*dim*/, int vdim = 1)
   {
      return DIM * vdim;
   }

   using base = tensor<real_t, DIM>;
   using base::base;

   MFEM_HOST_DEVICE grad_t() : base{}
   {
      for (int i = 0; i < DIM; ++i) { (*this)[i] = real_t(0); }
   }

   MFEM_HOST_DEVICE grad_t(const base &t) : base(t) {}

   MFEM_HOST_DEVICE grad_t &operator=(const base &t)
   {
      for (int i = 0; i < DIM; ++i) { (*this)[i] = t[i]; }
      return *this;
   }
};

/** Curl field at a quadrature point.
    Layout matches handwritten Piola: length 1 in 2D (scalar curl), 3 in 3D. */
template <int DIM>
struct curl_t : tensor<real_t, (DIM == 2 ? 1 : 3)>
{
   static constexpr field_kind kind = field_kind::Curl;
   static constexpr bool needs_basis = true;
   static constexpr bool is_grad = false;
   static constexpr bool is_curl = true;
   static constexpr bool is_div = false;
   static constexpr int spatial_dim = DIM;
   static constexpr int curl_dim = (DIM == 2 ? 1 : 3);

   static constexpr int planes(int /*dim*/, int vdim = 1)
   {
      return curl_dim * vdim;
   }

   using base = tensor<real_t, curl_dim>;
   using base::base;

   MFEM_HOST_DEVICE curl_t() : base{}
   {
      for (int i = 0; i < curl_dim; ++i) { (*this)[i] = real_t(0); }
   }

   MFEM_HOST_DEVICE curl_t(const base &t) : base(t) {}

   MFEM_HOST_DEVICE curl_t &operator=(const base &t)
   {
      for (int i = 0; i < curl_dim; ++i) { (*this)[i] = t[i]; }
      return *this;
   }
};

/** Divergence (scalar) at a quadrature point — same storage as eval_t. */
struct div_t : eval_t
{
   static constexpr field_kind kind = field_kind::Div;
   static constexpr bool needs_basis = true;
   static constexpr bool is_grad = false;
   static constexpr bool is_curl = false;
   static constexpr bool is_div = true;

   using eval_t::eval_t;
   using eval_t::operator=;
};

/** No trial DOF field (linear-form style). Marker type only. */
struct none_t
{
   static constexpr field_kind kind = field_kind::None;
   static constexpr bool needs_basis = false;
   static constexpr bool is_grad = false;
   static constexpr bool is_curl = false;
   static constexpr bool is_div = false;

   static constexpr int planes(int /*dim*/, int vdim = 1) { return vdim; }
};

using value_t = eval_t;

template <typename T>
struct is_eval : std::false_type {};
template <>
struct is_eval<eval_t> : std::true_type {};

template <typename T>
struct is_grad_field : std::false_type {};
template <int Dim>
struct is_grad_field<grad_t<Dim>> : std::true_type {};

template <typename T>
struct is_curl_field : std::false_type {};
template <int Dim>
struct is_curl_field<curl_t<Dim>> : std::true_type {};

template <typename T>
struct is_div_field : std::false_type {};
template <>
struct is_div_field<div_t> : std::true_type {};

template <typename T>
struct is_none : std::false_type {};
template <>
struct is_none<none_t> : std::true_type {};

template <typename T>
struct is_field_kind
   : std::integral_constant<bool, is_eval<T>::value ||
     is_grad_field<T>::value ||
     is_curl_field<T>::value ||
     is_div_field<T>::value ||
     is_none<T>::value> {};

template <typename T>
struct field_traits
{
   static constexpr field_kind kind = T::kind;
   static constexpr bool needs_basis = T::needs_basis;
   static constexpr bool is_grad = T::is_grad;
   static constexpr bool is_curl = T::is_curl;
   static constexpr bool is_div = T::is_div;
   static constexpr int planes(int dim, int vdim = 1)
   {
      return T::planes(dim, vdim);
   }
};


/** Compile-time description of a pointwise QFn for the MMA pipeline.
    Specialize `qfn_traits<MyQFn>` next to your QFn (typically in the
    integrator header under fem/integ/), inheriting a helper below. */
template <typename QFn>
struct qfn_traits;

// ---- Generic trait helpers (no form-specific physics) ----------------------

/** Eval×Eval bilinear: operator()(const eval_t&, eval_t&, real_t). */
struct EvalEvalQFnTraits
{
   using trial_kind = eval_t;
   using test_kind = eval_t;
   using coeff_type = real_t;

   static constexpr bool load_x = true;
   static constexpr bool has_trial = true;
   static constexpr bool trial_is_grad = false;
   static constexpr bool test_is_grad = false;
   static constexpr bool trial_is_curl = false;
   static constexpr bool test_is_curl = false;
   static constexpr bool trial_is_div = false;
   static constexpr bool test_is_div = false;
   static constexpr bool trial_is_vec_eval = false;

   static constexpr int u_planes(int dim, int vdim = 1)
   {
      return field_traits<trial_kind>::planes(dim, vdim);
   }
};

/** None×Eval linear form: operator()(eval_t&, real_t). */
struct NoneEvalQFnTraits
{
   using trial_kind = none_t;
   using test_kind = eval_t;
   using coeff_type = real_t;

   static constexpr bool load_x = false;
   static constexpr bool has_trial = false;
   static constexpr bool trial_is_grad = false;
   static constexpr bool test_is_grad = false;
   static constexpr bool trial_is_curl = false;
   static constexpr bool test_is_curl = false;
   static constexpr bool trial_is_div = false;
   static constexpr bool test_is_div = false;
   static constexpr bool trial_is_vec_eval = false;

   static constexpr int u_planes(int dim, int vdim = 1)
   {
      return field_traits<test_kind>::planes(dim, vdim);
   }
};

/** Grad×Grad: operator()(const grad_t<DIM>&, grad_t<DIM>&, tensor). */
template <int DIM, bool SYM = true>
struct GradGradQFnTraits
{
   using trial_kind = grad_t<DIM>;
   using test_kind = grad_t<DIM>;
   using coeff_type = tensor<real_t, DIM, DIM>;

   static constexpr bool load_x = true;
   static constexpr bool has_trial = true;
   static constexpr bool trial_is_grad = true;
   static constexpr bool test_is_grad = true;
   static constexpr bool trial_is_curl = false;
   static constexpr bool test_is_curl = false;
   static constexpr bool trial_is_div = false;
   static constexpr bool test_is_div = false;
   static constexpr bool trial_is_vec_eval = false;
   static constexpr bool symmetric_pa = SYM;
   static constexpr int spatial_dim = DIM;

   static constexpr int u_planes(int /*dim*/, int vdim = 1)
   {
      return field_traits<trial_kind>::planes(DIM, vdim);
   }
};

/** Curl×Curl: operator()(const curl_t<DIM>&, curl_t<DIM>&, coeff). */
template <int DIM, bool SYM = true>
struct CurlCurlQFnTraits
{
   using trial_kind = curl_t<DIM>;
   using test_kind = curl_t<DIM>;
   using coeff_type = typename std::conditional<
                      (DIM == 2), real_t, tensor<real_t, 3, 3>>::type;

   static constexpr bool load_x = true;
   static constexpr bool has_trial = true;
   static constexpr bool trial_is_grad = false;
   static constexpr bool test_is_grad = false;
   static constexpr bool trial_is_curl = true;
   static constexpr bool test_is_curl = true;
   static constexpr bool trial_is_div = false;
   static constexpr bool test_is_div = false;
   static constexpr bool trial_is_vec_eval = false;
   static constexpr bool symmetric_pa = SYM;
   static constexpr int spatial_dim = DIM;

   static constexpr int u_planes(int /*dim*/, int vdim = 1)
   {
      return field_traits<trial_kind>::planes(DIM, vdim);
   }
};

/** Div×Div: operator()(const div_t&, div_t&, real_t). */
struct DivDivQFnTraits
{
   using trial_kind = div_t;
   using test_kind = div_t;
   using coeff_type = real_t;

   static constexpr bool load_x = true;
   static constexpr bool has_trial = true;
   static constexpr bool trial_is_grad = false;
   static constexpr bool test_is_grad = false;
   static constexpr bool trial_is_curl = false;
   static constexpr bool test_is_curl = false;
   static constexpr bool trial_is_div = true;
   static constexpr bool test_is_div = true;
   static constexpr bool trial_is_vec_eval = false;

   static constexpr int u_planes(int dim, int vdim = 1)
   {
      return field_traits<trial_kind>::planes(dim, vdim);
   }
};

/** Vector Eval×Eval for H(curl)/H(div) mass at Q (Piola-mapped vector).
    OPEN_ON_COMP: true = H(curl) (open along component axis); false = H(div).
    SYM_PACK_O21: true = Diffusion/Hcurl packed [11,21,22]; false = Hdiv [11,12,22]. */
template <int DIM, bool SYM = true, bool OPEN_ON_COMP = true,
          bool SYM_PACK_O21 = OPEN_ON_COMP>
struct VecEvalEvalQFnTraits
{
   using trial_kind = grad_t<DIM>; // vector value storage
   using test_kind = grad_t<DIM>;
   using coeff_type = tensor<real_t, DIM, DIM>;

   static constexpr bool load_x = true;
   static constexpr bool has_trial = true;
   static constexpr bool trial_is_grad = false;
   static constexpr bool test_is_grad = false;
   static constexpr bool trial_is_curl = false;
   static constexpr bool test_is_curl = false;
   static constexpr bool trial_is_div = false;
   static constexpr bool test_is_div = false;
   static constexpr bool trial_is_vec_eval = true;
   static constexpr bool open_on_component = OPEN_ON_COMP;
   static constexpr bool sym_pack_o21 = SYM_PACK_O21;
   static constexpr bool symmetric_pa = SYM;
   static constexpr int spatial_dim = DIM;

   static constexpr int u_planes(int /*dim*/, int vdim = 1)
   {
      return field_traits<trial_kind>::planes(DIM, vdim);
   }
};

// ---- QFn invoke (arity from traits; no hard-coded operator() shape at call sites) --

/** Bilinear: qfn(const trial&, test&, coeff). */
template <typename QFn, typename Trial, typename Test, typename Coeff>
MFEM_HOST_DEVICE inline
std::enable_if_t<qfn_traits<QFn>::has_trial, void>
InvokeQFn(QFn qfn, const Trial &u, Test &y, const Coeff &c)
{
   qfn(u, y, c);
}

/** Linear form: qfn(test&, coeff) — no trial. */
template <typename QFn, typename Test, typename Coeff>
MFEM_HOST_DEVICE inline
std::enable_if_t<!qfn_traits<QFn>::has_trial, void>
InvokeQFn(QFn qfn, Test &y, const Coeff &c)
{
   qfn(y, c);
}

/** Eval field in place on a scalar (trial×test or test-only via traits). */
template <typename QFn>
MFEM_HOST_DEVICE inline void ApplyEvalQFn(real_t &u, real_t d)
{
   using Tr = qfn_traits<QFn>;
   static_assert(!Tr::trial_is_grad && !Tr::trial_is_curl && !Tr::trial_is_div &&
                  !Tr::trial_is_vec_eval,
                  "ApplyEvalQFn expects scalar Eval (or None) trial");
   if constexpr (Tr::has_trial)
   {
      eval_t trial(u), test;
      InvokeQFn(QFn{}, trial, test, d);
      u = real_t(test);
   }
   else
   {
      eval_t test;
      InvokeQFn(QFn{}, test, d);
      u = real_t(test);
   }
}


} // namespace mfem::internal::mma::form

/// \endcond
