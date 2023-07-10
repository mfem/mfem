// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "curlcurlmass.hpp"

#include "../../../../config/config.hpp"
#ifdef MFEM_USE_CEED
#include "curlcurlmass_qf.h"
#endif

namespace mfem
{

namespace ceed
{

#ifdef MFEM_USE_CEED
struct CurlCurlMassOperatorInfo : public OperatorInfo
{
   CurlCurlMassContext ctx = {0};
   bool ctx_coeff = false;
   template <typename CoeffType1, typename CoeffType2>
   CurlCurlMassOperatorInfo(const mfem::FiniteElementSpace &fes,
                            CoeffType1 *Qd, CoeffType2 *Qm,
                            bool use_bdr = false)
   {
      MFEM_VERIFY(fes.GetVDim() == 1,
                  "libCEED interface for vector FE does not support vdim > 1!");
      ctx.dim = fes.GetMesh()->Dimension() - use_bdr;
      MFEM_VERIFY(ctx.dim == 2 || ctx.dim == 3,
                  "CurlCurlMassIntegrator requires dim == 2 or dim == 3!");
      ctx.space_dim = fes.GetMesh()->SpaceDimension();
      ctx.curl_dim = (ctx.dim < 3) ? 1 : ctx.dim;
      apply_func = ":f_apply_curlcurl_mass";
      apply_qf = &f_apply_curlcurl_mass;
      // This integrator always has the coefficient stored as QFunction input.
      MFEM_VERIFY(Qd && Qm, "libCEED CurlCurlMassIntegrator requires both a "
                  "curl-curl and a mass integrator coefficient!");
      InitCoefficients(*Qd, *Qm);
      header = "/integrators/curlcurlmass/curlcurlmass_qf.h";
      trial_op = EvalMode::InterpAndCurl;
      test_op = EvalMode::InterpAndCurl;
      qdatasize = (ctx.curl_dim * (ctx.curl_dim + 1)) / 2 +
                  (ctx.dim * (ctx.dim + 1)) / 2;
   }
   void InitCoefficients(mfem::Coefficient &Qd, mfem::Coefficient &Qm)
   {
      build_func = ":f_build_curlcurl_mass_quad_scalar_scalar";
      build_qf = &f_build_curlcurl_mass_quad_scalar_scalar;
   }
   void InitCoefficients(mfem::Coefficient &Qd, mfem::VectorCoefficient &VQm)
   {
      build_func = ":f_build_curlcurl_mass_quad_scalar_vector";
      build_qf = &f_build_curlcurl_mass_quad_scalar_vector;
   }
   void InitCoefficients(mfem::Coefficient &Qd, mfem::MatrixCoefficient &MQm)
   {
      build_func = ":f_build_curlcurl_mass_quad_scalar_matrix";
      build_qf = &f_build_curlcurl_mass_quad_scalar_matrix;
   }
   void InitCoefficients(mfem::VectorCoefficient &Qd, mfem::Coefficient &Qm)
   {
      build_func = ":f_build_curlcurl_mass_quad_vector_scalar";
      build_qf = &f_build_curlcurl_mass_quad_vector_scalar;
   }
   void InitCoefficients(mfem::VectorCoefficient &Qd, mfem::VectorCoefficient &VQm)
   {
      build_func = ":f_build_curlcurl_mass_quad_vector_vector";
      build_qf = &f_build_curlcurl_mass_quad_vector_vector;
   }
   void InitCoefficients(mfem::VectorCoefficient &Qd, mfem::MatrixCoefficient &MQm)
   {
      build_func = ":f_build_curlcurl_mass_quad_vector_matrix";
      build_qf = &f_build_curlcurl_mass_quad_vector_matrix;
   }
   void InitCoefficients(mfem::MatrixCoefficient &Qd, mfem::Coefficient &Qm)
   {
      build_func = ":f_build_curlcurl_mass_quad_matrix_scalar";
      build_qf = &f_build_curlcurl_mass_quad_matrix_scalar;
   }
   void InitCoefficients(mfem::MatrixCoefficient &Qd, mfem::VectorCoefficient &VQm)
   {
      build_func = ":f_build_curlcurl_mass_quad_matrix_vector";
      build_qf = &f_build_curlcurl_mass_quad_matrix_vector;
   }
   void InitCoefficients(mfem::MatrixCoefficient &Qd, mfem::MatrixCoefficient &MQm)
   {
      build_func = ":f_build_curlcurl_mass_quad_matrix_matrix";
      build_qf = &f_build_curlcurl_mass_quad_matrix_matrix;
   }
};
#endif

template <typename CoeffType1, typename CoeffType2>
PACurlCurlMassIntegrator::PACurlCurlMassIntegrator(
   const mfem::CurlCurlMassIntegrator &integ,
   const mfem::FiniteElementSpace &fes,
   CoeffType1 *Qd,
   CoeffType2 *Qm,
   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   CurlCurlMassOperatorInfo info(fes, Qd, Qm, use_bdr);
   Assemble(integ, info, fes, !info.ctx_coeff ? Qd : nullptr,
            !info.ctx_coeff ? Qm : nullptr, use_bdr);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

// @cond DOXYGEN_SKIP

template PACurlCurlMassIntegrator::PACurlCurlMassIntegrator(
   const mfem::CurlCurlMassIntegrator &, const mfem::FiniteElementSpace &,
   mfem::Coefficient *, mfem::Coefficient *, const bool);
template PACurlCurlMassIntegrator::PACurlCurlMassIntegrator(
   const mfem::CurlCurlMassIntegrator &, const mfem::FiniteElementSpace &,
   mfem::Coefficient *, mfem::VectorCoefficient *, const bool);
template PACurlCurlMassIntegrator::PACurlCurlMassIntegrator(
   const mfem::CurlCurlMassIntegrator &, const mfem::FiniteElementSpace &,
   mfem::Coefficient *, mfem::MatrixCoefficient *, const bool);

template PACurlCurlMassIntegrator::PACurlCurlMassIntegrator(
   const mfem::CurlCurlMassIntegrator &, const mfem::FiniteElementSpace &,
   mfem::VectorCoefficient *, mfem::Coefficient *, const bool);
template PACurlCurlMassIntegrator::PACurlCurlMassIntegrator(
   const mfem::CurlCurlMassIntegrator &, const mfem::FiniteElementSpace &,
   mfem::VectorCoefficient *, mfem::VectorCoefficient *, const bool);
template PACurlCurlMassIntegrator::PACurlCurlMassIntegrator(
   const mfem::CurlCurlMassIntegrator &, const mfem::FiniteElementSpace &,
   mfem::VectorCoefficient *, mfem::MatrixCoefficient *, const bool);

template PACurlCurlMassIntegrator::PACurlCurlMassIntegrator(
   const mfem::CurlCurlMassIntegrator &, const mfem::FiniteElementSpace &,
   mfem::MatrixCoefficient *, mfem::Coefficient *, const bool);
template PACurlCurlMassIntegrator::PACurlCurlMassIntegrator(
   const mfem::CurlCurlMassIntegrator &, const mfem::FiniteElementSpace &,
   mfem::MatrixCoefficient *, mfem::VectorCoefficient *, const bool);
template PACurlCurlMassIntegrator::PACurlCurlMassIntegrator(
   const mfem::CurlCurlMassIntegrator &, const mfem::FiniteElementSpace &,
   mfem::MatrixCoefficient *, mfem::MatrixCoefficient *, const bool);

// @endcond

} // namespace ceed

} // namespace mfem
