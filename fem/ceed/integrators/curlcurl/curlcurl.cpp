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

#include "curlcurl.hpp"

#include "../../../../config/config.hpp"
#ifdef MFEM_USE_CEED
#include "curlcurl_qf.h"
#endif

namespace mfem
{

namespace ceed
{

#ifdef MFEM_USE_CEED
struct CurlCurlOperatorInfo : public OperatorInfo
{
   CurlCurlContext ctx = {0};
   bool ctx_coeff = false;
   template <typename CoeffType>
   CurlCurlOperatorInfo(const mfem::FiniteElementSpace &fes, CoeffType *Q,
                        bool use_bdr = false, bool use_mf = false)
   {
      MFEM_VERIFY(fes.GetVDim() == 1,
                  "libCEED interface for vector FE does not support vdim > 1!");
      ctx.dim = fes.GetMesh()->Dimension() - use_bdr;
      MFEM_VERIFY(ctx.dim == 2 || ctx.dim == 3,
                  "CurlCurlIntegrator requires dim == 2 or dim == 3!");
      ctx.space_dim = fes.GetMesh()->SpaceDimension();
      ctx.curl_dim = (ctx.dim < 3) ? 1 : ctx.dim;
      if (!use_mf)
      {
         apply_func = ":f_apply_curlcurl";
         apply_qf = &f_apply_curlcurl;
      }
      else
      {
         build_func = "";
         build_qf = nullptr;
      }
      if (Q == nullptr)
      {
         ctx_coeff = true;
         ctx.coeff[0] = 1.0;
         if (!use_mf)
         {
            build_func = ":f_build_curlcurl_const_scalar";
            build_qf = &f_build_curlcurl_const_scalar;
         }
         else
         {
            apply_func = ":f_apply_curlcurl_mf_const_scalar";
            apply_qf = &f_apply_curlcurl_mf_const_scalar;
         }
      }
      else
      {
         InitCoefficient(*Q, use_mf);
      }
      header = "/integrators/curlcurl/curlcurl_qf.h";
      trial_op = EvalMode::Curl;
      test_op = EvalMode::Curl;
      qdatasize = (ctx.curl_dim * (ctx.curl_dim + 1)) / 2;
   }
   void InitCoefficient(mfem::Coefficient &Q, bool use_mf)
   {
      if (mfem::ConstantCoefficient *const_coeff =
             dynamic_cast<mfem::ConstantCoefficient *>(&Q))
      {
         ctx_coeff = true;
         ctx.coeff[0] = const_coeff->constant;
         if (!use_mf)
         {
            build_func = ":f_build_curlcurl_const_scalar";
            build_qf = &f_build_curlcurl_const_scalar;
         }
         else
         {
            apply_func = ":f_apply_curlcurl_mf_const_scalar";
            apply_qf = &f_apply_curlcurl_mf_const_scalar;
         }
      }
      else
      {
         if (!use_mf)
         {
            build_func = ":f_build_curlcurl_quad_scalar";
            build_qf = &f_build_curlcurl_quad_scalar;
         }
         else
         {
            apply_func = ":f_apply_curlcurl_mf_quad_scalar";
            apply_qf = &f_apply_curlcurl_mf_quad_scalar;
         }
      }
   }
   void InitCoefficient(mfem::VectorCoefficient &VQ, bool use_mf)
   {
      if (mfem::VectorConstantCoefficient *const_coeff =
             dynamic_cast<mfem::VectorConstantCoefficient *>(&VQ))
      {
         ctx_coeff = true;
         const int vdim = VQ.GetVDim();
         MFEM_VERIFY(vdim <= LIBCEED_CURLCURL_COEFF_COMP_MAX,
                     "VectorCoefficient dimension exceeds context storage!");
         const mfem::Vector &val = const_coeff->GetVec();
         for (int i = 0; i < vdim; i++)
         {
            ctx.coeff[i] = val[i];
         }
         if (!use_mf)
         {
            build_func = ":f_build_curlcurl_const_vector";
            build_qf = &f_build_curlcurl_const_vector;
         }
         else
         {
            apply_func = ":f_apply_curlcurl_mf_const_vector";
            apply_qf = &f_apply_curlcurl_mf_const_vector;
         }
      }
      else
      {
         if (!use_mf)
         {
            build_func = ":f_build_curlcurl_quad_vector";
            build_qf = &f_build_curlcurl_quad_vector;
         }
         else
         {
            apply_func = ":f_apply_curlcurl_mf_quad_vector";
            apply_qf = &f_apply_curlcurl_mf_quad_vector;
         }
      }
   }
   void InitCoefficient(mfem::MatrixCoefficient &MQ, bool use_mf)
   {
      // Assumes matrix coefficient is symmetric
      if (mfem::MatrixConstantCoefficient *const_coeff =
             dynamic_cast<mfem::MatrixConstantCoefficient *>(&MQ))
      {
         ctx_coeff = true;
         const int vdim = MQ.GetVDim();
         MFEM_VERIFY((vdim * (vdim + 1)) / 2 <= LIBCEED_CURLCURL_COEFF_COMP_MAX,
                     "MatrixCoefficient dimensions exceed context storage!");
         const mfem::DenseMatrix &val = const_coeff->GetMatrix();
         for (int j = 0; j < vdim; j++)
         {
            for (int i = j; i < vdim; i++)
            {
               const int idx = (j * vdim) - (((j - 1) * j) / 2) + i - j;
               ctx.coeff[idx] = val(i, j);
            }
         }
         if (!use_mf)
         {
            build_func = ":f_build_curlcurl_const_matrix";
            build_qf = &f_build_curlcurl_const_matrix;
         }
         else
         {
            apply_func = ":f_apply_curlcurl_mf_const_matrix";
            apply_qf = &f_apply_curlcurl_mf_const_matrix;
         }
      }
      else
      {
         if (!use_mf)
         {
            build_func = ":f_build_curlcurl_quad_matrix";
            build_qf = &f_build_curlcurl_quad_matrix;
         }
         else
         {
            apply_func = ":f_apply_curlcurl_mf_quad_matrix";
            apply_qf = &f_apply_curlcurl_mf_quad_matrix;
         }
      }
   }
};
#endif

template <typename CoeffType>
PACurlCurlIntegrator::PACurlCurlIntegrator(
   const mfem::CurlCurlIntegrator &integ,
   const mfem::FiniteElementSpace &fes,
   CoeffType *Q,
   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   CurlCurlOperatorInfo info(fes, Q, use_bdr);
   Assemble(integ, info, fes, !info.ctx_coeff ? Q : nullptr, use_bdr);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

template <typename CoeffType>
MFCurlCurlIntegrator::MFCurlCurlIntegrator(
   const mfem::CurlCurlIntegrator &integ,
   const mfem::FiniteElementSpace &fes,
   CoeffType *Q,
   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   CurlCurlOperatorInfo info(fes, Q, use_bdr, true);
   Assemble(integ, info, fes, !info.ctx_coeff ? Q : nullptr, use_bdr, true);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

// @cond DOXYGEN_SKIP

template PACurlCurlIntegrator::PACurlCurlIntegrator(
   const mfem::CurlCurlIntegrator &, const mfem::FiniteElementSpace &,
   mfem::Coefficient *, const bool);
template PACurlCurlIntegrator::PACurlCurlIntegrator(
   const mfem::CurlCurlIntegrator &, const mfem::FiniteElementSpace &,
   mfem::VectorCoefficient *, const bool);
template PACurlCurlIntegrator::PACurlCurlIntegrator(
   const mfem::CurlCurlIntegrator &, const mfem::FiniteElementSpace &,
   mfem::MatrixCoefficient *, const bool);

template MFCurlCurlIntegrator::MFCurlCurlIntegrator(
   const mfem::CurlCurlIntegrator &, const mfem::FiniteElementSpace &,
   mfem::Coefficient *, const bool);
template MFCurlCurlIntegrator::MFCurlCurlIntegrator(
   const mfem::CurlCurlIntegrator &, const mfem::FiniteElementSpace &,
   mfem::VectorCoefficient *, const bool);
template MFCurlCurlIntegrator::MFCurlCurlIntegrator(
   const mfem::CurlCurlIntegrator &, const mfem::FiniteElementSpace &,
   mfem::MatrixCoefficient *, const bool);

// @endcond

} // namespace ceed

} // namespace mfem
