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
   CurlCurlContext ctx;
   template <typename CoeffType>
   CurlCurlOperatorInfo(const mfem::FiniteElementSpace &fes, CoeffType *Q,
                        bool use_bdr)
   {
      MFEM_VERIFY(fes.GetVDim() == 1,
                  "libCEED interface for vector FE does not support VDim > 1!");
      ctx.dim = fes.GetMesh()->Dimension() - use_bdr;
      MFEM_VERIFY(ctx.dim == 2 || ctx.dim == 3,
                  "CurlCurlIntegrator requires dim == 2 or dim == 3!");
      ctx.space_dim = fes.GetMesh()->SpaceDimension();
      ctx.curl_dim = (ctx.dim < 3) ? 1 : ctx.dim;
      InitCoefficient(Q);

      header = "/integrators/curlcurl/curlcurl_qf.h";
      build_func_const = ":f_build_curlcurl_const";
      build_qf_const = &f_build_curlcurl_const;
      build_func_quad = ":f_build_curlcurl_quad";
      build_qf_quad = &f_build_curlcurl_quad;
      apply_func = ":f_apply_curlcurl";
      apply_qf = &f_apply_curlcurl;
      apply_func_mf_const = ":f_apply_curlcurl_mf_const";
      apply_qf_mf_const = &f_apply_curlcurl_mf_const;
      apply_func_mf_quad = ":f_apply_curlcurl_mf_quad";
      apply_qf_mf_quad = &f_apply_curlcurl_mf_quad;
      trial_op = EvalMode::Curl;
      test_op = EvalMode::Curl;
      qdatasize = (ctx.curl_dim * (ctx.curl_dim + 1)) / 2;
   }
   void InitCoefficient(mfem::Coefficient *Q)
   {
      ctx.coeff_comp = 1;
      if (Q == nullptr)
      {
         ctx.coeff[0] = 1.0;
      }
      else if (ConstantCoefficient *const_coeff =
                  dynamic_cast<ConstantCoefficient *>(Q))
      {
         ctx.coeff[0] = const_coeff->constant;
      }
   }
   void InitCoefficient(mfem::VectorCoefficient *VQ)
   {
      if (VQ == nullptr)
      {
         ctx.coeff_comp = 1;
         ctx.coeff[0] = 1.0;
         return;
      }
      const int vdim = VQ->GetVDim();
      ctx.coeff_comp = vdim;
      if (VectorConstantCoefficient *const_coeff =
             dynamic_cast<VectorConstantCoefficient *>(VQ))
      {
         MFEM_VERIFY(ctx.coeff_comp <= LIBCEED_CURLCURL_COEFF_COMP_MAX,
                     "VectorCoefficient dimension exceeds context storage!");
         const mfem::Vector &val = const_coeff->GetVec();
         for (int i = 0; i < vdim; i++)
         {
            ctx.coeff[i] = val[i];
         }
      }
   }
   void InitCoefficient(mfem::MatrixCoefficient *MQ)
   {
      if (MQ == nullptr)
      {
         ctx.coeff_comp = 1;
         ctx.coeff[0] = 1.0;
         return;
      }
      // Assumes matrix coefficient is symmetric
      const int vdim = MQ->GetVDim();
      ctx.coeff_comp = (vdim * (vdim + 1)) / 2;
      if (MatrixConstantCoefficient *const_coeff =
             dynamic_cast<MatrixConstantCoefficient *>(MQ))
      {
         MFEM_VERIFY(ctx.coeff_comp <= LIBCEED_CURLCURL_COEFF_COMP_MAX,
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
   Assemble(integ, info, fes, Q, use_bdr);
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
   CurlCurlOperatorInfo info(fes, Q, use_bdr);
   Assemble(integ, info, fes, Q, use_bdr, true);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

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

} // namespace ceed

} // namespace mfem
