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

#include "mixedveccurl.hpp"

#include "../../../../config/config.hpp"
#ifdef MFEM_USE_CEED
#include "../curlcurl/curlcurl_qf.h"
#endif

namespace mfem
{

namespace ceed
{

#ifdef MFEM_USE_CEED
struct MixedVectorCurlOperatorInfo : public OperatorInfo
{
   CurlCurlContext ctx;
   template <typename CoeffType>
   MixedVectorCurlOperatorInfo(const mfem::FiniteElementSpace &trial_fes,
                               const mfem::FiniteElementSpace &test_fes,
                               CoeffType *Q, bool use_bdr, bool weak_curl)
   {
      MFEM_VERIFY(trial_fes.GetVDim() == 1 && test_fes.GetVDim() == 1,
                  "libCEED interface for vector FE does not support VDim > 1!");
      ctx.dim = trial_fes.GetMesh()->Dimension() - use_bdr;
      MFEM_VERIFY(ctx.dim == 3,
                  "MixedVectorCurlIntegrator and MixedVectorWeakCurlIntegrator "
                  "require dim == 3!");
      MFEM_VERIFY(
         weak_curl ||
         (trial_fes.FEColl()->GetDerivMapType(ctx.dim) == mfem::FiniteElement::H_DIV &&
          test_fes.FEColl()->GetMapType(ctx.dim) == mfem::FiniteElement::H_DIV),
         "libCEED interface for MixedVectorCurlIntegrator requires "
         "H(curl) domain and H(div) range FE spaces!");
      MFEM_VERIFY(
         !weak_curl ||
         (trial_fes.FEColl()->GetMapType(ctx.dim) == mfem::FiniteElement::H_DIV &&
          test_fes.FEColl()->GetDerivMapType(ctx.dim) == mfem::FiniteElement::H_DIV),
         "libCEED interface for MixedVectorWeakCurlIntegrator requires "
         "H(div) domain and H(curl) range FE spaces!");
      ctx.space_dim = trial_fes.GetMesh()->SpaceDimension();
      ctx.curl_dim = (ctx.dim < 3) ? 1 : ctx.dim;
      if (Q == nullptr)
      {
         ctx.coeff_comp = 1;
         ctx.coeff[0] = 1.0;
      }
      else
      {
         InitCoefficient(*Q);
      }

      // Reuse H(div) quadrature functions for CurlCurlIntegrator
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
      trial_op = weak_curl ? EvalMode::Interp : EvalMode::Curl;
      test_op = weak_curl ? EvalMode::Curl : EvalMode::Interp;
      qdatasize = (ctx.curl_dim * (ctx.curl_dim + 1)) / 2;
   }
   void InitCoefficient(mfem::Coefficient &Q)
   {
      ctx.coeff_comp = 1;
      if (ConstantCoefficient *const_coeff =
             dynamic_cast<ConstantCoefficient *>(&Q))
      {
         ctx.coeff[0] = const_coeff->constant;
      }
   }
   void InitCoefficient(mfem::VectorCoefficient &VQ)
   {
      const int vdim = VQ.GetVDim();
      ctx.coeff_comp = vdim;
      if (VectorConstantCoefficient *const_coeff =
             dynamic_cast<VectorConstantCoefficient *>(&VQ))
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
   void InitCoefficient(mfem::MatrixCoefficient &MQ)
   {
      // Assumes matrix coefficient is symmetric
      const int vdim = MQ.GetVDim();
      ctx.coeff_comp = (vdim * (vdim + 1)) / 2;
      if (MatrixConstantCoefficient *const_coeff =
             dynamic_cast<MatrixConstantCoefficient *>(&MQ))
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
PAMixedVectorCurlIntegrator::PAMixedVectorCurlIntegrator(
   const mfem::MixedVectorCurlIntegrator &integ,
   const mfem::FiniteElementSpace &trial_fes,
   const mfem::FiniteElementSpace &test_fes,
   CoeffType *Q,
   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   MixedVectorCurlOperatorInfo info(trial_fes, test_fes, Q, use_bdr, false);
   Assemble(integ, info, trial_fes, test_fes, Q, use_bdr);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

template <typename CoeffType>
MFMixedVectorCurlIntegrator::MFMixedVectorCurlIntegrator(
   const mfem::MixedVectorCurlIntegrator &integ,
   const mfem::FiniteElementSpace &trial_fes,
   const mfem::FiniteElementSpace &test_fes,
   CoeffType *Q,
   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   MixedVectorCurlOperatorInfo info(trial_fes, test_fes, Q, use_bdr, false);
   Assemble(integ, info, trial_fes, test_fes, Q, use_bdr, true);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

template <typename CoeffType>
PAMixedVectorWeakCurlIntegrator::PAMixedVectorWeakCurlIntegrator(
   const mfem::MixedVectorWeakCurlIntegrator &integ,
   const mfem::FiniteElementSpace &trial_fes,
   const mfem::FiniteElementSpace &test_fes,
   CoeffType *Q,
   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   MixedVectorCurlOperatorInfo info(trial_fes, test_fes, Q, use_bdr, true);
   Assemble(integ, info, trial_fes, test_fes, Q, use_bdr);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

template <typename CoeffType>
MFMixedVectorWeakCurlIntegrator::MFMixedVectorWeakCurlIntegrator(
   const mfem::MixedVectorWeakCurlIntegrator &integ,
   const mfem::FiniteElementSpace &trial_fes,
   const mfem::FiniteElementSpace &test_fes,
   CoeffType *Q,
   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   MixedVectorCurlOperatorInfo info(trial_fes, test_fes, Q, use_bdr, true);
   Assemble(integ, info, trial_fes, test_fes, Q, use_bdr, true);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

template PAMixedVectorCurlIntegrator::PAMixedVectorCurlIntegrator(
   const mfem::MixedVectorCurlIntegrator &, const mfem::FiniteElementSpace &,
   const mfem::FiniteElementSpace &, mfem::Coefficient *, const bool);
template PAMixedVectorCurlIntegrator::PAMixedVectorCurlIntegrator(
   const mfem::MixedVectorCurlIntegrator &, const mfem::FiniteElementSpace &,
   const mfem::FiniteElementSpace &, mfem::VectorCoefficient *, const bool);
template PAMixedVectorCurlIntegrator::PAMixedVectorCurlIntegrator(
   const mfem::MixedVectorCurlIntegrator &, const mfem::FiniteElementSpace &,
   const mfem::FiniteElementSpace &, mfem::MatrixCoefficient *, const bool);

template MFMixedVectorCurlIntegrator::MFMixedVectorCurlIntegrator(
   const mfem::MixedVectorCurlIntegrator &, const mfem::FiniteElementSpace &,
   const mfem::FiniteElementSpace &, mfem::Coefficient *, const bool);
template MFMixedVectorCurlIntegrator::MFMixedVectorCurlIntegrator(
   const mfem::MixedVectorCurlIntegrator &, const mfem::FiniteElementSpace &,
   const mfem::FiniteElementSpace &, mfem::VectorCoefficient *, const bool);
template MFMixedVectorCurlIntegrator::MFMixedVectorCurlIntegrator(
   const mfem::MixedVectorCurlIntegrator &, const mfem::FiniteElementSpace &,
   const mfem::FiniteElementSpace &, mfem::MatrixCoefficient *, const bool);

template PAMixedVectorWeakCurlIntegrator::PAMixedVectorWeakCurlIntegrator(
   const mfem::MixedVectorWeakCurlIntegrator &, const mfem::FiniteElementSpace &,
   const mfem::FiniteElementSpace &, mfem::Coefficient *, const bool);
template PAMixedVectorWeakCurlIntegrator::PAMixedVectorWeakCurlIntegrator(
   const mfem::MixedVectorWeakCurlIntegrator &, const mfem::FiniteElementSpace &,
   const mfem::FiniteElementSpace &, mfem::VectorCoefficient *, const bool);
template PAMixedVectorWeakCurlIntegrator::PAMixedVectorWeakCurlIntegrator(
   const mfem::MixedVectorWeakCurlIntegrator &, const mfem::FiniteElementSpace &,
   const mfem::FiniteElementSpace &, mfem::MatrixCoefficient *, const bool);

template MFMixedVectorWeakCurlIntegrator::MFMixedVectorWeakCurlIntegrator(
   const mfem::MixedVectorWeakCurlIntegrator &, const mfem::FiniteElementSpace &,
   const mfem::FiniteElementSpace &, mfem::Coefficient *, const bool);
template MFMixedVectorWeakCurlIntegrator::MFMixedVectorWeakCurlIntegrator(
   const mfem::MixedVectorWeakCurlIntegrator &, const mfem::FiniteElementSpace &,
   const mfem::FiniteElementSpace &, mfem::VectorCoefficient *, const bool);
template MFMixedVectorWeakCurlIntegrator::MFMixedVectorWeakCurlIntegrator(
   const mfem::MixedVectorWeakCurlIntegrator &, const mfem::FiniteElementSpace &,
   const mfem::FiniteElementSpace &, mfem::MatrixCoefficient *, const bool);

} // namespace ceed

} // namespace mfem
