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

#include "mixedvecgrad.hpp"

#include "../../../../config/config.hpp"
#ifdef MFEM_USE_CEED
#include "../diffusion/diffusion_qf.h"
#endif

namespace mfem
{

namespace ceed
{

#ifdef MFEM_USE_CEED
struct MixedVectorGradientOperatorInfo : public OperatorInfo
{
   DiffusionContext ctx;
   template <typename CoeffType>
   MixedVectorGradientOperatorInfo(const mfem::FiniteElementSpace &trial_fes,
                                   const mfem::FiniteElementSpace &test_fes,
                                   CoeffType *Q, bool use_bdr, bool mixed_vector_grad)
   {
      MFEM_VERIFY(trial_fes.GetVDim() == 1 && test_fes.GetVDim() == 1,
                  "libCEED interface for vector FE does not support VDim > 1!");
      ctx.dim = trial_fes.GetMesh()->Dimension() - use_bdr;
      MFEM_VERIFY(ctx.dim == 2 || ctx.dim == 3,
                  "MixedVectorGradientIntegrator and MixedVectorWeakDivergenceIntegrator "
                  "require dim == 2 or dim == 3!");
      MFEM_VERIFY(
         !mixed_vector_grad ||
         (trial_fes.FEColl()->GetDerivMapType(ctx.dim) == mfem::FiniteElement::H_CURL &&
          test_fes.FEColl()->GetMapType(ctx.dim) == mfem::FiniteElement::H_CURL),
         "libCEED interface for MixedVectorGradientIntegrator requires "
         "H^1 domain and H(curl) range FE spaces!");
      MFEM_VERIFY(
         mixed_vector_grad ||
         (trial_fes.FEColl()->GetMapType(ctx.dim) == mfem::FiniteElement::H_CURL &&
          test_fes.FEColl()->GetDerivMapType(ctx.dim) == mfem::FiniteElement::H_CURL),
         "libCEED interface for MixedVectorWeakDivergenceIntegrator requires "
         "H(curl) domain and H^1 range FE spaces!");
      ctx.space_dim = trial_fes.GetMesh()->SpaceDimension();
      ctx.vdim = 1;
      if (Q == nullptr)
      {
         ctx.coeff_comp = 1;
         ctx.coeff[0] = mixed_vector_grad ? 1.0 : -1.0;
      }
      else
      {
         InitCoefficient(*Q, mixed_vector_grad ? 1.0 : -1.0);
      }

      // Reuse H(curl) quadrature functions for DiffusionIntegrator
      header = "/integrators/diffusion/diffusion_qf.h";
      build_func_const = ":f_build_diff_const";
      build_qf_const = &f_build_diff_const;
      build_func_quad = ":f_build_diff_quad";
      build_qf_quad = &f_build_diff_quad;
      apply_func = ":f_apply_diff";
      apply_qf = &f_apply_diff;
      apply_func_mf_const = ":f_apply_diff_mf_const";
      apply_qf_mf_const = &f_apply_diff_mf_const;
      apply_func_mf_quad = ":f_apply_diff_mf_quad";
      apply_qf_mf_quad = &f_apply_diff_mf_quad;
      trial_op = mixed_vector_grad ? EvalMode::Grad : EvalMode::Interp;
      test_op = mixed_vector_grad ? EvalMode::Interp : EvalMode::Grad;
      qdatasize = (ctx.dim * (ctx.dim + 1)) / 2;
   }
   void InitCoefficient(mfem::Coefficient &Q, double sign)
   {
      ctx.coeff_comp = 1;
      if (ConstantCoefficient *const_coeff =
             dynamic_cast<ConstantCoefficient *>(&Q))
      {
         ctx.coeff[0] = sign * const_coeff->constant;
      }
   }
   void InitCoefficient(mfem::VectorCoefficient &VQ, double sign)
   {
      const int vdim = VQ.GetVDim();
      ctx.coeff_comp = vdim;
      if (VectorConstantCoefficient *const_coeff =
             dynamic_cast<VectorConstantCoefficient *>(&VQ))
      {
         MFEM_VERIFY(ctx.coeff_comp <= LIBCEED_DIFF_COEFF_COMP_MAX,
                     "VectorCoefficient dimension exceeds context storage!");
         const mfem::Vector &val = const_coeff->GetVec();
         for (int i = 0; i < vdim; i++)
         {
            ctx.coeff[i] = sign * val[i];
         }
      }
   }
   void InitCoefficient(mfem::MatrixCoefficient &MQ, double sign)
   {
      // Assumes matrix coefficient is symmetric
      const int vdim = MQ.GetVDim();
      ctx.coeff_comp = (vdim * (vdim + 1)) / 2;
      if (MatrixConstantCoefficient *const_coeff =
             dynamic_cast<MatrixConstantCoefficient *>(&MQ))
      {
         MFEM_VERIFY(ctx.coeff_comp <= LIBCEED_DIFF_COEFF_COMP_MAX,
                     "MatrixCoefficient dimensions exceed context storage!");
         const mfem::DenseMatrix &val = const_coeff->GetMatrix();
         for (int j = 0; j < vdim; j++)
         {
            for (int i = j; i < vdim; i++)
            {
               const int idx = (j * vdim) - (((j - 1) * j) / 2) + i - j;
               ctx.coeff[idx] = sign * val(i, j);
            }
         }
      }
   }
};
#endif

template <typename CoeffType>
PAMixedVectorGradientIntegrator::PAMixedVectorGradientIntegrator(
   const mfem::MixedVectorGradientIntegrator &integ,
   const mfem::FiniteElementSpace &trial_fes,
   const mfem::FiniteElementSpace &test_fes,
   CoeffType *Q,
   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   MixedVectorGradientOperatorInfo info(trial_fes, test_fes, Q, use_bdr, true);
   Assemble(integ, info, trial_fes, test_fes, Q, use_bdr);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

template <typename CoeffType>
MFMixedVectorGradientIntegrator::MFMixedVectorGradientIntegrator(
   const mfem::MixedVectorGradientIntegrator &integ,
   const mfem::FiniteElementSpace &trial_fes,
   const mfem::FiniteElementSpace &test_fes,
   CoeffType *Q,
   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   MixedVectorGradientOperatorInfo info(trial_fes, test_fes, Q, use_bdr, true);
   Assemble(integ, info, trial_fes, test_fes, Q, use_bdr, true);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

template <>
PAMixedVectorWeakDivergenceIntegrator::PAMixedVectorWeakDivergenceIntegrator(
   const mfem::MixedVectorWeakDivergenceIntegrator &integ,
   const mfem::FiniteElementSpace &trial_fes,
   const mfem::FiniteElementSpace &test_fes,
   mfem::Coefficient *Q,
   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   MixedVectorGradientOperatorInfo info(trial_fes, test_fes, Q, use_bdr, false);
   if (Q)
   {
      // Does not inherit ownership of old Q
      Q = new ProductCoefficient(-1.0, *Q);
   }
   Assemble(integ, info, trial_fes, test_fes, Q, use_bdr);
   delete Q;
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

template <>
PAMixedVectorWeakDivergenceIntegrator::PAMixedVectorWeakDivergenceIntegrator(
   const mfem::MixedVectorWeakDivergenceIntegrator &integ,
   const mfem::FiniteElementSpace &trial_fes,
   const mfem::FiniteElementSpace &test_fes,
   mfem::VectorCoefficient *Q,
   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   MixedVectorGradientOperatorInfo info(trial_fes, test_fes, Q, use_bdr, false);
   if (Q)
   {
      // Does not inherit ownership of old Q
      Q = new ScalarVectorProductCoefficient(-1.0, *Q);
   }
   Assemble(integ, info, trial_fes, test_fes, Q, use_bdr);
   delete Q;
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

template <>
PAMixedVectorWeakDivergenceIntegrator::PAMixedVectorWeakDivergenceIntegrator(
   const mfem::MixedVectorWeakDivergenceIntegrator &integ,
   const mfem::FiniteElementSpace &trial_fes,
   const mfem::FiniteElementSpace &test_fes,
   mfem::MatrixCoefficient *Q,
   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   MixedVectorGradientOperatorInfo info(trial_fes, test_fes, Q, use_bdr, false);
   if (Q)
   {
      // Does not inherit ownership of old Q
      Q = new ScalarMatrixProductCoefficient(-1.0, *Q);
   }
   Assemble(integ, info, trial_fes, test_fes, Q, use_bdr);
   delete Q;
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

template <>
MFMixedVectorWeakDivergenceIntegrator::MFMixedVectorWeakDivergenceIntegrator(
   const mfem::MixedVectorWeakDivergenceIntegrator &integ,
   const mfem::FiniteElementSpace &trial_fes,
   const mfem::FiniteElementSpace &test_fes,
   mfem::Coefficient *Q,
   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   MixedVectorGradientOperatorInfo info(trial_fes, test_fes, Q, use_bdr, false);
   if (Q)
   {
      // Does not inherit ownership of old Q
      Q = new ProductCoefficient(-1.0, *Q);
   }
   Assemble(integ, info, trial_fes, test_fes, Q, use_bdr, true);
   delete Q;
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

template <>
MFMixedVectorWeakDivergenceIntegrator::MFMixedVectorWeakDivergenceIntegrator(
   const mfem::MixedVectorWeakDivergenceIntegrator &integ,
   const mfem::FiniteElementSpace &trial_fes,
   const mfem::FiniteElementSpace &test_fes,
   mfem::VectorCoefficient *Q,
   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   MixedVectorGradientOperatorInfo info(trial_fes, test_fes, Q, use_bdr, false);
   if (Q)
   {
      // Does not inherit ownership of old Q
      Q = new ScalarVectorProductCoefficient(-1.0, *Q);
   }
   Assemble(integ, info, trial_fes, test_fes, Q, use_bdr, true);
   delete Q;
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

template <>
MFMixedVectorWeakDivergenceIntegrator::MFMixedVectorWeakDivergenceIntegrator(
   const mfem::MixedVectorWeakDivergenceIntegrator &integ,
   const mfem::FiniteElementSpace &trial_fes,
   const mfem::FiniteElementSpace &test_fes,
   mfem::MatrixCoefficient *Q,
   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   MixedVectorGradientOperatorInfo info(trial_fes, test_fes, Q, use_bdr, false);
   if (Q)
   {
      // Does not inherit ownership of old Q
      Q = new ScalarMatrixProductCoefficient(-1.0, *Q);
   }
   Assemble(integ, info, trial_fes, test_fes, Q, use_bdr, true);
   delete Q;
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

template PAMixedVectorGradientIntegrator::PAMixedVectorGradientIntegrator(
   const mfem::MixedVectorGradientIntegrator &, const mfem::FiniteElementSpace &,
   const mfem::FiniteElementSpace &, mfem::Coefficient *, const bool);
template PAMixedVectorGradientIntegrator::PAMixedVectorGradientIntegrator(
   const mfem::MixedVectorGradientIntegrator &, const mfem::FiniteElementSpace &,
   const mfem::FiniteElementSpace &, mfem::VectorCoefficient *, const bool);
template PAMixedVectorGradientIntegrator::PAMixedVectorGradientIntegrator(
   const mfem::MixedVectorGradientIntegrator &, const mfem::FiniteElementSpace &,
   const mfem::FiniteElementSpace &, mfem::MatrixCoefficient *, const bool);

template MFMixedVectorGradientIntegrator::MFMixedVectorGradientIntegrator(
   const mfem::MixedVectorGradientIntegrator &, const mfem::FiniteElementSpace &,
   const mfem::FiniteElementSpace &, mfem::Coefficient *, const bool);
template MFMixedVectorGradientIntegrator::MFMixedVectorGradientIntegrator(
   const mfem::MixedVectorGradientIntegrator &, const mfem::FiniteElementSpace &,
   const mfem::FiniteElementSpace &, mfem::VectorCoefficient *, const bool);
template MFMixedVectorGradientIntegrator::MFMixedVectorGradientIntegrator(
   const mfem::MixedVectorGradientIntegrator &, const mfem::FiniteElementSpace &,
   const mfem::FiniteElementSpace &, mfem::MatrixCoefficient *, const bool);

} // namespace ceed

} // namespace mfem
