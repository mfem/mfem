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

#include "diffusion.hpp"

#include "../../../../config/config.hpp"
#ifdef MFEM_USE_CEED
#include "diffusion_qf.h"
#endif

namespace mfem
{

namespace ceed
{

#ifdef MFEM_USE_CEED
struct DiffusionOperatorInfo : public OperatorInfo
{
   DiffusionContext ctx;
   template <typename CoeffType>
   DiffusionOperatorInfo(const mfem::FiniteElementSpace &fes, CoeffType *Q)
   {
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
      trial_op = EvalMode::Grad;
      test_op = EvalMode::Grad;
      qdatasize =
         fes.GetMesh()->Dimension() * (fes.GetMesh()->Dimension() + 1) / 2;

      ctx.dim = fes.GetMesh()->Dimension();
      ctx.space_dim = fes.GetMesh()->SpaceDimension();
      ctx.vdim = fes.GetVDim();
      InitCoefficient(Q);
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
         MFEM_VERIFY(ctx.coeff_comp <= LIBCEED_DIFF_COEFF_COMP_MAX,
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
      ctx.coeff_comp = vdim * (vdim + 1) / 2;
      if (MatrixConstantCoefficient *const_coeff =
             dynamic_cast<MatrixConstantCoefficient *>(MQ))
      {
         MFEM_VERIFY(ctx.coeff_comp <= LIBCEED_DIFF_COEFF_COMP_MAX,
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
PADiffusionIntegrator::PADiffusionIntegrator(
   const mfem::FiniteElementSpace &fes,
   const mfem::IntegrationRule &irm,
   CoeffType *Q)
   : PAIntegrator()
{
#ifdef MFEM_USE_CEED
   DiffusionOperatorInfo info(fes, Q);
   Assemble(info, fes, irm, Q);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

template <typename CoeffType>
MixedPADiffusionIntegrator::MixedPADiffusionIntegrator(
   const DiffusionIntegrator &integ,
   const mfem::FiniteElementSpace &fes,
   CoeffType *Q)
{
#ifdef MFEM_USE_CEED
   DiffusionOperatorInfo info(fes, Q);
   Assemble(integ, info, fes, Q);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

template <typename CoeffType>
MixedPADiffusionIntegrator::MixedPADiffusionIntegrator(
   const VectorDiffusionIntegrator &integ,
   const mfem::FiniteElementSpace &fes,
   CoeffType *Q)
{
#ifdef MFEM_USE_CEED
   DiffusionOperatorInfo info(fes, Q);
   Assemble(integ, info, fes, Q);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

template <typename CoeffType>
MFDiffusionIntegrator::MFDiffusionIntegrator(
   const mfem::FiniteElementSpace &fes,
   const mfem::IntegrationRule &irm,
   CoeffType *Q)
   : MFIntegrator()
{
#ifdef MFEM_USE_CEED
   DiffusionOperatorInfo info(fes, Q);
   Assemble(info, fes, irm, Q);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

template <typename CoeffType>
MixedMFDiffusionIntegrator::MixedMFDiffusionIntegrator(
   const DiffusionIntegrator &integ,
   const mfem::FiniteElementSpace &fes,
   CoeffType *Q)
{
#ifdef MFEM_USE_CEED
   DiffusionOperatorInfo info(fes, Q);
   Assemble(integ, info, fes, Q);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

template <typename CoeffType>
MixedMFDiffusionIntegrator::MixedMFDiffusionIntegrator(
   const VectorDiffusionIntegrator &integ,
   const mfem::FiniteElementSpace &fes,
   CoeffType *Q)
{
#ifdef MFEM_USE_CEED
   DiffusionOperatorInfo info(fes, Q);
   Assemble(integ, info, fes, Q);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

template PADiffusionIntegrator::PADiffusionIntegrator(
   const mfem::FiniteElementSpace &, const mfem::IntegrationRule &,
   mfem::Coefficient *);
template PADiffusionIntegrator::PADiffusionIntegrator(
   const mfem::FiniteElementSpace &, const mfem::IntegrationRule &,
   mfem::VectorCoefficient *);
template PADiffusionIntegrator::PADiffusionIntegrator(
   const mfem::FiniteElementSpace &, const mfem::IntegrationRule &,
   mfem::MatrixCoefficient *);

template MixedPADiffusionIntegrator::MixedPADiffusionIntegrator(
   const DiffusionIntegrator &, const mfem::FiniteElementSpace &,
   mfem::Coefficient *);
template MixedPADiffusionIntegrator::MixedPADiffusionIntegrator(
   const DiffusionIntegrator &, const mfem::FiniteElementSpace &,
   mfem::VectorCoefficient *);
template MixedPADiffusionIntegrator::MixedPADiffusionIntegrator(
   const DiffusionIntegrator &, const mfem::FiniteElementSpace &,
   mfem::MatrixCoefficient *);

template MixedPADiffusionIntegrator::MixedPADiffusionIntegrator(
   const VectorDiffusionIntegrator &, const mfem::FiniteElementSpace &,
   mfem::Coefficient *);
template MixedPADiffusionIntegrator::MixedPADiffusionIntegrator(
   const VectorDiffusionIntegrator &, const mfem::FiniteElementSpace &,
   mfem::VectorCoefficient *);
template MixedPADiffusionIntegrator::MixedPADiffusionIntegrator(
   const VectorDiffusionIntegrator &, const mfem::FiniteElementSpace &,
   mfem::MatrixCoefficient *);

template MFDiffusionIntegrator::MFDiffusionIntegrator(
   const mfem::FiniteElementSpace &, const mfem::IntegrationRule &,
   mfem::Coefficient *);
template MFDiffusionIntegrator::MFDiffusionIntegrator(
   const mfem::FiniteElementSpace &, const mfem::IntegrationRule &,
   mfem::VectorCoefficient *);
template MFDiffusionIntegrator::MFDiffusionIntegrator(
   const mfem::FiniteElementSpace &, const mfem::IntegrationRule &,
   mfem::MatrixCoefficient *);

template MixedMFDiffusionIntegrator::MixedMFDiffusionIntegrator(
   const DiffusionIntegrator &, const mfem::FiniteElementSpace &,
   mfem::Coefficient *);
template MixedMFDiffusionIntegrator::MixedMFDiffusionIntegrator(
   const DiffusionIntegrator &, const mfem::FiniteElementSpace &,
   mfem::VectorCoefficient *);
template MixedMFDiffusionIntegrator::MixedMFDiffusionIntegrator(
   const DiffusionIntegrator &, const mfem::FiniteElementSpace &,
   mfem::MatrixCoefficient *);

template MixedMFDiffusionIntegrator::MixedMFDiffusionIntegrator(
   const VectorDiffusionIntegrator &, const mfem::FiniteElementSpace &,
   mfem::Coefficient *);
template MixedMFDiffusionIntegrator::MixedMFDiffusionIntegrator(
   const VectorDiffusionIntegrator &, const mfem::FiniteElementSpace &,
   mfem::VectorCoefficient *);
template MixedMFDiffusionIntegrator::MixedMFDiffusionIntegrator(
   const VectorDiffusionIntegrator &, const mfem::FiniteElementSpace &,
   mfem::MatrixCoefficient *);

} // namespace ceed

} // namespace mfem
