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

#include "vecfemass.hpp"

#include "../../../../config/config.hpp"
#ifdef MFEM_USE_CEED
#include "vecfemass_qf.h"
#endif

namespace mfem
{

namespace ceed
{

#ifdef MFEM_USE_CEED
struct VectorFEMassOperatorInfo : public OperatorInfo
{
   VectorFEMassContext ctx;
   template <typename CoeffType>
   VectorFEMassOperatorInfo(const mfem::FiniteElementSpace &fes, CoeffType *Q,
                            bool use_bdr)
   {
      MFEM_VERIFY(fes.GetVDim() == 1,
                  "libCEED interface for vector FE does not support VDim > 1!");
      ctx.dim = fes.GetMesh()->Dimension() - use_bdr;
      ctx.space_dim = fes.GetMesh()->SpaceDimension();
      ctx.is_hdiv = (fes.FEColl()->GetDerivType(ctx.dim) == mfem::FiniteElement::DIV);
      MFEM_VERIFY(ctx.is_hdiv ||
                  fes.FEColl()->GetDerivType(ctx.dim) == mfem::FiniteElement::CURL,
                  "VectorFEMassIntegrator requires H(div) or H(curl) FE space!");
      InitCoefficient(Q);

      header = "/integrators/vecfemass/vecfemass_qf.h";
      build_func_const = ":f_build_vecfemass_const";
      build_qf_const = &f_build_vecfemass_const;
      build_func_quad = ":f_build_vecfemass_quad";
      build_qf_quad = &f_build_vecfemass_quad;
      apply_func = ":f_apply_vecfemass";
      apply_qf = &f_apply_vecfemass;
      apply_func_mf_const = ":f_apply_vecfemass_mf_const";
      apply_qf_mf_const = &f_apply_vecfemass_mf_const;
      apply_func_mf_quad = ":f_apply_vecfemass_mf_quad";
      apply_qf_mf_quad = &f_apply_vecfemass_mf_quad;
      trial_op = EvalMode::Interp;
      test_op = EvalMode::Interp;
      qdatasize = (ctx.dim * (ctx.dim + 1)) / 2;
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
         MFEM_VERIFY(ctx.coeff_comp <= LIBCEED_VECFEMASS_COEFF_COMP_MAX,
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
         MFEM_VERIFY(ctx.coeff_comp <= LIBCEED_VECFEMASS_COEFF_COMP_MAX,
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
PAVectorFEMassIntegrator::PAVectorFEMassIntegrator(
   const mfem::VectorFEMassIntegrator &integ,
   const mfem::FiniteElementSpace &fes,
   CoeffType *Q,
   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   VectorFEMassOperatorInfo info(fes, Q, use_bdr);
   Assemble(integ, info, fes, Q, use_bdr);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

template <typename CoeffType>
MFVectorFEMassIntegrator::MFVectorFEMassIntegrator(
   const mfem::VectorFEMassIntegrator &integ,
   const mfem::FiniteElementSpace &fes,
   CoeffType *Q,
   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   VectorFEMassOperatorInfo info(fes, Q, use_bdr);
   Assemble(integ, info, fes, Q, use_bdr, true);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

template PAVectorFEMassIntegrator::PAVectorFEMassIntegrator(
   const mfem::VectorFEMassIntegrator &, const mfem::FiniteElementSpace &,
   mfem::Coefficient *, const bool);
template PAVectorFEMassIntegrator::PAVectorFEMassIntegrator(
   const mfem::VectorFEMassIntegrator &, const mfem::FiniteElementSpace &,
   mfem::VectorCoefficient *, const bool);
template PAVectorFEMassIntegrator::PAVectorFEMassIntegrator(
   const mfem::VectorFEMassIntegrator &, const mfem::FiniteElementSpace &,
   mfem::MatrixCoefficient *, const bool);

template MFVectorFEMassIntegrator::MFVectorFEMassIntegrator(
   const mfem::VectorFEMassIntegrator &, const mfem::FiniteElementSpace &,
   mfem::Coefficient *, const bool);
template MFVectorFEMassIntegrator::MFVectorFEMassIntegrator(
   const mfem::VectorFEMassIntegrator &, const mfem::FiniteElementSpace &,
   mfem::VectorCoefficient *, const bool);
template MFVectorFEMassIntegrator::MFVectorFEMassIntegrator(
   const mfem::VectorFEMassIntegrator &, const mfem::FiniteElementSpace &,
   mfem::MatrixCoefficient *, const bool);

} // namespace ceed

} // namespace mfem
