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

#include "convection.hpp"

#include "../../../../config/config.hpp"
#ifdef MFEM_USE_CEED
#include "convection_qf.h"
#endif

namespace mfem
{

namespace ceed
{

#ifdef MFEM_USE_CEED
struct ConvectionOperatorInfo : public OperatorInfo
{
   ConvectionContext ctx = {0};
   ConvectionOperatorInfo(const mfem::FiniteElementSpace &fes,
                          mfem::VectorCoefficient *VQ, double alpha,
                          bool use_bdr = false, bool use_mf = false)
   {
      MFEM_VERIFY(VQ && VQ->GetVDim() == fes.GetMesh()->SpaceDimension(),
                  "Incorrect coefficient dimensions in ceed::ConvectionOperatorInfo!");
      ctx.dim = fes.GetMesh()->Dimension() - use_bdr;
      ctx.space_dim = fes.GetMesh()->SpaceDimension();
      ctx.alpha = alpha;
      if (!use_mf)
      {
         apply_func = ":f_apply_conv";
         apply_qf = &f_apply_conv;
      }
      else
      {
         build_func = "";
         build_qf = nullptr;
      }
      if (mfem::VectorConstantCoefficient *const_coeff =
             dynamic_cast<mfem::VectorConstantCoefficient *>(VQ))
      {
         const int vdim = VQ->GetVDim();
         MFEM_VERIFY(vdim <= LIBCEED_CONV_COEFF_COMP_MAX,
                     "VectorCoefficient dimension exceeds context storage!");
         const mfem::Vector &val = const_coeff->GetVec();
         for (int i = 0; i < vdim; i++)
         {
            ctx.coeff[i] = val[i];
         }
         if (!use_mf)
         {
            build_func = ":f_build_conv_const";
            build_qf = &f_build_conv_const;
         }
         else
         {
            apply_func = ":f_apply_conv_mf_const";
            apply_qf = &f_apply_conv_mf_const;
         }
      }
      else
      {
         if (!use_mf)
         {
            build_func = ":f_build_conv_quad";
            build_qf = &f_build_conv_quad;
         }
         else
         {
            apply_func = ":f_apply_conv_mf_quad";
            apply_qf = &f_apply_conv_mf_quad;
         }
      }
      header = "/integrators/convection/convection_qf.h";
      trial_op = EvalMode::Grad;
      test_op = EvalMode::Interp;
      qdatasize = ctx.dim;
   }
};
#endif

PAConvectionIntegrator::PAConvectionIntegrator(
   const mfem::ConvectionIntegrator &integ,
   const mfem::FiniteElementSpace &fes,
   mfem::VectorCoefficient *VQ,
   const double alpha,
   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   ConvectionOperatorInfo info(fes, VQ, alpha, use_bdr);
   Assemble(integ, info, fes, VQ, use_bdr);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

MFConvectionIntegrator::MFConvectionIntegrator(
   const mfem::ConvectionIntegrator &integ,
   const mfem::FiniteElementSpace &fes,
   mfem::VectorCoefficient *VQ,
   const double alpha,
   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   ConvectionOperatorInfo info(fes, VQ, alpha, use_bdr, true);
   Assemble(integ, info, fes, VQ, use_bdr, true);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

} // namespace ceed

} // namespace mfem
