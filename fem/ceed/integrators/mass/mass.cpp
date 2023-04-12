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

#include "mass.hpp"

#include "../../../../config/config.hpp"
#ifdef MFEM_USE_CEED
#include "mass_qf.h"
#endif

namespace mfem
{

namespace ceed
{

#ifdef MFEM_USE_CEED
struct MassOperatorInfo : public OperatorInfo
{
   MassContext ctx;
   MassOperatorInfo(const mfem::FiniteElementSpace &fes, mfem::Coefficient *Q,
                    bool use_bdr)
   {
      ctx.dim = fes.GetMesh()->Dimension() - use_bdr;
      ctx.space_dim = fes.GetMesh()->SpaceDimension();
      ctx.vdim = fes.GetVDim();
      if (Q == nullptr)
      {
         ctx.coeff = 1.0;
      }
      else if (ConstantCoefficient *const_coeff =
                  dynamic_cast<ConstantCoefficient *>(Q))
      {
         ctx.coeff = const_coeff->constant;
      }

      header = "/integrators/mass/mass_qf.h";
      build_func_const = ":f_build_mass_const";
      build_qf_const = &f_build_mass_const;
      build_func_quad = ":f_build_mass_quad";
      build_qf_quad = &f_build_mass_quad;
      apply_func = ":f_apply_mass";
      apply_qf = &f_apply_mass;
      apply_func_mf_const = ":f_apply_mass_mf_const";
      apply_qf_mf_const = &f_apply_mass_mf_const;
      apply_func_mf_quad = ":f_apply_mass_mf_quad";
      apply_qf_mf_quad = &f_apply_mass_mf_quad;
      trial_op = EvalMode::Interp;
      test_op = EvalMode::Interp;
      qdatasize = 1;
   }
};
#endif

PAMassIntegrator::PAMassIntegrator(const mfem::MassIntegrator &integ,
                                   const mfem::FiniteElementSpace &fes,
                                   mfem::Coefficient *Q,
                                   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   MassOperatorInfo info(fes, Q, use_bdr);
   Assemble(integ, info, fes, Q, use_bdr);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

PAMassIntegrator::PAMassIntegrator(const mfem::VectorMassIntegrator &integ,
                                   const mfem::FiniteElementSpace &fes,
                                   mfem::Coefficient *Q,
                                   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   MassOperatorInfo info(fes, Q, use_bdr);
   Assemble(integ, info, fes, Q, use_bdr);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

MFMassIntegrator::MFMassIntegrator(const mfem::MassIntegrator &integ,
                                   const mfem::FiniteElementSpace &fes,
                                   mfem::Coefficient *Q,
                                   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   MassOperatorInfo info(fes, Q, use_bdr);
   Assemble(integ, info, fes, Q, use_bdr, true);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

MFMassIntegrator::MFMassIntegrator(const mfem::VectorMassIntegrator &integ,
                                   const mfem::FiniteElementSpace &fes,
                                   mfem::Coefficient *Q,
                                   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   MassOperatorInfo info(fes, Q, use_bdr);
   Assemble(integ, info, fes, Q, use_bdr, true);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

} // namespace ceed

} // namespace mfem
