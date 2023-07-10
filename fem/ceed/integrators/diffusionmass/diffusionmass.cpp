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

#include "diffusionmass.hpp"

#include "../../../../config/config.hpp"
#ifdef MFEM_USE_CEED
#include "diffusionmass_qf.h"
#endif

namespace mfem
{

namespace ceed
{

#ifdef MFEM_USE_CEED
struct DiffusionMassOperatorInfo : public OperatorInfo
{
   DiffusionMassContext ctx = {0};
   bool ctx_coeff = false;
   template <typename CoeffType>
   DiffusionMassOperatorInfo(const mfem::FiniteElementSpace &fes, CoeffType *Qd,
                             mfem::Coefficient *Qm, bool use_bdr = false)
   {
      MFEM_VERIFY(fes.GetVDim() == 1,
                  "libCEED DiffusionMassIntegrator for does not support vdim > 1!");
      ctx.dim = fes.GetMesh()->Dimension() - use_bdr;
      ctx.space_dim = fes.GetMesh()->SpaceDimension();
      apply_func = ":f_apply_diff_mass";
      apply_qf = &f_apply_diff_mass;
      // This integrator always has the coefficient stored as QFunction input.
      MFEM_VERIFY(Qd && Qm, "libCEED DiffusionMassIntegrator requires both a "
                  "diffusion and a mass integrator coefficient!");
      InitCoefficients(*Qd, *Qm);
      header = "/integrators/diffusionmass/diffusionmass_qf.h";
      trial_op = EvalMode::InterpAndGrad;
      test_op = EvalMode::InterpAndGrad;
      qdatasize = (ctx.dim * (ctx.dim + 1)) / 2 + 1;
   }
   void InitCoefficients(mfem::Coefficient &Qd, mfem::Coefficient &Qm)
   {
      build_func = ":f_build_diff_mass_quad_scalar";
      build_qf = &f_build_diff_mass_quad_scalar;
   }
   void InitCoefficients(mfem::VectorCoefficient &VQd, mfem::Coefficient &Qm)
   {
      build_func = ":f_build_diff_mass_quad_vector";
      build_qf = &f_build_diff_mass_quad_vector;
   }
   void InitCoefficients(mfem::MatrixCoefficient &MQd, mfem::Coefficient &Qm)
   {
      build_func = ":f_build_diff_mass_quad_matrix";
      build_qf = &f_build_diff_mass_quad_matrix;
   }
};
#endif

template <typename CoeffType>
PADiffusionMassIntegrator::PADiffusionMassIntegrator(
   const mfem::DiffusionMassIntegrator &integ,
   const mfem::FiniteElementSpace &fes,
   CoeffType *Qd,
   mfem::Coefficient *Qm,
   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   DiffusionMassOperatorInfo info(fes, Qd, Qm, use_bdr);
   Assemble(integ, info, fes, !info.ctx_coeff ? Qd : nullptr,
            !info.ctx_coeff ? Qm : nullptr, use_bdr);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

// @cond DOXYGEN_SKIP

template PADiffusionMassIntegrator::PADiffusionMassIntegrator(
   const mfem::DiffusionMassIntegrator &, const mfem::FiniteElementSpace &,
   mfem::Coefficient *, mfem::Coefficient *, const bool);
template PADiffusionMassIntegrator::PADiffusionMassIntegrator(
   const mfem::DiffusionMassIntegrator &, const mfem::FiniteElementSpace &,
   mfem::VectorCoefficient *, mfem::Coefficient *, const bool);
template PADiffusionMassIntegrator::PADiffusionMassIntegrator(
   const mfem::DiffusionMassIntegrator &, const mfem::FiniteElementSpace &,
   mfem::MatrixCoefficient *, mfem::Coefficient *, const bool);

// @endcond

} // namespace ceed

} // namespace mfem
