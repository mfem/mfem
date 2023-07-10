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

#include "divdivmass.hpp"

#include "../../../../config/config.hpp"
#ifdef MFEM_USE_CEED
#include "divdivmass_qf.h"
#endif

namespace mfem
{

namespace ceed
{

#ifdef MFEM_USE_CEED
struct DivDivMassOperatorInfo : public OperatorInfo
{
   DivDivMassContext ctx = {0};
   bool ctx_coeff = false;
   template <typename CoeffType>
   DivDivMassOperatorInfo(const mfem::FiniteElementSpace &fes,
                          mfem::Coefficient *Qd, CoeffType *Qm,
                          bool use_bdr = false)
   {
      MFEM_VERIFY(fes.GetVDim() == 1,
                  "libCEED interface for vector FE does not support vdim > 1!");
      ctx.dim = fes.GetMesh()->Dimension() - use_bdr;
      ctx.space_dim = fes.GetMesh()->SpaceDimension();
      apply_func = ":f_apply_divdiv_mass";
      apply_qf = &f_apply_divdiv_mass;
      // This integrator always has the coefficient stored as QFunction input.
      MFEM_VERIFY(Qd && Qm, "libCEED DivDivMassIntegrator requires both a "
                  "div-div and a mass integrator coefficient!");
      InitCoefficients(*Qd, *Qm);
      header = "/integrators/divdivmass/divdivmass_qf.h";
      trial_op = EvalMode::InterpAndDiv;
      test_op = EvalMode::InterpAndDiv;
      qdatasize = 1 + (ctx.dim * (ctx.dim + 1)) / 2;
   }
   void InitCoefficients(mfem::Coefficient &Qd, mfem::Coefficient &Qm)
   {
      build_func = ":f_build_divdiv_mass_quad_scalar";
      build_qf = &f_build_divdiv_mass_quad_scalar;
   }
   void InitCoefficients(mfem::Coefficient &Qd, mfem::VectorCoefficient &VQm)
   {
      build_func = ":f_build_divdiv_mass_quad_vector";
      build_qf = &f_build_divdiv_mass_quad_vector;
   }
   void InitCoefficients(mfem::Coefficient &Qd, mfem::MatrixCoefficient &MQm)
   {
      build_func = ":f_build_divdiv_mass_quad_matrix";
      build_qf = &f_build_divdiv_mass_quad_matrix;
   }
};
#endif

template <typename CoeffType>
PADivDivMassIntegrator::PADivDivMassIntegrator(
   const mfem::DivDivMassIntegrator &integ,
   const mfem::FiniteElementSpace &fes,
   mfem::Coefficient *Qd,
   CoeffType *Qm,
   const bool use_bdr)
{
#ifdef MFEM_USE_CEED
   DivDivMassOperatorInfo info(fes, Qd, Qm, use_bdr);
   Assemble(integ, info, fes, !info.ctx_coeff ? Qd : nullptr,
            !info.ctx_coeff ? Qm : nullptr, use_bdr);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

// @cond DOXYGEN_SKIP

template PADivDivMassIntegrator::PADivDivMassIntegrator(
   const mfem::DivDivMassIntegrator &, const mfem::FiniteElementSpace &,
   mfem::Coefficient *, mfem::Coefficient *, const bool);
template PADivDivMassIntegrator::PADivDivMassIntegrator(
   const mfem::DivDivMassIntegrator &, const mfem::FiniteElementSpace &,
   mfem::Coefficient *, mfem::VectorCoefficient *, const bool);
template PADivDivMassIntegrator::PADivDivMassIntegrator(
   const mfem::DivDivMassIntegrator &, const mfem::FiniteElementSpace &,
   mfem::Coefficient *, mfem::MatrixCoefficient *, const bool);

// @endcond

} // namespace ceed

} // namespace mfem
