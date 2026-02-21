// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#pragma once

#include "../common/particles_extras.hpp"
#include "mfem.hpp"

/** Field solver responsible for updating the electrostatic potential and field
    from the particle charge density. Assembles and solves the periodic Poisson
    problem, computes the electric field via a discrete gradient operator, and
    provides utilities for field diagnostics (e.g. global field energy). */
class FieldSolver
{
private:
   mfem::real_t domain_volume;
   mfem::real_t neutralizing_const;
   mfem::ParLinearForm* precomputed_neutralizing_lf = nullptr;
   bool precompute_neutralizing_const = false;
   // Diffusion matrices: epsilon (for Poisson) and diffusivity c
   mfem::HypreParMatrix* diffusion_matrix_e;
   mfem::HypreParMatrix* diffusion_matrix_c;
   // Gradient operator for computing E = -∇phi
   mfem::ParDiscreteLinearOperator* grad_interpolator;
   mfem::FindPointsGSLIB& E_finder;
   mfem::ParLinearForm b;

protected:
   /** Compute neutralizing constant and initialize with the constant.
       Returns a reference to the precomputed neutralizing ParLinearForm. */
   const mfem::ParLinearForm& ComputeNeutralizingRHS(
      mfem::ParFiniteElementSpace* pfes,
      const mfem::ParticleVector& Q,
      MPI_Comm comm);

   /** Deposit charge from particles into a ParLinearForm (RHS b).
       b_i = sum_p q_p * phi_i(x_p) */
   void DepositCharge(mfem::ParFiniteElementSpace* pfes,
                      const mfem::ParticleVector& Q,
                      mfem::ParLinearForm& b);

public:
   FieldSolver(mfem::ParFiniteElementSpace* phi_fes,
               mfem::ParFiniteElementSpace* E_fes,
               mfem::FindPointsGSLIB& E_finder_,
               mfem::real_t diffusivity,
               bool precompute_neutralizing_const_ = false);

   ~FieldSolver();

   /** Update the phi_gf grid function from the particles.
       Solve periodic Poisson: diffusion_matrix * phi = (rho - <rho>)
       with zero-mean enforcement via OrthoSolver. */
   void UpdatePhiGridFunction(mfem::ParticleSet& particles,
                              mfem::ParGridFunction& phi_gf);

   /** Update E_gf grid function from phi_gf grid function.
       Compute the gradient: E = -∇phi. */
   void UpdateEGridFunction(mfem::ParGridFunction& phi_gf,
                            mfem::ParGridFunction& E_gf);

   /// Compute (global) field energy: 0.5 * ∫ ||E||^2 dx
   mfem::real_t ComputeFieldEnergy(const mfem::ParGridFunction& E_gf) const;
};
