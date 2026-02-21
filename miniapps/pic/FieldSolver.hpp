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

using namespace mfem;
using namespace mfem::common;
class FieldSolver
{
private:
   real_t domain_volume;
   real_t neutralizing_const;
   ParLinearForm* precomputed_neutralizing_lf = nullptr;
   bool precompute_neutralizing_const = false;
   // Diffusion matrix with epsilon (for Poisson solve)
   HypreParMatrix* diffusion_matrix;
   // LHS (M + c*K) for DiffuseRHS, from one ParBilinearForm (Mass + Diffusion(c))
   HypreParMatrix* M_plus_cK_matrix;
   // Gradient operator for computing E = -∇phi
   ParDiscreteLinearOperator* grad_interpolator;
   FindPointsGSLIB& E_finder;
   ParLinearForm b;

protected:
   /** Compute neutralizing constant and initialize with the constant.
       Returns a reference to the precomputed neutralizing ParLinearForm. */
   const ParLinearForm& ComputeNeutralizingRHS(ParFiniteElementSpace* pfes,
                                               const ParticleVector& Q,
                                               MPI_Comm comm);

   /** Deposit charge from particles into a ParLinearForm (RHS b).
       b_i = sum_p q_p * phi_i(x_p) */
   void DepositCharge(ParFiniteElementSpace* pfes, const ParticleVector& Q,
                      ParLinearForm& b);

public:
   FieldSolver(ParFiniteElementSpace* phi_fes, ParFiniteElementSpace* E_fes,
               FindPointsGSLIB& E_finder_, real_t diffusivity,
               bool precompute_neutralizing_const_ = false);

   ~FieldSolver();

   /** Update the phi_gf grid function from the particles.
       Solve periodic Poisson: diffusion_matrix * phi = (rho - <rho>)
       with zero-mean enforcement via OrthoSolver. */
   void UpdatePhiGridFunction(ParticleSet& particles, ParGridFunction& phi_gf, ParGridFunction& rho_gf);

   /** Update E_gf grid function from phi_gf grid function.
       Compute the gradient: E = -∇phi. */
   void UpdateEGridFunction(ParGridFunction& phi_gf, ParGridFunction& E_gf);

   /** Diffuse RHS by solving (M + c*K) u = M*rhs; overwrites rhs with u.
       When c=0, M*u = M*rhs so u = rhs. */
   void DiffuseRHS(ParLinearForm& b, ParGridFunction& rho_gf);

   /// Compute (global) field energy: 0.5 * ∫ ||E||^2 dx
   real_t ComputeFieldEnergy(const ParGridFunction& E_gf) const;
};
