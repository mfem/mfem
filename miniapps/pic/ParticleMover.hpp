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

#include <memory>

#include "../common/particles_extras.hpp"
#include "mfem.hpp"

/** This class implements explicit time integration for charged particles
    in an electric field using ParticleSet. */
class ParticleMover
{
public:
   enum Fields
   {
      MASS,    // vdim = 1
      CHARGE,  // vdim = 1
      MOM,     // vdim = dim
      EFIELD,  // vdim = dim
      PHI,     // vdim = 1
      RHO      // vdim = 1
   };

protected:
   /// Pointers to E field GridFunctions
   mfem::ParGridFunction* E_gf;
   mfem::ParGridFunction* phi_gf;
   mfem::ParGridFunction* rho_gf;

   /// FindPointsGSLIB object for E field mesh
   mfem::FindPointsGSLIB& E_finder;

   /// ParticleSet of charged particles
   std::unique_ptr<mfem::ParticleSet> charged_particles;

   /// Temporary vectors for particle computation
   mutable mfem::Vector pm_, pp_;

public:
   ParticleMover(MPI_Comm comm, mfem::ParGridFunction* E_gf_,
                 mfem::ParGridFunction* phi_gf_,
                 mfem::ParGridFunction* rho_gf_,
                 mfem::FindPointsGSLIB& E_finder_, int num_particles,
                 mfem::Ordering::Type pdata_ordering);

   /// Initialize charged particles with given parameters.
   /// @a init_case: 0 = Landau, 1 = two-stream, 2 = bump-on-tail, 3 = cold-beam.
   /// @a bump_fraction is the bump weight in f0 (case 2), not Landau alpha.
   /// @a landau_x: case 0 only; perturb density along x, not all axes.
   /// @a use_its: case 0 only; sample x from n(x)=[1+alpha cos(k_exc x)]/L via ITS.
   /// @a k sets domain length L = 2*pi/k; excitation uses k_exc = mode*k.
   /// @a alpha: excitation magnitude (cases 0 and 3).
   void InitializeChargedParticles(const mfem::real_t& k, int mode,
                                   const mfem::real_t& alpha, mfem::real_t m,
                                   mfem::real_t q, mfem::real_t L,
                                   int init_case, mfem::real_t v0,
                                   mfem::real_t beam_variance,
                                   mfem::real_t bump_fraction, mfem::real_t vb,
                                   mfem::real_t vth, mfem::real_t vtb,
                                   bool landau_x = false,
                                   bool use_its = false,
                                   bool reproduce = false);

   /// Find Particles in mesh corresponding to E and field
   void FindParticles();

   /// Advance particles one time step using Boris algorithm
   void Step(mfem::real_t& t, mfem::real_t dt, mfem::real_t L,
             bool first_step = false);

   /// Sample phi and rho at the current particle positions for CSV output.
   void UpdateParticleOutputFields();

   /// Redistribute particles across processors
   void Redistribute();

   /// Get reference to ParticleSet
   mfem::ParticleSet& GetParticles() { return *charged_particles; }

   /// Compute (global) kinetic energy from particles
   mfem::real_t ComputeKineticEnergy() const;
};
