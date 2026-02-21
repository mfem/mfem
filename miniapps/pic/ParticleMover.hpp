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
      EFIELD   // vdim = dim
   };

protected:
   /// Pointers to E field GridFunctions
   mfem::ParGridFunction* E_gf;

   /// FindPointsGSLIB object for E field mesh
   mfem::FindPointsGSLIB& E_finder;

   /// ParticleSet of charged particles
   std::unique_ptr<mfem::ParticleSet> charged_particles;

   /// Temporary vectors for particle computation
   mutable mfem::Vector pm_, pp_;

public:
   ParticleMover(MPI_Comm comm, mfem::ParGridFunction* E_gf_,
                 mfem::FindPointsGSLIB& E_finder_, int num_particles,
                 mfem::Ordering::Type pdata_ordering);

   /// Initialize charged particles with given parameters
   void InitializeChargedParticles(const mfem::real_t& k,
                                   const mfem::real_t& alpha, mfem::real_t m,
                                   mfem::real_t q, mfem::real_t L,
                                   bool reproduce = false);

   /// Find Particles in mesh corresponding to E and field
   void FindParticles();

   /// Advance particles one time step using Boris algorithm
   void Step(mfem::real_t& t, mfem::real_t dt, mfem::real_t L,
             bool first_step = false);

   /// Redistribute particles across processors
   void Redistribute();

   /// Get reference to ParticleSet
   mfem::ParticleSet& GetParticles() { return *charged_particles; }

   /// Compute (global) kinetic energy from particles
   mfem::real_t ComputeKineticEnergy() const;
};
