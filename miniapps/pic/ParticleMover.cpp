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

#include "ParticleMover.hpp"

#include <cmath>
#include <ctime>
#include <random>

using namespace mfem;
using namespace mfem::common;

ParticleMover::ParticleMover(MPI_Comm comm, ParGridFunction* E_gf_,
                             FindPointsGSLIB& E_finder_, int num_particles,
                             Ordering::Type pdata_ordering)
   : E_gf(E_gf_), E_finder(E_finder_)
{
   MFEM_ASSERT(E_gf, "Must pass an E field to ParticleMover.");

   const int dim = E_gf->ParFESpace()->GetMesh()->SpaceDimension();

   pm_.SetSize(dim);
   pp_.SetSize(dim);

   // Create particle set: 2 scalars of mass and charge,
   // 2 vectors of size space dim for momentum and e field.
   Array<int> field_vdims({1, 1, dim, dim});
   charged_particles = std::make_unique<ParticleSet>(
      comm, num_particles, dim, field_vdims, 1, pdata_ordering);
}

void ParticleMover::InitializeChargedParticles(const real_t& k,
                                               const real_t& alpha,
                                               real_t m, real_t q,
                                               real_t L, bool reproduce)
{
   int rank;
   MPI_Comm_rank(charged_particles->GetComm(), &rank);
   std::mt19937 gen(
      reproduce ? rank : (rank + static_cast<unsigned int>(time(nullptr))));
   std::uniform_real_distribution<> real_dist(0.0, 1.0);
   std::normal_distribution<> norm_dist(0.0, 1.0);

   const int dim = charged_particles->Coords().GetVDim();

   ParticleVector& X = charged_particles->Coords();
   ParticleVector& P = charged_particles->Field(ParticleMover::MOM);
   ParticleVector& M = charged_particles->Field(ParticleMover::MASS);
   ParticleVector& Q = charged_particles->Field(ParticleMover::CHARGE);

   for (int i = 0; i < charged_particles->GetNParticles(); i++)
   {
      // Initialize momentum.
      for (int d = 0; d < dim; d++) { P(i, d) = m * norm_dist(gen); }

      // Uniform positions (no accept-reject).
      for (int d = 0; d < dim; d++) { X(i, d) = real_dist(gen) * L; }

      // Displacement along x for perturbation ~ cos(k x).
      for (int d = 0; d < dim; d++)
      {
         real_t x = X(i, d);
         x -= (alpha / k) * std::sin(k * x);

         // Periodic wrap to [0, L).
         x = std::fmod(x, L);
         if (x < 0) { x += L; }

         X(i, d) = x;
      }

      M(i) = m;
      Q(i) = q;
   }
   FindParticles();
}

void ParticleMover::FindParticles()
{
   E_finder.FindPoints(charged_particles->Coords());
}

void ParticleMover::Step(real_t& t, real_t dt, real_t L, bool first_step)
{
   // Update E field at particles.
   ParticleVector& E = charged_particles->Field(EFIELD);
   E_finder.Interpolate(*E_gf, E, E.GetOrdering());

   // Extract particle data.
   ParticleVector& X = charged_particles->Coords();
   ParticleVector& P = charged_particles->Field(MOM);
   ParticleVector& M = charged_particles->Field(MASS);
   ParticleVector& Q = charged_particles->Field(CHARGE);

   const int npt = charged_particles->GetNParticles();
   const int dim = X.GetVDim();

   for (int particle = 0; particle < npt; ++particle)
   {
      for (int d = 0; d < dim; ++d)
      {
         P(particle, d) +=
            (first_step ? dt / 2.0 : dt) * Q(particle) * E(particle, d);
      }
   }

   // Periodic boundary: wrap coordinates to [0, L).
   for (int particle = 0; particle < npt; ++particle)
   {
      for (int d = 0; d < dim; ++d)
      {
         X(particle, d) += dt / M(particle) * P(particle, d);
         while (X(particle, d) > L) { X(particle, d) -= L; }
         while (X(particle, d) < 0.0) { X(particle, d) += L; }
      }
   }

   FindParticles();

   t += dt;
}

void ParticleMover::Redistribute()
{
   charged_particles->Redistribute(E_finder.GetProc());
   FindParticles();
}

real_t ParticleMover::ComputeKineticEnergy() const
{
   const ParticleVector& P = charged_particles->Field(MOM);
   const ParticleVector& M = charged_particles->Field(MASS);

   real_t kinetic_energy = 0.0;
   for (int p = 0; p < charged_particles->GetNParticles(); ++p)
   {
      real_t p_square_p = 0.0;
      for (int d = 0; d < P.GetVDim(); ++d) { p_square_p += P(p, d) * P(p, d); }
      kinetic_energy += 0.5 * p_square_p / M(p);
   }

   real_t global_kinetic_energy = 0.0;
   MPI_Allreduce(&kinetic_energy, &global_kinetic_energy, 1, MPI_DOUBLE,
                 MPI_SUM, charged_particles->GetComm());
   return global_kinetic_energy;
}
