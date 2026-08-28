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
#include <vector>

using namespace mfem;
using namespace mfem::common;

namespace
{
/// Invert F(x) = x/L + (alpha/(k*L)) sin(kx) for p(x) = [1 + alpha cos(kx)]/L.
real_t LandauInverseCDF(real_t u, real_t L, real_t k, real_t alpha)
{
   MFEM_VERIFY(u >= 0.0 && u <= 1.0, "CDF sample u must be in [0, 1].");
   const real_t target = L * u;
   real_t x = target;
   const real_t coeff = alpha / k;
   for (int iter = 0; iter < 8; ++iter)
   {
      const real_t kx = k * x;
      const real_t f = x + coeff * std::sin(kx) - target;
      const real_t df = 1.0 + alpha * std::cos(kx);
      const real_t dx = f / df;
      x -= dx;
      if (std::abs(dx) <= 1e-14 * (1.0 + std::abs(x))) { break; }
   }
   x = std::fmod(x, L);
   if (x < 0.0) { x += L; }
   return x;
}
}  // namespace

ParticleMover::ParticleMover(MPI_Comm comm, ParGridFunction* E_gf_,
                             ParGridFunction* phi_gf_,
                             ParGridFunction* rho_gf_,
                             FindPointsGSLIB& E_finder_, int num_particles,
                             Ordering::Type pdata_ordering)
   : E_gf(E_gf_), phi_gf(phi_gf_), rho_gf(rho_gf_), E_finder(E_finder_)
{
   MFEM_ASSERT(E_gf, "Must pass an E field to ParticleMover.");
   MFEM_ASSERT(phi_gf, "Must pass a phi field to ParticleMover.");
   MFEM_ASSERT(rho_gf, "Must pass a rho field to ParticleMover.");

   const int dim = E_gf->ParFESpace()->GetMesh()->SpaceDimension();

   pm_.SetSize(dim);
   pp_.SetSize(dim);

   // Create particle set: 4 scalars of mass, charge, phi, and rho,
   // plus 2 vectors of size space dim for momentum and E field.
   Array<int> field_vdims({1, 1, dim, dim, 1, 1});
   Array<const char*> field_names(
      {"mass", "charge", "momentum", "efield", "phi", "rho"});
   Array<const char*> tag_names({"tag0"});
   charged_particles = std::make_unique<ParticleSet>(
      comm, num_particles, dim, field_vdims, field_names, 1, tag_names,
      pdata_ordering);
}

void ParticleMover::InitializeChargedParticles(
   const real_t& k, const real_t& alpha, real_t m, real_t q, real_t L,
   int init_case, real_t v0, real_t beam_variance, real_t bump_fraction,
   real_t vb, real_t vth, real_t vtb, bool landau_x, bool use_its,
   bool reproduce)
{
   MFEM_VERIFY(init_case == 0 || init_case == 1 || init_case == 2,
               "init_case must be 0 (Landau), 1 (two-stream), or 2 "
               "(bump-on-tail).");
   MFEM_VERIFY(!use_its || init_case == 0,
               "use_its is only valid for init_case 0 (Landau).");
   MFEM_VERIFY(beam_variance >= 0.0, "beam_variance must be non-negative.");
   MFEM_VERIFY(bump_fraction >= 0.0 && bump_fraction <= 1.0,
               "bump_fraction must be in [0, 1].");
   MFEM_VERIFY(vth > 0.0, "vth must be positive.");
   MFEM_VERIFY(vtb > 0.0, "vtb must be positive.");

   int rank;
   MPI_Comm_rank(charged_particles->GetComm(), &rank);
   std::mt19937 gen(
      reproduce ? rank : (rank + static_cast<unsigned int>(time(nullptr))));
   std::uniform_real_distribution<> real_dist(0.0, 1.0);
   std::normal_distribution<> norm_dist(0.0, 1.0);

   const int dim = charged_particles->Coords().GetVDim();
   const real_t beam_std = std::sqrt(beam_variance);
   const int local_n = charged_particles->GetNParticles();

   int global_offset = 0;
   int global_npt = local_n;
   if (init_case == 0 && use_its)
   {
      MPI_Comm comm = charged_particles->GetComm();
      int comm_size = 1;
      MPI_Comm_size(comm, &comm_size);
      std::vector<int> counts(comm_size);
      MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
      global_npt = 0;
      for (int r = 0; r < comm_size; ++r)
      {
         if (r < rank) { global_offset += counts[r]; }
         global_npt += counts[r];
      }
      MFEM_VERIFY(global_npt > 0, "Need at least one particle for Landau ITS.");
   }

   ParticleVector& X = charged_particles->Coords();
   ParticleVector& P = charged_particles->Field(ParticleMover::MOM);
   ParticleVector& M = charged_particles->Field(ParticleMover::MASS);
   ParticleVector& Q = charged_particles->Field(ParticleMover::CHARGE);

   for (int i = 0; i < charged_particles->GetNParticles(); i++)
   {
      if (init_case == 0)
      {
         // Landau damping: Maxwellian momentum, cos(kx) density perturbation in x.
         for (int d = 0; d < dim; d++) { P(i, d) = m * norm_dist(gen); }

         if (use_its)
         {
            // Quiet inverse-transform sampling for n(x) ~ [1 + alpha cos(kx)]/L.
            const real_t u = (global_offset + i + 0.5) / global_npt;
            X(i, 0) = LandauInverseCDF(u, L, k, alpha);
            for (int d = 1; d < dim; d++) { X(i, d) = real_dist(gen) * L; }
         }
         else
         {
            for (int d = 0; d < dim; d++) { X(i, d) = real_dist(gen) * L; }

            const int d_end = landau_x ? 1 : dim;
            for (int d = 0; d < d_end; d++)
            {
               real_t x = X(i, d);
               x -= (alpha / k) * std::sin(k * x);
               x = std::fmod(x, L);
               if (x < 0) { x += L; }
               X(i, d) = x;
            }
         }
      }
      else if (init_case == 1)
      {
         // Two-stream: Gaussian beams at +/-v0 in x, Maxwellian transverse.
         const real_t vx_mean = (real_dist(gen) < 0.5 ? v0 : -v0);
         P(i, 0) = m * (vx_mean + beam_std * norm_dist(gen));

         for (int d = 1; d < dim; d++) { P(i, d) = m * norm_dist(gen); }

         for (int d = 0; d < dim; d++) { X(i, d) = real_dist(gen) * L; }
      }
      else  // init_case == 2
      {
         // Bump-on-tail: (1-bf)*N(0,vth^2) + bf*N(vb,vtb^2) in vx.
         const real_t vx =
            (real_dist(gen) < bump_fraction)
               ? (vb + vtb * norm_dist(gen))
               : (vth * norm_dist(gen));
         P(i, 0) = m * vx;

         for (int d = 1; d < dim; d++) { P(i, d) = m * vth * norm_dist(gen); }

         for (int d = 0; d < dim; d++) { X(i, d) = real_dist(gen) * L; }
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
   // Keep finder cache in sync in case other modules query different points.
   FindParticles();

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

void ParticleMover::UpdateParticleOutputFields()
{
   FindParticles();

   ParticleVector& phi = charged_particles->Field(PHI);
   ParticleVector& rho = charged_particles->Field(RHO);

   E_finder.Interpolate(*phi_gf, phi, phi.GetOrdering());
   E_finder.Interpolate(*rho_gf, rho, rho.GetOrdering());
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
