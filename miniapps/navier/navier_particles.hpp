
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

#ifndef MFEM_NAVIER_PARTICLES_HPP
#define MFEM_NAVIER_PARTICLES_HPP

#include "mfem.hpp"

namespace mfem
{
namespace navier
{

class NavierParticles
{
protected:
   const real_t kappa, gamma, zeta;
   ParticleSet fluid_particles;
   FindPointsGSLIB finder;

   Array<real_t> dthist{0.0, 0.0, 0.0};
   Array<real_t> beta{0.0, 0.0, 0.0, 0.0}; // BDFk coefficients
   Array<real_t> alpha{0.0, 0.0, 0.0}; // EXTk coefficients

   struct FluidParticleData
   {
      ParticleVector *u[4];
      ParticleVector *v[4];
      ParticleVector *w[4];
      ParticleVector *x[4];
   } fp_data;

   void SetTimeIntegrationCoefficients(int step);

   void ParticleStep2D(const real_t &dt, int p);

   mutable Vector up, vp, xpn, xp;
   mutable Vector r, C;

public:
   NavierParticles(MPI_Comm comm, const real_t kappa_, const real_t gamma_, const real_t zeta_, int num_particles, Mesh &m);

   void Step(const real_t &dt, int cur_step, const ParGridFunction &u_gf, const ParGridFunction &w_gf);

   void InterpolateUW(const ParGridFunction &u_gf, const ParGridFunction &w_gf, const ParticleVector &x, ParticleVector &u, ParticleVector &w);

   ParticleSet& GetParticles() { return fluid_particles; }

   ParticleVector& U(int nm=0) { return *fp_data.u[nm]; }

   ParticleVector& V(int nm=0) { return *fp_data.v[nm]; }

   ParticleVector& W(int nm=0) { return *fp_data.w[nm]; }

   ParticleVector& X(int nm=0) { return *fp_data.x[nm]; }

   void Apply2DReflectionBC(const Vector &line_start, const Vector &line_end, real_t e, bool invert_normal);

};


} // namespace navier

} // namespace mfem

#endif // MFEM_NAVIER_PARTICLES_HPP