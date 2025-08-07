
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
   std::array<Array<real_t>, 3> beta_k; // BDFk coefficients, k=1,2,3
   std::array<Array<real_t>, 3> alpha_k; // EXTk coefficients, k=1,2,3

   struct FluidParticleData
   {
      NodeFunction *u[4];
      NodeFunction *v[4];
      NodeFunction *w[4];
      NodeFunction *x[4];
      Array<int> *order;
   } fp_data;

   void SetTimeIntegrationCoefficients();

   void ParticleStep2D(const real_t &dt, int p);

   mutable Vector up, vp, xpn, xp;
   mutable Vector r, C;

public:
   NavierParticles(MPI_Comm comm, const real_t kappa_, const real_t gamma_, const real_t zeta_, int num_particles, Mesh &m);

   void Step(const real_t &dt, int cur_step, const ParGridFunction &u_gf, const ParGridFunction &w_gf);

   void InterpolateUW(const ParGridFunction &u_gf, const ParGridFunction &w_gf, const NodeFunction &x, NodeFunction &u, NodeFunction &w);

   ParticleSet& GetParticles() { return fluid_particles; }

   NodeFunction& U(int nm=0) { return *fp_data.u[nm]; }

   NodeFunction& V(int nm=0) { return *fp_data.v[nm]; }

   NodeFunction& W(int nm=0) { return *fp_data.w[nm]; }

   NodeFunction& X(int nm=0) { return *fp_data.x[nm]; }

   void Apply2DReflectionBC(const Vector &line_start, const Vector &line_end, real_t e, bool invert_normal);

};


} // namespace navier

} // namespace mfem

#endif // MFEM_NAVIER_PARTICLES_HPP