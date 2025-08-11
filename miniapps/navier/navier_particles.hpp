
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

#include <vector>
#include <type_traits>

namespace mfem
{
namespace navier
{

/** @brief Transient Navier-Stokes fluid-particles solver
 * 
 *  @details TODO: Citation, model, equation...
 * 
 */
class NavierParticles
{
protected:
   const real_t kappa, gamma, zeta;
   ParticleSet fluid_particles;
   FindPointsGSLIB finder;

   Array<real_t> dthist{0.0, 0.0, 0.0};

   // BDFk coefficients, k=1,2,3
   std::array<Array<real_t>, 3> beta_k;

   // EXTk coefficients, k=1,2,3
   std::array<Array<real_t>, 3> alpha_k;

   struct FluidParticleData
   {
      MultiVector *u[4];
      MultiVector *v[4];
      MultiVector *w[4];
      MultiVector *x[4];
      Array<int> *order;
   } fp_data;

   struct ReflectionBC_2D
   {
      const Vector line_start;
      const Vector line_end;
      const real_t e;
      const bool invert_normal;

   };
   struct RecirculationBC_2D
   {
      const Vector inlet_start;
      const Vector inlet_end;
      const bool invert_inlet_normal;
      const Vector outlet_start;
      const Vector outlet_end;
      const bool invert_outlet_normal;
   };

   using BCVariant = std::variant<ReflectionBC_2D, RecirculationBC_2D>;
   std::vector<BCVariant> bcs;

   void SetTimeIntegrationCoefficients();

   void ParticleStep2D(const real_t &dt, int p);

   static void Get2DNormal(const Vector &p1, const Vector &p2, bool inv_normal, Vector &normal);
   static bool Get2DSegmentIntersection(const Vector &s1_start, const Vector &s1_end, const Vector &s2_start, const Vector &s2_end, Vector &x_int, real_t *t1_ptr=nullptr, real_t *t2_ptr=nullptr);

   void Apply2DReflectionBC(const ReflectionBC_2D &bc);
   void Apply2DRecirculationBC(const RecirculationBC_2D &bc);
   void ApplyBCs();

   mutable Vector up, vp, xpn, xp;
   mutable Vector r, C;

public:

   
   NavierParticles(MPI_Comm comm, const real_t kappa_, const real_t gamma_, const real_t zeta_, int num_particles, Mesh &m);

   void Step(const real_t &dt, int cur_step, const ParGridFunction &u_gf, const ParGridFunction &w_gf);

   void InterpolateUW(const ParGridFunction &u_gf, const ParGridFunction &w_gf, const MultiVector &x, MultiVector &u, MultiVector &w);

   ParticleSet& GetParticles() { return fluid_particles; }

   MultiVector& U(int nm=0) { return *fp_data.u[nm]; }

   MultiVector& V(int nm=0) { return *fp_data.v[nm]; }

   MultiVector& W(int nm=0) { return *fp_data.w[nm]; }

   MultiVector& X(int nm=0) { return *fp_data.x[nm]; }

   Array<int>& Order()       { return *fp_data.order; }
   
   void Add2DReflectionBC(const Vector &line_start, const Vector &line_end, real_t e, bool invert_normal)
   { bcs.push_back(ReflectionBC_2D{line_start, line_end, e, invert_normal}); }

   void Add2DRecirculationBC(const Vector &inlet_start, const Vector &inlet_end, bool invert_inlet_normal, const Vector &outlet_start, const Vector &outlet_end, bool invert_outlet_normal)
   { 
      MFEM_ASSERT([&]()
      {
         real_t inlet_dist = inlet_start.DistanceTo(inlet_end);
         real_t outlet_dist = outlet_start.DistanceTo(outlet_end);

         return abs(inlet_dist-outlet_dist)/inlet_dist < 1e-12;
         
      }(), "Inlet + outlet must be same length.");
      bcs.push_back(RecirculationBC_2D{inlet_start, inlet_end, invert_inlet_normal, outlet_start, outlet_end, invert_outlet_normal}); 
   }
};


} // namespace navier

} // namespace mfem

#endif // MFEM_NAVIER_PARTICLES_HPP