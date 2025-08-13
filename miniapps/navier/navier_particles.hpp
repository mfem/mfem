
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
 *  @details TODO.
 * 
 * 
 * 
 * 
 *  [1] Dutta, Som (2017). Ph.D. Dissertation: Bulle-Effect and its implications for morphodynamics of river diversions. https://www.ideals.illinois.edu/items/102343.
 * 
 */
class NavierParticles
{
protected:

   /// Active fluid particle set.
   ParticleSet fluid_particles;

   /// Inactive fluid particle set. Particles that leave the domain are added here.
   ParticleSet inactive_fluid_particles;

   FindPointsGSLIB finder;

   /// Timestep history.
   Array<real_t> dthist{0.0, 0.0, 0.0};

   /// BDFk coefficients, k=1,2,3.
   std::array<Array<real_t>, 3> beta_k;

   /// EXTk coefficients, k=1,2,3.
   std::array<Array<real_t>, 3> alpha_k;

   /// Carrier of field + tag indices.
   struct FluidParticleIndices
   {
      struct FieldIndices
      {
         int kappa, zeta, gamma;
         int u[4];
         int v[4];
         int w[4];
         int x[3];
      } field;
      
      struct TagIndices
      {
         int order;
      } tag;

   } fp_idx;

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

   
   NavierParticles(MPI_Comm comm, int num_particles, Mesh &m);

   void Step(const real_t &dt, int cur_step, const ParGridFunction &u_gf, const ParGridFunction &w_gf);

   void InterpolateUW(const ParGridFunction &u_gf, const ParGridFunction &w_gf);

   void DeactivateLostParticles(bool findpts);

   ParticleSet& GetParticles() { return fluid_particles; }

   ParticleSet& GetInactiveParticles() { return inactive_fluid_particles; }

   MultiVector& Kappa()     { return fluid_particles.Field(fp_idx.field.kappa); }

   MultiVector& Zeta()      { return fluid_particles.Field(fp_idx.field.zeta); }

   MultiVector& Gamma()     { return fluid_particles.Field(fp_idx.field.gamma); }

   MultiVector& U(int nm=0) { return fluid_particles.Field(fp_idx.field.u[nm]); }

   MultiVector& V(int nm=0) { return fluid_particles.Field(fp_idx.field.v[nm]); }

   MultiVector& W(int nm=0) { return fluid_particles.Field(fp_idx.field.w[nm]); }

   MultiVector& X(int nm=0) { return nm == 0 ? fluid_particles.Coords() : fluid_particles.Field(fp_idx.field.x[nm-1]); }

   Array<int>& Order()      { return fluid_particles.Tag(fp_idx.tag.order); }

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