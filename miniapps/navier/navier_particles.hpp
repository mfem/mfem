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

   /// std::vector of all boundary condition structs
   std::vector<BCVariant> bcs;

   /// Update \ref beta_k and \ref alpha_k using current \ref dthist
   void SetTimeIntegrationCoefficients();

   /// 2D particle step for particle at index \p p
   void ParticleStep2D(const real_t &dt, int p);

   /** @brief Given two 2D points, get the unit normal to the line connecting them
    *
    *  For \p inv_normal *false*, the normal is +90 degrees from the line connecting \p p1 to \p p2 .
    *  For \p inv_normal *true*, the normal is -90 degrees from the line connecting \p p1 to \p p2 .
    */
   static void Get2DNormal(const Vector &p1, const Vector &p2, bool inv_normal,
                           Vector &normal);

   /** @brief Given two 2D segments, get the intersection point if it exists.
    *
    *  Specifically this function solves the following system of equations:
    *       r_1 = s1_start + t1*[s1_end - s1_start]
    *       r_2 = s2_start + t2*[s2_end - s2_start]
    *
    *  and then checks if 0<=t1<=1 and 0<=t2<=1 .
    *
    *  @param[in] s1_start         First line segment start point.
    *  @param[in] s1_end           First line segment end point.
    *  @param[in] s2_start         Second line segment start point.
    *  @param[in] s2_end           Second line segment end point.
    *  @param[in] x_int            Computed intersection point (if it exists)
    *  @param[in] t1_ptr           (Optional) Computed t1
    *  @param[in] t2_ptr           (Optional) Computed t2
    *
    *  @return *true* if intersection point exists, *false* otherwise
    */
   static bool Get2DSegmentIntersection(const Vector &s1_start,
                                        const Vector &s1_end, const Vector &s2_start, const Vector &s2_end,
                                        Vector &x_int, real_t *t1_ptr=nullptr, real_t *t2_ptr=nullptr);

   void Apply2DReflectionBC(const ReflectionBC_2D &bc);
   void Apply2DRecirculationBC(const RecirculationBC_2D &bc);
   void ApplyBCs();


   /** @brief Move any particles that have left the domain to the inactive ParticleSet.
    *
    *  This method uses the FindPointsGSLIB object internal to the class to detect if particles are within the domain or not.
    *
    *  @param[in] findpts     If true, call FindPointsGSLIB::FindPoints prior to deactivation (if particle coordinates out of sync with FindPointsGSLIB)
    */
   void DeactivateLostParticles(bool findpts);

   mutable Vector up, vp, xpn, xp;
   mutable Vector r, C;

public:

   /// Initialize NavierParticles with \p num_particles using fluid mesh \p m .
   NavierParticles(MPI_Comm comm, int num_particles, Mesh &m);

   /// Set initial timestep in time history array
   void Setup(const real_t &dt) { dthist[0] = dt; }

   /** @brief Step the particles in time.
    *
    *  @param[in] dt
    *  @param[in] u_gf     Fluid velocity on fluid mesh.
    *  @param[in] w_gf     Fluid vorticity on fluid mesh.
    */
   void Step(const real_t &dt, const ParGridFunction &u_gf,
             const ParGridFunction &w_gf);

   /** @brief Interpolate fluid velocity and vorticity onto current particles' location.
    *
    *  @param[in] u_gf     Fluid velocity on fluid mesh.
    *  @param[in] w_gf     Fluid vorticity on fluid mesh.
    */
   void InterpolateUW(const ParGridFunction &u_gf, const ParGridFunction &w_gf);

   /// Get reference to the active ParticleSet.
   ParticleSet& GetParticles() { return fluid_particles; }

   /// Get reference to the inactive ParticleSet.
   ParticleSet& GetInactiveParticles() { return inactive_fluid_particles; }

   /// Get reference to the kappa MultiVector.
   MultiVector& Kappa()     { return fluid_particles.Field(fp_idx.field.kappa); }

   /// Get reference to the Zeta MultiVector.
   MultiVector& Zeta()      { return fluid_particles.Field(fp_idx.field.zeta); }

   /// Get reference to the Gamma MultiVector.
   MultiVector& Gamma()     { return fluid_particles.Field(fp_idx.field.gamma); }

   /// Get reference to the fluid velocity-interpolated MultiVector at time n - \p nm .
   MultiVector& U(int nm=0) { MFEM_ASSERT(nm < 4, "nm must be <= 3"); return fluid_particles.Field(fp_idx.field.u[nm]); }

   /// Get reference to the particle velocity MultiVector at time n - \p nm .
   MultiVector& V(int nm=0) { MFEM_ASSERT(nm < 4, "nm must be <= 3"); return fluid_particles.Field(fp_idx.field.v[nm]); }

   /// Get reference to the fluid vorticity-interpolated MultiVector at time n - \p nm .
   MultiVector& W(int nm=0) { MFEM_ASSERT(nm < 4, "nm must be <= 3"); return fluid_particles.Field(fp_idx.field.w[nm]); }

   /// Get reference to the position MultiVector at time n - \p nm .
   MultiVector& X(int nm=0) { MFEM_ASSERT(nm < 4, "nm must be <= 3"); return nm == 0 ? fluid_particles.Coords() : fluid_particles.Field(fp_idx.field.x[nm-1]); }

   /// Get reference to the order Array<int>.
   Array<int>& Order()      { return fluid_particles.Tag(fp_idx.tag.order); }

   /** @brief Add a 2D wall reflective boundary condition.
    *
    *  @warning The normal must be facing into the domain. See \ref Get2DNormal for details on the normal direction.
    *
    *  @param[in] line_start     Wall line segment start point.
    *  @param[in] line_end       Wall line segment end point.
    *  @param[in] e              Boundary collision reconstitution constant. 1 for elastic.
    *  @param[in] invert_normal  Invert direction of the normal.
    *
    */
   void Add2DReflectionBC(const Vector &line_start, const Vector &line_end,
                          real_t e, bool invert_normal)
   { bcs.push_back(ReflectionBC_2D{line_start, line_end, e, invert_normal}); }

   /** @brief Add a 2D recirculation / one-way periodic boundary condition.
    *
    *  @warning *Both* normals must be facing into the domain. See \ref Get2DNormal for details on the normal direction.
    *
    *  @param[in] inlet_start             Inlet line segment start point.
    *  @param[in] inlet_end               Inlet line segment end point.
    *  @param[in] invert_inlet_normal     Invert direction of the inlet normal.
    *  @param[in] outlet_start              Outlet line segment start point.
    *  @param[in] outlet_end              Outlet line segment end point.
    *  @param[in] invert_outlet_normal    Invert direction of the outlet normal.
    *
    */
   void Add2DRecirculationBC(const Vector &inlet_start, const Vector &inlet_end,
                             bool invert_inlet_normal, const Vector &outlet_start, const Vector &outlet_end,
                             bool invert_outlet_normal)
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
