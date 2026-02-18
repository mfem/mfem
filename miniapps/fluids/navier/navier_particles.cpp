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

#include "navier_particles.hpp"

#ifdef MFEM_USE_GSLIB

using namespace std;

namespace mfem
{
namespace navier
{

void NavierParticles::SetTimeIntegrationCoefficients()
{
   // Ratio of time step history at dt(t_{n}) - dt(t_{n-1})
   real_t rho1 = 0.0;

   // Ratio of time step history at dt(t_{n-1}) - dt(t_{n-2})
   real_t rho2 = 0.0;

   rho1 = dthist[0] / dthist[1];
   rho2 = dthist[1] / dthist[2];

   for (int o = 0; o < 3; o++)
   {
      if (o == 0) // k=1
      {
         beta_k[o] = 0.0;
         beta_k[o][0] = 1.0;
         beta_k[o][1] = -1.0;

         alpha_k[o] = 0.0;
         alpha_k[o][0] = 1.0;
      }
      else if (o == 1) // k=2
      {
         beta_k[o][0] = (1.0 + 2.0 * rho1) / (1.0 + rho1);
         beta_k[o][1] = -(1.0 + rho1);
         beta_k[o][2] = pow(rho1, 2.0) / (1.0 + rho1);
         beta_k[o][3] = 0.0;

         alpha_k[o][0] = 1.0 + rho1;
         alpha_k[o][1] = -rho1;
         alpha_k[o][2] = 0.0;
      }
      else // k=3
      {
         beta_k[o][0] = 1.0 + rho1 / (1.0 + rho1)
                        + (rho2 * rho1) / (1.0 + rho2 * (1 + rho1));
         beta_k[o][1] = -1.0 - rho1 -
                        (rho2 * rho1 * (1.0 + rho1)) / (1.0 + rho2);
         beta_k[o][2] = pow(rho1, 2.0) * (rho2 + 1.0 / (1.0 + rho1));
         beta_k[o][3] = -(pow(rho2, 3.0) * pow(rho1, 2.0) * (1.0 + rho1))
                        / ((1.0 + rho2) * (1.0 + rho2 + rho2 * rho1));

         alpha_k[o][0] = ((1.0 + rho1) *
                          (1.0 + rho2 * (1.0 + rho1))) / (1.0 + rho2);
         alpha_k[o][1] = -rho1 * (1.0 + rho2 * (1.0 + rho1));
         alpha_k[o][2] = (pow(rho2, 2.0) * rho1 * (1.0 + rho1)) / (1.0 + rho2);
      }
   }
}

void NavierParticles::ParticleStep2D(const real_t &dt, int p)
{
   real_t w_n_ext = 0.0;

   int order_idx = Order()[p] - 1;
   const Array<real_t> &beta = beta_k[order_idx];
   const Array<real_t> &alpha = alpha_k[order_idx];

   const real_t &kappa = Kappa()[p];
   const real_t &zeta = Zeta()[p];
   const real_t &gamma = Gamma()[p];

   // Extrapolate particle vorticity using EXTk (w_n is new vorticity at old
   // particle loc)
   // w_n_ext = alpha1*w_nm1 + alpha2*w_nm2 + alpha3*w_nm3
   for (int j = 1; j <= 3; j++)
   {
      w_n_ext += alpha[j-1]*W(j)(p, 0);
   }

   // Assemble the 2D matrix B with implicit terms
   DenseMatrix B({{beta[0]+dt*kappa, zeta*dt*w_n_ext},
      {           -zeta*dt*w_n_ext, beta[0]+dt*kappa}});

   // Assemble the RHS with BDF and EXT terms
   r = 0.0;
   for (int j = 1; j <= 3; j++)
   {
      U(j).GetValuesRef(p, up);
      V(j).GetValuesRef(p, vp);

      // Add particle velocity component
      add(r, -beta[j], vp, r);

      // Create C
      C = up;
      C *= kappa;
      add(C, -gamma, Vector({0_r, 1_r}), C);
      add(C, zeta*w_n_ext, Vector({ up[1], -up[0]}), C);

      // Add C
      add(r, dt*alpha[j-1], C, r);
   }

   // Solve for particle velocity
   DenseMatrixInverse B_inv(B);
   V(0).GetValuesRef(p, vp);
   B_inv.Mult(r, vp);

   // Compute updated particle position
   X(0).GetValuesRef(p, xpn);
   xpn = 0.0;
   for (int j = 1; j <= 3; j++)
   {
      X(j).GetValuesRef(p, xp);
      add(xpn, -beta[j], xp, xpn);
   }
   V(0).GetValuesRef(p, vp);
   add(xpn, dt, vp, xpn);
   xpn *= 1.0/beta[0];
}

void NavierParticles::Get2DNormal(const Vector &p1, const Vector &p2,
                                  bool inv_normal, Vector &normal)
{
   normal.SetSize(2);
   Vector diff(p2);
   diff -= p1;
   if (inv_normal)
   {
      normal[0] = diff[1];
      normal[1] = -diff[0];
   }
   else
   {
      normal[0] = -diff[1];
      normal[1] = diff[0];
   }
   normal /= normal.Norml2(); // normalize
}

bool NavierParticles::Get2DSegmentIntersection(const Vector &s1_start,
                                               const Vector &s1_end, const Vector &s2_start, const Vector &s2_end,
                                               Vector &x_int, real_t *t1_ptr, real_t *t2_ptr)
{
   // Compute the intersection parametrically
   // r_1 = s1_start + t1*[s1_end - s1_start]
   // r_2 = s2_start + t2*[s2_end - s2_start]
   real_t denom = (s1_end[0]-s1_start[0])*(s2_start[1] - s2_end[1]) -
                  (s1_end[1]-s1_start[1])*(s2_start[0] - s2_end[0]);

   // If line is parallel, don't compute at all
   // Note that nearly-parallel intersections are not well-posed (denom >>> 0)
   real_t rho = abs(denom)/(s1_start.DistanceTo(s2_end)*s2_start.DistanceTo(
                               s2_end));
   if (rho < 1e-12)
   {
      return false;
   }

   real_t t1 = ( (s2_start[0] - s1_start[0])*(s2_start[1]-s2_end[1]) -
                 (s2_start[1] - s1_start[1])*(s2_start[0]-s2_end[0]) ) / denom;
   real_t t2 = ( (s1_end[0] - s1_start[0])*(s2_start[1] - s1_start[1]) -
                 (s1_end[1] - s1_start[1])*(s2_start[0] - s1_start[0]))/ denom;

   // If intersection falls on line segment of s1_start to s1_end AND s2_start to s2_end, set x_int and return true
   if ((0_r <= t1 && t1 <= 1_r) && (0_r <= t2 && t2 <= 1_r))
   {
      // Get the point of intersection
      x_int = s2_end;
      x_int -= s2_start;
      x_int *= t2;
      x_int += s2_start;
      if (t1_ptr)
      {
         *t1_ptr = t1;
      }
      if (t2_ptr)
      {
         *t2_ptr = t2;
      }
      return true;
   }
   return false;
}

void NavierParticles::Apply2DReflectionBC(const ReflectionBC_2D &bc)
{
   Vector normal;
   Get2DNormal(bc.line_start, bc.line_end, bc.invert_normal, normal);

   Vector p_xn(2), p_xnm1(2), p_xdiff(2), x_int(2), p_vn(2), p_vdiff(2);
   for (int i = 0; i < fluid_particles.GetNParticles(); i++)
   {
      X().GetValuesRef(i, p_xn);
      X(1).GetValuesRef(i, p_xnm1);
      V().GetValuesRef(i, p_vn);

      // If line_start to line_end and x_nm1 to x_n intersect, apply reflection
      if (Get2DSegmentIntersection(bc.line_start, bc.line_end,
                                   p_xnm1, p_xn, x_int))
      {
         // Verify that the particle moved INTO the wall
         // (Important for cases where p_xnm1 is on the wall within
         //  machine precision)
         p_xdiff = p_xn;
         p_xdiff -= p_xnm1;
         if (p_xdiff*normal > 0)
         {
            continue;
         }

         real_t dt_c = p_xnm1.DistanceTo(x_int)/p_vn.Norml2();

         // Correct the velocity
         p_vdiff = p_vn;
         add(p_vn, -(1+bc.e)*(p_vn*normal), normal, p_vn);
         p_vdiff -= p_vn;

         // Correct the position
         int &o = Order()[i];
         add(p_xn, (1.0/beta_k[o][0])*(dt_c - dthist[0]), p_vdiff, p_xn);

         // Set order to 0 (so that it becomes 1 on next iteration)
         o = 0;
      }
   }
}

void NavierParticles::Apply2DRecirculationBC(const RecirculationBC_2D &bc)
{
   Vector inlet_normal(2), outlet_normal(2);
   Get2DNormal(bc.inlet_start, bc.inlet_end, bc.invert_inlet_normal, inlet_normal);
   Get2DNormal(bc.outlet_start, bc.outlet_end, bc.invert_outlet_normal,
               outlet_normal);

   real_t inlet_length = bc.inlet_start.DistanceTo(bc.inlet_end);
   real_t outlet_length = bc.outlet_start.DistanceTo(
                             bc.outlet_end); // should be == inlet_length

   Vector inlet_tan(2), outlet_tan(2);
   inlet_tan = bc.inlet_end;
   inlet_tan -= bc.inlet_start;
   inlet_tan /= inlet_length;
   outlet_tan = bc.outlet_end;
   outlet_tan -= bc.outlet_start;
   outlet_tan /= outlet_length;

   real_t t1;
   Vector p_xn(2), p_xnm1(2), x_int(2), p_xc(2), p_x_int_diff(2);
   for (int i = 0; i < fluid_particles.GetNParticles(); i++)
   {
      X().GetValuesRef(i, p_xn);
      X(1).GetValuesRef(i, p_xnm1);

      // If outlet_start to outlet_end and x_nm1 to x_n intersect,
      // apply recirculation
      if (Get2DSegmentIntersection(bc.outlet_start, bc.outlet_end, p_xnm1, p_xn,
                                   x_int, &t1))
      {
         // Compute the corresponding intersection location on inlet
         p_xc = 0.0;
         add(bc.inlet_start, t1*inlet_length, inlet_tan, p_xc);

         // Compute the normal distance from p_xn to x_int, and add
         p_x_int_diff = p_xn;
         p_x_int_diff -= x_int;
         real_t normal_dist = abs(p_x_int_diff*outlet_normal);
         add(p_xc, normal_dist, inlet_normal, p_xc);

         // Compute tangential distance from p_xn to x_int, and add
         real_t tan_dist = abs(p_x_int_diff*outlet_tan);
         add(p_xc, tan_dist, inlet_tan, p_xc);

         // Update the position
         p_xn = p_xc;

         // Set order to 0, to avoid re-computing x history as well
         Order()[i] = 0;
      }

   }
}

void NavierParticles::ApplyBCs()
{
   for (BCVariant &bc_v : bcs)
   {
      std::visit(
         [this](auto &bc)
      {
         using T = std::decay_t<decltype(bc)>;
         if constexpr(std::is_same_v<T, ReflectionBC_2D>)
         {
            Apply2DReflectionBC(bc);
         }
         else if constexpr(std::is_same_v<T, RecirculationBC_2D>)
         {
            Apply2DRecirculationBC(bc);
         }
      },bc_v);

   }
}

NavierParticles::NavierParticles(MPI_Comm comm, int num_particles, Mesh &m)
   : fluid_particles(comm, num_particles, m.SpaceDimension()),
     inactive_fluid_particles(comm, 0, m.SpaceDimension()),
     finder(comm)
{

   for (int o = 0; o < 3; o++)
   {
      beta_k[o].SetSize(4);
      alpha_k[o].SetSize(3);
   }

   finder.Setup(m);

   int dim = fluid_particles.GetDim();

   // Initialize kappa, zeta, gamma
   // Ordering defaults to byVDIM and name defaults to 'Field_{field_idx}' if
   // unspecified.
   fp_idx.field.kappa = fluid_particles.AddField(1, Ordering::byVDIM, "kappa");
   fp_idx.field.zeta = fluid_particles.AddField(1, Ordering::byVDIM, "zeta");
   fp_idx.field.gamma = fluid_particles.AddNamedField(1, "gamma");

   inactive_fluid_particles.AddField(1);
   inactive_fluid_particles.AddField(1);
   inactive_fluid_particles.AddField(1);

   // Initialize fluid particle fields
   for (int i = 0; i < N_HIST; i++)
   {
      fp_idx.field.u[i] = fluid_particles.AddField(dim);
      fp_idx.field.v[i] = fluid_particles.AddField(dim);
      fp_idx.field.w[i] = fluid_particles.AddField(dim);
      if (i > 0)
      {
         fp_idx.field.x[i-1] = fluid_particles.AddField(dim);
      }
   }

   // Initialize order (tag)
   fp_idx.tag.order = fluid_particles.AddTag("Order");
   fluid_particles.Tag(fp_idx.tag.order) = 0;

   // Reserve num_particles for inactive_fluid_particles
   inactive_fluid_particles.Reserve(num_particles);

   r.SetSize(dim);
   C.SetSize(dim);
}

void NavierParticles::Step(const real_t dt, const ParGridFunction &u_gf,
                           const ParGridFunction &w_gf)
{
   // Shift fluid velocity, fluid vorticity, particle velocity, and particle position
   for (int i = N_HIST-1; i > 0; i--)
   {
      U(i) = std::move(U(i-1));
      V(i) = std::move(V(i-1));
      W(i) = std::move(W(i-1));
      X(i) = std::move(X(i-1));
   }
   U(0).SetSize(U(1).Size());
   V(0).SetSize(V(1).Size());
   W(0).SetSize(W(1).Size());
   X(0).SetSize(X(1).Size());

   SetTimeIntegrationCoefficients();

   if (fluid_particles.GetDim() == 2)
   {
      for (int i = 0; i < fluid_particles.GetNParticles(); i++)
      {
         // Increment particle order
         int &order = Order()[i];
         if (order < 3)
         {
            order++;
         }
         ParticleStep2D(dt, i);
      }
   }
   else
   {
      MFEM_ABORT("3D particles not yet implemented.");
   }

   // Apply any BCs
   ApplyBCs();

   // Re-interpolate fluid velocity + vorticity onto particles' new location
   InterpolateUW(u_gf, w_gf);

   // Move lost particles from active to inactive. We don't search for points again
   // because that is already done in InterpolateUW.
   DeactivateLostParticles(false);

   // Rotate values in time step history
   dthist[2] = dthist[1];
   dthist[1] = dthist[0];
   dthist[0] = dt;
}

void NavierParticles::InterpolateUW(const ParGridFunction &u_gf,
                                    const ParGridFunction &w_gf)
{
   finder.FindPoints(X());

   finder.Interpolate(u_gf, U(), U().GetOrdering());

   finder.Interpolate(w_gf, W(), W().GetOrdering());
}

void NavierParticles::DeactivateLostParticles(bool findpts)
{
   if (findpts)
   {
      finder.FindPoints(X());
   }

   const Array<unsigned int> lost_idxs = finder.GetPointsNotFoundIndices();
   Array<int> inactive_add_idxs;
   inactive_fluid_particles.AddParticles(lost_idxs.Size(), &inactive_add_idxs);

   Vector coords;
   for (int i = 0; i < lost_idxs.Size(); i++)
   {
      X().GetValues(lost_idxs[i], coords);
      inactive_fluid_particles.Coords().SetValues(inactive_add_idxs[i], coords);
      inactive_fluid_particles.Field(0)[inactive_add_idxs[i]] = Kappa()[lost_idxs[i]];
      inactive_fluid_particles.Field(1)[inactive_add_idxs[i]] = Zeta()[lost_idxs[i]];
      inactive_fluid_particles.Field(2)[inactive_add_idxs[i]] = Gamma()[lost_idxs[i]];
   }

   fluid_particles.RemoveParticles(lost_idxs);

}

} // namespace navier
} // namespace mfem
#endif // MFEM_USE_GSLIB
