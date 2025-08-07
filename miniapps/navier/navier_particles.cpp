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

using namespace std;

namespace mfem
{
namespace navier
{

void NavierParticles::SetTimeIntegrationCoefficients(int step)
{
   // Ratio of time step history at dt(t_{n}) - dt(t_{n-1})
   real_t rho1 = 0.0;

   // Ratio of time step history at dt(t_{n-1}) - dt(t_{n-2})
   real_t rho2 = 0.0;

   rho1 = dthist[0] / dthist[1];
   rho2 = dthist[1] / dthist[2];

   if (step == 0)
   {
      beta = 0.0;
      beta[0] = 1.0;
      beta[1] = -1.0;

      alpha = 0.0;
      alpha[0] = 1.0;
   }
   else if (step == 1)
   {
      beta[0] = (1.0 + 2.0 * rho1) / (1.0 + rho1);
      beta[1] = -(1.0 + rho1);
      beta[2] = pow(rho1, 2.0) / (1.0 + rho1);
      beta[3] = 0.0;

      alpha[0] = 1.0 + rho1;
      alpha[1] = -rho1;
      alpha[2] = 0.0;
   }
   else // (step >= 2)
   {
      beta[0] = 1.0 + rho1 / (1.0 + rho1)
            + (rho2 * rho1) / (1.0 + rho2 * (1 + rho1));
      beta[1] = -1.0 - rho1 - (rho2 * rho1 * (1.0 + rho1)) / (1.0 + rho2);
      beta[2] = pow(rho1, 2.0) * (rho2 + 1.0 / (1.0 + rho1));
      beta[3] = -(pow(rho2, 3.0) * pow(rho1, 2.0) * (1.0 + rho1))
            / ((1.0 + rho2) * (1.0 + rho2 + rho2 * rho1));
      alpha[0] = ((1.0 + rho1) * (1.0 + rho2 * (1.0 + rho1))) / (1.0 + rho2);
      alpha[1] = -rho1 * (1.0 + rho2 * (1.0 + rho1));
      alpha[2] = (pow(rho2, 2.0) * rho1 * (1.0 + rho1)) / (1.0 + rho2);
   }
}

void NavierParticles::ParticleStep2D(const real_t &dt, int p)
{
   real_t w_n_ext = 0.0;

   // Extrapolate particle vorticity using EXTk (w_n is new vorticity at old particle loc)
   // w_n_ext = alpha1*w_nm1 + alpha2*w_nm2 + alpha3*w_nm3
   for (int j = 1; j <= 3; j++)
   {
      w_n_ext += alpha[j-1]*(*fp_data.w[j])(p, 0);
   }

   // Assemble the 2D matrix B
   DenseMatrix B({{beta[0]+dt*kappa, zeta*dt*w_n_ext},
                  {-zeta*dt*w_n_ext, beta[0]+dt*kappa}});

   // Assemble the RHS
   r = 0.0;
   for (int j = 1; j <= 3; j++)
   {
      fp_data.u[j]->GetRefNodeValues(p, up);
      fp_data.v[j]->GetRefNodeValues(p, vp);

      // Add particle velocity component
      add(r, -beta[j], vp, r);

      // Create C
      C = up;
      C *= kappa;
      add(C, -gamma, Vector({0.0, 1.0}), C);
      add(C, zeta*w_n_ext, Vector({ up[1], -up[0]}), C);

      // Add C
      add(r, dt*alpha[j-1], C, r);
   }

   // Solve for particle velocity
   DenseMatrixInverse B_inv(B);
   fp_data.v[0]->GetRefNodeValues(p, vp);
   B_inv.Mult(r, vp);

   // Compute updated particle position
   fp_data.x[0]->GetRefNodeValues(p, xpn);
   xpn = 0.0;
   for (int j = 1; j <= 3; j++)
   {
      fp_data.x[j]->GetRefNodeValues(p, xp);
      add(xpn, -beta[j], xp, xpn);
   }
   fp_data.v[0]->GetRefNodeValues(p, vp);
   add(xpn, dt, vp, xpn);
   xpn *= 1.0/beta[0];
}

NavierParticles::NavierParticles(MPI_Comm comm, const real_t kappa_, const real_t gamma_, const real_t zeta_, int num_particles, Mesh &m)
: kappa(kappa_),
  gamma(gamma_),
  zeta(zeta_),
  fluid_particles(comm, num_particles, m.SpaceDimension()),
  finder(comm),
  beta(4),
  alpha(3)
{
   finder.Setup(m);

   int dim = fluid_particles.GetDim();

   // Initialize fluid particle fields
   for (int i = 0; i < 4; i++)
   {
      string suffix = i > 0 ? "_nm" + to_string(i) : "_n";
      fp_data.u[i] = &fluid_particles.AddField(dim, Ordering::byVDIM, ("u" + suffix).c_str());
      fp_data.v[i] = &fluid_particles.AddField(dim, Ordering::byVDIM, ("v" + suffix).c_str());
      fp_data.w[i] = &fluid_particles.AddField(dim, Ordering::byVDIM, ("w" + suffix).c_str());
      if (i == 0)
      {
         fp_data.x[i] = &fluid_particles.Coords();
      }
      else
      {
         fp_data.x[i] = &fluid_particles.AddField(dim, Ordering::byVDIM, ("x" + suffix).c_str());
      }
   }

   r.SetSize(dim);
   C.SetSize(dim);
}

void NavierParticles::Step(const real_t &dt, int cur_step, const ParGridFunction &u_gf, const ParGridFunction &w_gf)
{
   // Shift fluid velocity, fluid vorticity, particle velocity, and particle position
   for (int i = 3; i > 0; i--)
   {
      *fp_data.u[i] = fp_data.u[i-1]->GetData();
      *fp_data.v[i] = fp_data.v[i-1]->GetData();
      *fp_data.w[i] = fp_data.w[i-1]->GetData();
      *fp_data.x[i] = fp_data.x[i-1]->GetData();
   }

   if (cur_step == 0)
   {
      dthist[0] = dt;
   }
   SetTimeIntegrationCoefficients(cur_step);

   if (fluid_particles.GetDim() == 2)
   {
      for (int i = 0; i < fluid_particles.GetNP(); i++)
      {
         ParticleStep2D(dt, i);
      }
   }
   else
   {
      MFEM_ABORT("3D particles not yet implemented.");
   }

   // Re-interpolate fluid velocity + vorticity onto particles' new location
   InterpolateUW(u_gf, w_gf, X(), U(), W());

   // Rotate values in time step history
   dthist[2] = dthist[1];
   dthist[1] = dthist[0];
   dthist[0] = dt;
}

void NavierParticles::InterpolateUW(const ParGridFunction &u_gf, const ParGridFunction &w_gf, const NodeFunction &x, NodeFunction &u, NodeFunction &w)
{
   finder.FindPoints(x, x.GetOrdering());

   finder.Interpolate(u_gf, u);
   Ordering::Reorder(u, u.GetVDim(), u_gf.ParFESpace()->GetOrdering(), u.GetOrdering());

   finder.Interpolate(w_gf, w);
   Ordering::Reorder(w, w.GetVDim(), w_gf.ParFESpace()->GetOrdering(), w.GetOrdering());
}

void NavierParticles::Apply2DReflectionBC(const Vector &line_start, const Vector &line_end, real_t e, bool invert_normal)
{
   Vector normal(2);
   Vector diff(2);
   diff = line_end;
   diff -= line_start;
   if (invert_normal)
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

   Vector p_xn(2), p_xnm1(2), x_int(2), p_vn(2), p_vdiff(2);
   for (int i = 0; i < fluid_particles.GetNP(); i++)
   {
      X().GetRefNodeValues(i, p_xn);
      X(1).GetRefNodeValues(i, p_xnm1);
      V().GetRefNodeValues(i, p_vn);

      // Compute the intersection parametrically
      // r_1 = line_start + t1*[line_end - line_start]
      // r_2 = x_nm1 + t2*[x_n - x_nm1]
      real_t denom = (line_end[0]-line_start[0])*(p_xnm1[1] - p_xn[1]) - (line_end[1]-line_start[1])*(p_xnm1[0] - p_xn[0]);

      // If line is parallel, don't compute at all
      // Note that nearly-parallel intersections are not well-posed (denom >>> 0)...
      if (abs(denom) < 1e-12)
      {
         continue;
      }

      real_t t1 = ( (p_xnm1[0] - line_start[0])*(p_xnm1[1]-p_xn[1]) - (p_xnm1[1] - line_start[1])*(p_xnm1[0]-p_xn[0]) ) / denom;
      real_t t2 = ( (line_end[0] - line_start[0])*(p_xnm1[1] - line_start[1]) - (line_end[1] - line_start[1])*(p_xnm1[0] - line_start[0]) ) / denom;

      // If intersection falls on line segment of x_nm1 to x_n AND line_start to line_end, apply reflection
      if ((0 < t1 && t1 < 1) && (0 < t2 && t2 < 1))
      {
         // Get the point of intersection
         x_int = p_xn;
         x_int -= p_xnm1;
         x_int *= t2;
         x_int += p_xnm1;

         real_t dt_c = p_xnm1.DistanceTo(x_int)/p_vn.Norml2();

         // cout << "line_start: "; line_start.Print();
         // cout << "line_end: "; line_end.Print();
         // cout << "x_nm1: "; p_xnmi.Print();
         // cout << "x_n: "; p_xn.Print();
         // cout << "t1: " << t1 << endl;
         // cout << "intersection: "; x_int.Print();

         // Correct the velocity
         p_vdiff = p_vn;
         add(p_vn, -(1+e)*(p_vn*normal), normal, p_vn);
         p_vdiff -= p_vn;

         // Correct the position
         p_xc = 0.0;
         p_vdiff = p_vnmi;
         p_vdiff -= p_vc;
         add(p_xn, (1.0/beta[0])*(dt_c - dthist[0]), p_vdiff, p_xc);
         X().SetParticleValues(i, p_xc);

         // ... and history
         for (int n = 3; n > 0; n--)
         {
            X(n).GetParticleValues(i, p_xnmi);
            p_xdiff = x_int;
            p_xdiff -= p_xnmi;
            p_xc = 0.0;
            add(p_xnmi, 2*(p_xdiff*normal), normal, p_xc);
            X(n).SetParticleValues(i, p_xc);
         }

         
      }

   }
}

} // namespace navier
} // namespace mfem