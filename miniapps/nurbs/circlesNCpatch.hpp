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

#ifndef MFEM_CIRCLES_NCPATCH_HPP
#define MFEM_CIRCLES_NCPATCH_HPP

#include "mfem.hpp"

namespace mfem
{

void GetCircleCP_A7(real_t radius, int n, real_t theta0, bool is90,
                    Array2D<real_t> &cp);

// For testing and demonstration, this function constructs and returns a 2D
// NC-patch NURBS mesh with exact circles as boundary edges.
Mesh* CirclesMesh(int num_refinements)
{
   Mesh *mesh_ptr = new Mesh("../../data/nc3-nurbs.mesh", 1, 1);
   Mesh &mesh = *mesh_ptr;
   const int dim = mesh.Dimension();

   mesh.DegreeElevate(1);

   Vector &w = mesh.NURBSext->GetWeights();

   mesh.NURBSext->ConvertToPatches(*mesh.GetNodes());

   Array<NURBSPatch*> mpatch;
   mesh.NURBSext->GetPatches(mpatch);
   mesh.NURBSext->SetNumCoarsePatches(mesh.NURBSext->GetNP());

   for (int p=0; p<mesh.NURBSext->GetNP(); ++p)
   {
      const KnotVector *kv[3];
      NURBSPatchMap p2g(mesh.NURBSext);
      p2g.SetPatchDofMap(p, kv);

      Array<int> dofs;
      mesh.NURBSext->GetPatchDofs(p, dofs);

      MFEM_VERIFY(p2g.nx() == p2g.ny(), "");
      Vector pw((p2g.nx() + 1) * (p2g.ny() + 1));
      Vector pcp(2 * (p2g.nx() + 1) * (p2g.ny() + 1));

      const int nj = p2g.ny() + 1;

      // For each i, the curve in j is set as part of a circle.
      const real_t hr = 1.0 / p2g.nx();
      for (int i=0; i<=p2g.nx(); ++i)
      {
         Array2D<real_t> cp;
         if (p == 0) // Master patch
         {
            const real_t radius = (i * hr) + 1.0;
            GetCircleCP_A7(radius, p2g.ny() + 1, 0.0, true, cp);
         }
         else if (p == 1) // Slave patch on bottom
         {
            const real_t radius = (i * hr) + 2.0;
            GetCircleCP_A7(radius, p2g.ny() + 1, 0.0, false, cp);
         }
         else // p == 2 slave patch on top
         {
            const real_t radius = (i * hr) + 2.0;
            GetCircleCP_A7(radius, p2g.ny() + 1, 45.0, false, cp);
         }

         for (int j=0; j<=p2g.ny(); ++j)
         {
            if (p == 1)
            {
               w[p2g(2 - i, 2 - j)] = cp(j,2);
            }
            else if (p == 2)
            {
               w[p2g(j, 2 - i)] = cp(j,2);
            }
            else // p == 0
            {
               w[p2g(i,j)] = cp(j,2);
            }

            if (p == 1)
            {
               const int idx = ((2 - i) * nj) + 2 - j;
               pw[idx] = cp(j,2);
               for (int k=0; k<2; ++k)
               {
                  pcp[(2 * idx) + k] = cp(j,k);
               }

               for (int k=0; k<3; ++k)
               {
                  (*mpatch[p])(2 - i, 2 - j, k) = cp(j,k);
               }
            }
            else if (p == 2)
            {
               const int idx = (j * nj) + 2 - i;
               pw[idx] = cp(j,2);
               for (int k=0; k<2; ++k)
               {
                  pcp((2 * idx) + k) = cp(j,k);
               }

               for (int k=0; k<3; ++k)
               {
                  (*mpatch[p])(j, 2 - i, k) = cp(j,k);
               }
            }
            else // p == 0
            {
               const int idx = (i * nj) + j;
               pw[idx] = cp(j,2);
               for (int k=0; k<2; ++k)
               {
                  pcp((2 * idx) + k) = cp(j,k);
               }

               for (int k=0; k<3; ++k)
               {
                  (*mpatch[p])(i, j, k) = cp(j,k);
               }
            }

            for (int k=0; k<2; ++k)
            {
               if (p == 1)
               {
                  (*mesh.GetNodes())[(dim * p2g(2 - i, 2 - j)) + k] = cp(j,k);
               }
               else if (p == 2)
               {
                  (*mesh.GetNodes())[(dim * p2g(j, 2 - i)) + k] = cp(j,k);
               }
               else // p == 0
               {
                  (*mesh.GetNodes())[(dim * p2g(i,j)) + k] = cp(j,k);
               }
            }
         } // j
      } // i

      mesh.NURBSext->SetPatchControlPoints(p, *mpatch[p]);
   } // p

   // Set coarse patch CP
   for (int p=0; p<mesh.NURBSext->GetNP(); ++p)
   {
      constexpr int ncp = 3;
      constexpr real_t hr = 1.0 / ((real_t) (ncp - 1));

      Array2D<real_t> coarseCP(ncp * ncp, 3); // Coarse control points

      for (int i=0; i<ncp; ++i)
      {
         Array2D<real_t> cp;
         if (p == 0) // Master patch
         {
            const real_t radius = (i * hr) + 1.0;
            GetCircleCP_A7(radius, ncp, 0.0, true, cp);
         }
         else if (p == 1) // Slave patch on bottom
         {
            const real_t radius = (i * hr) + 2.0;
            GetCircleCP_A7(radius, ncp, 0.0, false, cp);
         }
         else // p == 2 slave patch on top
         {
            const real_t radius = (i * hr) + 2.0;
            GetCircleCP_A7(radius, ncp, 45.0, false, cp);
         }

         for (int j=0; j<ncp; ++j)
         {
            if (p == 1)
            {
               const int idx = ((2 - j) * ncp) + 2 - i;
               for (int k=0; k<3; ++k)
               {
                  coarseCP(idx, k) = cp(j,k);
               }
            }
            else if (p == 2)
            {
               const int idx = ((2 - i) * ncp) + j;
               for (int k=0; k<3; ++k)
               {
                  coarseCP(idx, k) = cp(j,k);
               }
            }
            else // p == 0
            {
               const int idx = (j * ncp) + i;
               for (int k=0; k<3; ++k)
               {
                  coarseCP(idx, k) = cp(j,k);
               }
            }
         }
      }

      mesh.NURBSext->SetCoarsePatchCP(p, coarseCP);
   } // p

   mesh.NURBSUniformRefinement(num_refinements);

   return mesh_ptr;
}

void Intersect3DLines(const Vector &p0, const Vector &t0, const Vector &p1,
                      const Vector &t1, Vector &p)
{
   // First line:  p0 + x * t0
   // Second line: p1 + y * t1
   // Find p = p0 + x * t0 = p1 + y * t1
   // Overdetermined system, assuming p1 - p0 is in the span of t0 and t1.
   // x * t0 - y * t1 = p1 - p0
   Vector rhs(3);
   rhs = p1;
   rhs -= p0;

   DenseMatrix A(3,2);
   DenseMatrix Ainv(2,3);

   for (int i=0; i<3; ++i)
   {
      A(i,0) = t0[i];
      A(i,1) = -t1[i];
   }

   CalcInverse(A, Ainv);

   Vector sol(2);
   Ainv.Mult(rhs, sol);

   // Verify the solution
   Vector Ax(3);
   A.Mult(sol, Ax);
   Ax -= rhs;
   const real_t error = Ax.Norml2();
   MFEM_ASSERT(error < 1.0e-6, "");

   // Set p from solution, p = p0 + x * t0
   p = t0;
   p *= sol[0];
   p += p0;
}

// Compute the control points for a circle, based on Algorithm A7.1 in The NURBS
// Book, assuming one arc.
void GetCircleCP_A7(real_t radius, int n, real_t theta0, bool is90,
                    Array2D<real_t> &cp)
{
   cp.SetSize(n, 3);

   // First solve for 3 control points.

   // Based on the fixed weights, compute the control points by solving a linear
   // system. Solve the 1D problem in a 2D patch.

   // The following method of constructing a single 2D patch is based on
   // tests/unit/mesh/test_mesh.cpp TEST_CASE("MakeNurbs", "[Mesh]")

   Array<real_t> intervals_array({1});
   Vector intervals(intervals_array.GetData(), intervals_array.Size());
   Array<int> continuity({-1, -1});

   const KnotVector kv(2, intervals, continuity);

   Vector knots(kv.GetNCP());
   const real_t hk = 1.0 / (kv.GetNCP() - 1);
   for (int i = 0; i < kv.GetNCP(); ++i)
   {
      knots[i] = i * hk;
   }

   Array<real_t> pts_2d(3 * kv.GetNCP() * kv.GetNCP());
   int count = 0;
   for (int j = 0; j < kv.GetNCP(); ++j)
      for (int i = 0; i < kv.GetNCP(); ++i)
      {
         pts_2d[count + 0] = knots[i];
         pts_2d[count + 1] = knots[j];
         pts_2d[count + 2] = 1.0;
         count += 3;
      }

   Array<NURBSPatch*> patches;
   patches.Append(new NURBSPatch(&kv, &kv, 3, pts_2d.GetData()));
   Mesh patch_topology_2d =
      Mesh::MakeCartesian2D(1, 1, Element::Type::QUADRILATERAL);

   NURBSExtension ne(&patch_topology_2d, patches);
   Mesh mesh(ne);

   // Set the weights. The following uses a simplified version of Algorithm A7.1
   // in The NURBS Book, assuming one arc.
   Vector &w = mesh.NURBSext->GetWeights();
   w = 1.0;

   // cos(90 / 2) or cos(45 / 2)
   const real_t wc = is90 ? 1.0 / sqrt(2.0) : 0.9238795325112867;
   {
      const KnotVector *kvp[3];
      NURBSPatchMap p2g(mesh.NURBSext);
      p2g.SetPatchDofMap(0, kvp);

      MFEM_VERIFY(p2g.nx() == 2 && p2g.ny() == 2, "");

      for (int i=0; i<=p2g.nx(); ++i)
      {
         w[p2g(i,1)] = wc;
      }
   }

   const real_t dtheta = is90 ? 45.0 : 22.5; // Sweep 90 or 45 degrees in 2 steps
   constexpr int dim = 2;

   for (int k=2; k>=0; --k) // k = 0 is the curve of interest
   {
      const KnotVector *kvp[3];
      NURBSPatchMap p2g(mesh.NURBSext);
      p2g.SetPatchDofMap(0, kvp);

      MFEM_VERIFY(p2g.nx() == 2 && p2g.ny() == 2, "");

      Vector p0(3), p1(3), p2(3), t0(3), t2(3);

      p0 = 0.0;
      p1 = 0.0;
      p2 = 0.0;
      t0 = 0.0;
      t2 = 0.0;

      const real_t r = radius + (k * 0.5);

      cp(0,0) = r * cos(theta0 * M_PI / 180.0);
      cp(0,1) = r * sin(theta0 * M_PI / 180.0);

      t0[0] = -sin(theta0 * M_PI / 180.0);
      t0[1] = cos(theta0 * M_PI / 180.0);

      const real_t theta = theta0 + (2.0 * dtheta);

      cp(2,0) = r * cos(theta * M_PI / 180.0);
      cp(2,1) = r * sin(theta * M_PI / 180.0);

      t2[0] = -sin(theta * M_PI / 180.0);
      t2[1] = cos(theta * M_PI / 180.0);

      for (int j=0; j<2; ++j)
      {
         p0[j] = cp(0,j);
         p2[j] = cp(2,j);
      }

      Intersect3DLines(p0, t0, p2, t2, p1);

      for (int j=0; j<2; ++j)
      {
         cp(1,j) = p1[j];
      }

      for (int i=0; i<3; ++i)
      {
         for (int j=0; j<2; ++j)
         {
            (*mesh.GetNodes())[(dim * p2g(k,i)) + j] = cp(i,j);
         }

         cp(i,2) = w[p2g(k,i)];
      }

      if (n == 3 && k == 0)
      {
         return;
      }
   }

   // Now that 3 control points are known, get more points by refinement.
   MFEM_VERIFY(n == 5 && is90, "");

   Vector kvmidpoint(1);
   kvmidpoint[0] = 0.5;

   Array<Vector*> kvi(2);
   kvi[0] = &kvmidpoint;
   kvi[1] = &kvmidpoint;
   mesh.KnotInsert(kvi);
   mesh.KnotInsert(kvi);

   {
      const KnotVector *kvp[3];
      NURBSPatchMap p2g(mesh.NURBSext);
      p2g.SetPatchDofMap(0, kvp);

      MFEM_VERIFY(p2g.nx() == 4 && p2g.ny() == 4, "");
      MFEM_VERIFY(w.Size() == (p2g.nx() + 1) * (p2g.ny() + 1), "");

      for (int i=0; i<n; ++i)
      {
         for (int j=0; j<2; ++j)
         {
            cp(i,j) = (*mesh.GetNodes())[(dim * p2g(0,i)) + j];
         }
         cp(i,2) = w[p2g(0,i)];
      }
   }
}

}

#endif
