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

#include "mfem.hpp"
using namespace mfem;

#include "unit_tests.hpp"

void SetMinMax(real_t v, std::array<real_t, 2> &m)
{
   if (m[0] < 0.0)
   {
      m = {v, v}; // Initialize
   }

   m[0] = std::min(m[0], v);
   m[1] = std::max(m[1], v);
}

void CheckRadius(Mesh &mesh)
{
   IntegrationPoint ip;
   Vector x;

   // Define a grid of points at which to check the radius.
   constexpr int n = 4;
   constexpr real_t h = 0.25;
   constexpr real_t tol = 1.0e-2;

   // Radius min/max on each element side.
   std::vector<std::array<real_t, 2>> side(4);

   real_t maxErr = 0.0;
   for (int e=0; e<mesh.GetNE(); ++e)
   {
      ElementTransformation *tr = mesh.GetElementTransformation(e);

      for (int s=0; s<4; ++s)
      {
         side[s] = {-1.0, -1.0}; // Initialize radius min/max as negative
      }

      for (int i=0; i<=n; ++i)
         for (int j=0; j<=n; ++j)
         {
            ip.Set2(i*h, j*h);
            tr->Transform(ip, x);
            const real_t r = sqrt((x[0] * x[0]) + (x[1] * x[1]));

            if (i == 0)
            {
               SetMinMax(r, side[3]);
            }
            if (i == n)
            {
               SetMinMax(r, side[1]);
            }
            if (j == 0)
            {
               SetMinMax(r, side[0]);
            }
            if (j == n)
            {
               SetMinMax(r, side[2]);
            }
         }

      int numCurved = 0;
      for (int s=0; s<4; ++s)
      {
         if (side[s][1] - side[s][0] < tol) // If edge is curved
         {
            numCurved++;
            maxErr = std::max(maxErr, side[s][1] - side[s][0]);
         }
      }

      REQUIRE(numCurved == 2);
   }

   REQUIRE(maxErr < 1.0e-14);
}

void SolveFaceNURBS(const Array3D<real_t> &grid, Mesh &mesh)
{
   const int dim = mesh.Dimension();
   FiniteElementCollection* fec = mesh.GetNodes()->OwnFEC();
   FiniteElementSpace fespace(&mesh, fec);
   GridFunction x(&fespace);

   IntegrationPoint ip;

   const int nz = 3;
   const double h = 1.0 / 2.0;

   DenseMatrix A(x.Size());
   std::vector<Vector> b(2);
   for (int i = 0; i < 2; ++i)
   {
      b[i].SetSize(x.Size());
   }

   for (int i = 0; i < nz; ++i)
   {
      for (int j = 0; j < nz; ++j)
      {
         ip.Set2(i * h, j * h);

         for (int l = 0; l < x.Size(); ++l)
         {
            x = 0.0;
            x[l] = 1.0;
            A(nz * i + j, l) = x.GetValue(0, ip);
         }

         for (int l = 0; l < 2; ++l)
         {
            b[l][(nz * i) + j] = grid(i, j, l);
         }
      }
   }

   A.Invert();

   for (int i = 0; i < 2; ++i)
   {
      A.Mult(b[i], x);

      for (int k = 0; k < x.Size(); ++k)
      {
         (*mesh.GetNodes())[(dim * k) + i] = x[k];
      }
   }
}

void GetCircleCP(real_t radius, int n, real_t theta0, real_t theta,
                 Array2D<real_t> &cp)
{
   cp.SetSize(n, 3);

   // First solve for 3 control points.

   // Based on the fixed weights, compute the control points by solving a linear
   // system, with the 1D problem formulated in a 2D patch.

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

   // Set the weights, using a simplified version of Algorithm A7.1 in
   // The NURBS Book, assuming one arc.
   MFEM_VERIFY(theta <= 0.5 * M_PI, "");

   Vector &w = mesh.NURBSext->GetWeights();
   w = 1.0;

   const real_t wc = cos(0.5 * theta);
   {
      const KnotVector *kvp[3];
      NURBSPatchMap p2g(mesh.NURBSext);
      p2g.SetPatchDofMap(0, kvp);

      for (int i=0; i<=p2g.nx(); ++i)
      {
         w[p2g(i,1)] = wc;
      }
   }

   const real_t dtheta = 0.5 * theta; // Sweep in 2 steps

   Array3D<real_t> grid(3, 3, 2);
   // i = 0 is the 1D curve of interest
   for (int i=0; i<3; ++i)
   {
      const real_t r = radius + (i * 0.5);
      for (int j=0; j<3; ++j)
      {
         const real_t theta = theta0 + (j * dtheta);
         grid(i, j, 0) = r * cos(theta);
         grid(i, j, 1) = r * sin(theta);
      }
   }

   SolveFaceNURBS(grid, mesh);

   if (n == 3)
   {
      const KnotVector *kvp[3];
      NURBSPatchMap p2g(mesh.NURBSext);
      p2g.SetPatchDofMap(0, kvp);

      constexpr int dim = 2;
      for (int i=0; i<3; ++i)
      {
         for (int j=0; j<2; ++j)
         {
            cp(i,j) = (*mesh.GetNodes())[(dim * p2g(0,i)) + j];
         }

         cp(i,2) = w[p2g(0,i)];
      }

      return;
   }

   // Now that 3 control points are known, get more points by refinement.

   Vector kvmidpoint(1);
   kvmidpoint[0] = 0.5;

   Array<Vector*> kvi(2);
   kvi[0] = &kvmidpoint;
   kvi[1] = &kvmidpoint;
   mesh.KnotInsert(kvi);
   mesh.KnotInsert(kvi);

   const KnotVector *kvp[3];
   NURBSPatchMap p2g(mesh.NURBSext);
   p2g.SetPatchDofMap(0, kvp);

   MFEM_VERIFY(p2g.nx() == 4 && p2g.ny() == 4, "");
   MFEM_VERIFY(w.Size() == (p2g.nx() + 1) * (p2g.ny() + 1), "");

   constexpr int dim = 2;

   for (int i=0; i<5; ++i)
   {
      for (int j=0; j<2; ++j)
      {
         cp(i,j) = (*mesh.GetNodes())[(dim * p2g(0,i)) + j];
      }
      cp(i,2) = w[p2g(0,i)];
   }
}

Mesh* NCPatchCircles()
{
   Mesh *mesh_ptr = new Mesh("../../data/nc3-nurbs.mesh", 1, 1);
   Mesh &mesh = *mesh_ptr;

   mesh.DegreeElevate(1); // Elevate from degree 1 to 2.

   const int dim = mesh.Dimension();
   Vector &w = mesh.NURBSext->GetWeights();
   Array<NURBSPatch*> mpatch;
   mesh.NURBSext->ConvertToPatches(*mesh.GetNodes());
   mesh.NURBSext->GetPatches(mpatch);

   for (int p=0; p<mesh.NURBSext->GetNP(); ++p)
   {
      const KnotVector *kv[3];
      NURBSPatchMap p2g(mesh.NURBSext);
      p2g.SetPatchDofMap(p, kv);

      Array<int> dofs;
      mesh.NURBSext->GetPatchDofs(p, dofs);

      const int nj = p2g.ny() + 1;

      // For each i, the curve in j is set as part of a circle.
      const real_t hr = 1.0 / p2g.nx();
      for (int i=0; i<=p2g.nx(); ++i)
      {
         Array2D<real_t> cp;
         if (p == 0) // 90-degree patch on the bottom
         {
            const real_t radius = (i * hr) + 1.0;
            const real_t theta = 0.5 * M_PI;
            GetCircleCP(radius, p2g.ny() + 1, 0.0, theta, cp);
         }
         else if (p == 1) // 45-degree patch on bottom
         {
            const real_t radius = (i * hr) + 2.0;
            const real_t theta = 0.25 * M_PI;
            GetCircleCP(radius, p2g.ny() + 1, 0.0, theta, cp);
         }
         else // p == 2, 45-degree patch on top
         {
            const real_t radius = (i * hr) + 2.0;
            const real_t theta = 0.25 * M_PI;
            GetCircleCP(radius, p2g.ny() + 1, 0.25 * M_PI, theta, cp);
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
               for (int k=0; k<3; ++k)
               {
                  (*mpatch[p])(2 - i, 2 - j, k) = cp(j,k);
               }
            }
            else if (p == 2)
            {
               for (int k=0; k<3; ++k)
               {
                  (*mpatch[p])(j, 2 - i, k) = cp(j,k);
               }
            }
            else // p == 0
            {
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

   mesh.NURBSUniformRefinement(2);
   return mesh_ptr;
}

TEST_CASE("NURBS NC-patch non-unit weights", "[NURBS]")
{
   Mesh *mesh = NCPatchCircles();

   CheckRadius(*mesh);

   /*
   mesh->NURBSext->ConvertToPatches(*mesh->GetNodes());
   std::ofstream mesh_ofs("test.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);
   */

   delete mesh;
}
