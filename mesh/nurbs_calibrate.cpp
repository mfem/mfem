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

#include "nurbs.hpp"
#include "../fem/gridfunc.hpp"

namespace mfem
{

using namespace std;

// Given a grid of physical points in 2D or 3D, solve for a single patch in the
// mesh interpolating those points at Demko reference points. The mesh must have
// exactly 1 patch. The number of DOFs in 1D per element minus 1 is given by
// ned, and the number of elements in each direction of the patch is in nel.
// Note that SolvePhysicalGridSweep is more efficient, using a banded linear
// solver for 1D sweeping solves. This function uses a dense matrix inversion on
// the entire patch, which is not scalable.
void SolvePhysicalGridInterior(const Array3D<real_t> &grid, int ned,
                               std::array<int, 3> nel, Mesh &mesh)
{
   constexpr int patch = 0; // Assuming one patch

   const int dim = mesh.Dimension();
   const int sdim = mesh.SpaceDimension();

   const int ncp0 = (nel[0] * ned) + 1;
   const int ncp1 = (nel[1] * ned) + 1;
   const int ncp2 = dim == 2 ? 1 : (nel[2] * ned) + 1;
   const int ncps[3] = {ncp0, ncp1, ncp2};

   FiniteElementCollection* fec = mesh.GetNodes()->OwnFEC();
   FiniteElementSpace fespace(&mesh, fec);
   GridFunction x(&fespace);

   MFEM_VERIFY(ncp0 * ncp1 * ncp2 == x.Size(), "");

   // Parameter space [0,1]^dim grid point coordinates
   std::vector<Vector> ugrid(dim);

   std::vector<Vector> xi_args(dim);
   std::vector<Array<int>> i_args(dim);

   Array<int> pdofs;
   mesh.NURBSext->GetPatchDofs(patch, pdofs);

   {
      for (int i = 0; i < dim; ++i)
      {
         const KnotVector *kv_i = mesh.NURBSext->GetKnotVector(i);
         mesh.NURBSext->GetKnotVector(i)->GetDemko(ugrid[i]);
         MFEM_VERIFY(ugrid[i].Size() == ncps[i], "");

         i_args[i].SetSize(ncps[i]);
         xi_args[i].SetSize(ncps[i]);

         for (int j=0; j<ncps[i]; ++j)
         {
            i_args[i][j] = kv_i->GetSpan(ugrid[i][j]);
            xi_args[i][j] = kv_i->GetRefPoint(ugrid[i][j], i_args[i][j]);
         }
      }

      // Convert i_args from knot-spans to element indices.
      for (int i = 0; i < dim; ++i)
      {
         const KnotVector *kv_i = mesh.NURBSext->GetKnotVector(i);
         for (int cp=0; cp<ncps[i]; ++cp)
         {
            const int k_i = i_args[i][cp];
            const int el_i = kv_i->ElementIndex(k_i);
            i_args[i][cp] = el_i;
         }
      }
   }

   const int niz = dim == 2 ? 1 : ncps[2] - 2;
   const int ncp = (ncps[0] - 2) * (ncps[1] - 2) * niz;

   DenseMatrix A(sdim);
   GridFunction x0(&fespace);
   Vector sol(ncp);
   std::vector<Vector> b(sdim);

   for (int i = 0; i < sdim; ++i)
   {
      b[i].SetSize(ncp);
   }

   IntegrationPoint ip;

   // Set the system matrix A and RHS vector b.
   A.SetSize(ncp);
   A = 0.0;
   x = 0.0;

   int dofprev = -1;
   for (int dof_k=1; dof_k <= niz; ++dof_k)
   {
      const int osi_k = (ncps[0] - 2) * (ncps[1] - 2) * (dof_k - 1);

      for (int dof_i=1; dof_i < ncp0 - 1; ++dof_i)
         for (int dof_j=1; dof_j < ncp1 - 1; ++dof_j)
         {
            const int el_i = i_args[0][dof_i];
            const int el_j = i_args[1][dof_j];
            const int el_k = dim == 2 ? 0 : i_args[2][dof_k];

            // Element index, assuming a single patch in the mesh.
            const int elem = el_i + (nel[0] * el_j) + (nel[0] * nel[1] * el_k);

            if (dim == 2)
            {
               ip.Set2(xi_args[0][dof_i], xi_args[1][dof_j]);
            }
            else
            {
               ip.Set3(xi_args[0][dof_i], xi_args[1][dof_j], xi_args[2][dof_k]);
            }

            for (int l_i=1; l_i<ncps[0] - 1; ++l_i)
               for (int l_j=1; l_j<ncps[1] - 1; ++l_j)
                  for (int l_k=1; l_k<=niz; ++l_k)
                  {
                     const int l = l_i - 1 + ((ncps[0] - 2) * (l_j - 1)) +
                                   ((ncps[0] - 2) * (ncps[1] - 2) * (l_k - 1));
                     const int os_k = dim == 2 ? 0 : (ncps[0] * ncps[1] * l_k);
                     const int l_full = l_i + (ncps[0] * l_j) + os_k;

                     if (dofprev >= 0)
                     {
                        x[dofprev] = 0.0;
                     }

                     const int dof = pdofs[l_full];
                     x[dof] = 1.0;
                     dofprev = dof;
                     const real_t v = x.GetValue(elem, ip);
                     A(dof_i - 1 + ((ncp0 - 2) * (dof_j - 1)) + osi_k, l) = v;
                  }
         }
   }

   for (int dir=0; dir<sdim; ++dir)
   {
      // Set b[dir]
      x = 0.0;

      for (int l_k=0; l_k<ncps[2]; ++l_k)
      {
         for (int side=0; side<2; ++side)
         {
            // Left and right (2D)
            const int l_i = side * (ncps[0] - 1);
            for (int l_j=0; l_j<ncps[1]; ++l_j)
            {
               const int l_full = l_i + (ncps[0] * l_j) + (ncps[0] * ncps[1] * l_k);
               const int dof = pdofs[l_full];
               x[dof] = (*mesh.GetNodes())[(sdim * dof) + dir];
            }
         }

         for (int side=0; side<2; ++side)
         {
            // Bottom and top (2D)
            const int l_j = side * (ncps[1] - 1);
            for (int l_i=0; l_i<ncps[0]; ++l_i)
            {
               const int l_full = l_i + (ncps[0] * l_j) + (ncps[0] * ncps[1] * l_k);
               const int dof = pdofs[l_full];
               x[dof] = (*mesh.GetNodes())[(sdim * dof) + dir];
            }
         }
      }

      if (dim == 3)
      {
         // Set the interiors of the 3D bottom and top faces.
         for (int side=0; side<2; ++side)
         {
            const int l_k = side * (ncps[2] - 1);
            for (int l_j=1; l_j<ncps[1]-1; ++l_j)
            {
               for (int l_i=1; l_i<ncps[0]-1; ++l_i)
               {
                  const int l_full = l_i + (ncps[0] * l_j) + (ncps[0] * ncps[1] * l_k);
                  const int dof = pdofs[l_full];
                  x[dof] = (*mesh.GetNodes())[(sdim * dof) + dir];
               }
            }
         }
      }

      x0 = x;
      x = 0.0;

      // Set the interior DOFs
      for (int dof_k=1; dof_k <= niz; ++dof_k)
      {
         const int osi_k = (ncps[0] - 2) * (ncps[1] - 2) * (dof_k - 1);

         for (int dof_i=1; dof_i < ncp0 - 1; ++dof_i)
            for (int dof_j=1; dof_j < ncp1 - 1; ++dof_j)
            {
               const int el_i = i_args[0][dof_i];
               const int el_j = i_args[1][dof_j];
               const int el_k = dim == 2 ? 0 : i_args[2][dof_k];

               // Element index, assuming a single patch in the mesh.
               const int elem = el_i + (nel[0] * el_j) + (nel[0] * nel[1] * el_k);

               if (dim == 2)
               {
                  ip.Set2(xi_args[0][dof_i], xi_args[1][dof_j]);
               }
               else
               {
                  ip.Set3(xi_args[0][dof_i], xi_args[1][dof_j], xi_args[2][dof_k]);
               }

               const real_t v0 = x0.GetValue(elem, ip);

               // Set the physical point
               const real_t g = dim == 2 ? grid(dof_i, dof_j, dir) :
                                grid(dof_i, dof_j, dir + (dof_k * dim));

               b[dir][dof_i - 1 + ((ncp0 - 2) * (dof_j - 1)) + osi_k] = g - v0;
            }
      }
   } // dir loop

   A.Invert();

   for (int i = 0; i < sdim; ++i)
   {
      A.Mult(b[i], sol);

      for (int j=1; j<ncp0 - 1; ++j)
         for (int k=1; k<ncp1 - 1; ++k)
            for (int l=0; l<niz; ++l)
            {
               const int os_l = dim == 2 ? 0 : ncp0 * ncp1 * (l + 1);
               const int dof = pdofs[j + (ncp0 * k) + os_l];
               const int sdof = j - 1 + ((ncp0 - 2) * (k - 1)) +
                                ((ncp0 - 2) * (ncp1 - 2) * l);
               (*mesh.GetNodes())[(sdim * dof) + i] = sol[sdof];
            }
   }
}

// See documentation for SolvePhysicalGridInterior.
void SolvePhysicalGridSweep(const Array3D<real_t> &grid, int ned,
                            std::array<int, 3> nel, Mesh &mesh)
{
   constexpr int patch = 0; // Assuming one patch

   const int dim = mesh.Dimension();
   const int sdim = mesh.SpaceDimension();

   const int ncp0 = (nel[0] * ned) + 1;
   const int ncp1 = (nel[1] * ned) + 1;
   const int ncp2 = dim == 2 ? 1 : (nel[2] * ned) + 1;
   const int ncps[3] = {ncp0, ncp1, ncp2};

   MFEM_VERIFY(grid.GetSize1() == ncp0 && grid.GetSize2() == ncp1 &&
               grid.GetSize3() == sdim * ncp2, "");

   Array<Vector*> x;
   for (int i = 0; i < sdim; ++i) { x.Append(new Vector(ncps[0])); }

   // Parameter space [0,1]^dim grid point coordinates
   std::vector<Vector> ugrid(dim);

   for (int i = 0; i < dim; ++i)
   {
      mesh.NURBSext->GetKnotVector(i)->GetDemko(ugrid[i]);
   }

   Array<int> pdofs;
   mesh.NURBSext->GetPatchDofs(patch, pdofs);

   // Interpolate a 2D surface or 3D volume by sweeping curve interpolations in
   // each direction. See Algorithm A9.4 of "The NURBS Book" - 2nd ed - Piegl and
   // Tiller.

   // Sweep in the first direction
   for (int k = 0; k < ncps[2]; ++k)
   {
      for (int j = 0; j < ncps[1]; ++j)
      {
         for (int i = 0; i < ncps[0]; i++)
         {
            for (int c=0; c<sdim; ++c)
            {
               if (dim == 2)
               {
                  (*x[c])[i] = grid(i, j, c);
               }
               else
               {
                  (*x[c])[i] = grid(i, j, c + (k * sdim));
               }
            }
         }

         const bool reuse_factorization = j > 0 || k > 0;
         mesh.NURBSext->GetKnotVector(0)->GetInterpolant(x, ugrid[0],
                                                         reuse_factorization);

         for (int i = 0; i < ncps[0]; i++)
         {
            const int dof = pdofs[i + (ncp0 * j) + (ncp0 * ncp1 * k)];
            for (int c=0; c<sdim; ++c)
            {
               (*mesh.GetNodes())[(sdim * dof) + c] = (*x[c])[i];
            }
         }
      }
   }

   // Resize for sweep in second direction
   for (int i = 0; i < sdim; ++i) { x[i]->SetSize(ncps[1]); }

   // Do another sweep in the second direction
   for (int k = 0; k < ncps[2]; ++k)
   {
      for (int i = 0; i < ncps[0]; i++)
      {
         for (int j = 0; j < ncps[1]; ++j)
         {
            const int dof = pdofs[i + (ncp0 * j) + (ncp0 * ncp1 * k)];
            for (int c=0; c<sdim; ++c)
            {
               (*x[c])[j] = (*mesh.GetNodes())[(sdim * dof) + c];
            }
         }

         const bool reuse_factorization = i > 0 || k > 0;
         mesh.NURBSext->GetKnotVector(1)->GetInterpolant(x, ugrid[1],
                                                         reuse_factorization);

         for (int j = 0; j < ncps[1]; ++j)
         {
            const int dof = pdofs[i + (ncp0 * j) + (ncp0 * ncp1 * k)];
            for (int c=0; c<sdim; ++c)
            {
               (*mesh.GetNodes())[(sdim * dof) + c] = (*x[c])[j];
            }
         }
      }
   }

   // Sweep in third direction
   if (dim == 3)
   {
      // Resize for sweep in third direction
      for (int i = 0; i < sdim; ++i) { x[i]->SetSize(ncps[2]); }

      // Do another sweep in the third direction
      for (int j = 0; j < ncps[1]; ++j)
      {
         for (int i = 0; i < ncps[0]; i++)
         {
            for (int k = 0; k < ncps[2]; ++k)
            {
               const int dof = pdofs[i + (ncp0 * j) + (ncp0 * ncp1 * k)];
               for (int c=0; c<sdim; ++c)
               {
                  (*x[c])[k] = (*mesh.GetNodes())[(sdim * dof) + c];
               }
            }

            const bool reuse_factorization = i > 0 || j > 0;
            mesh.NURBSext->GetKnotVector(2)->GetInterpolant(x, ugrid[2],
                                                            reuse_factorization);

            for (int k = 0; k < ncps[2]; ++k)
            {
               const int dof = pdofs[i + (ncp0 * j) + (ncp0 * ncp1 * k)];
               for (int c=0; c<sdim; ++c)
               {
                  (*mesh.GetNodes())[(sdim * dof) + c] = (*x[c])[k];
               }
            }
         }
      }
   }

   for (auto p : x) { delete p; }
}

// For a given side of a single quadrilateral patch in 2D, solve for control
// points only on that side to interpolate a 1D side grid extracted from the
// input 2D `grid`. Mesh `mesh0_` has a single patch and element in 2D, to be
// refined to match the number of elements in `mesh`.
void SolveBoundarySegment(const Mesh &mesh0_, int ned, std::array<int, 3> nel,
                          const Array3D<real_t> &grid, int dir, int side,
                          Mesh &mesh)
{
   Mesh mesh0(mesh0_); // Deep copy to be modified
   MFEM_VERIFY(mesh0.GetNE() == 1, "");

   const int sdim = mesh0.SpaceDimension();
   MFEM_VERIFY(grid.GetSize3() == sdim, "");

   const int odir = 1 - dir;
   const int ncp = (nel[dir] * ned) + 1;
   const int ncp_o = (nel[odir] * ned) + 1;

   // Set the grid points for this boundary segment, omitting the endpoints.
   std::vector<Vector> bgrid(ncp - 2);

   const int sideIndex = (side == 0) ? 0 : ncp_o - 1;

   for (int i = 1; i < ncp-1; ++i) // No endpoints
   {
      bgrid[i - 1].SetSize(sdim);
      const int ig = (dir == 0) ? i : sideIndex;
      const int jg = (dir == 0) ? sideIndex : i;
      for (int k=0; k<sdim; ++k)
      {
         bgrid[i - 1][k] = grid(ig, jg, k);
      }
   }

   // For each interior grid point on this boundary segment, find the
   // corresponding knot parameters (reference coordinates).

   IntegrationPoint ip;
   ip.Init(0);

   auto ComputePointOnSegment0 = [&](real_t u, Vector &p)
   {
      if (dir == 0)
      {
         ip.Set2(u, (real_t) side);
      }
      else
      {
         ip.Set2((real_t) side, u);
      }

      constexpr int elem = 0;

      mesh0.GetNodes()->GetVectorValue(elem, ip, p);
   };

   auto FindPointOnSegment = [&](const Vector &p)
   {
      // Use bisection.
      real_t u0 = 0.0;
      real_t u1 = 1.0;

      constexpr real_t conv_tol = 1.0e-12;

      Vector v2(sdim);
      Vector v0(sdim);
      Vector tp(sdim);
      Vector tv(sdim);

      ComputePointOnSegment0(0.0, v0);

      for (int j=0; j<sdim; ++j)
      {
         tp[j] = p[j] - v0[j];
      }

      if (tp.Norml2() < 1.0e-6)
      {
         return (real_t) 0.0;
      }

      int iter = 0;
      while (iter < 200)
      {
         iter++;

         const real_t um = 0.5 * (u0 + u1);

         ComputePointOnSegment0(um, v2);

         // Check whether v2 is before or after p. This method assumes the
         // curvature is limited.
         // TODO: can the implementation be generalized to avoid this assumption?

         for (int j=0; j<sdim; ++j)
         {
            tv[j] = v2[j] - p[j];
         }

         const bool past = tv * tp > 0.0;

         if (past)
         {
            u1 = um;
         }
         else
         {
            u0 = um;
         }

         MFEM_VERIFY(u0 < u1, "");
         if (u1 - u0 < conv_tol)
         {
            break;
         }
      }

      MFEM_VERIFY(0.0 < u1 - u0 && u1 - u0 < conv_tol, "");
      return u0;
   };

   Vector knots(nel[dir] - 1);
   for (int i=0; i<nel[dir] - 1; ++i)
   {
      const int cp = (i + 1) * ned;
      knots[i] = FindPointOnSegment(bgrid[cp - 1]);
   }

   // Insert the knots into the single-element patch.
   Array<Vector*> kv(2);

   Vector knotsEmpty(0);
   kv[dir] = &knots;
   kv[odir] = &knotsEmpty;

   const int degree = mesh0.NURBSext->GetOrder();
   for (int i=0; i<degree; ++i) { mesh0.KnotInsert(kv); }

   // Extract the weights.
   mesh0.NURBSext->ConvertToPatches(*mesh0.GetNodes());
   mesh.NURBSext->ConvertToPatches(*mesh.GetNodes());

   Array<NURBSPatch*> mpatch, patch;
   mesh.NURBSext->GetPatches(mpatch);
   MFEM_VERIFY(mpatch.Size() == 1, "");
   patch.SetSize(1);
   patch[0] = new NURBSPatch(*mpatch[0]); // Deep copy
   patch[0]->DivideOutWeights();

   Array<NURBSPatch*> patch0;
   mesh0.NURBSext->GetPatches(patch0);
   MFEM_VERIFY(patch0.Size() == 1, "");

   const int ncp0 = degree + 1;

   for (int i=0; i<ncp; ++i)
   {
      // Loop over coordinates for 2D control point, with weight included.
      for (int k = 0; k < sdim + 1; ++k)
      {
         const int j = side * (ncp_o - 1);
         if (dir == 0)
         {
            const real_t w = (*patch0[0])(i, side * (ncp0 - 1), sdim);
            if (k < sdim)
            {
               (*patch[0])(i, j, k) = (*patch0[0])(i, side * (ncp0 - 1), k) / w;
            }
            else
            {
               (*patch[0])(i, j, k) = (*patch0[0])(i, side * (ncp0 - 1), k);
            }
         }
         else
         {
            const real_t w = (*patch0[0])(side * (ncp0 - 1), i, sdim);
            if (k < sdim)
            {
               (*patch[0])(j, i, k) = (*patch0[0])(side * (ncp0 - 1), i, k) / w;
            }
            else
            {
               (*patch[0])(j, i, k) = (*patch0[0])(side * (ncp0 - 1), i, k);
            }
         }
      }
   }

   // NOTE: this function scales by w, so we had to divide by w above in setting patch[0].
   mesh.NURBSext->SetPatchControlPoints(0, *patch[0]);
   delete patch[0];

   mesh.NURBSext->SetKnotsFromPatches();
   mesh.NURBSext->SetCoordsFromPatches(*mesh.GetNodes(), sdim);
   mesh.GetNodes()->FESpace()->Update();
}

Mesh GetPatchMesh(int p, int dim, int sdim, int degree, int ncp,
                  const Array3D<double> &patchCP);

void SolvePhysicalGridBdry(Mesh &mesh, const Mesh &mesh0, int patchIndex,
                           const Array3D<double> &coarsePatchCP, int order,
                           int ned, std::array<int, 3> nel,
                           const Array3D<real_t> &grid, bool sweep1D);

// For a given side of a single hexahedral patch in 3D, solve for control points
// only on that side to interpolate a 2D side grid extracted from the input 3D
// point `grid`. Mesh `mesh0_` has a single patch and element in 2D, to be
// refined to match the number of elements in `mesh`.
void SolveBoundaryFace(const Mesh &mesh0_, int patchIndex,
                       const Array3D<double> &coarsePatchCP, int order, int ned,
                       std::array<int, 3> nel, const Array3D<real_t> &grid,
                       int dir, int side, bool sweep1D, Mesh &mesh)
{
   const int sdim = mesh.SpaceDimension(); // Space dimension

   Mesh mesh0(mesh0_); // Deep copy to be modified
   MFEM_VERIFY(mesh0.GetNE() == 1 && sdim == 3, "");

   // Extract a 2D mesh in 3D space for the face; extract the 2D grid for the face;
   // solve for the physical spacing on the 2D face; copy the DOFs
   // for the face to the 3D mesh.

   const int ncp0 = order + 1;

   Array3D<double> facePatchCP0(1, ncp0 * ncp0, 4);
   Array3D<double> facePatchCP(1, ncp0 * ncp0, 4);

   constexpr int p = 0; // Working with 1 patch

   std::array<int, 3> c;

   c[dir] = side * (ncp0 - 1);

   std::array<int, 2> odir, ncp;
   std::array<int, 3> nelFace = {0, 0, 0};
   std::array<int, 3> ncp3D;

   {
      int dcnt = 0;
      for (int i=0; i<3; ++i)
      {
         if (i != dir)
         {
            nelFace[dcnt] = nel[i];
            odir[dcnt++] = i;
         }

         ncp3D[i] = (nel[i] * ned) + 1;
      }

      MFEM_VERIFY(dcnt == 2, "");
   }

   for (int i=0; i<2; ++i)
   {
      ncp[i] = (nelFace[i] * ned) + 1;
   }

   const int ncp_dir = (nel[dir] * ned) + 1;

   for (int j = 0; j < ncp0; ++j)
   {
      c[odir[1]] = j;
      for (int i = 0; i < ncp0; ++i)
      {
         c[odir[0]] = i;

         const int cp = c[0] + (ncp0 * c[1]) + (ncp0 * ncp0 * c[2]);

         // Loop over control point coordinates, including weight
         for (int k=0; k<4; ++k)
         {
            facePatchCP0(p, i + (ncp0 * j), k) = coarsePatchCP(patchIndex, cp, k);
         }
      }
   }

   Mesh faceMesh0 = GetPatchMesh(p, 2, 3, order, ncp0, facePatchCP0);
   Mesh faceMesh(faceMesh0);

   {
      // Refine faceMesh to match the number of elements and control points on
      // the face of `mesh`.
      MFEM_VERIFY(faceMesh.NURBSext->GetNKV() == 2, "");
      Array<Vector*> knots(2);

      for (int i=0; i<2; ++i)
      {
         knots[i] = new Vector;
         if (nelFace[i] > 1)
         {
            faceMesh.NURBSext->GetKnotVector(i)->UniformRefinement(*knots[i],
                                                                   nelFace[i]);
         }
      }

      const int degree = faceMesh.NURBSext->GetOrder();
      for (int i=0; i<degree; ++i)
      {
         faceMesh.KnotInsert(knots);
      }

      for (int i=0; i<2; ++i)
      {
         MFEM_VERIFY(faceMesh.NURBSext->GetKnotVector(i)->GetNCP() == ncp[i],
                     "");
         MFEM_VERIFY(faceMesh.NURBSext->GetKnotVector(i)->GetNE() == nelFace[i],
                     "");
      }
   }

   // Refine faceMesh and set CP from `mesh`

   Array3D<real_t> faceGrid(ncp[0], ncp[1], 3);

   c[dir] = side * (ncp_dir - 1);
   for (int i=0; i<ncp[0]; ++i)
   {
      c[odir[0]] = i;
      for (int j=0; j<ncp[1]; ++j)
      {
         c[odir[1]] = j;
         for (int k=0; k<3; ++k)
         {
            faceGrid(i, j, k) = grid(c[0], c[1], k + (3 * c[2]));
         }
      }
   }

   SolvePhysicalGridBdry(faceMesh, faceMesh0, p, facePatchCP0, order, ned,
                         nelFace, faceGrid, sweep1D);

   // Copy control points from physically spaced faceMesh to mesh.

   Array<int> pdofs;
   mesh.NURBSext->GetPatchDofs(0, pdofs); // Assuming 1 patch

   Array<int> fdofs;
   faceMesh.NURBSext->GetPatchDofs(0, fdofs); // Assuming 1 patch

   MFEM_VERIFY(pdofs.Size() == ncp3D[0] * ncp3D[1] * ncp3D[2], "");
   MFEM_VERIFY(pdofs.Size() == ncp[0] * ncp[1] * ncp_dir, "");
   MFEM_VERIFY(fdofs.Size() == ncp[0] * ncp[1], "");

   c[dir] = side * (ncp_dir - 1);
   for (int i=0; i<ncp[0]; ++i)
   {
      c[odir[0]] = i;
      for (int j=0; j<ncp[1]; ++j)
      {
         c[odir[1]] = j;

         const int dof = pdofs[c[0] + (ncp3D[0] * c[1]) + (ncp3D[0] * ncp3D[1] * c[2])];
         const int fdof = fdofs[i + (ncp[0] * j)];
         for (int l=0; l<sdim; ++l)
         {
            (*mesh.GetNodes())[(sdim * dof) + l] = (*faceMesh.GetNodes())[(sdim * fdof) +
                                                                          l];
         }
      }
   }
}

// Solve for new control points in `mesh` (assuming a single patch) to
// interpolate the input point grid on the patch boundary.
void SolvePhysicalGridBdry(Mesh &mesh, const Mesh &mesh0, int patchIndex,
                           const Array3D<double> &coarsePatchCP, int order,
                           int ned, std::array<int, 3> nel,
                           const Array3D<real_t> &grid, bool sweep1D)
{
   const int dim = mesh0.Dimension(); // Reference space dimension

   if (dim == 2)
   {
      SolveBoundarySegment(mesh0, ned, nel, grid, 0, 0, mesh);
      SolveBoundarySegment(mesh0, ned, nel, grid, 0, 1, mesh);
      SolveBoundarySegment(mesh0, ned, nel, grid, 1, 0, mesh);
      SolveBoundarySegment(mesh0, ned, nel, grid, 1, 1, mesh);
   }
   else // dim == 3
   {
      // Note that the boundary edges of the patch are solved twice, which is an
      // insignificant inefficiency.
      for (int dir=0; dir<3; ++dir)
         for (int side=0; side<2; ++side)
            SolveBoundaryFace(mesh0, patchIndex, coarsePatchCP, order, ned, nel,
                              grid, dir, side, sweep1D, mesh);
   }

   if (sweep1D)
   {
      SolvePhysicalGridSweep(grid, ned, nel, mesh);
   }
   else
   {
      SolvePhysicalGridInterior(grid, ned, nel, mesh);
   }
}

// For the input mesh in 2D or 3D with a single patch, compute a grid of physical
// points such that the relative arc length of each edge in each direction
// approximately matches the spacing function `sf` in that direction. The number
// of DOFs in 1D per element minus 1 is given by ned, and the number of elements
// in each direction of the patch is in nel.
void GetSpacedPatchGrid(Mesh &mesh,
                        const std::vector<const SpacingFunction*> &sf,
                        int ned, std::array<int, 3> nel, Array3D<real_t> &grid)
{
   Array3D<real_t> ugrid(grid.GetSize1(), grid.GetSize2(), grid.GetSize3());

   const int dim = mesh.Dimension();
   MFEM_VERIFY(mesh.GetNodes() && (dim == 2 || dim == 3), "");

   FiniteElementCollection* fec = mesh.GetNodes()->OwnFEC();
   FiniteElementSpace fespace(&mesh, fec);
   GridFunction x(&fespace);

   Array<int> pdofs;
   mesh.NURBSext->GetPatchDofs(0, pdofs);

   const IntegrationRule *ir = &IntRules.Get(Geometry::SEGMENT, (4 * ned) + 1);

   IntegrationPoint ip;
   ip.Init(0);

   // ncp = (ne * ned) + 1 where ned is the number of DOFs per element minus 1.
   const int ncp0 = (nel[0] * ned) + 1;
   const int ncp1 = (nel[1] * ned) + 1;
   const int ncp2 = dim == 2 ? 1 : (nel[2] * ned) + 1;

   const std::array<int, 3> ncp = {ncp0, ncp1, ncp2};

   MFEM_VERIFY(grid.GetSize1() == ncp0 && grid.GetSize2() == ncp1, "");

   MFEM_VERIFY(x.Size() == ncp0 * ncp1 * ncp2, "");
   MFEM_VERIFY(dim * x.Size() == mesh.GetNodes()->Size(), "");
   MFEM_VERIFY(pdofs.Size() == ncp0 * ncp1 * ncp2, "");

   // Keep the grid samples aligned with the interpolation collocation points.
   // In FindMaxima mode the active element side of a shared knot matters.
   std::vector<Vector> sample_xi(3);
   std::vector<Vector> sample_u(3);
   std::vector<Array<int>> sample_el(3);

   for (int d=0; d<dim; ++d)
   {
      sample_xi[d].SetSize(ncp[d]);
      sample_u[d].SetSize(ncp[d]);
      sample_el[d].SetSize(ncp[d]);

      const KnotVector *kv = mesh.NURBSext->GetKnotVector(d);
      kv->GetDemko(sample_u[d]);
      MFEM_VERIFY(sample_u[d].Size() == ncp[d], "");
      sample_xi[d].SetSize(ncp[d]);

      for (int cp=0; cp<ncp[d]; ++cp)
      {
         const int knot_span = kv->GetSpan(sample_u[d][cp]);
         sample_xi[d][cp] = kv->GetRefPoint(sample_u[d][cp], knot_span);
         sample_el[d][cp] = kv->ElementIndex(knot_span);
      }

      for (int cp=0; cp<ncp[d]; ++cp)
      {
         MFEM_VERIFY(0 <= sample_el[d][cp] && sample_el[d][cp] < nel[d], "");
         MFEM_VERIFY(0.0 <= sample_xi[d][cp] && sample_xi[d][cp] <= 1.0, "");
         sample_u[d][cp] = (sample_el[d][cp] + sample_xi[d][cp]) /
                           ((real_t) nel[d]);
      }
   }

   for (int i=0; i<ncp0; ++i)
   {
      for (int j=0; j<ncp1; ++j)
      {
         for (int k=0; k<ncp2; ++k)
         {
            const int kos = k*dim;

            ugrid(i,j,0 + kos) = sample_u[0][i];
            ugrid(i,j,1 + kos) = sample_u[1][j];

            if (dim == 3)
            {
               ugrid(i,j,2 + kos) = sample_u[2][k];
            }
         }
      }
   }

   Vector v2(dim);

   // Parameter within element
   auto ElementParameter = [](real_t u_g, int ne, int &ei)
   {
      const int ei_ = u_g * ne;
      ei = std::min(ei_, ne - 1);
      real_t u = std::max(u_g - (ei / (real_t) ne), (real_t) 0.0);
      u *= (real_t) ne;
      u = std::min(u, (real_t) 1.0);
      return u;
   };

   // Adjust arc lengths in the given direction `dir`.
   auto AdjustGridArcLengths = [&](int dir)
   {
      std::array<int, 2> odir;  // Other, orthogonal directions
      {
         int dcnt = 0;
         for (int i=0; i<3; ++i)
         {
            if (i != dir)
            {
               odir[dcnt++] = i;
            }
         }
         MFEM_VERIFY(dcnt == 2, "Invalid direction");
      }

      auto ElementIndex = [&nel, &dir](int i, int j, int k)
      {
         if (dir == 0)
         {
            return i + (j * nel[0]) + (k * nel[0] * nel[1]);
         }
         else if (dir == 1)
         {
            return j + (i * nel[0]) + (k * nel[0] * nel[1]);
         }
         else // (dir == 2)
         {
            return j + (k * nel[0]) + (i * nel[0] * nel[1]);
         }
      };

      auto SetIP = [&dim, &dir, &ip](real_t v, real_t x, real_t y)
      {
         if (dim == 2)
         {
            if (dir == 1)
            {
               ip.Set2(x, v);
            }
            else
            {
               ip.Set2(v, x);
            }
         }
         else // dim == 3
         {
            if (dir == 0)
            {
               ip.Set3(v, x, y);
            }
            else if (dir == 1)
            {
               ip.Set3(x, v, y);
            }
            else // dir == 2
            {
               ip.Set3(x, y, v);
            }
         }
      };

      auto ArcLengthElement = [&](int ei, int ej, int ek, real_t x, real_t y,
                                  real_t u)
      {
         const int el = ElementIndex(ei, ej, ek);

         ElementTransformation *Tr =
            mesh.GetNodes()->FESpace()->GetElementTransformation(el);
         DenseMatrix grad;

         real_t arc = 0.0;
         for (auto irp : *ir)
         {
            SetIP(irp.x * u, x, y);
            Tr->SetIntPoint(&ip);
            mesh.GetNodes()->GetVectorGradientHat(*Tr, grad);

            MFEM_VERIFY(grad.NumRows() == dim, "");
            MFEM_VERIFY(grad.NumCols() == dim, "");
            v2.SetSize(dim);
            const int col = dir;
            for (int i=0; i<dim; ++i)
            {
               v2[i] = grad(i, col);
            }

            arc += v2.Norml2() * irp.weight * u;
         }

         return arc;
      };

      for (int dof_j=0; dof_j<ncp[odir[0]]; ++dof_j)
      {
         const int ej = sample_el[odir[0]][dof_j];
         const real_t xr = sample_xi[odir[0]][dof_j];

         for (int dof_k=0; dof_k<ncp[odir[1]]; ++dof_k)
         {
            const int ek = dim == 2 ? 0 : sample_el[odir[1]][dof_k];
            const real_t yr = dim == 2 ? 0.0 : sample_xi[odir[1]][dof_k];

            real_t totalArc = 0.0;
            Vector arcs(nel[dir]);
            Vector arcL(nel[dir]);
            for (int ei=0; ei<nel[dir]; ++ei)
            {
               arcs[ei] = totalArc;
               arcL[ei] = ArcLengthElement(ei, ej, ek, xr, yr, 1.0);
               totalArc += arcL[ei];
            } // ei

            auto ArcLength = [&](real_t u_p)
            {
               int ei;
               const real_t u = ElementParameter(u_p, nel[dir], ei);
               real_t a = arcs[ei]; // Arc length at start of this element.
               a += ArcLengthElement(ei, ej, ek, xr, yr, u);
               return a;
            }; // ArcLength

            const SpacingFunction *spacing_dir = sf[dir];
            std::unique_ptr<SpacingFunction> spacing =
               spacing_dir ? spacing_dir->Clone() :
               std::make_unique<UniformSpacingFunction>(nel[dir]);

            spacing->SetSize(nel[dir]);

            Vector spacing_prefix(nel[dir] + 1);
            spacing_prefix[0] = 0.0;
            for (int e=0; e<nel[dir]; ++e)
            {
               spacing_prefix[e + 1] = spacing_prefix[e] + spacing->Eval(e);
            }

            for (int l=1; l<ncp[dir]-1; ++l)
            {
               // Set a_l by using spacing function for non-uniform grids.
               const int el_l = sample_el[dir][l];
               const real_t s_l = spacing->Eval(el_l);
               const real_t a_l = (spacing_prefix[el_l] +
                                   (sample_xi[dir][l] * s_l)) * totalArc;

               // Find the reference point with arc length equal to a_l.
               // Use bisection method.
               real_t u0 = 0.0;
               real_t u1 = 1.0;

               constexpr real_t conv_tol = 1.0e-12;

               int iter = 0;
               while (iter < 200)
               {
                  iter++;

                  const real_t um = 0.5 * (u0 + u1);
                  const real_t a_i = ArcLength(um);
                  if (a_i < a_l)
                  {
                     u0 = um;
                  }
                  else
                  {
                     u1 = um;
                  }

                  MFEM_VERIFY(u0 < u1, "");
                  if (u1 - u0 < conv_tol)
                  {
                     break;
                  }
               }

               MFEM_VERIFY(0.0 < u1 - u0 && u1 - u0 < conv_tol, "");

               // Update ugrid. grid(i,j,*) follows local patch directions 0 and 1.
               if (dir == 0)
               {
                  ugrid(l, dof_j, 0 + (dof_k * dim)) = u0;
               }
               else if (dir == 1)
               {
                  ugrid(dof_j, l, 1 + (dof_k * dim)) = u0;
               }
               else // dir == 2
               {
                  ugrid(dof_j, dof_k, 2 + (l * dim)) = u0;
               }
            } // l
         } // dof_k
      } // dof_j
   }; // AdjustGridArcLengths

   for (int i=0; i<dim; ++i)
   {
      for (int d=0; d<dim; ++d)
      {
         AdjustGridArcLengths(d);
      }
   }

   // Set physical grid from the reference ugrid.
   const int nk = grid.GetSize3() / dim;
   for (int i=0; i<grid.GetSize1(); ++i)
   {
      for (int j=0; j<grid.GetSize2(); ++j)
      {
         for (int k=0; k<nk; ++k)
         {
            const int kos = k * dim;
            const real_t u0 = ugrid(i,j,0 + kos);
            const real_t u1 = ugrid(i,j,1 + kos);
            const real_t u2 = dim == 3 ? ugrid(i,j,2 + kos) : 0.0;

            std::array<real_t, 3> u = {u0, u1, u2};
            std::array<real_t, 3> u_e = {0.0, 0.0, 0.0};
            std::array<int, 3> el = {0, 0, 0};
            for (int l=0; l<dim; ++l)
            {
               u_e[l] = ElementParameter(u[l], nel[l], el[l]);
            }

            if (dim == 2)
            {
               ip.Set2(u_e[0], u_e[1]);
            }
            else
            {
               ip.Set3(u_e[0], u_e[1], u_e[2]);
            }

            const int el_ij = el[0] + (el[1] * nel[0]) + (el[2] * nel[0] * nel[1]);
            mesh.GetNodes()->GetVectorValue(el_ij, ip, v2);
            for (int l=0; l<dim; ++l)
            {
               grid(i, j, l + kos) = v2[l];
            }
         }
      }
   }
}

// Adjust a single patch to have relative physical spacing given by knots. The
// number of DOFs in 1D per element minus 1 is given by ned, and the number of
// elements in each direction of the patch is in nel.
void PatchPhysicalSpacing(NURBSPatch *patch, int patchIndex,
                          const Array3D<double> &coarsePatchCP,
                          const Mesh &mesh0, int order, int ned,
                          std::array<int, 3> nel, bool sweep1D)
{
   const int dim = mesh0.Dimension(); // Reference space dimension
   const int sdim = mesh0.SpaceDimension(); // Space dimension

   MFEM_VERIFY(patch->GetNKV() == dim, "");
   MFEM_VERIFY(patch->GetKV(0)->GetNE() == nel[0], "");
   MFEM_VERIFY(patch->GetKV(1)->GetNE() == nel[1], "");
   MFEM_VERIFY(dim == 2 || patch->GetKV(2)->GetNE() == nel[2], "");

   Array<NURBSPatch*> patches;
   patches.Append(patch);
   Mesh patch_topology = dim == 2 ?
                         Mesh::MakeCartesian2D(1, 1, Element::Type::QUADRILATERAL) :
                         Mesh::MakeCartesian3D(1, 1, 1, Element::Type::HEXAHEDRON);

   NURBSExtension ne(&patch_topology, patches);
   Mesh mesh(ne);

   const int ncpx = (nel[0] * ned) + 1;
   const int ncpy = (nel[1] * ned) + 1;
   const int ncpz = dim == 2 ? 1 : (nel[2] * ned) + 1;

   Array3D<real_t> grid(ncpx, ncpy, ncpz * dim);

   std::vector<const SpacingFunction*> sf(dim);
   for (int i=0; i<dim; ++i)
   {
      sf[i] = patch->GetKV(i)->spacing.get();
   }

   GetSpacedPatchGrid(mesh, sf, ned, nel, grid);
   SolvePhysicalGridBdry(mesh, mesh0, patchIndex, coarsePatchCP, order, ned,
                         nel, grid, sweep1D);

   // Copy control points from mesh to patch.

   Array<int> pdofs;
   mesh.NURBSext->GetPatchDofs(0, pdofs);

   const Vector &mweights = mesh.NURBSext->GetWeights();

   for (int i=0; i<ncpx; ++i)
      for (int j=0; j<ncpy; ++j)
         for (int k=0; k<ncpz; ++k)
         {
            const int dof = pdofs[i + (ncpx * j) + (ncpx * ncpy * k)];
            const real_t w = mweights[dof];
            for (int l=0; l<sdim; ++l)
            {
               if (dim == 2)
               {
                  (*patch)(i,j,l) = w * (*mesh.GetNodes())[(sdim * dof) + l];
               }
               else
               {
                  (*patch)(i,j,k,l) = w * (*mesh.GetNodes())[(sdim * dof) + l];
               }
            }

            if (dim == 2)
            {
               (*patch)(i,j,sdim) = w;
            }
            else
            {
               (*patch)(i,j,k,sdim) = w;
            }
         }
}

} // namespace mfem
