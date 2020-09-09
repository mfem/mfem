// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
//             ----------------------------------------------
//             Polar NC: Generate polar non-conforming meshes
//             ----------------------------------------------
//
// This miniapp generates a circular sector mesh that consist of quadrilaterals
// and triangles of similar sizes. The 3D version of the mesh is made of prisms
// and tetrahedra. The mesh is non-conforming by design, and can optionally be
// made curvilinear. The elements are ordered along a space-filling curve by
// default, which makes the mesh ready for parallel non-conforming AMR in MFEM.
//
// The implementation also demonstrates how to initialize a non-conforming mesh
// on the fly by marking hanging nodes with Mesh::AddVertexParents.
//
// Compile with: make polar-nc
//
// Sample runs:  polar-nc --radius 1 --nsteps 10
//               polar-nc --aspect 2
//               polar-nc --dim 3 --order 4

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;
using namespace std;


struct Params2
{
   double r, dr;
   double a, da;

   Params2() = default;
   Params2(double r0, double r1, double a0, double a1)
      : r(r0), dr(r1 - r0), a(a0), da(a1 - a0) {}
};

Mesh* Make2D(int nsteps, double rstep, double phi, double aspect, int order,
             bool sfc)
{
   Mesh *mesh = new Mesh(2, 0, 0);

   int origin = mesh->AddVertex(0.0, 0.0);

   // n is the number of steps in the polar direction
   int n = 1;
   while (phi * rstep/2 / n * aspect > rstep) { n++; }

   double r = rstep;
   int first = mesh->AddVertex(r, 0.0);

   Array<Params2> params;
   Array<Pair<int, int>> blocks;

   // create triangles around the origin
   double prev_alpha = 0.0;
   for (int i = 0; i < n; i++)
   {
      double alpha = phi * (i+1) / n;
      mesh->AddVertex(r*cos(alpha), r*sin(alpha));
      mesh->AddTriangle(origin, first+i, first+i+1);

      params.Append(Params2(0, r, prev_alpha, alpha));
      prev_alpha = alpha;
   }

   mesh->AddBdrSegment(origin, first, 1);
   mesh->AddBdrSegment(first+n, origin, 2);

   for (int k = 1; k < nsteps; k++)
   {
      // m is the number of polar steps of the previous row
      int m = n;
      int prev_first = first;

      double prev_r = r;
      r += rstep;

      if (phi * (r + prev_r)/2 / n * aspect < rstep * sqrt(2))
      {
         if (k == 1) { blocks.Append(Pair<int, int>(mesh->GetNE(), n)); }

         first = mesh->AddVertex(r, 0.0);
         mesh->AddBdrSegment(prev_first, first, 1);

         // create a row of quads, same number as in previous row
         prev_alpha = 0.0;
         for (int i = 0; i < n; i++)
         {
            double alpha = phi * (i+1) / n;
            mesh->AddVertex(r*cos(alpha), r*sin(alpha));
            mesh->AddQuad(prev_first+i, first+i, first+i+1, prev_first+i+1);

            params.Append(Params2(prev_r, r, prev_alpha, alpha));
            prev_alpha = alpha;
         }

         mesh->AddBdrSegment(first+n, prev_first+n, 2);
      }
      else // we need to double the number of elements per row
      {
         n *= 2;

         blocks.Append(Pair<int, int>(mesh->GetNE(), n));

         // first create hanging vertices
         int hang;
         for (int i = 0; i < m; i++)
         {
            double alpha = phi * (2*i+1) / n;
            int index = mesh->AddVertex(prev_r*cos(alpha), prev_r*sin(alpha));
            mesh->AddVertexParents(index, prev_first+i, prev_first+i+1);
            if (!i) { hang = index; }
         }

         first = mesh->AddVertex(r, 0.0);
         int a = prev_first, b = first;

         mesh->AddBdrSegment(a, b, 1);

         // create a row of quad pairs
         prev_alpha = 0.0;
         for (int i = 0; i < m; i++)
         {
            int c = hang+i, e = a+1;

            double alpha_half = phi * (2*i+1) / n;
            int d = mesh->AddVertex(r*cos(alpha_half), r*sin(alpha_half));

            double alpha = phi * (2*i+2) / n;
            int f = mesh->AddVertex(r*cos(alpha), r*sin(alpha));

            mesh->AddQuad(a, b, d, c);
            mesh->AddQuad(c, d, f, e);

            a = e, b = f;

            params.Append(Params2(prev_r, r, prev_alpha, alpha_half));
            params.Append(Params2(prev_r, r, alpha_half, alpha));
            prev_alpha = alpha;
         }

         mesh->AddBdrSegment(b, a, 2);
      }
   }

   for (int i = 0; i < n; i++)
   {
      mesh->AddBdrSegment(first+i, first+i+1, 3);
   }

   // reorder blocks of elements with Grid SFC ordering
   if (sfc)
   {
      blocks.Append(Pair<int, int>(mesh->GetNE(), 0));

      Array<Params2> new_params(params.Size());

      Array<int> ordering(mesh->GetNE());
      for (int i = 0; i < blocks[0].one; i++)
      {
         ordering[i] = i;
         new_params[i] = params[i];
      }

      Array<int> coords;
      for (int i = 0; i < blocks.Size()-1; i++)
      {
         int beg = blocks[i].one;
         int width = blocks[i].two;
         int height = (blocks[i+1].one - blocks[i].one) / width;

         NCMesh::GridSfcOrdering2D(width, height, coords);

         for (int j = 0, k = 0; j < coords.Size(); k++, j += 2)
         {
            int sfc = ((i & 1) ? coords[j] : (width-1 - coords[j]))
                      + coords[j+1]*width;
            int old_index = beg + sfc;

            ordering[old_index] = beg + k;
            new_params[beg + k] = params[old_index];
         }
      }

      mesh->ReorderElements(ordering, false);

      mfem::Swap(params, new_params);
   }

   // create high-order curvature
   if (order > 1)
   {
      mesh->SetCurvature(order);

      GridFunction *nodes = mesh->GetNodes();
      const FiniteElementSpace *fes = mesh->GetNodalFESpace();

      Array<int> dofs;
      MFEM_ASSERT(params.Size() == mesh->GetNE(), "");

      for (int i = 0; i < mesh->GetNE(); i++)
      {
         const Params2 &par = params[i];
         const IntegrationRule &ir = fes->GetFE(i)->GetNodes();
         Geometry::Type geom = mesh->GetElementBaseGeometry(i);
         fes->GetElementDofs(i, dofs);

         for (int j = 0; j < dofs.Size(); j++)
         {
            double r, a;
            if (geom == Geometry::SQUARE)
            {
               r = par.r + ir[j].x * par.dr;
               a = par.a + ir[j].y * par.da;
            }
            else
            {
               double rr = ir[j].x + ir[j].y;
               if (std::abs(rr) < 1e-12) { continue; }
               r = par.r + rr * par.dr;
               a = par.a + ir[j].y/rr * par.da;
            }
            (*nodes)(fes->DofToVDof(dofs[j], 0)) = r*cos(a);
            (*nodes)(fes->DofToVDof(dofs[j], 1)) = r*sin(a);
         }
      }

      nodes->RestrictConforming();
   }

   mesh->FinalizeMesh();

   return mesh;
}


const double pi2 = M_PI / 2;

struct Params3
{
   double r, dr;
   double u1, u2, u3;
   double v1, v2, v3;

   Params3() = default;
   Params3(double r0, double r1,
           double u1, double v1, double u2, double v2, double u3, double v3)
      : r(r0), dr(r1 - r0), u1(u1), u2(u2), u3(u3), v1(v1), v2(v2), v3(v3) {}
};

struct Vert : public Hashed2
{
   int id;
};

int GetMidVertex(int v1, int v2, double r, double u, double v, bool hanging,
                 Mesh *mesh, HashTable<Vert> &hash)
{
   int vmid = hash.FindId(v1, v2);
   if (vmid < 0)
   {
      vmid = hash.GetId(v1, v2);

      double w = 1.0 - u - v;
      double q = r / sqrt(u*u + v*v + w*w);
      int index = mesh->AddVertex(u*q, v*q, w*q);

      if (hanging) { mesh->AddVertexParents(index, v1, v2); }

      hash[vmid].id = index;
   }
   return hash[vmid].id;
}

void MakeLayer(int vx1, int vy1, int vz1, int vx2, int vy2, int vz2, int level,
               double r1, double r2, double u1, double v1, double u2, double v2,
               double u3, double v3, bool bnd1, bool bnd2, bool bnd3, bool bnd4,
               Mesh *mesh, HashTable<Vert> &hash, Array<Params3> &params)
{
   if (!level)
   {
      mesh->AddWedge(vx1, vy1, vz1, vx2, vy2, vz2);

      if (bnd1) { mesh->AddBdrQuad(vx1, vy1, vy2, vx2, 1); }
      if (bnd2) { mesh->AddBdrQuad(vy1, vz1, vz2, vy2, 2); }
      if (bnd3) { mesh->AddBdrQuad(vz1, vx1, vx2, vz2, 3); }
      if (bnd4) { mesh->AddBdrTriangle(vx2, vy2, vz2, 4); }

      params.Append(Params3(r1, r2, u1, v1, u2, v2, u3, v3));
   }
   else
   {
      double u12 = (u1+u2)/2, v12 = (v1+v2)/2;
      double u23 = (u2+u3)/2, v23 = (v2+v3)/2;
      double u31 = (u3+u1)/2, v31 = (v3+v1)/2;

      bool hang = (level == 1);

      int vxy1 = GetMidVertex(vx1, vy1, r1, u12, v12, hang, mesh, hash);
      int vyz1 = GetMidVertex(vy1, vz1, r1, u23, v23, hang, mesh, hash);
      int vxz1 = GetMidVertex(vx1, vz1, r1, u31, v31, hang, mesh, hash);
      int vxy2 = GetMidVertex(vx2, vy2, r2, u12, v12, false, mesh, hash);
      int vyz2 = GetMidVertex(vy2, vz2, r2, u23, v23, false, mesh, hash);
      int vxz2 = GetMidVertex(vx2, vz2, r2, u31, v31, false, mesh, hash);

      MakeLayer(vx1, vxy1, vxz1, vx2, vxy2, vxz2, level-1,
                r1, r2, u1, v1, u12, v12, u31, v31,
                bnd1, false, bnd3, bnd4, mesh, hash, params);
      MakeLayer(vxy1, vy1, vyz1, vxy2, vy2, vyz2, level-1,
                r1, r2, u12, v12, u2, v2, u23, v23,
                bnd1, bnd2, false, bnd4, mesh, hash, params);
      MakeLayer(vxz1, vyz1, vz1, vxz2, vyz2, vz2, level-1,
                r1, r2, u31, v31, u23, v23, u3, v3,
                false, bnd2, bnd3, bnd4, mesh, hash, params);
      MakeLayer(vyz1, vxz1, vxy1, vyz2, vxz2, vxy2, level-1,
                r1, r2, u23, v23, u31, v31, u12, v12,
                false, false, false, bnd4, mesh, hash, params);
   }
}

void MakeCenter(int origin, int vx, int vy, int vz, int level, double r,
                double u1, double v1, double u2, double v2, double u3, double v3,
                bool bnd1, bool bnd2, bool bnd3, bool bnd4,
                Mesh *mesh, HashTable<Vert> &hash, Array<Params3> &params)
{
   if (!level)
   {
      mesh->AddTet(origin, vx, vy, vz);

      if (bnd1) { mesh->AddBdrTriangle(0, vy, vx, 1); }
      if (bnd2) { mesh->AddBdrTriangle(0, vz, vy, 2); }
      if (bnd3) { mesh->AddBdrTriangle(0, vx, vz, 3); }
      if (bnd4) { mesh->AddBdrTriangle(vx, vy, vz, 4); }

      params.Append(Params3(0, r, u1, v1, u2, v2, u3, v3));
   }
   else
   {
      double u12 = (u1+u2)/2, v12 = (v1+v2)/2;
      double u23 = (u2+u3)/2, v23 = (v2+v3)/2;
      double u31 = (u3+u1)/2, v31 = (v3+v1)/2;

      int vxy = GetMidVertex(vx, vy, r, u12, v12, false, mesh, hash);
      int vyz = GetMidVertex(vy, vz, r, u23, v23, false, mesh, hash);
      int vxz = GetMidVertex(vx, vz, r, u31, v31, false, mesh, hash);

      MakeCenter(origin, vx, vxy, vxz, level-1, r, u1, v1, u12, v12, u31, v31,
                 bnd1, false, bnd3, bnd4, mesh, hash, params);
      MakeCenter(origin, vxy, vy, vyz, level-1, r, u12, v12, u2, v2, u23, v23,
                 bnd1, bnd2, false, bnd4, mesh, hash, params);
      MakeCenter(origin, vxz, vyz, vz, level-1, r, u31, v31, u23, v23, u3, v3,
                 false, bnd2, bnd3, bnd4, mesh, hash, params);
      MakeCenter(origin, vyz, vxz, vxy, level-1, r, u23, v23, u31, v31, u12, v12,
                 false, false, false, bnd4, mesh, hash, params);
   }
}

Mesh* Make3D(int nsteps, double rstep, double aspect, int order, bool sfc)
{
   Mesh *mesh = new Mesh(3, 0, 0);

   HashTable<Vert> hash;
   Array<Params3> params;

   int origin = mesh->AddVertex(0, 0, 0);

   double r = rstep;
   int a = mesh->AddVertex(r, 0, 0);
   int b = mesh->AddVertex(0, r, 0);
   int c = mesh->AddVertex(0, 0, r);

   int levels = 0;
   while (pi2 * rstep / (1 << levels) * aspect > rstep) { levels++; }

   MakeCenter(origin, a, b, c, levels, r, 1, 0, 0, 1, 0, 0,
              true, true, true, (nsteps == 1), mesh, hash, params);

   for (int k = 1; k < nsteps; k++)
   {
      double prev_r = r;
      r += rstep;

      if ((prev_r + rstep/2) * pi2 * aspect / (1 << levels) > rstep * sqrt(2))
      {
         levels++;
      }

      int d = mesh->AddVertex(r, 0, 0);
      int e = mesh->AddVertex(0, r, 0);
      int f = mesh->AddVertex(0, 0, r);

      MakeLayer(a, b, c, d, e, f, levels, prev_r, r,
                1, 0, 0, 1, 0, 0, true, true, true, (k == nsteps-1),
                mesh, hash, params);

      a = d;
      b = e;
      c = f;
   }

   // reorder mesh with Hilbert spatial sort
   if (sfc)
   {
      Array<int> ordering;
      mesh->GetHilbertElementOrdering(ordering);
      mesh->ReorderElements(ordering, false);

      Array<Params3> new_params(params.Size());
      for (int i = 0; i < ordering.Size(); i++)
      {
         new_params[ordering[i]] = params[i];
      }
      mfem::Swap(params, new_params);
   }

   mesh->FinalizeMesh();

   // create high-order curvature
   if (order > 1)
   {
      mesh->SetCurvature(order);

      GridFunction *nodes = mesh->GetNodes();
      const FiniteElementSpace *fes = mesh->GetNodalFESpace();

      Array<int> dofs;
      MFEM_ASSERT(params.Size() == mesh->GetNE(), "");

      for (int i = 0; i < mesh->GetNE(); i++)
      {
         const Params3 &par = params[i];
         const IntegrationRule &ir = fes->GetFE(i)->GetNodes();
         Geometry::Type geom = mesh->GetElementBaseGeometry(i);
         fes->GetElementDofs(i, dofs);

         for (int j = 0; j < dofs.Size(); j++)
         {
            const IntegrationPoint &ip = ir[j];

            double u, v, w, r;
            if (geom == Geometry::PRISM)
            {
               double l1 = 1.0 - ip.x - ip.y;
               double l2 = ip.x, l3 = ip.y;
               u = l1 * par.u1 + l2 * par.u2 + l3 * par.u3;
               v = l1 * par.v1 + l2 * par.v2 + l3 * par.v3;
               w = 1.0 - u - v;
               r = par.r + ip.z * par.dr;
            }
            else
            {
               u = ip.x * par.u1 + ip.y * par.u2 + ip.z * par.u3;
               v = ip.x * par.v1 + ip.y * par.v2 + ip.z * par.v3;
               double rr = ip.x + ip.y + ip.z;
               if (std::abs(rr) < 1e-12) { continue; }
               w = rr - u - v;
               r = par.r + rr * par.dr;
            }

            double q = r / sqrt(u*u + v*v + w*w);
            (*nodes)(fes->DofToVDof(dofs[j], 0)) = u*q;
            (*nodes)(fes->DofToVDof(dofs[j], 1)) = v*q;
            (*nodes)(fes->DofToVDof(dofs[j], 2)) = w*q;
         }
      }

      nodes->RestrictConforming();
   }

   return mesh;
}


int main(int argc, char *argv[])
{
   int dim = 2;
   double radius = 1.0;
   int nsteps = 10;
   double angle = 90;
   double aspect = 1.0;
   int order = 2;
   bool sfc = true;
   bool visualization = true;

   // parse command line
   OptionsParser args(argc, argv);
   args.AddOption(&dim, "-d", "--dim", "Mesh dimension (2 or 3).");
   args.AddOption(&radius, "-r", "--radius", "Radius of the domain.");
   args.AddOption(&nsteps, "-n", "--nsteps",
                  "Number of elements along the radial direction");
   args.AddOption(&aspect, "-a", "--aspect",
                  "Target aspect ratio of the elements.");
   args.AddOption(&angle, "-phi", "--phi", "Angular range (2D only).");
   args.AddOption(&order, "-o", "--order",
                  "Polynomial degree of mesh curvature.");
   args.AddOption(&sfc, "-sfc", "--sfc", "-no-sfc", "--no-sfc",
                  "Try to order elements along a space-filling curve.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return EXIT_FAILURE;
   }
   args.PrintOptions(cout);

   // validate options
   MFEM_VERIFY(radius > 0, "");
   MFEM_VERIFY(aspect > 0, "");
   MFEM_VERIFY(dim >= 2 && dim <= 3, "");
   MFEM_VERIFY(angle > 0 && angle < 360, "");
   MFEM_VERIFY(nsteps > 0, "");

   double phi = angle * M_PI / 180;

   // generate
   Mesh *mesh;
   if (dim == 2)
   {
      mesh = Make2D(nsteps, radius/nsteps, phi, aspect, order, sfc);
   }
   else
   {
      mesh = Make3D(nsteps, radius/nsteps, aspect, order, sfc);
   }

   // save the final mesh
   ofstream ofs("polar-nc.mesh");
   ofs.precision(8);
   mesh->Print(ofs);

   // output the mesh to GLVis
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "mesh\n" << *mesh << flush;
   }

   delete mesh;

   return EXIT_SUCCESS;
}

