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
//
//          --------------------------------------------------------
//          NURBS Surface: Interpolate a 3D Surface in a NURBS Patch
//          --------------------------------------------------------
//
// Compile with: make nurbs_surface
//
// Sample runs:  nurbs_surface -o 3 -nx 10 -ny 10 -fnx 10 -fny 10 -ex 1 -orig
//               nurbs_surface -o 3 -nx 10 -ny 10 -fnx 40 -fny 40 -ex 1
//               nurbs_surface -o 3 -nx 20 -ny 20 -fnx 10 -fny 10 -ex 1
//               nurbs_surface -o 3 -nx 10 -ny 10 -fnx 10 -fny 10 -ex 2 -orig
//               nurbs_surface -o 3 -nx 10 -ny 10 -fnx 40 -fny 40 -ex 2
//               nurbs_surface -o 3 -nx 20 -ny 20 -fnx 10 -fny 10 -ex 2
//               nurbs_surface -o 3 -nx 10 -ny 10 -fnx 10 -fny 10 -ex 3 -orig
//               nurbs_surface -o 3 -nx 10 -ny 10 -fnx 40 -fny 40 -ex 3
//               nurbs_surface -o 3 -nx 20 -ny 20 -fnx 10 -fny 10 -ex 3
//
// Description:  This example demonstrates the use of MFEM to interpolate a grid
//               of 3D points with a NURBS surface. The NURBS mesh of the
//               surface can be sampled to generate a linear mesh of arbitrary
//               resolution.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Compute error of interpolation with respect to an input grid of point data.
void CheckError(const Array3D<real_t> &a, const Array3D<real_t> &b, int c,
                int nx, int ny);

// Sample a high-order NURBS mesh to generate a fine first-order mesh.
void FineSampling(bool uniform, int nx, int ny, const Mesh &mesh,
                  const Array<int> &nks, const std::vector<Vector> &u_args,
                  Array3D<real_t> &vpos);

// Write a linear surface mesh with given vertex positions in v.
void WriteLinearMesh(int nx, int ny, const Array3D<real_t> &v,
                     const std::string &basename, bool visualization = false);

// Example data for 3D point grid on surface, given by an analytic function.
void SurfaceExample(int example, const std::vector<Vector> &ugrid,
                    Array3D<real_t> &v3D);

// Given an input grid of 3D points on a surface, this data structure computes
// a NURBS interpolation of the point grid, of given order.
class SurfaceInterpolator
{
public:
   /// Constructor for a given 2D point grid size and NURBS order.
   SurfaceInterpolator(int num_elem_x, int num_elem_y, int order,
                       bool vis = false);

   /// Create a surface interpolating the 2D grid of 3D points in @a input3D.
   void CreateSurface(const Array3D<real_t> &input3D);

   /// Sample the surface with the given grid size, storing points in
   /// @a output3D.
   void SampleSurface(int num_elem_x, int num_elem_y, bool compareOriginal,
                      Array3D<real_t> &output3D);

   /// Return the coordinates of grid points in the parameter space [0,1]^2.
   const std::vector<Vector>& GetParameterGrid() const { return ugrid; }

protected:
   /** @brief Compute the NURBS mesh interpolating the given coordinate of the
       grid of 3D points in @a input3D. */
   void ComputeNURBS(int coordinate, const Array3D<real_t> &input3D);

   /** @brief Write the NURBS surface mesh to file, defined coordinate-wise by
       the entries of @a cmesh. */
   void WriteNURBSMesh(std::vector<Mesh> &cmesh) const;

private:
   int nx, ny; // Number of elements in two directions of the surface grid
   int orderNURBS; // NURBS degree
   real_t hx, hy, hz; // Grid size in reference space

   bool visualization; // Whether to output glvis visualization

   Array3D<real_t> initial3D; // Initial grid of points

   static constexpr int dim = 3;
   Array<int> ncp; // Number of control points in each direction
   Array<int> nks; // Number of knot-spans in each direction

   std::vector<Vector> ugrid; // Parameter space [0,1]^2 grid point coordinates

   std::vector<KnotVector> kv; // Knotvectors in each direction

   std::unique_ptr<NURBSPatch> patch; // Pointer to the only patch in the mesh

   Mesh mesh; // NURBS mesh representing the surface
   std::vector<Mesh> cmesh; // NURBS meshes representing point components
};

int main(int argc, char *argv[])
{
   // Parse command-line options.
   int nx = 4;
   int ny = 4;
   int fnx = 40;
   int fny = 40;
   int order = 3;
   int example = 1;
   bool visualization = true;
   bool compareOriginal = false;

   OptionsParser args(argc, argv);
   args.AddOption(&example, "-ex", "--example",
                  "Example data");
   args.AddOption(&nx, "-nx", "--nx",
                  "Number of elements in x");
   args.AddOption(&ny, "-ny", "--ny",
                  "Number of elements in y");
   args.AddOption(&fnx, "-fnx", "--fnx",
                  "Number of resampled elements in x");
   args.AddOption(&fny, "-fny", "--fny",
                  "Number of resampled elements in y");
   args.AddOption(&order, "-o", "--order",
                  "NURBS finite element order (polynomial degree)");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&compareOriginal, "-orig", "--compare-original", "-no-orig",
                  "--no-compare-original",
                  "Whether to compare to original mesh.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   if (compareOriginal && (fnx != nx || fny != ny))
   {
      cout << "Comparing to the original mesh requires the same number of "
           << "samples!\n";
      return 1;
   }

   constexpr int dim = 3;
   Array3D<real_t> input3D(nx + 1, ny + 1, dim);
   Array3D<real_t> output3D(fnx + 1, fny + 1, dim);

   SurfaceInterpolator surf(nx, ny, order, visualization);
   SurfaceExample(example, surf.GetParameterGrid(), input3D);

   surf.CreateSurface(input3D);
   surf.SampleSurface(fnx, fny, compareOriginal, output3D);

   WriteLinearMesh(fnx, fny, output3D, "final-surf", visualization);

   return 0;
}

void FineSampling(bool uniform, int nx, int ny, const Mesh &mesh,
                  const Array<int> &nks, const std::vector<Vector> &ugrid,
                  Array3D<real_t> &vpos)
{
   const GridFunction *nodes = mesh.GetNodes();

   const real_t hx = 1.0 / (real_t) nx;
   const real_t hy = 1.0 / (real_t) ny;

   const real_t hxks = 1.0 / (real_t) nks[0];
   const real_t hyks = 1.0 / (real_t) nks[1];

   Vector vertex;
   IntegrationPoint ip;

   ip.z = 1.0;
   for (int i=0; i<=nx; ++i)
   {
      const real_t xref = uniform ? i * hx : ugrid[0][i];
      const int nurbsElem0 = std::min((int) (xref / hxks), nks[0] - 1);
      const real_t ipx = (xref - (nurbsElem0 * hxks)) / hxks;
      ip.x = ipx;

      for (int j=0; j<=ny; ++j)
      {
         const real_t yref = uniform ? j * hy : ugrid[1][j];
         const int nurbsElem1 = std::min((int) (yref / hyks), nks[1] - 1);
         const real_t ipy = (yref - (nurbsElem1 * hyks)) / hyks;
         ip.y = ipy;

         const int nurbsElem = nurbsElem0 + (nurbsElem1 * nks[0]);
         nodes->GetVectorValue(nurbsElem, ip, vertex);

         for (int k=0; k<3; ++k)
         {
            vpos(i, j, k) = vertex[k];
         }
      }
   }
}

void CheckError(const Array3D<real_t> &a, const Array3D<real_t> &b, int c,
                int nx, int ny)
{
   real_t maxErr = 0.0;
   for (int i=0; i<=nx; ++i)
      for (int j=0; j<=ny; ++j)
      {
         const real_t err_ij = std::abs(a(i, j, c) - b(i, j, 2));
         maxErr = std::max(maxErr, err_ij);
      }

   cout << "Max error " << maxErr << " for coordinate " << c << endl;
}

void WriteLinearMesh(int nx, int ny, const Array3D<real_t> &v,
                     const std::string &basename, bool visualization)
{
   const int nv = (nx + 1) * (ny + 1);
   const int nelem = nx * ny;
   constexpr int dim = 3; // Spatial dimension

   Mesh fmesh(2, nv, nelem, 0, dim);
   Vector vertex(dim);

   for (int i=0; i<=nx; ++i)
      for (int j=0; j<=ny; ++j)
      {
         for (int k=0; k<dim; ++k) { vertex[k] = v(i, j, k); }
         fmesh.AddVertex(vertex);
      }

   Array<int> verts(4);

   auto vID = [&](int i, int j)
   {
      return j + (i * (ny + 1));
   };

   for (int i=0; i<nx; ++i)
      for (int j=0; j<ny; ++j)
      {
         verts[0] = vID(i, j);
         verts[1] = vID(i+1, j);
         verts[2] = vID(i+1, j+1);
         verts[3] = vID(i, j+1);

         Element* el = fmesh.NewElement(Element::QUADRILATERAL);
         el->SetVertices(verts);
         fmesh.AddElement(el);
      }

   fmesh.FinalizeTopology();

   ofstream mesh_ofs(basename + ".mesh");
   mesh_ofs.precision(8);
   fmesh.Print(mesh_ofs);

   if (visualization)
   {
      char vishost[] = "localhost";
      constexpr int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "mesh\n" << fmesh << "window_title '"
               << basename << "'" << flush;
   }
}

// f(x,y) = sin(2 * pi * x) * sin(2 * pi * y)
void Function1(real_t u, real_t v, real_t &x, real_t &y, real_t &z)
{
   x = u;
   y = v;
   z = sin(2.0 * M_PI * u) * sin(2.0 * M_PI * v);
}

// Part of the parametric surface of a sphere, using spherical coordinates.
void Function2(real_t u, real_t v, real_t &x, real_t &y, real_t &z)
{
   constexpr real_t r = 1.0;
   constexpr real_t pi_4 = M_PI * 0.25;
   constexpr real_t phi0 = -3*pi_4;
   constexpr real_t phi1 = 3*pi_4;
   constexpr real_t theta0 = pi_4;
   constexpr real_t theta1 = 3 * pi_4;

   const real_t phi = (phi0 * (1.0 - v)) + (phi1 * v);
   const real_t theta = (theta0 * (1.0 - u)) + (theta1 * u);
   x = r * sin(theta) * cos(phi);
   y = r * sin(theta) * sin(phi);
   z = r * cos(theta);
}

// Helicoid surface
void Function3(real_t u, real_t v, real_t &x, real_t &y, real_t &z)
{
   x = u * cos(2.0 * M_PI * v);
   y = u * sin(2.0 * M_PI * v);
   z = v;
}

void SurfaceFunction(int example, real_t u, real_t v,
                     real_t &x, real_t &y, real_t &z)
{
   switch (example)
   {
      case 1:
         Function1(u, v, x, y, z);
         break;
      case 2:
         Function2(u, v, x, y, z);
         break;
      default:
         Function3(u, v, x, y, z);
   };
}

void SurfaceExample(int example, const std::vector<Vector> &ugrid,
                    Array3D<real_t> &v3D)
{
   for (int i = 0; i < ugrid[0].Size(); i++)
   {
      for (int j = 0; j < ugrid[1].Size(); j++)
      {
         SurfaceFunction(example, ugrid[0][i], ugrid[1][j],
                         v3D(i, j, 0), v3D(i, j, 1), v3D(i, j, 2));
      }
   }
}

SurfaceInterpolator::SurfaceInterpolator(int num_elem_x, int num_elem_y,
                                         int order, bool vis) :
   nx(num_elem_x), ny(num_elem_y), orderNURBS(order),
   ncp(dim), nks(dim), ugrid(dim - 1)
{
   ncp[0] = nx + 1;
   ncp[1] = ny + 1;
   ncp[2] = order + 1;

   for (int i=0; i<dim; ++i)
   {
      nks[i] = ncp[i] - order;

      Vector intervals(nks[i]);
      Array<int> continuity(nks[i] + 1);

      intervals = 1.0 / (real_t) nks[i];
      continuity = order - 1;
      continuity[0] = -1;
      continuity[nks[i]] = -1;

      kv.emplace_back(order, intervals, continuity);
   }

   patch.reset(new NURBSPatch(&kv[0], &kv[1], &kv[2], dim + 1));

   hx = 1.0 / (real_t) (ncp[0] - 1);
   hy = 1.0 / (real_t) (ncp[1] - 1);
   hz = 1.0 / (real_t) (ncp[2] - 1);

   Vector xi_args;
   Array<int> i_args;
   for (int i=0; i<2; ++i)
   {
      kv[i].FindMaxima(i_args, xi_args, ugrid[i]);
   }
}

void SurfaceInterpolator::CreateSurface(const Array3D<real_t> &input3D)
{
   WriteLinearMesh(nx, ny, input3D, "initial-surf", visualization);

   cmesh.clear();
   for (int c = 0; c < dim; ++c) // Loop over coordinates
   {
      ComputeNURBS(c, input3D);
      cmesh.emplace_back(mesh);
   }

   initial3D = input3D;
   WriteNURBSMesh(cmesh);
}

void SurfaceInterpolator::SampleSurface(int num_elem_x, int num_elem_y,
                                        bool compareOriginal,
                                        Array3D<real_t> &output3D)
{
   Array3D<real_t> vpos(num_elem_x + 1, num_elem_y + 1, dim);
   for (int c = 0; c < dim; ++c) // Loop over coordinates
   {
      FineSampling(true, num_elem_x, num_elem_y, cmesh[c], nks, ugrid, vpos);

      if (compareOriginal)
      {
         FineSampling(false, num_elem_x, num_elem_y, cmesh[c], nks, ugrid,
                      vpos);
         CheckError(initial3D, vpos, c, nx, ny);
      }

      for (int i=0; i<=num_elem_x; ++i)
      {
         for (int j=0; j<=num_elem_y; ++j)
         {
            output3D(i,j,c) = vpos(i,j,2);
         }
      }
   }
}

void SurfaceInterpolator::ComputeNURBS(int coordinate,
                                       const Array3D<real_t> &input3D)
{
   Array<Vector*> x;
   for (int i=0; i<dim; ++i) { x.Append(new Vector(ncp[0])); }

   for (int k = 0; k < ncp[2]; ++k)
   {
      const real_t z = k * hz;

      // For each horizontal slice (fixed k), interpolate a 2D surface by
      // sweeping curve interpolations in each direction. See Algorithm A9.4 of
      // "The NURBS Book" - 2nd ed - Piegl and Tiller.

      // Resize for sweep in first direction
      for (int i=0; i<dim; ++i) { x[i]->SetSize(ncp[0]); }

      // Sweep in the first direction
      for (int j = 0; j < ncp[1]; ++j)
      {
         for (int i = 0; i < ncp[0]; i++)
         {
            (*x[0])[i] = ugrid[0][i];
            (*x[1])[i] = ugrid[1][j];

            const real_t s_ij = input3D(i, j, coordinate);
            (*x[2])[i] = -1.0 + z + s_ij;
         }

         const bool reuse_factorization = j > 0;
         kv[0].FindInterpolant(x, reuse_factorization);

         for (int i = 0; i < ncp[0]; i++)
         {
            (*patch)(i,j,k,0) = (*x[0])[i];
            (*patch)(i,j,k,1) = (*x[1])[i];
            (*patch)(i,j,k,2) = (*x[2])[i];
            (*patch)(i,j,k,3) = 1.0; // weight
         }
      }

      // Resize for sweep in second direction
      for (int i=0; i<dim; ++i) { x[i]->SetSize(ncp[1]); }

      // Do another sweep in the second direction
      for (int i = 0; i < ncp[0]; i++)
      {
         for (int j = 0; j < ncp[1]; ++j)
         {
            (*x[0])[j] = (*patch)(i,j,k,0);
            (*x[1])[j] = (*patch)(i,j,k,1);
            (*x[2])[j] = (*patch)(i,j,k,2);
         }

         const bool reuse_factorization = i > 0;
         kv[1].FindInterpolant(x, reuse_factorization);

         for (int j = 0; j < ncp[1]; ++j)
         {
            (*patch)(i,j,k,0) = (*x[0])[j];
            (*patch)(i,j,k,1) = (*x[1])[j];
            (*patch)(i,j,k,2) = (*x[2])[j];
         }
      }
   }

   for (auto p : x) { delete p; }

   Array<const NURBSPatch*> patches(1);
   patches[0] = patch.get();
   Mesh patch_topology = Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON);
   NURBSExtension nurbsExt(&patch_topology, patches);

   mesh = Mesh(nurbsExt);
}

void SurfaceInterpolator::WriteNURBSMesh(std::vector<Mesh> &cmesh) const
{
   GridFunction *nodes = cmesh[0].GetNodes();
   NURBSPatch patch2D(&kv[0], &kv[1], dim);
   Array<const NURBSPatch*> patches(1);
   patches[0] = &patch2D;
   Mesh patch_topology = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL);
   Array<int> dofs;
   cmesh[0].NURBSext->GetPatchDofs(0, dofs);

   MFEM_VERIFY(dofs.Size() == (nx + 1) * (ny + 1) * (orderNURBS + 1), "");

   for (int j = 0; j < ncp[1]; ++j)
   {
      for (int i = 0; i < ncp[0]; i++)
      {
         const int dof = dofs[i + (ncp[0] * (j + (ncp[1] * orderNURBS)))];
         for (int k=0; k<2; ++k) { patch2D(i,j,k) = (*nodes)[dim*dof + k]; }

         patch2D(i,j,2) = 1.0; // weight
      }
   }

   NURBSExtension nurbsExt(&patch_topology, patches);
   Mesh mesh2D(nurbsExt);

   FiniteElementCollection *fec = nodes->OwnFEC();
   FiniteElementSpace fespace(&mesh2D, fec, dim, Ordering::byVDIM);
   GridFunction x(&fespace);

   const int n = mesh2D.GetNodes()->Size() / (dim - 1);
   MFEM_VERIFY((dim - 1) * n == mesh2D.GetNodes()->Size(), "");
   MFEM_VERIFY(dim * n == x.Size(), "");

   Array<int> dofs2D;
   mesh2D.NURBSext->GetPatchDofs(0, dofs2D);

   for (int k=0; k<dim; ++k)
   {
      const GridFunction &nodes_k = *cmesh[k].GetNodes();

      for (int j = 0; j < ncp[1]; ++j)
         for (int i = 0; i < ncp[0]; i++)
         {
            const int dof = dofs[i + (ncp[0] * (j + (ncp[1] * orderNURBS)))];
            const int dof2D = dofs2D[i + (ncp[0] * j)];
            x[(dim*dof2D) + k] = nodes_k[dim*dof + 2];
         }
   }

   ofstream mesh_ofs("nurbs-surf.mesh");
   mesh_ofs.precision(8);
   mesh2D.Print(mesh_ofs, "", &x);
}
