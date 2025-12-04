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
//               nurbs_surface -o 3 -nx 20 -ny 20 -fnx 40 -fny 40 -ex 1 -j 0.5
//               nurbs_surface -o 3 -nx 10 -ny 10 -fnx 10 -fny 10 -ex 2 -orig
//               nurbs_surface -o 3 -nx 10 -ny 10 -fnx 40 -fny 40 -ex 2
//               nurbs_surface -o 3 -nx 20 -ny 20 -fnx 10 -fny 10 -ex 2
//               nurbs_surface -o 3 -nx 10 -ny 10 -fnx 10 -fny 10 -ex 3 -orig
//               nurbs_surface -o 3 -nx 10 -ny 10 -fnx 40 -fny 40 -ex 3
//               nurbs_surface -o 3 -nx 20 -ny 20 -fnx 10 -fny 10 -ex 3
//               nurbs_surface -o 3 -nx 20 -ny 10 -fnx 20 -fny 10 -ex 4 -orig
//             * nurbs_surface -o 3 -nx 20 -ny 10 -fnx 80 -fny 40 -ex 4
//             * nurbs_surface -o 3 -nx 40 -ny 20 -fnx 20 -fny 10 -ex 4
//             * nurbs_surface -o 3 -nx 100 -ny 100 -fnx 100 -fny 100 -ex 5 -orig
//             * nurbs_surface -o 3 -nx 100 -ny 100 -fnx 400 -fny 400 -ex 5
//             * nurbs_surface -o 3 -nx 200 -ny 200 -fnx 100 -fny 100 -ex 5
//
// Description:  This example demonstrates the use of MFEM to interpolate an
//               input surface point grid in 3D using a NURBS surface. The NURBS
//               surface can then be sampled to generate an output mesh of
//               arbitrary resolution while staying close to the input geometry.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Example data for 3D point grid on surface, given by an analytic function.
void SurfaceGridExample(int example, int nx, int ny, Array3D<real_t> &vertices,
                        real_t jitter);

// Write a linear surface mesh with given vertex positions in v.
void WriteLinearMesh(int nx, int ny, const Array3D<real_t> &v,
                     const std::string &basename, bool visualization = false,
                     int x = 0, int y = 0, int w = 500, int h = 500);

// Given an input grid of 3D points on a surface, this class computes a NURBS
// surface of given order that interpolates the vertices of the input grid.
class SurfaceInterpolator
{
public:
   /// Constructor for a given 2D point grid size and NURBS order.
   SurfaceInterpolator(int num_elem_x, int num_elem_y, int order);

   /// Create a surface interpolating the 2D grid of 3D points in @a input3D.
   void CreateSurface(const Array3D<real_t> &input3D);

   /// Sample the surface with the given grid size, storing points in
   /// @a output3D.
   void SampleSurface(int num_elem_x, int num_elem_y, bool compareOriginal,
                      Array3D<real_t> &output3D);

   /** @brief Write the NURBS surface mesh to file, defined coordinate-wise by
       the entries of @a cmesh. */
   void WriteNURBSMesh(const std::string &basename, bool visualization = false,
                       int x = 0, int y = 0, int w = 500, int h = 500);

protected:
   /** @brief Compute the NURBS mesh interpolating the given coordinate of the
       grid of 3D points in @a input3D. */
   void ComputeNURBS(int coordinate, const Array3D<real_t> &input3D);

private:
   int nx, ny; // Number of elements in two directions of the surface grid
   int orderNURBS; // NURBS degree
   real_t hx, hy, hz; // Grid size in reference space

   Array3D<real_t> initial3D; // Initial grid of points

   static constexpr int dim = 3;
   Array<int> ncp; // Number of control points in each direction
   Array<int> nks; // Number of knot-spans in each direction

   std::vector<Vector> ugrid; // Parameter space [0,1]^2 grid point coordinates

   std::vector<KnotVector> kv; // KnotVectors in each direction

   std::unique_ptr<NURBSPatch> patch; // Pointer to the only patch in the mesh

   Mesh mesh; // NURBS mesh representing the surface
   std::vector<Mesh> cmesh; // NURBS meshes representing point components
};


int main(int argc, char *argv[])
{
   // Parse command-line options
   int nx = 4;
   int ny = 4;
   int fnx = 40;
   int fny = 40;
   int order = 3;
   int example = 1;
   bool visualization = true;
   bool compareOriginal = false;
   real_t jitter = 0.0;

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
                  "Compare to the original mesh?");
   args.AddOption(&jitter, "-j", "--jitter",
                  "Relative jittering in (0,1) to add to the input point "
                  "coordinates on a uniform nx x ny grid (0 by default)");
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

   // Dimensions of the 3 surfaces (Input, NURBS, Output)
   cout << "Input Surface:  " << nx << " x " << ny << " linear elements\n";
   cout << "NURBS Surface:  " << nx + 1 - order << " x " << ny + 1 - order
        << " knot elements of order " << order << "\n";
   cout << "Output Surface: " << fnx << " x " << fny << " linear elements\n";

   // Set the vertex coordinates of the initial linear mesh
   constexpr int dim = 3;
   Array3D<real_t> input3D(nx + 1, ny + 1, dim);
   SurfaceGridExample(example, nx, ny, input3D, jitter);

   // Create a NURBS surface for the given nx, ny and order parameters that
   // interpolates the input vertex coordinates
   SurfaceInterpolator surf(nx, ny, order);
   surf.CreateSurface(input3D);

   // Compute the vertex coordinates of the output linear mesh by sampling the
   // values from the NURBS surface
   Array3D<real_t> output3D(fnx + 1, fny + 1, dim);
   surf.SampleSurface(fnx, fny, compareOriginal, output3D);

   // Save and optionally visualize the 3 surfaces (Input, NURBS, Output)
   WriteLinearMesh(nx, ny, input3D, "Input-Surface", visualization, 0, 0);
   surf.WriteNURBSMesh("NURBS-Surface", visualization, 502, 0);
   WriteLinearMesh(fnx, fny, output3D, "Output-Surface", visualization, 1004, 0);

   return 0;
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

// Mobius strip
void Function4(real_t u, real_t v, real_t &x, real_t &y, real_t &z)
{
   constexpr int twists = 1;
   const real_t a = 1.0 + 0.5 * ((2.0 * v) - 1.0) * cos(2.0 * M_PI * twists * u);
   x = a * cos(2.0 * M_PI * u);
   y = a * sin(2.0 * M_PI * u);
   z = 0.5 * (2.0 * v - 1.0) * sin(2.0 * M_PI * twists * u);
}

// Breather surface
void Function5(real_t u, real_t v, real_t &x, real_t &y, real_t &z)
{
   const real_t m = 13.2 * ((2.0 * u) - 1.0);
   const real_t n = 37.4 * ((2.0 * v) - 1.0);
   constexpr real_t b = 0.4;
   constexpr real_t r = 1.0 - (b*b);
   const real_t w = sqrt(r);
   const real_t denom = b * (pow(w*cosh(b*m),2) + pow(b*sin(w*n),2));
   x = -m + (2*r*cosh(b*m)*sinh(b*m)) / denom;
   y = (2*w*cosh(b*m)*(-(w*cos(n)*cos(w*n)) - sin(n)*sin(w*n))) / denom;
   z = (2*w*cosh(b*m)*(-(w*sin(n)*cos(w*n)) + cos(n)*sin(w*n))) / denom;
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
      case 3:
         Function3(u, v, x, y, z);
         break;
      case 4:
         Function4(u, v, x, y, z);
         break;
      default:
         Function5(u, v, x, y, z);
   };
}

// Example data for 3D point grid on surface, given by an analytic function.
void SurfaceExample(int example, const std::vector<Vector> &grid,
                    Array3D<real_t> &v3D, real_t jitter)
{
   int seed = (int)time(0);
   srand((unsigned)seed);

   real_t h0 = grid[0][1]-grid[0][0], h1 = grid[1][1]-grid[1][0];
   for (int i = 0; i < grid[0].Size(); i++)
   {
      for (int j = 0; j < grid[1].Size(); j++)
      {
         if (i != 0 && i != grid[0].Size()-1 && j != 0 && j != grid[1].Size()-1)
         {
            SurfaceFunction(example, grid[0][i] + rand_real()*h0*jitter,
                            grid[1][j] + rand_real()*h1*jitter,
                            v3D(i, j, 0), v3D(i, j, 1), v3D(i, j, 2));
         }
         else
         {
            SurfaceFunction(example, grid[0][i], grid[1][j],
                            v3D(i, j, 0), v3D(i, j, 1), v3D(i, j, 2));
         }
      }
   }
}

void SurfaceGridExample(int example, int nx, int ny, Array3D<real_t> &vertices,
                        real_t jitter = 0)
{
   // Define a uniform grid of the reference parameter space [0,1]^2
   std::vector<Vector> uniformGrid(2);
   for (int i = 0; i < 2; ++i)
   {
      const int n = (i == 0) ? nx : ny;
      const real_t h = 1.0 / n;
      uniformGrid[i].SetSize(n + 1);
      for (int j = 0; j <= n; ++j) { uniformGrid[i][j] = j * h; }
   }

   SurfaceExample(example, uniformGrid, vertices, jitter);
}

// Write a linear surface mesh with given vertex positions in v.
void WriteLinearMesh(int nx, int ny, const Array3D<real_t> &v,
                     const std::string &basename, bool visualization,
                     int x, int y, int w, int h)
{
   const int nv = (nx + 1) * (ny + 1);
   const int nelem = nx * ny;
   constexpr int dim = 3; // Spatial dimension

   Mesh lmesh(2, nv, nelem, 0, dim);
   Vector vertex(dim);

   for (int i = 0; i <= nx; ++i)
   {
      for (int j = 0; j <= ny; ++j)
      {
         for (int k = 0; k < dim; ++k) { vertex[k] = v(i, j, k); }
         lmesh.AddVertex(vertex);
      }
   }

   Array<int> verts(4);

   auto vID = [&](int i, int j)
   {
      return j + (i * (ny + 1));
   };

   for (int i = 0; i < nx; ++i)
   {
      for (int j = 0; j < ny; ++j)
      {
         verts[0] = vID(i, j);
         verts[1] = vID(i+1, j);
         verts[2] = vID(i+1, j+1);
         verts[3] = vID(i, j+1);

         Element* el = lmesh.NewElement(Element::QUADRILATERAL);
         el->SetVertices(verts);
         lmesh.AddElement(el);
      }
   }

   lmesh.FinalizeTopology();

   ofstream mesh_ofs(basename + ".mesh");
   mesh_ofs.precision(8);
   lmesh.Print(mesh_ofs);

   if (visualization)
   {
      char vishost[] = "localhost";
      constexpr int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "mesh\n" << lmesh
               << "window_title '" << basename << "'"
               << "window_geometry "
               << x << " " << y << " " << w << " " << h << "\n"
               << "keys PPPPPPPPAattttt******\n"
               << flush;
   }
}


// Compute error of interpolation with respect to an input grid of point data.
void CheckError(const Array3D<real_t> &a, const Array3D<real_t> &b, int c,
                int nx, int ny)
{
   real_t maxErr = 0.0;
   for (int i = 0; i <= nx; ++i)
   {
      for (int j = 0; j <= ny; ++j)
      {
         const real_t err_ij = std::abs(a(i, j, c) - b(i, j, 2));
         maxErr = std::max(maxErr, err_ij);
      }
   }

   cout << "Max error: " << maxErr << " for coordinate " << c << endl;
}


// Sample a NURBS mesh to generate a first-order mesh.
void SampleNURBS(bool uniform, int nx, int ny, const Mesh &mesh,
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
   for (int i = 0; i <= nx; ++i)
   {
      const real_t xref = uniform ? i * hx : ugrid[0][i];
      const int nurbsElem0 = std::min((int) (xref / hxks), nks[0] - 1);
      const real_t ipx = (xref - (nurbsElem0 * hxks)) / hxks;
      ip.x = ipx;

      for (int j = 0; j <= ny; ++j)
      {
         const real_t yref = uniform ? j * hy : ugrid[1][j];
         const int nurbsElem1 = std::min((int) (yref / hyks), nks[1] - 1);
         const real_t ipy = (yref - (nurbsElem1 * hyks)) / hyks;
         ip.y = ipy;

         const int nurbsElem = nurbsElem0 + (nurbsElem1 * nks[0]);
         nodes->GetVectorValue(nurbsElem, ip, vertex);

         for (int k = 0; k < 3; ++k)
         {
            vpos(i, j, k) = vertex[k];
         }
      }
   }
}


SurfaceInterpolator::SurfaceInterpolator(int num_elem_x, int num_elem_y,
                                         int order) :
   nx(num_elem_x), ny(num_elem_y), orderNURBS(order),
   ncp(dim), nks(dim), ugrid(dim - 1)
{
   ncp[0] = nx + 1;
   ncp[1] = ny + 1;
   ncp[2] = order + 1;

   for (int i = 0; i < dim; ++i)
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
   for (int i = 0; i < 2; ++i)
   {
      kv[i].FindMaxima(i_args, xi_args, ugrid[i]);
   }
}

void SurfaceInterpolator::CreateSurface(const Array3D<real_t> &input3D)
{
   cmesh.clear();
   for (int c = 0; c < dim; ++c) // Loop over coordinates
   {
      ComputeNURBS(c, input3D);
      cmesh.emplace_back(mesh);
   }

   initial3D = input3D;
}

void SurfaceInterpolator::SampleSurface(int num_elem_x, int num_elem_y,
                                        bool compareOriginal,
                                        Array3D<real_t> &output3D)
{
   Array3D<real_t> vpos(num_elem_x + 1, num_elem_y + 1, dim);
   for (int c = 0; c < dim; ++c) // Loop over coordinates
   {
      SampleNURBS(true, num_elem_x, num_elem_y, cmesh[c], nks, ugrid, vpos);

      if (compareOriginal)
      {
         SampleNURBS(false, num_elem_x, num_elem_y, cmesh[c], nks, ugrid, vpos);
         CheckError(initial3D, vpos, c, nx, ny);
      }

      for (int i = 0; i <= num_elem_x; ++i)
      {
         for (int j = 0; j <= num_elem_y; ++j)
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
   for (int i = 0; i < dim; ++i) { x.Append(new Vector(ncp[0])); }

   for (int k = 0; k < ncp[2]; ++k)
   {
      const real_t z = k * hz;

      // For each horizontal slice (fixed k), interpolate a 2D surface by
      // sweeping curve interpolations in each direction. See Algorithm A9.4 of
      // "The NURBS Book" - 2nd ed - Piegl and Tiller.

      // Resize for sweep in first direction
      for (int i = 0; i < dim; ++i) { x[i]->SetSize(ncp[0]); }

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
      for (int i = 0; i < dim; ++i) { x[i]->SetSize(ncp[1]); }

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

void SurfaceInterpolator::WriteNURBSMesh(const std::string &basename,
                                         bool visualization,
                                         int x, int y, int w, int h)
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
         for (int k = 0; k < 2; ++k) { patch2D(i,j,k) = (*nodes)[dim*dof + k]; }
         patch2D(i,j,2) = 1.0; // weight
      }
   }

   NURBSExtension nurbsExt(&patch_topology, patches);
   Mesh mesh2D(nurbsExt);

   FiniteElementCollection *fec = nodes->OwnFEC();
   FiniteElementSpace fespace(&mesh2D, fec, dim, Ordering::byVDIM);
   GridFunction nodes2D(&fespace);

   const int n = mesh2D.GetNodes()->Size() / (dim - 1);
   MFEM_VERIFY((dim - 1) * n == mesh2D.GetNodes()->Size(), "");
   MFEM_VERIFY(dim * n == nodes2D.Size(), "");

   Array<int> dofs2D;
   mesh2D.NURBSext->GetPatchDofs(0, dofs2D);

   for (int k = 0; k < dim; ++k)
   {
      const GridFunction &nodes_k = *cmesh[k].GetNodes();

      for (int j = 0; j < ncp[1]; ++j)
      {
         for (int i = 0; i < ncp[0]; i++)
         {
            const int dof = dofs[i + (ncp[0] * (j + (ncp[1] * orderNURBS)))];
            const int dof2D = dofs2D[i + (ncp[0] * j)];
            nodes2D[(dim*dof2D) + k] = nodes_k[dim*dof + 2];
         }
      }
   }

   // Make mesh2D into a surface mesh with nodes given by nodes2D
   mesh2D.NewNodes(nodes2D);

   ofstream mesh_ofs(basename + ".mesh");
   mesh_ofs.precision(8);
   mesh2D.Print(mesh_ofs);

   if (visualization)
   {
      char vishost[] = "localhost";
      constexpr int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "mesh\n" << mesh2D
               << "window_title '" << basename << "'"
               << "window_geometry "
               << x << " " << y << " " << w << " " << h << "\n"
               << "keys PPPPPPPPAattttt******\n"
               << flush;
   }
}
