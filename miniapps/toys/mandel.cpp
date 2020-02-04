// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
//
//             ----------------------------------------------
//             Mandel Miniapp: Fractal visualization with AMR
//             ----------------------------------------------
//
// This miniapp is a specialized version of the Shaper miniapp to the Mandelbrot
// set. It provides a light-hearted example of AMR and GLVis integration.
//
// Compile with: make mandel
//
// Sample runs:  mandel
//               mandel -m ../../data/inline-tri.mesh
//               mandel -m ../../data/star-mixed-p2.mesh -a -ncl -1 -sd 4
//               mandel -m ../../data/klein-bottle.mesh
//               mandel -m ../../data/inline-hex.mesh

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;
using namespace std;

// Given a point x, return its material id as an integer. The ids should be
// positive. If the point is exactly on the interface, return 0.
//
// In this particular miniapp, the material value is based on the number of
// iterations for the point from the definition of the Mandelbrot set.
int material(Vector &x, Vector &xmin, Vector &xmax)
{
   // Rescaling to [0,1]^sdim
   for (int i = 0; i < x.Size(); i++)
   {
      x(i) = (x(i)-xmin(i))/(xmax(i)-xmin(i));
   }
   x(0) -= 0.1;
   double col = x(0), row = x(1);
   {
      int width = 1080, height = 1080;
      col *= width;
      row *= height;
      double c_re = (col - width/2)*4.0/width;
      double c_im = (row - height/2)*4.0/width;
      double x = 0, y = 0;
      int iteration = 0, maxit = 10000;
      while (x*x+y*y <= 4 && iteration < maxit)
      {
         double x_new = x*x - y*y + c_re;
         y = 2*x*y + c_im;
         x = x_new;
         iteration++;
      }
      if (iteration < maxit)
      {
         return iteration%10+2;
      }
      else
      {
         return 1;
      }
   }
}

int main(int argc, char *argv[])
{
   const char *mesh_file = "../../data/inline-quad.mesh";
   int sd = 2;
   int nclimit = 1;
   bool aniso = false;
   bool visualization = 1;

   // Parse command line
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Input mesh file to shape materials in.");
   args.AddOption(&sd, "-sd", "--sub-divisions",
                  "Number of element subdivisions for interface detection.");
   args.AddOption(&nclimit, "-ncl", "--nc-limit",
                  "Level of hanging nodes allowed (-1 = unlimited).");
   args.AddOption(&aniso, "-a", "--aniso", "-i", "--iso",
                  "Enable anisotropic refinement of quads and hexes.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   args.PrintOptions(cout);

   // Read initial mesh, get dimensions and bounding box
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   int sdim = mesh.SpaceDimension();
   Vector xmin, xmax;
   mesh.GetBoundingBox(xmin, xmax);

   // Increase the mesh resolution
   for (int l = 0; l < 3; l++) { mesh.UniformRefinement(); }

   // NURBS meshes don't support non-conforming refinement for now
   if (mesh.NURBSext) { mesh.SetCurvature(2); }

   // Anisotropic refinement not supported for simplex meshes.
   if (mesh.MeshGenerator() & 1) { aniso = false; }

   // Mesh attributes will be visualized as piece-wise constants
   L2_FECollection attr_fec(0, dim);
   FiniteElementSpace attr_fespace(&mesh, &attr_fec);
   GridFunction attr(&attr_fespace);

   // GLVis server to visualize to
   socketstream sol_sock;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sol_sock.open(vishost, visport);
      sol_sock.precision(8);
   }

   // Shaping loop
   for (int iter = 0; 1; iter++)
   {
      Array<Refinement> refs;
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         bool refine = false;

         // Sample materials in each element using "sd" sub-divisions
         Vector pt;
         Geometry::Type geom = mesh.GetElementBaseGeometry(i);
         ElementTransformation *T = mesh.GetElementTransformation(i);
         RefinedGeometry *RefG = GlobGeometryRefiner.Refine(geom, sd, 1);
         IntegrationRule &ir = RefG->RefPts;

         // Refine any element where different materials are detected. A more
         // sophisticated logic can be implemented here -- e.g. don't refine
         // the interfaces between certain materials.
         Array<int> mat(ir.GetNPoints());
         double matsum = 0.0;
         for (int j = 0; j < ir.GetNPoints(); j++)
         {
            T->Transform(ir.IntPoint(j), pt);
            int m = material(pt, xmin, xmax);
            mat[j] = m;
            matsum += m;
            if ((int)matsum != m*(j+1))
            {
               refine = true;
            }
         }

         // Set the element attribute as the "average". Other choices are
         // possible here too, e.g. attr(i) = mat;
         attr(i) = round(matsum/ir.GetNPoints());

         // Mark the element for refinement
         if (refine)
         {
            int type = 7;
            if (aniso)
            {
               // Determine the XYZ bitmask for anisotropic refinement.
               int dx = 0, dy = 0, dz = 0;
               const int s = sd+1;
               if (dim == 2)
               {
                  for (int j = 0; j <= sd; j++)
                     for (int i = 0; i < sd; i++)
                     {
                        dx += abs(mat[j*s + i+1] - mat[j*s + i]);
                        dy += abs(mat[(i+1)*s + j] - mat[i*s + j]);
                     }
               }
               else if (dim == 3)
               {
                  for (int k = 0; k <= sd; k++)
                     for (int j = 0; j <= sd; j++)
                        for (int i = 0; i < sd; i++)
                        {
                           dx += abs(mat[(k*s + j)*s + i+1] - mat[(k*s + j)*s + i]);
                           dy += abs(mat[(k*s + i+1)*s + j] - mat[(k*s + i)*s + j]);
                           dz += abs(mat[((i+1)*s + j)*s + k] - mat[(i*s + j)*s + k]);
                        }
               }
               type = 0;
               const int tol = mat.Size() / 10;
               if (dx > tol) { type |= 1; }
               if (dy > tol) { type |= 2; }
               if (dz > tol) { type |= 4; }
               if (!type) { type = 7; } // because of tol
            }

            refs.Append(Refinement(i, type));
         }
      }

      // Visualization
      if (visualization)
      {
         sol_sock << "solution\n" << mesh << attr;
         if (iter == 0 && sdim == 2)
         {
            sol_sock << "keys 'RjlppppppppppppppA*************'\n";
         }
         if (iter == 0 && sdim == 3)
         {
            sol_sock << "keys 'YYYYYYYYYXXXXXXXmA********8888888pppttt";
            if (dim == 3) { sol_sock << "iiM"; }
            sol_sock << "'\n";
         }
         sol_sock << flush;
      }

      // Ask the user if we should continue refining
      cout << "Iteration " << iter+1 << ": mesh has " << mesh.GetNE() <<
           " elements. \n";
      if ((iter+1) % 4 == 0)
      {
         if (!visualization) { break; }
         char yn;
         cout << "Continue shaping? --> ";
         cin >> yn;
         if (yn == 'n' || yn == 'q') { break; }
      }

      // Perform refinement, update spaces and grid functions
      mesh.GeneralRefinement(refs, -1, nclimit);
      attr_fespace.Update();
      attr.Update();
   }

   // Set element attributes in the mesh object before saving
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      mesh.SetAttribute(i, attr(i));
   }
   mesh.SetAttributes();

   // Save the final mesh
   ofstream mesh_ofs("mandel.mesh");
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);
}
