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
//           -------------------------------------------------
//           Mondrian Miniapp: Convert an image to an AMR mesh
//           -------------------------------------------------
//
// This miniapp is a specialized version of the Shaper miniapp that converts an
// input image to an AMR mesh. It allows the fast approximate meshing of any
// domain for which there is an image.
//
// The input to image should be in 8-bit grayscale PGM format. You can use a
// number of image manipulation tools, such as GIMP (gimp.org) and ImageMagick's
// convert utility (imagemagick.org/script/convert.php) to convert your image to
// this format as a pre-processing step, e.g.:
//
//   /usr/bin/convert australia.svg -compress none -depth 8 australia.pgm
//
// Compile with: make mondrian
//
// Sample runs:  mondrian -i australia.pgm
//               mondrian -i australia.pgm -m ../../data/inline-tri.mesh
//               mondrian -i australia.pgm -m ../../data/disc-nurbs.mesh
//               mondrian -i australia.pgm -sd 3 -a -ncl -1

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;
using namespace std;

// Simple class to parse portable graymap format (PGM) image files, see
// http://netpbm.sourceforge.net/doc/pgm.html
class ParsePGM
{
public:
   ParsePGM(const char *filename);
   ~ParsePGM();

   int Height() const { return N; }
   int Width() const { return M; }

   int operator()(int i, int j) const
   { return int((pgm8) ? pgm8[M*i+j] : pgm16[M*i+j]); }

private:
   int M, N;
   int depth;

   char *pgm8;
   unsigned short int *pgm16;

   void ReadMagicNumber(istream &in);
   void ReadComments(istream &in);
   void ReadDimensions(istream &in);
   void ReadDepth(istream &in);
   void ReadPGM(istream &in);
};

// Given a point x, return its "material" specification defined by the grayscale
// pixel values from the pgm image using NC different colors.
int material(const ParsePGM &pgm, int NC,
             Vector &x, Vector &xmin, Vector &xmax);

int main(int argc, char *argv[])
{
   const char *mesh_file = "../../data/inline-quad.mesh";
   const char *img_file = "australia.pgm";
   int sd = 2;
   int nclimit = 1;
   int ncolors = 3;
   bool aniso = false;
   bool visualization = 1;

   // Parse command line
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Input mesh file to shape materials in.");
   args.AddOption(&img_file, "-i", "--img",
                  "Input image.");
   args.AddOption(&sd, "-sd", "--sub-divisions",
                  "Number of element subdivisions for interface detection.");
   args.AddOption(&nclimit, "-ncl", "--nc-limit",
                  "Level of hanging nodes allowed (-1 = unlimited).");
   args.AddOption(&ncolors, "-nc", "--num-colors",
                  "Number of colors considered (1-256, based on binning).");
   args.AddOption(&aniso, "-a", "--aniso", "-i", "--iso",
                  "Enable anisotropic refinement of quads and hexes.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   args.PrintOptions(cout);

   // Read the image
   ParsePGM pgm(img_file);

   // Read initial mesh, get dimensions and bounding box
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   int sdim = mesh.SpaceDimension();
   Vector xmin, xmax;
   mesh.GetBoundingBox(xmin, xmax);

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
            int m = material(pgm, 256/ncolors, pt, xmin, xmax);
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
            sol_sock << "keys 'RjlmpppppppppppppA*************'\n";
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
      if ((iter+1) % 3 == 0)
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
   ofstream mesh_ofs("mondrian.mesh");
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);
}

ParsePGM::ParsePGM(const char *filename)
   : M(-1), N(-1), depth(-1), pgm8(NULL), pgm16(NULL)
{
   ifstream in(filename);
   if (!in)
   {
      // Abort with an error message
      MFEM_ABORT("Image file not found: " << filename << '\n');
   }

   ReadMagicNumber(in);
   ReadDimensions(in);
   ReadDepth(in);
   ReadPGM(in);

   in.close();
}

ParsePGM::~ParsePGM()
{
   if (pgm8  != NULL) { delete [] pgm8; }
   if (pgm16 != NULL) { delete [] pgm16; }
}

void ParsePGM::ReadMagicNumber(istream &in)
{
   char c;
   int p;
   in >> c >> p; // Read magic number which should be P2 or P5
   MFEM_VERIFY(c == 'P' && (p == 2 || p == 5),
               "Invalid PGM file! Unrecognized magic number\""
               << c << p << "\".");
   ReadComments(in);
}

void ParsePGM::ReadComments(istream &in)
{
   string buf;
   in >> std::ws; // absorb any white space
   while (in.peek() == '#')
   {
      std::getline(in,buf);
   }
   in >> std::ws; // absorb any white space
}

void ParsePGM::ReadDimensions(istream &in)
{
   in >> M;
   ReadComments(in);
   in >> N;
   ReadComments(in);
}

void ParsePGM::ReadDepth(istream &in)
{
   in >> depth;
   ReadComments(in);
}

void ParsePGM::ReadPGM(istream &in)
{
   if (depth < 16)
   {
      pgm8 = new char[M*N];
   }
   else
   {
      pgm16 = new unsigned short int[M*N];
   }

   if (pgm8)
   {
      for (int i=0; i<M*N; i++)
      {
         in >> pgm8[i];
      }
   }
   else
   {
      for (int i=0; i<M*N; i++)
      {
         in >> pgm16[i];
      }
   }
}

int material(const ParsePGM &pgm, int NC, Vector &x, Vector &xmin, Vector &xmax)
{
   // Rescaling to [0,1]^sdim
   for (int i = 0; i < x.Size(); i++)
   {
      x(i) = (x(i)-xmin(i))/(xmax(i)-xmin(i));
   }

   int M = pgm.Width();
   int N = pgm.Height();

   int i = x(1)*N, j = x(0)*M;
   if (i == N) { i = N-1; }
   if (j == M) { j = M-1; }
   i = N-1-i;

   return pgm(i,j)/NC+1;
}
