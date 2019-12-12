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
//               ----------------------------------------
//               Life Miniapp:  Model of the Game of Life
//               ----------------------------------------
//
// This miniapp provides a light-hearted example of mesh manipulation and
// GLVis integration.
//
// Compile with: make life
//
// Sample runs: life
//              life -nx 100 -ny 100 -r 0.3

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <bitset>
#include <vector>
#include <stdlib.h> // for random number functions

using namespace std;
using namespace mfem;

//void Update(char * b[], int r, int s);
//void ProjectStep(char b[], GridFunction & x, int ns, int s);

//void PrintRule(bitset<8> & r);
void Update(vector<bool> * b[], int nx, int ny);
void ProjectStep(const vector<bool> & b, GridFunction & x, int n);

void InitSketchPad(vector<bool> & b, int nx, int ny, const Array<int> & params);
void InitBlinker(vector<bool> & b, int nx, int ny, const Array<int> & params);
void InitGlider(vector<bool> & b, int nx, int ny, const Array<int> & params);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   // int ns = 20;
   int nx = 20;
   int ny = 20;
   int rs = -1;
   double r = 0.1;
   Array<int> sketch_pad_params(0);
   Array<int> blinker_params(0);
   Array<int> glider_params(0);
   bool visualization = 1;

   OptionsParser args(argc, argv);
   // args.AddOption(&ns, "-ns", "--num-steps",
   //             "Number of steps of the 2D cellular automaton.");
   args.AddOption(&nx, "-nx", "--num-elems-x",
                  "Number of elements in the x direction.");
   args.AddOption(&ny, "-ny", "--num-elems-y",
                  "Number of elements in the y direction.");
   args.AddOption(&r, "-r", "--random-fraction",
                  "Fraction of randomly chosen live cells.");
   args.AddOption(&rs, "-rs", "--random-seed",
                  "Seed for the random number generator.");
   args.AddOption(&sketch_pad_params, "-sp", "--sketch-pad",
                  "Specify the starting coordinates and values on a grid"
                  " of cells.");
   args.AddOption(&blinker_params, "-b", "--blinker",
                  "Specify the starting coordinates and orientation"
                  " of the blinker.");
   args.AddOption(&glider_params, "-g", "--glider",
                  "Specify the starting coordinates and orientation"
                  " of the glider.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(nx, ny, Element::QUADRILATERAL, 0, nx, ny);
   // mesh->Print(*(new ofstream("zzz")));
   int dim = mesh->Dimension();

   // 3. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec = new L2_FECollection(0, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

   // 4. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x = 0.0;

   // 5. Open a socket to GLVis
   socketstream sol_sock;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sol_sock.open(vishost, visport);
   }

   // 6. Initialize a pair of bit arrays
   int len = nx * ny;

   vector<bool> * vbp[2];
   vector<bool> * vbptmp = NULL;
   vector<bool> vb0(len);
   vector<bool> vb1(len);

   vbp[0] = &vb0;
   vbp[1] = &vb1;

   if ( r > 0.0 )
   {
      long seed;
      if ( rs < 0 )
      {
         // srandomdev(); // not available on Linux?
         srandom(time(NULL));
         seed = random();
         srand48(seed);
      }
      else
      {
         seed = (long)rs;
      }
      cout << "Using random seed:  " << seed << endl;
   }

   for (int i=0; i<len; i++)
   {
      double rv = drand48();
      // double rv = (double)random()/(pow(2.0,31)-1);
      vb0[i] = (rv <= r);
      vb1[i] = false;
   }
   if ( sketch_pad_params.Size() > 2 )
   {
      InitSketchPad(vb0, nx, ny, sketch_pad_params);
   }
   if ( blinker_params.Size() > 0 && (blinker_params.Size() % 3 == 0 ) )
   {
      InitBlinker(vb0, nx, ny, blinker_params);
   }
   if ( glider_params.Size() > 0 && (glider_params.Size() % 3 == 0 ) )
   {
      InitGlider(vb0, nx, ny, glider_params);
   }
   // InitBlinker(vb0, nx, ny, nx/2, ny/3);

   ProjectStep(*vbp[0], x, len);

   // 7. Create the rule as a bitset and display it for the user
   // bitset<8> rbs = r;
   // PrintRule(rbs);

   // 8. Apply the rule iteratively
   cout << endl << "Running the Game of Life..." << flush;
   // for (int s=1; s<ns; s++)
   bool is_good = true;
   while ( is_good && visualization )
   {
      Update(vbp, nx, ny);
      ProjectStep(*vbp[1], x, len);

      vbptmp = vbp[0];
      vbp[0] = vbp[1];
      vbp[1] = vbptmp;

      // 9. Send the solution by socket to a GLVis server.
      is_good = sol_sock.good();

      if (visualization && is_good )
      {
         sol_sock << "solution\n" << *mesh << x << flush;
         {
            static int once = 1;
            if (once)
            {
               sol_sock << "keys Ajlm\n";
               sol_sock << "view 0 0\n";
               sol_sock << "zoom 1.9\n";
               sol_sock << "palette 24\n";
               once = 0;
            }
         }
      }

      // char foo;
      //if (cin.eof()) cout << "cin.eof is true" << endl;
      // foo = getchar();
      // cout << "foo = " << foo << endl;
      // cout << cin.peek() << endl;
   }
   cout << "done." << endl;

   // 10. Save the refined mesh and the solution. This output can be
   //     viewed later using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("life.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);
   ofstream sol_ofs("life.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   // 11. Free the used memory.
   delete fespace;
   delete fec;
   delete mesh;

   return 0;
}
/*
bool Rule(bitset<8> & r, bool b0, bool b1, bool b2)
{
   return r[(b0?1:0)+(b1?2:0)+(b2?4:0)];
}

void PrintRule(bitset<8> & r)
{
   cout << endl << "Rule:" << endl;
   for (int i=7; i>=0; i--)
   {
      cout << " " << i/4 << (i/2)%2 << i%2;
   }
   cout << endl;
   for (int i=7; i>=0; i--)
   {
      cout << "  " << Rule(r,i%2,(i/2)%2,i/4) << " ";
   }
   cout << endl;
}
*/
inline int index(int i, int j, int nx, int ny)
{
   return ((j + ny) % ny) * nx + ((i + nx) % nx);
}

void Update(vector<bool> * b[], int nx, int ny)
{
   for (int j=0; j<ny; j++)
   {
      for (int i=0; i<nx; i++)
      {
         int c =
            (int)(*b[0])[index(i+0,j+0,nx,ny)] +
            (int)(*b[0])[index(i+1,j+0,nx,ny)] +
            (int)(*b[0])[index(i+1,j+1,nx,ny)] +
            (int)(*b[0])[index(i+0,j+1,nx,ny)] +
            (int)(*b[0])[index(i-1,j+1,nx,ny)] +
            (int)(*b[0])[index(i-1,j+0,nx,ny)] +
            (int)(*b[0])[index(i-1,j-1,nx,ny)] +
            (int)(*b[0])[index(i+0,j-1,nx,ny)] +
            (int)(*b[0])[index(i+1,j-1,nx,ny)];
         switch (c)
         {
            case 3:
               (*b[1])[index(i,j,nx,ny)] = true;
               break;
            case 4:
               (*b[1])[index(i,j,nx,ny)] = (*b[0])[index(i,j,nx,ny)];
               break;
            default:
               (*b[1])[index(i,j,nx,ny)] = false;
               break;
         }
      }
   }
}

void ProjectStep(const vector<bool> & b, GridFunction & x, int n)
{
   for (int i=0; i<n; i++)
   {
      x[i] = (double)b[i];
   }
}

void InitBlinker(vector<bool> & b, int nx, int ny, const Array<int> & params)
{
   for (int i=0; i<params.Size()/3; i++)
   {
      int cx   = params[3 * i + 0];
      int cy   = params[3 * i + 1];
      int ornt = params[3 * i + 2];

      switch (ornt % 2)
      {
         case 0:
            b[index(cx+0,cy+1,nx,ny)] = true;
            b[index(cx+0,cy+0,nx,ny)] = true;
            b[index(cx+0,cy-1,nx,ny)] = true;
            break;
         case 1:
            b[index(cx+1,cy+0,nx,ny)] = true;
            b[index(cx+0,cy+0,nx,ny)] = true;
            b[index(cx-1,cy+0,nx,ny)] = true;
            break;
      }
   }
}

void InitGlider(vector<bool> & b, int nx, int ny, const Array<int> & params)
{
   for (int i=0; i<params.Size()/3; i++)
   {
      int cx   = params[3 * i + 0];
      int cy   = params[3 * i + 1];
      int ornt = params[3 * i + 2];

      switch (ornt % 4)
      {
         case 0:
            b[index(cx-1,cy+0,nx,ny)] = true;
            b[index(cx+0,cy+1,nx,ny)] = true;
            b[index(cx+1,cy-1,nx,ny)] = true;
            b[index(cx+1,cy+0,nx,ny)] = true;
            b[index(cx+1,cy+1,nx,ny)] = true;
            break;
         case 1:
            b[index(cx+0,cy-1,nx,ny)] = true;
            b[index(cx-1,cy+0,nx,ny)] = true;
            b[index(cx-1,cy+1,nx,ny)] = true;
            b[index(cx+0,cy+1,nx,ny)] = true;
            b[index(cx+1,cy+1,nx,ny)] = true;
            break;
         case 2:
            b[index(cx+1,cy+0,nx,ny)] = true;
            b[index(cx+0,cy-1,nx,ny)] = true;
            b[index(cx-1,cy-1,nx,ny)] = true;
            b[index(cx-1,cy+0,nx,ny)] = true;
            b[index(cx-1,cy+1,nx,ny)] = true;
            break;
         case 3:
            b[index(cx+0,cy+1,nx,ny)] = true;
            b[index(cx+1,cy+0,nx,ny)] = true;
            b[index(cx-1,cy-1,nx,ny)] = true;
            b[index(cx+0,cy-1,nx,ny)] = true;
            b[index(cx+1,cy-1,nx,ny)] = true;
            break;
      }
   }
}

void InitSketchPad(vector<bool> & b, int nx, int ny, const Array<int> & params)
{
   int cx   = params[0];
   int cy   = params[1];

   int ox = 0;
   int oy = 0;

   for (int i=2; i<params.Size(); i++)
   {
      if ( params[i]/2 == 1 )
      {
         ox = 0;
         oy--;
      }
      else
      {
         b[index(cx+ox,cy+oy,nx,ny)] = (bool)params[i];
         ox++;
      }
   }
}
