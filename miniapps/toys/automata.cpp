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
//         ----------------------------------------------------
//         Automata Miniapp:  Model of simple cellular automata
//         ----------------------------------------------------
//
// This miniapp implements a one dimensional elementary cellular automata
// as described in: mathworld.wolfram.com/ElementaryCellularAutomaton.html
//
// This miniapp shows a completely unnecessary use of the finite element
// method to simply display binary data (but it's fun to play with).
//
// Compile with: make automata
//
// Sample runs: automata
//              automata -r 110 -ns 32
//              automata -r 30 -ns 96

#include "mfem.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <bitset>
#include <vector>

using namespace std;
using namespace mfem;

void PrintRule(bitset<8> & r);
void ApplyRule(vector<bool> * b[], bitset<8> & r, int ns, int s);
void ProjectStep(const vector<bool> & b, GridFunction & x, int ns, int s);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int ns = 16;
   int  r = 90;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&ns, "-ns", "--num-steps",
                  "Number of steps of the 1D cellular automaton.");
   args.AddOption(&r, "-r", "--rule",
                  "Elementary cellular automaton rule [0-255].");
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

   // 2. Build a rectangular mesh of quadrilateral elements nearly twice
   //    as wide as it is high.
   Mesh *mesh = new Mesh(2 * ns - 1, ns, Element::QUADRILATERAL,
                         0, 2 * ns - 1, ns, false);

   // 3. Define a finite element space on the mesh. Here we use discontinuous
   //    Lagrange finite elements of order zero i.e. piecewise constant basis
   //    functions.
   FiniteElementCollection *fec = new L2_FECollection(0, 2);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

   // 4. Initialize a pair of bit arrays to store two rows in the evolution
   //    of our cellular automaton.
   int len = 2 * ns - 1;

   vector<bool> * vbp[2];
   vector<bool> vb0(len);
   vector<bool> vb1(len);

   vbp[0] = &vb0;
   vbp[1] = &vb1;

   for (int i=0; i<len; i++)
   {
      vb0[i] = false;
      vb1[i] = false;
   }
   vb0[ns-1] = true;

   // 5. Define the vector x as a finite element grid function corresponding
   //    to fespace which will be used to visualize the cellular automata.
   //    Initialize x with initial condition of zero, which indicates a
   //    "white" or "off" cell in our automaton.
   GridFunction x(fespace);
   x = 0.0;

   // 6. Open a socket to GLVis to visualize the automaton.
   socketstream sol_sock;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sol_sock.open(vishost, visport);
   }

   // 7. Create the rule as a bitset and display it for the user
   bitset<8> rbs = r;
   PrintRule(rbs);

   // Transfer the current row of the automaton to the vector x.
   ProjectStep(*vbp[0], x, ns, 0);

   // 8. Apply the rule iteratively
   cout << endl << "Applying rule..." << flush;
   for (int s=1; s<ns; s++)
   {
      // Compute the next row from the current row
      ApplyRule(vbp, rbs, ns, s);

      // Transfer the new row of the automaton to the vector x.
      ProjectStep(*vbp[1], x, ns, s);

      // Swap bit arrays
      std::swap(vbp[0], vbp[1]);

      // 9. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         sol_sock << "solution\n" << *mesh << x << flush;
         {
            static int once = 1;
            if (once)
            {
               sol_sock << "keys Ajl\n";
               sol_sock << "view 0 180\n";
               sol_sock << "zoom 2.2\n";
               sol_sock << "palette 24\n";
               once = 0;
            }
         }
      }
   }
   cout << "done." << endl;

   // 10. Save the mesh and the final state of the automaton. This output can be
   //     viewed later using GLVis: "glvis -m automata.mesh -g automata.gf".
   ofstream mesh_ofs("automata.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);
   ofstream sol_ofs("automata.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   // 11. Free the used memory.
   delete fespace;
   delete fec;
   delete mesh;

   return 0;
}

bool Rule(bitset<8> & r, bool b0, bool b1, bool b2)
{
   return r[(b0 ? 1 : 0) + (b1 ? 2 : 0) + (b2 ? 4 : 0)];
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

void ApplyRule(vector<bool> * b[], bitset<8> & r, int ns, int s)
{
   for (int i=0; i<2*ns-1; i++)
   {
      int i0 = (i + 2 * ns - 2) % (2 * ns - 1);
      int i2 = (i + 1) % (2 * ns - 1);
      (*b[1])[i] = Rule(r, (*b[0])[i0], (*b[0])[i], (*b[0])[i2]);
   }
}

void ProjectStep(const vector<bool> & b, GridFunction & x, int ns, int s)
{
   for (int i=0; i<2*ns-1; i++)
   {
      x[s*(2*ns-1)+i] = (double)b[i];
   }
}
