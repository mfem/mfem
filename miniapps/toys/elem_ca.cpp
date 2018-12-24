//                                MFEM Example
//
// Compile with: make ex_ca

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <bitset>
#include <vector>

using namespace std;
using namespace mfem;

//void Update(char * b[], int r, int s);
//void ProjectStep(char b[], GridFunction & x, int ns, int s);

void PrintRule(bitset<8> & r);
void Update(vector<bool> * b[], bitset<8> & r, int ns, int s);
void ProjectStep(const vector<bool> & b, GridFunction & x, int ns, int s);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int ns = 20;
   int  r = 90;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&ns, "-ns", "--num-steps",
                  "Number of steps of the 1D cellular automaton.");
   args.AddOption(&r, "-r", "--rule",
                  "Elementary cellular automaton rule.");
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
   Mesh *mesh = new Mesh(2 * ns + 3, ns, Element::QUADRILATERAL,
                         0, 2 * ns + 3, ns);
   mesh->Print(*(new ofstream("zzz")));
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
   int len = 2 * ns + 3;

   vector<bool> * vbp[2];
   vector<bool> * vbptmp = NULL;
   vector<bool> vb0(len);
   vector<bool> vb1(len);

   vbp[0] = &vb0;
   vbp[1] = &vb1;

   for (int i=0; i<len; i++)
   {
      vb0[i] = false;
      vb1[i] = false;
   }
   vb0[ns+1] = true;
   vb0[0] = true;

   ProjectStep(*vbp[0], x, ns, 0);

   // 7. Create the rule as a bitset and display it for the user
   bitset<8> rbs = r;
   PrintRule(rbs);

   // 8. Apply the rule iteratively
   cout << endl << "Applying rule..." << flush;
   for (int s=1; s<ns; s++)
   {
      Update(vbp, rbs, ns, s);
      ProjectStep(*vbp[1], x, ns, s);

      vbptmp = vbp[0];
      vbp[0] = vbp[1];
      vbp[1] = vbptmp;

      // 9. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         // sol_sock.precision(2);
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

   // 10. Save the refined mesh and the solution. This output can be
   //     viewed later using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
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

void Update(vector<bool> * b[], bitset<8> & r, int ns, int s)
{
   for (int i=ns-s+1; i<=ns+s+1; i++)
   {
      (*b[1])[i] = Rule(r, (*b[0])[i-1], (*b[0])[i], (*b[0])[i+1]);
   }
}

void ProjectStep(const vector<bool> & b, GridFunction & x, int ns, int s)
{
   for (int i=ns-s+1; i<=ns+s+1; i++)
   {
      x[s*(2*ns+3)+i] = (double)b[i];
   }
}
