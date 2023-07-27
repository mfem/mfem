//                                Contact example
//
// Compile with: make contact
//
// Sample runs:  ./contact -m1 block1.mesh -m2 block2.mesh -at "5 6 7 8"
// Sample runs:  ./contact -m1 block1_d.mesh -m2 block2_d.mesh -at "5 6 7 8"

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "problems/problems.hpp"
#include "util/util.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file1 = "meshes/block1.mesh";
   const char *mesh_file2 = "meshes/rotatedblock2.mesh";
   int order = 1;
   Array<int> attr;
   Array<int> m_attr;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file1, "-m1", "--mesh1",
                  "First mesh file to use.");
   args.AddOption(&mesh_file2, "-m2", "--mesh2",
                  "Second mesh file to use.");
   args.AddOption(&attr, "-at", "--attributes-surf",
                  "Attributes of boundary faces on contact surface for mesh 2.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);


   ElasticityProblem prob1(mesh_file1,order);
   ElasticityProblem prob2(mesh_file2,order);

   ContactProblem contact(&prob1, &prob2);

   int ndofs1 = prob1.GetNumDofs();
   int ndofs2 = prob2.GetNumDofs();
   int ndofs = ndofs1 + ndofs2;

   Array<int> ess_tdofs1 = prob1.GetEssentialDofs();
   Array<int> ess_tdofs2 = prob2.GetEssentialDofs();
   int sz1 = ess_tdofs1.Size();
   int sz2 = ess_tdofs2.Size();
   Array<int> DirichletDofs(sz1+sz2);
   for (int i = 0; i<sz1; i++)
   {
      DirichletDofs[i] = ess_tdofs1[i];
   }
   for (int i = 0; i<sz2; i++)
   {
      DirichletDofs[i+sz1] = ess_tdofs2[i]+ndofs1;
   }

   GridFunction x1 = prob1.GetDisplacementGridFunction();
   GridFunction x2 = prob2.GetDisplacementGridFunction();
   
   Vector d0(ndofs); d0 = 0.0;
   Vector f;
   contact.DdE(d0,f);
   SparseMatrix * K = contact.DddE(d0);
   d0.SetVector(x1,0);
   d0.SetVector(x2,x1.Size());
   
   Vector g0; 
   contact.g(d0,g0, true);
   SparseMatrix * J = contact.Ddg(d0);

   int nconstraints = J->Height();
   Vector temp(nconstraints);
   J->Mult(d0, temp);
   g0.Add(-1.0, temp);

   return 0;
}
