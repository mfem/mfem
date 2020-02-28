//                                MFEM modified from Example 10 and 16
//
// Compile with: make imMHDp
//
// Description:  It solves a time dependent resistive MHD problem 
//               There are three versions:
//               1. explicit scheme
//               2. implicit scheme using a very simple linear preconditioner
//               3. implicit scheme using physcis-based preconditioner
// Author: QT

#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;

double beta=1e-3; //a global value of magnetude for the pertubation
double Lx;  //size of x domain
double lambda=.5/M_PI;
double ep=.2;
double tau=15.;

double InitialPsi3(const Vector &x)
{
   return -lambda*log( cosh(x(1)/lambda) +ep*cos(x(0)/lambda) )
          +beta*cos(M_PI*.5*x(1))*cos(M_PI*x(0));
}

int main(int argc, char *argv[])
{
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   //++++Parse command-line options.
   const char *mesh_file = "./Meshes/xperiodic-square.mesh";
   int ser_ref_levels = 2;
   int par_ref_levels = 0;
   int order = 2;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refineP",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0) args.PrintOptions(cout);

   //+++++Read the mesh from the given mesh file.    
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   //++++++Refine the mesh to increase the resolution.    
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   H1_FECollection fe_coll(order, dim);
   ParFiniteElementSpace fespace(pmesh, &fe_coll); 

   ParGridFunction psi(&fespace);
   FunctionCoefficient psiInit3(InitialPsi3);
   psi.ProjectCoefficient(psiInit3);
   psi.SetTrueVector();
   psi.SetFromTrueVector(); 

   {
      ostringstream mesh_name, psi_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      psi_name << "sol_psi." << setfill('0') << setw(6) << myid;

      ofstream omesh(mesh_name.str().c_str());
      omesh.precision(8);
      pmesh->Print(omesh);

      ofstream osol3(psi_name.str().c_str());
      osol3.precision(8);
      psi.Save(osol3);
   }

   delete pmesh;
   MPI_Finalize();
   return 0;
}



