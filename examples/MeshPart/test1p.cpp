
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void SolveEx1p(ParMesh & pmesh, int order,MPI_Session & mpi);


int main(int argc, char *argv[])
{
   MPI_Session mpi;
   int num_procs = mpi.WorldSize();
   int myid = mpi.WorldRank();

   // 1. Parse command line options
   const char *mesh_file = "../../data/periodic-annulus-sector.msh";
   int order = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.ParseCheck();

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   Mesh orig_mesh(mesh_file);
   orig_mesh.CheckElementOrientation(true);
   orig_mesh.CheckBdrElementOrientation(true);

   ParMesh orig_pmesh(MPI_COMM_WORLD, orig_mesh);



   // orig_pmesh.Save("orig_mesh.mesh");

   // mesh.EnsureNodes();
   Array<int> elems;
   if (myid == 0)
   {
      elems.Append(1);
      elems.Append(3);
   }
   else if (myid == 1)
   {
      elems.Append(0);
      elems.Append(4);
   }
   else if (myid == 2)
   {
      elems.Append(0);
      elems.Append(4);
   }
   // mesh.UniformRefinement();
   // Array<int> elems({1,3,21,10,20,2,0});
   int nel = orig_pmesh.GetNE();
   // int nel = elems.Size();
   // elems.SetSize(nel);
   // for (int i = 0; i<nel; i++)
   // {
   //    elems[i] = i;
   // }
   // elems.Print();

   // ParMesh new_pmesh = ParMesh::MakeRefined(orig_pmesh,1,1);
   ParMesh new_pmesh = ParMesh::ExtractMesh(orig_pmesh,elems);
   new_pmesh.CheckElementOrientation(true);
   new_pmesh.CheckBdrElementOrientation(true);
   cout << "new mesh extracted " << endl;
   // new_pmesh.Save("new_mesh.mesh");


   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream mesh0_sock(vishost, visport);
      mesh0_sock << "parallel " << num_procs << " " << myid << "\n";
      mesh0_sock.precision(8);
      mesh0_sock << "mesh\n" << orig_pmesh << "keys n \n" << flush;

      socketstream mesh1_sock(vishost, visport);
      mesh1_sock << "parallel " << num_procs << " " << myid << "\n";
      mesh1_sock.precision(8);
      mesh1_sock << "mesh\n" << new_pmesh << "keys n \n" << flush;
   }


   // SolveEx1p(orig_pmesh,2,mpi);
   // SolveEx1p(new_pmesh,2,mpi);

   cout << "myid = " << myid << ", NGroups = " << orig_pmesh.GetNGroups() << endl;



   return 0;
}


void SolveEx1p(ParMesh & pmesh, int order, MPI_Session & mpi)
{

   int num_procs = mpi.WorldSize();
   int myid = mpi.WorldRank();

   int dim = pmesh.Dimension();
   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   ParFiniteElementSpace fespace(&pmesh, fec);
   HYPRE_BigInt size = fespace.GlobalTrueVSize();

   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   Array<int> ess_tdof_list;
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   ParLinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   ParGridFunction x(&fespace);
   x = 0.0;

   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   HypreBoomerAMG *prec = new HypreBoomerAMG;

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetPreconditioner(*prec);
   cg.SetOperator(*A);
   cg.Mult(B, X);
   delete prec;

   a.RecoverFEMSolution(X, b, x);

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock << "parallel " << num_procs << " " << myid << "\n";
   sol_sock.precision(8);
   sol_sock << "solution\n" << pmesh << x << flush;

   delete fec;

}