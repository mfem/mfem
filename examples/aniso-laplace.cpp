#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

enum problem
{
   Lap2D, // Laplacian
   ALap2D, // Anisotropic
   Rlap2D, // Rotated Anisotropic
   FRegLap2D // Four Region Domain
};


int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();
   // 2. Parse command-line options.
   const char *mesh_file = "../data/ref-square.mesh";
   int order = 1;
   int ref = 1;
   bool visualization = true;
   int iprob = 0;
   problem prob;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&iprob, "-prob", "--problem", "Problem case"
                  " 0: Laplacian2D, 1: Anisotropic Laplacian 2D"
                  "2: Rotated anisotropic Laplacian 2D"
                  "4 region domain Laplacian 2D");
   args.AddOption(&ref, "-r", "--refinements",
                  "Number of mesh refinements");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   prob = (problem)iprob;

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   for (int l = 0; l < ref; l++)
   {
      mesh.UniformRefinement();
   }

   //             3
   //      _______________
   //     |       |       |
   //     |   4   |   3   |
   //   4 |_______|_______| 2
   //     |       |       |
   //     |   1   |   2   |
   //     |_______|_______|
   //             1
   // set attributes
   // Element attributes
   Vector center;
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      mesh.GetElementCenter(i,center);
      if (center(0)<0.5 && center(1)<0.5)
      {
         mesh.SetAttribute(i,1);
      }
      else if (center(0)>0.5 && center(1)<0.5)
      {
         mesh.SetAttribute(i,2);
      }
      else if (center(0)<0.5 && center(1)>0.5)
      {
         mesh.SetAttribute(i,4);
      }
      else
      {
         mesh.SetAttribute(i,3);
      }
   }
   mesh.SetAttributes();


   // Boundary attributes;
   for (int i = 0; i< mesh.GetNBE(); i++)
   {
      int geom = mesh.GetBdrElementBaseGeometry(i);
      ElementTransformation * T = mesh.GetBdrElementTransformation(i);
      T->Transform(Geometries.GetCenter(geom), center);
      if (center(0) == 0.0)
      {
         mesh.SetBdrAttribute(i, 4);
      }
      else if (center(0) == 1.0)
      {
         mesh.SetBdrAttribute(i,2);
      }
      else if (center(1) == 0.0)
      {
         mesh.SetBdrAttribute(i,1);
      }
      else
      {
         mesh.SetBdrAttribute(i,3);
      }
   }
   mesh.SetAttributes();

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   PWMatrixCoefficient K(dim);
   int nattr = pmesh.attributes.Max();
   Vector dvalues(nattr);
   Array<MatrixCoefficient*> Kcoeffs(nattr);
   Array<DenseMatrix*> kmats(nattr);
   Vector epsilons(nattr);
   Vector thetas(nattr);

   switch (prob)
   {
      case problem::Lap2D:
      {
         dvalues = 0.0;
         epsilons = 1.0;
         thetas = 0.0;
      }
      break;
      case problem::ALap2D:
      {
         dvalues = 0.0;
         epsilons = 0.01;
         thetas = 0.0;
      }
      break;
      case problem::Rlap2D:
      {
         dvalues = 0.0;
         epsilons = 0.01;
         thetas = M_PI/3.0;
      }
      break;
      default:
      {
         dvalues = 0.0;
         dvalues[0] = 1e4;
         epsilons[0] = 1.0; thetas[0] = 0.0;
         epsilons[1] = 1.0; thetas[1] = 0.0;
         epsilons[2] = 0.01; thetas[2] = 0.0;
         epsilons[3] = 0.01; thetas[3] = M_PI/2;
      }
      break;
   }
   PWConstCoefficient  d(dvalues);

   for (int i = 0; i<nattr; i++)
   {
      kmats[i] = new DenseMatrix(dim);
      (*kmats[i])(0,0) = cos(thetas(i)) * cos(thetas(i))
                         + epsilons(i) * sin(thetas(i)) * sin(thetas(i));
      (*kmats[i])(1,1) = epsilons(i) * cos(thetas(i)) * cos(thetas(i))
                         + sin(thetas(i)) * sin(thetas(i));
      (*kmats[i])(0,1) = (1.0 - epsilons(i)) * cos(thetas(i)) * sin(thetas(i));
      (*kmats[i])(1,0) = (*kmats[i])(0,1);
      Kcoeffs[i] = new MatrixConstantCoefficient(*kmats[i]);
      K.UpdateCoefficient(i+1,*Kcoeffs[i]);
   }

   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fespace(&pmesh, &fec);
   HYPRE_BigInt size = fespace.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   Array<int> ess_tdof_list;
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 0;
      switch (prob)
      {
         case problem::Lap2D:
         case problem::ALap2D:
         case problem::Rlap2D:
         {
            ess_bdr[1] = 1;
            ess_bdr[3] = 1;
         }
         break;
         default:
         {
            ess_bdr = 1;
         }
         break;
      }
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   ParLinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   ParGridFunction x(&fespace);
   x = 0.0;

   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator(K));
   a.AddDomainIntegrator(new MassIntegrator(d));
   a.Assemble();

   for (int i = 0; i<nattr; i++)
   {
      delete kmats[i];
      delete Kcoeffs[i];
   }

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   HypreBoomerAMG amg;
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetPreconditioner(amg);
   cg.SetOperator(*A);
   cg.Mult(B, X);

   a.RecoverFEMSolution(X, b, x);

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << x << flush;
   }

   return 0;

}