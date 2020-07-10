//                                MFEM Example 14
//
// Compile with: make ex14
//
// Sample runs:  ex14 -m ../data/inline-quad.mesh -o 0
//               ex14 -m ../data/star.mesh -r 4 -o 2
//               ex14 -m ../data/star-mixed.mesh -r 4 -o 2
//               ex14 -m ../data/escher.mesh -s 1
//               ex14 -m ../data/fichera.mesh -s 1 -k 1
//               ex14 -m ../data/fichera-mixed.mesh -s 1 -k 1
//               ex14 -m ../data/square-disc-p2.vtk -r 3 -o 2
//               ex14 -m ../data/square-disc-p3.mesh -r 2 -o 3
//               ex14 -m ../data/square-disc-nurbs.mesh -o 1
//               ex14 -m ../data/disc-nurbs.mesh -r 3 -o 2 -s 1 -k 0
//               ex14 -m ../data/pipe-nurbs.mesh -o 1
//               ex14 -m ../data/inline-segment.mesh -r 5
//               ex14 -m ../data/amr-quad.mesh -r 3
//               ex14 -m ../data/amr-hex.mesh
//               ex14 -m ../data/fichera-amr.mesh
//
// Description:  This example code demonstrates the use of MFEM to define a
//               discontinuous Galerkin (DG) finite element discretization of
//               the Laplace problem -Delta u = 1 with homogeneous Dirichlet
//               boundary conditions. Finite element spaces of any order,
//               including zero on regular grids, are supported. The example
//               highlights the use of discontinuous spaces and DG-specific face
//               integrators.
//
//               We recommend viewing examples 1 and 9 before viewing this
//               example.

#include "mfem.hpp"
#include "conv_rates.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double u_exact(const Vector &x);
double f_exact(const Vector &x);
void gradu_exact(const Vector &x, Vector &gradu);
Vector alpha;
int dim;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int ref_levels = -1;
   int order = 1;
   double sigma = -1.0;
   double kappa = -1.0;
   bool visualization = 1;
   int sr = 1;
   int pr = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&sr, "-sr", "--serial_ref",
                  "Number of serial refinements.");
   args.AddOption(&pr, "-pr", "--parallel_ref",
                  "Number of serial refinements.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&sigma, "-s", "--sigma",
                  "One of the two DG penalty parameters, typically +1/-1."
                  " See the documentation of class DGDiffusionIntegrator.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "One of the two DG penalty parameters, should be positive."
                  " Negative values are replaced with (order+1)^2.");
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
      MPI_Finalize();
      return 1;
   }
   if (kappa < 0)
   {
      kappa = (order+1)*(order+1);
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
   //    NURBS meshes are projected to second order meshes.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();

   for (int l = 0; l < sr; l++)
   {
      mesh->UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // 3. Set up parameters for exact solution
   alpha.SetSize(dim); // x,y,z coefficients of the solution
   for (int i=0; i<dim; i++) { alpha(i) = M_PI*(double)(i+1);}


   // 4. Define a finite element space on the mesh. Here we use discontinuous
   //    finite elements of the specified order >= 0.
   FiniteElementCollection *fec = new DG_FECollection(order, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);

   // 6. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero.
   ParGridFunction x(fespace);
   x = 0.0;
   // 5. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system.
   ParLinearForm *b = new ParLinearForm(fespace);
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   FunctionCoefficient u_ex(u_exact);
   FunctionCoefficient f(f_exact);
   b->AddDomainIntegrator(new DomainLFIntegrator(f));
   b->AddBdrFaceIntegrator(
      new DGDirichletLFIntegrator(u_ex, one, sigma, kappa));

   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   a->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
   a->AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));

   cout << myid << ": " << fespace->GetTrueVSize() << endl;

   Convergence rates(MPI_COMM_WORLD);
   rates.Clear();
   VectorFunctionCoefficient u_grad(dim,gradu_exact);

   for (int l = 0; l <= pr; l++)
   {
      b->Assemble();
      a->Assemble();
      a->Finalize();

      HypreParMatrix *A = a->ParallelAssemble();
      HypreParVector *B = b->ParallelAssemble();
      HypreParVector *X = x.ParallelProject();

      HypreBoomerAMG *amg = new HypreBoomerAMG(*A);
      amg->SetPrintLevel(0);
      if (sigma == -1.0)
      {
         HyprePCG pcg(*A);
         pcg.SetTol(1e-12);
         pcg.SetMaxIter(200);
         pcg.SetPrintLevel(0);
         pcg.SetPreconditioner(*amg);
         pcg.Mult(*B, *X);
      }
      else
      {
         GMRESSolver gmres(MPI_COMM_WORLD);
         gmres.SetAbsTol(0.0);
         gmres.SetRelTol(1e-12);
         gmres.SetMaxIter(200);
         gmres.SetKDim(10);
         gmres.SetPrintLevel(0);
         gmres.SetOperator(*A);
         gmres.SetPreconditioner(*amg);
         gmres.Mult(*B, *X);
      }
      delete amg;

      // // 12. Extract the parallel grid function corresponding to the finite element
      // //     approximation X. This is the local solution on each processor.
      x = *X;

      rates.AddGridFunction(&x,&u_ex,&u_grad,&one);
      // cout << myid << ": " << x.ComputeDGFaceJumpError(&u_ex, &one, 1.0) << endl;

      if (l==pr) break;

      pmesh->UniformRefinement();
      fespace->Update();
      a->Update();
      b->Update();
      x.Update();
   }
   rates.Print();

   cout << "nr sharedfaces = " << pmesh->GetNSharedFaces() << endl;
   cout << "nr faces = " << pmesh->GetNumFaces() << endl;

   // 10. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x << flush;
   }

   // 11. Free the used memory.
   delete a;
   delete b;
   delete fespace;
   delete fec;
   delete pmesh;
   MPI_Finalize();
   return 0;
}

double f_exact(const Vector &x)
{
   double s = 0.0;
   for (int i=0; i<dim; i++)
   {
      s+= alpha(i) * x(i);
   }
   double d2u = 0.0;
   for (int i=0;i<dim; i++)
   {
      d2u += -alpha(i)*alpha(i)*cos(s);
   }
   return -d2u;
}

double u_exact(const Vector &x)
{
   double u;
   double y=0;
   for (int i=0; i<dim; i++)
   {
      y+= alpha(i) * x(i);
   }
   u = cos(y);
   return u;
}

void gradu_exact(const Vector &x, Vector &du)
{
   double s=0.0;
   for (int i=0; i<dim; i++)
   {
      s+= alpha(i) * x(i);
   }
   for (int i=0; i<dim; i++)
   {
      du[i] = -alpha(i) * sin(s);
   }
}
