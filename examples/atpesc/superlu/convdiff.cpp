#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <cmath>

using namespace std;
using namespace mfem;

double SourceField(const Vector &x);

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   int ref_levels = 1;
   int order = 1;
   double velocity = 100.0;
   bool visit = false;
   bool slu_solver = false;

   OptionsParser args(argc, argv);
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&velocity, "-vel", "--velocity",
                  "Constant velocity in x that the fluid is flowing with.");   
   args.AddOption(&visit, "-v", "--visit", "-nov", "--no-visit", 
                  "Enable VisIt visualization.");
#ifdef MFEM_USE_SUPERLU
   args.AddOption(&slu_solver, "-slu", "--superlu", "-no-slu", "--no-superlu", 
                  "Use the SuperLU Solver.");
#endif
   args.Parse();
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }


   // 3. Read the (serial) mesh from the given mesh file on all processors. We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(100, 100, Element::QUADRILATERAL, 1, 1.0, 1.0);
   int dim = mesh->Dimension();

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution (1 time by
   //    default, or specified on the command line with -rp). Once the parallel
   //    mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
   }
   else if (pmesh->GetNodes())
   {
      fec = pmesh->GetNodes()->OwnFEC();
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
   }
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << size << endl;
   }

   // 7. Set up the parallel bilinear form representing the convection-diffusion system.
   Array<int> ess_tdof_list;
   if (pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   ConstantCoefficient diffcoef(1.0);
   Vector V(dim);
   V = 0.0;
   V[0] = velocity;
   VectorConstantCoefficient velocitycoef(V);
   ParBilinearForm *cd = new ParBilinearForm(fespace);
   cd->AddDomainIntegrator(new ConvectionIntegrator(velocitycoef));
   cd->AddDomainIntegrator(new DiffusionIntegrator(diffcoef));
   cd->Assemble();

   FunctionCoefficient source(SourceField);
   ParLinearForm *b = new ParLinearForm(fespace);
   b->AddDomainIntegrator(new DomainLFIntegrator(source));
   b->Assemble();

   HypreParMatrix CD;
   ParGridFunction x(fespace);
   x = 0.0;
   Vector B, X;
   cd->FormLinearSystem(ess_tdof_list, x, *b, CD, X, B);

   // 8. Define and configure the solver.  We will use HYPRE
   //    with  BoomerAMG as the preconditioner, or use 
   //    SuperLU which will handle the full solve in one go.
   Solver *solver = NULL;
   Operator *Mrow = NULL;
   HypreBoomerAMG *amg = NULL;
   if (!slu_solver)
   {
      HypreBoomerAMG *amg = new HypreBoomerAMG(CD);
      amg->SetPrintLevel(0);
      HypreGMRES *gmres = new HypreGMRES(CD);
      gmres->SetTol(1e-12);
      gmres->SetMaxIter(200);
      gmres->SetPrintLevel(2);
      gmres->SetPreconditioner(*amg);
      solver = gmres;
   }
   else
   {
#ifdef MFEM_USE_SUPERLU
      Mrow = new SuperLURowLocMatrix(CD);
      SuperLUSolver *superlu = new SuperLUSolver(MPI_COMM_WORLD);
      superlu->SetPrintStatistics(false);
      superlu->SetSymmetricPattern(false);
      superlu->SetColumnPermutation(superlu::PARMETIS);
      superlu->SetOperator(*Mrow);
      solver = superlu;
#endif
   }

   // 9. Complete the solve and recover the concentration in the grid function
   solver->Mult(B, X);
   cd->RecoverFEMSolution(X, *b, x);

   // 10. Dump the concentrantion values out to a visit file
   if (visit)
   {
      VisItDataCollection visit_dc("dump", pmesh);
      visit_dc.RegisterField("concentration", &x);
      visit_dc.Save();
   }

   // 11. Free the used memory.
   delete cd;
   delete b;

   delete solver;
   if (slu_solver)
   {
#if defined(MFEM_USE_SUPERLU)
      delete Mrow;
#endif
   }
   else
   {
      delete amg;
   }

   delete fespace;
   if (order > 0)
   {
      delete fec;
   }
   delete pmesh;

   MPI_Finalize();

   return 0;
}


//This will represent a disc of constant rate input at (0.5, 0.5)
double SourceField(const Vector &x)
{
   double R = 0.0;
   if (abs(x[0] - 0.5) < 0.05 && abs(x[1] - 0.5) < 0.05)
   {
      R = 1.0;
   }

   return R;
}