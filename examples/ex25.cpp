//                                MFEM Example 25
//
// Compile with: make ex25
//
// Sample runs:  ex25 -m ../data/star.mesh
//               ex25 -m ../data/fichera.mesh
//               ex25 -m ../data/beam-hex.mesh
//
// Device sample runs:
//               ex25 -d cuda
//               ex25 -d raja-cuda
//               ex25 -d occa-cuda
//               ex25 -d raja-omp
//               ex25 -d occa-omp
//               ex25 -d ceed-cpu
//               ex25 -d ceed-cuda
//               ex25 -m ../data/beam-hex.mesh -d cuda
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions
//               as in example 1. It highlights on the creation of a hierarchy
//               of discretization spaces with partial assembly and the
//               construction of an efficient p-multigrid preconditioner for the
//               iterative solver.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class MultigridDiffusionOperator : public MultigridOperator
{
private:
   Array<BilinearForm*> bfs;
   Array<Array<int>*> essentialTrueDofs;
   ConstantCoefficient one;

public:
   /// Constructor for a multigrid diffusion operator for a given SpaceHierarchy. Uses Chebyshev accelerated smoothing.
   MultigridDiffusionOperator(SpaceHierarchy& spaceHierarchy,
                              Array<int>& ess_bdr, int chebyshevOrder = 2)
      : one(1.0)
   {
      BilinearForm* form = new BilinearForm(&spaceHierarchy.GetFESpaceAtLevel(0));
      form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      AddIntegrators(form);
      form->Assemble();
      bfs.Append(form);

      essentialTrueDofs.Append(new Array<int>());
      spaceHierarchy.GetFESpaceAtLevel(0).GetEssentialTrueDofs(
         ess_bdr, *essentialTrueDofs.Last());

      OperatorPtr opr;
      opr.SetType(Operator::ANY_TYPE);
      form->FormSystemMatrix(*essentialTrueDofs.Last(), opr);
      opr.SetOperatorOwner(false);

      CGSolver* pcg = new CGSolver();
      pcg->SetPrintLevel(0);
      pcg->SetMaxIter(200);
      pcg->SetRelTol(sqrt(1e-4));
      pcg->SetAbsTol(0.0);
      pcg->SetOperator(*opr.Ptr());

      AddCoarsestLevel(opr.Ptr(), pcg, true, true);

      for (int level = 1; level < spaceHierarchy.GetNumLevels(); ++level)
      {
         BilinearForm* form = new BilinearForm(&spaceHierarchy.GetFESpaceAtLevel(level));
         form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
         AddIntegrators(form);
         form->Assemble();
         bfs.Append(form);

         essentialTrueDofs.Append(new Array<int>());
         spaceHierarchy.GetFESpaceAtLevel(level).GetEssentialTrueDofs(
            ess_bdr, *essentialTrueDofs.Last());

         OperatorPtr opr;
         opr.SetType(Operator::ANY_TYPE);
         form->FormSystemMatrix(*essentialTrueDofs.Last(), opr);
         opr.SetOperatorOwner(false);

         Vector diag(spaceHierarchy.GetFESpaceAtLevel(level).GetTrueVSize());
         form->AssembleDiagonal(diag);

         Solver* smoother = new OperatorChebyshevSmoother(
            opr.Ptr(), diag, *essentialTrueDofs.Last(), chebyshevOrder);

         Operator* P =
            new TransferOperator(spaceHierarchy.GetFESpaceAtLevel(level - 1),
                                 spaceHierarchy.GetFESpaceAtLevel(level));

         AddLevel(opr.Ptr(), smoother, P, true, true, true);
      }
   }

   virtual ~MultigridDiffusionOperator()
   {
      for (int i = 0; i < bfs.Size(); ++i)
      {
         delete bfs[i];
      }

      for (int i = 0; i < essentialTrueDofs.Size(); ++i)
      {
         delete essentialTrueDofs[i];
      }
   }

   void EliminateBCs(Vector& x, Vector& b, Vector& X, Vector& B)
   {
      OperatorPtr oper;
      bfs.Last()->FormLinearSystem(*essentialTrueDofs.Last(), x, b,
                                   oper, X, B);
   }

   void RecoverFEMSolution(const Vector& X, const Vector& b, Vector& x)
   {
      bfs.Last()->RecoverFEMSolution(X, b, x);
   }

private:
   void AddIntegrators(BilinearForm* form)
   {
      form->AddDomainIntegrator(new DiffusionIntegrator(one));
   }
};

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int orderrefinements = 2;
   const char *device_config = "cpu";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&orderrefinements, "-or", "--orderrefinements",
                  "Number of order refinements. Finest level in the hierarchy has order 2^{or}");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
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

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(5000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec = new H1_FECollection(1, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

   Array<FiniteElementCollection*> collections;
   collections.Append(fec);
   SpaceHierarchy spaceHierarchy(mesh, fespace, true, true);
   for (int level = 0; level < orderrefinements; ++level)
   {
      collections.Append(new H1_FECollection(std::pow(2, level+1), dim));
      spaceHierarchy.AddOrderRefinedLevel(collections.Last());
   }

   cout << "Number of finite element unknowns: "
        << spaceHierarchy.GetFinestFESpace().GetTrueVSize() << endl;

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   LinearForm *b = new LinearForm(&spaceHierarchy.GetFinestFESpace());
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(&spaceHierarchy.GetFinestFESpace());
   x = 0.0;

   Vector B, X;
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;
   MultigridDiffusionOperator mgOperator(spaceHierarchy, ess_bdr);
   MultigridSolver M(&mgOperator, MultigridSolver::CycleType::VCYCLE, 1, 1);

   cout << "Size of linear system: " <<
        mgOperator.GetOperatorAtFinestLevel()->Height() << endl;

   mgOperator.EliminateBCs(x, *b, X, B);

   // 12. Solve the linear system A X = B.
   PCG(mgOperator, M, B, X, 1, 2000, 1e-12, 0.0);

   // 13. Recover the solution as a finite element grid function.
   mgOperator.RecoverFEMSolution(X, *b, x);

   // 14. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   // 15. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }

   // 16. Free the used memory.
   delete b;
   for (int level = 0; level < collections.Size(); ++level)
   {
      delete collections[level];
   }

   return 0;
}
