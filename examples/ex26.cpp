//                                MFEM Example 26
//
// Compile with: make ex26
//
// Sample runs:  ex26 -m ../data/star.mesh
//               ex26 -m ../data/fichera.mesh
//               ex26 -m ../data/beam-hex.mesh
//
// Device sample runs:
//               ex26 -d cuda
//               ex26 -d raja-cuda
//               ex26 -d occa-cuda
//               ex26 -d raja-omp
//               ex26 -d occa-omp
//               ex26 -d ceed-cpu
//               ex26 -d ceed-cuda
//               ex26 -m ../data/beam-hex.mesh -d cuda
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions
//               as in Example 1.
//
//               It highlights on the creation of a hierarchy of discretization
//               spaces with partial assembly and the construction of an
//               efficient multigrid preconditioner for the iterative solver.
//
//               We recommend viewing Example 1 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Class for constructing a multigrid preconditioner for the diffusion operator.
// This example multigrid preconditioner class demonstrates the creation of the
// diffusion bilinear forms and operators using partial assembly for all spaces
// in the FiniteElementSpaceHierarchy. The preconditioner uses a CG solver on
// the coarsest level and second order Chebyshev accelerated smoothers on the
// other levels.
class DiffusionMultigrid : public GeometricMultigrid
{
private:
   ConstantCoefficient coeff;

public:
   // Constructs a diffusion multigrid for the given FiniteElementSpaceHierarchy
   // and the array of essential boundaries
   DiffusionMultigrid(FiniteElementSpaceHierarchy& fespaces, Array<int>& ess_bdr)
      : GeometricMultigrid(fespaces, ess_bdr), coeff(1.0)
   {
      ConstructCoarseOperatorAndSolver(fespaces.GetFESpaceAtLevel(0));
      for (int level = 1; level < fespaces.GetNumLevels(); ++level)
      {
         ConstructOperatorAndSmoother(fespaces.GetFESpaceAtLevel(level), level);
      }
   }

private:
   void ConstructBilinearForm(FiniteElementSpace& fespace)
   {
      BilinearForm* form = new BilinearForm(&fespace);
      form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      form->AddDomainIntegrator(new DiffusionIntegrator(coeff));
      form->Assemble();
      bfs.Append(form);
   }

   void ConstructCoarseOperatorAndSolver(FiniteElementSpace& coarse_fespace)
   {
      ConstructBilinearForm(coarse_fespace);

      OperatorPtr opr;
      opr.SetType(Operator::ANY_TYPE);
      bfs[0]->FormSystemMatrix(*essentialTrueDofs[0], opr);
      opr.SetOperatorOwner(false);

      CGSolver* pcg = new CGSolver();
      pcg->SetPrintLevel(-1);
      pcg->SetMaxIter(200);
      pcg->SetRelTol(sqrt(1e-4));
      pcg->SetAbsTol(0.0);
      pcg->SetOperator(*opr.Ptr());

      AddLevel(opr.Ptr(), pcg, true, true);
   }

   void ConstructOperatorAndSmoother(FiniteElementSpace& fespace, int level)
   {
      const Array<int> &ess_tdof_list = *essentialTrueDofs[level];
      ConstructBilinearForm(fespace);

      OperatorPtr opr;
      opr.SetType(Operator::ANY_TYPE);
      bfs[level]->FormSystemMatrix(ess_tdof_list, opr);
      opr.SetOperatorOwner(false);

      Vector diag(fespace.GetTrueVSize());
      bfs[level]->AssembleDiagonal(diag);

      Solver* smoother = new OperatorChebyshevSmoother(*opr, diag, ess_tdof_list, 2);
      AddLevel(opr.Ptr(), smoother, true, true);
   }
};


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int geometric_refinements = 0;
   int order_refinements = 2;
   const char *device_config = "cpu";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&geometric_refinements, "-gr", "--geometric-refinements",
                  "Number of geometric refinements done prior to order refinements.");
   args.AddOption(&order_refinements, "-or", "--order-refinements",
                  "Number of order refinements. Finest level in the hierarchy has order 2^{or}.");
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

   // 5. Define a finite element space hierarchy on the mesh. Here we use
   //    continuous Lagrange finite elements. We start with order 1 on the
   //    coarse level and geometrically refine the spaces by the specified
   //    amount. Afterwards, we increase the order of the finite elements
   //    by a factor of 2 for each additional level.
   FiniteElementCollection *fec = new H1_FECollection(1, dim);
   FiniteElementSpace *coarse_fespace = new FiniteElementSpace(mesh, fec);
   FiniteElementSpaceHierarchy fespaces(mesh, coarse_fespace, true, true);

   Array<FiniteElementCollection*> collections;
   collections.Append(fec);
   for (int level = 0; level < geometric_refinements; ++level)
   {
      fespaces.AddUniformlyRefinedLevel();
   }
   for (int level = 0; level < order_refinements; ++level)
   {
      collections.Append(new H1_FECollection((int)std::pow(2, level+1), dim));
      fespaces.AddOrderRefinedLevel(collections.Last());
   }

   cout << "Number of finite element unknowns: "
        << fespaces.GetFinestFESpace().GetTrueVSize() << endl;

   // 6. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   LinearForm *b = new LinearForm(&fespaces.GetFinestFESpace());
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   // 7. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(&fespaces.GetFinestFESpace());
   x = 0.0;

   // 8. Create the multigrid operator using the previously created
   //    FiniteElementSpaceHierarchy and additional boundary information. This
   //    operator is then used to create the MultigridSolver as a preconditioner
   //    in the iterative solver.
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;

   DiffusionMultigrid M(fespaces, ess_bdr);
   M.SetCycleType(Multigrid::CycleType::VCYCLE, 1, 1);

   OperatorPtr A;
   Vector B, X;
   M.FormFineLinearSystem(x, *b, A, X, B);
   cout << "Size of linear system: " << A->Height() << endl;

   // 9. Solve the linear system A X = B.
   PCG(*A, M, B, X, 1, 2000, 1e-12, 0.0);

   // 10. Recover the solution as a finite element grid function.
   M.RecoverFineFEMSolution(X, *b, x);

   // 11. Save the refined mesh and the solution. This output can be viewed
   //     later using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   fespaces.GetFinestFESpace().GetMesh()->Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   // 12. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *fespaces.GetFinestFESpace().GetMesh() << x <<
               flush;
   }

   // 13. Free the used memory.
   delete b;
   for (int level = 0; level < collections.Size(); ++level)
   {
      delete collections[level];
   }

   return 0;
}
