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

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 4;
   const char *device_config = "cpu";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
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
   FiniteElementCollection *fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
   }
   else if (mesh->GetNodes())
   {
      fec = mesh->GetNodes()->OwnFEC();
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
   }
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   if (mesh->bdr_attributes.Size())
   {
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   LinearForm *b = new LinearForm(fespace);
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x = 0.0;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   BilinearForm *a = new BilinearForm(fespace);
   a->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));

   // 10. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR, etc.
   a->Assemble();

   OperatorPtr A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   cout << "Size of linear system: " << A->Height() << endl;

   // 11. Set up the p-multigrid preconditioner
   Array<int> orders;
   orders.Append(order);
   int coarseOrder = order / 2;
   while (coarseOrder > 0)
   {
      orders.Append(coarseOrder);
      coarseOrder /= 2;
   }
   orders.Sort();
   SpaceHierarchy spaceHierarchy;
   Array<H1_FECollection*> collections;
   for (int level = 0; level < orders.Size() - 1; ++level)
   {
      collections.Append(new H1_FECollection(orders[level], dim));
      FiniteElementSpace *fesp = new FiniteElementSpace(mesh, collections.Last());
      spaceHierarchy.AddLevel(mesh, fesp, false, true);
   }
   spaceHierarchy.AddLevel(mesh, fespace, false, false);
   MultigridBilinearForm mgOperator(spaceHierarchy, *a, ess_bdr);
   MultigridSolver M(&mgOperator, MultigridSolver::CycleType::VCYCLE, 1, 1);

   // 12. Solve the linear system A X = B.
   PCG(mgOperator, M, B, X, 1, 2000, 1e-12, 0.0);

   // 13. Recover the solution as a finite element grid function.
   a->RecoverFEMSolution(X, *b, x);

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
   for (int level = 0; level < orders.Size() - 1; ++level)
   {
      delete collections[level];
   }
   delete a;
   delete b;
   delete fespace;
   if (order > 0) { delete fec; }
   delete mesh;

   return 0;
}
