//                                MFEM Example hp check
//
// Compile with: make ex1_hp_check
//
// Description:  This example checks variable order space with a random
//               mesh refinement and a random polynomial order distribution
//               Exact solution is known. L2 error is checked,
//               Returns failure if L2 error > eps
//
// Run: seed=1; while ./ex1_hp_check -o 2 -r 8 -s $seed -no-vis; do (( seed++ )); done
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Exact solution
double exact_solution(const Vector &p)
{
   double x = p(0), y = p(1);
   return x*x + y*y;
}

double exact_laplace(const Vector &p)
{
   //double x = p(0), y = p(1);
   return -4.0;
}

GridFunction* ProlongToMaxOrder(const GridFunction *x);

int main(int argc, char *argv[])
{
   // Parse command-line options.
   const char *mesh_file = "quad.mesh";
   int order = 2;
   int ref_levels = 1;
   int seed = 1;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   bool relaxed_hp = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&ref_levels, "-r", "--ref-levels",
                  "Number of mesh refinement levels.");
   args.AddOption(&seed, "-s", "--seed", "Random seed");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&relaxed_hp, "-x", "--relaxed-hp", "-no-x",
                  "--no-relaxed-hp", "Set relaxed hp conformity.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   //args.PrintOptions(cout);

   // Enable hardware devices such as GPUs, and programming models such as
   // CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // Read the mesh from the given mesh file.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();
   mesh->EnsureNCMesh();

   // Refine the mesh to increase the resolution. In this example we do
   // 'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   // largest number that gives a final mesh with no more than 50,000
   // elements.
   srand(seed);
   {
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->RandomRefinement(0.5, true);
      }
   }
   srand(seed);

   FunctionCoefficient exsol(exact_solution);
   FunctionCoefficient rhs(exact_laplace);

   // Define a finite element space on the mesh. Here we use continuous
   // Lagrange finite elements of the specified order.
   FiniteElementCollection *fec;
   fec = new H1_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   fespace->SetRelaxedHpConformity(relaxed_hp);

   // At this point all elements have the default order (specified when
   // construction the FECollection). Now we can p-refine some of them to
   // obtain a variable-order space...
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      fespace->SetElementOrder(i, (rand()%5)+order);
   }
   fespace->Update(false);

   cout << "Space size (all DOFs): " << fespace->GetNDofs() << endl;
   cout << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;

   // All boundary attributes will be used for essential (Dirichlet) BC.
   // Project exact solution on boundary
   Array<int> ess_tdof_list;
   MFEM_VERIFY(mesh->bdr_attributes.Size() > 0,
              "Boundary attributes required in the mesh.");
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;

   GridFunction x(fespace);
   x = 0.0;
   x.ProjectBdrCoefficient(exsol, ess_bdr);
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // Assemble the linear form. The right hand side is manufactured
   // so that the solution is the analytic solution.
   LinearForm lf(fespace);
   DomainLFIntegrator *dlfi = new DomainLFIntegrator(rhs);
   lf.AddDomainIntegrator(dlfi);
   lf.Assemble();

   // Assemble bilinear form.
   BilinearForm bf(fespace);
   if (pa) { bf.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   bf.AddDomainIntegrator(new DiffusionIntegrator());
   bf.Assemble();

   OperatorPtr A;
   Vector B, X;
   const int copy_interior = 1;
   bf.FormLinearSystem(ess_tdof_list, x, lf, A, X, B, copy_interior);

   // Solve the linear system A X = B.
   if (!pa)
   {
      // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 0, 1000, 1e-15, 0.0);
   }
   else // No preconditioning for now in partial assembly mode.
   {
      CG(*A, B, X, 3, 2000, 1e-12, 0.0);
   }

   // After solving the linear system, reconstruct the solution as a
   // finite element GridFunction. Constrained nodes are interpolated
   // from true DOFs (it may therefore happen that x.Size() >= X.Size()).
   bf.RecoverFEMSolution(X, lf, x);

   double eps = 1e-5;
   // Compute L2 error from the exact solution and check if < eps
   double error = x.ComputeL2Error(exsol);
   cout << "L2 error: " << error << endl;
   MFEM_VERIFY(error < eps,
              "Failure: L2 error bigger than given threshold.");

   // Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;

      // Prolong the solution vector onto L2 space of max order (for GLVis)
      GridFunction *vis_x = ProlongToMaxOrder(&x);

#if 1
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << *vis_x
               //<< "keys Rjlm\n"
               << flush;
#endif
      delete vis_x;

#if 1
      L2_FECollection l2fec(0, dim);
      FiniteElementSpace l2fes(mesh, &l2fec);
      GridFunction orders(&l2fes);

      for (int i = 0; i < orders.Size(); i++)
      {
         orders(i) = fespace->GetElementOrder(i);
      }

      socketstream ord_sock(vishost, visport);
      ord_sock.precision(8);
      ord_sock << "solution\n" << *mesh << orders
               //<< "keys Rjlmc\n"
               << flush;
#endif

   }

   // Free the used memory.
   delete fespace;
   if (order > 0) { delete fec; }
   delete mesh;

   return 0;
}

GridFunction* ProlongToMaxOrder(const GridFunction *x)
{
   const FiniteElementSpace *fespace = x->FESpace();
   Mesh *mesh = fespace->GetMesh();
   const FiniteElementCollection *fec = fespace->FEColl();

   // Find the max order in the space
   int max_order = 1;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      max_order = std::max(fespace->GetElementOrder(i), max_order);
   }

   // Create a visualization space of max order for all elements
   FiniteElementCollection *visualization_fec =
      new L2_FECollection(max_order, mesh->Dimension(), BasisType::GaussLobatto);
   FiniteElementSpace *visualization_space =
      new FiniteElementSpace(mesh, visualization_fec);

   IsoparametricTransformation T;
   DenseMatrix I;

   GridFunction *visualization_x = new GridFunction(visualization_space);

   // Interpolate solution vector in the visualization space
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      Geometry::Type geometry = mesh->GetElementGeometry(i);
      T.SetIdentityTransformation(geometry);

      Array<int> dofs;
      fespace->GetElementDofs(i, dofs);
      Vector elemvect, visualization_vect;
      x->GetSubVector(dofs, elemvect);

      const auto *fe = fec->GetFE(geometry, fespace->GetElementOrder(i));
      const auto *visualization_fe = visualization_fec->GetFE(geometry, max_order);

      visualization_fe->GetTransferMatrix(*fe, T, I);
      visualization_space->GetElementDofs(i, dofs);
      visualization_vect.SetSize(dofs.Size());

      I.Mult(elemvect, visualization_vect);
      visualization_x->SetSubVector(dofs, visualization_vect);
   }

   visualization_x->MakeOwner(visualization_fec);
   return visualization_x;
}
