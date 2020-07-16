//                                MFEM Example hp check
//
// Compile with: make ex1_hp_check
//
// Description:  This example checks variable order space with a random
//               mesh refinement and a random polynomial order distribution
//               Exact solution is known. L2 error is checked.
//               Returns failure if L2 error > eps
//
// Run: seed=1; while ./ex1_hp_check -o 2 -r 8 -s $seed -no-vis; do (( seed++ )); done
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Exact solution: x^2 + y^2 + z^2
double exact_sln_1(const Vector &p)
{
   double x = p(0), y = p(1);
   if (p.Size() == 3)
   {
      double z = p(2);
      return x*x + y*y + z*z;
   }
   else
   {
      return x*x + y*y;
   }
}

double exact_sln_2(const Vector &p)
{
   MFEM_ASSERT(p.Size() == 2, "");
   double x = p(0), y = p(1);
   return x*(1.0 - x)*y*(1.0 - y);
}

double exact_rhs_1(const Vector &p)
{
   return (p.Size() == 3) ? -6.0 : -4.0;
}

double exact_rhs_2(const Vector &p)
{
   MFEM_ASSERT(p.Size() == 2, "");
   double x = p(0), y = p(1);
   return -(2*x*(x - 1.0) + 2*y*(y - 1.0));
}


GridFunction* ProlongToMaxOrder(const GridFunction *x);


int main(int argc, char *argv[])
{
   // Parse command-line options.
   const char *mesh_file = "../data/inline-quad.mesh";
   int order = 2;
   int problem = 1;
   int ref_levels = 1;
   int seed = 1;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = false;
   bool relaxed_hp = false;

   char vishost[] = "localhost";
   int  visport   = 19916;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem 1 or 2.");
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
   args.PrintOptions(cout);

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
#if 1
   srand(seed);
   {
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->RandomRefinement(0.5, true);
      }
   }
#else
   Array<Refinement> refs;
   refs.Append(Refinement(0, 1));
   mesh->GeneralRefinement(refs);
   refs[0].index = 1;
   refs[0].ref_type = 2;
   mesh->GeneralRefinement(refs);
   refs[0].ref_type = 1;
   mesh->GeneralRefinement(refs);
#endif

   FunctionCoefficient exsol((problem == 1) ? exact_sln_1 : exact_sln_2);
   FunctionCoefficient rhs((problem == 1) ? exact_rhs_1 : exact_rhs_2);

   // Define a finite element space on the mesh. Here we use continuous
   // Lagrange finite elements of the specified order.
   FiniteElementCollection *fec;
   fec = new H1_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   fespace->SetRelaxedHpConformity(relaxed_hp);

   // At this point all elements have the default order (specified when
   // construction the FECollection). Now we can p-refine some of them to
   // obtain a variable-order space...
#if 1
   srand(seed);
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      fespace->SetElementOrder(i, (rand()%3)+order);
   }
#else
   fespace->SetElementOrder(0, 2);
   fespace->SetElementOrder(1, 2);
   fespace->SetElementOrder(2, 2);
   fespace->SetElementOrder(3, 4);
#endif
   fespace->Update(false);

   /*mesh->UniformRefinement();
   fespace->Update(false);
   mesh->UniformRefinement();
   fespace->Update(false);*/

   if (1)
   {
      std::ofstream f("mesh.dump");
      mesh->ncmesh->DebugDump(f);
   }

   const char* keys = (dim == 2) ? "Rjlmciiiiii" : "Amcooooo";

   if (visualization)
   {
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
               << "window_title 'Polynomial orders'\n"
               "keys " << keys << "\n" << flush;
   }

   cout << "Space size (all DOFs): " << fespace->GetNDofs() << endl;
   cout << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;

   // All boundary attributes will be used for essential (Dirichlet) BC.
   // Project exact solution on boundary
   Array<int> ess_tdof_list;
   MFEM_VERIFY(mesh->bdr_attributes.Size() > 0,
              "Boundary attributes required in the mesh.");
   Array<int> ess_attr(mesh->bdr_attributes.Max());
   ess_attr = 1;

   GridFunction x(fespace);
   x = 0.0;
   x.ProjectBdrCoefficient(exsol, ess_attr);
   fespace->GetEssentialTrueDofs(ess_attr, ess_tdof_list);

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
   if (relaxed_hp) {
      //bf.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(-1, -1));
   }
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
      PCG(*A, M, B, X, 1, 1000, 1e-30, 0.0);
   }
   else // No preconditioning for now in partial assembly mode.
   {
      CG(*A, B, X, 3, 2000, 1e-12, 0.0);
   }

   // After solving the linear system, reconstruct the solution as a
   // finite element GridFunction. Constrained nodes are interpolated
   // from true DOFs (it may therefore happen that x.Size() >= X.Size()).
   bf.RecoverFEMSolution(X, lf, x);

   // Compute L2 error from the exact solution and check if < eps
   double error = x.ComputeL2Error(exsol);
   cout << "\nFE solution L2 error: " << error << endl;

   // Do nodal interpolation of the exact solution
   GridFunction y(fespace);
   y.ProjectCoefficient(exsol);
   if (fespace->GetProlongationMatrix())
   {
      Vector tmp(fespace->GetTrueVSize());
      fespace->GetRestrictionInterpolationMatrix()->Mult(y, tmp);
      fespace->GetProlongationMatrix()->Mult(tmp, y);
   }
   double error2 = y.ComputeL2Error(exsol);
   cout << "Nodal projection L2 error: " << error2 << endl;

   const SparseMatrix *R = fespace->GetRestrictionMatrix();
   if (R)
   {
      // z = Ry
      Vector Z(R->Height());
      R->Mult(y, Z);

      // compute Az - B
      SparseMatrix *spA = A.Is<SparseMatrix>();
      Vector check(spA->Height());
      spA->Mult(Z, check);
      check -= B;
      cout << "|Az - B|: " << check.Norml2() << endl;

      // w = Rx
      Vector W(R->Height());
      R->Mult(x, W);

      // compute 1/2 z'Az - B'z
      Vector Aw(spA->Height());
      spA->Mult(W, Aw);
      double energy = W*Aw/2 - B*W;
      cout << "(1/2 w'Aw - B'w): " << energy << endl;
   }

   // Send the solution by socket to a GLVis server.
   if (visualization)
   {
      // Prolong the solution vector onto L2 space of max order (for GLVis)
      GridFunction *vis_x = ProlongToMaxOrder(&x);

#if 1
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << *vis_x
               << "window_title 'FE solution'\n"
                  "keys " << keys << "\n" << flush;
#endif

#if 1
      Vector tmp = *vis_x;
      vis_x->ProjectCoefficient(exsol);
      *vis_x -= tmp;

      socketstream err_sock(vishost, visport);
      err_sock.precision(8);
      err_sock << "solution\n" << *mesh << *vis_x
               << "window_title 'Residual'\n"
                  "keys " << keys << "\n" << flush;
#endif
      delete vis_x;

      // visualize the basis functions
      if (0)
      {
         socketstream b_sock(vishost, visport);
         b_sock.precision(8);

         int first = 0;
         for (int i = first; i < X.Size(); i++)
         {
            X = 0.0;
            X(i) = 1.0;
            bf.RecoverFEMSolution(X, lf, x);
            vis_x = ProlongToMaxOrder(&x);

            b_sock << "solution\n" << *mesh << *vis_x << flush;
            if (i == first)
            {
               b_sock << "window_title 'Basis functions'\n"
                         "keys " << keys << "\n";
            }
            b_sock << "pause\n" << flush;
            delete vis_x;
         }
      }
   }

   double eps = 1e-10;
   MFEM_VERIFY(error < eps,
              "Failure: L2 error bigger than given threshold.");


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
