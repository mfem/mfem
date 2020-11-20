//                     hp refinement demo (NOT TO BE MERGED)
//
// Compile with: make hpdemo
//
// Sample runs:  hpdemo
//               hpdemo -m ../data/inline-hex.mesh
//
// Description:  Demonstrates creation of a variable order space and solution
//               of the Laplace problem -Delta u = 1. The mesh may contain
//               nonconforming refinements. Works in both 2D and 3D.
//               Serial FiniteElementSpace only, so far.
//
//               As there is no support in GLVis yet (mainly variable-order
//               space serialization), we need to prolong the solution to the
//               maximum order before visualization.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

GridFunction* ProlongToMaxOrder(const GridFunction *x);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   int ref_levels = 1;
   int seed = 1;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   bool vis_orders = true, vis_basis = true;
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
                  "Enable or disable GLVis visualization of the solution.");
   args.AddOption(&vis_orders, "-vis-o", "--visualize-orders", "-no-vis-o",
                  "--dont-visualize_orders",
                  "Enable or disable visualization of element orders.");
   args.AddOption(&vis_basis, "-vis-b", "--visualize-bases", "-no-vis-b",
                  "--dont-visualize-bases",
                  "Enable or disable visualization of the basis functions.");

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
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   mesh.EnsureNCMesh();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   srand(seed);
   {
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.RandomRefinement(0.5, true);
      }
   }

   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   bool delete_fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
      delete_fec = true;
   }
   else if (mesh.GetNodes())
   {
      fec = mesh.GetNodes()->OwnFEC();
      delete_fec = false;
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
      delete_fec = true;
   }

   FiniteElementSpace fespace(&mesh, fec);
   fespace.SetRelaxedHpConformity(relaxed_hp);

   // 6. At this point all elements have the default order (specified when
   //    construction the FECollection). Now we can p-refine some of them to
   //    obtain a variable-order space...
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      fespace.SetElementOrder(i, order + (rand()%5));
   }
   fespace.Update(false);

   cout << "Number of finite element unknowns: "
        << fespace.GetTrueVSize() << endl;

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
   cout << "Essential DOFs: " << ess_tdof_list.Size() << endl;

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   LinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(&fespace);
   x = 0.0;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   BilinearForm a(&fespace);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a.AddDomainIntegrator(new DiffusionIntegrator(one));

   // 10. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   if (static_cond) { a.EnableStaticCondensation(); }
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   cout << "Size of linear system: " << A->Height() << endl;

   // 11. Solve the linear system A X = B.
   if (!pa)
   {
      // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);
   }
   else // Jacobi preconditioning in partial assembly mode
   {
      if (UsesTensorBasis(fespace))
      {
         OperatorJacobiSmoother M(a, ess_tdof_list);
         PCG(*A, M, B, X, 1, 400, 1e-12, 0.0);
      }
      else
      {
         CG(*A, B, X, 1, 400, 1e-12, 0.0);
      }
   }

   // 12. Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);

   // 13. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   const char vishost[] = "localhost";
   const int  visport   = 19916;

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      // Prolong the solution vector onto L2 space of max order (for GLVis)
      GridFunction *vis_x = ProlongToMaxOrder(&x);
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << *vis_x;
      if (dim < 3) { sol_sock << "keys Rjlm\n"; }
      delete vis_x;
   }

   // 15. Visualize element orders
   if (vis_orders)
   {
      L2_FECollection l2fec(0, dim);
      FiniteElementSpace l2fes(&mesh, &l2fec);
      GridFunction orders(&l2fes);

      for (int i = 0; i < orders.Size(); i++)
      {
         orders(i) = fespace.GetElementOrder(i);
      }

      socketstream ord_sock(vishost, visport);
      ord_sock.precision(8);
      ord_sock << "solution\n" << mesh << orders;
      if (dim < 3) { ord_sock << "keys Rjlmc\n"; }
   }

   // 16. Visualize the basis functions
   if (vis_basis)
   {
      socketstream b_sock(vishost, visport);
      b_sock.precision(8);
      cout << "Press SPACE to cycle through basis functions..." << endl;

      for (int i = 0; i < X.Size(); i++)
      {
         X = 0.0;
         X(i) = 1.0;
         a.RecoverFEMSolution(X, b, x);

         GridFunction *vis_x = ProlongToMaxOrder(&x);
         b_sock << "solution\n" << mesh << *vis_x << flush;
         if (!i) { b_sock << "keys miIMA\n"; }
         b_sock << "pause\n" << flush;
         delete vis_x;
      }
   }

   // 17. Free the used memory.
   if (delete_fec)
   {
      delete fec;
   }

   return 0;
}


GridFunction* ProlongToMaxOrder(const GridFunction *x)
{
   const FiniteElementSpace *fespace = x->FESpace();
   const FiniteElementCollection *fec = fespace->FEColl();
   Mesh *mesh = fespace->GetMesh();

   int max_order = fespace->GetMaxElementOrder();

   // Create a visualization space of max order for all elements
   FiniteElementCollection *vis_fec =
      new L2_FECollection(max_order, mesh->Dimension(), BasisType::GaussLobatto);
   FiniteElementSpace *vis_space =
      new FiniteElementSpace(mesh, vis_fec);

   IsoparametricTransformation T;
   DenseMatrix I;

   GridFunction *visualization_x = new GridFunction(vis_space);

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
      const auto *visualization_fe = vis_fec->GetFE(geometry, max_order);

      visualization_fe->GetTransferMatrix(*fe, T, I);
      vis_space->GetElementDofs(i, dofs);
      visualization_vect.SetSize(dofs.Size());

      I.Mult(elemvect, visualization_vect);
      visualization_x->SetSubVector(dofs, visualization_vect);
   }

   visualization_x->MakeOwner(vis_fec);
   return visualization_x;
}

