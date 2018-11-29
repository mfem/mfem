//                                MFEM Example 1
//
// Compile with: make ex1
//
// Sample runs:  ex1 -m ../data/square-disc.mesh
//               ex1 -m ../data/star.mesh
//               ex1 -m ../data/star-mixed.mesh
//               ex1 -m ../data/escher.mesh
//               ex1 -m ../data/fichera.mesh
//               ex1 -m ../data/fichera-mixed.mesh
//               ex1 -m ../data/toroid-wedge.mesh
//               ex1 -m ../data/square-disc-p2.vtk -o 2
//               ex1 -m ../data/square-disc-p3.mesh -o 3
//               ex1 -m ../data/square-disc-nurbs.mesh -o -1
//               ex1 -m ../data/star-mixed-p2.mesh -o 2
//               ex1 -m ../data/disc-nurbs.mesh -o -1
//               ex1 -m ../data/pipe-nurbs.mesh -o -1
//               ex1 -m ../data/fichera-mixed-p2.mesh -o 2
//               ex1 -m ../data/star-surf.mesh
//               ex1 -m ../data/square-disc-surf.mesh
//               ex1 -m ../data/inline-segment.mesh
//               ex1 -m ../data/amr-quad.mesh
//               ex1 -m ../data/amr-hex.mesh
//               ex1 -m ../data/fichera-amr.mesh
//               ex1 -m ../data/mobius-strip.mesh
//               ex1 -m ../data/mobius-strip.mesh -o -1 -sc
//
// Description:  This is a mock-up of an alternate interface for creating
//               FEM linear systems.
//

#include "mfem.hpp"
#include "mfem4/mfem4.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem4;


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool partial = false;
   bool static_cond = false;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&partial,
                  "-pa", "--partial-assembly", "-fa", "--full-assembly",
                  "Enable partial assembly.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
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

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(50000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 4. Define a finite element space on the mesh. Here we use continuous
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

   // 5. Determine the list of essential boundary dofs.
   //    NOTE: I wanted to make this less confusing.
   Array<int> attributes; // NOTE: attribute list, not markers
   attributes.Append(1);  // NOTE: empty list means all boundary attributes (?)

   Array<int> ess_dof_list; // NOTE: regular DOFs
   fespace->GetBoundaryDofs(attributes, ess_dof_list); // NOTE: more general function

   // 6. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   LinearForm *b = new LinearForm(fespace);
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   // 7. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x.SetTo(0.0); // NOTE: it would be nice to get rid of operator overloads

   // 8. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   mfem4::BilinearForm *a = new mfem4::BilinearForm(fespace);
   a->AddDomainIntegrator(new mfem4::DiffusionIntegrator(one)); // NOTE: integrators can MultAdd (if implemented)
   a->SetAssemblyLevel(partial ? AssemblyLevel::PARTIAL
                               : AssemblyLevel::FULL);
   a->Assemble(); // NOTE: BilinearForm is as simple as before, no FormLinearSystem

   // 9. Create the linear system, applying any necessary transformations
   //    such as: eliminating boundary conditions, applying conforming
   //    constraints for non-conforming AMR, static condensation, etc.
   //    NOTE: LinearSystem is a new class that holds (owns) A, X, B and knows
   //    how to form them.
   LinearSystem ls(a, b); // NOTE: ParLinearSystem in parallel
   ls.SetEssentialDofs(ess_dof_list, x); // NOTE: regular DOFs; values taken from x
   ls.SetOperatorType(Operator::MFEM_SPARSEMAT); // or PETSC_xxx...
   ls.EnableStaticCondensation(static_cond); // NOTE: does nothing for PA
   ls.Assemble(); // NOTE: this is like FormLinearSystem
   // NOTE: if the BF has partial assembly, ls.Assemble() constructs the constrained operator

   // 10. Solve the linear system with the supplied solver and optionally
   //     a preconditioner too.
   GSSmoother prec;
   CGSolver cg;
   cg.SetAbsTol(1e-12);
   cg.SetMaxIter(200);
   cg.SetPrintLevel(1);
   ls.Solve(prec, cg, x); // NOTE: this just calls the solver and recovers 'x'

   // 11. Alternatively, one can still access the system matrix and RHS if
   //     necessary (after LinearSystem::Assemble()).
   const Operator &A = ls.GetMatrix(); // NOTE: this fails if partial==true
   const Operator &A = ls.GetOperator(); // NOTE: this always works
   const Vector &B = ls.GetRHS(); // NOTE: this always works
   cout << "Size of linear system: " << A.Height() << endl;
   // NOTE: you can do ls.RecoverFEMSolution(X, x); if you obtain X your way

   // 12. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);

   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs); // NOTE: should we call this Print?

   // 13. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }

   // 14. Free the used memory.
   delete a;
   delete b;
   delete fespace;
   if (order > 0) { delete fec; }
   delete mesh;

   return 0;
}
