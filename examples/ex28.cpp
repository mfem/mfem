//                                MFEM Example 28
//
// Compile with: make ex28
//
// Sample runs:  ex28
//               ex28 --visit-datafiles
//               ex28 --order 2
//
// Description:  Demonstrates a sliding boundary condition in an elasticity
//               problem. A trapezoid, roughly as pictured below, is pushed
//               from the right into a rigid notch. Normal displacement is
//               restricted, but tangential movement is allowed, so the
//               trapezoid compresses into the notch.
//
//                                       /-------+
//               normal constrained --->/        | <--- boundary force (2)
//               boundary (4)          /---------+
//                                          ^
//                                          |
//                                normal constrained boundary (1)
//
//               This example demonstrates the use of the ConstrainedSolver
//               framework.
//
//               We recommend viewing Example 2 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <set>

using namespace std;
using namespace mfem;

// Return a mesh with a single element with vertices (0, 0), (1, 0), (1, 1),
// (offset, 1) to demonstrate boundary conditions on a surface that is not
// axis-aligned.
Mesh * build_trapezoid_mesh(real_t offset)
{
   MFEM_VERIFY(offset < 0.9, "offset is too large!");

   const int dimension = 2;
   const int nvt = 4; // vertices
   const int nbe = 4; // num boundary elements
   Mesh * mesh = new Mesh(dimension, nvt, 1, nbe);

   // vertices
   real_t vc[dimension];
   vc[0] = 0.0; vc[1] = 0.0;
   mesh->AddVertex(vc);
   vc[0] = 1.0; vc[1] = 0.0;
   mesh->AddVertex(vc);
   vc[0] = offset; vc[1] = 1.0;
   mesh->AddVertex(vc);
   vc[0] = 1.0; vc[1] = 1.0;
   mesh->AddVertex(vc);

   // element
   Array<int> vert(4);
   vert[0] = 0; vert[1] = 1; vert[2] = 3; vert[3] = 2;
   mesh->AddQuad(vert, 1);

   // boundary
   Array<int> sv(2);
   sv[0] = 0; sv[1] = 1;
   mesh->AddBdrSegment(sv, 1);
   sv[0] = 1; sv[1] = 3;
   mesh->AddBdrSegment(sv, 2);
   sv[0] = 2; sv[1] = 3;
   mesh->AddBdrSegment(sv, 3);
   sv[0] = 0; sv[1] = 2;
   mesh->AddBdrSegment(sv, 4);

   mesh->FinalizeQuadMesh(1, 0, true);

   return mesh;
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int order = 1;
   bool visualization = 1;
   real_t offset = 0.3;
   bool visit = false;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&offset, "--offset", "--offset",
                  "How much to offset the trapezoid.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Build a trapezoidal mesh with a single quadrilateral element, where
   //    'offset' determines how far off it is from a rectangle.
   Mesh *mesh = build_trapezoid_mesh(offset);
   int dim = mesh->Dimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 1,000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(1000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 4. Define a finite element space on the mesh. Here we use vector finite
   //    elements, i.e. dim copies of a scalar finite element space. The vector
   //    dimension is specified by the last argument of the FiniteElementSpace
   //    constructor.
   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec, dim);
   cout << "Number of finite element unknowns: " << fespace->GetTrueVSize()
        << endl;
   cout << "Assembling matrix and r.h.s... " << flush;

   // 5. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, there are no essential boundary
   //    conditions in the usual sense, but we leave the machinery here for
   //    users to modify if they wish.
   Array<int> ess_tdof_list, ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 0;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 6. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system. In this case, b_i equals the boundary integral
   //    of f*phi_i where f represents a "push" force on the right side of the
   //    trapezoid.
   VectorArrayCoefficient f(dim);
   for (int i = 0; i < dim-1; i++)
   {
      f.Set(i, new ConstantCoefficient(0.0));
   }
   {
      Vector push_force(mesh->bdr_attributes.Max());
      push_force = 0.0;
      push_force(1) = -5.0e-2; // index 1 attribute 2
      f.Set(0, new PWConstCoefficient(push_force));
   }
   LinearForm *b = new LinearForm(fespace);
   b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
   b->Assemble();

   // 7. Define the solution vector x as a finite element grid function
   //    corresponding to fespace.
   GridFunction x(fespace);
   x = 0.0;

   // 8. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the linear elasticity integrator with piece-wise
   //    constants coefficient lambda and mu. We use constant coefficients,
   //    but see ex2 for how to set up piecewise constant coefficients based
   //    on attribute.
   Vector lambda(mesh->attributes.Max());
   lambda = 1.0;
   PWConstCoefficient lambda_func(lambda);
   Vector mu(mesh->attributes.Max());
   mu = 1.0;
   PWConstCoefficient mu_func(mu);

   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new ElasticityIntegrator(lambda_func, mu_func));

   // 9. Assemble the bilinear form and the corresponding linear system,
   //    applying any necessary transformations such as: eliminating boundary
   //    conditions, applying conforming constraints for non-conforming AMR,
   //    static condensation, etc.
   a->Assemble();

   SparseMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
   cout << "done." << endl;
   cout << "Size of linear system: " << A.Height() << endl;

   // 10. Set up constraint matrix to constrain normal displacement (but
   //     allow tangential displacement) on specified boundaries.
   Array<int> constraint_atts(2);
   constraint_atts[0] = 1;  // attribute 1 bottom
   constraint_atts[1] = 4;  // attribute 4 left side
   Array<int> lagrange_rowstarts;
   SparseMatrix* local_constraints =
      BuildNormalConstraints(*fespace, constraint_atts, lagrange_rowstarts);

   // 11. Define and apply an iterative solver for the constrained system
   //     in saddle-point form with a Gauss-Seidel smoother for the
   //     displacement block.
   GSSmoother M(A);
   SchurConstrainedSolver * solver =
      new SchurConstrainedSolver(A, *local_constraints, M);
   solver->SetRelTol(1e-5);
   solver->SetMaxIter(2000);
   solver->SetPrintLevel(1);
   solver->Mult(B, X);

   // 12. Recover the solution as a finite element grid function. Move the
   //     mesh to reflect the displacement of the elastic body being
   //     simulated, for purposes of output.
   a->RecoverFEMSolution(X, *b, x);
   mesh->SetNodalFESpace(fespace);
   GridFunction *nodes = mesh->GetNodes();
   *nodes += x;

   // 13. Save the refined mesh and the solution in VisIt format.
   if (visit)
   {
      VisItDataCollection visit_dc("ex28", mesh);
      visit_dc.SetLevelsOfDetail(4);
      visit_dc.RegisterField("displacement", &x);
      visit_dc.Save();
   }

   // 14. Save the displaced mesh and the inverted solution (which gives the
   //     backward displacements to the original grid). This output can be
   //     viewed later using GLVis: "glvis -m displaced.mesh -g sol.gf".
   {
      x *= -1; // sign convention for GLVis displacements
      ofstream mesh_ofs("displaced.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 15. Send the above data by socket to a GLVis server.  Use the "n" and "b"
   //     keys in GLVis to visualize the displacements.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }

   // 16. Free the used memory.
   delete local_constraints;
   delete solver;
   delete a;
   delete b;
   if (fec)
   {
      delete fespace;
      delete fec;
   }
   delete mesh;

   return 0;
}
