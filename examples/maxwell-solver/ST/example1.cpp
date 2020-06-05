
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "additive_schwarz.hpp"
#include "schwarz.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../../data/star.mesh";
   // const char *mesh_file = "../../../data/beam-quad.mesh";
   int order = 1;
   int ref_levels = 1;
   bool visualization = true;
   StopWatch chrono;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&ref_levels, "-ref", "--ref_levels",
                  "Number of uniform h-refinements");
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


   Mesh *mesh;
   // mesh = new Mesh(mesh_file, 1, 1);
   mesh = new Mesh(1, 1, Element::QUADRILATERAL, true, 1, 1, false);
   int dim = mesh->Dimension();

   for (int l = 0; l < ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   // FiniteElementCollection *fec = new ND_FECollection(order, dim);
   FiniteElementSpace * fespace = new FiniteElementSpace(mesh, fec);
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   if (mesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh->bdr_attributes.Max());
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
   x = 1.0;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   BilinearForm *a = new BilinearForm(fespace);
   a->SetDiagonalPolicy(mfem::Matrix::DIAG_ONE);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));

   // 10. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   a->Assemble();

   OperatorPtr A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   cout << "Size of linear system: " << A->Height() << endl;

   AddSchwarz * prec = new AddSchwarz(a,ess_tdof_list, 0);
   prec->SetOperator((SparseMatrix&)(*A));
   prec->SetNumSmoothSteps(1);
   prec->SetDumpingParam(0.5);

   SchwarzSmoother * prec2 = new SchwarzSmoother(mesh,0,fespace,&(SparseMatrix&)(*A),ess_bdr);
   prec2->SetNumSmoothSteps(1);
   prec2->SetDumpingParam(0.5);


   int maxit = 2000;
   double rtol = 1e-8;
   double atol = 1e-8;
   Vector X0(X);
   CGSolver pcg;
   pcg.iterative_mode = false;
   pcg.SetPrintLevel(1);
   pcg.SetMaxIter(maxit);
   pcg.SetRelTol(rtol);
   pcg.SetAbsTol(atol);
   pcg.SetPreconditioner(*prec);
   pcg.SetOperator((SparseMatrix&)(*A));
   pcg.Mult(B, X0);

   X0 = X;
   pcg.SetPreconditioner(*prec2);
   pcg.Mult(B, X0);

   // 12. Recover the solution as a finite element grid function.
   a->RecoverFEMSolution(X0, *b, x);

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream mesh_sock(vishost, visport);
      mesh_sock.precision(8);
      mesh_sock << "mesh\n" << *mesh << flush;

      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << "keys rRjmc" << flush;
   }

   // 15. Free the used memory.
   delete prec;
   delete a;
   delete b;
   delete fespace;
   delete fec;
   delete mesh;

   return 0;
}
