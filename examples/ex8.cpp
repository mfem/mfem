//                                MFEM Example 8
//
// Compile with: make ex8
//
// Sample runs:  ex8 -m ../data/square-disc.mesh
//               ex8 -m ../data/star.mesh
//               ex8 -m ../data/star-mixed.mesh
//               ex8 -m ../data/escher.mesh
//               ex8 -m ../data/fichera.mesh
//               ex8 -m ../data/fichera-mixed.mesh
//               ex8 -m ../data/square-disc-p2.vtk
//               ex8 -m ../data/square-disc-p3.mesh
//               ex8 -m ../data/star-surf.mesh -o 2
//               ex8 -m ../data/mobius-strip.mesh
//
// Description:  This example code demonstrates the use of the Discontinuous
//               Petrov-Galerkin (DPG) method in its primal 2x2 block form as a
//               simple finite element discretization of the Laplace problem
//               -Delta u = f with homogeneous Dirichlet boundary conditions. We
//               use high-order continuous trial space, a high-order interfacial
//               (trace) space, and a high-order discontinuous test space
//               defining a local dual (H^{-1}) norm.
//
//               We use the primal form of DPG, see "A primal DPG method without
//               a first-order reformulation", Demkowicz and Gopalakrishnan, CAM
//               2013, DOI:10.1016/j.camwa.2013.06.029.
//
//               The example highlights the use of interfacial (trace) finite
//               elements and spaces, trace face integrators and the definition
//               of block operators and preconditioners.
//
//               We recommend viewing examples 1-5 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
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
   //    largest number that gives a final mesh with no more than 10,000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 4. Define the trial, interfacial (trace) and test DPG spaces:
   //    - The trial space, x0_space, contains the non-interfacial unknowns and
   //      has the essential BC.
   //    - The interfacial space, xhat_space, contains the interfacial unknowns
   //      and does not have essential BC.
   //    - The test space, test_space, is an enriched space where the enrichment
   //      degree may depend on the spatial dimension of the domain, the type of
   //      the mesh and the trial space order.
   unsigned int trial_order = order;
   unsigned int trace_order = order - 1;
   unsigned int test_order  = order; /* reduced order, full order is
                                        (order + dim - 1) */
   if (dim == 2 && (order%2 == 0 || (mesh->MeshGenerator() & 2 && order > 1)))
   {
      test_order++;
   }
   if (test_order < trial_order)
      cerr << "Warning, test space not enriched enough to handle primal"
           << " trial space\n";

   FiniteElementCollection *x0_fec, *xhat_fec, *test_fec;

   x0_fec   = new H1_FECollection(trial_order, dim);
   xhat_fec = new RT_Trace_FECollection(trace_order, dim);
   test_fec = new L2_FECollection(test_order, dim);

   FiniteElementSpace *x0_space   = new FiniteElementSpace(mesh, x0_fec);
   FiniteElementSpace *xhat_space = new FiniteElementSpace(mesh, xhat_fec);
   FiniteElementSpace *test_space = new FiniteElementSpace(mesh, test_fec);

   // 5. Define the block structure of the problem, by creating the offset
   //    variables. Also allocate two BlockVector objects to store the solution
   //    and rhs.
   enum {x0_var, xhat_var, NVAR};

   int s0 = x0_space->GetVSize();
   int s1 = xhat_space->GetVSize();
   int s_test = test_space->GetVSize();

   Array<int> offsets(NVAR+1);
   offsets[0] = 0;
   offsets[1] = s0;
   offsets[2] = s0+s1;

   Array<int> offsets_test(2);
   offsets_test[0] = 0;
   offsets_test[1] = s_test;

   std::cout << "\nNumber of Unknowns:\n"
             << " Trial space,     X0   : " << s0
             << " (order " << trial_order << ")\n"
             << " Interface space, Xhat : " << s1
             << " (order " << trace_order << ")\n"
             << " Test space,      Y    : " << s_test
             << " (order " << test_order << ")\n\n";

   BlockVector x(offsets), b(offsets);
   x = 0.;

   // 6. Set up the linear form F(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (f,phi_i) where f=1.0 and
   //    phi_i are the basis functions in the test finite element fespace.
   ConstantCoefficient one(1.0);
   LinearForm F(test_space);
   F.AddDomainIntegrator(new DomainLFIntegrator(one));
   F.Assemble();

   // 7. Set up the mixed bilinear form for the primal trial unknowns, B0,
   //    the mixed bilinear form for the interfacial unknowns, Bhat,
   //    the inverse stiffness matrix on the discontinuous test space, Sinv,
   //    and the stiffness matrix on the continuous trial space, S0.
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;

   MixedBilinearForm *B0 = new MixedBilinearForm(x0_space,test_space);
   B0->AddDomainIntegrator(new DiffusionIntegrator(one));
   B0->Assemble();
   B0->EliminateTrialDofs(ess_bdr, x.GetBlock(x0_var), F);
   B0->Finalize();

   MixedBilinearForm *Bhat = new MixedBilinearForm(xhat_space,test_space);
   Bhat->AddTraceFaceIntegrator(new TraceJumpIntegrator());
   Bhat->Assemble();
   Bhat->Finalize();

   BilinearForm *Sinv = new BilinearForm(test_space);
   SumIntegrator *Sum = new SumIntegrator;
   Sum->AddIntegrator(new DiffusionIntegrator(one));
   Sum->AddIntegrator(new MassIntegrator(one));
   Sinv->AddDomainIntegrator(new InverseIntegrator(Sum));
   Sinv->Assemble();
   Sinv->Finalize();

   BilinearForm *S0 = new BilinearForm(x0_space);
   S0->AddDomainIntegrator(new DiffusionIntegrator(one));
   S0->Assemble();
   S0->EliminateEssentialBC(ess_bdr);
   S0->Finalize();

   SparseMatrix &matB0   = B0->SpMat();
   SparseMatrix &matBhat = Bhat->SpMat();
   SparseMatrix &matSinv = Sinv->SpMat();
   SparseMatrix &matS0   = S0->SpMat();

   // 8. Set up the 1x2 block Least Squares DPG operator, B = [B0  Bhat],
   //    the normal equation operator, A = B^t Sinv B, and
   //    the normal equation right-hand-size, b = B^t Sinv F.
   BlockOperator B(offsets_test, offsets);
   B.SetBlock(0,0,&matB0);
   B.SetBlock(0,1,&matBhat);
   RAPOperator A(B, matSinv, B);
   {
      Vector SinvF(s_test);
      matSinv.Mult(F,SinvF);
      B.MultTranspose(SinvF, b);
   }

   // 9. Set up a block-diagonal preconditioner for the 2x2 normal equation
   //
   //        [ S0^{-1}     0     ]
   //        [   0     Shat^{-1} ]      Shat = (Bhat^T Sinv Bhat)
   //
   //    corresponding to the primal (x0) and interfacial (xhat) unknowns.
   SparseMatrix * Shat = RAP(matBhat, matSinv, matBhat);

#ifndef MFEM_USE_SUITESPARSE
   const double prec_rtol = 1e-3;
   const int prec_maxit = 200;
   CGSolver *S0inv = new CGSolver;
   S0inv->SetOperator(matS0);
   S0inv->SetPrintLevel(-1);
   S0inv->SetRelTol(prec_rtol);
   S0inv->SetMaxIter(prec_maxit);
   CGSolver *Shatinv = new CGSolver;
   Shatinv->SetOperator(*Shat);
   Shatinv->SetPrintLevel(-1);
   Shatinv->SetRelTol(prec_rtol);
   Shatinv->SetMaxIter(prec_maxit);
   // Disable 'iterative_mode' when using CGSolver (or any IterativeSolver) as
   // a preconditioner:
   S0inv->iterative_mode = false;
   Shatinv->iterative_mode = false;
#else
   Operator *S0inv = new UMFPackSolver(matS0);
   Operator *Shatinv = new UMFPackSolver(*Shat);
#endif

   BlockDiagonalPreconditioner P(offsets);
   P.SetDiagonalBlock(0, S0inv);
   P.SetDiagonalBlock(1, Shatinv);

   // 10. Solve the normal equation system using the PCG iterative solver.
   //     Check the weighted norm of residual for the DPG least square problem.
   //     Wrap the primal variable in a GridFunction for visualization purposes.
   PCG(A, P, b, x, 1, 200, 1e-12, 0.0);

   {
      Vector LSres(s_test);
      B.Mult(x, LSres);
      LSres -= F;
      double res = sqrt(matSinv.InnerProduct(LSres, LSres));
      cout << "\n|| B0*x0 + Bhat*xhat - F ||_{S^-1} = " << res << endl;
   }

   GridFunction x0;
   x0.MakeRef(x0_space, x.GetBlock(x0_var), 0);

   // 11. Save the refined mesh and the solution. This output can be viewed
   //     later using GLVis: "glvis -m refined.mesh -g sol.gf".
   {
      ofstream mesh_ofs("refined.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      x0.Save(sol_ofs);
   }

   // 12. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x0 << flush;
   }

   // 13. Free the used memory.
   delete S0inv;
   delete Shatinv;
   delete Shat;
   delete Bhat;
   delete B0;
   delete S0;
   delete Sinv;
   delete test_space;
   delete test_fec;
   delete xhat_space;
   delete xhat_fec;
   delete x0_space;
   delete x0_fec;
   delete mesh;

   return 0;
}
