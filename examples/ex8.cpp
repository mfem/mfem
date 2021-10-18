//                                MFEM Example 8
//
// Compile with: make ex8
//
// Sample runs:  ex8 -m ../data/inline-quad.mesh

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/inline-quad.mesh";
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

   cout << "mesh attr max = " << mesh->bdr_attributes.Max() << endl;
   cout << "mesh bdr elemens = " << mesh->GetNBE() << endl;


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

   // 7. Set up the mixed bilinear form for the primal trial unknowns, B0,
   //    the mixed bilinear form for the interfacial unknowns, Bhat,
   //    the inverse stiffness matrix on the discontinuous test space, Sinv,
   //    and the stiffness matrix on the continuous trial space, S0.
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   Array<int> ess_hat_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;
   ess_bdr[1] = 1;
   ess_hat_bdr = 1;
   ess_hat_bdr[0] = 0;
   ess_hat_bdr[1] = 0;


   // 6. Set up the linear form F(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (f,phi_i) where f=1.0 and
   //    phi_i are the basis functions in the test finite element fespace.
   ConstantCoefficient one(1.0);
   LinearForm F(test_space);
   F.AddDomainIntegrator(new DomainLFIntegrator(one));
   F.Assemble();


   MixedBilinearForm *B0 = new MixedBilinearForm(x0_space,test_space);
   B0->AddDomainIntegrator(new DiffusionIntegrator(one));
   B0->Assemble();
   // B0->EliminateTrialDofs(ess_bdr, x.GetBlock(x0_var), F); // will be taken care at the matrix level
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
   BlockMatrix B(offsets_test, offsets);
   B.SetBlock(0,0,&matB0);
   B.SetBlock(0,1,&matBhat);

   SparseMatrix & B1 = *B.CreateMonolithic();
   SparseMatrix & A = *RAP(B1,matSinv,B1);

   {
      Vector SinvF(s_test);
      matSinv.Mult(F,SinvF);
      B1.MultTranspose(SinvF, b);
   }

   Array<int> ess_tdofs0;
   Array<int> ess_tdofs1;
   x0_space->GetEssentialTrueDofs(ess_bdr,ess_tdofs0);
   xhat_space->GetEssentialTrueDofs(ess_hat_bdr,ess_tdofs1);


   // Esential BC on the field variable
   for (int i = 0; i<ess_tdofs0.Size(); i++)
   {
      int j = ess_tdofs0[i];
      A.EliminateRowCol(j,x[j],b);
   }

   // Neuman BC on the field variable (equivalently essential BC on the flux variable)
   for (int i = 0; i<ess_tdofs1.Size(); i++)
   {
      int j = ess_tdofs1[i] + x0_space->GetTrueVSize();
      A.EliminateRowCol(j,x[j],b);
   }


   // 9. Set up a block-diagonal preconditioner for the 2x2 normal equation
   //
   //        [ S0^{-1}     0     ]
   //        [   0     Shat^{-1} ]      Shat = (Bhat^T Sinv Bhat)
   //
   //    corresponding to the primal (x0) and interfacial (xhat) unknowns.
   SparseMatrix * Shat = RAP(matBhat, matSinv, matBhat);
   for (int i = 0; i<ess_tdofs1.Size(); i++)
   {
      int j = ess_tdofs1[i];
      Shat->EliminateRowCol(j);
   }

   Operator *S0inv = new UMFPackSolver(matS0);
   Operator *Shatinv = new UMFPackSolver(*Shat);

   BlockDiagonalPreconditioner P(offsets);
   P.SetDiagonalBlock(0, S0inv);
   P.SetDiagonalBlock(1, Shatinv);

   // 10. Solve the normal equation system using the PCG iterative solver.
   //     Check the weighted norm of residual for the DPG least square problem.
   //     Wrap the primal variable in a GridFunction for visualization purposes.
   PCG(A, P, b, x, 1, 200, 1e-12, 0.0);
   {
      Vector LSres(s_test);
      B1.Mult(x, LSres);
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
