//                       MFEM Example 8 - Parallel Version
//
// Compile with: make ex8p
//
// Sample runs:  mpirun -np 4 ex8p -m ../data/square-disc.mesh
//               mpirun -np 4 ex8p -m ../data/star.mesh
//               mpirun -np 4 ex8p -m ../data/star-mixed.mesh
//               mpirun -np 4 ex8p -m ../data/escher.mesh
//               mpirun -np 4 ex8p -m ../data/fichera.mesh
//               mpirun -np 4 ex8p -m ../data/fichera-mixed.mesh
//               mpirun -np 4 ex8p -m ../data/square-disc-p2.vtk
//               mpirun -np 4 ex8p -m ../data/square-disc-p3.mesh
//               mpirun -np 4 ex8p -m ../data/star-surf.mesh -o 2
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
//               of block operators and preconditioners. The use of the ADS
//               preconditioner from hypre for interfacially-reduced H(div)
//               problems is also illustrated.
//
//               We recommend viewing examples 1-5 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
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
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
      int ref_levels =
         (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      int par_ref_levels = 1;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }

   // 6. Define the trial, interfacial (trace) and test DPG spaces:
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
   if (dim == 2 && (order%2 == 0 || (pmesh->MeshGenerator() & 2 && order > 1)))
   {
      test_order++;
   }
   if (test_order < trial_order)
   {
      if (myid == 0)
      {
         cerr << "Warning, test space not enriched enough to handle primal"
              << " trial space\n";
      }
   }

   FiniteElementCollection *x0_fec, *xhat_fec, *test_fec;

   x0_fec   = new H1_FECollection(trial_order, dim);
   xhat_fec = new RT_Trace_FECollection(trace_order, dim);
   test_fec = new L2_FECollection(test_order, dim);

   ParFiniteElementSpace *x0_space, *xhat_space, *test_space;

   x0_space   = new ParFiniteElementSpace(pmesh, x0_fec);
   xhat_space = new ParFiniteElementSpace(pmesh, xhat_fec);
   test_space = new ParFiniteElementSpace(pmesh, test_fec);

   HYPRE_BigInt glob_true_s0     =   x0_space->GlobalTrueVSize();
   HYPRE_BigInt glob_true_s1     = xhat_space->GlobalTrueVSize();
   HYPRE_BigInt glob_true_s_test = test_space->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "\nNumber of Unknowns:\n"
           << " Trial space,     X0   : " << glob_true_s0
           << " (order " << trial_order << ")\n"
           << " Interface space, Xhat : " << glob_true_s1
           << " (order " << trace_order << ")\n"
           << " Test space,      Y    : " << glob_true_s_test
           << " (order " << test_order << ")\n\n";
   }

   // 7. Set up the linear form F(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (f,phi_i) where f=1.0 and
   //    phi_i are the basis functions in the test finite element fespace.
   ConstantCoefficient one(1.0);
   ParLinearForm * F = new ParLinearForm(test_space);
   F->AddDomainIntegrator(new DomainLFIntegrator(one));
   F->Assemble();

   ParGridFunction * x0 = new ParGridFunction(x0_space);
   *x0 = 0.;

   // 8. Set up the mixed bilinear form for the primal trial unknowns, B0,
   //    the mixed bilinear form for the interfacial unknowns, Bhat,
   //    the inverse stiffness matrix on the discontinuous test space, Sinv,
   //    and the stiffness matrix on the continuous trial space, S0.
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   Array<int> ess_dof;
   x0_space->GetEssentialVDofs(ess_bdr, ess_dof);

   ParMixedBilinearForm *B0 = new ParMixedBilinearForm(x0_space,test_space);
   B0->AddDomainIntegrator(new DiffusionIntegrator(one));
   B0->Assemble();
   B0->EliminateEssentialBCFromTrialDofs(ess_dof, *x0, *F);
   B0->Finalize();

   ParMixedBilinearForm *Bhat = new ParMixedBilinearForm(xhat_space,test_space);
   Bhat->AddTraceFaceIntegrator(new TraceJumpIntegrator());
   Bhat->Assemble();
   Bhat->Finalize();

   ParBilinearForm *Sinv = new ParBilinearForm(test_space);
   SumIntegrator *Sum = new SumIntegrator;
   Sum->AddIntegrator(new DiffusionIntegrator(one));
   Sum->AddIntegrator(new MassIntegrator(one));
   Sinv->AddDomainIntegrator(new InverseIntegrator(Sum));
   Sinv->Assemble();
   Sinv->Finalize();

   ParBilinearForm *S0 = new ParBilinearForm(x0_space);
   S0->AddDomainIntegrator(new DiffusionIntegrator(one));
   S0->Assemble();
   S0->EliminateEssentialBC(ess_bdr);
   S0->Finalize();

   HypreParMatrix * matB0   = B0->ParallelAssemble();    delete B0;
   HypreParMatrix * matBhat = Bhat->ParallelAssemble();  delete Bhat;
   HypreParMatrix * matSinv = Sinv->ParallelAssemble();  delete Sinv;
   HypreParMatrix * matS0   = S0->ParallelAssemble();    delete S0;

   // 9. Define the block structure of the problem, by creating the offset
   //    variables. Also allocate two BlockVector objects to store the solution
   //    and rhs.
   enum {x0_var, xhat_var, NVAR};

   int true_s0     = x0_space->TrueVSize();
   int true_s1     = xhat_space->TrueVSize();
   int true_s_test = test_space->TrueVSize();

   Array<int> true_offsets(NVAR+1);
   true_offsets[0] = 0;
   true_offsets[1] = true_s0;
   true_offsets[2] = true_s0+true_s1;

   Array<int> true_offsets_test(2);
   true_offsets_test[0] = 0;
   true_offsets_test[1] = true_s_test;

   BlockVector x(true_offsets), b(true_offsets);
   x = 0.0;
   b = 0.0;

   // 10. Set up the 1x2 block Least Squares DPG operator, B = [B0 Bhat],
   //     the normal equation operator, A = B^t Sinv B, and
   //     the normal equation right-hand-size, b = B^t Sinv F.
   BlockOperator B(true_offsets_test, true_offsets);
   B.SetBlock(0, 0, matB0);
   B.SetBlock(0, 1, matBhat);

   RAPOperator A(B, *matSinv, B);

   HypreParVector *trueF = F->ParallelAssemble();
   {
      HypreParVector SinvF(test_space);
      matSinv->Mult(*trueF, SinvF);
      B.MultTranspose(SinvF, b);
   }

   // 11. Set up a block-diagonal preconditioner for the 2x2 normal equation
   //
   //        [ S0^{-1}     0     ]
   //        [   0     Shat^{-1} ]      Shat = (Bhat^T Sinv Bhat)
   //
   //     corresponding to the primal (x0) and interfacial (xhat) unknowns.
   //     Since the Shat operator is equivalent to an H(div) matrix reduced to
   //     the interfacial skeleton, we approximate its inverse with one V-cycle
   //     of the ADS preconditioner from the hypre library (in 2D we use AMS for
   //     the rotated H(curl) problem).
   HypreBoomerAMG *S0inv = new HypreBoomerAMG(*matS0);
   S0inv->SetPrintLevel(0);

   HypreParMatrix *Shat = RAP(matSinv, matBhat);
   HypreSolver *Shatinv;
   if (dim == 2) { Shatinv = new HypreAMS(*Shat, xhat_space); }
   else          { Shatinv = new HypreADS(*Shat, xhat_space); }

   BlockDiagonalPreconditioner P(true_offsets);
   P.SetDiagonalBlock(0, S0inv);
   P.SetDiagonalBlock(1, Shatinv);

   // 12. Solve the normal equation system using the PCG iterative solver.
   //     Check the weighted norm of residual for the DPG least square problem.
   //     Wrap the primal variable in a GridFunction for visualization purposes.
   CGSolver pcg(MPI_COMM_WORLD);
   pcg.SetOperator(A);
   pcg.SetPreconditioner(P);
   pcg.SetRelTol(1e-6);
   pcg.SetMaxIter(200);
   pcg.SetPrintLevel(1);
   pcg.Mult(b, x);

   {
      HypreParVector LSres(test_space), tmp(test_space);
      B.Mult(x, LSres);
      LSres -= *trueF;
      matSinv->Mult(LSres, tmp);
      real_t res = sqrt(InnerProduct(LSres, tmp));
      if (myid == 0)
      {
         cout << "\n|| B0*x0 + Bhat*xhat - F ||_{S^-1} = " << res << endl;
      }
   }

   x0->Distribute(x.GetBlock(x0_var));

   // 13. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x0->Save(sol_ofs);
   }

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << *x0 << flush;
   }

   // 15. Free the used memory.
   delete trueF;
   delete Shatinv;
   delete S0inv;
   delete Shat;
   delete matB0;
   delete matBhat;
   delete matSinv;
   delete matS0;
   delete x0;
   delete F;
   delete test_space;
   delete xhat_space;
   delete x0_space;
   delete test_fec;
   delete xhat_fec;
   delete x0_fec;
   delete pmesh;

   return 0;
}
