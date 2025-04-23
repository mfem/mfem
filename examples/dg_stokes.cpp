//                                          MFEM Example dg_stokes
//
// Compile with: make dg_stokes
//
// Sample run: mpirun -np 4 ex34
//
// Description:   This example code solves 2D Stokes equation over the square [-1,1]x[-1,1]
//
//                                     -Delta u + grad p = f
//                                                 div u = g
//
//                with nonzero dirichlet boundary conditions, zero mean condition for pressure,
//                and source terms f and g defined by a manufactured solution
//
//                                           u_x = (1 - y^2)
//                                           u_y = (1 - x^2)
//                                           p = x
//
//                The problem is discretized using the SIP DG method described in chapter 6 of
//                B. Rivière (2006) [1]. Specifically, we discretize with L2 finite elements
//                for velocity u and pressure p of order k and k-1 respectively.
//
//                This example demonstrates the use of user-defined bilinear form integrators, block
//                vectors and matrices, how to create vector forms of exisiting scalar integrators,
//                and how to apply a zero mean condition on part of a block vector.
//
//                We recommend viewing examples 5, 9, and 14 before viewing this example.
//
//                [1]  Rivière B. DG Methods for Solving Elliptic and Parabolic Equations Theory and
//                     Implementation. Society for Industrial and Applied Mathematics; 2008.
//                     doi: 10.1137/1.9780898717440

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "dg_stokes.hpp"
#include <unistd.h>

using namespace std;
using namespace mfem;

// Manufactured solution
double p_exact(const Vector &xvec);
void u_exact(const Vector &xvec, Vector &u);

// Source Terms
void f_source(const Vector &xvec, Vector &u);
double g_source(const Vector &xvec);

// Remove mean from a ParGridFunction (modified from Navier Miniapp)
void MeanZero(ParGridFunction &v);

int main(int argc, char *argv[])
{

   // 1. Specify options
   Mpi::Init(argc, argv);
   Hypre::Init();
   // const char *mesh_file = "../data/square.mesh";
   // int ref_levels = 2;
   const char *mesh_file = "../data/StokesMixedMesh.msh";
   int ref_levels = 0;
   int order = 2;

   bool visualization = true;
   int vdim = 2;
   int precision = 12;
   cout.precision(precision);

   // 2. Read in mesh from the given mesh file and print out its
   //    characteristics
   Mesh mesh(mesh_file,1,1);
   int dim = mesh.Dimension();
   mesh.PrintCharacteristics();

   // 3. Perform any uniform mesh refinement, with the number of
   //    uniform refinements given by 'ref_levels' and print out
   //    the characteristics of the refined mesh. Also form parallel mesh
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }
   if (ref_levels > 0)
   {
      cout << "After " << ref_levels << " of uniform refinement:\n"
           << endl;
      mesh.PrintCharacteristics();
   }
   auto *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // 4. Define and finite element space on the mesh. Here we use discontinuous
   //    DG finite element space of specified order for velocity and one order less
   //    for pressure on the refined mesh.
   DG_FECollection ufec(order, dim, BasisType::GaussLobatto);
   DG_FECollection pfec(order-1, dim, BasisType::GaussLobatto);
   ParFiniteElementSpace ufes(pmesh, &ufec, vdim, Ordering::byNODES);
   ParFiniteElementSpace pfes(pmesh, &pfec);
         
   if (Mpi::Root())
   {
      cout << "Number of unknowns: " << ufes.GlobalTrueVSize() +
           pfes.GlobalTrueVSize() << endl;
   }

   // 6. Setup the BlockStructure of the problem, i.e. define the array of
   //    offsets for each variable. The last component of the Array is the sum
   //    of the dimensions of each block.
   Array<int> block_offsets(3); // number of variables + 1
   block_offsets[0] = 0;
   block_offsets[1] = ufes.GetVSize();
   block_offsets[2] = pfes.GetVSize();
   block_offsets.PartialSum();

   Array<int> trueblock_offsets(3); // number of variables + 1
   trueblock_offsets[0] = 0;
   trueblock_offsets[1] = ufes.GetTrueVSize();
   trueblock_offsets[2] = pfes.GetTrueVSize();
   trueblock_offsets.PartialSum();

   HYPRE_BigInt dimU = ufes.GlobalTrueVSize();
   HYPRE_BigInt dimP = pfes.GlobalTrueVSize();

   if (Mpi::Root())
   {
      std::cout << "***********************************************************\n";
      std::cout << "dim(u) = " << dimU << "\n";
      std::cout << "dim(p) = " << dimP << "\n";
      std::cout << "dim(u+p) = " << dimU + dimP << "\n";
      std::cout << "***********************************************************\n";
   }

   const char *device_config = "cpu";
   Device device(device_config);
   MemoryType mt = device.GetMemoryType();

   BlockVector x(block_offsets, mt);
   BlockVector rhs(block_offsets, mt);
   BlockVector truex(trueblock_offsets, mt);
   BlockVector truerhs(trueblock_offsets, mt);
   x = 0.0;
   rhs = 0.0;
   truex = 0.0;
   truerhs = 0.0;

   /*
      x = [ u ]      rhs = [ f ]
          [ p ]            [ g ]
   */

   // 7. Setup and assemble the bilinear and linear forms corresponding to the
   //    DG discretization.

   // diffusion term
   ParBilinearForm a(&ufes);
   double sigma = -1.0, kappa = 10 * (order + 1) * (order + 1);
   a.AddDomainIntegrator(new VectorDiffusionIntegrator(vdim));
   a.AddInteriorFaceIntegrator(new VectorDGDiffusionIntegrator(sigma,kappa,vdim));
   a.AddBdrFaceIntegrator(new VectorDGDiffusionIntegrator(sigma,kappa,vdim));
   a.Assemble();
   a.Finalize();

   // rhs of momentum equation
   ParLinearForm *f(new ParLinearForm);
   f->Update(&ufes, rhs.GetBlock(0),0);
   VectorFunctionCoefficient momentum_source(vdim,f_source);
   VectorFunctionCoefficient velocity_dbc(vdim,u_exact);
   f->AddBdrFaceIntegrator(new VectorDGDirichletLFIntegrator(velocity_dbc,sigma,
                                                             kappa,vdim));
   f->AddDomainIntegrator(new VectorDomainLFIntegrator(momentum_source));
   f->Assemble();
   f->SyncAliasMemory(rhs);
   f->ParallelAssemble(truerhs.GetBlock(0));
   truerhs.GetBlock(0).SyncAliasMemory(truerhs);

   // grad pressure term
   pfes.ExchangeFaceNbrData();
   ufes.ExchangeFaceNbrData();
   ParMixedBilinearForm b(&pfes,&ufes); // (trial,test)
   ConstantCoefficient minusOne(-1.0);
   b.AddDomainIntegrator(new TransposeIntegrator(new VectorDivergenceIntegrator(
                                                    minusOne)));
   b.AddInteriorFaceIntegrator(new DGAvgNormalJumpIntegrator(vdim));
   b.AddBdrFaceIntegrator(new DGAvgNormalJumpIntegrator(vdim));
   b.Assemble();
   b.Finalize();

   // rhs for continuity equation
   ParLinearForm *g(new ParLinearForm);
   g->Update(&pfes, rhs.GetBlock(1), 0);
   FunctionCoefficient mass_source(g_source);
   g->AddBdrFaceIntegrator(new DG_BoundaryNormalLFIntegrator(velocity_dbc));
   g->AddDomainIntegrator(new DomainLFIntegrator(mass_source));
   g->Assemble();
   g->SyncAliasMemory(rhs);
   g->ParallelAssemble(truerhs.GetBlock(1));
   truerhs.GetBlock(1).SyncAliasMemory(truerhs);

   // 8. Setup a block matrix for the Stokes operator
   /*
      S = [ A    B ] [ u ] = [ f ]
          [ B^T  0 ] [ p ] = [ g ]
   */
   BlockOperator stokesOp(block_offsets);

   HypreParMatrix *A = a.ParallelAssemble();
   HypreParMatrix *B = b.ParallelAssemble();
   HypreParMatrix *Bt = B->Transpose();

   stokesOp.SetBlock(0, 0, A);
   stokesOp.SetBlock(0, 1, B);
   stokesOp.SetBlock(1, 0, Bt);

   // 9. Setup a very simple block diagonal preconditioner
   /*
      P = [ diag(A)             0       ]
          [ 0                   I       ]
   */
   BlockDiagonalPreconditioner stokesPrec(block_offsets);
   Solver *invM;

   invM = new HypreDiagScale(*A);
   invM->iterative_mode = false;

   Vector id_vec(b.Width());
   id_vec = 1.0;
   SparseMatrix id_mat(id_vec);

   stokesPrec.SetDiagonalBlock(0, invM);
   stokesPrec.SetDiagonalBlock(1, &id_mat);

   // 10. Solve the linear system with MINRES solver
   int maxIter(50000);
   double rtol(1.e-14);
   double atol(1.e-14);

   //Wrapper to ensure mean of pressure is zero after application of preconditioner
   BlockOrthoSolver stokesPrec_wrap(block_offsets);
   stokesPrec_wrap.SetSolver(stokesPrec);

   StopWatch chrono;
   chrono.Clear();
   chrono.Start();

   MINRESSolver solver(MPI_COMM_WORLD);
   solver.SetAbsTol(atol);
   solver.SetRelTol(rtol);
   solver.SetMaxIter(maxIter);
   solver.SetPreconditioner(stokesPrec_wrap);
   solver.SetOperator(stokesOp);
   solver.SetPrintLevel(1);

   // Wrapper to ensure mean of pressure is zero after each iteration of MINRES
   BlockOrthoSolver solver_wrap(block_offsets, MPI_COMM_WORLD);
   solver_wrap.SetSolver(solver);

   // Solver the system
   x = 0.0; // initial guess of 0
   solver_wrap.Mult(rhs, x);
   chrono.Stop();

   if (Mpi::Root())
   {
      if (solver.GetConverged())
      {
         std::cout << "MINRES converged in " << solver.GetNumIterations()
                   << " iterations with a residual norm of "
                   << solver.GetFinalNorm() << ".\n";
      }
      else
      {
         std::cout << "MINRES did not converge in " << solver.GetNumIterations()
                   << " iterations. Residual norm is " << solver.GetFinalNorm()
                   << ".\n";
      }
      std::cout << "Solver solver took " << chrono.RealTime() << "s.\n";
   }

   // 11. Create the grid functions u and p. Compute the L2 error norms.
   ParGridFunction u, p;
   u.MakeRef(&ufes, x.GetBlock(0), 0);
   p.MakeRef(&pfes, x.GetBlock(1), 0);
   FunctionCoefficient pressure(p_exact);
   VectorFunctionCoefficient velocity(vdim,u_exact);

   // remove mean from pressure approximation
   MeanZero(p);

   // zero mean of exact pressure solution
   ParGridFunction p_exact_gf(&pfes);
   p_exact_gf.ProjectCoefficient(pressure);
   MeanZero(p_exact_gf);
   GridFunctionCoefficient p_ex_zero_mean_coeff(&p_exact_gf);

   // compute L2 error
   int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   double err_u  = u.ComputeL2Error(velocity, irs);
   double norm_u = ComputeGlobalLpNorm(2., velocity, *pmesh, irs);
   double err_p = p.ComputeL2Error(p_ex_zero_mean_coeff, irs);
   double norm_p = ComputeGlobalLpNorm(2, p_ex_zero_mean_coeff, *pmesh, irs);

   if (Mpi::Root())
   {
      std::cout << "|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";
      std::cout << "|| p_h - p_ex || / || p_ex || = " << err_p / norm_p << "\n";
   }

   // 12. Save the mesh and the solution. This output can be viewed later using
   //     GLVis: "glvis -m Stokes.mesh -g sol_u.gf" or "glvis -m Stokes.mesh -g
   //     sol_p.gf".
   {
      ofstream mesh_ofs("dg_stokes.mesh");
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream u_ofs("dg_stokes_sol_u.gf");
      u_ofs.precision(8);
      u.Save(u_ofs);

      ofstream p_ofs("dg_stokes_sol_p.gf");
      p_ofs.precision(8);
      p.Save(p_ofs);
   }

   // 13. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream u_sock(vishost, visport);
      u_sock.precision(8);
      u_sock << "parallel " << Mpi::WorldSize() << " "
             << Mpi::WorldRank() << "\n";
      u_sock << "solution\n" << *pmesh << u << "window_title 'Velocity'" << endl;
      MPI_Barrier(pmesh->GetComm());

      socketstream p_sock(vishost, visport);
      p_sock.precision(8);
      p_sock << "parallel " << Mpi::WorldSize() << " "
             << Mpi::WorldRank() << "\n";
      p_sock << "solution\n" << *pmesh << p << "window_title 'Pressure'" << endl;
      MPI_Barrier(pmesh->GetComm());
   }

   // 14. Free the used memory.
   delete f;
   delete g;
   delete invM;

   return 0;
}

double p_exact(const Vector &xvec)
{
   double x = xvec(0);
   double y = xvec(1);

   return x;
}

void u_exact(const Vector &xvec, Vector &u)
{
   double x = xvec(0);
   double y = xvec(1);

   u(0) = (1-y*y);
   u(1) = (1-x*x);
}

void f_source(const Vector &xvec, Vector &u)
{
   u(0) = 3;
   u(1) = 2;
}

double g_source(const Vector &xvec)
{
   return 0.0;
}

void MeanZero(ParGridFunction &v)
{
   ConstantCoefficient onecoeff(1.0);
   ParLinearForm mass_lf(v.ParFESpace());
   auto *dlfi = new DomainLFIntegrator(onecoeff);

   mass_lf.AddDomainIntegrator(dlfi);
   mass_lf.Assemble();

   ParGridFunction one_gf(v.ParFESpace());
   one_gf.ProjectCoefficient(onecoeff);

   double volume = mass_lf.operator()(one_gf);
   double integ = mass_lf.operator()(v);

   v -= integ / volume;
}