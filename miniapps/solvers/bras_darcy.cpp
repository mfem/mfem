// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
//            -------------------------------------------------------
//            BRAS Darcy Miniapp: BRAS in a Block Preconditioner
//            -------------------------------------------------------
//
// This miniapp illustrates the broken-space additive Schwarz (BRAS) mass
// matrix preconditioner, see BRASPreconditioner and [1], inside a block
// preconditioner for a saddle point system. It solves the mixed Darcy problem
// of Example 5,
//
//                                 u + grad p = f
//                                 - div u    = g
//
// with natural boundary condition -p = <given pressure>, discretized with
// Raviart-Thomas elements for the velocity u and discontinuous elements for
// the pressure p. The resulting saddle point system
//
//                           [ M  B^T ] [u] = [f]
//                           [ B   0  ] [p] = [g]
//
// is solved with MINRES, preconditioned with the block diagonal operator
//
//                           [ M_0^{-1}     0      ]
//                           [    0      AMG(S_0)  ],
//
// where S_0 = B M_0^{-1} B^T approximates the Schur complement B M^{-1} B^T.
// Two choices of M_0^{-1} are compared:
//
//    1. Jacobi, M_0^{-1} = diag(M)^{-1}, as in Example 5.
//    2. BRAS, M_0^{-1} = B_W, applied matrix-free in the velocity block and
//       assembled in S_0 (see BRASPreconditioner::ParallelAssembleMatrix).
//
// Since BRAS satisfies M^{-1} <= B_W <= kappa M^{-1}, where kappa depends only
// on the element-local spectral equivalence of the element matrices and their
// diagonals, the BRAS Schur complement approximation satisfies
// B M^{-1} B^T <= S_0 <= kappa B M^{-1} B^T, independently of the mesh size and
// of the refinement level of nonconforming meshes. The Jacobi approximation
// has no such bound, and deteriorates for some high-order bases, e.g. the
// default closed uniform basis (-b U). The price is a wider stencil of S_0,
// whose number of nonzeros is printed.
//
// [1]. O. A. Krzysik, B. S. Southworth, and G. A. Wimmer. Broken-space
//      additive Schwarz mass inverse approximations and (block)
//      preconditioning. arXiv:2607.12138. (2026)
//
// We recommend viewing Example 5 before viewing this miniapp.
//
// Compile with: make bras_darcy
//
// Sample runs:
//
//    mpirun -np 4 bras_darcy
//    mpirun -np 4 bras_darcy -o 3
//    mpirun -np 4 bras_darcy -o 3 -b G
//    mpirun -np 4 bras_darcy -m ../../data/amr-quad.mesh -rs 2 -o 2
//    mpirun -np 4 bras_darcy -m ../../data/inline-hex.mesh -rs 0 -o 1
//    mpirun -np 4 bras_darcy -m ../../data/star.mesh -rs 1 -o 2
//
// Device sample runs:
//  * mpirun -np 4 bras_darcy -m ../../data/inline-hex.mesh -rs 1 -o 3 -d cuda

#include "mfem.hpp"
#include <iostream>
#include <memory>

using namespace std;
using namespace mfem;

// Exact solution, u and p, and r.h.s., f and g, of Example 5.
void u_exact(const Vector &x, Vector &u);
real_t p_exact(const Vector &x);
void f_exact(const Vector &x, Vector &f);
real_t g_exact(const Vector &x);
real_t natural_bc(const Vector &x);

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   const char *mesh_file = "../../data/inline-quad.mesh";
   int ser_ref_levels = 2, par_ref_levels = 0;
   int order = 2;
   const char *basis = "U";
   const char *device_config = "cpu";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Order of the Raviart-Thomas and L2 spaces.");
   args.AddOption(&basis, "-b", "--basis-type",
                  "Closed basis of the Raviart-Thomas space: G - Gauss-Lobatto, "
                  "U - closed uniform, c - closed Gauss-Legendre.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.ParseCheck();

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   Mesh serial_mesh(mesh_file, 1, 1);
   const int dim = serial_mesh.Dimension();
   for (int l = 0; l < ser_ref_levels; l++) { serial_mesh.UniformRefinement(); }
#ifndef MFEM_USE_METIS
   // Without METIS, the parallel mesh is partitioned along the space-filling
   // curve of a nonconforming mesh.
   serial_mesh.EnsureNCMesh(true);
#endif
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();
   for (int l = 0; l < par_ref_levels; l++) { mesh.UniformRefinement(); }

   // The open basis of the Raviart-Thomas space belongs to the same family as
   // the closed basis given by -b.
   const int b1 = BasisType::GetType(basis[0]);
   const int b2 = (b1 == BasisType::ClosedUniform) ? BasisType::OpenUniform :
                  (b1 == BasisType::ClosedGL) ? BasisType::OpenHalfUniform :
                  BasisType::GaussLegendre;
   RT_FECollection hdiv_fec(order, dim, b1, b2);
   L2_FECollection l2_fec(order, dim);
   ParFiniteElementSpace R_space(&mesh, &hdiv_fec);
   ParFiniteElementSpace W_space(&mesh, &l2_fec);

   const HYPRE_BigInt dimR = R_space.GlobalTrueVSize();
   const HYPRE_BigInt dimW = W_space.GlobalTrueVSize();
   const long long nelem = mesh.GetGlobalNE();
   if (Mpi::Root())
   {
      cout << "Number of elements: " << nelem << '\n'
           << "dim(R) = " << dimR << ", dim(W) = " << dimW << endl;
   }

   Array<int> offsets({0, R_space.GetTrueVSize(), W_space.GetTrueVSize()});
   offsets.PartialSum();

   // Right-hand side, as in Example 5.
   VectorFunctionCoefficient fcoeff(dim, f_exact), ucoeff(dim, u_exact);
   FunctionCoefficient natcoeff(natural_bc), gcoeff(g_exact), pcoeff(p_exact);
   const MemoryType mt = device.GetMemoryType();
   BlockVector rhs(offsets, mt), x(offsets, mt);
   ParLinearForm fform(&R_space);
   fform.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
   fform.AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(natcoeff));
   fform.Assemble();
   fform.ParallelAssemble(rhs.GetBlock(0));
   ParLinearForm gform(&W_space);
   gform.AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
   gform.Assemble();
   gform.ParallelAssemble(rhs.GetBlock(1));

   // The Darcy operator. Natural boundary conditions are imposed on the whole
   // boundary, so there are no essential DOFs.
   Array<int> ess_tdof_list;
   ParBilinearForm m(&R_space);
   m.AddDomainIntegrator(new VectorFEMassIntegrator);
   m.Assemble();
   m.Finalize();
   unique_ptr<HypreParMatrix> M(m.ParallelAssemble());
   ParMixedBilinearForm bform(&R_space, &W_space);
   bform.AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   bform.Assemble();
   bform.Finalize();
   unique_ptr<HypreParMatrix> B(bform.ParallelAssemble());
   *B *= -1.0;
   unique_ptr<HypreParMatrix> Bt(B->Transpose());

   BlockOperator darcy(offsets);
   darcy.SetBlock(0, 0, M.get());
   darcy.SetBlock(0, 1, Bt.get());
   darcy.SetBlock(1, 0, B.get());

   ParGridFunction u(&R_space), p(&W_space);
   const int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }
   const real_t norm_u = ComputeGlobalLpNorm(2, ucoeff, mesh, irs);
   const real_t norm_p = ComputeGlobalLpNorm(2, pcoeff, mesh, irs);

   // Solve with MINRES and the block diagonal preconditioner built from
   // invM and S, and print the results. The setup time is measured by the
   // caller; the solve time includes the setup of AMG(S).
   auto solve = [&](Solver &invM, const HypreParMatrix &S, const char *name,
                    real_t setup_time)
   {
      HypreBoomerAMG invS(S);
      invS.SetPrintLevel(0);
      invM.iterative_mode = false;
      invS.iterative_mode = false;
      BlockDiagonalPreconditioner prec(offsets);
      prec.SetDiagonalBlock(0, &invM);
      prec.SetDiagonalBlock(1, &invS);

      MINRESSolver minres(MPI_COMM_WORLD);
      minres.SetRelTol(1e-10);
      minres.SetAbsTol(0.0);
      minres.SetMaxIter(2000);
      minres.SetPrintLevel(0);
      minres.SetOperator(darcy);
      minres.SetPreconditioner(prec);
      x = 0.0;
      StopWatch chrono;
      chrono.Start();
      minres.Mult(rhs, x);
      chrono.Stop();

      u.Distribute(x.GetBlock(0));
      p.Distribute(x.GetBlock(1));
      const real_t err_u = u.ComputeL2Error(ucoeff, irs);
      const real_t err_p = p.ComputeL2Error(pcoeff, irs);
      const HYPRE_BigInt nnz = S.NNZ();
      if (Mpi::Root())
      {
         cout << name << ": " << minres.GetNumIterations() << " iterations"
              << (minres.GetConverged() ? "" : " (not converged)")
              << ", nnz(S) " << nnz << ", setup " << setup_time
              << " s, solve " << chrono.RealTime() << " s\n"
              << "   || u_h - u_ex || / || u_ex || = " << err_u / norm_u
              << "\n   || p_h - p_ex || / || p_ex || = " << err_p / norm_p
              << endl;
      }
   };

   // 1. Jacobi: S = B diag(M)^{-1} B^T.
   {
      StopWatch chrono;
      chrono.Start();
      HypreDiagScale invM(*M);
      Vector Md;
      M->GetDiag(Md);
      unique_ptr<HypreParMatrix> MinvBt(B->Transpose());
      MinvBt->InvScaleRows(Md);
      unique_ptr<HypreParMatrix> S(ParMult(B.get(), MinvBt.get()));
      chrono.Stop();
      solve(invM, *S, "Jacobi", chrono.RealTime());
   }

   // 2. BRAS: S = B B_W B^T.
   {
      StopWatch chrono;
      chrono.Start();
      BRASPreconditioner invM(m, ess_tdof_list);
      unique_ptr<HypreParMatrix> BW(invM.ParallelAssembleMatrix());
      unique_ptr<HypreParMatrix> S(RAP(BW.get(), Bt.get()));
      chrono.Stop();
      solve(invM, *S, "BRAS", chrono.RealTime());
   }

   // Send the BRAS solution by socket to a GLVis server.
   if (visualization)
   {
      socketstream u_sock("localhost", 19916);
      u_sock << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank()
             << "\n";
      u_sock.precision(8);
      u_sock << "solution\n" << mesh << u << "window_title 'Velocity'" << endl;
      // Make sure all ranks have sent their 'u' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(mesh.GetComm());
      socketstream p_sock("localhost", 19916);
      p_sock << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank()
             << "\n";
      p_sock.precision(8);
      p_sock << "solution\n" << mesh << p << "window_title 'Pressure'" << endl;
   }

   return 0;
}

void u_exact(const Vector &x, Vector &u)
{
   const real_t xi(x(0));
   const real_t yi(x(1));
   const real_t zi(x.Size() == 3 ? x(2) : 0.0);

   u(0) = - exp(xi)*sin(yi)*cos(zi);
   u(1) = - exp(xi)*cos(yi)*cos(zi);
   if (x.Size() == 3)
   {
      u(2) = exp(xi)*sin(yi)*sin(zi);
   }
}

real_t p_exact(const Vector &x)
{
   const real_t xi(x(0));
   const real_t yi(x(1));
   const real_t zi(x.Size() == 3 ? x(2) : 0.0);
   return exp(xi)*sin(yi)*cos(zi);
}

void f_exact(const Vector &x, Vector &f)
{
   f = 0.0;
}

real_t g_exact(const Vector &x)
{
   if (x.Size() == 3) { return -p_exact(x); }
   return 0.0;
}

real_t natural_bc(const Vector &x)
{
   return (-p_exact(x));
}
