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
//                  ---------------------------------------------
//                  BRAS Mass Matrix Preconditioner Miniapp
//                  ---------------------------------------------
//
// This miniapp illustrates the broken-space additive Schwarz (BRAS)
// preconditioner for mass matrices, see BRASPreconditioner and [1]. The L2
// projection M x = b of a given function onto an H1, H(curl), H(div), or L2
// finite element space is solved with CG, preconditioned with Jacobi (the
// diagonal of M) and with BRAS, and the iteration counts and times of the two
// solves are compared.
//
// BRAS applies one element-local solve per element. With the default diagonal
// weighting, its condition number bound depends only on the element-local
// spectral equivalence of the element mass matrices and their diagonals, so it
// is independent of the mesh size and of the local refinement. On
// nonconforming meshes (option -amr), the hanging DOFs are treated with the
// left-inverse approximation of BRAS (default) or a Chebyshev approximation of
// degree -cheb. Option -t selects the sum factorization of the element solves,
// which is available on affine quadrilateral and hexahedral meshes (and, for
// H(curl) and H(div), only on rectangular elements).
//
// [1]. O. A. Krzysik, B. S. Southworth, and M. Wimmer. Weighted broken-space
//      additive Schwarz preconditioning for high-order mass matrices.
//      arXiv:2607.12138. (2026)
//
// Compile with: make bras_mass
//
// Sample runs:
//
//    mpirun -np 4 bras_mass -fe h
//    mpirun -np 4 bras_mass -fe n
//    mpirun -np 4 bras_mass -fe r
//    mpirun -np 4 bras_mass -fe l
//    mpirun -np 4 bras_mass -fe h -b U -o 5
//    mpirun -np 4 bras_mass -fe r -amr 4 -t
//    mpirun -np 4 bras_mass -fe n -amr 4 -cheb 3
//    mpirun -np 4 bras_mass -m ../../data/inline-hex.mesh -rs 1 -fe r -amr 2
//    mpirun -np 4 bras_mass -m ../../data/star.mesh -fe h -iw
//    mpirun -np 4 bras_mass -m ../../data/inline-tet.mesh -rs 0 -fe n
//
// Device sample runs:
//  * mpirun -np 4 bras_mass -m ../../data/inline-hex.mesh -fe h -pa -t -d cuda
//  * mpirun -np 4 bras_mass -m ../../data/inline-hex.mesh -fe r -pa -d cuda

#include "mfem.hpp"
#include <iostream>
#include <memory>

#include "lor_mms.hpp"

using namespace std;
using namespace mfem;

// Refine the elements that cross the circle (2D) or sphere (3D) with the given
// center and radius, i.e. that have vertices on both sides of it.
void RefineFeature(Mesh &mesh, const Vector &center, real_t radius,
                   int nc_limit)
{
   Array<int> refs, v;
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      mesh.GetElementVertices(e, v);
      bool inside = false, outside = false;
      for (int i : v)
      {
         const real_t d = center.DistanceTo(mesh.GetVertex(i));
         inside = inside || d < radius;
         outside = outside || d >= radius;
      }
      if (inside && outside) { refs.Append(e); }
   }
   mesh.GeneralRefinement(refs, 1, nc_limit);
}

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   const char *mesh_file = "../../data/inline-quad.mesh";
   int ser_ref_levels = 2, par_ref_levels = 0;
   int amr_steps = 0, nc_limit = 0;
   int order = 3;
   const char *fe = "h";
   const char *basis = "G";
   bool identity_weighting = false;
   bool tensor = false;
   int cheb_degree = 0;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&amr_steps, "-amr", "--amr-steps",
                  "Number of nonconforming refinements of the elements "
                  "crossing a circle (2D) or sphere (3D) around the center "
                  "of the domain, after the serial uniform refinements.");
   args.AddOption(&nc_limit, "-ncl", "--nc-limit",
                  "Maximum level of hanging nodes (0 = unlimited).");
   args.AddOption(&order, "-o", "--order", "Polynomial degree.");
   args.AddOption(&fe, "-fe", "--fe-type",
                  "FE type. h for H1, n for Hcurl, r for Hdiv, l for L2");
   args.AddOption(&basis, "-b", "--basis-type",
                  "Basis: G - Gauss-Lobatto, U - closed uniform, "
                  "c - closed Gauss-Legendre, P - positive (H1 and L2 only).");
   args.AddOption(&identity_weighting, "-iw", "--identity-weighting", "-dw",
                  "--diagonal-weighting",
                  "Use the identity or the diagonal BRAS weighting.");
   args.AddOption(&tensor, "-t", "--tensor", "-no-t", "--no-tensor",
                  "Apply the element inverses by sum factorization.");
   args.AddOption(&cheb_degree, "-cheb", "--chebyshev-degree",
                  "Degree of the Chebyshev approximation of A_W^{-1} on "
                  "nonconforming spaces; 0 selects the left-inverse "
                  "approximation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly",
                  "Use partial assembly for the mass operator.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.ParseCheck();

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   bool H1 = false, ND = false, RT = false, L2 = false;
   if (string(fe) == "h") { H1 = true; }
   else if (string(fe) == "n") { ND = true; }
   else if (string(fe) == "r") { RT = true; }
   else if (string(fe) == "l") { L2 = true; }
   else { MFEM_ABORT("Bad FE type. Must be 'h', 'n', 'r', or 'l'."); }

   Mesh serial_mesh(mesh_file, 1, 1);
   const int dim = serial_mesh.Dimension();
   const int sdim = serial_mesh.SpaceDimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "Mesh dimension must be 2 or 3.");
   for (int l = 0; l < ser_ref_levels; l++) { serial_mesh.UniformRefinement(); }
#ifndef MFEM_USE_METIS
   // Without METIS, the parallel mesh is partitioned along the space-filling
   // curve of a nonconforming mesh.
   serial_mesh.EnsureNCMesh(true);
#endif
   if (amr_steps > 0)
   {
      serial_mesh.EnsureNCMesh(true);
      Vector pmin, pmax, center(sdim);
      serial_mesh.GetBoundingBox(pmin, pmax);
      add(0.5, pmin, 0.5, pmax, center);
      real_t radius = infinity();
      for (int d = 0; d < sdim; d++)
      {
         radius = min(radius, 0.25*(pmax(d) - pmin(d)));
      }
      for (int l = 0; l < amr_steps; l++)
      {
         RefineFeature(serial_mesh, center, radius, nc_limit);
      }
   }
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();
   for (int l = 0; l < par_ref_levels; l++) { mesh.UniformRefinement(); }

   // The H(curl) and H(div) bases combine the closed basis given by -b with
   // an open basis of the same family.
   const int b1 = BasisType::GetType(basis[0]);
   const int b2 = (b1 == BasisType::ClosedUniform) ? BasisType::OpenUniform :
                  (b1 == BasisType::ClosedGL) ? BasisType::OpenHalfUniform :
                  BasisType::GaussLegendre;
   unique_ptr<FiniteElementCollection> fec;
   if (H1) { fec.reset(new H1_FECollection(order, dim, b1)); }
   else if (ND) { fec.reset(new ND_FECollection(order, dim, b1, b2)); }
   else if (RT) { fec.reset(new RT_FECollection(order-1, dim, b1, b2)); }
   else { fec.reset(new L2_FECollection(order, dim, b1)); }

   ParFiniteElementSpace fes(&mesh, fec.get());
   const HYPRE_BigInt ndofs = fes.GlobalTrueVSize();
   const long long nelem = mesh.GetGlobalNE();
   if (Mpi::Root())
   {
      cout << "Number of elements: " << nelem << '\n'
           << "Number of DOFs: " << ndofs << endl;
   }

   ParBilinearForm m(&fes);
   if (H1 || L2) { m.AddDomainIntegrator(new MassIntegrator); }
   else { m.AddDomainIntegrator(new VectorFEMassIntegrator); }
   if (pa) { m.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   m.Assemble();
   Array<int> ess_dofs; // The L2 projection has no essential DOFs.
   OperatorPtr M;
   m.FormSystemMatrix(ess_dofs, M);

   FunctionCoefficient u_coeff(u);
   VectorFunctionCoefficient u_vec_coeff(sdim, u_vec);
   ParLinearForm b(&fes);
   if (H1 || L2) { b.AddDomainIntegrator(new DomainLFIntegrator(u_coeff)); }
   else { b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(u_vec_coeff)); }
   b.Assemble();
   Vector B(fes.GetTrueVSize()), X(fes.GetTrueVSize());
   B.UseDevice(true);
   X.UseDevice(true);
   b.ParallelAssemble(B);
   ParGridFunction x(&fes);

   // Solve with CG and the given preconditioner, and print the results.
   auto solve = [&](Solver &prec, const char *name, real_t setup_time)
   {
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetAbsTol(0.0);
      cg.SetMaxIter(2000);
      cg.SetPrintLevel(0);
      cg.SetOperator(*M);
      cg.SetPreconditioner(prec);
      X = 0.0;
      StopWatch chrono;
      chrono.Start();
      cg.Mult(B, X);
      chrono.Stop();
      x.Distribute(X);
      const real_t er = (H1 || L2) ? x.ComputeL2Error(u_coeff) :
                        x.ComputeL2Error(u_vec_coeff);
      if (Mpi::Root())
      {
         cout << name << ": " << cg.GetNumIterations() << " iterations"
              << (cg.GetConverged() ? "" : " (not converged)")
              << ", setup " << setup_time << " s, solve " << chrono.RealTime()
              << " s, L2 error " << er << endl;
      }
   };

   StopWatch chrono;
   chrono.Start();
   OperatorJacobiSmoother jacobi(m, ess_dofs);
   chrono.Stop();
   solve(jacobi, "Jacobi", chrono.RealTime());

   using BRAS = BRASPreconditioner;
   chrono.Restart();
   BRAS bras(m, ess_dofs, identity_weighting ? BRAS::Weighting::IDENTITY :
             BRAS::Weighting::DIAGONAL, tensor ? BRAS::ElementInverse::TENSOR :
             BRAS::ElementInverse::DENSE);
   if (cheb_degree > 0)
   {
      bras.SetNCMethod(BRAS::NCMethod::CHEBYSHEV, cheb_degree);
   }
   chrono.Stop();
   solve(bras, "BRAS", chrono.RealTime());

   // Send the BRAS solution by socket to a GLVis server.
   if (visualization)
   {
      socketstream sol_sock("localhost", 19916);
      sol_sock << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank()
               << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << x << flush;
   }

   return 0;
}
