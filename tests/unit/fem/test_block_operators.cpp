// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

TEST_CASE("BlockOperators", "[BlockOperators], [GPU]")
{
   const int dim = 2, nx = 3, ny = 3, order = 2;
   Element::Type e_type = Element::QUADRILATERAL;
   Mesh mesh = Mesh::MakeCartesian2D(nx, ny, e_type);

   RT_FECollection rt_fe(order, dim);
   L2_FECollection l2_fe(order, dim);

   FiniteElementSpace R_fes(&mesh, &rt_fe), W_fes(&mesh, &l2_fe);

   Array<int> block_offsets(3);
   block_offsets[0] = 0;
   block_offsets[1] = R_fes.GetVSize();
   block_offsets[2] = W_fes.GetVSize();
   block_offsets.PartialSum();

   VectorFunctionCoefficient fcoeff(dim, [](const Vector &, Vector &f) { f = M_PI; }),
                             ucoeff(dim, [](const Vector &x, Vector &u)
   {
      const real_t xi(x(0)), yi(x(1)), zi(0.0);
      u(0) = -std::exp(xi) * std::sin(yi) * std::cos(zi);
      u(1) = -std::exp(xi) * std::cos(yi) * std::cos(zi);
   });
   auto pFun_ex = [](const Vector &x)
   {
      real_t xi(x(0)), yi(x(1)), zi(0.0);
      return std::exp(xi) * std::sin(yi) * std::cos(zi);
   };

   FunctionCoefficient fnatcoeff([&](const Vector &x) { return (-pFun_ex(x)); }),
   gcoeff([](const Vector &) {return M_PI_2;}), pcoeff(pFun_ex);

   const MemoryType mt = Device::GetMemoryType();
   BlockVector x(block_offsets, mt), y(block_offsets, mt), rhs(block_offsets, mt);

   LinearForm lf, lg;
   lf.Update(&R_fes, rhs.GetBlock(0), 0);
   lf.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
   lf.AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(fnatcoeff));
   lf.Assemble();
   lf.SyncAliasMemory(rhs);

   lg.Update(&W_fes, rhs.GetBlock(1), 0);
   lg.AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
   lg.Assemble();
   lg.SyncAliasMemory(rhs);

   BilinearForm vmass(&R_fes);
   ConstantCoefficient k(1.0);
   vmass.AddDomainIntegrator(new VectorFEMassIntegrator(k));
   vmass.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   vmass.Assemble();

   MixedBilinearForm vdiv(&R_fes, &W_fes);
   vdiv.AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   vdiv.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   vdiv.Assemble();

   BlockOperator blockOp(block_offsets);
   TransposeOperator Bt(vdiv);

   blockOp.SetBlock(0, 0, &vmass);
   blockOp.SetBlock(0, 1, &Bt, -1.0);
   blockOp.SetBlock(1, 0, &vdiv, -1.0);

   Vector Md(vmass.Height());
   vmass.AssembleDiagonal(Md);
   BlockDiagonalPreconditioner darcyDiagonalPrec(block_offsets);
   BlockLowerTriangularPreconditioner darcyLowerTriangularPrec(block_offsets);

   Vector invMd(Md);
   invMd.Reciprocal();

   Vector BMBt_diag(vdiv.Height());
   vdiv.AssembleDiagonal_ADAt(invMd, BMBt_diag);
   OperatorJacobiSmoother invM(Md, {}), invS(BMBt_diag, {});

   darcyDiagonalPrec.SetDiagonalBlock(0, &invM);
   darcyDiagonalPrec.SetDiagonalBlock(1, &invS);
   darcyLowerTriangularPrec.SetDiagonalBlock(0, &invM);
   darcyLowerTriangularPrec.SetDiagonalBlock(1, &invS);

   MINRESSolver solver;
   const auto rtol = 1e-6, atol = 1e-8;
   const auto print_lvl = 3, max_it = 100;
   solver.SetAbsTol(atol);
   solver.SetRelTol(rtol);
   solver.SetPrintLevel(print_lvl);
   solver.SetMaxIter(max_it);
   solver.SetOperator(blockOp);

   solver.SetPreconditioner(darcyDiagonalPrec);
   x = 0.0, solver.Mult(rhs, x);
   REQUIRE(solver.GetConverged());

   solver.SetPreconditioner(darcyLowerTriangularPrec);
   y = 0.0, solver.Mult(rhs, y);
   REQUIRE(solver.GetConverged());

   x -= y;
   REQUIRE(x.Normlinf() == MFEM_Approx(0.0));
}
