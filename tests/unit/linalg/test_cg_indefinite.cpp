// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

class FakeSolver : public Solver
{
private:
   Operator& op;

public:
   FakeSolver(Operator& op_) : Solver(op_.Height()), op(op_) { }
   void SetOperator(const Operator &op_) override { }
   void Mult(const Vector& x, Vector& y) const override { op.Mult(x, y); }
};

TEST_CASE("CGSolver", "[Indefinite]")
{
   mfem::out << "===> BEGIN: Expected PCG warning messages" << std::endl;

   // Define indefinite SparseMatrix
   SparseMatrix indefinite(2, 2);
   indefinite.Add(0, 0, -1.0);
   indefinite.Finalize();

   Vector v(2);
   v(0) = 1.0;
   v(1) = 1.0;
   Vector x(2);
   x = 0.0;

   // Check indefinite operator - this particular example does not make sense
   // with CG and should not converge
   CGSolver cg;
   cg.SetOperator(indefinite);
   cg.SetPrintLevel(1);
   cg.Mult(v, x);
   REQUIRE(!cg.GetConverged());

   // Check indefinite preconditioner - this particular example does not make
   // sense with CG and should not converge
   IdentityOperator identity(2);
   FakeSolver indefprec(indefinite);
   CGSolver cg2;
   cg2.SetOperator(identity);
   cg2.SetPreconditioner(indefprec);
   cg2.SetPrintLevel(1);
   x = 0.0;
   cg2.Mult(v, x);
   REQUIRE(!cg2.GetConverged());

   mfem::out << "===> END: Expected PCG warning messages" << std::endl;
}
