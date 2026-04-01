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

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_PETSC)

namespace
{
struct PetscSession
{
   PetscSession() { MFEMInitializePetsc(); }
   ~PetscSession() { MFEMFinalizePetsc(); }
};

class IdentityGradientOperator : public IdentityOperator
{
public:
   IdentityGradientOperator() : IdentityOperator(1), _jac(1)
   {
      _jac.Add(0, 0, 1.0);
      _jac.Finalize();
   }

   Operator &GetGradient(const Vector &) const override
   {
      return const_cast<SparseMatrix &>(_jac);
   }

private:
   SparseMatrix _jac;
};
}

TEST_CASE("PetscNonlinearSolver accepts non-empty rhs", "[Parallel][PETSc]")
{
   static PetscSession petsc_session;

   IdentityGradientOperator oper;
   PetscNonlinearSolver solver(MPI_COMM_WORLD, "nl_");
   solver.SetRelTol(1.0e-12);
   solver.SetAbsTol(1.0e-12);
   solver.SetMaxIter(5);
   solver.SetPrintLevel(0);
   solver.SetJacobianType(Operator::PETSC_MATAIJ);
   solver.SetOperator(oper);

   Vector rhs(1);
   rhs(0) = 2.5;

   Vector x(1);
   x = 0.0;

   solver.Mult(rhs, x);

   REQUIRE(x.Size() == 1);
   REQUIRE(x(0) == MFEM_Approx(rhs(0)));
}

#endif
