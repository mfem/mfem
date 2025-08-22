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

#ifndef MFEM_FILTEREDSOLVER
#define MFEM_FILTEREDSOLVER

#include "../config/config.hpp"
#include "operator.hpp"
#include "handle.hpp"


namespace mfem
{

class FilteredSolver : public Solver
{
public:
   FilteredSolver() : Solver() { }

   void SetOperator(const Operator &A);
   void SetSolver(Solver &M);

   // Transfer operator from Filtered subspace to the original space
   void SetFilteredSubspaceTransferOperator(const Operator &P);
   void SetFilteredSubspaceSolver(Solver &Mf);

   void Mult(const Vector &x, Vector &y) const;

   ~FilteredSolver();

private:

   const Operator * A = nullptr;
   const Operator * P = nullptr;
   const mutable Operator * PtAP = nullptr;
   Solver * M = nullptr;
   Solver * Mf = nullptr;

   bool mutable solver_set = false;

   const Operator * GetPtAP(const Operator *Aop, const Operator *Pop) const;
   void MakeSolver() const;

   mutable Vector z;
   mutable Vector rf;
   mutable Vector xf;
   mutable Vector r;

}; // mfem::FilteredSolver class

} // namespace mfem

#endif
