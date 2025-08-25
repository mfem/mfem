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
#include <memory>

namespace mfem
{

class FilteredSolver : public Solver
{
public:
   FilteredSolver() : Solver() { }

   virtual void SetOperator(const Operator &A) override;
   virtual void SetSolver(Solver &M);

   // Transfer operator from Filtered subspace to the original space
   void SetFilteredSubspaceTransferOperator(const Operator &P);
   void SetFilteredSubspaceSolver(Solver &Mf);

   void Mult(const Vector &x, Vector &y) const override;

   virtual ~FilteredSolver() = default;

   FilteredSolver(const FilteredSolver&) = delete;
   FilteredSolver& operator=(const FilteredSolver&) = delete;
   FilteredSolver(FilteredSolver&&) = default;
   FilteredSolver& operator=(FilteredSolver&&) = default;

protected:
   const Operator * A = nullptr;
   const Operator * P = nullptr;
   Solver * M = nullptr;
   Solver * Mf = nullptr;
   mutable std::unique_ptr<const Operator> PtAP = nullptr;
   void InitVectors() const;
   bool mutable solver_set = false;
private:

   const Operator * GetPtAP(const Operator *Aop, const Operator *Pop) const;
   void MakeSolver() const;

   mutable Vector z;
   mutable Vector rf;
   mutable Vector xf;
   mutable Vector r;

}; // mfem::FilteredSolver class


#ifdef MFEM_USE_MPI
class AMGFSolver : public FilteredSolver
{
private:
   std::unique_ptr<HypreBoomerAMG> amg;
public:
   AMGFSolver() : FilteredSolver()
   {
      amg = std::make_unique<HypreBoomerAMG>();
      FilteredSolver::SetSolver(*amg);
   }
   virtual void SetOperator(const Operator &A) override;

   HypreBoomerAMG& AMG() { solver_set = false; return *amg; }
   const HypreBoomerAMG& AMG() const { return *amg; }


   void SetSolver(Solver &M) override
   {
      MFEM_ABORT("SetSolver is not supported in AMGFSolver. It is set to AMG by default");
   }

   // Transfer operator from Filtered subspace to the original space
   void SetFilteredSubspaceTransferOperator(const HypreParMatrix &Pop);

   ~AMGFSolver() override = default;

};
#endif


} // namespace mfem

#endif
