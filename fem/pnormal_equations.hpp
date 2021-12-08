// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_PNORMALEQUATIONS
#define MFEM_PNORMALEQUATIONS

#include "../config/config.hpp"
#include "../linalg/linalg.hpp"

#ifdef MFEM_USE_MPI

#include <mpi.h>
#include "pfespace.hpp"
#include "normal_equations.hpp"

namespace mfem
{

/** @brief Class representing the whole weak formulation. (Convenient for DPG or
    Normal Equations) */
class ParNormalEquations : public NormalEquations
{

protected:
   // Domain FE spaces
   Array<ParFiniteElementSpace * > domain_pfes;

   // Trace FE Spaces
   Array<ParFiniteElementSpace * > trace_pfes;

   // All FE Spaces
   Array<ParFiniteElementSpace * > pfes;

   // Block operator of HypreParMatrix
   BlockOperator * P = nullptr; // Block Prolongation
   BlockOperator * R = nullptr; // Block Restriction


   // Block operator of HypreParMatrix
   BlockOperator * p_mat = nullptr;
   BlockOperator * p_mat_e = nullptr;


   // void Init();

   // Allocate appropriate SparseMatrix and assign it to mat
   void pAllocMat();

   void BuildProlongation();

private:

public:

   /// Creates bilinear form associated with FE spaces @a *fespaces.
   ParNormalEquations(Array<FiniteElementSpace* > & fes_,
                      Array<FiniteElementSpace* > & trace_fes_,
                      Array<FiniteElementCollection *> & fecol_, bool store_mat_ = false)
      : NormalEquations(fes_,trace_fes_,fecol_,store_mat_), domain_pfes(fes_),
        trace_pfes(trace_fes_)
   {
      pfes.Append(domain_pfes);
      pfes.Append(trace_pfes);
   }

   /// Assembles the form i.e. sums over all domain integrators.
   void Assemble(int skip_zeros = 1);

   /// Returns the matrix assembled on the true dofs, i.e. P^t A P.
   /** The returned matrix has to be deleted by the caller. */
   BlockOperator *ParallelAssemble() { return ParallelAssemble(mat); }

   /// Returns the eliminated matrix assembled on the true dofs, i.e. P^t A_e P.
   /** The returned matrix has to be deleted by the caller. */
   BlockOperator *ParallelAssembleElim() { return ParallelAssemble(mat_e); }

   /// Return the matrix @a m assembled on the true dofs, i.e. P^t A P.
   /** The returned matrix has to be deleted by the caller. */
   BlockOperator *ParallelAssemble(BlockMatrix *m);

   void FormLinearSystem(const Array<int> &ess_tdof_list, Vector &x,
                         OperatorHandle &A, Vector &X,
                         Vector &B, int copy_interior = 0);

   void FormSystemMatrix(const Array<int> &ess_tdof_list,
                         OperatorHandle &A);

   /** Call this method after solving a linear system constructed using the
       FormLinearSystem method to recover the solution as a ParGridFunction-size
       vector in x. Use the same arguments as in the FormLinearSystem call. */
   virtual void RecoverFEMSolution(const Vector &X, Vector &x);

   void EliminateVDofs(const Array<int> &vdofs,
                       Operator::DiagonalPolicy dpolicy = Operator::DIAG_ONE);

   void EliminateVDofsInRHS(const Array<int> &vdofs, const Vector &x, Vector &b);

   /// Destroys bilinear form.
   ~ParNormalEquations();

};

} // namespace mfem


#endif // MFEM_USE_MPI


#endif
