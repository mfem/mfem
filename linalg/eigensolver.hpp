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

/**
 * @file eigensolver.hpp
 *
 * @brief This file contains a common interface for all eigensolver classes
 */

#ifndef MFEM_EIGENSOLVER
#define MFEM_EIGENSOLVER

#ifdef MFEM_HYPRE
#include "hypre.hpp"
#endif

#ifdef MFEM_SLEPC
#include "slepc.hpp"
#endif

namespace mfem
{

enum class EigenSolverType
{
   HYPRE,
   SLEPC,
   INVALID_TYPE
};

/// Provides base class for MFEM Eigensolvers
class EigenSolverBase
{
public:
   EigenSolverBase() {}

   /// Destructor
   virtual ~EigenSolverBase() = default;

   /// Solves the eigenvalue problem
   virtual void Solve() = 0;

   /// Set the required number of modes
   virtual void SetNumModes(int num_Modes) 
   {
      numModes=num_Modes;
   }

   /// @brief  Set the operator to the eigenvalue problem
   /// @param A - operator
   virtual void SetOperator(Operator& A) = 0;

   /// @brief Sets operators for the generalized eigenvalue problem
   /// @param A - operator
   /// @param M - mass matrix
   virtual void SetOperator(Operator& A, Operator& M)
   {
      MFEM_ABORT("Generalized eigensolver is not supported!");
   }

   /// Optional method - sets preconditioner for the
   /// eigenvalue solver.
   virtual void SetPreconditioner(Solver& precond)
   {
      MFEM_ABORT("Preconditioner is not supported!");
   }

   /// Returns the converged eigenvalues
   virtual void GetEigenvalues(Array<real_t>& eigen_vals) = 0;

   /// Returns the vec_index eigenvector.
   virtual void GetEigenvector(int vec_index, Vector& vector) = 0;

   /// Returns the eigensolver type.
   EigenSolverType GetSolverType() { return eigSolverType; }

protected:
   int numModes = 0;
   EigenSolverType eigSolverType = EigenSolverType::INVALID_TYPE;
};

#ifdef MFEM_HYPRE
class EigenSolverHypreLOBPCG : public EigenSolverBase
{
public:
   EigenSolverHypreLOBPCG(MPI_Comm comm)
   {
      eigenSolver = std::make_unique<HypreLOBPCG>(comm);
      eigSolverType = EigenSolverType::HYPRE;
   }

   ~EigenSolverHypreLOBPCG() {}

   void Solve() override { eigenSolver->Solve(); }
   void SetNumModes(int num_Modes) override
   {
      eigenSolver->SetNumModes(num_Modes);
      numModes = num_Modes;
   }

   void SetOperator(Operator& A) override { eigenSolver->SetOperator(A); }

   void SetOperator(Operator& A, Operator& M) override
   {
      eigenSolver->SetOperator(A);
      eigenSolver->SetMassMatrix(M);
   }

   void SetPreconditioner(Solver& precond) override { eigenSolver->SetPreconditioner(precond); }
   void GetEigenvalues(Array<real_t>& eigen_vals) override { eigenSolver->GetEigenvalues(eigen_vals); }
   void GetEigenvector(int vec_index, Vector& vector) override
   {
      const HypreParVector& eigenvec = eigenSolver->GetEigenvector(vec_index);
      vector = eigenvec;
   }

   void SetTol(real_t tol) { eigenSolver->SetTol(tol); }
   void SetRelTol(real_t rel_tol) { eigenSolver->SetRelTol(rel_tol); }
   void SetMaxIter(int max_iter) { eigenSolver->SetMaxIter(max_iter); }
   void SetPrintLevel(int logging) { eigenSolver->SetPrintLevel(logging); }
   void SetRandomSeed(int seed) { eigenSolver->SetRandomSeed(seed); }
   void SetPrecondUsageMode(int usage_mode) { eigenSolver->SetPrecondUsageMode(usage_mode); }

private:
   std::unique_ptr<HypreLOBPCG> eigenSolver = nullptr;
};
#endif

#ifdef MFEM_SLEPC
class EigenSolverSlepc : public EigenSolverBase
{
public:
   EigenSolverSlepc(MPI_Comm comm)
   {
      eigSolverType = EigenSolverType::SLEPC;
      eigenSolver = std::make_unique<SlepcEigenSolver>(comm);

      eigenSolver->SetWhichEigenpairs(SlepcEigenSolver::TARGET_REAL);
      eigenSolver->SetTarget(0.0);
      eigenSolver->SetSpectralTransformation(SlepcEigenSolver::SHIFT_INVERT);
   }

   ~EigenSolverSlepc() {}

   void Solve() override { eigenSolver->Solve(); }
   void SetNumModes(int num_Modes) override
   {
      eigenSolver->SetNumModes(num_Modes);
      numModes = num_Modes;
   }
   /// @brief  Set the operator to the slepc eigenvalue problem. This method deep copies data to create a PetscParMatrix
   /// @param A - operator, must be of type HypreParMatrix.
   void SetOperator(Operator& A) override
   {
      petscMatA = std::make_unique<PetscParMatrix>
                  (dynamic_cast<HypreParMatrix*>(&A));
      eigenSolver->SetOperator(*petscMatA);
   }
   /// @brief  Set the operators to the slepc eigenvalue problem. This method deep copies data to create a PetscParMatrix
   /// @param A - operator, must be of type HypreParMatrix.
   /// @param M - operator, must be of type HypreParMatrix.
   void SetOperator(Operator& A, Operator& M) override
   {
      petscMatA = std::make_unique<PetscParMatrix>
                  (dynamic_cast<const HypreParMatrix*>(&A));
      petscMatM = std::make_unique<PetscParMatrix>
                  (dynamic_cast<const HypreParMatrix*>(&M));

      eigenSolver->SetOperators(*petscMatA, *petscMatM);
   }
   void SetPreconditioner([[maybe_unused]] Solver& precond) override {}
   void GetEigenvalues(Array<real_t>& eigen_vals) override
   {
      eigen_vals.SetSize(numModes);
      for (int ik = 0; ik < numModes; ik++)
      {
         eigenSolver->GetEigenvalue(static_cast<unsigned int>(ik), eigen_vals[ik]);
      }
   }
   void GetEigenvector( int vec_index, Vector& vector) override
   { eigenSolver->GetEigenvector(vec_index, vector); }

   void SetTol(real_t tol) { eigenSolver->SetTol(tol); }
   void SetMaxIter(int max_iter) { eigenSolver->SetMaxIter(max_iter); }

private:
   std::unique_ptr<SlepcEigenSolver> eigenSolver = nullptr;
   std::unique_ptr<PetscParMatrix> petscMatA = nullptr;
   std::unique_ptr<PetscParMatrix> petscMatM = nullptr;
};
#endif

}  // namespace mfem

#endif
