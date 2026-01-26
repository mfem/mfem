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
 * @brief This file contains a comman interface for all eigensolver classes
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

enum EigenSolverType
{
   HYPRE = 0,
   SLEPC,
   ENDENUM
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
   virtual void SetNumModes(int num_eigs) = 0;

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
   virtual void GetEigenvector(unsigned int vec_index, Vector& vector) = 0;

   /// Returns the eigensolver type.
   EigenSolverType GetSolverType() { return eigSolverType; };

protected:
   int numModes = 0;
   EigenSolverType eigSolverType = ENDENUM;
};

//------------------------------------------------------------------------------------------

class EigenSolverHypreLOBPCG : public EigenSolverBase
{
public:
   EigenSolverHypreLOBPCG(MPI_Comm comm)
   {
      eigenSolver_ = std::make_unique<HypreLOBPCG>(comm);
      eigSolverType = HYPRE;
   }

   ~EigenSolverHypreLOBPCG() {}

   void Solve() override { eigenSolver_->Solve(); };
   void SetNumModes(int num_eigs) override
   {
      eigenSolver_->SetNumModes(num_eigs);
      numModes = num_eigs;
   };

   void SetOperator(Operator& A) override { eigenSolver_->SetOperator(A); };

   void SetOperator(Operator& A, Operator& M) override
   {
      eigenSolver_->SetOperator(A);
      eigenSolver_->SetMassMatrix(M);
   };

   void SetPreconditioner(Solver& precond) override { eigenSolver_->SetPreconditioner(precond); };
   void GetEigenvalues(Array<real_t>& eigen_vals) override { eigenSolver_->GetEigenvalues(eigen_vals); };
   void GetEigenvector(unsigned int vec_index, Vector& vector) override
   {
      const HypreParVector& eigenvec = eigenSolver_->GetEigenvector(vec_index);
      vector = eigenvec;
   };

   void SetTol(real_t tol) { eigenSolver_->SetTol(tol); };
   void SetRelTol(real_t rel_tol) { eigenSolver_->SetRelTol(rel_tol); };
   void SetMaxIter(int max_iter) { eigenSolver_->SetMaxIter(max_iter); };
   void SetPrintLevel(int logging) { eigenSolver_->SetPrintLevel(logging); };
   void SetRandomSeed(int seed) { eigenSolver_->SetRandomSeed(seed); };
   void SetPrecondUsageMode(int usage_mode) { eigenSolver_->SetPrecondUsageMode(usage_mode); };

   // void   SetInitialVectors (int num_vecs, HypreParVector **vecs);
   // void   SetSubSpaceProjector (Operator &proj);
   // HypreParVector **  StealEigenvectors ()

private:
   std::unique_ptr<HypreLOBPCG> eigenSolver_ = nullptr;
};


//------------------------------------------------------------------------------------------

#ifdef MFEM_SLEPC
class EigenSolverSlepc : public EigenSolverBase
{
public:
   EigenSolverSlepc(MPI_Comm comm)
   {
      eigSolverType = SLEPC;
      eigenSolver_ = std::make_unique<SlepcEigenSolver>(comm);

      eigenSolver_->SetWhichEigenpairs(SlepcEigenSolver::TARGET_REAL);
      eigenSolver_->SetTarget(0.0);
      eigenSolver_->SetSpectralTransformation(SlepcEigenSolver::SHIFT_INVERT);
   }

   ~EigenSolverSlepc() {}

   void Solve() override { eigenSolver_->Solve(); };
   void SetNumModes(int num_eigs) override
   {
      eigenSolver_->SetNumModes(num_eigs);
      numModes = num_eigs;
   };
   void SetOperator(Operator& A) override
   {
      petscMatA = std::make_unique<PetscParMatrix>
                  (dynamic_cast<HypreParMatrix*>(&A));
      eigenSolver_->SetOperator(*petscMatA);
   };
   void SetOperator(Operator& A, Operator& M) override
   {
      petscMatA = std::make_unique<PetscParMatrix>
                  (dynamic_cast<const HypreParMatrix*>(&A));
      petscMatM = std::make_unique<PetscParMatrix>
                  (dynamic_cast<const HypreParMatrix*>(&M));

      eigenSolver_->SetOperators(*petscMatA, *petscMatM);
   };
   void SetPreconditioner([[maybe_unused]] Solver& precond) {};
   void GetEigenvalues(Array<real_t>& eigen_vals) override
   {
      for (int ik = 0; ik < numModes; ik++)
      {
         eigenSolver_->GetEigenvalue(static_cast<unsigned int>(ik), eigen_vals[ik]);
      }
   };
   void GetEigenvector(unsigned int vec_index, Vector& vector) override
   { eigenSolver_->GetEigenvector(vec_index, vector); };

   void SetTol(real_t tol) { eigenSolver_->SetTol(tol); };
   void SetMaxIter(int max_iter) { eigenSolver_->SetMaxIter(max_iter); };

   // void   Customize (bool customize=true) const
   // int    GetNumConverged ()
   // void   SetWhichEigenpairs (Which which)

private:
   std::unique_ptr<SlepcEigenSolver> eigenSolver_ = nullptr;
   std::unique_ptr<PetscParMatrix> petscMatA = nullptr;
   std::unique_ptr<PetscParMatrix> petscMatM = nullptr;
};
#endif

}  // namespace mfem

#endif
