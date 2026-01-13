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

namespace mfem {

enum EigenSolverType
{
  HYPRE = 0,
  SLEPC,
  ENDENUM
};

class EigenSolverBase {
 public:
  EigenSolverBase() {}

  virtual ~EigenSolverBase() {};

  virtual void Solve() = 0;
  virtual void SetNumModes(int num_eigs) = 0;
  virtual void SetOperator(mfem::Operator& A) = 0;
  virtual void SetOperator(mfem::Operator& A, mfem::Operator& M) = 0;
  virtual void SetPreconditioner(mfem::Solver& precond) = 0;

  virtual void GetEigenvalues(mfem::Array<mfem::real_t>& eigen_vals) = 0;
  virtual void GetEigenvector(unsigned int vec_index, mfem::Vector& vector) = 0;
  EigenSolverType GetSolverType() { return eigSolverType; };

 private:
 protected:
  int numModes = 0;
  EigenSolverType eigSolverType = ENDENUM;
};

//------------------------------------------------------------------------------------------

class EigenSolverHypreLOBPCG : public mfem::EigenSolverBase {
 public:
  EigenSolverHypreLOBPCG(MPI_Comm comm)
  {
    eigenSolver_ = std::make_unique<mfem::HypreLOBPCG>(comm);
    eigSolverType = HYPRE;
  }

  ~EigenSolverHypreLOBPCG() {}

  void Solve() override { eigenSolver_->Solve(); };
  void SetNumModes(int num_eigs) override
  {
    eigenSolver_->SetNumModes(num_eigs);
    numModes = num_eigs;
  };
  void SetOperator(mfem::Operator& A) override { eigenSolver_->SetOperator(A); };
  void SetOperator(mfem::Operator& A, mfem::Operator& M) override
  {
    eigenSolver_->SetOperator(A);
    eigenSolver_->SetMassMatrix(M);
  };
  void SetPreconditioner(mfem::Solver& precond) override { eigenSolver_->SetPreconditioner(precond); };
  void GetEigenvalues(mfem::Array<mfem::real_t>& eigen_vals) override { eigenSolver_->GetEigenvalues(eigen_vals); };
  void GetEigenvector(unsigned int vec_index, mfem::Vector& vector) override
  {
    const mfem::HypreParVector& eigenvec = eigenSolver_->GetEigenvector(vec_index);
    vector = eigenvec;
  };

  void SetTol(mfem::real_t tol) { eigenSolver_->SetTol(tol); };
  void SetRelTol(mfem::real_t rel_tol) { eigenSolver_->SetRelTol(rel_tol); };
  void SetMaxIter(int max_iter) { eigenSolver_->SetMaxIter(max_iter); };
  void SetPrintLevel(int logging) { eigenSolver_->SetPrintLevel(logging); };
  void SetRandomSeed(int seed) { eigenSolver_->SetRandomSeed(seed); };
  void SetPrecondUsageMode(int usage_mode) { eigenSolver_->SetPrecondUsageMode(usage_mode); };

  // void 	SetInitialVectors (int num_vecs, HypreParVector **vecs);
  // void 	SetSubSpaceProjector (Operator &proj);
  // HypreParVector ** 	StealEigenvectors ()

 private:
  std::unique_ptr<mfem::HypreLOBPCG> eigenSolver_ = nullptr;
};


//------------------------------------------------------------------------------------------

#ifdef MFEM_SLEPC
class EigenSolverSlepc : public mfem::EigenSolverBase {
 public:
  EigenSolverSlepc(MPI_Comm comm)
  {
    eigSolverType = SLEPC;
    eigenSolver_ = std::make_unique<mfem::SlepcEigenSolver>(comm);

    eigenSolver_->SetWhichEigenpairs(mfem::SlepcEigenSolver::TARGET_REAL);
    eigenSolver_->SetTarget(0.0);
    eigenSolver_->SetSpectralTransformation(mfem::SlepcEigenSolver::SHIFT_INVERT);
  }

  ~EigenSolverSlepc() {}

  void Solve() override { eigenSolver_->Solve(); };
  void SetNumModes(int num_eigs) override
  {
    eigenSolver_->SetNumModes(num_eigs);
    numModes = num_eigs;
  };
  void SetOperator(mfem::Operator& A) override
  {
    petscMatA = std::make_unique<mfem::PetscParMatrix>(dynamic_cast<mfem::HypreParMatrix*>(&A));
    eigenSolver_->SetOperator(*petscMatA);
  };
  void SetOperator(mfem::Operator& A, mfem::Operator& M) override
  {
    petscMatA = std::make_unique<mfem::PetscParMatrix>(dynamic_cast<const mfem::HypreParMatrix*>(&A));
    petscMatM = std::make_unique<mfem::PetscParMatrix>(dynamic_cast<const mfem::HypreParMatrix*>(&M));

    eigenSolver_->SetOperators(*petscMatA, *petscMatM);
  };
  void SetPreconditioner([[maybe_unused]] mfem::Solver& precond) {};
  void GetEigenvalues(mfem::Array<mfem::real_t>& eigen_vals) override
  {
    for (int ik = 0; ik < numModes; ik++) {
      eigenSolver_->GetEigenvalue(static_cast<unsigned int>(ik), eigen_vals[ik]);
    }
  };
  void GetEigenvector(unsigned int vec_index, mfem::Vector& vector) override
  { eigenSolver_->GetEigenvector(vec_index, vector); };

  void SetTol(mfem::real_t tol) { eigenSolver_->SetTol(tol); };
  void SetMaxIter(int max_iter) { eigenSolver_->SetMaxIter(max_iter); };

  // void 	Customize (bool customize=true) const
  // int 	GetNumConverged ()
  // void 	SetWhichEigenpairs (Which which)

 private:
  std::unique_ptr<mfem::SlepcEigenSolver> eigenSolver_ = nullptr;
  std::unique_ptr<mfem::PetscParMatrix> petscMatA = nullptr;
  std::unique_ptr<mfem::PetscParMatrix> petscMatM = nullptr;
};
#endif

}  // namespace mfem

#endif
