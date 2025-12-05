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

#ifndef MFEM_SLEPC
#define MFEM_SLEPC

#include "../config/config.hpp"

#ifdef MFEM_USE_SLEPC
#ifdef MFEM_USE_MPI

#include "petsc.hpp"

// Forward declaration of SLEPc's internal struct _p_EPS:
struct _p_EPS;

namespace mfem
{

// Declare an alias of SLEPc's EPS type, mfem::slepc::EPS:
namespace slepc { typedef struct ::_p_EPS *EPS; }

void MFEMInitializeSlepc();
void MFEMInitializeSlepc(int*,char***);
void MFEMInitializeSlepc(int*,char***,const char[],const char[]);
void MFEMFinalizeSlepc();

class SlepcEigenSolver
{
private:
   /// Boolean to handle SetFromOptions calls
   mutable bool clcustom;

   /// SLEPc linear eigensolver object
   slepc::EPS eps;

   /// Real and imaginary part of eigenvector
   mutable PetscParVector *VR, *VC;

public:
   /// Constructors
   SlepcEigenSolver(MPI_Comm comm, const std::string &prefix = std::string());

   virtual ~SlepcEigenSolver();

   /** @brief Set solver convergence tolerance relative to the magnitude of the
       eigenvalue.

       @note Default value is 1e-8
   */
   void SetTol(real_t tol);

   /** @brief Set maximum number of iterations allowed in the call to
       SlepcEigenSolver::Solve */
   void SetMaxIter(int max_iter);

   /// Set the number of eigenmodes to compute
   void SetNumModes(int num_eigs);

   /// Set operator for standard eigenvalue problem
   void SetOperator(const PetscParMatrix &op);

   /// Set operators for generalized eigenvalue problem
   void SetOperators(const PetscParMatrix &op, const PetscParMatrix &opB);

   /// Customize object with options set
   void Customize(bool customize = true) const;

   /// Solve the eigenvalue problem for the specified number of eigenvalues
   void Solve();

   /** @brief Get the number of converged eigenvalues after the call to
       SlepcEigenSolver::Solve */
   int GetNumConverged();

   /** @brief Get the ith eigenvalue after the system has been solved
      @param[in] i The index for the eigenvalue you want ordered by
                 SlepcEigenSolver::SetWhichEigenpairs
      @param[out] lr The real component of the eigenvalue
      @note the index @a i must be between 0 and
            SlepcEigenSolver::GetNumConverged - 1
   */
   void GetEigenvalue(unsigned int i, real_t & lr) const;

   /** @brief Get the ith eigenvalue after the system has been solved
      @param[in] i The index for the eigenvalue you want ordered by
                 SlepcEigenSolver::SetWhichEigenpairs
      @param[out] lr The real component of the eigenvalue
      @param[out] lc The imaginary component of the eigenvalue
      @note the index @a i must be between 0 and
            SlepcEigenSolver::GetNumConverged - 1
   */
   void GetEigenvalue(unsigned int i, real_t & lr, real_t & lc) const;

   /** @brief Get the ith eigenvector after the system has been solved
      @param[in] i The index for the eigenvector you want ordered by
                 SlepcEigenSolver::SetWhichEigenpairs
      @param[out] vr The real components of the eigenvector
      @note the index @a i must be between 0 and
            SlepcEigenSolver::GetNumConverged - 1
   */
   void GetEigenvector(unsigned int i, Vector & vr) const;

   /** @brief Get the ith eigenvector after the system has been solved
      @param[in] i The index for the eigenvector you want ordered by
                 SlepcEigenSolver::SetWhichEigenpairs
      @param[out] vr The real components of the eigenvector
      @param[out] vc The imaginary components of the eigenvector
      @note the index @a i must be between 0 and
            SlepcEigenSolver::GetNumConverged - 1
   */
   void GetEigenvector(unsigned int i, Vector & vr, Vector & vc) const;

   /** @brief Target spectrum for the eigensolver.

       This will define the order in which the eigenvalues/eigenvectors are
       indexed after the call to SlepcEigenSolver::Solve.
       @note Target imaginary is not supported without complex support in SLEPc,
       and intervals are not implemented.
   */
   enum Which
   {
      /// The eigenvalues with the largest complex magnitude (default)
      LARGEST_MAGNITUDE,
      /// The eigenvalues with the smallest complex magnitude
      SMALLEST_MAGNITUDE,
      /// The eigenvalues with the largest real component
      LARGEST_REAL,
      /// The eigenvalues with the smallest real component
      SMALLEST_REAL,
      /// The eigenvalues with the largest imaginary component
      LARGEST_IMAGINARY,
      /// The eigenvalues with the smallest imaginary component
      SMALLEST_IMAGINARY,
      /// The eigenvalues with complex magnitude closest to the target value
      TARGET_MAGNITUDE,
      /// The eigenvalues with the real component closest to the target value
      TARGET_REAL
   };

   /** @brief Spectral transformations that can be used by the solver in order
       to accelerate the convergence to the target eignevalues
   */
   enum SpectralTransformation
   {
      /// Utilize the shift of origin strategy
      SHIFT,
      /// Utilize the shift and invert strategy
      SHIFT_INVERT
   };

   /** @brief Set the which eigenvalues the solver will target and the order
       they will be indexed in.

       For SlepcEigenSolver::TARGET_MAGNITUDE or SlepcEigenSolver::TARGET_REAL
       you will also need to set the target value with
       SlepcEigenSolver::SetTarget.
   */
   void SetWhichEigenpairs(Which which);

   /** @brief Set the target value for the eigenpairs you want when using
       SlepcEigenSolver::TARGET_MAGNITUDE or SlepcEigenSolver::TARGET_REAL in
       the SlepcEigenSolver::SetWhichEigenpairs method.
   */
   void SetTarget(real_t target);

   /** @brief Set the spectral transformation strategy for acceletating
       convergenvce. Both SlepcEigenSolver::SHIFT and
       SlepcEigenSolver::SHIFT_INVERT are available.
   */
   void SetSpectralTransformation(SpectralTransformation transformation);

   /// Conversion function to SLEPc's EPS type.
   operator slepc::EPS() const { return eps; }

   /// Conversion function to PetscObject
   operator PetscObject() const {return (PetscObject)eps; }
};

}

#endif // MFEM_USE_MPI
#endif // MFEM_USE_SLEPC

#endif // MFEM_SLEPC
