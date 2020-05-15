// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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
#ifdef MFEM_USE_PETSC
#ifdef MFEM_USE_MPI

#include "petsc.hpp"
#include "slepc.h"

namespace mfem
{

void MFEMInitializeSlepc();
void MFEMInitializeSlepc(int*,char***);
void MFEMInitializeSlepc(int*,char***,const char[],const char[]);
void MFEMFinalizeSlepc();

class SlepcEigenSolver
{
private:
   /// SLEPc linear eigensolver object
   EPS eps;
   /// Solver tolerance
   double _tol = PETSC_DEFAULT;

   /// Maximum number of iterations
   int _max_its = PETSC_DEFAULT;

   /// Number of converged eigenvectors. Start with a negative value before the solver is run.
   int _num_conv = -1;

   /// Real and imaginary part of eigenvector
   mutable PetscParVector *VR = NULL, *VC = NULL;
public:
   /// Constructors
   SlepcEigenSolver(MPI_Comm comm, const std::string &prefix = std::string());

   virtual ~SlepcEigenSolver();

   /// Set solver tolerance
   void SetTol(double tol);

   /// Set maximum number of iterations
   void SetMaxIter(int max_iter);
   void SetNumModes(int num_eigs);
   /// Set operator for standard eigenvalue problem
   void SetOperator(const Operator &op);
   /// Set operator for generalized eigenvalue problem
   void SetOperators(const Operator &op, const Operator &opB);

   /// Customize object with options set
   void Customize();

   /// Solve the eigenvalue problem for the specified number of eigenvalues
   void Solve();

   /// Get the corresponding eigenvalue
   void GetEigenvalue(unsigned int i, double & lr) const;
   void GetEigenvalue(unsigned int i, double & lr, double & lc) const;

   /// Get the corresponding eigenvector
   void GetEigenvector(unsigned int i, Vector & vr) const;
   void GetEigenvector(unsigned int i, Vector & vr, Vector & vc) const;

   /// Target spectrum for the eigensolver. Target imaginary is not supported without complex support in SLEPc, and intervals are not implemented.
   enum Which
   {
      LARGEST_MAGNITUDE,
      SMALLEST_MAGNITUDE,
      LARGEST_REAL,
      SMALLEST_REAL,
      LARGEST_IMAGINARY,
      SMALLEST_IMAGINARY,
      TARGET_MAGNITUDE,
      TARGET_REAL
   };

   enum SpectralTransformation
   {
      SHIFT,
      SHIFT_INVERT
   };

   void SetWhichEigenpairs(Which which);
   void SetTarget(double target);
   void SetSpectralTransformation(SpectralTransformation transformation);

   /// Conversion function to SLEPc's EPS type.
   operator EPS() const { return eps; }
};

}
#endif // MFEM_USE_MPI
#endif // MFEM_USE_PETSC
#endif // MFEM_USE_SLEPC

#endif // MFEM_SLEPC
