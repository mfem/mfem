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

#ifndef MFEM_EIGENSOLVERS
#define MFEM_EIGENSOLVERS

#include "vector.hpp"

namespace mfem
{

/// Abstract Eigenequation
/// Defines the operator of the linear eigenvalue equation
///    A x_i = lambda_i x_i
/// Where the lambda_i are the eigenvalues and x_i are the eigenvectors.
class Eigenequation
{
protected:
   Eigenequation() = default;
   virtual ~Eigenequation() = default;

public:
   /// @brief Set the operator A of the eigenvalue equation
   virtual void SetOperator(const Operator & A) = 0;
};

/// Abstract Complex-valued Eigenequation
/// Defines the operator of the linear eigenvalue equation
///    A x_i = lambda_i x_i
/// Where A is complex-valued, the lambda_i are the eigenvalues and
/// x_i are the eigenvectors.
class ComplexEigenequation : public Eigenequation
{
protected:
   ComplexEigenequation() = default;
   virtual ~ComplexEigenequation() = default;

public:
   using Eigenequation::SetOperator;

   /// @brief Set the real and imaginary parts of the operator A
   virtual void SetOperator(const Operator & Ar, const Operator & Ai) = 0;
};

/// Abstract Generalized Eigenequation
/// Defines the operator of the linear eigenvalue equation
///    A x_i = lambda_i M x_i
/// Where the lambda_i are the eigenvalues and x_i are the eigenvectors.
class GenEigenequation : public Eigenequation
{
protected:
   GenEigenequation() = default;
   virtual ~GenEigenequation() = default;

public:
   /// @brief Set the mass operator M of the generalized eigenvalue equation
   virtual void SetMassMatrix(const Operator & M) = 0;
};

/// Abstract Complex-valued Generalized Eigenequation
/// Defines the operator of the linear eigenvalue equation
///    A x_i = lambda_i M x_i
/// Where the lambda_i are the eigenvalues and x_i are the eigenvectors.
class ComplexGenEigenequation :
   public ComplexEigenequation, public GenEigenequation
{
protected:
   ComplexGenEigenequation() = default;
   virtual ~ComplexGenEigenequation() = default;

public:
   using GenEigenequation::SetMassMatrix;

   /// @brief Set the real and imaginary parts of the mass operator M
   virtual void SetMassMatrix(const Operator & Mr, const Operator & Mi) = 0;
};

/// Abstract Eigensolver
/// Computes eigenvalue/eigenvector pairs for the linear system
///    A x_i = lambda_i x_i
/// Where the lambda_i are the eigenvalues and x_i are the eigenvectors.
class Eigensolver
{
protected:
   Eigensolver() = default;
   virtual ~Eigensolver() = default;

public:
   /// @brief Stopping criteria based on numerical tolerance
   ///
   /// @note This may be defined differently by different solvers.
   virtual void SetTol(real_t tol) = 0;

   /// @brief Stopping criteria based on number of iterations required to
   /// reach convergence.
   ///
   /// @note This may also be defined differently in different solvers.
   virtual void SetMaxIter(int max_iter) = 0;

   /// @brief Controls the type and amount of information printed to
   ///        standard output.
   virtual void SetPrintLevel(int logging) = 0;

   /// @brief Set the number of desired eigenmodes to compute
   virtual void SetNumModes(int num_eigs) = 0;

   /// @brief Get the number of converged eigenmodes
   virtual int GetNumConverged() const = 0;

   /// @brief Perform the eigenvalue solve
   virtual void Solve() = 0;
};

/// Symmetric Eigensolver
/// If A^T = A and M^T = M the linear system must have real-valued eigenvalues
/// and eigenvectors.
class SymEigensolver : public Eigensolver
{
protected:
   SymEigensolver() = default;
   virtual ~SymEigensolver() = default;

public:
   /// @brief Collect the converged eigenvalues
   ///
   /// The length of the array should equal the number of converged eigenvalues.
   virtual void GetEigenvalues(Array<real_t> & eigenvalues) const = 0;

   /// @brief Extract a single eigenvector
   ///
   /// The index i should be in the range [0, numConverged). The
   virtual const Vector & GetEigenvector(unsigned int i) const = 0;

   /// @brief Transfer ownership of the converged eigenvectors
   ///
   /// The array should contain numConverged vectors.
   virtual Vector ** StealEigenvectors() = 0;
};

/// Hermetian Eigensolver
/// If A^H = A and M^H = M the linear system must have real-valued eigenvalues
/// but may have complex-valued eigenvectors.
class HermEigensolver : public Eigensolver
{
protected:
   HermEigensolver() = default;
   virtual ~HermEigensolver() = default;

public:
   /// @brief Collect the converged eigenvalues
   ///
   /// The length of the array should equal the number of converged eigenvalues.
   virtual void GetEigenvalues(Array<real_t> & eigenvalues) const = 0;

   /// @brief Extract a single eigenvector
   ///
   /// The index i should be in the range [0, 2*numConverged). The
   /// vectors corresponding to even indices are the real parts of the
   /// converged eigenvectors and the odd indices correspond to the
   /// imaginary parts.
   virtual const Vector & GetEigenvector(unsigned int i) const = 0;

   /// @brief Transfer ownership of the converged eigenvectors
   ///
   /// The array should contain 2*numConverged vectors with the even
   /// indices corresponding to the real parts of the converged
   /// eigenvectors and the odd indices corresponding to the imaginary
   /// parts.
   virtual Vector ** StealEigenvectors() = 0;
};

/// Non-Symmetric Eigensolver
/// For general real-valued operators A the linear system must have
/// eigenvalues and eigenvectors which form complex conjugate pairs.
class NonSymEigensolver : public Eigensolver
{
protected:
   NonSymEigensolver() = default;
   virtual ~NonSymEigensolver() = default;

public:
   /// @brief Collect the converged eigenvalues
   ///
   /// The length of the array should be the number of converged
   /// eigenvalues. The complex-valued eigenvalues can be constructed
   /// as: lambda_{2*j} = eig[2*j]+i*eig[2*j+1] and
   /// lambda_{2*j+1} = eig[2*j]-i*eig[2*j+1]
   /// With j in the range [0, numConverged/2)
   virtual void GetEigenvalues(Array<real_t> & eig) const = 0;

   /// @brief Extract a single eigenvector
   ///
   /// The index i should be in the range [0, numConverged). The
   /// vectors corresponding to even indices are the real parts of the
   /// converged eigenvectors and the odd indices correspond to the
   /// imaginary parts. If needed, the complex conjugate pairs of
   /// eigenvectors can be constructed in the same manner described
   /// for the eigenvalues.
   virtual const Vector & GetEigenvector(unsigned int i) const = 0;

   /// @brief Transfer ownership of the converged eigenvectors
   ///
   /// The array should contain numConverged vectors with the even
   /// indices corresponding to the real parts of the converged
   /// eigenvectors and the odd indices corresponding to the imaginary
   /// parts.
   virtual Vector ** StealEigenvectors() = 0;
};

/// Complex Eigensolver
/// Can have arbitrary complex-valued eigenvalues and eigenvectors
class ComplexEigensolver : public NonSymEigensolver
{
protected:
   ComplexEigensolver() = default;
   virtual ~ComplexEigensolver() = default;

public:
   /// @brief Collect the converged eigenvalues
   ///
   /// The length of the array should be twice the number of converged
   /// eigenvalues. The complex-valued eigenvalues can be constructed
   /// as: lambda_j = eig[2*j]+i*eig[2*j+1]
   virtual void GetEigenvalues(Array<real_t> & eig) const = 0;

   /// @brief Extract a single eigenvector
   ///
   /// The index i should be in the range [0, 2*numConverged). The
   /// vectors corresponding to even indices are the real parts of the
   /// converged eigenvectors and the odd indices correspond to the
   /// imaginary parts.
   virtual const Vector & GetEigenvector(unsigned int i) const = 0;

   /// @brief Transfer ownership of the converged eigenvectors
   ///
   /// The array should contain 2*numConverged vectors with the even
   /// indices corresponding to the real parts of the converged
   /// eigenvectors and the odd indices corresponding to the imaginary
   /// parts.
   virtual Vector ** StealEigenvectors() = 0;
};

}

#endif
