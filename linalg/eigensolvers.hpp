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
/// Where A is a real-valued operator, the lambda_i are the eigenvalues,
/// and x_i are the eigenvectors.
class Eigenequation
{
protected:
   Eigenequation() = default;

public:
   virtual ~Eigenequation() = default;

   /// @brief Set the operator A of the eigenvalue equation
   virtual void SetOperator(const Operator & A) = 0;
};

/// Abstract Complex-valued Eigenequation
/// Defines the operator of the linear eigenvalue equation
///    A x_i = lambda_i x_i
/// Where A is a complex-valued operator, the lambda_i are the eigenvalues,
/// and x_i are the eigenvectors.
class ComplexEigenequation
{
protected:
   ComplexEigenequation() = default;

public:
   virtual ~ComplexEigenequation() = default;

   /// @brief Set the real and imaginary parts of the operator A
   virtual void SetOperator(const Operator & Ar, const Operator & Ai) = 0;
};

/// Abstract Generalized Eigenequation
/// Defines the operator of the linear eigenvalue equation
///    A x_i = lambda_i B x_i
/// Where A and B are real-valued operators, the lambda_i are the eigenvalues,
/// and x_i are the eigenvectors.
class GenEigenequation
{
protected:
   GenEigenequation() = default;

public:
   virtual ~GenEigenequation() = default;

   /// @brief Set the operators A and B of the generalized eigenvalue equation
   virtual void SetOperators(const Operator & A, const Operator & B) = 0;
};

/// Abstract Complex-valued Generalized Eigenequation
/// Defines the operator of the linear eigenvalue equation
///    A x_i = lambda_i B x_i
/// Where A and B are complex-valued operators, the lambda_i are the
/// eigenvalues, and x_i are the eigenvectors.
class ComplexGenEigenequation
{
protected:
   ComplexGenEigenequation() = default;

public:
   virtual ~ComplexGenEigenequation() = default;

   /// @brief Set the real and imaginary parts of the operators A and B
   virtual void SetOperators(const Operator & Ar, const Operator & Ai,
                             const Operator & Br, const Operator & Bi) = 0;
};

/// Abstract Eigensolver
/// Computes eigenvalue/eigenvector pairs for the linear system
///    A x_i = lambda_i x_i
/// Where the lambda_i are the eigenvalues and x_i are the eigenvectors.
class EigensolverBase
{
protected:
   EigensolverBase() = default;

public:
   virtual ~EigensolverBase() = default;

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
/// If A^T = A the linear system must have real-valued eigenvalues
/// and eigenvectors.
class SymEigensolver : public EigensolverBase, public Eigenequation
{
protected:
   SymEigensolver() = default;

public:
   virtual ~SymEigensolver() = default;

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

/// Symmetric Generalized Eigensolver
/// If A^T = A and M^T = M the linear system must have real-valued eigenvalues
/// and eigenvectors.
class SymGenEigensolver : public EigensolverBase, public GenEigenequation
{
protected:
   SymGenEigensolver() = default;

public:
   virtual ~SymGenEigensolver() = default;

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
/// If A^H = A the linear system must have real-valued eigenvalues
/// but may have complex-valued eigenvectors.
class HermEigensolver : public EigensolverBase, public ComplexEigenequation
{
protected:
   HermEigensolver() = default;

public:
   virtual ~HermEigensolver() = default;

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

/// Hermetian Generalized Eigensolver
/// If A^H = A and M^H = M the linear system must have real-valued eigenvalues
/// but may have complex-valued eigenvectors.
class HermGenEigensolver :
   public EigensolverBase, public ComplexGenEigenequation
{
protected:
   HermGenEigensolver() = default;

public:
   virtual ~HermGenEigensolver() = default;

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
class NonSymEigensolver : public EigensolverBase, public Eigenequation
{
protected:
   NonSymEigensolver() = default;

public:
   virtual ~NonSymEigensolver() = default;

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

/// Non-Symmetric Eigensolver
/// For general real-valued operators A and M the linear system must have
/// eigenvalues and eigenvectors which form complex conjugate pairs.
class NonSymGenEigensolver : public EigensolverBase, public GenEigenequation
{
protected:
   NonSymGenEigensolver() = default;

public:
   virtual ~NonSymGenEigensolver() = default;

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
class ComplexEigensolver : public EigensolverBase, public ComplexEigenequation
{
protected:
   ComplexEigensolver() = default;

public:
   virtual ~ComplexEigensolver() = default;

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

/// Complex Generalized Eigensolver
/// Can have arbitrary complex-valued eigenvalues and eigenvectors
class ComplexGenEigensolver :
   public EigensolverBase, public ComplexGenEigenequation
{
protected:
   ComplexGenEigensolver() = default;

public:
   virtual ~ComplexGenEigensolver() = default;

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
