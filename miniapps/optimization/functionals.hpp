// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_OPT_FUNCTIONALS_HPP
#define MFEM_OPT_FUNCTIONALS_HPP

#include "mfem.hpp"

namespace mfem
{

/** @brief f(x) = 1/2 x^T A x + b^T x + c with A symmetric; A and b are not
    copied and must outlive this. In parallel, x is a true-dof vector, A x its
    local part of the global product, and the value is reduced over @a comm.
    We do not check whether A is a linear operator or symmetric.
    Instead, we assume that the user has provided a suitable opeartor (e.g., SparseMatrix, HypreParMatrix, etc.)

    GetGradient(x): (A x + b)^T (1 x n with fixed x); its
    GetGradient() returns A. */
class QuadraticFunctional : public Operator
{
public:
   QuadraticFunctional(Operator &A_, const Vector &b_, real_t c_ = 0.0)
      : QuadraticFunctional(A_, &b_, c_) {}
   QuadraticFunctional(Operator &A_, real_t c_ = 0.0)
      : QuadraticFunctional(A_, nullptr, c_) {}
#ifdef MFEM_USE_MPI
   QuadraticFunctional(MPI_Comm comm_, Operator &A_, const Vector &b_,
                       real_t c_ = 0.0)
      : QuadraticFunctional(A_, &b_, c_) { comm = comm_; }
   QuadraticFunctional(MPI_Comm comm_, Operator &A_, real_t c_ = 0.0)
      : QuadraticFunctional(A_, nullptr, c_) { comm = comm_; }
#endif

   void Mult(const Vector &x, Vector &y) const override;
   Operator &GetGradient(const Vector &x) const override;

private:
   /// The derivative at the point of the last GetGradient(): g = A x + b.
   class Derivative : public Operator
   {
   public:
      explicit Derivative(const QuadraticFunctional &f_)
         : Operator(1, f_.Width()), f(f_) {}
      void SetPoint(const Vector &x);
      /// y = <g, v> = (Ax + b)^T v
      void Mult(const Vector &v, Vector &y) const override;
      /// dx = s g = s*(Ax + b), s scalar
      void MultTranspose(const Vector &s, Vector &dx) const override;
      /// return A.
      Operator &GetGradient(const Vector &) const override { return f.A; }
   private:
      const QuadraticFunctional &f;
      Vector g;
   };

   QuadraticFunctional(Operator &A_, const Vector *b_, real_t c_);

   real_t reduce(const real_t x) const
   {
#ifdef MFEM_USE_MPI
      if (comm != MPI_COMM_NULL)
      {
         real_t y = x;
         MPI_Allreduce(MPI_IN_PLACE, &y, 1, MFEM_MPI_REAL_T, MPI_SUM, comm);
         return y;
      }
#endif
      return x;
   }

#ifdef MFEM_USE_MPI
   MPI_Comm comm = MPI_COMM_NULL;
#endif
   Operator &A;
   const Vector *b;
   const real_t c;
   mutable Vector tmp;
   mutable Derivative der;
};

/** @brief f(x) = b^T x + c; b is not copied and must outlive this. In
    parallel, the value is reduced over @a comm.

    GetGradient() returns the derivative b^T (1 x n); its GetGradient()
    returns the zero second derivative. */
class LinearFunctional : public Operator
{
public:
   LinearFunctional(const Vector &b_, real_t c_ = 0.0)
      : Operator(1, b_.Size()), b(b_), c(c_), der(*this) {}
#ifdef MFEM_USE_MPI
   LinearFunctional(MPI_Comm comm_, const Vector &b_, real_t c_ = 0.0)
      : LinearFunctional(b_, c_) { comm = comm_; }
#endif

   void Mult(const Vector &x, Vector &y) const override;
   Operator &GetGradient(const Vector &) const override { return der; }

private:
   /// The zero second derivative (n x n).
   class Zero : public Operator
   {
   public:
      explicit Zero(int n) : Operator(n) {}
      void Mult(const Vector &, Vector &y) const override
      { y.SetSize(height); y = 0.0; }
      void MultTranspose(const Vector &, Vector &y) const override
      { y.SetSize(width); y = 0.0; }
   };

   /// The derivative b^T.
   class Derivative : public Operator
   {
   public:
      explicit Derivative(const LinearFunctional &f_)
         : Operator(1, f_.Width()), f(f_), zero(f_.Width()) {}
      /// y = <b, v>
      void Mult(const Vector &v, Vector &y) const override;
      /// dx = s b
      void MultTranspose(const Vector &s, Vector &dx) const override;
      Operator &GetGradient(const Vector &) const override { return zero; }
   private:
      const LinearFunctional &f;
      mutable Zero zero;
   };
   real_t reduce(const real_t x) const
   {
#ifdef MFEM_USE_MPI
      if (comm != MPI_COMM_NULL)
      {
         real_t y = x;
         MPI_Allreduce(MPI_IN_PLACE, &y, 1, MFEM_MPI_REAL_T, MPI_SUM, comm);
         return y;
      }
#endif
      return x;
   }

#ifdef MFEM_USE_MPI
   MPI_Comm comm = MPI_COMM_NULL;
#endif
   const Vector &b;
   const real_t c;
   mutable Derivative der;
};

} // namespace mfem

#endif // MFEM_OPT_FUNCTIONALS_HPP
