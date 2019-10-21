// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_ELEMENTWISESMOOTHER
#define MFEM_ELEMENTWISESMOOTHER

#include "../config/config.hpp"
#include "fespace.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

namespace mfem
{

/**
   Applies a smoother element-by-element.

   The interface basically requires you to implement GetElementFromMatVec()
   and LocalSmoother()
*/
class ElementWiseSmoother : public mfem::Solver
{
public:
   ElementWiseSmoother(const mfem::FiniteElementSpace& fespace);
   virtual ~ElementWiseSmoother() {}

   virtual void SetOperator(const mfem::Operator &op) {}

   virtual void Mult(const mfem::Vector& in, mfem::Vector& out) const;

   /**
      b and r are local to the element, but x has to be global.
   */
   virtual void ElementResidual(int element, const mfem::Vector& b,
                                const mfem::Vector& x,
                                mfem::Vector& r) const;

   /**
      This produces the same result in y_element as if you did a
      global matvec with (global) x, and then extracted the
      dofs corresponding to a particular element.

      Since elements have a bounded number of neighbors, this can
      be done more efficiently than a global matvec.
   */
   virtual void GetElementFromMatVec(int element, const mfem::Vector& x,
                                     mfem::Vector& y_element) const = 0;

   /**
      Interface is that `in` should be a residual, and then `out` is a correction
      to the solution. (Ie, in is not a right-hand side, and out is not a current
      iterate; out gets totally overwritten, not updated.)

      I am not sure this is the *correct* interface for this use, but it is the
      *current* interface.
   */
   virtual void LocalSmoother(int element, const mfem::Vector& in,
                              mfem::Vector& out) const = 0;

protected:
   const mfem::FiniteElementSpace& fespace_;
};

class AdditiveSchwarzLORSmoother : public Solver
{
 public:
   AdditiveSchwarzLORSmoother(const mfem::FiniteElementSpace& fespace,
                              const Array<int>& ess_tdof_list,
                              mfem::BilinearForm& aform,
                              const mfem::Vector& diag,
                              mfem::SparseMatrix* LORmat, double scale);
   virtual ~AdditiveSchwarzLORSmoother() {}

   virtual void SetOperator(const mfem::Operator& op) {}

   virtual void Mult(const mfem::Vector& in, mfem::Vector& out) const;

 private:
   virtual void LocalSmoother(int element, const mfem::Vector& in,
                              mfem::Vector& out) const;

 protected:
   const mfem::FiniteElementSpace& fespace_;
   const Array<int>& ess_tdof_list_;
   mfem::BilinearForm& aform_;
   mfem::Vector diag_;
   mfem::SparseMatrix* LORmat_;
   double scale_;
   mfem::Table el_to_el_;
   Vector countingVector;
   mutable DenseMatrixInverse inv;
};


class AdditiveSchwarzApproxLORSmoother : public Solver
{
 public:
   AdditiveSchwarzApproxLORSmoother(const mfem::FiniteElementSpace& fespace,
                              const Array<int>& ess_tdof_list,
                              mfem::BilinearForm& aform,
                              const mfem::Vector& diag,
                              const mfem::Vector& LORdiag,
                              mfem::SparseMatrix* LORmat, double scale);
   virtual ~AdditiveSchwarzApproxLORSmoother() {}

   virtual void SetOperator(const mfem::Operator& op) {}

   virtual void Mult(const mfem::Vector& in, mfem::Vector& out) const;

 private:
   virtual void LocalSmoother(int element, const mfem::Vector& in,
                              mfem::Vector& out) const;

 protected:
   const mfem::FiniteElementSpace& fespace_;
   const Array<int>& ess_tdof_list_;
   mfem::BilinearForm& aform_;
   mfem::Vector diag_;
   mfem::Vector LORdiag_;
   mfem::SparseMatrix* LORmat_;
   double scale_;
   mfem::Table el_to_el_;
   Vector countingVector;
   DenseMatrix elmat;
   mutable DenseMatrixInverse inv;
};

/**
   Our first implementation is Jacobi within elements and Gauss-Seidel
   between them.
*/
class ElementWiseJacobi : public ElementWiseSmoother
{
public:
   ElementWiseJacobi(const mfem::FiniteElementSpace& fespace,
                     mfem::BilinearForm& aform,
                     const mfem::Vector& global_diag,
                     double scale=1.0);

   virtual void GetElementFromMatVec(int element, const mfem::Vector& x,
                                     mfem::Vector& y) const;

   virtual void LocalSmoother(int element, const mfem::Vector& in,
                              mfem::Vector& out) const;

protected:
   mfem::BilinearForm& aform_;
   const mfem::Vector& global_diag_;
   double scale_;

   mfem::Table el_to_el_;
};

}

#endif // MFEM_ELEMENTWISESMOOTHER
