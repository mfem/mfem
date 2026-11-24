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

#ifndef MFEM_FUNCTIONAL_HPP
#define MFEM_FUNCTIONAL_HPP

#include "mfem.hpp"
#include <functional>
#include <memory>

namespace mfem
{
class Functional : public Operator
{
private:
   Array<int> input_sizes;
public:
   Functional(Array<int> input_sizes)
      : Operator(1, input_sizes.Sum()), input_sizes(std::move(input_sizes)) {}
   /// Evaluate the functional at x.
   virtual real_t Eval(const MultiVector &x) const = 0;
   /// Get state-less derivative of the functional at x
   /// The derivative should support MultMV().
   /// To support Hessian, the returned operator should support GetGradientMV().
   /// GetGradient() for each block.
   /// ```
   /// auto df = f.Derivative(); // get derivative, state-less
   /// df->MultMV(X, dFdX); // compute derivative at X
   /// auto H = df->GetGradientMV(X); // get Hessian at X
   /// ```
   virtual std::shared_ptr<Operator> Derivative() const
   { MFEM_ABORT("Derivative not implemented for this functional."); }

   /// Evaluate the functional at X, returning the value in Y.
   void MultMV(const MultiVector &X, MultiVector &Y) const override final
   { Y.SetNumBlocks(1); Y[0].SetSize(1); Y[0].HostWrite()[0] = Eval(X); }

   /// Evaluate the functional at x, returning the value in y.
   void Mult(const Vector &x, Vector &y) const override final
   {
      MFEM_VERIFY(x.Size() == Width(), "x must be of size " << Width());
      const MultiVector X(const_cast<Vector&>(x), input_sizes);
      y.SetSize(1); y.HostWrite()[0] = Eval(X);
   }
   Operator &GetGradient(const Vector &) const override final
   {
      MFEM_ABORT("Functional does not support GetGradient. Use Derivative()");
   }
   Operator &GetGradientMV(const MultiVector &) const override final
   {
      MFEM_ABORT("Functional does not support GetGradientMV. Use Derivative()");
   }
   const Array<int> &InputSizes() const { return input_sizes; }
   void SetSizes(Array<int> new_sizes)
   {
      input_sizes = std::move(new_sizes);
      width = input_sizes.Sum();
   }
};

/// @brief Adapts a linear Operator acting on a single Vector to the
/// MultiVector interface: MultMV() calls Mult() on the only block, and
/// GetGradientMV() returns the operator itself.
///
/// Use it for the derivative of a quadratic functional, e.g. the gradient
/// x -> A x of 1/2 x^T A x, where A is a SparseMatrix or HypreParMatrix
/// (neither implements MultMV()).
class LinearMVOperator : public Operator
{
   std::shared_ptr<Operator> op;
public:
   explicit LinearMVOperator(std::shared_ptr<Operator> op_)
      : Operator(op_ ? op_->Height() : 0, op_ ? op_->Width() : 0),
        op(std::move(op_))
   { MFEM_VERIFY(op, "Operator must not be null."); }
   LinearMVOperator(Operator *op_, bool take_ownership = false)
      : Operator(op_ ? op_->Height() : 0, op_ ? op_->Width() : 0)
   {
      MFEM_VERIFY(op_, "Operator must not be null.");
      if (take_ownership) { op = std::shared_ptr<Operator>(op_); }
      else { op = std::shared_ptr<Operator>(op_, [](Operator*) {}); }
   }

   void Mult(const Vector &x, Vector &y) const override { op->Mult(x, y); }

   void MultMV(const MultiVector &X, MultiVector &Y) const override
   {
      MFEM_VERIFY(X.NumBlocks() == 1,
                  "LinearMVOperator expects a single input block.");
      Y.SetNumBlocks(1);
      Y[0].SetSize(height);
      op->Mult(X[0], Y[0]);
   }

   Operator &GetGradient(const Vector &) const override { return *op; }
   Operator &GetGradientMV(const MultiVector &) const override { return *op; }
};

/// @brief Functional defined by a callable and a derivative operator.
///
/// The derivative operator maps the input X to the gradient dF/dX, so it is
/// square with size equal to the total input size, and it must implement
/// MultMV() (wrap a plain linear operator in LinearMVOperator).
class FunctionalWrapper : public Functional
{
protected:
   using EvalFunc = std::function<real_t(const MultiVector&)>;
   EvalFunc eval_func;
   std::shared_ptr<Operator> df;
protected:
   /// For derived classes that set the derivative later with SetDerivative().
   FunctionalWrapper(Array<int> input_sizes, EvalFunc eval_func_)
      : Functional(std::move(input_sizes)), eval_func(std::move(eval_func_))
   {
      for (auto s : InputSizes())
      { MFEM_VERIFY(s >= 0, "Input sizes must be non-negative."); }
   }

   void SetDerivative(Operator* derivative_op, bool own)
   {
      MFEM_VERIFY(derivative_op, "Derivative operator must not be null.");
      MFEM_VERIFY(derivative_op->Height() == derivative_op->Width(),
                  "Derivative operator must be square.");
      MFEM_VERIFY(derivative_op->Width() == Width(),
                  "Derivative operator size must match the total size of the "
                  "input fields.");
      if (own) { df = std::shared_ptr<Operator>(derivative_op); }
      else { df = std::shared_ptr<Operator>(derivative_op, [](Operator*) {}); }
   }
public:
   /// If @a take_ownership is false, @a derivative_op must outlive this
   /// functional (and any copy of it).
   FunctionalWrapper(Array<int> input_sizes, EvalFunc eval_func_,
                     Operator *derivative_op, bool take_ownership = false)
      : FunctionalWrapper(std::move(input_sizes), std::move(eval_func_))
   {
      if (derivative_op) { SetDerivative(derivative_op, take_ownership); }
   }

   real_t Eval(const MultiVector &x) const override
   {
      MFEM_VERIFY(eval_func, "Eval function must not be null.");
      return eval_func(x);
   }

   std::shared_ptr<Operator> Derivative() const override
   {
      if (!df) { return Functional::Derivative(); }
      return df;
   }
};
}

#endif
