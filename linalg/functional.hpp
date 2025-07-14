#ifndef MFEM_FUNCTIONAL_HPP
#define MFEM_FUNCTIONAL_HPP
#include "operator.hpp"
#include "blockvector.hpp"
#include "solvers.hpp"

namespace mfem
{
/// @brief A base class for functionals F:R^n->R
///
/// This class provides an interface for evaluating
/// F:R^n->R, \nabla F:R^n->R^n, and \nabla^2 F:R^n x R^n ->R^n.
/// F.Mult(x, y) evaluates the functional at a point x, and stores the result in y[0]
/// F.GetGradient() returns an operator that evaluates the gradient
/// F.GetGradient().GetGradient(x) returns an Hessian action operator.
///
/// The usual Operator::GetGradient(const Vector &x) method for this method
/// is deprecated as \nabla F only takes a single argument x.
/// It is redundant to use F.GetGradient(x).Mult(x, y) to evaluate the gradient.
/// Instead, use F.GetGradient().Mult(x, y) to evaluate the gradient at x.
///
/// The gradient and Hessian can be defined in two ways:
/// 1. If the gradient is available as a sperate operator,
///    then override Functional::GetGradient().
///    In this case, Functional::HessianMult() will not be called.
///
/// 2. Otherwise, override Functional::EvalGradient() and Functional::HessianMult()
///    to evaluate the gradient and Hessian action, respectively.
///    The helper classes, GradientOperator and HessianActionOperator,
///    will call these methods to evaluate the gradient and Hessian action.
///    If Hessian is a seperate operator, then you can override
///    Functional::GetHessian() to return the Hessian operator.
class Functional : public Operator
{
public:
   /// @brief Create a Functional with optional gradient and hessian
   /// @param n number of variables
   /// @param differentiability_order differentiability order
   /// 0: not differentiable, 1: differentiable, 2: twice differentiable
   /// @note EvalGradient() and HessianMult() should be implemented
   /// if differentiability_order is 1 or 2.
   Functional(int n=0)
      : Operator(1, n)
      , grad_operator(*this)
      , hessian_action_operator(*this)
      , parallel(false)
   { }

#ifdef MFEM_USE_MPI
   Functional(MPI_Comm comm, int n=0)
      : Operator(1, n)
      , grad_operator(*this)
      , hessian_action_operator(*this)
      , parallel(true), comm(comm)
   { }

   void SetComm(MPI_Comm comm) { parallel = true; this->comm = comm; }
   MPI_Comm GetComm() const { return comm; }
#endif
   bool IsParallel() const { return parallel; }

   /// @brief return the GradientOperator that evaluates the gradient
   ///        input x is not used. Use GetGradient().Mult(x,y) to evaluate the gradient
   ///        we recommend using GetGradient() instead of GetGradient(x)

   /// Deprecated. See Functional::GetGradient()
   MFEM_DEPRECATED
   Operator &GetGradient(const Vector &dummy) const override final { return GetGradient(); }

   /// @brief Return the GradientOperator that wraps Functional::EvalGradient() for Mult().
   /// @note If the functional has a corresponding standalone gradient operator,
   /// override this method to return the gradient operator.
   virtual Operator &GetGradient() const { return grad_operator; }

   /// @brief Evaluate the functional at a point x that will be called by GradientOperator::Mult()
   /// @note This method is not meant to be called directly. See, GradientOperator
   virtual void EvalGradient(const Vector &x, Vector &y) const
   {
      MFEM_ABORT("Functional::EvalGradient() not implemented");
   }
   /// @brief Evaluate the Hessian action at a point x and direction d
   /// that will be called by Functional::GetGradient().GetHessian(x).Mult(d,y)
   /// @note This method is not meant to be called directly. See, HessianActionOperator
   virtual void HessianMult(const Vector &x, const Vector &d, Vector &y) const
   {
      MFEM_ABORT("Functional::HessianMult() not implemented.");
   }
   /// @brief Return the HessianActionOperator at evaluation point x
   ///        that wraps Functional::HessianMult() for Mult().
   ///        See, HessianActionOperator and Functional::HessianMult().
   ///
   /// @note If the Hessian is available as a seperate operator, override this method.
   ///
   /// @warning If GetGradient() is overridden, this method will not be used.
   virtual Operator &GetHessian(const Vector &x) const
   {
      hessian_action_operator.SetX(x);
      return hessian_action_operator;
   }
private:
   bool parallel;
#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif
   /// @brief A helper class to return an operator that evaluates the gradient
   ///        using Functional::EvalGradient() method.
   class GradientOperator : public Operator
   {
   private: const Functional &f;
   public:
      GradientOperator(const Functional &f) : Operator(f.Width()), f(f) {}
      /// @brief Evaluate the gradient of Functional at a point x
      void Mult(const Vector &x, Vector &y) const override final { f.EvalGradient(x, y); }
      /// @brief Evaluate the Hessian of Functional at a point x
      Operator &GetGradient(const Vector &x) const override final { return f.GetHessian(x); }
   };
   friend class GradientOperator;

   /// @brief A helper class to return an operator that applies the Hessian action
   ///        using Functional::HessianMult() method.
   class HessianActionOperator : public Operator
   {
   private:
      const Functional &f;
      const Vector *x;
   public:
      HessianActionOperator(const Functional &f) : Operator(f.Width()), f(f) {}
      void SetX(const Vector &x) { this->x = &x; }
      void Mult(const Vector &d, Vector &y) const override { f.HessianMult(*x, d, y); }
   };
   friend class HessianActionOperator;

   mutable GradientOperator grad_operator;
   mutable HessianActionOperator hessian_action_operator;
};

/// @brief Quadratic functional of the form
///        f(u) = 0.5 * <A u, u> + beta<b, u> + c
///        where A is a square (possibly nonlinear) operator,
///        beta is a scalar (defaults to 1.0, not used when b is nullptr),
///        b is a vector (independent of u, optional),
///        c is a constant (independent of u, optional).
///        GetHessian() returns the operator A.
///
class QuadraticFunctional : public Functional
{
public:
   QuadraticFunctional()
      : Functional(0)
      , A(nullptr), b(nullptr), c(0.0)
   {}
   QuadraticFunctional(const Operator *A, const Vector *b=nullptr,
                       const real_t beta=1.0,
                       const real_t c=0.0);
#ifdef MFEM_USE_MPI
   QuadraticFunctional(MPI_Comm comm)
      : QuadraticFunctional()
   { SetComm(comm); }

   QuadraticFunctional(MPI_Comm comm, const Operator *A, const Vector *b=nullptr,
                       const real_t beta=1.0, const real_t c=0.0)
      : QuadraticFunctional(A, b, beta, c)
   { SetComm(comm); }
#endif

   void SetOperator(const Operator &A)
   {
      MFEM_VERIFY(A.Width() == A.Height(),
                  "QuadraticFunctional: A must be a square operator.");
      this->A = &A;
      width = A.Width();
      aux.SetSize(width);
   }

   void SetVector(const Vector &b, const real_t beta=1.0)
   {
      MFEM_VERIFY(A != nullptr && A->Width() == b.Size(),
                  "QuadraticFunctional: A and b must have compatible sizes.");
      this->b = &b;
      this->beta = beta;
   }
   void SetConstant(real_t c)
   {
      this->c = c;
   }

   void Mult(const Vector &x, Vector &y) const override;

   void EvalGradient(const Vector &x, Vector &y) const override;
protected:
   const Operator *A;
   real_t beta;
   const Vector *b;
   real_t c;
   mutable Vector aux;
protected:
   /// @brief return the underlying Operator A
   /// @warning Modifying the returned operator leads to undefined behavior.
   Operator& GetHessian(const Vector &dummy) const override
   {
      return const_cast<Operator&>(*A);
   }
};

} // namespace mfem
#endif // MFEM_FUNCTIONAL_HPP
