#ifndef MFEM_FUNCTIONAL_HPP
#define MFEM_FUNCTIONAL_HPP
#include "operator.hpp"
#include "blockvector.hpp"
#include "solvers.hpp"
#include <cxxabi.h>

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
      , parallel(false)
      , grad_operator(*this)
      , hessian_action_operator(*this)
   { }

#ifdef MFEM_USE_MPI
   Functional(MPI_Comm comm, int n=0)
      : Functional(n)
   { SetComm(comm); }

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

/// @brief A base class for functionals that shares evaluation points
/// either across multiple functionals or across multiple member functions.
/// This requires to update the evaluation point x manually by calling SharedFunctional::Update(x)
/// Usually, x is a T-vector, and processed_x is an L-vector.
/// It also supports caching of the evaluation results of Mult() and EvalGradient().
/// Currently, the Hessian will not be cached, as it requires to store a matrix (which may not be sparse).
/// Therefore, we always call the HessianMultProcessed() to evaluate the Hessian action at the processed point.
/// As `hessianCached` will be set to false when the evaluation point is updated,
/// the derived class can use it to check evaluation point update and assemble the Hessian if needed.
///
/// The derived class must implement
/// 1) ProcessX(const Vector &x) which sets the processed_x,
/// 2) ShallowCopyProcessedX(OtherSharedFunctional &owner) which copies the processed_x from another functional
/// 3) Mult(Vector &y) which evaluates the functional at processed_x,
/// 4) (optional) EvalGradient(Vector &y) which evaluates the gradient at processed_x,
/// 5) (optional) HessianMultProcessed(const Vector &d, Vector &y) which evaluates the Hessian action at processed_x.
class SharedFunctional : public Functional
{
public:
   SharedFunctional(int n=0)
      : Functional(n)
      , cache_enabled(false)
      , multCached(false)
      , gradCached(false)
      , hessianCached(false)
      , processedX_owner(nullptr)
   { }
#ifdef MFEM_USE_MPI
   SharedFunctional(MPI_Comm comm, int n=0)
      : SharedFunctional(n)
   { SetComm(comm); }
#endif


   /// @brief Share the processed point with another functional. Input is a viewer.
   /// The viewer should implement ShallowCopyProcessedX() to copy the processed point data.
   void SharePoint(SharedFunctional &viewer);

   /// @brief Enable of disable caching of the functional.
   /// Cahche will be invalidated when Update() is called, and Mult(y), EvalGradient(y) will cache (or use) result.
   void EnableCache(bool enable=true);
   bool IsCacheEnabled() const { return cache_enabled; }
   bool IsMultCached() const { return multCached; }
   bool IsGradCached() const { return gradCached; }

   /// @brief Update the processed point with a new evaluation point x.
   /// When caching is enabled, this will invalidate the cached values of Mult() and EvalGradient()
   void Update(const Vector &x) const;

   /// @brief Helper method to evaluate the functional at the processed point. X is not used.
   void Mult(const Vector &dummy, Vector &y) const override final;

   /// Helper method to evaluate the gradient of functional at the processed point. Use cached value if available.
   void EvalGradient(const Vector &dummy, Vector &y) const override final;

   /// Helper method to evaluate the gradient of functional at the processed point.
   /// Caching is not used here to avoid assembling possibly full or expensive Hessian assembly.
   /// If the Hessian can be stored, then assemble it in HessianMult(const Vector &, Vector &).
   /// After assembling the Hessian, set hessianCached to true and reuse the assembled Hessian until hessianCached is false.
   /// hessianCached will be set to false when Update() is called.
   void HessianMult(const Vector &dummy, const Vector &d,
                    Vector &y) const override final
   { HessianMultCurrent(d, y); }

   /// Evaluate the functional at the processed point. Use processed evaluation point.
   virtual void MultCurrent(Vector &y) const = 0;
   /// Evaluate the gradient of functional at the processed point. Use processed evaluation point.
   virtual void EvalGradientCurrent(Vector &y) const
   {
      MFEM_ABORT("SharedFunctional::EvalGrad() is not implemented.");
   }
   /// Evaluate the Hessian action of functional at the processed point. Use processed evaluation point.
   virtual void HessianMultCurrent(const Vector &d, Vector &y) const
   {
      MFEM_ABORT("SharedFunctional::HessianMult() is not implemented.");
   }

   // return the reference to the cached value of Mult()
   real_t &GetCachedMult() const
   {
      MFEM_VERIFY(cache_enabled,
                  "SharedFunctional::GetCachedGradient() called without cache enabled.");
      return current_mult_value;
   }
   // return the reference to the cached value of EvalGradient()
   Vector &GetCachedGradient() const
   {
      MFEM_VERIFY(cache_enabled,
                  "SharedFunctional::GetCachedGradient() called without cache enabled.");
      return current_grad_value;
   }

protected:
   mutable bool cache_enabled;
   mutable bool multCached;
   mutable real_t current_mult_value;
   mutable bool gradCached;
   mutable Vector current_grad_value;
   mutable bool hessianCached;

   /// Derived classes must implement the following methods:

   /// Process evaluation point x. This will be called by Update(x).
   virtual void ProcessX(const Vector &x) const = 0;

   /// By default, this method will abort. Derived classes should implement for specific types using dynamic_cast.
   virtual void ShallowCopyProcessedX(SharedFunctional &owner);

private:
   SharedFunctional *processedX_owner; // null if owner==this
   std::vector<SharedFunctional*> processedX_viewers;


};

/// @brief A base class for evaluating multiple functionals [f_1, ..., f_k] at once.
/// Typical usage of this class is to evaluate and get access to the Hessian of multiple constraints. See, OptimizationProblem.
/// Let F = [f_1, ..., f_k] be a multi-functional F:u |-> [f_1(u), f_2(u), ..., f_k(u)] where u in R^n.
/// Then its gradient is an Operator G(k, n) such that F.GetGradient(u).Mult(d, z) returns [df_1(u)/du1(d), ..., df_k(u)/duk(d)]
/// Hessian action is not supported by this class.
/// That is,
///   F.GetGradient(u).Mult(d, z) = [<df_1(u)/du1, d>, ..., <df_k(u)/duk, d>]
///   F.GetGradient(u).MultTranspose(lambda, z) = sum_i lambda_i * df_i(u)/du
///
/// All functionals share the same evaluation point x with the first functional.
/// Therefore, `ShallowCopyProcessedX()` must be implemented for all shared functionals added to this class for first functional's type.
/// See, SharedFunctional::ShallowCopyProcessedX() and MultiSharedFunctional::AddFunctional()
class MultiSharedFunctional : public Operator
{

public:
   MultiSharedFunctional(int n=0)
      : Operator(0, n)
      , funcs(0)
      , grad_helper_op(*this)
      , grad_vecs(0)
   {}

   void AddFunctional(SharedFunctional &f)
   {
      // share point with the first functional
      if (!funcs.empty())
      {
         funcs[0]->SharePoint(f);
      }
      else
      {
         cache_enabled = f.IsCacheEnabled();
      }
      MFEM_VERIFY(f.IsCacheEnabled() == cache_enabled,
                  "MultiSharedFunctional::AddFunctional: All functionals must have the same cache enabled state.");

      funcs.push_back(&f);
      height++;
      if (cache_enabled)
      {
         grad_vecs.push_back(&f.GetCachedGradient());
      }
   }

   void Update(const Vector &x) const
   {
      funcs[0]->Update(x);
      // other functionals will use the same processed point
   }
   bool IsCacheEnabled() const { return cache_enabled; }

   void Mult(const Vector &x, Vector &y) const override
   {
      y.SetSize(height);
      Vector yview;
      for (int i=0; i<funcs.size(); i++)
      {
         yview.MakeRef(y, i, 1);
         funcs[i]->MultCurrent(yview);
      }
   }

   Operator &GetGradient(const Vector &x) const override
   {
      funcs[0]->Update(x);
      return grad_helper_op;
   }
   std::vector<Vector*> &GetGradientVectors() const
   {
      MFEM_VERIFY(cache_enabled,
                  "MultiSharedFunctional::GetGradientVectors() called without cache enabled.");
      return grad_vecs;
   }
protected:
   std::vector<SharedFunctional*> funcs; // List of functionals
   mutable std::vector<Vector*> grad_vecs;
   bool share_point;
   bool cache_enabled;
private:
   class GradientOperator : public Operator
   {
   private:
      const MultiSharedFunctional &op;
      mutable Vector tmp_grad;
   public:
      GradientOperator(const MultiSharedFunctional &op)
         : Operator(op.Height(), op.Width())
         , op(op)
         , tmp_grad(op.cache_enabled ? 0 : op.Width())
      { }

      void Mult(const Vector &d, Vector &y) const override
      {
         y.SetSize(op.Width());
         Vector yview;
         for (int i=0; i<op.funcs.size(); i++)
         {
            op.funcs[i]->GetGradient().Mult(d, yview);
         }
      }

      void MultTranspose(const Vector &lambda, Vector &y) const override
      {
         y.SetSize(op.Width());
         y = 0.0;
         for (int i=0; i<op.funcs.size(); i++)
         {
            if (op.cache_enabled)
            {
               // tmp_grad points to the cached gradient to avoid copy
               tmp_grad.MakeRef(*op.grad_vecs[i], 0);
            }
            // this will reuse the cached gradient if available
            op.funcs[i]->EvalGradientCurrent(tmp_grad);
            y.Add(lambda[i], tmp_grad);
         }
      }

      Operator &GetGradient(const Vector &lambda) const override
      {
         MFEM_ABORT("MultiSharedFunctional::GradientOperator::GetGradient(): GetGradient() is not supported.");
      }
   };
   friend class GradientOperator;
   mutable GradientOperator grad_helper_op;
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

   void SetOperator(const Operator &A);
   void SetVector(const Vector &b, const real_t beta=1.0);
   void SetConstant(real_t c);

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
