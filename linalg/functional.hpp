#ifndef MFEM_FUNCTIONAL_HPP
#define MFEM_FUNCTIONAL_HPP

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI
#include "general/communication.hpp"
#endif

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
///    The GradientOperator::GetGradient(x) will call Functional::GetHessian(x)
///
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

/// @brief Stacked functioanl operator, [f1, ..., fk] where fi:R^n->R are functionals
/// Typical usage of this class is to provide a single operator for multiple constraints.
/// For example, if we have a minimization problem with k constraints,
/// min f0(u) s.t. fi(u)=0, i=1,...,k.
/// The Lagrangian functional is
/// L(u, lambda) = F0(u) + sum_i lambda_i * fi(u)
/// The first-order optimality conditions are
/// grad f0(u) + sum_i lambda_i * grad fi(u) = 0
/// fi(u) = 0
/// where lambda_i are the Lagrange multipliers.
///
/// StackedFunctional::GetGradient(u) returns an column-stacked gradient
/// That is, [grad f0(u), ..., grad fk(u)] in R^{n x k}
///
/// StackedFunctional::GetGradient(u).Mult(lambda, y) will contract the gradients with the Lagrange multipliers
/// y = sum lambda_i * grad fi(u)
/// StackedFunctional::GetGradient(u).MultTranspose(d, y) will return the directional derivative for each k
/// y[i] = <grad fi(u), d>
/// If you want to extract the gradient as a list of vectors, use
/// StackedFunctional::GetGradientColumns(const Vector &x, DenseMatrix &grad)
///
/// StackedFunctional::GetHessian(u, lambda).Mult(d, y) will return the contracted Hessian action
/// y = sum_i lambda_i * H_{fi}(u, d)
///
/// @warning Functionals should be all serial or all parallel.
///
class StackedFunctional : public Operator
{
public:
   StackedFunctional(int n=0)
      : Operator(0, n)
      , funcs(0)
      , grad_helper_op(*this)
      , hessian_helper_op(*this)
   {}
   StackedFunctional(Functional &f)
      : Operator(0, f.Width())
      , grad_helper_op(*this)
      , hessian_helper_op(*this)
   { AddFunctional(f); }

   StackedFunctional(const std::vector<Functional*> &funcs)
      : Operator(funcs.size(), funcs[0]->Width())
      , grad_helper_op(*this)
      , hessian_helper_op(*this)
   { for (auto &f : funcs) { AddFunctional(*f); } }

   void AddFunctional(Functional &f)
   {
#ifdef MFEM_USE_MPI
      if (funcs.empty()) { if (parallel) { SetComm(f.GetComm()); } }
#endif
      MFEM_VERIFY(f.Width() == Width(),
                  "StackedFunctional::AddFunctional: Functional width does not match with the operator.");
      MFEM_VERIFY(parallel == f.IsParallel(),
                  "StackedFunctional::AddFunctional: Parallelism mismatch.");
      funcs.push_back(&f);
      height++;
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      y.SetSize(funcs.size());
      Vector yview;
      for (int i=0; i<funcs.size(); i++)
      {
         yview.MakeRef(y, i, 1);
         funcs[i]->Mult(x, yview);
      }
   }

   Operator &GetGradient(const Vector &x) const override
   {
      return grad_helper_op;
   }

   void GetGradientMatrix(const Vector &x, DenseMatrix &grads) const
   {
      grads.SetSize(Width(), funcs.size());
      Vector grad;
      for (int i=0; i<funcs.size(); i++)
      {
         grads.GetColumnReference(i, grad);
         funcs[i]->GetGradient().Mult(x, grad);
      }
   }

   Operator &GetHessian(const Vector &x, const Vector &lambda) const
   {
      hessian_helper_op.SetX(x, lambda);
      return hessian_helper_op;
   }

   bool parallel;
   bool IsParallel() const { return parallel; }
#ifdef MFEM_USE_MPI
   void SetComm(MPI_Comm comm) { parallel = true; this->comm = comm; }
   MPI_Comm GetComm() const { return comm; }
#endif

protected:
   MPI_Comm comm;
   std::vector<Functional*> funcs;

   class GradientOperator : public Operator
   {
   public:
      GradientOperator(const StackedFunctional &op)
         : Operator(op.Width(), op.Height())
         , op(op)
         , tmp_grad(op.Width())
      {}
      void SetX(const Vector &x) const { x_curr = &x; }
      Operator &GetGradient(const Vector &lambda) const override
      {
         op.hessian_helper_op.SetX(*x_curr, lambda);
         return op.hessian_helper_op;
      }

      void Mult(const Vector &lambda, Vector &y) const override
      {
         y.SetSize(op.Width());
         y = 0.0;
         for (int i=0; i<op.Height(); i++)
         {
            op.funcs[i]->GetGradient().Mult(*x_curr, tmp_grad);
            y.Add(lambda[i], tmp_grad);
         }
      }
      void MultTranspose(const Vector &x, Vector &y) const override
      {
         y.SetSize(op.Height());
         for (int i=0; i<op.Height(); i++)
         {
            op.funcs[i]->GetGradient().Mult(x, tmp_grad);
            y[i] = InnerProduct(tmp_grad, *x_curr);
         }
#ifdef MFEM_USE_MPI
         if (op.IsParallel())
         {
            MPI_Allreduce(MPI_IN_PLACE, y.GetData(), op.Height(),
                          MPITypeMap<real_t>::mpi_type, MPI_SUM,
                          op.GetComm());
         }
#endif
      }

   private:
      const StackedFunctional &op;
      mutable const Vector *x_curr;
      mutable Vector tmp_grad;

   };
   class HessianActionOperator : public Operator
   {
   public:
      HessianActionOperator(const StackedFunctional &op)
         : Operator(op.Width()), op(op)
      {}
      void SetX(const Vector &x, const Vector &lambda) const { x_curr = &x; lambda_curr = &lambda; }

      void Mult(const Vector &d, Vector &y) const override
      {
         y.SetSize(op.Width());
         y = 0.0;
         for (int i=0; i<op.Height(); i++)
         {
            op.funcs[i]->GetGradient().GetGradient(*x_curr).Mult(d, tmp_hessian);
            y.Add((*lambda_curr)[i], tmp_hessian);
         }
      }
   private:
      const StackedFunctional &op;
      mutable Vector tmp_hessian;
      mutable const Vector *x_curr;
      mutable const Vector *lambda_curr;
   };
   friend class GradientOperator;
   friend class HessianActionOperator;
   mutable GradientOperator grad_helper_op;
   mutable HessianActionOperator hessian_helper_op;
private:
};

/// @brief A StackedFunctional that shares the evaluation point with the first functional.
/// All functionals should be of type SharedFunctional and support ShallowCopyProcessedX() with the first functional.
/// See, SharedFunctional::SharePoint() and SharedFunctional::ShallowCopyProcessedX()
/// @warning The caching should be enabled for all/none of the functionals before adding them to the StackedSharedFunctional.
class StackedSharedFunctional : public StackedFunctional
{
public:
   StackedSharedFunctional(int n=0)
      : StackedFunctional(n)
      , cache_enabled(false)
   {}
   StackedSharedFunctional(SharedFunctional &f)
      : StackedFunctional(f.Width())
      , cache_enabled(false)
   {
      AddFunctional(f);
   }
   StackedSharedFunctional(const std::vector<SharedFunctional*> &funcs)
      : StackedFunctional(funcs[0]->Width())
      , cache_enabled(false)
   { for (auto &f : funcs) { AddFunctional(*f); } }

   void AddFunctional(SharedFunctional &f)
   {
      if (!funcs.empty())
      {
         static_cast<SharedFunctional*>(funcs[0])->SharePoint(f);
         MFEM_VERIFY(f.IsCacheEnabled() == cache_enabled,
                     "StackedSharedFunctional: All functionals must have the same cache enabled state.");
      }
      else
      {
         cache_enabled = f.IsCacheEnabled();
      }
      StackedFunctional::AddFunctional(f);
   }
   bool IsCacheEnabled() const { return cache_enabled; }
   void Update(const Vector &x) const
   {
      MFEM_VERIFY(cache_enabled,
                  "StackedSharedFunctional::Update() called without cache enabled.");
      static_cast<SharedFunctional*>(funcs[0])->Update(x);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      if (!cache_enabled) { static_cast<SharedFunctional*>(funcs[0])->Update(x); }
      StackedFunctional::Mult(x, y);
   }

   Operator &GetGradient(const Vector &x) const override
   {
      if (!cache_enabled) { static_cast<SharedFunctional*>(funcs[0])->Update(x); }
      return StackedFunctional::GetGradient(x);
   }

   void GetGradientMatrix(const Vector &x, DenseMatrix &grads) const
   {
      if (!cache_enabled) { static_cast<SharedFunctional*>(funcs[0])->Update(x); }
      StackedFunctional::GetGradientMatrix(x, grads);
   }

   Operator &GetHessian(const Vector &x, const Vector &lambda) const
   {
      if (!cache_enabled) { static_cast<SharedFunctional*>(funcs[0])->Update(x); }
      return StackedFunctional::GetHessian(x, lambda);
   }
private:
   bool cache_enabled;

};

/// @brief A Lagrangian functional for
/// min F(u)
/// subject to C(u) = 0
/// That is, L(u, lambda) = F(u) + <lambda, C(u)>
///
/// We assume that F:R^n -> R is a functional,
///                C:R^n -> R^k is an equality constraint operator,
/// C should return a residual. That is, if you have
/// C(u) = c, then C.Mult(u, y) should return y[i] = C_i(u) - c_i.
///
/// C.GetGradient(u):R^k -> R^n that takes lambda and returns the contracted gradient at x
/// C.GetGradient(u).Mult(lambda, y) returns y = sum lambda_i * grad C_i(u)
///
/// C's gradient should support both Mult and MultTranspose methods
/// That is, C.GetGradient(u).Mult(lambda, y) returns y = sum lambda_i + grad C_i(u)
///          C.GetGradient(u).MultTranspose(d, y) returns y[i] = <grad C_i(u), d>
///
class LagrangianFunctional : public Functional
{
public:
   LagrangianFunctional(Functional &objective,
                        Operator &eq_constraints)
      : Functional(objective.Width() + eq_constraints.Height())
      , objective(objective)
      , eq_constraints(eq_constraints)
   {
      // Check Size
      MFEM_VERIFY(eq_constraints.Width() == objective.Width(),
                  "LagrangianFunctional: Equality constraints width does not match with the objective.");

#ifdef MFEM_USE_MPI
      if (objective.IsParallel()) { SetComm(objective.GetComm()); }
      /// No machanism to detect parallelism for Operator.
      /// Just check if eq_constraints and ineq_constraints are StackedFunctional
      if (auto *stacked_op = dynamic_cast<StackedFunctional*>(&eq_constraints))
      {
         MFEM_VERIFY(stacked_op->IsParallel() == objective.IsParallel(),
                     "LagrangianFunctional: Parallelism mismatch for equality constraints.");
      }
#endif
      offsets.SetSize(3);
      offsets[0] = 0;
      offsets[1] = objective.Width();
      offsets[2] = eq_constraints.Height();
      offsets.PartialSum();
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      const BlockVector input_block(const_cast<Vector&>(x), offsets);
      const Vector &u = input_block.GetBlock(0);
      const Vector &lambda = input_block.GetBlock(1);

      y.SetSize(1);
      y[0] = 0.0;
      objective.Mult(u, y);
      Vector eq_residual(eq_constraints.Height());
      eq_constraints.Mult(u, eq_residual);
      y[0] += InnerProduct(lambda, eq_residual);
   }

   void EvalGradient(const Vector &x, Vector &y) const override
   {
      const BlockVector input_block(const_cast<Vector&>(x), offsets);
      const Vector &u = input_block.GetBlock(0);
      const Vector &lambda = input_block.GetBlock(1);

      y.SetSize(Width());
      BlockVector output_block(y, offsets);
      Vector &opt_residual = output_block.GetBlock(0);
      Vector &eq_residual = output_block.GetBlock(1);
      y = 0.0;
      objective.GetGradient().Mult(u, opt_residual);
      eq_constraints.Mult(u, eq_residual);
      eq_constraints.GetGradient(u).AddMult(lambda, opt_residual);
   }

   /// @brief Evaluate the Hessian action at a point x=[u, lambda]
   /// and direction d=[v, mu]
   /// [H_F(u,d) + \sum_i lambda_i H_{C_i}(u, d) + <grad C_i(u), mu>
   void HessianMult(const Vector &x, const Vector &d, Vector &y) const override
   {
      const BlockVector input_block(const_cast<Vector&>(x), offsets);
      const Vector &u = input_block.GetBlock(0);
      const Vector &lambda = input_block.GetBlock(1);

      const BlockVector direction_block(const_cast<Vector&>(x), offsets);
      const Vector &v = direction_block.GetBlock(0);
      const Vector &mu = direction_block.GetBlock(1);

      y.SetSize(Width());
      BlockVector output_block(y, offsets);
      Vector &opt_H = output_block.GetBlock(0); // Optimality Hessian
      Vector &eq_H = output_block.GetBlock(1); // Equality Hessian

      // H_F(u,d) + \sum_i lambda_i H_{C_i}(u, d) + mu_i grad C_i(u)
      objective.GetGradient().GetGradient(u).Mult(v, opt_H);
      eq_constraints.GetGradient(u).GetGradient(lambda).AddMult(v, opt_H);
      eq_constraints.GetGradient(u).Mult(mu, eq_H);
      // <grad C_i(u), d>
      eq_constraints.GetGradient(u).MultTranspose(d, eq_H);
   }

protected:
   Functional &objective;
   Operator &eq_constraints;
   Array<int> offsets; // offsets for [x, lambda, mu]
};

/// @brief An augmented Lagrangian functional of the form
/// F(u) + 0.5 mu * C(u)^T C(u) + <lambda, C(u)>
/// where F is the objective functional,
/// C is the equality constraint operator,
/// lambda is the Lagrange multiplier vector (initialized to zero),
/// mu is the penalty parameter (defaults to 1.0)
///
/// AugLagrangianFunctional::Update() will update the penalty and Lagrange multiplier vectors
/// By default, lambda <- lambda + mu * C(u)
///             mu <- mu (no update)
class AugLagrangianFunctional : public Functional
{
public:
   AugLagrangianFunctional(Functional &objective,
                           Operator &eq_constraints)
      : Functional(objective.Width() + eq_constraints.Height())
      , objective(objective)
      , eq_constraints(eq_constraints)
      , lambda(eq_constraints.Height())
      , mu(1.0)
      , eq_residual(eq_constraints.Height())
      , eq_dir(eq_constraints.Height())
   {
      lambda = 0.0;
      // Check Size
      MFEM_VERIFY(eq_constraints.Width() == objective.Width(),
                  "AugLagrangianFunctional: Equality constraints width does not match with the objective.");

#ifdef MFEM_USE_MPI
      if (objective.IsParallel()) { SetComm(objective.GetComm()); }
      /// No machanism to detect parallelism for Operator.
      /// Just check if eq_constraints and ineq_constraints are StackedFunctional
      if (auto *stacked_op = dynamic_cast<StackedFunctional*>(&eq_constraints))
      {
         MFEM_VERIFY(stacked_op->IsParallel() == objective.IsParallel(),
                     "AugLagrangianFunctional: Parallelism mismatch for equality constraints.");
      }
#endif
   }

   void SetLambda(const Vector &lambda)
   {
      MFEM_VERIFY(lambda.Size() == eq_constraints.Height(),
                  "AugLagrangianFunctional: Lambda size does not match with the equality constraints.");
      this->lambda = lambda;
   }

   void SetPenalty(real_t mu)
   {
      MFEM_VERIFY(mu >= 0.0,
                  "AugLagrangianFunctional: Penalty parameter mu must be non-negative.");
      this->mu = mu;
   }

   virtual void Update(const Vector &x)
   {
      // Update the Lagrange multipliers
      eq_constraints.AddMult(x, lambda, mu);
      // Update the penalty parameter
      // Do nothing
   }


   void Mult(const Vector &x, Vector &y) const override
   {
      y.SetSize(1);
      objective.Mult(x, y);
      Vector eq_residual(eq_constraints.Height());
      eq_constraints.Mult(x, eq_residual);
      y[0] += InnerProduct(lambda, eq_residual);
      y[0] += 0.5 * mu * InnerProduct(eq_residual, eq_residual);
   }

   void EvalGradient(const Vector &x, Vector &y) const override
   {
      y.SetSize(Width());
      // grad F(u) + \sum_i (lambda_i + mu * C_i(u)) grad C_i(u)
      Vector curr_lambda = lambda;
      objective.GetGradient().Mult(x, y);
      eq_constraints.Mult(x, eq_residual);
      curr_lambda.Add(mu, eq_residual);
      eq_constraints.GetGradient(x).AddMult(curr_lambda, y);
   }

   /// @brief Evaluate the Hessian action at a point x=[u, lambda]
   /// and direction d=[v, mu]
   /// [H_F(u,d) + \sum_i lambda_i H_{C_i}(u, d) + <grad C_i(u), mu>
   void HessianMult(const Vector &x, const Vector &d, Vector &y) const override
   {
      // H_F(u,d) + \sum_i lambda_i H_{C_i}(u, d) + mu_i grad C_i(u) <grad C_i(u), d>
      objective.GetGradient().GetGradient(x).Mult(d, y);

      Vector curr_lambda = lambda;
      eq_constraints.Mult(x, eq_residual);
      curr_lambda.Add(mu, eq_residual);

      eq_constraints.GetGradient(x).GetGradient(curr_lambda).AddMult(d, y);
      // eq_dir = <grad C_i(u), d>
      eq_constraints.GetGradient(x).MultTranspose(d, eq_dir);
      // mu_i <grad C_i(u), eq_dir>
      eq_constraints.GetGradient(x).AddMult(eq_dir, y, mu);
   }

protected:
   Functional &objective;
   Operator &eq_constraints;
   Vector lambda;
   real_t mu;
   mutable Vector eq_residual; // residual of the equality constraints, R^k
   mutable Vector
   eq_dir; // directional derivative of the equality constraints, R^k
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
