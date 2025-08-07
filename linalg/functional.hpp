#ifndef MFEM_FUNCTIONAL_HPP
#define MFEM_FUNCTIONAL_HPP

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI
#include "../general/communication.hpp"
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
   Operator * riesz_map = nullptr; ///< Riesz map operator, if available
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
   Functional(MPI_Comm comm, int n)
      : Functional(n)
   { SetComm(comm); }

   void SetComm(MPI_Comm comm) { parallel = true; this->comm = comm; }
   MPI_Comm GetComm() const { return comm; }
   bool IsParallel() const { return parallel; }
#else
   constexpr bool IsParallel() const { return false; }
#endif

   void SetRieszMap(Operator &riesz_map) { this->riesz_map = &riesz_map; }
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
   private: const Functional &f; mutable Vector der;
   public:
      GradientOperator(const Functional &f) : Operator(f.Width()), f(f) {}
      /// @brief Evaluate the gradient of Functional at a point x
      void Mult(const Vector &x, Vector &y) const override final
      {
         if (f.riesz_map)
         {
            der.SetSize(f.Width());
            f.EvalGradient(x, der);
            f.riesz_map->Mult(der, y);
         }
         else
         {
            f.EvalGradient(x, y);
         }
      }
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
///
/// When a functional shares evaluation points, then the evaluation point
/// for the "viewer' functional will ignore the input x.
/// The owner is responsible for processing the evaluation point,
/// and the viewer have shallow copy of the processed evaluation point.
/// @warning Evaluating the viewer functional prior to the owner functional will results in
/// unsynchronized evaluation point, which is not recommended.
///
/// A derived class must implement ProcessX() and ShallowCopyprocessedX().
/// See, SharedFunctional::SharePoint() and SharedFunctional::ProcessX().
///
/// When the manual update is enabled, the input x will be ignored,
/// Users must call SharedFunctional::Update() at the owner functional to update x.
///
/// The derived class must implement
/// 1) ProcessX(const Vector &x) which sets internally processed evaluation point
/// 2) ShallowCopyProcessedX(OtherSharedFunctional &owner) which copies processed evaluation point
/// 3) MultCurrent(Vector &y)
/// 4) (optional) EvalGradientCurrent(Vector &y)
/// 5) (optional) HessianMultCurrent(const Vector &d, Vector &y)
class SharedFunctional : public Functional
{
public:
   SharedFunctional(int n=0)
      : Functional(n)
      , owner(nullptr)
      , viewers(0)
      , manual_update(false)
   { }
#ifdef MFEM_USE_MPI
   SharedFunctional(MPI_Comm comm, int n)
      : SharedFunctional(n)
   { SetComm(comm); }
#endif

   void EnableManualUpdate(bool enable=true) { manual_update = enable; }
   bool IsManualUpdateEnabled() const { return manual_update; }


   /// @brief Share the processed point with another functional. Input is a viewer.
   /// The viewer should implement ShallowCopyProcessedX() to copy the processed point data.
   void SharePointTo(SharedFunctional &viewer);

   /// @brief Update the processed point with a new evaluation point x.
   /// When caching is enabled, this will invalidate the cached values of Mult() and EvalGradient()
   /// This will increase owner's sequence number, not the viewer's.
   void Update(const Vector &x) const;
   /// @brief Get evaluation point update sequence number of the current functional.
   int GetSequence() const { return sequence; }
   /// @brief Get the owner sequence number of the current functional.
   int GetOwnerSequence() const { return owner ? owner->GetSequence() : sequence; }

   /// @brief Helper method to evaluate the functional at the processed point. X is not used.
   void Mult(const Vector &x, Vector &y) const override final
   {
      if (owner == nullptr && !manual_update) { ProcessX(x); }
      MultCurrent(y);
   }
   /// Helper method to evaluate the gradient of functional at the processed point. Use cached value if available.
   void EvalGradient(const Vector &x, Vector &y) const override final
   {
      if (owner == nullptr && !manual_update) { ProcessX(x); }
      EvalGradientCurrent(y);
   }
   /// Helper method to evaluate the gradient of functional at the processed point.
   /// Caching is not used here to avoid assembling possibly full or expensive Hessian assembly.
   /// If the Hessian can be stored, then assemble it in HessianMult(const Vector &, Vector &).
   /// After assembling the Hessian, set hessianCached to true and reuse the assembled Hessian until hessianCached is false.
   /// hessianCached will be set to false when Update() is called.
   void HessianMult(const Vector &x, const Vector &d,
                    Vector &y) const override final
   {
      if (owner == nullptr && !manual_update) { ProcessX(x); }
      HessianMultCurrent(d, y);
   }

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

protected:
   /// Derived classes must implement the following methods:

   /// Process evaluation point x. This will be called by Update(x).
   virtual void ProcessX(const Vector &x) const = 0;

   /// By default, this method will abort. Derived classes should implement for specific types using dynamic_cast.
   virtual void ShallowCopyProcessedX(SharedFunctional &owner);
   // This can be used to track the evaluation point update sequence.
   // One use case is to track Mult() calls to avoid re-evaluating the same point.
   // In that case, create a multSequence variable in the derived class,
   // and reset it to multSequence=sequence when Mult() actually evaluates.
   // Otherwise, return the cached value.
   mutable int sequence;

private:
   SharedFunctional *owner; // null if owner==this
   std::vector<SharedFunctional*> viewers;
   bool manual_update;
};

/// @brief Stacked functioanl operator, [f1, ..., fk] where fi:R^n->R are functionals
/*
   Typical usage of this class is to provide a single operator for multiple constraints.
   For example, consider a minimization problem with k constraints,
   min f0(u) s.t. fi(u)=0, i=1,...,k.
   The Lagrangian functional is
   L(u, lambda) = F0(u) + sum_i lambda_i * fi(u)
   The first-order optimality conditions are
   grad f0(u) + sum_i lambda_i * grad fi(u) = 0
   fi(u) = 0
   where lambda_i are the Lagrange multipliers.
   The StackedFunctional class can be used to represent the list of constraints fi(u).

   StackedFunctional::Mult(u, y) will evaluate each functional y[i]=fi(u)

   StackedFunctional::GetGradient(u) represents an operator, column-stacked gradient
   That is, [grad f0(u), ..., grad fk(u)] in R^{n x k}
   If you want to extract the gradient as a matrix, use
   StackedFunctional::GetGradientMatrix(const Vector &x, DenseMatrix &grad)
   As functionals are not assumed to return a sparse vector, the gradient is dense.

   StackedFunctional::GetGradient(u).Mult(lambda, y) contract the gradients with the Lagrange multipliers
   y = sum lambda_i * grad fi(u)
   StackedFunctional::GetGradient(u).MultTranspose(d, y) return the directional derivative for each k
   y[i] = <grad fi(u), d>

   StackedFunctional::GetHessian(u, lambda).Mult(d, y) will return the contracted Hessian action
   y = sum_i lambda_i * H_{fi}(u, d)
*/
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
      if (funcs.empty()) { if (f.IsParallel()) { SetComm(f.GetComm()); } }
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
      grad_helper_op.SetX(x);
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
   Functional &GetFunctional(int i) const
   {
      MFEM_VERIFY(i >= 0 && i < funcs.size(),
                  "StackedFunctional::GetFunctional: Index out of bounds.");
      return *funcs[i];
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
/// @warning The manual update should be enabled for all/none of the functionals before adding them to the StackedSharedFunctional.
class StackedSharedFunctional : public StackedFunctional
{
public:
   StackedSharedFunctional(int n=0)
      : StackedFunctional(n)
   {}
   StackedSharedFunctional(SharedFunctional &f)
      : StackedFunctional(f.Width())
   { AddFunctional(f); }
   StackedSharedFunctional(const std::vector<SharedFunctional*> &funcs)
      : StackedFunctional(funcs[0]->Width())
   { for (auto &f : funcs) { AddFunctional(*f); } }
   void EnableManualUpdate(bool enable=true)
   {
      manual_update = enable;
      for (auto &f : funcs) { static_cast<SharedFunctional*>(f)->EnableManualUpdate(enable); }
   }
   bool IsManualUpdateEnabled() const { return manual_update; }
   void Update(const Vector &x)
   {
      SharedFunctional *owner = static_cast<SharedFunctional*>(funcs[0]);
      owner->Update(x);
      sequence = owner->GetSequence();
   }

   void AddFunctional(SharedFunctional &f)
   {
      if (funcs.empty())
      {
         manual_update = f.IsManualUpdateEnabled();
      }
      else
      {
         MFEM_VERIFY(f.IsManualUpdateEnabled() == manual_update,
                     "StackedSharedFunctional: Cannot add a functional with different manual update setting.");
         static_cast<SharedFunctional*>(funcs[0])->SharePointTo(f);
      }
      StackedFunctional::AddFunctional(f);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      if (!manual_update) { static_cast<SharedFunctional*>(funcs[0])->Update(x); }
      StackedFunctional::Mult(x, y);
   }

   Operator &GetGradient(const Vector &x) const override
   {
      if (!manual_update) { static_cast<SharedFunctional*>(funcs[0])->Update(x); }
      return StackedFunctional::GetGradient(x);
   }

   void GetGradientMatrix(const Vector &x, DenseMatrix &grads) const
   {
      if (!manual_update) { static_cast<SharedFunctional*>(funcs[0])->Update(x); }
      StackedFunctional::GetGradientMatrix(x, grads);
   }

   Operator &GetHessian(const Vector &x, const Vector &lambda) const
   {
      if (!manual_update) { static_cast<SharedFunctional*>(funcs[0])->Update(x); }
      return StackedFunctional::GetHessian(x, lambda);
   }
private:
   bool manual_update = false;
   int sequence = 0;
};

class ConstrainedOptimizationProblem : public Functional
{
public:
   ConstrainedOptimizationProblem(Functional &objective,
                                  Operator *eq_constraints=nullptr,
                                  Operator *ineq_constraints=nullptr)
      : Functional(objective.Width())
      , objective(objective)
      , eq_constraints(eq_constraints)
      , ineq_constraints(ineq_constraints)
   {
      // Check Size
      MFEM_VERIFY((eq_constraints == nullptr ||
                   eq_constraints->Width() == objective.Width()),
                  "ConstrainedFunctional: Equality constraints width does not match with the objective.");
      MFEM_VERIFY((ineq_constraints == nullptr ||
                   ineq_constraints->Width() == objective.Width()),
                  "ConstrainedFunctional: Inequality constraints width does not match with the objective.");
#ifdef MFEM_USE_MPI
      if (objective.IsParallel()) { SetComm(objective.GetComm()); }
#endif
   }

   Functional &GetObjective() { return objective; }
   const Functional &GetObjective() const { return objective; }
   Operator *GetEqualityConstraints() { return eq_constraints; }
   const Operator *GetEqualityConstraints() const { return eq_constraints; }
   Operator *GetInequalityConstraints() { return ineq_constraints; }
   const Operator *GetInequalityConstraints() const { return ineq_constraints; }
protected:
   Functional &objective;
   Operator *eq_constraints;
   Operator *ineq_constraints;
};

/// @brief A Lagrangian functional for
/// min F(u)
/// subject to C(u) = 0
/// That is, L(u, lambda) = F(u) + <lambda, C(u)>
///
/// We assume that F:R^n -> R is a functional,
///                C:R^n -> R^k is an equality constraint operator,
/// C should return a residual. That is,
/// C(u) = c, then C.Mult(u, y) should return y[i] = C_i(u) - c_i.
///
/// C.GetGradient(u):R^k -> R^n that takes lambda and returns the contracted gradient at x
/// C.GetGradient(u).Mult(lambda, y) returns y = sum lambda_i * grad C_i(u)
///
/// C's gradient should support MultTranspose method
/// That is, C.GetGradient(u).MultTranspose(d, y) returns y[i] = <grad C_i(u), d>
///
class LagrangianFunctional : public ConstrainedOptimizationProblem
{
private:
   mutable Vector eq_residual;
public:
   LagrangianFunctional(Functional &objective,
                        Operator &eq_constraints)
      : ConstrainedOptimizationProblem(objective, &eq_constraints)
      , eq_residual(eq_constraints.Height())
   {
      width = objective.Width() + eq_constraints.Height();

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
      eq_constraints->Mult(u, eq_residual);
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
      // grad F(u) + \sum_i lambda_i grad C_i(u)
      objective.GetGradient().Mult(u, opt_residual);
      eq_constraints->GetGradient(u).AddMult(lambda, opt_residual);
      // grad C_i(u)^T
      eq_constraints->GetGradient(u).MultTranspose(u, eq_residual);
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
      eq_constraints->GetGradient(u).GetGradient(lambda).AddMult(v, opt_H);
      eq_constraints->GetGradient(u).Mult(mu, eq_H);
      // <grad C_i(u), d>
      eq_constraints->GetGradient(u).MultTranspose(d, eq_H);
   }

protected:
   Array<int> offsets; // offsets for [x, lambda, mu]
};

/// @brief An augmented Lagrangian functional of the form
/// F(u) + 0.5 mu * ||C(u)||^2 + <lambda, C(u)>
/// where F is the objective functional,
/// C is the equality constraint operator,
/// lambda is the Lagrange multiplier vector (initialized to zero),
/// mu is the penalty parameter (defaults to 1.0)
///
/// Currently, only equality constraints are supported.
///
/// AugLagrangianFunctional::Update() will update the penalty and Lagrange multiplier vectors
/// By default, lambda <- lambda + mu * C(u)
///             mu <- mu (no update)
class AugLagrangianFunctional : public ConstrainedOptimizationProblem
{
public:
   AugLagrangianFunctional(Functional &objective,
                           Operator &eq_constraints)
      : ConstrainedOptimizationProblem(objective, &eq_constraints)
      , lambda(eq_constraints.Height())
      , mu(1.0)
      , eq_residual(eq_constraints.Height())
      , eq_dir(eq_constraints.Height())
   {
      lambda = 0.0;
   }

   void SetLambda(const Vector &lambda)
   {
      MFEM_VERIFY(lambda.Size() == eq_constraints->Height(),
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
      eq_constraints->AddMult(x, lambda, mu);
      // Update the penalty parameter
      // Do nothing
   }
   const Vector &GetLambda() const { return lambda; }
   real_t GetPenalty() const { return mu; }


   void Mult(const Vector &x, Vector &y) const override
   {
      y.SetSize(1);
      objective.Mult(x, y);
      Vector eq_residual(eq_constraints->Height());
      eq_constraints->Mult(x, eq_residual);
      y[0] += lambda*eq_residual;
      y[0] += 0.5 * mu * (eq_residual*eq_residual);
   }

   void EvalGradient(const Vector &x, Vector &y) const override
   {
      y.SetSize(Width());
      // grad F(x) + \sum_i (lambda_i + mu * C_i(x)) grad C_i(x)
      Vector curr_lambda = lambda; // store lambda + mu * C(x)
      objective.GetGradient().Mult(x, y);
      eq_constraints->Mult(x, eq_residual);
      curr_lambda.Add(mu, eq_residual);
      eq_constraints->GetGradient(x).AddMult(curr_lambda, y);
   }

   /// @brief Evaluate the Hessian action at a point x=[u, lambda]
   /// and direction d=[v, mu]
   /// [H_F(u,d) + \sum_i lambda_i H_{C_i}(u, d) + <grad C_i(u), mu>
   void HessianMult(const Vector &x, const Vector &d, Vector &y) const override
   {
      // H_F(u,d) + \sum_i lambda_i H_{C_i}(u, d) + mu_i grad C_i(u) <grad C_i(u), d>
      objective.GetGradient().GetGradient(x).Mult(d, y);

      Vector curr_lambda = lambda;
      eq_constraints->Mult(x, eq_residual);
      curr_lambda.Add(mu, eq_residual);

      eq_constraints->GetGradient(x).GetGradient(curr_lambda).AddMult(d, y);
      // eq_dir = <grad C_i(u), d>
      eq_constraints->GetGradient(x).MultTranspose(d, eq_dir);
      // mu_i <grad C_i(u), eq_dir>
      eq_constraints->GetGradient(x).AddMult(eq_dir, y, mu);
   }

protected:
   Vector lambda;
   real_t mu;
   mutable Vector eq_residual; // residual of the equality constraints, R^k
   // directional derivative of the equality constraints, R^k
   mutable Vector eq_dir;
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

class Optimizer : public IterativeSolver
{
public:
   Optimizer() : IterativeSolver(), subproblem(nullptr) { }
#ifdef MFEM_USE_MPI
   Optimizer(MPI_Comm comm) : IterativeSolver(comm), subproblem(nullptr) { }
#endif
   // @brief Set the subproblem functional operator
   // @param op the functional operator
   // @note The functional will be stored in subproblem, and oper will be set to the gradient of the functional.
   void SetOperator(const Functional &f)
   {
      subproblem = &f;
      IterativeSolver::SetOperator(f.GetGradient());
   }
   virtual void SetLinearSolver(Solver &prec) { IterativeSolver::SetPreconditioner(prec); }

   /// @brief This will abort. Should be called only with a Functional operator.
   void SetOperator(const Operator &op) override
   {
      MFEM_ABORT("OptSolver::SetOperator() should not be called directly. Use SetFunctional() instead.");
   }
protected:
   const Functional * subproblem;
};

class NewtonOptimizer : public Optimizer
{
private:
   real_t step_size = 1.0; // default step size
public:
   NewtonOptimizer() : Optimizer() { }
#ifdef MFEM_USE_MPI
   NewtonOptimizer(MPI_Comm comm) : Optimizer(comm) { }
#endif
   void SetStepSize(real_t step_size) { this->step_size = step_size; }

   void Mult(const Vector &x, Vector &y) const override
   {
      dx.SetSize(x.Size());
      y.SetSize(x.Size());
      y = x;
      MFEM_ASSERT(subproblem != nullptr,
                  "NewtonOptimizer::Mult() called without a functional operator.");
      MFEM_ASSERT(prec != nullptr,
                  "NewtonOptimizer::Mult() called without a linear solver.");
      for (int i=0; i<max_iter; i++)
      {
         Step(y, dx, step_size);
         if (Dot(dx, dx) < abs_tol*abs_tol)
         {
            break;
         }
      }
   }
private:
   // inplace Newton step
   // x <- x - step_size * dx, where H_f(x) dx = grad_f(x)
   void Step(Vector &x, Vector &dx, real_t step_size=1.0) const
   {
      oper->Mult(x, grad);
      Operator &hess = oper->GetGradient(x);
      prec->SetOperator(hess);
      prec->Mult(grad, dx);
      x.Add(-step_size, dx);
   }
   mutable Vector grad;
   mutable Vector dx;
};

class GradientDescentOptimizer : public Optimizer
{
};


} // namespace mfem
#endif // MFEM_FUNCTIONAL_HPP
