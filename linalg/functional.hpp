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
#include <vector>

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
      , grad_operator(*this)
      , hessian_action_operator(*this)
   { }

#ifdef MFEM_USE_MPI
   Functional(MPI_Comm comm, int n=0)
      : Functional(n)
   { SetComm(comm); }

   void SetComm(MPI_Comm comm_)
   {
      parallel = comm_ != MPI_COMM_NULL;
      comm = comm_;
   }
   MPI_Comm GetComm() const { return comm; }
   bool IsParallel() const { return parallel; }
#else
   constexpr bool IsParallel() const { return false; }
#endif

   void SetRieszMap(Operator &op) { riesz_map = &op; }
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
#ifdef MFEM_USE_MPI
   bool parallel=false;
#else
   const static bool parallel=false;
#endif
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
      void SetX(const Vector &new_x) { x = &new_x; }
      void Mult(const Vector &d, Vector &y) const override { f.HessianMult(*x, d, y); }
   };
   friend class HessianActionOperator;

   mutable GradientOperator grad_operator;
   mutable HessianActionOperator hessian_action_operator;
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
      : Operator((int)funcs.size(), funcs[0]->Width())
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
      y.SetSize(Height());
      Vector yview;
      for (int i=0; i<Height(); i++)
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
      grads.SetSize(Width(), Height());
      Vector grad;
      for (int i=0; i<Height(); i++)
      {
         grads.GetColumnReference(i, grad);
         funcs[i]->GetGradient().Mult(x, grad);
      }
   }
   Functional &GetFunctional(int i) const
   {
      MFEM_VERIFY(i >= 0 && i < Height(),
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
   void SetComm(MPI_Comm comm_)
   {
      parallel = comm != MPI_COMM_NULL;
      comm = comm_;
   }
   MPI_Comm GetComm() const { return comm; }
#endif

protected:
#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif
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

class ConstrainedOptimizationProblem : public Functional
{
public:
   ConstrainedOptimizationProblem(Functional &objective_,
                                  Operator *eq_constraints_=nullptr,
                                  Operator *ineq_constraints_=nullptr)
      : Functional(objective_.Width())
      , objective(objective_)
      , eq_constraints(eq_constraints_)
      , ineq_constraints(ineq_constraints_)
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
      eq_residual = output_block.GetBlock(1);
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
   AugLagrangianFunctional(Functional &objective_,
                           Operator &eq_constraints_)
      : ConstrainedOptimizationProblem(objective_, &eq_constraints_)
      , lambda(eq_constraints_.Height())
      , mu(1.0)
      , eq_residual(eq_constraints_.Height())
      , eq_dir(eq_constraints_.Height())
   {
      lambda = 0.0;
   }

   void SetLambda(const Vector &lambda_)
   {
      MFEM_VERIFY(lambda_.Size() == eq_constraints->Height(),
                  "AugLagrangianFunctional: Lambda size does not match with the equality constraints.");
      lambda = lambda_;
   }

   void SetPenalty(real_t mu_)
   {
      MFEM_VERIFY(mu_ >= 0.0,
                  "AugLagrangianFunctional: Penalty parameter mu must be non-negative.");
      mu = mu_;
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
   QuadraticFunctional(const Operator *A_, const Vector *b_=nullptr,
                       const real_t beta_=1.0,
                       const real_t c_=0.0);
#ifdef MFEM_USE_MPI
   QuadraticFunctional(MPI_Comm comm_)
      : QuadraticFunctional()
   { SetComm(comm_); }

   QuadraticFunctional(MPI_Comm comm_, const Operator *A_,
                       const Vector *b_=nullptr,
                       const real_t beta_=1.0, const real_t c_=0.0)
      : QuadraticFunctional(A_, b_, beta_, c_)
   { SetComm(comm_); }
#endif

   void SetOperator(const Operator &A_);
   void SetVector(const Vector &b_, const real_t beta_=1.0);
   void SetConstant(real_t c_);

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
   Optimizer() : IterativeSolver(), f(nullptr) { }
#ifdef MFEM_USE_MPI
   Optimizer(MPI_Comm comm) : IterativeSolver(comm), f(nullptr) { }
#endif
   // @brief Set the subproblem functional operator
   // @param op the functional operator
   // @note The functional will be stored in subproblem, and oper will be set to the gradient of the functional.
   void SetOperator(const Functional &f_)
   {
      f = &f_;
      IterativeSolver::SetOperator(f_.GetGradient());
   }
   virtual void SetLinearSolver(Solver &prec) { IterativeSolver::SetPreconditioner(prec); }

   /// @brief This will abort. Should be called only with a Functional operator.
   void SetOperator(const Operator &op) override
   {
      MFEM_ABORT("OptSolver::SetOperator() should not be called directly. Use SetFunctional() instead.");
   }
protected:
   const Functional * f;
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
   void SetStepSize(real_t step_size_) { step_size = step_size_; }

   void Mult(const Vector &x, Vector &y) const override
   {
      dx.SetSize(x.Size());
      y.SetSize(x.Size());
      y = x;
      MFEM_ASSERT(f != nullptr,
                  "NewtonOptimizer::Mult() called without a functional operator.");
      MFEM_ASSERT(prec != nullptr,
                  "NewtonOptimizer::Mult() called without a linear solver.");
      for (int i=0; i<max_iter; i++)
      {
         oper->Mult(y, grad);
         Operator &hess = oper->GetGradient(y);
         prec->SetOperator(hess);
         prec->Mult(grad, dx);
         y.Add(-step_size, dx);
         if (Dot(dx, dx) < abs_tol*abs_tol)
         {
            break;
         }
      }
   }
private:
   mutable Vector grad;
   mutable Vector dx;
};

class GradientDescentOptimizer : public Optimizer
{
private:
   real_t step_size = 1.0; // default step size
public:
   GradientDescentOptimizer() : Optimizer() { }
#ifdef MFEM_USE_MPI
   GradientDescentOptimizer(MPI_Comm comm) : Optimizer(comm) { }
#endif
   void SetStepSize(real_t step_size_) { step_size = step_size_; }

   void Mult(const Vector &x, Vector &y) const override
   {
      grad.SetSize(x.Size());
      y.SetSize(x.Size());
      y = x;
      MFEM_ASSERT(f != nullptr,
                  "NewtonOptimizer::Mult() called without a functional operator.");
      MFEM_ASSERT(prec != nullptr,
                  "NewtonOptimizer::Mult() called without a linear solver.");
      for (int i=0; i<max_iter; i++)
      {
         oper->Mult(y, grad);
         y.Add(-step_size, grad);
         if (Dot(grad, grad) < abs_tol*abs_tol)
         {
            break;
         }
      }
   }
private:
   mutable Vector grad;
};


} // namespace mfem
#endif // MFEM_FUNCTIONAL_HPP
