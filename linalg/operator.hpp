// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_OPERATOR
#define MFEM_OPERATOR

#include "vector.hpp"

namespace mfem
{

class ConstrainedOperator;
class RectangularConstrainedOperator;

/// Abstract operator
class Operator
{
protected:
   int height; ///< Dimension of the output / number of rows in the matrix.
   int width;  ///< Dimension of the input / number of columns in the matrix.

   /// see FormSystemOperator()
   void FormConstrainedSystemOperator(
      const Array<int> &ess_tdof_list, ConstrainedOperator* &Aout);

   /// see FormRectangularSystemOperator()
   void FormRectangularConstrainedSystemOperator(
      const Array<int> &trial_tdof_list,
      const Array<int> &test_tdof_list,
      RectangularConstrainedOperator* &Aout);

   /// Returns RAP Operator of this, taking in input/output Prolongation matrices
   Operator *SetupRAP(const Operator *Pi, const Operator *Po);

public:
   /// Defines operator diagonal policy upon elimination of rows and/or columns.
   enum DiagonalPolicy
   {
      DIAG_ZERO, ///< Set the diagonal value to zero
      DIAG_ONE,  ///< Set the diagonal value to one
      DIAG_KEEP  ///< Keep the diagonal value
   };

   /// Initializes memory for true vectors of linear system
   void InitTVectors(const Operator *Po, const Operator *Ri, const Operator *Pi,
                     Vector &x, Vector &b,
                     Vector &X, Vector &B) const;

   /// Construct a square Operator with given size s (default 0).
   explicit Operator(int s = 0) { height = width = s; }

   /** @brief Construct an Operator with the given height (output size) and
       width (input size). */
   Operator(int h, int w) { height = h; width = w; }

   /// Get the height (size of output) of the Operator. Synonym with NumRows().
   inline int Height() const { return height; }
   /** @brief Get the number of rows (size of output) of the Operator. Synonym
       with Height(). */
   inline int NumRows() const { return height; }

   /// Get the width (size of input) of the Operator. Synonym with NumCols().
   inline int Width() const { return width; }
   /** @brief Get the number of columns (size of input) of the Operator. Synonym
       with Width(). */
   inline int NumCols() const { return width; }

   /// Return the MemoryClass preferred by the Operator.
   /** This is the MemoryClass that will be used to access the input and output
       vectors in the Mult() and MultTranspose() methods.

       For example, classes using the MFEM_FORALL macro for implementation can
       return the value returned by Device::GetMemoryClass().

       The default implementation of this method in class Operator returns
       MemoryClass::HOST. */
   virtual MemoryClass GetMemoryClass() const { return MemoryClass::HOST; }

   /// Operator application: `y=A(x)`.
   virtual void Mult(const Vector &x, Vector &y) const = 0;

   /** @brief Action of the transpose operator: `y=A^t(x)`. The default behavior
       in class Operator is to generate an error. */
   virtual void MultTranspose(const Vector &x, Vector &y) const
   { mfem_error("Operator::MultTranspose() is not overloaded!"); }

   /** @brief Evaluate the gradient operator at the point @a x. The default
       behavior in class Operator is to generate an error. */
   virtual Operator &GetGradient(const Vector &x) const
   {
      mfem_error("Operator::GetGradient() is not overloaded!");
      return const_cast<Operator &>(*this);
   }

   /** @brief Prolongation operator from linear algebra (linear system) vectors,
       to input vectors for the operator. `NULL` means identity. */
   virtual const Operator *GetProlongation() const { return NULL; }
   /** @brief Restriction operator from input vectors for the operator to linear
       algebra (linear system) vectors. `NULL` means identity. */
   virtual const Operator *GetRestriction() const  { return NULL; }
   /** @brief Prolongation operator from linear algebra (linear system) vectors,
       to output vectors for the operator. `NULL` means identity. */
   virtual const Operator *GetOutputProlongation() const
   {
      return GetProlongation(); // Assume square unless specialized
   }
   /** @brief Restriction operator from output vectors for the operator to linear
       algebra (linear system) vectors. `NULL` means identity. */
   virtual const Operator *GetOutputRestriction() const
   {
      return GetRestriction(); // Assume square unless specialized
   }

   /** @brief Form a constrained linear system using a matrix-free approach.

       Assuming square operator, form the operator linear system `A(X)=B`,
       corresponding to it and the right-hand side @a b, by applying any
       necessary transformations such as: parallel assembly, conforming
       constraints for non-conforming AMR and eliminating boundary conditions.
       @note Static condensation and hybridization are not supported for general
       operators (cf. the analogous methods BilinearForm::FormLinearSystem() and
       ParBilinearForm::FormLinearSystem()).

       The constraints are specified through the prolongation P from
       GetProlongation(), and restriction R from GetRestriction() methods, which
       are e.g. available through the (parallel) finite element space of any
       (parallel) bilinear form operator. We assume that the operator is square,
       using the same input and output space, so we have: `A(X)=[P^t (*this)
       P](X)`, `B=P^t(b)`, and `X=R(x)`.

       The vector @a x must contain the essential boundary condition values.
       These are eliminated through the ConstrainedOperator class and the vector
       @a X is initialized by setting its essential entries to the boundary
       conditions and all other entries to zero (@a copy_interior == 0) or
       copied from @a x (@a copy_interior != 0).

       After solving the system `A(X)=B`, the (finite element) solution @a x can
       be recovered by calling Operator::RecoverFEMSolution() with the same
       vectors @a X, @a b, and @a x.

       @note The caller is responsible for destroying the output operator @a A!
       @note If there are no transformations, @a X simply reuses the data of @a
       x. */
   void FormLinearSystem(const Array<int> &ess_tdof_list,
                         Vector &x, Vector &b,
                         Operator* &A, Vector &X, Vector &B,
                         int copy_interior = 0);

   /** @brief Form a column-constrained linear system using a matrix-free approach.

       Form the operator linear system `A(X)=B` corresponding to the operator
       and the right-hand side @a b, by applying any necessary transformations
       such as: parallel assembly, conforming constraints for non-conforming AMR
       and eliminating boundary conditions.  @note Static condensation and
       hybridization are not supported for general operators (cf. the method
       MixedBilinearForm::FormRectangularLinearSystem())

       The constraints are specified through the input prolongation Pi from
       GetProlongation(), and output restriction Ro from GetOutputRestriction()
       methods, which are e.g. available through the (parallel) finite element
       spaces of any (parallel) mixed bilinear form operator. So we have:
       `A(X)=[Ro (*this) Pi](X)`, `B=Ro(b)`, and `X=Pi^T(x)`.

       The vector @a x must contain the essential boundary condition values.
       The "columns" in this operator corresponding to these values are
       eliminated through the RectangularConstrainedOperator class.

       After solving the system `A(X)=B`, the (finite element) solution @a x can
       be recovered by calling Operator::RecoverFEMSolution() with the same
       vectors @a X, @a b, and @a x.

       @note The caller is responsible for destroying the output operator @a A!
       @note If there are no transformations, @a X simply reuses the data of @a
       x. */
   void FormRectangularLinearSystem(const Array<int> &trial_tdof_list,
                                    const Array<int> &test_tdof_list,
                                    Vector &x, Vector &b,
                                    Operator* &A, Vector &X, Vector &B);

   /** @brief Reconstruct a solution vector @a x (e.g. a GridFunction) from the
       solution @a X of a constrained linear system obtained from
       Operator::FormLinearSystem() or Operator::FormRectangularLinearSystem().

       Call this method after solving a linear system constructed using
       Operator::FormLinearSystem() to recover the solution as an input vector,
       @a x, for this Operator (presumably a finite element grid function). This
       method has identical signature to the analogous method for bilinear
       forms, though currently @a b is not used in the implementation. */
   virtual void RecoverFEMSolution(const Vector &X, const Vector &b, Vector &x);

   /** @brief Return in @a A a parallel (on truedofs) version of this square
       operator.

       This returns the same operator as FormLinearSystem(), but does without
       the transformations of the right-hand side and initial guess. */
   void FormSystemOperator(const Array<int> &ess_tdof_list,
                           Operator* &A);

   /** @brief Return in @a A a parallel (on truedofs) version of this
       rectangular operator (including constraints).

       This returns the same operator as FormRectangularLinearSystem(), but does
       without the transformations of the right-hand side. */
   void FormRectangularSystemOperator(const Array<int> &trial_tdof_list,
                                      const Array<int> &test_tdof_list,
                                      Operator* &A);

   /** @brief Return in @a A a parallel (on truedofs) version of this
       rectangular operator.

       This is similar to FormSystemOperator(), but for dof-to-dof mappings
       (discrete linear operators), which can also correspond to rectangular
       matrices. The user should provide specializations of GetProlongation()
       for the input dofs and GetOutputRestriction() for the output dofs in
       their Operator implementation that are appropriate for the two spaces the
       Operator maps between. These are e.g. available through the (parallel)
       finite element space of any (parallel) bilinear form operator. We have:
       `A(X)=[Rout (*this) Pin](X)`. */
   void FormDiscreteOperator(Operator* &A);

   /// Prints operator with input size n and output size m in Matlab format.
   void PrintMatlab(std::ostream & out, int n = 0, int m = 0) const;

   /// Virtual destructor.
   virtual ~Operator() { }

   /// Enumeration defining IDs for some classes derived from Operator.
   /** This enumeration is primarily used with class OperatorHandle. */
   enum Type
   {
      ANY_TYPE,         ///< ID for the base class Operator, i.e. any type.
      MFEM_SPARSEMAT,   ///< ID for class SparseMatrix.
      Hypre_ParCSR,     ///< ID for class HypreParMatrix.
      PETSC_MATAIJ,     ///< ID for class PetscParMatrix, MATAIJ format.
      PETSC_MATIS,      ///< ID for class PetscParMatrix, MATIS format.
      PETSC_MATSHELL,   ///< ID for class PetscParMatrix, MATSHELL format.
      PETSC_MATNEST,    ///< ID for class PetscParMatrix, MATNEST format.
      PETSC_MATHYPRE,   ///< ID for class PetscParMatrix, MATHYPRE format.
      PETSC_MATGENERIC, ///< ID for class PetscParMatrix, unspecified format.
      Complex_Operator, ///< ID for class ComplexOperator.
      MFEM_ComplexSparseMat, ///< ID for class ComplexSparseMatrix.
      Complex_Hypre_ParCSR   ///< ID for class ComplexHypreParMatrix.
   };

   /// Return the type ID of the Operator class.
   /** This method is intentionally non-virtual, so that it returns the ID of
       the specific pointer or reference type used when calling this method. If
       not overridden by derived classes, they will automatically use the type ID
       of the base Operator class, ANY_TYPE. */
   Type GetType() const { return ANY_TYPE; }
};


/// Base abstract class for first order time dependent operators.
/** Operator of the form: (x,t) -> f(x,t), where k = f(x,t) generally solves the
    algebraic equation F(x,k,t) = G(x,t). The functions F and G represent the
    _implicit_ and _explicit_ parts of the operator, respectively. For explicit
    operators, F(x,k,t) = k, so f(x,t) = G(x,t). */
class TimeDependentOperator : public Operator
{
public:
   enum Type
   {
      EXPLICIT,   ///< This type assumes F(x,k,t) = k, i.e. k = f(x,t) = G(x,t).
      IMPLICIT,   ///< This is the most general type, no assumptions on F and G.
      HOMOGENEOUS ///< This type assumes that G(x,t) = 0.
   };

   /// Evaluation mode. See SetEvalMode() for details.
   enum EvalMode
   {
      /** Normal evaluation. */
      NORMAL,
      /** Assuming additive split, f(x,t) = f1(x,t) + f2(x,t), evaluate the
          first term, f1. */
      ADDITIVE_TERM_1,
      /** Assuming additive split, f(x,t) = f1(x,t) + f2(x,t), evaluate the
          second term, f2. */
      ADDITIVE_TERM_2
   };

protected:
   double t;  ///< Current time.
   Type type; ///< Describes the form of the TimeDependentOperator.
   EvalMode eval_mode; ///< Current evaluation mode.

public:
   /** @brief Construct a "square" TimeDependentOperator y = f(x,t), where x and
       y have the same dimension @a n. */
   explicit TimeDependentOperator(int n = 0, double t_ = 0.0,
                                  Type type_ = EXPLICIT)
      : Operator(n) { t = t_; type = type_; eval_mode = NORMAL; }

   /** @brief Construct a TimeDependentOperator y = f(x,t), where x and y have
       dimensions @a w and @a h, respectively. */
   TimeDependentOperator(int h, int w, double t_ = 0.0, Type type_ = EXPLICIT)
      : Operator(h, w) { t = t_; type = type_; eval_mode = NORMAL; }

   /// Read the currently set time.
   virtual double GetTime() const { return t; }

   /// Set the current time.
   virtual void SetTime(const double _t) { t = _t; }

   /// True if #type is #EXPLICIT.
   bool isExplicit() const { return (type == EXPLICIT); }
   /// True if #type is #IMPLICIT or #HOMOGENEOUS.
   bool isImplicit() const { return !isExplicit(); }
   /// True if #type is #HOMOGENEOUS.
   bool isHomogeneous() const { return (type == HOMOGENEOUS); }

   /// Return the current evaluation mode. See SetEvalMode() for details.
   EvalMode GetEvalMode() const { return eval_mode; }

   /// Set the evaluation mode of the time-dependent operator.
   /** The evaluation mode is a switch that allows time-stepping methods to
       request evaluation of separate components/terms of the time-dependent
       operator. For example, IMEX methods typically assume additive split of
       the operator: f(x,t) = f1(x,t) + f2(x,t) and they rely on the ability to
       evaluate the two terms separately.

       Generally, setting the evaluation mode should affect the behavior of all
       evaluation-related methods in the class, such as Mult(), ImplicitSolve(),
       etc. However, the exact list of methods that need to support a specific
       mode will depend on the used time-stepping method. */
   virtual void SetEvalMode(const EvalMode new_eval_mode)
   { eval_mode = new_eval_mode; }

   /** @brief Perform the action of the explicit part of the operator, G:
       @a y = G(@a x, t) where t is the current time.

       Presently, this method is used by some PETSc ODE solvers, for more
       details, see the PETSc Manual. */
   virtual void ExplicitMult(const Vector &x, Vector &y) const;

   /** @brief Perform the action of the implicit part of the operator, F:
       @a y = F(@a x, @a k, t) where t is the current time.

       Presently, this method is used by some PETSc ODE solvers, for more
       details, see the PETSc Manual.*/
   virtual void ImplicitMult(const Vector &x, const Vector &k, Vector &y) const;

   /** @brief Perform the action of the operator: @a y = k = f(@a x, t), where
       k solves the algebraic equation F(@a x, k, t) = G(@a x, t) and t is the
       current time. */
   virtual void Mult(const Vector &x, Vector &y) const;

   /** @brief Solve the equation: @a k = f(@a x + @a dt @a k, t), for the
       unknown @a k at the current time t.

       For general F and G, the equation for @a k becomes:
       F(@a x + @a dt @a k, @a k, t) = G(@a x + @a dt @a k, t).

       The input vector @a x corresponds to time index (or cycle) n, while the
       currently set time, #t, and the result vector @a k correspond to time
       index n+1. The time step @a dt corresponds to the time interval between
       cycles n and n+1.

       This method allows for the abstract implementation of some time
       integration methods, including diagonal implicit Runge-Kutta (DIRK)
       methods and the backward Euler method in particular.

       If not re-implemented, this method simply generates an error. */
   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k);

   /** @brief Return an Operator representing (dF/dk @a shift + dF/dx) at the
       given @a x, @a k, and the currently set time.

       Presently, this method is used by some PETSc ODE solvers, for more
       details, see the PETSc Manual. */
   virtual Operator& GetImplicitGradient(const Vector &x, const Vector &k,
                                         double shift) const;

   /** @brief Return an Operator representing dG/dx at the given point @a x and
       the currently set time.

       Presently, this method is used by some PETSc ODE solvers, for more
       details, see the PETSc Manual. */
   virtual Operator& GetExplicitGradient(const Vector &x) const;

   /** @brief Setup the ODE linear system \f$ A(x,t) = (I - gamma J) \f$ or
       \f$ A = (M - gamma J) \f$, where \f$ J(x,t) = \frac{df}{dt(x,t)} \f$.

       @param[in]  x     The state at which \f$A(x,t)\f$ should be evaluated.
       @param[in]  fx    The current value of the ODE rhs function, \f$f(x,t)\f$.
       @param[in]  jok   Flag indicating if the Jacobian should be updated.
       @param[out] jcur  Flag to signal if the Jacobian was updated.
       @param[in]  gamma The scaled time step value.

       If not re-implemented, this method simply generates an error.

       Presently, this method is used by SUNDIALS ODE solvers, for more
       details, see the SUNDIALS User Guides. */
   virtual int SUNImplicitSetup(const Vector &x, const Vector &fx,
                                int jok, int *jcur, double gamma);

   /** @brief Solve the ODE linear system \f$ A x = b \f$ as setup by
       the method SUNImplicitSetup().

       @param[in]      b   The linear system right-hand side.
       @param[in,out]  x   On input, the initial guess. On output, the solution.
       @param[in]      tol Linear solve tolerance.

       If not re-implemented, this method simply generates an error.

       Presently, this method is used by SUNDIALS ODE solvers, for more
       details, see the SUNDIALS User Guides. */
   virtual int SUNImplicitSolve(const Vector &b, Vector &x, double tol);

   /** @brief Setup the mass matrix in the ODE system \f$ M y' = f(y,t) \f$ .

       If not re-implemented, this method simply generates an error.

       Presently, this method is used by SUNDIALS ARKStep integrator, for more
       details, see the ARKode User Guide. */
   virtual int SUNMassSetup();

   /** @brief Solve the mass matrix linear system \f$ M x = b \f$
       as setup by the method SUNMassSetup().

       @param[in]      b   The linear system right-hand side.
       @param[in,out]  x   On input, the initial guess. On output, the solution.
       @param[in]      tol Linear solve tolerance.

       If not re-implemented, this method simply generates an error.

       Presently, this method is used by SUNDIALS ARKStep integrator, for more
       details, see the ARKode User Guide. */
   virtual int SUNMassSolve(const Vector &b, Vector &x, double tol);

   /** @brief Compute the mass matrix-vector product \f$ v = M x \f$ .

       @param[in]   x The vector to multiply.
       @param[out]  v The result of the matrix-vector product.

       If not re-implemented, this method simply generates an error.

       Presently, this method is used by SUNDIALS ARKStep integrator, for more
       details, see the ARKode User Guide. */
   virtual int SUNMassMult(const Vector &x, Vector &v);

   virtual ~TimeDependentOperator() { }
};


/** TimeDependentAdjointOperator is a TimeDependentOperator with Adjoint rate
    equations to be used with CVODESSolver. */
class TimeDependentAdjointOperator : public TimeDependentOperator
{
public:

   /**
      \brief The TimedependentAdjointOperator extends the TimeDependentOperator
      class to use features in SUNDIALS CVODESSolver for computing quadratures
      and solving adjoint problems.

      To solve adjoint problems one needs to implement the AdjointRateMult
      method to tell CVODES what the adjoint rate equation is.

      QuadratureIntegration (optional) can be used to compute values over the
      forward problem

      QuadratureSensitivityMult (optional) can be used to find the sensitivity
      of the quadrature using the adjoint solution in part.

      SUNImplicitSetupB (optional) can be used to setup custom solvers for the
      newton solve for the adjoint problem.

      SUNImplicitSolveB (optional) actually uses the solvers from
      SUNImplicitSetupB to solve the adjoint problem.

      See SUNDIALS user manuals for specifics.

      \param[in] dim Dimension of the forward operator
      \param[in] adjdim Dimension of the adjoint operator. Typically it is the
      same size as dim. However, SUNDIALS allows users to specify the size if
      one wants to perform custom operations.
      \param[in] t Starting time to set
      \param[in] type The TimeDependentOperator type
   */
   TimeDependentAdjointOperator(int dim, int adjdim, double t = 0.,
                                Type type = EXPLICIT) :
      TimeDependentOperator(dim, t, type),
      adjoint_height(adjdim)
   {}

   /// Destructor
   virtual ~TimeDependentAdjointOperator() {};

   /**
      \brief Provide the operator integration of a quadrature equation

      \param[in] y The current value at time t
      \param[out] qdot The current quadrature rate value at t
   */
   virtual void QuadratureIntegration(const Vector &y, Vector &qdot) const {};

   /** @brief Perform the action of the operator:
       @a yBdot = k = f(@a y,@2 yB, t), where

       @param[in] y The primal solution at time t
       @param[in] yB The adjoint solution at time t
       @param[out] yBdot the rate at time t
   */
   virtual void AdjointRateMult(const Vector &y, Vector & yB,
                                Vector &yBdot) const = 0;

   /**
      \brief Provides the sensitivity of the quadrature w.r.t to primal and
      adjoint solutions

      \param[in] y the value of the primal solution at time t
      \param[in] yB the value of the adjoint solution at time t
      \param[out] qBdot the value of the sensitivity of the quadrature rate at
      time t
   */
   virtual void QuadratureSensitivityMult(const Vector &y, const Vector &yB,
                                          Vector &qBdot) const {}

   /** @brief Setup the ODE linear system \f$ A(x,t) = (I - gamma J) \f$ or
       \f$ A = (M - gamma J) \f$, where \f$ J(x,t) = \frac{df}{dt(x,t)} \f$.

       @param[in]  t     The current time
       @param[in]  x     The state at which \f$A(x,xB,t)\f$ should be evaluated.
       @param[in]  xB    The state at which \f$A(x,xB,t)\f$ should be evaluated.
       @param[in]  fxB   The current value of the ODE rhs function, \f$f(x,t)\f$.
       @param[in]  jokB   Flag indicating if the Jacobian should be updated.
       @param[out] jcurB  Flag to signal if the Jacobian was updated.
       @param[in]  gammaB The scaled time step value.

       If not re-implemented, this method simply generates an error.

       Presently, this method is used by SUNDIALS ODE solvers, for more details,
       see the SUNDIALS User Guides.
   */
   virtual int SUNImplicitSetupB(const double t, const Vector &x,
                                 const Vector &xB, const Vector &fxB,
                                 int jokB, int *jcurB, double gammaB)
   {
      mfem_error("TimeDependentAdjointOperator::SUNImplicitSetupB() is not "
                 "overridden!");
      return (-1);
   }

   /** @brief Solve the ODE linear system \f$ A(x,xB,t) xB = b \f$ as setup by
       the method SUNImplicitSetup().

       @param[in]      b   The linear system right-hand side.
       @param[in,out]  x   On input, the initial guess. On output, the solution.
       @param[in]      tol Linear solve tolerance.

       If not re-implemented, this method simply generates an error.

       Presently, this method is used by SUNDIALS ODE solvers, for more details,
       see the SUNDIALS User Guides. */
   virtual int SUNImplicitSolveB(Vector &x, const Vector &b, double tol)
   {
      mfem_error("TimeDependentAdjointOperator::SUNImplicitSolveB() is not "
                 "overridden!");
      return (-1);
   }

   /// Returns the size of the adjoint problem state space
   int GetAdjointHeight() {return adjoint_height;}

protected:
   int adjoint_height; /// Size of the adjoint problem
};


/// Base abstract class for second order time dependent operators.
/** Operator of the form: (x,dxdt,t) -> f(x,dxdt,t), where k = f(x,dxdt,t)
    generally solves the algebraic equation F(x,dxdt,k,t) = G(x,dxdt,t).
    The functions F and G represent the_implicit_ and _explicit_ parts of
    the operator, respectively. For explicit operators,
    F(x,dxdt,k,t) = k, so f(x,dxdt,t) = G(x,dxdt,t). */
class SecondOrderTimeDependentOperator : public TimeDependentOperator
{
public:
   /** @brief Construct a "square" SecondOrderTimeDependentOperator
       y = f(x,dxdt,t), where x, dxdt and y have the same dimension @a n. */
   explicit SecondOrderTimeDependentOperator(int n = 0, double t_ = 0.0,
                                             Type type_ = EXPLICIT)
      : TimeDependentOperator(n, t_,type_) { }

   /** @brief Construct a SecondOrderTimeDependentOperator y = f(x,dxdt,t),
       where x, dxdt and y have the same dimension @a n. */
   SecondOrderTimeDependentOperator(int h, int w, double t_ = 0.0,
                                    Type type_ = EXPLICIT)
      : TimeDependentOperator(h, w, t_,type_) { }

   using TimeDependentOperator::Mult;

   /** @brief Perform the action of the operator: @a y = k = f(@a x,@ dxdt, t),
       where k solves the algebraic equation
       F(@a x,@ dxdt, k, t) = G(@a x,@ dxdt, t) and t is the current time. */
   virtual void Mult(const Vector &x, const Vector &dxdt, Vector &y) const;

   using TimeDependentOperator::ImplicitSolve;
   /** @brief Solve the equation:
       @a k = f(@a x + 1/2 @a dt0^2 @a k, @a dxdt + @a dt1 @a k, t), for the
       unknown @a k at the current time t.

       For general F and G, the equation for @a k becomes:
       F(@a x + 1/2 @a dt0^2 @a k, @a dxdt + @a dt1 @a k, t)
                        = G(@a x + 1/2 @a dt0^2 @a k, @a dxdt + @a dt1 @a k, t).

       The input vector @a x corresponds to time index (or cycle) n, while the
       currently set time, #t, and the result vector @a k correspond to time
       index n+1. The time step @a dt corresponds to the time interval between
       cycles n and n+1.

       This method allows for the abstract implementation of some time
       integration methods.

       If not re-implemented, this method simply generates an error. */
   virtual void ImplicitSolve(const double dt0, const double dt1,
                              const Vector &x, const Vector &dxdt, Vector &k);


   virtual ~SecondOrderTimeDependentOperator() { }
};


/// Base class for solvers
class Solver : public Operator
{
public:
   /// If true, use the second argument of Mult() as an initial guess.
   bool iterative_mode;

   /** @brief Initialize a square Solver with size @a s.

       @warning Use a Boolean expression for the second parameter (not an int)
       to distinguish this call from the general rectangular constructor. */
   explicit Solver(int s = 0, bool iter_mode = false)
      : Operator(s) { iterative_mode = iter_mode; }

   /// Initialize a Solver with height @a h and width @a w.
   Solver(int h, int w, bool iter_mode = false)
      : Operator(h, w) { iterative_mode = iter_mode; }

   /// Set/update the solver for the given operator.
   virtual void SetOperator(const Operator &op) = 0;
};


/// Identity Operator I: x -> x.
class IdentityOperator : public Operator
{
public:
   /// Create an identity operator of size @a n.
   explicit IdentityOperator(int n) : Operator(n) { }

   /// Operator application
   virtual void Mult(const Vector &x, Vector &y) const { y = x; }

   /// Application of the transpose
   virtual void MultTranspose(const Vector &x, Vector &y) const { y = x; }
};

/// Returns true if P is the identity prolongation, i.e. if it is either NULL or
/// an IdentityOperator.
inline bool IsIdentityProlongation(const Operator *P)
{
   return !P || dynamic_cast<const IdentityOperator*>(P);
}

/// Scaled Operator B: x -> a A(x).
class ScaledOperator : public Operator
{
private:
   const Operator &A_;
   double a_;

public:
   /// Create an operator which is a scalar multiple of A.
   explicit ScaledOperator(const Operator *A, double a)
      : Operator(A->Width(), A->Height()), A_(*A), a_(a) { }

   /// Operator application
   virtual void Mult(const Vector &x, Vector &y) const
   { A_.Mult(x, y); y *= a_; }
};


/** @brief The transpose of a given operator. Switches the roles of the methods
    Mult() and MultTranspose(). */
class TransposeOperator : public Operator
{
private:
   const Operator &A;

public:
   /// Construct the transpose of a given operator @a *a.
   TransposeOperator(const Operator *a)
      : Operator(a->Width(), a->Height()), A(*a) { }

   /// Construct the transpose of a given operator @a a.
   TransposeOperator(const Operator &a)
      : Operator(a.Width(), a.Height()), A(a) { }

   /// Operator application. Apply the transpose of the original Operator.
   virtual void Mult(const Vector &x, Vector &y) const
   { A.MultTranspose(x, y); }

   /// Application of the transpose. Apply the original Operator.
   virtual void MultTranspose(const Vector &x, Vector &y) const
   { A.Mult(x, y); }
};


/// General product operator: x -> (A*B)(x) = A(B(x)).
class ProductOperator : public Operator
{
   const Operator *A, *B;
   bool ownA, ownB;
   mutable Vector z;

public:
   ProductOperator(const Operator *A, const Operator *B, bool ownA, bool ownB);

   virtual void Mult(const Vector &x, Vector &y) const
   { B->Mult(x, z); A->Mult(z, y); }

   virtual void MultTranspose(const Vector &x, Vector &y) const
   { A->MultTranspose(x, z); B->MultTranspose(z, y); }

   virtual ~ProductOperator();
};


/// The operator x -> R*A*P*x constructed through the actions of R^T, A and P
class RAPOperator : public Operator
{
private:
   const Operator & Rt;
   const Operator & A;
   const Operator & P;
   mutable Vector Px;
   mutable Vector APx;
   MemoryClass mem_class;

public:
   /// Construct the RAP operator given R^T, A and P.
   RAPOperator(const Operator &Rt_, const Operator &A_, const Operator &P_);

   virtual MemoryClass GetMemoryClass() const { return mem_class; }

   /// Operator application.
   virtual void Mult(const Vector & x, Vector & y) const
   { P.Mult(x, Px); A.Mult(Px, APx); Rt.MultTranspose(APx, y); }

   /// Application of the transpose.
   virtual void MultTranspose(const Vector & x, Vector & y) const
   { Rt.Mult(x, APx); A.MultTranspose(APx, Px); P.MultTranspose(Px, y); }
};


/// General triple product operator x -> A*B*C*x, with ownership of the factors.
class TripleProductOperator : public Operator
{
   const Operator *A;
   const Operator *B;
   const Operator *C;
   bool ownA, ownB, ownC;
   mutable Vector t1, t2;
   MemoryClass mem_class;

public:
   TripleProductOperator(const Operator *A, const Operator *B,
                         const Operator *C, bool ownA, bool ownB, bool ownC);

   virtual MemoryClass GetMemoryClass() const { return mem_class; }

   virtual void Mult(const Vector &x, Vector &y) const
   { C->Mult(x, t1); B->Mult(t1, t2); A->Mult(t2, y); }

   virtual void MultTranspose(const Vector &x, Vector &y) const
   { A->MultTranspose(x, t2); B->MultTranspose(t2, t1); C->MultTranspose(t1, y); }

   virtual ~TripleProductOperator();
};


/** @brief Square Operator for imposing essential boundary conditions using only
    the action, Mult(), of a given unconstrained Operator.

    Square operator constrained by fixing certain entries in the solution to
    given "essential boundary condition" values. This class is used by the
    general, matrix-free system formulation of Operator::FormLinearSystem. */
class ConstrainedOperator : public Operator
{
protected:
   Array<int> constraint_list;  ///< List of constrained indices/dofs.
   Operator *A;                 ///< The unconstrained Operator.
   bool own_A;                  ///< Ownership flag for A.
   mutable Vector z, w;         ///< Auxiliary vectors.
   MemoryClass mem_class;
   DiagonalPolicy diag_policy;  ///< Diagonal policy for constrained dofs

public:
   /** @brief Constructor from a general Operator and a list of essential
       indices/dofs.

       Specify the unconstrained operator @a *A and a @a list of indices to
       constrain, i.e. each entry @a list[i] represents an essential dof. If the
       ownership flag @a own_A is true, the operator @a *A will be destroyed
       when this object is destroyed. The @a diag_policy determines how the
       operator sets entries corresponding to essential dofs. */
   ConstrainedOperator(Operator *A, const Array<int> &list, bool own_A = false,
                       DiagonalPolicy diag_policy = DIAG_ONE);

   /// Returns the type of memory in which the solution and temporaries are stored.
   virtual MemoryClass GetMemoryClass() const { return mem_class; }

   /// Set the diagonal policy for the constrained operator.
   void SetDiagonalPolicy(const DiagonalPolicy _diag_policy)
   { diag_policy = _diag_policy; }

   /** @brief Eliminate "essential boundary condition" values specified in @a x
       from the given right-hand side @a b.

       Performs the following steps:

           z = A((0,x_b));  b_i -= z_i;  b_b = x_b;

       where the "_b" subscripts denote the essential (boundary) indices/dofs of
       the vectors, and "_i" -- the rest of the entries. */
   void EliminateRHS(const Vector &x, Vector &b) const;

   /** @brief Constrained operator action.

       Performs the following steps:

           z = A((x_i,0));  y_i = z_i;  y_b = x_b;

       where the "_b" subscripts denote the essential (boundary) indices/dofs of
       the vectors, and "_i" -- the rest of the entries. */
   virtual void Mult(const Vector &x, Vector &y) const;

   /// Destructor: destroys the unconstrained Operator, if owned.
   virtual ~ConstrainedOperator() { if (own_A) { delete A; } }
};

/** @brief Rectangular Operator for imposing essential boundary conditions on
    the input space using only the action, Mult(), of a given unconstrained
    Operator.

    Rectangular operator constrained by fixing certain entries in the solution
    to given "essential boundary condition" values. This class is used by the
    general matrix-free formulation of Operator::FormRectangularLinearSystem. */
class RectangularConstrainedOperator : public Operator
{
protected:
   Array<int> trial_constraints, test_constraints;
   Operator *A;
   bool own_A;
   mutable Vector z, w;
   MemoryClass mem_class;

public:
   /** @brief Constructor from a general Operator and a list of essential
       indices/dofs.

       Specify the unconstrained operator @a *A and two lists of indices to
       constrain, i.e. each entry @a trial_list[i] represents an essential trial
       dof. If the ownership flag @a own_A is true, the operator @a *A will be
       destroyed when this object is destroyed. */
   RectangularConstrainedOperator(Operator *A, const Array<int> &trial_list,
                                  const Array<int> &test_list, bool own_A = false);
   /// Returns the type of memory in which the solution and temporaries are stored.
   virtual MemoryClass GetMemoryClass() const { return mem_class; }
   /** @brief Eliminate columns corresponding to "essential boundary condition"
       values specified in @a x from the given right-hand side @a b.

       Performs the following steps:

           b -= A((0,x_b));
           b_j = 0

       where the "_b" subscripts denote the essential (boundary) indices and the
       "_j" subscript denotes the essential test indices */
   void EliminateRHS(const Vector &x, Vector &b) const;
   /** @brief Rectangular-constrained operator action.

       Performs the following steps:

           y = A((x_i,0));
           y_j = 0

       where the "_i" subscripts denote all the nonessential (boundary) trial
       indices and the "_j" subscript denotes the essential test indices */
   virtual void Mult(const Vector &x, Vector &y) const;
   virtual void MultTranspose(const Vector &x, Vector &y) const;
   virtual ~RectangularConstrainedOperator() { if (own_A) { delete A; } }
};

/** @brief PowerMethod helper class to estimate the largest eigenvalue of an
           operator using the iterative power method. */
class PowerMethod
{
   Vector v1;
#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif

public:

#ifdef MFEM_USE_MPI
   PowerMethod() : comm(MPI_COMM_NULL) {}
#else
   PowerMethod() {}
#endif

#ifdef MFEM_USE_MPI
   PowerMethod(MPI_Comm _comm) : comm(_comm) {}
#endif

   /// @brief Returns an estimate of the largest eigenvalue of the operator \p opr
   /// using the iterative power method.
   /** \p v0 is being used as the vector for the iterative process and will contain
       the eigenvector corresponding to the largest eigenvalue after convergence.
       The maximum number of iterations may set with \p numSteps, the relative
       tolerance with \p tolerance and the seed of the random initialization of
       \p v0 with \p seed. */
   double EstimateLargestEigenvalue(Operator& opr, Vector& v0,
                                    int numSteps = 10, double tolerance = 1e-8,
                                    int seed = 12345);
};

}

#endif
