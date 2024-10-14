// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
   /** @note Uses DiagonalPolicy::DIAG_ONE. */
   void FormConstrainedSystemOperator(
      const Array<int> &ess_tdof_list, ConstrainedOperator* &Aout);

   /// see FormRectangularSystemOperator()
   void FormRectangularConstrainedSystemOperator(
      const Array<int> &trial_tdof_list,
      const Array<int> &test_tdof_list,
      RectangularConstrainedOperator* &Aout);

   /** @brief Returns RAP Operator of this, using input/output Prolongation matrices
       @a Pi corresponds to "P", @a Po corresponds to "Rt" */
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
                     Vector &x, Vector &b, Vector &X, Vector &B) const;

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

       For example, classes using the mfem::forall macro for implementation can
       return the value returned by Device::GetMemoryClass().

       The default implementation of this method in class Operator returns
       MemoryClass::HOST. */
   virtual MemoryClass GetMemoryClass() const { return MemoryClass::HOST; }

   /// Operator application: `y=A(x)`.
   virtual void Mult(const Vector &x, Vector &y) const = 0;

   /** @brief Action of the transpose operator: `y=A^t(x)`. The default behavior
       in class Operator is to generate an error. */
   virtual void MultTranspose(const Vector &x, Vector &y) const
   { mfem_error("Operator::MultTranspose() is not overridden!"); }

   /// Operator application: `y+=A(x)` (default) or `y+=a*A(x)`.
   virtual void AddMult(const Vector &x, Vector &y, const real_t a = 1.0) const;

   /// Operator transpose application: `y+=A^t(x)` (default) or `y+=a*A^t(x)`.
   virtual void AddMultTranspose(const Vector &x, Vector &y,
                                 const real_t a = 1.0) const;

   /// Operator application on a matrix: `Y=A(X)`.
   virtual void ArrayMult(const Array<const Vector *> &X,
                          Array<Vector *> &Y) const;

   /// Action of the transpose operator on a matrix: `Y=A^t(X)`.
   virtual void ArrayMultTranspose(const Array<const Vector *> &X,
                                   Array<Vector *> &Y) const;

   /// Operator application on a matrix: `Y+=A(X)` (default) or `Y+=a*A(X)`.
   virtual void ArrayAddMult(const Array<const Vector *> &X, Array<Vector *> &Y,
                             const real_t a = 1.0) const;

   /** @brief Operator transpose application on a matrix: `Y+=A^t(X)` (default)
       or `Y+=a*A^t(X)`. */
   virtual void ArrayAddMultTranspose(const Array<const Vector *> &X,
                                      Array<Vector *> &Y, const real_t a = 1.0) const;

   /** @brief Evaluate the gradient operator at the point @a x. The default
       behavior in class Operator is to generate an error. */
   virtual Operator &GetGradient(const Vector &x) const
   {
      mfem_error("Operator::GetGradient() is not overridden!");
      return const_cast<Operator &>(*this);
   }

   /** @brief Computes the diagonal entries into @a diag. Typically, this
       operation only makes sense for linear Operator%s. In some cases, only an
       approximation of the diagonal is computed. */
   virtual void AssembleDiagonal(Vector &diag) const
   {
      MFEM_CONTRACT_VAR(diag);
      MFEM_ABORT("Not relevant or not implemented for this Operator.");
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

   /** @brief Transpose of GetOutputRestriction, directly available in this
       form to facilitate matrix-free RAP-type operators.

       `NULL` means identity. */
   virtual const Operator *GetOutputRestrictionTranspose() const { return NULL; }

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
   void PrintMatlab(std::ostream & out, int n, int m = 0) const;

   /// Prints operator in Matlab format.
   virtual void PrintMatlab(std::ostream & out) const;

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
      Complex_Hypre_ParCSR,   ///< ID for class ComplexHypreParMatrix.
      Complex_DenseMat,  ///< ID for class ComplexDenseMatrix
      MFEM_Block_Matrix,     ///< ID for class BlockMatrix.
      MFEM_Block_Operator   ///< ID for the base class BlockOperator.
   };

   /// Return the type ID of the Operator class.
   /** This method is intentionally non-virtual, so that it returns the ID of
       the specific pointer or reference type used when calling this method. If
       not overridden by derived classes, they will automatically use the type ID
       of the base Operator class, ANY_TYPE. */
   Type GetType() const { return ANY_TYPE; }
};


/// Base abstract class for first order time dependent operators.
/** Operator of the form: (u,t) -> k(u,t), where k generally solves the
    algebraic equation F(u,k,t) = G(u,t). The functions F and G represent the
    _implicit_ and _explicit_ parts of the operator, respectively.

    A common use for this class is representing a differential algebraic
    equation of the form $ F(y,\frac{dy}{dt},t) = G(y,t) $.

    For example, consider an ordinary differential equation of the form
    $ M \frac{dy}{dt} = g(y,t) $. There are various ways of expressing this ODE
    as a TimeDependentOperator depending on the choices for F and G. Here are
    some common choices:

      1. F(u,k,t) = k and G(u,t) = inv(M) g(u,t),
      2. F(u,k,t) = M k and G(u,t) = g(u,t),
      3. F(u,k,t) = M k - g(u,t) and G(u,t) = 0.

    Note that depending on the ODE solver, some of the above choices may be
    preferable to the others.
*/
class TimeDependentOperator : public Operator
{
public:
   /// Enum used to describe the form of the time-dependent operator.
   /** The type should be set by classes derived from TimeDependentOperator to
       describe the form, in terms of the functions F and G, used by the
       specific derived class. This information can be queried by classes or
       functions (like time stepping algorithms) to make choices about the
       algorithm to use, or to ensure that the TimeDependentOperator uses the
       form expected by the class/function.

       For example, assume that a derived class is implementing the ODE
       $M \frac{dy}{dt} = g(y,t)$ and chooses to define $F(u,k,t) = M k$ and
       $G(u,t) = g(u,t)$. Then it cannot use type EXPLICIT, unless $M = I$, or
       type HOMOGENEOUS, unless $g(u,t) = 0$. If, on the other hand, the derived
       class chooses to define $F(u,k,t) = k$ and $G(u,t) = M^{-1} g(y,t)$, then
       the natural choice is to set the type to EXPLICIT, even though setting it
       to IMPLICIT is also not wrong -- doing so will simply fail to inform
       methods that query this information that it uses a more specific
       implementation, EXPLICIT, that may allow the use of algorithms that
       support only the EXPLICIT type. */
   enum Type
   {
      EXPLICIT,   ///< This type assumes F(u,k,t) = k.
      IMPLICIT,   ///< This is the most general type, no assumptions on F and G.
      HOMOGENEOUS ///< This type assumes that G(u,t) = 0.
   };

   /// Evaluation mode. See SetEvalMode() for details.
   enum EvalMode
   {
      /** Normal evaluation. */
      NORMAL,
      /** Assuming additive split, k(u,t) = k1(u,t) + k2(u,t), evaluate the
          first term, k1. */
      ADDITIVE_TERM_1,
      /** Assuming additive split, k(u,t) = k1(u,t) + k2(u,t), evaluate the
          second term, k2. */
      ADDITIVE_TERM_2
   };

protected:
   real_t t;  ///< Current time.
   Type type; /**< @brief Describes the form of the TimeDependentOperator, see
                   the documentation of #Type. */
   EvalMode eval_mode; ///< Current evaluation mode.

public:
   /** @brief Construct a "square" TimeDependentOperator (u,t) -> k(u,t), where
       u and k have the same dimension @a n. */
   explicit TimeDependentOperator(int n = 0, real_t t_ = 0.0,
                                  Type type_ = EXPLICIT)
      : Operator(n) { t = t_; type = type_; eval_mode = NORMAL; }

   /** @brief Construct a TimeDependentOperator (u,t) -> k(u,t), where u and k
       have dimensions @a w and @a h, respectively. */
   TimeDependentOperator(int h, int w, double t_ = 0.0, Type type_ = EXPLICIT)
      : Operator(h, w) { t = t_; type = type_; eval_mode = NORMAL; }

   /// Read the currently set time.
   virtual real_t GetTime() const { return t; }

   /// Set the current time.
   virtual void SetTime(const real_t t_) { t = t_; }

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
       the operator: k(u,t) = k1(u,t) + k2(u,t) and they rely on the ability to
       evaluate the two terms separately.

       Generally, setting the evaluation mode should affect the behavior of all
       evaluation-related methods in the class, such as Mult(), ImplicitSolve(),
       etc. However, the exact list of methods that need to support a specific
       mode will depend on the used time-stepping method. */
   virtual void SetEvalMode(const EvalMode new_eval_mode)
   { eval_mode = new_eval_mode; }

   /** @brief Perform the action of the explicit part of the operator, G:
       @a v = G(@a u, t) where t is the current time.

       Presently, this method is used by some PETSc ODE solvers and the
       SUNDIALS ARKStep integrator, for more details, see either the PETSc
       Manual or the ARKode User Guide, respectively. */
   virtual void ExplicitMult(const Vector &u, Vector &v) const;

   /** @brief Perform the action of the implicit part of the operator, F:
       @a v = F(@a u, @a k, t) where t is the current time.

       Presently, this method is used by some PETSc ODE solvers, for more
       details, see the PETSc Manual.*/
   virtual void ImplicitMult(const Vector &u, const Vector &k, Vector &v) const;

   /** @brief Perform the action of the operator (u,t) -> k(u,t) where t is the
       current time set by SetTime() and @a k satisfies
       F(@a u, @a k, t) = G(@a u, t).

       For solving an ordinary differential equation of the form
       $ M \frac{dy}{dt} = g(y,t) $, recall that F and G can be defined in
       various ways, e.g.:

         1. F(u,k,t) = k and G(u,t) = inv(M) g(u,t)
         2. F(u,k,t) = M k and G(u,t) = g(u,t)
         3. F(u,k,t) = M k - g(u,t) and G(u,t) = 0.

       Regardless of the choice of F and G, this function should always compute
       @a k = inv(M) g(@a u, t). */
   void Mult(const Vector &u, Vector &k) const override;

   /** @brief Solve for the unknown @a k, at the current time t, the following
       equation:
       F(@a u + @a gamma @a k, @a k, t) = G(@a u + @a gamma @a k, t).

       For solving an ordinary differential equation of the form
       $ M \frac{dy}{dt} = g(y,t) $, recall that F and G can be defined in
       various ways, e.g.:

         1. F(u,k,t) = k and G(u,t) = inv(M) g(u,t)
         2. F(u,k,t) = M k and G(u,t) = g(u,t)
         3. F(u,k,t) = M k - g(u,t) and G(u,t) = 0

       Regardless of the choice of F and G, this function should solve for @a k
       in M @a k = g(@a u + @a gamma @a k, t).

       To see how @a k can be useful, consider the backward Euler method defined
       by $ y(t + \Delta t) = y(t) + \Delta t k_0 $ where
       $ M k_0 = g \big( y(t) + \Delta t k_0, t + \Delta t \big) $. A backward
       Euler integrator can use @a k from this function for $k_0$, with the call
       using @a u set to $ y(t) $, @a gamma set to $ \Delta t$, and time set to
       $t + \Delta t$. See class BackwardEulerSolver.

       Generalizing further, consider a diagonally implicit Runge-Kutta (DIRK)
       method defined by
       $ y(t + \Delta t) = y(t) + \Delta t \sum_{i=1}^s b_i k_i $ where
       $ M k_i = g \big( y(t) + \Delta t \sum_{j=1}^i a_{ij} k_j,
                         t + c_i \Delta t \big) $.
       A DIRK integrator can use @a k from this function, with @a u set to
       $ y(t) + \Delta t \sum_{j=1}^{i-1} a_{ij} k_j $ and @a gamma set to
       $ a_{ii} \Delta t $, for $ k_i $. For example, see class SDIRK33Solver.

       If not re-implemented, this method simply generates an error. */
   virtual void ImplicitSolve(const real_t gamma, const Vector &u, Vector &k);

   /** @brief Return an Operator representing (dF/dk @a shift + dF/du) at the
       given @a u, @a k, and the currently set time.

       Presently, this method is used by some PETSc ODE solvers, for more
       details, see the PETSc Manual. */
   virtual Operator& GetImplicitGradient(const Vector &u, const Vector &k,
                                         real_t shift) const;

   /** @brief Return an Operator representing dG/du at the given point @a u and
       the currently set time.

       Presently, this method is used by some PETSc ODE solvers, for more
       details, see the PETSc Manual. */
   virtual Operator& GetExplicitGradient(const Vector &u) const;

   /** @brief Setup a linear system as needed by some SUNDIALS ODE solvers to
       perform a similar action to ImplicitSolve, i.e., solve for k, at the
       current time t, in F(u + gamma k, k, t) = G(u + gamma k, t).

       The SUNDIALS ODE solvers iteratively solve for k, as knew = kold + dk.
       The linear system here is for dk, obtained by linearizing the nonlinear
       system F(u + gamma knew, knew, t) = G(u + gamma knew, t) about dk = 0:
          F(u + gamma (kold + dk), kold + dk, t) = G(u + gamma (kold + dk), t)
          => [dF/dk + gamma (dF/du - dG/du)] dk = G - F + O(dk^2)
       In other words, the linear system to be setup here is A dk = r, where
       A = [dF/dk + gamma (dF/du - dG/du)] and r = G - F.

       For solving an ordinary differential equation of the form
       $ M \frac{dy}{dt} = g(y,t) $, recall that F and G can be defined as one
       of the following:

         1. F(u,k,t) = k and G(u,t) = inv(M) g(u,t)
         2. F(u,k,t) = M k and G(u,t) = g(u,t)
         3. F(u,k,t) = M k - g(u,t) and G(u,t) = 0

       This function performs setup to solve $ A dk = r $ where A is either

         1. A(@a y,t) = I - @a gamma inv(M) J(@a y,t)
         2. A(@a y,t) = M - @a gamma J(@a y,t)
         3. A(@a y,t) = M - @a gamma J(@a y,t)

       with J = dg/dy (or a reasonable approximation thereof).

       @param[in]  y     The state at which A(@a y,t) should be evaluated.
       @param[in]  v     The value of inv(M) g(y,t) for 1 or g(y,t) for 2 & 3.
       @param[in]  jok   Flag indicating if the Jacobian should be updated.
       @param[out] jcur  Flag to signal if the Jacobian was updated.
       @param[in]  gamma The scaled time step value.

       If not re-implemented, this method simply generates an error.

       Presently, this method is used by SUNDIALS ODE solvers, for more
       details, see the SUNDIALS User Guides. */
   virtual int SUNImplicitSetup(const Vector &y, const Vector &v,
                                int jok, int *jcur, real_t gamma);

   /** @brief Solve the ODE linear system A @a dk = @a r , where A and r are
       defined by the method SUNImplicitSetup().

       For solving an ordinary differential equation of the form
       $ M \frac{dy}{dt} = g(y,t) $, recall that F and G can be defined as one
       of the following:

         1. F(u,k,t) = k and G(u,t) = inv(M) g(u,t)
         2. F(u,k,t) = M k and G(u,t) = g(u,t)
         3. F(u,k,t) = M k - g(u,t) and G(u,t) = 0

       @param[in]      r   inv(M) g(y,t) - k for 1 or g(y,t) - M k for 2 & 3.
       @param[in,out]  dk  On input, the initial guess. On output, the solution.
       @param[in]      tol Linear solve tolerance.

       If not re-implemented, this method simply generates an error.

       Presently, this method is used by SUNDIALS ODE solvers, for more
       details, see the SUNDIALS User Guides. */
   virtual int SUNImplicitSolve(const Vector &r, Vector &dk, real_t tol);

   /** @brief Setup the mass matrix in the ODE system
       $ M \frac{dy}{dt} = g(y,t) $ .

       If not re-implemented, this method simply generates an error.

       Presently, this method is used by SUNDIALS ARKStep integrator, for more
       details, see the ARKode User Guide. */
   virtual int SUNMassSetup();

   /** @brief Solve the mass matrix linear system  M @a x = @a b, where M is
       defined by the method SUNMassSetup().

       @param[in]      b   The linear system right-hand side.
       @param[in,out]  x   On input, the initial guess. On output, the solution.
       @param[in]      tol Linear solve tolerance.

       If not re-implemented, this method simply generates an error.

       Presently, this method is used by SUNDIALS ARKStep integrator, for more
       details, see the ARKode User Guide. */
   virtual int SUNMassSolve(const Vector &b, Vector &x, real_t tol);

   /** @brief Compute the mass matrix-vector product @a v = M @a x, where M is
       defined by the method SUNMassSetup().

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
   TimeDependentAdjointOperator(int dim, int adjdim, real_t t = 0.,
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

   /** @brief Setup the ODE linear system $ A(x,t) = (I - gamma J) $ or
       $ A = (M - gamma J) $, where $ J(x,t) = \frac{df}{dt(x,t)} $.

       @param[in]  t     The current time
       @param[in]  x     The state at which $A(x,xB,t)$ should be evaluated.
       @param[in]  xB    The state at which $A(x,xB,t)$ should be evaluated.
       @param[in]  fxB   The current value of the ODE rhs function, $f(x,t)$.
       @param[in]  jokB   Flag indicating if the Jacobian should be updated.
       @param[out] jcurB  Flag to signal if the Jacobian was updated.
       @param[in]  gammaB The scaled time step value.

       If not re-implemented, this method simply generates an error.

       Presently, this method is used by SUNDIALS ODE solvers, for more details,
       see the SUNDIALS User Guides.
   */
   virtual int SUNImplicitSetupB(const real_t t, const Vector &x,
                                 const Vector &xB, const Vector &fxB,
                                 int jokB, int *jcurB, real_t gammaB)
   {
      mfem_error("TimeDependentAdjointOperator::SUNImplicitSetupB() is not "
                 "overridden!");
      return (-1);
   }

   /** @brief Solve the ODE linear system $ A(x,xB,t) xB = b $ as setup by
       the method SUNImplicitSetup().

       @param[in]      b   The linear system right-hand side.
       @param[in,out]  x   On input, the initial guess. On output, the solution.
       @param[in]      tol Linear solve tolerance.

       If not re-implemented, this method simply generates an error.

       Presently, this method is used by SUNDIALS ODE solvers, for more details,
       see the SUNDIALS User Guides. */
   virtual int SUNImplicitSolveB(Vector &x, const Vector &b, real_t tol)
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
   explicit SecondOrderTimeDependentOperator(int n = 0, real_t t_ = 0.0,
                                             Type type_ = EXPLICIT)
      : TimeDependentOperator(n, t_,type_) { }

   /** @brief Construct a SecondOrderTimeDependentOperator y = f(x,dxdt,t),
       where x, dxdt and y have the same dimension @a n. */
   SecondOrderTimeDependentOperator(int h, int w, real_t t_ = 0.0,
                                    Type type_ = EXPLICIT)
      : TimeDependentOperator(h, w, t_,type_) { }

   using TimeDependentOperator::Mult;

   /** @brief Perform the action of the operator: @a y = k = f(@a x,@ dxdt, t),
       where k solves the algebraic equation
       F(@a x,@ dxdt, k, t) = G(@a x,@ dxdt, t) and t is the current time. */
   virtual void Mult(const Vector &x, const Vector &dxdt, Vector &y) const;

   using TimeDependentOperator::ImplicitSolve;
   /** @brief Solve the equation:
       @a k = f(@a x + @a fac0 @a k, @a dxdt + @a fac1 @a k, t), for the
       unknown @a k at the current time t.

       For general F and G, the equation for @a k becomes:
       F(@a x +  @a fac0 @a k, @a dxdt + @a fac1 @a k, t)
                        = G(@a x +  @a fac0 @a k, @a dxdt + @a fac1 @a k, t).

       The input vectors @a x and @a dxdt corresponds to time index (or cycle) n, while the
       currently set time, #t, and the result vector @a k correspond to time
       index n+1.

       This method allows for the abstract implementation of some time
       integration methods.

       If not re-implemented, this method simply generates an error. */
   virtual void ImplicitSolve(const real_t fac0, const real_t fac1,
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
   void Mult(const Vector &x, Vector &y) const override { y = x; }

   /// Application of the transpose
   void MultTranspose(const Vector &x, Vector &y) const override { y = x; }
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
   real_t a_;

public:
   /// Create an operator which is a scalar multiple of A.
   explicit ScaledOperator(const Operator *A, real_t a)
      : Operator(A->Height(), A->Width()), A_(*A), a_(a) { }

   /// Operator application
   void Mult(const Vector &x, Vector &y) const override
   { A_.Mult(x, y); y *= a_; }

   /// Application of the transpose.
   void MultTranspose(const Vector &x, Vector &y) const override
   { A_.MultTranspose(x, y); y *= a_; }
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
   void Mult(const Vector &x, Vector &y) const override
   { A.MultTranspose(x, y); }

   /// Application of the transpose. Apply the original Operator.
   void MultTranspose(const Vector &x, Vector &y) const override
   { A.Mult(x, y); }
};

/// General linear combination operator: x -> a A(x) + b B(x).
class SumOperator : public Operator
{
   const Operator *A, *B;
   const real_t alpha, beta;
   bool ownA, ownB;
   mutable Vector z;

public:
   SumOperator(
      const Operator *A, const real_t alpha,
      const Operator *B, const real_t beta,
      bool ownA, bool ownB);

   void Mult(const Vector &x, Vector &y) const override
   { z.SetSize(A->Height()); A->Mult(x, z); B->Mult(x, y); add(alpha, z, beta, y, y); }

   void MultTranspose(const Vector &x, Vector &y) const override
   { z.SetSize(A->Width()); A->MultTranspose(x, z); B->MultTranspose(x, y); add(alpha, z, beta, y, y); }

   virtual ~SumOperator();
};

/// General product operator: x -> (A*B)(x) = A(B(x)).
class ProductOperator : public Operator
{
   const Operator *A, *B;
   bool ownA, ownB;
   mutable Vector z;

public:
   ProductOperator(const Operator *A, const Operator *B, bool ownA, bool ownB);

   void Mult(const Vector &x, Vector &y) const override
   { B->Mult(x, z); A->Mult(z, y); }

   void MultTranspose(const Vector &x, Vector &y) const override
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

   MemoryClass GetMemoryClass() const override { return mem_class; }

   /// Operator application.
   void Mult(const Vector & x, Vector & y) const override
   { P.Mult(x, Px); A.Mult(Px, APx); Rt.MultTranspose(APx, y); }

   /// Approximate diagonal of the RAP Operator.
   /** Returns the diagonal of A, as returned by its AssembleDiagonal method,
       multiplied be P^T.

       When P is the FE space prolongation operator on a mesh without hanging
       nodes and Rt = P, the returned diagonal is exact, as long as the diagonal
       of A is also exact. */
   void AssembleDiagonal(Vector &diag) const override
   {
      A.AssembleDiagonal(APx);
      P.MultTranspose(APx, diag);

      // TODO: For an AMR mesh, a convergent diagonal can be assembled with
      // |P^T| APx, where |P^T| has entry-wise absolute values of the conforming
      // prolongation transpose operator. See BilinearForm::AssembleDiagonal.
   }

   /// Application of the transpose.
   void MultTranspose(const Vector & x, Vector & y) const override
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

   MemoryClass GetMemoryClass() const override { return mem_class; }

   void Mult(const Vector &x, Vector &y) const override
   { C->Mult(x, t1); B->Mult(t1, t2); A->Mult(t2, y); }

   void MultTranspose(const Vector &x, Vector &y) const override
   { A->MultTranspose(x, t2); B->MultTranspose(t2, t1); C->MultTranspose(t1, y); }

   virtual ~TripleProductOperator();
};


/** @brief Square Operator for imposing essential boundary conditions using only
    the action, Mult(), of a given unconstrained Operator.

    Square operator constrained by fixing certain entries in the solution to
    given "essential boundary condition" values. This class is used by the
    general, matrix-free system formulation of Operator::FormLinearSystem.

    Do not confuse with ConstrainedSolver, which despite the name has very
    different functionality. */
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
   MemoryClass GetMemoryClass() const override { return mem_class; }

   /// Set the diagonal policy for the constrained operator.
   void SetDiagonalPolicy(const DiagonalPolicy diag_policy_)
   { diag_policy = diag_policy_; }

   /// Diagonal of A, modified according to the used DiagonalPolicy.
   void AssembleDiagonal(Vector &diag) const override;

   /** @brief Eliminate "essential boundary condition" values specified in @a x
       from the given right-hand side @a b.

       Performs the following steps:

           z = A((0,x_b));  b_i -= z_i;  b_b = x_b;

       where the "_b" subscripts denote the essential (boundary) indices/dofs of
       the vectors, and "_i" -- the rest of the entries.

       @note This method is consistent with `DiagonalPolicy::DIAG_ONE`. */
   void EliminateRHS(const Vector &x, Vector &b) const;

   /** @brief Constrained operator action.

       Performs the following steps:

           z = A((x_i,0));  y_i = z_i;  y_b = x_b;

       where the "_b" subscripts denote the essential (boundary) indices/dofs of
       the vectors, and "_i" -- the rest of the entries. */
   void Mult(const Vector &x, Vector &y) const override;

   void AddMult(const Vector &x, Vector &y, const real_t a = 1.0) const override;

   void MultTranspose(const Vector &x, Vector &y) const override;

   /** @brief Implementation of Mult or MultTranspose.
    *  TODO - Generalize to allow constraining rows and columns differently.
   */
   void ConstrainedMult(const Vector &x, Vector &y, const bool transpose) const;

   /// Destructor: destroys the unconstrained Operator, if owned.
   ~ConstrainedOperator() override { if (own_A) { delete A; } }
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
   MemoryClass GetMemoryClass() const override { return mem_class; }
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
   void Mult(const Vector &x, Vector &y) const override;
   void MultTranspose(const Vector &x, Vector &y) const override;
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
   PowerMethod(MPI_Comm comm_) : comm(comm_) {}
#endif

   /// @brief Returns an estimate of the largest eigenvalue of the operator \p opr
   /// using the iterative power method.
   /** \p v0 is being used as the vector for the iterative process and will contain
       the eigenvector corresponding to the largest eigenvalue after convergence.
       The maximum number of iterations may set with \p numSteps, the relative
       tolerance with \p tolerance and the seed of the random initialization of
       \p v0 with \p seed. If \p seed is 0 \p v0 will not be random-initialized. */
   real_t EstimateLargestEigenvalue(Operator& opr, Vector& v0,
                                    int numSteps = 10, real_t tolerance = 1e-8,
                                    int seed = 12345);
};

}

#endif
