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

#ifndef MFEM_OPERATOR
#define MFEM_OPERATOR

#include "vector.hpp"

namespace mfem
{

class ConstrainedOperator;

/// Abstract operator
class Operator
{
protected:
   int height; ///< Dimension of the output / number of rows in the matrix.
   int width;  ///< Dimension of the input / number of columns in the matrix.

   /// see FormSystemOperator()
   void FormConstrainedSystemOperator(
      const Array<int> &ess_tdof_list, ConstrainedOperator* &Aout);

public:
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
   /** @brief Restriction operator from output vectors for the operator to linear
       algebra (linear system) vectors. `NULL` means identity. */
   virtual const Operator *GetOutputRestriction() const  { return NULL; }

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

   /** @brief Reconstruct a solution vector @a x (e.g. a GridFunction) from the
       solution @a X of a constrained linear system obtained from
       Operator::FormLinearSystem().

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
      MFEM_SPARSEMAT,   ///< ID for class SparseMatrix
      Hypre_ParCSR,     ///< ID for class HypreParMatrix.
      PETSC_MATAIJ,     ///< ID for class PetscParMatrix, MATAIJ format.
      PETSC_MATIS,      ///< ID for class PetscParMatrix, MATIS format.
      PETSC_MATSHELL,   ///< ID for class PetscParMatrix, MATSHELL format.
      PETSC_MATNEST,    ///< ID for class PetscParMatrix, MATNEST format.
      PETSC_MATHYPRE,   ///< ID for class PetscParMatrix, MATHYPRE format.
      PETSC_MATGENERIC  ///< ID for class PetscParMatrix, unspecified format.
   };

   /// Return the type ID of the Operator class.
   /** This method is intentionally non-virtual, so that it returns the ID of
       the specific pointer or reference type used when calling this method. If
       not overridden by derived classes, they will automatically use the type ID
       of the base Operator class, ANY_TYPE. */
   Type GetType() const { return ANY_TYPE; }
};


/// Base abstract class for time dependent operators.
/** Operator of the form: (x,t) -> f(x,t), where k = f(x,t) generally solves the
    algebraic equation F(x,k,t) = G(x,t). The functions F and G represent the
    _implicit_ and _explicit_ parts of the operator, respectively. For explicit
    operators, F(x,k,t) = k, so f(x,t) = G(x,t).*/
class TimeDependentOperator : public Operator
{
public:
   enum Type
   {
      EXPLICIT,   ///< This type assumes F(x,k,t) = k, i.e. k = f(x,t) = G(x,t).
      IMPLICIT,   ///< This is the most general type, no assumptions on F and G.
      HOMOGENEOUS ///< This type assumes that G(x,t) = 0.
   };

protected:
   double t;  ///< Current time.
   Type type; ///< Describes the form of the TimeDependentOperator.

public:
   /** @brief Construct a "square" TimeDependentOperator y = f(x,t), where x and
       y have the same dimension @a n. */
   explicit TimeDependentOperator(int n = 0, double t_ = 0.0,
                                  Type type_ = EXPLICIT)
      : Operator(n) { t = t_; type = type_; }

   /** @brief Construct a TimeDependentOperator y = f(x,t), where x and y have
       dimensions @a w and @a h, respectively. */
   TimeDependentOperator(int h, int w, double t_ = 0.0, Type type_ = EXPLICIT)
      : Operator(h, w) { t = t_; type = type_; }

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

   /** @brief Perform the action of the explicit part of the operator, G:
       @a y = G(@a x, t) where t is the current time.

       Presently, this method is used by some PETSc ODE solvers, for more
       details, see the PETSc Manual. */
   virtual void ExplicitMult(const Vector &x, Vector &y) const
   {
      mfem_error("TimeDependentOperator::ExplicitMult() is not overridden!");
   }

   /** @brief Perform the action of the implicit part of the operator, F:
       @a y = F(@a x, @a k, t) where t is the current time.

       Presently, this method is used by some PETSc ODE solvers, for more
       details, see the PETSc Manual.*/
   virtual void ImplicitMult(const Vector &x, const Vector &k, Vector &y) const
   {
      mfem_error("TimeDependentOperator::ImplicitMult() is not overridden!");
   }

   /** @brief Perform the action of the operator: @a y = k = f(@a x, t), where
       k solves the algebraic equation F(@a x, k, t) = G(@a x, t) and t is the
       current time. */
   virtual void Mult(const Vector &x, Vector &y) const
   {
      mfem_error("TimeDependentOperator::Mult() is not overridden!");
   }

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
   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k)
   {
      mfem_error("TimeDependentOperator::ImplicitSolve() is not overridden!");
   }

   /** @brief Return an Operator representing (dF/dk @a shift + dF/dx) at the
       given @a x, @a k, and the currently set time.

       Presently, this method is used by some PETSc ODE solvers, for more
       details, see the PETSc Manual. */
   virtual Operator& GetImplicitGradient(const Vector &x, const Vector &k,
                                         double shift) const
   {
      mfem_error("TimeDependentOperator::GetImplicitGradient() is "
                 "not overridden!");
      return const_cast<Operator &>(dynamic_cast<const Operator &>(*this));
   }

   /** @brief Return an Operator representing dG/dx at the given point @a x and
       the currently set time.

       Presently, this method is used by some PETSc ODE solvers, for more
       details, see the PETSc Manual. */
   virtual Operator& GetExplicitGradient(const Vector &x) const
   {
      mfem_error("TimeDependentOperator::GetExplicitGradient() is "
                 "not overridden!");
      return const_cast<Operator &>(dynamic_cast<const Operator &>(*this));
   }

   virtual ~TimeDependentOperator() { }
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

public:
   /** @brief Constructor from a general Operator and a list of essential
       indices/dofs.

       Specify the unconstrained operator @a *A and a @a list of indices to
       constrain, i.e. each entry @a list[i] represents an essential-dof. If the
       ownership flag @a own_A is true, the operator @a *A will be destroyed
       when this object is destroyed. */
   ConstrainedOperator(Operator *A, const Array<int> &list, bool own_A = false);

   virtual MemoryClass GetMemoryClass() const { return mem_class; }

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

}

#endif
