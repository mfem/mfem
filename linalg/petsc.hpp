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

// Author: Stefano Zampini <stefano.zampini@gmail.com>

#ifndef MFEM_PETSC
#define MFEM_PETSC

#include "../config/config.hpp"

#ifdef MFEM_USE_PETSC
#ifdef MFEM_USE_MPI

#include <limits>

#include "handle.hpp"
#include "hypre.hpp"
#include "ode.hpp"

#include "petscconf.h"
#if !defined(PETSC_USE_REAL_DOUBLE)
#error "MFEM does not work with PETSc compiled without double precision"
#endif
#if defined(PETSC_USE_COMPLEX)
#error "MFEM does not work with PETSc compiled with complex numbers support"
#endif
#if defined(PETSC_USE_64BIT_INDICES) && !defined(HYPRE_BIGINT)
#error "Mismatch between HYPRE (32bit) and PETSc (64bit) integer types"
#endif
#if !defined(PETSC_USE_64BIT_INDICES) && defined(HYPRE_BIGINT)
#error "Mismatch between HYPRE (64bit) and PETSc (32bit) integer types"
#endif

#include "petscversion.h"
#if PETSC_VERSION_GE(3,12,0)
#include "petscsystypes.h"
#else
typedef HYPRE_Int PetscInt;
typedef double PetscScalar;
typedef double PetscReal;
typedef int PetscClassId;
typedef struct _p_PetscObject *PetscObject;
#endif

// forward declarations of PETSc objects
typedef struct _p_Vec *Vec;
typedef struct _p_Mat *Mat;
typedef struct _p_KSP *KSP;
typedef struct _p_PC *PC;
typedef struct _p_SNES *SNES;
typedef struct _p_TS *TS;


namespace mfem
{

/// Convenience functions to initialize/finalize PETSc
void MFEMInitializePetsc();
void MFEMInitializePetsc(int*,char***);
void MFEMInitializePetsc(int*,char***,const char[],const char[]);
void MFEMFinalizePetsc();

class ParFiniteElementSpace;
class PetscParMatrix;

/// Wrapper for PETSc's vector class
class PetscParVector : public Vector
{
protected:
   /// The actual PETSc object
   Vec x;

   friend class PetscParMatrix;
   friend class PetscODESolver;
   friend class PetscLinearSolver;
   friend class PetscPreconditioner;
   friend class PetscNonlinearSolver;
   friend class PetscBDDCSolver;

   // Set Vector::data and Vector::size from x
   void _SetDataAndSize_();

public:
   /// Creates vector with given global size and partitioning of the columns.
   /** If @a col is provided, processor P owns columns [col[P],col[P+1]).
       Otherwise, PETSc decides the partitioning */
   PetscParVector(MPI_Comm comm, PetscInt glob_size, PetscInt *col = NULL);

   /** @brief Creates vector with given global size, partitioning of the
       columns, and data.

       The data must be allocated and destroyed outside. If @a _data is NULL, a
       dummy vector without a valid data array will be created. */
   PetscParVector(MPI_Comm comm, PetscInt glob_size, PetscScalar *_data,
                  PetscInt *col);

   /// Creates vector compatible with @a y
   PetscParVector(const PetscParVector &y);

   /** @brief Creates a PetscParVector from a Vector
       @param[in] comm  MPI communicator on which the new object lives
       @param[in] _x    The mfem Vector (data is not shared)
       @param[in] copy  Whether to copy the data in _x or not */
   PetscParVector(MPI_Comm comm, const Vector &_x, bool copy = false);

   /** @brief Creates vector compatible with the Operator (i.e. in the domain
       of) @a op or its adjoint. */
   /** The argument @a allocate determines if the memory is actually allocated
       to store the data. */
   explicit PetscParVector(MPI_Comm comm, const Operator &op,
                           bool transpose = false, bool allocate = true);

   /// Creates vector compatible with (i.e. in the domain of) @a A or @a A^T
   /** The argument @a allocate determines if the memory is actually allocated
       to store the data. */
   explicit PetscParVector(const PetscParMatrix &A, bool transpose = false,
                           bool allocate = true);

   /// Creates PetscParVector out of PETSc Vec object.
   /** @param[in] y    The PETSc Vec object.
       @param[in] ref  If true, we increase the reference count of @a y. */
   explicit PetscParVector(Vec y, bool ref=false);

   /// Create a true dof parallel vector on a given ParFiniteElementSpace
   explicit PetscParVector(ParFiniteElementSpace *pfes);

   /// Calls PETSc's destroy function
   virtual ~PetscParVector();

   /// Get the associated MPI communicator
   MPI_Comm GetComm() const;

   /// Returns the global number of rows
   PetscInt GlobalSize() const;

   /// Typecasting to PETSc's Vec type
   operator Vec() const { return x; }

   /// Typecasting to PETSc object
   operator PetscObject() const { return (PetscObject)x; }

   /// Returns the global vector in each processor
   Vector* GlobalVector() const;

   /// Set constant values
   PetscParVector& operator= (PetscScalar d);

   /** @brief Set values in a vector.

       @note any process can insert in any location
       @note This is a collective operation, so all process needs to call it  */
   PetscParVector& SetValues(const Array<PetscInt>&, const Array<PetscScalar>&);

   /** @brief Add values in a vector.

       @note any process can add to any location
       @note This is a collective operation, so all process needs to call it  */
   PetscParVector& AddValues(const Array<PetscInt>&, const Array<PetscScalar>&);

   /// Define operators for PETSc vectors.
   PetscParVector& operator= (const PetscParVector &y);
   PetscParVector& operator+= (const PetscParVector &y);
   PetscParVector& operator-= (const PetscParVector &y);
   PetscParVector& operator*= (PetscScalar d);
   PetscParVector& operator+= (PetscScalar d);

   /** @brief Temporarily replace the data of the PETSc Vec object. To return to
       the original data array, call ResetArray().

       @note This method calls PETSc's VecPlaceArray() function.
       @note The inherited Vector::data pointer is not affected by this call. */
   void PlaceArray(PetscScalar *temp_data);

   /** @brief Reset the PETSc Vec object to use its default data. Call this
       method after the use of PlaceArray().

       @note This method calls PETSc's VecResetArray() function. */
   void ResetArray();

   /// Set random values
   void Randomize(PetscInt seed = 0);

   /// Prints the vector (to stdout if @a fname is NULL)
   void Print(const char *fname = NULL, bool binary = false) const;
};


/// Wrapper for PETSc's matrix class
class PetscParMatrix : public Operator
{
protected:
   /// The actual PETSc object
   Mat A;

   /// Auxiliary vectors for typecasting
   mutable PetscParVector *X, *Y;

   /// Initialize with defaults. Does not initialize inherited members.
   void Init();

   /// Delete all owned data. Does not perform re-initialization with defaults.
   void Destroy();

   /** @brief Creates a wrapper around a mfem::Operator @a op using PETSc's
       MATSHELL object and returns the Mat in @a B.

       This does not take any reference to @a op, that should not be destroyed
       until @a B is needed. */
   void MakeWrapper(MPI_Comm comm, const Operator* op, Mat *B);

   /// Convert an mfem::Operator into a Mat @a B; @a op can be destroyed unless
   /// tid == PETSC_MATSHELL or tid == PETSC_MATHYPRE
   /// if op is a BlockOperator, the operator type is relevant to the individual
   /// blocks
   void ConvertOperator(MPI_Comm comm, const Operator& op, Mat *B,
                        Operator::Type tid);

   friend class PetscLinearSolver;
   friend class PetscPreconditioner;

private:
   /// Constructs a block-diagonal Mat object
   void BlockDiagonalConstructor(MPI_Comm comm, PetscInt *row_starts,
                                 PetscInt *col_starts, SparseMatrix *diag,
                                 bool assembled, Mat *A);

public:
   /// Create an empty matrix to be used as a reference to an existing matrix.
   PetscParMatrix();

   /// Creates PetscParMatrix out of PETSc's Mat.
   /** @param[in]  a    The PETSc Mat object.
       @param[in]  ref  If true, we increase the reference count of @a a. */
   PetscParMatrix(Mat a, bool ref=false);

   /** @brief Convert a PetscParMatrix @a pa with a new PETSc format @a tid.
       Note that if @a pa is already a PetscParMatrix of the same type as
       @a tid, the resulting PetscParMatrix will share the same Mat object */
   explicit PetscParMatrix(const PetscParMatrix *pa, Operator::Type tid);

   /** @brief Creates a PetscParMatrix extracting the submatrix of @a A with
       @a rows row indices and @a cols column indices */
   PetscParMatrix(const PetscParMatrix& A, const mfem::Array<PetscInt>& rows,
                  const mfem::Array<PetscInt>& cols);

   /** @brief Convert a HypreParMatrix @a ha to a PetscParMatrix in the given
       PETSc format @a tid. */
   /** The supported type ids are: Operator::PETSC_MATAIJ,
       Operator::PETSC_MATIS, Operator::PETSC_MATSHELL and
       Operator::PETSC_MATHYPRE
       @a ha can be destroyed unless tid == PETSC_MATSHELL or
       tid == PETSC_MATHYPRE */
   explicit PetscParMatrix(const HypreParMatrix *ha,
                           Operator::Type tid = Operator::PETSC_MATAIJ);

   /** @brief Convert a SparseMatrix @a ha to a PetscParMatrix in the given
       PETSc format @a tid. */
   explicit PetscParMatrix(const SparseMatrix *sa,
                           Operator::Type tid = Operator::PETSC_MATAIJ);

   /** @brief Convert an mfem::Operator into a PetscParMatrix in the given PETSc
       format @a tid. */
   /** If @a tid is Operator::PETSC_MATSHELL and @a op is not a PetscParMatrix,
       it converts any mfem::Operator @a op implementing Operator::Mult() and
       Operator::MultTranspose() into a PetscParMatrix. The Operator @a op
       should not be deleted while the constructed PetscParMatrix is used.

       Otherwise, it tries to convert the operator in PETSc's classes.
       @a op cannot be destroyed if tid == PETSC_MATHYPRE.

       In particular, if @a op is a BlockOperator, then a MATNEST Mat object is
       created using @a tid as the type for the blocks.
       Note that if @a op is already a PetscParMatrix of the same type as
       @a tid, the resulting PetscParMatrix will share the same Mat object */
   PetscParMatrix(MPI_Comm comm, const Operator *op,
                  Operator::Type tid = Operator::PETSC_MATSHELL);

   /// Creates block-diagonal square parallel matrix.
   /** The block-diagonal is given by @a diag which must be in CSR format
       (finalized). The new PetscParMatrix does not take ownership of any of the
       input arrays. The type id @a tid can be either PETSC_MATAIJ (parallel
       distributed CSR) or PETSC_MATIS. */
   PetscParMatrix(MPI_Comm comm, PetscInt glob_size, PetscInt *row_starts,
                  SparseMatrix *diag, Operator::Type tid);

   /// Creates block-diagonal rectangular parallel matrix.
   /** The block-diagonal is given by @a diag which must be in CSR format
       (finalized). The new PetscParMatrix does not take ownership of any of the
       input arrays. The type id @a tid can be either PETSC_MATAIJ (parallel
       distributed CSR) or PETSC_MATIS. */
   PetscParMatrix(MPI_Comm comm, PetscInt global_num_rows,
                  PetscInt global_num_cols, PetscInt *row_starts,
                  PetscInt *col_starts, SparseMatrix *diag,
                  Operator::Type tid);

   /// Calls PETSc's destroy function.
   virtual ~PetscParMatrix() { Destroy(); }

   /// Replace the inner Mat Object. The reference count of newA is increased
   void SetMat(Mat newA);

   /// @name Assignment operators
   ///@{
   PetscParMatrix& operator=(const PetscParMatrix& B);
   PetscParMatrix& operator=(const HypreParMatrix& B);
   PetscParMatrix& operator+=(const PetscParMatrix& B);
   PetscParMatrix& operator-=(const PetscParMatrix& B);
   ///@}

   /// Matvec: @a y = @a a A @a x + @a b @a y.
   void Mult(double a, const Vector &x, double b, Vector &y) const;

   /// Matvec transpose: @a y = @a a A^T @a x + @a b @a y.
   void MultTranspose(double a, const Vector &x, double b, Vector &y) const;

   virtual void Mult(const Vector &x, Vector &y) const
   { Mult(1.0, x, 0.0, y); }

   virtual void MultTranspose(const Vector &x, Vector &y) const
   { MultTranspose(1.0, x, 0.0, y); }

   /// Get the associated MPI communicator
   MPI_Comm GetComm() const;

   /// Typecasting to PETSc's Mat type
   operator Mat() const { return A; }

   /// Typecasting to PETSc object
   operator PetscObject() const { return (PetscObject)A; }

   /// Returns the global index of the first local row
   PetscInt GetRowStart() const;

   /// Returns the global index of the first local column
   PetscInt GetColStart() const;

   /// Returns the local number of rows
   PetscInt GetNumRows() const;

   /// Returns the local number of columns
   PetscInt GetNumCols() const;

   /// Returns the global number of rows
   PetscInt M() const;

   /// Returns the global number of columns
   PetscInt N() const;

   /// Returns the global number of rows
   PetscInt GetGlobalNumRows() const { return M(); }

   /// Returns the global number of columns
   PetscInt GetGlobalNumCols() const { return N(); }

   /// Returns the number of nonzeros.
   /** Differently from HYPRE, this call is collective on the communicator,
       as this number is not stored inside PETSc, but needs to be computed. */
   PetscInt NNZ() const;

   /// Returns the inner vector in the domain of A (it creates it if needed)
   PetscParVector* GetX() const;

   /// Returns the inner vector in the range of A (it creates it if needed)
   PetscParVector* GetY() const;

   /// Returns the transpose of the PetscParMatrix.
   /** If @a action is false, the new matrix is constructed with the PETSc
       function MatTranspose().
       If @a action is true, then the matrix is not actually transposed.
       Instead, an object that behaves like the transpose is returned. */
   PetscParMatrix* Transpose(bool action = false);

   /// Prints the matrix (to stdout if fname is NULL)
   void Print(const char *fname = NULL, bool binary = false) const;

   /// Scale all entries by s: A_scaled = s*A.
   void operator*=(double s);

   /** @brief Eliminate rows and columns from the matrix, and rows from the
       vector @a B. Modify @a B with the BC values in @a X. Put @a diag
       on the diagonal corresponding to eliminated entries */
   void EliminateRowsCols(const Array<int> &rows_cols, const PetscParVector &X,
                          PetscParVector &B, double diag = 1.);
   void EliminateRowsCols(const Array<int> &rows_cols, const HypreParVector &X,
                          HypreParVector &B, double diag = 1.);

   /** @brief Eliminate rows and columns from the matrix and store the
       eliminated elements in a new matrix Ae (returned).

       The sum of the modified matrix and the returned matrix, Ae, is equal to
       the original matrix. */
   PetscParMatrix* EliminateRowsCols(const Array<int> &rows_cols);

   /// Scale the local row i by s(i).
   void ScaleRows(const Vector & s);

   /// Scale the local col i by s(i).
   void ScaleCols(const Vector & s);

   /// Shift diagonal by a constant
   void Shift(double s);

   /// Shift diagonal by a vector
   void Shift(const Vector & s);

   /** @brief Eliminate only the rows from the matrix */
   void EliminateRows(const Array<int> &rows);

   /// Makes this object a reference to another PetscParMatrix
   void MakeRef(const PetscParMatrix &master);

   /** @brief Release the PETSc Mat object. If @a dereference is true, decrement
       the refcount of the Mat object. */
   Mat ReleaseMat(bool dereference);

   Type GetType() const;
};

/// Returns the matrix A * B
PetscParMatrix * ParMult(const PetscParMatrix *A, const PetscParMatrix *B);

/// Returns the matrix Rt^t * A * P
PetscParMatrix * RAP(PetscParMatrix *Rt, PetscParMatrix *A, PetscParMatrix *P);

/// Returns the matrix P^t * A * P
PetscParMatrix * RAP(PetscParMatrix *A, PetscParMatrix *P);

/// Returns the matrix P^t * A * P
PetscParMatrix * RAP(HypreParMatrix *A, PetscParMatrix *P);

/** @brief Eliminate essential BC specified by @a ess_dof_list from the solution
    @a X to the r.h.s. @a B.

    Here, @a A is a matrix with eliminated BC, while @a Ae is such that
    (@a A + @a Ae) is the original (Neumann) matrix before elimination. */
void EliminateBC(PetscParMatrix &A, PetscParMatrix &Ae,
                 const Array<int> &ess_dof_list, const Vector &X, Vector &B);

/// Helper class for handling essential boundary conditions.
class PetscBCHandler
{
public:
   enum Type
   {
      ZERO,
      CONSTANT,  ///< Constant in time b.c.
      TIME_DEPENDENT
   };

   PetscBCHandler(Type _type = ZERO) :
      bctype(_type), setup(false), eval_t(0.0),
      eval_t_cached(std::numeric_limits<double>::min()) {}
   PetscBCHandler(Array<int>& ess_tdof_list, Type _type = ZERO);

   virtual ~PetscBCHandler() {}

   /// Returns the type of boundary conditions
   Type GetType() const { return bctype; }

   /// Sets the type of boundary conditions
   void SetType(enum Type _type) { bctype = _type; setup = false; }

   /// Boundary conditions evaluation
   /** In the result vector, @a g, only values at the essential dofs need to be
       set. */
   virtual void Eval(double t, Vector &g)
   { mfem_error("PetscBCHandler::Eval method not overloaded"); }

   /// Sets essential dofs (local, per-process numbering)
   void SetTDofs(Array<int>& list);

   /// Gets essential dofs (local, per-process numbering)
   Array<int>& GetTDofs() { return ess_tdof_list; }

   /// Sets the current time
   void SetTime(double t) { eval_t = t; }

   /// SetUp the helper object, where @a n is the size of the solution vector
   void SetUp(PetscInt n);

   /// y = x on ess_tdof_list_c and y = g (internally evaluated) on ess_tdof_list
   void ApplyBC(const Vector &x, Vector &y);

   /// Replace boundary dofs with the current value
   void ApplyBC(Vector &x);

   /// y = x-g on ess_tdof_list, the rest of y is unchanged
   void FixResidualBC(const Vector& x, Vector& y);

private:
   enum Type bctype;
   bool setup;

   double eval_t;
   double eval_t_cached;
   Vector eval_g;

   Array<int> ess_tdof_list;    //Essential true dofs
};

// Helper class for user-defined preconditioners that needs to be setup
class PetscPreconditionerFactory
{
private:
   std::string name;
public:
   PetscPreconditionerFactory(const std::string &_name = "MFEM Factory")
      : name(_name) { }
   const char* GetName() { return name.c_str(); }
   virtual Solver *NewPreconditioner(const OperatorHandle& oh) = 0;
   virtual ~PetscPreconditionerFactory() {}
};

// Forward declarations of helper classes
class PetscSolverMonitor;

/// Abstract class for PETSc's solvers.
class PetscSolver
{
protected:
   /// Boolean to handle SetFromOptions calls.
   mutable bool clcustom;

   /// The actual PETSc object (KSP, PC, SNES or TS).
   PetscObject obj;

   /// The class id of the actual PETSc object
   PetscClassId cid;

   /// Right-hand side and solution vector
   mutable PetscParVector *B, *X;

   /// Handler for boundary conditions
   PetscBCHandler *bchandler;

   /// Private context for solver
   void *private_ctx;

   /// Boolean to handle SetOperator calls.
   mutable bool operatorset;

public:
   /// Construct an empty PetscSolver. Initialize protected objects to NULL.
   PetscSolver();

   /// Destroy the PetscParVectors allocated (if any).
   virtual ~PetscSolver();

   /** @name Update of PETSc options.
       The following Set methods can be used to update the internal PETSc
       options.
       @note They will be overwritten by the options in the input PETSc file. */
   ///@{
   void SetTol(double tol);
   void SetRelTol(double tol);
   void SetAbsTol(double tol);
   void SetMaxIter(int max_iter);
   void SetPrintLevel(int plev);
   ///@}

   /// Customize object with options set
   /** If @a customize is false, it disables any options customization. */
   void Customize(bool customize = true) const;
   int GetConverged();
   int GetNumIterations();
   double GetFinalNorm();

   /// Sets user-defined monitoring routine.
   void SetMonitor(PetscSolverMonitor *ctx);

   /// Sets the object to handle essential boundary conditions
   void SetBCHandler(PetscBCHandler *bch);

   /// Sets the object for the creation of the preconditioner
   void SetPreconditionerFactory(PetscPreconditionerFactory *factory);

   /// Conversion function to PetscObject.
   operator PetscObject() const { return obj; }

   /// Get the associated MPI communicator
   MPI_Comm GetComm() const;

protected:
   /// These two methods handle creation and destructions of
   /// private data for the Solver objects
   void CreatePrivateContext();
   void FreePrivateContext();
};


/// Abstract class for PETSc's linear solvers.
class PetscLinearSolver : public PetscSolver, public Solver
{
private:
   /// Internal flag to handle HypreParMatrix conversion or not.
   bool wrap;
   void MultKernel(const Vector &b, Vector &x, bool trans) const;

public:
   PetscLinearSolver(MPI_Comm comm, const std::string &prefix = std::string(),
                     bool wrap = true);
   PetscLinearSolver(const PetscParMatrix &A,
                     const std::string &prefix = std::string());
   /// Constructs a solver using a HypreParMatrix.
   /** If @a wrap is true, then the MatMult ops of HypreParMatrix are wrapped.
       No preconditioner can be automatically constructed from PETSc. If
       @a wrap is false, the HypreParMatrix is converted into a the AIJ
       PETSc format, which is suitable for most preconditioning methods. */
   PetscLinearSolver(const HypreParMatrix &A, bool wrap = true,
                     const std::string &prefix = std::string());
   virtual ~PetscLinearSolver();

   virtual void SetOperator(const Operator &op);

   /// Allows to prescribe a different operator (@a pop) to construct
   /// the preconditioner
   void SetOperator(const Operator &op, const Operator &pop);

   /// Sets the solver to perform preconditioning
   void SetPreconditioner(Solver &precond);

   /// Application of the solver.
   virtual void Mult(const Vector &b, Vector &x) const;
   virtual void MultTranspose(const Vector &b, Vector &x) const;

   /// Conversion function to PETSc's KSP type.
   operator KSP() const { return (KSP)obj; }
};


class PetscPCGSolver : public PetscLinearSolver
{
public:
   PetscPCGSolver(MPI_Comm comm, const std::string &prefix = std::string());
   PetscPCGSolver(PetscParMatrix &A, const std::string &prefix = std::string());
   PetscPCGSolver(HypreParMatrix &A,bool wrap=true,
                  const std::string &prefix = std::string());
};


/// Abstract class for PETSc's preconditioners.
class PetscPreconditioner : public PetscSolver, public Solver
{
private:
   void MultKernel(const Vector &b, Vector &x, bool trans) const;

public:
   PetscPreconditioner(MPI_Comm comm,
                       const std::string &prefix = std::string());
   PetscPreconditioner(PetscParMatrix &A,
                       const std::string &prefix = std::string());
   PetscPreconditioner(MPI_Comm comm, Operator &op,
                       const std::string &prefix = std::string());
   virtual ~PetscPreconditioner();

   virtual void SetOperator(const Operator &op);

   /// Application of the preconditioner.
   virtual void Mult(const Vector &b, Vector &x) const;
   virtual void MultTranspose(const Vector &b, Vector &x) const;

   /// Conversion function to PETSc's PC type.
   operator PC() const { return (PC)obj; }
};


/// Auxiliary class for BDDC customization.
class PetscBDDCSolverParams
{
protected:
   ParFiniteElementSpace *fespace;
   const Array<int>      *ess_dof;
   bool                  ess_dof_local;
   const Array<int>      *nat_dof;
   bool                  nat_dof_local;
   bool                  netflux;
   friend class PetscBDDCSolver;

public:
   PetscBDDCSolverParams() : fespace(NULL), ess_dof(NULL), ess_dof_local(false),
      nat_dof(NULL), nat_dof_local(false), netflux(false)
   {}
   void SetSpace(ParFiniteElementSpace *fe) { fespace = fe; }

   /// Specify dofs on the essential boundary.
   /** If @a loc is false, it is a list of true dofs in local ordering.
       If @a loc is true, it is a marker for Vdofs in local ordering. */
   void SetEssBdrDofs(const Array<int> *essdofs, bool loc = false)
   {
      ess_dof = essdofs;
      ess_dof_local = loc;
   }
   /// Specify dofs on the natural boundary.
   /** If @a loc is false, it is a list of true dofs in local ordering.
       If @a loc is true, it is a marker for Vdofs in local ordering. */
   void SetNatBdrDofs(const Array<int> *natdofs, bool loc = false)
   {
      nat_dof = natdofs;
      nat_dof_local = loc;
   }
   /// Setup BDDC with no-net-flux local solvers. Needs a ParFiniteElementSpace attached
   void SetComputeNetFlux(bool net = true)
   {
      netflux = net;
   }
};


class PetscBDDCSolver : public PetscPreconditioner
{
private:
   void BDDCSolverConstructor(const PetscBDDCSolverParams &opts);

public:
   PetscBDDCSolver(MPI_Comm comm, Operator &op,
                   const PetscBDDCSolverParams &opts = PetscBDDCSolverParams(),
                   const std::string &prefix = std::string());
   PetscBDDCSolver(PetscParMatrix &op,
                   const PetscBDDCSolverParams &opts = PetscBDDCSolverParams(),
                   const std::string &prefix = std::string());
};


class PetscFieldSplitSolver : public PetscPreconditioner
{
public:
   PetscFieldSplitSolver(MPI_Comm comm, Operator &op,
                         const std::string &prefix = std::string());
};


/// Abstract class for PETSc's nonlinear solvers.
class PetscNonlinearSolver : public PetscSolver, public Solver
{
public:
   PetscNonlinearSolver(MPI_Comm comm,
                        const std::string &prefix = std::string());
   PetscNonlinearSolver(MPI_Comm comm, Operator &op,
                        const std::string &prefix = std::string());
   virtual ~PetscNonlinearSolver();

   /// Specification of the nonlinear operator.
   virtual void SetOperator(const Operator &op);

   /// Specifies the desired format of the Jacobian in case a PetscParMatrix
   /// is not returned by the GetGradient method.
   void SetJacobianType(Operator::Type type);

   /// Application of the solver.
   virtual void Mult(const Vector &b, Vector &x) const;

   /// Specification of an objective function to be used for line search.
   void SetObjective(void (*obj)(Operator* op, const Vector &x, double *f));

   /// User-defined routine to be applied after a successful line search step.
   /// The user can change the current direction Y and/or the updated solution W
   /// (with W = X - lambda * Y) but not the previous solution X.
   /// If Y or W have been changed, the corresponding booleans need to updated.
   void SetPostCheck(void (*post)(Operator *op, const Vector &X, Vector &Y,
                                  Vector &W, bool &changed_y, bool &changed_w));

   /// General purpose update function to be called at the beginning of each step
   /// it is the current nonlinear iteration number
   /// F is the current function value, X the current solution
   /// D the previous step taken, and P the previous solution
   void SetUpdate(void (*update)(Operator *op, int it,
                                 const mfem::Vector& F, const mfem::Vector& X,
                                 const mfem::Vector& D, const mfem::Vector& P));

   /// Conversion function to PETSc's SNES type.
   operator SNES() const { return (SNES)obj; }
};


/// Abstract class for PETSc's ODE solvers.
class PetscODESolver : public PetscSolver, public ODESolver
{
public:
   /// The type of the ODE. Use ODE_SOLVER_LINEAR if the jacobians
   /// are linear and independent of time.
   enum Type
   {
      ODE_SOLVER_LINEAR,
      ODE_SOLVER_GENERAL
   };

   PetscODESolver(MPI_Comm comm, const std::string &prefix = std::string());
   virtual ~PetscODESolver();

   /// Initialize the ODE solver.
   virtual void Init(TimeDependentOperator &f_,
                     enum PetscODESolver::Type type);
   virtual void Init(TimeDependentOperator &f_) { Init(f_,ODE_SOLVER_GENERAL); }

   void SetType(PetscODESolver::Type);
   PetscODESolver::Type GetType() const;

   /// Specifies the desired format of the Jacobian in case a PetscParMatrix
   /// is not returned by the GetGradient methods
   void SetJacobianType(Operator::Type type);

   virtual void Step(Vector &x, double &t, double &dt);
   virtual void Run(Vector &x, double &t, double &dt, double t_final);

   /// Conversion function to PETSc's TS type.
   operator TS() const { return (TS)obj; }
};


/// Abstract class for monitoring PETSc's solvers.
class PetscSolverMonitor
{
public:
   bool mon_sol;
   bool mon_res;
   PetscSolverMonitor(bool monitor_sol = false, bool monitor_res = true)
      : mon_sol(monitor_sol), mon_res(monitor_res) {}
   virtual ~PetscSolverMonitor() {}

   /// Monitor the solution vector x
   virtual void MonitorSolution(PetscInt it, PetscReal norm, const Vector &x)
   {
      MFEM_ABORT("MonitorSolution() is not implemented!")
   }

   /// Monitor the residual vector r
   virtual void MonitorResidual(PetscInt it, PetscReal norm, const Vector &r)
   {
      MFEM_ABORT("MonitorResidual() is not implemented!")
   }

   /// Generic monitor to take access to the solver
   virtual void MonitorSolver(PetscSolver* solver) {}
};


} // namespace mfem

#endif // MFEM_USE_MPI
#endif // MFEM_USE_PETSC

#endif
