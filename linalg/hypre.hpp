// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_HYPRE
#define MFEM_HYPRE

// Enable internal hypre timing routines
#define HYPRE_TIMING

// hypre header files
#include "seq_mv.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_parcsr_ls.h"

/// Wrapper for hypre's parallel vector class
class HypreParVector : public Vector
{
private:
   int own_ParVector;

   /// The actual object
   hypre_ParVector *x;

   friend class HypreParMatrix;

public:
   /** Creates vector with given global size and partitioning of the columns.
       Processor P owns columns [col[P],col[P+1]) */
   HypreParVector(int glob_size, int *col);
   /** Creates vector with given global size, partitioning of the columns,
       and data. The data must be allocated and destroyed outside. */
   HypreParVector(int glob_size, double *_data, int *col);
   /// Creates vector compatible with y
   HypreParVector(const HypreParVector &y);
   /// Creates vector wrapping y
   HypreParVector(HYPRE_ParVector y);

   /// Typecasting to hypre's hypre_ParVector*
   operator hypre_ParVector*() const;
   /// Typecasting to hypre's HYPRE_ParVector, a.k.a. void *
   operator HYPRE_ParVector() const;
   /// Changes the ownership of the the vector
   hypre_ParVector *StealParVector() { own_ParVector = 0; return x; }

   /// Returns the global vector in each processor
   Vector *GlobalVector();

   /// Set constant values
   HypreParVector& operator= (double d);
   /// Define '=' for hypre vectors.
   HypreParVector& operator= (const HypreParVector &y);

   /** Sets the data of the Vector and the hypre_ParVector to _data.
       Must be used only for HypreParVectors that do not own the data,
       e.g. created with the constructor:
       HypreParVector(int glob_size, double *_data, int *col).  */
   void SetData(double *_data);

   /// Set random values
   int Randomize(int seed);

   /// Prints the locally owned rows in parallel
   void Print(const char *fname);

   /// Calls hypre's destroy function
   ~HypreParVector();
};

/// Returns the inner product of x and y
double InnerProduct(HypreParVector &x, HypreParVector &y);
double InnerProduct(HypreParVector *x, HypreParVector *y);


/// Wrapper for hypre's ParCSR matrix class
class HypreParMatrix : public Operator
{
private:
   /// The actual object
   hypre_ParCSRMatrix *A;

   /// Internal communication object associated with A
   hypre_ParCSRCommPkg *CommPkg;

   /// Auxiliary vectors for typecasting
   mutable HypreParVector *X, *Y;

public:
   /// Converts hypre's format to HypreParMatrix
   HypreParMatrix(hypre_ParCSRMatrix *a) : A(a)
   { X = Y = 0; CommPkg = 0; }
   /// Creates block-diagonal square parallel matrix. Diagonal given by diag.
   HypreParMatrix(int size, int *row, SparseMatrix *diag);
   /** Creates block-diagonal rectangular parallel matrix. Diagonal
       given by diag. */
   HypreParMatrix(int M, int N, int *row, int *col, SparseMatrix *diag);
   /// Creates general (rectangular) parallel matrix
   HypreParMatrix(int M, int N, int *row, int *col, SparseMatrix *diag,
                  SparseMatrix *offd, int *cmap);

   /// Creates a parallel matrix from SparseMatrix on processor 0.
   HypreParMatrix(int *row, int *col, SparseMatrix *a);

   /// Creates boolean block-diagonal rectangular parallel matrix.
   HypreParMatrix(int M, int N, int *row, int *col, Table *diag);
   /// Creates boolean rectangular parallel matrix (which owns its data)
   HypreParMatrix(MPI_Comm comm, int id, int np, int *row, int *col,
                  int *i_diag, int *j_diag, int *i_offd, int *j_offd,
                  int *cmap, int cmap_size);

   // hypre's communication package object
   void SetCommPkg(hypre_ParCSRCommPkg *comm_pkg);
   void CheckCommPkg();
   void DestroyCommPkg();

   /// Typecasting to hypre's hypre_ParCSRMatrix*
   operator hypre_ParCSRMatrix*();
   /// Typecasting to hypre's HYPRE_ParCSRMatrix, a.k.a. void *
   operator HYPRE_ParCSRMatrix();
   /// Changes the ownership of the the matrix
   hypre_ParCSRMatrix* StealData();

   /// Returns the number of nonzeros
   inline int NNZ() { return A->num_nonzeros; }
   /// Returns the row partitioning
   inline int * RowPart() { return A->row_starts; }
   /// Returns the row partitioning
   inline int * ColPart() { return A->col_starts; }
   /// Returns the global number of rows
   inline int M() { return A -> global_num_rows; }
   /// Returns the global number of columns
   inline int N() { return A -> global_num_cols; }

   /// Returns the transpose of *this
   HypreParMatrix * Transpose();

   /// Returns the number of rows in the diagonal block of the ParCSRMatrix
   int GetNumRows() const
   { return hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A)); }

   int GetGlobalNumRows() const { return hypre_ParCSRMatrixGlobalNumRows(A); }

   int GetGlobalNumCols() const { return hypre_ParCSRMatrixGlobalNumCols(A); }

   int *GetRowStarts() const { return hypre_ParCSRMatrixRowStarts(A); }

   int *GetColStarts() const { return hypre_ParCSRMatrixColStarts(A); }

   /// Computes y = alpha * A * x + beta * y
   int Mult(HypreParVector &x, HypreParVector &y,
            double alpha = 1.0, double beta = 0.0);
   /// Computes y = alpha * A * x + beta * y
   int Mult(HYPRE_ParVector x, HYPRE_ParVector y,
            double alpha = 1.0, double beta = 0.0);
   /// Computes y = alpha * A^t * x + beta * y
   int MultTranspose(HypreParVector &x, HypreParVector &y,
                     double alpha = 1.0, double beta = 0.0);

   virtual void Mult(const Vector &x, Vector &y) const;

   /// Prints the locally owned rows in parallel
   void Print(const char *fname, int offi = 0, int offj = 0);
   /// Reads the matrix from a file
   void Read(const char *fname);

   /// Calls hypre's destroy function
   virtual ~HypreParMatrix();
};

/// Returns the matrix A * B
HypreParMatrix * ParMult(HypreParMatrix *A, HypreParMatrix *B);

/// Returns the matrix P^t * A * P
HypreParMatrix * RAP(HypreParMatrix *A, HypreParMatrix *P);


/// Abstract class for hypre's solvers and preconditioners
class HypreSolver : public Operator
{
protected:
   /// The linear system matrix
   HypreParMatrix *A;

   /// Right-hand side and solution vector
   mutable HypreParVector *B, *X;

   /// Was hypre's Setup function called already?
   mutable int setup_called;

public:
   HypreSolver();

   HypreSolver(HypreParMatrix *_A);

   /// Typecast to HYPRE_Solver -- return the solver
   virtual operator HYPRE_Solver() const = 0;

   /// hypre's internal Setup function
   virtual HYPRE_PtrToParSolverFcn SetupFcn() const = 0;
   /// hypre's internal Solve function
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const = 0;

   /// Solve the linear system Ax=b
   virtual void Mult(const HypreParVector &b, HypreParVector &x) const;
   virtual void Mult(const Vector &b, Vector &x) const;

   virtual ~HypreSolver();
};

/// PCG solver in hypre
class HyprePCG : public HypreSolver
{
private:
   int print_level, use_zero_initial_iterate;
   HYPRE_Solver pcg_solver;

public:
   HyprePCG(HypreParMatrix &_A);

   void SetTol(double tol);
   void SetMaxIter(int max_iter);
   void SetLogging(int logging);
   void SetPrintLevel(int print_lvl);

   /// Set the hypre solver to be used as a preconditioner
   void SetPreconditioner(HypreSolver &precond);

   /// non-hypre setting
   void SetZeroInintialIterate() { use_zero_initial_iterate = 1; }

   /// The typecast to HYPRE_Solver returns the internal pcg_solver
   virtual operator HYPRE_Solver() const { return pcg_solver; }

   /// PCG Setup function
   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRPCGSetup; }
   /// PCG Solve function
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRPCGSolve; }

   /// Solve Ax=b with hypre's PCG
   virtual void Mult(const HypreParVector &b, HypreParVector &x) const;

   virtual ~HyprePCG();
};

/// GMRES solver in hypre
class HypreGMRES : public HypreSolver
{
private:
   int print_level, use_zero_initial_iterate;
   HYPRE_Solver gmres_solver;

public:
   HypreGMRES(HypreParMatrix &_A);

   void SetTol(double tol);
   void SetMaxIter(int max_iter);
   void SetKDim(int dim);
   void SetLogging(int logging);
   void SetPrintLevel(int print_lvl);

   /// Set the hypre solver to be used as a preconditioner
   void SetPreconditioner(HypreSolver &precond);

   /// non-hypre setting
   void SetZeroInintialIterate() { use_zero_initial_iterate = 1; }

   /// The typecast to HYPRE_Solver returns the internal gmres_solver
   virtual operator HYPRE_Solver() const  { return gmres_solver; }

   /// GMRES Setup function
   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRGMRESSetup; }
   /// GMRES Solve function
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRGMRESSolve; }

   /// Solve Ax=b with hypre's GMRES
   virtual void Mult (const HypreParVector &b, HypreParVector &x) const;

   virtual ~HypreGMRES();
};

/// The identity operator as a hypre solver
class HypreIdentity : public HypreSolver
{
public:
   virtual operator HYPRE_Solver() const { return NULL; }

   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) hypre_ParKrylovIdentitySetup; }
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) hypre_ParKrylovIdentity; }

   virtual ~HypreIdentity() { }
};

/// Jacobi preconditioner in hypre
class HypreDiagScale : public HypreSolver
{
public:
   virtual operator HYPRE_Solver() const { return NULL; }

   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRDiagScaleSetup; }
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRDiagScale; }

   virtual ~HypreDiagScale() { }
};

/// The ParaSails preconditioner in hypre
class HypreParaSails : public HypreSolver
{
private:
   HYPRE_Solver sai_precond;

public:
   HypreParaSails(HypreParMatrix &A);

   void SetSymmetry(int sym);

   /// The typecast to HYPRE_Solver returns the internal sai_precond
   virtual operator HYPRE_Solver() const { return sai_precond; }

   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParaSailsSetup; }
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ParaSailsSolve; }

   virtual ~HypreParaSails();
};

/// The BoomerAMG solver in hypre
class HypreBoomerAMG : public HypreSolver
{
private:
   HYPRE_Solver amg_precond;

public:
   HypreBoomerAMG(HypreParMatrix &A);

   /** More robust options for systems, such as elastisity. Note that BoomerAMG
       assumes Ordering::byVDIM in the finite element space used to generate the
       matrix A. */
   void SetSystemsOptions(int dim);

   /// The typecast to HYPRE_Solver returns the internal amg_precond
   virtual operator HYPRE_Solver() const { return amg_precond; }

   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_BoomerAMGSetup; }
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_BoomerAMGSolve; }

   virtual ~HypreBoomerAMG();
};

class ParFiniteElementSpace;

/// The Auxiliary-space Maxwell Solver in hypre
class HypreAMS : public HypreSolver
{
private:
   HYPRE_Solver ams;

   /// Vertex coordinates
   HypreParVector *x, *y, *z;
   /// Discrete gradient matrix
   HypreParMatrix *G;

public:
   HypreAMS(HypreParMatrix &A, ParFiniteElementSpace *edge_fespace);

   /// The typecast to HYPRE_Solver returns the internal ams object
   virtual operator HYPRE_Solver() const { return ams; }

   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_AMSSetup; }
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_AMSSolve; }

   virtual ~HypreAMS();
};

#endif
