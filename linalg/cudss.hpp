#ifndef MFEM_CUDSS
#define MFEM_CUDSS

#include "../config/config.hpp"

#ifdef MFEM_USE_CUDSS
#ifdef MFEM_USE_MPI

#include "operator.hpp"
#include "cudss.h"
#include "hypre.hpp"
#include <mpi.h>

#ifdef MFEM_USE_SINGLE
#define CUDA_REAL_T CUDA_R_32F
#else
#define CUDA_REAL_T CUDA_R_64F
#endif

namespace mfem
{

/**
 * @brief cuDSS: A high-performance CUDA Library for Direct Sparse Solvers
 *
 * Interface for the distributed cuDSS solver
 */
class CuDSSSolver : public Solver
{
public:
   /// Specify the type of matrix we are applying the solver to
   enum MatType
   {
      /// CUDSS_MTYPE_GENERAL: General matrix [default] $LDU$ factorization will be computed with optional local or global pivoting
      UNSYMMETRIC = 0,
      /// CUDSS_MTYPE_SYMMETRIC: Real symmetric matrix. $LDL^{T}$ factorization will be computed with optional local pivoting
      SYMMETRIC_INDEFINITE = 1,
      /// CUDSS_MTYPE_SPD: Symmetric positive-definite matrix Cholesky factorization will be computed with optional local pivoting
      SYMMETRIC_POSITIVE_DEFINITE = 2,
   };

   /// Specify the view type of matrix we are applying the solver to
   enum MatViewType
   {
      /// CUDSS_MVIEW_FULL: Full matrix [default]
      FULL = 0,
      /// CUDSS_MVIEW_LOWER: Lower-triangular matrix (including the diagonal) All values above the main diagonal will be ignored.
      LOWER = 1,
      /// CUDSS_MVIEW_UPPER: Upper-triangular matrix (including the diagonal) All values below the main diagonal will be ignored.
      UPPER = 2,
   };

   CuDSSSolver(MPI_Comm comm);

   /**
    * @brief Set the matrix type
    *
    * Supported matrix types:
    *          CuDSSSolver::UNSYMMETRIC,
    *          CuDSSSolver::SYMMETRIC_INDEFINITE,
    *      and CuDSSSolver::SYMMETRIC_INDEFINITE,
    *
    * @param mtype_ Matrix type
    *
    * @note This method has to be called before SetOperator
    */
   void SetMatrixSymType(MatType mtype_);

   /**
    * @brief Set the matrix view type
    *
    * Supported matrix types:
    *          CuDSSSolver::FULL,
    *          CuDSSSolver::LOWER,
    *      and CuDSSSolver::UPPER,
    *
    * @param mvtype Matrix view type
    *
    * @note This method has to be called before SetOperator
    */
   void SetMatrixViewType(MatViewType mvtype);

   /**
    * @brief Set the flag controlling reuse of the symbolic factorization
    * for multiple operators
    *
    * @param reuse Flag to reuse symbolic factorization
    *
    * @note This method has to be called before repeated calls to SetOperator
    */
   void SetReorderingReuse(bool reuse);

   /**
    * @brief Set the flag controlling sort the rows of CSR matrix
    *
    * @param sort_row_ Flag to sort the row of CSR matrix
    *
    * @note This method has to be called before repeated calls to SetOperator
    */
   void SetMatrixSortRow(bool sort_row_);

   void SetOperator(const Operator &op) override;
   /**
    * @brief Solve $ y = Op^{-1} x $
    *
    * @param x RHS vector
    * @param y Solution vector
    */
   void Mult(const Vector &x, Vector &y) const override;

   /**
    * @brief Solve $ Y_i = Op^{-1} X_i $
    *
    * @param X Array of RHS vectors
    * @param Y Array of Solution vectors
    */
   void ArrayMult(const Array<const Vector *> &X, Array<Vector *> &Y) const;

   ~CuDSSSolver();

private:
   // MPI_Comm
   MPI_Comm *mpi_comm = NULL;
   // MPI rank
   int myid;

   // Parameter controlling whether or not to reuse the symbolic factorization
   // for multiple calls to SetOperator
   bool reorder_reuse = false;

   // Parameter controlling whether or not to sort the row of CSR matrix
   bool sort_row = false;

   // Matrix type
   MatType mtype = MatType::UNSYMMETRIC;
   // Parameter controlling the matrix type
   cudssMatrixType_t mat_type = CUDSS_MTYPE_GENERAL;

   mutable int nrhs = 0; // the number of the RHSs
   int n = 0; // global number of rows

   int nnz = 0; // the number of non zeros
   int row_start = 0;  // the first row index in CSR matrix operator
   int row_end = 0;   // the end row index in CSR matrix operator
   int n_loc = 0;    // the number of the rows in CSR matrix operator

   // copy and keep the I and J arrays in device memory when skipping analysis
   // phase
   int *csr_offsets_d = NULL; // copy and keep I in device
   int *csr_columns_d = NULL; // copy and keep J in device

   // cuDSS object specifies available matrix types for sparse matrices
   cudssMatrixViewType_t mview = CUDSS_MVIEW_FULL;

   // cuDSS objects storage for sparse matrix Ac, RHS yc and solution xc
   std::unique_ptr<cudssMatrix_t> Ac;
   mutable cudssMatrix_t xc, yc;

   // cuDSS object holds the cuDSS library context
   mutable cudssHandle_t handle;
   // cuDSS object stores configuration settings for the solver
   mutable cudssConfig_t solverConfig;
   // cuDSS object holds internal data
   mutable cudssData_t solverData;

   /// Method for initialization
   void Init(MPI_Comm comm_);

   /// Method for the initialization of cudss Handle
   void InitHandle();

   /// Method for configuring storage for distributed/centralized RHS and
   /// solution
   void InitRhsSol(int nrhs_) const;
};

} // namespace mfem

#endif // MFEM_USE_MPI
#endif // MFEM_USE_CUDSS
#endif // MFEM_CUDSS