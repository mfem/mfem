#ifndef MFEM_CUDSS
#define MFEM_CUDSS

#include "../config/config.hpp"

#ifdef MFEM_USE_CUDSS

#include "cudss.h"
#include <memory>

#ifdef MFEM_USE_MPI
#include <mpi.h>
#include "hypre.hpp"
#else
#include "operator.hpp"
#include "sparsemat.hpp"
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
   enum class MatType
   {
      /// CUDSS_MTYPE_GENERAL: General matrix [default].
      NONSYMMETRIC = 0,
      /// CUDSS_MTYPE_SYMMETRIC: Real symmetric matrix.
      SYMMETRIC_INDEFINITE = 1,
      /// CUDSS_MTYPE_SPD: Symmetric positive-definite matrix.
      SYMMETRIC_POSITIVE_DEFINITE = 2,
   };

   /// Specify the view type of matrix we are applying the solver to
   enum class MatViewType
   {
      /// CUDSS_MVIEW_FULL: Full matrix [default]
      FULL = 0,
      /// CUDSS_MVIEW_LOWER: Lower-triangular matrix (including the diagonal).
      LOWER = 1,
      /// CUDSS_MVIEW_UPPER: Upper-triangular matrix (including the diagonal).
      UPPER = 2,
   };

   /**
    * @brief Summary statistics reported by cuDSS after each solver phase.
    */
   struct CuDSSSummary
   {
      /// Times are in seconds. analysis_time_seconds == 0 means analysis was skipped.
      double analysis_time_seconds = -1.0;
      double factorization_time_seconds = -1.0;
      double solve_time_seconds = -1.0;

      /// Factor statistics.
      size_t lu_nnz = 0;
      size_t input_nnz = 0;
      size_t num_pivots = 0;

      /// Estimated memory from cuDSS analysis.
      size_t est_device_mem_permanent_bytes = 0;
      size_t est_device_mem_peak_bytes = 0;
      size_t est_host_mem_permanent_bytes = 0;
      size_t est_host_mem_peak_bytes = 0;

   #ifdef MFEM_USE_MPI
      /// Return the summary reduced across all ranks in @a comm.
      /// Timing fields use MPI_MAX; count/memory fields use MPI_SUM.
      CuDSSSummary GetGlobalSummary(MPI_Comm comm) const;
   #else
      /// Serial fallback: return this summary unchanged.
      CuDSSSummary GetGlobalSummary() const { return *this; }
   #endif

      /// Print this summary to mfem::out.
      void PrintSummary() const;
   };

   /**
    * @brief Constructor.
    */
   CuDSSSolver();

#ifdef MFEM_USE_MPI
   /**
    * @brief Constructor with MPI_Comm parameter.
    */
   CuDSSSolver(MPI_Comm comm);
#endif

   // Note: CuDSSSolver disables the move copy constructor and move assignment
   // operator
   CuDSSSolver(CuDSSSolver &&) = delete;
   CuDSSSolver &operator=(CuDSSSolver &&) = delete;

   /**
    * @brief Set the matrix type
    *
    * Supported matrix types:
    *          CuDSSSolver::NONSYMMETRIC,
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

   /// Return a const reference to the latest solver statistics.
   const CuDSSSummary &GetSummary() const { return summary; }

   /// Return summary statistics reduced across all MPI ranks (or local in serial).
   CuDSSSummary GetGlobalSummary() const;

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
#ifdef MFEM_USE_MPI
   // MPI_Comm
   MPI_Comm mpi_comm = MPI_COMM_NULL;

   int row_start = 0;  // the first row index in CSR matrix operator
   int row_end = 0;    // the end row index in CSR matrix operator
#endif

   // Parameter controlling whether or not to reuse the symbolic factorization
   // for multiple calls to SetOperator
   bool reorder_reuse = false;

   // Parameter controlling whether or not to sort the row of CSR matrix
   bool sort_row = false;

   // Matrix type
   MatType mtype = MatType::NONSYMMETRIC;
   // Parameter controlling the matrix type
   cudssMatrixType_t mat_type = CUDSS_MTYPE_GENERAL;

   int n_global = 0;      // global number of rows
   int n_loc = 0;         // the number of the rows in CSR matrix operator

   mutable int nrhs = 0;  // the number of the RHSs
   int nnz = 0;        // the number of non zeros

   // copy and keep the I and J arrays in device memory when skipping analysis
   // phase
   void *csr_offsets_d = NULL; // copy and keep I in device
   void *csr_columns_d = NULL; // copy and keep J in device

   // cuDSS object specifies available matrix types for sparse matrices
   cudssMatrixViewType_t mview = CUDSS_MVIEW_FULL;

   // cuDSS objects storage for sparse matrix Ac, RHS yc and solution xc
   std::unique_ptr<cudssMatrix_t> Ac;
   mutable cudssMatrix_t xc, yc;

   // common for all cuDSS solver instances.
   // cuDSS object holds the cuDSS library context
   static cudssHandle_t handle;
   static int CuDSSSolverCount;

   // cuDSS object stores configuration settings for the solver
   mutable cudssConfig_t solverConfig;
   // cuDSS object holds internal data
   mutable cudssData_t solverData;

   // Solver statistics, updated after each phase
   mutable CuDSSSummary summary;

   /// Method for configuring storage for distributed/centralized RHS and
   /// solution
   void SetNumRHS(int nrhs_) const;

#ifdef MFEM_USE_MPI
   /**
    * @brief Set the HypreParMatrix object
    *
    * @param op HypreParMatrix object
    *
    * @note This method is called inside SetOperator
    */
   void SetMatrix(const HypreParMatrix &op);
#endif

   /**
    * @brief Set the SparseMatrix object
    *
    * @param op SparseMatrix object
    *
    * @note This method is called inside SetOperator
    */
   void SetMatrix(const SparseMatrix &op);
};

} // namespace mfem

#endif // MFEM_USE_CUDSS
#endif // MFEM_CUDSS
