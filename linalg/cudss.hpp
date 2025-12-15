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
 * @brief A simple singleton class for cuDSS handle.
 * 1) Call Init() to initialize the cuDSS library.
 * 2) Call Finalize() at destruction.
 */
class CuDSSHandle
{
public:
   /// Initialize the cuDSS library
   static void Init();

   /// Finalize the cuDSS library
   static void Finalize();

   /// Get the cuDSS handle
   static cudssHandle_t Get();

   /// Get the initialization state of cuDSS handle
   static bool IsInitialized() { return state == State::INITIALIZED; }

private:
   /// Private constructor for singleton pattern
   CuDSSHandle() = default;

   /// Copy constructor. Deleted.
   CuDSSHandle(const CuDSSHandle &) = delete;

   /// Move constructor. Deleted.
   CuDSSHandle(CuDSSHandle &&) = delete;

   /// The singleton destructor (called at program exit) finalizes cudss handle.
   ~CuDSSHandle() {Finalize();}

   /// State of the cuDSS library
   enum class State { UNINITIALIZED, INITIALIZED };

   /// Tracks whether CuDSSHandle was initialized or finalized by this class.
   static State state;

   /// The global cuDSS handle
   static cudssHandle_t global_cudss_handle;
};

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
    * @brief Constructor with MPI_Comm parameter.
    */
   CuDSSSolver(MPI_Comm comm);

   // Note: CuDSSSolver disables the move copy constructor and move assignment
   // operator
   CuDSSSolver(CuDSSSolver &&) = delete;
   CuDSSSolver &operator=(CuDSSSolver &&) = delete;

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
   MPI_Comm mpi_comm = MPI_COMM_NULL;

   // Parameter controlling whether or not to reuse the symbolic factorization
   // for multiple calls to SetOperator
   bool reorder_reuse = false;

   // Parameter controlling whether or not to sort the row of CSR matrix
   bool sort_row = false;

   // Matrix type
   MatType mtype = MatType::NONSYMMETRIC;
   // Parameter controlling the matrix type
   cudssMatrixType_t mat_type = CUDSS_MTYPE_GENERAL;

   mutable int nrhs = 0; // the number of the RHSs
   int n_global = 0;     // global number of rows

   int nnz = 0; // the number of non zeros
   int row_start = 0;  // the first row index in CSR matrix operator
   int row_end = 0;   // the end row index in CSR matrix operator
   int n_loc = 0;    // the number of the rows in CSR matrix operator

   // copy and keep the I and J arrays in device memory when skipping analysis
   // phase
   void *csr_offsets_d = NULL; // copy and keep I in device
   void *csr_columns_d = NULL; // copy and keep J in device

   // cuDSS object specifies available matrix types for sparse matrices
   cudssMatrixViewType_t mview = CUDSS_MVIEW_FULL;

   // cuDSS objects storage for sparse matrix Ac, RHS yc and solution xc
   std::unique_ptr<cudssMatrix_t> Ac;
   mutable cudssMatrix_t xc, yc;

   // cuDSS object stores configuration settings for the solver
   mutable cudssConfig_t solverConfig;
   // cuDSS object holds internal data
   mutable cudssData_t solverData;

   /// Method for the initialization of cuDSS solver
   void Init();

   /// Method for configuring storage for distributed/centralized RHS and
   /// solution
   void SetNumRHS(int nrhs_) const;
};

} // namespace mfem

#endif // MFEM_USE_MPI
#endif // MFEM_USE_CUDSS
#endif // MFEM_CUDSS
