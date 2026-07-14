// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "cudss.hpp"
#include "../general/communication.hpp"
#include <chrono>
#include <iomanip>
#include <string>

#ifdef MFEM_USE_CUDSS

#ifdef MFEM_USE_SINGLE
#define CUDA_REAL_T CUDA_R_32F
#else
#define CUDA_REAL_T CUDA_R_64F
#endif

// Define a cuDSS error check macro, MFEM_CUDSS_CHECK(x), where x returns/is of
// type 'cudssStatus_t'. This macro evaluates 'x' and raises an error if the
// result is not CUDSS_STATUS_SUCCESS.
#define MFEM_CUDSS_CHECK(x)                                          \
  do {                                                               \
    cudssStatus_t mfem_err_internal_var_name = (x);                  \
    if (mfem_err_internal_var_name != CUDSS_STATUS_SUCCESS) {        \
      ::mfem::mfem_cudss_error(mfem_err_internal_var_name, #x,       \
                               _MFEM_FUNC_NAME, __FILE__, __LINE__); \
    }                                                                \
  } while (0)

namespace mfem
{
// Forward declaration needed by MFEM_CUDSS_CHECK uses before the definition.
static void mfem_cudss_error(cudssStatus_t status, const char *expr,
                             const char *func, const char *file, int line);

cudssHandle_t CuDSSSolver::handle = nullptr;
int CuDSSSolver::CuDSSSolverCount = 0;

constexpr double BytesToMiB(size_t bytes)
{
   return static_cast<double>(bytes) / static_cast<double>(1u << 20);
}

static size_t ToSizeTOrZero(int64_t value)
{
   return (value > 0) ? static_cast<size_t>(value) : 0;
}

static void UpdateMemoryEstimates(cudssHandle_t handle,
                                  cudssData_t solverData,
                                  CuDSSSolver::CuDSSSummary &summary)
{
   // Reset in case estimates are unavailable for the current operator/config.
   summary.est_device_mem_permanent_bytes = 0;
   summary.est_device_mem_peak_bytes = 0;
   summary.est_host_mem_permanent_bytes = 0;
   summary.est_host_mem_peak_bytes = 0;

   int64_t estimates[16] = {0};
   cudssStatus_t status = cudssDataGet(handle, solverData,
                                       CUDSS_DATA_MEMORY_ESTIMATES,
                                       estimates, sizeof(estimates), nullptr);
   if (status == CUDSS_STATUS_SUCCESS)
   {
      summary.est_device_mem_permanent_bytes = ToSizeTOrZero(estimates[0]);
      summary.est_device_mem_peak_bytes = ToSizeTOrZero(estimates[1]);
      summary.est_host_mem_permanent_bytes = ToSizeTOrZero(estimates[2]);
      summary.est_host_mem_peak_bytes = ToSizeTOrZero(estimates[3]);
   }
   else if (status == CUDSS_STATUS_NOT_SUPPORTED) { }
   else
   {
      MFEM_CUDSS_CHECK(status);
   }
}

static bool HasMemoryEstimates(const CuDSSSolver::CuDSSSummary &summary)
{
   return summary.est_device_mem_permanent_bytes ||
          summary.est_device_mem_peak_bytes ||
          summary.est_host_mem_permanent_bytes ||
          summary.est_host_mem_peak_bytes;
}

#ifdef MFEM_USE_MPI
CuDSSSolver::CuDSSSummary
CuDSSSolver::CuDSSSummary::GetGlobalSummary(MPI_Comm comm) const
{
   if (comm == MPI_COMM_NULL) { return *this; }

   CuDSSSummary global = *this;

   MPI_Allreduce(&analysis_time_seconds, &global.analysis_time_seconds, 1,
                 MPI_DOUBLE, MPI_MAX, comm);
   MPI_Allreduce(&factorization_time_seconds, &global.factorization_time_seconds,
                 1, MPI_DOUBLE, MPI_MAX, comm);
   MPI_Allreduce(&solve_time_seconds, &global.solve_time_seconds, 1,
                 MPI_DOUBLE, MPI_MAX, comm);

   auto sum_size_t = [comm](size_t local_value) -> size_t
   {
      unsigned long long local_ull = static_cast<unsigned long long>(local_value);
      unsigned long long global_ull = 0;
      MPI_Allreduce(&local_ull, &global_ull, 1, MPI_UNSIGNED_LONG_LONG,
                    MPI_SUM, comm);
      return static_cast<size_t>(global_ull);
   };

   global.lu_nnz = sum_size_t(lu_nnz);
   global.input_nnz = sum_size_t(input_nnz);
   global.num_pivots = sum_size_t(num_pivots);
   global.est_device_mem_permanent_bytes =
      sum_size_t(est_device_mem_permanent_bytes);
   global.est_device_mem_peak_bytes = sum_size_t(est_device_mem_peak_bytes);
   global.est_host_mem_permanent_bytes = sum_size_t(est_host_mem_permanent_bytes);
   global.est_host_mem_peak_bytes = sum_size_t(est_host_mem_peak_bytes);

   return global;
}
#endif

// Function used by the macro MFEM_CUDSS_CHECK.
static void mfem_cudss_error(cudssStatus_t status, const char *expr,
                             const char *func, const char *file, int line)
{
   mfem::err << "\n\nCUDSS error: (" << expr << ") failed with error:\n --> "
             << "CUDSS call ended unsuccessfully"
             << " [code: " << static_cast<int>(status) << ']'
             << "\n ... in function: " << func << "\n ... in file: " << file
             << ':' << line << '\n';
   mfem_error();
}

CuDSSSolver::CuDSSSolver() { InitCuDSS(); }

#ifdef MFEM_USE_MPI
CuDSSSolver::CuDSSSolver(MPI_Comm comm_) : mpi_comm(comm_)
{
   InitCuDSS();

   // NOTE: Set the communication layer to NULL so that cuDSS picks it
   // from the environment variable "CUDSS_COMM_LIB"
   const char* comm_lib = GetEnv("CUDSS_COMM_LIB");
#ifdef MFEM_CUDSS_COMM_LIB
   if (comm_lib == nullptr)
   {
      comm_lib = MFEM_CUDSS_COMM_LIB;
   }
#endif
   MFEM_CUDSS_CHECK(cudssSetCommLayer(handle, comm_lib));

   MFEM_CUDSS_CHECK(cudssDataSet(handle, solverData, CUDSS_DATA_COMM,
                                 &mpi_comm, sizeof(MPI_Comm *)));
}
#endif // MFEM_USE_MPI

CuDSSSolver::~CuDSSSolver()
{
   // Destroy the system Matrix, RHS vector and solution vector
   if (Ac)
   {
      MFEM_CUDSS_CHECK(cudssMatrixDestroy(*Ac));
      MFEM_CUDSS_CHECK(cudssMatrixDestroy(xc));
      MFEM_CUDSS_CHECK(cudssMatrixDestroy(yc));
   }

   // Destroy the cuDSS handle, solver config and solver data
   MFEM_CUDSS_CHECK(cudssDataDestroy(handle, solverData));
   MFEM_CUDSS_CHECK(cudssConfigDestroy(solverConfig));


   MFEM_CUDSS_CHECK(cudssDestroy(handle));
   handle = nullptr;


   if (csr_offsets_d != NULL)
   {
      CuMemFree(csr_offsets_d);
   }

   if (csr_columns_d != NULL)
   {
      CuMemFree(csr_columns_d);
   }

   if (csr_values_d != NULL)
   {
      CuMemFree(csr_values_d);
   }
}

void CuDSSSolver::InitCuDSS()
{
   // Create the cuDSS handle
   MFEM_CUDSS_CHECK(cudssCreate(&handle));

#ifdef MFEM_USE_OPENMP
   // NOTE: Set the threading layer library name to NULL so that cuDSS picks
   // it from the environment variable "CUDSS_THREADING_LIB"
   const char* threading_lib = GetEnv("CUDSS_THREADING_LIB");
#ifdef MFEM_CUDSS_THREADING_LIB
   if (threading_lib == nullptr)
   {
      threading_lib = MFEM_CUDSS_THREADING_LIB;
   }
#endif
   MFEM_CUDSS_CHECK(cudssSetThreadingLayer(handle, threading_lib));
#endif  // MFEM_USE_OPENMP

   // Create the solver configuration and data objects
   MFEM_CUDSS_CHECK(cudssConfigCreate(&solverConfig));
   MFEM_CUDSS_CHECK(cudssDataCreate(handle, &solverData));
}

void CuDSSSolver::SetMatrixSymType(MatType mtype_)
{
   switch (mtype_)
   {
      case MatType::SYMMETRIC_INDEFINITE:
         mat_type = CUDSS_MTYPE_SYMMETRIC;
         break;
      case MatType::SYMMETRIC_POSITIVE_DEFINITE:
         mat_type = CUDSS_MTYPE_SPD;
         break;
      default:
         mat_type = CUDSS_MTYPE_GENERAL;
         break;
   }
}

void CuDSSSolver::SetMatrixViewType(MatViewType mvtype_)
{
   // If the MatType is NONSYMMETRIC, the matrix view type must be FULL.
   if (mat_type == CUDSS_MTYPE_GENERAL)
   {
      mview = CUDSS_MVIEW_FULL;
      return;
   }

   // If the matrix is symmetric, the following view type will be optional.
   switch (mvtype_)
   {
      case MatViewType::LOWER:
         mview = CUDSS_MVIEW_LOWER;
         break;
      case MatViewType::UPPER:
         mview = CUDSS_MVIEW_UPPER;
         break;
      default:
         mview = CUDSS_MVIEW_FULL;
         break;
   }
}

void CuDSSSolver::SetReorderingReuse(bool reuse)
{
   MFEM_VERIFY(Ac == nullptr,
               "Set reordering reuse before setting the operator!");
   reorder_reuse = reuse;
}

#ifdef MFEM_USE_MPI
void CuDSSSolver::SetMatrix(const HypreParMatrix &op)
{
   bool cuDSSObjectInitialized = (Ac != nullptr);

   hypre_ParCSRMatrix *parcsr_op = op;
   op.HypreRead();
   hypre_CSRMatrix *csr_op = hypre_MergeDiagAndOffd(parcsr_op);
   op.HypreRead();
#if MFEM_HYPRE_VERSION >= 21600
   hypre_CSRMatrixBigJtoJ(csr_op);
#endif

   // Parameters of the Operator
   n_loc = height;  // Equal to the csr_op->num_rows
   n_global = internal::to_int(parcsr_op->global_num_rows);
   row_start = parcsr_op->first_row_index;
   row_end = row_start + n_loc - 1;
   MFEM_VERIFY(!cuDSSObjectInitialized || !reorder_reuse ||
               (reorder_reuse && (nnz == csr_op->num_nonzeros)),
               "Inconsistent new matrix pattern!");
   nnz = csr_op->num_nonzeros;

   SetMatrixCuDSS(csr_op->i, csr_op->j, csr_op->data);
   hypre_CSRMatrixDestroy(csr_op);
}
#endif // MFEM_USE_MPI

void CuDSSSolver::SetMatrix(const SparseMatrix &op)
{
   bool cuDSSObjectInitialized = (Ac != nullptr);

   // Parameters of the Operator
   MFEM_VERIFY(!cuDSSObjectInitialized || !reorder_reuse ||
               (reorder_reuse && (nnz == op.NumNonZeroElems())),
               "Inconsistent new matrix pattern!");

   SparseMatrix *A = const_cast<SparseMatrix *>(&op);

   nnz = A->NumNonZeroElems();
   n_global = height;  // Equal to the height in serial
   n_loc = height;     // Equal to the height in serial

   int *csr_offsets = const_cast<int *>(A->ReadI());
   int *csr_columns = const_cast<int *>(A->ReadJ());
   real_t *csr_values = const_cast<real_t *>(A->ReadData());

   SetMatrixCuDSS(csr_offsets, csr_columns, csr_values);
}

void CuDSSSolver::SetMatrixCuDSS(int *csr_offsets, int *csr_columns,
                                 real_t *csr_values)
{
   bool cuDSSObjectInitialized = (Ac != nullptr);
   // Initial the cudssMatrix objects
   if (!cuDSSObjectInitialized)
   {
      // Set the cudssMatrix object of csr operator
      Ac = std::make_unique<cudssMatrix_t>();
      // Create empty RHS and solution vectors
      SetNumRHS(1);
      // Allocate device memory for csr values
      CuMemAlloc(&csr_values_d, nnz * sizeof(real_t));
   }

   CuMemcpyDtoD(csr_values_d, csr_values, nnz * sizeof(real_t));

   // New cuDSS CSR matrix object and analysis or reuse the one from a previous
   // matrix
   if (!cuDSSObjectInitialized || !reorder_reuse)
   {
      if (reorder_reuse)  // !cuDSSObjectInitialized && reorder_reuse
      {
         // NOTE: For CuDSS solver to reuse the reordering (skipping analysis
         // phase), it needs to access the I and J arrays of the **initial**
         // matrix. Therefore, we need to copy and keep I and J in device memory.
         CuMemAlloc(&csr_offsets_d, (n_loc + 1) * sizeof(int));
         CuMemAlloc(&csr_columns_d, nnz * sizeof(int));

         CuMemcpyDtoD(csr_offsets_d, csr_offsets, (n_loc + 1) * sizeof(int));
         CuMemcpyDtoD(csr_columns_d, csr_columns, nnz * sizeof(int));

         MFEM_CUDSS_CHECK(
            cudssMatrixCreateCsr(
               Ac.get(), n_global, n_global, nnz, csr_offsets_d, NULL,
               csr_columns_d, csr_values_d, CUDA_R_32I, CUDA_REAL_T, mat_type, mview,
               CUDSS_BASE_ZERO));
      }
      else    // !reorder_reuse
      {
         if (cuDSSObjectInitialized)
         {
            MFEM_CUDSS_CHECK(cudssMatrixDestroy(*Ac));
         }
         MFEM_CUDSS_CHECK(
            cudssMatrixCreateCsr(
               Ac.get(), n_global, n_global, nnz, csr_offsets, NULL, csr_columns,
               csr_values_d, CUDA_R_32I, CUDA_REAL_T, mat_type, mview,
               CUDSS_BASE_ZERO));
      }
#ifdef MFEM_USE_MPI
      if (Mpi::IsInitialized())
      {
         MFEM_CUDSS_CHECK(cudssMatrixSetDistributionRow1d(*Ac, row_start, row_end));
      }
#endif
      // Analysis
      {
         auto t0 = std::chrono::steady_clock::now();
         MFEM_CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig,
                                       solverData, *Ac, yc, xc));
         auto t1 = std::chrono::steady_clock::now();
         summary.analysis_time_seconds = std::chrono::duration<double>(t1 - t0).count();
         UpdateMemoryEstimates(handle, solverData, summary);
      }
   }
   else    // cuDSSObjectInitialized && reorder_reuse
   {
      summary.analysis_time_seconds =
         0.0; // analysis skipped (reusing symbolic factorization)
      // NOTE: When reusing analysis result, we only update the Data array,
      // without changing the I and J arrays.
      MFEM_CUDSS_CHECK(cudssMatrixSetValues(*Ac, csr_values_d));
   }

   // Factorization
   {
      auto t0 = std::chrono::steady_clock::now();
      MFEM_CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig,
                                    solverData, *Ac, yc, xc));
      auto t1 = std::chrono::steady_clock::now();
      summary.factorization_time_seconds = std::chrono::duration<double>
                                           (t1 - t0).count();
   }

   // Query solver statistics
   summary.input_nnz = nnz;
   int64_t lu_nnz = 0;
   int num_pivots = 0;
   int info = 0;
   cudssDataGet(handle, solverData, CUDSS_DATA_LU_NNZ,
                &lu_nnz, sizeof(int64_t), nullptr);
   cudssDataGet(handle, solverData, CUDSS_DATA_NPIVOTS,
                &num_pivots, sizeof(int), nullptr);
   cudssDataGet(handle, solverData, CUDSS_DATA_INFO,
                &info, sizeof(int), nullptr);
   MFEM_VERIFY(info == 0, "cuDSS factorization info returned nonzero status: "
               << info);
   summary.lu_nnz = (lu_nnz > 0) ? static_cast<size_t>(lu_nnz) : 0;
   summary.num_pivots = (num_pivots > 0) ? static_cast<size_t>(num_pivots) : 0;
}

void CuDSSSolver::SetOperator(const Operator &op)
{
   bool cuDSSObjectInitialized = (Ac != nullptr);
   MFEM_VERIFY(
      !cuDSSObjectInitialized || (height == op.Height() && width == op.Width()),
      "Inconsistent new matrix size!");
   height = op.Height();
   width = op.Width();
   if (const SparseMatrix *A = dynamic_cast<const SparseMatrix *>(&op))
   {
      SetMatrix(*A);
   }
#ifdef MFEM_USE_MPI
   else if (const HypreParMatrix *A =
               dynamic_cast<const HypreParMatrix *>(&op))
   {
      SetMatrix(*A);
   }
#endif // MFEM_USE_MPI
   else
   {
      MFEM_ABORT("Unsupported Operator Type \n");
   }
}

void CuDSSSolver::SetNumRHS(int nrhs_) const
{
   if (nrhs != nrhs_)
   {
      if (nrhs > 0)
      {
         // Destroy the previous RHS vector and solution vector
         MFEM_CUDSS_CHECK(cudssMatrixDestroy(xc));
         MFEM_CUDSS_CHECK(cudssMatrixDestroy(yc));
      }
      // Create empty RHS and solution vectors
      MFEM_CUDSS_CHECK(cudssMatrixCreateDn(&xc, n_global, nrhs_, n_global, NULL,
                                           CUDA_REAL_T, CUDSS_LAYOUT_COL_MAJOR));

      MFEM_CUDSS_CHECK(cudssMatrixCreateDn(&yc, n_global, nrhs_, n_global, NULL,
                                           CUDA_REAL_T, CUDSS_LAYOUT_COL_MAJOR));

#ifdef MFEM_USE_MPI
      MFEM_CUDSS_CHECK(cudssMatrixSetDistributionRow1d(xc, row_start, row_end));
      MFEM_CUDSS_CHECK(cudssMatrixSetDistributionRow1d(yc, row_start, row_end));
#endif // MFEM_USE_MPI
   }
   nrhs = nrhs_;
}

void CuDSSSolver::Mult(const Vector &x, Vector &y) const
{
   Array<const Vector *> X(1);
   Array<Vector *> Y(1);
   X[0] = &x;
   Y[0] = &y;
   ArrayMult(X, Y);
}

void CuDSSSolver::ArrayMult(const Array<const Vector *> &X,
                            Array<Vector *> &Y) const
{
   SetNumRHS(X.Size());

   Vector RHS, SOL;

   if (nrhs == 1)
   {
      RHS.MakeRef(*(const_cast<Vector *>(X[0])), 0, X[0]->Size());
      SOL.MakeRef(*Y[0], 0, Y[0]->Size());
   }
   else
   {
      // NOTE: RHS must have **global** num_rows and nrhs columns
      RHS.SetSize(nrhs * n_global, *X[0]);
      for (int i = 0; i < nrhs; i++)
      {
         Vector s(RHS, i * n_global, n_loc);
         s = *X[i];
      }

      // NOTE: SOL must have **global** num_rows and nrhs columns
      SOL.SetSize(nrhs * n_global, *Y[0]);
   }

   MFEM_CUDSS_CHECK(cudssMatrixSetValues(xc, const_cast<real_t *>(RHS.Read())));
   MFEM_CUDSS_CHECK(cudssMatrixSetValues(yc, SOL.Write()));

   // Solve
   {
      auto t0 = std::chrono::steady_clock::now();
      MFEM_CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig,
                                    solverData, *Ac, yc, xc));
      auto t1 = std::chrono::steady_clock::now();
      summary.solve_time_seconds = std::chrono::duration<double>(t1 - t0).count();
   }
   int info = 0;
   cudssDataGet(handle, solverData, CUDSS_DATA_INFO,
                &info, sizeof(int), nullptr);
   MFEM_VERIFY(info == 0, "cuDSS solve info returned nonzero status: " << info);

   if (nrhs == 1)
   {
      SOL.SyncAliasMemory(*Y[0]);
   }

   if (nrhs > 1)
   {
      // Get solution for each right-hand side
      for (int i = 0; i < nrhs; i++)
      {
         Vector s(SOL, i * n_global, n_loc);
         *Y[i] = s;
      }
   }
}

CuDSSSolver::CuDSSSummary CuDSSSolver::GetGlobalSummary() const
{
#ifdef MFEM_USE_MPI
   return summary.GetGlobalSummary(mpi_comm);
#else
   return summary.GetGlobalSummary();
#endif
}

void CuDSSSolver::CuDSSSummary::PrintSummary() const
{
   mfem::out << "\nCuDSSSolver statistics:\n";
   if (analysis_time_seconds == 0.0)
   {
      mfem::out <<
                "  Analysis time:       (skipped — reusing symbolic factorization)\n";
   }
   else if (analysis_time_seconds > 0.0)
      mfem::out << "  Analysis time:       "
                << std::fixed << std::setprecision(4) << analysis_time_seconds << " s\n";
   if (factorization_time_seconds >= 0.0)
      mfem::out << "  Factorization time:  "
                << std::fixed << std::setprecision(4) << factorization_time_seconds << " s\n";
   if (solve_time_seconds >= 0.0)
      mfem::out << "  Solve time:          "
                << std::fixed << std::setprecision(4) << solve_time_seconds << " s\n";
   mfem::out << "  LU nnz:              " << lu_nnz
             << "  (fill ratio: "
             << std::fixed << std::setprecision(2)
             << (input_nnz > 0 ? (double)lu_nnz / input_nnz : 0.0) << "x)\n";
   mfem::out << "  Pivots used:         " << num_pivots << "\n";
   if (HasMemoryEstimates(*this))
   {
      mfem::out << "  Est. device memory:  "
                << std::fixed << std::setprecision(1)
                << BytesToMiB(est_device_mem_permanent_bytes)
                << " MiB perm, "
                << BytesToMiB(est_device_mem_peak_bytes)
                << " MiB peak\n";
      mfem::out << "  Est. host memory:    "
                << std::fixed << std::setprecision(1)
                << BytesToMiB(est_host_mem_permanent_bytes)
                << " MiB perm, "
                << BytesToMiB(est_host_mem_peak_bytes)
                << " MiB peak\n";
   }
}

} // namespace mfem
#endif // MFEM_USE_CUDSS
