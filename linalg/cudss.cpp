#include "cudss.hpp"
#include "../general/communication.hpp"
#include <string>

#ifdef MFEM_USE_CUDSS
#ifdef MFEM_USE_MPI

namespace mfem
{
namespace
{
    void CheckCudssAPI(cudssStatus_t status, std::string msg)
    {
        MFEM_VERIFY(status == CUDSS_STATUS_SUCCESS,
                    "mpi rank [" << Mpi::WorldRank() << "] "
                                 << "CUDSS call ended unsuccessfully with status = "
                                 << static_cast<int>(status)
                                 << ", details : " << msg << " Failed!");
#ifdef MFEM_DEBUG
        mfem::out << "mpi rank [" << Mpi::WorldRank() << "] "
                  << "CUDSS call ended successfully! "
                  << "Details : " << msg << " successfully!\n";
#endif
    }

    void CheckCudaAPI(cudaError_t status, std::string msg)
    {
        MFEM_VERIFY(status == cudaSuccess,
                    "mpi rank [" << Mpi::WorldRank() << "] "
                                 << "CUDA call ended unsuccessfully with status = "
                                 << static_cast<int>(status)
                                 << ", details : " << msg << " Failed!");
#ifdef MFEM_DEBUG
        mfem::out << "mpi rank [" << Mpi::WorldRank() << "] "
                  << "CUDA call ended successfully! "
                  << "Details : " << msg << " successfully!\n";
#endif
    }
}

CuDSSSolver::CuDSSSolver(MPI_Comm comm_)
{
    Init(comm_);
    InitHandle();
}

void CuDSSSolver::Init(MPI_Comm comm_)
{
    mpi_comm = (MPI_Comm *)malloc(sizeof(MPI_Comm));
    mpi_comm[0] = comm_; // MPI_COMM_WORLD
}

void CuDSSSolver::InitHandle()
{
    // Checkout the cudss API
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;

    status = cudssCreate(&handle);
    CheckCudssAPI(status, "cudssCreate");
    status = cudssSetCommLayer(handle, NULL);
    CheckCudssAPI(status, "cudssSetCommLayer");

    status = cudssConfigCreate(&solverConfig);
    CheckCudssAPI(status, "cudssConfigCreate");
    status = cudssDataCreate(handle, &solverData);
    CheckCudssAPI(status, "cudssDataCreate");

    status = cudssDataSet(handle, solverData, CUDSS_DATA_COMM, mpi_comm, sizeof(MPI_Comm *));
    CheckCudssAPI(status, "cudssDataSet for CUDSS_DATA_COMM");
}

CuDSSSolver::~CuDSSSolver()
{
    // Checkout the cudss API
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;
    cudaError_t cuda_status = cudaSuccess;

    // Destroy the system Matrix
    if(Ac.get())
    {
        status = cudssMatrixDestroy(*Ac);
        CheckCudssAPI(status, "cudssMatrixDestroy for Operator");
    }
    // Destroy the empty RHS vector and solution vector
    status = cudssMatrixDestroy(xc);
    CheckCudssAPI(status, "cudssMatrixDestroy for RHS");
    status = cudssMatrixDestroy(yc);
    CheckCudssAPI(status, "cudssMatrixDestroy for solution Vector");

    // Destroy the cudss handle, solver config and solver data
    status = cudssDataDestroy(handle, solverData);
    CheckCudssAPI(status, "cudssDataDestroy");
    status = cudssConfigDestroy(solverConfig);
    CheckCudssAPI(status, "cudssConfigDestroy");
    status = cudssDestroy(handle);
    CheckCudssAPI(status, "cudssDestroy");

    if (mpi_comm != NULL)
    {
        free(mpi_comm);
    }

    if (csr_offsets_d !=NULL)
    {
        cuda_status = cudaFree(csr_offsets_d);
        CheckCudaAPI(cuda_status, "cudaFree for CSR I vector");
    }

    if(csr_columns_d!=NULL)
    {
        cuda_status = cudaFree(csr_columns_d);
        CheckCudaAPI(cuda_status, "cudaFree for CSR J vector");
    }
}

void CuDSSSolver::SetMatrixSymType(MatType mtype_)
{
    mtype = mtype_;

    switch (mtype)
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
    // If the MatType is UNSYMMETRIC, the matrix view type must be FULL.
    if (mtype == MatType::UNSYMMETRIC)
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
    reorder_reuse = reuse;
}

void CuDSSSolver::SetMatrixSortRow(bool sort_row_)
{
    sort_row = sort_row_;
}

void CuDSSSolver::SetOperator(const Operator &op)
{
    bool cuDSSObjectInitialized = (Ac.get() != nullptr);
    MFEM_VERIFY(!cuDSSObjectInitialized ||
                    (height == op.Height() && width == op.Width()),
                "CuDSSSolver::SetOperator: Inconsistent new matrix size!");
    height = op.Height();
    width = op.Width();

    const HypreParMatrix *A = dynamic_cast<const HypreParMatrix *>(&op);
    MFEM_VERIFY(A, "Not a compatible matrix type");

    hypre_ParCSRMatrix *parcsr_op = (hypre_ParCSRMatrix *)const_cast<HypreParMatrix &>(*A);
    hypre_CSRMatrix *csr_op = hypre_MergeDiagAndOffd(parcsr_op);
    A->HypreRead();
#if MFEM_HYPRE_VERSION >= 21600
    hypre_CSRMatrixBigJtoJ(csr_op);
#endif

    if (sort_row)
    {
        hypre_CSRMatrixSortRow(csr_op);
    }

    // parameters of the Operator
    n_loc = csr_op->num_rows;
    n = parcsr_op->global_num_rows;
    int64_t nrows = n;
    row_start = parcsr_op->first_row_index;
    row_end = row_start + n_loc - 1;
    MFEM_VERIFY(!cuDSSObjectInitialized || !reorder_reuse ||
                    (reorder_reuse && (nnz == csr_op->num_nonzeros)),
                "CuDSSSolver::SetOperator: Inconsistent new matrix pattern!");
    nnz = csr_op->num_nonzeros;

    // Checkout the cudss API
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;

    // Initial the cudssMatrix objects
    if (!cuDSSObjectInitialized)
    {
        // Set the cudssMatrix object of csr operator
        Ac = std::make_unique<cudssMatrix_t>();
        // Create empty RHS and solution vectors
        InitRhsSol(1);
    }

    // New cudss CSR matrix object and analysis or reuse the one from a previous matrix
    if (!cuDSSObjectInitialized || !reorder_reuse)
    {
        if (reorder_reuse) // !cuDSSObjectInitialized && reorder_reuse
        {
            // NOTE: For CuDSS solver to reuse the reordering (skipping analysis phase),
            // it needs to access the I and J arrays of the **initial** matrix.
            // Therefore, we need to copy and keep I and J in device memory.
            cudaError_t cuda_status = cudaSuccess;

            cuda_status = cudaMalloc(&csr_offsets_d, (n_loc + 1) * sizeof(int));
            CheckCudaAPI(cuda_status, "cudaMalloc for CSR I vector");

            cuda_status = cudaMalloc(&csr_columns_d, nnz * sizeof(int));
            CheckCudaAPI(cuda_status, "cudaMalloc for CSR J vector");

            cuda_status = cudaMemcpy(csr_offsets_d, csr_op->i, (n_loc + 1) * sizeof(int),
                                     cudaMemcpyDeviceToDevice);
            CheckCudaAPI(cuda_status, "cudaMemcpy for CSR I vector");

            cuda_status = cudaMemcpy(csr_columns_d, csr_op->j, nnz * sizeof(int),
                                     cudaMemcpyDeviceToDevice);
            CheckCudaAPI(cuda_status, "cudaMemcpy for CSR J vector");

            status = cudssMatrixCreateCsr(Ac.get(), nrows, nrows, nnz, csr_offsets_d, NULL,
                                          csr_columns_d, csr_op->data, CUDA_R_32I, CUDA_REAL_T,
                                          mat_type, mview, CUDSS_BASE_ZERO);
            CheckCudssAPI(status, "cudssMatrixCreateCsr for Operator");
        }
        else // !reorder_reuse
        {
            if (cuDSSObjectInitialized)
            {
                status = cudssMatrixDestroy(*Ac);
                CheckCudssAPI(status, "cudssMatrixDestroy for Operator");
            }
            status = cudssMatrixCreateCsr(Ac.get(), nrows, nrows, nnz, csr_op->i, NULL,
                                          csr_op->j, csr_op->data, CUDA_R_32I, CUDA_REAL_T,
                                          mat_type, mview, CUDSS_BASE_ZERO);
            CheckCudssAPI(status, "cudssMatrixCreateCsr for Operator");
        }

        status = cudssMatrixSetDistributionRow1d(*Ac, row_start, row_end);
        CheckCudssAPI(status, "cudssMatrixSetDistributionRow1d for Operator");

        // Analysis
        status = cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData,
                              *Ac, yc, xc);
        CheckCudssAPI(status, "cudssExecute for CUDSS_PHASE_ANALYSIS");
    }
    else // cuDSSObjectInitialized && reorder_reuse
    {
        // NOTE: When reusing analysis result, we only update the Data array,
        // without changing the I and J arrays.
        status = cudssMatrixSetValues(*Ac, csr_op->data);
        CheckCudssAPI(status, "cudssMatrixSetValues");
    }

    // Factorization
    status = cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig,
                          solverData, *Ac, yc, xc);
    CheckCudssAPI(status, "cudssExecute for CUDSS_PHASE_FACTORIZATION");

    hypre_CSRMatrixDestroy(csr_op);
}

void CuDSSSolver::InitRhsSol(int nrhs_) const
{
   if (nrhs != nrhs_)
   {
        cudssStatus_t status = CUDSS_STATUS_SUCCESS;

        if (nrhs > 0)
        {
            // Destroy the previous RHS vector and solution vector
            status = cudssMatrixDestroy(xc);
            CheckCudssAPI(status, "cudssMatrixDestroy for RHS");
            status = cudssMatrixDestroy(yc);
            CheckCudssAPI(status, "cudssMatrixDestroy for solution Vector");
        }
        // Create empty RHS and solution vectors
        status = cudssMatrixCreateDn(&xc, n, nrhs_, n, NULL, CUDA_REAL_T,
                                     CUDSS_LAYOUT_COL_MAJOR);
        CheckCudssAPI(status, "cudssMatrixCreateDn for RHS");
        status = cudssMatrixSetDistributionRow1d(xc, row_start, row_end);
        CheckCudssAPI(status, "cudssMatrixSetDistributionRow1d for RHS");
        status = cudssMatrixCreateDn(&yc, n, nrhs_, n, NULL, CUDA_REAL_T,
                                     CUDSS_LAYOUT_COL_MAJOR);
        CheckCudssAPI(status, "cudssMatrixCreateDn for solution Vector");
        status = cudssMatrixSetDistributionRow1d(yc, row_start, row_end);
        CheckCudssAPI(status, "cudssMatrixSetDistributionRow1d for solution Vector");
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

void CuDSSSolver::ArrayMult(const Array<const Vector *> &X, Array<Vector *> &Y) const
{
    InitRhsSol(X.Size());

    Vector RHS, SOL;

    if (nrhs == 1)
    {
        RHS.MakeRef(*(const_cast<Vector *>(X[0])), 0, X[0]->Size());
        SOL.MakeRef(*Y[0], 0, Y[0]->Size());
    }
    else
    {
        // NOTE: RHS must have **global** num_rows and nrhs columns
        RHS.SetSize(nrhs * n, *X[0]);
        for (int i = 0; i < nrhs; i++)
        {
            Vector s(RHS, i * n, n_loc);
            s = *X[i];
        }

        // NOTE: SOL must have **global** num_rows and nrhs columns
        SOL.SetSize(nrhs * n, *Y[0]);
    }

    cudssStatus_t status = CUDSS_STATUS_SUCCESS;

    status = cudssMatrixSetValues(xc, const_cast<real_t *>(RHS.Read()));
    CheckCudssAPI(status, "cudssMatrixSetValues for RHS");
    status = cudssMatrixSetValues(yc, SOL.Write());
    CheckCudssAPI(status, "cudssMatrixSetValues for solution Vector");

    // Solve
    status = cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig, solverData,
                          *Ac, yc, xc);
    CheckCudssAPI(status, "cudssExecute for CUDSS_PHASE_SOLVE");

    if (nrhs > 1)
    {
        // Get solution for each right-hand side
        for (int i = 0; i < nrhs; i++)
        {
            Vector s(SOL, i * n, n_loc);
            *Y[i] = s;
        }
    }
}

} // namespace mfem
#endif // MFEM_USE_MPI
#endif // MFEM_USE_CUDSS
