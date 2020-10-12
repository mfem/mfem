#include "cpardiso.hpp"
#include "hypre.hpp"
#include <algorithm>
#include <vector>
#include <numeric>

#ifdef MFEM_USE_MPI
#ifdef MFEM_USE_MKL_CPARDISO

namespace mfem
{
CPardisoSolver::CPardisoSolver(MPI_Comm comm) : comm_(comm)
{
   // Indicate that default parameters are changed
   iparm[0] = 1;
   // Use METIS for fill-in reordering
   iparm[1] = 2;
   // Do not write the solution into the x vector data
   iparm[5] = 0;
   // Maximum number of iterative refinement steps
   iparm[7] = 2;
   // Perturb the pivot elements with 1E-13
   iparm[9] = 13;
   // Use nonsymmetric permutation
   iparm[10] = 1;
   // Perform a check on the input data
   iparm[26] = 1;
   // 0-based indexing in CSR data structure
   iparm[34] = 1;
   // All inputs are distributed between MPI processes
   iparm[39] = 2;
   // Maximum number of numerical factorizations
   maxfct = 1;
   // Which factorization to use. This parameter is ignored and always assumed
   // to be equal to 1. See MKL documentation.
   mnum = 1;
   // Print statistical information in file
   msglvl = 0;
   // Initialize error flag
   error = 0;
   // Real nonsymmetric matrix
   mtype = MatType::REAL_NONSYMMETRIC;
   // Number of right hand sides
   nrhs = 1;
};

void CPardisoSolver::SetOperator(const Operator &op)
{
   auto hypreParMat = dynamic_cast<const HypreParMatrix &>(op);

   MFEM_ASSERT(hypreParMat, "Must pass HypreParMatrix as Operator");

   auto parcsr_op = static_cast<hypre_ParCSRMatrix *>(
                       const_cast<HypreParMatrix &>(hypreParMat));

   hypre_CSRMatrix *csr_op = hypre_MergeDiagAndOffd(parcsr_op);
#if MFEM_HYPRE_VERSION >= 21600
   hypre_CSRMatrixBigJtoJ(csr_op);
#endif

   m = parcsr_op->global_num_rows;
   first_row = parcsr_op->first_row_index;
   nnz_loc = csr_op->num_nonzeros;
   m_loc = csr_op->num_rows;

   height = m_loc;
   width = m_loc;

   double *csr_nzval = csr_op->data;
   int *csr_colind = csr_op->j;

   delete[] csr_rowptr;
   delete[] reordered_csr_colind;
   delete[] reordered_csr_nzval;
   csr_rowptr = new int[m_loc + 1];
   reordered_csr_colind = new int[nnz_loc];
   reordered_csr_nzval = new double[nnz_loc];

   for (int i = 0; i <= m_loc; i++)
   {
      csr_rowptr[i] = (csr_op->i)[i];
   }

   // CPardiso expects the column indices to be sorted for each row
   std::vector<int> permutation_idx(nnz_loc);
   std::iota(permutation_idx.begin(), permutation_idx.end(), 0);
   for (int i = 0; i < m_loc; i++)
   {
      std::sort(permutation_idx.begin() + csr_rowptr[i],
                permutation_idx.begin() + csr_rowptr[i + 1],
                [csr_colind](int i1, int i2)
      {
         return csr_colind[i1] < csr_colind[i2];
      });
   }

   for (int i = 0; i < nnz_loc; i++)
   {
      reordered_csr_colind[i] = csr_colind[permutation_idx[i]];
      reordered_csr_nzval[i] = csr_nzval[permutation_idx[i]];
   }

   hypre_CSRMatrixDestroy(csr_op);

   // The number of row in global matrix, rhs element and solution vector that
   // begins the input domain belonging to this MPI process
   iparm[40] = first_row;

   // The number of row in global matrix, rhs element and solution vector that
   // ends the input domain belonging to this MPI process
   iparm[41] = first_row + m_loc - 1;

   // Analyze inputs
   phase = 11;
   cluster_sparse_solver(pt,
                         &maxfct,
                         &mnum,
                         &mtype,
                         &phase,
                         &m,
                         reordered_csr_nzval,
                         csr_rowptr,
                         reordered_csr_colind,
                         &idum,
                         &nrhs,
                         iparm,
                         &msglvl,
                         &ddum,
                         &ddum,
                         &comm_,
                         &error);

   MFEM_ASSERT(error == 0, "CPardiso analyze input error");

   // Numerical factorization
   phase = 22;
   cluster_sparse_solver(pt,
                         &maxfct,
                         &mnum,
                         &mtype,
                         &phase,
                         &m,
                         reordered_csr_nzval,
                         csr_rowptr,
                         reordered_csr_colind,
                         &idum,
                         &nrhs,
                         iparm,
                         &msglvl,
                         &ddum,
                         &ddum,
                         &comm_,
                         &error);

   MFEM_ASSERT(error == 0, "CPardiso factorization input error");
}

void CPardisoSolver::Mult(const Vector &b, Vector &x) const
{
   // Solve
   phase = 33;
   cluster_sparse_solver(pt,
                         &maxfct,
                         &mnum,
                         &mtype,
                         &phase,
                         &m,
                         reordered_csr_nzval,
                         csr_rowptr,
                         reordered_csr_colind,
                         &idum,
                         &nrhs,
                         iparm,
                         &msglvl,
                         b.GetData(),
                         x.GetData(),
                         &comm_,
                         &error);

   MFEM_ASSERT(error == 0, "Pardiso solve error");
}

void CPardisoSolver::SetPrintLevel(int print_level)
{
   msglvl = print_level;
}

void CPardisoSolver::SetMatrixType(MatType mat_type)
{
   mtype = mat_type;
}

CPardisoSolver::~CPardisoSolver()
{
   // Release all internal memory
   phase = -1;
   cluster_sparse_solver(pt,
                         &maxfct,
                         &mnum,
                         &mtype,
                         &phase,
                         &m,
                         reordered_csr_nzval,
                         csr_rowptr,
                         reordered_csr_colind,
                         &idum,
                         &nrhs,
                         iparm,
                         &msglvl,
                         &ddum,
                         &ddum,
                         &comm_,
                         &error);

   MFEM_ASSERT(error == 0, "CPardiso free error");

   delete[] csr_rowptr;
   delete[] reordered_csr_colind;
   delete[] reordered_csr_nzval;
}

} // namespace mfem

#endif // MFEM_USE_MKL_CPARDISO
#endif // MFEM_USE_MPI
