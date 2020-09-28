// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../config/config.hpp"

#ifdef MFEM_USE_MUMPS
#ifdef MFEM_USE_MPI

#include "mumps.hpp"

// mumps headers
#define USE_COMM_WORLD -987654
using namespace std;


namespace mfem
{

MUMPSSolver::MUMPSSolver( MPI_Comm comm )
   : comm_(comm)
{
   this->Init();
}

MUMPSSolver::MUMPSSolver( HypreParMatrix & A)
   : comm_(A.GetComm())
{


   APtr = dynamic_cast<const HypreParMatrix*>(&A);
   if ( APtr == NULL )
   {
      mfem_error("MUMPSSolver::SetOperator : not HypreParMatrix!");
   }
   height = A.Height();
   width  = A.Width();
   this->Init();
}

MUMPSSolver::~MUMPSSolver()
{
   id->job=-2;
   dmumps_c(id);
}

void MUMPSSolver::Init()
{
   MPI_Comm_size(comm_, &numProcs_);
   MPI_Comm_rank(comm_, &myid_);

   cout << "In MUMPS init" << endl;
   hypre_ParCSRMatrix * parcsr_op = (hypre_ParCSRMatrix *)
                                    const_cast<HypreParMatrix&>(*APtr);

   csr_op = hypre_MergeDiagAndOffd(parcsr_op);
#if MFEM_HYPRE_VERSION >= 21600
   hypre_CSRMatrixBigJtoJ(csr_op);
#endif

   int * Iptr = csr_op->i;
   int * Jptr = csr_op->j;

   int nnz = csr_op->num_nonzeros;
   I = new int[nnz];
   J = new int[nnz];

   n_loc = csr_op->num_rows;
   int k = 0;
   for (int i = 0; i<n_loc; i++)
   {
      for (int j = Iptr[i]; j<Iptr[i+1]; j++)
      {
         // "tdof offsets" can be determined by parcsr_op -> rowstarts
         I[k] = parcsr_op->first_row_index + i + 1;
         J[k] = Jptr[k]+1;
         k++;
      }
   }

   id = new DMUMPS_STRUC_C;

   MUMPS_INT ierr;
   int error = 0;
   /* Initialize a MUMPS instance. Use MPI_COMM_WORLD */
   id->comm_fortran=USE_COMM_WORLD;

   id->job=-1; id->par=1; id->sym=0;
   // Mumps init
   dmumps_c(id);

#define ICNTL(I) icntl[(I)-1] /* macro s.t. indices match documentation */
#define INFO(I) info[(I)-1] /* macro s.t. indices match documentation */
   /* No outputs */
   //   id.ICNTL(1)=-1; id.ICNTL(2)=-1; id.ICNTL(3)=-1; id.ICNTL(4)=0;
   // id.ICNTL(5) = 0;
   id->ICNTL(18) = 3; // distributed matrix
   // id->ICNTL(20) = 10; // distributed rhs
   id->ICNTL(20) = 0; //  rhs on host
   // id->ICNTL(21) = 1; // distributed solution
   id->ICNTL(21) = 0; // solution on host

   id->n = parcsr_op->global_num_rows;

   // on all procs
   id->nnz_loc = nnz;
   id->irn_loc = I;
   id->jcn_loc = J;
   id->a_loc = csr_op->data;

   // Analysis
   id->job=1;
   dmumps_c(id);

   // Factorization
   id->job=2;
   dmumps_c(id);

   if (myid_ == 0)
   {
      rhs_glob.SetSize(parcsr_op->global_num_rows);
      recv_counts.SetSize(numProcs_);
      displs.SetSize(numProcs_);
   }
   int n_loc = csr_op->num_rows;
   MPI_Gather(&n_loc,1,MPI_INT,recv_counts,1,MPI_INT,0,comm_);


   if (myid_ == 0)
   {
      displs[0] = 0;
      for (int k=0; k<numProcs_-1; k++)
      {
         displs[k+1] = displs[k] + recv_counts[k];
      }
   }
}

void MUMPSSolver::Mult( const Vector & x, Vector & y ) const
{
   MPI_Gatherv(x.GetData(),x.Size(),MPI_DOUBLE,rhs_glob.GetData(),
               recv_counts,displs,MPI_DOUBLE,0,comm_);
   id->rhs = rhs_glob.GetData();
   id->job=3;
   dmumps_c(id);

   MPI_Scatterv(id->rhs,recv_counts,displs,MPI_DOUBLE,
                y.GetData(),y.Size(),MPI_DOUBLE,0,comm_);
}

void MUMPSSolver::SetOperator( const Operator & op )
{
   // Verify that we have a compatible operator
   APtr = dynamic_cast<const HypreParMatrix*>(&op);
   if ( APtr == NULL )
   {
      mfem_error("MUMPSSolver::SetOperator : not HypreParMatrix!");
   }

}

} // mfem namespace

#endif // MFEM_USE_MPI
#endif // MFEM_USE_MUMPS
