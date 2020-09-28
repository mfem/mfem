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

MUMPSSolver::MUMPSSolver( MPI_Comm comm_ )
   : comm(comm_)
{}

MUMPSSolver::~MUMPSSolver()
{
   id->job=-2;
   dmumps_c(id);
}

void MUMPSSolver::SetParameters()
{
   id->ICNTL(1) = -1; // output messages 
   id->ICNTL(2) = -1; // Diagnosting printing
   id->ICNTL(3) = -1;  // Global info on host
   id->ICNTL(4) =  0; // Level of error printing
   id->ICNTL(5) =  0; //inpute matrix format (distributed)
   id->ICNTL(9) =  1; // Use A or A^T
   id->ICNTL(10) = 0; // Iterative refinement (disabled)
   id->ICNTL(11) = 0; // Error analysis-statistics (disabled)
   id->ICNTL(13) = 0; // Use of ScaLAPACK (Parallel factorization on root)
   id->ICNTL(14) = 20; // Percentage increase of estimated workspace (default = 20%)
   id->ICNTL(16) = 0; // Number of OpenMP threads (default)
   id->ICNTL(18) = 3; // Matrix input format (distributed)
   id->ICNTL(19) = 0; // Schur complement (no Schur complement matrix returned)
   id->ICNTL(20) = (dist_rhs) ? 10 : 0; // RHS input format (distributed or only on host)
   id->ICNTL(21) = (dist_sol) ? 1  : 0; // Sol input format (distributed or only on host)
   id->ICNTL(22) = 0; // Out of core factorization and solve (disabled)
   id->ICNTL(23) = 0; // Maximum size of working memory (default = based on estimates)
}

void MUMPSSolver::Init()
{
   MPI_Comm_size(comm, &numProcs);
   MPI_Comm_rank(comm, &myid);

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
   row_start = parcsr_op->first_row_index;
   int k = 0;
   for (int i = 0; i<n_loc; i++)
   {
      for (int j = Iptr[i]; j<Iptr[i+1]; j++)
      {
         // "tdof offsets" can be determined by parcsr_op -> rowstarts
         I[k] = row_start + i + 1;
         J[k] = Jptr[k]+1; // This can be avoided
         k++;
      }
   }

   // new MUMPS object
   id = new DMUMPS_STRUC_C;
   // Initialize a MUMPS instance. Use MPI_COMM_WORLD 
   id->comm_fortran=USE_COMM_WORLD;

   // Host is involved in computation
   id->par=1; 
   // Unsymmetric matrix
   id->sym=0;
   // Mumps init
   id->job=-1; 
   dmumps_c(id);

   SetParameters(); // Set MUMPS default parameters

   id->n = parcsr_op->global_num_rows;
   // on all procs
   id->nnz_loc = nnz;
   id->irn_loc = I; // Distributed rows array
   id->jcn_loc = J; // Distributed column array
   id->a_loc = csr_op->data; // Distributed data array

   // Analysis
   id->job=1;
   dmumps_c(id);

   // Factorization
   id->job=2;
   dmumps_c(id);


   if (!(dist_rhs && dist_sol)) // if at least one is on host
   {
      if (myid == 0)
      {
         rhs_glob.SetSize(parcsr_op->global_num_rows); rhs_glob = 0.0;
         recv_counts.SetSize(numProcs);
         displs.SetSize(numProcs);
      }
      MPI_Gather(&n_loc,1,MPI_INT,recv_counts,1,MPI_INT,0,comm);
      if (myid == 0)
      {
         displs[0] = 0;
         for (int k=0; k<numProcs-1; k++)
         {
            displs[k+1] = displs[k] + recv_counts[k];
         }
      }
   }

   if (dist_rhs)
   {
      irhs_loc.SetSize(n_loc);
      for (int i = 0; i<n_loc; i++)
      {
         irhs_loc[i] = row_start + i + 1;
      }
   }

   if (dist_sol)
   {
      row_starts.resize(numProcs);
      MPI_Allgather(&row_start,1,MPI_INT,&row_starts[0],1,MPI_INT,comm);
      sol_loc.SetSize(id->INFO(23));
      isol_loc = new int[id->INFO(23)];
   }
}

void MUMPSSolver::Mult( const Vector & x, Vector & y ) const
{

   // cases for the RHS
   if (dist_rhs) 
   {  // distribute RHS
      id->nloc_rhs = x.Size();
      id->lrhs_loc = x.Size();
      id->rhs_loc = x.GetData();
      id->irhs_loc = const_cast<int*>(irhs_loc.GetData());
   }
   else
   {  // RHS gathered on the host
      MPI_Gatherv(x.GetData(),x.Size(),MPI_DOUBLE,rhs_glob.GetData(),
                  recv_counts,displs,MPI_DOUBLE,0,comm);
      if (myid == 0)
      {
         id->rhs = rhs_glob.GetData();
      }            
   }

   // cases for the SOL
   if (dist_sol)
   {  
      id->sol_loc = sol_loc.GetData();
      id->lsol_loc = id->INFO(23);
      id->isol_loc = isol_loc;
   }
   else
   {  // if sol on host on the host  
      if (myid == 0)
      {
         id->rhs = rhs_glob.GetData();
      }      
   }
   

   id->job = 3;
   dmumps_c(id);

   if (dist_sol)
   {
      // sol_loc.Print();
      Array<int> temp(sol_loc.Size());
      for (int i = 0; i<sol_loc.Size(); i++)
      {
         temp[i] = isol_loc[i]-1;
      }
      RedistributeSol(temp, sol_loc, y); 
   }
   else
   {
      MPI_Scatterv(rhs_glob.GetData(),recv_counts,displs,MPI_DOUBLE,
                   y.GetData(),y.Size(),MPI_DOUBLE,0,comm);
   }
}

void MUMPSSolver::SetOperator( const Operator & op )
{
   // Verify that we have a compatible operator
   APtr = dynamic_cast<const HypreParMatrix*>(&op);
   if ( APtr == NULL )
   {
      mfem_error("MUMPSSolver::SetOperator : not HypreParMatrix!");
   }
   height = op.Height();
   width  = op.Width();
   this->Init();

   Array<int> test;
   
}

int MUMPSSolver::GetRowRank(int i, const std::vector<int> & row_starts_) const
{
   int size = row_starts_.size();
   if (size == 1) { return 0; }
   auto up=std::upper_bound(row_starts_.begin(), row_starts_.end(),i); 
   return std::distance(row_starts_.begin(),up)-1;
}

void MUMPSSolver::RedistributeSol(const Array<int> & row_map, const Vector & x, Vector &y) const
{
   MFEM_VERIFY(row_map.Size() == x.Size(), "Inconcistent sizes");
   int size = x.Size();


   // workspace for MPI_ALLtoALL
   Array<int> send_count(numProcs); send_count = 0;
   Array<int> send_displ(numProcs); send_displ = 0;
   Array<int> recv_count(numProcs); recv_count = 0;
   Array<int> recv_displ(numProcs); recv_displ = 0;

   // compute send_count
   for (int i = 0; i<size; i++)
   {
      int j = row_map[i];
      int row_rank = GetRowRank(j,row_starts);
      send_count[row_rank]++; // the dof value 
   }

   // compute recv_count
   MPI_Alltoall(send_count,1,MPI_INT,recv_count,1,MPI_INT,comm);
   for (int k=0; k<numProcs-1; k++)
   {
      send_displ[k+1] = send_displ[k] + send_count[k];
      recv_displ[k+1] = recv_displ[k] + recv_count[k];
   }
   int sbuff_size = send_count.Sum();
   int rbuff_size = recv_count.Sum();

   Array<int> sendbuf_index(sbuff_size);  sendbuf_index = 0;
   Array<double> sendbuf_value(sbuff_size);  sendbuf_value = 0;
   Array<int> soffs(numProcs); soffs = 0;

   // Fill in send buffers
   for (int i = 0; i<size; i++)
   {
      int j = row_map[i];
      int row_rank = GetRowRank(j,row_starts);
      int k = send_displ[row_rank] + soffs[row_rank];
      sendbuf_index[k] = j;
      sendbuf_value[k] = x(i);
      soffs[row_rank]++;
   }

   // communicate
   Array<int> recvbuf_index(rbuff_size);
   Array<double> recvbuf_value(rbuff_size);
   MPI_Alltoallv(sendbuf_index, send_count, send_displ, MPI_INT, recvbuf_index,
                 recv_count, recv_displ, MPI_INT, comm);
   MPI_Alltoallv(sendbuf_value, send_count, send_displ, MPI_DOUBLE, recvbuf_value,
                 recv_count, recv_displ, MPI_DOUBLE, comm);     

   // Unpack recv buffer
   for (int i = 0; i<rbuff_size; i++)
   {
      int local_index = recvbuf_index[i] - row_start;
      double val = recvbuf_value[i];
      y(local_index) = val;
   }                          
}



} // mfem namespace

#endif // MFEM_USE_MPI
#endif // MFEM_USE_MUMPS
