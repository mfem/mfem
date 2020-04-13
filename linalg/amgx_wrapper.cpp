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

#include "amgx_wrapper.hpp"


namespace mfem
{

int NvidiaAMGX::count = 0;

AMGX_resources_handle NvidiaAMGX::rsrc{nullptr};

NvidiaAMGX::NvidiaAMGX(const MPI_Comm &comm,
                       const std::string &modeStr, const std::string &cfgFile)
{
  Init(comm, modeStr, cfgFile);
}

void NvidiaAMGX::Init(const MPI_Comm &comm,
                      const std::string &modeStr, const std::string &cfgFile)
{
  if(modeStr == "dDDI") { amgx_mode = AMGX_mode_dDDI;}
  else{ mfem_error("dDDI only supported \n");}


  count++;
  MPI_Comm_dup(comm, &amgx_comm);
  MPI_Comm_size(amgx_comm, &MPI_SZ);
  MPI_Comm_rank(amgx_comm, &MPI_RANK);

  //Get device count. Using lrun will enable 1 MPI rank to see 1 GPU.
  cudaGetDeviceCount(&nDevs);
  cudaGetDevice(&deviceId);

  printf("No of visible devices per rank %d, deviceId %d, myrank %d, mpi_sz %d\n",
         nDevs, deviceId, MPI_RANK, MPI_SZ);

  //Init AMGX
  if(count == 1)
  {
    AMGX_SAFE_CALL(AMGX_initialize());

    AMGX_SAFE_CALL(AMGX_initialize_plugins());

    AMGX_SAFE_CALL(AMGX_install_signal_handler());
  }

  AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, cfgFile.c_str()));

  //Create a resource object
  if (count == 1) AMGX_resources_create(&rsrc, cfg, &amgx_comm, 1, &deviceId);

  //Create vector objects
  AMGX_vector_create(&x, rsrc, amgx_mode);
  AMGX_vector_create(&b, rsrc, amgx_mode);

  AMGX_matrix_create(&A, rsrc, amgx_mode);

  AMGX_solver_create(&solver, rsrc, amgx_mode, cfg);

  // obtain the default number of rings based on current configuration
  AMGX_config_get_default_number_of_rings(cfg, &ring);

  isEnabled = true;
}

void NvidiaAMGX::GetLocalA(const HypreParMatrix &in_A,
                             Array<HYPRE_Int> &I, Array<int64_t> &J, Array<double> &Data)
{

  mfem::SparseMatrix Diag, Offd;
  HYPRE_Int* cmap; //column map

  in_A.GetDiag(Diag); Diag.SortColumnIndices();
  in_A.GetOffd(Offd, cmap); Offd.SortColumnIndices();

  //Number of rows in this partition
  int row_len = std::abs(in_A.RowPart()[1] - in_A.RowPart()[0]); //end of row partition

  //Note Amgx requires 64 bit integers for column array
  //So we promote in this routine
  int *DiagI = Diag.GetI();
  int *DiagJ = Diag.GetJ();
  double *DiagA = Diag.GetData();

  int *OffI = Offd.GetI();
  int *OffJ = Offd.GetJ();
  double *OffA = Offd.GetData();

  I.SetSize(row_len+1);

  //Enumerate the local rows [0, num rows in proc)
  I[0]=0;
  for (int i=0; i<row_len; i++) {
    I[i+1] = I[i] + (DiagI[i+1] - DiagI[i]) + (OffI[i+1] - OffI[i]);
  }

  const HYPRE_Int *colPart = in_A.ColPart();
  J.SetSize(1+I[row_len]);
  Data.SetSize(1+I[row_len]);

  int cstart = colPart[0];

  int k    = 0;
  for (int i=0; i<row_len; i++) {

    int jo, icol;
    int ncols_o = OffI[i+1] - OffI[i];
    int ncols_d = DiagI[i+1] - DiagI[i];

    //OffDiagonal
    for (jo=0; jo<ncols_o; jo++) {
      icol = cmap[*OffJ];
      if (icol >= cstart) break;
      J[k]   = icol; OffJ++;
      Data[k++] = *OffA++;
    }

    //Diagonal matrix
    for (int j=0; j<ncols_d; j++) {
      J[k]   = cstart + *DiagJ++;
      Data[k++] = *DiagA++;
    }

    //OffDiagonal
    for (int j=jo; j<ncols_o; j++) {
      J[k]   = cmap[*OffJ++];
      Data[k++] = *OffA++;
    }
  }

}

void NvidiaAMGX::SetA(const HypreParMatrix &in_A)
{

  int nGlobalRows, nLocalRows;
  nGlobalRows = in_A.M();

  Array<HYPRE_Int> I;
  Array<int64_t> J;
  Array<double> Aloc;

  //Step 1.
  //Merge Diag and Offdiag to a single CSR matrix
  GetLocalA(in_A, I, J, Aloc);

  //Step 2.
  //Create a vector of offsets describing matrix row partitions
  mfem::Array<int64_t> rowPart(MPI_SZ+1); rowPart = 0.0;

  //Must be promoted to int64!  --consider typedef?
  int64_t myStart = in_A.GetRowStarts()[0];
  MPI_Allgather(&myStart, 1, MPI_INT64_T,
                rowPart.GetData(),1, MPI_INT64_T
                ,amgx_comm);
  MPI_Barrier(amgx_comm);

  rowPart[MPI_SZ] = in_A.M();
  //rowPart.Print();
  for(int i=0; i<rowPart.Size(); ++i){
    std::cout<<rowPart[i]<<std::endl;
  }

  AMGX_distribution_handle dist;
  AMGX_distribution_create(&dist, cfg);
  AMGX_distribution_set_partition_data(dist, AMGX_DIST_PARTITION_OFFSETS,
                                       rowPart.GetData());

  //Step 3.
  //Upload matrix to AMGX
  nLocalRows = I.Size()-1;

  if (!std::is_sorted(rowPart.GetData(), rowPart.GetData() + MPI_SZ + 1)) {
    mfem_error("Not sorted \n");
  }

  AMGX_matrix_upload_distributed(A, nGlobalRows, nLocalRows, I.HostRead()[nLocalRows],
                                 1, 1, I.ReadWrite(), J.ReadWrite(), Aloc.ReadWrite(),
                                 nullptr, dist);


  AMGX_distribution_destroy(dist);

  MPI_Barrier(amgx_comm);
  AMGX_solver_setup(solver, A);

  //Step 4. Bind vectors to A
  AMGX_vector_bind(x, A);
  AMGX_vector_bind(b, A);
}

void NvidiaAMGX::Solve(Vector &in_x, Vector &in_b)
{

  //Upload vectors to amgx
  AMGX_vector_upload(x, in_x.Size(), 1, in_x.GetData());

  AMGX_vector_upload(b, in_b.Size(), 1, in_b.GetData());

  MPI_Barrier(amgx_comm);

  AMGX_solver_solve(solver, b, x);

  AMGX_SOLVE_STATUS   status;
  AMGX_solver_get_status(solver, &status);
  if (status != AMGX_SOLVE_SUCCESS)
  {
    printf("Amgx failed to solve system, error code %d. \n", status);
  }

  AMGX_vector_download(x, in_x.GetData());

}

NvidiaAMGX::~NvidiaAMGX()
{
  if(isEnabled)
  {
    AMGX_solver_destroy(solver);
    AMGX_matrix_destroy(A);
    AMGX_vector_destroy(x);
    AMGX_vector_destroy(b);

    if (count == 1)
      {
        AMGX_resources_destroy(rsrc);
        AMGX_SAFE_CALL(AMGX_config_destroy(cfg));

        AMGX_SAFE_CALL(AMGX_finalize_plugins());
        AMGX_SAFE_CALL(AMGX_finalize());
      }
    else
      {
        AMGX_config_destroy(cfg);
      }
    MPI_Comm_free(&amgx_comm);
  }
  count -= 1;
  isEnabled=false;
}

}//mfem namespace
