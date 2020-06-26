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
#include "amgx.hpp"
#ifdef MFEM_USE_MPI
#ifdef MFEM_USE_AMGX

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
   if (modeStr == "dDDI") { amgx_mode = AMGX_mode_dDDI;}
   else { mfem_error("dDDI only supported \n");}


   count++;

   MPI_Comm_dup(comm, &amgx_comm);
   MPI_Comm_size(amgx_comm, &MPI_SZ);
   MPI_Comm_rank(amgx_comm, &MPI_RANK);

   //Get device count. Using lrun will enable 1 MPI rank to see 1 GPU.
   cudaGetDeviceCount(&nDevs);
   cudaGetDevice(&deviceId);

   //Init AMGX
   if (count == 1)
   {
      AMGX_SAFE_CALL(AMGX_initialize());

      AMGX_SAFE_CALL(AMGX_initialize_plugins());

      AMGX_SAFE_CALL(AMGX_install_signal_handler());
   }

   AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, cfgFile.c_str()));

   //Create a resource object
   if (count == 1) { AMGX_resources_create(&rsrc, cfg, &amgx_comm, 1, &deviceId); }

   //Create vector objects
   AMGX_vector_create(&x, rsrc, amgx_mode);
   AMGX_vector_create(&b, rsrc, amgx_mode);

   AMGX_matrix_create(&A, rsrc, amgx_mode);

   AMGX_solver_create(&solver, rsrc, amgx_mode, cfg);

   // obtain the default number of rings based on current configuration
   AMGX_config_get_default_number_of_rings(cfg, &ring);

   isEnabled = true;
   isMPIEnabled = true;
}

void NvidiaAMGX::Init(const std::string &modeStr, const std::string &cfgFile)
{
   if (modeStr == "dDDI") { amgx_mode = AMGX_mode_dDDI;}
   else { mfem_error("dDDI only supported \n");}
   AMGX_SAFE_CALL(AMGX_initialize());
   AMGX_SAFE_CALL(AMGX_initialize_plugins());
   AMGX_SAFE_CALL(AMGX_install_signal_handler());
   AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, cfgFile.c_str()));
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
   int row_len = std::abs(in_A.RowPart()[1] -
                          in_A.RowPart()[0]); //end of row partition

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
   for (int i=0; i<row_len; i++)
   {
      I[i+1] = I[i] + (DiagI[i+1] - DiagI[i]) + (OffI[i+1] - OffI[i]);
   }

   const HYPRE_Int *colPart = in_A.ColPart();
   J.SetSize(1+I[row_len]);
   Data.SetSize(1+I[row_len]);

   int cstart = colPart[0];

   int k    = 0;
   for (int i=0; i<row_len; i++)
   {

      int jo, icol;
      int ncols_o = OffI[i+1] - OffI[i];
      int ncols_d = DiagI[i+1] - DiagI[i];

      //OffDiagonal
      for (jo=0; jo<ncols_o; jo++)
      {
         icol = cmap[*OffJ];
         if (icol >= cstart) { break; }
         J[k]   = icol; OffJ++;
         Data[k++] = *OffA++;
      }

      //Diagonal matrix
      for (int j=0; j<ncols_d; j++)
      {
         J[k]   = cstart + *DiagJ++;
         Data[k++] = *DiagA++;
      }

      //OffDiagonal
      for (int j=jo; j<ncols_o; j++)
      {
         J[k]   = cmap[*OffJ++];
         Data[k++] = *OffA++;
      }
   }

}

void NvidiaAMGX::SetA(const HypreParMatrix &in_A)
{

   int nGlobalRows, nLocalRows;
   nGlobalRows = in_A.M();

   //Step 1.
   //Merge Diag and Offdiag to a single CSR matrix
   GetLocalA(in_A, m_I, m_J, m_Aloc);

   //Step 2.
   //Create a vector of offsets describing matrix row partitions
   mfem::Array<int64_t> rowPart(MPI_SZ+1); rowPart = 0.0;

   //Must be promoted to int64!  --consider typedef?
   int64_t myStart = in_A.GetRowStarts()[0];
   /*
   MPI_Allgather(&myStart, 1, MPI_INT64_T,
                 rowPart.GetData(),1, MPI_INT64_T
                 ,amgx_comm);
   MPI_Barrier(amgx_comm);
    */
   rowPart[MPI_SZ] = in_A.M();

   AMGX_distribution_handle dist;

   AMGX_distribution_create(&dist, cfg);
  /*
   AMGX_distribution_set_partition_data(dist, AMGX_DIST_PARTITION_OFFSETS,
                                        rowPart.GetData());

   //Step 3.
   //Upload matrix to AMGX
   nLocalRows = m_I.Size()-1;

   const int m_I_nLocalRows = m_I.HostRead()[nLocalRows];
   AMGX_pin_memory(m_I.HostReadWrite(),m_I.Size()*sizeof(int));
   AMGX_pin_memory(m_J.HostReadWrite(),m_J.Size()*sizeof(int64_t));
   AMGX_pin_memory(m_Aloc.HostReadWrite(),m_Aloc.Size()*sizeof(double));

   AMGX_matrix_upload_distributed(A, nGlobalRows, nLocalRows,
                                  m_I_nLocalRows,
                                  1, 1, m_I.HostReadWrite(),
                                  m_J.HostReadWrite(), m_Aloc.HostReadWrite(),
                                  nullptr, dist);


   AMGX_distribution_destroy(dist);

   MPI_Barrier(amgx_comm);
   AMGX_solver_setup(solver, A);

   //Step 4. Bind vectors to A
   AMGX_vector_bind(x, A);
   AMGX_vector_bind(b, A);
   */
}


void NvidiaAMGX::SetA(const Operator& op)
{
   amgx_mode = AMGX_mode_dDDI;

   AMGX_resources_create_simple(&rsrc, cfg);
   AMGX_vector_create(&x, rsrc, amgx_mode);
   AMGX_vector_create(&b, rsrc, amgx_mode);
   AMGX_matrix_create(&A, rsrc, amgx_mode);
   AMGX_solver_create(&solver, rsrc, amgx_mode, cfg);


   spop = const_cast<SparseMatrix*>(dynamic_cast<const SparseMatrix*>(&op));
   MFEM_VERIFY(spop, "operator is not of correct type!");

   AMGX_matrix_upload_all(A, spop->Height(),
                          spop->NumNonZeroElems(),
                          1, 1,
                          spop->ReadWriteI(),
                          spop->ReadWriteJ(),
                          spop->ReadWriteData(), NULL);

   AMGX_solver_setup(solver, A);

   AMGX_vector_bind(x, A);
   AMGX_vector_bind(b, A);
}


void NvidiaAMGX::Solve(Vector &in_x, Vector &in_b)
{

   //Upload vectors to amgx
   AMGX_pin_memory(in_x.HostReadWrite(), in_x.Size()*sizeof(double));
   AMGX_pin_memory(in_b.HostReadWrite(), in_b.Size()*sizeof(double));

   AMGX_vector_upload(x, in_x.Size(), 1, in_x.HostReadWrite());

   AMGX_vector_upload(b, in_b.Size(), 1, in_b.HostReadWrite());

   MPI_Barrier(amgx_comm);

   AMGX_solver_solve(solver, b, x);

   AMGX_SOLVE_STATUS   status;
   AMGX_solver_get_status(solver, &status);
   if (status != AMGX_SOLVE_SUCCESS)
   {
      printf("Amgx failed to solve system, error code %d. \n", status);
   }

   AMGX_vector_download(x, in_x.HostWrite());

   AMGX_unpin_memory(in_x.HostReadWrite());
   AMGX_unpin_memory(in_b.HostReadWrite());
}


void NvidiaAMGX::Mult(Vector &in_b, Vector &in_x)
{

   //Upload vectors to amgx
   AMGX_pin_memory(in_x.HostReadWrite(), in_x.Size()*sizeof(double));
   AMGX_pin_memory(in_b.HostReadWrite(), in_b.Size()*sizeof(double));

   AMGX_vector_upload(x, in_x.Size(), 1, in_x.HostReadWrite());

   AMGX_vector_upload(b, in_b.Size(), 1, in_b.HostReadWrite());

   AMGX_solver_solve(solver, b, x);

   AMGX_vector_download(x, in_x.HostWrite());

   AMGX_unpin_memory(in_x.HostReadWrite());
   AMGX_unpin_memory(in_b.HostReadWrite());
}




NvidiaAMGX::~NvidiaAMGX()
{
   if (isEnabled)
   {

      AMGX_unpin_memory(m_I.HostReadWrite());
      AMGX_unpin_memory(m_J.HostReadWrite());
      AMGX_unpin_memory(m_Aloc.HostReadWrite());

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
      if (isMPIEnabled) { MPI_Comm_free(&amgx_comm); }
   }
   count -= 1;
   isEnabled=false;
}

////BEGIN ADAM
/* \implements AmgXSolver::initialize */
void NvidiaAMGX::initialize_new(const MPI_Comm &comm,
        const std::string &modeStr, const std::string &cfgFile)
{
    if (modeStr == "dDDI") { amgx_mode = AMGX_mode_dDDI;}
    else { mfem_error("dDDI only supported \n");}
    // increase the number of AmgXSolver instances
    count++;

    // get the name of this node
    int     len;
    char    name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(name, &len);
    nodeName = name;
    // initialize communicators and corresponding information
    initMPIcomms_new(comm);

    // only processes in gpuWorld are required to initialize AmgX
    if (gpuProc == 0)
    {
        initAmgX_new(cfgFile);
    }

    // a bool indicating if this instance is initialized
    //isInitialized = true;
}

/* \implements AmgXSolver::initMPIcomms */
void NvidiaAMGX::initMPIcomms_new(const MPI_Comm &comm)
{
    // duplicate the global communicator
    MPI_Comm_dup(comm, &amgx_comm);
    MPI_Comm_set_name(amgx_comm, "globalCpuWorld");

    // get size and rank for global communicator
    MPI_Comm_size(amgx_comm, &MPI_SZ);
    MPI_Comm_rank(amgx_comm, &MPI_RANK);


    // Get the communicator for processors on the same node (local world)
    MPI_Comm_split_type(amgx_comm,
            MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localCpuWorld);
    MPI_Comm_set_name(localCpuWorld, "localCpuWorld");

    // get size and rank for local communicator
    MPI_Comm_size(localCpuWorld, &localSize);
    MPI_Comm_rank(localCpuWorld, &myLocalRank);

    cudaGetDeviceCount(&nDevs);

    cudaGetDevice(&deviceId);

    // set up corresponding ID of the device used by each local process
    setDeviceIDs_new();
    MPI_Barrier(amgx_comm);


    // split the global world into a world involved in AmgX and a null world
    MPI_Comm_split(amgx_comm, gpuProc, 0, &gpuWorld);

    // get size and rank for the communicator corresponding to gpuWorld
    if (gpuWorld != MPI_COMM_NULL)
    {
        MPI_Comm_set_name(gpuWorld, "gpuWorld");
        MPI_Comm_size(gpuWorld, &gpuWorldSize);
        MPI_Comm_rank(gpuWorld, &myGpuWorldRank);
    }
    else // for those can not communicate with GPU devices
    {
        gpuWorldSize = MPI_UNDEFINED;
        myGpuWorldRank = MPI_UNDEFINED;
    }


    // split local world into worlds corresponding to each CUDA device
    MPI_Comm_split(localCpuWorld, deviceId, 0, &devWorld);
    MPI_Comm_set_name(devWorld, "devWorld");

    // get size and rank for the communicator corresponding to myWorld
    MPI_Comm_size(devWorld, &devWorldSize);
    MPI_Comm_rank(devWorld, &myDevWorldRank);

    MPI_Barrier(amgx_comm);
}

/* \implements AmgXSolver::setDeviceIDs */
void NvidiaAMGX::setDeviceIDs_new()
{
    // set the ID of device that each local process will use
    if (nDevs == localSize) // # of the devices and local precosses are the same
    {
        deviceId = myLocalRank;
        gpuProc = 0;
    }
    else if (nDevs > localSize) // there are more devices than processes
    {
        deviceId = myLocalRank;
        gpuProc = 0;
    }
    else // there more processes than devices
    {
        int     nBasic = localSize / nDevs,
                nRemain = localSize % nDevs;

        if (myLocalRank < (nBasic+1)*nRemain)
        {
            deviceId = myLocalRank / (nBasic + 1);
            if (myLocalRank % (nBasic + 1) == 0)  gpuProc = 0;
        }
        else
        {
            deviceId = (myLocalRank - (nBasic+1)*nRemain) / nBasic + nRemain;
            if ((myLocalRank - (nBasic+1)*nRemain) % nBasic == 0) gpuProc = 0;
        }
    }
}


/* \implements AmgXSolver::initAmgX */
void NvidiaAMGX::initAmgX_new(const std::string &cfgFile)
{

    // only the first instance (AmgX solver) is in charge of initializing AmgX
    if (count == 1)
    {
        // initialize AmgX
        AMGX_SAFE_CALL(AMGX_initialize());

        // intialize AmgX plugings
        AMGX_SAFE_CALL(AMGX_initialize_plugins());

        // only the master process can output something on the screen **** NOT SURE ABOUT THIS
      //  AMGX_SAFE_CALL(AMGX_register_print_callback(
      //              [](const char *msg, int length)->void
      //              {PetscPrintf(PETSC_COMM_WORLD, "%s", msg);}));

        // let AmgX to handle errors returned
        AMGX_SAFE_CALL(AMGX_install_signal_handler());
    }

    // create an AmgX configure object
    AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, cfgFile.c_str()));

    // let AmgX handle returned error codes internally
    AMGX_SAFE_CALL(AMGX_config_add_parameters(&cfg, "exception_handling=1"));

    // create an AmgX resource object, only the first instance is in charge
    if (count == 1) AMGX_resources_create(&rsrc, cfg, &amgx_comm, 1, &deviceId);

    // create AmgX vector object for unknowns and RHS
    AMGX_vector_create(&x, rsrc, amgx_mode);
    AMGX_vector_create(&b, rsrc, amgx_mode);

    // create AmgX matrix object for unknowns and RHS
    AMGX_matrix_create(&A, rsrc, amgx_mode);

    // create an AmgX solver object
    AMGX_solver_create(&solver, rsrc, amgx_mode, cfg);

    // obtain the default number of rings based on current configuration
    AMGX_config_get_default_number_of_rings(cfg, &ring);

}

//END ADAM

}//mfem namespace

#endif
#endif
