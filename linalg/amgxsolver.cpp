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

// Implementation of the MFEM wrapper for Nvidia's multigrid library, AmgX
//
// This work is partially based on:
//
//    Pi-Yueh Chuang and Lorena A. Barba (2017).
//    AmgXWrapper: An interface between PETSc and the NVIDIA AmgX library.
//    J. Open Source Software, 2(16):280, doi:10.21105/joss.00280
//
// See https://github.com/barbagroup/AmgXWrapper.

#include "../config/config.hpp"
#include "amgxsolver.hpp"
#ifdef MFEM_USE_AMGX

namespace mfem
{

int AmgXSolver::count = 0;

AMGX_resources_handle AmgXSolver::rsrc = nullptr;

AmgXSolver::AmgXSolver(const AMGX_MODE amgxMode_, const bool verbose)
{
   amgxMode = amgxMode_;

   DefaultParameters(amgxMode, verbose);

   InitSerial();
}

#ifdef MFEM_USE_MPI

AmgXSolver::AmgXSolver(const MPI_Comm &comm,
                       const AMGX_MODE amgxMode_, const bool verbose)
{
   std::string config;
   amgxMode = amgxMode_;

   DefaultParameters(amgxMode, verbose);

   InitExclusiveGPU(comm);
}

AmgXSolver::AmgXSolver(const MPI_Comm &comm, const int nDevs,
                       const AMGX_MODE amgxMode_, const bool verbose)
{
   std::string config;
   amgxMode = amgxMode_;

   DefaultParameters(amgxMode_, verbose);

   InitMPITeams(comm, nDevs);
}

#endif

AmgXSolver::~AmgXSolver()
{
   if (isInitialized) { Finalize(); }
}

void AmgXSolver::InitSerial()
{
   count++;

   mpi_gpu_mode = "serial";

   AMGX_SAFE_CALL(AMGX_initialize());

   AMGX_SAFE_CALL(AMGX_initialize_plugins());

   AMGX_SAFE_CALL(AMGX_install_signal_handler());

   MFEM_VERIFY(configSrc != CONFIG_SRC::UNDEFINED,
               "AmgX configuration is not defined \n");

   if (configSrc == CONFIG_SRC::EXTERNAL)
   {
      AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, amgx_config.c_str()));
   }
   else
   {
      AMGX_SAFE_CALL(AMGX_config_create(&cfg, amgx_config.c_str()));
   }

   AMGX_resources_create_simple(&rsrc, cfg);
   AMGX_solver_create(&solver, rsrc, precision_mode, cfg);
   AMGX_matrix_create(&AmgXA, rsrc, precision_mode);
   AMGX_vector_create(&AmgXP, rsrc, precision_mode);
   AMGX_vector_create(&AmgXRHS, rsrc, precision_mode);

   isInitialized = true;
}

#ifdef MFEM_USE_MPI

void AmgXSolver::InitExclusiveGPU(const MPI_Comm &comm)
{
   // If this instance has already been initialized, skip
   if (isInitialized)
   {
      mfem_error("This AmgXSolver instance has been initialized on this process.");
   }

   // Note that every MPI rank may talk to a GPU
   mpi_gpu_mode = "mpi-gpu-exclusive";
   gpuProc = 0;

   // Increment number of AmgX instances
   count++;

   MPI_Comm_dup(comm, &gpuWorld);
   MPI_Comm_size(gpuWorld, &gpuWorldSize);
   MPI_Comm_rank(gpuWorld, &myGpuWorldRank);

   // Each rank will only see 1 device call it device 0
   nDevs = 1, devID = 0;

   InitAmgX();

   isInitialized = true;
}

// Initialize for MPI ranks > GPUs, all devices are visible to all of the MPI
// ranks
void AmgXSolver::InitMPITeams(const MPI_Comm &comm,
                              const int nDevs)
{
   // If this instance has already been initialized, skip
   if (isInitialized)
   {
      mfem_error("This AmgXSolver instance has been initialized on this process.");
   }

   mpi_gpu_mode = "mpi-teams";

   // Increment number of AmgX instances
   count++;

   // Get the name of this node
   int     len;
   char    name[MPI_MAX_PROCESSOR_NAME];
   MPI_Get_processor_name(name, &len);
   nodeName = name;
   int globalcommrank;

   MPI_Comm_rank(comm, &globalcommrank);

   // Initialize communicators and corresponding information
   InitMPIcomms(comm, nDevs);

   // Only processes in gpuWorld are required to initialize AmgX
   if (gpuProc == 0)
   {
      InitAmgX();
   }

   isInitialized = true;
}

#endif

void AmgXSolver::ReadParameters(const std::string config,
                                const CONFIG_SRC source)
{
   amgx_config = config;
   configSrc = source;
}

void AmgXSolver::DefaultParameters(const AMGX_MODE amgxMode_,
                                   const bool verbose)
{
   amgxMode = amgxMode_;

   configSrc = INTERNAL;

   if (amgxMode == AMGX_MODE::PRECONDITIONER)
   {
      amgx_config = "{\n"
                    " \"config_version\": 2, \n"
                    " \"solver\": { \n"
                    "   \"solver\": \"AMG\", \n"
                    "   \"presweeps\": 1, \n"
                    "   \"postsweeps\": 1, \n"
                    "   \"interpolator\": \"D2\", \n"
                    "   \"max_iters\": 2, \n"
                    "   \"convergence\": \"ABSOLUTE\", \n"
                    "   \"cycle\": \"V\"";
      if (verbose)
      {
         amgx_config = amgx_config + ",\n"
                       "   \"obtain_timings\": 1, \n"
                       "   \"monitor_residual\": 1, \n"
                       "   \"print_grid_stats\": 1, \n"
                       "   \"print_solve_stats\": 1 \n";
      }
      else
      {
         amgx_config = amgx_config + "\n";
      }
      amgx_config = amgx_config + " }\n" + "}\n";
   }
   else if (amgxMode == AMGX_MODE::SOLVER)
   {
      amgx_config = "{ \n"
                    " \"config_version\": 2, \n"
                    " \"solver\": { \n"
                    "   \"preconditioner\": { \n"
                    "     \"solver\": \"AMG\", \n"
                    "     \"smoother\": { \n"
                    "     \"scope\": \"jacobi\", \n"
                    "     \"solver\": \"BLOCK_JACOBI\", \n"
                    "     \"relaxation_factor\": 0.7 \n"
                    "       }, \n"
                    "     \"presweeps\": 1, \n"
                    "     \"interpolator\": \"D2\", \n"
                    "     \"max_row_sum\" : 0.9, \n"
                    "     \"strength_threshold\" : 0.25, \n"
                    "     \"max_iters\": 2, \n"
                    "     \"scope\": \"amg\", \n"
                    "     \"max_levels\": 100, \n"
                    "     \"cycle\": \"V\", \n"
                    "     \"postsweeps\": 1 \n"
                    "    }, \n"
                    "  \"solver\": \"PCG\", \n"
                    "  \"max_iters\": 100, \n"
                    "  \"convergence\": \"RELATIVE_MAX\", \n"
                    "  \"scope\": \"main\", \n"
                    "  \"tolerance\": 1e-12, \n"
                    "  \"norm\": \"L2\" ";
      if (verbose)
      {
         amgx_config = amgx_config + ", \n"
                       "        \"obtain_timings\": 1, \n"
                       "        \"monitor_residual\": 1, \n"
                       "        \"print_grid_stats\": 1, \n"
                       "        \"print_solve_stats\": 1 \n";
      }
      else
      {
         amgx_config = amgx_config + "\n";
      }
      amgx_config = amgx_config + "   } \n" + "} \n";
   }
   else
   {
      mfem_error("AmgX mode not supported \n");
   }
}

// Sets up AmgX library for MPI builds
#ifdef MFEM_USE_MPI
void AmgXSolver::InitAmgX()
{
   // Set up once
   if (count == 1)
   {
      AMGX_SAFE_CALL(AMGX_initialize());

      AMGX_SAFE_CALL(AMGX_initialize_plugins());

      AMGX_SAFE_CALL(AMGX_install_signal_handler());

      AMGX_SAFE_CALL(AMGX_register_print_callback(
                        [](const char *msg, int length)->void
      {
         int irank; MPI_Comm_rank(MPI_COMM_WORLD, &irank);
         if (irank == 0) { mfem::out<<msg;} }));
   }

   MFEM_VERIFY(configSrc != CONFIG_SRC::UNDEFINED,
               "AmgX configuration is not defined \n");

   if (configSrc == CONFIG_SRC::EXTERNAL)
   {
      AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, amgx_config.c_str()));
   }
   else
   {
      AMGX_SAFE_CALL(AMGX_config_create(&cfg, amgx_config.c_str()));
   }

   // Let AmgX handle returned error codes internally
   AMGX_SAFE_CALL(AMGX_config_add_parameters(&cfg, "exception_handling=1"));

   // Create an AmgX resource object, only the first instance needs to create
   // the resource object.
   if (count == 1) { AMGX_resources_create(&rsrc, cfg, &gpuWorld, 1, &devID); }

   // Create AmgX vector object for unknowns and RHS
   AMGX_vector_create(&AmgXP, rsrc, precision_mode);
   AMGX_vector_create(&AmgXRHS, rsrc, precision_mode);

   // Create AmgX matrix object for unknowns and RHS
   AMGX_matrix_create(&AmgXA, rsrc, precision_mode);

   // Create an AmgX solver object
   AMGX_solver_create(&solver, rsrc, precision_mode, cfg);

   // Obtain the default number of rings based on current configuration
   AMGX_config_get_default_number_of_rings(cfg, &ring);
}

// Groups MPI ranks into teams and assigns the roots to talk to GPUs
void AmgXSolver::InitMPIcomms(const MPI_Comm &comm, const int nDevs)
{
   // Duplicate the global communicator
   MPI_Comm_dup(comm, &globalCpuWorld);
   MPI_Comm_set_name(globalCpuWorld, "globalCpuWorld");

   // Get size and rank for global communicator
   MPI_Comm_size(globalCpuWorld, &globalSize);
   MPI_Comm_rank(globalCpuWorld, &myGlobalRank);

   // Get the communicator for processors on the same node (local world)
   MPI_Comm_split_type(globalCpuWorld,
                       MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localCpuWorld);
   MPI_Comm_set_name(localCpuWorld, "localCpuWorld");

   // Get size and rank for local communicator
   MPI_Comm_size(localCpuWorld, &localSize);
   MPI_Comm_rank(localCpuWorld, &myLocalRank);

   // Set up corresponding ID of the device used by each local process
   SetDeviceIDs(nDevs);

   MPI_Barrier(globalCpuWorld);

   // Split the global world into a world involved in AmgX and a null world
   MPI_Comm_split(globalCpuWorld, gpuProc, 0, &gpuWorld);

   // Get size and rank for the communicator corresponding to gpuWorld
   if (gpuWorld != MPI_COMM_NULL)
   {
      MPI_Comm_set_name(gpuWorld, "gpuWorld");
      MPI_Comm_size(gpuWorld, &gpuWorldSize);
      MPI_Comm_rank(gpuWorld, &myGpuWorldRank);
   }
   else // for those that will not communicate with the GPU
   {
      gpuWorldSize = MPI_UNDEFINED;
      myGpuWorldRank = MPI_UNDEFINED;
   }

   // Split local world into worlds corresponding to each CUDA device
   MPI_Comm_split(localCpuWorld, devID, 0, &devWorld);
   MPI_Comm_set_name(devWorld, "devWorld");

   // Get size and rank for the communicator corresponding to myWorld
   MPI_Comm_size(devWorld, &devWorldSize);
   MPI_Comm_rank(devWorld, &myDevWorldRank);

   MPI_Barrier(globalCpuWorld);
}

// Determine MPI teams based on available devices
void AmgXSolver::SetDeviceIDs(const int nDevs)
{
   // Set the ID of device that each local process will use
   if (nDevs == localSize) // # of the devices and local process are the same
   {
      devID = myLocalRank;
      gpuProc = 0;
   }
   else if (nDevs > localSize) // there are more devices than processes
   {
      MFEM_WARNING("CUDA devices on the node " << nodeName.c_str() <<
                   " are more than the MPI processes launched. Only "<<
                   nDevs << " devices will be used.\n");
      devID = myLocalRank;
      gpuProc = 0;
   }
   else // in case there are more ranks than devices
   {
      int     nBasic = localSize / nDevs,
              nRemain = localSize % nDevs;

      if (myLocalRank < (nBasic+1)*nRemain)
      {
         devID = myLocalRank / (nBasic + 1);
         if (myLocalRank % (nBasic + 1) == 0) { gpuProc = 0; }
      }
      else
      {
         devID = (myLocalRank - (nBasic+1)*nRemain) / nBasic + nRemain;
         if ((myLocalRank - (nBasic+1)*nRemain) % nBasic == 0) { gpuProc = 0; }
      }
   }
}

void AmgXSolver::GatherArray(const Array<double> &inArr, Array<double> &outArr,
                             const int mpiTeamSz, const MPI_Comm &mpiTeamComm) const
{
   // Calculate number of elements to be collected from each process
   Array<int> Apart(mpiTeamSz);
   int locAsz = inArr.Size();
   MPI_Gather(&locAsz, 1, MPI_INT,
              Apart.HostWrite(),1, MPI_INT,0,mpiTeamComm);

   MPI_Barrier(mpiTeamComm);

   // Determine stride for process (to be used by root)
   Array<int> Adisp(mpiTeamSz);
   int myid; MPI_Comm_rank(mpiTeamComm, &myid);
   if (myid == 0)
   {
      Adisp[0] = 0;
      for (int i=1; i<mpiTeamSz; ++i)
      {
         Adisp[i] = Adisp[i-1] + Apart[i-1];
      }
   }

   MPI_Gatherv(inArr.HostRead(), inArr.Size(), MPI_DOUBLE,
               outArr.HostWrite(), Apart.HostRead(), Adisp.HostRead(),
               MPI_DOUBLE, 0, mpiTeamComm);
}

void AmgXSolver::GatherArray(const Vector &inArr, Vector &outArr,
                             const int mpiTeamSz, const MPI_Comm &mpiTeamComm) const
{
   // Calculate number of elements to be collected from each process
   Array<int> Apart(mpiTeamSz);
   int locAsz = inArr.Size();
   MPI_Gather(&locAsz, 1, MPI_INT,
              Apart.HostWrite(),1, MPI_INT,0,mpiTeamComm);

   MPI_Barrier(mpiTeamComm);

   // Determine stride for process (to be used by root)
   Array<int> Adisp(mpiTeamSz);
   int myid; MPI_Comm_rank(mpiTeamComm, &myid);
   if (myid == 0)
   {
      Adisp[0] = 0;
      for (int i=1; i<mpiTeamSz; ++i)
      {
         Adisp[i] = Adisp[i-1] + Apart[i-1];
      }
   }

   MPI_Gatherv(inArr.HostRead(), inArr.Size(), MPI_DOUBLE,
               outArr.HostWrite(), Apart.HostRead(), Adisp.HostRead(),
               MPI_DOUBLE, 0, mpiTeamComm);
}

void AmgXSolver::GatherArray(const Array<int> &inArr, Array<int> &outArr,
                             const int mpiTeamSz, const MPI_Comm &mpiTeamComm) const
{
   // Calculate number of elements to be collected from each process
   Array<int> Apart(mpiTeamSz);
   int locAsz = inArr.Size();
   MPI_Gather(&locAsz, 1, MPI_INT,
              Apart.GetData(),1, MPI_INT,0,mpiTeamComm);

   MPI_Barrier(mpiTeamComm);

   // Determine stride for process (to be used by root)
   Array<int> Adisp(mpiTeamSz);
   int myid; MPI_Comm_rank(mpiTeamComm, &myid);
   if (myid == 0)
   {
      Adisp[0] = 0;
      for (int i=1; i<mpiTeamSz; ++i)
      {
         Adisp[i] = Adisp[i-1] + Apart[i-1];
      }
   }

   MPI_Gatherv(inArr.HostRead(), inArr.Size(), MPI_INT,
               outArr.HostWrite(), Apart.HostRead(), Adisp.HostRead(),
               MPI_INT, 0, mpiTeamComm);
}


void AmgXSolver::GatherArray(const Array<int64_t> &inArr,
                             Array<int64_t> &outArr,
                             const int mpiTeamSz, const MPI_Comm &mpiTeamComm) const
{
   // Calculate number of elements to be collected from each process
   Array<int> Apart(mpiTeamSz);
   int locAsz = inArr.Size();
   MPI_Gather(&locAsz, 1, MPI_INT,
              Apart.GetData(),1, MPI_INT,0,mpiTeamComm);

   MPI_Barrier(mpiTeamComm);

   // Determine stride for process
   Array<int> Adisp(mpiTeamSz);
   int myid; MPI_Comm_rank(mpiTeamComm, &myid);
   if (myid == 0)
   {
      Adisp[0] = 0;
      for (int i=1; i<mpiTeamSz; ++i)
      {
         Adisp[i] = Adisp[i-1] + Apart[i-1];
      }
   }

   MPI_Gatherv(inArr.HostRead(), inArr.Size(), MPI_INT64_T,
               outArr.HostWrite(), Apart.HostRead(), Adisp.HostRead(),
               MPI_INT64_T, 0, mpiTeamComm);

   MPI_Barrier(mpiTeamComm);
}

void AmgXSolver::GatherArray(const Vector &inArr, Vector &outArr,
                             const int mpiTeamSz, const MPI_Comm &mpiTeamComm,
                             Array<int> &Apart, Array<int> &Adisp) const
{
   // Calculate number of elements to be collected from each process
   int locAsz = inArr.Size();
   MPI_Allgather(&locAsz, 1, MPI_INT,
                 Apart.HostWrite(),1, MPI_INT, mpiTeamComm);

   MPI_Barrier(mpiTeamComm);

   // Determine stride for process
   Adisp[0] = 0;
   for (int i=1; i<mpiTeamSz; ++i)
   {
      Adisp[i] = Adisp[i-1] + Apart[i-1];
   }

   MPI_Gatherv(inArr.HostRead(), inArr.Size(), MPI_DOUBLE,
               outArr.HostWrite(), Apart.HostRead(), Adisp.HostRead(),
               MPI_DOUBLE, 0, mpiTeamComm);
}

void AmgXSolver::ScatterArray(const Vector &inArr, Vector &outArr,
                              const int mpiTeamSz, const MPI_Comm &mpiTeamComm,
                              Array<int> &Apart, Array<int> &Adisp) const
{
   MPI_Scatterv(inArr.HostRead(),Apart.HostRead(),Adisp.HostRead(),
                MPI_DOUBLE,outArr.HostWrite(),outArr.Size(),
                MPI_DOUBLE, 0, mpiTeamComm);
}
#endif

void AmgXSolver::SetMatrix(const SparseMatrix &in_A, const bool update_mat)
{
   if (update_mat == false)
   {
      AMGX_matrix_upload_all(AmgXA, in_A.Height(),
                             in_A.NumNonZeroElems(),
                             1, 1,
                             in_A.ReadI(),
                             in_A.ReadJ(),
                             in_A.ReadData(), NULL);

      AMGX_solver_setup(solver, AmgXA);
      AMGX_vector_bind(AmgXP, AmgXA);
      AMGX_vector_bind(AmgXRHS, AmgXA);
   }
   else
   {
      AMGX_matrix_replace_coefficients(AmgXA,
                                       in_A.Height(),
                                       in_A.NumNonZeroElems(),
                                       in_A.ReadData(), NULL);
   }
}

#ifdef MFEM_USE_MPI

void AmgXSolver::SetMatrix(const HypreParMatrix &A, const bool update_mat)
{
   // Require hypre >= 2.16.
#if MFEM_HYPRE_VERSION < 21600
   mfem_error("Hypre version 2.16+ is required when using AmgX \n");
#endif

   hypre_ParCSRMatrix * A_ptr =
      (hypre_ParCSRMatrix *)const_cast<HypreParMatrix&>(A);

   hypre_CSRMatrix *A_csr = hypre_MergeDiagAndOffd(A_ptr);

   Array<double> loc_A(A_csr->data, (int)A_csr->num_nonzeros);
   const Array<int> loc_I(A_csr->i, (int)A_csr->num_rows+1);

   // Column index must be int64_t so we must promote here
   Array<int64_t> loc_J((int)A_csr->num_nonzeros);
   for (int i=0; i<A_csr->num_nonzeros; ++i)
   {
      loc_J[i] = A_csr->big_j[i];
   }

   // Assumes one GPU per MPI rank
   if (mpi_gpu_mode=="mpi-gpu-exclusive")
   {
      return SetMatrixMPIGPUExclusive(A, loc_A, loc_I, loc_J, update_mat);
   }

   // Assumes teams of MPI ranks are sharing a GPU
   if (mpi_gpu_mode == "mpi-teams")
   {
      return SetMatrixMPITeams(A, loc_A, loc_I, loc_J, update_mat);
   }

   mfem_error("Unsupported MPI_GPU combination \n");
}

void AmgXSolver::SetMatrixMPIGPUExclusive(const HypreParMatrix &A,
                                          const Array<double> &loc_A,
                                          const Array<int> &loc_I,
                                          const Array<int64_t> &loc_J,
                                          const bool update_mat)
{
   // Create a vector of offsets describing matrix row partitions
   Array<int64_t> rowPart(gpuWorldSize+1); rowPart = 0.0;

   int64_t myStart = A.GetRowStarts()[0];

   MPI_Allgather(&myStart, 1, MPI_INT64_T,
                 rowPart.GetData(),1, MPI_INT64_T
                 ,gpuWorld);
   MPI_Barrier(gpuWorld);

   rowPart[gpuWorldSize] = A.M();

   const int nGlobalRows = A.M();
   const int local_rows = loc_I.Size()-1;
   const int num_nnz = loc_I[local_rows];

   if (update_mat == false)
   {
      AMGX_distribution_handle dist;
      AMGX_distribution_create(&dist, cfg);
      AMGX_distribution_set_partition_data(dist, AMGX_DIST_PARTITION_OFFSETS,
                                           rowPart.GetData());

      AMGX_matrix_upload_distributed(AmgXA, nGlobalRows, local_rows,
                                     num_nnz, 1, 1, loc_I.Read(),
                                     loc_J.Read(), loc_A.Read(),
                                     NULL, dist);

      AMGX_distribution_destroy(dist);

      MPI_Barrier(gpuWorld);

      AMGX_solver_setup(solver, AmgXA);

      AMGX_vector_bind(AmgXP, AmgXA);
      AMGX_vector_bind(AmgXRHS, AmgXA);
   }
   else
   {
      AMGX_matrix_replace_coefficients(AmgXA,nGlobalRows,num_nnz,loc_A, NULL);
   }
}

void AmgXSolver::SetMatrixMPITeams(const HypreParMatrix &A,
                                   const Array<double> &loc_A,
                                   const Array<int> &loc_I,
                                   const Array<int64_t> &loc_J,
                                   const bool update_mat)
{
   // The following arrays hold the consolidated diagonal + off-diagonal matrix
   // data
   Array<int> all_I;
   Array<int64_t> all_J;
   Array<double> all_A;

   // Determine array sizes
   int J_allsz(0), all_NNZ(0), nDevRows(0);
   const int loc_row_len = std::abs(A.RowPart()[1] -
                                    A.RowPart()[0]); // end of row partition
   const int loc_Jz_sz = loc_J.Size();
   const int loc_A_sz = loc_A.Size();

   MPI_Reduce(&loc_row_len, &nDevRows, 1, MPI_INT, MPI_SUM, 0, devWorld);
   MPI_Reduce(&loc_Jz_sz, &J_allsz, 1, MPI_INT, MPI_SUM, 0, devWorld);
   MPI_Reduce(&loc_A_sz, &all_NNZ, 1, MPI_INT, MPI_SUM, 0, devWorld);

   MPI_Barrier(devWorld);

   if (myDevWorldRank == 0)
   {
      all_I.SetSize(nDevRows+devWorldSize);
      all_J.SetSize(J_allsz); all_J = 0.0;
      all_A.SetSize(all_NNZ);
   }

   GatherArray(loc_I, all_I, devWorldSize, devWorld);
   GatherArray(loc_J, all_J, devWorldSize, devWorld);
   GatherArray(loc_A, all_A, devWorldSize, devWorld);

   MPI_Barrier(devWorld);

   int local_nnz(0);
   int64_t local_rows(0);

   if (myDevWorldRank == 0)
   {
      // A fix up step is needed for the array holding row data to remove extra
      // zeros when consolidating team data.
      Array<int> z_ind(devWorldSize+1);
      int iter = 1;
      while (iter < devWorldSize-1)
      {
         // Determine the indices of zeros in global all_I array
         int counter = 0;
         z_ind[counter] = counter;
         counter++;
         for (int idx=1; idx<all_I.Size()-1; idx++)
         {
            if (all_I[idx]==0)
            {
               z_ind[counter] = idx-1;
               counter++;
            }
         }
         z_ind[devWorldSize] = all_I.Size()-1;
         // End of determining indices of zeros in global all_I Array

         // Bump all_I
         for (int idx=z_ind[1]+1; idx < z_ind[2]; idx++)
         {
            all_I[idx] = all_I[idx-1] + (all_I[idx+1] - all_I[idx]);
         }

         // Shift array after bump to remove unnecessary values in middle of
         // array
         for (int idx=z_ind[2]; idx < all_I.Size()-1; ++idx)
         {
            all_I[idx] = all_I[idx+1];
         }
         iter++;
      }

      // LAST TIME THROUGH ARRAY
      // Determine the indices of zeros in global row_ptr array
      int counter = 0;
      z_ind[counter] = counter;
      counter++;
      for (int idx=1; idx<all_I.Size()-1; idx++)
      {
         if (all_I[idx]==0)
         {
            z_ind[counter] = idx-1;
            counter++;
         }
      }

      z_ind[devWorldSize] = all_I.Size()-1;
      // End of determining indices of zeros in global all_I Array BUMP all_I
      // one last time
      for (int idx=z_ind[1]+1; idx < all_I.Size()-1; idx++)
      {
         all_I[idx] = all_I[idx-1] + (all_I[idx+1] - all_I[idx]);
      }
      local_nnz = all_I[all_I.Size()-devWorldSize];
      local_rows = nDevRows;
   }

   // Create row partition
   mat_local_rows = local_rows; // class copy
   Array<int64_t> rowPart;
   if (gpuProc == 0)
   {
      rowPart.SetSize(gpuWorldSize+1); rowPart=0;

      MPI_Allgather(&local_rows, 1, MPI_INT64_T,
                    &rowPart.GetData()[1], 1, MPI_INT64_T,
                    gpuWorld);
      MPI_Barrier(gpuWorld);

      // Fixup step
      for (int i=1; i<rowPart.Size(); ++i)
      {
         rowPart[i] += rowPart[i-1];
      }

      // Upload A matrix to AmgX
      MPI_Barrier(gpuWorld);

      int nGlobalRows = A.M();
      if (update_mat == false)
      {
         AMGX_distribution_handle dist;
         AMGX_distribution_create(&dist, cfg);
         AMGX_distribution_set_partition_data(dist, AMGX_DIST_PARTITION_OFFSETS,
                                              rowPart.GetData());

         AMGX_matrix_upload_distributed(AmgXA, nGlobalRows, local_rows,
                                        local_nnz,
                                        1, 1, all_I.ReadWrite(),
                                        all_J.Read(),
                                        all_A.Read(),
                                        nullptr, dist);

         AMGX_distribution_destroy(dist);
         MPI_Barrier(gpuWorld);

         AMGX_solver_setup(solver, AmgXA);

         // Bind vectors to A
         AMGX_vector_bind(AmgXP, AmgXA);
         AMGX_vector_bind(AmgXRHS, AmgXA);
      }
      else
      {
         AMGX_matrix_replace_coefficients(AmgXA,nGlobalRows,local_nnz,all_A,NULL);
      }
   }
}

#endif

void AmgXSolver::SetOperator(const Operator& op)
{
   height = op.Height();
   width = op.Width();

   if (const SparseMatrix* Aptr =
          dynamic_cast<const SparseMatrix*>(&op))
   {
      SetMatrix(*Aptr);
   }
#ifdef MFEM_USE_MPI
   else if (const HypreParMatrix* Aptr =
               dynamic_cast<const HypreParMatrix*>(&op))
   {
      SetMatrix(*Aptr);
   }
#endif
   else
   {
      mfem_error("Unsupported Operator Type \n");
   }
}

void AmgXSolver::UpdateOperator(const Operator& op)
{
   if (const SparseMatrix* Aptr =
          dynamic_cast<const SparseMatrix*>(&op))
   {
      SetMatrix(*Aptr, true);
   }
#ifdef MFEM_USE_MPI
   else if (const HypreParMatrix* Aptr =
               dynamic_cast<const HypreParMatrix*>(&op))
   {
      SetMatrix(*Aptr, true);
   }
#endif
   else
   {
      mfem_error("Unsupported Operator Type \n");
   }
}

void AmgXSolver::Mult(const Vector& B, Vector& X) const
{
   // Set initial guess to zero
   X.UseDevice(true);
   X = 0.0;

   // Mult for serial, and mpi-exclusive modes
   if (mpi_gpu_mode != "mpi-teams")
   {
      AMGX_vector_upload(AmgXP, X.Size(), 1, X.ReadWrite());
      AMGX_vector_upload(AmgXRHS, B.Size(), 1, B.Read());

      if (mpi_gpu_mode != "serial")
      {
#ifdef MFEM_USE_MPI
         MPI_Barrier(gpuWorld);
#endif
      }

      AMGX_solver_solve(solver,AmgXRHS, AmgXP);

      AMGX_SOLVE_STATUS   status;
      AMGX_solver_get_status(solver, &status);
      if (status != AMGX_SOLVE_SUCCESS && amgxMode == SOLVER)
      {
         if (status == AMGX_SOLVE_DIVERGED)
         {
            mfem_error("AmgX solver failed to solve system \n");
         }
         else
         {
            mfem_error("AmgX solver diverged \n");
         }
      }

      AMGX_vector_download(AmgXP, X.Write());
      return;
   }

#ifdef MFEM_USE_MPI
   Vector all_X(mat_local_rows);
   Vector all_B(mat_local_rows);
   Array<int> Apart_X(devWorldSize);
   Array<int> Adisp_X(devWorldSize);
   Array<int> Apart_B(devWorldSize);
   Array<int> Adisp_B(devWorldSize);

   GatherArray(X, all_X, devWorldSize, devWorld, Apart_X, Adisp_X);
   GatherArray(B, all_B, devWorldSize, devWorld, Apart_B, Adisp_B);
   MPI_Barrier(devWorld);

   if (gpuWorld != MPI_COMM_NULL)
   {
      AMGX_vector_upload(AmgXP, all_X.Size(), 1, all_X.ReadWrite());
      AMGX_vector_upload(AmgXRHS, all_B.Size(), 1, all_B.ReadWrite());

      MPI_Barrier(gpuWorld);

      AMGX_solver_solve(solver,AmgXRHS, AmgXP);

      AMGX_SOLVE_STATUS   status;
      AMGX_solver_get_status(solver, &status);
      if (status != AMGX_SOLVE_SUCCESS && amgxMode == SOLVER)
      {
         if (status == AMGX_SOLVE_DIVERGED)
         {
            mfem_error("AmgX solver failed to solve system \n");
         }
         else
         {
            mfem_error("AmgX solver diverged \n");
         }
      }

      AMGX_vector_download(AmgXP, all_X.Write());
   }

   ScatterArray(all_X, X, devWorldSize, devWorld, Apart_X, Adisp_X);
#endif
}

int AmgXSolver::GetNumIterations()
{
   int getIters;
   AMGX_solver_get_iterations_number(solver, &getIters);
   return getIters;
}

void AmgXSolver::Finalize()
{
   // Check instance is initialized
   if (! isInitialized || count < 1)
   {
      mfem_error("Error in AmgXSolver::Finalize(). \n"
                 "This AmgXWrapper has not been initialized. \n"
                 "Please initialize it before finalization.\n");
   }

   // Only processes using GPU are required to destroy AmgX content
#ifdef MFEM_USE_MPI
   if (gpuProc == 0 || mpi_gpu_mode == "serial")
#endif
   {
      // Destroy solver instance
      AMGX_solver_destroy(solver);

      // Destroy matrix instance
      AMGX_matrix_destroy(AmgXA);

      // Destroy RHS and unknown vectors
      AMGX_vector_destroy(AmgXP);
      AMGX_vector_destroy(AmgXRHS);

      // Only the last instance need to destroy resource and finalizing AmgX
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
#ifdef MFEM_USE_MPI
      // destroy gpuWorld
      if (mpi_gpu_mode != "serial")
      {
         MPI_Comm_free(&gpuWorld);
      }
#endif
   }

   // re-set necessary variables in case users want to reuse the variable of
   // this instance for a new instance
#ifdef MFEM_USE_MPI
   gpuProc = MPI_UNDEFINED;
   if (globalCpuWorld != MPI_COMM_NULL)
   {
      MPI_Comm_free(&globalCpuWorld);
      MPI_Comm_free(&localCpuWorld);
      MPI_Comm_free(&devWorld);
   }
#endif
   // decrease the number of instances
   count -= 1;

   // change status
   isInitialized = false;
}

} // mfem namespace

#endif
