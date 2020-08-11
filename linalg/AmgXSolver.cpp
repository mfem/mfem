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
//Reference:
//Pi-Yueh Chuang, & Lorena A. Barba (2017).
//AmgXWrapper: An interface between PETSc and the NVIDIA AmgX library. J. Open Source Software, 2(16):280, doi:10.21105/joss.00280

#include "../config/config.hpp"
#include "AmgXSolver.hpp"
#include <chrono>
#ifdef MFEM_USE_MPI
#ifdef MFEM_USE_AMGX

namespace mfem
{

// initialize AmgXSolver::count to 0
int AmgXSolver::count = 0;

int AmgXSolver::count2 = 0;

// initialize AmgXSolver::rsrc to nullptr;
AMGX_resources_handle AmgXSolver::rsrc = nullptr;

/* \implements AmgXSolver::AmgXSolver */
AmgXSolver::AmgXSolver(const MPI_Comm &comm,
                       const std::string &modeStr, const std::string &cfgFile, int &nDevs)
{
   initialize(comm, modeStr, cfgFile, nDevs);
}

AmgXSolver::AmgXSolver(const std::string &modeStr, const std::string &cfgFile)
{
   initialize(modeStr, cfgFile);
}

/* \implements AmgXSolver::~AmgXSolver */
AmgXSolver::~AmgXSolver()
{
   if (isInitialized) { finalize(); }
}

void AmgXSolver::setMode(const std::string &modeStr)
{
   if (modeStr == "dDDI")
   {
      mode = AMGX_mode_dDDI;
   }
   else { mfem_error("Mode not supported \n"); }
}

void AmgXSolver::getNumIterations(int &getIters)
{
   AMGX_solver_get_iterations_number(solver, &getIters);
}



/* \implements AmgXSolver::initialize */
void AmgXSolver::initialize(const MPI_Comm &comm,
                            const std::string &modeStr, const std::string &cfgFile, int &nDevs)
{

   // if this instance has already been initialized, skip
   if (isInitialized)
   {
      mfem_error("This AmgXSolver instance has been initialized on this process.");
   }

   // increase the number of AmgXSolver instances
   count += 1;

   // get the name of this node
   int     len;
   char    name[MPI_MAX_PROCESSOR_NAME];
   MPI_Get_processor_name(name, &len);
   nodeName = name;
   int globalcommrank;

   MPI_Comm_rank(comm, &globalcommrank);

   // get the mode of AmgX solver
   setMode(modeStr);

   // initialize communicators and corresponding information
   initMPIcomms(comm, nDevs);

   // only processes in gpuWorld are required to initialize AmgX
   if (gpuProc == 0)
   {
      initAmgX(cfgFile);
   }

   // a bool indicating if this instance is initialized
   isInitialized = true;
}

/* \implements AmgXSolver::initialize  for serial runs*/
void AmgXSolver::initialize(const std::string &modeStr,
                            const std::string &cfgFile)
{

   // get the mode of AmgX solver
   setMode(modeStr);

   // initialize AmgX
   AMGX_SAFE_CALL(AMGX_initialize());
   // intialize AmgX plugings
   AMGX_SAFE_CALL(AMGX_initialize_plugins());
   // let AmgX to handle errors returned
   AMGX_SAFE_CALL(AMGX_install_signal_handler());

   AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, cfgFile.c_str()));

   AMGX_resources_create_simple(&rsrc, cfg);
   AMGX_solver_create(&solver, rsrc, mode, cfg);
   AMGX_matrix_create(&AmgXA, rsrc, mode);
   AMGX_vector_create(&AmgXP, rsrc, mode);
   AMGX_vector_create(&AmgXRHS, rsrc, mode);

   // a bool indicating if this instance is initialized
   isInitialized = true;
}


/* \implements AmgXSolver::initAmgX */
void AmgXSolver::initAmgX(const std::string &cfgFile)
{

   // only the first instance (AmgX solver) is in charge of initializing AmgX
   if (count == 1)
   {

      // initialize AmgX
      AMGX_SAFE_CALL(AMGX_initialize());


      // intialize AmgX plugings
      AMGX_SAFE_CALL(AMGX_initialize_plugins());

      // let AmgX to handle errors returned
      AMGX_SAFE_CALL(AMGX_install_signal_handler());
   }



   // create an AmgX configure object
   AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, cfgFile.c_str()));


   // let AmgX handle returned error codes internally
   AMGX_SAFE_CALL(AMGX_config_add_parameters(&cfg, "exception_handling=1"));


   //AMGX_config_create(&cfg, "communicator=MPI, min_rows_latency_hiding=50000");

   // create an AmgX resource object, only the first instance is in charge
   if (count == 1) { AMGX_resources_create(&rsrc, cfg, &gpuWorld, 1, &devID); }



   // create AmgX vector object for unknowns and RHS
   AMGX_vector_create(&AmgXP, rsrc, mode);
   AMGX_vector_create(&AmgXRHS, rsrc, mode);
   // create AmgX matrix object for unknowns and RHS
   AMGX_matrix_create(&AmgXA, rsrc, mode);

   // create an AmgX solver object
   AMGX_solver_create(&solver, rsrc, mode, cfg);
   // obtain the default number of rings based on current configuration
   AMGX_config_get_default_number_of_rings(cfg, &ring);
}

void AmgXSolver::InitializeAsPreconditioner(bool verbose,
                                            const std::string &modeStr)
{

   // get the mode of AmgX solver
   setMode(modeStr);

   // initialize AmgX
   AMGX_SAFE_CALL(AMGX_initialize());
   // intialize AmgX plugings
   AMGX_SAFE_CALL(AMGX_initialize_plugins());
   // let AmgX to handle errors returned
   AMGX_SAFE_CALL(AMGX_install_signal_handler());

   std::string configs = "{\n"
                         "    \"config_version\": 2, \n"
                         "    \"solver\": {\n"
                         "        \"max_uncolored_percentage\": 0.15, \n"
                         "        \"algorithm\": \"AGGREGATION\", \n"
                         "        \"solver\": \"AMG\", \n"
                         "        \"smoother\": \"MULTICOLOR_GS\", \n"
                         "        \"presweeps\": 1, \n"
                         "        \"symmetric_GS\": 1, \n"
                         "        \"selector\": \"SIZE_2\", \n"
                         "        \"coarsest_sweeps\": 10, \n"
                         "        \"max_iters\": 200, \n"
                         "        \"postsweeps\": 1, \n"
                         "        \"scope\": \"main\", \n"
                         "        \"max_levels\": 1000, \n"
                         "        \"matrix_coloring_scheme\": \"MIN_MAX\", \n"
                         "        \"tolerance\": 0.0, \n"
                         "        \"norm\": \"L2\", \n"
                         "        \"cycle\": \"V\"";

   if (verbose)
   {
      configs = configs + ",\n"
                "        \"obtain_timings\": 1, \n"
                "        \"monitor_residual\": 1, \n"
                "        \"print_grid_stats\": 1, \n"
                "        \"print_solve_stats\": 1 \n";
   }
   else
   {
      configs = configs + "\n";
   }
   configs = configs + "    }\n" + "}\n";

   AMGX_SAFE_CALL(AMGX_config_create(&cfg, configs.c_str()));

   AMGX_resources_create_simple(&rsrc, cfg);
   AMGX_solver_create(&solver, rsrc, mode, cfg);
   AMGX_matrix_create(&AmgXA, rsrc, mode);
   AMGX_vector_create(&AmgXP, rsrc, mode);
   AMGX_vector_create(&AmgXRHS, rsrc, mode);

   // a bool indicating if this instance is initialized
   isInitialized = true;
}


/* \implements AmgXSolver::initMPIcomms */
void AmgXSolver::initMPIcomms(const MPI_Comm &comm, int &nDevs)
{
   // duplicate the global communicator
   MPI_Comm_dup(comm, &globalCpuWorld);
   MPI_Comm_set_name(globalCpuWorld, "globalCpuWorld");

   // get size and rank for global communicator
   MPI_Comm_size(globalCpuWorld, &globalSize);
   MPI_Comm_rank(globalCpuWorld, &myGlobalRank);

   // Get the communicator for processors on the same node (local world)
   MPI_Comm_split_type(globalCpuWorld,
                       MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localCpuWorld);
   MPI_Comm_set_name(localCpuWorld, "localCpuWorld");

   // get size and rank for local communicator
   MPI_Comm_size(localCpuWorld, &localSize);
   MPI_Comm_rank(localCpuWorld, &myLocalRank);

   // set up corresponding ID of the device used by each local process
   setDeviceIDs(nDevs);

   MPI_Barrier(globalCpuWorld);

   // split the global world into a world involved in AmgX and a null world
   MPI_Comm_split(globalCpuWorld, gpuProc, 0, &gpuWorld);

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
   MPI_Comm_split(localCpuWorld, devID, 0, &devWorld);
   MPI_Comm_set_name(devWorld, "devWorld");

   // get size and rank for the communicator corresponding to myWorld
   MPI_Comm_size(devWorld, &devWorldSize);
   MPI_Comm_rank(devWorld, &myDevWorldRank);

   MPI_Barrier(globalCpuWorld);
}

/* \implements AmgXSolver::setDeviceCount */
void AmgXSolver::setDeviceCount()
{

   // get the number of devices that AmgX solvers can use
   switch (mode)
   {
      case AMGX_mode_dDDI: // for GPU cases, nDevs is the # of local GPUs
      case AMGX_mode_dDFI: // for GPU cases, nDevs is the # of local GPUs
      case AMGX_mode_dFFI: // for GPU cases, nDevs is the # of local GPUs
         // get the number of total cuda devices
         CHECK(cudaGetDeviceCount(&nDevs));

         // Check whether there is at least one CUDA device on this node
         if (nDevs == 0)
         {
            printf("There is no CUDA device on the node %s !\n", nodeName.c_str());
            mfem_error("No CUDA devices found \n");
         }

         break;
      case AMGX_mode_hDDI: // for CPU cases, nDevs is the # of local processes
      case AMGX_mode_hDFI: // for CPU cases, nDevs is the # of local processes
      case AMGX_mode_hFFI: // for CPU cases, nDevs is the # of local processes
      default:
         nDevs = localSize;
         break;
   }

}


/* \implements AmgXSolver::setDeviceIDs */
void AmgXSolver::setDeviceIDs(int &nDevs)
{

   // set the ID of device that each local process will use
   if (nDevs == localSize) // # of the devices and local precosses are the same
   {
      devID = myLocalRank;
      gpuProc = 0;
   }
   else if (nDevs > localSize) // there are more devices than processes
   {
      printf("CUDA devices on the node %s "
             "are more than the MPI processes launched. Only %d CUDA "
             "devices will be used.\n", nodeName.c_str(),nDevs);

      devID = myLocalRank;
      gpuProc = 0;
   }
   else // there more processes than devices
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


void AmgXSolver::GetLocalA(const HypreParMatrix &in_A, Array<HYPRE_Int> &I,
                           Array<int64_t> &J, Array<double> &Data)
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
   J.SetSize(I[row_len]);
   Data.SetSize(I[row_len]);

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



void AmgXSolver::GatherArray(Array<double> &inArr, Array<double> &outArr,
                             int MPI_SZ, MPI_Comm &mpiTeam)
{
   //Calculate number of elements to be collected from each process
   mfem::Array<int> Apart(MPI_SZ);
   int locAsz = inArr.Size();
   MPI_Allgather(&locAsz, 1, MPI_INT,
                 Apart.GetData(),1, MPI_INT,mpiTeam);

   MPI_Barrier(mpiTeam);

   //Determine stride for process
   mfem::Array<int> Adisp(MPI_SZ);
   Adisp[0] = 0;
   for (int i=1; i<MPI_SZ; ++i)
   {
      Adisp[i] = Adisp[i-1] + Apart[i-1];
   }

   MPI_Gatherv(inArr.HostReadWrite(), inArr.Size(), MPI_DOUBLE,
               outArr.HostWrite(), Apart.HostRead(), Adisp.HostRead(),
               MPI_DOUBLE, 0, mpiTeam);
}

void AmgXSolver::GatherArray(Vector &inArr, Vector &outArr,
                             int MPI_SZ, MPI_Comm &mpiTeam)
{
   //Calculate number of elements to be collected from each process
   mfem::Array<int> Apart(MPI_SZ);
   int locAsz = inArr.Size();
   MPI_Allgather(&locAsz, 1, MPI_INT,
                 Apart.GetData(),1, MPI_INT,mpiTeam);

   MPI_Barrier(mpiTeam);

   //Determine stride for process
   mfem::Array<int> Adisp(MPI_SZ);
   Adisp[0] = 0;
   for (int i=1; i<MPI_SZ; ++i)
   {
      Adisp[i] = Adisp[i-1] + Apart[i-1];
   }

   MPI_Gatherv(inArr.HostReadWrite(), inArr.Size(), MPI_DOUBLE,
               outArr.HostWrite(), Apart.HostRead(), Adisp.HostRead(),
               MPI_DOUBLE, 0, mpiTeam);
}

void AmgXSolver::GatherArray(Array<int>&Apart, Array<int> &inArr,
                             Array<int> &outArr,
                             int MPI_SZ, MPI_Comm &mpiTeam)
{
   //Calculate number of elements to be collected from each process
   Apart.SetSize(MPI_SZ);
   int locAsz = inArr.Size();
   MPI_Allgather(&locAsz, 1, MPI_INT,
                 Apart.GetData(),1, MPI_INT,mpiTeam);

   MPI_Barrier(mpiTeam);

   //Determine stride for process
   mfem::Array<int> Adisp(MPI_SZ);
   Adisp[0] = 0;
   for (int i=1; i<MPI_SZ; ++i)
   {
      Adisp[i] = Adisp[i-1] + Apart[i-1];
   }

   MPI_Gatherv(inArr.HostReadWrite(), inArr.Size(), MPI_INT,
               outArr.HostWrite(), Apart.HostRead(), Adisp.HostRead(),
               MPI_INT, 0, mpiTeam);
}

void AmgXSolver::GatherArray(Array<int64_t> &inArr, Array<int64_t> &outArr,
                             int MPI_SZ, MPI_Comm &mpiTeam)
{
   //Calculate number of elements to be collected from each process
   mfem::Array<int> Apart(MPI_SZ);
   int locAsz = inArr.Size();
   MPI_Allgather(&locAsz, 1, MPI_INT,
                 Apart.GetData(),1, MPI_INT,mpiTeam);

   MPI_Barrier(mpiTeam);

   //Determine stride for process
   mfem::Array<int> Adisp(MPI_SZ);
   Adisp[0] = 0;
   for (int i=1; i<MPI_SZ; ++i)
   {
      Adisp[i] = Adisp[i-1] + Apart[i-1];
   }

   MPI_Gatherv(inArr.HostReadWrite(), inArr.Size(), MPI_INT64_T,
               outArr.HostWrite(), Apart.HostRead(), Adisp.HostRead(),
               MPI_INT64_T, 0, mpiTeam);

   MPI_Barrier(mpiTeam);
}


void AmgXSolver::setA(const HypreParMatrix &A)
{
   //Want to work in devWorld, rank 0 is team leader
   //and will talk to the gpu
   //printf("devWorld rank %d device id %d gpuProc %d \n",
   //myDevWorldRank, devID, gpuProc);
   //Local processor data
   Array<int> loc_I;
   Array<int64_t> loc_J;
   Array<double> loc_A;


   // create an AmgX solver object
   GetLocalA(A, loc_I, loc_J, loc_A);

   //
   //Send data to devWorld team lead
   //
   Array<int> all_I;
   Array<int64_t> all_J;
   Array<double> all_A;

   //
   //Determine array sizes
   int J_allsz(0), all_NNZ(0), nDevRows(0);
   const int loc_row_len = std::abs(A.RowPart()[1] -
                                    A.RowPart()[0]); //end of row partition
   const int loc_Jz_sz = loc_J.Size();
   const int loc_A_sz = loc_A.Size();

   //printf("loc_jz_sz %d\n",loc_Jz_sz);
   //printf("loc_A_sz %d\n",loc_A_sz);
   MPI_Allreduce(&loc_row_len, &nDevRows, 1, MPI_INT, MPI_SUM, devWorld);
   MPI_Allreduce(&loc_Jz_sz, &J_allsz, 1, MPI_INT, MPI_SUM, devWorld);
   MPI_Allreduce(&loc_A_sz, &all_NNZ, 1, MPI_INT, MPI_SUM, devWorld);

   MPI_Barrier(devWorld);

   if (myDevWorldRank == 0)
   {
      all_I.SetSize(nDevRows+devWorldSize);
      all_J.SetSize(J_allsz); all_J = 0.0;
      all_A.SetSize(all_NNZ);
   }

   mfem::Array<int> I_rowInfo;


   GatherArray(I_rowInfo, loc_I, all_I, devWorldSize, devWorld);
   GatherArray(loc_J, all_J, devWorldSize, devWorld);
   GatherArray(loc_A, all_A, devWorldSize, devWorld);

   MPI_Barrier(devWorld);

   int local_nnz(0);
   int64_t local_rows(0);

   if (myDevWorldRank == 0)
   {

      Array<int> z_ind(devWorldSize+1);
      int iter = 1;
      while (iter < devWorldSize-1)
      {

         //Determine the indices of zeros in global all_I array
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
         //End of determining indices of zeros in global all_I Array

         //Bump all_I
         for (int idx=z_ind[1]+1; idx < z_ind[2]; idx++)
         {
            all_I[idx] = all_I[idx-1] + (all_I[idx+1] - all_I[idx]);
         }

         //Shift array after bump to remove uncesssary values in middle of array
         for (int idx=z_ind[2]; idx < all_I.Size()-1; ++idx)
         {
            all_I[idx] = all_I[idx+1];
         }
         iter++;
      }

      // LAST TIME THROUGH ARRAY
      //Determine the indices of zeros in global row_ptr array
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
      //End of determining indices of zeros in global all_I Array\
      //BUMP all_I one last time
      for (int idx=z_ind[1]+1; idx < all_I.Size()-1; idx++)
      {
         all_I[idx] = all_I[idx-1] + (all_I[idx+1] - all_I[idx]);
      }
      local_nnz = all_I[all_I.Size()-devWorldSize];
      local_rows = nDevRows;

   }

   //Create row partition
   m_local_rows = local_rows; //class copy
   mfem::Array<int64_t> rowPart;


   if (gpuProc == 0)
   {
      rowPart.SetSize(gpuWorldSize+1); rowPart=0;

      MPI_Allgather(&local_rows, 1, MPI_INT64_T,
                    &rowPart.GetData()[1], 1, MPI_INT64_T,
                    gpuWorld);
      MPI_Barrier(gpuWorld);
      //rowPart[gpuWorldSize] = A.M();
      //Fixup step
      for (int i=1; i<rowPart.Size(); ++i)
      {
         rowPart[i] += rowPart[i-1];
      }

      //upload A matrix to AmgX
      MPI_Barrier(gpuWorld);
      AMGX_distribution_handle dist;
      AMGX_distribution_create(&dist, cfg);
      AMGX_distribution_set_partition_data(dist, AMGX_DIST_PARTITION_OFFSETS,
                                           rowPart.GetData());


      int nGlobalRows = A.M();
      AMGX_matrix_upload_distributed(AmgXA, nGlobalRows, local_rows,
                                     local_nnz,
                                     1, 1, all_I.HostReadWrite(),
                                     all_J.HostReadWrite(),
                                     all_A.HostReadWrite(),
                                     nullptr, dist);

      AMGX_distribution_destroy(dist);
      MPI_Barrier(gpuWorld);

      AMGX_solver_setup(solver, AmgXA);


      //Bind vectors to A
      AMGX_vector_bind(AmgXP, AmgXA);
      AMGX_vector_bind(AmgXRHS, AmgXA);
   }

}

void AmgXSolver::SetOperator(const Operator& op)
{
   spop = const_cast<SparseMatrix*>(dynamic_cast<const SparseMatrix*>(&op));
   MFEM_VERIFY(spop, "Operator is not of correct type!");

   AMGX_matrix_upload_all(AmgXA, spop->Height(),
                          spop->NumNonZeroElems(),
                          1, 1,
                          spop->ReadWriteI(),
                          spop->ReadWriteJ(),
                          spop->ReadWriteData(), NULL);

   AMGX_solver_setup(solver, AmgXA);
   AMGX_vector_bind(AmgXP, AmgXA);
   AMGX_vector_bind(AmgXRHS, AmgXA);
}

void AmgXSolver::SetOperator(const SparseMatrix &in_A)
{

   AMGX_matrix_upload_all(AmgXA, in_A.Size(),
                          in_A.NumNonZeroElems(),
                          1, 1,
                          in_A.GetI(),
                          in_A.GetJ(),
                          in_A.GetData(), NULL);

   AMGX_solver_setup(solver, AmgXA);
   AMGX_vector_bind(AmgXP, AmgXA);
   AMGX_vector_bind(AmgXRHS, AmgXA);
}

void AmgXSolver::updateA(const HypreParMatrix &A)
{
   //Want to work in devWorld, rank 0 is team leader
   //and will talk to the gpu
   //printf("devWorld rank %d device id %d gpuProc %d \n",
   //myDevWorldRank, devID, gpuProc);
   //Local processor data
   Array<int> loc_I;
   Array<int64_t> loc_J;
   Array<double> loc_A;


   // create an AmgX solver object
   GetLocalA(A, loc_I, loc_J, loc_A);

   //
   //Send data to devWorld team lead
   //
   Array<int> all_I;
   Array<int64_t> all_J;
   Array<double> all_A;

   //
   //Determine array sizes
   int J_allsz(0), all_NNZ(0), nDevRows(0);
   const int loc_row_len = std::abs(A.RowPart()[1] -
                                    A.RowPart()[0]); //end of row partition
   const int loc_Jz_sz = loc_J.Size();
   const int loc_A_sz = loc_A.Size();

   //printf("loc_jz_sz %d\n",loc_Jz_sz);
   //printf("loc_A_sz %d\n",loc_A_sz);
   MPI_Allreduce(&loc_row_len, &nDevRows, 1, MPI_INT, MPI_SUM, devWorld);
   MPI_Allreduce(&loc_Jz_sz, &J_allsz, 1, MPI_INT, MPI_SUM, devWorld);
   MPI_Allreduce(&loc_A_sz, &all_NNZ, 1, MPI_INT, MPI_SUM, devWorld);

   MPI_Barrier(devWorld);

   if (myDevWorldRank == 0)
   {
      all_I.SetSize(nDevRows+devWorldSize);
      all_J.SetSize(J_allsz); all_J = 0.0;
      all_A.SetSize(all_NNZ);
   }

   mfem::Array<int> I_rowInfo;


   GatherArray(I_rowInfo, loc_I, all_I, devWorldSize, devWorld);
   GatherArray(loc_J, all_J, devWorldSize, devWorld);
   GatherArray(loc_A, all_A, devWorldSize, devWorld);

   MPI_Barrier(devWorld);

   int local_nnz(0);
   int64_t local_rows(0);

   if (myDevWorldRank == 0)
   {

      Array<int> z_ind(devWorldSize+1);
      int iter = 1;
      while (iter < devWorldSize-1)
      {

         //Determine the indices of zeros in global all_I array
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
         //End of determining indices of zeros in global all_I Array

         //Bump all_I
         for (int idx=z_ind[1]+1; idx < z_ind[2]; idx++)
         {
            all_I[idx] = all_I[idx-1] + (all_I[idx+1] - all_I[idx]);
         }

         //Shift array after bump to remove uncesssary values in middle of array
         for (int idx=z_ind[2]; idx < all_I.Size()-1; ++idx)
         {
            all_I[idx] = all_I[idx+1];
         }
         iter++;
      }

      // LAST TIME THROUGH ARRAY
      //Determine the indices of zeros in global row_ptr array
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
      //End of determining indices of zeros in global all_I Array\
      //BUMP all_I one last time
      for (int idx=z_ind[1]+1; idx < all_I.Size()-1; idx++)
      {
         all_I[idx] = all_I[idx-1] + (all_I[idx+1] - all_I[idx]);
      }
      local_nnz = all_I[all_I.Size()-devWorldSize];
      local_rows = nDevRows;

   }

   //Create row partition
   m_local_rows = local_rows; //class copy
   mfem::Array<int64_t> rowPart;


   if (gpuProc == 0)
   {
      AMGX_matrix_replace_coefficients(AmgXA,A.M(),local_nnz,all_A,0);
   }

}

void AmgXSolver::GatherArray(Vector &inArr, Vector &outArr,
                             int MPI_SZ, MPI_Comm &mpi_comm, Array<int> &Apart, Array<int> &Adisp)
{
   //Calculate number of elements to be collected from each process
   int locAsz = inArr.Size();
   MPI_Allgather(&locAsz, 1, MPI_INT,
                 Apart.GetData(),1, MPI_INT,mpi_comm);

   MPI_Barrier(mpi_comm);

   //Determine stride for process
   Adisp[0] = 0;
   for (int i=1; i<MPI_SZ; ++i)
   {
      Adisp[i] = Adisp[i-1] + Apart[i-1];
   }

   MPI_Gatherv(inArr.HostReadWrite(), inArr.Size(), MPI_DOUBLE,
               outArr.HostWrite(), Apart.HostRead(), Adisp.HostRead(),
               MPI_DOUBLE, 0, mpi_comm);
}

void AmgXSolver::ScatterArray(Vector &inArr, Vector &outArr,
                              int MPI_SZ, MPI_Comm &mpi_comm, Array<int> &Apart, Array<int> &Adisp)
{

   MPI_Scatterv(inArr.HostReadWrite(),Apart.HostRead(),Adisp.HostRead(),
                MPI_DOUBLE,outArr.HostWrite(),outArr.Size(),
                MPI_DOUBLE, 0, mpi_comm);
}

void AmgXSolver::solve(mfem::Vector &X, mfem::Vector &B)
{

   Vector all_X(m_local_rows);
   Vector all_B(m_local_rows);
   Array<int> Apart_X(devWorldSize);
   Array<int> Adisp_X(devWorldSize);
   Array<int> Apart_B(devWorldSize);
   Array<int> Adisp_B(devWorldSize);

   GatherArray(X, all_X, devWorldSize, devWorld, Apart_X, Adisp_X);
   GatherArray(B, all_B, devWorldSize, devWorld, Apart_B, Adisp_B);
   MPI_Barrier(devWorld);

   if (gpuWorld != MPI_COMM_NULL)
   {

      AMGX_vector_upload(AmgXP, all_X.Size(), 1, all_X.HostReadWrite());
      AMGX_vector_upload(AmgXRHS, all_B.Size(), 1, all_B.HostReadWrite());

      MPI_Barrier(gpuWorld);

      AMGX_solver_solve(solver,AmgXRHS, AmgXP);

      AMGX_SOLVE_STATUS   status;
      AMGX_solver_get_status(solver, &status);
      if (status != AMGX_SOLVE_SUCCESS)
      {
         printf("Amgx failed to solve system, error code %d. \n", status);
      }


      AMGX_vector_download(AmgXP, all_X.HostWrite());

   }

   ScatterArray(all_X, X, devWorldSize, devWorld, Apart_X, Adisp_X);

}

void AmgXSolver::Mult(const Vector& b, Vector& x) const
{
   AMGX_vector_upload(AmgXRHS, b.Size(), 1, b.Read());

   AMGX_vector_set_zero(AmgXP, x.Size(), 1);
   // AMGX_vector_upload(amgx_x, x.Size(), 1, x.Read()); // leads to nans...

   AMGX_solver_solve(solver, AmgXRHS, AmgXP);

   AMGX_vector_download(AmgXP, x.HostWrite());
}





/* \implements AmgXSolver::finalize */
void AmgXSolver::finalize()
{
   // skip if this instance has not been initialized
   if (! isInitialized)
   {
      printf("This AmgXWrapper has not been initialized. "
             "Please initialize it before finalization.\n");
   }

   // only processes using GPU are required to destroy AmgX content
   if (gpuProc == 0)
   {
      // destroy solver instance
      AMGX_solver_destroy(solver);

      // destroy matrix instance
      AMGX_matrix_destroy(AmgXA);

      // destroy RHS and unknown vectors
      AMGX_vector_destroy(AmgXP);
      AMGX_vector_destroy(AmgXRHS);

      // only the last instance need to destroy resource and finalizing AmgX
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

      // destroy gpuWorld
      //MPI_Comm_free(&gpuWorld);
   }

   // re-set necessary variables in case users want to reuse
   // the variable of this instance for a new instance

   gpuProc = MPI_UNDEFINED;
   if (globalCpuWorld != MPI_COMM_NULL)
   {
      MPI_Comm_free(&globalCpuWorld);
      MPI_Comm_free(&localCpuWorld);
      MPI_Comm_free(&devWorld);
   }
   // decrease the number of instances
   count -= 1;

   // change status
   isInitialized = false;
}

}//mfem namespace

#endif
#endif
