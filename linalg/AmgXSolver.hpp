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

#ifndef MFEM_AMGX_SOLVER
#define MFEM_AMGX_SOLVER

#include "../config/config.hpp"

#ifdef MFEM_USE_AMGX
#ifdef MFEM_USE_MPI

#include <amgx_c.h>
#include <mpi.h>
#include "hypre.hpp"

# define CHECK(call)                                                        \
  {                                                                           \
  const cudaError_t       error = call;                                   \
  if (error != cudaSuccess)                                               \
    {                                                                       \
  printf("Error: %s:%d, code:%d, reason: %s\n",                         \
         __FILE__, __LINE__, error, cudaGetErrorString(error));          \
    }                                                                       \
  }


namespace mfem
{

class AmgXSolver : public Solver
{

public:

   AmgXSolver() = default;

   /* Constructor for serial builds - Should be supported without Hypre and MPI */
   AmgXSolver(const std::string &modeStr, const std::string &cfgFile);

   /* Constructor for mpi exclusive - Needs Hypre and MPI */
   AmgXSolver(const MPI_Comm &comm,
              const std::string &modeStr, const std::string &cfgFile);

   /* Constructor for mpi teams - Needs Hypre and MPI */
   AmgXSolver(const MPI_Comm &comm,
              const std::string &modeStr, const std::string &cfgFile, int &nDevs);

   ~AmgXSolver();

   void Initialize_Serial(const std::string &modeStr, const std::string &cfgFile);

   void Initialize_ExclusiveGPU(const MPI_Comm &comm, const std::string &modeStr,
                                const std::string &cfgFile);

   void Initialize_MPITeams(const MPI_Comm &comm,
                            const std::string &modeStr, const std::string &cfgFile,
                            const int nDevs);

   void finalize();

   void SetA(const HypreParMatrix &A);

   void SetA(const SparseMatrix &A);

   //Setup AMGX
   void SetA_MPI_GPU_Exclusive(const HypreParMatrix &A, const Array<double> &loc_A,
                               const Array<int> &loc_I, const Array<int64_t> &loc_J);

   void SetA_MPI_Teams(const HypreParMatrix &A, const Array<double> &loc_A,
                       const Array<int> &loc_I, const Array<int64_t> &loc_J);

   virtual void SetOperator(const Operator &op);

   virtual void Mult(const Vector& b, Vector& x) const;

   int getNumIterations();

private:

   //The following methods send vectors to root node in a MPI-Team
   void GatherArray(const Array<double> &inArr, Array<double> &outArr,
                    const int mpiTeamSz, MPI_Comm &mpiTeam);

   void GatherArray(const Vector &inArr, Vector &outArr,
                    const int mpiTeamSz, MPI_Comm &mpiTeam);

   void GatherArray(const Array<int> &inArr, Array<int> &outArr,
                    const int mpiTeamSz, MPI_Comm &mpiTeam);

   void GatherArray(const Array<int64_t> &inArr, Array<int64_t> &outArr,
                    const int mpiTeamSz, MPI_Comm &mpiTeam);

   //The following methods send vectors to root node in a MPI-Team
   // partitions and displacements are reused.
   void GatherArray(const Vector &inArr, Vector &outArr,
                    const int mpiTeamSz, const MPI_Comm &mpiTeamComm,
                    Array<int> &Apart, Array<int> &Adisp) const;

   void ScatterArray(const Vector &inArr, Vector &outArr,
                     const int mpiTeamSz, const MPI_Comm &mpi_comm,
                     Array<int> &Apart, Array<int> &Adisp) const;

private:

   static int              count;

   // \brief A flag indicating if this instance has been initialized.
   bool                    isInitialized = false;

   // \brief The name of the node that this MPI process belongs to.
   std::string             nodeName;

   // \brief Number of local GPU devices used by AmgX.
   int                     nDevs;

   // \brief The ID of corresponding GPU device used by this MPI process.
   int                     devID;

   // \brief A flag indicating if this process will talk to GPU.
   int                     gpuProc = MPI_UNDEFINED;

   // \brief A communicator for global world.
   MPI_Comm                globalCpuWorld = MPI_COMM_NULL;

   // \brief A communicator for local world (i.e., in-node).
   MPI_Comm                localCpuWorld;

   // \brief A communicator for processes sharing the same devices.
   MPI_Comm                devWorld;

   // \brief A communicator for MPI processes that can talk to GPUs.
   MPI_Comm                gpuWorld;

   // \brief Size of \ref AmgXSolver::globalCpuWorld "globalCpuWorld".
   int                     globalSize;

   // \brief Size of \ref AmgXSolver::localCpuWorld "localCpuWorld".
   int                     localSize;

   // \brief Size of \ref AmgXSolver::gpuWorld "gpuWorld".
   int                     gpuWorldSize;

   // \brief Size of \ref AmgXSolver::devWorld "devWorld".
   int                     devWorldSize;

   // \brief Rank in \ref AmgXSolver::globalCpuWorld "globalCpuWorld".
   int                     myGlobalRank;

   // \brief Rank in \ref AmgXSolver::localCpuWorld "localCpuWorld".
   int                     myLocalRank;

   // \brief Rank in \ref AmgXSolver::gpuWorld "gpuWorld".
   int                     myGpuWorldRank;

   // \brief Rank in \ref AmgXSolver::devWorld "devWorld".
   int                     myDevWorldRank;

   // \brief A parameter used by AmgX.
   int                     ring;

   // \brief AmgX solver mode.
   AMGX_Mode               mode;

   // \brief AmgX config object.
   AMGX_config_handle      cfg = nullptr;

   // \brief AmgX matrix object.
   AMGX_matrix_handle      AmgXA = nullptr;

   // \brief AmgX vector object representing unknowns.
   AMGX_vector_handle      AmgXP = nullptr;

   // \brief AmgX vector object representing RHS.
   AMGX_vector_handle      AmgXRHS = nullptr;

   // \brief AmgX solver object.
   AMGX_solver_handle      solver = nullptr;

   // \brief AmgX resource object.
   static AMGX_resources_handle   rsrc;

   // \brief Set AmgX solver mode based on the user-provided string.
   void setMode(const std::string &modeStr);

   // \brief Set the ID of the corresponding GPU used by this process.
   void setDeviceIDs(const int nDevs);

   // \brief Initialize all MPI communicators.
   void initMPIcomms(const MPI_Comm &comm, const int nDevs);

   void initAmgX(const std::string &cfgFile);

   int64_t m_local_rows;  //mlocal rows for ranks that talk to the gpu

   std::string mpi_gpu_mode;

};

}

#endif

#endif
#endif
