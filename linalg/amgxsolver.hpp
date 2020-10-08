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

#ifndef MFEM_AMGX_SOLVER
#define MFEM_AMGX_SOLVER

#include "../config/config.hpp"

#ifdef MFEM_USE_AMGX

#include <amgx_c.h>
#ifdef MFEM_USE_MPI
#include <mpi.h>
#include "hypre.hpp"
#else
#include "operator.hpp"
#include "sparsemat.hpp"
#endif

namespace mfem
{

/**
   MFEM's wrapper for Nvidia's multigrid library,
   AmgX (https://github.com/NVIDIA/AMGX).

   AmgX requires building MFEM with CUDA, AMGX enabled.
   For distributed memory parallism MPI and Hypre
   (version 16.0+) are also required.
   Although CUDA is required for building,
   the AmgX wrapper is compatible with
   a MFEM CPU device configuration.

   The AmgXSolver class is designed to work in
   conjunction with existing MFEM solvers either
   as a preconditioner or a solver. The AmgX
   solver class supports configuration of
   three different modes and uses exising
   MFEM sparse matrix formats.

   Serial - Takes a MFEM::SparseMatrix and
   solves and assumes no MPI communication.

   Exclusive GPU - Takes a HypreParMatrix
   and assumes each MPI rank is paired with
   an Nvidia GPU.

   MPI Teams - Takes a HypreParMatrix and
   enables flexibility between number of MPI
   ranks, and GPUs. Specifically, MPI ranks
   are grouped with GPUs and a matrix consolidation
   step is taken so the MPI root of each team
   performs the necessary AmgX library calls.
   The solution is then broadcasted to appropriate ranks.
   This is particularly useful for codes which
   are not fully ported to GPUs.

   Examples 1,1p demonstrate basic usage with
   default parameters, while examples under the
   amgx folder demonstrate configuring AmgX as
   solver, preconditioner and configuring and
   running with exclusive GPU or MPI teams.

   Reference:
   Pi-Yueh Chuang, & Lorena A. Barba (2017).
   AmgXWrapper: An interface between PETSc and the NVIDIA AmgX library.
   J. Open Source Software, 2(16):280, doi:10.21105/joss.00280

 */
class AmgXSolver : public Solver
{
public:

   enum AMGX_MODE {SOLVER, PRECONDITIONER};
   enum CONFIG_SRC {INTERNAL, EXTERNAL, UNDEFINED};

   AmgXSolver() = default;

   /* Constructor for serial builds - supported without Hypre and MPI */
   AmgXSolver(const AMGX_MODE amgxMode_, const bool verbose);

   void InitSerial();

#ifdef MFEM_USE_MPI

   /* Constructor for MPI-GPU exclusive (1 MPI per GPU) with default parameters*/
   AmgXSolver(const MPI_Comm &comm, const AMGX_MODE amgxMode_, const bool verbose);

   /* Constructor for MPI teams (MPI procs share a GPU) */
   /* nDevs specifies number of devices per node */
   /* with default parameters */
   AmgXSolver(const MPI_Comm &comm, const int nDevs,
              const AMGX_MODE amgx_Mode_, const bool verbose);

   void InitExclusiveGPU(const MPI_Comm &comm);

   void InitMPITeams(const MPI_Comm &comm,
                     const int nDevs);

   void SetMatrixMPIGPUExclusive(const HypreParMatrix &A,
                                 const Array<double> &loc_A,
                                 const Array<int> &loc_I, const Array<int64_t> &loc_J,
                                 const bool update_mat = false);

   void SetMatrixMPITeams(const HypreParMatrix &A, const Array<double> &loc_A,
                          const Array<int> &loc_I, const Array<int64_t> &loc_J,
                          const bool update_mat = false);
#endif

   virtual void SetOperator(const Operator &op);

   void UpdateOperator(const Operator &op);

   virtual void Mult(const Vector& b, Vector& x) const;

   int GetNumIterations();

   void ReadParameters(const std::string config, CONFIG_SRC source);

   /**
      @param [in] AMGX_MODE AmgXSolver::PRECONDITIONER,
                            AmgXSolver::SOLVER.

      @param [in] verbose  true, false. Specifies the level
                           of verbosity.

      When configured as a preconditioner, the default configuration
      is to apply two iterations of an AMG V cycle with AmgX's default
      smoother (block Jacobi).
      As a solver the preconditioned conjugate gradient method with
      the AMG V cycle with a block Jacobi smoother is used a
      preconditioner is used.
   */
   void DefaultParameters(const AMGX_MODE amgxMode_, const bool verbose);

   ~AmgXSolver();

   void Finalize();

private:

   AMGX_MODE amgxMode;

   std::string amgx_config = "";

   CONFIG_SRC configSrc = UNDEFINED;

#ifdef MFEM_USE_MPI
   //The following methods send vectors to the root node in a MPI team
   void GatherArray(const Array<double> &inArr, Array<double> &outArr,
                    const int mpiTeamSz, const MPI_Comm &mpiTeam) const;

   void GatherArray(const Vector &inArr, Vector &outArr,
                    const int mpiTeamSz, const MPI_Comm &mpiTeam) const;

   void GatherArray(const Array<int> &inArr, Array<int> &outArr,
                    const int mpiTeamSz, const MPI_Comm &mpiTeam) const;

   void GatherArray(const Array<int64_t> &inArr, Array<int64_t> &outArr,
                    const int mpiTeamSz, const MPI_Comm &mpiTeam) const;

   //The following methods send vectors to the root node in a MPI team
   //and store array partitions and displacements
   void GatherArray(const Vector &inArr, Vector &outArr,
                    const int mpiTeamSz, const MPI_Comm &mpiTeamComm,
                    Array<int> &Apart, Array<int> &Adisp) const;

   void ScatterArray(const Vector &inArr, Vector &outArr,
                     const int mpiTeamSz, const MPI_Comm &mpi_comm,
                     Array<int> &Apart, Array<int> &Adisp) const;

   void SetMatrix(const HypreParMatrix &A, const bool update_mat = false);
#endif

   void SetMatrix(const SparseMatrix &A, const bool update_mat = false);

   static int              count;

   // Indicate if this instance has been initialized.
   bool                    isInitialized = false;

#ifdef MFEM_USE_MPI
   // The name of the node that this MPI process belongs to.
   std::string             nodeName;

   // Number of local GPU devices used by AmgX.
   int                     nDevs;

   // The ID of corresponding GPU device used by this MPI process.
   int                     devID;

   // A flag indicating if this process will invoke AmgX
   int                     gpuProc = MPI_UNDEFINED;

   // Communicator for all MPI ranks
   MPI_Comm                globalCpuWorld = MPI_COMM_NULL;

   // Communicator for ranks in same node
   MPI_Comm                localCpuWorld;

   // Communicator for ranks sharing a device
   MPI_Comm                devWorld;

   // A communicator for MPI processes that will launch AmgX (root of devWorld)
   MPI_Comm                gpuWorld;

   // Global number of MPI procs + rank id
   int                     globalSize;

   int                     myGlobalRank;

   // Total number of MPI procs in a node
   // + rank id
   int                     localSize;

   int                     myLocalRank;

   // Total number of MPI ranks sharing a device
   // + rank id
   int                     devWorldSize;

   int                     myDevWorldRank;

   // Total number of MPI procs calling AmgX
   // + rank id
   int                     gpuWorldSize;

   int                     myGpuWorldRank;
#endif

   // \brief A parameter used by AmgX.
   int                     ring;

   // \brief AmgX precision.
   AMGX_Mode               precision_mode = AMGX_mode_dDDI;

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

   // \brief Set the ID of the corresponding GPU used by this process.
   void SetDeviceIDs(const int nDevs);

   // \brief Initialize all MPI communicators.
#ifdef MFEM_USE_MPI
   void InitMPIcomms(const MPI_Comm &comm, const int nDevs);
#endif

   void InitAmgX();

   int64_t mat_local_rows;  //mlocal rows for ranks that talk to the gpu

   std::string mpi_gpu_mode;
};
}
#endif //MFEM_USE_AMGX
#endif //MFEM_AMGX_SOLVER
