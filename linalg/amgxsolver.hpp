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

class AmgXSolver : public Solver
{

public:

   AmgXSolver() = default;

   /* Constructor for serial builds - supported without Hypre and MPI */
   AmgXSolver(const std::string &cfgFile);

   void Initialize_Serial(const std::string &cfgFile);

#ifdef MFEM_USE_MPI
   /* Constructor for MPI-GPU exclusive (1 MPI per GPU) */
   AmgXSolver(const MPI_Comm &comm,
              const std::string &cfgFile);

   /* Constructor for MPI teams (MPI procs share a GPU) */
   /* nDevs specifies number of devices per node */
   AmgXSolver(const MPI_Comm &comm,
              const std::string &cfgFile, int &nDevs);

   void Initialize_ExclusiveGPU(const MPI_Comm &comm, const std::string &cfgFile);

   void Initialize_MPITeams(const MPI_Comm &comm,
                            const std::string &cfgFile,
                            const int nDevs);

   void SetA_MPI_GPU_Exclusive(const HypreParMatrix &A, const Array<double> &loc_A,
                               const Array<int> &loc_I, const Array<int64_t> &loc_J);

   void SetA_MPI_Teams(const HypreParMatrix &A, const Array<double> &loc_A,
                       const Array<int> &loc_I, const Array<int64_t> &loc_J);
#endif

   virtual void SetOperator(const Operator &op);

   virtual void Mult(const Vector& b, Vector& x) const;

   int GetNumIterations();

   enum AMGX_MODE {SOLVER, PRECONDITIONER};

   void SetMode(AMGX_MODE mode) {m_AmgxMode = mode;}

   ~AmgXSolver();

   void Finalize();

private:

   AMGX_MODE m_AmgxMode = SOLVER;

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

   void SetMatrix(const HypreParMatrix &A);
#endif

   void SetMatrix(const SparseMatrix &A);

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

   void InitAmgX(const std::string &cfgFile);

   int64_t m_local_rows;  //mlocal rows for ranks that talk to the gpu

   std::string mpi_gpu_mode;

};

}

#endif //MFEM_USE_AMGX
#endif //MFEM_AMGX_SOLVER
