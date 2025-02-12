// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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

/** @brief
   MFEM wrapper for Nvidia's multigrid library, AmgX (github.com/NVIDIA/AMGX)

   AmgX requires building MFEM with CUDA, and AMGX enabled. For distributed
   memory parallism, MPI and Hypre (version 16.0+) are also required. Although
   CUDA is required for building, the AmgX solver is compatible with a MFEM CPU
   device configuration.

   The AmgXSolver class is designed to work as a solver or preconditioner for
   existing MFEM solvers. The AmgX solver class may be configured in one of
   three ways:

   Serial - Takes a SparseMatrix solves on a single GPU and assumes no MPI
   communication.

   Exclusive GPU - Takes a HypreParMatrix and assumes each MPI rank is paired
   with an Nvidia GPU.

   MPI Teams - Takes a HypreParMatrix and enables flexibility between number of
   MPI ranks, and GPUs. Specifically, MPI ranks are grouped with GPUs and a
   matrix consolidation step is taken so the MPI root of each team performs the
   necessary AmgX library calls. The solution is then broadcasted to appropriate
   ranks. This is particularly useful when configuring MFEM's device as
   CPU. This work is based on the AmgXWrapper of Chuang and Barba. Routines were
   adopted and modified for setting up MPI communicators.

   Examples 1/1p in the examples/amgx directory demonstrate configuring the
   wrapper as a solver and preconditioner, as well as configuring and running
   with exclusive GPU or MPI teams modes.

   This work is partially based on:

      Pi-Yueh Chuang and Lorena A. Barba (2017).
      AmgXWrapper: An interface between PETSc and the NVIDIA AmgX library.
      J. Open Source Software, 2(16):280, doi:10.21105/joss.00280

   See https://github.com/barbagroup/AmgXWrapper.
*/
class AmgXSolver : public Solver
{
public:

   /// Flags to configure AmgXSolver as a solver or preconditioner
   enum AMGX_MODE
   {
      /// Use the preconditioned conjugate gradient method with the AMG
      /// V-cycle used as a proconditioner.  With the default configuration
      /// a block Jacobi smoother is used.
      SOLVER,
      /// Directly apply iterations of the AMG V cycle to the matrix
      /// With the default configuration this will be 2 iterations
      /// with block Jacobi smoother.
      PRECONDITIONER
   };

   /// Flag to check for convergence
   bool ConvergenceCheck;

   /** @brief
      Flags to determine whether user solver settings are defined internally in
      the source code or will be read through an external JSON file.
   */
   enum CONFIG_SRC
   {
      /// Configuration will be read directly from a string
      INTERNAL,
      /// Configure will be read from a specified file
      EXTERNAL,
      UNDEFINED
   };

   AmgXSolver();

   /** @brief
      Configures AmgX with a default configuration based on the AmgX mode, and
      verbosity. Assumes no MPI parallelism.
   */
   AmgXSolver(const AMGX_MODE amgxMode_, const bool verbose);

   /** @brief Initialize the AmgX library for serial execution once
      the solver configuration has been established through either the
      AmgXSolver::ReadParameters method or the constructor.  The constructor
      will make this call.
   */
   void InitSerial();

#ifdef MFEM_USE_MPI

   /** @brief
      Configures AmgX with a default configuration based on the AMGX_MODE
      (AmgXSolver::SOLVER, AmgXSolver::PRECONDITIONER)
      and verbosity. Pairs each MPI rank with one GPU.
   */
   AmgXSolver(const MPI_Comm &comm, const AMGX_MODE amgxMode_,
              const bool verbose);

   /** @brief
      Configures AmgX with a default configuration based on the AMGX_MODE
      (AmgXSolver::SOLVER, AmgXSolver::PRECONDITIONER)
      and verbosity. Creates MPI teams around GPUs to support more ranks than
      GPUs. Consolidates linear solver data to avoid multiple ranks sharing
      GPUs. Requires specifying the number  of devices in each compute node as
      @a nDevs.
   */
   AmgXSolver(const MPI_Comm &comm, const int nDevs,
              const AMGX_MODE amgx_Mode_, const bool verbose);

   /** @brief Initialize the AmgX library in parallel mode with exactly one
      GPU per rank after the solver configuration has been established,
      either through the constructor or the AmgXSolver::ReadParameters
      method.  If configuring with a constructor, the constructor will make
      this call.
   */
   void InitExclusiveGPU(const MPI_Comm &comm);

   /** @brief Initialize the AmgX library and create MPI teams based on the
      number of devices on each node @a nDevs.  If configuring with a
      constructor, the constructor will make this call, otherwise this will need
      to be called after the solver configuration has been established through
      the AmgXSolver::ReadParameters call.
   */
   void InitMPITeams(const MPI_Comm &comm,
                     const int nDevs);
#endif

   /** @brief  Sets the Operator that is going to be solved via AmgX.
      Supports operators based on either an MFEM SparseMatrix or
      HypreParMatrix.
   */
   void SetOperator(const Operator &op) override;

   /** @brief Change the input operator that is being solved via AmgX.
      Supports operators based on either an MFEM SparseMatrix or
      HypreParMatrix.
   */
   void UpdateOperator(const Operator &op);

   /** @brief Utilize the AmgX library to solve the linear system
       where the "matrix" is the AMG approximation to the operator set
       by AmgXSolver::SetOperator.  If the mode is set to
       AmgXSolver::PRECONDITIONER the initial guess for the
       @a x vector will be set to zero, otherwise the value of @a x passed
       in will be used.
   */
   void Mult(const Vector& b, Vector& x) const override;

   /** @brief Return the number of iterations that were executed during the
       last solve phase. */
   int GetNumIterations();

   /** @brief Read in the AmgX parameters either through a file or directly
       through a properly formated string.  If @a source is set to
       AmgXSolver::EXTERNAL the parameters are loaded from a filename set by
       @a config.  If @a source is set to AmgXSolver::INTERNAL the parameters
       are set directly by the string defined by @a config.
   */
   void ReadParameters(const std::string config, CONFIG_SRC source);

   /** @brief  Set up the AmgX library with the default paramaters.
      @param [in] amgxMode_ AmgXSolver::PRECONDITIONER,
                            AmgXSolver::SOLVER.

      @param [in] verbose  true, false. Specifies the level
                           of verbosity.

      When configured as a preconditioner, the default configuration is to apply
      two iterations of an AMG V cycle with AmgX's default smoother (block
      Jacobi).

      When configured as a solver the preconditioned conjugate gradient method
      is used with the AMG V-cycle and a block Jacobi smoother is used as a
      preconditioner.
   */
   void DefaultParameters(const AMGX_MODE amgxMode_, const bool verbose);

   /// Add a check for convergence after applying Mult.
   void SetConvergenceCheck(bool setConvergenceCheck_=true);

   /// Close down the AmgX library and free up any MPI Comms set up for it
   ~AmgXSolver();

   /// Close down the AmgX library and free up any MPI Comms set up for it
   void Finalize();

private:

   AMGX_MODE amgxMode;

   std::string amgx_config = "";

   CONFIG_SRC configSrc = UNDEFINED;

#ifdef MFEM_USE_MPI
   /** @brief  Consolidates matrix diagonal and off diagonal data and uploads
       matrix to AmgX. */
   void SetMatrixMPIGPUExclusive(const HypreParMatrix &A,
                                 const Array<double> &loc_A,
                                 const Array<int> &loc_I,
                                 const Array<int64_t> &loc_J,
                                 const bool update_mat = false);

   /** @brief Consolidates matrix diagonal and off diagonal data for all ranks
       in an MPI team. Root rank of each MPI team holds the consolidated
       data and matrix. */
   void SetMatrixMPITeams(const HypreParMatrix &A, const Array<double> &loc_A,
                          const Array<int> &loc_I, const Array<int64_t> &loc_J,
                          const bool update_mat = false);

   /// Consolidate array data to the root node in a MPI team.
   void GatherArray(const Array<double> &inArr, Array<double> &outArr,
                    const int mpiTeamSz, const MPI_Comm &mpiTeam) const;

   /// Consolidate array data to the root node in a MPI team.
   void GatherArray(const Vector &inArr, Vector &outArr,
                    const int mpiTeamSz, const MPI_Comm &mpiTeam) const;

   /// Consolidate array data to the root node in a MPI team.
   void GatherArray(const Array<int> &inArr, Array<int> &outArr,
                    const int mpiTeamSz, const MPI_Comm &mpiTeam) const;

   /// Consolidate array data to the root node in a MPI team.
   void GatherArray(const Array<int64_t> &inArr, Array<int64_t> &outArr,
                    const int mpiTeamSz, const MPI_Comm &mpiTeam) const;

   /** @brief Consolidate array data to the root node in a MPI
      team as well as store array partitions and displacements in
      @a Apart and @a Adisp.
   */
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

   // Total number of MPI procs in a node + rank id
   int                     localSize;

   int                     myLocalRank;

   // Total number of MPI ranks sharing a device + rank id
   int                     devWorldSize;

   int                     myDevWorldRank;

   // Total number of MPI procs calling AmgX + rank id
   int                     gpuWorldSize;

   int                     myGpuWorldRank;
#endif

   // A parameter used by AmgX.
   int                     ring;

   // Sets AmgX precision (currently on double is supported)
   AMGX_Mode               precision_mode = AMGX_mode_dDDI;

   // AmgX config object.
   AMGX_config_handle      cfg = nullptr;

   // AmgX matrix object.
   AMGX_matrix_handle      AmgXA = nullptr;

   // AmgX vector object representing unknowns.
   AMGX_vector_handle      AmgXP = nullptr;

   // AmgX vector object representing RHS.
   AMGX_vector_handle      AmgXRHS = nullptr;

   // AmgX solver object.
   AMGX_solver_handle      solver = nullptr;

   // AmgX resource object.
   static AMGX_resources_handle   rsrc;

   /// Set the ID of the corresponding GPU used by this process.
   void SetDeviceIDs(const int nDevs);

   /// Initialize all MPI communicators.
#ifdef MFEM_USE_MPI
   void InitMPIcomms(const MPI_Comm &comm, const int nDevs);
#endif

   void InitAmgX();

   // Row partition for the HypreParMatrix
   int64_t mat_local_rows;

   std::string mpi_gpu_mode;
};
}
#endif // MFEM_USE_AMGX
#endif // MFEM_AMGX_SOLVER
