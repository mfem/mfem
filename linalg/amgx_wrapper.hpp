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

#ifndef MFEM_AMGX
#define MFEM_AMGX

#include <amgx_c.h>
#include <mpi.h>
#include "hypre.hpp"

//using Int_64 = long long int; //needed for Amgx

//Reference:
//Pi-Yueh Chuang, & Lorena A. Barba (2017).
//AmgXWrapper: An interface between PETSc and the NVIDIA AmgX library. J. Open Source Software, 2(16):280, doi:10.21105/joss.00280

namespace mfem
{

class NvidiaAMGX
{

private:

  //Only first instance will setup/teardown AMGX
  static int count;

  bool isEnabled{false};

  //Number of gpus - assume same as MPI procs
  int nDevs, deviceId;

  int MPI_SZ, MPI_RANK;

  AMGX_Mode amgx_mode;

  MPI_Comm amgx_comm; //amgx communicator

  //Amgx matrices and vectors
  int ring;
  AMGX_matrix_handle      A{nullptr};
  AMGX_vector_handle x{nullptr}, b{nullptr};
  AMGX_solver_handle solver{nullptr};

  AMGX_config_handle  cfg;

  static AMGX_resources_handle   rsrc;

//Reference impl: PETSc MatMPIAIJGetLocalMat method
//used to merge Diagonal and OffDiagonal blocks in a ParCSR matrix
void GetLocalA(const HypreParMatrix &in_A, Array<int> &I,
               Array<int64_t> &J, Array<double> &Aloc);


public:

  NvidiaAMGX() = default;

  //Constructor
  NvidiaAMGX(const MPI_Comm &comm,
             const std::string &modeStr, const std::string &cfgFile);

  void Init(const MPI_Comm &comm,
            const std::string &modeStr, const std::string &cfgFile);

  void SetA(const HypreParMatrix &A);

  void Solve(Vector &x, Vector &b);

  //Destructor
  ~NvidiaAMGX();
};

}

#endif
