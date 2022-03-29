// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_LOR_ADS
#define MFEM_LOR_ADS

#include "../../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "lor_rt.hpp"
#include "lor_ams.hpp"
#include "../../linalg/hypre.hpp"

namespace mfem
{

// Helper class for assembling the discrete curl, gradient and coordinate
// vectors needed by the ADS solver. Generally, this class should *not* be
// directly used by users, instead use LORSolver<HypreADS> (which internally
// uses this class).
class BatchedLOR_ADS
{
protected:
   ParFiniteElementSpace &face_fes;
   static constexpr int dim = 3;
   const int order;
   ND_FECollection edge_fec;
   ParFiniteElementSpace edge_fes;
   BatchedLOR_AMS ams;
   HypreParMatrix *C;

   void Form3DFaceToEdge(Array<int> &face2edge);
public:
   BatchedLOR_ADS(ParFiniteElementSpace &pfes_ho_,
                  const Vector &X_vert);
   HypreParMatrix *StealCurlMatrix();
   HypreParMatrix *GetCurlMatrix() const { return C; };
   BatchedLOR_AMS &GetAMS() { return ams; }
   void FormCurlMatrix();
   ~BatchedLOR_ADS();
};

}

#endif

#endif
