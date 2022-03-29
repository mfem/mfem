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

#ifndef MFEM_LOR_AMS
#define MFEM_LOR_AMS

#include "../../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "lor_nd.hpp"
#include "../../linalg/hypre.hpp"

namespace mfem
{

// Helper class for assembling the discrete gradient and coordinate vectors
// needed by the AMS solver. Generally, this class should *not* be directly used
// by users, instead use LORSolver<HypreAMS> (which internally uses this class).
class BatchedLOR_AMS
{
protected:
   ParFiniteElementSpace &edge_fes;
   const int dim;
   const int order;
   H1_FECollection vert_fec;
   ParFiniteElementSpace vert_fes;
   OperatorHandle A;
   Vector *xyz_tvec;
   HypreParMatrix *G;
   HypreParVector *x, *y, *z;

   void Form2DEdgeToVertex(Array<int> &edge2vert);
   void Form2DEdgeToVertex_ND(Array<int> &edge2vert);
   void Form2DEdgeToVertex_RT(Array<int> &edge2vert);
   void Form3DEdgeToVertex(Array<int> &edge2vert);
public:
   BatchedLOR_AMS(ParFiniteElementSpace &pfes_ho_,
                  const Vector &X_vert);
   HypreParMatrix *StealGradientMatrix();
   Vector *StealCoordinateVector();
   HypreParVector *StealXCoordinate();
   HypreParVector *StealYCoordinate();
   HypreParVector *StealZCoordinate();

   HypreParMatrix *GetGradientMatrix() const { return G; };
   HypreParVector *GetXCoordinate() const { return x; };
   HypreParVector *GetYCoordinate() const { return y; };
   HypreParVector *GetZCoordinate() const { return z; };

   // The following should be protected, but contain MFEM_FORALL kernels
   void FormCoordinateVectors(const Vector &X_vert);
   void FormGradientMatrix();
   ~BatchedLOR_AMS();
};

template <typename T> T *StealPointer(T *&ptr)
{
   T *tmp = ptr;
   ptr = nullptr;
   return tmp;
}

}

#endif

#endif
