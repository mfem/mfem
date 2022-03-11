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
class BatchedLOR_AMS : public BatchedLOR_ND
{
protected:
   ParFiniteElementSpace &edge_fes;
   const int dim;
   const int order;
   H1_FECollection vert_fec;
   ParFiniteElementSpace vert_fes;
   // *WARNING*: the following data members are allocated but not freed by
   // this class. It is the responsibility of the caller to delete the
   // following objects: *A, xyz_tvec, G, x, y, z
   OperatorHandle A;
   Vector *xyz_tvec;
   HypreParMatrix *G;
   HypreParVector *x, *y, *z;

   void Form2DEdgeToVertex(DenseMatrix &edge2vert);
   void Form3DEdgeToVertex(DenseMatrix &edge2vert);
public:
   BatchedLOR_AMS(BilinearForm &a_,
                  ParFiniteElementSpace &pfes_ho_,
                  const Array<int> &ess_dofs_);
   HypreParMatrix &GetAssembledMatrix() const { return *A.As<HypreParMatrix>(); }
   HypreParMatrix *GetGradientMatrix() const { return G; };
   Vector *GetCoordinateVector() const { return xyz_tvec; };
   HypreParVector *GetXCoordinate() const { return x; };
   HypreParVector *GetYCoordinate() const { return y; };
   HypreParVector *GetZCoordinate() const { return z; };
   // The following should be protected, but contain MFEM_FORALL kernels
   void FormCoordinateVectors();
   void FormGradientMatrix();
};

}

#endif

#endif
