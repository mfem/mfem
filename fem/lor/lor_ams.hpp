// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
   ParFiniteElementSpace &edge_fes; ///< The Nedelec space.
   const int dim; ///< Spatial dimension.
   const int order; ///< Polynomial degree.
   H1_FECollection vert_fec; ///< The corresponding H1 collection.
   ParFiniteElementSpace vert_fes; ///< The corresponding H1 space.
   Vector *xyz_tvec; ///< Mesh vertex coordinates in true-vector format.
   HypreParMatrix *G; ///< Discrete gradient matrix.

   /// @name Mesh coordinate vectors in HypreParVector format
   ///@{
   HypreParVector *x, *y, *z;
   ///@}

   /// @name Construct the local (elementwise) discrete gradient
   ///@{
   void Form2DEdgeToVertex(Array<int> &edge2vert);
   void Form2DEdgeToVertex_ND(Array<int> &edge2vert);
   void Form2DEdgeToVertex_RT(Array<int> &edge2vert);
   void Form3DEdgeToVertex(Array<int> &edge2vert);
   ///@}
public:
   /// @brief Construct the BatchedLOR_AMS object associated with the Nedelec
   /// space (or RT in 2D) @a pfes_ho_.
   ///
   /// The vector @a X_vert represents the LOR mesh coordinates in E-vector
   /// format, see BatchedLORAssembly::GetLORVertexCoordinates.
   BatchedLOR_AMS(ParFiniteElementSpace &pfes_ho_,
                  const Vector &X_vert);

   /// @name These functions steal the discrete gradient and coordinate vectors.
   ///@{
   /// The caller assumes ownership and @b must delete the returned objects.
   /// Subsequent calls will return nullptr.

   HypreParMatrix *StealGradientMatrix();
   Vector *StealCoordinateVector();
   HypreParVector *StealXCoordinate();
   HypreParVector *StealYCoordinate();
   HypreParVector *StealZCoordinate();

   ///@}

   /// @name These functions return the discrete gradient and coordinate vectors.
   ///@{
   /// The caller does not assume ownership, and must not delete the returned
   /// objects.

   HypreParMatrix *GetGradientMatrix() const { return G; };
   HypreParVector *GetXCoordinate() const { return x; };
   HypreParVector *GetYCoordinate() const { return y; };
   HypreParVector *GetZCoordinate() const { return z; };

   ///@}

   // The following should be protected, but contain mfem::forall kernels

   /// Construct the mesh coordinate vectors (not part of the public API).
   void FormCoordinateVectors(const Vector &X_vert);

   /// Construct the discrete gradient matrix (not part of the public API).
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
