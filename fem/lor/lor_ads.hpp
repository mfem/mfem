// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
   ParFiniteElementSpace &face_fes; ///< The RT space.
   static constexpr int dim = 3; ///< Spatial dimension, always 3.
   const int order; ///< Polynomial degree.
   ND_FECollection edge_fec; ///< The associated Nedelec collection.
   ParFiniteElementSpace edge_fes; ///< The associated Nedelec space.
   BatchedLOR_AMS ams; ///< The associated AMS object.
   HypreParMatrix *C; ///< The discrete curl matrix.

   /// Form the local elementwise discrete curl matrix.
   void Form3DFaceToEdge(Array<int> &face2edge);
public:
   /// @brief Construct the BatchedLOR_AMS object associated with the 3D RT
   /// space @a pfes_ho_.
   ///
   /// The vector @a X_vert represents the LOR mesh coordinates in E-vector
   /// format, see BatchedLORAssembly::GetLORVertexCoordinates.
   BatchedLOR_ADS(ParFiniteElementSpace &pfes_ho_,
                  const Vector &X_vert);
   /// @brief Steal ownership of the discrete curl matrix.
   ///
   /// The caller assumes ownership and must delete the object. Subsequent calls
   /// will return nullptr.
   HypreParMatrix *StealCurlMatrix();
   /// @brief Return the discrete curl matrix.
   ///
   /// The caller does not assume ownership, and must not delete the object.
   HypreParMatrix *GetCurlMatrix() const { return C; };
   /// Return the associated BatchedLOR_AMS object.
   BatchedLOR_AMS &GetAMS() { return ams; }

   // The following should be protected, but contain mfem::forall kernels

   /// Form the discrete curl matrix (not part of the public API).
   void FormCurlMatrix();
   ~BatchedLOR_ADS();
};

}

#endif

#endif
