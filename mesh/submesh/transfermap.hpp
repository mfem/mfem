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

#ifndef MFEM_TRANSFERMAP
#define MFEM_TRANSFERMAP

#include "../../fem/gridfunc.hpp"
#include "transfer_category.hpp"
#include <memory>

namespace mfem
{

/**
 * @brief TransferMap represents a mapping of degrees of freedom from a source
 * GridFunction to a destination GridFunction.
 *
 * This map can be constructed from a parent Mesh to a SubMesh or vice versa.
 * Additionally one can create it between two SubMeshes that share the same root
 * parent. In this case, a supplemental FiniteElementSpace is created on the
 * root parent Mesh to transfer degrees of freedom.
 */
class TransferMap
{
public:
   /**
    * @brief Construct a new TransferMap object which transfers degrees of
    * freedom from the source FiniteElementSpace to the destination
    * FiniteElementSpace.
    *
    * @param src The source FiniteElementSpace
    * @param dst The destination FiniteElementSpace
    */
   TransferMap(const FiniteElementSpace &src, const FiniteElementSpace &dst);

   /**
    * @brief Construct a new TransferMap object which transfers degrees of
    * freedom from the source GridFunction to the destination GridFunction.
    *
    * Equivalent to creating the TransferMap from the finite element spaces of
    * each of the GridFunction%s.
    *
    * @param src The source GridFunction
    * @param dst The destination GridFunction
    */
   TransferMap(const GridFunction &src, const GridFunction &dst);

   /**
    * @brief Transfer the source GridFunction to the destination GridFunction.
    *
    * Uses the precomputed maps for the transfer.
    *
    * @param src The source GridFunction
    * @param dst The destination GridFunction
    */
   void Transfer(const GridFunction &src, GridFunction &dst) const;

private:

   static void CorrectFaceOrientations(const FiniteElementSpace &fes,
                                       const Vector &src,
                                       Vector &dst,
                                       const Array<int> *s2p_map = NULL);

   TransferCategory category_;

   /// Mapping of the GridFunction defined on the SubMesh to the GridFunction
   /// of its parent Mesh.
   Array<int> sub_to_parent_map_;

   /// Pointer to the finite element space defined on the SubMesh.
   const FiniteElementSpace *sub_fes_ = nullptr;

   /// @name Needed for SubMesh-to-SubMesh transfer
   ///@{

   /// Pointer to the supplemental FiniteElementSpace on the common root parent
   /// Mesh. This is only used if this TransferMap represents a SubMesh to
   /// SubMesh transfer.
   std::unique_ptr<FiniteElementSpace> root_fes_;

   /// Pointer to the supplemental FiniteElementCollection used with root_fes_.
   /// This is only used if this TransferMap represents a SubMesh to SubMesh
   /// transfer where the root requires a different type of collection than the
   /// SubMesh objects. For example, when the subpaces are L2 on boundaries of
   /// the parent mesh and the root space can be RT.
   std::unique_ptr<const FiniteElementCollection> root_fec_;

   /// Transfer mapping from the source to the parent (root).
   std::unique_ptr<TransferMap> src_to_parent;

   /// @brief Transfer mapping from the destination to the parent (root).
   ///
   /// SubMesh-to-SubMesh transfer works by bringing both the source and
   /// destination data to their common parent, and then transferring back to
   /// the destination.
   std::unique_ptr<TransferMap> dst_to_parent;

   /// Transfer mapping from the parent to the destination.
   std::unique_ptr<TransferMap> parent_to_dst;

   ///@}

   /// Temporary vector
   mutable GridFunction z_;
};

} // namespace mfem

#endif // MFEM_TRANSFERMAP
