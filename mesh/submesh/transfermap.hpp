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
    * freedom from the source GridFunction to the destination GridFunction.
    *
    * @param src The source GridFunction
    * @param dst The destination Gridfunction
    */
   TransferMap(const GridFunction &src,
               const GridFunction &dst);

   /**
    * @brief Transfer the source GridFunction to the destination GridFunction.
    *
    * Uses the precomputed maps for the transfer.
    *
    * @param src The source GridFunction
    * @param dst The destination Gridfunction
    */
   void Transfer(const GridFunction &src, GridFunction &dst) const;

private:
   TransferCategory category_;

   /// Mapping of the GridFunction defined on the SubMesh to the Gridfunction
   /// of its parent Mesh.
   Array<int> sub1_to_parent_map_;

   /// Mapping of the GridFunction defined on the second SubMesh to the
   /// GridFunction of its parent Mesh. This is only used if this TransferMap
   /// represents a SubMesh to SubMesh transfer.
   Array<int> sub2_to_parent_map_;

   /// Pointer to the supplemental FiniteElementSpace on the common root parent
   /// Mesh. This is only used if this TransferMap represents a SubMesh to
   /// SubMesh transfer.
   std::unique_ptr<const FiniteElementSpace> root_fes_;

   /// Temporary vector
   mutable Vector z_;
};

} // namespace mfem

#endif // MFEM_TRANSFERMAP
