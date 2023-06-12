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

#ifndef MFEM_PTRANSFERMAP
#define MFEM_PTRANSFERMAP

#include "../../fem/pgridfunc.hpp"
#include "transfer_category.hpp"
#include <memory>

namespace mfem
{

/**
 * @brief ParTransferMap represents a mapping of degrees of freedom from a
 * source ParGridFunction to a destination ParGridFunction.
 *
 * This map can be constructed from a parent ParMesh to a ParSubMesh or vice
 * versa. Additionally one can create it between two ParSubMeshes that share the
 * same root parent. In this case, a supplemental ParFiniteElementSpace is
 * created on the root parent ParMesh to transfer degrees of freedom.
 */
class ParTransferMap
{
public:
   /**
    * @brief Construct a new ParTransferMap object which transfers degrees of
    * freedom from the source ParGridFunction to the destination
    * ParGridFunction.
    *
    * @param src The source ParGridFunction
    * @param dst The destination ParGridFunction
    */
   ParTransferMap(const ParGridFunction &src,
                  const ParGridFunction &dst);

   /**
    * @brief Transfer the source ParGridFunction to the destination
    * ParGridFunction.
    *
    * Uses the precomputed maps for the transfer.
    *
    * @param src The source ParGridFunction
    * @param dst The destination ParGridFunction
    */
   void Transfer(const ParGridFunction &src, ParGridFunction &dst) const;

private:
   /**
    * @brief Communicate from each local processor which index in map is set.
    *
    * The result is accumulated in the member variable indices_set_global_ and
    * indicates which and how many processors in total will set a certain degree
    * of freedom.
    *
    * Convenience method for tidyness. Uses and changes member variables.
    */
   void CommunicateIndicesSet(Array<int> &map, int dst_sz);

   /**
    * @brief Communicate shared vdofs in Vector f.
    *
    * Guarantees that all ranks have the appropriate dofs set. See comments in
    * implementation for more details.
    *
    * Convenience method for tidyness. Uses and changes member variables.
    */
   void CommunicateSharedVdofs(Vector &f) const;

   TransferCategory category_;

   /// Mapping of the ParGridFunction defined on the SubMesh to the
   /// ParGridFunction of its parent ParMesh.
   Array<int> sub1_to_parent_map_;

   /// Mapping of the ParGridFunction defined on the second SubMesh to the
   /// ParGridFunction of its parent ParMesh. This is only used if this
   /// ParTransferMap represents a ParSubMesh to ParSubMesh transfer.
   Array<int> sub2_to_parent_map_;

   /// Set of indices in the dof map that are set by the local rank.
   Array<int> indices_set_local_;

   /// Set of indices in the dof map that are set by all ranks. The number is
   /// accumulated by summation.
   Array<int> indices_set_global_;

   /// Pointer to the supplemental ParFiniteElementSpace on the common root
   /// parent ParMesh. This is only used if this ParTransferMap represents a
   /// ParSubMesh to ParSubMesh transfer.
   std::unique_ptr<const ParFiniteElementSpace> root_fes_;

   const GroupCommunicator *root_gc_ = nullptr;

   /// Temporary vector
   mutable Vector z_;
};

} // namespace mfem

#endif // MFEM_PTRANSFERMAP
