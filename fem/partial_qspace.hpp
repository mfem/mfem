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

#pragma once

#include "../config/config.hpp"
#include "qspace.hpp"

namespace mfem
{

/// Class representing a subset of a QuadratureSpace for efficient operations on subdomains.
/** PartialQuadratureSpace extends MFEM's QuadratureSpaceBase to provide efficient finite
    element operations on subsets of mesh elements. This is particularly useful for handling
    multi-material simulations where different constitutive models apply to different regions
    of the mesh.

    The class maintains bidirectional mappings between local element indices (within the
    partial set) and global element indices (in the full mesh), enabling efficient data
    access and computation while maintaining compatibility with MFEM's finite element
    framework. */
class PartialQuadratureSpace : public mfem::QuadratureSpaceBase
{
protected:
   friend class PartialQuadratureFunction; ///< Uses the offsets.

   /// Maps local element indices to global mesh element indices.
   mfem::Array<int> local2global;

   /// Maps global mesh element indices to local element indices (-1 if not in partial set).
   mfem::Array<int> global2local;

   /// Maps global mesh element indices to quadrature point offsets.
   mfem::Array<int> global_offsets;

   /// Implementation of GetGeometricFactorWeights required by the base class.
   virtual const mfem::Vector &GetGeometricFactorWeights() const override;

   /// Constructs the offset arrays for quadrature points in partial elements.
   void ConstructOffsets();

   /// Constructs global offset arrays for full mesh compatibility.
   void ConstructGlobalOffsets();

   /// Main construction method that builds all internal data structures.
   void Construct();

   /// Constructs local-to-global and global-to-local element mappings.
   void ConstructMappings(mfem::Mesh *mesh_, mfem::Array<bool> &partial_index);

public:
   /// Create a PartialQuadratureSpace based on the global rules from IntRules.
   /** @param[in] mesh_ Pointer to the mesh object.
       @param[in] order_ Integration order for automatic quadrature rule selection.
       @param[in] partial_index Boolean array indicating which elements to include. */
   PartialQuadratureSpace(mfem::Mesh *mesh_, int order_,
                          mfem::Array<bool> &partial_index);

   /// Create a PartialQuadratureSpace with an IntegrationRule.
   /** This constructor is valid only when the mesh has one element type.
       @param[in] mesh_ Pointer to the mesh object.
       @param[in] ir Integration rule to use for all elements.
       @param[in] partial_index Boolean array indicating which elements to include. */
   PartialQuadratureSpace(mfem::Mesh *mesh_, const mfem::IntegrationRule &ir,
                          mfem::Array<bool> &partial_index);

   /// Read a PartialQuadratureSpace from the stream.
   /** @param[in] mesh_ Pointer to the mesh object.
       @param[in] in Input stream containing serialized PartialQuadratureSpace data. */
   PartialQuadratureSpace(mfem::Mesh *mesh_, std::istream &in);

   /// Converts a local element index to the corresponding global element index.
   /** @param[in] local_idx Local element index in the partial quadrature space.
       @return Global element index in the full mesh, or -1 if invalid. */
   inline int LocalToGlobal(int local_idx) const;

   /// Converts a global element index to the corresponding local element index.
   /** @param[in] global_idx Global element index in the full mesh.
       @return Local element index in the partial space, or -1 if not in partial set. */
   inline int GlobalToLocal(int global_idx) const;

   /// Get read-only access to the global-to-local mapping array.
   const mfem::Array<int> &GetGlobal2Local() const { return global2local; }

   /// Get read-only access to the local-to-global mapping array.
   const mfem::Array<int> &GetLocal2Global() const { return local2global; }

   /// Get read-only access to the global offset array.
   const mfem::Array<int> &GetGlobalOffset() const { return global_offsets; }

   /// Get the number of elements in the local partial space.
   int GetNumLocalElements() const { return local2global.Size(); }

   /// Check if this partial space covers the entire mesh.
   bool IsFullSpace() const { return (global2local.Size() == 1); }

   /// Get the element transformation for a local entity index.
   virtual mfem::ElementTransformation *GetTransformation(int idx) override
   {
      int global_idx = LocalToGlobal(idx);
      return mesh.GetElementTransformation(global_idx);
   }

   /// Return the geometry type of the entity with local index idx.
   virtual mfem::Geometry::Type GetGeometry(int idx) const override
   {
      int global_idx = LocalToGlobal(idx);
      return mesh.GetElementGeometry(global_idx);
   }

   /// Get the permuted quadrature point index (trivial for element spaces).
   /** For element quadrature spaces, the permutation is trivial. */
   virtual int GetPermutedIndex(int idx, int iq) const override
   {
      // For element quadrature spaces, the permutation is trivial
      return iq;
   }

   /// Write the PartialQuadratureSpace to the stream.
   virtual void Save(std::ostream &out) const override;

   /// Returns the element index for the given ElementTransformation.
   virtual int GetEntityIndex(const mfem::ElementTransformation &T) const override
   {
      return T.ElementNo;
   }
};


// Inline methods

inline int PartialQuadratureSpace::LocalToGlobal(int local_idx) const
{
   if (local_idx >= 0 && local_idx < local2global.Size())
   {
      return local2global[local_idx];
   }
   return -1;
}

inline int PartialQuadratureSpace::GlobalToLocal(int global_idx) const
{
   if (global_idx >= 0 && global_idx < global2local.Size())
   {
      return global2local[global_idx];
   }
   else if (global_idx >= 0 && global2local.Size() == 1)
   {
      return global_idx;
   }
   return -1;
}

} // namespace mfem