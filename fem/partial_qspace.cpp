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

#include "partial_qspace.hpp"

namespace mfem
{

const mfem::Vector &
PartialQuadratureSpace::GetGeometricFactorWeights() const
{
   // We'll create a partial weight vector from the full mesh's geometric factors
   auto flags = mfem::GeometricFactors::DETERMINANTS;
   // TODO: assumes only one integration rule. This should be fixed once
   // Mesh::GetGeometricFactors accepts a QuadratureSpace instead of
   // IntegrationRule.
   const mfem::IntegrationRule &ir = GetIntRule(0);
   auto *geom = mesh.GetGeometricFactors(ir, flags);
   
   // We need to extract only the weights for our partial elements
   mfem::Vector &partial_weights = const_cast<mfem::Vector&>(weights);
   partial_weights.SetSize(size);
   partial_weights = 0.0;
   
   // Fill in the weights for our partial elements
   for (int i = 0; i < local2global.Size(); i++)
   {
      int global_idx = local2global[i];
      const int s_offset = offsets[i];
      const int e_offset = offsets[i + 1];
      const int num_qpoints = e_offset - s_offset;
      
      // Copy the weights for this element from the full mesh
      for (int j = 0; j < num_qpoints; j++)
      {
         // This is a simplified approach - a more accurate implementation would
         // need to map to the correct quadrature point indices in the full mesh
         partial_weights(s_offset + j) = geom->detJ(global_idx * num_qpoints + j);
      }
   }
   return weights;
}

void PartialQuadratureSpace::ConstructOffsets()
{
   // Set up offsets based on our partial element set
   const int num_partial_elem = local2global.Size();
   offsets.SetSize(num_partial_elem + 1);
   int offset = 0;
   for (int i = 0; i < num_partial_elem; i++)
   {
      offsets[i] = offset;
      // Get the global element index
      int global_elem_idx = local2global[i];
      // Get geometry for the element
      const size_t geom = static_cast<size_t>(mesh.GetElementBaseGeometry(global_elem_idx));
      MFEM_ASSERT(int_rule[geom] != NULL, "Missing integration rule.");
      offset += int_rule[geom]->GetNPoints();
   }
   offsets[num_partial_elem] = size = offset;
}

void PartialQuadratureSpace::ConstructGlobalOffsets()
{
   // Set up offsets based on our partial element set
   const int num_elems = global2local.Size();
   if (num_elems != 1)
   {
      global_offsets.SetSize(num_elems + 1);
      int offset = 0;
      for (int i = 0; i < num_elems; i++)
      {
         global_offsets[i] = offset;
         // Get geometry for the element
         const size_t geom = static_cast<size_t>(mesh.GetElementBaseGeometry(i));
         MFEM_ASSERT(int_rule[geom] != NULL, "Missing integration rule.");
         offset += int_rule[geom]->GetNPoints();
      }
      global_offsets[num_elems] = offset;
   }
   else
   {
      global_offsets.SetSize(1);
      global_offsets[0] = 0;
   }
}

void PartialQuadratureSpace::Construct()
{
   ConstructIntRules(mesh.Dimension());
   ConstructOffsets();
   ConstructGlobalOffsets();
}

void PartialQuadratureSpace::ConstructMappings(mfem::Mesh *mesh_,
                                                mfem::Array<bool> &partial_index)
{
   // First, construct the mapping arrays
   int num_elements = mesh_->GetNE();

   int partial_count = 0;
   if (partial_index.Size() == 0)
   {
      partial_count = num_elements;
   }
   else
   {
      // Count how many elements are in our partial set
      for (int i = 0; i < num_elements; i++)
      {
         if (partial_index[i])
         {
            partial_count++;
         }
      }
   }
   
   // Initialize local2global array
   local2global.SetSize(partial_count);
   
   // Set up global2local mapping with -1 as default (not in partial set)
   // If partial_count == num_elements then this is a quadspace equiv
   if (partial_count != num_elements)
   {
      global2local.SetSize(num_elements);
      for (int i = 0; i < num_elements; i++)
      {
         global2local[i] = -1;
      }
   
      // Fill the mapping arrays
      int local_idx = 0;
      for (int i = 0; i < num_elements; i++)
      {
         if (partial_index[i])
         {
            local2global[local_idx] = i;
            global2local[i] = local_idx;
            local_idx++;
         }
      }
   }
   else
   {
      for (int i = 0; i < num_elements; i++)
      {
         local2global[i] = i;
      }
      global2local.SetSize(1);
      global2local[0] = 0;
   }
}

PartialQuadratureSpace::PartialQuadratureSpace(mfem::Mesh *mesh_, int order_,
                                                mfem::Array<bool> &partial_index)
   : QuadratureSpaceBase(*mesh_, order_)
{
   ConstructMappings(mesh_, partial_index);
   Construct();
}

PartialQuadratureSpace::PartialQuadratureSpace(mfem::Mesh *mesh_,
                                                const mfem::IntegrationRule &ir, 
                                                mfem::Array<bool> &partial_index)
   : QuadratureSpaceBase(*mesh_, mesh_->GetTypicalElementGeometry(), ir)
{
   MFEM_VERIFY(mesh.GetNumGeometries(mesh.Dimension()) <= 1,
               "Constructor not valid for mixed meshes");
   ConstructMappings(mesh_, partial_index);
   ConstructOffsets();
}

PartialQuadratureSpace::PartialQuadratureSpace(mfem::Mesh *mesh_,
                                                std::istream &in)
   : QuadratureSpaceBase(*mesh_)
{
   const char *msg = "invalid input stream";
   std::string ident;

   // Read header information
   in >> ident; MFEM_VERIFY(ident == "PartialQuadratureSpace", msg);
   in >> ident; MFEM_VERIFY(ident == "Type:", msg);
   in >> ident;
   if (ident == "default_quadrature")
   {
      in >> ident; MFEM_VERIFY(ident == "Order:", msg);
      in >> order;
   }
   else
   {
      MFEM_ABORT("unknown PartialQuadratureSpace type: " << ident);
      return;
   }

   // Read partial space mapping information
   in >> ident; MFEM_VERIFY(ident == "PartialIndices:", msg);
   int size;
   in >> size;
   local2global.SetSize(size);
   
   // Read local2global array
   for (int i = 0; i < size; i++)
   {
      in >> local2global[i];
   }
   
   // Set up global2local mapping
   int num_elements = mesh.GetNE();
   if (size != num_elements)
   {
      global2local.SetSize(num_elements);
      for (int i = 0; i < num_elements; i++)
      {
         global2local[i] = -1;
      }
      
      // Build the inverse mapping
      for (int i = 0; i < size; i++)
      {
         int global_idx = local2global[i];
         global2local[global_idx] = i;
      }
   }
   else
   {
      global2local.SetSize(1);
      global2local[0] = 0;
   }
   
   // Now construct the quadrature space internals
   Construct();
}

void PartialQuadratureSpace::Save(std::ostream &os) const
{
   os << "PartialQuadratureSpace\n"
       << "Type: default_quadrature\n"
       << "Order: " << order << '\n';
   
   // Save the partial space mapping information
   os << "PartialIndices: " << local2global.Size() << '\n';
   for (int i = 0; i < local2global.Size(); i++)
   {
      os << local2global[i] << " ";
   }
   os << "\n";
}

} // namespace mfem