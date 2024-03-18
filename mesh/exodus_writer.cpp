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

#include "mesh_headers.hpp"

#ifdef MFEM_USE_NETCDF
#include "netcdf.h"
#endif

namespace mfem
{
/// Function Prototypes.
static void HandleNetCDFStatus(int status);
static void GenerateExodusIIElementBlocksFromMesh(Mesh & mesh,
                                                  std::set<mfem::Element::Type> & element_types,
                                                  std::map<mfem::Element::Type, std::vector<int>> & element_ids_for_type);
static void GenerateExodusIISideSetsFromMesh(Mesh & mesh,
                                             std::set<int> & boundary_ids);

static void GenerateExodusIINodeIDsFromMesh(Mesh & mesh, int & num_nodes);

void Mesh::WriteExodusII(const std::string fpath)
{
   int status = NC_NOERR;
   int ncid  = 0;  // File descriptor.

   //
   // Open file
   //
   int flags = NC_CLOBBER; // Overwrite existing files.

   status = nc_create(fpath.c_str(), flags, &ncid);
   HandleNetCDFStatus(status);

   //
   // Generate Initialization Parameters
   //

   //
   // Add title.
   //
   const char *title = "MFEM mesh";
   status = nc_put_att_text(ncid, NC_GLOBAL, "title", strlen(title), title);
   HandleNetCDFStatus(status);

   //
   // Set dimension.
   //
   int num_dim_id;
   status = nc_def_dim(ncid, "num_dim", Dim, &num_dim_id);
   HandleNetCDFStatus(status);

   //
   // Set # nodes.
   //
   int num_nodes_id;
   int num_nodes;

   GenerateExodusIINodeIDsFromMesh(*this, num_nodes);

   status = nc_def_dim(ncid, "num_nodes", num_nodes, &num_nodes_id);
   HandleNetCDFStatus(status);

   //
   // Set # elements.
   //
   int num_elem_id;
   status = nc_def_dim(ncid, "num_elem", NumOfElements, &num_elem_id);
   HandleNetCDFStatus(status);

   //
   // Set # element blocks.
   //
   std::set<mfem::Element::Type> element_types;
   std::map<mfem::Element::Type, std::vector<int>> element_ids_for_type;
   GenerateExodusIIElementBlocksFromMesh(*this, element_types,
                                         element_ids_for_type);

   int num_elem_blk_id;
   status = nc_def_dim(ncid, "num_elem_blk", (int)element_types.size(),
                       &num_elem_blk_id);
   HandleNetCDFStatus(status);

   //
   // Set # side sets ("boundaries")
   //
   std::set<int> boundary_ids;
   GenerateExodusIISideSetsFromMesh(*this, boundary_ids);

   int num_side_sets_ids;
   status = nc_def_dim(ncid, "num_side_sets", (int)boundary_ids.size(),
                       &num_side_sets_ids);
   HandleNetCDFStatus(status);

   //
   // Set # node sets - TODO: add this (currently, set to 0).
   //
   int num_node_sets_ids;
   status = nc_def_dim(ncid, "num_node_sets", 0, &num_node_sets_ids);
   HandleNetCDFStatus(status);

   //
   // Close file
   //
   status = nc_close(ncid);
   HandleNetCDFStatus(status);

   mfem::out << "Mesh successfully written to Exodus II file" << std::endl;
}

/// @brief Aborts with description of error.
static void HandleNetCDFStatus(int status)
{
   if (status != NC_NOERR)
   {
      MFEM_ABORT("NetCDF error: " << nc_strerror(status));
   }
}

/// @brief Generates blocks based on the elements in the mesh. We've lost information
/// if we previously had an Exodus II mesh that we read-in. If all elements are the
/// same then we create a single block. Otherwise, we create a block for each element
/// type in the mesh.
static void GenerateExodusIIElementBlocksFromMesh(Mesh & mesh,
                                                  std::set<mfem::Element::Type> & element_types,
                                                  std::map<mfem::Element::Type, std::vector<int>> & element_ids_for_type)
{
   element_types.clear();
   element_ids_for_type.clear();

   // Iterate over the elements in the mesh.
   for (int ielement = 0; ielement < mesh.GetNE(); ielement++)
   {
      mfem::Element::Type element_type = mesh.GetElementType(ielement);

      if (element_types.count(element_type) == 0)  // Encountered a new element type!
      {
         element_types.insert(element_type);

         element_ids_for_type[element_type] = { ielement };
      }
      else
      {
         std::vector<int> & element_ids = element_ids_for_type.at(element_type);

         element_ids.push_back(ielement);
      }
   }
}

/// @brief Generates sidesets from the mesh. We iterate over the boundary elements and look at the
/// element attributes (each one matches a sideset ID). We can then build a set of unique sideset
/// IDs and a mapping from sideset ID to a vector of all element IDs.
static void GenerateExodusIISideSetsFromMesh(Mesh & mesh,
                                             std::set<int> & boundary_ids)
{
   boundary_ids.clear();

   for (int ielement = 0; ielement < mesh.GetNBE(); ielement++)
   {
      mfem::Element * boundary_element = mesh.GetBdrElement(ielement);

      int boundary_id = boundary_element->GetAttribute();

      boundary_ids.insert(boundary_id);
   }
}

/// @brief Iterates over the elements of the mesh to extract a unique set of node IDs (or vertex IDs if first-order).
static void GenerateExodusIINodeIDsFromMesh(Mesh & mesh, int & num_nodes)
{
   std::set<int> node_ids;

   const FiniteElementSpace * fespace = mesh.GetNodalFESpace();

   mfem::Array<int> dofs;

   for (int ielement = 0; ielement < mesh.GetNE(); ielement++)
   {
      if (fespace)   // Higher-order
      {
         fespace->GetElementDofs(ielement, dofs);

         for (int dof : dofs) { node_ids.insert(dof); }
      }
      else
      {
         mfem::Array<int> vertex_indices;
         mesh.GetElementVertices(ielement, vertex_indices);

         // TODO: - Hmmmm. These are not actually the dofs. Just the vertex offsets.
         for (int vertex_index : vertex_indices)
         {
            node_ids.insert(vertex_index);
         }
      }
   }

   num_nodes = (int)node_ids.size();
}
}
