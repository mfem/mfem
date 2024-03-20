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
                                                  std::vector<int> & unique_block_ids,
                                                  std::map<int, std::vector<int>>  & element_ids_for_block_id,
                                                  std::map<int, Element::Type> & element_type_for_block_id);

static void GenerateExodusIISideSetsFromMesh(Mesh & mesh,
                                             std::vector<int> & boundary_ids,
                                             std::map<int, int> & num_elements_for_boundary_id,
                                             std::map<int, std::vector<int>> & element_ids_for_boundary_id,
                                             std::map<int, std::vector<int>> & side_ids_for_boundary_id);

static void GenerateExodusIINodeIDsFromMesh(Mesh & mesh, int & num_nodes);

static void ExtractVertexCoordinatesFromMesh(int ncid, Mesh & mesh,
                                             std::vector<double> & coordx, std::vector<double> & coordy,
                                             std::vector<double> & coordz);

static void WriteNodalCoordinatesFromMesh(int ncid, Mesh & mesh);

static void WriteNodeConnectivityForBlock(int ncid, Mesh & mesh,
                                          const int * num_dim_id_ptr,
                                          const int block_id,
                                          const std::map<int, std::vector<int>> & element_ids_for_block_id);

static void WriteSideSetInformationForMesh(int ncid, Mesh & mesh,
                                           const int * num_dim_id_ptr,
                                           const std::vector<int> & boundary_ids,
                                           const std::map<int, int> & num_elements_for_boundary_id,
                                           const std::map<int, std::vector<int>> & element_ids_for_boundary_id,
                                           const std::map<int, std::vector<int>> & side_ids_for_boundary_id);

static void WriteBlockIDs(int ncid, const int * num_dim_id_ptr,
                          const std::vector<int> & unique_block_ids);


void Mesh::WriteExodusII(const std::string fpath)
{
   int status = NC_NOERR;
   int ncid  = 0;  // File descriptor.

   //
   // Open file
   //

   int flags = NC_CLOBBER |
               NC_NETCDF4; // Overwrite existing files and use NetCDF4 mode (avoid classic).

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
   // Set # nodes. NB: - Assume 1st order currently so NumOfVertices == # nodes
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
   std::vector<int> unique_block_ids;
   std::map<int, Element::Type> element_type_for_block_id;
   std::map<int, std::vector<int>> element_ids_for_block_id;
   GenerateExodusIIElementBlocksFromMesh(*this, unique_block_ids,
                                         element_ids_for_block_id, element_type_for_block_id);

   int num_elem_blk_id;
   status = nc_def_dim(ncid, "num_el_blk", (int)unique_block_ids.size(),
                       &num_elem_blk_id);
   HandleNetCDFStatus(status);

   //
   // Set # side sets ("boundaries")
   //
   std::vector<int> boundary_ids;
   std::map<int, int> num_elements_for_boundary_id;
   std::map<int, std::vector<int>> element_ids_for_boundary_id;
   std::map<int, std::vector<int>> side_ids_for_boundary_id;

   GenerateExodusIISideSetsFromMesh(*this, boundary_ids,
                                    num_elements_for_boundary_id,
                                    element_ids_for_boundary_id,
                                    side_ids_for_boundary_id);

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
   // Define nodal coordinates.
   // https://docs.unidata.ucar.edu/netcdf-c/current/group__variables.html#gac7e8662c51f3bb07d1fc6d6c6d9052c8
   // NB: assume we have doubles (could be floats!)
   // ndims = 1 (vectors).
   int coordx_id, coordy_id, coordz_id;

   status = nc_def_var(ncid, "coordx", NC_DOUBLE, 1, &num_dim_id, &coordx_id);
   HandleNetCDFStatus(status);

   status = nc_def_var(ncid, "coordy", NC_DOUBLE, 1, &num_dim_id, &coordy_id);
   HandleNetCDFStatus(status);

   if (Dim == 3)
   {
      status = nc_def_var(ncid, "coordz", NC_DOUBLE, 1, &num_dim_id, &coordz_id);
      HandleNetCDFStatus(status);
   }

   //
   // Write nodal coordinates.
   //
   WriteNodalCoordinatesFromMesh(ncid, *this);

   //
   // Write element block parameters.
   //
   char name_buffer[100];

   for (int block_id : unique_block_ids)
   {
      const auto & block_element_ids = element_ids_for_block_id.at(block_id);
      //
      // TODO: - element block IDs 0-indexed MFEM --> 1-indexed Exodus II
      //

      Element * front_element = GetElement(block_element_ids.front());

      //
      // Define # elements in the block.
      //
      sprintf(name_buffer, "num_el_in_blk%d", block_id);

      int num_el_in_blk_id;
      status = nc_def_dim(ncid, name_buffer, block_element_ids.size(),
                          &num_el_in_blk_id);
      HandleNetCDFStatus(status);


      //
      // Define # nodes per element. NB: - assume first-order elements currently!!
      //
      sprintf(name_buffer, "num_nod_per_el%d", block_id);

      int num_node_per_el_id;
      status = nc_def_dim(ncid, name_buffer, front_element->GetNVertices(),
                          &num_node_per_el_id);
      HandleNetCDFStatus(status);

      //
      // Define # edges per element:
      //
      sprintf(name_buffer, "num_edg_per_el%d", block_id);

      int num_edg_per_el_id;
      status = nc_def_dim(ncid, name_buffer, front_element->GetNEdges(),
                          &num_edg_per_el_id);
      HandleNetCDFStatus(status);

      //
      // Define # faces per element.
      //
      sprintf(name_buffer, "num_fac_per_el%d", block_id);

      int num_fac_per_el_id;
      status = nc_def_dim(ncid, name_buffer, front_element->GetNFaces(),
                          &num_fac_per_el_id);
      HandleNetCDFStatus(status);

      //
      // Define element node connectivity for block.
      //
      WriteNodeConnectivityForBlock(ncid, *this, &num_dim_id, block_id,
                                    element_ids_for_block_id);
   }

   //
   // Write block IDs.
   //
   WriteBlockIDs(ncid, &num_dim_id, unique_block_ids);

   //
   // Write sideset information.
   //
   WriteSideSetInformationForMesh(ncid, *this, &num_dim_id, boundary_ids,
                                  num_elements_for_boundary_id,
                                  element_ids_for_boundary_id,
                                  side_ids_for_boundary_id);

   //
   // Close file
   //
   status = nc_close(ncid);
   HandleNetCDFStatus(status);

   mfem::out << "Mesh successfully written to Exodus II file" << std::endl;
}

static void WriteBlockIDs(int ncid, const int * num_dim_id_ptr,
                          const std::vector<int> & unique_block_ids)
{
   int status, unique_block_ids_ptr;

   status = nc_def_var(ncid, "eb_prop1", NC_INT, 1, num_dim_id_ptr,
                       &unique_block_ids_ptr);
   HandleNetCDFStatus(status);

   status = nc_put_var_int(ncid, unique_block_ids_ptr, unique_block_ids.data());
   HandleNetCDFStatus(status);
}

static void WriteSideSetInformationForMesh(int ncid, Mesh & mesh,
                                           const int * num_dim_id_ptr,
                                           const std::vector<int> & boundary_ids,
                                           const std::map<int, int> & num_elements_for_boundary_id,
                                           const std::map<int, std::vector<int>> & element_ids_for_boundary_id,
                                           const std::map<int, std::vector<int>> & side_ids_for_boundary_id)
{
   //
   // Add the boundary IDs
   //
   int status, boundary_ids_ptr;
   status = nc_def_var(ncid, "ss_prop1", NC_INT, 1, num_dim_id_ptr,
                       &boundary_ids_ptr);
   HandleNetCDFStatus(status);

   status = nc_put_var_int(ncid, boundary_ids_ptr, boundary_ids.data());
   HandleNetCDFStatus(status);

   //
   // Add the number of elements for each boundary_id.
   //
   char name_buffer[100];

   for (int boundary_id : boundary_ids)
   {
      sprintf(name_buffer, "num_side_ss%d", boundary_id);

      int num_side_ss_id;
      status = nc_def_dim(ncid, name_buffer,
                          (size_t)num_elements_for_boundary_id.at(boundary_id), &num_side_ss_id);
      HandleNetCDFStatus(status);
   }

   //
   // Add the faces here.
   //
   for (int boundary_id : boundary_ids)
   {
      const std::vector<int> & side_ids = side_ids_for_boundary_id.at(boundary_id);

      sprintf(name_buffer, "side_ss%d", boundary_id);

      int side_id_ptr;
      status = nc_def_var(ncid, name_buffer, NC_INT, 1, num_dim_id_ptr, &side_id_ptr);
      HandleNetCDFStatus(status);

      status = nc_put_var_int(ncid, side_id_ptr, side_ids.data());
      HandleNetCDFStatus(status);
   }

   //
   // Add the boundary elements here.
   //
   for (int boundary_id : boundary_ids)
   {
      // TODO: - need to figure-out correct element ids (we only have local element indexes into the boundaries array!)
      const std::vector<int> & element_ids = element_ids_for_boundary_id.at(
                                                boundary_id);

      sprintf(name_buffer, "elem_ss%d", boundary_id);

      int elem_id_ptr;
      status = nc_def_var(ncid, name_buffer, NC_INT, 1, num_dim_id_ptr, &elem_id_ptr);
      HandleNetCDFStatus(status);

      status = nc_put_var_int(ncid, elem_id_ptr, element_ids.data());
      HandleNetCDFStatus(status);
   }
}

static void WriteNodeConnectivityForBlock(int ncid, Mesh & mesh,
                                          const int * num_dim_id_ptr,
                                          const int block_id,
                                          const std::map<int, std::vector<int>> & element_ids_for_block_id)
{
   int status{NC_NOERR}, connect_id;

   // Generate arbitrary name:
   char name_buffer[100];
   sprintf(name_buffer, "connect%d", block_id);

   // NB: 1 == vector!; name is arbitrary; NC_INT or NCINT64??
   status = nc_def_var(ncid, name_buffer, NC_INT, 1, num_dim_id_ptr, &connect_id);
   HandleNetCDFStatus(status);

   std::vector<int> block_node_connectivity;

   for (int element_id : element_ids_for_block_id.at(block_id))
   {
      // NB: assume first-order elements only for now.
      // NB: - need to convert from 0-based indexing --> 1-based indexing.
      mfem::Array<int> element_vertices;
      mesh.GetElementVertices(element_id, element_vertices);

      for (int vertex_id : element_vertices)
      {
         block_node_connectivity.push_back(vertex_id);
      }
   }

   status = nc_put_var_int(ncid, connect_id, block_node_connectivity.data());
   HandleNetCDFStatus(status);
}


static void ExtractVertexCoordinatesFromMesh(int ncid, Mesh & mesh,
                                             std::vector<double> & coordx, std::vector<double> & coordy,
                                             std::vector<double> & coordz)
{
   coordx.resize(mesh.GetNV());
   coordy.resize(mesh.GetNV());
   coordz.resize(mesh.Dimension() == 3 ? mesh.GetNV() : 0);

   for (int ivertex = 0; ivertex < mesh.GetNV(); ivertex++)
   {
      double * coordinates = mesh.GetVertex(ivertex);

      coordx[ivertex] = coordinates[0];
      coordy[ivertex] = coordinates[1];

      if (mesh.Dimension() == 3)
      {
         coordz[ivertex] = coordinates[2];
      }
   }
}

static void WriteNodalCoordinatesFromMesh(int ncid, Mesh & mesh)
{
   std::vector<double> coordx, coordy, coordz;
   ExtractVertexCoordinatesFromMesh(ncid, mesh, coordx, coordy, coordz);

   int coordx_id, coordy_id, coordz_id;
   int status = NC_NOERR;

   status = nc_inq_varid(ncid, "coordx", &coordx_id);
   HandleNetCDFStatus(status);

   status = nc_put_var_double(ncid, coordx_id, coordx.data());
   HandleNetCDFStatus(status);

   status = nc_inq_varid(ncid, "coordy", &coordy_id);
   HandleNetCDFStatus(status);

   status = nc_put_var_double(ncid, coordy_id, coordy.data());
   HandleNetCDFStatus(status);

   if (mesh.Dimension() == 3)
   {
      status = nc_inq_varid(ncid, "coordz", &coordz_id);
      HandleNetCDFStatus(status);

      status = nc_put_var_double(ncid, coordz_id, coordz.data());
      HandleNetCDFStatus(status);
   }
}

/// @brief Aborts with description of error.
static void HandleNetCDFStatus(int status)
{
   if (status != NC_NOERR)
   {
      MFEM_ABORT("NetCDF error: " << nc_strerror(status));
   }
}

/// @brief Generates blocks based on the elements in the mesh. We assume that this was originally an Exodus II
/// mesh. Therefore, we iterate over the elements and use the attributes as the element blocks. We assume that
/// all elements belonging to the same block will have the same attribute. We can perform a safety check as well
/// by ensuring that all elements in the block have the same element type. If this is not the case then something
/// has gone horribly wrong!
static void GenerateExodusIIElementBlocksFromMesh(Mesh & mesh,
                                                  std::vector<int> & unique_block_ids,
                                                  std::map<int, std::vector<int>>  & element_ids_for_block_id,
                                                  std::map<int, Element::Type> & element_type_for_block_id)
{
   unique_block_ids.clear();
   element_ids_for_block_id.clear();
   element_type_for_block_id.clear();

   std::set<int> observed_block_ids;

   // Iterate over the elements in the mesh.
   for (int ielement = 0; ielement < mesh.GetNE(); ielement++)
   {
      Element::Type element_type = mesh.GetElementType(ielement);

      int block_id = mesh.GetAttribute(ielement);
      printf("block_id = %d\n", block_id);

      if (observed_block_ids.count(block_id) == 0)
      {
         unique_block_ids.push_back(block_id);

         element_type_for_block_id[block_id] = element_type;
         element_ids_for_block_id[block_id] = { ielement };

         observed_block_ids.insert(block_id);
      }
      else
      {
         auto & block_element_ids = element_ids_for_block_id.at(block_id);
         block_element_ids.push_back(ielement);

         // Safety check: ensure that the element type matches what we have on record for the block.
         if (element_type != element_type_for_block_id.at(block_id))
         {
            MFEM_ABORT("Multiple element types are defined for block: " << block_id);
         }
      }
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


static void GenerateExodusIISideSetsFromMesh(Mesh & mesh,
                                             std::vector<int> & boundary_ids,
                                             std::map<int, int> & num_elements_for_boundary_id,
                                             std::map<int, std::vector<int>> & element_ids_for_boundary_id,
                                             std::map<int, std::vector<int>> & side_ids_for_boundary_id)
{
   boundary_ids.clear();
   num_elements_for_boundary_id.clear();
   element_ids_for_boundary_id.clear();
   side_ids_for_boundary_id.clear();

   // Iterate over boundary elements.
   std::set<int> boundaries_seen;

   for (int ibdr_element = 0; ibdr_element < mesh.GetNBE(); ibdr_element++)
   {
      int boundary_id = mesh.GetBdrAttribute(ibdr_element);

      if (boundaries_seen.count(boundary_id) == 0)
      {
         boundaries_seen.insert(boundary_id);
         boundary_ids.push_back(boundary_id);
         num_elements_for_boundary_id[boundary_id] = 1;
      }
      else
      {
         num_elements_for_boundary_id[boundary_id]++;
      }

      // Returns the boundary face index of the boundary element. We need to convert these to Exodus II face indices. For now, we assume that MFEM and
      // Exodus II use the same mappings (this is almost certainly not the case). We also need to add 1 since MFEM uses 0-based indexing and Exodus II
      // uses 1-based indexing.
      int face_index, face_orientation;
      mesh.GetBdrElementFace(ibdr_element, &face_index, &face_orientation);

      side_ids_for_boundary_id[boundary_id].push_back(face_orientation + 1);

      // TODO: - Need to create a vector containing all elements for that boundary ID. Since these are the boundary elements, we need to
      // figure-out what MFEM element (stored in elements) this is!
      element_ids_for_boundary_id[boundary_id].push_back(
         ibdr_element); // NOT CORRECT!!!
   }
}

}
