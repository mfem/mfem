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
#include <unordered_set>

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

static void GenerateExodusIIBoundaryInfo(Mesh & mesh,
                                         std::vector<int> & unique_boundary_ids,
                                         std::map<int, std::vector<int>> & exodusII_element_ids_for_boundary_id,
                                         std::map<int, std::vector<int>> & exodusII_side_ids_for_boundary_id);

static void GenerateExodusIINodeIDsFromMesh(Mesh & mesh, int & num_nodes);

static void ExtractVertexCoordinatesFromMesh(int ncid, Mesh & mesh,
                                             std::vector<double> & coordx, std::vector<double> & coordy,
                                             std::vector<double> & coordz);

static void WriteNodalCoordinatesFromMesh(int ncid,
                                          std::vector<double> & coordx, std::vector<double> & coordy,
                                          std::vector<double> & coordz);

static void WriteNodeConnectivityForBlock(int ncid, Mesh & mesh,
                                          const int block_id,
                                          const std::map<int, std::vector<int>> & element_ids_for_block_id);

static void WriteSideSetInformationForMesh(int ncid, Mesh & mesh,
                                           const std::vector<int> & boundary_ids,
                                           const std::map<int, std::vector<int>> & element_ids_for_boundary_id,
                                           const std::map<int, std::vector<int>> & side_ids_for_boundary_id);

static void WriteBlockIDs(int ncid, const std::vector<int> & unique_block_ids);


// Returns the Exodus II face ID for the MFEM face index.
const int mfem_to_exodusII_side_map_tet4[] =
{
   2, 3, 1, 4
};


void Mesh::WriteExodusII(const std::string fpath)
{
   int status = NC_NOERR;
   int ncid  = 0;  // File descriptor.

   //
   // Open file
   //

   int flags =
      NC_CLOBBER; // Overwrite existing files and use NetCDF4 mode (avoid classic).

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
   // Set # nodes. NB: - Assume 1st order currently so NumOfVertices == # nodes
   //
   int num_nodes_id;
   int num_nodes;

   GenerateExodusIINodeIDsFromMesh(*this, num_nodes);

   status = nc_def_dim(ncid, "num_nodes", num_nodes, &num_nodes_id);
   HandleNetCDFStatus(status);

   //
   // Set dimension.
   //
   int num_dim_id;
   status = nc_def_dim(ncid, "num_dim", Dim, &num_dim_id);
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
   // Set # node sets - TODO: add this (currently, set to 0).
   //
   int num_node_sets_ids;
   status = nc_def_dim(ncid, "num_node_sets", 0, &num_node_sets_ids);
   HandleNetCDFStatus(status);

   //
   // Set # side sets ("boundaries")
   //
   std::vector<int> boundary_ids;
   std::map<int, std::vector<int>> exodusII_element_ids_for_boundary_id;
   std::map<int, std::vector<int>> exodusII_side_ids_for_boundary_id;
   GenerateExodusIIBoundaryInfo(*this, boundary_ids,
                                exodusII_element_ids_for_boundary_id,
                                exodusII_side_ids_for_boundary_id);

   int num_side_sets_ids;
   status = nc_def_dim(ncid, "num_side_sets", (int)boundary_ids.size(),
                       &num_side_sets_ids);
   HandleNetCDFStatus(status);

   //
   // Set database version #
   //
   float database_version = 4.72;
   status = nc_put_att_float(ncid, NC_GLOBAL, "version", NC_FLOAT, 1,
                             &database_version);
   HandleNetCDFStatus(status);

   //
   // Set API version #
   //
   float version = 4.72;   // Current version as of 2024-03-21.
   status = nc_put_att_float(ncid, NC_GLOBAL, "api_version", NC_FLOAT, 1,
                             &version);
   HandleNetCDFStatus(status);

   //
   // Set I/O word size as an attribute (float == 4; double == 8)
   //
   const int word_size = 8;
   status = nc_put_att_int(ncid, NC_GLOBAL, "floating_point_word_size", NC_INT, 1,
                           &word_size);
   HandleNetCDFStatus(status);

   //
   // Store Exodus file size (normal==0; large==1). NB: coordinates specifed separately as components
   // for large file.
   //
   const int file_size = 1;
   status = nc_put_att_int(ncid, NC_GLOBAL, "file_size", NC_INT, 1, &file_size);
   HandleNetCDFStatus(status);

   //
   // Set length of character strings.
   //
   const int max_name_length = 80;
   status = nc_put_att_int(ncid, NC_GLOBAL, "maxiumum_name_length", NC_INT, 1,
                           &max_name_length);
   HandleNetCDFStatus(status);

   //
   // Set length of character lines.
   //
   const int max_line_length = 80;
   status = nc_put_att_int(ncid, NC_GLOBAL, "maximum_line_length", NC_INT, 1,
                           &max_name_length);
   HandleNetCDFStatus(status);

   //
   // Set # timesteps (ASSUME no timesteps for initial verision)
   //
   int timesteps_dim;
   status = nc_def_dim(ncid, "time_step", 1, &timesteps_dim);
   HandleNetCDFStatus(status);

   //
   // ---------------------------------------------------------------------------------
   //

   //
   // Quality Assurance Data
   //

   //
   // Information Data
   //


   //
   // NB: LibMesh has a dodgy bug where it will skip the x-coordinate if coordx_id == 0.
   // To prevent this, the first variable to be defined will be a dummy variable which will
   // have a variable id of 0.
   {
      int dummy_var_id, dummy_var_dim_id, dummy_value = 1;

      status = nc_def_dim(ncid, "dummy_var_dim", 1, &dummy_var_dim_id);
      HandleNetCDFStatus(status);

      status = nc_def_var(ncid, "dummy_var", NC_INT, 1, &dummy_var_dim_id,
                          &dummy_var_id);
      HandleNetCDFStatus(status);

      nc_enddef(ncid);
      status = nc_put_var_int(ncid, dummy_var_id, &dummy_value);
      HandleNetCDFStatus(status);
      nc_redef(ncid);
   }

   //
   // Define nodal coordinates.
   // https://docs.unidata.ucar.edu/netcdf-c/current/group__variables.html#gac7e8662c51f3bb07d1fc6d6c6d9052c8
   // NB: assume we have doubles (could be floats!)
   // ndims = 1 (vectors).
   int coordx_id, coordy_id, coordz_id;

   std::vector<double> coordx(num_nodes), coordy(num_nodes),
       coordz(Dim == 3 ? num_nodes : 0);
   ExtractVertexCoordinatesFromMesh(ncid, *this, coordx, coordy, coordz);

   status = nc_def_var(ncid, "coordx", NC_DOUBLE, 1, &num_nodes_id, &coordx_id);
   HandleNetCDFStatus(status);

   status = nc_def_var(ncid, "coordy", NC_DOUBLE, 1, &num_nodes_id, &coordy_id);
   HandleNetCDFStatus(status);

   if (Dim == 3)
   {
      status = nc_def_var(ncid, "coordz", NC_DOUBLE, 1, &num_nodes_id, &coordz_id);
      HandleNetCDFStatus(status);
   }

   //
   // Write nodal coordinates.
   //
   WriteNodalCoordinatesFromMesh(ncid, coordx, coordy, coordz);

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
      WriteNodeConnectivityForBlock(ncid, *this, block_id,
                                    element_ids_for_block_id);

      //
      // Define the element type.
      //
      std::string element_type;

      switch (front_element->GetType())
      {
         case Geometry::Type::CUBE:
            element_type = "hex";
            break;
         case Geometry::Type::TETRAHEDRON:
            element_type = "tet";
            break;
         case Geometry::Type::PRISM:
            element_type = "wedge";
            break;
         case Geometry::Type::TRIANGLE:
            element_type = "tri";
            break;
         default:
            MFEM_ABORT("Unsupported MFEM element type: " << front_element->GetType());
      }

      int elem_type_dim;
      status = nc_def_dim(ncid, "elem_type", element_type.length(), &elem_type_dim);
      HandleNetCDFStatus(status);

      char name_buffer[100];
      sprintf(name_buffer, "connect%d", block_id);

      int connect_id;
      status = nc_inq_varid(ncid, name_buffer, &connect_id);
      HandleNetCDFStatus(status);

      status = nc_put_att_text(ncid, connect_id, "elem_type", element_type.length(),
                               element_type.c_str());
      HandleNetCDFStatus(status);
   }

   //
   // Write block IDs.
   //
   WriteBlockIDs(ncid, unique_block_ids);

   //
   // Write sideset information.
   //
   WriteSideSetInformationForMesh(ncid, *this, boundary_ids,
                                  exodusII_element_ids_for_boundary_id,
                                  exodusII_side_ids_for_boundary_id);

   //
   // Close file
   //
   status = nc_close(ncid);
   HandleNetCDFStatus(status);

   mfem::out << "Mesh successfully written to Exodus II file" << std::endl;
}

static void WriteBlockIDs(int ncid, const std::vector<int> & unique_block_ids)
{
   int status, unique_block_ids_ptr;

   int block_dim;
   status = nc_def_dim(ncid, "block_dim", unique_block_ids.size(), &block_dim);
   HandleNetCDFStatus(status);

   status = nc_def_var(ncid, "eb_prop1", NC_INT, 1, &block_dim,
                       &unique_block_ids_ptr);
   HandleNetCDFStatus(status);

   nc_enddef(ncid);
   status = nc_put_var_int(ncid, unique_block_ids_ptr, unique_block_ids.data());
   HandleNetCDFStatus(status);
   nc_redef(ncid);
}

static void WriteSideSetInformationForMesh(int ncid, Mesh & mesh,
                                           const std::vector<int> & boundary_ids,
                                           const std::map<int, std::vector<int>> & element_ids_for_boundary_id,
                                           const std::map<int, std::vector<int>> & side_ids_for_boundary_id)
{
   //
   // Add the boundary IDs
   //
   int status, boundary_ids_ptr;

   int boundary_ids_dim;
   status = nc_def_dim(ncid, "boundary_ids_dim", boundary_ids.size(),
                       &boundary_ids_dim);
   HandleNetCDFStatus(status);

   status = nc_def_var(ncid, "ss_prop1", NC_INT, 1, &boundary_ids_dim,
                       &boundary_ids_ptr);
   HandleNetCDFStatus(status);

   nc_enddef(ncid);
   status = nc_put_var_int(ncid, boundary_ids_ptr, boundary_ids.data());
   HandleNetCDFStatus(status);
   nc_redef(ncid);

   //
   // Add the number of elements for each boundary_id.
   //
   char name_buffer[100];

   for (int boundary_id : boundary_ids)
   {
      size_t num_elements_for_boundary = element_ids_for_boundary_id.at(
                                            boundary_id).size();

      sprintf(name_buffer, "num_side_ss%d", boundary_id);

      int num_side_ss_id;
      status = nc_def_dim(ncid, name_buffer, num_elements_for_boundary,
                          &num_side_ss_id);
      HandleNetCDFStatus(status);
   }

   //
   // Add the faces here.
   //
   for (int boundary_id : boundary_ids)
   {
      const std::vector<int> & side_ids = side_ids_for_boundary_id.at(boundary_id);

      sprintf(name_buffer, "side_ss%d_dim", boundary_id);

      int side_id_dim;
      status = nc_def_dim(ncid, name_buffer, side_ids.size(), &side_id_dim);
      HandleNetCDFStatus(status);

      sprintf(name_buffer, "side_ss%d", boundary_id);

      int side_id_ptr;
      status = nc_def_var(ncid, name_buffer, NC_INT, 1,  &side_id_dim, &side_id_ptr);
      HandleNetCDFStatus(status);

      nc_enddef(ncid);
      status = nc_put_var_int(ncid, side_id_ptr, side_ids.data());
      HandleNetCDFStatus(status);
      nc_redef(ncid);
   }

   //
   // Add the boundary elements here.
   //
   for (int boundary_id : boundary_ids)
   {
      // TODO: - need to figure-out correct element ids (we only have local element indexes into the boundaries array!)
      const std::vector<int> & element_ids = element_ids_for_boundary_id.at(
                                                boundary_id);

      sprintf(name_buffer, "elem_ss%d_dim", boundary_id);

      int elem_ids_dim;
      status = nc_def_dim(ncid, name_buffer, element_ids.size(), &elem_ids_dim);
      HandleNetCDFStatus(status);

      sprintf(name_buffer, "elem_ss%d", boundary_id);

      int elem_ids_ptr;
      status = nc_def_var(ncid, name_buffer, NC_INT, 1, &elem_ids_dim, &elem_ids_ptr);
      HandleNetCDFStatus(status);

      nc_enddef(ncid);
      status = nc_put_var_int(ncid, elem_ids_ptr, element_ids.data());
      HandleNetCDFStatus(status);
      nc_redef(ncid);
   }
}

static void WriteNodeConnectivityForBlock(int ncid, Mesh & mesh,
                                          const int block_id,
                                          const std::map<int, std::vector<int>> & element_ids_for_block_id)
{
   int status{NC_NOERR}, connect_id;

   // Generate arbitrary name:
   char name_buffer[100];

   std::vector<int> block_node_connectivity;

   for (int element_id : element_ids_for_block_id.at(block_id))
   {
      // NB: assume first-order elements only for now.
      // NB: - need to convert from 0-based indexing --> 1-based indexing.
      mfem::Array<int> element_vertices;
      mesh.GetElementVertices(element_id, element_vertices);

      for (int vertex_id : element_vertices)
      {
         block_node_connectivity.push_back(vertex_id + 1);  // 1-based indexing.
      }
   }

   sprintf(name_buffer, "connect%d_dim", block_id);

   int node_connectivity_dim;
   status = nc_def_dim(ncid, name_buffer, block_node_connectivity.size(),
                       &node_connectivity_dim);
   HandleNetCDFStatus(status);

   // NB: 1 == vector!; name is arbitrary; NC_INT or NCINT64??
   sprintf(name_buffer, "connect%d", block_id);

   status = nc_def_var(ncid, name_buffer, NC_INT, 1, &node_connectivity_dim,
                       &connect_id);
   HandleNetCDFStatus(status);

   nc_enddef(ncid);
   status = nc_put_var_int(ncid, connect_id, block_node_connectivity.data());
   HandleNetCDFStatus(status);
   nc_redef(ncid);
}


static void ExtractVertexCoordinatesFromMesh(int ncid, Mesh & mesh,
                                             std::vector<double> & coordx, std::vector<double> & coordy,
                                             std::vector<double> & coordz)
{
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

static void WriteNodalCoordinatesFromMesh(int ncid,
                                          std::vector<double> & coordx, std::vector<double> & coordy,
                                          std::vector<double> & coordz)
{
   int coordx_id, coordy_id, coordz_id;
   int status = NC_NOERR;

   nc_enddef(ncid);

   status = nc_inq_varid(ncid, "coordx", &coordx_id);
   HandleNetCDFStatus(status);

   status = nc_put_var_double(ncid, coordx_id, coordx.data());
   HandleNetCDFStatus(status);

   status = nc_inq_varid(ncid, "coordy", &coordy_id);
   HandleNetCDFStatus(status);

   status = nc_put_var_double(ncid, coordy_id, coordy.data());
   HandleNetCDFStatus(status);

   if (coordz.size() != 0)
   {
      status = nc_inq_varid(ncid, "coordz", &coordz_id);
      HandleNetCDFStatus(status);

      status = nc_put_var_double(ncid, coordz_id, coordz.data());
      HandleNetCDFStatus(status);
   }

   nc_redef(ncid);
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

static void GenerateExodusIIBoundaryInfo(Mesh & mesh,
                                         std::vector<int> & unique_boundary_ids,
                                         std::map<int, std::vector<int>> & exodusII_element_ids_for_boundary_id,
                                         std::map<int, std::vector<int>> & exodusII_side_ids_for_boundary_id)
{
   // Store the unique boundary IDs.
   unique_boundary_ids.clear();
   exodusII_element_ids_for_boundary_id.clear();
   exodusII_side_ids_for_boundary_id.clear();

   for (int bdr_attribute : mesh.bdr_attributes)
   {
      unique_boundary_ids.push_back(bdr_attribute);
   }

   // Generate a mapping from the MFEM face index to the MFEM element ID.
   // Note that if we have multiple element IDs for a face index then the
   // face is shared between them and it cannot possibly be a boundary face
   // since that can only have a single element associated with it. Therefore
   // we remove it from the array.
   struct GlobalFaceIndexInfo
   {
      int element_index;
      int local_face_index;
   };

   std::unordered_map<int, GlobalFaceIndexInfo>
   mfem_face_index_info_for_global_face_index;
   std::unordered_set<int> blacklisted_global_face_indices;

   Array<int> global_face_indices, orient;
   for (int ielement = 0; ielement < mesh.GetNE(); ielement++)
   {
      mesh.GetElementFaces(ielement, global_face_indices, orient);

      for (int iface = 0; iface < global_face_indices.Size(); iface++)
      {
         int face_index = global_face_indices[iface];

         if (blacklisted_global_face_indices.count(face_index))
         {
            continue;
         }

         if (mfem_face_index_info_for_global_face_index.count(
                face_index)) // Now we've seen it twice!
         {
            blacklisted_global_face_indices.insert(face_index);
            mfem_face_index_info_for_global_face_index.erase(face_index);
            continue;
         }

         mfem_face_index_info_for_global_face_index[face_index] = { .element_index = ielement, .local_face_index = iface };
      }
   }

   for (int ibdr_element = 0; ibdr_element < mesh.GetNBE(); ibdr_element++)
   {
      int boundary_id = mesh.GetBdrAttribute(ibdr_element);
      int bdr_element_face_index = mesh.GetBdrElementFaceIndex(ibdr_element);

      // Locate match.
      auto & element_face_info = mfem_face_index_info_for_global_face_index.at(
                                    bdr_element_face_index);

      int ielement = element_face_info.element_index;
      int iface = element_face_info.local_face_index;

      // 1. Convert MFEM 0-based element index to 1-based Exodus II element ID.
      int exodusII_element_id = ielement + 1;

      // 2. Convert 0-based MFEM face index to Exodus II 1-based face ID (different ordering).
      int exodusII_face_id;

      Element::Type element_type = mesh.GetElementType(ielement);
      switch (element_type)
      {
         case Element::Type::TETRAHEDRON:
            exodusII_face_id = mfem_to_exodusII_side_map_tet4[iface];
            break;
         default:
            MFEM_ABORT("Cannot handle element of type " << element_type);
      }

      exodusII_element_ids_for_boundary_id[boundary_id].push_back(
         exodusII_element_id);
      exodusII_side_ids_for_boundary_id[boundary_id].push_back(exodusII_face_id);
   }
}

}
