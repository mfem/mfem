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

class ExodusIIWriter
{
public:
   ExodusIIWriter(Mesh & mesh) : _mesh{mesh} {}

   inline int GetExodusFileID() const { return _exid; }

   void HandleNetCDFStatus(int status);

   void GenerateExodusIIElementBlocksFromMesh();

   void GenerateExodusIIBoundaryInfo();

   void GenerateExodusIINodeIDsFromMesh();

   void ExtractVertexCoordinatesFromMesh(std::vector<double> & coordx,
                                         std::vector<double> & coordy,
                                         std::vector<double> & coordz);

   void WriteNodalCoordinatesFromMesh(std::vector<double> & coordx,
                                      std::vector<double> & coordy,
                                      std::vector<double> & coordz);

   void WriteNodeConnectivityForBlock(const int block_id);

   void WriteSideSetInformationForMesh();

   void WriteBlockIDs();

   void CreateEmptyFile(const std::string & fpath);

   void WriteTitle();

   void WriteNumOfElements();

   void WriteFloatingPointWordSize();

   void WriteAPIVersion();

   void WriteDatabaseVersion();

   void WriteMaxLineLength();

   void WriteMaxNameLength();

   void WriteNumElementBlocks();

   void WriteElementBlockParameters();

   void WriteElementBlockParameters(int block_id);

   void WriteNumBoundaries();

   void WriteNumNodes();

   void WriteNodalCoordinates();

   void WriteFileSize();

   void WriteNodeSets();

   void WriteDimension();

   void WriteTimesteps();

   void CloseExodusIIFile();

   void WriteDummyVariable();

private:
   // ExodusII file ID.
   int _exid{0};

   // NetCDF status.
   int _status{NC_NOERR};

   // Reference to mesh we would like to write-out.
   Mesh & _mesh;

   // Mesh info.
   std::vector<int> _block_ids;
   std::map<int, Element::Type> _element_type_for_block_id;
   std::map<int, std::vector<int>> _element_ids_for_block_id;

   std::vector<int> _boundary_ids;
   std::map<int, std::vector<int>> _exodusII_element_ids_for_boundary_id;
   std::map<int, std::vector<int>> _exodusII_side_ids_for_boundary_id;

   int _num_nodes;
   int _num_nodes_id;
   int _num_dim_id;
   int _coordx_id;
   int _coordy_id;
   int _coordz_id;
};


// Returns the Exodus II face ID for the MFEM face index.
const int mfem_to_exodusII_side_map_tet4[] =
{
   2, 3, 1, 4
};

const int mfem_to_exodusII_side_map_hex8[] =
{
   5, 1, 2, 3, 4, 6
};

const int mfem_to_exodusII_side_map_wedge6[] =
{
   4, 5, 1, 2, 3
};

const int mfem_to_exodusII_side_map_pyramid5[] =
{
   5, 1, 2, 3, 4
};


void Mesh::WriteExodusII(const std::string fpath)
{
   int status = NC_NOERR;

   ExodusIIWriter writer(*this);

   //
   // Open file
   //
   writer.CreateEmptyFile(fpath);
   int ncid = writer.GetExodusFileID();

   //
   // Generate Initialization Parameters
   //

   //
   // Add title.
   //
   writer.WriteTitle();

   //
   // Set # nodes. NB: - Assume 1st order currently so NumOfVertices == # nodes
   //
   writer.GenerateExodusIINodeIDsFromMesh();
   writer.WriteNumNodes();

   //
   // Set dimension.
   //
   writer.WriteDimension();

   //
   // Set # elements.
   //
   writer.WriteNumOfElements();

   //
   // Set # element blocks.
   //
   writer.GenerateExodusIIElementBlocksFromMesh();

   writer.WriteNumElementBlocks();

   //
   // Set # node sets - TODO: add this (currently, set to 0).
   //
   writer.WriteNodeSets();

   //
   // Set # side sets ("boundaries")
   //
   writer.GenerateExodusIIBoundaryInfo();

   writer.WriteNumBoundaries();

   //
   // Set database version #
   //
   writer.WriteDatabaseVersion();

   //
   // Set API version #
   //
   writer.WriteAPIVersion();

   //
   // Set I/O word size as an attribute (float == 4; double == 8)
   //
   writer.WriteFloatingPointWordSize();

   //
   // Store Exodus file size (normal==0; large==1). NB: coordinates specifed separately as components
   // for large file.
   //
   writer.WriteFileSize();

   //
   // Set length of character strings.
   //
   writer.WriteMaxNameLength();

   //
   // Set length of character lines.
   //
   writer.WriteMaxLineLength();

   //
   // Set # timesteps (ASSUME no timesteps for initial verision)
   //
   writer.WriteTimesteps();

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
   writer.WriteDummyVariable();

   //
   // Write nodal coordinates.
   //
   writer.WriteNodalCoordinates();

   //
   // Write element block parameters.
   //
   writer.WriteElementBlockParameters();

   //
   // Write block IDs.
   //
   writer.WriteBlockIDs();

   //
   // Write sideset information.
   //
   writer.WriteSideSetInformationForMesh();

   //
   // Close file
   //
   writer.CloseExodusIIFile();

   mfem::out << "Mesh successfully written to Exodus II file" << std::endl;
}

void ExodusIIWriter::CreateEmptyFile(const std::string & fpath)
{
   // Overwrite existing file; use older file version.
   int flags = NC_CLOBBER;

   int status = nc_create(fpath.c_str(), flags, &_exid);
   HandleNetCDFStatus(status);
}

void ExodusIIWriter::CloseExodusIIFile()
{
   int status = nc_close(_exid);
   HandleNetCDFStatus(status);
}

void ExodusIIWriter::WriteTitle()
{
   const char *title = "MFEM mesh";

   int status = nc_put_att_text(_exid, NC_GLOBAL, "title", strlen(title), title);
   HandleNetCDFStatus(status);
}

void ExodusIIWriter::WriteNumOfElements()
{
   int num_elem_id;

   int status = nc_def_dim(_exid, "num_elem", _mesh.GetNE(), &num_elem_id);
   HandleNetCDFStatus(status);
}

void ExodusIIWriter::WriteFloatingPointWordSize()
{
   const int word_size = 8;
   int status = nc_put_att_int(_exid, NC_GLOBAL, "floating_point_word_size",
                               NC_INT, 1,
                               &word_size);
   HandleNetCDFStatus(status);
}

void ExodusIIWriter::WriteAPIVersion()
{
   float version = 4.72;   // Current version as of 2024-03-21.
   int status = nc_put_att_float(_exid, NC_GLOBAL, "api_version", NC_FLOAT, 1,
                                 &version);
   HandleNetCDFStatus(status);
}

void ExodusIIWriter::WriteDatabaseVersion()
{
   float database_version = 4.72;
   int status = nc_put_att_float(_exid, NC_GLOBAL, "version", NC_FLOAT, 1,
                                 &database_version);
   HandleNetCDFStatus(status);
}

void ExodusIIWriter::WriteMaxNameLength()
{
   const int max_name_length = 80;
   int status = nc_put_att_int(_exid, NC_GLOBAL, "maximum_name_length", NC_INT, 1,
                               &max_name_length);
   HandleNetCDFStatus(status);
}

void ExodusIIWriter::WriteMaxLineLength()
{
   const int max_line_length = 80;
   int status = nc_put_att_int(_exid, NC_GLOBAL, "maximum_line_length", NC_INT, 1,
                               &max_line_length);
   HandleNetCDFStatus(status);
}



void ExodusIIWriter::WriteBlockIDs()
{
   int status, unique_block_ids_ptr;

   int block_dim;
   status = nc_def_dim(_exid, "block_dim", _block_ids.size(), &block_dim);
   HandleNetCDFStatus(status);

   status = nc_def_var(_exid, "eb_prop1", NC_INT, 1, &block_dim,
                       &unique_block_ids_ptr);
   HandleNetCDFStatus(status);

   nc_enddef(_exid);
   status = nc_put_var_int(_exid, unique_block_ids_ptr, _block_ids.data());
   HandleNetCDFStatus(status);
   nc_redef(_exid);
}

void ExodusIIWriter::WriteElementBlockParameters()
{
   for (int block_id : _block_ids)
   {
      WriteElementBlockParameters(block_id);
   }
}

void ExodusIIWriter::WriteElementBlockParameters(int block_id)
{
   int status{NC_NOERR};
   char name_buffer[100];

   const auto & block_element_ids = _element_ids_for_block_id.at(block_id);
   //
   // TODO: - element block IDs 0-indexed MFEM --> 1-indexed Exodus II
   //

   Element * front_element = _mesh.GetElement(block_element_ids.front());

   //
   // Define # elements in the block.
   //
   sprintf(name_buffer, "num_el_in_blk%d", block_id);

   int num_el_in_blk_id;
   status = nc_def_dim(_exid, name_buffer, block_element_ids.size(),
                       &num_el_in_blk_id);
   HandleNetCDFStatus(status);


   //
   // Define # nodes per element. NB: - assume first-order elements currently!!
   //
   sprintf(name_buffer, "num_nod_per_el%d", block_id);

   int num_node_per_el_id;
   status = nc_def_dim(_exid, name_buffer, front_element->GetNVertices(),
                       &num_node_per_el_id);
   HandleNetCDFStatus(status);

   //
   // Define # edges per element:
   //
   sprintf(name_buffer, "num_edg_per_el%d", block_id);

   int num_edg_per_el_id;
   status = nc_def_dim(_exid, name_buffer, front_element->GetNEdges(),
                       &num_edg_per_el_id);
   HandleNetCDFStatus(status);

   //
   // Define # faces per element.
   //
   sprintf(name_buffer, "num_fac_per_el%d", block_id);

   int num_fac_per_el_id;
   status = nc_def_dim(_exid, name_buffer, front_element->GetNFaces(),
                       &num_fac_per_el_id);
   HandleNetCDFStatus(status);

   //
   // Define element node connectivity for block.
   //
   WriteNodeConnectivityForBlock(block_id);

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
      case Geometry::Type::PYRAMID:
         element_type = "pyramid";
         break;
      default:
         MFEM_ABORT("Unsupported MFEM element type: " << front_element->GetType());
   }

   sprintf(name_buffer, "connect%d", block_id);

   int connect_id;
   status = nc_inq_varid(_exid, name_buffer, &connect_id);
   HandleNetCDFStatus(status);

   status = nc_put_att_text(_exid, connect_id, "elem_type", element_type.length(),
                            element_type.c_str());
   HandleNetCDFStatus(status);
}

void ExodusIIWriter::WriteNumNodes()
{
   int status = nc_def_dim(_exid, "num_nodes", _num_nodes, &_num_nodes_id);
   HandleNetCDFStatus(status);
}

void ExodusIIWriter::WriteNodalCoordinates()
{
   //
   // Define nodal coordinates.
   // https://docs.unidata.ucar.edu/netcdf-c/current/group__variables.html#gac7e8662c51f3bb07d1fc6d6c6d9052c8
   // NB: assume we have doubles (could be floats!)
   // ndims = 1 (vectors).
   std::vector<double> coordx(_num_nodes), coordy(_num_nodes),
       coordz(_mesh.Dimension() == 3 ? _num_nodes : 0);

   ExtractVertexCoordinatesFromMesh(coordx, coordy, coordz);

   int status = nc_def_var(_exid, "coordx", NC_DOUBLE, 1, &_num_nodes_id,
                           &_coordx_id);
   HandleNetCDFStatus(status);

   status = nc_def_var(_exid, "coordy", NC_DOUBLE, 1, &_num_nodes_id, &_coordy_id);
   HandleNetCDFStatus(status);

   if (_mesh.Dimension() == 3)
   {
      status = nc_def_var(_exid, "coordz", NC_DOUBLE, 1, &_num_nodes_id, &_coordz_id);
      HandleNetCDFStatus(status);
   }

   //
   // Write nodal coordinates.
   //
   WriteNodalCoordinatesFromMesh(coordx, coordy, coordz);
}

void ExodusIIWriter::WriteSideSetInformationForMesh()
{
   //
   // Add the boundary IDs
   //
   int status, boundary_ids_ptr;

   int boundary_ids_dim;
   status = nc_def_dim(_exid, "boundary_ids_dim", _boundary_ids.size(),
                       &boundary_ids_dim);
   HandleNetCDFStatus(status);

   status = nc_def_var(_exid, "ss_prop1", NC_INT, 1, &boundary_ids_dim,
                       &boundary_ids_ptr);
   HandleNetCDFStatus(status);

   nc_enddef(_exid);
   status = nc_put_var_int(_exid, boundary_ids_ptr, _boundary_ids.data());
   HandleNetCDFStatus(status);
   nc_redef(_exid);

   //
   // Add the number of elements for each boundary_id.
   //
   char name_buffer[100];

   for (int boundary_id : _boundary_ids)
   {
      size_t num_elements_for_boundary = _exodusII_element_ids_for_boundary_id.at(
                                            boundary_id).size();

      sprintf(name_buffer, "num_side_ss%d", boundary_id);

      int num_side_ss_id;
      status = nc_def_dim(_exid, name_buffer, num_elements_for_boundary,
                          &num_side_ss_id);
      HandleNetCDFStatus(status);
   }

   //
   // Add the faces here.
   //
   for (int boundary_id : _boundary_ids)
   {
      const std::vector<int> & side_ids = _exodusII_side_ids_for_boundary_id.at(
                                             boundary_id);

      sprintf(name_buffer, "side_ss%d_dim", boundary_id);

      int side_id_dim;
      status = nc_def_dim(_exid, name_buffer, side_ids.size(), &side_id_dim);
      HandleNetCDFStatus(status);

      sprintf(name_buffer, "side_ss%d", boundary_id);

      int side_id_ptr;
      status = nc_def_var(_exid, name_buffer, NC_INT, 1,  &side_id_dim, &side_id_ptr);
      HandleNetCDFStatus(status);

      nc_enddef(_exid);
      status = nc_put_var_int(_exid, side_id_ptr, side_ids.data());
      HandleNetCDFStatus(status);
      nc_redef(_exid);
   }

   //
   // Add the boundary elements here.
   //
   for (int boundary_id : _boundary_ids)
   {
      // TODO: - need to figure-out correct element ids (we only have local element indexes into the boundaries array!)
      const std::vector<int> & element_ids = _exodusII_element_ids_for_boundary_id.at(
                                                boundary_id);

      sprintf(name_buffer, "elem_ss%d_dim", boundary_id);

      int elem_ids_dim;
      status = nc_def_dim(_exid, name_buffer, element_ids.size(), &elem_ids_dim);
      HandleNetCDFStatus(status);

      sprintf(name_buffer, "elem_ss%d", boundary_id);

      int elem_ids_ptr;
      status = nc_def_var(_exid, name_buffer, NC_INT, 1, &elem_ids_dim,
                          &elem_ids_ptr);
      HandleNetCDFStatus(status);

      nc_enddef(_exid);
      status = nc_put_var_int(_exid, elem_ids_ptr, element_ids.data());
      HandleNetCDFStatus(status);
      nc_redef(_exid);
   }
}

void ExodusIIWriter::WriteNodeConnectivityForBlock(const int block_id)
{
   int status{NC_NOERR}, connect_id;

   // Generate arbitrary name:
   char name_buffer[100];

   std::vector<int> block_node_connectivity;

   for (int element_id : _element_ids_for_block_id.at(block_id))
   {
      // NB: assume first-order elements only for now.
      // NB: - need to convert from 0-based indexing --> 1-based indexing.
      mfem::Array<int> element_vertices;
      _mesh.GetElementVertices(element_id, element_vertices);

      for (int vertex_id : element_vertices)
      {
         block_node_connectivity.push_back(vertex_id + 1);  // 1-based indexing.
      }
   }

   sprintf(name_buffer, "connect%d_dim", block_id);

   int node_connectivity_dim;
   status = nc_def_dim(_exid, name_buffer, block_node_connectivity.size(),
                       &node_connectivity_dim);
   HandleNetCDFStatus(status);

   // NB: 1 == vector!; name is arbitrary; NC_INT or NCINT64??
   sprintf(name_buffer, "connect%d", block_id);

   status = nc_def_var(_exid, name_buffer, NC_INT, 1, &node_connectivity_dim,
                       &connect_id);
   HandleNetCDFStatus(status);

   nc_enddef(_exid);
   status = nc_put_var_int(_exid, connect_id, block_node_connectivity.data());
   HandleNetCDFStatus(status);
   nc_redef(_exid);
}


void ExodusIIWriter::ExtractVertexCoordinatesFromMesh(std::vector<double> &
                                                      coordx, std::vector<double> & coordy,
                                                      std::vector<double> & coordz)
{
   for (int ivertex = 0; ivertex < _mesh.GetNV(); ivertex++)
   {
      double * coordinates = _mesh.GetVertex(ivertex);

      coordx[ivertex] = coordinates[0];
      coordy[ivertex] = coordinates[1];

      if (_mesh.Dimension() == 3)
      {
         coordz[ivertex] = coordinates[2];
      }
   }
}

void ExodusIIWriter::WriteNodalCoordinatesFromMesh(std::vector<double> & coordx,
                                                   std::vector<double> & coordy,
                                                   std::vector<double> & coordz)
{
   int status = NC_NOERR;

   nc_enddef(_exid);

   status = nc_put_var_double(_exid, _coordx_id, coordx.data());
   HandleNetCDFStatus(status);

   status = nc_put_var_double(_exid, _coordy_id, coordy.data());
   HandleNetCDFStatus(status);

   if (coordz.size() != 0)
   {
      status = nc_put_var_double(_exid, _coordz_id, coordz.data());
      HandleNetCDFStatus(status);
   }

   nc_redef(_exid);
}

/// @brief Aborts with description of error.
void ExodusIIWriter::HandleNetCDFStatus(int status)
{
   if (status != NC_NOERR)
   {
      MFEM_ABORT("NetCDF error: " << nc_strerror(status));
   }
}

void ExodusIIWriter::WriteFileSize()
{
   // Store Exodus file size (normal==0; large==1). NB: coordinates specifed separately as components
   // for large file.
   const int file_size = 1;

   int status = nc_put_att_int(_exid, NC_GLOBAL, "file_size", NC_INT, 1,
                               &file_size);
   HandleNetCDFStatus(status);
}

void ExodusIIWriter::WriteDimension()
{
   int status = nc_def_dim(_exid, "num_dim", _mesh.Dimension(), &_num_dim_id);
   HandleNetCDFStatus(status);
}

void ExodusIIWriter::WriteNodeSets()
{
   // Set # node sets - TODO: add this (currently, set to 0).
   int num_node_sets_ids;

   int status = nc_def_dim(_exid, "num_node_sets", 0, &num_node_sets_ids);
   HandleNetCDFStatus(status);
}

void ExodusIIWriter::WriteTimesteps()
{
   // Set # timesteps (ASSUME no timesteps for initial verision)
   int timesteps_dim;

   int status = nc_def_dim(_exid, "time_step", 1, &timesteps_dim);
   HandleNetCDFStatus(status);
}

void ExodusIIWriter::WriteDummyVariable()
{
   // NB: LibMesh has a dodgy bug where it will skip the x-coordinate if coordx_id == 0.
   // To prevent this, the first variable to be defined will be a dummy variable which will
   // have a variable id of 0.
   int dummy_var_id, dummy_var_dim_id, dummy_value = 1;

   int status = nc_def_dim(_exid, "dummy_var_dim", 1, &dummy_var_dim_id);
   HandleNetCDFStatus(status);

   status = nc_def_var(_exid, "dummy_var", NC_INT, 1, &dummy_var_dim_id,
                       &dummy_var_id);
   HandleNetCDFStatus(status);

   nc_enddef(_exid);
   status = nc_put_var_int(_exid, dummy_var_id, &dummy_value);
   HandleNetCDFStatus(status);
   nc_redef(_exid);
}

/// @brief Generates blocks based on the elements in the mesh. We assume that this was originally an Exodus II
/// mesh. Therefore, we iterate over the elements and use the attributes as the element blocks. We assume that
/// all elements belonging to the same block will have the same attribute. We can perform a safety check as well
/// by ensuring that all elements in the block have the same element type. If this is not the case then something
/// has gone horribly wrong!
void ExodusIIWriter::GenerateExodusIIElementBlocksFromMesh()
{
   _block_ids.clear();
   _element_ids_for_block_id.clear();
   _element_type_for_block_id.clear();

   std::set<int> observed_block_ids;

   // Iterate over the elements in the mesh.
   for (int ielement = 0; ielement < _mesh.GetNE(); ielement++)
   {
      Element::Type element_type = _mesh.GetElementType(ielement);

      int block_id = _mesh.GetAttribute(ielement);

      if (observed_block_ids.count(block_id) == 0)
      {
         _block_ids.push_back(block_id);

         _element_type_for_block_id[block_id] = element_type;
         _element_ids_for_block_id[block_id] = { ielement };

         observed_block_ids.insert(block_id);
      }
      else
      {
         auto & block_element_ids = _element_ids_for_block_id.at(block_id);
         block_element_ids.push_back(ielement);

         // Safety check: ensure that the element type matches what we have on record for the block.
         if (element_type != _element_type_for_block_id.at(block_id))
         {
            MFEM_ABORT("Multiple element types are defined for block: " << block_id);
         }
      }
   }
}

void ExodusIIWriter::WriteNumElementBlocks()
{
   int num_elem_blk_id;
   int status = nc_def_dim(_exid, "num_el_blk", (int)_block_ids.size(),
                           &num_elem_blk_id);
   HandleNetCDFStatus(status);
}

/// @brief Iterates over the elements of the mesh to extract a unique set of node IDs (or vertex IDs if first-order).
void ExodusIIWriter::GenerateExodusIINodeIDsFromMesh()
{
   std::set<int> node_ids;

   const FiniteElementSpace * fespace = _mesh.GetNodalFESpace();

   mfem::Array<int> dofs;

   for (int ielement = 0; ielement < _mesh.GetNE(); ielement++)
   {
      if (fespace)   // Higher-order
      {
         fespace->GetElementDofs(ielement, dofs);

         for (int dof : dofs) { node_ids.insert(dof); }
      }
      else
      {
         mfem::Array<int> vertex_indices;
         _mesh.GetElementVertices(ielement, vertex_indices);

         // TODO: - Hmmmm. These are not actually the dofs. Just the vertex offsets.
         for (int vertex_index : vertex_indices)
         {
            node_ids.insert(vertex_index);
         }
      }
   }

   _num_nodes = (int)node_ids.size();
}

void ExodusIIWriter::WriteNumBoundaries()
{
   int num_side_sets_ids;
   int status = nc_def_dim(_exid, "num_side_sets", _boundary_ids.size(),
                           &num_side_sets_ids);
   HandleNetCDFStatus(status);
}

void ExodusIIWriter::GenerateExodusIIBoundaryInfo()
{
   // Store the unique boundary IDs.
   _boundary_ids.clear();
   _exodusII_element_ids_for_boundary_id.clear();
   _exodusII_side_ids_for_boundary_id.clear();

   for (int bdr_attribute : _mesh.bdr_attributes)
   {
      _boundary_ids.push_back(bdr_attribute);
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
   for (int ielement = 0; ielement < _mesh.GetNE(); ielement++)
   {
      _mesh.GetElementFaces(ielement, global_face_indices, orient);

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

   for (int ibdr_element = 0; ibdr_element < _mesh.GetNBE(); ibdr_element++)
   {
      int boundary_id = _mesh.GetBdrAttribute(ibdr_element);
      int bdr_element_face_index = _mesh.GetBdrElementFaceIndex(ibdr_element);

      // Locate match.
      auto & element_face_info = mfem_face_index_info_for_global_face_index.at(
                                    bdr_element_face_index);

      int ielement = element_face_info.element_index;
      int iface = element_face_info.local_face_index;

      // 1. Convert MFEM 0-based element index to 1-based Exodus II element ID.
      int exodusII_element_id = ielement + 1;

      // 2. Convert 0-based MFEM face index to Exodus II 1-based face ID (different ordering).
      int exodusII_face_id;

      Element::Type element_type = _mesh.GetElementType(ielement);
      switch (element_type)
      {
         case Element::Type::TETRAHEDRON:
            exodusII_face_id = mfem_to_exodusII_side_map_tet4[iface];
            break;
         case Element::Type::HEXAHEDRON:
            exodusII_face_id = mfem_to_exodusII_side_map_hex8[iface];
            break;
         case Element::Type::WEDGE:
            exodusII_face_id = mfem_to_exodusII_side_map_wedge6[iface];
            break;
         case Element::Type::PYRAMID:
            exodusII_face_id = mfem_to_exodusII_side_map_pyramid5[iface];
            break;
         default:
            MFEM_ABORT("Cannot handle element of type " << element_type);
      }

      _exodusII_element_ids_for_boundary_id[boundary_id].push_back(
         exodusII_element_id);
      _exodusII_side_ids_for_boundary_id[boundary_id].push_back(exodusII_face_id);
   }
}

}
