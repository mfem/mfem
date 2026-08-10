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

#include "mesh_headers.hpp"
#include <unordered_set>
#include <cstdarg>

#ifdef MFEM_USE_NETCDF
#include "netcdf.h"
#endif

// Call NetCDF functions inside the macro. This will provide basic error-handling.
#define CHECK_NETCDF_CODE(return_code)\
{\
   if ((return_code) != NC_NOERR)\
   {\
      MFEM_ABORT("NetCDF error: " << nc_strerror((return_code)));\
   }\
}

#if defined(MFEM_USE_DOUBLE)
#define MFEM_NETCDF_REAL_T NC_DOUBLE
#elif defined(MFEM_USE_SINGLE)
#define MFEM_NETCDF_REAL_T NC_FLOAT
#endif

namespace mfem
{

#ifdef MFEM_USE_NETCDF

namespace ExodusIISideMaps
{
/// Convert from the MFEM face numbering to the ExodusII face numbering.
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
}

namespace ExodusIINodeOrderings
{
/// Convert from the MFEM (0-based) node ordering to the ExodusII 1-based node
/// ordering.
const int mfem_to_exodusII_node_ordering_tet10[] =
{
   1, 2, 3, 4, 5, 8, 6, 7, 9, 10
};

const int mfem_to_exodusII_node_ordering_hex27[] =
{
   1, 2, 3, 4, 5, 6, 7, 8, 9,
   10, 11, 12, 17, 18, 19, 20, 13, 14,
   15, 16, 27, 21, 26, 25, 23, 22, 24
};

const int mfem_to_exodusII_node_ordering_wedge18[] =
{
   1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 10, 11, 12, 16, 17, 18
};

const int mfem_to_exodusII_node_ordering_pyramid14[] =
{
   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
};
}


namespace ExodusIILabels
{
// Variable labels
const char * EXODUS_TITLE_LABEL = "title";
const char * EXODUS_NUM_ELEM_LABEL = "num_elem";
const char * EXODUS_FLOATING_POINT_WORD_SIZE_LABEL = "floating_point_word_size";
const char * EXODUS_API_VERSION_LABEL = "api_version";
const char * EXODUS_DATABASE_VERSION_LABEL = "version";
const char * EXODUS_MAX_NAME_LENGTH_LABEL = "maximum_name_length";
const char * EXODUS_MAX_LINE_LENGTH_LABEL = "maximum_line_length";
const char * EXODUS_NUM_BLOCKS_LABEL = "block_dim";
const char * EXODUS_COORDX_LABEL = "coordx";
const char * EXODUS_COORDY_LABEL = "coordy";
const char * EXODUS_COORDZ_LABEL = "coordz";
const char * EXODUS_NUM_BOUNDARIES_LABEL = "boundary_ids_dim";
const char * EXODUS_FILE_SIZE_LABEL = "file_size";
const char * EXODUS_NUM_DIM_LABEL = "num_dim";
const char * EXODUS_NUM_NODE_SETS_LABEL = "num_node_sets";
const char * EXODUS_TIME_STEP_LABEL = "time_step";
const char * EXODUS_ELEMENT_TYPE_LABEL = "elem_type";
const char * EXODUS_NUM_SIDE_SETS_LABEL = "num_side_sets";
const char * EXODUS_SIDE_SET_IDS_LABEL = "ss_prop1";
const char * EXODUS_ELEMENT_BLOCK_IDS_LABEL = "eb_prop1";
const char * EXODUS_NUM_ELEMENT_BLOCKS_LABEL = "num_el_blk";
const char * EXODUS_MESH_TITLE = "MFEM mesh";

// Current version as of 2024-03-21.
const float EXODUS_API_VERSION = 4.72;
const float EXODUS_DATABASE_VERSION = 4.72;

const int EXODUS_MAX_NAME_LENGTH = 80;
const int EXODUS_MAX_LINE_LENGTH = 80;
}

/**
 * Helper class for writing a mesh to an ExodusII file.
 */
class ExodusIIWriter
{
public:
   /// @brief Default constructor. Opens ExodusII file.
   /// @param mesh The mesh to write to the file.
   ExodusIIWriter(Mesh & mesh) : mesh{mesh} {}

   ExodusIIWriter() = delete;

   /// @brief Closes ExodusII file if it has been opened.
   ~ExodusIIWriter();

   /// @brief Writes the mesh to an ExodusII file.
   /// @param fpath The path to the file.
   /// @param flags NC_CLOBBER will overwrite existing file.
   void PrintExodusII(const std::string &fpath, int flags = NC_CLOBBER);

   /// @brief Static method for writing a mesh to an ExodusII file.
   /// @param mesh The mesh to write to the file.
   /// @param fpath The path to the file.
   /// @param flags NetCDF file flags.
   static void PrintExodusII(Mesh & mesh, const std::string &fpath,
                             int flags = NC_CLOBBER);

protected:
   /// @brief Closes any open file and creates a NetCDF file using selected flags.
   void OpenExodusII(const std::string &fpath, int flags);

   /// @brief Closes any open file.
   void CloseExodusII();

   /// @brief Generates blocks based on the elements in the mesh. We iterate
   /// over the mesh elements and use the attributes as the element blocks. We
   /// assume that all elements belonging to the same block will share the same
   /// attribute. We perform a safety check to verify that all elements in the
   /// block have the same element type.
   void GenerateExodusIIElementBlocks();

   /// @brief Extracts boundary ids and determines the element IDs and side IDs
   /// (Exodus II) for each boundary element.
   void GenerateExodusIIBoundaryInfo();

   /// @brief Iterates over the elements to extract a unique set of node IDs
   /// (or vertex IDs if first-order).
   std::unordered_set<int> GenerateUniqueNodeIDs();

   /// @brief Populates vectors with x, y, z coordinates from mesh.
   void ExtractVertexCoordinates(std::vector<real_t> &coordx,
                                 std::vector<real_t> &coordy,
                                 std::vector<real_t> &coordz);

   /// @brief Writes node connectivity for a particular block.
   /// @param block_id The block to write to the file.
   void WriteNodeConnectivityForBlock(const int block_id);

   /// @brief Writes boundary information to file.
   void WriteBoundaries();

   /// @brief Writes the block IDs to the file.
   void WriteBlockIDs();

   /// @brief Writes a title to the file.
   void WriteTitle();

   /// @brief Writes the number of elements in the mesh.
   void WriteNumOfElements();

   /// @brief Writes the floating-point word size (sizeof(real_t)).
   void WriteFloatingPointWordSize();

   /// @brief Writes the API version.
   void WriteAPIVersion();

   /// @brief Writes the database version.
   void WriteDatabaseVersion();

   /// @brief Writes the maximum length of a line.
   void WriteMaxLineLength();

   /// @brief Writes the maximum length of a name.
   void WriteMaxNameLength();

   /// @brief Writes the number of blocks.
   void WriteNumElementBlocks();

   /// @brief Writes all element block parameters.
   void WriteElementBlocks();

   /// @brief Called by @a WriteElementBlockParameters in for-loop.
   /// @param block_id Block to write parameters.
   void WriteElementBlockParameters(int block_id);

   /// @brief Writes the coordinates of nodes.
   void WriteNodalCoordinates();

   /// @brief Writes the file size (normal=0; large=1). Coordinates are specified
   /// separately as components for large files (i.e. xxx, yyy, zzz) as opposed
   /// to (xyz, xyz, xyz) for normal files.
   void WriteFileSize();

   /// @brief Writes the nodesets. Currently, we do not support nodesets.
   void WriteNodeSets();

   /// @brief Writes the mesh dimension.
   void WriteMeshDimension();

   /// @brief Writes the number of timesteps. Currently, we do not support
   /// multiple timesteps.
   void WriteTimesteps();

   /// @brief Writes a dummy variable. This is to circumvent a bug in LibMesh where
   /// it will skip the x-coordinate when reading in an ExodusII file if the id of
   /// the x-coordinates is 0. To prevent this, we define a dummy variable before
   /// defining the coordinates. This ensures that the coordinate variable IDs have
   /// values greater than zero. See: https://github.com/libMesh/libmesh/issues/3823
   void WriteDummyVariable();

   /// @brief Wrapper around @a nc_def_dim with error handling.
   void DefineDimension(const char *name, size_t len, int *dim_id);

   /// @brief Wrapper around @a nc_def_var with error handling.
   void DefineVar(const char *name, nc_type xtype, int ndims, const int *dimidsp,
                  int *varidp);

   /// @brief Write variable data to the file. This is a wrapper around
   /// @a nc_put_var with error handling.
   void PutVar(int varid, const void * data);

   /// @brief Combine @a DefineVar with @a PutVar.
   void DefineAndPutVar(const char *name, nc_type xtype, int ndims,
                        const int *dimidsp, const void *data);

   /// @brief Write attribute to the file. This is a wrapper around @a nc_put_att
   /// with error handling.
   void PutAtt(int varid, const char *name, nc_type xtype, size_t len,
               const void * data);

   /// @brief Returns a pointer to a static buffer containing the character
   /// string with formatting. Used to generate variable labels.
   char * GenerateLabel(const char * format, ...);

   /// @brief Writes boiler-plate information for ExodusII file format including
   /// title, database version, file size etc.
   void WriteExodusIIFileInformation();

   /// @brief Writes all information about the mesh to the ExodusII file.
   void WriteExodusIIMeshInformation();

private:
   /// @brief Verifies that the nodal FESpace exists and is H1, order 2.
   void CheckNodalFESpaceIsSecondOrderH1() const;

   // ExodusII file ID.
   int exid{-1};

   /// Flag to check if a file is currently open.
   bool file_open{false};

   // Reference to mesh we would like to write-out.
   Mesh & mesh;

   // Block information.
   std::vector<int> block_ids;
   std::map<int, Element::Type> element_type_for_block_id;
   std::map<int, std::vector<int>> element_ids_for_block_id;

   std::vector<int> boundary_ids;
   std::map<int, std::vector<int>> exodusII_element_ids_for_boundary_id;
   std::map<int, std::vector<int>> exodusII_side_ids_for_boundary_id;
};

void Mesh::PrintExodusII(const std::string &fpath)
{
   ExodusIIWriter::PrintExodusII(*this, fpath);
}

void ExodusIIWriter::DefineDimension(const char *name, size_t len, int *dim_id)
{
   nc_redef(exid);
   CHECK_NETCDF_CODE(nc_def_dim(exid, name, len, dim_id));
}

void ExodusIIWriter::DefineVar(const char *name, nc_type xtype, int ndims,
                               const int *dimidsp, int *varidp)
{
   nc_redef(exid);  // Switch to define mode.
   CHECK_NETCDF_CODE(nc_def_var(exid, name, xtype, ndims, dimidsp,
                                varidp));
}

void ExodusIIWriter::PutAtt(int varid, const char *name, nc_type xtype,
                            size_t len, const void * data)
{
   nc_redef(exid);
   CHECK_NETCDF_CODE(nc_put_att(exid, varid, name, xtype, len, data));
}

void ExodusIIWriter::PutVar(int varid, const void * data)
{
   nc_enddef(exid); // Switch to data mode.
   CHECK_NETCDF_CODE(nc_put_var(exid, varid, data));
}

void ExodusIIWriter::DefineAndPutVar(const char *name, nc_type xtype, int ndims,
                                     const int *dimidsp, const void *data)
{
   int varid;
   DefineVar(name, xtype, ndims, dimidsp, &varid);
   PutVar(varid, data);
}

void ExodusIIWriter::WriteExodusIIFileInformation()
{
   WriteTitle();

   WriteDatabaseVersion();
   WriteAPIVersion();

   WriteFloatingPointWordSize();

   WriteFileSize();

   WriteMaxNameLength();
   WriteMaxLineLength();

   WriteDummyVariable();
}

void ExodusIIWriter::WriteExodusIIMeshInformation()
{
   WriteMeshDimension();
   WriteNumOfElements();

   WriteTimesteps();

   WriteNodalCoordinates();

   WriteElementBlocks();
   WriteBoundaries();
   WriteNodeSets();
}

void ExodusIIWriter::PrintExodusII(const std::string &fpath, int flags)
{
   OpenExodusII(fpath, flags);

   WriteExodusIIFileInformation();
   WriteExodusIIMeshInformation();

   CloseExodusII();

   mfem::out << "Mesh successfully written to Exodus II file" << std::endl;
}

void ExodusIIWriter::PrintExodusII(Mesh &mesh, const std::string &fpath,
                                   int flags)
{
   ExodusIIWriter writer(mesh);

   writer.PrintExodusII(fpath, flags);
}

void ExodusIIWriter::OpenExodusII(const std::string &fpath, int flags)
{
   CloseExodusII();  // Close any open files.

   CHECK_NETCDF_CODE(nc_create(fpath.c_str(), flags, &exid));

   file_open = true;
}

void ExodusIIWriter::CloseExodusII()
{
   if (!file_open) { return; }   // No files open.

   CHECK_NETCDF_CODE(nc_close(exid));

   file_open = false;
   exid = (-1);  // Set to negative value (valid IDs are positive!)
}

ExodusIIWriter::~ExodusIIWriter()
{
   CloseExodusII();
}

void ExodusIIWriter::WriteTitle()
{
   PutAtt(NC_GLOBAL, ExodusIILabels::EXODUS_TITLE_LABEL, NC_CHAR,
          strlen(ExodusIILabels::EXODUS_MESH_TITLE),
          ExodusIILabels::EXODUS_MESH_TITLE);
}

void ExodusIIWriter::WriteNumOfElements()
{
   int num_elem_id;
   DefineDimension(ExodusIILabels::EXODUS_NUM_ELEM_LABEL, mesh.GetNE(),
                   &num_elem_id);
}

void ExodusIIWriter::WriteFloatingPointWordSize()
{
   const int word_size = sizeof(real_t);
   PutAtt(NC_GLOBAL, ExodusIILabels::EXODUS_FLOATING_POINT_WORD_SIZE_LABEL,
          NC_INT, 1,
          &word_size);
}

void ExodusIIWriter::WriteAPIVersion()
{
   PutAtt(NC_GLOBAL, ExodusIILabels::EXODUS_API_VERSION_LABEL, MFEM_NETCDF_REAL_T,
          1,
          &ExodusIILabels::EXODUS_API_VERSION);
}

void ExodusIIWriter::WriteDatabaseVersion()
{
   PutAtt(NC_GLOBAL, ExodusIILabels::EXODUS_DATABASE_VERSION_LABEL,
          MFEM_NETCDF_REAL_T, 1,
          &ExodusIILabels::EXODUS_DATABASE_VERSION);
}

void ExodusIIWriter::WriteMaxNameLength()
{
   PutAtt(NC_GLOBAL, ExodusIILabels::EXODUS_MAX_NAME_LENGTH_LABEL, NC_INT, 1,
          &ExodusIILabels::EXODUS_MAX_NAME_LENGTH);
}

void ExodusIIWriter::WriteMaxLineLength()
{
   PutAtt(NC_GLOBAL, ExodusIILabels::EXODUS_MAX_LINE_LENGTH_LABEL, NC_INT, 1,
          &ExodusIILabels::EXODUS_MAX_LINE_LENGTH);
}

void ExodusIIWriter::WriteBlockIDs()
{
   int block_dim;
   DefineDimension(ExodusIILabels::EXODUS_NUM_BLOCKS_LABEL, block_ids.size(),
                   &block_dim);

   DefineAndPutVar(ExodusIILabels::EXODUS_ELEMENT_BLOCK_IDS_LABEL, NC_INT, 1,
                   &block_dim,
                   block_ids.data());
}

void ExodusIIWriter::WriteElementBlocks()
{
   GenerateExodusIIElementBlocks();

   WriteNumElementBlocks();
   WriteBlockIDs();

   for (int block_id : block_ids)
   {
      WriteElementBlockParameters(block_id);
   }
}

char * ExodusIIWriter::GenerateLabel(const char * format, ...)
{
   va_list arglist;
   va_start(arglist, format);

   const int buffer_size = 100;

   static char buffer[buffer_size];
   int nwritten = vsnprintf(buffer, buffer_size, format, arglist);

   bool ok = (nwritten > 0 && nwritten < buffer_size);
   if (!ok)
   {
      MFEM_ABORT("Unable to write characters to buffer.");
   }

   va_end(arglist);
   return buffer;
}

void ExodusIIWriter::WriteElementBlockParameters(int block_id)
{
   char * label{nullptr};

   const std::vector<int> & block_element_ids = element_ids_for_block_id.at(
                                                   block_id);
   const Element * front_element = mesh.GetElement(block_element_ids.front());

   // 1. Define number of elements in the block.
   label = GenerateLabel("num_el_in_blk%d", block_id);

   int num_el_in_blk_id;
   DefineDimension(label, block_element_ids.size(),
                   &num_el_in_blk_id);

   // 2. Define number of nodes per element.
   label = GenerateLabel("num_nod_per_el%d", block_id);

   int num_node_per_el_id;
   if (mesh.GetNodes())
   {
      // Safety check: H1, order 2 fespace.
      CheckNodalFESpaceIsSecondOrderH1();

      // Higher order. Get the first element from the block.
      const FiniteElementSpace * fespace = mesh.GetNodalFESpace();

      auto & block_elements = element_ids_for_block_id.at(block_id);

      int first_element_id = block_elements.front();

      Array<int> dofs;
      fespace->GetElementDofs(first_element_id, dofs);

      DefineDimension(label, dofs.Size(),
                      &num_node_per_el_id);
   }
   else
   {
      DefineDimension(label, front_element->GetNVertices(),
                      &num_node_per_el_id);
   }

   // 3. Define number of edges per element:
   label = GenerateLabel("num_edg_per_el%d", block_id);

   int num_edg_per_el_id;
   DefineDimension(label, front_element->GetNEdges(),
                   &num_edg_per_el_id);

   // 4. Define number of faces per element.
   label = GenerateLabel("num_fac_per_el%d", block_id);

   int num_fac_per_el_id;
   DefineDimension(label, front_element->GetNFaces(),
                   &num_fac_per_el_id);

   // 5. Define element node connectivity for block.
   WriteNodeConnectivityForBlock(block_id);

   // 6. Define the element type.
   std::string element_type;

   const FiniteElementSpace * fespace = mesh.GetNodalFESpace();

   // Safety check: assume that the elements are of the same order.
   MFEM_ASSERT((!fespace || (fespace &&
                             !fespace->IsVariableOrder())),
               "Spaces with varying element orders are not supported.");

   bool higher_order = (fespace && fespace->GetMaxElementOrder() > 1);

   switch (front_element->GetType())
   {
      case Element::HEXAHEDRON:
         element_type = higher_order ? "HEX27" : "Hex8";
         break;
      case Element::TETRAHEDRON:
         element_type = higher_order ? "TETRA10" : "TETRA4";
         break;
      case Element::WEDGE:
         element_type = higher_order ? "WEDGE18" : "WEDGE6";
         break;
      case Element::PYRAMID:
         element_type = higher_order ? "PYRAMID14" : "PYRAMID5";
         break;
      default:
         MFEM_ABORT("Unsupported MFEM element type: " << front_element->GetType());
   }

   label = GenerateLabel("connect%d", block_id);

   int connect_id;
   CHECK_NETCDF_CODE(nc_inq_varid(exid, label, &connect_id));

   PutAtt(connect_id, ExodusIILabels::EXODUS_ELEMENT_TYPE_LABEL, NC_CHAR,
          element_type.length(),
          element_type.c_str());
}

void ExodusIIWriter::WriteNodalCoordinates()
{
   // 1. Generate the unique node IDs.
   std::unordered_set<int> unique_node_ids = GenerateUniqueNodeIDs();
   const size_t num_nodes = unique_node_ids.size();

   // 2. Define the "num_nodes" dimension.
   int num_nodes_id;
   DefineDimension("num_nodes", num_nodes, &num_nodes_id);

   // 3. Extract the nodal coordinates.
   // NB: writes in format real_t (double or float); ndims = 1 (vector).
   // https://docs.unidata.ucar.edu/netcdf-c/current/group__variables.html#gac7e8662c51f3bb07d1fc6d6c6d9052c8
   std::vector<real_t> coordx(num_nodes);
   std::vector<real_t> coordy(num_nodes);
   std::vector<real_t> coordz(mesh.Dimension() == 3 ? num_nodes : 0);

   ExtractVertexCoordinates(coordx, coordy, coordz);

   // 4. Define and put the nodal coordinates.
   DefineAndPutVar(ExodusIILabels::EXODUS_COORDX_LABEL, MFEM_NETCDF_REAL_T, 1,
                   &num_nodes_id,
                   coordx.data());
   DefineAndPutVar(ExodusIILabels::EXODUS_COORDY_LABEL, MFEM_NETCDF_REAL_T, 1,
                   &num_nodes_id,
                   coordy.data());

   if (mesh.Dimension() == 3)
   {
      DefineAndPutVar(ExodusIILabels::EXODUS_COORDZ_LABEL, MFEM_NETCDF_REAL_T, 1,
                      &num_nodes_id,
                      coordz.data());
   }
}

void ExodusIIWriter::WriteBoundaries()
{
   // 1. Generate boundary info.
   GenerateExodusIIBoundaryInfo();

   // 2. Define the number of boundaries.
   int num_side_sets_ids;
   DefineDimension(ExodusIILabels::EXODUS_NUM_SIDE_SETS_LABEL,
                   boundary_ids.size(),
                   &num_side_sets_ids);

   // 3. Boundary IDs.
   int boundary_ids_dim;
   DefineDimension(ExodusIILabels::EXODUS_NUM_BOUNDARIES_LABEL,
                   boundary_ids.size(),
                   &boundary_ids_dim);

   DefineAndPutVar(ExodusIILabels::EXODUS_SIDE_SET_IDS_LABEL, NC_INT, 1,
                   &boundary_ids_dim,
                   boundary_ids.data());

   // 4. Number of boundary elements.
   for (int boundary_id : boundary_ids)
   {
      size_t num_elements_for_boundary = exodusII_element_ids_for_boundary_id.at(
                                            boundary_id).size();

      char * label = GenerateLabel("num_side_ss%d", boundary_id);

      int num_side_ss_id;
      DefineDimension(label, num_elements_for_boundary,
                      &num_side_ss_id);
   }

   // 5. Boundary side IDs.
   for (int boundary_id : boundary_ids)
   {
      const std::vector<int> & side_ids = exodusII_side_ids_for_boundary_id.at(
                                             boundary_id);

      char * label = GenerateLabel("side_ss%d_dim", boundary_id);

      int side_id_dim;
      DefineDimension(label, side_ids.size(), &side_id_dim);

      label = GenerateLabel("side_ss%d", boundary_id);
      DefineAndPutVar(label, NC_INT, 1,  &side_id_dim, side_ids.data());
   }

   // 6. Boundary element IDs.
   for (int boundary_id : boundary_ids)
   {
      const std::vector<int> & element_ids = exodusII_element_ids_for_boundary_id.at(
                                                boundary_id);

      char * label = GenerateLabel("elem_ss%d_dim", boundary_id);

      int elem_ids_dim;
      DefineDimension(label, element_ids.size(), &elem_ids_dim);

      label = GenerateLabel("elem_ss%d", boundary_id);
      DefineAndPutVar(label, NC_INT, 1, &elem_ids_dim,
                      element_ids.data());
   }
}

void ExodusIIWriter::WriteNodeConnectivityForBlock(const int block_id)
{
   std::vector<int> block_node_connectivity;

   int * node_ordering_map = nullptr;

   // Apply mappings to convert from MFEM --> ExodusII orderings.
   Element::Type block_type = element_type_for_block_id.at(block_id);

   switch (block_type)
   {
      case Element::Type::TETRAHEDRON:
         node_ordering_map = (int *)
                             ExodusIINodeOrderings::mfem_to_exodusII_node_ordering_tet10;
         break;
      case Element::Type::HEXAHEDRON:
         node_ordering_map = (int *)
                             ExodusIINodeOrderings::mfem_to_exodusII_node_ordering_hex27;
         break;
      case Element::Type::WEDGE:
         node_ordering_map = (int *)
                             ExodusIINodeOrderings::mfem_to_exodusII_node_ordering_wedge18;
         break;
      case Element::Type::PYRAMID:
         node_ordering_map = (int *)
                             ExodusIINodeOrderings::mfem_to_exodusII_node_ordering_pyramid14;
         break;
      default:
         MFEM_ABORT("Higher-order elements of type '" << block_type <<
                    "' are not supported.");
   }

   const FiniteElementSpace * fespace = mesh.GetNodalFESpace();

   Array<int> element_dofs;
   for (int element_id : element_ids_for_block_id.at(block_id))
   {
      if (fespace)
      {
         fespace->GetElementDofs(element_id, element_dofs);

         for (int j = 0; j < element_dofs.Size(); j++)
         {
            int dof_index = node_ordering_map[j] - 1;
            int dof = element_dofs[dof_index];

            block_node_connectivity.push_back(dof + 1);  // 1-based indexing.
         }
      }
      else
      {
         mesh.GetElementVertices(element_id, element_dofs);

         for (int vertex_id : element_dofs)
         {
            block_node_connectivity.push_back(vertex_id + 1);  // 1-based indexing.
         }
      }
   }

   char * label = GenerateLabel("connect%d_dim", block_id);

   int node_connectivity_dim;
   DefineDimension(label, block_node_connectivity.size(),
                   &node_connectivity_dim);

   // NB: 1 == vector!; name is arbitrary; NC_INT or NCINT64?
   label = GenerateLabel("connect%d", block_id);
   DefineAndPutVar(label, NC_INT, 1, &node_connectivity_dim,
                   block_node_connectivity.data());
}


void ExodusIIWriter::ExtractVertexCoordinates(std::vector<real_t> & coordx,
                                              std::vector<real_t> & coordy,
                                              std::vector<real_t> & coordz)
{
   if (mesh.GetNodes()) // Higher-order.
   {
      std::unordered_set<int> unordered_node_ids = GenerateUniqueNodeIDs();

      std::vector<int> sorted_node_ids(unordered_node_ids.size());
      sorted_node_ids.assign(unordered_node_ids.begin(), unordered_node_ids.end());
      std::sort(sorted_node_ids.begin(), sorted_node_ids.end());

      real_t coordinates[3];
      for (size_t i = 0; i < sorted_node_ids.size(); i++)
      {
         int node_id = sorted_node_ids[i];

         mesh.GetNode(node_id, coordinates);

         coordx[node_id] = coordinates[0];
         coordy[node_id] = coordinates[1];

         if (mesh.Dimension() == 3)
         {
            coordz[node_id] = coordinates[2];
         }
      }
   }
   else // First-order.
   {
      for (int ivertex = 0; ivertex < mesh.GetNV(); ivertex++)
      {
         real_t *coordinates = mesh.GetVertex(ivertex);

         coordx[ivertex] = coordinates[0];
         coordy[ivertex] = coordinates[1];

         if (mesh.Dimension() == 3)
         {
            coordz[ivertex] = coordinates[2];
         }
      }
   }
}

void ExodusIIWriter::WriteFileSize()
{
   // Store Exodus file size (normal==0; large==1). NB: coordinates specifed
   // separately as components for large file.
   const int file_size = 1;

   PutAtt(NC_GLOBAL, ExodusIILabels::EXODUS_FILE_SIZE_LABEL, NC_INT, 1,
          &file_size);
}

void ExodusIIWriter::WriteMeshDimension()
{
   int num_dim_id;
   DefineDimension(ExodusIILabels::EXODUS_NUM_DIM_LABEL, mesh.Dimension(),
                   &num_dim_id);
}

void ExodusIIWriter::WriteNodeSets()
{
   // Nodesets are not currently implemented; set to zero.
   int num_node_sets_ids;
   DefineDimension(ExodusIILabels::EXODUS_NUM_NODE_SETS_LABEL, 0,
                   &num_node_sets_ids);
}

void ExodusIIWriter::WriteTimesteps()
{
   // Set number of timesteps (ASSUME single timestep for initial verision).
   int timesteps_dim;
   DefineDimension(ExodusIILabels::EXODUS_TIME_STEP_LABEL, 1, &timesteps_dim);
}

void ExodusIIWriter::WriteDummyVariable()
{
   int dummy_var_dim_id, dummy_value = 1;

   DefineDimension("dummy_var_dim", 1, &dummy_var_dim_id);

   DefineAndPutVar("dummy_var", NC_INT, 1, &dummy_var_dim_id,
                   &dummy_value);
}

void ExodusIIWriter::GenerateExodusIIElementBlocks()
{
   block_ids.clear();
   element_ids_for_block_id.clear();
   element_type_for_block_id.clear();

   std::unordered_set<int> observed_block_ids;

   // Iterate over the elements in the mesh.
   for (int ielement = 0; ielement < mesh.GetNE(); ielement++)
   {
      Element::Type element_type = mesh.GetElementType(ielement);

      int block_id = mesh.GetAttribute(ielement);

      if (observed_block_ids.count(block_id) == 0)
      {
         block_ids.push_back(block_id);

         element_type_for_block_id[block_id] = element_type;
         element_ids_for_block_id[block_id] = { ielement };

         observed_block_ids.insert(block_id);
      }
      else
      {
         auto & block_element_ids = element_ids_for_block_id.at(block_id);
         block_element_ids.push_back(ielement);

         // Safety check: ensure that the element type matches what we have on record
         // for the block.
         if (element_type != element_type_for_block_id.at(block_id))
         {
            MFEM_ABORT("Multiple element types are defined for block: " << block_id);
         }
      }
   }
}

void ExodusIIWriter::WriteNumElementBlocks()
{
   int num_elem_blk_id;
   DefineDimension(ExodusIILabels::EXODUS_NUM_ELEMENT_BLOCKS_LABEL,
                   block_ids.size(),
                   &num_elem_blk_id);
}


std::unordered_set<int> ExodusIIWriter::GenerateUniqueNodeIDs()
{
   std::unordered_set<int> unique_node_ids;

   const FiniteElementSpace * fespace = mesh.GetNodalFESpace();

   mfem::Array<int> element_dofs;
   for (int ielement = 0; ielement < mesh.GetNE(); ielement++)
   {
      if (fespace)   // Higher-order
      {
         fespace->GetElementDofs(ielement, element_dofs);
      }
      else
      {
         mesh.GetElementVertices(ielement, element_dofs);
      }

      for (int dof : element_dofs)
      {
         unique_node_ids.insert(dof);
      }
   }

   return unique_node_ids;
}

void ExodusIIWriter::GenerateExodusIIBoundaryInfo()
{
   // Store the unique boundary IDs.
   boundary_ids.clear();
   exodusII_element_ids_for_boundary_id.clear();
   exodusII_side_ids_for_boundary_id.clear();

   // Generate a mapping from the MFEM face index to the MFEM element ID.
   // Note that if we have multiple element IDs for a face index then the
   // face is shared between them and it cannot possibly be an external boundary
   // face since that can only have a single element associated with it. Therefore
   // we remove it from the array.
   struct GlobalFaceIndexInfo
   {
      int element_index;
      int local_face_index;

      GlobalFaceIndexInfo() : element_index{0}, local_face_index{0} {}

      GlobalFaceIndexInfo(int element_index, int local_face_index)
      {
         this->element_index = element_index;
         this->local_face_index = local_face_index;
      }
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

         if (mfem_face_index_info_for_global_face_index.count(face_index))
         {
            // Now we've seen it twice!
            blacklisted_global_face_indices.insert(face_index);
            mfem_face_index_info_for_global_face_index.erase(face_index);
            continue;
         }

         mfem_face_index_info_for_global_face_index[face_index] = GlobalFaceIndexInfo(
                                                                     ielement, iface);
      }
   }

   std::unordered_set<int> unique_boundary_attributes;

   for (int ibdr_element = 0; ibdr_element < mesh.GetNBE(); ibdr_element++)
   {
      int boundary_id = mesh.GetBdrAttribute(ibdr_element);
      int bdr_element_face_index = mesh.GetBdrElementFaceIndex(ibdr_element);

      // Skip any interior boundary faces.
      if (mesh.FaceIsInterior(bdr_element_face_index))
      {
         MFEM_WARNING("Skipping internal boundary " << ibdr_element);
         continue;
      }

      // Locate match.
      auto & element_face_info = mfem_face_index_info_for_global_face_index.at(
                                    bdr_element_face_index);

      int ielement = element_face_info.element_index;
      int iface = element_face_info.local_face_index;

      // 1. Convert MFEM 0-based element index to ExodusII 1-based element ID.
      int exodusII_element_id = ielement + 1;

      // 2. Convert MFEM 0-based face index to ExodusII 1-based face ID (different ordering).
      int exodusII_face_id;

      Element::Type element_type = mesh.GetElementType(ielement);
      switch (element_type)
      {
         case Element::Type::TETRAHEDRON:
            exodusII_face_id = ExodusIISideMaps::mfem_to_exodusII_side_map_tet4[iface];
            break;
         case Element::Type::HEXAHEDRON:
            exodusII_face_id = ExodusIISideMaps::mfem_to_exodusII_side_map_hex8[iface];
            break;
         case Element::Type::WEDGE:
            exodusII_face_id = ExodusIISideMaps::mfem_to_exodusII_side_map_wedge6[iface];
            break;
         case Element::Type::PYRAMID:
            exodusII_face_id = ExodusIISideMaps::mfem_to_exodusII_side_map_pyramid5[iface];
            break;
         default:
            MFEM_ABORT("Cannot handle element of type " << element_type);
      }

      unique_boundary_attributes.insert(boundary_id);

      exodusII_element_ids_for_boundary_id[boundary_id].push_back(
         exodusII_element_id);
      exodusII_side_ids_for_boundary_id[boundary_id].push_back(exodusII_face_id);
   }

   boundary_ids.assign(unique_boundary_attributes.begin(),
                       unique_boundary_attributes.end());
   std::sort(boundary_ids.begin(), boundary_ids.end());
}

void ExodusIIWriter::CheckNodalFESpaceIsSecondOrderH1() const
{
   const FiniteElementSpace * fespace = mesh.GetNodalFESpace();
   if (!fespace)  // Mesh does not have nodes.
   {
      MFEM_ABORT("The mesh has no nodal fespace.");
   }

   // Expect order 2.
   const int fespace_order = fespace->GetMaxElementOrder();
   if (fespace_order != 2)
   {
      MFEM_ABORT("Nodal fespace is of order " << fespace_order <<
                 ". Expected 2nd order.");
   }

   // Get a pointer to the FE collection associated with the fespace.
   const FiniteElementCollection * fec = fespace->FEColl();
   if (!fec)
   {
      MFEM_ABORT("No FECollection associated with nodal fespace.");
   }

   // Expect H1 FEC.
   if (strncmp(fec->Name(), "H1", 2) != 0)
   {
      MFEM_ABORT("Nodal fespace's FECollection is '" << fec->Name() <<
                 "'. Expected H1.");
   }
}

#endif

}
