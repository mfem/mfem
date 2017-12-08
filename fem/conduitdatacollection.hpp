// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_CONDUITDATACOLLECTION
#define MFEM_CONDUITDATACOLLECTION

#include "../config/config.hpp"

#ifdef MFEM_USE_CONDUIT

#include "datacollection.hpp"
#include <conduit.hpp>

namespace mfem
{

/** @brief Data collection that uses the the Conduit mesh blueprint
    specification. */
/** ConduitDataCollection provides json, simple binary, and HDF5-based 
    file formats for visualization or restart.

    For more information, see:
    - LLNL conduit/blueprint library, https://github.com/LLNL/conduit
    - HDF5 library, https://support.hdfgroup.org/HDF5

    @note The ConduitDataCollection only wraps the mfem objects to
    save them and creates them on load, Conduit does not own any of the 
    data.

    The SidreDataCollection provides more features (for example it 
    help allocate and own data).

    @note QuadratureFunction%s (q-fields) are not supported.

    @warning This class is still _experimental_, meaning that in future
    releases, it may not be backward compatible, and the output files generated
    by the current version may become unreadable.
*/
   
/// Data collection with Conduit I/O routines
class ConduitDataCollection : public DataCollection
{
protected:
   
   // file name helpers
   
   /// Returns blueprint root file name for the current cycle
   std::string RootFileName();
      
   /// Returns the mesh file name for a given domain at the current cycle
   std::string MeshFileName(int domain_id,
                            const std::string &file_protocol="hdf5");

   /// Returns the mesh output directory for the current cycle
   std::string MeshDirectoryName();
   
   /// Returns the mesh file pattern for the current cycle
   std::string MeshFilePattern(const std::string &file_protocol="hdf5");

   // Helper functions for Save()
   void SaveRootFile(int num_domains,
                     const conduit::Node &n_mesh,
                     const std::string &file_protocol);

   void SaveMeshAndFields(int domain_id,
                          const conduit::Node &n_mesh,
                          const std::string &file_protocol);

   // Helper functions for Load()
   void LoadRootFile(conduit::Node &n_root_out);
   void LoadMeshAndFields(int domain_id,
                          const std::string &file_protocol);

   // holds currently active conduit relay i/o protocol
   std::string relay_protocol;
   

public:
   /// Constructor. The collection name is used when saving the data.
   /** If @a mesh_ is NULL, then the mesh can be set later by calling either
       SetMesh() or Load(). The latter works only in serial. */
   ConduitDataCollection(const std::string& collection_name,
                         Mesh *mesh = NULL);

   /// We will delete the mesh and fields if we own them
   virtual ~ConduitDataCollection();

   /// Set the Conduit relay i/o protocol to use
   /** Supported options: hdf5 (default), json, conduit_json, conduit_bin */
   void SetProtocol(const std::string &protocol);

   /// Save the collection and a Conduit blueprint root file
   virtual void Save();
   
   /// Load the collection based blueprint data
   virtual void Load(int cycle = 0);

private:

   /// Converts from MFEM element type enum to mesh bp element name
   static std::string ElementTypeToShapeName(Element::Type element_type);

   static mfem::Geometry::Type ShapeNameToGeomType(const std::string &shape_name);
   
   ////////
   // NOTE: These below could be public Mesh and GF members in the future
   ////////
   
   /// Describes a mesh using the mesh blueprint into the
   /// passed  conduit::Node ( Conduit does NOT own the data)
   static void MeshToBlueprintMesh(Mesh *m, conduit::Node &out);
   
   /// Describes a grid function using the mesh blueprint into the
   /// passed  conduit::Node ( Conduit does NOT own the data)
   static void GridFunctionToBlueprintField(GridFunction *gf, conduit::Node &out);

   /// Constructs and MFEM mesh from a Conduit Blueprint Description 
   static Mesh         *BlueprintMeshToMesh(const conduit::Node &n_mesh);
   /// Constructs and MFEM Grid Function from a Conduit Blueprint Description 
   static GridFunction *BlueprintFieldToGridFunction(Mesh *mesh,
                                                     const conduit::Node &n_field);

   ////////
   // NOTE: The above could be public Mesh and GF members in the future
   ////////
   
};


} // end namespace mfem

#endif

#endif
