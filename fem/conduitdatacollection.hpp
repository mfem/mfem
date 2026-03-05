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

#ifndef MFEM_CONDUITDATACOLLECTION
#define MFEM_CONDUITDATACOLLECTION

#include "../config/config.hpp"

#ifdef MFEM_USE_CONDUIT

#include "datacollection.hpp"
#include <conduit.hpp>

namespace mfem
{

/** @brief Data collection that uses the Conduit Mesh Blueprint
    specification. */
/** ConduitDataCollection provides json, simple binary, and HDF5-based file
    formats for visualization or restart. It also provides methods that convert
    between MFEM Meshes and GridFunctions and Conduit Mesh Blueprint
    descriptions.

    For more information, see:
    - LLNL conduit project, https://github.com/LLNL/conduit
    - HDF5 library, https://support.hdfgroup.org/HDF5

    @note The ConduitDataCollection only wraps the mfem objects to save them and
    creates them on load, Conduit does not own any of the data. The
    SidreDataCollection provides more features, for example the
    SidreDataCollection allocates and will own the data backing the mfem objects
    in the data collection.

    This class also provides public static methods that convert between MFEM
    Meshes and GridFunctions and Conduit Mesh Blueprint descriptions.

    Those that describe MFEM data using Conduit (MFEM to Conduit Blueprint) try
    to zero-copy as much of data MFEM as possible. The Conduit node result will
    not own all of the data, however you can easily make a copy of the result
    node using Conduit's API when necessary.

    Those that construct MFEM objects from Conduit Nodes (Conduit Blueprint to
    MFEM) provide a zero-copy option. Zero-copy is only possible if the
    blueprint data matches the data types provided by the MFEM API, for example:
    ints for connectivity arrays, real_t (double/float) for field value arrays,
    allocations that match MFEM's striding options, etc. If these constraints
    are not met, MFEM objects that own the data are created and returned. In
    either case pointers to new MFEM object instances are returned, the
    zero-copy only applies to data backing the MFEM object instances.

    @note QuadratureFunction%s (q-fields) are not supported.

    @note AMR Meshes are not fully supported.

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

   /// Saves root file for the current cycle
   void SaveRootFile(int num_domains,
                     const conduit::Node &n_mesh,
                     const std::string &file_protocol);

   /// Saves all meshes and fields for the current cycle
   void SaveMeshAndFields(int domain_id,
                          const conduit::Node &n_mesh,
                          const std::string &file_protocol);

   // Helper functions for Load()

   /// Loads contents of the root field for the current cycle into n_root_out
   void LoadRootFile(conduit::Node &n_root_out);

   /// Loads all meshes and fields of a given domain id for the current cycle
   void LoadMeshAndFields(int domain_id,
                          const std::string &file_protocol);

   // holds currently active conduit relay i/o protocol
   std::string relay_protocol;

public:
   /// Constructor. The collection name is used when saving the data.
   /** If @a mesh is NULL, then the mesh can be set later by calling either
       SetMesh() or Load(). The latter works only in serial. */
   ConduitDataCollection(const std::string& collection_name,
                         Mesh *mesh = NULL);
#ifdef MFEM_USE_MPI
   /// Construct a parallel ConduitDataCollection.
   ConduitDataCollection(MPI_Comm comm, const std::string& collection_name,
                         Mesh *mesh = NULL);
#endif

   /// We will delete the mesh and fields if we own them
   virtual ~ConduitDataCollection();

   /// Set the Conduit relay i/o protocol to use
   /** Supported options: hdf5 (default), json, conduit_json, conduit_bin */
   void SetProtocol(const std::string &protocol);

   /// Save the collection and a Conduit blueprint root file
   virtual void Save();

   /// Load the collection based blueprint data
   virtual void Load(int cycle = 0);

   /* Methods that convert to and from MFEM objects and Conduit Nodes that
      conform to the mesh blueprint.

      These methods could be public Mesh and GF members in the future.  One draw
      back is that would require use of MFEM_USE_CONDUIT in the mesh and
      gridfunc sources vs having everything contained in the
      conduitdatacollection sources.
   */

   /// Describes a MFEM mesh using the mesh blueprint
   /** Sets up passed conduit::Node to describe the given mesh using the mesh
       blueprint.

       Zero-copies as much data as possible.

       @a coordset_name, @a main_topology_name, and @a boundary_topology_name
       control the names used for the mesh blueprint entries.

       With the default set of names, this method describes the mesh's
       coordinates with a coordinate set entry named `coords`. Describes the
       mesh with a topology entry named 'main'.  If the mesh has nodes, these
       are described in a field entry named `mesh_nodes`. If the mesh has an
       attribute field, this is described in a field entry named
       `mesh_attribute`.

       If the mesh has boundary info, this is described in a topology entry
       named `boundary`. If the boundary has an attribute field, this is
       described in a field entry named `boundary_attribute`.
   */
   static void MeshToBlueprintMesh(Mesh *m,
                                   conduit::Node &out,
                                   const std::string &coordset_name = "coords",
                                   const std::string &main_topology_name = "main",
                                   const std::string &boundary_topology_name = "boundary",
                                   const std::string &main_adjset_name = "main_adjset");

   /// Describes a MFEM grid function using the mesh blueprint
   /** Sets up passed conduit::Node out to describe the given grid function
       using the mesh field blueprint.

       Zero-copies as much data as possible.

       @a main_toplogy_name is used to set the associated topology name.
       With the default setting, the resulting field is associated with the
       topology `main`.
   */
   static void GridFunctionToBlueprintField(GridFunction *gf,
                                            conduit::Node &out,
                                            const std::string &main_topology_name = "main");

   /// Describes a MFEM quadrature function using the mesh blueprint
   /** Sets up passed conduit::Node out to describe the given quadrature function
       using the mesh field blueprint.

       Zero-copies as much data as possible.

       @a main_toplogy_name is used to set the associated topology name.
       With the default setting, the resulting field is associated with the
       topology `main`.
   */
   static void QuadratureFunctionToBlueprintField(QuadratureFunction *qf,
                                                  conduit::Node &out,
                                                  const std::string &main_topology_name = "main");


   /// Constructs and MFEM mesh from a Conduit Blueprint Description
   /** @a main_topology_name is used to select which topology to use, when
       empty ("") the first topology entry will be used.

       If zero_copy == true, tries to construct a mesh that points to the data
       described by the conduit node. This is only possible if the data in the
       node matches the data types needed for the MFEM API (ints for
       connectivity, real_t for field values, etc). If these constraints are
       not met, a mesh that owns the data is created and returned.
   */
   static Mesh *BlueprintMeshToMesh(const conduit::Node &n_mesh,
                                    const std::string &main_toplogy_name = "",
                                    bool zero_copy = false);

   /// Constructs and MFEM Grid Function from a Conduit Blueprint Description
   /** If zero_copy == true, tries to construct a grid function that points to
       the data described by the conduit node. This is only possible if the data
       in the node matches the data types needed for the MFEM API (real_t for
       field values, allocated in soa or aos ordering, etc).  If these
       constraints are not met, a grid function that owns the data is created
       and returned.
   */
   static GridFunction *BlueprintFieldToGridFunction(Mesh *mesh,
                                                     const conduit::Node &n_field,
                                                     bool zero_copy = false);
   /// Constructs and MFEM Quadrature Function from a Conduit Blueprint Description
   /** If zero_copy == true, tries to construct a quadrature function that points to
       the data described by the conduit node. This is only possible if the data
       in the node matches the data types needed for the MFEM API (real_t for
       field values, allocated in an interleavred/byVDIM order, etc). If these
       constraints are not met, a grid function that owns the data is created
       and returned.
   */
   static QuadratureFunction *BlueprintFieldToQuadratureFunction(Mesh *mesh,
                                                                 const conduit::Node &n_field,
                                                                 bool zero_copy = false);

private:
   /// Converts from MFEM element type enum to mesh bp shape name
   static std::string ElementTypeToShapeName(Element::Type element_type);

   /// Converts a mesh bp shape name to a MFEM geom type
   static mfem::Geometry::Type ShapeNameToGeomType(const std::string &shape_name);
};

} // end namespace mfem

#endif

#endif
