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

#ifndef MFEM_DATACOLLECTION
#define MFEM_DATACOLLECTION

#include "../config/config.hpp"
#include "gridfunc.hpp"
#include <string>
#include <map>

namespace mfem
{

/** A class for collecting finite element data that is part of the same
    simulation. Currently, this class groups together several grid functions
    (fields) and the mesh that they are defined on. */
class DataCollection
{
protected:
   /// Name of the collection, used as a directory name when saving
   std::string name;

   /// A path where the directory with results is saved
   std::string prefix_path;

   /// The fields and their names (used when saving)
   std::map<std::string, GridFunction*> field_map;
   /// The (common) mesh for the collected fields
   Mesh *mesh;

   /** Time cycle; for time-dependent simulations cycle >= 0, otherwise = -1.
       When cycle >= 0, it is appended to directory names. */
   int cycle;
   /// Physical time (for time-dependent simulations)
   double time;

   /// Serial or parallel run? If false, append rank (myid) to file names
   bool serial;
   /// MPI rank (in parallel)
   int myid;
   /// Number of MPI ranks (in parallel)
   int num_procs;

   /// Precision (number of digits) used for the text output of doubles
   int precision;
   /// Number of digits used for the cycle and MPI rank in filenames
   int pad_digits;

   /// Default value for precision
   static const int precision_default = 6;
   /// Default value for pad_digits
   static const int pad_digits_default = 6;

   /// Should the collection delete its mesh and fields
   bool own_data;

   /// Error state
   int error;

   /// Create an empty collection with the given name.
   DataCollection(const char *collection_name);
   /// Delete data owned by the DataCollection keeping field information
   void DeleteData();
   /// Delete data owned by the DataCollection including field information
   void DeleteAll();

   /// Save one field to disk, assuming the collection directory exists
   void SaveOneField(const std::map<std::string,GridFunction*>::iterator &it);

public:
   /// Initialize the collection with its name and Mesh.
   DataCollection(const char *collection_name, Mesh *_mesh);

   /// Add a grid function to the collection
   virtual void RegisterField(const char *field_name, GridFunction *gf);
   /** Get a pointer to a grid function in the collection. Returns NULL if
       'field_name' is not in the collection. */
   GridFunction *GetField(const char *field_name);
   /// Check if a grid function is part of the collection
   bool HasField(const char *name) { return field_map.count(name) == 1; }

   /// Get a pointer to the mesh in the collection
   Mesh *GetMesh() { return mesh; }
   /// Set/change the mesh associated with the collection
   virtual void SetMesh(Mesh *new_mesh);

   /// Set time cycle (for time-dependent simulations)
   void SetCycle(int c) { cycle = c; }
   /// Set physical time (for time-dependent simulations)
   void SetTime(double t) { time = t; }

   /// Get time cycle (for time-dependent simulations)
   int GetCycle() { return cycle; }
   /// Get physical time (for time-dependent simulations)
   double GetTime() { return time; }
   /// Get the name of the collection
   const char* GetCollectionName() { return name.c_str(); }
   /// Set the ownership of collection data
   void SetOwnData(bool o) { own_data = o; }

   /// Set the precision (number of digits) used for the text output of doubles
   void SetPrecision(int prec) { precision = prec; }
   /// Set the number of digits used for the cycle and MPI rank in filenames
   void SetPadDigits(int digits) { pad_digits = digits; }
   /// Set the path where the DataCollection will be saved.
   void SetPrefixPath(const char *prefix);

   /** Save the collection to disk. By default, everything is saved in a
       directory with name "collection_name" or "collection_name_cycle" for
       time-dependent simulations. */
   virtual void Save();
   /// Save the mesh, creating the collection directory.
   virtual void SaveMesh();
   /// Save one field, assuming the collection directory already exists.
   virtual void SaveField(const char *field_name);

   /// Delete the mesh and fields if owned by the collection
   virtual ~DataCollection();

   /// Errors returned by Error()
   enum { NO_ERROR = 0, READ_ERROR = 1, WRITE_ERROR = 2 };

   /// Get the current error state
   int Error() const { return error; }
   /// Reset the error state
   void ResetError(int err = NO_ERROR) { error = err; }
};


/// Helper class for VisIt visualization data
class VisItFieldInfo
{
public:
   std::string association;
   int num_components;
   VisItFieldInfo() { association = ""; num_components = 0; }
   VisItFieldInfo(std::string _association, int _num_components)
   { association = _association; num_components = _num_components; }
};

/// Data collection with VisIt I/O routines
class VisItDataCollection : public DataCollection
{
protected:
   // Additional data needed in the VisIt root file, which describes the mesh
   // and all the fields in the collection
   int spatial_dim, topo_dim;
   int visit_max_levels_of_detail;
   std::map<std::string, VisItFieldInfo> field_info_map;

   /// Prepare the VisIt root file in JSON format for the current collection
   std::string GetVisItRootString();
   /// Read in a VisIt root file in JSON format
   void ParseVisItRootString(std::string json);

   // Helper functions for Load()
   void LoadVisItRootFile(std::string root_name);
   void LoadMesh();
   void LoadFields();

public:
   /** Create an empty collection with the given name, that will be filled in
       later with the Load() function. Currently this only works in serial! */
   VisItDataCollection(const char *collection_name);
   /// Initialize the collection with its mesh, fill-in the extra VisIt data
   VisItDataCollection(const char *collection_name, Mesh *_mesh);

   /// Set/change the mesh associated with the collection
   virtual void SetMesh(Mesh *new_mesh);

   /// Add a grid function to the collection and update the root file
   virtual void RegisterField(const char *field_name, GridFunction *gf);

   /// Set VisIt parameter: maximum levels of detail for the MultiresControl
   void SetMaxLevelsOfDetail(int max_levels_of_detail);

   /** Delete all data owned by VisItDataCollection including field data
       information. */
   void DeleteAll();

   /// Save the collection and a VisIt root file
   virtual void Save();

   /// Save a VisIt root file for the collection
   void SaveRootFile();

   /// Load the collection based on its VisIt data (described in its root file)
   void Load(int _cycle = 0);

   /// We will delete the mesh and fields if we own them
   virtual ~VisItDataCollection() {}
};

}

#endif
