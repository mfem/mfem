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

#ifndef MFEM_FMSDATACOLLECTION
#define MFEM_FMSDATACOLLECTION

#include "../config/config.hpp"

#ifdef MFEM_USE_FMS

#include "datacollection.hpp"
#include <fms.h>

namespace mfem
{

/** @brief Data collection that uses FMS. */
/** FMSDataCollection lets MFEM read/write data using FMS.

    For more information, see:
    - FMS project, https://ceed.exascaleproject.org/fms/
*/

/// Data collection with FMS I/O routines
class FMSDataCollection : public DataCollection
{
protected:
   // file name helpers

   /// Returns file name for the current cycle
   std::string RootFileName();

   // holds currently active i/o protocol
   std::string fms_protocol;

public:
   /// Constructor. The collection name is used when saving the data.
   /** If @a mesh is NULL, then the mesh can be set later by calling either
       SetMesh() or Load(). The latter works only in serial. */
   FMSDataCollection(const std::string& collection_name,
                     Mesh *mesh = NULL);
#ifdef MFEM_USE_MPI
   /// Construct a parallel FMSDataCollection.
   FMSDataCollection(MPI_Comm comm, const std::string& collection_name,
                     Mesh *mesh = NULL);
#endif

   /// We will delete the mesh and fields if we own them
   virtual ~FMSDataCollection();

   /// Set the FMS relay i/o protocol to use
   /** Supported options: ascii (default), json, yaml, hdf5 */
   void SetProtocol(const std::string &protocol);

   /// Save the collection and a FMS blueprint root file
   void Save() override;

   /// Load the collection based blueprint data
   void Load(int cycle = 0) override;
};

} // namespace mfem

#endif

#endif
