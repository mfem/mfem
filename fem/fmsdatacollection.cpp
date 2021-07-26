// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../config/config.hpp"

#ifdef MFEM_USE_FMS

#include "fem.hpp"
#include "../general/text.hpp"

#include <fmsio.h>

#include <string>
#include <sstream>

namespace mfem
{

// class FMSDataCollection implementation

FMSDataCollection::FMSDataCollection(const std::string& coll_name,
                                     Mesh *mesh)
   : DataCollection(coll_name, mesh),
     fms_protocol("ascii")
{
   appendRankToFileName = false; // always include rank in file names
   cycle = 0;                   // always include cycle in directory names
}

#ifdef MFEM_USE_MPI

FMSDataCollection::FMSDataCollection(MPI_Comm comm,
                                     const std::string& coll_name,
                                     Mesh *mesh)
   : DataCollection(coll_name, mesh),
     fms_protocol("ascii")
{
   m_comm = comm;
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &num_procs);
   appendRankToFileName = true; // always include rank in file names
   cycle = 0;                   // always include cycle in directory names
}

#endif

FMSDataCollection::~FMSDataCollection()
{
   // empty
}

void FMSDataCollection::Save()
{
   // Convert this to FmsDataCollection.

   FmsDataCollection dc;
   if (DataCollectionToFmsDataCollection(this, &dc) == 0)
   {
      std::string root(RootFileName());
      int err = FmsIOWrite(root.c_str(), fms_protocol.c_str(), dc);
      FmsDataCollectionDestroy(&dc);
      if (err)
      {
         MFEM_ABORT("Error creating FMS file: " << root);
      }
   }
   else
   {
      MFEM_ABORT("Error converting data collection");
   }
}

void FMSDataCollection::Load(int cycle)
{
   DeleteAll();
   this->cycle = cycle;

   FmsDataCollection dc;
   std::string root(RootFileName());
   int err = FmsIORead(root.c_str(), fms_protocol.c_str(), &dc);

   if (err == 0)
   {
      DataCollection *mdc = nullptr;
      if (FmsDataCollectionToDataCollection(dc,&mdc) == 0)
      {
         // Tell the data collection we read that it does not own data.
         // We will steal its data.
         mdc->SetOwnData(false);

         SetCycle(mdc->GetCycle());
         SetTime(mdc->GetTime());
         SetTimeStep(mdc->GetTimeStep());
         name = mdc->GetCollectionName();

         // Set mdc's mesh as our mesh.
         SetMesh(mdc->GetMesh());

         // Set mdc's fields/qfields as ours.
         std::vector<std::string> names;
         for (const auto &pair : mdc->GetFieldMap())
         {
            names.push_back(pair.first);
            RegisterField(pair.first, pair.second);
         }
         for (const auto &name : names)
         {
            mdc->DeregisterField(name);
         }

         names.clear();
         for (const auto &pair : mdc->GetQFieldMap())
         {
            names.push_back(pair.first);
            RegisterQField(pair.first, pair.second);
         }
         for (const auto &name : names)
         {
            mdc->DeregisterField(name);
         }

         // Indicate that we own the data.
         SetOwnData(true);

         // Delete mdc. We stole its contents.
         delete mdc;
      }
      FmsDataCollectionDestroy(&dc);
   }
   else
   {
      MFEM_ABORT("Error reading data collection" << root);
   }
}

void FMSDataCollection::SetProtocol(const std::string &protocol)
{
   fms_protocol = protocol;
}

std::string FMSDataCollection::RootFileName()
{
   std::string res;
   if (pad_digits_cycle)
   {
      res = prefix_path + name + "_" +
            to_padded_string(cycle, pad_digits_cycle) +
            ".fms";
   }
   else
   {
      res = prefix_path + name + ".fms";
   }
   return res;
}

} // namespace mfem

#endif
