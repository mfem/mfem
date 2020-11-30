// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
// Created on: Jan 7, 2020
// Author: William F Godoy godoywf@ornl.gov
// adios2: Adaptable Input/Output System https://github.com/ornladios/ADIOS2

#include "adios2datacollection.hpp"

#ifdef MFEM_USE_ADIOS2

namespace mfem
{

#ifdef MFEM_USE_MPI
ADIOS2DataCollection::ADIOS2DataCollection(MPI_Comm comm,
                                           const std::string& collection_name, Mesh* mesh,
                                           const std::string engine_type) : DataCollection(collection_name, mesh),
   stream( new adios2stream(name, adios2stream::openmode::out, comm, engine_type) )
{
   SetMesh(mesh);
}
#else
ADIOS2DataCollection::ADIOS2DataCollection(
   const std::string& collection_name, Mesh* mesh,
   const std::string engine_type): DataCollection(collection_name, mesh),
   stream( new adios2stream(name, adios2stream::openmode::out, engine_type) )
{
   SetMesh(mesh);
}
#endif

ADIOS2DataCollection::~ADIOS2DataCollection()
{
   stream->Close();
}

void ADIOS2DataCollection::Save()
{
   stream->BeginStep();

   // only save mesh once (moving mesh, not yet supported)
   if (stream->CurrentStep() == 0)
   {
      if (mesh == nullptr)
      {
         const std::string error_message =
            "MFEM ADIOS2DataCollection Save error: Mesh is null. Please call SetMesh before Save\n";
         mfem_error(error_message.c_str());
      }
      stream->Print(*mesh);
   }

   // reduce footprint
   if (myid == 0)
   {
      stream->SetTime(time);
      stream->SetCycle(cycle);
   }

   for (const auto& field : field_map)
   {
      const std::string& variable_name = field.first;
      field.second->Save(*stream.get(), variable_name);
   }

   stream->EndStep();
}

void ADIOS2DataCollection::SetParameter(const std::string key,
                                        const std::string value) noexcept
{
   stream->SetParameter(key, value);
}

void ADIOS2DataCollection::SetLevelsOfDetail(const int levels_of_detail)
noexcept
{
   stream->SetRefinementLevel(levels_of_detail);
}

} // namespace mfem

#endif // MFEM_USE_ADIOS2
