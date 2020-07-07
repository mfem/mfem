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

#ifndef MFEM_ADIOS2DATACOLLECTION
#define MFEM_ADIOS2DATACOLLECTION

#include "../config/config.hpp"

#ifdef MFEM_USE_ADIOS2

#include "../general/adios2stream.hpp"
#include "datacollection.hpp"

#include <memory>  // std::unique_ptr
#include <string>

namespace mfem
{

class ADIOS2DataCollection : public DataCollection
{
public:
#ifdef MFEM_USE_MPI
   /**
    * Parallel constructor. Important: scope of this object must be within
    * MPI_Init and MPI_Finalize otherwise. The destructor will call the Close
    * function. Either object must live in a try/catch block (inside try) or use
    * raw pointers calling delete before MPI_Finalize.
    * @param comm MPI communicator setting the datacollection domain
    * @param collection_name unique name for saving data
    * @param mesh can be set at the constructor level or later by calling
    * SetMesh()
    * @param engine_type adios2 engine type
    */
   ADIOS2DataCollection(MPI_Comm comm, const std::string& collection_name,
                        Mesh* mesh = nullptr,
                        const std::string engine_type = "BPFile");
#else
   /**
    * Serial constructor
    * @param collection_name unique name for saving data
    * @param mesh can be set at the constructor level or later by calling
    * SetMesh()
    * @param engine_type adios2 engine type
    * @throws std::invalid_argument (user input error) or std::runtime_error
    *         (system error)
    */
   ADIOS2DataCollection(const std::string& collection_name, Mesh* mesh = nullptr,
                        const std::string engine_type = "BPFile");
#endif

   virtual ~ADIOS2DataCollection();

   /** Save the collection */
   virtual void Save();

   /**
    * Pass a parameter unique to adios2datacollection
    * For available parameters:
    * See https://adios2.readthedocs.io/en/latest/engines/engines.html
    * The most common is: key=SubStreams value=1 to nprocs (MPI processes)
    * @param key parameter key
    * @param value parameter value
    */
   void SetParameter(const std::string key, const std::string value) noexcept;

   /**
    * Sets the levels of detail for the global grid refinement
    * @param levels_of_detail (default = 1)
    */
   void SetLevelsOfDetail(const int levels_of_detail) noexcept;

private:
   std::unique_ptr<adios2stream> stream;
};

}  // namespace mfem

#endif // MFEM_USE_ADIOS2

#endif /* MFEM_ADIOS2DATACOLLECTION */
