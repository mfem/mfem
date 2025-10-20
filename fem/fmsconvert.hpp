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

#ifndef FMS_CONVERT
#define FMS_CONVERT

#include "../config/config.hpp"
#include "datacollection.hpp"

#ifdef MFEM_USE_FMS
#include <fms.h>

namespace mfem
{

/** In-memory conversion of FMS data collection to an MFEM data collection.
    @param dc The FMS data collection to convert.
    @param[out] mfem_dc A pointer to a new MFEM DataCollection containing the
    FMS data.
    @return 0 on success; non-zero on failure.
*/
int FmsDataCollectionToDataCollection(FmsDataCollection dc,
                                      DataCollection **mfem_dc);

/** In-memory conversion of MFEM data collection to an FMS data collection.
    @param mfem_dc The MFEM data collection to convert.
    @param[out] dc A pointer to a new FmsDataCollection containing the MFEM
    data.
    @return 0 on success; non-zero on failure.
*/
int DataCollectionToFmsDataCollection(DataCollection *mfem_dc,
                                      FmsDataCollection *dc);

} // namespace mfem

#endif

#endif
