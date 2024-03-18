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

#ifdef MFEM_USE_NETCDF
#include "netcdf.h"
#endif

namespace mfem
{
/// Function Prototypes.
static void HandleNetCDFStatus(int status);


void Mesh::WriteExodusII(const std::string fpath)
{
   int status = NC_NOERR;
   int ncid  = 0;  // File descriptor.

   //
   // Open file
   //
   int flags = NC_CLOBBER; // Overwrite existing files.

   status = nc_create(fpath.c_str(), flags, &ncid);
   HandleNetCDFStatus(status);

   //
   // Add title.
   //
   const char *title = "MFEM mesh";
   status = nc_put_att_text(ncid, NC_GLOBAL, "title", strlen(title), title);
   HandleNetCDFStatus(status);

   //
   // Set dimension.
   //
   int num_dim_id;
   status = nc_def_dim(ncid, "num_dim", Dim, &num_dim_id);
   HandleNetCDFStatus(status);

   //
   // Set # elements.
   //
   int num_elem_id;
   status = nc_def_dim(ncid, "num_elem", NumOfElements, &num_elem_id);
   HandleNetCDFStatus(status);

   //
   // Close file
   //
   status = nc_close(ncid);
   HandleNetCDFStatus(status);

   mfem::out << "Mesh successfully written to Exodus II file" << std::endl;
}

/// @brief Aborts with description of error.
static void HandleNetCDFStatus(int status)
{
   if (status != NC_NOERR)
   {
      MFEM_ABORT("NetCDF error: " << nc_strerror(status));
   }
}



}
