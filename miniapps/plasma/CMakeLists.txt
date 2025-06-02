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

add_mfem_miniapp(g_eqdsk_viewer
  MAIN g_eqdsk_viewer.cpp
  EXTRA_SOURCES g_eqdsk_data.cpp
  EXTRA_HEADERS g_eqdsk_data.hpp plasma.hpp ${MFEM_MINIAPPS_COMMON_HEADERS}
  LIBRARIES mfem mfem-common)
