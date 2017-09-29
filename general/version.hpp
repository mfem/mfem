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

#ifndef MFEM_VERSION_HPP
#define MFEM_VERSION_HPP

namespace mfem
{

int get_version();
int get_version_major();
int get_version_minor();
int get_version_patch();

const char *get_version_str();
const char *get_config_str();

} // namespace mfem

#endif
