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

/// Return the version number as a single integer.
int GetVersion();

/// Return the major version number as an integer.
int GetVersionMajor();

/// Return the minor version number as an integer.
int GetVersionMinor();

/// Return the version patch number as an integer.
int GetVersionPatch();

/// Return the version number as a string.
const char *GetVersionStr();

/// Return the Git hash as a string.
const char *GetGitStr();

/// Return the MFEM configuration as a string.
const char *GetConfigStr();

} // namespace mfem

#endif
