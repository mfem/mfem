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

#ifndef MFEM_VERSION_HPP
#define MFEM_VERSION_HPP

namespace mfem
{

/// Return the MFEM version number as a single integer.
int GetVersion();

/// Return the MFEM major version number as an integer.
int GetVersionMajor();

/// Return the MFEM minor version number as an integer.
int GetVersionMinor();

/// Return the MFEM version patch number as an integer.
int GetVersionPatch();

/// Return the MFEM version number as a string.
const char *GetVersionStr();

/// Return the MFEM Git hash as a string.
const char *GetGitStr();

/// Return the MFEM configuration as a string.
const char *GetConfigStr();

} // namespace mfem

#endif
