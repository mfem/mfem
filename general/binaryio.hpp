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

#ifndef MFEM_BINARYIO
#define MFEM_BINARYIO

#include "../config/config.hpp"

#include <iostream>
#include <vector>

namespace mfem
{

// binary I/O helpers

namespace bin_io
{

template<typename T>
inline void write(std::ostream& os, T value)
{
   os.write((char*) &value, sizeof(T));
}

template<typename T>
inline T read(std::istream& is)
{
   T value;
   is.read((char*) &value, sizeof(T));
   return value;
}

template <typename T>
void AppendBytes(std::vector<char> &vec, const T &val)
{
   const char *ptr = reinterpret_cast<const char*>(&val);
   vec.insert(vec.end(), ptr, ptr + sizeof(T));
}

void WriteBase64(std::ostream &out, const void *bytes, size_t length);

} // namespace mfem::bin_io

} // namespace mfem

#endif
