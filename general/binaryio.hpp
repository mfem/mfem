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
