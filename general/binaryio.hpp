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

/// Enum to specify if values should be read in binary or ASCII format.
enum BinaryOrASCII : bool
{
   ASCII = false,
   BINARY = true
};

/// Write 'value' to stream.
template<typename T>
inline void write(std::ostream& os, T value)
{
   os.write((char*) &value, sizeof(T));
}

/// Read a value from the stream and return it.
template<typename T>
inline T read(std::istream& is)
{
   T value;
   is.read((char*) &value, sizeof(T));
   return value;
}

/// Read a value of type @a T from a binary buffer and return it.
template <typename T>
inline T read(const char *buf)
{
   T value;
   std::copy(buf, buf + sizeof(T), reinterpret_cast<char*>(&value));
   return value;
}

/// Append the binary representation of @a val to the byte buffer @a vec.
template <typename T>
void AppendBytes(std::vector<char> &vec, const T &val)
{
   const char *ptr = reinterpret_cast<const char*>(&val);
   vec.insert(vec.end(), ptr, ptr + sizeof(T));
}

/// @brief Given a buffer @a bytes of length @a nbytes, encode the data in
/// base-64 format, and write the encoded data to the output stream @a out.
void WriteBase64(std::ostream &out, const void *bytes, size_t nbytes);

/// @brief Decode @a len base-64 encoded characters in the buffer @a src, and
/// store the resulting decoded data in @a buf. @a buf will be resized as
/// needed.
void DecodeBase64(const char *src, size_t len, std::vector<char> &buf);

/// @brief Return the number of characters needed to encode @a nbytes in
/// base-64.
///
/// This is equal to 4*nbytes/3, rounded up to the nearest multiple of 4.
size_t NumBase64Chars(size_t nbytes);

/// @brief Read and return a value of type @a T from the input stream, in either
/// binary or ASCII format, depending on the value of @a binary.
template <typename T>
T ReadBinaryOrASCII(std::istream &input, BinaryOrASCII binary)
{
   if (binary)
   {
      return read<T>(input);
   }
   else
   {
      T val;
      input >> val;
      return val;
   }
}

/// @brief Skip @a num values of type @a T from the input stream, in either
/// binary or ASCII format, depending on the value of @a binary.
template <typename T>
void Skip(std::istream &input, int num, BinaryOrASCII binary)
{
   if (binary)
   {
      input.ignore(sizeof(T) * num);
   }
   else
   {
      for (int i = 0; i < num; ++i) { ReadBinaryOrASCII<T>(input, ASCII); }
   }
}

} // namespace mfem::bin_io

} // namespace mfem

#endif
