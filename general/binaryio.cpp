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

#include "binaryio.hpp"
#include "error.hpp"

namespace mfem
{
namespace bin_io
{

static const char *b64str
   = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
     "abcdefghijklmnopqrstuvwxyz"
     "0123456789+/";

void WriteBase64(std::ostream &out, const void *bytes, size_t nbytes)
{
   const unsigned char *in = static_cast<const unsigned char *>(bytes);
   const unsigned char *end = in + nbytes;
   while (end - in >= 3)
   {
      out << b64str[in[0] >> 2];
      out << b64str[((in[0] & 0x03) << 4) | (in[1] >> 4)];
      out << b64str[((in[1] & 0x0f) << 2) | (in[2] >> 6)];
      out << b64str[in[2] & 0x3f];
      in += 3;
   }
   if (end - in > 0) // Padding
   {
      out << b64str[in[0] >> 2];
      if (end - in == 1)
      {
         out << b64str[(in[0] & 0x03) << 4];
         out << '=';
      }
      else // end - in == 2
      {
         out << b64str[((in[0] & 0x03) << 4) | (in[1] >> 4)];
         out << b64str[(in[1] & 0x0f) << 2];
      }
      out << '=';
   }
}

} // namespace mfem::bin_io
} // namespace mfem
