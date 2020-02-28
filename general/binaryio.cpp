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
