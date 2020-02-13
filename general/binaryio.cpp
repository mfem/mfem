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
#ifdef MFEM_USE_GZSTREAM
#include <vector>
#include <zlib.h>
#endif

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

void WriteEncodedCompressed(std::ostream &out, const void *bytes,
                            uint32_t nbytes, int compression_level)
{
   if (compression_level == 0)
   {
      // First write size of buffer (as uint32_t), encoded with base 64
      WriteBase64(out, &nbytes, sizeof(nbytes));
      // Then write all the bytes in the buffer, encoded with base 64
      WriteBase64(out, bytes, nbytes);
   }
   else
   {
#ifdef MFEM_USE_GZSTREAM
      MFEM_ASSERT(compression_level >= 0 && compression_level <= 9,
                  "Compression level must be between 0 and 9 (inclusive).");
      uLongf buf_sz = compressBound(nbytes);
      std::vector<unsigned char> buf(buf_sz);
      compress2(buf.data(), &buf_sz, static_cast<const Bytef *>(bytes), nbytes,
                compression_level);

      // Write the header
      std::vector<uint32_t> header(4);
      header[0] = 1; // number of blocks
      header[1] = nbytes; // uncompressed size
      header[2] = 0; // size of partial block
      header[3] = buf_sz; // compressed size
      WriteBase64(out, header.data(), header.size()*sizeof(uint32_t));
      // Write the compressed data
      WriteBase64(out, buf.data(), buf_sz);
#else
      MFEM_ABORT("MFEM must be compiled with gzstream support to output "
                 "compressed binary data.")
#endif
   }
}

} // namespace mfem::bin_io
} // namespace mfem
