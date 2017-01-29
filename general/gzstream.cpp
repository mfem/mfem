// ============================================================================
// gzstream, C++ iostream classes wrapping the zlib compression library.
// Copyright (C) 2001  Deepak Bandyopadhyay, Lutz Kettner
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
// ============================================================================
//
// File          : gzstream.C
// Revision      : $Revision: 1.7 $
// Revision_date : $Date: 2003/01/08 14:41:27 $
// Author(s)     : Deepak Bandyopadhyay, Lutz Kettner
//
// Standard streambuf implementation following Nicolai Josuttis, "The
// Standard C++ Library".
// ============================================================================

#include "../config/config.hpp"
#include "gzstream.hpp"
#include <fstream>
#include <cstring>

namespace mfem
{

#ifdef MFEM_USE_GZSTREAM
// ----------------------------------------------------------------------------
// Internal classes to implement gzstream. See header file for user classes.
// ----------------------------------------------------------------------------

// --------------------------------------
// class gzstreambuf:
// --------------------------------------

gzstreambuf* gzstreambuf::open(char const *name, char const *_mode)
{
   if (is_open())
   {
      return (gzstreambuf*)0;
   }
   file = gzopen(name, _mode);
   if (file == 0)
   {
      return (gzstreambuf*)0;
   }
   strncpy(mode, _mode, sizeof(mode)-1);
   opened = 1;
   return this;
}

gzstreambuf * gzstreambuf::close()
{
   if ( is_open())
   {
      sync();
      opened = 0;
      if ( gzclose( file) == Z_OK)
      {
         return this;
      }
   }
   return (gzstreambuf*)0;
}

int gzstreambuf::underflow()   // used for input buffer only
{
   if ( gptr() && ( gptr() < egptr()))
   {
      return * reinterpret_cast<unsigned char *>( gptr());
   }

   if ( ! strchr(mode,'r') || ! opened)
   {
      return EOF;
   }
   // Josuttis' implementation of inbuf
   int n_putback = gptr() - eback();
   if ( n_putback > 4)
   {
      n_putback = 4;
   }
   memcpy( buffer + (4 - n_putback), gptr() - n_putback, n_putback);

   int num = gzread( file, buffer+4, bufferSize-4);
   if (num <= 0) // ERROR or EOF
   {
      return EOF;
   }

   // reset buffer pointers
   setg( buffer + (4 - n_putback),   // beginning of putback area
         buffer + 4,                 // read position
         buffer + 4 + num);          // end of buffer

   // return next character
   return * reinterpret_cast<unsigned char *>( gptr());
}

int gzstreambuf::flush_buffer()
{
   // Separate the writing of the buffer from overflow() and
   // sync() operation.
   int w = pptr() - pbase();
   if ( gzwrite( file, pbase(), w) != w)
   {
      return EOF;
   }
   pbump( -w);
   return w;
}

int gzstreambuf::overflow( int c)   // used for output buffer only
{

   if ( ! (strchr(mode,'w') || strchr(mode,'a')) || ! opened)
   {
      return EOF;
   }
   if (c != EOF)
   {
      *pptr() = c;
      pbump(1);
   }
   if ( flush_buffer() == EOF)
   {
      return EOF;
   }
   return c;
}

int gzstreambuf::sync()
{
   // Changed to use flush_buffer() instead of overflow( EOF)
   // which caused improper behavior with std::endl and flush(),
   // bug reported by Vincent Ricard.
   if ( pptr() && pptr() > pbase())
   {
      if ( flush_buffer() == EOF)
      {
         return -1;
      }
   }
   return 0;
}

// --------------------------------------
// class gzstreambase:
// --------------------------------------

gzstreambase::gzstreambase(char const *name, char const *_mode)
{
   init( &buf);
   open(name, _mode);
}

gzstreambase::~gzstreambase()
{
   buf.close();
}

void gzstreambase::open(char const *name, char const *_mode)
{
   if ( ! buf.open( name, _mode))
   {
      clear( rdstate() | std::ios::badbit);
   }
}

void gzstreambase::close()
{
   if ( buf.is_open())
      if ( ! buf.close())
      {
         clear( rdstate() | std::ios::badbit);
      }
}

#endif // MFEM_USE_GZSTREAM


// static method
bool ifgzstream::maybe_gz(const char *fn)
{
   unsigned short byt = 0x0000;
   std::ifstream strm(fn,std::ios_base::binary|std::ios_base::in);
   strm.read(reinterpret_cast<char*>(&byt),2);
   if (byt==0x1f8b||byt==0x8b1f) { return true; }
   return false;
}

ifgzstream::ifgzstream(char const *name, char const *mode)
   : std::istream(0)
{
   bool err;
#ifdef MFEM_USE_GZSTREAM
   if (maybe_gz(name))
   {
      gzstreambuf *gzbuf = new gzstreambuf;
      err = gzbuf != gzbuf->open(name, mode);
      buf = gzbuf;
   }
   else
#endif
   {
      std::filebuf *fbuf = new std::filebuf;
      err = fbuf != fbuf->open(name, std::ios_base::in); // 'mode' is ignored
      buf = fbuf;
   }
   if (!err)
   {
      rdbuf(buf);
   }
   else
   {
      delete buf;
      buf = NULL;
      setstate(std::ios::failbit);
   }
}


// static class member, ofgzstream::default_mode
#ifdef MFEM_USE_GZSTREAM
char const *ofgzstream::default_mode = "zwb6";
#else
char const *ofgzstream::default_mode = "w";
#endif

ofgzstream::ofgzstream(char const *name, char const *mode)
   : std::ostream(0)
{
   bool err;
#ifdef MFEM_USE_GZSTREAM
   if (strchr(mode,'z'))
   {
      gzstreambuf *gzbuf = new gzstreambuf;
      err = gzbuf != gzbuf->open(name, mode);
      buf = gzbuf;
   }
   else
#endif
   {
      std::filebuf *fbuf = new std::filebuf;
      err = fbuf != fbuf->open(name, std::ios_base::out); // 'mode' is ignored
      buf = fbuf;
   }
   if (!err)
   {
      rdbuf(buf);
   }
   else
   {
      delete buf;
      buf = NULL;
      setstate(std::ios::failbit);
   }
}

} // namespace mfem
