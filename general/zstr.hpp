//---------------------------------------------------------
// Copyright 2015 Ontario Institute for Cancer Research
// Written by Matei David (matei@cs.toronto.edu)
//---------------------------------------------------------

// Original version, https://github.com/mateidavid/zstr, distributed under MIT
// license. This file is a combination of the zstr.hpp and strict_fstream.hpp
// files in the original src/ directory with additional MFEM modifactions.

// The MIT License (MIT)
//
// Copyright (c) 2015 Matei David, Ontario Institute for Cancer Research
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Reference:
// http://stackoverflow.com/questions/14086417/how-to-write-custom-input-stream-in-c

#ifndef __ZSTR_HPP
#define __ZSTR_HPP

#include <cassert>
#include <fstream>
#include <sstream>
#include <cstring>
#include <string>

#ifdef MFEM_USE_ZLIB
#include <zlib.h>
#endif

// The section below is a modified content of the src/strict_fstream.hpp
// file from https://github.com/mateidavid/zstr.

/**
 * This namespace defines wrappers for std::ifstream, std::ofstream, and
 * std::fstream objects. The wrappers perform the following steps:
 * - check the open modes make sense
 * - check that the call to open() is successful
 * - (for input streams) check that the opened file is peek-able
 * - turn on the badbit in the exception mask
 */
namespace strict_fstream
{

/// Overload of error-reporting function, to enable use with VS.
/// Ref: http://stackoverflow.com/a/901316/717706
static std::string strerror()
{
   std::string buff(80, '\0');
#ifdef _WIN32
   if (strerror_s(&buff[0], buff.size(), errno) != 0)
   {
      buff = "Unknown error";
   }
#elif (_POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600) && ! _GNU_SOURCE || defined(__APPLE__)
   // XSI-compliant strerror_r()
   if (strerror_r(errno, &buff[0], buff.size()) != 0)
   {
      buff = "Unknown error";
   }
#else
   // GNU-specific strerror_r()
   auto p = strerror_r(errno, &buff[0], buff.size());
   std::string tmp(p, std::strlen(p));
   std::swap(buff, tmp);
#endif
   buff.resize(buff.find('\0'));
   return buff;
}

/// Exception class thrown by failed operations.
class Exception
   : public std::exception
{
public:
   Exception(const std::string& msg) : _msg(msg) {}
   const char * what() const noexcept { return _msg.c_str(); }
private:
   std::string _msg;
}; // class Exception

namespace detail
{

struct static_method_holder
{
   static std::string mode_to_string(std::ios_base::openmode mode)
   {
      static const int n_modes = 6;
      static const std::ios_base::openmode mode_val_v[n_modes] =
      {
         std::ios_base::in,
         std::ios_base::out,
         std::ios_base::app,
         std::ios_base::ate,
         std::ios_base::trunc,
         std::ios_base::binary
      };

      static const char * mode_name_v[n_modes] =
      {
         "in",
         "out",
         "app",
         "ate",
         "trunc",
         "binary"
      };
      std::string res;
      for (int i = 0; i < n_modes; ++i)
      {
         if (mode & mode_val_v[i])
         {
            res += (! res.empty()? "|" : "");
            res += mode_name_v[i];
         }
      }
      if (res.empty()) { res = "none"; }
      return res;
   }
   static void check_mode(const std::string& filename,
                          std::ios_base::openmode mode)
   {
      if ((mode & std::ios_base::trunc) && ! (mode & std::ios_base::out))
      {
         throw Exception(std::string("strict_fstream: open('") + filename +
                         "'): mode error: trunc and not out");
      }
      else if ((mode & std::ios_base::app) && ! (mode & std::ios_base::out))
      {
         throw Exception(std::string("strict_fstream: open('") + filename +
                         "'): mode error: app and not out");
      }
      else if ((mode & std::ios_base::trunc) && (mode & std::ios_base::app))
      {
         throw Exception(std::string("strict_fstream: open('") + filename +
                         "'): mode error: trunc and app");
      }
   }
   static void check_open(std::ios * s_p, const std::string& filename,
                          std::ios_base::openmode mode)
   {
      if (s_p->fail())
      {
         throw Exception(std::string("strict_fstream: open('")
                         + filename + "'," + mode_to_string(mode) + "): open failed: "
                         + strerror());
      }
   }
   static void check_peek(std::istream * is_p, const std::string& filename,
                          std::ios_base::openmode mode)
   {
      bool peek_failed = true;
      try
      {
         is_p->peek();
         peek_failed = is_p->fail();
      }
      catch (std::ios_base::failure &e) {}
      if (peek_failed)
      {
         throw Exception(std::string("strict_fstream: open('")
                         + filename + "'," + mode_to_string(mode) + "): peek failed: "
                         + strerror());
      }
      is_p->clear();
   }
}; // struct static_method_holder

} // namespace detail

class ifstream
   : public std::ifstream
{
public:
   ifstream() = default;
   ifstream(const std::string& filename,
            std::ios_base::openmode mode = std::ios_base::in)
   {
      open(filename, mode);
   }
   void open(const std::string& filename,
             std::ios_base::openmode mode = std::ios_base::in)
   {
      mode |= std::ios_base::in;
      exceptions(std::ios_base::badbit);
      detail::static_method_holder::check_mode(filename, mode);
      std::ifstream::open(filename, mode);
      detail::static_method_holder::check_open(this, filename, mode);
      detail::static_method_holder::check_peek(this, filename, mode);
   }
}; // class ifstream

class ofstream
   : public std::ofstream
{
public:
   ofstream() = default;
   ofstream(const std::string& filename,
            std::ios_base::openmode mode = std::ios_base::out)
   {
      open(filename, mode);
   }
   void open(const std::string& filename,
             std::ios_base::openmode mode = std::ios_base::out)
   {
      mode |= std::ios_base::out;
      exceptions(std::ios_base::badbit);
      detail::static_method_holder::check_mode(filename, mode);
      std::ofstream::open(filename, mode);
      detail::static_method_holder::check_open(this, filename, mode);
   }
}; // class ofstream

class fstream
   : public std::fstream
{
public:
   fstream() = default;
   fstream(const std::string& filename,
           std::ios_base::openmode mode = std::ios_base::in)
   {
      open(filename, mode);
   }
   void open(const std::string& filename,
             std::ios_base::openmode mode = std::ios_base::in)
   {
      if (! (mode & std::ios_base::out)) { mode |= std::ios_base::in; }
      exceptions(std::ios_base::badbit);
      detail::static_method_holder::check_mode(filename, mode);
      std::fstream::open(filename, mode);
      detail::static_method_holder::check_open(this, filename, mode);
      detail::static_method_holder::check_peek(this, filename, mode);
   }
}; // class fstream

} // namespace strict_fstream


// The section below is a modified content of the src/zstr.hpp file from
// https://github.com/mateidavid/zstr.

namespace zstr
{
#ifdef MFEM_USE_ZLIB
/// Exception class thrown by failed zlib operations.
class Exception
   : public std::exception
{
public:
   Exception(z_stream *zstrm_p, int ret)
      : _msg("zlib: ")
   {
      switch (ret)
      {
         case Z_STREAM_ERROR:
            _msg += "Z_STREAM_ERROR: ";
            break;
         case Z_DATA_ERROR:
            _msg += "Z_DATA_ERROR: ";
            break;
         case Z_MEM_ERROR:
            _msg += "Z_MEM_ERROR: ";
            break;
         case Z_VERSION_ERROR:
            _msg += "Z_VERSION_ERROR: ";
            break;
         case Z_BUF_ERROR:
            _msg += "Z_BUF_ERROR: ";
            break;
         default:
            std::ostringstream oss;
            oss << ret;
            _msg += "[" + oss.str() + "]: ";
            break;
      }
      _msg += zstrm_p->msg;
   }
   Exception(const std::string msg) : _msg(msg) {}
   const char *what() const noexcept { return _msg.c_str(); }

private:
   std::string _msg;
}; // class Exception
#endif

#ifdef MFEM_USE_ZLIB
namespace detail
{
class z_stream_wrapper
   : public z_stream
{
public:
   z_stream_wrapper(bool _is_input = true, int _level = Z_DEFAULT_COMPRESSION)
      : is_input(_is_input)
   {
      this->zalloc = Z_NULL;
      this->zfree = Z_NULL;
      this->opaque = Z_NULL;
      int ret;
      if (is_input)
      {
         this->avail_in = 0;
         this->next_in = Z_NULL;
         ret = inflateInit2(this, 15 + 32);
      }
      else
      {
         ret = deflateInit2(this, _level, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY);
      }
      if (ret != Z_OK)
      {
         throw Exception(this, ret);
      }
   }
   ~z_stream_wrapper()
   {
      if (is_input)
      {
         inflateEnd(this);
      }
      else
      {
         deflateEnd(this);
      }
   }

private:
   bool is_input;
}; // class z_stream_wrapper

} // namespace detail

class istreambuf
   : public std::streambuf
{
public:
   istreambuf(std::streambuf *_sbuf_p,
              std::size_t _buff_size = default_buff_size, bool _auto_detect = true)
      : sbuf_p(_sbuf_p),
        zstrm_p(nullptr),
        buff_size(_buff_size),
        auto_detect(_auto_detect),
        auto_detect_run(false),
        is_text(false)
   {
      assert(sbuf_p);
      in_buff = new char[buff_size];
      in_buff_start = in_buff;
      in_buff_end = in_buff;
      out_buff = new char[buff_size];
      setg(out_buff, out_buff, out_buff);
   }

   istreambuf(const istreambuf &) = delete;
   istreambuf(istreambuf &&) = default;
   istreambuf &operator=(const istreambuf &) = delete;
   istreambuf &operator=(istreambuf &&) = default;

   virtual ~istreambuf()
   {
      delete[] in_buff;
      delete[] out_buff;
      if (zstrm_p)
      {
         delete zstrm_p;
      }
   }

   virtual std::streambuf::int_type underflow()
   {
      if (this->gptr() == this->egptr())
      {
         // pointers for free region in output buffer
         char *out_buff_free_start = out_buff;
         do
         {
            // read more input if none available
            if (in_buff_start == in_buff_end)
            {
               // empty input buffer: refill from the start
               in_buff_start = in_buff;
               std::streamsize sz = sbuf_p->sgetn(in_buff, buff_size);
               in_buff_end = in_buff + sz;
               if (in_buff_end == in_buff_start)
               {
                  break;
               } // end of input
            }
            // auto detect if the stream contains text or deflate data
            if (auto_detect && !auto_detect_run)
            {
               auto_detect_run = true;
               unsigned char b0 = *reinterpret_cast<unsigned char *>(in_buff_start);
               unsigned char b1 = *reinterpret_cast<unsigned char *>(in_buff_start + 1);
               // Ref:
               // http://en.wikipedia.org/wiki/Gzip
               // http://stackoverflow.com/questions/9050260/what-does-a-zlib-header-look-like
               is_text = !(in_buff_start + 2 <= in_buff_end && ((b0 == 0x1F &&
                                                                 b1 == 0x8B)                  // gzip header
                                                                || (b0 == 0x78 && (b1 == 0x01 // zlib header
                                                                                   || b1 == 0x9C || b1 == 0xDA))));
            }
            if (is_text)
            {
               // simply swap in_buff and out_buff, and adjust pointers
               assert(in_buff_start == in_buff);
               std::swap(in_buff, out_buff);
               out_buff_free_start = in_buff_end;
               in_buff_start = in_buff;
               in_buff_end = in_buff;
            }
            else
            {
               // run inflate() on input
               if (!zstrm_p)
               {
                  zstrm_p = new detail::z_stream_wrapper(true);
               }
               zstrm_p->next_in = reinterpret_cast<decltype(zstrm_p->next_in)>(in_buff_start);
               zstrm_p->avail_in = in_buff_end - in_buff_start;
               zstrm_p->next_out = reinterpret_cast<decltype(zstrm_p->next_out)>
                                   (out_buff_free_start);
               zstrm_p->avail_out = (out_buff + buff_size) - out_buff_free_start;
               int ret = inflate(zstrm_p, Z_NO_FLUSH);
               // process return code
               if (ret != Z_OK && ret != Z_STREAM_END)
               {
                  throw Exception(zstrm_p, ret);
               }
               // update in&out pointers following inflate()
               in_buff_start = reinterpret_cast<decltype(in_buff_start)>(zstrm_p->next_in);
               in_buff_end = in_buff_start + zstrm_p->avail_in;
               out_buff_free_start = reinterpret_cast<decltype(out_buff_free_start)>
                                     (zstrm_p->next_out);
               assert(out_buff_free_start + zstrm_p->avail_out == out_buff + buff_size);
               // if stream ended, deallocate inflator
               if (ret == Z_STREAM_END)
               {
                  delete zstrm_p;
                  zstrm_p = nullptr;
               }
            }
         }
         while (out_buff_free_start == out_buff);
         // 2 exit conditions:
         // - end of input: there might or might not be output available
         // - out_buff_free_start != out_buff: output available
         this->setg(out_buff, out_buff, out_buff_free_start);
      }
      return this->gptr() == this->egptr()
             ? traits_type::eof()
             : traits_type::to_int_type(*this->gptr());
   }

private:
   std::streambuf *sbuf_p;
   char *in_buff;
   char *in_buff_start;
   char *in_buff_end;
   char *out_buff;
   detail::z_stream_wrapper *zstrm_p;
   std::size_t buff_size;
   bool auto_detect;
   bool auto_detect_run;
   bool is_text;

   static const std::size_t default_buff_size = (std::size_t)1 << 20;
}; // class istreambuf

class ostreambuf
   : public std::streambuf
{
public:
   ostreambuf(std::streambuf *_sbuf_p,
              std::size_t _buff_size = default_buff_size, int _level = Z_DEFAULT_COMPRESSION)
      : sbuf_p(_sbuf_p),
        zstrm_p(new detail::z_stream_wrapper(false, _level)),
        buff_size(_buff_size)
   {
      assert(sbuf_p);
      in_buff = new char[buff_size];
      out_buff = new char[buff_size];
      setp(in_buff, in_buff + buff_size);
   }

   ostreambuf(const ostreambuf &) = delete;
   ostreambuf(ostreambuf &&) = default;
   ostreambuf &operator=(const ostreambuf &) = delete;
   ostreambuf &operator=(ostreambuf &&) = default;

   int deflate_loop(int flush)
   {
      while (true)
      {
         zstrm_p->next_out = reinterpret_cast<decltype(zstrm_p->next_out)>(out_buff);
         zstrm_p->avail_out = buff_size;
         int ret = deflate(zstrm_p, flush);
         if (ret != Z_OK && ret != Z_STREAM_END && ret != Z_BUF_ERROR)
         {
            throw Exception(zstrm_p, ret);
         }
         std::streamsize sz = sbuf_p->sputn(out_buff,
                                            reinterpret_cast<decltype(out_buff)>(zstrm_p->next_out) - out_buff);
         if (sz != reinterpret_cast<decltype(out_buff)>(zstrm_p->next_out) - out_buff)
         {
            // there was an error in the sink stream
            return -1;
         }
         if (ret == Z_STREAM_END || ret == Z_BUF_ERROR || sz == 0)
         {
            break;
         }
      }
      return 0;
   }

   virtual ~ostreambuf()
   {
      // flush the zlib stream
      //
      // NOTE: Errors here (sync() return value not 0) are ignored, because we
      // cannot throw in a destructor. This mirrors the behaviour of
      // std::basic_filebuf::~basic_filebuf(). To see an exception on error,
      // close the ofstream with an explicit call to close(), and do not rely
      // on the implicit call in the destructor.
      //
      sync();
      delete[] in_buff;
      delete[] out_buff;
      delete zstrm_p;
   }
   virtual std::streambuf::int_type overflow(std::streambuf::int_type c =
                                                traits_type::eof())
   {
      zstrm_p->next_in = reinterpret_cast<decltype(zstrm_p->next_in)>(pbase());
      zstrm_p->avail_in = pptr() - pbase();
      while (zstrm_p->avail_in > 0)
      {
         int r = deflate_loop(Z_NO_FLUSH);
         if (r != 0)
         {
            setp(nullptr, nullptr);
            return traits_type::eof();
         }
      }
      setp(in_buff, in_buff + buff_size);
      return traits_type::eq_int_type(c,
                                      traits_type::eof())
             ? traits_type::eof()
             : sputc(c);
   }
   virtual int sync()
   {
      // first, call overflow to clear in_buff
      overflow();
      if (!pptr())
      {
         return -1;
      }
      // then, call deflate asking to finish the zlib stream
      zstrm_p->next_in = nullptr;
      zstrm_p->avail_in = 0;
      if (deflate_loop(Z_FINISH) != 0)
      {
         return -1;
      }
      deflateReset(zstrm_p);
      return 0;
   }

private:
   std::streambuf *sbuf_p;
   char *in_buff;
   char *out_buff;
   detail::z_stream_wrapper *zstrm_p;
   std::size_t buff_size;

   static const std::size_t default_buff_size = (std::size_t)1 << 20;
}; // class ostreambuf

class istream
   : public std::istream
{
public:
   istream(std::istream &is)
      : std::istream(new istreambuf(is.rdbuf()))
   {
      exceptions(std::ios_base::badbit);
   }
   explicit istream(std::streambuf *sbuf_p)
      : std::istream(new istreambuf(sbuf_p))
   {
      exceptions(std::ios_base::badbit);
   }
   virtual ~istream()
   {
      delete rdbuf();
   }
}; // class istream

class ostream
   : public std::ostream
{
public:
   ostream(std::ostream &os)
      : std::ostream(new ostreambuf(os.rdbuf()))
   {
      exceptions(std::ios_base::badbit);
   }
   explicit ostream(std::streambuf *sbuf_p)
      : std::ostream(new ostreambuf(sbuf_p))
   {
      exceptions(std::ios_base::badbit);
   }
   virtual ~ostream()
   {
      delete rdbuf();
   }
}; // class ostream
#endif

namespace detail
{

template <typename FStream_Type>
struct strict_fstream_holder
{
   strict_fstream_holder(const std::string &filename,
                         std::ios_base::openmode mode = std::ios_base::in)
      : _fs(filename, mode)
   {
   }
   FStream_Type _fs;
}; // class strict_fstream_holder

} // namespace detail

#ifdef MFEM_USE_ZLIB
class ifstream
   : private detail::strict_fstream_holder<strict_fstream::ifstream>,
     public std::istream
{
public:
   explicit ifstream(const std::string &filename,
                     std::ios_base::openmode mode = std::ios_base::in)
      : detail::strict_fstream_holder<strict_fstream::ifstream>(filename, mode),
        std::istream(new istreambuf(_fs.rdbuf()))
   {
      exceptions(std::ios_base::badbit);
   }
   virtual ~ifstream()
   {
      if (rdbuf())
      {
         delete rdbuf();
      }
   }
}; // class ifstream

class ofstream
   : private detail::strict_fstream_holder<strict_fstream::ofstream>,
     public std::ostream
{
public:
   explicit ofstream(const std::string &filename,
                     std::ios_base::openmode mode = std::ios_base::out)
      : detail::strict_fstream_holder<strict_fstream::ofstream>(filename,
                                                                mode | std::ios_base::binary),
        std::ostream(new ostreambuf(_fs.rdbuf()))
   {
      exceptions(std::ios_base::badbit);
   }
   virtual ~ofstream()
   {
      if (rdbuf())
      {
         delete rdbuf();
      }
   }
}; // class ofstream
#endif

} // namespace zstr


// The section below contains MFEM-specific additions.

namespace mfem
{

class ofgzstream
   : private zstr::detail::strict_fstream_holder<strict_fstream::ofstream>,
     public std::ostream
{
public:
   explicit ofgzstream(const std::string &filename,
                       bool compression = false)
      : zstr::detail::strict_fstream_holder<strict_fstream::ofstream>(filename,
                                                                      std::ios_base::binary),
        std::ostream(nullptr)
   {
#ifdef MFEM_USE_ZLIB
      if (compression)
      {
         strbuf = new zstr::ostreambuf(_fs.rdbuf());
         rdbuf(strbuf);
      }
      else
#endif
      {
         rdbuf(_fs.rdbuf());
      }
      exceptions(std::ios_base::badbit);
   }

   explicit ofgzstream(const std::string &filename,
                       char const *open_mode_chars)
      : zstr::detail::strict_fstream_holder<strict_fstream::ofstream>(filename,
                                                                      std::ios_base::binary),
        std::ostream(nullptr)
   {
#ifdef MFEM_USE_ZLIB
      // If open_mode_chars contains any combination of open mode chars
      // containing the 'z' char, compression is enabled. This preserves the
      // behavior of the old interface but ignores the choice of the compression
      // level (it is always set to 6).
      if (std::string(open_mode_chars).find('z') != std::string::npos)
      {
         strbuf = new zstr::ostreambuf(_fs.rdbuf());
         rdbuf(strbuf);
      }
      else
#endif
      {
         rdbuf(_fs.rdbuf());
      }
      exceptions(std::ios_base::badbit);
   }

   virtual ~ofgzstream()
   {
      delete strbuf;
   }

   std::streambuf *strbuf = nullptr;
};

class ifgzstream
   : private zstr::detail::strict_fstream_holder<strict_fstream::ifstream>,
     public std::istream
{
public:
   explicit ifgzstream(const std::string &filename)
      : zstr::detail::strict_fstream_holder<strict_fstream::ifstream>(filename,
                                                                      std::ios_base::in),
        std::istream(nullptr)
   {
#ifdef MFEM_USE_ZLIB
      strbuf = new zstr::istreambuf(_fs.rdbuf());
      rdbuf(strbuf);
#else
      rdbuf(_fs.rdbuf());
#endif
      exceptions(std::ios_base::badbit);
   }

   virtual ~ifgzstream()
   {
      delete strbuf;
   }

   std::streambuf *strbuf = nullptr;
};

/// Input file stream that remembers the input file name (useful for example
/// when reading NetCDF meshes) and supports optional zlib decompression.
class named_ifgzstream : public ifgzstream
{
public:
   named_ifgzstream(const std::string &mesh_name) : ifgzstream(mesh_name),
      filename(mesh_name) {}

   const std::string filename;
};

} // namespace mfem

#endif
