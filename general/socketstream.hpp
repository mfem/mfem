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

#ifndef MFEM_SOCKETSTREAM
#define MFEM_SOCKETSTREAM

#include "../config/config.hpp"
#include <iostream>

namespace mfem
{

class socketbuf : public std::streambuf
{
private:
   int socket_descriptor;
   static const int buflen = 1024;
   char ibuf[buflen], obuf[buflen];

public:
   socketbuf()
   {
      socket_descriptor = -1;
   }

   explicit socketbuf(int sd)
   {
      socket_descriptor = sd;
      setp(obuf, obuf + buflen);
   }

   socketbuf(const char hostname[], int port)
   {
      socket_descriptor = -1;
      open(hostname, port);
   }

   /** Attach a new socket descriptor to the socketbuf.
       Returns the old socket descriptor which is NOT closed. */
   int attach(int sd);

   int detach() { return attach(-1); }

   int open(const char hostname[], int port);

   int close();

   int getsocketdescriptor() { return socket_descriptor; }

   bool is_open() { return (socket_descriptor >= 0); }

   ~socketbuf() { close(); }

protected:
   virtual int sync();

   virtual int_type underflow();

   virtual int_type overflow(int_type c = traits_type::eof());

   virtual std::streamsize xsgetn(char_type *__s, std::streamsize __n);

   virtual std::streamsize xsputn(const char_type *__s, std::streamsize __n);
};


class socketstream : public std::iostream
{
private:
   socketbuf __buf;

public:
   socketstream() : std::iostream(&__buf) { }

   explicit socketstream(int s) : std::iostream(&__buf), __buf(s) { }

   socketstream(const char hostname[], int port)
      : std::iostream(&__buf) { open(hostname, port); }

   socketbuf *rdbuf() { return &__buf; }

   int open(const char hostname[], int port)
   {
      int err = __buf.open(hostname, port);
      if (err)
      {
         setstate(std::ios::failbit);
      }
      else
      {
         clear();
      }
      return err;
   }

   int close() { return __buf.close(); }

   bool is_open() { return __buf.is_open(); }

   virtual ~socketstream() { }
};


class socketserver
{
private:
   int listen_socket;

public:
   explicit socketserver(int port);

   bool good() { return (listen_socket >= 0); }

   int close();

   int accept(socketstream &sockstr);

   ~socketserver() { close(); }
};

}

#endif
