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

#ifdef _WIN32
// Turn off CRT deprecation warnings for strerror (VS 2013)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "socketstream.hpp"

#include <cstring>      // memset, memcpy, strerror
#include <cerrno>       // errno
#ifndef _WIN32
#include <netdb.h>      // gethostbyname
#include <arpa/inet.h>  // htons
#include <sys/types.h>  // socket, setsockopt, connect, recv, send
#include <sys/socket.h> // socket, setsockopt, connect, recv, send
#include <unistd.h>     // close
#include <netinet/in.h> // sockaddr_in
#define closesocket (::close)
#else
#include <winsock.h>
typedef int ssize_t;
// Link with ws2_32.lib
#pragma comment(lib, "ws2_32.lib")
#endif

namespace mfem
{

int socketbuf::attach(int sd)
{
   int old_sd = socket_descriptor;
   pubsync();
   socket_descriptor = sd;
   setg(NULL, NULL, NULL);
   setp(obuf, obuf + buflen);
   return old_sd;
}

int socketbuf::open(const char hostname[], int port)
{
   struct sockaddr_in  sa;
   struct hostent     *hp;

   close();
   setg(NULL, NULL, NULL);
   setp(obuf, obuf + buflen);

   hp = gethostbyname(hostname);
   if (hp == NULL)
   {
      socket_descriptor = -3;
      return -1;
   }
   memset(&sa, 0, sizeof(sa));
   memcpy((char *)&sa.sin_addr, hp->h_addr, hp->h_length);
   sa.sin_family = hp->h_addrtype;
   sa.sin_port = htons(port);
   socket_descriptor = socket(hp->h_addrtype, SOCK_STREAM, 0);
   if (socket_descriptor < 0)
   {
      return -1;
   }

#if defined __APPLE__
   // OS X does not support the MSG_NOSIGNAL option of send().
   // Instead we can use the SO_NOSIGPIPE socket option.
   int on = 1;
   if (setsockopt(socket_descriptor, SOL_SOCKET, SO_NOSIGPIPE,
                  (char *)(&on), sizeof(on)) < 0)
   {
      closesocket(socket_descriptor);
      socket_descriptor = -2;
      return -1;
   }
#endif

   if (connect(socket_descriptor,
               (const struct sockaddr *)&sa, sizeof(sa)) < 0)
   {
      closesocket(socket_descriptor);
      socket_descriptor = -2;
      return -1;
   }
   return 0;
}

int socketbuf::close()
{
   if (is_open())
   {
      pubsync();
      int err = closesocket(socket_descriptor);
      socket_descriptor = -1;
      return err;
   }
   return 0;
}

int socketbuf::sync()
{
   ssize_t bw, n = pptr() - pbase();
   // std::cout << "[socketbuf::sync n=" << n << ']' << std::endl;
   while (n > 0)
   {
#ifdef MSG_NOSIGNAL
      bw = send(socket_descriptor, pptr() - n, n, MSG_NOSIGNAL);
#else
      bw = send(socket_descriptor, pptr() - n, n, 0);
#endif
      if (bw < 0)
      {
#ifdef MFEM_DEBUG
         std::cout << "Error in send(): " << strerror(errno) << std::endl;
#endif
         setp(pptr() - n, obuf + buflen);
         pbump(n);
         return -1;
      }
      n -= bw;
   }
   setp(obuf, obuf + buflen);
   return 0;
}

socketbuf::int_type socketbuf::underflow()
{
   // assuming (gptr() < egptr()) is false
   ssize_t br = recv(socket_descriptor, ibuf, buflen, 0);
   // std::cout << "[socketbuf::underflow br=" << br << ']'
   //           << std::endl;
   if (br <= 0)
   {
#ifdef MFEM_DEBUG
      if (br < 0)
      {
         std::cout << "Error in recv(): " << strerror(errno) << std::endl;
      }
#endif
      setg(NULL, NULL, NULL);
      return traits_type::eof();
   }
   setg(ibuf, ibuf, ibuf + br);
   return traits_type::to_int_type(*ibuf);
}

socketbuf::int_type socketbuf::overflow(int_type c)
{
   if (sync() < 0)
   {
      return traits_type::eof();
   }
   if (traits_type::eq_int_type(c, traits_type::eof()))
   {
      return traits_type::not_eof(c);
   }
   *pptr() = traits_type::to_char_type(c);
   pbump(1);
   return c;
}

std::streamsize socketbuf::xsgetn(char_type *__s, std::streamsize __n)
{
   // std::cout << "[socketbuf::xsgetn __n=" << __n << ']'
   //           << std::endl;
   const std::streamsize bn = egptr() - gptr();
   if (__n <= bn)
   {
      traits_type::copy(__s, gptr(), __n);
      gbump(__n);
      return __n;
   }
   traits_type::copy(__s, gptr(), bn);
   setg(NULL, NULL, NULL);
   std::streamsize remain = __n - bn;
   char_type *end = __s + __n;
   ssize_t br;
   while (remain > 0)
   {
      br = recv(socket_descriptor, end - remain, remain, 0);
      if (br <= 0)
      {
#ifdef MFEM_DEBUG
         if (br < 0)
         {
            std::cout << "Error in recv(): " << strerror(errno) << std::endl;
         }
#endif
         return (__n - remain);
      }
      remain -= br;
   }
   return __n;
}

std::streamsize socketbuf::xsputn(const char_type *__s, std::streamsize __n)
{
   // std::cout << "[socketbuf::xsputn __n=" << __n << ']'
   //           << std::endl;
   if (pptr() + __n <= epptr())
   {
      traits_type::copy(pptr(), __s, __n);
      pbump(__n);
      return __n;
   }
   if (sync() < 0)
   {
      return 0;
   }
   ssize_t bw;
   std::streamsize remain = __n;
   const char_type *end = __s + __n;
   while (remain > buflen)
   {
#ifdef MSG_NOSIGNAL
      bw = send(socket_descriptor, end - remain, remain, MSG_NOSIGNAL);
#else
      bw = send(socket_descriptor, end - remain, remain, 0);
#endif
      if (bw < 0)
      {
#ifdef MFEM_DEBUG
         std::cout << "Error in send(): " << strerror(errno) << std::endl;
#endif
         return (__n - remain);
      }
      remain -= bw;
   }
   if (remain > 0)
   {
      traits_type::copy(pptr(), end - remain, remain);
      pbump(remain);
   }
   return __n;
}


socketserver::socketserver(int port)
{
   listen_socket = socket(PF_INET, SOCK_STREAM, 0); // tcp socket
   if (listen_socket < 0)
   {
      return;
   }
   int on = 1;
   if (setsockopt(listen_socket, SOL_SOCKET, SO_REUSEADDR,
                  (char *)(&on), sizeof(on)) < 0)
   {
      closesocket(listen_socket);
      listen_socket = -2;
      return;
   }
   struct sockaddr_in sa;
   memset(&sa, 0, sizeof(sa));
   sa.sin_family = AF_INET;
   sa.sin_port = htons(port);
   sa.sin_addr.s_addr = INADDR_ANY;
   if (bind(listen_socket, (const struct sockaddr *)&sa, sizeof(sa)))
   {
      closesocket(listen_socket);
      listen_socket = -3;
      return;
   }
   const int backlog = 4;
   if (listen(listen_socket, backlog) < 0)
   {
      closesocket(listen_socket);
      listen_socket = -4;
      return;
   }
}

int socketserver::close()
{
   if (!good())
   {
      return 0;
   }
   int err = closesocket(listen_socket);
   listen_socket = -1;
   return err;
}

int socketserver::accept(socketstream &sockstr)
{
   if (!good())
   {
      return -1;
   }
   int socketd = ::accept(listen_socket, NULL, NULL);
   if (socketd >= 0)
   {
      sockstr.rdbuf()->close();
      sockstr.rdbuf()->attach(socketd);
   }
   return socketd;
}

}
