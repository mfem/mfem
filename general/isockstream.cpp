// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "isockstream.hpp"
#include "globals.hpp"
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <errno.h>
#ifndef _WIN32
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <unistd.h>
#else
#include <winsock2.h>
#include <ws2tcpip.h>
#ifdef _MSC_VER
typedef int ssize_t;
// Link with ws2_32.lib
#pragma comment(lib, "ws2_32.lib")
#endif
typedef int socklen_t;
#define close closesocket
#endif

namespace mfem
{

isockstream::isockstream(int port)
{
   portnum = port;

   if ( (portID = establish()) < 0)
      mfem::out << "Server couldn't be established on port "
                << portnum << std::endl;
   Buf = NULL;
}

int isockstream::establish()
{
   // char myname[129];
   char   myname[] = "localhost";
   int    sfd = -1;
   struct addrinfo hints, *res, *rp;

   memset(&hints, 0, sizeof(hints));
   hints.ai_family = AF_UNSPEC;
   hints.ai_socktype = SOCK_STREAM;
   hints.ai_protocol = 0;

   int s = getaddrinfo(myname, NULL, &hints, &res);
   if (s != 0)
   {
      mfem::err << "isockstream::establish(): getaddrinfo() failed!\n"
                << "isockstream::establish(): getaddrinfo() returned: '"
                << myname << "'" << std::endl;
      error = 1;
      return (-1);
   }

   // loop the list of address structures returned by getaddrinfo()
   for (rp = res; rp != NULL; rp = rp->ai_next)
   {
      if ((sfd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol)) < 0)
      {
         mfem::err << "isockstream::establish(): socket() failed!" << std::endl;
         error = 2;
         return (-1);
      }

      int on = 1;
      if (setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, (char *)&on, sizeof(on)) < 0)
      {
         mfem::err << "isockstream::establish(): setsockopt() failed!" << std::endl;
         return (-1);
      }

#if defined(__APPLE__)
      if (bind(sfd, (const struct sockaddr *)rp->ai_addr, rp->ai_addrlen) < 0)
#else
      if (bind(sfd, rp->ai_addr, static_cast<socklen_t>(rp->ai_addrlen)) < 0)
#endif
      {
         mfem::err << "isockstream::establish(): bind() failed!" << std::endl;
         close(sfd);
         error = 3;
         continue;
      }

      break;
   }

   // No address succeeded
   if (rp == NULL)
   {
      mfem::err << "Could not bind\n";
      return (-1);
   }

   freeaddrinfo(res);
   listen(sfd, 4);
   return (sfd);
}

int isockstream::read_data(int s, char *buf, int n)
{
   int bcount;                      // counts bytes read
   int br;                          // bytes read this pass

   bcount= 0;
   while (bcount < n)               // loop until full buffer
   {
      if ((br = recv(s, buf, n - bcount, 0)) > 0)
      {
         bcount += br;                // increment byte counter
         buf += br;                   // move buffer ptr for next read
      }
      else if (br < 0)               // signal an error to the caller
      {
         error = 4;
         return (-1);
      }
   }
   return (bcount);
}

void isockstream::receive(std::istringstream **in)
{
   int size;
   char length[32];

   if ((*in) != NULL)
   {
      delete (*in), *in = NULL;
   }

   if (portID == -1)
   {
      return;
   }

   if ((socketID = accept(portID, NULL, NULL)) < 0)
   {
      mfem::out << "Server failed to accept connection." << std::endl;
      error = 5;
      return;
   }

   if (recv(socketID, length, 32, 0) < 0)
   {
      error = 6;
      return;
   }
   size = atoi(length);

   if (Buf != NULL)
   {
      delete [] Buf;
   }
   Buf = new char[size+1];
   if (size != read_data(socketID, Buf, size))
   {
      mfem::out << "Not all the data has been read" << std::endl;
   }
#ifdef DEBUG
   else
   {
      mfem::out << "Reading " << size << " bytes is successful" << std::endl;
   }
#endif
   Buf[size] = '\0';

   close(socketID);
   (*in) = new std::istringstream(Buf);
}

isockstream::~isockstream()
{
   if (Buf != NULL)
   {
      delete [] Buf;
   }
   if (portID != -1)
   {
      close(portID);
   }
}

}
