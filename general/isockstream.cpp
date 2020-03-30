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

#include "isockstream.hpp"
#include "globals.hpp"
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <errno.h>
#ifndef _WIN32
#include <netinet/in.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <unistd.h>
#else
#include <winsock.h>
typedef int ssize_t;
typedef int socklen_t;
#define close closesocket
// Link with ws2_32.lib
#pragma comment(lib, "ws2_32.lib")
#endif

using namespace std;

namespace mfem
{

isockstream::isockstream(int port)
{
   portnum = port;

   if ( (portID = establish()) < 0)
      mfem::out << "Server couldn't be established on port "
                << portnum << endl;
   Buf = NULL;
}

int isockstream::establish()
{
   // char myname[129];
   char   myname[] = "localhost";
   int    port;
   struct sockaddr_in sa;
   struct hostent *hp;

   memset(&sa, 0, sizeof(struct sockaddr_in));
   // gethostname(myname, 128);
   hp= gethostbyname(myname);

   if (hp == NULL)
   {
      mfem::err << "isockstream::establish(): gethostbyname() failed!\n"
                << "isockstream::establish(): gethostname() returned: '"
                << myname << "'" << endl;
      error = 1;
      return (-1);
   }

   sa.sin_family= hp->h_addrtype;
   sa.sin_port= htons(portnum);

   if ((port = socket(AF_INET, SOCK_STREAM, 0)) < 0)
   {
      mfem::err << "isockstream::establish(): socket() failed!" << endl;
      error = 2;
      return (-1);
   }

   int on=1;
   setsockopt(port, SOL_SOCKET, SO_REUSEADDR, (char *)(&on), sizeof(on));

   if (bind(port,(const sockaddr*)&sa,(socklen_t)sizeof(struct sockaddr_in)) < 0)
   {
      mfem::err << "isockstream::establish(): bind() failed!" << endl;
      close(port);
      error = 3;
      return (-1);
   }

   listen(port, 4);
   error = 0;
   return (port);
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
      mfem::out << "Server failed to accept connection." << endl;
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
      mfem::out << "Not all the data has been read" << endl;
   }
#ifdef DEBUG
   else
   {
      mfem::out << "Reading " << size << " bytes is successful" << endl;
   }
#endif
   Buf[size] = '\0';

   close(socketID);
   (*in) = new istringstream(Buf);
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
