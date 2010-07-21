// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_OSOCKSTREAM
#define MFEM_OSOCKSTREAM

#include <sstream>
using namespace std;

/** Data type for output socket stream class. The class is used as client
    to send data to a server on a specified port number. One object of the
    class can be used for one time send of data to the server. The user
    writes in the stream, as in any other output stream and when the data
    is ready to be send function send() has to be executed. Otherwise (if
    not executed) the destructor will send the data. */
class osockstream : public ostringstream
{
private:
   int portnum, sent;
   char machine[128];

   // return socket number;
   // (-1) for unknown machine;
   // (-2) for socket fail;
   // (-3) for connect fail.
   int call_socket();
   int send_through_socket(int s, const char *Buf, int size);

public:

   /** The constructor takes as input the name of the server and the
       port number through which the communication will take place. */
   osockstream(int port, const char *hostname);

   /** Send the current in the stream data to the server specified by
       name "hostname" (in the constructor) on port number "port".
       Return -1 if data has already been sent or 0 for success. */
   int send();

   /** Virtual destructor. If the data hasn't been sent it sends it. */
   ~osockstream();
};

#endif
