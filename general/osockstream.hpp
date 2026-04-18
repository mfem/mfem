// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_OSOCKSTREAM
#define MFEM_OSOCKSTREAM

#include "socketstream.hpp"

namespace mfem
{

/** Data type for output socket stream class. The class is used as client
    to send data to a server on a specified port number. One object of the
    class can be used for one time send of data to the server. The user
    writes in the stream, as in any other output stream and when the data
    is ready to be send function send() has to be executed. Otherwise (if
    not executed) the destructor will send the data.
    @deprecated This class is DEPRECATED. New code should use class
    @ref socketstream (see socketstream.hpp). */
class osockstream : public socketstream
{
public:

   /** The constructor takes as input the name of the server and the
       port number through which the communication will take place. */
   osockstream(int port, const char *hostname);

   /** Send the current in the stream data to the server specified by
       name "hostname" (in the constructor) on port number "port".
       Return -1 if data has already been sent or 0 for success. */
   int send() { (*this) << std::flush; return 0; }

   /** Virtual destructor. If the data hasn't been sent it sends it. */
   virtual ~osockstream() { }
};

}

#endif
