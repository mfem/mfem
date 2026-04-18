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

#ifndef MFEM_ISOCKSTREAM
#define MFEM_ISOCKSTREAM

#include "../config/config.hpp"
#include <sstream>

namespace mfem
{

/** Data type for input socket stream class. The class is used as server
    to receive data from a client on specified port number. The user gets
    data from the stream as from any other input stream.
    @deprecated This class is DEPRECATED. New code should use class
    @ref socketserver (see socketstream.hpp). */
class isockstream
{
private:
   int portnum, portID, socketID, error;
   char *Buf;

   int establish();
   int read_data(int socketid, char *buf, int size);

public:

   /** The constructor takes as input the port number on which it
       establishes a server. */
   explicit isockstream(int port);

   bool good() { return (!error); }

   /// Start waiting for data and return it in an input stream.
   void receive(std::istringstream **in);

   /** Virtual destructor. If the data hasn't been sent it sends it. */
   ~isockstream();
};

}

#endif
