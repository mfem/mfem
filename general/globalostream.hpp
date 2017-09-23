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

#ifndef MFEM_GLOBAL_OUTPUT_STREAM
#define MFEM_GLOBAL_OUTPUT_STREAM

#include <iostream>
#include "communication.hpp"

namespace mfem
{

class NullBuffer : public std::streambuf
{
public:
   int overflow(int c) { return c; }
};

class NullOStream : private NullBuffer, public std::ostream
{
public:
   NullOStream() : std::ostream(this) {}
   NullBuffer* rdbuf() { return this; }
};

class WrappedOStream
{
private:
   std::ostream *theStream;
   NullOStream nullStream;
   bool enabled;
public:
   WrappedOStream() { theStream = &std::cout; Enable(); }
   WrappedOStream(std::ostream *stream) { theStream = stream; Enable(); }
   void SetStream(std::ostream *stream) { theStream = stream; }
   std::ostream& GetStream() { return *theStream; }
   inline void Enable();
   inline void Disable() { enabled = false; }
   template <typename T>
   inline std::ostream& operator<<(T val);
};

inline void WrappedOStream::Enable()
{
#ifdef MFEM_USE_MPI
   int rank;
   MPI_Comm_rank(global_mpi_comm, &world_rank);
   if (world_rank == 0)
   {
      enabled = true;
   }
   else
   {
      enabled = false;
   }
#else
   enabled = true;
#endif
}

template <typename T>
inline std::ostream& WrappedOStream::operator<<(T val)
{
   if (enabled)
   {
      (*theStream) << val;
      return *theStream;
   }

   nullStream << val;
   return nullStream;
}

extern WrappedOStream mout;
extern WrappedOStream merr;

}

#endif
