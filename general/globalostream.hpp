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
#include "globalcomm.hpp"

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

class WrappedOStream : public std::ostream
{
private:
   std::ostream *theStream;
   NullOStream nullStream;
   bool enabled;
public:
   WrappedOStream() { theStream = &std::cout; Enable(); }
   WrappedOStream(std::ostream *stream) { theStream = stream; Enable(); }
   inline void SetStream(std::ostream *stream) { theStream = stream; }
   inline std::ostream& GetStream();
   inline void Enable()  { enabled = true; }
   inline void Disable() { enabled = false; }
   inline bool IsEnabled() {return enabled;}
};


inline std::ostream& WrappedOStream::GetStream() 
{ 
#ifdef MFEM_USE_MPI
   int world_rank;
   MPI_Comm_rank(MFEM_COMM_WORLD, &world_rank);
   if (enabled && world_rank == 0)
   {
      return *theStream;
   }
#else
   if (enabled)
   {
      return *theStream;
   }
#endif

   return nullStream;
}


template <typename T>
std::ostream& operator <<(WrappedOStream& wos, T const& value) 
{
    wos.GetStream() << value;
    return wos.GetStream();
}


extern WrappedOStream out;
extern WrappedOStream err;

}

#endif
