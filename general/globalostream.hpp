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
    NullOStream() : std::ostream( this ) {}
    NullBuffer* rdbuf() { return this; }
};

class GlobalOutputStream
{
private:
   GlobalOutputStream() {theStream = &std::cout; enabled = true;}
   GlobalOutputStream(const GlobalOutputStream&);
   void operator=(const GlobalOutputStream&);

   std::ostream *theStream;
   NullOStream nullStream;
   bool enabled;
public:
   static GlobalOutputStream& Get()
   {
      static GlobalOutputStream instance;
      return instance;
   }

   inline void SetStream(std::ostream *stream) {theStream = stream;}
   inline std::ostream& GetStream();
   inline void Enable() {enabled = true;}
   inline void Disable() {enabled = false;}
};


inline std::ostream& GlobalOutputStream::GetStream()
{
   if (enabled)
   {
      return *theStream;
   }
   else
   {
      return nullStream;
   }
}


inline static std::ostream &mfem_out()
{
   return GlobalOutputStream::Get().GetStream();
}

}

#endif