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

#ifndef MFEM4_BUFFER_HPP
#define MFEM4_BUFFER_HPP

namespace mfem4
{

/** A memory block, optionally with a GPU mirror.
 */
class Buffer
{
public:
   Buffer() : host(NULL), device(NULL), size(0), own(false) {}

   /// Shallow, non-owning copy
   Buffer(const Buffer &other) { MakeRef(other); }

   ~Buffer() { Free(); }

   void* HostPtr() const { return host; }
   void* DevicePtr() const { return device; }

   int Size() const { return size; }

   void CopyToDevice();
   void CopyToHost();

protected:
   void* host;
   void* device;
   int size;
   bool own;
};


} // namespace mfem4

#endif // MFEM4_BUFFER_HPP
