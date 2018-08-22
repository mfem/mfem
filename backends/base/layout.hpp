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

#ifndef MFEM_BACKENDS_BASE_LAYOUT_HPP
#define MFEM_BACKENDS_BASE_LAYOUT_HPP

#include "../../config/config.hpp"
#ifdef MFEM_USE_BACKENDS

#include "smart_pointers.hpp"
#include "engine.hpp"

namespace mfem
{

/// Polymorphic layout (array/vector layout descriptor)
class PLayout : public RefCounted
{
protected:
   /// Engine with shared ownership
   SharedPtr<const Engine> engine;
   std::size_t size;

   template <typename DObject>
   struct Maker
   {
      template <typename entry_t>
      static DObject MakeNew(PLayout &layout);
   };

public:
   explicit PLayout(std::size_t s = 0) : engine(NULL), size(s) { }

   explicit PLayout(const Engine &e, std::size_t s = 0)
      : engine(&e), size(s) { }

   virtual ~PLayout() { }

   /**
       @name Virtual interface
    */
   ///@{

   /// Resize the layout
   virtual void Resize(std::size_t new_size) { size = new_size; }

   /// Resize the layout based on the given worker offsets
   virtual void Resize(const mfem::Array<std::size_t> &offsets)
   { MFEM_ABORT("method not supported"); }

   ///@}
   // End: Virtual interface

   /// Layouts without engine cannot create DArray, DVector, etc.
   bool HasEngine() const { return engine != NULL; }

   /// TODO: doxygen
   const Engine &GetEngine() const { return *engine; }

   /// TODO: doxygen
   std::size_t Size() const { return size; }

   /// TODO
   template <typename derived_t>
   derived_t &As() { return *util::As<derived_t>(this); }

   /// TODO
   template <typename derived_t>
   const derived_t &As() const { return *util::As<const derived_t>(this); }

   /// TODO: doxygen
   template <typename DObject, typename entry_t>
   DObject Make()
   {
      MFEM_ASSERT(HasEngine(), "this method requires an Engine");
      return Maker<DObject>::template MakeNew<entry_t>(*this);
   }
};

template <> struct PLayout::Maker<DArray>
{
   template <typename entry_t> static DArray MakeNew(PLayout &layout)
   { return layout.GetEngine().MakeArray(layout, sizeof(entry_t)); }
};

template <> struct PLayout::Maker<DVector>
{
   template <typename entry_t> static DVector MakeNew(PLayout &layout)
   { return layout.GetEngine().MakeVector(layout, ScalarId<entry_t>::value); }
};

} // namespace mfem

#endif // MFEM_USE_BACKENDS

#endif // MFEM_BACKENDS_BASE_LAYOUT_HPP
