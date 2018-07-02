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

#ifndef MFEM_BACKENDS_OMP_LAYOUT_HPP
#define MFEM_BACKENDS_OMP_LAYOUT_HPP

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)

#include "../base/layout.hpp"
#include "engine.hpp"

namespace mfem
{

namespace omp
{

class Layout : public mfem::PLayout
{
protected:
   //
   // Inherited fields
   //
   // SharedPtr<const mfem::Engine> engine;
   // std::size_t size;

public:
   Layout(const Engine &e, std::size_t s = 0) : PLayout(e, s) { }

   const Engine &OmpEngine() const
   { return *static_cast<const Engine *>(engine.Get()); }

   void *Alloc(std::size_t bytes) const
   { return OmpEngine().Malloc(bytes); }

   void Dealloc(void *ptr) const
   { return OmpEngine().Dealloc(ptr); }

   virtual ~Layout() { }

   /**
      @name Virtual interface
   */
   ///@{

   /// Resize the layout
   virtual void Resize(std::size_t new_size);

   /// Resize the layout based on the given worker offsets
   virtual void Resize(const Array<std::size_t> &offsets);

   ///@}
   // End: Virtual interface
};

} // namespace mfem::omp

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)

#endif // MFEM_BACKENDS_OMP_LAYOUT_HPP
