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

#ifndef MFEM_BACKENDS_KERNELS_LAYOUT_HPP
#define MFEM_BACKENDS_KERNELS_LAYOUT_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

namespace mfem
{

namespace kernels
{

class Layout : public mfem::PLayout
{
protected:

public:
   Layout(const Engine &e, std::size_t s = 0) : PLayout(e, s) { }

   const Engine &KernelsEngine() const
   { return *static_cast<const Engine *>(engine.Get()); }

   kernels::memory Alloc(std::size_t bytes) const
   { return KernelsEngine().GetDevice().malloc(bytes); }

   virtual ~Layout() { }

   /// Resize the layout
   virtual void Resize(std::size_t new_size);

   /// Resize the layout based on the given worker offsets
   virtual void Resize(const Array<std::size_t> &offsets);
};

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#endif // MFEM_BACKENDS_KERNELS_LAYOUT_HPP
