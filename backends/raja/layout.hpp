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
#ifndef MFEM_BACKENDS_RAJA_LAYOUT_HPP
#define MFEM_BACKENDS_RAJA_LAYOUT_HPP

namespace mfem
{

namespace raja
{

class Layout : public PLayout
{
protected:
   //
   // Inherited fields
   //
   // SharedPtr<const mfem::Engine> engine;
   // std::size_t size;

public:
   Layout(const Engine &e, std::size_t s = 0) : PLayout(e, s) { }

   const Engine& RajaEngine() const
   { return *static_cast<const Engine *>(engine.Get()); }

  raja::memory Alloc(std::size_t bytes) const;

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

} // namespace mfem::raja

} // namespace mfem

#endif // MFEM_BACKENDS_RAJA_LAYOUT_HPP
