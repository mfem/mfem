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

#ifndef MFEM_BACKENDS_RAJA_VECTOR_HPP
#define MFEM_BACKENDS_RAJA_VECTOR_HPP

namespace mfem
{

namespace raja
{

class Vector : virtual public Array, public PVector
{
protected:
   //
   // Inherited fields
   //
   // DLayout layout;

   /**
       @name Virtual interface
    */
   ///@{

   virtual PVector *DoVectorClone(bool copy_data, void **buffer,
                                  int buffer_type_id) const;

   virtual void DoDotProduct(const PVector &x, void *result,
                             int result_type_id) const;

   virtual void DoAxpby(const void *a, const PVector &x,
                        const void *b, const PVector &y,
                        int ab_type_id);

   ///@}
   // End: Virtual interface

public:
   Vector(Layout &lt)
      : PArray(lt), Array(lt, sizeof(double)), PVector(lt)
   { }

   mfem::Vector Wrap();

   const mfem::Vector Wrap() const;
};

} // namespace mfem::raja

} // namespace mfem

#endif // MFEM_BACKENDS_RAJA_VECTOR_HPP
