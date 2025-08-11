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

#ifndef MFEM_NODEFUNCTION
#define MFEM_NODEFUNCTION

#include "vector.hpp"

namespace mfem
{

/** @brief The ordering method used when the number of unknowns per node
    (vector dimension) is bigger than 1. */
class Ordering
{
public:
   /// %Ordering methods:
   enum Type
   {
      byNODES, /**< loop first over the nodes (inner loop) then over the vector
                    dimension (outer loop); symbolically it can be represented
                    as: XXX...,YYY...,ZZZ... */
      byVDIM   /**< loop first over the vector dimension (inner loop) then over
                    the nodes (outer loop); symbolically it can be represented
                    as: XYZ,XYZ,XYZ,... */
   };

   /// Map ldof \p dof to vdof component \p vd for \p ndof total ldofs
   template <Type Ord>
   static int Map(int ndofs, int vdim, int dof, int vd);

   template <Type Ord>
   static void DofsToVDofs(int ndofs, int vdim, Array<int> &dofs);

   /// Reorder Vector \p v from its current ordering \p in_ord to \p out_ord
   static void Reorder(Vector &v, int vdim, Type in_ord, Type out_ord);
};


class MultiVector : public Vector
{
protected:
   const int vdim;
   const Ordering::Type ordering;

public:

   using Vector::operator=;
   using Vector::operator();

   MultiVector(int vdim_, Ordering::Type ordering_);

   MultiVector(int vdim_, Ordering::Type ordering_, int num_nodes);

   int GetVDim() const { return vdim; }

   Ordering::Type GetOrdering() const { return ordering; }

   int GetNumNodes() const { return Size()/vdim; }

   void GetNodeValues(int i, Vector &nvals) const;

   void GetRefNodeValues(int i, Vector &nref);

   void SetNodeValues(int i, const Vector &nvals);

   real_t& operator()(int i, int comp);

   const real_t& operator()(int i, int comp) const;
   
};

} // namespace mfem


#endif // MFEM_NODEFUNCTION