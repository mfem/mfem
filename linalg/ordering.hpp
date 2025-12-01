#ifndef MFEM_ORDERING
#define MFEM_ORDERING

#include "../config/config.hpp"
#include "vector.hpp"

namespace mfem
{

/** @brief The ordering method used when the number of unknowns per mesh node
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

   template <Type Ord>
   static inline int Map(int ndofs, int vdim, int dof, int vd);

   template <Type Ord>
   static void DofsToVDofs(int ndofs, int vdim, Array<int> &dofs);

   /// Reorder Vector \p v from its current ordering \p in_ord to \p out_ord
   static void Reorder(Vector &v, int vdim, Type in_ord, Type out_ord);

};

template <> inline int
Ordering::Map<Ordering::byNODES>(int ndofs, int vdim, int dof, int vd)
{
   MFEM_ASSERT(dof < ndofs && -1-dof < ndofs && 0 <= vd && vd < vdim, "");
   return (dof >= 0) ? dof+ndofs*vd : dof-ndofs*vd;
}

template <> inline int
Ordering::Map<Ordering::byVDIM>(int ndofs, int vdim, int dof, int vd)
{
   MFEM_ASSERT(dof < ndofs && -1-dof < ndofs && 0 <= vd && vd < vdim, "");
   return (dof >= 0) ? vd+vdim*dof : -1-(vd+vdim*(-1-dof));
}

} // namespace mfem

#endif // MFEM_ORDERING
