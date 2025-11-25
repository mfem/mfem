#include "ordering.hpp"

namespace mfem
{

template <>
void Ordering::DofsToVDofs<Ordering::byNODES>(int ndofs, int vdim,
                                              Array<int> &dofs)
{
   // static method
   int size = dofs.Size();
   dofs.SetSize(size*vdim);
   for (int vd = 1; vd < vdim; vd++)
   {
      for (int i = 0; i < size; i++)
      {
         dofs[i+size*vd] = Map<byNODES>(ndofs, vdim, dofs[i], vd);
      }
   }
}

template <>
void Ordering::DofsToVDofs<Ordering::byVDIM>(int ndofs, int vdim,
                                             Array<int> &dofs)
{
   // static method
   int size = dofs.Size();
   dofs.SetSize(size*vdim);
   for (int vd = vdim-1; vd >= 0; vd--)
   {
      for (int i = 0; i < size; i++)
      {
         dofs[i+size*vd] = Map<byVDIM>(ndofs, vdim, dofs[i], vd);
      }
   }
}

void Ordering::Reorder(Vector &v, int vdim, Ordering::Type in_ord,
                       Ordering::Type out_ord)
{
   if (in_ord == out_ord)
   {
      return;
   }

   int nvdofs = v.Size();
   int nldofs = nvdofs/vdim;

   if (out_ord == Ordering::byNODES) // byVDIM -> byNODES
   {
      Vector temp = v;
      for (int d = 0; d < vdim; d++)
      {
         int off = d * nldofs;
         for (int i = 0; i < nldofs; i++)
         {
            v[i + off] = temp[Map<byVDIM>(nldofs,vdim,i,d)];
         }
      }
   }
   else // byNODES -> byVDIM
   {
      Vector temp = v;
      for (int i = 0; i < nldofs; i++)
      {
         int off = i*vdim;
         for (int d = 0; d < vdim; d++)
         {
            v[d + off] = temp[Map<byNODES>(nldofs,vdim,i,d)];
         }
      }
   }
}

} // namespace mfem
