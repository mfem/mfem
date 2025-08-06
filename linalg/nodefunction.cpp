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

#include "nodefunction.hpp"

namespace mfem
{

template <> int Ordering::
Map<Ordering::byNODES>(int ndofs, int vdim, int dof, int vd)
{
   MFEM_ASSERT(dof < ndofs && -1-dof < ndofs && 0 <= vd && vd < vdim, "");
   return (dof >= 0) ? dof+ndofs*vd : dof-ndofs*vd;
}

template <> int Ordering::
Map<Ordering::byVDIM>(int ndofs, int vdim, int dof, int vd)
{
   MFEM_ASSERT(dof < ndofs && -1-dof < ndofs && 0 <= vd && vd < vdim, "");
   return (dof >= 0) ? vd+vdim*dof : -1-(vd+vdim*(-1-dof));
}

template <> void Ordering::
DofsToVDofs<Ordering::byNODES>(int ndofs, int vdim, Array<int> &dofs)
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

template <> void Ordering::
DofsToVDofs<Ordering::byVDIM>(int ndofs, int vdim, Array<int> &dofs)
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
      for (int i = 0; i < nvdofs; i++)
      {
         v[i] = temp[Map<byVDIM>(nldofs,vdim,i%nldofs,i/nldofs)];
      }
   }
   else // byNODES -> byVDIM
   {
      Vector temp = v;
      for (int i = 0; i < nvdofs; i++)
      {
         v[i] = temp[Map<byNODES>(nldofs,vdim,i/vdim,i%vdim)];
      }
   }
}

void NodeFunction::GetNodeValues(int i, Vector &nvals) const
{
   nvals.SetSize(vdim);

   if (ordering == Ordering::byNODES)
   {
      for (int c = 0; c < vdim; c++)
      {
         nvals[c] = Vector::operator[](i + c*GetNumNodes());
      }
   }
   else
   {
      for (int c = 0; c < vdim; c++)
      {
         nvals[c] = Vector::operator[](c + i*vdim);
      }
   }
}

void NodeFunction::GetRefNodeValues(int i, Vector &nref)
{
   MFEM_VERIFY(ordering == Ordering::byVDIM, "GetRefNodeValues only valid when ordering byVDIM.");
   nref.MakeRef(*this, i*vdim, vdim);
}

void NodeFunction::SetNodeValues(int i, const Vector &nvals)
{
   if (ordering == Ordering::byNODES)
   {
      for (int c = 0; c < vdim; c++)
      {
         Vector::operator[](i + c*GetNumNodes()) = nvals[c];
      }
   }
   else
   {
      for (int c = 0; c < vdim; c++)
      {
         Vector::operator[](c + i*vdim) = nvals[c];
      }
   }
}

real_t& NodeFunction::operator()(int i, int comp)
{
   if (ordering == Ordering::byNODES)
   {
      return Vector::operator[](i + comp*GetNumNodes());
   }
   else
   {
      return Vector::operator[](comp + i*vdim);
   }
}

const real_t& NodeFunction::operator()(int i, int comp) const
{
   if (ordering == Ordering::byNODES)
   {
      return Vector::operator[](i + comp*GetNumNodes());
   }
   else
   {
      return Vector::operator[](comp + i*vdim);
   }
}

}