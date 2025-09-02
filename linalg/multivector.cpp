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

#include "multivector.hpp"

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

void MultiVector::GrowSize(int min_num_vectors)
{
   const int nsize = std::max(min_num_vectors*vdim, 2 * data.Capacity());
   Memory<real_t> p(nsize, data.GetMemoryType());
   p.CopyFrom(data, size);
   p.UseDevice(data.UseDevice());
   data.Delete();
   data = p;
}

MultiVector::MultiVector(int vdim_, Ordering::Type ordering_)
   : MultiVector(vdim_, ordering_, 0)
{

}

MultiVector::MultiVector(int vdim_, Ordering::Type ordering_, int num_nodes)
   : Vector(num_nodes*vdim_), vdim(vdim_), ordering(ordering_)
{
   Vector::operator=(0.0);
}

MultiVector::MultiVector(int vdim_, Ordering::Type ordering_, const Vector &vec)
   : Vector(vec), vdim(vdim_), ordering(ordering_)
{
   MFEM_ASSERT(vec.Size() % vdim == 0,
               "Incompatible Vector size of " << vec.Size() << " given vdim " << vdim);
}

void MultiVector::GetVectorValues(int i, Vector &nvals) const
{
   nvals.SetSize(vdim);

   if (ordering == Ordering::byNODES)
   {
      int nv = GetNumVectors();
      for (int c = 0; c < vdim; c++)
      {
         nvals[c] = Vector::operator[](i+nv*c);
      }
   }
   else
   {
      for (int c = 0; c < vdim; c++)
      {
         nvals[c] = Vector::operator[](c+vdim*i);
      }
   }
}

void MultiVector::GetVectorRef(int i, Vector &nref)
{
   MFEM_ASSERT(ordering == Ordering::byVDIM,
               "GetRefVector only valid when ordering byVDIM.");

   nref.MakeRef(*this, i*vdim, vdim);
}

void MultiVector::SetVectorValues(int i, const Vector &nvals)
{
   if (ordering == Ordering::byNODES)
   {
      int nv = GetNumVectors();
      for (int c = 0; c < vdim; c++)
      {
         Vector::operator[](i + c*nv) = nvals[c];
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

real_t& MultiVector::operator()(int i, int comp)
{
   MFEM_ASSERT(i < GetNumVectors(),
               "Vector index " << i << " is out-of-range for number of vectors " <<
               GetNumVectors());

   if (ordering == Ordering::byNODES)
   {
      return Vector::operator[](i + comp*GetNumVectors());
   }
   else
   {
      return Vector::operator[](comp + i*vdim);
   }
}

const real_t& MultiVector::operator()(int i, int comp) const
{
   if (ordering == Ordering::byNODES)
   {
      return Vector::operator[](i + comp*GetNumVectors());
   }
   else
   {
      return Vector::operator[](comp + i*vdim);
   }
}

void MultiVector::DeleteVectorsAt(const Array<int> &indices)
{
   // Convert list index array of "ldofs" to "vdofs"
   Array<int> v_list;
   v_list.Reserve(indices.Size()*vdim);
   if (ordering == Ordering::byNODES)
   {
      for (int l = 0; l < indices.Size(); l++)
      {
         for (int vd = 0; vd < vdim; vd++)
         {
            v_list.Append(Ordering::Map<Ordering::byNODES>(GetNumVectors(), vdim,
                                                           indices[l], vd));
         }
      }
   }
   else
   {
      for (int l = 0; l < indices.Size(); l++)
      {
         for (int vd = 0; vd < vdim; vd++)
         {
            v_list.Append(Ordering::Map<Ordering::byVDIM>(GetNumVectors(), vdim, indices[l],
                                                          vd));
         }
      }
   }

   Vector::DeleteAt(v_list);
}

void MultiVector::SetOrdering(Ordering::Type ordering_)
{
   Ordering::Reorder(*this, vdim, ordering, ordering_);
   ordering = ordering_;
}

void MultiVector::SetNumVectors(int num_vectors)
{
   int old_nv = GetNumVectors();

   if (num_vectors == old_nv)
   {
      return;
   }
   using namespace std;

   // If resizing larger...
   if (num_vectors > old_nv)
   {
      // Increase capacity if needed
      if (num_vectors*vdim > Vector::Capacity())
      {
         GrowSize(num_vectors);
      }

      // Set larger new size
      Vector::SetSize(num_vectors*vdim);

      if (ordering == Ordering::byNODES)
      {
         // Shift entries for byNODES
         for (int c = vdim-1; c > 0; c--)
         {
            for (int i = old_nv-1; i >= 0; i--)
            {
               Vector::operator[](i+c*num_vectors) = Vector::operator[](i+c*old_nv);
            }
         }

         // Zero-out data now associated with new Vectors
         for (int c = 0; c < vdim; c++)
         {
            for (int i = old_nv; i < num_vectors; i++)
            {
               Vector::operator[](i+c*num_vectors) = 0.0;
            }
         }
      }
      else // byVDIM
      {
         for (int i = old_nv*vdim; i < num_vectors*vdim; i++)
         {
            data[i] = 0.0;
         }
      }
   }
   else // Else just remove the trailing vector data
   {
      Array<int> rm_indices(old_nv-num_vectors);
      for (int i = 0; i < rm_indices.Size(); i++)
      {
         rm_indices[i] = old_nv - rm_indices.Size() + i;
      }
      DeleteVectorsAt(rm_indices);
   }
}

} // namespace mfem
