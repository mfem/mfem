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
               "GetVectorRef only valid when ordering byVDIM.");

   nref.MakeRef(*this, i*vdim, vdim);
}

void MultiVector::GetComponentValues(int vd, Vector &comp)
{
   int vdim_temp = vdim;

   // For byNODES: Treat each component as a vector temporarily
   // For byVDIM:  Treat each vector as a component temporarily
   vdim = GetNumVectors();
   ordering = ordering == Ordering::byNODES ? Ordering::byVDIM : Ordering::byNODES;

   GetVectorValues(vd, comp);

   // Reset ordering back to original
   ordering = ordering == Ordering::byNODES ? Ordering::byVDIM : Ordering::byNODES;

   vdim = vdim_temp;
}

void MultiVector::GetComponentRef(int vd, Vector &nref)
{
   MFEM_ASSERT(ordering == Ordering::byNODES,
               "GetComponentRef only valid when ordering byNODES.");
   nref.MakeRef(*this, vd*GetNumVectors(), GetNumVectors());
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

void MultiVector::SetComponentValues(int vd, const Vector &comp)
{
   int vdim_temp = vdim;

   // For byNODES: Treat each component as a vector temporarily
   // For byVDIM:  Treat each vector as a component temporarily
   vdim = GetNumVectors();
   ordering = ordering == Ordering::byNODES ? Ordering::byVDIM : Ordering::byNODES;

   SetVectorValues(vd, comp);

   // Reset ordering back to original
   ordering = ordering == Ordering::byNODES ? Ordering::byVDIM : Ordering::byNODES;

   vdim = vdim_temp;
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

void MultiVector::SetVDim(int vdim_)
{
   // Reorder/shift existing entries
   // For byNODES: Treat each component as a vector temporarily
   // For byVDIM:  Treat each vector as a component temporarily
   vdim = GetNumVectors();
   ordering = ordering == Ordering::byNODES ? Ordering::byVDIM : Ordering::byNODES;
   
   SetNumVectors(vdim_);

   // Reset ordering back to original
   ordering = ordering == Ordering::byNODES ? Ordering::byVDIM : Ordering::byNODES;

   vdim = vdim_;
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
