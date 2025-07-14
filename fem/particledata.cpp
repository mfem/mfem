
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
#include "particledata.hpp"
#include "particlespace.hpp"

#ifdef MFEM_USE_GSLIB

namespace mfem
{

template<typename T>
void ParticleData<T>::AddParticles(int num_new)
{
   // Update np (size-indicator)
   int np_old = np;
   np += num_new;

   // If now over-capacity, create new data sized to fit new
   if (np*vdim > data.Capacity())
   {
      Memory<T> p(np*vdim, data.GetMemoryType());
      p.CopyFrom(data, data.Capacity());
      p.UseDevice(data.UseDevice());
      data.Delete();
      data = p;
   }

   // Update data
   // Default-initialize any remaining new entires
   if (ordering == Ordering::byNODES)
   {
      // Shift down byNODES data...
      for (int c = vdim-1; c > 0; c--)
      {
         for (int i = np_old-1; i >= 0; i--)
         {
            data[i+c*np] = data[i+c*np_old];
         }
         for (int i = np_old; i < np; i++)
         {
            data[i+c*np] = T();
         } 
      }
   }
   else
   {
      for (int i = np_old*vdim; i < np*vdim; i++)
      {
         data[i] = T();
      }
   }

   SyncWrapper();

}

template<typename T>
void ParticleData<T>::RemoveParticles(const Array<int> &indices)
{
   // Update np (size-indicator)
   int np_old = np;
   np -= indices.Size();

   // Sort the indices
   Array<int> sorted_list(indices);
   sorted_list.Sort();

   // Shift non-removed data, maintain capacity AT END OF DATA
   if (ordering == Ordering::byNODES)
   {
      int rm_count = 0;
      for (int i = sorted_list[0]; i < np_old*vdim; i++)
      {
         if (i % np_old == sorted_list[rm_count % sorted_list.Size()])
         {
            rm_count++;
         }
         else
         {
            data[i-rm_count] = data[i]; // Shift elements rm_count
         }
      }
   }
   else
   {
      int rm_count = 0;
      for (int i = sorted_list[0]*vdim; i < np_old*vdim;  i++)
      {
         int p_idx = i/vdim; // particle index
         int rm_idx = rm_count/vdim;
         if (rm_idx < sorted_list.Size() && p_idx == sorted_list[rm_idx])
         {
            rm_count += vdim;
            i += vdim - 1;
         }
         else
         {
            data[i-rm_count] = data[i];
         }
      }
   }

   SyncWrapper();
}

template<typename T>
void ParticleData<T>::ShrinkToFit()
{
   // Copied from Array<T>::ShrinkToFit
   if (Capacity() == np*vdim) { return; }
   Memory<T> p(np*vdim, data.GetMemoryType());
   p.CopyFrom(data, np*vdim);
   p.UseDevice(data.UseDevice());
   data.Delete();
   data = p;

   SyncWrapper();
}

template<typename T>
T& ParticleData<T>::GetParticleData(int i, int comp)
{
   if (ordering == Ordering::byNODES)
   {
      return data[i + comp*np];
   }
   else
   {
      return data[comp + i*vdim];
   }
}

template<typename T>
const T& ParticleData<T>::GetParticleData(int i, int comp) const
{
   if (ordering == Ordering::byNODES)
   {
      return data[i + comp*np];
   }
   else
   {
      return data[comp + i*vdim];
   }
}

template<typename T>
void ParticleData<T>::GetParticleData(int i, Memory<T> &pdata) const
{
   if (ordering == Ordering::byNODES)
   {
      for (int c = 0; c < vdim; c++)
      {
         pdata[c] = data[i + c*np];
      }
   }
   else
   {
      for (int c = 0; c < vdim; c++)
      {
         pdata[c] = data[c + i*vdim];
      }
   }
}

template<typename T>
void ParticleData<T>::SetParticleData(int i, const T &pdata, int comp)
{
   if (ordering == Ordering::byNODES)
   {
      data[i + comp*np] = pdata;
   }
   else
   {
      data[comp + i*vdim] = pdata;
   }
}

template<typename T>
void ParticleData<T>::SetParticleData(const Array<int> &indices,
                              const Memory<T> &pdatas)
{
   int np_update = indices.Size();
   
   if (ordering == Ordering::byNODES)
   {
      for (int i = 0; i < np_update; i++)
      {
         for (int c = 0; c < vdim; c++)
         {
            data[indices[i]+c*np] = pdatas[i+c*np_update];
         }
      }
   }
   else
   {
      for (int i = 0; i < np_update; i++)
      {
         for (int c = 0; c < vdim; c++)
         {
            data[c+indices[i]*vdim] = pdatas[c+i*vdim];
         }
      }
   }
}

template class ParticleData<int>;
template class ParticleData<real_t>;

ParticleFunction::ParticleFunction(const ParticleSpace &pspace, int vdim_)
: ParticleData<real_t>(pspace.GetNP(), vdim_, pspace.GetOrdering()),
  pspace(pspace)
{
   // TODO: For now, capacity == num_particles...
   AddParticles(pspace.GetNP());
}

// void ParticleFunction::Interpolate(GridFunction &gf)
// {
//    Mesh *m = gf.FESpace()->GetMesh();

//    // Check if this mesh exists on the particlespace

// }

} // namespace mfem

#endif // MFEM_USE_GSLIB