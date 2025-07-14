
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
   np = np_old + num_new;

   // If now over-capacity, create new data sized to fit new
   if (np*vdim > data.Capacity())
   {
      Memory<T> old = data;
      data.Delete();
      data.New(np*vdim, old.GetMemoryType());
      data.UseDevice(old.UseDevice);
      data.CopyFrom(old, np_old*vdim);
      old.Delete();
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
   // TODO: Definitely some mistakes with memorytype + usedevice stuff...
   Memory<T> old = data;
   data.Delete();
   data.New(old.Capacity() - indices.Size()*vdim, old.GetMemoryType());
   data.UseDevice(old.UseDevice());
   np = data.Capacity()/vdim;

   int np_old = old.Capacity()/vdim;

   Array<int> mask(np_old);
   FiniteElementSpace::ListToMarker(indices, np_old, mask, 1)
   // Copy non-removed data over
   if (ordering == Ordering::byNODES)
   {
      int idx = 0;
      for (int i = 0; i < np_old; i++)
      {
         if (mask[i])
         {
            for (int c = 0; c < vdim; c++)
            {
               data[idx+c*np] = old[i+c*np_old];
            }
            idx++;
         }
      }
   }
   else
   {
      int idx = 0;
      for (int i = 0; i < np_old; i++)
      {
         if (mask[i])
         {
            for (int c = 0; c < vdim; c++)
            {
               data[c+idx*vdim] = old[c+i*vdim];
            }
            idx++;
         }
      }
   }

   old.Delete();
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
void ParticleData<T>::SetParticleData(int i, const Memory<T> &pdata)
{
   if (ordering == Ordering::byNODES)
   {
      for (int c = 0; c < vdim; c++)
      {
         data[i + c*np] = pdata[c];
      }
   }
   else
   {
      for (int c = 0; c < vdim; c++)
      {
         data[c + i*vdim] = pdata[c];
      }
   }
}

template<typename T>
void ParticleData<T>::SetParticleData(const Array<int> &indices,
                              const Memory<T> &pdatas)
{
   int np_update = pdatas.Capacity()/vdim;
   
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


ParticleFunction::ParticleFunction(const ParticleSpace &pspace, int vdim_)
: ParticleData<real_t>(pspace.GetNP(), pspace.GetOrdering(), vdim_),
  pspace(pspace)
{

}

void ParticleFunction::Interpolate(GridFunction &gf)
{
   Mesh *m = gf.FESpace()->GetMesh();

   // Check if this mesh exists on the particlespace

}

} // namespace mfem

#endif // MFEM_USE_GSLIB