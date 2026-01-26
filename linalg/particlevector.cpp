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

#include "particlevector.hpp"
#include "../general/forall.hpp"

namespace mfem
{

void ParticleVector::GrowSize(int min_num_vectors, bool keep_data)
{
   const int nsize = std::max(min_num_vectors*vdim, 2 * data.Capacity());
   Memory<real_t> p(nsize, data.GetMemoryType());
   if (keep_data) { p.CopyFrom(data, size); }
   p.UseDevice(data.UseDevice());
   data.Delete();
   data = p;
}

ParticleVector::ParticleVector(int vdim_, Ordering::Type ordering_)
   : ParticleVector(vdim_, ordering_, 0) { }

ParticleVector::ParticleVector(int vdim_, Ordering::Type ordering_,
                               int num_nodes)
   : Vector(num_nodes*vdim_), vdim(vdim_), ordering(ordering_)
{
   Vector::operator=(0.0);
}

ParticleVector::ParticleVector(int vdim_, Ordering::Type ordering_,
                               const Vector &vec)
   : Vector(vec), vdim(vdim_), ordering(ordering_)
{
   MFEM_ASSERT(vec.Size() % vdim == 0,
               "Incompatible Vector size of " << vec.Size() << " given vdim " << vdim);
}

void ParticleVector::GetValues(int i, Vector &nvals) const
{
   nvals.SetSize(vdim);

   const bool nvals_use_dev = nvals.UseDevice();
   // Use ParticleVector's device flag to minimize movement from large source
   const bool use_dev = UseDevice();
   const auto d_src = Read(use_dev);
   auto d_dest = nvals.Write(use_dev);

   const int vdim_ = vdim;
   const int ordering_ = (int)ordering;
   const int nv = (ordering == Ordering::byNODES) ? size / vdim : 0;

   MFEM_FORALL(c, vdim_,
   {
      if (ordering_ == Ordering::byNODES)
      {
         d_dest[c] = d_src[i + nv*c];
      }
      else
      {
         d_dest[c] = d_src[c + vdim_*i];
      }
   });

   // If nvals was not using device but ParticleVector is, copy back to host
   if (!nvals_use_dev && use_dev)
   {
      nvals.HostRead();
      nvals.UseDevice(false);
   }
   // If nvals was using device but ParticleVector is not, copy back to device
   if (!use_dev && nvals_use_dev)
   {
      nvals.Read();
   }
}

void ParticleVector::GetValuesRef(int i, Vector &nref)
{
   MFEM_ASSERT(ordering == Ordering::byVDIM,
               "GetValuesRef only valid when ordering byVDIM.");

   nref.MakeRef(*this, i*vdim, vdim);
}

void ParticleVector::GetComponents(int vd, Vector &comp)
{
   int vdim_temp = vdim;

   // For byNODES: Treat each component as a vector temporarily
   // For byVDIM:  Treat each vector as a component temporarily
   vdim = GetNumParticles();
   ordering = ordering == Ordering::byNODES ? Ordering::byVDIM :
              Ordering::byNODES;

   GetValues(vd, comp);

   // Reset ordering back to original
   ordering = ordering == Ordering::byNODES ? Ordering::byVDIM :
              Ordering::byNODES;

   vdim = vdim_temp;
}

void ParticleVector::GetComponentsRef(int vd, Vector &nref)
{
   MFEM_ASSERT(ordering == Ordering::byNODES,
               "GetComponentsRef only valid when ordering byNODES.");
   nref.MakeRef(*this, vd*GetNumParticles(), GetNumParticles());
}

void ParticleVector::SetValues(int i, const Vector &nvals)
{
   const bool use_dev = UseDevice(); // use ParticleVector's device flag
   const bool nvals_use_dev = nvals.UseDevice();
   auto d_dest = ReadWrite(use_dev);
   const auto d_src = nvals.Read(use_dev);

   const int vdim_ = vdim;
   const int ordering_ = (int)ordering;
   const int nv = (ordering == Ordering::byNODES) ? size / vdim : 0;

   MFEM_FORALL(c, vdim_,
   {
      if (ordering_ == Ordering::byNODES)
      {
         d_dest[i + c*nv] = d_src[c];
      }
      else
      {
         d_dest[c + i*vdim_] = d_src[c];
      }
   });
   nvals.UseDevice(nvals_use_dev);
}

void ParticleVector::SetComponents(int vd, const Vector &comp)
{
   int vdim_temp = vdim;

   // For byNODES: Treat each component as a vector temporarily
   // For byVDIM:  Treat each vector as a component temporarily
   vdim = GetNumParticles();
   ordering = ordering == Ordering::byNODES ? Ordering::byVDIM :
              Ordering::byNODES;

   SetValues(vd, comp);

   // Reset ordering back to original
   ordering = ordering == Ordering::byNODES ? Ordering::byVDIM :
              Ordering::byNODES;

   vdim = vdim_temp;
}

real_t& ParticleVector::operator()(int i, int comp)
{
   MFEM_ASSERT(i < GetNumParticles(),
               "Particle index " << i <<
               " is invalid for number of particles " << GetNumParticles());
   MFEM_ASSERT(comp < vdim,
               "Component index " << comp <<
               " is invalid for vector dimension " << vdim);

   HostRead();

   if (ordering == Ordering::byNODES)
   {
      return Vector::operator[](i + comp*GetNumParticles());
   }
   else
   {
      return Vector::operator[](comp + i*vdim);
   }
}

const real_t& ParticleVector::operator()(int i, int comp) const
{
   MFEM_ASSERT(i < GetNumParticles(),
               "Particle index " << i <<
               " is invalid for number of particles " << GetNumParticles());
   MFEM_ASSERT(comp < vdim,
               "Component index " << comp <<
               " is invalid for vector dimension " << vdim);

   HostRead();

   if (ordering == Ordering::byNODES)
   {
      return Vector::operator[](i + comp*GetNumParticles());
   }
   else
   {
      return Vector::operator[](comp + i*vdim);
   }
}

void ParticleVector::DeleteParticles(const Array<int> &indices)
{
   if (indices.Size() == 0) { return; }
   // Convert list index array of "ldofs" to "vdofs"
   Array<int> v_list;
   v_list.Reserve(indices.Size()*vdim);
   MFEM_VERIFY(indices.Max() < GetNumParticles(),
               "Particle index " << indices.Max() <<
               " is out-of-range for number of particles " <<
               GetNumParticles());
   if (ordering == Ordering::byNODES)
   {
      for (int l = 0; l < indices.Size(); l++)
      {
         for (int vd = 0; vd < vdim; vd++)
         {
            v_list.Append(Ordering::Map<Ordering::byNODES>(GetNumParticles(),
                                                           vdim,
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
            v_list.Append(Ordering::Map<Ordering::byVDIM>(GetNumParticles(),
                                                          vdim,
                                                          indices[l],
                                                          vd));
         }
      }
   }

   Vector::DeleteAt(v_list);
}

void ParticleVector::SetVDim(int vdim_, bool keep_data)
{
   if (!keep_data)
   {
      int num_particles = GetNumParticles();
      vdim = vdim_;
      Vector::SetSize(num_particles*vdim_);
      return;
   }

   // Reorder/shift existing entries
   // For byNODES: Treat each component as a vector temporarily
   // For byVDIM:  Treat each vector as a component temporarily
   vdim = GetNumParticles();
   ordering = ordering == Ordering::byNODES ? Ordering::byVDIM :
              Ordering::byNODES;

   SetNumParticles(vdim_, keep_data);

   // Reset ordering back to original
   ordering = ordering == Ordering::byNODES ? Ordering::byVDIM :
              Ordering::byNODES;

   vdim = vdim_;
}

void ParticleVector::SetOrdering(Ordering::Type ordering_, bool keep_data)
{
   if (keep_data)
   {
      Ordering::Reorder(*this, vdim, ordering, ordering_);
   }
   ordering = ordering_;
}

void ParticleVector::SetNumParticles(int num_vectors, bool keep_data)
{
   int old_nv = GetNumParticles();

   if (num_vectors == old_nv)
   {
      return;
   }

   // If resizing larger...
   if (num_vectors > old_nv)
   {
      // Increase capacity if needed
      if (num_vectors*vdim > Vector::Capacity())
      {
         GrowSize(num_vectors, keep_data);
      }

      // Set larger new size
      Vector::SetSize(num_vectors*vdim);

      if (!keep_data) { return; }

      auto d_dest = this->ReadWrite(UseDevice());

      if (ordering == Ordering::byNODES)
      {
         // create deep copy of old data that will be copied
         Vector old_slice;
         old_slice.MakeRef(*this, 0, old_nv * vdim);
         Vector old_copy(old_slice);

         const auto d_src = old_copy.Read(UseDevice());
         const int vdim_ = vdim;

         // Shift entries for byNODES
         MFEM_FORALL(k, old_nv * vdim_,
         {
            const int d = k / old_nv;
            const int i = k % old_nv;
            d_dest[i + d*num_vectors] = d_src[k];
         });

         // Zero-out new data slots
         const int diff = num_vectors - old_nv;
         MFEM_FORALL(k, diff * vdim,
         {
            const int d = k / diff;
            const int i = k % diff;
            d_dest[d * num_vectors + old_nv + i] = 0.0;
         });
      }
      else // byVDIM
      {
         const int start_idx = old_nv * vdim;
         const int end_idx = num_vectors * vdim;
         const int diff = end_idx - start_idx;
         MFEM_FORALL(i, diff,
         {
            d_dest[start_idx + i] = 0.0;
         });
      }
   }
   else // Else just remove the trailing vector data
   {
      if (!keep_data) { Vector::SetSize(num_vectors*vdim); return; }
      Array<int> rm_indices(old_nv-num_vectors);
      for (int i = 0; i < rm_indices.Size(); i++)
      {
         rm_indices[i] = old_nv - rm_indices.Size() + i;
      }
      DeleteParticles(rm_indices);
   }
}

} // namespace mfem
