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

   if (ordering == Ordering::byNODES)
   {
      int nv = GetNumParticles();
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
   if (ordering == Ordering::byNODES)
   {
      int nv = GetNumParticles();
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
