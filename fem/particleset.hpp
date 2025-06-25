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

#ifndef MFEM_PARTICLESET
#define MFEM_PARTICLESET

#include "../config/config.hpp"

#include "../mesh/mesh.hpp"
#include "fespace.hpp"

#include <vector>
#include <numeric>

namespace mfem
{

// -----------------------------------------------------------------------------------------------------
// Define Particle class

template<int SpaceDim, int NumScalars=0, int... VectorVDims>
class Particle
{
private:
   static constexpr std::array<int, sizeof...(VectorVDims)> VDims = { VectorVDims... };

   bool owning;

   Vector coords;
   std::array<real_t*, NumScalars> scalars;
   std::array<Vector, sizeof...(VectorVDims)> vectors;

   void Destroy();
   void Copy(const Particle &p);
   void Steal(Particle &p);

public:
   static constexpr int GetSpaceDim() { return SpaceDim; };
   static constexpr int GetNumScalars() { return NumScalars; };
   static constexpr int GetNumVectors() { return sizeof...(VectorVDims); };
   static constexpr int GetVDim(int v) { return VDims[v]; };

   /// Create a new particle which owns its own data
   Particle();

   /// Create a new particle whose data references external data
   Particle(real_t *in_coords, real_t *in_scalars[], real_t *in_vectors[]);

   /// Copy ctor
   explicit Particle(const Particle &p) : Particle() { Copy(p); }

   /// Move ctor
   explicit Particle(Particle &&p) : Particle() { Steal(p); }

   /// Copy assign
   Particle& operator=(const Particle &p) { Copy(p); return *this; }

   /// Move assign
   Particle& operator=(Particle &&p) { Steal(p); return *this; }   

   Vector& GetCoords() { return coords; }

   real_t& GetScalar(int s) { return *scalars[s]; }

   Vector& GetVector(int v) { return vectors[v]; }

   const Vector& GetCoords() const { return coords; }

   const real_t& GetScalar(int s) const { return *scalars[s]; }

   const Vector& GetVector(int v) const { return vectors[v]; }

   bool operator==(const Particle &rhs) const;

   bool operator!=(const Particle &rhs) const { return !operator==(rhs); }

   ~Particle() { Destroy(); }

   void Print(std::ostream &out=mfem::out);
};

// -----------------------------------------------------------------------------------------------------
// Define Base ParticleSet to restrict first type T
template<typename T, Ordering::Type VOrdering=Ordering::byNODES>
class ParticleSet { static_assert(sizeof(T)==0, "ParticleSet<T,VOrdering> requires that T is a Particle."); };


// Define ParticleSet
template<int SpaceDim, int NumScalars, int... VectorVDims, Ordering::Type VOrdering>
class ParticleSet<Particle<SpaceDim,NumScalars,VectorVDims...>, VOrdering>
{
protected:
   static constexpr std::array<int, sizeof...(VectorVDims)> VDims = { VectorVDims... }; // VDims of vectors
   static constexpr int TotalFields = 1 + NumScalars + sizeof...(VectorVDims); // Total number of fields / particle
   static constexpr int TotalComps = SpaceDim + NumScalars + (VectorVDims + ... + 0); // Total comps / particle
   
   static constexpr std::array<int, TotalFields> MakeFieldVDims()
   {
      std::array<int, TotalFields> temp{};
      temp[0] = SpaceDim;
      for (int s = 0; s < NumScalars; s++) temp[1+s] = 1;
      for (int v = 0; v < sizeof...(VectorVDims); v++) temp[1+NumScalars+v] = ParticleSet::VDims[v];
      return temp;
   }
   static constexpr std::array<int, TotalFields> FieldVDims = MakeFieldVDims(); // VDims of all fields (coords, scalars, vectors...)
   
   static const std::array<int, TotalFields> MakeExclScanFieldVDims() // std::exclusive_scan not constexpr until C++20
   {
      std::array<int, TotalFields> temp{};
      std::exclusive_scan(FieldVDims.begin(), FieldVDims.end(), temp.begin(), 0);
      return temp;
   }
   static inline const std::array<int, TotalFields> ExclScanFieldVDims = MakeExclScanFieldVDims();
   
   const int id_stride;
   int id_counter;

   std::vector<real_t> data; // Stores ALL particle data
   std::array<Vector, TotalFields> fields; // User-facing Vectors, referencing data
   Array<unsigned int> ids; // Particle IDs


   /// Sync Vectors in \ref fields
   void SyncVectors();

   /// Add particle w/ given ID
   void AddParticle(const Particle<SpaceDim, NumScalars, VectorVDims...> &p, int id);

   /// Private ctor to set ID stride + starting counter
   ParticleSet(const int id_stride_, const int id_0) : id_stride(id_stride_), id_counter(id_0) { };

public:

   static constexpr Ordering::Type GetOrdering() { return VOrdering; }
   
   ParticleSet() : ParticleSet(1, 0) {};

   /// Reserve room for \p res particles. Can help to avoid re-allocation for adding + removing particles.
   void Reserve(int res) { data.reserve(res*TotalComps); ids.Reserve(res); }

   /// Get the number of particles currently held by this ParticleSet.
   int GetNP() const { return ids.Size(); }

   /// Add particle
   void AddParticle(const Particle<SpaceDim, NumScalars, VectorVDims...> &p)
   {
      AddParticle(p, id_counter);
      id_counter += id_stride;
   };

   /// Remove particle data specified by \p list of particle indices.
   void RemoveParticles(const Array<int> &list);

   /// Get particle \p i referencing actual, data ONLY FOR Ordering byVDIM.
   Particle<SpaceDim, NumScalars, VectorVDims...> GetParticleRef(int i);

   const Particle<SpaceDim, NumScalars, VectorVDims...> GetParticleRef(int i) const { return const_cast<ParticleSet*>(this)->GetParticleRef(i); };
   
   /// Get copy of data in particle \p i .
   Particle<SpaceDim, NumScalars, VectorVDims...> GetParticleData(int i) const;

   const Array<unsigned int>& GetIDs() const { return ids; };

   Vector& GetSetCoords() { return fields[0]; }

   Vector& GetSetScalar(int s) { return fields[1+s]; }

   Vector& GetSetVector(int v) { return fields[1+NumScalars+v]; }

   const Vector& GetSetCoords() const { return fields[0]; }

   const Vector& GetSetScalar(int s) const { return fields[1+s]; }

   const Vector& GetSetVector(int v) const { return fields[1+NumScalars+v]; }

   /// Print in VisIt Point3D format (x y z ID)
   void PrintPoint3D(std::ostream &out);

   /// Print all data to CSV (can be read into ParaView easily)
   void PrintCSV(std::ostream &out);
};


// -----------------------------------------------------------------------------------------------------
// Particle Fxn implementations:

template<int SpaceDim, int NumScalars, int... VectorVDims>
void Particle<SpaceDim,NumScalars,VectorVDims...>::Destroy()
{
   coords.Destroy();
   coords.SetSize(SpaceDim);
   for (int v = 0; v < vectors.size(); v++)
   {
      vectors[v].Destroy();
      vectors[v].SetSize(VDims[v]);
   }
   if (owning)
   {
      for (int s = 0; s < scalars.size(); s++)
         delete scalars[s];
   }
   for (int s = 0; s < scalars.size(); s++)
      scalars[s] = nullptr;
   
}

template<int SpaceDim, int NumScalars, int... VectorVDims>
void Particle<SpaceDim,NumScalars,VectorVDims...>::Copy(const Particle<SpaceDim,NumScalars,VectorVDims...> &p)
{
   Destroy();
   owning = p.owning;
   if (owning)
   {
      coords = Vector(p.coords);
      for (int s = 0; s < scalars.size(); s++)
         scalars[s] = new real_t(p.GetScalar(s));
      for (int v = 0; v < vectors.size(); v++)
         vectors[v] = Vector(p.vectors[v]);
   }
   else
   {
      coords = Vector(p.coords.GetData(), SpaceDim);
      for (int s = 0; s < scalars.size(); s++)
         scalars[s] = p.scalars[s];
      for (int v = 0; v < vectors.size(); v++)
         vectors[v] = Vector(p.vectors[v].GetData(), VDims[v]);
   }
}

template<int SpaceDim, int NumScalars, int... VectorVDims>
void Particle<SpaceDim,NumScalars,VectorVDims...>::Steal(Particle<SpaceDim,NumScalars,VectorVDims...> &p)
{
   Destroy();
   owning = p.owning;
   coords = std::move(p.coords);
   for (int v = 0; v < vectors.size(); v++)
   vectors[v] = std::move(p.vectors[v]);
   for (int s = 0; s < scalars.size(); s++)
   {
      if (owning)
         scalars[s] = std::exchange(p.scalars[s], nullptr);
      else
         scalars[s] = p.scalars[s];
   }
}


template<int SpaceDim, int NumScalars, int... VectorVDims>
Particle<SpaceDim,NumScalars,VectorVDims...>::Particle()
: owning(true)
{
   coords.SetSize(SpaceDim);
   coords = 0.0;

   // Initialize scalar ptrs
   for (int i = 0; i < scalars.size(); i++)
      scalars[i] = new real_t(0.0);

   // Initialize vectors
   for (int i = 0; i < vectors.size(); i++)
   {
      vectors[i].SetSize(VDims[i]);
      vectors[i] = 0.0;
   }
}

template<int SpaceDim, int NumScalars, int... VectorVDims>
Particle<SpaceDim,NumScalars,VectorVDims...>::Particle(real_t *in_coords, real_t *in_scalars[], real_t *in_vectors[])
: owning(false)
{
   coords = Vector(in_coords, SpaceDim);
   for (int i = 0; i < scalars.size(); i++)
      scalars[i] = in_scalars[i];
   
   for (int i = 0; i < vectors.size(); i++)
      vectors[i] = Vector(in_vectors[i], VDims[i]);
}


template<int SpaceDim, int NumScalars, int... VectorVDims>
bool Particle<SpaceDim,NumScalars,VectorVDims...>::operator==(const Particle<SpaceDim,NumScalars,VectorVDims...> &rhs) const
{
   bool equal = true;
   for (int d = 0; d < SpaceDim; d++)
   {
      if (GetCoords()[d] != rhs.GetCoords()[d])
         equal = false;
   }
   for (int s = 0; s < NumScalars; s++)
   {
      if (GetScalar(s) != rhs.GetScalar(s))
         equal = false;
   }
   for (int v = 0; v < sizeof...(VectorVDims); v++)
   {
      for (int c = 0; c < VDims[v]; c++)
      {
         if (GetVector(v)[c] != rhs.GetVector(v)[c])
            equal = false;
      }
   }
   return equal;
}


template<int SpaceDim, int NumScalars, int... VectorVDims>
void Particle<SpaceDim,NumScalars,VectorVDims...>::Print(std::ostream &out)
{
   out << "Coords: (";
   for (int d = 0; d < SpaceDim; d++)
      out << GetCoords()[d] << ( (d+1 < SpaceDim) ? "," : ")\n");
   for (int s = 0; s < NumScalars; s++)
      out << "Scalar " << s << ": " << GetScalar(s) << "\n";
   for (int v = 0; v < sizeof...(VectorVDims); v++)
   {
      out << "Vector " << v << ": (";
      for (int c = 0; c < VDims[v]; c++)
         out << GetVector(v)[c] << ( (c+1 < VDims[v]) ? "," : ")\n");
   }
}

// -----------------------------------------------------------------------------------------------------
// ParticleSet Fxn implementations:

template<int SpaceDim, int NumScalars, int... VectorVDims, Ordering::Type VOrdering>
void ParticleSet<Particle<SpaceDim,NumScalars,VectorVDims...>, VOrdering>::SyncVectors()
{
   // Reset Vector references to data
   for (int f = 0; f < TotalFields; f++)
   {
      fields[f] = Vector(data.data() + GetNP()*ExclScanFieldVDims[f], GetNP()*FieldVDims[f]);
   }
}

template<int SpaceDim, int NumScalars, int... VectorVDims, Ordering::Type VOrdering>
void ParticleSet<Particle<SpaceDim,NumScalars,VectorVDims...>, VOrdering>::AddParticle(const Particle<SpaceDim, NumScalars, VectorVDims...> &p, int id)
{
   int old_np = GetNP();

   if constexpr (VOrdering == Ordering::byNODES)
   {
      real_t dat;
      int offset = old_np;

      for (int f = 0; f < TotalFields; f++)
      {
         for (int c = 0; c < FieldVDims[f]; c++)
         {
            if (f == 0) // If processing coord comps
            {
               dat = p.GetCoords()[c];
            }
            else if (f - 1 < NumScalars) // Else if processing scalars
            {
               dat = p.GetScalar(f - 1);
            }
            else // Else processing vector comps
            {
               dat = p.GetVector(f - 1 - NumScalars)[c];
            }
            data.insert(data.begin() + offset, dat);
            offset += old_np + 1; // 1 to account for added data each loop iteration
         }
      }
   }
   else // byVDIM
   {
      const real_t* dat;
      for (int f = 0; f < TotalFields; f++)
      {
         if (f == 0)
         {
            dat = p.GetCoords().GetData();
         }
         else if (f - 1 < NumScalars)
         {
            dat = &p.GetScalar(f-1);
         }
         else
         {
            dat = p.GetVector(f-1-NumScalars).GetData();
         }
         data.insert(data.begin() + old_np*(ExclScanFieldVDims[f] + FieldVDims[f]) + ExclScanFieldVDims[f], dat, dat + FieldVDims[f]);
      }
   }

   ids.Append(id); // Add ID
   SyncVectors();
}

template<int SpaceDim, int NumScalars, int... VectorVDims, Ordering::Type VOrdering>
void ParticleSet<Particle<SpaceDim,NumScalars,VectorVDims...>, VOrdering>::RemoveParticles(const Array<int> &list)
{
   int old_np = GetNP();

   // Sort the indices
   Array<int> sorted_list(list);
   sorted_list.Sort();

   if constexpr (VOrdering == Ordering::byNODES)
   {
      int rm_count = 0;
      for (int i = sorted_list[0]; i < data.size(); i++)
      {
         if (i % GetNP() == sorted_list[rm_count % sorted_list.Size()])
         {
            rm_count++;
         }
         else
         {
            data[i-rm_count] = data[i]; // Shift elements rm_count
         }
      }
   }
   else // byVDIM
   {
      int rm_count = 0;

      int f = 0;
      for (int i = sorted_list[0]*FieldVDims[0]; i < data.size();  i++)
      {
         if (f + 1 < FieldVDims.size() && i == ExclScanFieldVDims[f+1]*GetNP())
         {
            f++;
         }

         int d_idx = (i - ExclScanFieldVDims[f]*GetNP())/FieldVDims[f];
         int s_idx = ((rm_count - ExclScanFieldVDims[f]*sorted_list.Size())/FieldVDims[f]);
         if (s_idx < sorted_list.Size() && d_idx == sorted_list[s_idx])
         {
            rm_count += FieldVDims[f];
            i += FieldVDims[f] - 1;
         }
         else
         {
            data[i-rm_count] = data[i];
         }
      }
   }

   // Remove old IDs
   int rm_idx = 0;
   for (int i = 0; i < old_np; i++)
   {
      if (rm_idx < sorted_list.Size() && i == sorted_list[rm_idx])
      {
         rm_idx++;
      }
      else
      {
         ids[i-rm_idx] = ids[i];
      }
   }

   // Resize / remove tails
   int num_new = old_np - list.Size();
   data.resize(num_new*TotalComps);
   ids.SetSize(num_new);

   SyncVectors();

}


template<int SpaceDim, int NumScalars, int... VectorVDims, Ordering::Type VOrdering>
Particle<SpaceDim, NumScalars, VectorVDims...> ParticleSet<Particle<SpaceDim,NumScalars,VectorVDims...>, VOrdering>::GetParticleRef(int i)
{
   static_assert(VOrdering == Ordering::byVDIM, "GetParticleRef is only available when ordering is byVDIM.");

   real_t *coords = &data[i*SpaceDim];

   real_t *scalars[NumScalars];
   for (int s = 0; s < NumScalars; s++) scalars[s] = &data[(s+SpaceDim)*GetNP() + i];

   real_t *vectors[sizeof...(VectorVDims)];
   for (int v = 0; v < sizeof...(VectorVDims); v++) vectors[v] = &data[ExclScanFieldVDims[v+1+NumScalars]*GetNP() + i*VDims[v]];

   return Particle<SpaceDim, NumScalars, VectorVDims...>(coords, scalars, vectors);
   
}

template<int SpaceDim, int NumScalars, int... VectorVDims, Ordering::Type VOrdering>
Particle<SpaceDim, NumScalars, VectorVDims...> ParticleSet<Particle<SpaceDim,NumScalars,VectorVDims...>, VOrdering>::GetParticleData(int i) const
{
   if constexpr(VOrdering == Ordering::byNODES)
   {
      real_t *dat;
      Particle<SpaceDim, NumScalars, VectorVDims...> p;
      for (int f = 0; f < TotalFields; f++)
      {
         for (int c = 0; c < FieldVDims[f]; c++)
         {
            if (f == 0)
            {
               dat = &p.GetCoords()[c];
            }
            else if (f-1 < NumScalars)
            {
               dat = &p.GetScalar(f-1);
            }
            else
            {
               dat = &p.GetVector(f-1-NumScalars)[c];
            }
            *dat = data[i+(c+ExclScanFieldVDims[f])*GetNP()];
         }
      }
      return Particle(p);
   }
   else // byVDIM
   {
      return Particle<SpaceDim, NumScalars, VectorVDims...>(GetParticleRef(i));
   }
   
}

template<int SpaceDim, int NumScalars, int... VectorVDims, Ordering::Type VOrdering>
void ParticleSet<Particle<SpaceDim,NumScalars,VectorVDims...>, VOrdering>::PrintPoint3D(std::ostream &os)
{
   // Write column headers
   os << "x y z id\n";

   // Write the data
   for (int i = 0 ; i < GetNP(); i++)
   {
      for (int d = 0; d < 3; d++)
      {
         real_t coord;
         if constexpr (VOrdering == Ordering::byNODES)
         {
            coord = (d < SpaceDim) ? data[i + d*GetNP()] : 0.0;
         }
         else
         {
            coord = (d < SpaceDim) ? data[d + i*SpaceDim] : 0.0;
         }
         os << ZeroSubnormal(coord) << " ";
      }
      os << ids[i] << "\n";
   }

}

template<int SpaceDim, int NumScalars, int... VectorVDims, Ordering::Type VOrdering>
void ParticleSet<Particle<SpaceDim,NumScalars,VectorVDims...>, VOrdering>::PrintCSV(std::ostream &os)
{
   std::array<char, 3> ax = {'x', 'y', 'z'};
   // Write column headers (i=-1), and data
   for (int i = -1; i < GetNP(); i++)
   {
      if (i == -1)
      {
         os << "id,";
      }
      else
      {
         os << ids[i] << ",";
      }
      for (int f = 0; f < TotalFields; f++)
      {
         for (int c = 0; c < FieldVDims[f]; c++)
         {

            if (i == -1)
            {
               if (f == 0)
               {
                  os << ax[c];
               }
               else if (f-1 < NumScalars)
               {
                  os << "Scalar_" << f-1;
               }
               else
               {
                  os << "Vector_" << f-1-NumScalars << "_" << c;
               }
            }
            else
            {
               real_t dat;
               if constexpr (VOrdering == Ordering::byNODES)
               {
                  dat = data[i + (ExclScanFieldVDims[f]+c)*GetNP()];
               }
               else
               {
                  dat = data[c + FieldVDims[f]*i + ExclScanFieldVDims[f]*GetNP()];
               }
               os << dat;
            }
            os << ((f+1 == TotalFields && c+1 == FieldVDims[f]) ? "\n" : ",");
         }
      }
   }
}



} // namespace mfem


#endif // MFEM_PARTICLESET