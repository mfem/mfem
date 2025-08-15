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
#include "../linalg/linalg.hpp"


namespace mfem
{


/// Container for data associated with a single particle. See \ref ParticleSet for more information.
class Particle
{
protected:
   Vector coords;
   std::vector<Vector> fields;
   std::vector<Memory<int>> tags;
public:

   /** @brief Construct a Particle instance
    *  @param[in] dim          Spatial dimension.
    *  @param[in] field_vdims  Vector dimensions of particle fields.
    *  @param[in] num_tags     Number of integer tags.
    */
   Particle(int dim, const Array<int> &field_vdims, int num_tags);

   /// Get the spatial dimension of this particle.
   int GetDim() const { return coords.Size(); }

   /// Get the number of fields associated with this particle.
   int GetNF() const { return fields.size(); }

   /// Get the vector dimension of field \p f .
   int FieldVDim(int f) const { return fields[f].Size(); }

   /// Get the number of tags associated with this particle.
   int GetNT() const { return tags.size(); }

   /// Get reference to particle coordinates Vector.
   Vector& Coords() { return coords; }

   /// Get const reference to particle coordinates Vector.
   const Vector& Coords() const { return coords; }

   /// Get reference to field \p f , component \p c value.
   real_t& FieldValue(int f, int c=0) { return fields[f][c]; }

   /// Get const reference to field \p f , component \p c value.
   const real_t& FieldValue(int f, int c=0) const { return fields[f][c]; }

   /// Get reference to field \p f Vector.
   Vector& Field(int f) { return fields[f]; }

   /// Get const reference to field \p f Vector.
   const Vector& Field(int f) const { return fields[f]; }

   /// Get reference to tag \p t .
   int& Tag(int t) { return tags[t][0]; }

   /// Get const reference to tag \p t .
   const int& Tag(int t) const { return tags[t][0]; }

   /// Get reference to underlying memory object of tag \p t .
   Memory<int>& TagMemory(int t) { return tags[t]; }

   /// Particle equality operator.
   bool operator==(const Particle &rhs) const;

   bool operator!=(const Particle &rhs) const { return !operator==(rhs); }

   /// Print all particle data to \p out .
   void Print(std::ostream &out=mfem::out) const;

};


} // namespace mfem


#endif // MFEM_PARTICLESET