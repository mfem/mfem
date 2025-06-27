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

#include "particleset.hpp"


namespace mfem
{

void Particle::Destroy()
{
   coords.Destroy();
   if (owning)
   {
      for (int s = 0; s < scalars.size(); s++)
         delete scalars[s];
   }
   scalars.resize(0);
   for (int v = 0; v < vectors.size(); v++)
   {
      vectors[v].Destroy();
   }
   vectors.resize(0);
}

void Particle::Copy(const Particle &p)
{
   Destroy();
   owning = p.owning;
   scalars.resize(p.GetNumScalars());
   vectors.resize(p.GetNumVectors());

   if (owning)
   {
      coords = Vector(p.coords);
      for (int s = 0; s < scalars.size(); s++)
         scalars[s] = new real_t(p.scalars[s]);
      for (int v = 0; v < vectors.size(); v++)
         vectors[v] = Vector(p.vectors[v]);
   }
   else // data refers to external data
   {
      coords = Vector(p.coords.GetData(), p.GetSpaceDim());
      for (int s = 0; s < scalars.size(); s++)
         scalars[s] = p.scalars[s];
      for (int v = 0; v < vectors.size(); v++)
         vectors[v] = Vector(p.vectors[v].GetData(), p.GetVDim(v));
   }
}

void Particle::Steal(Particle &p)
{
   Destroy();
   owning = p.owning;
   coords = std::move(p.coords);
   scalars.resize(p.GetNumScalars());
   vectors.resize(p.GetNumVectors());

   for (int s = 0; s < scalars.size(); s++)
   {
      if (owning)
         scalars[s] = std::exchange(p.scalars[s], nullptr);
      else
         scalars[s] = p.scalars[s];
   }
      for (int v = 0; v < vectors.size(); v++)
      vectors[v] = std::move(p.vectors[v]);
}

Particle::Particle(int spaceDim, int numScalars, const Array<int> &vectorVDims)
: owning(true)
{
   coords.SetSize(SpaceDim);
   coords = 0.0;

   // Initialize scalar ptrs
   scalars.resize(numScalars);
   for (int i = 0; i < scalars.size(); i++)
      scalars[i] = new real_t(0.0);

   // Initialize vectors
   vectors.resize(vectorVDims.Size());
   for (int i = 0; i < vectors.size(); i++)
   {
      vectors[i].SetSize(vectorVDims[i]);
      vectors[i] = 0.0;
   }
}

Particle::Particle(int spaceDim, int numScalars, const Array<int> &vectorVDims, real_t *in_coords, real_t *in_scalars[], real_t *in_vectors[])
: owning(false)
{
   coords = Vector(in_coords, spaceDim);

   scalars.resize(numScalars);
   for (int i = 0; i < scalars.size(); i++)
      scalars[i] = in_scalars[i];

   vectors.resize(vectorVDims);
   for (int i = 0; i < vectors.size(); i++)
      vectors[i] = Vector(in_vectors[i], vectorVDims[i]);
}


bool Particle::operator==(const Particle &rhs) const
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


} // namespace mfem