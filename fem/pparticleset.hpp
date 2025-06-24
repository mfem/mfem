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

#ifndef MFEM_PPARTICLESET
#define MFEM_PPARTICLESET


#if defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB)

template<int SpaceDim, int NumScalars, int... VectorVDims, Ordering::Type VOrdering>
class ParParticleSet : public ParticleSet<Particle<SpaceDim, NumScalars, VectorVDims...>, VOrdering>
{
protected:
   MPI_Comm comm;

public:
   explicit ParticleSet(MPI_Comm comm_)
   : ParticleSet([&](){int s; MPI_Comm_size(comm_, &s); return s; }(), [&]() { int r; MPI_Comm_rank(comm_, &r); return r; }()),
     comm(comm_) {}

   /// Redistribute particles onto ranks specified in \p rank_list .
   void Redistribute(const Array<int> &rank_list);

   void SaveAsSerial(const char* fname, int precision=16, int save_rank = 0) const;
};

#endif //MFEM_USE_MPI && MFEM_USE_GSLIB

#endif // MFEM_PPARTICLESET