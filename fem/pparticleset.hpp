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

#include "particleset.hpp"

namespace mfem
{

template<typename T, Ordering::Type VOrdering>
class ParParticleSet : public ParticleSet<T, VOrdering> {};

template<int SpaceDim, int NumScalars, int... VectorVDims, Ordering::Type VOrdering>
class ParParticleSet<Particle<SpaceDim, NumScalars, VectorVDims...>, VOrdering> : public ParticleSet<Particle<SpaceDim, NumScalars, VectorVDims...>, VOrdering>
{
protected:
   MPI_Comm comm;
public:
   explicit ParParticleSet(MPI_Comm comm_)
   : ParticleSet<Particle<SpaceDim, NumScalars, VectorVDims...>, VOrdering>([&](){int s; MPI_Comm_size(comm_, &s); return s; }(), [&]() { int r; MPI_Comm_rank(comm_, &r); return r; }()),
     comm(comm_) {}

   /// Redistribute particles onto ranks specified in \p rank_list .
   void Redistribute(const Array<int> &rank_list);

   /// Parallel version of PrintCSV
   void PrintCSV(const char* fname, int precision=16) override;
};


template<int SpaceDim, int NumScalars, int... VectorVDims, Ordering::Type VOrdering>
void ParParticleSet<Particle<SpaceDim, NumScalars, VectorVDims...>, VOrdering>::PrintCSV(const char* fname, int precision)
{
   int rank; MPI_Comm_rank(comm, &rank);

   MPI_File file;
   MPI_File_open(comm, fname, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);

   std::stringstream ss_header;
   ParticleSet<Particle<SpaceDim, NumScalars, VectorVDims...>, VOrdering>::PrintCSVHeader(ss_header);
   std::string header = ss_header.str();

   // Print header
   if (rank == 0)
   {
      MPI_File_write_at(file, 0, header.data(), header.size(), MPI_CHAR, MPI_STATUS_IGNORE);
   }

   // Get data for each rank
   std::stringstream ss;
   ss.precision(precision);
   ParticleSet<Particle<SpaceDim, NumScalars, VectorVDims...>, VOrdering>::PrintCSV(ss, false);

   // Compute the size in bytes
   std::string s_dat = ss.str();
   MPI_Offset dat_size = s_dat.size();
   MPI_Offset offset;

   // int size; MPI_Comm_size(comm, &size);
   // for (int i = 0; i < size; i++)
   // {
   //    if (rank == i)
   //       std::cout << s_dat;
   //    MPI_Barrier(comm);
   // }
   
   // Compute the offsets using an exclusive scan
   MPI_Exscan(&dat_size, &offset, 1, MPI_OFFSET, MPI_SUM, comm);
   if (rank == 0)
   {
      offset = 0;
   }

   // Add offset from the header
   offset += header.size();

   // Write data collectively
   MPI_File_write_at_all(file, offset, s_dat.data(), dat_size, MPI_BYTE, MPI_STATUS_IGNORE);

   // Close file
   MPI_File_close(&file);


}


} // namespace mfem

#endif //MFEM_USE_MPI && MFEM_USE_GSLIB

#endif // MFEM_PPARTICLESET