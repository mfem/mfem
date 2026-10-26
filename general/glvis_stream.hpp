// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

namespace mfem
{

struct glvis_data
{
   bool serial;
   int mpi_size;
   bool mpi_root;
#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif
   std::stringstream stream;
   std::vector<std::unique_ptr<std::istream>> streams;
   int total_size;
   std::vector<int> offsets;
   std::string type;

#ifdef MFEM_USE_MPI
   explicit glvis_data(MPI_Comm comm)
      : serial(true), mpi_size(1), mpi_root(true), comm(comm),
        total_size(0), type({}) {}
#else
   glvis_data()
      : serial(true), mpi_size(1), mpi_root(true),
        total_size(0), type({}) {}
#endif
};

/// Gathers one GLVis payload. Rank 0 opens the window.
///
/// flush() and '<< std::flush' are collective. Every rank of the stored
/// communicator enters them together. Rank 0 blocks in the window. The other
/// ranks wait until it returns. A failure is raised on every rank.
/// That communicator is MPI_COMM_WORLD unless the caller passes one. An
/// explicit communicator is borrowed, not duplicated, so the caller keeps
/// it alive until this stream is destroyed. Ranks that do not flush pass a
/// subcommunicator of their own. A flush by only some ranks waits forever.
/// std::endl and a flush through std::ostream only sync this buffer: flush()
/// is not virtual, and Mesh::Print calls it on an ostream reference while the
/// mesh is still incomplete.
class glvis_stream: public std::iostream
{
   glvis_data data;
   bool visualize = true;
   void SyncComm();
   void serialize();

public:
   /// Stores MPI_COMM_WORLD when MPI is on. flush() is collective there.
   glvis_stream();
#ifdef MFEM_USE_MPI
   /// Borrows @a comm (not duplicated). Keep @a comm alive until this
   /// stream is destroyed. A subset of ranks passes its own subcommunicator.
   /// A flush by only some ranks of @a comm waits forever.
   explicit glvis_stream(MPI_Comm comm);
#endif

   glvis_stream(glvis_stream &&) = delete;
   glvis_stream(const glvis_stream &) = delete;
   glvis_stream &operator=(const glvis_stream &) = delete;
   glvis_stream &operator=(glvis_stream &&) = delete;

   ~glvis_stream() = default;

   size_t size() { return data.stream.tellp(); }
   std::streamsize precision() const { return std::iostream::precision(); }
   std::streamsize precision(std::streamsize new_prec)
   { return std::iostream::precision(new_prec); }

   using ostream_manipulator = std::ostream& (*)(std::ostream&);
   glvis_stream& operator<<(ostream_manipulator pf);

   template<typename T>
   glvis_stream& operator<<(const T& val)
   {
      static_cast<std::ostream&>(*this) << val;
      return *this;
   }

   /// Collective gather. Rank 0 then blocks in the GLVis window.
   ///
   /// Each payload starts with mesh, solution, or parallel. A parallel
   /// payload continues with the communicator size and the rank, as in
   /// "parallel 2 0". flush() consumes the buffer, so size() is 0
   /// afterwards. An empty or unknown header aborts on every rank.
   void flush();

   /// When @a on is false, flush gathers and does not open a window.
   void SetVisualize(bool on) { visualize = on; }

   void reset()
   {
      data.stream.str(std::string());
      data.stream.clear();
      std::iostream::clear();
      data.stream.seekg(0, std::ios::beg);
      data.stream.seekp(0, std::ios::beg);
      std::iostream::seekg(0, std::ios::beg);
      std::iostream::seekp(0, std::ios::beg);
   }

   void operator()();
};

} // namespace mfem
