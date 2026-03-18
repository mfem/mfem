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
#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace mfem
{

struct GLVisData
{
   const bool serial;
   const int mpi_size, mpi_rank, mpi_root;
   std::stringstream stream;
   std::vector<std::unique_ptr<std::stringstream>> streams;
   int total_size;
   std::vector<int> offsets;
   std::string type;

   GLVisData(const bool serial, const size_t size,
             const size_t rank, const bool root):
      serial(serial), mpi_size(size), mpi_rank(rank), mpi_root(root),
      stream(), streams(), total_size(0), offsets(), type({}) {}
};

class glvis_stream : public std::iostream
{
   void MpiGather();

public:
   explicit glvis_stream(const char*, int, int rank = -1);

   glvis_stream(glvis_stream &&) = delete;
   glvis_stream(const glvis_stream &) = delete;
   glvis_stream &operator=(const glvis_stream &) = delete;
   glvis_stream &operator=(glvis_stream &&) = delete;

   virtual ~glvis_stream() {}

   size_t size() { return data.stream.tellp(); }
   std::streamsize precision() const { return std::iostream::precision(); }
   std::streamsize precision(std::streamsize new_prec) { return std::iostream::precision(new_prec); }

   using ostream_manipulator = std::ostream& (*)(std::ostream&);
   glvis_stream& operator<<(ostream_manipulator pf);

   template<typename T>
   glvis_stream& operator<<(const T& val)
   {
      static_cast<std::ostream&>(*this) << val;
      return *this;
   }

   int open(const char *hostname, int port) { return 0; }

   bool is_open() const { return true; }

   int close() { return 0; }

   void flush() { std::iostream::flush();}

   void reset()
   {
      data.stream.clear();
      data.stream.seekg(0, std::ios::beg);
      data.stream.seekp(0, std::ios::beg);
   }

   void glvis();

private:
   GLVisData data;
};

} // namespace mfem
