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

#include <iostream>
#include <memory>

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

#include "../../general/glvis/data.hpp"

namespace mfem
{

class glvis_stream : public std::iostream
{
public:
   struct IBase
   {
      virtual ~IBase() = default;
      virtual size_t size() const = 0;
      virtual std::streambuf* get_buf() = 0;
      virtual std::streamsize precision() const = 0;
      virtual std::streamsize precision(std::streamsize) = 0;
      virtual int open(const char hostname[], int port) = 0;
      virtual bool is_open() const = 0;
      virtual int close() = 0;
      virtual void flush() = 0;
   };

   //////////////////////////////////////////////////////////////////
   struct SerialImpl : public IBase
   {
      const std::shared_ptr<GLVisData> &data;
      SerialImpl(const std::shared_ptr<GLVisData> &data): data(data)
      {
         data->stream.clear();
      }
      ~SerialImpl() override {}
      size_t size() const override { return data->stream.tellp(); }
      std::streambuf *get_buf() override { return data->stream.rdbuf(); }
      std::streamsize precision() const override { return data->stream.precision(); }
      std::streamsize precision(std::streamsize prec) override { return data->stream.precision(prec); }
      int open(const char[], int) override { return 0; }
      bool is_open() const override { return true; }
      int close() override { return 0; }
      void flush() override { data->stream.flush(); }
   };

#ifdef MFEM_USE_MPI
   //////////////////////////////////////////////////////////////////
   struct ParallelImpl : public IBase
   {
      const std::shared_ptr<GLVisData> &data;
      ParallelImpl(const std::shared_ptr<GLVisData> &data): data(data)
      {
         data->stream.clear();
      }
      ~ParallelImpl() override { }
      size_t size() const override { return data->stream.tellp(); }
      std::streamsize precision() const override { return data->stream.precision(); }
      std::streamsize precision(std::streamsize prec) override { return data->stream.precision(prec); }
      std::streambuf *get_buf() override { return data->stream.rdbuf(); }
      int open(const char hostname[], int port) override { return 0; }
      bool is_open() const override { return true; }
      int close() override { return 0; }
      void flush() override { data->stream.flush(); }
   };
#endif // MFEM_USE_MPI

   int MpiSize() const;
   int MpiRank() const;
   inline bool Root() const { return MpiRank() == 0; }

public:
   explicit glvis_stream(const char*, int, int rank = -1);

   glvis_stream(glvis_stream &&) = delete;
   glvis_stream(const glvis_stream &) = delete;
   glvis_stream &operator=(const glvis_stream &) = delete;
   glvis_stream &operator=(glvis_stream &&) = delete;

   virtual ~glvis_stream() {}

   std::streamsize precision() const { return impl->precision(); }
   std::streamsize precision(std::streamsize new_prec) { return impl->precision(new_prec); }

   using ostream_manipulator = std::ostream& (*)(std::ostream&);
   glvis_stream& operator<<(ostream_manipulator pf);

   template<typename T>
   glvis_stream& operator<<(const T& val)
   {
      static_cast<std::ostream&>(*this) << val;
      return *this;
   }

   int open(const char hostname[], int port) { return 0; }

   bool is_open() const { return true; }

   int close() { return 0; }

   void flush();

   void glvis();

private:
   const bool mpi_initialized;
   const int mpi_size, mpi_rank;
   const bool serial, mpi_root;
   std::shared_ptr<GLVisData> data;
   std::unique_ptr<IBase> impl;
};

} // namespace mfem
