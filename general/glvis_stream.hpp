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

#include "glvis_data.hpp"
#include "glvis_server.hpp"

namespace mfem
{

class glvis_stream : public std::iostream
{
   struct Impl
   {
      virtual ~Impl() = default;
      virtual size_t size() const = 0;
      virtual std::streambuf* get_buf() = 0;
      virtual std::streamsize precision() const = 0;
      virtual std::streamsize precision(std::streamsize) = 0;
      virtual int open(const char hostname[], int port) = 0;
      virtual bool is_open() const = 0;
      virtual int close() = 0;
      virtual void flush() = 0;
   };

   struct NullImpl: public Impl
   {
      // Null streambuf for no-op output: discards everything
      static class null_streambuf: public std::streambuf
      {
         int overflow(int c) override { return c; }
      } nullbuf;
      NullImpl();
      size_t size() const override { return 0; }
      std::streambuf* get_buf() override { return &nullbuf; }
      std::streamsize precision() const override { return 0; }
      std::streamsize precision(std::streamsize new_prec) override { return new_prec; }
      int open(const char hostname[], int port) override { return -1; }
      bool is_open() const override { return false; }
      int close() override { return -1; }
      void flush() override { }
   };

   struct SerialImpl : public Impl
   {
      std::shared_ptr<GLVisData> data;
      char_stream_uptr stream;
      SerialImpl(std::shared_ptr<GLVisData> data);
      ~SerialImpl() override;
      size_t size() const override;
      std::streambuf *get_buf() override;
      std::streamsize precision() const override;
      std::streamsize precision(std::streamsize) override;
      int open(const char[], int) override { return 0; }
      bool is_open() const override { return true; }
      int close() override { return 0; }
      void flush() override;
   };

public:
   glvis_stream();

   glvis_stream(glvis_stream &&) = delete;
   glvis_stream(const glvis_stream &) = delete;
   glvis_stream &operator=(const glvis_stream &) = delete;
   glvis_stream &operator=(glvis_stream &&) = delete;

   void Flush();

   virtual ~glvis_stream();

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
   int open(const char hostname[], int port);

   bool is_open() const;

   int close();

private:
   std::shared_ptr<GLVisData> data;
   std::unique_ptr<Impl> impl;
   GLVisServer glvis;
};

} // namespace mfem
