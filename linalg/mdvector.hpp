// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_MDVECTOR
#define MFEM_MDVECTOR

#include "../config/config.hpp"

#include "vector.hpp"
#include "general/mdspan.hpp"

namespace mfem
{

template<int N, typename Layout = MDLayoutLeft<N>>
struct MDVector : public MDSpan<Vector, N, Layout>
{
   using base_t = MDSpan<Vector, N, Layout>;

   /**
    * @brief MDVector default constructor (recursion)
    */
   MDVector(): base_t() { }

   /**
    * @brief MDVector recursion constructor
    * @param[in] n Dimension indice
    * @param[in] args Rest of dimension indices
    */
   template <typename... Ts>
   MDVector(int n, Ts... args): MDVector(args...) { base_t::Setup(n, args...); }

   /// Move constructor not supported
   MDVector(MDVector&&) = delete;

   /// Copy constructor not supported
   MDVector(const MDVector&) = delete;

   /// Move assignment not supported
   MDVector& operator=(MDVector&&) = delete;

   /// Copy assignment not supported
   MDVector& operator=(const MDVector&) = delete;

   using Vector::Read;
   using Vector::Write;
   using Vector::ReadWrite;
   using Vector::HostRead;
   using Vector::HostWrite;
   using Vector::HostReadWrite;

   using Vector::GetData;
   using Vector::SetData;

   using Vector::operator=;
};

} // namespace mfem

#endif // MFEM_MDVECTOR
