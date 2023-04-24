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

#ifndef MFEM_MDARRAY
#define MFEM_MDARRAY

#include "../config/config.hpp"

#include "array.hpp"
#include "mdspan.hpp"

namespace mfem
{

template<typename T, int N, typename Layout = MDLayoutLeft<N>>
struct MDArray : public MDSpan<Array<T>, N, Layout>
{
   using base_t = MDSpan<Array<T>, N, Layout>;

   /**
    * @brief MDArray default constructor (recursion)
    */
   MDArray(): base_t() { }

   /**
    * @brief MDArray recursion constructor
    * @param[in] n Dimension indice
    * @param[in] args Rest of dimension indices
    */
   template <typename... Ts>
   MDArray(int n, Ts... args): MDArray(args...) { base_t::Setup(n, args...); }

   /// Move constructor not supported
   MDArray(MDArray&&) = delete;

   /// Copy constructor not supported
   MDArray(const MDArray&) = delete;

   /// Move assignment not supported
   MDArray& operator=(MDArray&&) = delete;

   /// Copy assignment not supported
   MDArray& operator=(const MDArray&) = delete;

   using Array<T>::Read;
   using Array<T>::Write;
   using Array<T>::ReadWrite;
   using Array<T>::HostRead;
   using Array<T>::HostWrite;
   using Array<T>::HostReadWrite;

   using Array<T>::Assign;
   using Array<T>::Print;

   using Array<T>::GetData;

   using Array<T>::operator=;
};

} // namespace mfem

#endif // MFEM_MDARRAY
