// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_TENSOR_QDATA_TRAITS
#define MFEM_TENSOR_QDATA_TRAITS

namespace mfem
{

/// is_qdata
template <typename NotQData>
struct is_qdata_v
{
   static constexpr bool value = false;
};

template <typename QData>
constexpr bool is_qdata = is_qdata_v<QData>::value;

} // mfem namespace

#endif // MFEM_TENSOR_QDATA_TRAITS
