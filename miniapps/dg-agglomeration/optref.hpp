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

#ifndef MFEM_OPTREF_HPP
#define MFEM_OPTREF_HPP

#include "mfem.hpp"

namespace mfem
{

class NullOptT { };

template <typename T>
class OptRef
{
   T *ptr = nullptr;
public:
   OptRef() = default;
   OptRef(NullOptT) { }
   OptRef(T &t) : ptr(&t) { }
   operator bool() const { return ptr; }
   T &operator*() const { return *ptr; }
   T *operator->() const { return ptr; }
};

static constexpr NullOptT NullOpt;

} // namespace mfem

#endif
