// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_SCALARS_HPP
#define MFEM_SCALARS_HPP

#include "../config/config.hpp"
#include <complex>

namespace mfem
{

/// Map a scalar type to a scalar type Id.
template <typename T> struct ScalarId;

template<> struct ScalarId<double>
{
   static const int value = 0;
};

template<> struct ScalarId<std::complex<double> >
{
   static const int value = 1;
};


/// Basic operations on scalars
template <typename T> struct ScalarOps;

template <> struct ScalarOps<double>
{
   static inline const double &conj(const double &a) { return a; }
};

template <> struct ScalarOps<std::complex<double> >
{
   static inline std::complex<double> conj(const std::complex<double> &a)
   { return std::conj(a); }
};

} // namespace mfem

#endif // MFEM_SCALARS_HPP
