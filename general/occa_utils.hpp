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

#include "../config/config.hpp"

#ifdef MFEM_USE_OCCA
#  ifndef MFEM_OCCA_UTILS
#  define MFEM_OCCA_UTILS

#include "array.hpp"

#include "occa.hpp"

namespace mfem {
  template <class TM>
  occa::array<TM> occafy(occa::device device, const Array<TM> &array) {
    occa::array<TM> ret(device, array.Size());
    occa::memcpy(ret.memory(), array.GetData());
    return ret;
  }

  template <class TM>
  occa::array<TM> occafy(const Array<TM> &array) {
    occa::array<TM> ret(occa::currentDevice(), array.Size());
    occa::memcpy(ret.memory(), array.GetData());
    return ret;
  }
}

#  endif
#endif