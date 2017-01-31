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

// This file is only to separate declarations and definitions
// ovector.tpp is included at the end of ovector.hpp

#include "../config/config.hpp"

#ifdef MFEM_USE_OCCA
#  ifndef MFEM_OTYPES
#  define MFEM_OTYPES

#include <iostream>

#include "error.hpp"

namespace mfem {
  template <class TM>
  const std::string &typeToString() {
    static std::string s = "";
    mfem_error("Can't stringify type");
    return s;
  }

  template <>
  const std::string &typeToString<int>();

  template <>
  const std::string &typeToString<float>();

  template <>
  const std::string &typeToString<double>();
}

#  endif
#endif