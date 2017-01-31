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

#include "otypes.hpp"

namespace mfem {
  template <>
  const std::string &typeToString<int>() {
    static std::string s = "int";
    return s;
  }

  template <>
  const std::string &typeToString<float>() {
    static std::string s = "float";
    return s;
  }

  template <>
  const std::string &typeToString<double>() {
    static std::string s = "double";
    return s;
  }
}

#endif