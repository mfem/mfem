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

#include <vector>
#include "../linalg/vector.hpp"

namespace mfem {
  inline Vector& GetOccaHostVector(const int id, const int64_t size = -1) {
    static std::vector<Vector*> v;
    if (v.size() <= id) {
      for (int i = (int) v.size(); i < (id + 1); ++i) {
        v.push_back(new Vector);
      }
    }
    if (size >= 0) {
      v[id]->SetSize(size);
    }
    return *(v[id]);
  }
}

#  endif
#endif
