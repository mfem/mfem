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

#include "outils.hpp"

namespace mfem {
  Vector& GetOccaHostVector(const int id, const int64_t size) {
    static std::vector<Vector*> v;
    if (v.size() <= (size_t) id) {
      for (int i = (int) v.size(); i < (id + 1); ++i) {
        v.push_back(new Vector);
      }
    }
    if (size >= 0) {
      v[id]->SetSize(size);
    }
    return *(v[id]);
  }

  void OccaMult(const Operator &op,
                const OccaVector &x, OccaVector &y) {
    occa::device device = y.GetDevice();
    if (device.hasSeparateMemorySpace()) {
      Vector &hostX = GetOccaHostVector(0, op.Width());
      Vector &hostY = GetOccaHostVector(1, op.Height());
      hostX = x;
      op.Mult(hostX, hostY);
      y = hostY;
    } else {
      Vector hostX((double*) x.GetData().ptr(), x.Size());
      Vector hostY((double*) y.GetData().ptr(), y.Size());
      op.Mult(hostX, hostY);
    }
  }

  void OccaMultTranspose(const Operator &op,
                         const OccaVector &x, OccaVector &y) {
    occa::device device = y.GetDevice();
    if (device.hasSeparateMemorySpace()) {
      Vector &hostX = GetOccaHostVector(1, op.Height());
      Vector &hostY = GetOccaHostVector(0, op.Width());
      hostX = x;
      op.MultTranspose(hostX, hostY);
      y = hostY;
    } else {
      Vector hostX((double*) x.GetData().ptr(), x.Size());
      Vector hostY((double*) y.GetData().ptr(), y.Size());
      op.MultTranspose(hostX, hostY);
    }
  }
}

#endif
