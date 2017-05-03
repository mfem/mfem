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

#include "ogridfunc.hpp"

namespace mfem {
  OccaGridFunction::OccaGridFunction() :
    OccaVector(),
    ofespace(NULL),
    sequence(0) {}

  OccaGridFunction::OccaGridFunction(OccaFiniteElementSpace *ofespace_) :
    OccaVector(ofespace_->GetGlobalDofs()),
    ofespace(ofespace_),
    sequence(0) {}

  OccaGridFunction::OccaGridFunction(occa::device device_,
                                     OccaFiniteElementSpace *ofespace_) :
    OccaVector(device_, ofespace_->GetGlobalDofs()),
    ofespace(ofespace_),
    sequence(0) {}

  OccaGridFunction::OccaGridFunction(const OccaGridFunction &v) :
    OccaVector(v),
    ofespace(v.ofespace),
    sequence(v.sequence) {}

  OccaGridFunction& OccaGridFunction::operator = (double value) {
    OccaVector::operator = (value);
    return *this;
  }
  OccaGridFunction& OccaGridFunction::operator = (const OccaVector &v) {
    OccaVector::operator = (v);
    return *this;
  }
  OccaGridFunction& OccaGridFunction::operator = (const OccaGridFunction &v) {
    OccaVector::operator = (v);
    return *this;
  }

  void OccaGridFunction::GetTrueDofs(OccaVector &v) const {
    const Operator *R = ofespace->GetRestrictionOperator();
    if (!R) {
      v = *this;
    } else {
      v.SetSize(R->Height());
      R->Mult(*this, v);
    }
  }

  void OccaGridFunction::SetFromTrueDofs(const OccaVector &v) {
    const Operator *P = ofespace->GetProlongationOperator();
    if (!P) {
      *this = v;
    } else {
      SetSize(P->Height());
      P->Mult(v, *this);
    }
  }

  void OccaGridFunction::ProjectCoefficient(OccaCoefficient &coeff) {}
}

#endif
