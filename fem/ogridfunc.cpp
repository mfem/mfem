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
#include "obilininteg.hpp"

namespace mfem {
  std::map<occa::hash_t, occa::kernel> gridFunctionKernels;

  occa::kernel GetGridFunctionKernel(OccaIntegrator &integ) {
    occa::device device = integ.GetDevice();
    const int numQuad = integ.GetIntegrationRule().GetNPoints();

    const FiniteElementSpace &fespace = integ.GetFESpace();
    const FiniteElement &fe = *(fespace.GetFE(0));
    const int dim = fe.GetDim();

    occa::hash_t hash = (occa::hash(device)
                         ^ ("FEColl : " + std::string(fespace.FEColl()->Name()))
                         ^ ("Quad   : " + occa::toString(numQuad))
                         ^ ("Dim    : " + occa::toString(dim)));

    // DofToQuad
    OccaDofQuadMaps &maps = integ.GetDofQuadMaps();

    // Kernel defines
    occa::properties props;
    integ.SetupProperties(props);

    occa::kernel kernel = gridFunctionKernels[hash];
    if (!kernel.isInitialized()) {
      kernel = device.buildKernel("occa://mfem/fem/gridfunc.okl",
                                  stringWithDim("GridFuncToQuad", dim),
                                  props);
    }
    return kernel;
  }

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
      v.NewDataAndSize(data, size);
    } else {
      v.SetSize(data.getDevice(), R->Height());
      R->Mult(*this, v);
    }
  }

  void OccaGridFunction::SetFromTrueDofs(const OccaVector &v) {
    const Operator *P = ofespace->GetProlongationOperator();
    if (!P) {
      NewDataAndSize(v.GetData(), v.Size());
    } else {
      SetSize(v.GetDevice(), P->Height());
      P->Mult(v, *this);
    }
  }

  void OccaGridFunction::ToQuad(OccaIntegrator &integ,
                                OccaVector &quadValues) {

    occa::device device = integ.GetDevice();

    OccaDofQuadMaps &maps = integ.GetDofQuadMaps();

    const FiniteElementSpace &fespace = integ.GetFESpace();
    const FiniteElement &fe = *(fespace.GetFE(0));

    const int dim      = fe.GetDim();
    const int elements = fespace.GetNE();
    const int numQuad  = integ.GetIntegrationRule().GetNPoints();
    quadValues.SetSize(device,
                       numQuad * elements);

    occa::kernel gridFuncToQuad = GetGridFunctionKernel(integ);
    gridFuncToQuad(elements,
                   maps.dofToQuad,
                   ofespace->GetLocalToGlobalMap(),
                   *this,
                   quadValues);
  }
}

#endif
