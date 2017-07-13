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
  std::map<std::string, occa::kernel> gridFunctionKernels;

  occa::kernel GetGridFunctionKernel(OccaIntegrator &integ) {
    occa::device device = integ.GetDevice();
    const int numQuad = integ.GetIntegrationRule().GetNPoints();

    const FiniteElementSpace &fespace = integ.GetTrialFESpace();
    const FiniteElement &fe = *(fespace.GetFE(0));
    const int dim = fe.GetDim();
    const int vdim = fespace.GetVDim();

    std::stringstream ss;
    ss << occa::hash(device)
       << "FEColl : " << fespace.FEColl()->Name()
       << "Quad: "    << numQuad
       << "Dim: "     << dim
       << "VDim: "    << vdim;
    std::string hash = ss.str();

    // Kernel defines
    occa::properties props;
    props["defines/NUM_VDIM"] = vdim;
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
    OccaVector(ofespace_->GetVSize()),
    ofespace(ofespace_),
    sequence(0) {}

  OccaGridFunction::OccaGridFunction(occa::device device_,
                                     OccaFiniteElementSpace *ofespace_) :
    OccaVector(device_, ofespace_->GetVSize()),
    ofespace(ofespace_),
    sequence(0) {}

  OccaGridFunction::OccaGridFunction(OccaFiniteElementSpace *ofespace_,
                                     OccaVectorRef ref) :
    OccaVector(ref),
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

  OccaGridFunction& OccaGridFunction::operator = (const OccaVectorRef &v) {
    OccaVector::operator = (v);
    return *this;
  }

  OccaGridFunction& OccaGridFunction::operator = (const OccaGridFunction &v) {
    OccaVector::operator = (v);
    return *this;
  }

  void OccaGridFunction::SetGridFunction(GridFunction &gf) {
    Vector v = *this;
    gf.MakeRef(ofespace->GetFESpace(), v, 0);
    // Make gf the owner of the data
    v.Swap(gf);
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

  FiniteElementSpace* OccaGridFunction::GetFESpace() {
    return ofespace->GetFESpace();
  }

  const FiniteElementSpace* OccaGridFunction::GetFESpace() const {
    return ofespace->GetFESpace();
  }

  void OccaGridFunction::ToQuad(OccaIntegrator &integ,
                                OccaVector &quadValues) {

    occa::device device = integ.GetDevice();

    OccaDofQuadMaps &maps = integ.GetDofQuadMaps();

    const FiniteElementSpace &fespace = integ.GetTrialFESpace();
    const FiniteElement &fe = *(fespace.GetFE(0));

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
