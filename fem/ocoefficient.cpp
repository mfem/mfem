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

#include "ocoefficient.hpp"

namespace mfem {
  //---[ Parameter ]------------
  void OccaParameter::SetProps(occa::properties &props) {}

  occa::kernelArg OccaParameter::KernelArgs() {
    return occa::kernelArg();
  }
  //====================================

  //---[ Include Parameter ]------------
  OccaIncludeParameter::OccaIncludeParameter(const std::string &filename_) :
    filename(filename_) {}

  void OccaIncludeParameter::SetProps(occa::properties &props) {
    props["headers"].asArray() += "#include " + filename;
  }
  //====================================

  //---[ Source Parameter ]------------
  OccaSourceParameter::OccaSourceParameter(const std::string &source_) :
    source(source_) {}

  void OccaSourceParameter::SetProps(occa::properties &props) {
    props["headers"].asArray() += source;
  }
  //====================================

  //---[ Coefficient ]------------------
  OccaCoefficient::OccaCoefficient() {}

  OccaCoefficient::OccaCoefficient(const double value) {
    props["COEFF_ARGS"] = "";
    props["COEFF"]      = value;
  }

  OccaCoefficient::OccaCoefficient(const std::string &function) {
    props["COEFF_ARGS"] = "";
    props["COEFF"]      = function;
  }

  OccaCoefficient::OccaCoefficient(const OccaCoefficient &coeff) :
    params(coeff.params),
    props(coeff.props) {}

  void OccaCoefficient::SetCoeffProps(occa::properties &props_) {
    props_ += props;
  }

  occa::kernelArg OccaCoefficient::CoeffKernelArg() {
    return occa::kernelArg();
  }

  OccaCoefficient& OccaCoefficient::With(OccaParameter *param) {
    params.push_back(param);
    return *this;
  }

  void OccaCoefficient::SetProps(occa::properties &props_) {
    const int paramCount = (int) params.size();
    for (int i = 0; i < paramCount; ++i) {
      params[i]->SetProps(props_);
    }
    SetCoeffProps(props_);
  }

  OccaCoefficient::operator occa::kernelArg () {
    occa::kernelArg kArg;
    const int paramCount = (int) params.size();
    for (int i = 0; i < paramCount; ++i) {
      kArg.add(params[i]->KernelArgs());
    }
    kArg.add(CoeffKernelArg());
    return kArg;
  }
  //====================================
}

#endif
