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
  OccaParameter::~OccaParameter() {}

  void OccaParameter::SetProps(occa::properties &props) {}

  occa::kernelArg OccaParameter::KernelArgs() {
    return occa::kernelArg();
  }
  //====================================

  //---[ Include Parameter ]------------
  OccaIncludeParameter::OccaIncludeParameter(const std::string &filename_) :
    filename(filename_) {}

  OccaParameter* OccaIncludeParameter::Clone() {
    return new OccaIncludeParameter(filename);
  }

  void OccaIncludeParameter::SetProps(occa::properties &props) {
    props["headers"].asArray() += "#include " + filename;
  }
  //====================================

  //---[ Source Parameter ]------------
  OccaSourceParameter::OccaSourceParameter(const std::string &source_) :
    source(source_) {}

  OccaParameter* OccaSourceParameter::Clone() {
    return new OccaSourceParameter(source);
  }

  void OccaSourceParameter::SetProps(occa::properties &props) {
    props["headers"].asArray() += source;
  }
  //====================================

  //---[ Coefficient ]------------------
  OccaCoefficient::OccaCoefficient(const double value) :
    name("COEFF") {
    coeffValue = value;
    coeffArgs  = "";
  }

  OccaCoefficient::OccaCoefficient(const std::string &source) :
    name("COEFF") {
    coeffValue = source;
    coeffArgs  = "";
  }

  OccaCoefficient::OccaCoefficient(const OccaCoefficient &coeff) :
    name(coeff.name),
    coeffValue(coeff.coeffValue),
    coeffArgs(coeff.coeffArgs) {

    const int paramCount = (int) coeff.params.size();
    for (int i = 0; i < paramCount; ++i) {
      params.push_back(coeff.params[i]->Clone());
    }
  }

  OccaCoefficient::~OccaCoefficient() {
    const int paramCount = (int) params.size();
    for (int i = 0; i < paramCount; ++i) {
      delete params[i];
    }
  }

  OccaCoefficient& OccaCoefficient::SetName(const std::string &name_) {
    name = name_;
    return *this;
  }

  OccaCoefficient& OccaCoefficient::IncludeHeader(const std::string &filename) {
    params.push_back(new OccaIncludeParameter(filename));
    return *this;
  }

  OccaCoefficient& OccaCoefficient::IncludeSource(const std::string &source) {
    params.push_back(new OccaSourceParameter(source));
    return *this;
  }

  OccaCoefficient& OccaCoefficient::SetProps(occa::properties &props) {
    const int paramCount = (int) params.size();
    for (int i = 0; i < paramCount; ++i) {
      params[i]->SetProps(props);
    }
    props[name]           = coeffValue;
    props[name + "_ARGS"] = coeffArgs;
    return *this;
  }

  OccaCoefficient::operator occa::kernelArg () {
    occa::kernelArg kArg;
    const int paramCount = (int) params.size();
    for (int i = 0; i < paramCount; ++i) {
      kArg.add(params[i]->KernelArgs());
    }
    return kArg;
  }
  //====================================
}

#endif
