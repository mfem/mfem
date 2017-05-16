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
#include "obilininteg.hpp"

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

  //---[ Vector Parameter ]-------
  OccaVectorParameter::OccaVectorParameter(const std::string &name_,
                                           OccaVector &v_,
                                           const bool useRestrict_) :
    name(name_),
    v(v_),
    attr(""),
    useRestrict(useRestrict_) {}

  OccaVectorParameter::OccaVectorParameter(const std::string &name_,
                                           OccaVector &v_,
                                           const std::string &attr_,
                                           const bool useRestrict_) :
    name(name_),
    v(v_),
    attr(attr_),
    useRestrict(useRestrict_) {}

  OccaParameter* OccaVectorParameter::Clone() {
    return new OccaVectorParameter(name, v, attr, useRestrict);
  }

  void OccaVectorParameter::SetProps(occa::properties &props) {
    std::string &args = (props["defines/COEFF_ARGS"]
                         .asString()
                         .string());
    args += "const double *";
    if (useRestrict) {
      args += " restrict ";
    }
    args += name;
    if (attr.size()) {
      args += ' ';
      args += attr;
    }
    args += ",\n";
  }

  occa::kernelArg OccaVectorParameter::KernelArgs() {
    return occa::kernelArg(v);
  }
  //====================================

  //---[ GridFunction Parameter ]-------
  occa::kernel OccaGridFunctionParameter::gridFuncToQuad[3];

  OccaGridFunctionParameter::OccaGridFunctionParameter(const std::string &name_,
                                                       OccaVector &gf_,
                                                       const bool useRestrict_) :
    name(name_),
    gf(gf_),
    useRestrict(useRestrict_) {}

  OccaParameter* OccaGridFunctionParameter::Clone() {
    OccaGridFunctionParameter *param =
      new OccaGridFunctionParameter(name, gf, useRestrict);
    param->gfQuad = gfQuad;
    return param;
  }

  void OccaGridFunctionParameter::Setup(OccaIntegrator &integ) {
    occa::device device = integ.GetDevice();

    OccaDofQuadMaps &maps = integ.GetDofQuadMaps();

    const FiniteElementSpace &fespace = integ.GetFESpace();
    const FiniteElement &fe = *(fespace.GetFE(0));

    const int dim      = fe.GetDim();
    const int elements = fespace.GetNE();
    const int numQuad  = integ.GetIntegrationRule().GetNPoints();
    gfQuad.SetSize(device,
                   numQuad * elements);

    if (!gridFuncToQuad[dim].isInitialized()) {
      gridFuncToQuad[dim] = device.buildKernel("occa://mfem/fem/gridfunc.okl",
                                               stringWithDim("GridFuncToQuad", dim));
    }
    gridFuncToQuad[dim](elements, maps.dofToQuad, gf, gfQuad);
  }

  void OccaGridFunctionParameter::SetProps(occa::properties &props) {
      std::string &args = (props["defines/COEFF_ARGS"]
                           .asString()
                           .string());
      args += "const double *";
      if (useRestrict) {
        args += " restrict ";
      }
      args += name;
      args += " @(NUM_QUAD, numElements),\n";
  }

  occa::kernelArg OccaGridFunctionParameter::KernelArgs() {
    return occa::kernelArg(gf);
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

  void OccaCoefficient::Setup(OccaIntegrator &integ) {
    const int paramCount = (int) params.size();
    for (int i = 0; i < paramCount; ++i) {
      params[i]->Setup(integ);
    }
  }

  OccaCoefficient& OccaCoefficient::Add(OccaParameter *param) {
    params.push_back(param);
    return *this;
  }

  OccaCoefficient& OccaCoefficient::IncludeHeader(const std::string &filename) {
    return Add(new OccaIncludeParameter(filename));
  }

  OccaCoefficient& OccaCoefficient::IncludeSource(const std::string &source) {
    return Add(new OccaSourceParameter(source));
  }

  OccaCoefficient& OccaCoefficient::AddVector(const std::string &name_,
                                              OccaVector &v,
                                              const bool useRestrict) {
    return Add(new OccaVectorParameter(name_, v, useRestrict));
  }

  OccaCoefficient& OccaCoefficient::AddVector(const std::string &name_,
                                              OccaVector &v,
                                              const std::string &attr,
                                              const bool useRestrict) {
    return Add(new OccaVectorParameter(name_, v, attr, useRestrict));
  }

  OccaCoefficient& OccaCoefficient::AddGridFunction(const std::string &name_,
                                                    OccaVector &gf,
                                                    const bool useRestrict) {
    return Add(new OccaGridFunctionParameter(name_, gf, useRestrict));
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
