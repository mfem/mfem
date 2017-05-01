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
#  ifndef MFEM_OCCA_COEFFICIENT
#  define MFEM_OCCA_COEFFICIENT

#include <vector>

#include "occa.hpp"

namespace mfem {
  class OccaParameter {
  public:
    virtual void SetProps(occa::properties &props);
    virtual occa::kernelArg KernelArgs();
  };

  //---[ Defined Parameter ]------------
  template <class TM>
  class OccaDefinedParameter : public OccaParameter {
  private:
    const std::string name;
    TM value;

  public:
    OccaDefinedParameter(const std::string &name_,
                         const TM &value_) :
      name(name_),
      value(value_) {}

    virtual void SetProps(occa::properties &props) {
      props["defines"][name] = value;
    }
  };
  //====================================

  //---[ Variable Parameter ]-----------
  template <class TM>
  class OccaVariableParameter : public OccaParameter {
  private:
    const std::string name;
    TM &value;

  public:
    OccaVariableParameter(const std::string &name_,
                          const TM &value_) :
      name(name_),
      value(value_) {}

    virtual void SetProps(occa::properties &props) {
      std::string &args = (props["defines/COEFF_ARGS"]
                           .asString()
                           .string());
      // const TM name,\n"
      args += "const ";
      args += occa::primitiveinfo<TM>::name;
      args += ' ';
      args += name;
      args += ",\n";
    }

    virtual occa::kernelArg KernelArgs() {
      return occa::kernelArg(value);
    }
  };
  //====================================

  //---[ Include Parameter ]------------
  class OccaIncludeParameter : public OccaParameter {
  private:
    std::string filename;

  public:
    OccaIncludeParameter(const std::string &filename_);

    virtual void SetProps(occa::properties &props);
  };
  //====================================

  //---[ Source Parameter ]------------
  class OccaSourceParameter : public OccaParameter {
  private:
    std::string source;

  public:
    OccaSourceParameter(const std::string &filename_);

    virtual void SetProps(occa::properties &props);
  };
  //====================================

  //---[ Coefficient ]------------------
  class OccaCoefficient {
  private:
    std::vector<OccaParameter*> params;
    occa::properties props;

  public:
    OccaCoefficient();
    OccaCoefficient(const double value);
    OccaCoefficient(const std::string &function);

    OccaCoefficient(const OccaCoefficient &coeff);

    virtual void SetCoeffProps(occa::properties &props_);
    virtual occa::kernelArg CoeffKernelArg();

    OccaCoefficient& With(OccaParameter *param);

    void SetProps(occa::properties &props_);

    operator occa::kernelArg ();
  };
  //====================================
}

#  endif
#endif