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
#include "../linalg/ovector.hpp"
#include "ogridfunc.hpp"

namespace mfem {
  class OccaIntegrator;

  class OccaParameter {
  public:
    virtual ~OccaParameter();

    virtual OccaParameter* Clone() = 0;

    virtual void Setup(OccaIntegrator &integ,
                       occa::properties &props);

    virtual occa::kernelArg KernelArgs();
  };

  //---[ Include Parameter ]------------
  class OccaIncludeParameter : public OccaParameter {
  private:
    std::string filename;

  public:
    OccaIncludeParameter(const std::string &filename_);

    virtual OccaParameter* Clone();

    virtual void Setup(OccaIntegrator &integ,
                       occa::properties &props);
  };
  //====================================

  //---[ Source Parameter ]------------
  class OccaSourceParameter : public OccaParameter {
  private:
    std::string source;

  public:
    OccaSourceParameter(const std::string &filename_);

    virtual OccaParameter* Clone();

    virtual void Setup(OccaIntegrator &integ,
                       occa::properties &props);
  };
  //====================================

  //---[ Define Parameter ]------------
  template <class TM>
  class OccaDefineParameter : public OccaParameter {
  private:
    const std::string name;
    TM value;

  public:
    OccaDefineParameter(const std::string &name_,
                        const TM &value_) :
      name(name_),
      value(value_) {}

    virtual OccaParameter* Clone() {
      return new OccaDefineParameter(name, value);
    }

    virtual void Setup(OccaIntegrator &integ,
                       occa::properties &props) {
       props["defines"][name] = value;
    }
  };
  //====================================

  //---[ Variable Parameter ]-----------
  template <class TM>
  class OccaVariableParameter : public OccaParameter {
  private:
    const std::string name;
    const TM &value;

  public:
    OccaVariableParameter(const std::string &name_,
                          const TM &value_) :
      name(name_),
      value(value_) {}

    virtual OccaParameter* Clone() {
      return new OccaVariableParameter(name, value);
    }

    virtual void Setup(OccaIntegrator &integ,
                       occa::properties &props) {
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

  //---[ Vector Parameter ]-------
  class OccaVectorParameter : public OccaParameter {
  private:
    const std::string name;
    OccaVector v;
    bool useRestrict;
    std::string attr;

  public:
    OccaVectorParameter(const std::string &name_,
                        OccaVector &v_,
                        const bool useRestrict_ = false);

    OccaVectorParameter(const std::string &name_,
                        OccaVector &v_,
                        const std::string &attr_,
                        const bool useRestrict_ = false);

    virtual OccaParameter* Clone();

    virtual void Setup(OccaIntegrator &integ,
                       occa::properties &props);

    virtual occa::kernelArg KernelArgs();
  };
  //====================================

  //---[ GridFunction Parameter ]-------
  class OccaGridFunctionParameter : public OccaParameter {
  private:
    const std::string name;
    OccaGridFunction &gf;
    OccaVector gfQuad;
    bool useRestrict;

  public:
    OccaGridFunctionParameter(const std::string &name_,
                              OccaGridFunction &gf_,
                              const bool useRestrict_ = false);

    virtual OccaParameter* Clone();

    virtual void Setup(OccaIntegrator &integ,
                       occa::properties &props);

    virtual occa::kernelArg KernelArgs();
  };
  //====================================

  //---[ Coefficient ]------------------
  // [MISSING]
  // Needs to know about the integrator's
  //   - fespace
  //   - ir
  // Step where parameters that need the ir get called for setup
  // For example, GridFunction (d, e) -> (q, e)
  class OccaCoefficient {
  private:
    occa::device device;
    OccaIntegrator *integ;

    std::string name;
    occa::json coeffValue;

    occa::properties props;
    std::vector<OccaParameter*> params;

  public:
    OccaCoefficient(const double value = 1.0);
    OccaCoefficient(const std::string &source);
    OccaCoefficient(const char *source);
    ~OccaCoefficient();

    OccaCoefficient(const OccaCoefficient &coeff);

    OccaCoefficient& SetName(const std::string &name_);

    void Setup(OccaIntegrator &integ_,
               occa::properties &props_);

    OccaCoefficient& Add(OccaParameter *param);

    OccaCoefficient& IncludeHeader(const std::string &filename);
    OccaCoefficient& IncludeSource(const std::string &source);

    template <class TM>
    OccaCoefficient& AddDefine(const std::string &name_, const TM &value) {
      return Add(new OccaDefineParameter<TM>(name_, value));
    }

    template <class TM>
    OccaCoefficient& AddVariable(const std::string &name_, const TM &value) {
      return Add(new OccaVariableParameter<TM>(name_, value));
    }

    OccaCoefficient& AddVector(const std::string &name_,
                               OccaVector &v,
                               const bool useRestrict = false);


    OccaCoefficient& AddVector(const std::string &name_,
                               OccaVector &v,
                               const std::string &attr,
                               const bool useRestrict = false);

    OccaCoefficient& AddGridFunction(const std::string &name_,
                                     OccaGridFunction &gf,
                                     const bool useRestrict = false);

    bool IsConstant();
    double GetConstantValue();

    OccaVector Eval();
    void Eval(OccaVector &quadCoeff);

    operator occa::kernelArg ();
  };
  //====================================
}

#  endif
#endif
