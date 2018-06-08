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

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#include "../raja.hpp"

namespace mfem
{

namespace raja
{

//---[ Parameter ]------------
RajaParameter::~RajaParameter() {}

void RajaParameter::Setup(RajaIntegrator &integ) {}

//====================================


//---[ Include Parameter ]------------
RajaIncludeParameter::RajaIncludeParameter(const std::string &filename_) :
   filename(filename_) {}

RajaParameter* RajaIncludeParameter::Clone()
{
   return new RajaIncludeParameter(filename);
}

void RajaIncludeParameter::Setup(RajaIntegrator &integ)
{
}
//====================================


//---[ Source Parameter ]------------
RajaSourceParameter::RajaSourceParameter(const std::string &source_) :
   source(source_) {}

RajaParameter* RajaSourceParameter::Clone()
{
   return new RajaSourceParameter(source);
}

void RajaSourceParameter::Setup(RajaIntegrator &integ)
{
}
//====================================

//---[ Vector Parameter ]-------
RajaVectorParameter::RajaVectorParameter(const std::string &name_,
                                         Vector &v_,
                                         const bool useRestrict_) :
   name(name_),
   v(v_),
   useRestrict(useRestrict_),
   attr("") {}

RajaVectorParameter::RajaVectorParameter(const std::string &name_,
                                         Vector &v_,
                                         const std::string &attr_,
                                         const bool useRestrict_) :
   name(name_),
   v(v_),
   useRestrict(useRestrict_),
   attr(attr_) {}

RajaParameter* RajaVectorParameter::Clone()
{
   return new RajaVectorParameter(name, v, attr, useRestrict);
}

void RajaVectorParameter::Setup(RajaIntegrator &integ)
{
}

//====================================


//---[ GridFunction Parameter ]-------
RajaGridFunctionParameter::RajaGridFunctionParameter(const std::string &name_,
                                                     RajaGridFunction &gf_,
                                                     const bool useRestrict_)
   : name(name_),
     gf(gf_),
     gfQuad(*(new Layout(gf_.RajaLayout().RajaEngine(), 0))),
     useRestrict(useRestrict_) {}

RajaParameter* RajaGridFunctionParameter::Clone()
{
   RajaGridFunctionParameter *param =
      new RajaGridFunctionParameter(name, gf, useRestrict);
   param->gfQuad.MakeRef(gfQuad);
   return param;
}

void RajaGridFunctionParameter::Setup(RajaIntegrator &integ)
{

}
//====================================


//---[ Coefficient ]------------------
   RajaCoefficient::RajaCoefficient(const double value) :
   engine(NULL),
   integ(NULL),
   name("COEFF")
{
}
//---[ Coefficient ]------------------
   RajaCoefficient::RajaCoefficient(const Engine &e, const double value) :
   engine(&e),
   integ(NULL),
   name("COEFF")
{
}

RajaCoefficient::RajaCoefficient(const Engine &e, const std::string &source) :
   engine(&e),
   integ(NULL),
   name("COEFF")
{
}

RajaCoefficient::RajaCoefficient(const Engine &e, const char *source) :
   engine(&e),
   integ(NULL),
   name("COEFF")
{
}

RajaCoefficient::RajaCoefficient(const RajaCoefficient &coeff) :
   engine(coeff.engine),
   integ(NULL),
   name(coeff.name)
{

   const int paramCount = (int) coeff.params.size();
   for (int i = 0; i < paramCount; ++i)
   {
      params.push_back(coeff.params[i]->Clone());
   }
}

RajaCoefficient::~RajaCoefficient()
{
   const int paramCount = (int) params.size();
   for (int i = 0; i < paramCount; ++i)
   {
      delete params[i];
   }
}

RajaCoefficient& RajaCoefficient::SetName(const std::string &name_)
{
   name = name_;
   return *this;
}

void RajaCoefficient::Setup(RajaIntegrator &integ_)
{
   integ = &integ_;
}

RajaCoefficient& RajaCoefficient::Add(RajaParameter *param)
{
   params.push_back(param);
   return *this;
}

RajaCoefficient& RajaCoefficient::IncludeHeader(const std::string &filename)
{
   return Add(new RajaIncludeParameter(filename));
}

RajaCoefficient& RajaCoefficient::IncludeSource(const std::string &source)
{
   return Add(new RajaSourceParameter(source));
}

RajaCoefficient& RajaCoefficient::AddVector(const std::string &name_,
                                            Vector &v,
                                            const bool useRestrict)
{
   return Add(new RajaVectorParameter(name_, v, useRestrict));
}

RajaCoefficient& RajaCoefficient::AddVector(const std::string &name_,
                                            Vector &v,
                                            const std::string &attr,
                                            const bool useRestrict)
{
   return Add(new RajaVectorParameter(name_, v, attr, useRestrict));
}

RajaCoefficient& RajaCoefficient::AddGridFunction(const std::string &name_,
                                                  RajaGridFunction &gf,
                                                  const bool useRestrict)
{
   return Add(new RajaGridFunctionParameter(name_, gf, useRestrict));
}

bool RajaCoefficient::IsConstant()
{
   assert(false);
   return true;
}

double RajaCoefficient::GetConstantValue()
{
   if (!IsConstant())
   {
      mfem_error("RajaCoefficient is not constant");
   }
   assert(false);
   return 1.0;
}

Vector RajaCoefficient::Eval()
{
   if (integ == NULL)
   {
      mfem_error("RajaCoefficient requires a Setup() call before Eval()");
   }

   mfem::FiniteElementSpace &fespace = integ->GetTrialFESpace();
   const mfem::IntegrationRule &ir   = integ->GetIntegrationRule();

   const int elements = fespace.GetNE();
   const int numQuad  = ir.GetNPoints();

   Vector quadCoeff(*(new Layout(RajaEngine(), numQuad * elements)));
   Eval(quadCoeff);
   return quadCoeff;
}

void RajaCoefficient::Eval(Vector &quadCoeff)
{
   assert(false);
}

} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)
