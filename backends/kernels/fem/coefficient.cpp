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
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#include "../kernels.hpp"

namespace mfem
{

namespace kernels
{

//---[ Parameter ]------------
KernelsParameter::~KernelsParameter() {}

void KernelsParameter::Setup(KernelsIntegrator &integ) {push(); pop();}

//====================================


//---[ Include Parameter ]------------
KernelsIncludeParameter::KernelsIncludeParameter(const std::string &filename_) :
   filename(filename_) {}

KernelsParameter* KernelsIncludeParameter::Clone()
{
   return new KernelsIncludeParameter(filename);
}

void KernelsIncludeParameter::Setup(KernelsIntegrator &integ)
{
}
//====================================


//---[ Source Parameter ]------------
KernelsSourceParameter::KernelsSourceParameter(const std::string &source_) :
   source(source_) {}

KernelsParameter* KernelsSourceParameter::Clone()
{
   return new KernelsSourceParameter(source);
}

void KernelsSourceParameter::Setup(KernelsIntegrator &integ)
{
}
//====================================

//---[ Vector Parameter ]-------
KernelsVectorParameter::KernelsVectorParameter(const std::string &name_,
                                               Vector &v_,
                                               const bool useRestrict_) :
   name(name_),
   v(v_),
   useRestrict(useRestrict_),
   attr("") {}

KernelsVectorParameter::KernelsVectorParameter(const std::string &name_,
                                               Vector &v_,
                                               const std::string &attr_,
                                               const bool useRestrict_) :
   name(name_),
   v(v_),
   useRestrict(useRestrict_),
   attr(attr_) {}

KernelsParameter* KernelsVectorParameter::Clone()
{
   return new KernelsVectorParameter(name, v, attr, useRestrict);
}

void KernelsVectorParameter::Setup(KernelsIntegrator &integ)
{
}

//====================================


//---[ GridFunction Parameter ]-------
KernelsGridFunctionParameter::KernelsGridFunctionParameter(
   const std::string &name_,
   KernelsGridFunction &gf_,
   const bool useRestrict_)
   : name(name_),
     gf(gf_),
     gfQuad(*(new Layout(gf_.KernelsLayout().KernelsEngine(), 0))),
     useRestrict(useRestrict_) {}

KernelsParameter* KernelsGridFunctionParameter::Clone()
{
   KernelsGridFunctionParameter *param =
      new KernelsGridFunctionParameter(name, gf, useRestrict);
   param->gfQuad.MakeRef(gfQuad);
   return param;
}

void KernelsGridFunctionParameter::Setup(KernelsIntegrator &integ)
{
   push();
   pop();
}
//====================================


//---[ Coefficient ]------------------
KernelsCoefficient::KernelsCoefficient(const double value) :
   engine(NULL),
   integ(NULL),
   name("COEFF")
{
   push();
   pop();
}
//---[ Coefficient ]------------------
KernelsCoefficient::KernelsCoefficient(const Engine &e, const double value) :
   engine(&e),
   integ(NULL),
   name("COEFF")
{
   push();
   pop();
}

KernelsCoefficient::KernelsCoefficient(const Engine &e,
                                       const std::string &source) :
   engine(&e),
   integ(NULL),
   name("COEFF")
{
   push();
   pop();
}

KernelsCoefficient::KernelsCoefficient(const Engine &e, const char *source) :
   engine(&e),
   integ(NULL),
   name("COEFF")
{
   push();
   pop();
}

KernelsCoefficient::KernelsCoefficient(const KernelsCoefficient &coeff) :
   engine(coeff.engine),
   integ(NULL),
   name(coeff.name)
{
   push();
   const int paramCount = (int) coeff.params.size();
   for (int i = 0; i < paramCount; ++i)
   {
      params.push_back(coeff.params[i]->Clone());
   }
   pop();
}

KernelsCoefficient::~KernelsCoefficient()
{
   const int paramCount = (int) params.size();
   for (int i = 0; i < paramCount; ++i)
   {
      delete params[i];
   }
}

KernelsCoefficient& KernelsCoefficient::SetName(const std::string &name_)
{
   name = name_;
   return *this;
}

void KernelsCoefficient::Setup(KernelsIntegrator &integ_)
{
   push();
   integ = &integ_;
   pop();
}

KernelsCoefficient& KernelsCoefficient::Add(KernelsParameter *param)
{
   push();
   params.push_back(param);
   pop();
   return *this;
}

KernelsCoefficient& KernelsCoefficient::IncludeHeader(const std::string
                                                      &filename)
{
   return Add(new KernelsIncludeParameter(filename));
}

KernelsCoefficient& KernelsCoefficient::IncludeSource(const std::string &source)
{
   return Add(new KernelsSourceParameter(source));
}

KernelsCoefficient& KernelsCoefficient::AddVector(const std::string &name_,
                                                  Vector &v,
                                                  const bool useRestrict)
{
   return Add(new KernelsVectorParameter(name_, v, useRestrict));
}

KernelsCoefficient& KernelsCoefficient::AddVector(const std::string &name_,
                                                  Vector &v,
                                                  const std::string &attr,
                                                  const bool useRestrict)
{
   return Add(new KernelsVectorParameter(name_, v, attr, useRestrict));
}

KernelsCoefficient& KernelsCoefficient::AddGridFunction(
   const std::string &name_,
   KernelsGridFunction &gf,
   const bool useRestrict)
{
   return Add(new KernelsGridFunctionParameter(name_, gf, useRestrict));
}

bool KernelsCoefficient::IsConstant()
{
   assert(false);
   return true;
}

double KernelsCoefficient::GetConstantValue()
{
   if (!IsConstant())
   {
      mfem_error("KernelsCoefficient is not constant");
   }
   assert(false);
   return 1.0;
}

Vector KernelsCoefficient::Eval()
{
   if (integ == NULL)
   {
      mfem_error("KernelsCoefficient requires a Setup() call before Eval()");
   }

   mfem::FiniteElementSpace &fespace = integ->GetTrialFESpace();
   const mfem::IntegrationRule &ir   = integ->GetIntegrationRule();

   const int elements = fespace.GetNE();
   const int numQuad  = ir.GetNPoints();

   Vector quadCoeff(*(new Layout(KernelsEngine(), numQuad * elements)));
   Eval(quadCoeff);
   return quadCoeff;
}

void KernelsCoefficient::Eval(Vector &quadCoeff)
{
   assert(false);
}

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)
