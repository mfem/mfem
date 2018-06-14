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

#ifndef MFEM_BACKENDS_KERNELS_COEFFICIENT_HPP
#define MFEM_BACKENDS_KERNELS_COEFFICIENT_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

namespace mfem
{

namespace kernels
{

class KernelsIntegrator;


class KernelsParameter
{
public:
   virtual ~KernelsParameter();

   virtual KernelsParameter* Clone() = 0;

   virtual void Setup(KernelsIntegrator &integ);

   //virtual ::kernels::kernelArg KernelArgs();
};


//---[ Include Parameter ]------------
class KernelsIncludeParameter : public KernelsParameter
{
private:
   std::string filename;

public:
   KernelsIncludeParameter(const std::string &filename_);

   virtual KernelsParameter* Clone();

   virtual void Setup(KernelsIntegrator &integ);
};
//====================================


//---[ Source Parameter ]------------
class KernelsSourceParameter : public KernelsParameter
{
private:
   std::string source;

public:
   KernelsSourceParameter(const std::string &filename_);

   virtual KernelsParameter* Clone();

   virtual void Setup(KernelsIntegrator &integ);
};
//====================================


//---[ Define Parameter ]------------
template <class TM>
class KernelsDefineParameter : public KernelsParameter
{
private:
   const std::string name;
   TM value;

public:
   KernelsDefineParameter(const std::string &name_,
                          const TM &value_) :
      name(name_),
      value(value_) {}

   virtual KernelsParameter* Clone()
   {
      return new KernelsDefineParameter(name, value);
   }

   virtual void Setup(KernelsIntegrator &integ)
   {
   }
};
//====================================


//---[ Variable Parameter ]-----------
template <class TM>
class KernelsVariableParameter : public KernelsParameter
{
private:
   const std::string name;
   const TM &value;

public:
   KernelsVariableParameter(const std::string &name_,
                            const TM &value_) :
      name(name_),
      value(value_) {}

   virtual KernelsParameter* Clone()
   {
      return new KernelsVariableParameter(name, value);
   }

   virtual void Setup(KernelsIntegrator &integ)
   {
      assert(false);
   }


};
//====================================


//---[ Vector Parameter ]-------
class KernelsVectorParameter : public KernelsParameter
{
private:
   const std::string name;
   Vector v;
   bool useRestrict;
   std::string attr;

public:
   KernelsVectorParameter(const std::string &name_,
                          Vector &v_,
                          const bool useRestrict_ = false);

   KernelsVectorParameter(const std::string &name_,
                          Vector &v_,
                          const std::string &attr_,
                          const bool useRestrict_ = false);

   virtual KernelsParameter* Clone();

   virtual void Setup(KernelsIntegrator &integ);
};
//====================================


//---[ GridFunction Parameter ]-------
class KernelsGridFunctionParameter : public KernelsParameter
{
private:
   const std::string name;
   KernelsGridFunction &gf;
   Vector gfQuad;
   bool useRestrict;

public:
   KernelsGridFunctionParameter(const std::string &name_,
                                KernelsGridFunction &gf_,
                                const bool useRestrict_ = false);

   virtual KernelsParameter* Clone();

   virtual void Setup(KernelsIntegrator &integ);
};
//====================================


//---[ Coefficient ]------------------
// [MISSING]
// Needs to know about the integrator's
//   - fespace
//   - ir
// Step where parameters that need the ir get called for setup
// For example, GridFunction (d, e) -> (q, e)
class KernelsCoefficient
{
private:
   SharedPtr<const Engine> engine;

   KernelsIntegrator *integ;

   std::string name;

   std::vector<KernelsParameter*> params;

public:
   KernelsCoefficient(const double value = 1.0);
   KernelsCoefficient(const Engine &e, const double value = 1.0);
   KernelsCoefficient(const Engine &e, const std::string &source);
   KernelsCoefficient(const Engine &e, const char *source);
   ~KernelsCoefficient();

   KernelsCoefficient(const KernelsCoefficient &coeff);

   const Engine &KernelsEngine() const { return *engine; }

   kernels::device GetDevice(int idx = 0) const
   { return engine->GetDevice(idx); }

   KernelsCoefficient& SetName(const std::string &name_);

   void Setup(KernelsIntegrator &integ_);

   KernelsCoefficient& Add(KernelsParameter *param);

   KernelsCoefficient& IncludeHeader(const std::string &filename);
   KernelsCoefficient& IncludeSource(const std::string &source);

   template <class TM>
   KernelsCoefficient& AddDefine(const std::string &name_, const TM &value)
   {
      return Add(new KernelsDefineParameter<TM>(name_, value));
   }

   template <class TM>
   KernelsCoefficient& AddVariable(const std::string &name_, const TM &value)
   {
      return Add(new KernelsVariableParameter<TM>(name_, value));
   }

   KernelsCoefficient& AddVector(const std::string &name_,
                                 Vector &v,
                                 const bool useRestrict = false);


   KernelsCoefficient& AddVector(const std::string &name_,
                                 Vector &v,
                                 const std::string &attr,
                                 const bool useRestrict = false);

   KernelsCoefficient& AddGridFunction(const std::string &name_,
                                       KernelsGridFunction &gf,
                                       const bool useRestrict = false);

   bool IsConstant();
   double GetConstantValue();

   Vector Eval();
   void Eval(Vector &quadCoeff);

   //operator ::kernels::kernelArg ();
};

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#endif // MFEM_BACKENDS_KERNELS_COEFFICIENT_HPP
