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

#ifndef MFEM_BACKENDS_RAJA_COEFFICIENT_HPP
#define MFEM_BACKENDS_RAJA_COEFFICIENT_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#include "../linalg/vector.hpp"
#include "../fem/gridfunc.hpp"

namespace mfem
{

namespace raja
{

class RajaIntegrator;


class RajaParameter
{
public:
   virtual ~RajaParameter();

   virtual RajaParameter* Clone() = 0;

   virtual void Setup(RajaIntegrator &integ);

   //virtual ::raja::kernelArg KernelArgs();
};


//---[ Include Parameter ]------------
class RajaIncludeParameter : public RajaParameter
{
private:
   std::string filename;

public:
   RajaIncludeParameter(const std::string &filename_);

   virtual RajaParameter* Clone();

   virtual void Setup(RajaIntegrator &integ);
};
//====================================


//---[ Source Parameter ]------------
class RajaSourceParameter : public RajaParameter
{
private:
   std::string source;

public:
   RajaSourceParameter(const std::string &filename_);

   virtual RajaParameter* Clone();

   virtual void Setup(RajaIntegrator &integ);
};
//====================================


//---[ Define Parameter ]------------
template <class TM>
class RajaDefineParameter : public RajaParameter
{
private:
   const std::string name;
   TM value;

public:
   RajaDefineParameter(const std::string &name_,
                       const TM &value_) :
      name(name_),
      value(value_) {}

   virtual RajaParameter* Clone()
   {
      return new RajaDefineParameter(name, value);
   }

   virtual void Setup(RajaIntegrator &integ)
   {
   }
};
//====================================


//---[ Variable Parameter ]-----------
template <class TM>
class RajaVariableParameter : public RajaParameter
{
private:
   const std::string name;
   const TM &value;

public:
   RajaVariableParameter(const std::string &name_,
                         const TM &value_) :
      name(name_),
      value(value_) {}

   virtual RajaParameter* Clone()
   {
      return new RajaVariableParameter(name, value);
   }

   virtual void Setup(RajaIntegrator &integ)
   {
      assert(false);
   }

 
};
//====================================


//---[ Vector Parameter ]-------
class RajaVectorParameter : public RajaParameter
{
private:
   const std::string name;
   Vector v;
   bool useRestrict;
   std::string attr;

public:
   RajaVectorParameter(const std::string &name_,
                       Vector &v_,
                       const bool useRestrict_ = false);

   RajaVectorParameter(const std::string &name_,
                       Vector &v_,
                       const std::string &attr_,
                       const bool useRestrict_ = false);

   virtual RajaParameter* Clone();

   virtual void Setup(RajaIntegrator &integ);
};
//====================================


//---[ GridFunction Parameter ]-------
class RajaGridFunctionParameter : public RajaParameter
{
private:
   const std::string name;
   RajaGridFunction &gf;
   Vector gfQuad;
   bool useRestrict;

public:
   RajaGridFunctionParameter(const std::string &name_,
                             RajaGridFunction &gf_,
                             const bool useRestrict_ = false);

   virtual RajaParameter* Clone();

   virtual void Setup(RajaIntegrator &integ);
};
//====================================


//---[ Coefficient ]------------------
// [MISSING]
// Needs to know about the integrator's
//   - fespace
//   - ir
// Step where parameters that need the ir get called for setup
// For example, GridFunction (d, e) -> (q, e)
class RajaCoefficient
{
private:
   SharedPtr<const Engine> engine;

   RajaIntegrator *integ;

   std::string name;

   std::vector<RajaParameter*> params;

public:
   RajaCoefficient(const Engine &e, const double value = 1.0);
   RajaCoefficient(const Engine &e, const std::string &source);
   RajaCoefficient(const Engine &e, const char *source);
   ~RajaCoefficient();

   RajaCoefficient(const RajaCoefficient &coeff);

   const Engine &RajaEngine() const { return *engine; }

   raja::device GetDevice(int idx = 0) const
   { return engine->GetDevice(idx); }

   RajaCoefficient& SetName(const std::string &name_);

   void Setup(RajaIntegrator &integ_);

   RajaCoefficient& Add(RajaParameter *param);

   RajaCoefficient& IncludeHeader(const std::string &filename);
   RajaCoefficient& IncludeSource(const std::string &source);

   template <class TM>
   RajaCoefficient& AddDefine(const std::string &name_, const TM &value)
   {
      return Add(new RajaDefineParameter<TM>(name_, value));
   }

   template <class TM>
   RajaCoefficient& AddVariable(const std::string &name_, const TM &value)
   {
      return Add(new RajaVariableParameter<TM>(name_, value));
   }

   RajaCoefficient& AddVector(const std::string &name_,
                              Vector &v,
                              const bool useRestrict = false);


   RajaCoefficient& AddVector(const std::string &name_,
                              Vector &v,
                              const std::string &attr,
                              const bool useRestrict = false);

   RajaCoefficient& AddGridFunction(const std::string &name_,
                                    RajaGridFunction &gf,
                                    const bool useRestrict = false);

   bool IsConstant();
   double GetConstantValue();

   Vector Eval();
   void Eval(Vector &quadCoeff);

   //operator ::raja::kernelArg ();
};

} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#endif // MFEM_BACKENDS_RAJA_COEFFICIENT_HPP
