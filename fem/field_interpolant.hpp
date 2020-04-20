// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Implementation of Field Interpolants

#ifndef MFEM_FIELD_INTERPOLANT
#define MFEM_FIELD_INTERPOLANT

#include "../config/config.hpp"
#include "../linalg/linalg.hpp"
#include "intrules.hpp"
#include "eltrans.hpp"
#include "coefficient.hpp"
#include "bilininteg.hpp"
#include "lininteg.hpp"

namespace mfem
{

class FieldInterpolant
{
private:
   bool setup;
   Vector m_all_data;
   MassIntegrator mass_int;
   int NE;
public:
   FieldInterpolant(const IntegrationRule* ir) { mass_int.SetIntRule(ir); }
   void ProjectQuadratureDiscCoefficient(GridFunction &gf,
                                         VectorQuadratureFunctionCoefficient &vqfc,
                                         FiniteElementSpace &fes);
   void ProjectQuadratureDiscCoefficient(GridFunction &gf,
                                         QuadratureFunctionCoefficient &qfc,
                                         FiniteElementSpace &fes);
   void SetupReset() { setup = false; }
};

class VectorQuadratureIntegrator : public LinearFormIntegrator
{
private:
   VectorQuadratureFunctionCoefficient &vqfc;
public:
   VectorQuadratureIntegrator(VectorQuadratureFunctionCoefficient &vqfc) : vqfc(
         vqfc) { }
   using LinearFormIntegrator::AssembleRHSElementVect;
   void AssembleRHSElementVect(const FiniteElement &fe,
                               ElementTransformation &Tr,
                               Vector &elvect);
};

class QuadratureIntegrator : public LinearFormIntegrator
{
private:
   QuadratureFunctionCoefficient &qfc;
public:
   QuadratureIntegrator(QuadratureFunctionCoefficient &qfc) : qfc(qfc) { }
   using LinearFormIntegrator::AssembleRHSElementVect;
   void AssembleRHSElementVect(const FiniteElement &fe,
                               ElementTransformation &Tr,
                               Vector &elvect);
};

}

#endif