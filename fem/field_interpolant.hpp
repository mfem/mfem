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
   bool setup_disc;
   Vector m_all_data;
   MassIntegrator mass_int;
   int NE;
public:
   FieldInterpolant(const IntegrationRule* ir) { mass_int.SetIntRule(ir); }
   // This function takes a vector quadrature function coefficient and projects it onto a GridFunction that lives
   // in L2 space. This function requires tr_fes to be the finite element space that the VectorQuadratureFunctionCoefficient lives on
   // and fes is the L2 finite element space that we're projecting onto.
   void ProjectQuadratureDiscCoefficient(GridFunction &gf,
                                         VectorQuadratureFunctionCoefficient &vqfc,
                                         FiniteElementSpace &tr_fes,
                                         FiniteElementSpace &fes);
   // This function takes a quadrature function coefficient and projects it onto a GridFunction that lives
   // in L2 space. This function requires tr_fes to be the finite element space that the QuadratureFunctionCoefficient lives on
   // and fes is the L2 finite element space that we're projecting onto.
   void ProjectQuadratureDiscCoefficient(GridFunction &gf,
                                         QuadratureFunctionCoefficient &qfc,
                                         FiniteElementSpace &tr_fes,
                                         FiniteElementSpace &fes);
   //Parallel versions of the ProjectQuadratureCoefficient will need to be created once the serial version works
   
   // This function takes a vector quadrature function coefficient and projects it onto a GridFunction of the same space as vector
   // quadrature function coefficient.
   void ProjectQuadratureCoefficient(GridFunction &gf,
                                     VectorQuadratureFunctionCoefficient &vqfc,
                                     FiniteElementSpace &fes);
   // This function takes a quadrature function coefficient and projects it onto a GridFunction of the same space as 
   // quadrature function coefficient.
   void ProjectQuadratureCoefficient(GridFunction &gf,
                                     QuadratureFunctionCoefficient &qfc,
                                     FiniteElementSpace &fes);
   //Tells the ProjectQuadratureDiscCoefficient that they need to recalculate the data.
   void SetupDiscReset() { setup_disc = false; }
   ~FieldInterpolant() {}
};

class VectorQuadratureIntegrator : public LinearFormIntegrator
{
private:
   VectorQuadratureFunctionCoefficient &vqfc;
public:
   VectorQuadratureIntegrator(VectorQuadratureFunctionCoefficient &vqfc) : vqfc(
         vqfc) { }
   VectorQuadratureIntegrator(VectorQuadratureFunctionCoefficient &vqfc, const IntegrationRule *ir) : vqfc(
         vqfc), LinearFormIntegrator(ir) { }
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
   QuadratureIntegrator(QuadratureFunctionCoefficient &qfc, const IntegrationRule *ir) : qfc(qfc), LinearFormIntegrator(ir) { }
   using LinearFormIntegrator::AssembleRHSElementVect;
   void AssembleRHSElementVect(const FiniteElement &fe,
                               ElementTransformation &Tr,
                               Vector &elvect);
};

}

#endif