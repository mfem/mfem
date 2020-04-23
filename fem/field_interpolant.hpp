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
#include "bilinearform.hpp"
#include "lininteg.hpp"
#include "gridfunc.hpp"
#ifdef MFEM_USE_MPI
#include "pgridfunc.hpp"
#include "pbilinearform.hpp"
#endif

namespace mfem
{

class FieldInterpolant
{
protected:
   bool setup_disc;
   bool setup_full;
   Vector m_all_data;
   BilinearForm *L2;
   CGSolver *cg;
   int NE;
public:
   // The FiniteElementSpace passed into here should have a vdim set to 1 in order for the
   // MassIntegrator to work properly if the ProjectQuadratureCoefficient method is used with
   // a VectorQuadratureFunctionCoefficient.
   FieldInterpolant(FiniteElementSpace *fes) : setup_disc(false), setup_full(false)
   {
      L2 = new BilinearForm(fes);

      const FiniteElement &el = *fes->GetFE(0);
      const IntegrationRule *ir = &(IntRules.Get(el.GetGeomType(),
                                                 2 * el.GetOrder() + 1));

      L2->AddDomainIntegrator(new MassIntegrator(ir));
      L2->Assemble();
   }
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
   // Tells the ProjectQuadratureDiscCoefficient that they need to recalculate the data.
   void SetupDiscReset() { setup_disc = false; }
   // Tells the ProjectQuadratureCoefficient that they need to recalculate the data.
   virtual void FullReset()
   {
      L2->Update();
      L2->Assemble();
   }
   virtual void SetupCG()
   {
      cg = new CGSolver();
      cg->SetPrintLevel(0);
      cg->SetMaxIter(2000);
      cg->SetRelTol(sqrt(1e-30));
      cg->SetAbsTol(sqrt(0.0));
   }
   ~FieldInterpolant()
   {
      delete L2;
      delete cg;
   }
};

class ParFieldInterpolant : public FieldInterpolant
{
protected:
   ParBilinearForm *ParL2;
public:
   // The ParFiniteElementSpace passed into here should have a vdim set to 1 in order for the
   // MassIntegrator to work properly if the ProjectQuadratureCoefficient method is used with
   // a VectorQuadratureFunctionCoefficient.
   ParFieldInterpolant(ParFiniteElementSpace *pfes) : FieldInterpolant(pfes)
   {
      ParL2 = new ParBilinearForm(pfes);

      const FiniteElement &el = *pfes->GetFE(0);
      const IntegrationRule *ir = &(IntRules.Get(el.GetGeomType(),
                                                 2 * el.GetOrder() + 1));

      ParL2->AddDomainIntegrator(new MassIntegrator(ir));
      ParL2->Assemble();
   }
   // This function takes a vector quadrature function coefficient and projects it onto a GridFunction of the same space as vector
   // quadrature function coefficient.
   void ProjectQuadratureCoefficient(ParGridFunction &gf,
                                     VectorQuadratureFunctionCoefficient &vqfc,
                                     ParFiniteElementSpace &fes);
   // This function takes a quadrature function coefficient and projects it onto a GridFunction of the same space as
   // quadrature function coefficient.
   void ProjectQuadratureCoefficient(ParGridFunction &gf,
                                     QuadratureFunctionCoefficient &qfc,
                                     ParFiniteElementSpace &fes);
   // Tells the internal bilinearform needs to be reset in order to reset the sparse matrix
   virtual void FullReset() override
   {
      FieldInterpolant::FullReset();
      ParL2->Update();
      ParL2->Assemble();
   }
   using FieldInterpolant::SetupCG;
   // Setup the CG solver with an MPI communicator
   virtual void SetupCG(MPI_Comm _comm)
   {
      cg = new CGSolver(_comm);
      cg->SetPrintLevel(0);
      cg->SetMaxIter(2000);
      cg->SetRelTol(sqrt(1e-30));
      cg->SetAbsTol(sqrt(0.0));
   }

   ~ParFieldInterpolant()
   {
      delete ParL2;
   }
};

class VectorQuadratureIntegrator : public LinearFormIntegrator
{
private:
   VectorQuadratureFunctionCoefficient &vqfc;
public:
   VectorQuadratureIntegrator(VectorQuadratureFunctionCoefficient &vqfc) : vqfc(
         vqfc) { }
   VectorQuadratureIntegrator(VectorQuadratureFunctionCoefficient &vqfc,
                              const IntegrationRule *ir) : LinearFormIntegrator(ir), vqfc(
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
   QuadratureIntegrator(QuadratureFunctionCoefficient &qfc,
                        const IntegrationRule *ir) : LinearFormIntegrator(ir), qfc(qfc) { }
   using LinearFormIntegrator::AssembleRHSElementVect;
   void AssembleRHSElementVect(const FiniteElement &fe,
                               ElementTransformation &Tr,
                               Vector &elvect);
};

}

#endif