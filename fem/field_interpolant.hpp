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

/** @brief Provides methods to take quadrature data and project it onto a field
    within a H1 or L2 space.*/
class Quad2FieldInterpolant
{
protected:
   BilinearForm *L2 = nullptr;  // Owned.
   CGSolver *cg = nullptr;      // Owned.
public:
   /** The FiniteElementSpace passed into here should be of the same order
       and space as those used within the ProjectQuadratureCoefficient method.
       The vdim on the FES should be equal to 1, so the L2 method can work on
       either scalar or vector GridFunctions.*/
   Quad2FieldInterpolant(FiniteElementSpace *fes)
   {
      MFEM_VERIFY(fes->GetVDim() == 1, "FiniteElementSpace should have a \
                                        a vdim of 1");
      L2 = new BilinearForm(fes);

      const FiniteElement &el = *fes->GetFE(0);
      const IntegrationRule *ir = &(IntRules.Get(el.GetGeomType(),
                                                 2 * el.GetOrder() + 1));

      L2->AddDomainIntegrator(new MassIntegrator(ir));
      L2->Assemble();
   }
   /** @brief This function takes a vector quadrature function coefficient and projects
       it onto a GridFunction that lives either in a H1 or L2 space.*/
   /** Internally, this function makes use of the GridFunction::ProjectDiscCoefficient.*/
   void ProjectQuadratureDiscCoefficient(GridFunction &gf,
                                         VectorQuadratureFunctionCoefficient &vqfc);
   /** @brief This function takes a quadrature function coefficient and projects
       it onto a GridFunction lives either in a H1 or L2 space.*/
   /** Internally, this function makes use of the GridFunction::ProjectDiscCoefficient.*/
   void ProjectQuadratureDiscCoefficient(GridFunction &gf,
                                         QuadratureFunctionCoefficient &qfc);
   /** This function takes a vector quadrature function coefficient and projects
       it onto a GridFunction through the use of an L2 projection method.*/
   void ProjectQuadratureCoefficient(GridFunction &gf,
                                     VectorQuadratureFunctionCoefficient &vqfc);
   /** This function takes a quadrature function coefficient and projects it onto
       a GridFunction through the use of an L2 projection method. */
   void ProjectQuadratureCoefficient(GridFunction &gf,
                                     QuadratureFunctionCoefficient &qfc);
   /// This function resets the internal bilinearform due to any mesh changes.
   virtual void FullReset()
   {
      L2->Update();
      L2->Assemble();
   }
   virtual void SetupCG(double rel_tol = 1e-15, double abs_tol = 0.0,
                        int print_level = 0, int max_iter = 2000)
   {
      if (cg)
      {
         delete cg;
      }
      cg = new CGSolver();
      cg->SetPrintLevel(print_level);
      cg->SetMaxIter(max_iter);
      cg->SetRelTol(rel_tol);
      cg->SetAbsTol(abs_tol);
   }
   virtual ~Quad2FieldInterpolant()
   {
      if (L2)
      {
         delete L2;
      }
      if (cg)
      {
         delete cg;
      }
   }

protected:
   // Should only be needed for children classes to avoid unneeded resources
   // from being allocated.
   Quad2FieldInterpolant() {}
};

#ifdef MFEM_USE_MPI
class ParQuad2FieldInterpolant : public Quad2FieldInterpolant
{
protected:
   ParBilinearForm *ParL2 = nullptr; // Owned
public:
   /** The FiniteElementSpace passed into here should be of the same order
       and space as those used within the ProjectQuadratureCoefficient method.
       The vdim on the FES should be equal to 1, so the L2 method can work on
       either scalar or vector GridFunctions.*/
   ParQuad2FieldInterpolant(ParFiniteElementSpace *pfes)
   {
      MFEM_VERIFY(pfes->GetVDim() == 1, "FiniteElementSpace should have a \
                                    a vdim of 1");
      ParL2 = new ParBilinearForm(pfes);

      const FiniteElement &el = *pfes->GetFE(0);
      const IntegrationRule *ir = &(IntRules.Get(el.GetGeomType(),
                                                 2 * el.GetOrder() + 1));

      ParL2->AddDomainIntegrator(new MassIntegrator(ir));
      ParL2->Assemble();
   }
   /** @brief This function takes a vector quadrature function coefficient and projects
       it onto a GridFunction that lives either in a H1 or L2 space.*/
   /** Internally, this function makes use of the GridFunction::ProjectDiscCoefficient.*/
   void ProjectQuadratureCoefficient(ParGridFunction &gf,
                                     VectorQuadratureFunctionCoefficient &vqfc);
   /** @brief This function takes a quadrature function coefficient and projects
       it onto a GridFunction that lives either in a H1 or L2 space.*/
   /** Internally, this function makes use of the GridFunction::ProjectDiscCoefficient.*/
   void ProjectQuadratureCoefficient(ParGridFunction &gf,
                                     QuadratureFunctionCoefficient &qfc);
   /// This function resets the internal bilinearform due to any mesh changes.
   virtual void FullReset() override
   {
      ParL2->Update();
      ParL2->Assemble();
   }
   using Quad2FieldInterpolant::SetupCG;
   /// Setup the CG solver with an MPI communicator
   virtual void SetupCG(MPI_Comm _comm, double rel_tol = 1e-15,
                        double abs_tol = 0.0,
                        int print_level = 0, int max_iter = 2000)
   {
      if (cg)
      {
         delete cg;
      }
      cg = new CGSolver(_comm);
      cg->SetPrintLevel(print_level);
      cg->SetMaxIter(max_iter);
      cg->SetRelTol(rel_tol);
      cg->SetAbsTol(abs_tol);
   }

   virtual ~ParQuad2FieldInterpolant()
   {
      delete ParL2;
   }
};
#endif
}
#endif
