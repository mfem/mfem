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
#include "raja.hpp"

#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

namespace mfem
{

namespace raja
{
/*
void BilinearForm::InitRajaBilinearForm()
{
   // Init 'obform' using 'bform'
   MFEM_ASSERT(bform != NULL, "");
   MFEM_ASSERT(obform == NULL, "");

   FiniteElementSpace &ofes =
      bform->FESpace()->Get_PFESpace()->As<FiniteElementSpace>();
   obform = new RajaBilinearForm(&ofes);

   // Transfer domain integrators
   mfem::Array<mfem::BilinearFormIntegrator*> &dbfi = *bform->GetDBFI();
   for (int i = 0; i < dbfi.Size(); i++)
   {
      std::string integ_name(dbfi[i]->Name());
      Coefficient *scal_coeff = dbfi[i]->GetScalarCoefficient();
      ConstantCoefficient *const_coeff =
         dynamic_cast<ConstantCoefficient*>(scal_coeff);
      // TODO: other types of coefficients ...
      double val = const_coeff ? const_coeff->constant : 1.0;
      RajaCoefficient ocoeff(obform->RajaEngine(), val);

      RajaIntegrator *ointeg = NULL;

      if (integ_name == "(undefined)")
      {
         MFEM_ABORT("BilinearFormIntegrator does not define Name()");
      }
      else if (integ_name == "diffusion")
      {
         ointeg = new RajaDiffusionIntegrator(ocoeff);
      }
      else
      {
         MFEM_ABORT("BilinearFormIntegrator [Name() = " << integ_name
                    << "] is not supported");
      }

      const mfem::IntegrationRule *ir = dbfi[i]->GetIntRule();
      if (ir) { ointeg->SetIntegrationRule(*ir); }

      obform->AddDomainIntegrator(ointeg);
   }

   // TODO: other types of integrators ...
}

bool BilinearForm::Assemble()
{
   if (obform == NULL) { InitRajaBilinearForm(); }

   obform->Assemble();

   return true; // --> host assembly is not needed
}

void BilinearForm::FormSystemMatrix(const mfem::Array<int> &ess_tdof_list,
                                    mfem::OperatorHandle &A)
{
   if (A.Type() == mfem::Operator::ANY_TYPE)
   {
      mfem::Operator *Aout = NULL;
      obform->FormOperator(ess_tdof_list, Aout);
      A.Reset(Aout);
   }
   else
   {
      MFEM_ABORT("Operator::Type is not supported, type = " << A.Type());
   }
}

void BilinearForm::FormLinearSystem(const mfem::Array<int> &ess_tdof_list,
                                    mfem::Vector &x, mfem::Vector &b,
                                    mfem::OperatorHandle &A,
                                    mfem::Vector &X, mfem::Vector &B,
                                    int copy_interior)
{
   FormSystemMatrix(ess_tdof_list, A);
   obform->InitRHS(ess_tdof_list, x, b, A.Ptr(), X, B, copy_interior);
}

void BilinearForm::RecoverFEMSolution(const mfem::Vector &X,
                                      const mfem::Vector &b,
                                      mfem::Vector &x)
{
   obform->RajaRecoverFEMSolution(X, b, x);
}
*/
} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)
