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

// *****************************************************************************
bool BilinearForm::Assemble()
{
   push();
   if (kbform == NULL) InitKBilinearForm();
   kbform->Assemble();
   pop();
   return true; // --> host assembly is not needed
}

// *****************************************************************************
void BilinearForm::FormSystemMatrix(const mfem::Array<int> &ess_tdof_list,
                                    mfem::OperatorHandle &A)
{
   push();//assert(false);// ex1pd comes here, Laghos dont
   if (A.Type() == mfem::Operator::ANY_TYPE)
   {
      mfem::Operator *Aout = NULL;
      kbform->FormOperator(ess_tdof_list, Aout);
      A.Reset(Aout);
   }
   else
   {
      MFEM_ABORT("Operator::Type is not supported, type = " << A.Type());
   }
   pop();
}

// *****************************************************************************
void BilinearForm::FormLinearSystem(const mfem::Array<int> &ess_tdof_list,
                                    mfem::Vector &x, mfem::Vector &b,
                                    mfem::OperatorHandle &A,
                                    mfem::Vector &X, mfem::Vector &B,
                                    int copy_interior)
{
   push();//assert(false); // ex1pd comes here, Laghos dont
   FormSystemMatrix(ess_tdof_list, A);
   kbform->InitRHS(ess_tdof_list, x, b, A.Ptr(), X, B, copy_interior);
   pop();
}

// *****************************************************************************
void BilinearForm::RecoverFEMSolution(const mfem::Vector &X,
                                      const mfem::Vector &b,
                                      mfem::Vector &x)
{
   push();
   kbform->KernelsRecoverFEMSolution(X, b, x);
   pop();
}

// *****************************************************************************
void BilinearForm::InitKBilinearForm()
{
   push();
   // Init 'kbform' using 'bform'
   MFEM_ASSERT(bform != NULL, "");
   MFEM_ASSERT(kbform == NULL, "");

   kFiniteElementSpace &ofes =
      bform->FESpace()->Get_PFESpace()->As<kFiniteElementSpace>();
   kbform = new kBilinearForm(&ofes);

   dbg(", transfer domain integrators");
   mfem::Array<mfem::BilinearFormIntegrator*> &dbfi = *bform->GetDBFI();
   for (int i = 0; i < dbfi.Size(); i++)
   {
      std::string integ_name(dbfi[i]->Name());
      dbg(", integ_name: %s",integ_name.c_str());
      Coefficient *scal_coeff = dbfi[i]->GetScalarCoefficient();
      ConstantCoefficient *const_coeff =
         dynamic_cast<ConstantCoefficient*>(scal_coeff);
      // TODO: other types of coefficients ...
      double val = const_coeff ? const_coeff->constant : 1.0;
      KernelsCoefficient coeff(kbform->engine(), val);
      KernelsIntegrator *integ = NULL;

      if (integ_name == "mass")
      {
         //MFEM_ABORT("BilinearFormIntegrator does not define Name()");
         integ = new KernelsMassIntegrator(coeff);
      }
      else if (integ_name == "diffusion")
      {
        //assert(false);
         integ = new KernelsDiffusionIntegrator(coeff);
      }
      else
      {
         MFEM_ABORT("BilinearFormIntegrator [Name() = " << integ_name
                    << "] is not supported");
      }

      const mfem::IntegrationRule *ir = dbfi[i]->GetIntRule();
      if (ir) { integ->SetIntegrationRule(*ir); }

      kbform->AddDomainIntegrator(integ);
   }
   pop();
   // TODO: other types of integrators ...
}

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)
