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

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)

#include "backend.hpp"
#include "bilinearform.hpp"
#include "adiffusioninteg.hpp"

namespace mfem
{

namespace omp
{

void BilinearForm::TransferIntegrators()
{
   mfem::Array<mfem::BilinearFormIntegrator*> &dbfi = *bform->GetDBFI();
   for (int i = 0; i < dbfi.Size(); i++)
   {
      std::string integ_name(dbfi[i]->Name());
      Coefficient *scal_coeff = dbfi[i]->GetScalarCoefficient();
      ConstantCoefficient *const_coeff =
         dynamic_cast<ConstantCoefficient*>(scal_coeff);
      // TODO: other types of coefficients ...
      double val = const_coeff ? const_coeff->constant : 1.0;

      if (integ_name == "(undefined)")
      {
         MFEM_ABORT("BilinearFormIntegrator does not define Name()");
      }
      else if (integ_name == "diffusion")
      {
	 switch (OmpEngine().MultType())
	 {
	 case Acrotensor:
	    tbfi.Append(new AcroDiffusionIntegrator(*scal_coeff, bform->FESpace()->Get_PFESpace()->As<FiniteElementSpace>()));
	    break;
	 default:
	    mfem_error("integrator is not supported for any MultType");
	    break;
	 }
      }
      else
      {
         MFEM_ABORT("BilinearFormIntegrator [Name() = " << integ_name
                    << "] is not supported");
      }
   }
}

bool BilinearForm::Assemble()
{
   if (!has_assembled)
   {
      TransferIntegrators();
      has_assembled = true;
   }

   return true;
}


} // namespace mfem::omp

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)
