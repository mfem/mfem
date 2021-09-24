// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Implementations of classes FABilinearFormExtension, EABilinearFormExtension,
// PABilinearFormExtension and MFBilinearFormExtension.

#include "linearform.hpp"

namespace mfem
{

LinearFormExtension::LinearFormExtension(LinearForm *lf) : lf(lf) { }

LinearFormExtension::~LinearFormExtension() { }

PALinearFormExtension::PALinearFormExtension(LinearForm *lf):
   LinearFormExtension(lf),
   fes(lf->FESpace()),
   domain_integs(*lf->GetDLFI()),
   domain_integs_marker(*lf->GetDLFIM()) { }

void PALinearFormExtension::Assemble()
{
   const int NE = fes->GetNE();

   mark.SetSize(NE);
   mark.UseDevice(true);

   lf->Vector::operator=(0.0);

   MFEM_VERIFY(lf->GetBLFI()->Size() == 0 &&
               lf->GetDLFI_Delta()->Size() == 0 &&
               lf->GetBLFI()->Size() == 0 &&
               lf->GetIFLFI()->Size() == 0 &&
               lf->GetFLFI()->Size() == 0,
               "integrators are not supported yet");

   for (int k = 0; k < domain_integs.Size(); ++k)
   {
      if (domain_integs_marker[k] != NULL)
      {
         MFEM_VERIFY(fes->GetMesh()->attributes.Size() ==
                     domain_integs_marker[k]->Size(),
                     "invalid element marker for domain linear form "
                     "integrator #" << k << ", counting from zero");
      }

      mark = 0.0;
      mark.HostReadWrite();

      for (int i = 0; i < NE; i++)
      {
         const int elem_attr = fes->GetMesh()->GetAttribute(i);
         if (domain_integs_marker[k] == NULL ||
             (*(domain_integs_marker[k]))[elem_attr-1] == 1)
         {
            mark[i] = 1.0;
         }
      }
      domain_integs[k]->AssemblePA(*fes, mark, *lf);
   }
}

} // namespace mfem
