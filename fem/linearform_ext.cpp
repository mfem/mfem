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

#define MFEM_DEBUG_COLOR 201
#include "../general/debug.hpp"

#include "../general/forall.hpp"

namespace mfem
{

LinearFormExtension::LinearFormExtension(LinearForm *lf) : lf(lf) { }

LinearFormExtension::~LinearFormExtension() { }

PALinearFormExtension::PALinearFormExtension(LinearForm *lf):
   LinearFormExtension(lf),
   fes(*lf->FESpace()),
   mesh(*fes.GetMesh()),
   domain_integs(*lf->GetDLFI()),
   domain_integs_marker(*lf->GetDLFIM()), // element attribute marker
   NE(fes.GetNE()),
   mesh_attributes_size(fes.GetMesh()->attributes.Size())
{
   //dbg("NE:%d",NE);
   marks.SetSize(NE);
   marks.UseDevice(true);

   attributes.SetSize(NE);
   attributes.UseDevice(true);

   // Fill the attributes vector on host
   for (int i=0; i<NE; i++) { attributes[i] = fes.GetMesh()->GetAttribute(i); }
}

void PALinearFormExtension::Assemble()
{

   lf->Vector::operator=(0.0);

   MFEM_VERIFY(lf->GetBLFI()->Size() == 0 &&
               lf->GetDLFI_Delta()->Size() == 0 &&
               lf->GetBLFI()->Size() == 0 &&
               lf->GetIFLFI()->Size() == 0 &&
               lf->GetFLFI()->Size() == 0,
               "integrators are not supported yet");

   for (int k = 0; k < domain_integs.Size(); ++k)
   {
      if (domain_integs_marker[k] != nullptr)
      {
         MFEM_VERIFY(mesh_attributes_size == domain_integs_marker[k]->Size(),
                     "invalid element marker for domain linear form "
                     "integrator #" << k << ", counting from zero");
      }

      marks = 0.0;

      const Array<int> *dimks = domain_integs_marker[k];
      const bool no_dimk =  dimks == nullptr;
      const auto dimk = no_dimk ? nullptr : dimks->Read();
      const auto attr = attributes.Read();
      auto mark = marks.ReadWrite();

      MFEM_FORALL(i, NE,
      {
         const int elem_attr = attr[i];
         const bool elem_attr_eq_1 = no_dimk ? false : dimk[elem_attr-1] == 1;
         if (no_dimk || elem_attr_eq_1) { mark[i] = 1.0; }
      });

      domain_integs[k]->AssemblePA(fes, marks, *lf);
   }
   //assert(false);
}

} // namespace mfem
