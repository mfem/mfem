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

#include "lor.hpp"
#include "pbilinearform.hpp"

namespace mfem
{

using GetIntegratorsFn = Array<BilinearFormIntegrator*> *(BilinearForm::*)();
using GetMarkersFn = Array<Array<int>*> *(BilinearForm::*)();
using AddIntegratorFn = void (BilinearForm::*)(BilinearFormIntegrator*);
using AddIntegratorMarkersFn =
   void (BilinearForm::*)(BilinearFormIntegrator*, Array<int>&);

void AddIntegrators(BilinearForm &a_from, BilinearForm &a_to,
                    GetIntegratorsFn get_integrators,
                    AddIntegratorFn add_integrator)
{
   Array<BilinearFormIntegrator*> *integrators = (a_from.*get_integrators)();
   for (int i=0; i<integrators->Size(); ++i)
   {
      (a_to.*add_integrator)(*integrators[i]);
   }
}

void AddIntegratorsAndMarkers(BilinearForm &a_from, BilinearForm &a_to,
                              GetIntegratorsFn get_integrators,
                              GetMarkersFn get_markers,
                              AddIntegratorMarkersFn add_integrator)
{
   Array<BilinearFormIntegrator*> *integrators = (a_from.*get_integrators)();
   Array<Array<int>*> *markers = (a_from.*get_markers)();

   for (int i=0; i<integrators->Size(); ++i)
   {
      (a_to.*add_integrator)(*integrators[i], *(*markers[i]));
   }
}

LOR::LOR(BilinearForm &a_ho, const Array<int> &ess_tdof_list, int ref_type)
{
   FiniteElementSpace &fes_ho = *a_ho.FESpace();
   MFEM_VERIFY(!fes_ho.IsDGSpace(),
               "Cannot construct LOR operators on DG spaces");
   // TODO: support variable-order spaces
   MFEM_VERIFY(!fes_ho.IsVariableOrder(),
               "Cannot construct LOR operators on variable-order spaces");

   int order = fes_ho.GetMaxElementOrder();

   Mesh &mesh_ho = *fes_ho.GetMesh();
   mesh = Mesh::MakeRefined(mesh_ho, order, ref_type);

   fec = fes_ho.FEColl()->Clone(1);
   fes = new FiniteElementSpace(&mesh, fec);
   a = new BilinearForm(fes);

   a->UseExternalIntegrators();

   AddIntegrators(a_ho, *a, &BilinearForm::GetDBFI,
                  &BilinearForm::AddDomainIntegrator);
   AddIntegrators(a_ho, *a, &BilinearForm::GetFBFI,
                  &BilinearForm::AddInteriorFaceIntegrator);

   AddIntegratorsAndMarkers(a_ho, *a, &BilinearForm::GetBBFI,
                            &BilinearForm::GetBBFI_Marker,
                            &BilinearForm::AddBoundaryIntegrator);
   AddIntegratorsAndMarkers(a_ho, *a, &BilinearForm::GetBFBFI,
                            &BilinearForm::GetBFBFI_Marker,
                            &BilinearForm::AddBdrFaceIntegrator);

   a->FormSystemMatrix(ess_tdof_list, A);
}

SparseMatrix &LOR::GetAssembledMatrix()
{
   return A;
}

LOR::~LOR()
{
   delete a;
   delete fes;
   delete fec;
}

#ifdef MFEM_USE_MPI

ParLOR::ParLOR(ParBilinearForm &a_ho, const Array<int> &ess_tdof_list,
               int ref_type)
{
   ParFiniteElementSpace &fes_ho = *a_ho.ParFESpace();
   MFEM_VERIFY(!fes_ho.IsDGSpace(),
               "Cannot construct LOR operators on DG spaces");
   // TODO: support variable-order spaces
   MFEM_VERIFY(!fes_ho.IsVariableOrder(),
               "Cannot construct LOR operators on variable-order spaces");

   int order = fes_ho.GetMaxElementOrder();

   ParMesh &mesh_ho = *fes_ho.GetParMesh();
   mesh = ParMesh::MakeRefined(mesh_ho, order, ref_type);

   fec = fes_ho.FEColl()->Clone(1);
   fes = new ParFiniteElementSpace(&mesh, fec);
   a = new ParBilinearForm(fes);

   a->UseExternalIntegrators();

   AddIntegrators(a_ho, *a, &BilinearForm::GetDBFI,
                  &BilinearForm::AddDomainIntegrator);
   AddIntegrators(a_ho, *a, &BilinearForm::GetFBFI,
                  &BilinearForm::AddInteriorFaceIntegrator);

   AddIntegratorsAndMarkers(a_ho, *a, &BilinearForm::GetBBFI,
                            &BilinearForm::GetBBFI_Marker,
                            &BilinearForm::AddBoundaryIntegrator);
   AddIntegratorsAndMarkers(a_ho, *a, &BilinearForm::GetBFBFI,
                            &BilinearForm::GetBFBFI_Marker,
                            &BilinearForm::AddBdrFaceIntegrator);

   a->Assemble();
   a->FormSystemMatrix(ess_tdof_list, A);
}

HypreParMatrix &ParLOR::GetAssembledMatrix()
{
   return A;
}

ParLOR::~ParLOR()
{
   delete a;
   delete fes;
   delete fec;
}

#endif

} // namespace mfem
