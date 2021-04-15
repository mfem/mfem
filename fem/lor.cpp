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

void LORBase::AddIntegrators(BilinearForm &a_to,
                             GetIntegratorsFn get_integrators,
                             AddIntegratorFn add_integrator)
{
   Array<BilinearFormIntegrator*> *integrators = (a_ho.*get_integrators)();
   for (int i=0; i<integrators->Size(); ++i)
   {
      (a_to.*add_integrator)(*integrators[i]);
      ir_map[*integrators[i]] = (*integrators[i])->GetIntegrationRule();
      if (ir) { (*integrators[i])->SetIntegrationRule(*ir); }
   }
}

void LORBase::AddIntegratorsAndMarkers(BilinearForm &a_to,
                                       GetIntegratorsFn get_integrators,
                                       GetMarkersFn get_markers,
                                       AddIntegratorMarkersFn add_integrator)
{
   Array<BilinearFormIntegrator*> *integrators = (a_ho.*get_integrators)();
   Array<Array<int>*> *markers = (a_ho.*get_markers)();

   for (int i=0; i<integrators->Size(); ++i)
   {
      (a_to.*add_integrator)(*integrators[i], *(*markers[i]));
      ir_map[*integrators[i]] = (*integrators[i])->GetIntegrationRule();
      if (ir) { (*integrators[i])->SetIntegrationRule(*ir); }
   }
}

void LORBase::AddIntegrators(BilinearForm &a_to)
{
   a_to.UseExternalIntegrators();
   AddIntegrators(a_to, &BilinearForm::GetDBFI,
                  &BilinearForm::AddDomainIntegrator);
   AddIntegrators(a_to, &BilinearForm::GetFBFI,
                  &BilinearForm::AddInteriorFaceIntegrator);

   AddIntegratorsAndMarkers(a_to, &BilinearForm::GetBBFI,
                            &BilinearForm::GetBBFI_Marker,
                            &BilinearForm::AddBoundaryIntegrator);
   AddIntegratorsAndMarkers(a_to, &BilinearForm::GetBFBFI,
                            &BilinearForm::GetBFBFI_Marker,
                            &BilinearForm::AddBdrFaceIntegrator);
}

void LORBase::ResetIntegrationRules(GetIntegratorsFn get_integrators)
{
   Array<BilinearFormIntegrator*> *integrators = (a_ho.*get_integrators)();
   for (int i=0; i<integrators->Size(); ++i)
   {
      (*integrators[i])->SetIntegrationRule(*ir_map[*integrators[i]]);
   }
}

void LORBase::ResetIntegrationRules()
{
   ResetIntegrationRules(&BilinearForm::GetDBFI);
   ResetIntegrationRules(&BilinearForm::GetFBFI);
   ResetIntegrationRules(&BilinearForm::GetBBFI);
   ResetIntegrationRules(&BilinearForm::GetBFBFI);
}

OperatorHandle &LORBase::GetAssembledSystem()
{
   return A;
}

LORBase::LORBase(BilinearForm &a_)
   : irs(0, Quadrature1D::GaussLobatto), a_ho(a_)
{
   Mesh &mesh = *a_ho.FESpace()->GetMesh();
   int dim = mesh.Dimension();
   Array<Geometry::Type> geoms;
   mesh.GetGeometries(dim, geoms);
   if (geoms.Size() == 1 && Geometry::IsTensorProduct(geoms[0]))
   {
      ir = &irs.Get(geoms[0], 1);
   }
   else
   {
      ir = NULL;
   }
}

LORBase::~LORBase()
{
   delete a;
   delete fes;
   delete fec;
   delete mesh;
}

LOR::LOR(BilinearForm &a_ho_, const Array<int> &ess_tdof_list, int ref_type)
   : LORBase(a_ho_)
{
   FiniteElementSpace &fes_ho = *a_ho.FESpace();
   MFEM_VERIFY(!fes_ho.IsDGSpace(),
               "Cannot construct LOR operators on DG spaces");
   // TODO: support variable-order spaces
   MFEM_VERIFY(!fes_ho.IsVariableOrder(),
               "Cannot construct LOR operators on variable-order spaces");

   int order = fes_ho.GetMaxElementOrder();

   Mesh &mesh_ho = *fes_ho.GetMesh();
   mesh = new Mesh(Mesh::MakeRefined(mesh_ho, order, ref_type));

   fec = fes_ho.FEColl()->Clone(1);
   fes = new FiniteElementSpace(mesh, fec);
   a = new BilinearForm(fes);

   AddIntegrators(*a);
   a->Assemble();
   A.SetType(Operator::MFEM_SPARSEMAT);
   a->FormSystemMatrix(ess_tdof_list, A);

   ResetIntegrationRules();
}

SparseMatrix &LOR::GetAssembledMatrix()
{
   return *A.As<SparseMatrix>();
}

#ifdef MFEM_USE_MPI

ParLOR::ParLOR(ParBilinearForm &a_ho_, const Array<int> &ess_tdof_list,
               int ref_type) : LORBase(a_ho_)
{
   ParFiniteElementSpace &fes_ho = *a_ho_.ParFESpace();
   MFEM_VERIFY(!fes_ho.IsDGSpace(),
               "Cannot construct LOR operators on DG spaces");
   // TODO: support variable-order spaces
   MFEM_VERIFY(!fes_ho.IsVariableOrder(),
               "Cannot construct LOR operators on variable-order spaces");

   int order = fes_ho.GetMaxElementOrder();

   ParMesh &mesh_ho = *fes_ho.GetParMesh();
   ParMesh *pmesh = new ParMesh(ParMesh::MakeRefined(mesh_ho, order, ref_type));
   mesh = pmesh;

   fec = fes_ho.FEColl()->Clone(1);
   ParFiniteElementSpace *pfes = new ParFiniteElementSpace(pmesh, fec);
   fes = pfes;
   a = new ParBilinearForm(pfes);

   AddIntegrators(*a);
   a->Assemble();
   A.SetType(Operator::Hypre_ParCSR);
   a->FormSystemMatrix(ess_tdof_list, A);

   ResetIntegrationRules();
}

HypreParMatrix &ParLOR::GetAssembledMatrix()
{
   return *A.As<HypreParMatrix>();
}

#endif

} // namespace mfem
