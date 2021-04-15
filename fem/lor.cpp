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
      (a_to.*add_integrator)((*integrators)[i]);
      ir_map[(*integrators)[i]] = ((*integrators)[i])->GetIntegrationRule();
      if (ir) { ((*integrators)[i])->SetIntegrationRule(*ir); }
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
      (a_to.*add_integrator)((*integrators)[i], *(*markers[i]));
      ir_map[(*integrators)[i]] = ((*integrators)[i])->GetIntegrationRule();
      if (ir) { ((*integrators)[i])->SetIntegrationRule(*ir); }
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
      ((*integrators)[i])->SetIntegrationRule(*ir_map[(*integrators)[i]]);
   }
}

void LORBase::ResetIntegrationRules()
{
   ResetIntegrationRules(&BilinearForm::GetDBFI);
   ResetIntegrationRules(&BilinearForm::GetFBFI);
   ResetIntegrationRules(&BilinearForm::GetBBFI);
   ResetIntegrationRules(&BilinearForm::GetBFBFI);
}

LORBase::FESpaceType LORBase::GetFESpaceType() const
{
   const FiniteElementCollection *fec = a_ho.FESpace()->FEColl();
   if (dynamic_cast<const H1_FECollection*>(fec)) { return H1; }
   else if (dynamic_cast<const ND_FECollection*>(fec)) { return ND; }
   else if (dynamic_cast<const RT_FECollection*>(fec)) { return RT; }
   else if (dynamic_cast<const L2_FECollection*>(fec)) { return L2; }
   else { MFEM_ABORT("Bad LOR space type."); }
   return INVALID;
}

int LORBase::GetLOROrder() const
{
   FESpaceType type = GetFESpaceType();
   return (type == L2 || type == RT) ? 0 : 1;
}

void LORBase::ConstructDofPermutation() const
{
   FESpaceType type = GetFESpaceType();

   MFEM_VERIFY(type != L2, ""); // TODO: implement for DG

   if (type == H1)
   {
      // H1: no permutation necessary, return identity
      perm.SetSize(fes->GetVSize());
      for (int i=0; i<perm.Size(); ++i) { perm[i] = i; }
      return;
   }

   FiniteElementSpace &fes_ho = *a_ho.FESpace();
   FiniteElementSpace &fes_lor = *fes;


   auto get_dof_map = [](FiniteElementSpace &fes, int i)
   {
      const FiniteElement *fe = fes.GetFE(i);
      auto tfe = dynamic_cast<const TensorBasisElement*>(fe);
      MFEM_ASSERT(tfe != NULL, "");
      return tfe->GetDofMap();
   };

   perm.SetSize(fes_lor.GetVSize());
   Array<int> vdof_ho, vdof_lor;

   Mesh &mesh_lor = *fes_lor.GetMesh();
   int dim = mesh_lor.Dimension();
   const CoarseFineTransformations &cf_tr = mesh_lor.GetRefinementTransforms();
   for (int ilor=0; ilor<mesh_lor.GetNE(); ++ilor)
   {
      int iho = cf_tr.embeddings[ilor].parent;
      int lor_index = cf_tr.embeddings[ilor].matrix;

      int p = fes_ho.GetOrder(iho);
      int p1 = p+1;
      int ndof_per_dim = (dim == 2) ? p*p1 : type == ND ? p*p1*p1 : p*p*p1;

      fes_ho.GetElementVDofs(iho, vdof_ho);
      fes_lor.GetElementVDofs(ilor, vdof_lor);

      const Array<int> &dofmap_ho = get_dof_map(fes_ho, iho);
      const Array<int> &dofmap_lor = get_dof_map(fes_lor, ilor);

      int off_x = lor_index % p;
      int off_y = (lor_index / p) % p;
      int off_z = (lor_index / p) / p;

      auto set_perm = [&](int off_lor, int off_ho, int n1, int n2)
      {
         for (int i1=0; i1<2; ++i1)
         {
            int m = (dim == 2 || type == RT) ? 1 : 2;
            for (int i2=0; i2<m; ++i2)
            {
               int i;
               i = dofmap_lor[off_lor + i1 + i2*2];
               int s1 = i < 0 ? -1 : 1;
               int idof_lor = vdof_lor[absdof(i)];
               i = dofmap_ho[off_ho + i1*n1 + i2*n2];
               int s2 = i < 0 ? -1 : 1;
               int idof_ho = vdof_ho[absdof(i)];
               int s3 = idof_lor < 0 ? -1 : 1;
               int s4 = idof_ho < 0 ? -1 : 1;
               int s = s1*s2*s3*s4;
               i = absdof(idof_ho);
               perm[absdof(idof_lor)] = s < 0 ? -1-absdof(i) : absdof(i);
            }
         }
      };

      int offset;

      if (type == ND)
      {
         // x
         offset = off_x + off_y*p + off_z*p*p1;
         set_perm(0, offset, p, p*p1);
         // y
         offset = ndof_per_dim + off_x + off_y*(p1) + off_z*p1*p;
         set_perm(dim == 2 ? 2 : 4, offset, 1, p*p1);
         // z
         if (dim == 3)
         {
            offset = 2*ndof_per_dim + off_x + off_y*p1 + off_z*p1*p1;
            set_perm(8, offset, 1, p+1);
         }
      }
      else
      {
         // x
         offset = off_x + off_y*p1 + off_z*p*p1;
         set_perm(0, offset, 1, 0);
         // y
         offset = ndof_per_dim + off_x + off_y*p + off_z*p1*p;
         set_perm(2, offset, p, 0);
         // z
         if (dim == 3)
         {
            offset = 2*ndof_per_dim + off_x + off_y*p + off_z*p*p;
            set_perm(4, offset, p*p, 0);
         }
      }
   }
}

const Array<int> &LORBase::GetDofPermutation() const
{
   if (perm.Size() == 0) { ConstructDofPermutation(); }
   return perm;
}

bool LORBase::RequiresDofPermutation() const
{
   return (GetFESpaceType() == H1) ? false : true;
}

const OperatorHandle &LORBase::GetAssembledSystem() const
{
   return A;
}

void LORBase::AssembleSystem(const Array<int> &ess_tdof_list)
{
   AddIntegrators(*a);
   a->Assemble();
   if (RequiresDofPermutation())
   {
      const Array<int> &p = GetDofPermutation();
      // Form inverse permutation: given high-order dof i, pi[i] is corresp. LO
      Array<int> pi(p.Size());
      for (int i=0; i<p.Size(); ++i)
      {
         pi[absdof(p[i])] = i;
      }
      Array<int> ess_tdof_list_perm(ess_tdof_list.Size());
      for (int i=0; i<ess_tdof_list.Size(); ++i)
      {
         ess_tdof_list_perm[i] = pi[ess_tdof_list[i]];
      }
      a->FormSystemMatrix(ess_tdof_list_perm, A);
   }
   else
   {
      a->FormSystemMatrix(ess_tdof_list, A);
   }
   ResetIntegrationRules();
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

   fec = fes_ho.FEColl()->Clone(GetLOROrder());
   fes = new FiniteElementSpace(mesh, fec);
   a = new BilinearForm(fes);
   A.SetType(Operator::MFEM_SPARSEMAT);

   AssembleSystem(ess_tdof_list);
}

SparseMatrix &LOR::GetAssembledMatrix() const
{
   return *A.As<SparseMatrix>();
}

#ifdef MFEM_USE_MPI

ParLOR::ParLOR(ParBilinearForm &a_ho_, const Array<int> &ess_tdof_list,
               int ref_type) : LORBase(a_ho_)
{
   ParFiniteElementSpace &fes_ho = *a_ho_.ParFESpace();
   // TODO: support DG
   MFEM_VERIFY(!fes_ho.IsDGSpace(),
               "Cannot construct LOR operators on DG spaces");
   // TODO: support variable-order spaces
   MFEM_VERIFY(!fes_ho.IsVariableOrder(),
               "Cannot construct LOR operators on variable-order spaces");

   int order = fes_ho.GetMaxElementOrder();

   ParMesh &mesh_ho = *fes_ho.GetParMesh();
   ParMesh *pmesh = new ParMesh(ParMesh::MakeRefined(mesh_ho, order, ref_type));
   mesh = pmesh;

   fec = fes_ho.FEColl()->Clone(GetLOROrder());
   ParFiniteElementSpace *pfes = new ParFiniteElementSpace(pmesh, fec);
   fes = pfes;
   a = new ParBilinearForm(pfes);

   AssembleSystem(ess_tdof_list);
}

HypreParMatrix &ParLOR::GetAssembledMatrix() const
{
   return *A.As<HypreParMatrix>();
}

ParFiniteElementSpace &ParLOR::GetParFESpace() const
{
   return static_cast<ParFiniteElementSpace&>(*fes);
}

#endif

} // namespace mfem
