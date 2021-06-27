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

void LORBase::AddIntegrators(BilinearForm &a_from,
                             BilinearForm &a_to,
                             GetIntegratorsFn get_integrators,
                             AddIntegratorFn add_integrator,
                             const IntegrationRule *ir)
{
   Array<BilinearFormIntegrator*> *integrators = (a_from.*get_integrators)();
   for (int i=0; i<integrators->Size(); ++i)
   {
      (a_to.*add_integrator)((*integrators)[i]);
      ir_map[(*integrators)[i]] = ((*integrators)[i])->GetIntegrationRule();
      if (ir) { ((*integrators)[i])->SetIntegrationRule(*ir); }
   }
}

void LORBase::AddIntegratorsAndMarkers(BilinearForm &a_from,
                                       BilinearForm &a_to,
                                       GetIntegratorsFn get_integrators,
                                       GetMarkersFn get_markers,
                                       AddIntegratorMarkersFn add_integrator,
                                       const IntegrationRule *ir)
{
   Array<BilinearFormIntegrator*> *integrators = (a_from.*get_integrators)();
   Array<Array<int>*> *markers = (a_from.*get_markers)();

   for (int i=0; i<integrators->Size(); ++i)
   {
      (a_to.*add_integrator)((*integrators)[i], *(*markers[i]));
      ir_map[(*integrators)[i]] = ((*integrators)[i])->GetIntegrationRule();
      if (ir) { ((*integrators)[i])->SetIntegrationRule(*ir); }
   }
}

void LORBase::ResetIntegrationRules(GetIntegratorsFn get_integrators)
{
   Array<BilinearFormIntegrator*> *integrators = (a->*get_integrators)();
   for (int i=0; i<integrators->Size(); ++i)
   {
      ((*integrators)[i])->SetIntegrationRule(*ir_map[(*integrators)[i]]);
   }
}

LORBase::FESpaceType LORBase::GetFESpaceType() const
{
   const FiniteElementCollection *fec = fes_ho.FEColl();
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

void LORBase::ConstructLocalDofPermutation(Array<int> &perm_) const
{
   FESpaceType type = GetFESpaceType();
   MFEM_VERIFY(type != H1 && type != L2, "");

   auto get_dof_map = [](FiniteElementSpace &fes, int i)
   {
      const FiniteElement *fe = fes.GetFE(i);
      auto tfe = dynamic_cast<const TensorBasisElement*>(fe);
      MFEM_ASSERT(tfe != NULL, "");
      return tfe->GetDofMap();
   };

   FiniteElementSpace &fes_lor = *fes;
   Mesh &mesh_lor = *fes_lor.GetMesh();
   int dim = mesh_lor.Dimension();
   const CoarseFineTransformations &cf_tr = mesh_lor.GetRefinementTransforms();

   using GeomRef = std::pair<Geometry::Type, int>;
   std::map<GeomRef, int> point_matrices_offsets;
   perm_.SetSize(fes_lor.GetVSize());

   Array<int> vdof_ho, vdof_lor;
   for (int ilor=0; ilor<mesh_lor.GetNE(); ++ilor)
   {
      int iho = cf_tr.embeddings[ilor].parent;
      int p = fes_ho.GetOrder(iho);
      int lor_index = cf_tr.embeddings[ilor].matrix;
      // We use the point matrix index to identify the local LOR element index
      // within the high-order coarse element.
      //
      // In variable-order spaces, the point matrices for each order are
      // concatenated sequentially, so for the given element order, we need to
      // find the offset that will give us the point matrix index relative to
      // the current element order only.
      GeomRef id(mesh_lor.GetElementBaseGeometry(ilor), p);
      if (point_matrices_offsets.find(id) == point_matrices_offsets.end())
      {
         point_matrices_offsets[id] = lor_index;
      }
      lor_index -= point_matrices_offsets[id];

      fes_ho.GetElementVDofs(iho, vdof_ho);
      fes_lor.GetElementVDofs(ilor, vdof_lor);

      if (type == L2)
      {
         perm_[vdof_lor[0]] = vdof_ho[lor_index];
         continue;
      }

      int p1 = p+1;
      int ndof_per_dim = (dim == 2) ? p*p1 : type == ND ? p*p1*p1 : p*p*p1;

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
               perm_[absdof(idof_lor)] = s < 0 ? -1-absdof(i) : absdof(i);
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
      else if (type == RT)
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

void LORBase::ConstructDofPermutation() const
{
   FESpaceType type = GetFESpaceType();
   if (type == H1 || type == L2 || nonconforming)
   {
      // H1 and L2: no permutation necessary, return identity
      perm.SetSize(fes->GetTrueVSize());
      for (int i=0; i<perm.Size(); ++i) { perm[i] = i; }
      return;
   }

#ifdef MFEM_USE_MPI
   ParFiniteElementSpace *pfes_ho
      = dynamic_cast<ParFiniteElementSpace*>(&fes_ho);
   ParFiniteElementSpace *pfes_lor = dynamic_cast<ParFiniteElementSpace*>(fes);
   if (pfes_ho && pfes_lor)
   {
      Array<int> l_perm;
      ConstructLocalDofPermutation(l_perm);
      perm.SetSize(pfes_lor->GetTrueVSize());
      for (int i=0; i<l_perm.Size(); ++i)
      {
         int j = l_perm[i];
         int s = j < 0 ? -1 : 1;
         int t_i = pfes_lor->GetLocalTDofNumber(i);
         int t_j = pfes_ho->GetLocalTDofNumber(absdof(j));
         // Either t_i and t_j both -1, or both non-negative
         if ((t_i < 0 && t_j >=0) || (t_j < 0 && t_i >= 0))
         {
            MFEM_ABORT("Inconsistent DOF numbering");
         }
         if (t_i < 0) { continue; }
         perm[t_i] = s < 0 ? -1 - t_j : t_j;
      }
   }
   else
#endif
   {
      ConstructLocalDofPermutation(perm);
   }
}

const Array<int> &LORBase::GetDofPermutation() const
{
   if (perm.Size() == 0) { ConstructDofPermutation(); }
   return perm;
}

bool LORBase::HasSameDofNumbering() const
{
   FESpaceType type = GetFESpaceType();
   return type == H1 || type == L2;
}

bool LORBase::RequiresDofPermutation() const
{
   return (HasSameDofNumbering() || nonconforming) ? false : true;
}

const OperatorHandle &LORBase::GetAssembledSystem() const
{
   MFEM_VERIFY(a != NULL && A.Ptr() != NULL, "No LOR system assembled");
   return A;
}

void LORBase::AssembleSystem_(BilinearForm &a_ho, const Array<int> &ess_dofs)
{
   a->UseExternalIntegrators();
   AddIntegrators(a_ho, *a, &BilinearForm::GetDBFI,
                  &BilinearForm::AddDomainIntegrator, ir_el);
   AddIntegrators(a_ho, *a, &BilinearForm::GetFBFI,
                  &BilinearForm::AddInteriorFaceIntegrator, ir_face);
   AddIntegratorsAndMarkers(a_ho, *a, &BilinearForm::GetBBFI,
                            &BilinearForm::GetBBFI_Marker,
                            &BilinearForm::AddBoundaryIntegrator, ir_face);
   AddIntegratorsAndMarkers(a_ho, *a, &BilinearForm::GetBFBFI,
                            &BilinearForm::GetBFBFI_Marker,
                            &BilinearForm::AddBdrFaceIntegrator, ir_face);
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
      Array<int> ess_dofs_perm(ess_dofs.Size());
      for (int i=0; i<ess_dofs.Size(); ++i)
      {
         ess_dofs_perm[i] = pi[ess_dofs[i]];
      }
      a->FormSystemMatrix(ess_dofs_perm, A);
   }
   else
   {
      a->FormSystemMatrix(ess_dofs, A);
   }
   ResetIntegrationRules(&BilinearForm::GetDBFI);
   ResetIntegrationRules(&BilinearForm::GetFBFI);
   ResetIntegrationRules(&BilinearForm::GetBBFI);
   ResetIntegrationRules(&BilinearForm::GetBFBFI);
}

void LORBase::SetupNonconforming()
{
   if (!HasSameDofNumbering())
   {
      Array<int> p;
      ConstructLocalDofPermutation(p);
      fes->CopyProlongationAndRestriction(fes_ho, &p);
   }
   else
   {
      fes->CopyProlongationAndRestriction(fes_ho, NULL);
   }
   nonconforming = fes->Nonconforming();
}

template <typename FEC>
void CheckScalarBasisType(const FiniteElementSpace &fes)
{
   const FEC *fec = dynamic_cast<const FEC*>(fes.FEColl());
   if (fec)
   {
      int btype = fec->GetBasisType();
      if (btype != BasisType::GaussLobatto)
      {
         mfem::err << "\nWARNING: Constructing low-order refined "
                   << "discretization with basis type\n"
                   << BasisType::Name(btype) << ". "
                   << "The LOR discretization is only spectrally equivalent\n"
                   << "with Gauss-Lobatto basis.\n" << std::endl;
      }
   }
}

template <typename FEC>
void CheckVectorBasisType(const FiniteElementSpace &fes)
{
   const FEC *fec = dynamic_cast<const FEC*>(fes.FEColl());
   if (fec)
   {
      int cbtype = fec->GetClosedBasisType();
      int obtype = fec->GetOpenBasisType();
      if (cbtype != BasisType::GaussLobatto || obtype != BasisType::IntegratedGLL)
      {
         mfem::err << "\nWARNING: Constructing vector low-order refined "
                   << "discretization with basis type \npair ("
                   << BasisType::Name(cbtype) << ", "
                   << BasisType::Name(obtype) << "). "
                   << "The LOR discretization is only spectrally\nequivalent "
                   << "with basis types (Gauss-Lobatto, IntegratedGLL).\n"
                   << std::endl;
      }
   }
}

void CheckBasisType(const FiniteElementSpace &fes)
{
   CheckScalarBasisType<H1_FECollection>(fes);
   CheckVectorBasisType<ND_FECollection>(fes);
   CheckVectorBasisType<RT_FECollection>(fes);
   // L2 is a bit more complicated, for now don't verify basis type
}

LORBase::LORBase(FiniteElementSpace &fes_ho_)
   : irs(0, Quadrature1D::GaussLobatto), fes_ho(fes_ho_)
{
   Mesh &mesh_ = *fes_ho_.GetMesh();
   int dim = mesh_.Dimension();
   Array<Geometry::Type> geoms;
   mesh_.GetGeometries(dim, geoms);
   if (geoms.Size() == 1 && Geometry::IsTensorProduct(geoms[0]))
   {
      ir_el = &irs.Get(geoms[0], 1);
      ir_face = &irs.Get(Geometry::TensorProductGeometry(dim-1), 1);
   }
   else
   {
      ir_el = NULL;
      ir_face = NULL;
   }
   a = NULL;
}

LORBase::~LORBase()
{
   delete a;
   delete fes;
   delete fec;
   delete mesh;
}

LORDiscretization::LORDiscretization(BilinearForm &a_ho_,
                                     const Array<int> &ess_tdof_list,
                                     int ref_type)
   : LORDiscretization(*a_ho_.FESpace(), ref_type)
{
   AssembleSystem(a_ho_, ess_tdof_list);
}

LORDiscretization::LORDiscretization(FiniteElementSpace &fes_ho,
                                     int ref_type) : LORBase(fes_ho)
{
   CheckBasisType(fes_ho);

   Mesh &mesh_ho = *fes_ho.GetMesh();
   // For H1, ND and RT spaces, use refinement = element order, for DG spaces,
   // use refinement = element order + 1 (since LOR is p = 0 in this case).
   int increment = (GetFESpaceType() == L2) ? 1 : 0;
   Array<int> refinements(mesh_ho.GetNE());
   for (int i=0; i<refinements.Size(); ++i)
   {
      refinements[i] = fes_ho.GetOrder(i) + increment;
   }
   mesh = new Mesh(Mesh::MakeRefined(mesh_ho, refinements, ref_type));

   fec = fes_ho.FEColl()->Clone(GetLOROrder());
   fes = new FiniteElementSpace(mesh, fec);
   if (fes_ho.Nonconforming()) { SetupNonconforming(); }

   A.SetType(Operator::MFEM_SPARSEMAT);
}

void LORDiscretization::AssembleSystem(BilinearForm &a_ho,
                                       const Array<int> &ess_dofs)
{
   delete a;
   a = new BilinearForm(&GetFESpace());
   AssembleSystem_(a_ho, ess_dofs);
}

SparseMatrix &LORDiscretization::GetAssembledMatrix() const
{
   MFEM_VERIFY(a != NULL && A.Ptr() != NULL, "No LOR system assembled");
   return *A.As<SparseMatrix>();
}

#ifdef MFEM_USE_MPI

ParLORDiscretization::ParLORDiscretization(ParBilinearForm &a_ho_,
                                           const Array<int> &ess_tdof_list,
                                           int ref_type)
   : ParLORDiscretization(*a_ho_.ParFESpace(), ref_type)
{
   AssembleSystem(a_ho_, ess_tdof_list);
}

ParLORDiscretization::ParLORDiscretization(ParFiniteElementSpace &fes_ho,
                                           int ref_type) : LORBase(fes_ho)
{
   if (fes_ho.GetMyRank() == 0) { CheckBasisType(fes_ho); }
   // TODO: support variable-order spaces in parallel
   MFEM_VERIFY(!fes_ho.IsVariableOrder(),
               "Cannot construct LOR operators on variable-order spaces");

   int order = fes_ho.GetMaxElementOrder();
   if (GetFESpaceType() == L2) { ++order; }

   ParMesh &mesh_ho = *fes_ho.GetParMesh();
   ParMesh *pmesh = new ParMesh(ParMesh::MakeRefined(mesh_ho, order, ref_type));
   mesh = pmesh;

   fec = fes_ho.FEColl()->Clone(GetLOROrder());
   ParFiniteElementSpace *pfes = new ParFiniteElementSpace(pmesh, fec);
   fes = pfes;
   if (fes_ho.Nonconforming()) { SetupNonconforming(); }

   A.SetType(Operator::Hypre_ParCSR);
}

void ParLORDiscretization::AssembleSystem(ParBilinearForm &a_ho,
                                          const Array<int> &ess_dofs)
{
   delete a;
   a = new ParBilinearForm(&GetParFESpace());
   AssembleSystem_(a_ho, ess_dofs);
}

HypreParMatrix &ParLORDiscretization::GetAssembledMatrix() const
{
   MFEM_VERIFY(a != NULL && A.Ptr() != NULL, "No LOR system assembled");
   return *A.As<HypreParMatrix>();
}

ParFiniteElementSpace &ParLORDiscretization::GetParFESpace() const
{
   return static_cast<ParFiniteElementSpace&>(*fes);
}

#endif

} // namespace mfem
