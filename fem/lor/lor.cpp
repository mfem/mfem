// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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
#include "lor_batched.hpp"
#include "../restriction.hpp"
#include "../pbilinearform.hpp"
#include "../../general/forall.hpp"

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
      BilinearFormIntegrator *integrator = (*integrators)[i];
      (a_to.*add_integrator)(integrator);
      ir_map[integrator] = integrator->GetIntRule();
      if (ir) { integrator->SetIntegrationRule(*ir); }
   }
}

void LORBase::AddIntegratorsAndMarkers(BilinearForm &a_from,
                                       BilinearForm &a_to,
                                       GetIntegratorsFn get_integrators,
                                       GetMarkersFn get_markers,
                                       AddIntegratorMarkersFn add_integrator_marker,
                                       AddIntegratorFn add_integrator,
                                       const IntegrationRule *ir)
{
   Array<BilinearFormIntegrator*> *integrators = (a_from.*get_integrators)();
   Array<Array<int>*> &markers = *(a_from.*get_markers)();

   for (int i=0; i<integrators->Size(); ++i)
   {
      BilinearFormIntegrator *integrator = (*integrators)[i];
      if (markers[i] != nullptr)
      {
         (a_to.*add_integrator_marker)(integrator, *markers[i]);
      }
      else
      {
         (a_to.*add_integrator)(integrator);
      }
      ir_map[integrator] = integrator->GetIntRule();
      if (ir) { integrator->SetIntegrationRule(*ir); }
   }
}

void LORBase::ResetIntegrationRules(GetIntegratorsFn get_integrators)
{
   Array<BilinearFormIntegrator*> *integrators = (a->*get_integrators)();
   for (int i=0; i<integrators->Size(); ++i)
   {
      ((*integrators)[i])->SetIntRule(ir_map[(*integrators)[i]]);
   }
}

LORBase::FESpaceType LORBase::GetFESpaceType() const
{
   const FiniteElementCollection *fec_ho = fes_ho.FEColl();
   if (dynamic_cast<const H1_FECollection*>(fec_ho)) { return H1; }
   else if (dynamic_cast<const ND_FECollection*>(fec_ho)) { return ND; }
   else if (dynamic_cast<const RT_FECollection*>(fec_ho)) { return RT; }
   else if (dynamic_cast<const L2_FECollection*>(fec_ho)) { return L2; }
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

   auto get_dof_map = [](FiniteElementSpace &fes_, int i)
   {
      const FiniteElement *fe = fes_.GetFE(i);
      auto tfe = dynamic_cast<const TensorBasisElement*>(fe);
      MFEM_ASSERT(tfe != NULL, "");
      return tfe->GetDofMap();
   };

   FiniteElementSpace &fes_lor = GetFESpace();
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
   if (type == H1 || type == L2)
   {
      // H1 and L2: no permutation necessary, return identity
      perm.SetSize(fes_ho.GetTrueVSize());
      for (int i=0; i<perm.Size(); ++i) { perm[i] = i; }
      return;
   }

#ifdef MFEM_USE_MPI
   ParFiniteElementSpace *pfes_ho
      = dynamic_cast<ParFiniteElementSpace*>(&fes_ho);
   ParFiniteElementSpace *pfes_lor
      = dynamic_cast<ParFiniteElementSpace*>(&GetFESpace());
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

OperatorHandle &LORBase::GetAssembledSystem()
{
   MFEM_VERIFY(A.Ptr() != NULL, "No LOR system assembled");
   return A;
}

const OperatorHandle &LORBase::GetAssembledSystem() const
{
   MFEM_VERIFY(A.Ptr() != NULL, "No LOR system assembled");
   return A;
}

void LORBase::SetupProlongationAndRestriction()
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

LORBase::LORBase(FiniteElementSpace &fes_ho_, int ref_type_)
   : irs(0, Quadrature1D::GaussLobatto), ref_type(ref_type_), fes_ho(fes_ho_)
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

FiniteElementSpace &LORBase::GetFESpace() const
{
   // In the case of "batched assembly", the creation of the LOR mesh and
   // space can be completely omitted (for efficiency). In this case, the
   // fes object is NULL, and we need to create it when requested.
   if (fes == NULL) { const_cast<LORBase*>(this)->FormLORSpace(); }
   return *fes;
}

void LORBase::AssembleSystem(BilinearForm &a_ho, const Array<int> &ess_dofs)
{
   A.Clear();
   delete a;
   if (BatchedLORAssembly::FormIsSupported(a_ho))
   {
      // Skip forming the space
      a = nullptr;
      if (batched_lor == nullptr)
      {
         batched_lor = new BatchedLORAssembly(fes_ho);
      }
      batched_lor->Assemble(a_ho, ess_dofs, A);
   }
   else
   {
      LegacyAssembleSystem(a_ho, ess_dofs);
   }
}

void LORBase::LegacyAssembleSystem(BilinearForm &a_ho,
                                   const Array<int> &ess_dofs)
{
   // TODO: use AssemblyLevel::FULL here instead of AssemblyLevel::LEGACY.
   // This is waiting for parallel assembly + BCs with AssemblyLevel::FULL.
   // In that case, maybe "LegacyAssembleSystem" is not a very clear name.

   // If the space is not formed already, it will be constructed lazily in
   // GetFESpace.
   FiniteElementSpace &fes_lor = GetFESpace();
#ifdef MFEM_USE_MPI
   if (auto *pfes = dynamic_cast<ParFiniteElementSpace*>(&fes_lor))
   {
      a = new ParBilinearForm(pfes);
   }
   else
#endif
   {
      a = new BilinearForm(&fes_lor);
   }

   a->UseExternalIntegrators();
   AddIntegrators(a_ho, *a, &BilinearForm::GetDBFI,
                  &BilinearForm::AddDomainIntegrator, ir_el);
   AddIntegrators(a_ho, *a, &BilinearForm::GetFBFI,
                  &BilinearForm::AddInteriorFaceIntegrator, ir_face);
   AddIntegratorsAndMarkers(a_ho, *a, &BilinearForm::GetBBFI,
                            &BilinearForm::GetBBFI_Marker,
                            &BilinearForm::AddBoundaryIntegrator,
                            &BilinearForm::AddBoundaryIntegrator, ir_face);
   AddIntegratorsAndMarkers(a_ho, *a, &BilinearForm::GetBFBFI,
                            &BilinearForm::GetBFBFI_Marker,
                            &BilinearForm::AddBdrFaceIntegrator,
                            &BilinearForm::AddBdrFaceIntegrator, ir_face);

   a->Assemble();
   a->FormSystemMatrix(ess_dofs, A);

   ResetIntegrationRules(&BilinearForm::GetDBFI);
   ResetIntegrationRules(&BilinearForm::GetFBFI);
   ResetIntegrationRules(&BilinearForm::GetBBFI);
   ResetIntegrationRules(&BilinearForm::GetBFBFI);
}

LORBase::~LORBase()
{
   delete batched_lor;
   delete a;
   delete fes;
   delete fec;
   delete mesh;
}

LORDiscretization::LORDiscretization(BilinearForm &a_ho_,
                                     const Array<int> &ess_tdof_list,
                                     int ref_type_)
   : LORBase(*a_ho_.FESpace(), ref_type_)
{
   CheckBasisType(fes_ho);
   A.SetType(Operator::MFEM_SPARSEMAT);
   AssembleSystem(a_ho_, ess_tdof_list);
}

LORDiscretization::LORDiscretization(FiniteElementSpace &fes_ho,
                                     int ref_type_) : LORBase(fes_ho, ref_type_)
{
   CheckBasisType(fes_ho);
   A.SetType(Operator::MFEM_SPARSEMAT);
}

void LORDiscretization::FormLORSpace()
{
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
   const int vdim = fes_ho.GetVDim();
   const Ordering::Type ordering = fes_ho.GetOrdering();
   fes = new FiniteElementSpace(mesh, fec, vdim, ordering);
   SetupProlongationAndRestriction();
}

SparseMatrix &LORDiscretization::GetAssembledMatrix() const
{
   MFEM_VERIFY(A.Ptr() != nullptr, "No LOR system assembled");
   return *A.As<SparseMatrix>();
}

#ifdef MFEM_USE_MPI

ParLORDiscretization::ParLORDiscretization(ParBilinearForm &a_ho_,
                                           const Array<int> &ess_tdof_list,
                                           int ref_type_) : LORBase(*a_ho_.ParFESpace(), ref_type_)
{
   ParFiniteElementSpace *pfes_ho = a_ho_.ParFESpace();
   if (pfes_ho->GetMyRank() == 0) { CheckBasisType(fes_ho); }
   A.SetType(Operator::Hypre_ParCSR);
   AssembleSystem(a_ho_, ess_tdof_list);
}

ParLORDiscretization::ParLORDiscretization(
   ParFiniteElementSpace &fes_ho, int ref_type_) : LORBase(fes_ho, ref_type_)
{
   if (fes_ho.GetMyRank() == 0) { CheckBasisType(fes_ho); }
   A.SetType(Operator::Hypre_ParCSR);
}

void ParLORDiscretization::FormLORSpace()
{
   ParFiniteElementSpace &pfes_ho = static_cast<ParFiniteElementSpace&>(fes_ho);
   // TODO: support variable-order spaces in parallel
   MFEM_VERIFY(!pfes_ho.IsVariableOrder(),
               "Cannot construct LOR operators on variable-order spaces");

   int order = pfes_ho.GetMaxElementOrder();
   if (GetFESpaceType() == L2) { ++order; }

   ParMesh &mesh_ho = *pfes_ho.GetParMesh();
   ParMesh *pmesh = new ParMesh(ParMesh::MakeRefined(mesh_ho, order, ref_type));
   mesh = pmesh;

   fec = pfes_ho.FEColl()->Clone(GetLOROrder());
   const int vdim = fes_ho.GetVDim();
   const Ordering::Type ordering = fes_ho.GetOrdering();
   fes = new ParFiniteElementSpace(pmesh, fec, vdim, ordering);
   SetupProlongationAndRestriction();
}

HypreParMatrix &ParLORDiscretization::GetAssembledMatrix() const
{
   MFEM_VERIFY(A.Ptr() != nullptr, "No LOR system assembled");
   return *A.As<HypreParMatrix>();
}

ParFiniteElementSpace &ParLORDiscretization::GetParFESpace() const
{
   return static_cast<ParFiniteElementSpace&>(GetFESpace());
}

#endif // MFEM_USE_MPI

} // namespace mfem
