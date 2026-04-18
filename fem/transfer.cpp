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

#include "transfer.hpp"
#include "bilinearform.hpp"
#include "pbilinearform.hpp"
#include "../general/forall.hpp"

namespace mfem
{

GridTransfer::GridTransfer(FiniteElementSpace &dom_fes_,
                           FiniteElementSpace &ran_fes_)
   : dom_fes(dom_fes_), ran_fes(ran_fes_),
     oper_type(Operator::ANY_TYPE),
     fw_t_oper(), bw_t_oper(), use_ea(false), d_mt(Device::GetHostMemoryType())
{
#ifdef MFEM_USE_MPI
   const bool par_dom = dynamic_cast<ParFiniteElementSpace*>(&dom_fes);
   const bool par_ran = dynamic_cast<ParFiniteElementSpace*>(&ran_fes);
   MFEM_VERIFY(par_dom == par_ran, "the domain and range FE spaces must both"
               " be either serial or parallel");
   parallel = par_dom;
#endif
}

const Operator &GridTransfer::MakeTrueOperator(
   FiniteElementSpace &fes_in, FiniteElementSpace &fes_out,
   const Operator &oper, OperatorHandle &t_oper)
{
   if (t_oper.Ptr())
   {
      return *t_oper.Ptr();
   }

   if (!Parallel())
   {
      const SparseMatrix *in_cP = fes_in.GetConformingProlongation();
      const SparseMatrix *out_cR = fes_out.GetConformingRestriction();
      if (oper_type == Operator::MFEM_SPARSEMAT)
      {
         const SparseMatrix *mat = dynamic_cast<const SparseMatrix *>(&oper);
         MFEM_VERIFY(mat != NULL, "Operator is not a SparseMatrix");
         if (!out_cR)
         {
            t_oper.Reset(const_cast<SparseMatrix*>(mat), false);
         }
         else
         {
            t_oper.Reset(mfem::Mult(*out_cR, *mat));
         }
         if (in_cP)
         {
            t_oper.Reset(mfem::Mult(*t_oper.As<SparseMatrix>(), *in_cP));
         }
      }
      else if (oper_type == Operator::ANY_TYPE)
      {
         const int RP_case = bool(out_cR) + 2*bool(in_cP);
         switch (RP_case)
         {
            case 0:
               t_oper.Reset(const_cast<Operator*>(&oper), false);
               break;
            case 1:
               t_oper.Reset(
                  new ProductOperator(out_cR, &oper, false, false));
               break;
            case 2:
               t_oper.Reset(
                  new ProductOperator(&oper, in_cP, false, false));
               break;
            case 3:
               t_oper.Reset(
                  new TripleProductOperator(
                     out_cR, &oper, in_cP, false, false, false));
               break;
         }
      }
      else
      {
         MFEM_ABORT("Operator::Type is not supported: " << oper_type);
      }
   }
   else // Parallel() == true
   {
#ifdef MFEM_USE_MPI
      if (oper_type == Operator::Hypre_ParCSR)
      {
         const SparseMatrix *out_R = fes_out.GetRestrictionMatrix();
         const ParFiniteElementSpace *pfes_in =
            dynamic_cast<const ParFiniteElementSpace *>(&fes_in);
         const ParFiniteElementSpace *pfes_out =
            dynamic_cast<const ParFiniteElementSpace *>(&fes_out);
         const SparseMatrix *sp_mat = dynamic_cast<const SparseMatrix *>(&oper);
         const HypreParMatrix *hy_mat;
         if (sp_mat)
         {
            SparseMatrix *RA = mfem::Mult(*out_R, *sp_mat);
            t_oper.Reset(pfes_in->Dof_TrueDof_Matrix()->
                         LeftDiagMult(*RA, pfes_out->GetTrueDofOffsets()));
            delete RA;
         }
         else if ((hy_mat = dynamic_cast<const HypreParMatrix *>(&oper)))
         {
            HypreParMatrix *RA =
               hy_mat->LeftDiagMult(*out_R, pfes_out->GetTrueDofOffsets());
            t_oper.Reset(mfem::ParMult(RA, pfes_in->Dof_TrueDof_Matrix()));
            delete RA;
         }
         else
         {
            MFEM_ABORT("unknown Operator type");
         }
      }
      else if (oper_type == Operator::ANY_TYPE)
      {
         const Operator *out_R = fes_out.GetRestrictionOperator();
         t_oper.Reset(new TripleProductOperator(
                         out_R, &oper, fes_in.GetProlongationMatrix(),
                         false, false, false));
      }
      else
      {
         MFEM_ABORT("Operator::Type is not supported: " << oper_type);
      }
#endif
   }

   return *t_oper.Ptr();
}


InterpolationGridTransfer::~InterpolationGridTransfer()
{
   if (own_mass_integ) { delete mass_integ; }
}

void InterpolationGridTransfer::SetMassIntegrator(
   BilinearFormIntegrator *mass_integ_, bool own_mass_integ_)
{
   if (own_mass_integ) { delete mass_integ; }

   mass_integ = mass_integ_;
   own_mass_integ = own_mass_integ_;
}

const Operator &InterpolationGridTransfer::ForwardOperator()
{
   if (F.Ptr())
   {
      return *F.Ptr();
   }

   // Construct F
   if (oper_type == Operator::ANY_TYPE)
   {
      F.Reset(new FiniteElementSpace::RefinementOperator(&ran_fes, &dom_fes));
   }
   else if (oper_type == Operator::MFEM_SPARSEMAT)
   {
      Mesh::GeometryList elem_geoms(*ran_fes.GetMesh());

      DenseTensor localP[Geometry::NumGeom];
      for (int i = 0; i < elem_geoms.Size(); i++)
      {
         ran_fes.GetLocalRefinementMatrices(dom_fes, elem_geoms[i],
                                            localP[elem_geoms[i]]);
      }
      F.Reset(ran_fes.RefinementMatrix_main(
                 dom_fes.GetNDofs(), dom_fes.GetElementToDofTable(),
                 dom_fes.GetElementToFaceOrientationTable(), localP));
   }
   else
   {
      MFEM_ABORT("Operator::Type is not supported: " << oper_type);
   }

   return *F.Ptr();
}

const Operator &InterpolationGridTransfer::BackwardOperator()
{
   if (B.Ptr())
   {
      return *B.Ptr();
   }

   // Construct B, if not set, define a suitable mass_integ
   if (!mass_integ)
   {
      const FiniteElement *f_fe_0 = ran_fes.GetTypicalFE();
      const int map_type = f_fe_0->GetMapType();
      if (map_type == FiniteElement::VALUE ||
          map_type == FiniteElement::INTEGRAL)
      {
         mass_integ = new MassIntegrator;
      }
      else if (map_type == FiniteElement::H_DIV ||
               map_type == FiniteElement::H_CURL)
      {
         mass_integ = new VectorFEMassIntegrator;
      }
      else
      {
         MFEM_ABORT("unknown type of FE space");
      }
      own_mass_integ = true;
   }
   if (oper_type == Operator::ANY_TYPE)
   {
      B.Reset(new FiniteElementSpace::DerefinementOperator(
                 &ran_fes, &dom_fes, mass_integ));
   }
   else
   {
      MFEM_ABORT("Operator::Type is not supported: " << oper_type);
   }

   return *B.Ptr();
}


L2ProjectionGridTransfer::L2Projection::L2Projection(
   const FiniteElementSpace &fes_ho_, const FiniteElementSpace &fes_lor_,
   MemoryType d_mt_)
   : Operator(fes_lor_.GetVSize(), fes_ho_.GetVSize()),
     fes_ho(fes_ho_), fes_lor(fes_lor_), d_mt(d_mt_)
{ }

void L2ProjectionGridTransfer::L2Projection::BuildHo2Lor(
   int nel_ho, int nel_lor, const CoarseFineTransformations& cf_tr)
{
   // Construct the mapping from HO to LOR
   // ho2lor.GetRow(iho) will give all the LOR elements contained in iho
   ho2lor.MakeI(nel_ho);
   for (int ilor = 0; ilor < nel_lor; ++ilor)
   {
      int iho = cf_tr.embeddings[ilor].parent;
      ho2lor.AddAColumnInRow(iho);
   }
   ho2lor.MakeJ();
   for (int ilor = 0; ilor < nel_lor; ++ilor)
   {
      int iho = cf_tr.embeddings[ilor].parent;
      ho2lor.AddConnection(iho, ilor);
   }
   ho2lor.ShiftUpI();
}

void L2ProjectionGridTransfer::L2Projection::ElemMixedMass(
   Geometry::Type geom, const FiniteElement& fe_ho,
   const FiniteElement& fe_lor, ElementTransformation* tr_ho,
   ElementTransformation* tr_lor,
   IntegrationPointTransformation& ip_tr,
   DenseMatrix& M_mixed_el) const
{
   int order = fe_lor.GetOrder() + fe_ho.GetOrder() + tr_lor->OrderW();
   const IntegrationRule* ir = &IntRules.Get(geom, order);
   M_mixed_el = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint& ip_lor = ir->IntPoint(i);
      IntegrationPoint ip_ho;
      ip_tr.Transform(ip_lor, ip_ho);
      Vector shape_lor(fe_lor.GetDof());
      fe_lor.CalcShape(ip_lor, shape_lor);
      Vector shape_ho(fe_ho.GetDof());
      tr_ho->SetIntPoint(&ip_ho);
      fe_ho.CalcPhysShape(*tr_ho, shape_ho);
      tr_lor->SetIntPoint(&ip_lor);
      // For now we use the geometry information from the LOR space, which means
      // we won't be mass conservative if the mesh is curved
      real_t w = ip_lor.weight;
      if (fe_lor.GetMapType() == FiniteElement::VALUE)
      {
         w *= tr_lor->Weight();
      }
      shape_lor *= w;
      AddMultVWt(shape_lor, shape_ho, M_mixed_el);
   }
}

void L2ProjectionGridTransfer::L2Projection::ElemMixedMass(
   Geometry::Type geom, const FiniteElement& fe_ho,
   const FiniteElement& fe_lor, ElementTransformation* el_tr,
   IntegrationPointTransformation& ip_tr,
   DenseMatrix& B_L, DenseMatrix& B_H) const
{
   int order = fe_lor.GetOrder() + fe_ho.GetOrder() + el_tr->OrderW();
   const IntegrationRule* ir = &IntRules.Get(geom, order);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint& ip_lor = ir->IntPoint(i);
      IntegrationPoint ip_ho;

      // maps integration point ip_lor -> ip_ho
      ip_tr.Transform(ip_lor, ip_ho);
      Vector shape_lor(fe_lor.GetDof());
      fe_lor.CalcShape(ip_lor, shape_lor);
      Vector shape_ho(fe_ho.GetDof());
      fe_ho.CalcShape(ip_ho, shape_ho);

      for (int j=0; j<shape_lor.Size(); ++j)
      {
         B_L(i, j) = shape_lor(j);
      }

      for (int j=0; j<shape_ho.Size(); ++j)
      {
         B_H(i, j) = shape_ho(j);
      }
   }

}

void L2ProjectionGridTransfer::L2Projection::MixedMassEA(
   const FiniteElementSpace& fes_ho_ea,
   const FiniteElementSpace& fes_lor_ea,
   Vector &M_LH, MemoryType d_mt_)
{
   Mesh* mesh_ho = fes_ho_ea.GetMesh();
   Mesh* mesh_lor = fes_lor_ea.GetMesh();
   int nel_ho = mesh_ho->GetNE();
   int nel_lor = mesh_lor->GetNE();

   if (nel_ho == 0)
   {
      M_LH.SetSize(0);
      return;
   }

   const CoarseFineTransformations& cf_tr = mesh_lor->GetRefinementTransforms();

   int nref_max = 0;
   Array<Geometry::Type> geoms;
   mesh_ho->GetGeometries(mesh_ho->Dimension(), geoms);
   for (int ig = 0; ig < geoms.Size(); ++ig)
   {
      Geometry::Type geom = geoms[ig];
      nref_max = std::max(nref_max, cf_tr.point_matrices[geom].SizeK());
   }

   BuildHo2Lor(nel_ho, nel_lor, cf_tr);

   IntegrationPointTransformation ip_tr;
   IsoparametricTransformation &emb_tr = ip_tr.Transf;

   // Gather basis functions (B_L, B_HO) and data at quadrature points
   DenseTensor B_L, B_H, D;
   {
      // Assume all HO elements are LOR in the same way
      const int iho = 0;
      {
         Array<int> lor_els;
         ho2lor.GetRow(iho, lor_els);
         int nref = ho2lor.RowSize(iho);

         Geometry::Type geom = mesh_ho->GetElementBaseGeometry(iho);
         const FiniteElement &fe_ho = *fes_ho_ea.GetFE(iho);
         const FiniteElement &fe_lor = *fes_lor_ea.GetFE(lor_els[0]);

         // Allocate space for DenseTensors
         ElementTransformation *el_tr = fes_lor_ea.GetElementTransformation(0);
         int order = fe_lor.GetOrder() + fe_ho.GetOrder() + el_tr->OrderW();
         const IntegrationRule* ir_ea = &IntRules.Get(geom, order);
         int qPts = ir_ea->GetNPoints();

         // Containers for the basis functions sampled at quadrature points
         B_L.SetSize(qPts, fe_lor.GetDof(), nref, d_mt);
         B_H.SetSize(qPts, fe_ho.GetDof(), nref, d_mt);
         D.SetSize(qPts, nref, nel_ho, d_mt);

         const GeometricFactors *geo_facts =
            mesh_lor->GetGeometricFactors(*ir_ea, GeometricFactors::DETERMINANTS);

         MFEM_ASSERT(nel_ho*nref == nel_lor, "we expect nel_ho*nref == nel_lor");

         // Setup data at quadrature points
         // TODO add support for user coefficient
         const auto W = Reshape(ir_ea->GetWeights().Read(), qPts);
         const auto J = Reshape(geo_facts->detJ.Read(), qPts, nel_lor);
         const auto d_D = Reshape(D.Write(), qPts, nref, nel_ho);

         mfem::forall(qPts * nref * nel_ho, [=] MFEM_HOST_DEVICE (int tid)
         {
            const int q    = tid % qPts;
            const int iref = (tid / qPts) % nref;
            const int iho  = (tid / (qPts * nref)) % nel_ho;

            const int lo_el_id = iref + nref*iho;
            const real_t detJ = J(q, lo_el_id);

            d_D(q, iref, iho) = W(q) * detJ;

         });

         emb_tr.SetIdentityTransformation(geom);
         const DenseTensor &pmats = cf_tr.point_matrices[geom];

         // Collect the basis functions
         for (int iref = 0; iref < nref; ++iref)
         {
            int ilor = lor_els[iref];
            // Now assemble the block-row of the mixed mass matrix associated
            // with integrating HO functions against LOR functions on the LOR
            // sub-element.

            // Create the transformation that embeds the fine low-order element
            // within the coarse high-order element in reference space
            emb_tr.SetPointMat(pmats(cf_tr.embeddings[ilor].matrix));

            DenseMatrix &b_lo = B_L(ilor);
            DenseMatrix &b_ho = B_H(ilor);

            ElemMixedMass(geom, fe_ho, fe_lor, el_tr, ip_tr, b_lo, b_ho);

         } // loop over subcells of ho element
         // end of quadrature point setup
      }

   } // completed setup of basis function and quadrature point

   // Assemble mixed mass matrix
   {
      int iho = 0;
      Array<int> lor_els;
      ho2lor.GetRow(iho, lor_els);
      int nref = ho2lor.RowSize(iho);

      const FiniteElement &fe_ho = *fes_ho_ea.GetFE(iho);
      const FiniteElement &fe_lor = *fes_lor_ea.GetFE(lor_els[0]);
      const int ndof_ho = fe_ho.GetDof();
      const int ndof_lor = fe_lor.GetDof();

      const int qPts = D.SizeI();

      M_LH.SetSize(ndof_lor*ndof_ho*nref*nel_ho, d_mt);

      // Rows x columns
      // Recall MFEM is column major
      // rows x columns is inverted  - matrix is ndof_lor x ndof_ho
      auto v_M_LH = Reshape(M_LH.Write(), ndof_lor, ndof_ho, nref,
                            nel_ho);

      const int fe_ho_ndof = fe_ho.GetDof();
      const int fe_lor_ndof = fe_lor.GetDof();

      auto d_B_L = Reshape(B_L.Read(), qPts, fe_lor_ndof, nref);
      auto d_B_H = Reshape(B_H.Read(), qPts, fe_ho_ndof, nref);
      auto d_D   = Reshape(D.Read(), qPts, nref, nel_ho);

      mfem::forall(fe_ho_ndof*nref*nel_ho, [=] MFEM_HOST_DEVICE (int idx)
      {
         const int bh = idx % fe_ho_ndof;
         const int iref = (idx / fe_ho_ndof) % nref;
         const int iho = idx / fe_ho_ndof / nref;
         // (B_lo_dofs x Q) x (Q x B_ho_dofs)
         for (int bl = 0; bl < fe_lor_ndof; ++bl)
         {
            real_t dot = 0.0;
            for (int qi=0; qi<qPts; ++qi)
            {
               dot += d_B_L(qi, bl, iref) *  d_D(qi, iref, iho) * d_B_H(qi, bh, iref);
            }
            // column major storage
            v_M_LH(bl, bh, iref, iho) = dot;
         }
      });
   } // end of mixed assembly mass matrix
}

L2ProjectionGridTransfer::L2ProjectionL2Space::L2ProjectionL2Space
(const FiniteElementSpace &fes_ho_, const FiniteElementSpace &fes_lor_,
 const bool use_ea_, MemoryType d_mt_)
   : L2Projection(fes_ho_, fes_lor_, d_mt_),
     use_ea(use_ea_)
{
   if (use_ea)
   {
      EAL2ProjectionL2Space();
      return;
   }

   Mesh *mesh_ho = fes_ho.GetMesh();
   Mesh *mesh_lor = fes_lor.GetMesh();
   int nel_ho = mesh_ho->GetNE();
   int nel_lor = mesh_lor->GetNE();

   // The prolongation operation is only well-defined when the LOR space has at
   // least as many DOFs as the high-order space.
   const bool build_P = fes_lor.GetTrueVSize() >= fes_ho.GetTrueVSize();

   // If the local mesh is empty, skip all computations
   if (nel_ho == 0) { return; }

   const CoarseFineTransformations &cf_tr = mesh_lor->GetRefinementTransforms();

   int nref_max = 0;
   Array<Geometry::Type> geoms;
   mesh_ho->GetGeometries(mesh_ho->Dimension(), geoms);
   for (int ig = 0; ig < geoms.Size(); ++ig)
   {
      Geometry::Type geom = geoms[ig];
      nref_max = std::max(nref_max, cf_tr.point_matrices[geom].SizeK());
   }

   BuildHo2Lor(nel_ho, nel_lor, cf_tr);

   offsets.SetSize(nel_ho+1);
   offsets[0] = 0;
   for (int iho = 0; iho < nel_ho; ++iho)
   {
      int nref = ho2lor.RowSize(iho);
      const FiniteElement &fe_ho = *fes_ho.GetFE(iho);
      const FiniteElement &fe_lor = *fes_lor.GetFE(ho2lor.GetRow(iho)[0]);
      offsets[iho+1] = offsets[iho] + fe_ho.GetDof()*fe_lor.GetDof()*nref;
   }
   // R will contain the restriction (L^2 projection operator) defined on each
   // coarse HO element (and corresponding patch of LOR elements)
   R.SetSize(offsets[nel_ho]);
   if (build_P)
   {
      // P will contain the corresponding prolongation operator
      P.SetSize(offsets[nel_ho]);
   }

   IntegrationPointTransformation ip_tr;
   IsoparametricTransformation &emb_tr = ip_tr.Transf;

   for (int iho = 0; iho < nel_ho; ++iho)
   {
      Array<int> lor_els;
      ho2lor.GetRow(iho, lor_els);
      int nref = ho2lor.RowSize(iho);

      Geometry::Type geom = mesh_ho->GetElementBaseGeometry(iho);
      const FiniteElement &fe_ho = *fes_ho.GetFE(iho);
      const FiniteElement &fe_lor = *fes_lor.GetFE(lor_els[0]);
      int ndof_ho = fe_ho.GetDof();
      int ndof_lor = fe_lor.GetDof();

      ElementTransformation *tr_ho = fes_ho.GetElementTransformation(iho);

      emb_tr.SetIdentityTransformation(geom);
      const DenseTensor &pmats = cf_tr.point_matrices[geom];

      DenseMatrix R_iho(&R[offsets[iho]], ndof_lor*nref, ndof_ho);

      DenseMatrix Minv_lor(ndof_lor*nref, ndof_lor*nref);
      DenseMatrix M_mixed(ndof_lor*nref, ndof_ho);

      MassIntegrator mi;
      DenseMatrix M_lor_el(ndof_lor, ndof_lor);
      DenseMatrixInverse Minv_lor_el(&M_lor_el);
      DenseMatrix M_lor(ndof_lor*nref, ndof_lor*nref);
      DenseMatrix M_mixed_el(ndof_lor, ndof_ho);

      Minv_lor = 0.0;
      M_lor = 0.0;

      DenseMatrix RtMlor(ndof_ho, ndof_lor*nref);
      DenseMatrix RtMlorR(ndof_ho, ndof_ho);
      DenseMatrixInverse RtMlorR_inv(&RtMlorR);

      for (int iref = 0; iref < nref; ++iref)
      {
         // Assemble the low-order refined mass matrix and invert locally
         int ilor = lor_els[iref];
         ElementTransformation *tr_lor = fes_lor.GetElementTransformation(ilor);
         mi.AssembleElementMatrix(fe_lor, *tr_lor, M_lor_el);
         M_lor.CopyMN(M_lor_el, iref*ndof_lor, iref*ndof_lor);
         Minv_lor_el.Factor();
         Minv_lor_el.GetInverseMatrix(M_lor_el);
         // Insert into the diagonal of the patch LOR mass matrix
         Minv_lor.CopyMN(M_lor_el, iref*ndof_lor, iref*ndof_lor);

         // Now assemble the block-row of the mixed mass matrix associated
         // with integrating HO functions against LOR functions on the LOR
         // sub-element.

         // Create the transformation that embeds the fine low-order element
         // within the coarse high-order element in reference space
         emb_tr.SetPointMat(pmats(cf_tr.embeddings[ilor].matrix));

         ElemMixedMass(geom, fe_ho, fe_lor, tr_ho, tr_lor, ip_tr, M_mixed_el);

         M_mixed.CopyMN(M_mixed_el, iref*ndof_lor, 0);
      }
      mfem::Mult(Minv_lor, M_mixed, R_iho);

      if (build_P)
      {
         DenseMatrix P_iho(&P[offsets[iho]], ndof_ho, ndof_lor*nref);

         mfem::MultAtB(R_iho, M_lor, RtMlor);
         mfem::Mult(RtMlor, R_iho, RtMlorR);
         RtMlorR_inv.Factor();
         RtMlorR_inv.Mult(RtMlor, P_iho);
      }
   }

}


void L2ProjectionGridTransfer::L2ProjectionL2Space::EAL2ProjectionL2Space()
{
   Mesh *mesh_ho = fes_ho.GetMesh();
   Mesh *mesh_lor = fes_lor.GetMesh();
   int nel_ho = mesh_ho->GetNE();
   int nel_lor = mesh_lor->GetNE();

   // The prolongation operation is only well-defined when the LOR space has at
   // least as many DOFs as the high-order space.
   const bool build_P = fes_lor.GetTrueVSize() >= fes_ho.GetTrueVSize();

   // If the local mesh is empty, skip all computations
   if (nel_ho == 0) { return; }

   const CoarseFineTransformations &cf_tr = mesh_lor->GetRefinementTransforms();

   int nref_max = 0;
   Array<Geometry::Type> geoms;
   mesh_ho->GetGeometries(mesh_ho->Dimension(), geoms);
   for (int ig = 0; ig < geoms.Size(); ++ig)
   {
      Geometry::Type geom = geoms[ig];
      nref_max = std::max(nref_max, cf_tr.point_matrices[geom].SizeK());
   }

   BuildHo2Lor(nel_ho, nel_lor, cf_tr);

   offsets.SetSize(nel_ho+1);
   offsets[0] = 0;
   for (int iho = 0; iho < nel_ho; ++iho)
   {
      int nref = ho2lor.RowSize(iho);
      const FiniteElement &fe_ho = *fes_ho.GetFE(iho);
      const FiniteElement &fe_lor = *fes_lor.GetFE(ho2lor.GetRow(iho)[0]);
      offsets[iho+1] = offsets[iho] + fe_ho.GetDof()*fe_lor.GetDof()*nref;
   }

   // R will contain the restriction (L^2 projection operator) defined on each
   // coarse HO element (and corresponding patch of LOR elements)
   R.SetSize(offsets[nel_ho]);

   if (build_P)
   {
      // P will contain the corresponding prolongation operator
      P.SetSize(offsets[nel_ho]);
   }

   // Assemble mixed mass matrix
   Vector M_mixed_all;
   MixedMassEA(fes_ho, fes_lor, M_mixed_all, d_mt);


   // R = inv(M_L) * M_mixed
   // Need to compute M_L
   // Note: Using user-inputted M_LH IntegrationRule ir
   // (higher order than needed) in order to re-use coeff
   MassIntegrator mi;

   Vector M_ea_lor;
   int ndof_lor;
   int ndof_ho;
   int nref;
   {
      int iho = 0;
      Array<int> lor_els;
      ho2lor.GetRow(iho, lor_els);
      nref = ho2lor.RowSize(iho);

      const FiniteElement &fe_ho = *fes_ho.GetFE(iho);
      const FiniteElement &fe_lor = *fes_lor.GetFE(lor_els[0]);
      ndof_ho = fe_ho.GetDof();
      ndof_lor = fe_lor.GetDof();

      M_ea_lor.SetSize(ndof_lor*ndof_lor*nel_lor, d_mt);
   }

   const bool add = false;
   mi.AssembleEA(fes_lor, M_ea_lor, add);

   DenseTensor Minv_ear_lor;
   Minv_ear_lor.SetSize(ndof_lor, ndof_lor, nel_lor, d_mt);
   Minv_ear_lor.GetMemory().CopyFrom(M_ea_lor.GetMemory(), M_ea_lor.Size());

   BatchedLinAlg::Invert(Minv_ear_lor);
   {
      // Recall mfem is column major
      // ndof_lor x ndof_ho
      auto v_M_mixed_all = Reshape(M_mixed_all.Read(), ndof_lor, ndof_ho, nref,
                                   nel_ho);

      // matrix is symmetric
      auto v_Minv_ear_lor = Reshape(Minv_ear_lor.Read(), ndof_lor, ndof_lor,
                                    nel_lor);

      // ndof_lor x ndof_ho
      auto v_R = Reshape(R.Write(), ndof_lor, nref, ndof_ho, nel_ho);

      MFEM_VERIFY(nel_lor==nel_ho*nref, "nel_lor != nel_ho*nref");

      // (ndofs_lor x ndofs_lor) x (ndofs_lor x ndof_ho)
      mfem::forall(ndof_lor * nref * ndof_ho * nel_ho, [=] MFEM_HOST_DEVICE (int tid)
      {

         const int i = tid % ndof_lor;
         const int iref = (tid / ndof_lor) % nref;
         const int j    = (tid / (ndof_lor * nref) ) % ndof_ho;
         const int iho  = (tid / (ndof_lor * nref * ndof_ho)) % nel_ho;

         const int lor_idx = iref + iho * nref;

         //matrices are stored in the transpose position
         real_t dot = 0.0;
         for (int k=0; k<ndof_lor; ++k)
         {
            dot += v_Minv_ear_lor(i, k, lor_idx) * v_M_mixed_all(k, j, iref, iho);
         }
         v_R(i, iref, j, iho) = dot;

      });
   }

   if (build_P)
   {
      // P = inv(R^T M_L R) * R^T M_L

      // M_lor is size of ndof_lor x ndof_lor
      // R is size of (ndof_lor x nref x ndof_ho)
      auto v_M_ea_lor = Reshape(M_ea_lor.Read(), ndof_lor, ndof_lor, nel_lor);
      auto v_R = Reshape(R.Read(), ndof_lor, nref, ndof_ho, nel_ho);
      // R^T M_LO is of size nref x ndof_lor

      // Compute R^T M_L
      Vector RtM_L(ndof_ho*nref*ndof_lor*nel_ho, d_mt);
      auto v_RtM_L = Reshape(RtM_L.Write(), ndof_ho, ndof_lor, nref, nel_ho);

      mfem::forall(ndof_lor * nref * ndof_ho * nel_ho, [=] MFEM_HOST_DEVICE (int tid)
      {

         const int jlo = tid % ndof_lor;
         const int iref = (tid / ndof_lor) % nref;
         const int iho  = (tid / (ndof_lor * nref)) % ndof_ho;
         const int e    = (tid / (ndof_lor * nref * ndof_ho)) % nel_ho;

         const int lor_idx = iref + e * nref;

         real_t dot = 0.0;
         for (int t=0; t<ndof_lor; ++t)
         {
            dot += v_R(t, iref, iho, e) * v_M_ea_lor(t, jlo, lor_idx);
         }

         v_RtM_L(iho, jlo, iref, e) = dot;

      });

      // Resulting matrix should be: ndof_ho x ndof_ho
      // R^T M_L x R
      DenseTensor RtM_L_dt;
      RtM_L_dt.NewMemoryAndSize(RtM_L.GetMemory(), ndof_ho, ndof_lor*nref,
                                nel_ho, false);
      Vector R_vec;
      R_vec.NewMemoryAndSize(R.GetMemory(), R.Size(), false);
      Vector RtM_LR(ndof_ho * ndof_ho * nel_ho, d_mt);
      BatchedLinAlg::Mult(RtM_L_dt, R_vec, RtM_LR);
      // Ensure that changes to the alias R_vec are propagated to the base, R
      R_vec.GetMemory().SyncAlias(R.GetMemory(), P.Size());

      // Compute the inverse of InvRtM_LR
      DenseTensor InvRtM_LR;
      InvRtM_LR.NewMemoryAndSize(RtM_LR.GetMemory(), ndof_ho, ndof_ho, nel_ho, false);
      BatchedLinAlg::Invert(InvRtM_LR);

      // Form P
      // P should be of dimension (ndof_ho x ndof_ho) x (ndof_ho x nref*ndof_lor)
      // P ndof_ho x nref*ndof_lor
      Vector P_vec;
      P_vec.NewMemoryAndSize(P.GetMemory(), P.Size(), false);
      BatchedLinAlg::Mult(InvRtM_LR, RtM_L, P_vec);
      // Ensure that changes to the alias P_vec are propagated to the base, P
      P_vec.GetMemory().SyncAlias(P.GetMemory(), P.Size());
   }
}

void L2ProjectionGridTransfer::L2ProjectionL2Space::Mult(
   const Vector &x, Vector &y) const
{

   if (use_ea)
   {
      return EAMult(x,y);
   }


   int vdim = fes_ho.GetVDim();
   Array<int> vdofs;
   DenseMatrix xel_mat, yel_mat;
   for (int iho = 0; iho < fes_ho.GetNE(); ++iho)
   {
      int nref = ho2lor.RowSize(iho);
      int ndof_ho = fes_ho.GetFE(iho)->GetDof();
      int ndof_lor = fes_lor.GetFE(ho2lor.GetRow(iho)[0])->GetDof();
      xel_mat.SetSize(ndof_ho, vdim);
      yel_mat.SetSize(ndof_lor*nref, vdim);
      DenseMatrix R_iho(&R[offsets[iho]], ndof_lor*nref, ndof_ho);

      fes_ho.GetElementVDofs(iho, vdofs);
      x.GetSubVector(vdofs, xel_mat.GetData());
      mfem::Mult(R_iho, xel_mat, yel_mat);
      // Place result correctly into the low-order vector
      for (int iref = 0; iref < nref; ++iref)
      {
         int ilor = ho2lor.GetRow(iho)[iref];
         for (int vd=0; vd<vdim; ++vd)
         {
            fes_lor.GetElementDofs(ilor, vdofs);
            fes_lor.DofsToVDofs(vd, vdofs);
            y.SetSubVector(vdofs, &yel_mat(iref*ndof_lor,vd));
         }
      }
   }
}

void L2ProjectionGridTransfer::L2ProjectionL2Space::EAMult(
   const Vector &x, Vector &y) const
{
   const int nel_ho = fes_ho.GetMesh()->GetNE();

   if (nel_ho == 0)
   {
      return;
   }

   const int iho = 0;
   const int nref = ho2lor.RowSize(iho);
   const int ndof_ho = fes_ho.GetFE(iho)->GetDof();
   const int ndof_lor = fes_lor.GetFE(ho2lor.GetRow(iho)[0])->GetDof();

   DenseTensor R_dt;
   R_dt.NewMemoryAndSize(R.GetMemory(), ndof_lor*nref, ndof_ho, nel_ho, false);
   BatchedLinAlg::Mult(R_dt, x, y);
}

void L2ProjectionGridTransfer::L2ProjectionL2Space::MultTranspose(
   const Vector &x, Vector &y) const
{

   if (use_ea)
   {
      return EAMultTranspose(x,y);
   }

   int vdim = fes_ho.GetVDim();
   Array<int> vdofs;
   DenseMatrix xel_mat, yel_mat;
   y = 0.0;
   for (int iho = 0; iho < fes_ho.GetNE(); ++iho)
   {
      int nref = ho2lor.RowSize(iho);
      int ndof_ho = fes_ho.GetFE(iho)->GetDof();
      int ndof_lor = fes_lor.GetFE(ho2lor.GetRow(iho)[0])->GetDof();
      xel_mat.SetSize(ndof_lor*nref, vdim);
      yel_mat.SetSize(ndof_ho, vdim);
      DenseMatrix R_iho(&R[offsets[iho]], ndof_lor*nref, ndof_ho);

      // Extract the LOR DOFs
      for (int iref=0; iref<nref; ++iref)
      {
         int ilor = ho2lor.GetRow(iho)[iref];
         for (int vd=0; vd<vdim; ++vd)
         {
            fes_lor.GetElementDofs(ilor, vdofs);
            fes_lor.DofsToVDofs(vd, vdofs);
            x.GetSubVector(vdofs, &xel_mat(iref*ndof_lor, vd));
         }
      }
      // Multiply locally by the transpose
      mfem::MultAtB(R_iho, xel_mat, yel_mat);
      // Place the result in the HO vector
      fes_ho.GetElementVDofs(iho, vdofs);
      y.AddElementVector(vdofs, yel_mat.GetData());
   }

}

void L2ProjectionGridTransfer::L2ProjectionL2Space::EAMultTranspose(
   const Vector &x, Vector &y) const
{
   const int nel_ho = fes_ho.GetMesh()->GetNE();

   if (nel_ho == 0)
   {
      return;
   }

   const int iho = 0;
   const int nref = ho2lor.RowSize(iho);
   const int ndof_ho = fes_ho.GetFE(iho)->GetDof();
   const int ndof_lor = fes_lor.GetFE(ho2lor.GetRow(iho)[0])->GetDof();

   DenseTensor R_dt;
   R_dt.NewMemoryAndSize(R.GetMemory(), ndof_lor*nref, ndof_ho, nel_ho, false);
   BatchedLinAlg::MultTranspose(R_dt, x, y);
}

void L2ProjectionGridTransfer::L2ProjectionL2Space::Prolongate(
   const Vector &x, Vector &y) const
{
   if (fes_ho.GetNE() == 0) { return; }

   if (use_ea)
   {
      return EAProlongate(x,y);
   }

   MFEM_VERIFY(P.Size() > 0, "Prolongation not supported for these spaces.")
   int vdim = fes_ho.GetVDim();
   Array<int> vdofs;
   DenseMatrix xel_mat,yel_mat;
   y = 0.0;
   for (int iho = 0; iho < fes_ho.GetNE(); ++iho)
   {
      int nref = ho2lor.RowSize(iho);
      int ndof_ho = fes_ho.GetFE(iho)->GetDof();
      int ndof_lor = fes_lor.GetFE(ho2lor.GetRow(iho)[0])->GetDof();
      xel_mat.SetSize(ndof_lor*nref, vdim);
      yel_mat.SetSize(ndof_ho, vdim);
      DenseMatrix P_iho(&P[offsets[iho]], ndof_ho, ndof_lor*nref);

      // Extract the LOR DOFs
      for (int iref = 0; iref < nref; ++iref)
      {
         int ilor = ho2lor.GetRow(iho)[iref];
         for (int vd = 0; vd < vdim; ++vd)
         {
            fes_lor.GetElementDofs(ilor, vdofs);
            fes_lor.DofsToVDofs(vd, vdofs);
            x.GetSubVector(vdofs, &xel_mat(iref*ndof_lor, vd));
         }
      }
      // Locally prolongate
      mfem::Mult(P_iho, xel_mat, yel_mat);
      // Place the result in the HO vector
      fes_ho.GetElementVDofs(iho, vdofs);
      y.AddElementVector(vdofs, yel_mat.GetData());
   }

}

void L2ProjectionGridTransfer::L2ProjectionL2Space::EAProlongate(
   const Vector &x, Vector &y) const
{
   const int iho = 0;
   const int nref = ho2lor.RowSize(iho);
   const int ndof_ho = fes_ho.GetFE(iho)->GetDof();
   const int ndof_lor = fes_lor.GetFE(ho2lor.GetRow(iho)[0])->GetDof();
   const int nel_ho = fes_ho.GetMesh()->GetNE();

   DenseTensor P_dt;
   P_dt.NewMemoryAndSize(P.GetMemory(), ndof_ho, ndof_lor * nref, nel_ho, false);
   BatchedLinAlg::Mult(P_dt, x, y);
}

void L2ProjectionGridTransfer::L2ProjectionL2Space::ProlongateTranspose(
   const Vector &x, Vector &y) const
{
   if (fes_ho.GetNE() == 0) { return; }

   if (use_ea)
   {
      return EAProlongateTranspose(x,y);
   }

   MFEM_VERIFY(P.Size() > 0, "Prolongation not supported for these spaces.")
   int vdim = fes_ho.GetVDim();
   Array<int> vdofs;
   DenseMatrix xel_mat,yel_mat;
   for (int iho = 0; iho < fes_ho.GetNE(); ++iho)
   {
      int nref = ho2lor.RowSize(iho);
      int ndof_ho = fes_ho.GetFE(iho)->GetDof();
      int ndof_lor = fes_lor.GetFE(ho2lor.GetRow(iho)[0])->GetDof();
      xel_mat.SetSize(ndof_ho, vdim);
      yel_mat.SetSize(ndof_lor*nref, vdim);
      DenseMatrix P_iho(&P[offsets[iho]], ndof_ho, ndof_lor*nref);

      fes_ho.GetElementVDofs(iho, vdofs);
      x.GetSubVector(vdofs, xel_mat.GetData());
      mfem::MultAtB(P_iho, xel_mat, yel_mat);

      // Place result correctly into the low-order vector
      for (int iref = 0; iref < nref; ++iref)
      {
         int ilor = ho2lor.GetRow(iho)[iref];
         for (int vd=0; vd<vdim; ++vd)
         {
            fes_lor.GetElementDofs(ilor, vdofs);
            fes_lor.DofsToVDofs(vd, vdofs);
            y.SetSubVector(vdofs, &yel_mat(iref*ndof_lor,vd));
         }
      }
   }

}

void L2ProjectionGridTransfer::L2ProjectionL2Space::EAProlongateTranspose(
   const Vector &x, Vector &y) const
{
   const int iho = 0;
   const int nref = ho2lor.RowSize(iho);
   const int ndof_ho = fes_ho.GetFE(iho)->GetDof();
   const int ndof_lor = fes_lor.GetFE(ho2lor.GetRow(iho)[0])->GetDof();
   const int nel_ho = fes_ho.GetMesh()->GetNE();

   DenseTensor P_dt;
   P_dt.NewMemoryAndSize(P.GetMemory(), ndof_ho, ndof_lor * nref, nel_ho, false);
   BatchedLinAlg::MultTranspose(P_dt, x, y);
}

L2ProjectionGridTransfer::L2ProjectionH1Space::L2ProjectionH1Space(
   const FiniteElementSpace& fes_ho_, const FiniteElementSpace& fes_lor_,
   const bool use_ea_, MemoryType d_mt_)
   : L2Projection(fes_ho_, fes_lor_, d_mt_),
     use_ea(use_ea_)
{

   // need scalar to keep dimensions matching (operators are built to apply
   // individually on each vdim)
   // needed in both matrix and element based versions
   fes_ho_scalar.reset(new FiniteElementSpace(fes_ho.GetMesh(),
                                              fes_ho.FEColl(), 1));
   fes_lor_scalar.reset(new FiniteElementSpace(fes_lor.GetMesh(),
                                               fes_lor.FEColl(), 1));

   if (use_ea)
   {
      EAL2ProjectionH1Space();
      return;
   }

   std::unique_ptr<SparseMatrix> R_mat, M_LH_mat;

   std::tie(R_mat, M_LH_mat) = ComputeSparseRAndM_LH();

   const SparseMatrix *P_ho = fes_ho_scalar->GetConformingProlongation();
   const SparseMatrix *P_lor = fes_lor_scalar->GetConformingProlongation();

   if (P_ho || P_lor)
   {
      if (P_ho && P_lor)
      {
         R_mat.reset(RAP(*P_lor, *R_mat, *P_ho));
         M_LH_mat.reset(RAP(*P_lor, *M_LH_mat, *P_ho));
      }
      else if (P_ho)
      {
         R_mat.reset(mfem::Mult(*R_mat, *P_ho));
         M_LH_mat.reset(mfem::Mult(*M_LH_mat, *P_ho));
      }
      else // P_lor != nullptr
      {
         R_mat.reset(mfem::Mult(*P_lor, *R_mat));
         M_LH_mat.reset(mfem::Mult(*P_lor, *M_LH_mat));
      }
   }

   SparseMatrix *RTxM_LH_mat = TransposeMult(*R_mat, *M_LH_mat);
   precon.reset(new DSmoother(*RTxM_LH_mat));

   // Set ownership
   RTxM_LH.reset(RTxM_LH_mat);
   R = std::move(R_mat);
   M_LH = std::move(M_LH_mat);

   SetupPCG();
}

#ifdef MFEM_USE_MPI

L2ProjectionGridTransfer::L2ProjectionH1Space::L2ProjectionH1Space(
   const ParFiniteElementSpace& pfes_ho, const ParFiniteElementSpace& pfes_lor,
   const bool use_ea_, MemoryType d_mt_)
   : L2Projection(pfes_ho, pfes_lor, d_mt_),
     use_ea(use_ea_), pcg(pfes_ho.GetComm())
{

   // need scalar to keep dimensions matching (operators are built to apply
   // individually on each vdim)
   // needed in both matrix and element based versions
   pfes_ho_scalar.reset(new ParFiniteElementSpace(pfes_ho.GetParMesh(),
                                                  pfes_ho.FEColl(), 1));
   pfes_lor_scalar.reset(new ParFiniteElementSpace(pfes_lor.GetParMesh(),
                                                   pfes_lor.FEColl(), 1));

   if (use_ea)
   {
      EAL2ProjectionH1Space(pfes_ho, pfes_lor);
      return;
   }

   std::tie(R, M_LH) = ComputeSparseRAndM_LH();


   HypreParMatrix R_local = HypreParMatrix(pfes_ho.GetComm(),
                                           pfes_lor_scalar->GlobalVSize(),
                                           pfes_ho_scalar->GlobalVSize(),
                                           pfes_lor_scalar->GetDofOffsets(),
                                           pfes_ho_scalar->GetDofOffsets(),
                                           static_cast<SparseMatrix*>(R.get()));
   HypreParMatrix M_LH_local = HypreParMatrix(pfes_ho.GetComm(),
                                              pfes_lor_scalar->GlobalVSize(),
                                              pfes_ho_scalar->GlobalVSize(),
                                              pfes_lor_scalar->GetDofOffsets(),
                                              pfes_ho_scalar->GetDofOffsets(),
                                              static_cast<SparseMatrix*>(M_LH.get()));

   HypreParMatrix *R_mat = RAP(pfes_lor_scalar->Dof_TrueDof_Matrix(),
                               &R_local, pfes_ho_scalar->Dof_TrueDof_Matrix());
   HypreParMatrix *M_LH_mat = RAP(pfes_lor_scalar->Dof_TrueDof_Matrix(),
                                  &M_LH_local, pfes_ho_scalar->Dof_TrueDof_Matrix());

   std::unique_ptr<HypreParMatrix> R_T(R_mat->Transpose());
   HypreParMatrix *RTxM_LH_mat = ParMult(R_T.get(), M_LH_mat, true);

   HypreBoomerAMG *amg = new HypreBoomerAMG(*RTxM_LH_mat);
   amg->SetPrintLevel(0);

   R.reset(R_mat);
   M_LH.reset(M_LH_mat);
   RTxM_LH.reset(RTxM_LH_mat);
   precon.reset(amg);

   SetupPCG();
   pcg.SetPreconditioner(*precon);
   pcg.SetOperator(*RTxM_LH);
}

#endif

void L2ProjectionGridTransfer::L2ProjectionH1Space::SetupPCG()
{
   // Basic PCG solver setup
   pcg.SetPrintLevel(0);
   // pcg.SetPrintLevel(IterativeSolver::PrintLevel().Summary());
   pcg.SetMaxIter(1000);
   // initial values for relative and absolute tolerance
   pcg.SetRelTol(1e-13);
   pcg.SetAbsTol(1e-13);
   pcg.SetPreconditioner(*precon);
   pcg.SetOperator(*RTxM_LH);
}

void L2ProjectionGridTransfer::L2ProjectionH1Space::EAL2ProjectionH1Space()
{
   Mesh* mesh_ho = fes_ho.GetMesh();
   Mesh* mesh_lor = fes_lor.GetMesh();
   int nel_ho = mesh_ho->GetNE();
   int nel_lor = mesh_lor->GetNE();
   int ndof_ho = fes_ho.GetNDofs();
   int ndof_lor = fes_lor.GetNDofs();

   // If the local mesh is empty, skip all computations
   if (nel_ho == 0)
   {
      return;
   }

   const CoarseFineTransformations& cf_tr = mesh_lor->GetRefinementTransforms();

   int nref_max = 0;
   Array<Geometry::Type> geoms;
   mesh_ho->GetGeometries(mesh_ho->Dimension(), geoms);
   for (int ig = 0; ig < geoms.Size(); ++ig)
   {
      Geometry::Type geom = geoms[ig];
      nref_max = std::max(nref_max, cf_tr.point_matrices[geom].SizeK());
   }

   BuildHo2Lor(nel_ho, nel_lor, cf_tr);

   // lumped M_H and inv lumped M_L

   // M_H contains the lumped (row sum) high order mass matrix. This is built for
   // preconditioning the inverse needed to build the prolongation operator P
   Vector M_H(ndof_ho);
   M_H = 0.0;
   // ML_inv_ea contains the inverse lumped (row sum) mass matrix. Note that the
   // method will also work with a full (consistent) mass matrix, though this is
   // not implemented here. L refers to the low-order refined mesh
   ML_inv_ea.SetSize(ndof_lor);
   ML_inv_ea = 0.0;

   BilinearForm Mho(fes_ho_scalar.get());
   Mho.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   Mho.AddDomainIntegrator(new MassIntegrator);
   Mho.Assemble();

   // Processor local lumped Mass
   Vector ones_ho(Mho.Width()); ones_ho = 1.0;
   M_H = 0.0;
   Mho.Mult(ones_ho, M_H);

   BilinearForm Mlor(fes_lor_scalar.get());
   Mlor.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   Mlor.AddDomainIntegrator(new MassIntegrator);
   Mlor.Assemble();

   Vector ones_lor(Mlor.Width()); ones_lor = 1.0;
   Mlor.Mult(ones_lor, ML_inv_ea);

   // DOF by DOF inverse of non-zero entries
   LumpedMassInverse(ML_inv_ea);

   // mixed mass M_LH
   MixedMassEA(fes_ho, fes_lor, M_LH_ea, d_mt);

   // Set ownership
   M_LH_local_op = new H1SpaceMixedMassOperator(fes_ho_scalar.get(),
                                                fes_lor_scalar.get(),
                                                &ho2lor,
                                                &M_LH_ea);

   ML_inv_vea.reset(new H1SpaceLumpedMassOperator(fes_ho_scalar.get(),
                                                  fes_lor_scalar.get(),
                                                  ML_inv_ea));
   M_LH.reset(M_LH_local_op);
   R.reset(new ProductOperator(ML_inv_vea.get(), M_LH.get(), false,
                               false));

   Array<int> ess_tdof_list;  // leave empty
   precon.reset(new OperatorJacobiSmoother(M_H, ess_tdof_list));

   TransposeOperator* RT = new TransposeOperator(R.get());
   RTxM_LH.reset(new ProductOperator(RT, M_LH.get(), true, false));

   SetupPCG();
}

#ifdef MFEM_USE_MPI
void L2ProjectionGridTransfer::L2ProjectionH1Space::EAL2ProjectionH1Space
(const ParFiniteElementSpace& pfes_ho, const ParFiniteElementSpace& pfes_lor)
{
   Mesh* mesh_ho = pfes_ho.GetParMesh();
   Mesh* mesh_lor = pfes_lor.GetParMesh();
   int nel_ho = mesh_ho->GetNE();
   int nel_lor = mesh_lor->GetNE();
   int ndof_ho = pfes_ho.GetNDofs();
   int ndof_lor = pfes_lor.GetNDofs();

   const CoarseFineTransformations& cf_tr = mesh_lor->GetRefinementTransforms();

   int nref_max = 0;
   Array<Geometry::Type> geoms;
   mesh_ho->GetGeometries(mesh_ho->Dimension(), geoms);
   for (int ig = 0; ig < geoms.Size(); ++ig)
   {
      Geometry::Type geom = geoms[ig];
      nref_max = std::max(nref_max, cf_tr.point_matrices[geom].SizeK());
   }

   BuildHo2Lor(nel_ho, nel_lor, cf_tr);

   // lumped M_H and inv lumped M_L

   // M_H contains the lumped (row sum) high order mass matrix. This is built for
   // preconditioning the inverse needed to build the prolongation operator P
   Vector M_H(ndof_ho);
   M_H = 0.0;
   // ML_inv_ea contains the inverse lumped (row sum) mass matrix. Note that the
   // method will also work with a full (consistent) mass matrix, though this is
   // not implemented here. L refers to the low-order refined mesh
   ML_inv_ea.SetSize(ndof_lor);
   ML_inv_ea = 0.0;

   ParBilinearForm pMho(pfes_ho_scalar.get());
   pMho.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   pMho.AddDomainIntegrator(new MassIntegrator);
   pMho.Assemble();

   // Processor local lumped Mass
   Vector ones_ho(pMho.Width()); ones_ho = 1.0;
   M_H = 0.0;
   pMho.Mult(ones_ho, M_H);

   ParBilinearForm pMlor(pfes_lor_scalar.get());
   pMlor.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   pMlor.AddDomainIntegrator(new MassIntegrator);
   pMlor.Assemble();

   Vector ones_lor(pMlor.Width()); ones_lor = 1.0;
   pMlor.Mult(ones_lor, ML_inv_ea);


   // DOF by DOF inverse of non-zero entries
   LumpedMassInverse(ML_inv_ea);

   // mixed mass M_LH
   MixedMassEA(*pfes_ho_scalar.get(), *pfes_lor_scalar.get(), M_LH_ea, d_mt);

   // Set ownership
   M_LH_local_op = new H1SpaceMixedMassOperator(pfes_ho_scalar.get(),
                                                pfes_lor_scalar.get(),
                                                &ho2lor, &M_LH_ea);

   const Operator *P_ho = pfes_ho_scalar->GetProlongationMatrix();
   const Operator *P_lor = pfes_lor_scalar->GetProlongationMatrix();

   Array<int> ess_tdof_list;  // leave empty

   if (P_ho || P_lor)
   {
      if (P_ho && P_lor)
      {
         Operator *Pt_lor = new TransposeOperator(P_lor);
         RML_inv.SetSize(pfes_lor_scalar->GetTrueVSize());
         GetTDofs(*pfes_lor_scalar, ML_inv_ea, RML_inv);
         ML_inv_vea.reset(new H1SpaceLumpedMassOperator(pfes_ho_scalar.get(),
                                                        pfes_lor_scalar.get(),
                                                        RML_inv));
         M_LH.reset(new TripleProductOperator(Pt_lor, M_LH_local_op, P_ho, true,
                                              true, false));

         Vector RM_H(pfes_ho_scalar->GetTrueVSize());
         GetTDofsTranspose(*pfes_ho_scalar, M_H, RM_H);
         precon.reset(new OperatorJacobiSmoother(RM_H, ess_tdof_list));
      }
      else if (P_ho)
      {
         ML_inv_vea.reset(new H1SpaceLumpedMassOperator(pfes_ho_scalar.get(),
                                                        pfes_lor_scalar.get(),
                                                        ML_inv_ea));
         M_LH.reset(new ProductOperator(M_LH_local_op, P_ho, true, false));

         Vector RM_H(pfes_ho_scalar->GetTrueVSize());
         GetTDofsTranspose(*pfes_ho_scalar.get(), M_H, RM_H);
         precon.reset(new OperatorJacobiSmoother(RM_H, ess_tdof_list));
      }
      else if (P_lor)
      {
         Operator *Pt_lor = new TransposeOperator(P_lor);
         RML_inv.SetSize(pfes_lor_scalar->GetTrueVSize());
         GetTDofsTranspose(*pfes_lor_scalar, ML_inv_ea, RML_inv);
         ML_inv_vea.reset(new H1SpaceLumpedMassOperator(pfes_ho_scalar.get(),
                                                        pfes_lor_scalar.get(),
                                                        RML_inv));
         M_LH.reset(new ProductOperator(Pt_lor, M_LH_local_op, true, true));
         R.reset(new ProductOperator(ML_inv_vea.get(), M_LH.get(), false,
                                     false));

         precon.reset(new OperatorJacobiSmoother(M_H, ess_tdof_list));
      }
      else
      {
         ML_inv_vea.reset(new H1SpaceLumpedMassOperator(pfes_ho_scalar.get(),
                                                        pfes_lor_scalar.get(),
                                                        ML_inv_ea));
         M_LH.reset(M_LH_local_op);

         precon.reset(new OperatorJacobiSmoother(M_H, ess_tdof_list));
      }
   }
   R.reset(new ProductOperator(ML_inv_vea.get(), M_LH.get(), false,
                               false));

   TransposeOperator* RT = new TransposeOperator(R.get());
   RTxM_LH.reset(new ProductOperator(RT, M_LH.get(), true, false));

   SetupPCG();
}

#endif

void L2ProjectionGridTransfer::L2ProjectionH1Space::Mult(
   const Vector& x, Vector& y) const
{
   Vector X(fes_ho.GetTrueVSize());
   Vector X_dim(R->Width());

   Vector Y_dim(R->Height());
   Vector Y(fes_lor.GetTrueVSize());

   Array<int> vdofs_list;

   GetTDofs(fes_ho, x, X);

   for (int d = 0; d < fes_ho.GetVDim(); ++d)
   {
      TDofsListByVDim(fes_ho, d, vdofs_list);
      X.GetSubVector(vdofs_list, X_dim);
      R->Mult(X_dim, Y_dim);
      TDofsListByVDim(fes_lor, d, vdofs_list);
      Y.SetSubVector(vdofs_list, Y_dim);
   }

   SetFromTDofs(fes_lor, Y, y);

}

void L2ProjectionGridTransfer::L2ProjectionH1Space::MultTranspose(
   const Vector& x, Vector& y) const
{
   Vector X(fes_lor.GetTrueVSize());
   Vector X_dim(R->Height());

   Vector Y_dim(R->Width());
   Vector Y(fes_ho.GetTrueVSize());

   Array<int> vdofs_list;

   GetTDofsTranspose(fes_lor, x, X);

   for (int d = 0; d < fes_ho.GetVDim(); ++d)
   {
      TDofsListByVDim(fes_lor, d, vdofs_list);
      X.GetSubVector(vdofs_list, X_dim);
      R->MultTranspose(X_dim, Y_dim);
      TDofsListByVDim(fes_ho, d, vdofs_list);
      Y.SetSubVector(vdofs_list, Y_dim);
   }

   SetFromTDofsTranspose(fes_ho, Y, y);

}

void L2ProjectionGridTransfer::L2ProjectionH1Space::Prolongate(
   const Vector& x, Vector& y) const
{

   Vector X(fes_lor.GetTrueVSize());
   Vector X_dim(M_LH->Height());
   Vector Xbar(pcg.Width());

   Vector Y_dim(pcg.Height());
   Y_dim = 0.0;
   Vector Y(fes_ho.GetTrueVSize());

   Array<int> vdofs_list;

   GetTDofs(fes_lor, x, X);

   for (int d = 0; d < fes_ho.GetVDim(); ++d)
   {
      TDofsListByVDim(fes_lor, d, vdofs_list);
      X.GetSubVector(vdofs_list, X_dim);
      // Compute y = P x = (R^T M_LH)^(-1) M_LH^T X = (R^T M_LH)^(-1) Xbar
      M_LH->MultTranspose(X_dim, Xbar);
      pcg.Mult(Xbar, Y_dim);
      TDofsListByVDim(fes_ho, d, vdofs_list);
      Y.SetSubVector(vdofs_list, Y_dim);
   }

   SetFromTDofs(fes_ho, Y, y);

}

void L2ProjectionGridTransfer::L2ProjectionH1Space::ProlongateTranspose(
   const Vector& x, Vector& y) const
{
   Vector X(fes_ho.GetTrueVSize());
   Vector X_dim(pcg.Width());
   Vector Xbar(pcg.Height());

   Vector Y_dim(M_LH->Height());
   Vector Y(fes_lor.GetTrueVSize());

   Array<int> vdofs_list;

   GetTDofsTranspose(fes_ho, x, X);

   for (int d = 0; d < fes_ho.GetVDim(); ++d)
   {
      TDofsListByVDim(fes_ho, d, vdofs_list);
      X.GetSubVector(vdofs_list, X_dim);
      // Compute y = P^T x = M_LH (R^T M_LH)^(-1) X = M_LH Xbar
      Xbar = 0.0;
      pcg.Mult(X_dim, Xbar);
      M_LH->Mult(Xbar, Y_dim);
      TDofsListByVDim(fes_lor, d, vdofs_list);
      Y.SetSubVector(vdofs_list, Y_dim);
   }

   SetFromTDofsTranspose(fes_lor, Y, y);

}

void L2ProjectionGridTransfer::L2ProjectionH1Space::SetRelTol(real_t p_rtol_)
{
   pcg.SetRelTol(p_rtol_);
}

void L2ProjectionGridTransfer::L2ProjectionH1Space::SetAbsTol(real_t p_atol_)
{
   pcg.SetAbsTol(p_atol_);
}

std::pair<
std::unique_ptr<SparseMatrix>,
std::unique_ptr<SparseMatrix>>
                            L2ProjectionGridTransfer::L2ProjectionH1Space::ComputeSparseRAndM_LH()
{
   std::pair<std::unique_ptr<SparseMatrix>,
       std::unique_ptr<SparseMatrix>> r_and_mlh;

   Mesh* mesh_ho = fes_ho.GetMesh();
   Mesh* mesh_lor = fes_lor.GetMesh();
   int nel_ho = mesh_ho->GetNE();
   int nel_lor = mesh_lor->GetNE();
   int ndof_lor = fes_lor.GetNDofs();

   // If the local mesh is empty, skip all computations
   if (nel_ho == 0)
   {
      return std::make_pair(
                std::unique_ptr<SparseMatrix>(new SparseMatrix),
                std::unique_ptr<SparseMatrix>(new SparseMatrix)
             );
   }

   const CoarseFineTransformations& cf_tr = mesh_lor->GetRefinementTransforms();

   int nref_max = 0;
   Array<Geometry::Type> geoms;
   mesh_ho->GetGeometries(mesh_ho->Dimension(), geoms);
   for (int ig = 0; ig < geoms.Size(); ++ig)
   {
      Geometry::Type geom = geoms[ig];
      nref_max = std::max(nref_max, cf_tr.point_matrices[geom].SizeK());
   }

   BuildHo2Lor(nel_ho, nel_lor, cf_tr);

   // ML_inv contains the inverse lumped (row sum) mass matrix. Note that the
   // method will also work with a full (consistent) mass matrix, though this is
   // not implemented here. L refers to the low-order refined mesh
   Vector ML_inv(ndof_lor);
   ML_inv = 0.0;

   // Compute ML_inv
   for (int iho = 0; iho < nel_ho; ++iho)
   {
      Array<int> lor_els;
      ho2lor.GetRow(iho, lor_els);
      int nref = ho2lor.RowSize(iho);

      Geometry::Type geom = mesh_ho->GetElementBaseGeometry(iho);
      const FiniteElement& fe_lor = *fes_lor.GetFE(lor_els[0]);
      int nedof_lor = fe_lor.GetDof();

      // Instead of using a MassIntegrator, manually loop over integration
      // points so we can row sum and store the diagonal as a Vector.
      Vector ML_el(nedof_lor);
      Vector shape_lor(nedof_lor);
      Array<int> dofs_lor(nedof_lor);

      for (int iref = 0; iref < nref; ++iref)
      {
         int ilor = lor_els[iref];
         ElementTransformation* el_tr = fes_lor.GetElementTransformation(ilor);

         int order = 2 * fe_lor.GetOrder() + el_tr->OrderW();
         const IntegrationRule* ir = &IntRules.Get(geom, order);
         ML_el = 0.0;
         for (int i = 0; i < ir->GetNPoints(); ++i)
         {
            const IntegrationPoint& ip_lor = ir->IntPoint(i);
            fe_lor.CalcShape(ip_lor, shape_lor);
            el_tr->SetIntPoint(&ip_lor);
            ML_el += (shape_lor *= (el_tr->Weight() * ip_lor.weight));
         }
         fes_lor.GetElementDofs(ilor, dofs_lor);
         ML_inv.AddElementVector(dofs_lor, ML_el);
      }
   }
   // DOF by DOF inverse of non-zero entries
   LumpedMassInverse(ML_inv);

   // Compute sparsity pattern for R = M_L^(-1) M_LH and allocate
   r_and_mlh.first = AllocR();
   // Allocate M_LH (same sparsity pattern as R)
   // L refers to the low-order refined mesh (DOFs correspond to rows)
   // H refers to the higher-order mesh (DOFs correspond to columns)
   Memory<int> I(r_and_mlh.first->Height() + 1);
   for (int icol = 0; icol < r_and_mlh.first->Height() + 1; ++icol)
   {
      I[icol] = r_and_mlh.first->GetI()[icol];
   }
   Memory<int> J(r_and_mlh.first->NumNonZeroElems());
   for (int jcol = 0; jcol < r_and_mlh.first->NumNonZeroElems(); ++jcol)
   {
      J[jcol] = r_and_mlh.first->GetJ()[jcol];
   }
   r_and_mlh.second = std::unique_ptr<SparseMatrix>(
                         new SparseMatrix(I, J, NULL, r_and_mlh.first->Height(),
                                          r_and_mlh.first->Width(), true, true, true));

   IntegrationPointTransformation ip_tr;
   IsoparametricTransformation& emb_tr = ip_tr.Transf;

   // Compute M_LH and R
   offsets.SetSize(nel_ho+1);
   offsets[0] = 0;
   for (int iho = 0; iho < nel_ho; ++iho)
   {
      Array<int> lor_els;
      ho2lor.GetRow(iho, lor_els);
      int nref = ho2lor.RowSize(iho);

      Geometry::Type geom = mesh_ho->GetElementBaseGeometry(iho);
      const FiniteElement& fe_ho = *fes_ho.GetFE(iho);
      const FiniteElement& fe_lor = *fes_lor.GetFE(lor_els[0]);
      offsets[iho+1] = offsets[iho] + fe_ho.GetDof()*fe_lor.GetDof()*nref;

      ElementTransformation *tr_ho = fes_ho.GetElementTransformation(iho);

      emb_tr.SetIdentityTransformation(geom);
      const DenseTensor& pmats = cf_tr.point_matrices[geom];

      int nedof_ho = fe_ho.GetDof();
      int nedof_lor = fe_lor.GetDof();
      DenseMatrix M_LH_el(nedof_lor, nedof_ho);
      DenseMatrix R_el(nedof_lor, nedof_ho);

      for (int iref = 0; iref < nref; ++iref)
      {
         int ilor = lor_els[iref];
         ElementTransformation* tr_lor = fes_lor.GetElementTransformation(ilor);

         // Create the transformation that embeds the fine low-order element
         // within the coarse high-order element in reference space
         emb_tr.SetPointMat(pmats(cf_tr.embeddings[ilor].matrix));

         ElemMixedMass(geom, fe_ho, fe_lor, tr_ho, tr_lor, ip_tr, M_LH_el);

         Array<int> dofs_lor(nedof_lor);
         fes_lor.GetElementDofs(ilor, dofs_lor);
         Vector R_row;
         for (int i = 0; i < nedof_lor; ++i)
         {
            M_LH_el.GetRow(i, R_row);
            R_el.SetRow(i, R_row.Set(ML_inv[dofs_lor[i]], R_row));
         }
         Array<int> dofs_ho(nedof_ho);
         fes_ho.GetElementDofs(iho, dofs_ho);
         r_and_mlh.second->AddSubMatrix(dofs_lor, dofs_ho, M_LH_el);
         r_and_mlh.first->AddSubMatrix(dofs_lor, dofs_ho, R_el);

      }
   }

   return r_and_mlh;
}

void L2ProjectionGridTransfer::L2ProjectionH1Space::GetTDofs(
   const FiniteElementSpace& fes, const Vector& x, Vector& X) const
{
   const Operator* res = fes.GetRestrictionOperator();
   if (res)
   {
      res->Mult(x, X);
   }
   else
   {
      X = x;
   }
}

void L2ProjectionGridTransfer::L2ProjectionH1Space::SetFromTDofs(
   const FiniteElementSpace& fes, const Vector &X, Vector& x) const
{
   const Operator* P = fes.GetProlongationMatrix();
   if (P)
   {
      P->Mult(X, x);
   }
   else
   {
      x = X;
   }
}

void L2ProjectionGridTransfer::L2ProjectionH1Space::GetTDofsTranspose(
   const FiniteElementSpace& fes, const Vector& x, Vector& X) const
{
   const Operator* P = fes.GetProlongationMatrix();
   if (P)
   {
      P->MultTranspose(x, X);
   }
   else
   {
      X = x;
   }
}

void L2ProjectionGridTransfer::L2ProjectionH1Space::SetFromTDofsTranspose(
   const FiniteElementSpace& fes, const Vector &X, Vector& x) const
{
   const Operator *R_op = fes.GetRestrictionOperator();
   if (R_op)
   {
      R_op->MultTranspose(X, x);
   }
   else
   {
      x = X;
   }
}

void L2ProjectionGridTransfer::L2ProjectionH1Space::TDofsListByVDim(
   const FiniteElementSpace& fes, int vdim, Array<int>& vdofs_list) const
{
   const SparseMatrix *R_mat = fes.GetRestrictionMatrix();
   if (R_mat)
   {
      Array<int> x_vdofs_list(fes.GetNDofs());
      Array<int> x_vdofs_marker(fes.GetVSize());
      Array<int> X_vdofs_marker(fes.GetTrueVSize());
      fes.GetVDofs(vdim, x_vdofs_list);
      FiniteElementSpace::ListToMarker(x_vdofs_list, fes.GetVSize(), x_vdofs_marker);
      R_mat->BooleanMult(x_vdofs_marker, X_vdofs_marker);
      FiniteElementSpace::MarkerToList(X_vdofs_marker, vdofs_list);
   }
   else
   {
      vdofs_list.SetSize(fes.GetNDofs());
      fes.GetVDofs(vdim, vdofs_list);
   }
}

void L2ProjectionGridTransfer::L2ProjectionH1Space::LumpedMassInverse(
   Vector& ML_inv) const
{
#ifdef MFEM_USE_MPI
   // LumpedMassInverse may get called from serial and MPI parallel routines
   // since we do not know which code path is calling it we must check if
   // the pfes pointer is null when MPI is available.
   auto * fes = pfes_lor_scalar == nullptr ? fes_lor_scalar.get() :
                pfes_lor_scalar.get();
#else
   auto * fes = fes_lor_scalar.get();
#endif
   MFEM_ASSERT(fes != nullptr, "[p]fes_lor_scalar is nullptr");

   Vector ML_inv_true(fes->GetTrueVSize());
   const Operator *P = fes->GetProlongationMatrix();
   if (P) { P->MultTranspose(ML_inv, ML_inv_true); }
   else { ML_inv_true = ML_inv; }

   ML_inv_true.Reciprocal();

   if (P) { P->Mult(ML_inv_true, ML_inv); }
   else { ML_inv = ML_inv_true; }

}

std::unique_ptr<SparseMatrix>
L2ProjectionGridTransfer::L2ProjectionH1Space::AllocR()
{
   const Table& elem_dof_ho = fes_ho.GetElementToDofTable();
   const Table& elem_dof_lor = fes_lor.GetElementToDofTable();
   const int ndof_ho = fes_ho.GetNDofs();
   const int ndof_lor = fes_lor.GetNDofs();

   Table dof_elem_lor;
   Transpose(elem_dof_lor, dof_elem_lor, ndof_lor);

   Mesh* mesh_lor = fes_lor.GetMesh();
   const CoarseFineTransformations& cf_tr = mesh_lor->GetRefinementTransforms();

   // mfem::Mult but uses ho2lor to map HO elements to LOR elements
   const int* elem_dof_hoI = elem_dof_ho.GetI();
   const int* elem_dof_hoJ = elem_dof_ho.GetJ();
   const int* dof_elem_lorI = dof_elem_lor.GetI();
   const int* dof_elem_lorJ = dof_elem_lor.GetJ();

   Array<int> I(ndof_lor + 1);

   // figure out the size of J
   Array<int> dof_used_ho;
   dof_used_ho.SetSize(ndof_ho, -1);

   int sizeJ = 0;
   for (int ilor = 0; ilor < ndof_lor; ++ilor)
   {
      for (int jlor = dof_elem_lorI[ilor]; jlor < dof_elem_lorI[ilor + 1]; ++jlor)
      {
         int el_lor = dof_elem_lorJ[jlor];
         int iho = cf_tr.embeddings[el_lor].parent;
         for (int jho = elem_dof_hoI[iho]; jho < elem_dof_hoI[iho + 1]; ++jho)
         {
            int dof_ho = elem_dof_hoJ[jho];
            if (dof_used_ho[dof_ho] != ilor)
            {
               dof_used_ho[dof_ho] = ilor;
               ++sizeJ;
            }
         }
      }
   }

   // initialize dof_ho_dof_lor
   Table dof_lor_dof_ho;
   dof_lor_dof_ho.SetDims(ndof_lor, sizeJ);

   for (int i = 0; i < ndof_ho; ++i)
   {
      dof_used_ho[i] = -1;
   }

   // set values of J
   int* dof_dofI = dof_lor_dof_ho.GetI();
   int* dof_dofJ = dof_lor_dof_ho.GetJ();
   sizeJ = 0;
   for (int ilor = 0; ilor < ndof_lor; ++ilor)
   {
      dof_dofI[ilor] = sizeJ;
      for (int jlor = dof_elem_lorI[ilor]; jlor < dof_elem_lorI[ilor + 1]; ++jlor)
      {
         int el_lor = dof_elem_lorJ[jlor];
         int iho = cf_tr.embeddings[el_lor].parent;
         for (int jho = elem_dof_hoI[iho]; jho < elem_dof_hoI[iho + 1]; ++jho)
         {
            int dof_ho = elem_dof_hoJ[jho];
            if (dof_used_ho[dof_ho] != ilor)
            {
               dof_used_ho[dof_ho] = ilor;
               dof_dofJ[sizeJ] = dof_ho;
               ++sizeJ;
            }
         }
      }
   }

   dof_lor_dof_ho.SortRows();
   real_t* data = Memory<real_t>(dof_dofI[ndof_lor]);

   std::unique_ptr<SparseMatrix> R_local(new SparseMatrix(
                                            dof_dofI, dof_dofJ, data, ndof_lor,
                                            ndof_ho, true, true, true));
   (*R_local) = 0.0;

   dof_lor_dof_ho.LoseData();

   return R_local;
}

L2ProjectionGridTransfer::H1SpaceMixedMassOperator::H1SpaceMixedMassOperator(
   const FiniteElementSpace* fes_ho_, const FiniteElementSpace* fes_lor_,
   Table* ho2lor_, Vector* M_LH_ea_) :
   Operator(fes_lor_->GetElementRestriction(ElementDofOrdering::NATIVE)->Width(),
            fes_ho_->GetElementRestriction(ElementDofOrdering::NATIVE)->Width()),
   fes_ho(fes_ho_), fes_lor(fes_lor_), ho2lor(ho2lor_),
   M_LH_ea(M_LH_ea_)
{ }

void L2ProjectionGridTransfer::H1SpaceMixedMassOperator::Mult(const Vector &x,
                                                              Vector &y) const
{
   if (fes_ho->GetNE() == 0)
   {
      return;
   }

   const Operator* elem_restrict_ho = fes_ho->GetElementRestriction(
                                         ElementDofOrdering::NATIVE);
   const Operator* elem_restrict_lor = fes_lor->GetElementRestriction(
                                          ElementDofOrdering::NATIVE);

   const int vdim = fes_ho->GetVDim();
   const int iho = 0;
   const int nref = ho2lor->RowSize(iho);
   const int ndof_ho = fes_ho->GetFE(iho)->GetDof();
   const int ndof_lor = fes_lor->GetFE(ho2lor->GetRow(iho)[0])->GetDof();
   const Mesh *mesh_ho = fes_ho->GetMesh();
   const int nel_ho = mesh_ho->GetNE();

   Vector tempx(elem_restrict_ho->Height());
   elem_restrict_ho->Mult(x, tempx);

   Vector tempy(ndof_lor*nref*vdim*nel_ho);

   auto v_M_mixed_ea = Reshape(M_LH_ea->Read(), ndof_lor, ndof_ho, nref,
                               nel_ho);
   auto v_tempx    = Reshape(tempx.Read(), ndof_ho, vdim, nel_ho);
   auto v_tempy    = Reshape(tempy.Write(), ndof_lor, nref, vdim, nel_ho);


   mfem::forall(ndof_lor * nref * vdim * nel_ho, [=] MFEM_HOST_DEVICE (int tid)
   {
      const int j   = tid % ndof_lor;
      const int i   = (tid / ndof_lor) % nref;
      const int v   = (tid / (ndof_lor * nref)) % vdim;
      const int iho = (tid / (ndof_lor * nref * vdim)) % nel_ho;

      real_t dot = 0.0;
      for (int k=0; k<ndof_ho; ++k)
      {
         dot += v_M_mixed_ea(j, k, i, iho) * v_tempx(k, v, iho);
      }

      v_tempy(j, i, v, iho) = dot;
   });

   elem_restrict_lor->MultTranspose(tempy, y);
}

void L2ProjectionGridTransfer::H1SpaceMixedMassOperator::MultTranspose(
   const Vector &x, Vector &y) const
{
   if (fes_ho->GetNE() == 0)
   {
      return;
   }

   const Operator* elem_restrict_ho = fes_ho->GetElementRestriction(
                                         ElementDofOrdering::NATIVE);
   const Operator* elem_restrict_lor = fes_lor->GetElementRestriction(
                                          ElementDofOrdering::NATIVE);

   const int vdim = fes_ho->GetVDim();
   const int iho = 0;
   const int nref = ho2lor->RowSize(iho);
   const int ndof_ho = fes_ho->GetFE(iho)->GetDof();
   const int ndof_lor = fes_lor->GetFE(ho2lor->GetRow(iho)[0])->GetDof();
   const Mesh *mesh_ho = fes_ho->GetMesh();
   const int nel_ho = mesh_ho->GetNE();

   Vector tempx(elem_restrict_lor->Height());
   elem_restrict_lor->Mult(x, tempx);

   Vector tempy(ndof_ho*vdim*nel_ho);

   auto v_M_mixed_ea = Reshape(M_LH_ea->Read(), ndof_lor, ndof_ho, nref,
                               nel_ho);
   auto v_tempx    = Reshape(tempx.Read(), ndof_lor, nref, vdim, nel_ho);
   auto v_tempy    = Reshape(tempy.Write(), ndof_ho, vdim, nel_ho);

   mfem::forall(ndof_ho * vdim * nel_ho, [=] MFEM_HOST_DEVICE (int tid)
   {
      const int k = tid % ndof_ho;
      const int v = (tid / ndof_ho) % vdim;
      const int iho = (tid / (ndof_ho * vdim)) % nel_ho;

      real_t dot = 0.0;
      for (int i=0; i<nref; ++i)
      {
         for (int j=0; j<ndof_lor; ++j)
         {
            dot += v_M_mixed_ea(j, k, i, iho) * v_tempx(j, i, v, iho);
         }
         v_tempy(k, v, iho) = dot;
      }
   });

   elem_restrict_ho->MultTranspose(tempy, y);
}

L2ProjectionGridTransfer::H1SpaceLumpedMassOperator::H1SpaceLumpedMassOperator(
   const FiniteElementSpace* fes_ho_,
   const FiniteElementSpace* fes_lor_,
   Vector& ML_inv_) :
   Operator(ML_inv_.Size(), ML_inv_.Size()),
   fes_ho(fes_ho_), fes_lor(fes_lor_),
   ML_inv(&ML_inv_)
{ }

void L2ProjectionGridTransfer::H1SpaceLumpedMassOperator::Mult(const Vector &x,
                                                               Vector &y) const
{
   MFEM_ASSERT(ML_inv->Size() == x.Size(), "sizes not the same");
   auto v_ML_inv = Reshape(ML_inv->Read(), ML_inv->Size());
   auto v_x = Reshape(x.Read(), x.Size());
   auto v_y = Reshape(y.Write(), y.Size());

   mfem::forall(ML_inv->Size(), [=] MFEM_HOST_DEVICE(int i)
   { v_y(i) = v_ML_inv(i) * v_x(i); });
}

void L2ProjectionGridTransfer::H1SpaceLumpedMassOperator::MultTranspose(
   const Vector &x, Vector &y) const
{
   this->Mult(x,y); // lumped diagonal has the same Mult and MultTranspose behavior
}

L2ProjectionGridTransfer::~L2ProjectionGridTransfer()
{
   delete F;
   delete B;
}

const Operator &L2ProjectionGridTransfer::ForwardOperator()
{
   if (!F) { BuildF(); }
   return *F;
}

const Operator &L2ProjectionGridTransfer::BackwardOperator()
{
   if (!B)
   {
      if (!F) { BuildF(); }
      B = new L2Prolongation(*F);
   }
   return *B;
}

void L2ProjectionGridTransfer::BuildF()
{
   if (!force_l2_space &&
       dom_fes.FEColl()->GetContType() == FiniteElementCollection::CONTINUOUS)
   {
      if (!Parallel())
      {
         F = new L2ProjectionH1Space(dom_fes, ran_fes,
                                     use_ea, d_mt);
      }
      else
      {
#ifdef MFEM_USE_MPI
         const mfem::ParFiniteElementSpace& dom_pfes =
            static_cast<mfem::ParFiniteElementSpace&>(dom_fes);
         const mfem::ParFiniteElementSpace& ran_pfes =
            static_cast<mfem::ParFiniteElementSpace&>(ran_fes);
         F = new L2ProjectionH1Space(dom_pfes, ran_pfes,
                                     use_ea, d_mt);
#endif
      }
   }
   else
   {
      F = new L2ProjectionL2Space(dom_fes, ran_fes,
                                  use_ea, d_mt);
   }
}

bool L2ProjectionGridTransfer::SupportsBackwardsOperator() const
{
   return ran_fes.GetTrueVSize() >= dom_fes.GetTrueVSize();
}


TransferOperator::TransferOperator(const FiniteElementSpace& lFESpace_,
                                   const FiniteElementSpace& hFESpace_)
   : Operator(hFESpace_.GetVSize(), lFESpace_.GetVSize())
{
   bool isvar_order = lFESpace_.IsVariableOrder() || hFESpace_.IsVariableOrder();
   if (lFESpace_.FEColl() == hFESpace_.FEColl() && !isvar_order)
   {
      OperatorPtr P(Operator::ANY_TYPE);
      hFESpace_.GetTransferOperator(lFESpace_, P);
      P.SetOperatorOwner(false);
      opr = P.Ptr();
   }
   else if (lFESpace_.GetVDim() == 1
            && hFESpace_.GetVDim() == 1
            && dynamic_cast<const TensorBasisElement*>(lFESpace_.GetTypicalFE())
            && dynamic_cast<const TensorBasisElement*>(hFESpace_.GetTypicalFE())
            && !isvar_order
            && (hFESpace_.FEColl()->GetContType() ==
                mfem::FiniteElementCollection::CONTINUOUS ||
                hFESpace_.FEColl()->GetContType() ==
                mfem::FiniteElementCollection::DISCONTINUOUS))
   {
      opr = new TensorProductPRefinementTransferOperator(lFESpace_, hFESpace_);
   }
   else
   {
      opr = new PRefinementTransferOperator(lFESpace_, hFESpace_);
   }
}

TransferOperator::~TransferOperator() { delete opr; }

void TransferOperator::Mult(const Vector& x, Vector& y) const
{
   opr->Mult(x, y);
}

void TransferOperator::MultTranspose(const Vector& x, Vector& y) const
{
   opr->MultTranspose(x, y);
}


PRefinementTransferOperator::PRefinementTransferOperator(
   const FiniteElementSpace& lFESpace_, const FiniteElementSpace& hFESpace_)
   : Operator(hFESpace_.GetVSize(), lFESpace_.GetVSize()), lFESpace(lFESpace_),
     hFESpace(hFESpace_)
{
   isvar_order = lFESpace_.IsVariableOrder() || hFESpace_.IsVariableOrder();
}

void PRefinementTransferOperator::Mult(const Vector& x, Vector& y) const
{
   Mesh* mesh = hFESpace.GetMesh();
   Array<int> l_dofs, h_dofs, l_vdofs, h_vdofs;
   DenseMatrix loc_prol;
   Vector subY, subX;

   Geometry::Type cached_geom = Geometry::INVALID;
   const FiniteElement* h_fe = NULL;
   const FiniteElement* l_fe = NULL;
   IsoparametricTransformation T;

   int vdim = lFESpace.GetVDim();

   y = 0.0;

   DofTransformation doftrans_h, doftrans_l;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      hFESpace.GetElementDofs(i, h_dofs, doftrans_h);
      lFESpace.GetElementDofs(i, l_dofs, doftrans_l);

      const Geometry::Type geom = mesh->GetElementBaseGeometry(i);
      if (geom != cached_geom || isvar_order)
      {
         h_fe = hFESpace.GetFE(i);
         l_fe = lFESpace.GetFE(i);
         T.SetIdentityTransformation(h_fe->GetGeomType());
         h_fe->GetTransferMatrix(*l_fe, T, loc_prol);
         subY.SetSize(loc_prol.Height());
         cached_geom = geom;
      }

      for (int vd = 0; vd < vdim; vd++)
      {
         l_dofs.Copy(l_vdofs);
         lFESpace.DofsToVDofs(vd, l_vdofs);
         h_dofs.Copy(h_vdofs);
         hFESpace.DofsToVDofs(vd, h_vdofs);
         x.GetSubVector(l_vdofs, subX);
         doftrans_l.InvTransformPrimal(subX);
         loc_prol.Mult(subX, subY);
         doftrans_h.TransformPrimal(subY);
         y.SetSubVector(h_vdofs, subY);
      }
   }
}

void PRefinementTransferOperator::MultTranspose(const Vector& x,
                                                Vector& y) const
{
   y = 0.0;

   Mesh* mesh = hFESpace.GetMesh();
   Array<int> l_dofs, h_dofs, l_vdofs, h_vdofs;
   DenseMatrix loc_prol;
   Vector subY, subX;

   Array<char> processed(hFESpace.GetVSize());
   processed = 0;

   Geometry::Type cached_geom = Geometry::INVALID;
   const FiniteElement* h_fe = NULL;
   const FiniteElement* l_fe = NULL;
   IsoparametricTransformation T;

   int vdim = lFESpace.GetVDim();

   DofTransformation doftrans_h, doftrans_l;

   for (int i = 0; i < mesh->GetNE(); i++)
   {
      hFESpace.GetElementDofs(i, h_dofs, doftrans_h);
      lFESpace.GetElementDofs(i, l_dofs, doftrans_l);

      const Geometry::Type geom = mesh->GetElementBaseGeometry(i);
      if (geom != cached_geom || isvar_order)
      {
         h_fe = hFESpace.GetFE(i);
         l_fe = lFESpace.GetFE(i);
         T.SetIdentityTransformation(h_fe->GetGeomType());
         h_fe->GetTransferMatrix(*l_fe, T, loc_prol);
         loc_prol.Transpose();
         subY.SetSize(loc_prol.Height());
         cached_geom = geom;
      }

      for (int vd = 0; vd < vdim; vd++)
      {
         l_dofs.Copy(l_vdofs);
         lFESpace.DofsToVDofs(vd, l_vdofs);
         h_dofs.Copy(h_vdofs);
         hFESpace.DofsToVDofs(vd, h_vdofs);

         x.GetSubVector(h_vdofs, subX);
         doftrans_h.InvTransformDual(subX);
         for (int p = 0; p < h_dofs.Size(); ++p)
         {
            if (processed[lFESpace.DecodeDof(h_dofs[p])])
            {
               subX[p] = 0.0;
            }
         }

         loc_prol.Mult(subX, subY);
         doftrans_l.TransformDual(subY);
         y.AddElementVector(l_vdofs, subY);
      }

      for (int p = 0; p < h_dofs.Size(); ++p)
      {
         processed[lFESpace.DecodeDof(h_dofs[p])] = 1;
      }
   }
}


TensorProductPRefinementTransferOperator::
TensorProductPRefinementTransferOperator(
   const FiniteElementSpace& lFESpace_,
   const FiniteElementSpace& hFESpace_)
   : Operator(hFESpace_.GetVSize(), lFESpace_.GetVSize()), lFESpace(lFESpace_),
     hFESpace(hFESpace_)
{
   // Assuming the same element type
   Mesh* mesh = lFESpace.GetMesh();
   dim = mesh->Dimension();
   const FiniteElement& el = *lFESpace.GetTypicalFE();

   const TensorBasisElement* ltel =
      dynamic_cast<const TensorBasisElement*>(&el);
   MFEM_VERIFY(ltel, "Low order FE space must be tensor product space");

   const TensorBasisElement* htel =
      dynamic_cast<const TensorBasisElement*>(hFESpace.GetTypicalFE());
   MFEM_VERIFY(htel, "High order FE space must be tensor product space");
   const Array<int>& hdofmap = htel->GetDofMap();

   const IntegrationRule& ir = hFESpace.GetTypicalFE()->GetNodes();
   IntegrationRule irLex = ir;

   // The quadrature points, or equivalently, the dofs of the high order space
   // must be sorted in lexicographical order
   for (int i = 0; i < ir.GetNPoints(); ++i)
   {
      int j = hdofmap[i] >=0 ? hdofmap[i] : -1 - hdofmap[i];
      irLex.IntPoint(i) = ir.IntPoint(j);
   }

   NE = lFESpace.GetNE();
   const DofToQuad& maps = el.GetDofToQuad(irLex, DofToQuad::TENSOR);

   D1D = maps.ndof;
   Q1D = maps.nqpt;
   B = maps.B;
   Bt = maps.Bt;

   elem_restrict_lex_l =
      lFESpace.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);

   MFEM_VERIFY(elem_restrict_lex_l,
               "Low order ElementRestriction not available");

   elem_restrict_lex_h =
      hFESpace.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);

   MFEM_VERIFY(elem_restrict_lex_h,
               "High order ElementRestriction not available");

   localL.SetSize(elem_restrict_lex_l->Height(), Device::GetMemoryType());
   localH.SetSize(elem_restrict_lex_h->Height(), Device::GetMemoryType());
   localL.UseDevice(true);
   localH.UseDevice(true);

   MFEM_VERIFY(dynamic_cast<const ElementRestriction*>(elem_restrict_lex_h),
               "High order element restriction is of unsupported type");

   mask.SetSize(localH.Size(), Device::GetMemoryType());
   static_cast<const ElementRestriction*>(elem_restrict_lex_h)
   ->BooleanMask(mask);
   mask.UseDevice(true);
}

namespace TransferKernels
{
void Prolongation2D(const int NE, const int D1D, const int Q1D,
                    const Vector& localL, Vector& localH,
                    const Array<real_t>& B, const Vector& mask)
{
   auto x_ = Reshape(localL.Read(), D1D, D1D, NE);
   auto y_ = Reshape(localH.Write(), Q1D, Q1D, NE);
   auto B_ = Reshape(B.Read(), Q1D, D1D);
   auto m_ = Reshape(mask.Read(), Q1D, Q1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            y_(qx, qy, e) = 0.0;
         }
      }

      for (int dy = 0; dy < D1D; ++dy)
      {
         real_t sol_x[DofQuadLimits::MAX_Q1D];
         for (int qy = 0; qy < Q1D; ++qy)
         {
            sol_x[qy] = 0.0;
         }
         for (int dx = 0; dx < D1D; ++dx)
         {
            const real_t s = x_(dx, dy, e);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_x[qx] += B_(qx, dx) * s;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const real_t d2q = B_(qy, dy);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               y_(qx, qy, e) += d2q * sol_x[qx];
            }
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            y_(qx, qy, e) *= m_(qx, qy, e);
         }
      }
   });
}

void Prolongation3D(const int NE, const int D1D, const int Q1D,
                    const Vector& localL, Vector& localH,
                    const Array<real_t>& B, const Vector& mask)
{
   auto x_ = Reshape(localL.Read(), D1D, D1D, D1D, NE);
   auto y_ = Reshape(localH.Write(), Q1D, Q1D, Q1D, NE);
   auto B_ = Reshape(B.Read(), Q1D, D1D);
   auto m_ = Reshape(mask.Read(), Q1D, Q1D, Q1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               y_(qx, qy, qz, e) = 0.0;
            }
         }
      }

      for (int dz = 0; dz < D1D; ++dz)
      {
         real_t sol_xy[DofQuadLimits::MAX_Q1D][DofQuadLimits::MAX_Q1D];
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_xy[qy][qx] = 0.0;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            real_t sol_x[DofQuadLimits::MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_x[qx] = 0;
            }
            for (int dx = 0; dx < D1D; ++dx)
            {
               const real_t s = x_(dx, dy, dz, e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  sol_x[qx] += B_(qx, dx) * s;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const real_t wy = B_(qy, dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  sol_xy[qy][qx] += wy * sol_x[qx];
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            const real_t wz = B_(qz, dz);
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  y_(qx, qy, qz, e) += wz * sol_xy[qy][qx];
               }
            }
         }
      }
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               y_(qx, qy, qz, e) *= m_(qx, qy, qz, e);
            }
         }
      }
   });
}

void Restriction2D(const int NE, const int D1D, const int Q1D,
                   const Vector& localH, Vector& localL,
                   const Array<real_t>& Bt, const Vector& mask)
{
   auto x_ = Reshape(localH.Read(), Q1D, Q1D, NE);
   auto y_ = Reshape(localL.Write(), D1D, D1D, NE);
   auto Bt_ = Reshape(Bt.Read(), D1D, Q1D);
   auto m_ = Reshape(mask.Read(), Q1D, Q1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      for (int dy = 0; dy < D1D; ++dy)
      {
         for (int dx = 0; dx < D1D; ++dx)
         {
            y_(dx, dy, e) = 0.0;
         }
      }

      for (int qy = 0; qy < Q1D; ++qy)
      {
         real_t sol_x[DofQuadLimits::MAX_D1D];
         for (int dx = 0; dx < D1D; ++dx)
         {
            sol_x[dx] = 0.0;
         }
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const real_t s = m_(qx, qy, e) * x_(qx, qy, e);
            for (int dx = 0; dx < D1D; ++dx)
            {
               sol_x[dx] += Bt_(dx, qx) * s;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            const real_t q2d = Bt_(dy, qy);
            for (int dx = 0; dx < D1D; ++dx)
            {
               y_(dx, dy, e) += q2d * sol_x[dx];
            }
         }
      }
   });
}
void Restriction3D(const int NE, const int D1D, const int Q1D,
                   const Vector& localH, Vector& localL,
                   const Array<real_t>& Bt, const Vector& mask)
{
   auto x_ = Reshape(localH.Read(), Q1D, Q1D, Q1D, NE);
   auto y_ = Reshape(localL.Write(), D1D, D1D, D1D, NE);
   auto Bt_ = Reshape(Bt.Read(), D1D, Q1D);
   auto m_ = Reshape(mask.Read(), Q1D, Q1D, Q1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      for (int dz = 0; dz < D1D; ++dz)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               y_(dx, dy, dz, e) = 0.0;
            }
         }
      }

      for (int qz = 0; qz < Q1D; ++qz)
      {
         real_t sol_xy[DofQuadLimits::MAX_D1D][DofQuadLimits::MAX_D1D];
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               sol_xy[dy][dx] = 0;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            real_t sol_x[DofQuadLimits::MAX_D1D];
            for (int dx = 0; dx < D1D; ++dx)
            {
               sol_x[dx] = 0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const real_t s = m_(qx, qy, qz, e) * x_(qx, qy, qz, e);
               for (int dx = 0; dx < D1D; ++dx)
               {
                  sol_x[dx] += Bt_(dx, qx) * s;
               }
            }
            for (int dy = 0; dy < D1D; ++dy)
            {
               const real_t wy = Bt_(dy, qy);
               for (int dx = 0; dx < D1D; ++dx)
               {
                  sol_xy[dy][dx] += wy * sol_x[dx];
               }
            }
         }
         for (int dz = 0; dz < D1D; ++dz)
         {
            const real_t wz = Bt_(dz, qz);
            for (int dy = 0; dy < D1D; ++dy)
            {
               for (int dx = 0; dx < D1D; ++dx)
               {
                  y_(dx, dy, dz, e) += wz * sol_xy[dy][dx];
               }
            }
         }
      }
   });
}
} // namespace TransferKernels

void TensorProductPRefinementTransferOperator::Mult(const Vector& x,
                                                    Vector& y) const
{
   if (lFESpace.GetMesh()->GetNE() == 0)
   {
      return;
   }

   elem_restrict_lex_l->Mult(x, localL);
   if (dim == 2)
   {
      TransferKernels::Prolongation2D(NE, D1D, Q1D, localL, localH, B, mask);
   }
   else if (dim == 3)
   {
      TransferKernels::Prolongation3D(NE, D1D, Q1D, localL, localH, B, mask);
   }
   else
   {
      MFEM_ABORT("TensorProductPRefinementTransferOperator::Mult not "
                 "implemented for dim = "
                 << dim);
   }
   elem_restrict_lex_h->MultTranspose(localH, y);
}

void TensorProductPRefinementTransferOperator::MultTranspose(const Vector& x,
                                                             Vector& y) const
{
   if (lFESpace.GetMesh()->GetNE() == 0)
   {
      return;
   }

   elem_restrict_lex_h->Mult(x, localH);
   if (dim == 2)
   {
      TransferKernels::Restriction2D(NE, D1D, Q1D, localH, localL, Bt, mask);
   }
   else if (dim == 3)
   {
      TransferKernels::Restriction3D(NE, D1D, Q1D, localH, localL, Bt, mask);
   }
   else
   {
      MFEM_ABORT("TensorProductPRefinementTransferOperator::MultTranspose not "
                 "implemented for dim = "
                 << dim);
   }
   elem_restrict_lex_l->MultTranspose(localL, y);
}


TrueTransferOperator::TrueTransferOperator(const FiniteElementSpace& lFESpace_,
                                           const FiniteElementSpace& hFESpace_)
   : Operator(hFESpace_.GetTrueVSize(), lFESpace_.GetTrueVSize()),
     lFESpace(lFESpace_),
     hFESpace(hFESpace_)
{
   localTransferOperator = new TransferOperator(lFESpace_, hFESpace_);

   P = lFESpace.GetProlongationMatrix();
   R = hFESpace.IsVariableOrder() ? hFESpace.GetHpRestrictionMatrix() :
       hFESpace.GetRestrictionMatrix();

   // P and R can be both null
   // P can be null and R not null
   // If P is not null it is assumed that R is not null as well
   if (P) { MFEM_VERIFY(R, "Both P and R have to be not NULL") }

   if (P)
   {
      tmpL.SetSize(lFESpace_.GetVSize());
      tmpH.SetSize(hFESpace_.GetVSize());
   }
   // P can be null and R not null
   else if (R)
   {
      tmpH.SetSize(hFESpace_.GetVSize());
   }
}

TrueTransferOperator::~TrueTransferOperator()
{
   delete localTransferOperator;
}

void TrueTransferOperator::Mult(const Vector& x, Vector& y) const
{
   if (P)
   {
      P->Mult(x, tmpL);
      localTransferOperator->Mult(tmpL, tmpH);
      R->Mult(tmpH, y);
   }
   else if (R)
   {
      localTransferOperator->Mult(x, tmpH);
      R->Mult(tmpH, y);
   }
   else
   {
      localTransferOperator->Mult(x, y);
   }
}

void TrueTransferOperator::MultTranspose(const Vector& x, Vector& y) const
{
   if (P)
   {
      R->MultTranspose(x, tmpH);
      localTransferOperator->MultTranspose(tmpH, tmpL);
      P->MultTranspose(tmpL, y);
   }
   else if (R)
   {
      R->MultTranspose(x, tmpH);
      localTransferOperator->MultTranspose(tmpH, y);
   }
   else
   {
      localTransferOperator->MultTranspose(x, y);
   }
}

} // namespace mfem
