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

#include "lor_assembly.hpp"
#include "../linalg/dtensor.hpp"

namespace mfem
{

void AssembleBatchedLOR(BilinearForm &form_lor, FiniteElementSpace &fes_ho,
                        const Array<int> &ess_dofs, OperatorHandle &A)
{
   MFEM_VERIFY(UsesTensorBasis(fes_ho),
               "Batched LOR assembly requires tensor basis");
   // ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   // const Operator *restrict = fes_ho.GetElementRestriction(ordering);

   FiniteElementSpace &fes_lor = *form_lor.FESpace();
   Mesh &mesh_lor = *fes_lor.GetMesh();
   Mesh &mesh_ho = *fes_ho.GetMesh();

   int nel_ho = mesh_ho.GetNE();
   int nel_lor = mesh_lor.GetNE();
   int order = fes_ho.GetMaxElementOrder();
   int dim = mesh_ho.Dimension();
   MFEM_VERIFY(dim == 1, "Only 1D supported now...");

   IntegrationRules irs(0, Quadrature1D::GaussLobatto);
   const IntegrationRule &ir = irs.Get(mesh_lor.GetElementGeometry(0), 1);
   int nq = ir.Size();

   // const GeometricFactors *geom
   //    = mesh_lor.GetGeometricFactors(ir, GeometricFactors::JACOBIANS);
   // const Vector &J_data = geom->J;
   Array<double> J_data(nq*dim*dim*nel_lor);


   const CoarseFineTransformations &cf_tr = mesh_lor.GetRefinementTransforms();
   Array<double> invJ_data(nel_ho*pow(order,dim)*nq);
   auto invJ = Reshape(invJ_data.ReadWrite(), nq, order, nel_ho);
   auto J = Reshape(J_data.ReadWrite(), nq, dim, dim, nel_lor);
   for (int iel_lor=0; iel_lor<mesh_lor.GetNE(); ++iel_lor)
   {
      ElementTransformation *T = mesh_lor.GetElementTransformation(iel_lor);
      for (int iq=0; iq<nq; ++iq)
      {
         T->SetIntPoint(&ir[iq]);
         const DenseMatrix &Jq = T->Jacobian();
         J(iq, 0, 0, iel_lor) = Jq(0,0);
      }

      int iel_ho = cf_tr.embeddings[iel_lor].parent;
      int iref = cf_tr.embeddings[iel_lor].matrix;
      for (int iq=0; iq<nq; ++iq)
      {
         invJ(iq, iref, iel_ho) = ir[iq].weight/J(iq, 0, 0, iel_lor);
      }
   }

   int ndofs = fes_lor.GetTrueVSize();

   SparseMatrix *A_mat = new SparseMatrix(ndofs, ndofs);

   const Array<int> &lex_map = dynamic_cast<const NodalFiniteElement&>
                               (*fes_ho.GetFE(0)).GetLexicographicOrdering();

   Array<int> dofs, dofs_lex;
   for (int iel_ho=0; iel_ho<nel_ho; ++iel_ho)
   {
      fes_ho.GetElementDofs(iel_ho, dofs);
      int n = dofs.Size();

      int i0 = dofs[lex_map[0]];
      int i1 = dofs[lex_map[1]];
      int inm1 = dofs[lex_map[n-1]];
      int inm2 = dofs[lex_map[n-2]];

      // Left boundary
      A_mat->Add(i0, i0, invJ(0,0,iel_ho) + invJ(1,0,iel_ho));
      A_mat->Add(i0, i1, -(invJ(0,0,iel_ho) + invJ(1,0,iel_ho)));

      // Right boundary
      A_mat->Add(inm1, inm1, invJ(0,n-2,iel_ho) + invJ(1,n-2,iel_ho));
      A_mat->Add(inm1, inm2, -(invJ(0,n-2,iel_ho) + invJ(1,n-2,iel_ho)));

      // Interior
      for (int i=1; i<n-1; ++i)
      {
         int ii = dofs[lex_map[i]];
         int iim1 = dofs[lex_map[i-1]];
         int iip1 = dofs[lex_map[i+1]];

         A_mat->Add(ii, ii, invJ(0,i-1,iel_ho) + invJ(1,i-1,iel_ho) + invJ(0,i,
                                                                           iel_ho) + invJ(1,i,iel_ho));
         A_mat->Add(ii, iim1, -(invJ(0,i-1,iel_ho) + invJ(1,i-1,iel_ho)));
         A_mat->Add(ii, iip1, -(invJ(0,i,iel_ho) + invJ(1,i,iel_ho)));
      }
   }

   for (int i : ess_dofs)
   {
      A_mat->EliminateRowCol(i, Operator::DIAG_KEEP);
   }

   A_mat->Finalize();
   A.Reset(A_mat); // A now owns A_mat
}

}
