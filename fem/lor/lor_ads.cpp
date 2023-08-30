// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "lor_ads.hpp"
#include "../../general/forall.hpp"
#include "../../fem/pbilinearform.hpp"

namespace mfem
{

#ifdef MFEM_USE_MPI

BatchedLOR_ADS::BatchedLOR_ADS(ParFiniteElementSpace &pfes_ho_,
                               const Vector &X_vert)
   : face_fes(pfes_ho_),
     order(face_fes.GetMaxElementOrder()),
     edge_fec(order, dim),
     edge_fes(face_fes.GetParMesh(), &edge_fec),
     ams(edge_fes, X_vert)
{
   MFEM_VERIFY(face_fes.GetParMesh()->Dimension() == dim, "Bad dimension.")
   FormCurlMatrix();
}

void BatchedLOR_ADS::Form3DFaceToEdge(Array<int> &face2edge)
{
   const int o = order;
   const int op1 = o + 1;
   const int nface = dim*o*o*op1;

   face2edge.SetSize(4*nface);
   auto f2e = Reshape(face2edge.HostWrite(), 4, nface);

   for (int c=0; c<dim; ++c)
   {
      const int nx = (c == 0) ? op1 : o;
      const int ny = (c == 1) ? op1 : o;

      const int c1 = (c+1)%dim;
      const int c2 = (c+2)%dim;

      const int nx_e1 = (c1 == 0) ? o : op1;
      const int ny_e1 = (c1 == 1) ? o : op1;
      const int nx_e2 = (c2 == 0) ? o : op1;
      const int ny_e2 = (c2 == 1) ? o : op1;

      for (int i=0; i<o*o*op1; ++i)
      {
         int i1[dim], i2[dim];
         const int ix = i1[0] = i2[0] = i%nx;
         const int iy = i1[1] = i2[1] = (i/nx)%ny;
         const int iz = i1[2] = i2[2] = i/nx/ny;

         const int iface = ix + iy*nx + iz*nx*ny + c*o*o*op1;

         ++i1[c2];
         ++i2[c1];

         const int ie0 = c1*o*op1*op1 + ix + iy*nx_e1 + iz*nx_e1*ny_e1;
         const int ie1 = c1*o*op1*op1 + i1[0] + i1[1]*nx_e1 + i1[2]*nx_e1*ny_e1;

         const int ie2 = c2*o*op1*op1 + ix + iy*nx_e2 + iz*nx_e2*ny_e2;
         const int ie3 = c2*o*op1*op1 + i2[0] + i2[1]*nx_e2 + i2[2]*nx_e2*ny_e2;

         f2e(0, iface) = ie0;
         f2e(1, iface) = ie1;
         f2e(2, iface) = ie2;
         f2e(3, iface) = ie3;
      }
   }
}

void BatchedLOR_ADS::FormCurlMatrix()
{
   // The curl matrix maps from LOR edges to LOR faces. Given a quadrilateral
   // face (defined by its four edges) f_i = (e_j1, e_j2, e_j3, e_j4), the
   // matrix has nonzeros A(i, jk), so there are always exactly four nonzeros
   // per row.
   const int nface_dof = face_fes.GetNDofs();
   const int nedge_dof = edge_fes.GetNDofs();

   SparseMatrix C_local;
   C_local.OverrideSize(nface_dof, nedge_dof);

   C_local.GetMemoryI().New(nedge_dof+1, Device::GetDeviceMemoryType());
   // Each row always has four nonzeros
   const int nnz = 4*nedge_dof;
   auto I = C_local.WriteI();
   mfem::forall(nedge_dof+1, [=] MFEM_HOST_DEVICE (int i) { I[i] = 4*i; });

   // face2edge is a mapping of size (4, nface_per_el), such that with a macro
   // element, face i (in lexicographic ordering) has four edges given by the
   // entries (k, i), for k=1,2,3,4.
   Array<int> face2edge;
   Form3DFaceToEdge(face2edge);

   ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const auto *R_f = dynamic_cast<const ElementRestriction*>(
                        face_fes.GetElementRestriction(ordering));
   const auto *R_e = dynamic_cast<const ElementRestriction*>(
                        edge_fes.GetElementRestriction(ordering));
   MFEM_VERIFY(R_f != NULL && R_e != NULL, "");

   const int nel_ho = edge_fes.GetNE();
   const int nedge_per_el = dim*order*(order+1)*(order+1);
   const int nface_per_el = dim*order*order*(order+1);

   const auto offsets_f = R_f->Offsets().Read();
   const auto indices_f = R_f->Indices().Read();
   const auto gather_e = Reshape(R_e->GatherMap().Read(), nedge_per_el, nel_ho);

   const auto f2e = Reshape(face2edge.Read(), 4, nface_per_el);

   // Fill J and data
   C_local.GetMemoryJ().New(nnz, Device::GetDeviceMemoryType());
   C_local.GetMemoryData().New(nnz, Device::GetDeviceMemoryType());

   auto J = C_local.WriteJ();
   auto V = C_local.WriteData();

   // Loop over Raviart-Thomas L-DOFs
   mfem::forall(nface_dof, [=] MFEM_HOST_DEVICE (int i)
   {
      const int sj = indices_f[offsets_f[i]]; // signed
      const int j = (sj >= 0) ? sj : -1 - sj;
      const int sgn_f = (sj >= 0) ? 1 : -1;
      const int j_loc = j%nface_per_el;
      const int j_el = j/nface_per_el;

      for (int k=0; k<4; ++k)
      {
         const int je_loc = f2e(k, j_loc);
         const int sje = gather_e(je_loc, j_el); // signed
         const int je = (sje >= 0) ? sje : -1 - sje;
         const int sgn_e = (sje >= 0) ? 1 : -1;
         const int sgn = (k == 1 || k == 2) ? -1 : 1;
         J[i*4 + k] = je;
         V[i*4 + k] = sgn*sgn_f*sgn_e;
      }
   });

   // Create a block diagonal parallel matrix
   OperatorHandle C_diag(Operator::Hypre_ParCSR);
   C_diag.MakeRectangularBlockDiag(edge_fes.GetComm(),
                                   face_fes.GlobalVSize(),
                                   edge_fes.GlobalVSize(),
                                   face_fes.GetDofOffsets(),
                                   edge_fes.GetDofOffsets(),
                                   &C_local);

   // Assemble the parallel gradient matrix, must be deleted by the caller
   if (IsIdentityProlongation(edge_fes.GetProlongationMatrix()))
   {
      C = C_diag.As<HypreParMatrix>();
      C_diag.SetOperatorOwner(false);
      HypreStealOwnership(*C, C_local);
   }
   else
   {
      OperatorHandle Rt(Transpose(*face_fes.GetRestrictionMatrix()));
      OperatorHandle Rt_diag(Operator::Hypre_ParCSR);
      Rt_diag.MakeRectangularBlockDiag(face_fes.GetComm(),
                                       face_fes.GlobalVSize(),
                                       face_fes.GlobalTrueVSize(),
                                       face_fes.GetDofOffsets(),
                                       face_fes.GetTrueDofOffsets(),
                                       Rt.As<SparseMatrix>());
      C = RAP(Rt_diag.As<HypreParMatrix>(),
              C_diag.As<HypreParMatrix>(),
              edge_fes.Dof_TrueDof_Matrix());
   }
   C->CopyRowStarts();
   C->CopyColStarts();
}

HypreParMatrix *BatchedLOR_ADS::StealCurlMatrix()
{
   return StealPointer(C);
}

BatchedLOR_ADS::~BatchedLOR_ADS()
{
   delete C;
}

LORSolver<HypreADS>::LORSolver(
   ParBilinearForm &a_ho, const Array<int> &ess_tdof_list, int ref_type)
{
   MFEM_VERIFY(a_ho.FESpace()->GetMesh()->Dimension() == 3,
               "The ADS solver is only valid in 3D.");
   if (BatchedLORAssembly::FormIsSupported(a_ho))
   {
      ParFiniteElementSpace &pfes = *a_ho.ParFESpace();
      BatchedLORAssembly batched_lor(pfes);
      BatchedLOR_ADS lor_ads(pfes, batched_lor.GetLORVertexCoordinates());
      BatchedLOR_AMS &lor_ams = lor_ads.GetAMS();
      batched_lor.Assemble(a_ho, ess_tdof_list, A);
      xyz = lor_ams.StealCoordinateVector();
      solver = new HypreADS(*A.As<HypreParMatrix>(),
                            lor_ads.StealCurlMatrix(),
                            lor_ams.StealGradientMatrix(),
                            lor_ams.StealXCoordinate(),
                            lor_ams.StealYCoordinate(),
                            lor_ams.StealZCoordinate());
   }
   else
   {
      ParLORDiscretization lor(a_ho, ess_tdof_list, ref_type);
      // Assume ownership of the system matrix so that `lor` can be safely
      // deleted
      A.Reset(lor.GetAssembledSystem().Ptr());
      lor.GetAssembledSystem().SetOperatorOwner(false);
      solver = new HypreADS(lor.GetAssembledMatrix(), &lor.GetParFESpace());
   }
   width = solver->Width();
   height = solver->Height();
}

void LORSolver<HypreADS>::SetOperator(const Operator &op)
{
   solver->SetOperator(op);
}

void LORSolver<HypreADS>::Mult(const Vector &x, Vector &y) const
{
   solver->Mult(x, y);
}

HypreADS &LORSolver<HypreADS>::GetSolver() { return *solver; }

const HypreADS &LORSolver<HypreADS>::GetSolver() const { return *solver; }

LORSolver<HypreADS>::~LORSolver()
{
   delete solver;
   delete xyz;
}

#endif

} // namespace mfem
