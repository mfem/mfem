// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "lor_ams.hpp"
#include "../../general/forall.hpp"
#include "../../fem/pbilinearform.hpp"

namespace mfem
{

#ifdef MFEM_USE_MPI

void BatchedLOR_AMS::Form2DEdgeToVertex(DenseMatrix &edge2vert)
{
   const int o = order;
   const int op1 = o + 1;
   const int nedge = dim*o*pow(op1, dim-1);

   edge2vert.SetSize(2, nedge);

   for (int c=0; c<dim; ++c)
   {
      const int nx = (c == 0) ? o : op1;
      for (int i2=0; i2<op1; ++i2)
      {
         for (int i1=0; i1<o; ++i1)
         {
            const int ix = (c == 0) ? i1 : i2;
            const int iy = (c == 0) ? i2 : i1;

            const int iedge = ix + iy*nx + c*o*op1;

            const int ix1 = (c == 0) ? ix + 1 : ix;
            const int iy1 = (c == 1) ? iy + 1 : iy;

            const int iv0 = ix + iy*op1;
            const int iv1 = ix1 + iy1*op1;

            edge2vert(0, iedge) = iv0;
            edge2vert(1, iedge) = iv1;
         }
      }
   }
}

void BatchedLOR_AMS::Form3DEdgeToVertex(DenseMatrix &edge2vert)
{
   const int o = order;
   const int op1 = o + 1;
   const int nedge = dim*o*pow(op1, dim-1);

   edge2vert.SetSize(2, nedge);

   for (int c=0; c<dim; ++c)
   {
      const int nx = (c == 0) ? o : op1;
      const int ny = (c == 1) ? o : op1;
      for (int i=0; i<o*op1*op1; ++i)
      {
         const int ix = i%nx;
         const int iy = (i/nx)%ny;
         const int iz = i/nx/ny;

         const int iedge = ix + iy*nx + iz*nx*ny + c*o*op1*op1;

         const int ix1 = (c == 0) ? ix + 1 : ix;
         const int iy1 = (c == 1) ? iy + 1 : iy;
         const int iz1 = (c == 2) ? iz + 1 : iz;

         const int iv0 = ix + iy*op1 + iz*op1*op1;
         const int iv1 = ix1 + iy1*op1 + iz1*op1*op1;

         edge2vert(0, iedge) = iv0;
         edge2vert(1, iedge) = iv1;
      }
   }
}

void BatchedLOR_AMS::FormGradientMatrix()
{
   // The gradient matrix maps from LOR vertices to LOR edges. Given an edge
   // (defined by its two vertices) e_i = (v_j1, v_j2), the matrix has nonzeros
   // A(i, j1) = -1 and A(i, j2) = 1, so there are always exactly two nonzeros
   // per row.
   const int nedge_dof = fes_ho.GetNDofs();
   const int nvert_dof = vert_fes.GetNDofs();

   SparseMatrix G_local;
   G_local.OverrideSize(nedge_dof, nvert_dof);

   G_local.GetMemoryI().New(nedge_dof+1, Device::GetDeviceMemoryType());
   // Each row always has two nonzeros
   const int nnz = 2*nedge_dof;
   auto I = G_local.WriteI();
   MFEM_FORALL(i, nedge_dof+1, I[i] = 2*i;);

   // edge2vertex is a mapping of size (2, nedge_per_el), such that with a macro
   // element, edge i (in lexicographic ordering) has vertices (also in
   // lexicographic ordering) given by the entries (0, i) and (1, i) of the
   // matrix.
   DenseMatrix edge2vertex;
   if (dim == 2) { Form2DEdgeToVertex(edge2vertex); }
   else { Form3DEdgeToVertex(edge2vertex); }

   ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const auto *R_v = dynamic_cast<const ElementRestriction*>(
                        vert_fes.GetElementRestriction(ordering));
   const auto *R_e = dynamic_cast<const ElementRestriction*>(
                        fes_ho.GetElementRestriction(ordering));
   MFEM_VERIFY(R_v != NULL && R_e != NULL, "");

   const int nel_ho = fes_ho.GetNE();
   const int nedge_per_el = dim*order*pow(order + 1, dim - 1);
   const int nvert_per_el = pow(order + 1, dim);

   const auto offsets_e = R_e->Offsets().Read();
   const auto indices_e = R_e->Indices().Read();
   const auto gather_v = Reshape(R_v->GatherMap().Read(), nvert_per_el, nel_ho);

   const auto e2v = Reshape(edge2vertex.Read(), 2, nedge_per_el);

   // Fill J and data
   G_local.GetMemoryJ().New(nnz, Device::GetDeviceMemoryType());
   G_local.GetMemoryData().New(nnz, Device::GetDeviceMemoryType());

   auto J = G_local.WriteJ();
   auto V = G_local.WriteData();

   // Loop over Nedelec L-DOFs
   MFEM_FORALL(i, nedge_dof,
   {
      const int sj = indices_e[offsets_e[i]]; // signed
      const int j = (sj >= 0) ? sj : -1 - sj;
      const int sgn = (sj >= 0) ? 1 : -1;
      const int j_loc = j%nedge_per_el;
      const int j_el = j/nedge_per_el;

      const int jv0_loc = e2v(0, j_loc);
      const int jv1_loc = e2v(1, j_loc);

      J[i*2 + 0] = gather_v(jv0_loc, j_el);
      J[i*2 + 1] = gather_v(jv1_loc, j_el);

      V[i*2 + 0] = -sgn;
      V[i*2 + 1] = sgn;
   });

   // Create a block diagonal parallel matrix
   OperatorHandle G_diag(Operator::Hypre_ParCSR);
   G_diag.MakeRectangularBlockDiag(vert_fes.GetComm(),
                                   edge_fes.GlobalVSize(),
                                   vert_fes.GlobalVSize(),
                                   edge_fes.GetDofOffsets(),
                                   vert_fes.GetDofOffsets(),
                                   &G_local);

   // Assemble the parallel gradient matrix, must be deleted by the caller
   if (IsIdentityProlongation(vert_fes.GetProlongationMatrix()))
   {
      G = G_diag.As<HypreParMatrix>();
      G_diag.SetOperatorOwner(false);
      HypreStealOwnership(*G, G_local);
   }
   else
   {
      OperatorHandle Rt(Transpose(*edge_fes.GetRestrictionMatrix()));
      OperatorHandle Rt_diag(Operator::Hypre_ParCSR);
      Rt_diag.MakeRectangularBlockDiag(edge_fes.GetComm(),
                                       edge_fes.GlobalVSize(),
                                       edge_fes.GlobalTrueVSize(),
                                       edge_fes.GetDofOffsets(),
                                       edge_fes.GetTrueDofOffsets(),
                                       Rt.As<SparseMatrix>());
      G = RAP(Rt_diag.As<HypreParMatrix>(),
              G_diag.As<HypreParMatrix>(),
              vert_fes.Dof_TrueDof_Matrix());
   }
   G->CopyRowStarts();
   G->CopyColStarts();
}

void BatchedLOR_AMS::FormCoordinateVectors()
{
   // Create true-DOF vectors x, y, and z that contain the coordinates of the
   // vertices of the LOR mesh. The vertex coordinates are already computed in
   // E-vector format (and stored in X_vert) in the function (called already)
   // BatchedLORAssembly::GetLORVertexCoordinates.
   //
   // In this function, we just need to convert X_vert (which has the shape
   // (dim, ndof_per_el, nel_ho)) to T-DOF format.
   //
   // We place the results in the vector xyz_tvec, which has shape (ntdofs, dim)
   // and then make the hypre vectors x, y, and z point to subvectors.
   //
   // In 2D, z is simply NULL.

   // Create the H1 vertex space and get the element restriction
   ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *op = vert_fes.GetElementRestriction(ordering);
   const auto *el_restr = dynamic_cast<const ElementRestriction*>(op);
   MFEM_VERIFY(el_restr != NULL, "");
   const SparseMatrix *R = vert_fes.GetRestrictionMatrix();

   const int nel_ho = vert_fes.GetNE();
   const int order = vert_fes.GetMaxElementOrder();
   const int ndp1 = order + 1;
   const int ndof_per_el = pow(ndp1, dim);
   const int sdim = dim;
   const int ntdofs = R->Height();

   xyz_tvec = new Vector(ntdofs*dim);

   auto xyz_tv = Reshape(xyz_tvec->Write(), ntdofs, dim);
   const auto xyz_e = Reshape(X_vert.Read(), dim, ndof_per_el, nel_ho);
   const auto d_offsets = el_restr->Offsets().Read();
   const auto d_indices = el_restr->Indices().Read();
   const auto ltdof_ldof = R->ReadJ();

   // Go from E-vector format directly to T-vector format
   MFEM_FORALL(i, ntdofs,
   {
      const int j = d_offsets[ltdof_ldof[i]];
      for (int c = 0; c < sdim; ++c)
      {
         const int idx_j = d_indices[j];
         xyz_tv(i,c) = xyz_e(c, idx_j%ndof_per_el, idx_j/ndof_per_el);
      }
   });

   // Make x, y, z HypreParVectors point to T-vector data
   HYPRE_BigInt glob_size = vert_fes.GlobalTrueVSize();
   HYPRE_BigInt *cols = vert_fes.GetTrueDofOffsets();

   bool dev = Device::GetDeviceMemoryClass() == MemoryClass::DEVICE;

   double *d_x_ptr = xyz_tv + 0*ntdofs;
   x = new HypreParVector(vert_fes.GetComm(), glob_size, d_x_ptr, cols, dev);
   double *d_y_ptr = xyz_tv + 1*ntdofs;
   y = new HypreParVector(vert_fes.GetComm(), glob_size, d_y_ptr, cols, dev);
   if (dim == 3)
   {
      double *d_z_ptr = xyz_tv + 2*ntdofs;
      z = new HypreParVector(vert_fes.GetComm(), glob_size, d_z_ptr, cols, dev);
   }
   else
   {
      z = NULL;
   }
}

template <typename T> T *StealPointer(T *&ptr)
{
   T *tmp = ptr;
   ptr = nullptr;
   return tmp;
}

HypreParMatrix *BatchedLOR_AMS::StealGradientMatrix()
{
   return StealPointer(G);
};
Vector *BatchedLOR_AMS::StealCoordinateVector()
{
   return StealPointer(xyz_tvec);
};
HypreParVector *BatchedLOR_AMS::StealXCoordinate()
{
   return StealPointer(x);
};
HypreParVector *BatchedLOR_AMS::StealYCoordinate()
{
   return StealPointer(y);
};
HypreParVector *BatchedLOR_AMS::StealZCoordinate()
{
   return StealPointer(z);
};

BatchedLOR_AMS::~BatchedLOR_AMS()
{
   delete x;
   delete y;
   delete z;
   delete xyz_tvec;
   delete G;
}

BatchedLOR_AMS::BatchedLOR_AMS(BilinearForm &a_,
                               ParFiniteElementSpace &pfes_ho_,
                               const Array<int> &ess_dofs_)
   : BatchedLOR_ND(a_, pfes_ho_, ess_dofs_),
     edge_fes(pfes_ho_),
     dim(edge_fes.GetParMesh()->Dimension()),
     order(edge_fes.GetMaxElementOrder()),
     vert_fec(order, dim),
     vert_fes(edge_fes.GetParMesh(), &vert_fec)
{
   // Form the coordinate vectors (uses X_vert) and gradient matrix
   FormCoordinateVectors();
   FormGradientMatrix();
}

LORSolver<HypreAMS>::LORSolver(
   ParBilinearForm &a_ho, const Array<int> &ess_tdof_list, int ref_type)
{
   if (BatchedLORAssembly::FormIsSupported(a_ho))
   {
      ParFiniteElementSpace &pfes = *a_ho.ParFESpace();
      BatchedLOR_AMS batched_lor(a_ho, pfes, ess_tdof_list);
      batched_lor.Assemble(A);
      xyz = batched_lor.StealCoordinateVector();
      solver = new HypreAMS(*A.As<HypreParMatrix>(),
                            batched_lor.StealGradientMatrix(),
                            batched_lor.StealXCoordinate(),
                            batched_lor.StealYCoordinate(),
                            batched_lor.StealZCoordinate());
   }
   else
   {
      ParLORDiscretization lor(a_ho, ess_tdof_list, ref_type);
      // Assume ownership of the system matrix so that `lor` can be safely
      // deleted
      A.Reset(lor.GetAssembledSystem().Ptr());
      lor.GetAssembledSystem().SetOperatorOwner(false);
      solver = new HypreAMS(lor.GetAssembledMatrix(), &lor.GetParFESpace());
   }
   width = solver->Width();
   height = solver->Height();
}

void LORSolver<HypreAMS>::SetOperator(const Operator &op)
{
   solver->SetOperator(op);
}

void LORSolver<HypreAMS>::Mult(const Vector &x, Vector &y) const
{
   solver->Mult(x, y);
}

HypreAMS &LORSolver<HypreAMS>::GetSolver() { return *solver; }

const HypreAMS &LORSolver<HypreAMS>::GetSolver() const { return *solver; }

LORSolver<HypreAMS>::~LORSolver()
{
   delete solver;
   delete xyz;
}

#endif

} // namespace mfem
