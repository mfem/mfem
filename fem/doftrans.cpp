// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "fem.hpp"

namespace mfem
{

void DofTransformation::TransformPrimal(real_t *v) const
{
   if (IsIdentity()) { return; }
   int size = dof_trans_->Size();

   if (vdim_ == 1 || (Ordering::Type)ordering_ == Ordering::byNODES)
   {
      for (int i=0; i<vdim_; i++)
      {
         dof_trans_->TransformPrimal(Fo_, &v[i*size]);
      }
   }
   else
   {
      Vector vec(size);
      for (int i=0; i<vdim_; i++)
      {
         for (int j=0; j<size; j++)
         {
            vec(j) = v[j*vdim_+i];
         }
         dof_trans_->TransformPrimal(Fo_, vec);
         for (int j=0; j<size; j++)
         {
            v[j*vdim_+i] = vec(j);
         }
      }
   }
}

void DofTransformation::InvTransformPrimal(real_t *v) const
{
   if (IsIdentity()) { return; }
   int size = dof_trans_->Height();

   if (vdim_ == 1 || (Ordering::Type)ordering_ == Ordering::byNODES)
   {
      for (int i=0; i<vdim_; i++)
      {
         dof_trans_->InvTransformPrimal(Fo_, &v[i*size]);
      }
   }
   else
   {
      Vector vec(size);
      for (int i=0; i<vdim_; i++)
      {
         for (int j=0; j<size; j++)
         {
            vec(j) = v[j*vdim_+i];
         }
         dof_trans_->InvTransformPrimal(Fo_, vec);
         for (int j=0; j<size; j++)
         {
            v[j*vdim_+i] = vec(j);
         }
      }
   }
}

void DofTransformation::TransformDual(real_t *v) const
{
   if (IsIdentity()) { return; }
   int size = dof_trans_->Size();

   if (vdim_ == 1 || (Ordering::Type)ordering_ == Ordering::byNODES)
   {
      for (int i=0; i<vdim_; i++)
      {
         dof_trans_->TransformDual(Fo_, &v[i*size]);
      }
   }
   else
   {
      Vector vec(size);
      for (int i=0; i<vdim_; i++)
      {
         for (int j=0; j<size; j++)
         {
            vec(j) = v[j*vdim_+i];
         }
         dof_trans_->TransformDual(Fo_, vec);
         for (int j=0; j<size; j++)
         {
            v[j*vdim_+i] = vec(j);
         }
      }
   }
}

void DofTransformation::InvTransformDual(real_t *v) const
{
   if (IsIdentity()) { return; }
   int size = dof_trans_->Size();

   if (vdim_ == 1 || (Ordering::Type)ordering_ == Ordering::byNODES)
   {
      for (int i=0; i<vdim_; i++)
      {
         dof_trans_->InvTransformDual(Fo_, &v[i*size]);
      }
   }
   else
   {
      Vector vec(size);
      for (int i=0; i<vdim_; i++)
      {
         for (int j=0; j<size; j++)
         {
            vec(j) = v[j*vdim_+i];
         }
         dof_trans_->InvTransformDual(Fo_, vec);
         for (int j=0; j<size; j++)
         {
            v[j*vdim_+i] = vec(j);
         }
      }
   }
}

void TransformPrimal(const DofTransformation &ran_dof_trans,
                     const DofTransformation &dom_dof_trans,
                     DenseMatrix &elmat)
{
   // No action if both transformations are NULL
   if (!ran_dof_trans.IsIdentity())
   {
      ran_dof_trans.TransformPrimalCols(elmat);
   }
   if (!dom_dof_trans.IsIdentity())
   {
      dom_dof_trans.TransformDualRows(elmat);
   }
}

void TransformDual(const DofTransformation &ran_dof_trans,
                   const DofTransformation &dom_dof_trans,
                   DenseMatrix &elmat)
{
   // No action if both transformations are NULL
   if (!ran_dof_trans.IsIdentity())
   {
      ran_dof_trans.TransformDualCols(elmat);
   }
   if (!dom_dof_trans.IsIdentity())
   {
      dom_dof_trans.TransformDualRows(elmat);
   }
}

// ordering (i0j0, i1j0, i0j1, i1j1), each row is a column major matrix
const real_t ND_DofTransformation::T_data[24] =
{
   1.0,  0.0,  0.0,  1.0,
   -1.0, -1.0,  0.0,  1.0,
   0.0,  1.0, -1.0, -1.0,
   1.0,  0.0, -1.0, -1.0,
   -1.0, -1.0,  1.0,  0.0,
   0.0,  1.0,  1.0,  0.0
};

const DenseTensor ND_DofTransformation
::T(const_cast<real_t *>(ND_DofTransformation::T_data), 2, 2, 6);

// ordering (i0j0, i1j0, i0j1, i1j1), each row is a column major matrix
const real_t ND_DofTransformation::TInv_data[24] =
{
   1.0,  0.0,  0.0,  1.0,
   -1.0, -1.0,  0.0,  1.0,
   -1.0, -1.0,  1.0,  0.0,
   1.0,  0.0, -1.0, -1.0,
   0.0,  1.0, -1.0, -1.0,
   0.0,  1.0,  1.0,  0.0
};

const DenseTensor ND_DofTransformation
::TInv(const_cast<real_t *>(TInv_data), 2, 2, 6);

ND_DofTransformation::ND_DofTransformation(int size, int p, int num_edges,
                                           int num_faces,
                                           int face_types[])
   : StatelessDofTransformation(size)
   , order(p)
   , nedofs(p)
   , ntdofs(p*(p-1))
   , nqdofs(2*p*(p-1))
   , nedges(num_edges)
   , nfaces(num_faces)
   , ftypes(face_types)
{
}

void ND_DofTransformation::TransformPrimal(const Array<int> & Fo,
                                           real_t *v) const
{
   // Return immediately when no face DoFs are present
   if (IsIdentity()) { return; }

   MFEM_VERIFY(Fo.Size() >= nfaces,
               "Face orientation array is shorter than the number of faces in "
               "ND_DofTransformation");

   int of = 0;
   real_t data[2];
   Vector v2(data, 2);
   DenseMatrix T2;

   // Transform face DoFs
   for (int f=0; f<nfaces; f++)
   {
      if (ftypes[f] == Geometry::TRIANGLE)
      {
         for (int i=0; i<ntdofs/2; i++)
         {
            v2 = &v[nedges*nedofs + of + 2*i];
            T2.UseExternalData(const_cast<real_t *>(T.GetData(Fo[f])), 2, 2);
            T2.Mult(v2, &v[nedges*nedofs + of + 2*i]);
         }
         of += ntdofs;
      }
      else
      {
         of += nqdofs;
      }
   }
}

void ND_DofTransformation::InvTransformPrimal(const Array<int> & Fo,
                                              real_t *v) const
{
   // Return immediately when no face DoFs are present
   if (IsIdentity()) { return; }

   MFEM_VERIFY(Fo.Size() >= nfaces,
               "Face orientation array is shorter than the number of faces in "
               "ND_DofTransformation");

   int of = 0;
   real_t data[2];
   Vector v2(data, 2);
   DenseMatrix T2Inv;

   // Transform face DoFs
   for (int f=0; f<nfaces; f++)
   {
      if (ftypes[f] == Geometry::TRIANGLE)
      {
         for (int i=0; i<ntdofs/2; i++)
         {
            v2 = &v[nedges*nedofs + of + 2*i];
            T2Inv.UseExternalData(const_cast<real_t *>(TInv.GetData(Fo[f])), 2, 2);
            T2Inv.Mult(v2, &v[nedges*nedofs + of + 2*i]);
         }
         of += ntdofs;
      }
      else
      {
         of += nqdofs;
      }
   }
}

void ND_DofTransformation::TransformDual(const Array<int> & Fo, real_t *v) const
{
   // Return immediately when no face DoFs are present
   if (IsIdentity()) { return; }

   MFEM_VERIFY(Fo.Size() >= nfaces,
               "Face orientation array is shorter than the number of faces in "
               "ND_DofTransformation");

   int of = 0;
   real_t data[2];
   Vector v2(data, 2);
   DenseMatrix T2Inv;

   // Transform face DoFs
   for (int f=0; f<nfaces; f++)
   {
      if (ftypes[f] == Geometry::TRIANGLE)
      {
         for (int i=0; i<ntdofs/2; i++)
         {
            v2 = &v[nedges*nedofs + of + 2*i];
            T2Inv.UseExternalData(const_cast<real_t *>(TInv.GetData(Fo[f])), 2, 2);
            T2Inv.MultTranspose(v2, &v[nedges*nedofs + of + 2*i]);
         }
         of += ntdofs;
      }
      else
      {
         of += nqdofs;
      }

   }
}

void ND_DofTransformation::InvTransformDual(const Array<int> & Fo,
                                            real_t *v) const
{
   // Return immediately when no face DoFs are present
   if (IsIdentity()) { return; }

   MFEM_VERIFY(Fo.Size() >= nfaces,
               "Face orientation array is shorter than the number of faces in "
               "ND_DofTransformation");

   int of = 0;
   real_t data[2];
   Vector v2(data, 2);
   DenseMatrix T2;

   // Transform face DoFs
   for (int f=0; f<nfaces; f++)
   {
      if (ftypes[f] == Geometry::TRIANGLE)
      {
         for (int i=0; i<ntdofs/2; i++)
         {
            v2 = &v[nedges*nedofs + of + 2*i];
            T2.UseExternalData(const_cast<real_t *>(T.GetData(Fo[f])), 2, 2);
            T2.MultTranspose(v2, &v[nedges*nedofs + of + 2*i]);
         }
         of += ntdofs;
      }
      else
      {
         of += nqdofs;
      }
   }
}

namespace
{

int TriBernsteinMultiindexTo1D(int p, const int alpha[3])
{
   MFEM_ASSERT(p >= 0 && alpha[0] >= 0 && alpha[1] >= 0 &&
               alpha[2] >= 0 && alpha[0] + alpha[1] + alpha[2] == p,
               "Invalid triangular Bernstein multi-index.");
   const int i = alpha[1];
   const int j = alpha[2];
   const int p2p3 = 2*p + 3;
   return ((p2p3 - j)*j)/2 + i;
}

/** Given the scalar Bernstein basis functions in the triangular face,
    construct the H(curl) face shape functions {v_F} supported on the face,
    and use them to evaluate the functional L_{q,a}(v_F) := v_F(xm) * t_a.
    The resulting values are stored in the specified row of the shape matrix,
    with pairs of rows corresponding to the domain point (a0/(p+1), a1/(p+1))
    and unit directions (1,0) and (0,1). */
void NDPosFillFaceShapeRow(int order, const real_t *bernstein,
                           const real_t grad[3][2], int tangent,
                           DenseMatrix &shape, int row)
{
   int n = 0;
   // face-supported gradient fields (Type 2 basis functions)
   for (int a0 = 1; a0 <= order - 2; a0++)
   {
      for (int a1 = 1; a1 <= order - a0 - 1; a1++)
      {
         int alpha[3] = {a0, a1, order - a0 - a1};
         real_t value = 0.0;
         for (int i = 0; i < 3; i++)
         {
            for (int l = 0; l < 3; l++)
            {
               int beta[3] = {alpha[0], alpha[1], alpha[2]};
               beta[i]--; beta[l]++;
               const real_t coefficient = alpha[l] - (i == l) + 1;
               value += coefficient*
                        bernstein[TriBernsteinMultiindexTo1D(order, beta)]*
                        grad[i][tangent];
            }
         }
         shape(row,n++) = value;
      }
   }

   // non-gradient face bubbles (Type 3 basis functions). The dependent
   // multiindex (order-1)*e_2 is the excluded multiindex.
   const int p = order - 1;
   for (int a0 = 0; a0 <= p; a0++)
   {
      for (int a1 = 0; a1 <= p - a0; a1++)
      {
         int alpha[3] = {a0, a1, p - a0 - a1};
         if (alpha[2] == p) { continue; }
         real_t value = 0.0;
         for (int s = 0; s < 3; s++)
         {
            const int i = s;
            const int j = (s + 1)%3;
            const int k = (s + 2)%3;
            int beta[3] = {alpha[0], alpha[1], alpha[2]};
            beta[i]++;
            value += (alpha[i] + 1)*
                     bernstein[TriBernsteinMultiindexTo1D(order, beta)]*
                     (alpha[k]*grad[j][tangent] -
                      alpha[j]*grad[k][tangent]);
         }
         shape(row,n++) = value;
      }
   }
   MFEM_ASSERT(n == order*(order - 1),
               "Incorrect Bernstein H(curl) face basis size.");
}

/** returns the transformation matrix T such that
       \phi^oriented_j = \sum_i T_{ij} \phi^canonical_i,
    where \phi^canonical_i are the canonical basis functions and
   \phi^oriented_j are the basis functions with the given orientation.

   Let L_{q,a}(v) := v(xm) * t_a, where t_a is the unit basis vector in direction a.
   Define C to be the matrix with entries C_{(q,a),(i)} = L_{q,a}(\phi^canonical_i)
   and O to be the matrix with entries O_{(q,a),(j)} = L_{q,a}(\phi^oriented_j).
   Writing
      L_{q,a}(\phi^oriented_j) = \sum_i T_{ij} L_{q,a}(\phi^canonical_i),
   we have
      O = C * T,
   so T = C^{-1} * O. */
void NDPosBuildFaceTransform(int order, int orientation,
                             DenseMatrix &transform)
{
   const int nfdofs = order*(order - 1);
   MFEM_ASSERT(transform.Height() == nfdofs && transform.Width() == nfdofs,
               "Incorrect NDPos face transformation size.");
   if (nfdofs == 0) { return; }

   DenseMatrix canonical(nfdofs), oriented(nfdofs);
   const int scalar_dof = (order + 1)*(order + 2)/2;
   Vector canonical_bernstein(scalar_dof), oriented_bernstein(scalar_dof);
   const real_t canonical_grad[3][2] = {{-1., -1.}, {1., 0.}, {0., 1.}};
   const int inv_orientation =
      Geometry::Constants<Geometry::TRIANGLE>::InvOrient[orientation];
   const int *inv_perm = Geometry::Constants<Geometry::TRIANGLE>::
                         Orient[inv_orientation];

   int row = 0;
   const int denominator = order + 1;
   for (int a0 = 1; a0 <= denominator - 2; a0++)
   {
      for (int a1 = 1; a1 <= denominator - a0 - 1; a1++)
      {
         // use the interior domain points of degree p+1 to
         // evaluate the shape functions
         real_t lambda[3] =
         {
            real_t(a0)/denominator,
            real_t(a1)/denominator,
            real_t(denominator - a0 - a1)/denominator
         };
         real_t local_lambda[3], local_grad[3][2];
         for (int k = 0; k < 3; k++)
         {
            local_lambda[k] = lambda[inv_perm[k]];
            local_grad[k][0] = canonical_grad[inv_perm[k]][0];
            local_grad[k][1] = canonical_grad[inv_perm[k]][1];
         }

         H1Pos_TriangleElement::CalcShape(order, lambda[1], lambda[2],
                                          canonical_bernstein.GetData());
         H1Pos_TriangleElement::CalcShape(order, local_lambda[1],
                                          local_lambda[2],
                                          oriented_bernstein.GetData());

         for (int tangent = 0; tangent < 2; tangent++, row++)
         {
            NDPosFillFaceShapeRow(order, canonical_bernstein.GetData(),
                                  canonical_grad, tangent, canonical, row);
            NDPosFillFaceShapeRow(order, oriented_bernstein.GetData(),
                                  local_grad, tangent, oriented, row);
         }
      }
   }
   MFEM_ASSERT(row == nfdofs, "Incorrect face sampling point count.");
   DenseMatrixInverse inverse(canonical);
   inverse.Mult(oriented, transform);
}

void NDPosApplyFaceMatrices(const DenseTensor &matrices,
                            int order, int nedges, int nfaces,
                            const Array<int> &Fo, real_t *v,
                            bool transpose)
{
   const int nfdofs = order*(order - 1);
   if (nfdofs == 0) { return; }
   MFEM_VERIFY(Fo.Size() >= nfaces,
               "Face orientation array is too short for NDPos transformation.");
   Vector output(nfdofs);
   DenseMatrix matrix;
   for (int f = 0; f < nfaces; f++)
   {
      real_t *face_data = v + nedges*order + f*nfdofs;
      Vector input(face_data, nfdofs);
      const DenseMatrix &face_matrix = matrices(Fo[f], matrix);
      if (transpose) { face_matrix.MultTranspose(input, output); }
      else { face_matrix.Mult(input, output); }
      for (int i = 0; i < nfdofs; i++) { face_data[i] = output[i]; }
   }
}

} // anonymous namespace

NDPos_DofTransformation::NDPos_DofTransformation(
   int size, int p, int num_edges, int num_faces)
   : StatelessDofTransformation(size), order(p), nedges(num_edges),
     nfaces(num_faces), nfdofs(p*(p - 1)),
     T(nfdofs, nfdofs, 6), TInv(nfdofs, nfdofs, 6)
{
   if (nfdofs == 0) { return; }

   DenseMatrix transform, inverse_transform;
   T(0, transform) = 0.0;
   TInv(0, inverse_transform) = 0.0;
   for (int i = 0; i < nfdofs; i++)
   {
      transform(i,i) = 1.0;
      inverse_transform(i,i) = 1.0;
   }

   for (int ori = 1; ori < 6; ori++)
   {
      NDPosBuildFaceTransform(order, ori, T(ori, transform));
      DenseMatrixInverse inverse(transform);
      inverse.GetInverseMatrix(TInv(ori, inverse_transform));
   }
}

void NDPos_DofTransformation::TransformPrimal(const Array<int> &Fo,
                                              real_t *v) const
{
   NDPosApplyFaceMatrices(T, order, nedges, nfaces, Fo, v, false);
}

void NDPos_DofTransformation::InvTransformPrimal(const Array<int> &Fo,
                                                 real_t *v) const
{
   NDPosApplyFaceMatrices(TInv, order, nedges, nfaces, Fo, v, false);
}

void NDPos_DofTransformation::TransformDual(const Array<int> &Fo,
                                            real_t *v) const
{
   NDPosApplyFaceMatrices(TInv, order, nedges, nfaces, Fo, v, true);
}

void NDPos_DofTransformation::InvTransformDual(const Array<int> &Fo,
                                               real_t *v) const
{
   NDPosApplyFaceMatrices(T, order, nedges, nfaces, Fo, v, true);
}

} // namespace mfem
