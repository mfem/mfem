// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "change_basis.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/dtensor.hpp"

namespace mfem
{

/// @brief Compute the inverse of the matrix A and store the result in Ainv.
///
/// The input A is an array of size n*n, interpreted as a matrix with column
/// major ordering.
void ComputeInverse(const Array<real_t> &A, Array<real_t> &Ainv)
{
   Array<real_t> A2 = A;
   const int n2 = A.Size();
   const int n = sqrt(n2);
   Array<int> ipiv(n);
   LUFactors lu(A2.GetData(), ipiv.GetData());
   lu.Factor(n);
   Ainv.SetSize(n2);
   lu.GetInverseMatrix(n, Ainv.GetData());
}

void SubcellIntegrals(int n, const Poly_1D::Basis &basis, Array<real_t> &B)
{
   const IntegrationRule &ir = IntRules.Get(Geometry::SEGMENT, n);
   const real_t *gll_pts = poly1d.GetPoints(n, BasisType::GaussLobatto);
   Vector u(n);
   B.SetSize(n*n);
   B = 0.0;

   for (int i = 0; i < n; ++i)
   {
      const real_t h = gll_pts[i+1] - gll_pts[i];
      // Loop over subcell quadrature points
      for (int iq = 0; iq < ir.Size(); ++iq)
      {
         const IntegrationPoint &ip = ir[iq];
         const real_t x = gll_pts[i] + h*ip.x;
         const real_t w = h*ip.weight;
         basis.Eval(x, u);
         for (int j = 0; j < n; ++j)
         {
            B[i + j*n] += w*u[j];
         }
      }
   }
}

void Transpose(const Array<real_t> &B, Array<real_t> &Bt)
{
   const int n = sqrt(B.Size());
   Bt.SetSize(n*n);
   for (int i=0; i<n; ++i) for (int j=0; j<n; ++j) { Bt[i+j*n] = B[j+i*n]; }
}

ChangeOfBasis_L2::ChangeOfBasis_L2(FiniteElementSpace &fes)
   : Operator(fes.GetTrueVSize()),
     ne(fes.GetNE())
{
   auto *fec1 = dynamic_cast<const L2_FECollection*>(fes.FEColl());
   MFEM_VERIFY(fec1, "Must be L2 finite element space");

   const int btype = fec1->GetBasisType();

   // If the basis types are the same, don't need to perform change of basis.
   no_op = (btype == BasisType::IntegratedGLL);
   if (no_op) { return; }

   // Convert from the given basis to the "integrated GLL basis".
   // The degrees of freedom are integrals over subcells.
   const FiniteElement *fe = fes.GetFE(0);
   auto *tbe = dynamic_cast<const TensorBasisElement*>(fe);
   MFEM_VERIFY(tbe != nullptr, "Must be a tensor element.");
   const Poly_1D::Basis &basis = tbe->GetBasis1D();

   const int p = fes.GetMaxElementOrder();
   const int pp1 = p + 1;

   Array<real_t> B_inv;
   SubcellIntegrals(pp1, basis, B_inv);

   ComputeInverse(B_inv, B_1d);
   Transpose(B_1d, Bt_1d);

   // Set up the DofToQuad object, used in TensorValues
   dof2quad.FE = fe;
   dof2quad.mode = DofToQuad::TENSOR;
   dof2quad.ndof = pp1;
   dof2quad.nqpt = pp1;
}

void ChangeOfBasis_L2::Mult(const Vector &x, Vector &y) const
{
   if (no_op) { y = x; return; }
   dof2quad.B.MakeRef(B_1d);
   const int dim = dof2quad.FE->GetDim();
   const int nd = dof2quad.ndof;
   const int nq = dof2quad.nqpt;
   QuadratureInterpolator::TensorEvalKernels::Run(
      dim, QVectorLayout::byVDIM, 1, nd, nq, ne, dof2quad.B.Read(), x.Read(),
      y.Write(), 1, nd, nq);
}

void ChangeOfBasis_L2::MultTranspose(const Vector &x, Vector &y) const
{
   if (no_op) { y = x; return; }
   dof2quad.B.MakeRef(Bt_1d);
   const int dim = dof2quad.FE->GetDim();
   const int nd = dof2quad.ndof;
   const int nq = dof2quad.nqpt;
   QuadratureInterpolator::TensorEvalKernels::Run(
      dim, QVectorLayout::byVDIM, 1, nd, nq, ne, dof2quad.B.Read(), x.Read(),
      y.Write(), 1, nd, nq);
}

ChangeOfBasis_RT::ChangeOfBasis_RT(FiniteElementSpace &fes)
   : Operator(fes.GetTrueVSize()),
     fes(fes),
     dim(fes.GetMesh()->Dimension()),
     ne(fes.GetNE()),
     p(fes.GetMaxElementOrder())
{
   auto op = fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   elem_restr = dynamic_cast<const ElementRestriction*>(op);
   MFEM_VERIFY(elem_restr != NULL, "Missing element restriction.");

   const auto *rt_fec = dynamic_cast<const RT_FECollection*>(fes.FEColl());
   MFEM_VERIFY(rt_fec, "Must be RT finite element space.");

   const int cb_type = rt_fec->GetClosedBasisType();
   const int ob_type = rt_fec->GetOpenBasisType();

   no_op = (cb_type == BasisType::GaussLobatto &&
            ob_type == BasisType::IntegratedGLL);
   if (no_op) { return; }

   const int pp1 = p + 1;

   Poly_1D::Basis &cbasis = poly1d.GetBasis(p, cb_type);
   Poly_1D::Basis &obasis = poly1d.GetBasis(p-1, ob_type);

   const real_t *cpts2 = poly1d.GetPoints(p, BasisType::GaussLobatto);

   Bci_1d.SetSize(pp1*pp1);
   Vector b(pp1);
   for (int i = 0; i < pp1; ++i)
   {
      cbasis.Eval(cpts2[i], b);
      for (int j = 0; j < pp1; ++j)
      {
         Bci_1d[i + j*pp1] = b[j];
      }
   }
   SubcellIntegrals(p, obasis, Boi_1d);

   ComputeInverse(Boi_1d, Bo_1d);
   Transpose(Bo_1d, Bot_1d);
   ComputeInverse(Bci_1d, Bc_1d);
   Transpose(Bc_1d, Bct_1d);
}

const real_t *ChangeOfBasis_RT::GetOpenMap(Mode mode) const
{
   switch (mode)
   {
      case NORMAL: return Bo_1d.Read();
      case TRANSPOSE: return Bot_1d.Read();
      case INVERSE: return Boi_1d.Read();
   }
   return nullptr;
}

const real_t *ChangeOfBasis_RT::GetClosedMap(Mode mode) const
{
   switch (mode)
   {
      case NORMAL: return Bc_1d.Read();
      case TRANSPOSE: return Bct_1d.Read();
      case INVERSE: return Bci_1d.Read();
   }
   return nullptr;
}

void ChangeOfBasis_RT::MultRT_2D(const Vector &x, Vector &y, Mode mode) const
{
   const int DIM = dim;
   const int NE = ne;
   const int D1D = p + 1;
   const int ND = (p+1)*p;
   const real_t *BC = GetClosedMap(mode);
   const real_t *BO = GetOpenMap(mode);
   const auto X = Reshape(x.Read(), DIM*ND, ne);
   auto Y = Reshape(y.Write(), DIM*ND, ne);

   MFEM_FORALL(e, NE,
   {
      for (int c = 0; c < DIM; ++c)
      {
         const int nx = (c == 0) ? D1D : D1D-1;
         const int ny = (c == 1) ? D1D : D1D-1;
         const real_t *Bx = (c == 0) ? BC : BO;
         const real_t *By = (c == 1) ? BC : BO;

         for (int i = 0; i < ND; ++i)
         {
            Y(i + c*ND, e) = 0.0;
         }
         for (int iy = 0; iy < ny; ++ iy)
         {
            real_t xx[DofQuadLimits::MAX_D1D];
            for (int ix = 0; ix < nx; ++ix) { xx[ix] = 0.0; }
            for (int jx = 0; jx < nx; ++jx)
            {
               const real_t val = X(jx + iy*nx + c*nx*ny, e);
               for (int ix = 0; ix < nx; ++ix)
               {
                  xx[ix] += val*Bx[ix + jx*nx];
               }
            }
            for (int jy = 0; jy < ny; ++jy)
            {
               const real_t b = By[jy + iy*ny];
               for (int ix = 0; ix < nx; ++ix)
               {
                  Y(ix + jy*nx + c*nx*ny, e) += xx[ix]*b;
               }
            }
         }
      }
   });
}

void ChangeOfBasis_RT::MultRT_3D(const Vector &x, Vector &y, Mode mode) const
{
   const int DIM = dim;
   const int NE = ne;
   const int D1D = p + 1;
   const int ND = (p+1)*p*p;
   const real_t *BC = GetClosedMap(mode);
   const real_t *BO = GetOpenMap(mode);
   const auto X = Reshape(x.Read(), DIM*ND, ne);
   auto Y = Reshape(y.Write(), DIM*ND, ne);

   MFEM_FORALL(e, NE,
   {
      for (int c = 0; c < DIM; ++c)
      {
         const int nx = (c == 0) ? D1D : D1D-1;
         const int ny = (c == 1) ? D1D : D1D-1;
         const int nz = (c == 2) ? D1D : D1D-1;
         const real_t *Bx = (c == 0) ? BC : BO;
         const real_t *By = (c == 1) ? BC : BO;
         const real_t *Bz = (c == 2) ? BC : BO;

         for (int i = 0; i < ND; ++i)
         {
            Y(i + c*ND, e) = 0.0;
         }
         for (int iz = 0; iz < nz; ++ iz)
         {
            real_t xy[DofQuadLimits::MAX_D1D][DofQuadLimits::MAX_D1D];
            for (int iy = 0; iy < ny; ++iy)
            {
               for (int ix = 0; ix < nx; ++ix)
               {
                  xy[iy][ix] = 0.0;
               }
            }
            for (int iy = 0; iy < ny; ++iy)
            {
               real_t xx[DofQuadLimits::MAX_D1D];
               for (int ix = 0; ix < nx; ++ix) { xx[ix] = 0.0; }
               for (int ix = 0; ix < nx; ++ix)
               {
                  const real_t val = X(ix + iy*nx + iz*nx*ny + c*ND, e);
                  for (int jx = 0; jx < nx; ++jx)
                  {
                     xx[jx] += val*Bx[jx + ix*nx];
                  }
               }
               for (int jy = 0; jy < ny; ++jy)
               {
                  const real_t b = By[jy + iy*ny];
                  for (int jx = 0; jx < nx; ++jx)
                  {
                     xy[jy][jx] += xx[jx] * b;
                  }
               }
            }
            for (int jz = 0; jz < nz; ++jz)
            {
               const real_t b = Bz[jz + iz*nz];
               for (int jy = 0; jy < ny; ++jy)
               {
                  for (int jx = 0; jx < nx; ++jx)
                  {
                     Y(jx + jy*nx + jz*nx*ny + c*ND, e) += xy[jy][jx] * b;
                  }
               }
            }
         }
      }
   });
}

void ChangeOfBasis_RT::Mult(const Vector &x, Vector &y, Mode mode) const
{
   if (no_op) { y = x; return; }

   const Operator *P = fes.GetProlongationMatrix();

   if (IsIdentityProlongation(P))
   {
      x_l.MakeRef(const_cast<Vector&>(x), 0, fes.GetVSize());
      y_l.MakeRef(y, 0, fes.GetVSize());
   }
   else
   {
      x_l.SetSize(fes.GetVSize());
      y_l.SetSize(fes.GetVSize());
      P->Mult(x, x_l);
   }

   x_e.SetSize(elem_restr->Height());
   y_e.SetSize(elem_restr->Height());

   elem_restr->Mult(x_l, x_e);

   if (dim == 2) { MultRT_2D(x_e, y_e, mode); }
   else { MultRT_3D(x_e, y_e, mode); }

   elem_restr->MultLeftInverse(y_e, y_l);

   const Operator *R = fes.GetRestrictionOperator();
   if (R) { R->Mult(y_l, y); }
   else { MFEM_VERIFY(P == NULL, "Invalid state."); }
}

void ChangeOfBasis_RT::Mult(const Vector &x, Vector &y) const
{
   Mult(x, y, NORMAL);
}

void ChangeOfBasis_RT::MultTranspose(const Vector &x, Vector &y) const
{
   Mult(x, y, TRANSPOSE);
}

void ChangeOfBasis_RT::MultInverse(const Vector &x, Vector &y) const
{
   Mult(x, y, INVERSE);
}

} // namespace mfem
