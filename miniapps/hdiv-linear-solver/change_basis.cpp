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

#include "change_basis.hpp"
#include "fem/qinterp/dispatch.hpp"
#include "general/forall.hpp"
#include "linalg/dtensor.hpp"

namespace mfem
{

ChangeOfBasis_L2::ChangeOfBasis_L2(FiniteElementSpace &fes1,
                                   FiniteElementSpace &fes2)
   : Operator(fes1.GetTrueVSize()),
     ne(fes1.GetNE())
{
   auto *fec1 = dynamic_cast<const L2_FECollection*>(fes1.FEColl());
   auto *fec2 = dynamic_cast<const L2_FECollection*>(fes2.FEColl());
   MFEM_VERIFY(fec1 && fec2, "Must be L2 finite element space");
   int btype1 = fec1->GetBasisType();
   int btype2 = fec2->GetBasisType();

   // If the basis types are the same, don't need to perform change of basis.
   no_op = (btype1 == btype2);
   if (no_op) { return; }

   BasisType::CheckNodal(btype1);

   const IntegrationRule &ir = fes1.GetFE(0)->GetNodes();
   const auto mode = DofToQuad::TENSOR;

   // NOTE: this assumes that fes1 uses a *nodal basis*
   // This creates a *copy* of dof2quad.
   dof2quad = fes2.GetFE(0)->GetDofToQuad(ir, mode);

   // Make copies of the 1D matrices.
   B_1d = dof2quad.B;
   Bt_1d = dof2quad.Bt;
}

void ChangeOfBasis_L2::Mult(const Vector &x, Vector &y) const
{
   if (no_op) { y = x; return; }
   using namespace internal::quadrature_interpolator;
   dof2quad.B.MakeRef(B_1d);
   TensorValues<QVectorLayout::byVDIM>(ne, 1, dof2quad, x, y);
}

void ChangeOfBasis_L2::MultTranspose(const Vector &x, Vector &y) const
{
   if (no_op) { y = x; return; }
   using namespace internal::quadrature_interpolator;
   dof2quad.B.MakeRef(Bt_1d);
   TensorValues<QVectorLayout::byVDIM>(ne, 1, dof2quad, x, y);
}

ChangeOfBasis_RT::ChangeOfBasis_RT(FiniteElementSpace &fes1,
                                   FiniteElementSpace &fes2)
   : Operator(fes1.GetTrueVSize()),
     fes(fes1),
     dim(fes1.GetMesh()->Dimension()),
     ne(fes1.GetNE()),
     p(fes1.GetMaxElementOrder())
{
   auto op = fes1.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   elem_restr = dynamic_cast<const ElementRestriction*>(op);
   MFEM_VERIFY(elem_restr != NULL, "Missing element restriciton.");

   const auto *rt_fec1 = dynamic_cast<const RT_FECollection*>(fes1.FEColl());
   const auto *rt_fec2 = dynamic_cast<const RT_FECollection*>(fes2.FEColl());
   MFEM_VERIFY(rt_fec1 && rt_fec2, "Must be RT finite element space.");

   const int cb_type1 = rt_fec1->GetClosedBasisType();
   const int ob_type1 = rt_fec1->GetOpenBasisType();

   const int cb_type2 = rt_fec2->GetClosedBasisType();
   const int ob_type2 = rt_fec2->GetOpenBasisType();

   no_op = (cb_type1 == cb_type2 && ob_type1 == ob_type2);
   if (no_op) { return; }

   const int pp1 = p + 1;

   const double *cpts1 = poly1d.GetPoints(p, cb_type1);
   const double *opts1 = poly1d.GetPoints(p - 1, ob_type1);

   const auto &cb2 = poly1d.GetBasis(p, BasisType::GaussLobatto);
   auto &ob2 = poly1d.GetBasis(p - 1, BasisType::IntegratedGLL);
   ob2.ScaleIntegrated(false);

   Vector b;

   // Evaluate cb2 at cb1
   Bc_1d.SetSize(pp1*pp1);
   Bct_1d.SetSize(pp1*pp1);
   b.SetSize(pp1);
   for (int i = 0; i < pp1; ++i)
   {
      cb2.Eval(cpts1[i], b);
      for (int j = 0; j < pp1; ++j)
      {
         Bc_1d[i + j*pp1] = b[j];
         Bct_1d[j + i*pp1] = b[j];
      }
   }

   // Evaluate ob2 at ob1
   Bo_1d.SetSize(p*p);
   Bot_1d.SetSize(p*p);
   b.SetSize(p);
   for (int i = 0; i < p; ++i)
   {
      ob2.Eval(opts1[i], b);
      for (int j = 0; j < p; ++j)
      {
         Bo_1d[i + j*p] = b[j];
         Bot_1d[j + i*p] = b[j];
      }
   }

   auto compute_inverse = [](const Array<double> &A, Array<double> &Ainv)
   {
      Array<double> A2 = A;
      const int n2 = A.Size();
      const int n = sqrt(n2);
      Array<int> ipiv(n);
      LUFactors lu(A2.GetData(), ipiv.GetData());
      lu.Factor(n);
      Ainv.SetSize(n2);
      lu.GetInverseMatrix(n, Ainv.GetData());
   };

   compute_inverse(Bo_1d, Boi_1d);
   compute_inverse(Bc_1d, Bci_1d);
}

const double *ChangeOfBasis_RT::GetOpenMap(Mode mode) const
{
   switch (mode)
   {
      case NORMAL: return Bo_1d.Read();
      case TRANSPOSE: return Bot_1d.Read();
      case INVERSE: return Boi_1d.Read();
   }
   return nullptr;
}

const double *ChangeOfBasis_RT::GetClosedMap(Mode mode) const
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
   const double *BC = GetClosedMap(mode);
   const double *BO = GetOpenMap(mode);
   const auto X = Reshape(x.Read(), DIM*ND, ne);
   auto Y = Reshape(y.Write(), DIM*ND, ne);

   MFEM_FORALL(e, NE,
   {
      for (int c = 0; c < DIM; ++c)
      {
         const int nx = (c == 0) ? D1D : D1D-1;
         const int ny = (c == 1) ? D1D : D1D-1;
         const double *Bx = (c == 0) ? BC : BO;
         const double *By = (c == 1) ? BC : BO;

         for (int i = 0; i < ND; ++i)
         {
            Y(i + c*ND, e) = 0.0;
         }
         for (int iy = 0; iy < ny; ++ iy)
         {
            double xx[MAX_D1D];
            for (int ix = 0; ix < nx; ++ix) { xx[ix] = 0.0; }
            for (int jx = 0; jx < nx; ++jx)
            {
               const double val = X(jx + iy*nx + c*nx*ny, e);
               for (int ix = 0; ix < nx; ++ix)
               {
                  xx[ix] += val*Bx[ix + jx*nx];
               }
            }
            for (int jy = 0; jy < ny; ++jy)
            {
               const double b = By[jy + iy*ny];
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
   const double *BC = GetClosedMap(mode);
   const double *BO = GetOpenMap(mode);
   const auto X = Reshape(x.Read(), DIM*ND, ne);
   auto Y = Reshape(y.Write(), DIM*ND, ne);

   MFEM_FORALL(e, NE,
   {
      for (int c = 0; c < DIM; ++c)
      {
         const int nx = (c == 0) ? D1D : D1D-1;
         const int ny = (c == 1) ? D1D : D1D-1;
         const int nz = (c == 2) ? D1D : D1D-1;
         const double *Bx = (c == 0) ? BC : BO;
         const double *By = (c == 1) ? BC : BO;
         const double *Bz = (c == 2) ? BC : BO;

         for (int i = 0; i < ND; ++i)
         {
            Y(i + c*ND, e) = 0.0;
         }
         for (int iz = 0; iz < nz; ++ iz)
         {
            double xy[MAX_D1D][MAX_D1D];
            for (int iy = 0; iy < ny; ++iy)
            {
               for (int ix = 0; ix < nx; ++ix)
               {
                  xy[iy][ix] = 0.0;
               }
            }
            for (int iy = 0; iy < ny; ++iy)
            {
               double xx[MAX_D1D];
               for (int ix = 0; ix < nx; ++ix) { xx[ix] = 0.0; }
               for (int ix = 0; ix < nx; ++ix)
               {
                  const double val = X(ix + iy*nx + iz*nx*ny + c*ND, e);
                  for (int jx = 0; jx < nx; ++jx)
                  {
                     xx[jx] += val*Bx[jx + ix*nx];
                  }
               }
               for (int jy = 0; jy < ny; ++jy)
               {
                  const double b = By[jy + iy*ny];
                  for (int jx = 0; jx < nx; ++jx)
                  {
                     xy[jy][jx] += xx[jx] * b;
                  }
               }
            }
            for (int jz = 0; jz < nz; ++jz)
            {
               const double b = Bz[jz + iz*nz];
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

   if (P)
   {
      x_l.SetSize(fes.GetVSize());
      y_l.SetSize(fes.GetVSize());
      P->Mult(x, x_l);
   }
   else
   {
      x_l.MakeRef(const_cast<Vector&>(x), 0);
      y_l.MakeRef(y, 0);
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
