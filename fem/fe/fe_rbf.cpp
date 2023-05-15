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

// Radial basis function classes

#include "fe_rbf.hpp"
#include "../bilininteg.hpp"
#include "../lininteg.hpp"
#include "../coefficient.hpp"

namespace mfem
{

const double RBFFunction::GlobalRadius = 1.e10;

const double GaussianRBF::hNorm = 0.37619048746124223;

double GaussianRBF::BaseFunction(double r) const
{
   return exp(-r * r);
}

double GaussianRBF::BaseDerivative(double r) const
{
   return -2.0 * r * exp(-r * r);
}

double GaussianRBF::BaseDerivative2(double r) const
{
   const double r2 = r * r;
   return (-2 * 4 * r2) * exp(-r2);
}

const double MultiquadricRBF::hNorm = 0.17889;

double MultiquadricRBF::BaseFunction(double r) const
{
   return sqrt(1 + r * r);
}

double MultiquadricRBF::BaseDerivative(double r) const
{
   const double f = BaseFunction(r);
   return r / f;
}

double MultiquadricRBF::BaseDerivative2(double r) const
{
   const double f = BaseFunction(r);
   return 1.0 / (f * f * f);
}

const double InvMultiquadricRBF::hNorm = 0.17889;

double InvMultiquadricRBF::BaseFunction(double r) const
{
   return 1. / sqrt(1 + r * r);
}

double InvMultiquadricRBF::BaseDerivative(double r) const
{
   const double f = BaseFunction(r);
   return -r * f * f * f;
}

double InvMultiquadricRBF::BaseDerivative2(double r) const
{
   const double f = BaseFunction(r);
   return (2 * r * r - 1) * f * f * f * f * f;
}

const double CompactGaussianRBF::hNorm = 0.37619048746124223;

CompactGaussianRBF::CompactGaussianRBF(const double rad)
   : radius(rad)
{
   double k = exp(-radius*radius);
   multK = 1. / (1. - k);
   shiftK = k / (1. - k);
}

double CompactGaussianRBF::BaseFunction(double r) const
{
   if (r < radius)
   {
      return multK * exp(-r * r) - shiftK;
   }
   else
   {
      return 0.0;
   }
}

double CompactGaussianRBF::BaseDerivative(double r) const
{
   if (r < radius)
   {
      return -2.0 * multK * r * exp(-r * r);
   }
   else
   {
      return 0.0;
   }
}

double CompactGaussianRBF::BaseDerivative2(double r) const
{
   if (r < radius)
   {
      const double r2 = r * r;
      return (-2 * 4 * r2) * multK * exp(-r2);
   }
   else
   {
      return 0.0;
   }
}

const double TruncatedGaussianRBF::hNorm = 0.37619048746124223;

double TruncatedGaussianRBF::BaseFunction(double r) const
{
   if (r < radius)
   {
      return exp(-r * r);
   }
   else
   {
      return 0.0;
   }
}

double TruncatedGaussianRBF::BaseDerivative(double r) const
{
   if (r < radius)
   {
      return -2.0 * r * exp(-r * r);
   }
   else
   {
      return 0.0;
   }
}

double TruncatedGaussianRBF::BaseDerivative2(double r) const
{
   if (r < radius)
   {
      const double r2 = r * r;
      return (-2 * 4 * r2) * exp(-r2);
   }
   else
   {
      return 0.0;
   }
}

const double Wendland11RBF::radius = 1.0;

double Wendland11RBF::BaseFunction(double r) const
{
   if (r < radius)
   {
      return pow(1 - r, 3) * (1 + 3 * r);
   }
   else
   {
      return 0.0;
   }
}

double Wendland11RBF::BaseDerivative(double r) const
{
   if (r < radius)
   {
      return -12 * pow(-1 + r, 2) * r;
   }
   else
   {
      return 0.0;
   }
}

double Wendland11RBF::BaseDerivative2(double r) const
{
   if (r < radius)
   {
      return -12 * (1 - 4 * r + 3 * r * r);
   }
   else
   {
      return 0.0;
   }
}

const double Wendland31RBF::radius = 1.0;

double Wendland31RBF::BaseFunction(double r) const
{
   if (r < radius)
   {
      return pow(1 - r, 4) * (4 * r + 1);
   }
   else
   {
      return 0.0;
   }
}

double Wendland31RBF::BaseDerivative(double r) const
{
   if (r < radius)
   {
      return -20 * pow(1. - r, 3) * r;
   }
   else
   {
      return 0.0;
   }
}

double Wendland31RBF::BaseDerivative2(double r) const
{
   if (r < radius)
   {
      return 20. * pow(1 - r, 2) * (4. * r - 1.);
   }
   else
   {
      return 0.0;
   }
}

const double Wendland33RBF::radius = 1.0;

double Wendland33RBF::BaseFunction(double r) const
{
   if (r < radius)
   {
      return pow(1 - r, 8) * (1 + 8 * r + 25 * r * r + 32 * r * r * r);
   }
   else
   {
      return 0.0;
   }
}

double Wendland33RBF::BaseDerivative(double r) const
{
   if (r < radius)
   {
      return 22 * pow(-1 + r, 7) * r * (1 + 7 * r + 16 * r * r);
   }
   else
   {
      return 0.0;
   }
}

double Wendland33RBF::BaseDerivative2(double r) const
{
   if (r < radius)
   {
      return 22 * pow(-1 + r, 6) * (-1 - 6 * r + 15 * r * r + 160 * r * r * r);
   }
   else
   {
      return 0.0;
   }
}

DistanceMetric *DistanceMetric::GetDistance(const int dim,
                                            const int pnorm)
{
   if (pnorm == 1) { return new L1Distance(dim); }
   else if (pnorm == 2) { return new L2Distance(dim); }
   else if (pnorm > 2) { return new LpDistance(dim, pnorm); }
   else { MFEM_ABORT("distance norm not recognized: " << pnorm); }
   return NULL;
}

void L1Distance::Distance(const Vector &x,
                          double &r) const
{
   r = 0.0;
   for (int d = 0; d < dim; ++d)
   {
      r += abs(x(d));
   }
}

void L1Distance::DDistance(const Vector &x,
                           Vector &dr) const
{
   for (int d = 0; d < dim; ++d)
   {
      if (x(d) > 0)
      {
         dr(d) = 1.0;
      }
      else if (x(d) < 0)
      {
         dr(d) = -1.0;
      }
      else
      {
         dr(d) = 0.0;
      }
   }
}

void L1Distance::DDDistance(const Vector &x,
                            DenseMatrix &ddr) const
{
   for (int d1 = 0; d1 < dim; ++d1)
   {
      for (int d2 = 0; d2 < dim; ++d2)
      {
         ddr(d1, d2) = 0.0;
      }
   }
}

void L2Distance::Distance(const Vector &x,
                          double &r) const
{
   r = 0.0;
   for (int d = 0; d < dim; ++d)
   {
      r += x(d) * x(d);
   }
   r = sqrt(r);
}

void L2Distance::DDistance(const Vector &x,
                           Vector &dr) const
{
   double r;
   Distance(x, r);
   double rinv = r == 0.0 ? 0.0 : 1.0 / r;
   for (int d = 0; d < dim; ++d)
   {
      dr(d) = x(d) * rinv;
   }
}

void L2Distance::DDDistance(const Vector &x,
                            DenseMatrix &ddr) const
{
   double r;
   Distance(x, r);
   double rinv = r == 0.0 ? 0.0 : 1.0 / r;
   for (int d1 = 0; d1 < dim; ++d1)
   {
      for (int d2 = 0; d2 < dim; ++d2)
      {
         if (d1 == d2)
         {
            ddr(d1, d2) = rinv - x(d1) * x(d2) * rinv * rinv * rinv;
         }
         else
         {
            ddr(d1, d2) = -x(d1) * x(d2) * rinv * rinv * rinv;
         }
      }
   }
}

void LpDistance::Distance(const Vector &x,
                          double &r) const
{
   r = 0.0;
   for (int d = 0; d < dim; ++d)
   {
      r += pow(abs(x(d)), p);
   }
   r = pow(r, pinv);
}

void LpDistance::DDistance(const Vector &x,
                           Vector &dr) const
{
   double r;
   Distance(x, r);
   double rinv = r == 0.0 ? 0.0 : 1.0 / r;
   for (int d = 0; d < dim; ++d)
   {
      dr(d) = pow(abs(x(d)) * rinv, p-1) * copysign(1.0, x(d));
   }
}

void LpDistance::DDDistance(const Vector &x,
                            DenseMatrix &ddr) const
{
   double r;
   Distance(x, r);
   for (int d1 = 0; d1 < dim; ++d1)
   {
      for (int d2 = 0; d2 < dim; ++d2)
      {
         if (d1 == d2)
         {
            ddr(d1, d2) = (p-1) * (pow(abs(x(d1)), p-2) * pow(r, 1-p)
                                   - pow(x(d1), 2*p-2) * pow(r, 1-2*p));
         }
         else
         {
            ddr(d1, d2) = (1-p) * (abs(x(d1) * x(d2)) * pow(r, 1-2*p)
                                   * copysign(1.0, x(d1))
                                   * copysign(1.0, x(d2)));
         }
      }
   }
}

void KernelFiniteElement::IntRuleToVec(const IntegrationPoint &ip,
                                       Vector &vec) const
{
   switch (dim)
   {
      case 1:
         vec(0) = ip.x;
         break;
      case 2:
         vec(0) = ip.x;
         vec(1) = ip.y;
         break;
      case 3:
         vec(0) = ip.x;
         vec(1) = ip.y;
         vec(2) = ip.z;
         break;
      default:
         MFEM_ABORT("invalid dimension: " << dim);
   }
}

void KernelFiniteElement::Project(
   Coefficient &coeff, ElementTransformation &Trans, Vector &dofs) const
{
   if (interpolate)
   {
      // Copied from PositiveFiniteElement
      for (int i = 0; i < dof; i++)
      {
         const IntegrationPoint &ip = Nodes.IntPoint(i);
         Trans.SetIntPoint(&ip);
         dofs(i) = coeff.Eval(Trans, ip);
      }
   }
   else
   {
      DenseMatrix M;
      MassIntegrator Minteg;
      Minteg.AssembleElementMatrix(*this, Trans, M);

      Vector q;
      DomainLFIntegrator qinteg(coeff);
      qinteg.AssembleRHSElementVect(*this, Trans, q);

      DenseMatrixInverse Minv(M);
      Minv.Mult(q, dofs);
   }
}

void KernelFiniteElement::Project(
   VectorCoefficient &vc, ElementTransformation &Trans, Vector &dofs) const
{
   MFEM_ASSERT(dofs.Size() == vc.GetVDim()*dof, "");
   const int vdim = vc.GetVDim();
   Vector x(vdim);

   if (interpolate)
   {
      // Copied from PositiveFiniteElement
      for (int i = 0; i < dof; i++)
      {
         const IntegrationPoint &ip = Nodes.IntPoint(i);
         Trans.SetIntPoint(&ip);
         vc.Eval(x, Trans, ip);
         for (int j = 0; j < vdim; j++)
         {
            dofs(dof*j+i) = x(j);
         }
      }
   }
   else
   {
      Vector q;
      VectorDomainLFIntegrator qinteg(vc);
      qinteg.AssembleRHSElementVect(*this, Trans, q);

      DenseMatrix M;
      MassIntegrator Minteg;
      Minteg.AssembleElementMatrix(*this, Trans, M);

      DenseMatrixInverse Minv(M);
      for (int d = 0; d < vdim; ++d)
      {
         Vector qd(q, d * dof, dof);
         Vector dofsd(dofs, d * dof, dof);
         Minv.Mult(qd, dofsd);
      }
   }
}

void KernelFiniteElement::Project(
   const FiniteElement &fe, ElementTransformation &Trans, DenseMatrix &I) const
{
   // Copied from PositiveFiniteElement
   DenseMatrix pos_mass, mixed_mass;
   MassIntegrator mass_integ;

   mass_integ.AssembleElementMatrix(*this, Trans, pos_mass);
   mass_integ.AssembleElementMatrix2(fe, *this, Trans, mixed_mass);

   DenseMatrixInverse pos_mass_inv(pos_mass);
   I.SetSize(dof, fe.GetDof());
   pos_mass_inv.Mult(mixed_mass, I);
}

RBFFiniteElement::RBFFiniteElement(const int D,
                                   const int numPointsD,
                                   const double h,
                                   const int rbfType,
                                   const int distNorm,
                                   const int intOrder)
   : KernelFiniteElement(D,
                         TensorBasisElement::GetTensorProductGeometry(D),
                         TensorBasisElement::Pow(numPointsD, D),
                         intOrder*numPointsD,
                         FunctionSpace::Qk),
#ifndef MFEM_THREAD_SAFE
     x_scr(D),
     y_scr(D),
     dy_scr(D),
     dr_scr(D),
     ddr_scr(D, D),
#endif
     numPointsD(numPointsD),
     h(h),
     rbf(RBFType::GetRBF(rbfType)),
     distance(DistanceMetric::GetDistance(D, distNorm))
{
   InitializeGeometry();
}

void RBFFiniteElement::GetCompactIndices(const Vector &ip,
                                         int (&indices)[3][2]) const
{
   // This assumes the points lie on vertices of equally-spaced Cartesian grid
   // Should work for any Lp norm, as the regions are assumed to be square
   for (int d = 0; d < dim; ++d)
   {
      indices[d][0] = std::max(int(ceil((ip(d) - radPhys) / delta)), 0);
      indices[d][1] = std::min(int(floor((ip(d) + radPhys) / delta)), numPointsD - 1);
   }

   // Fill remainder of indices with zeros so we don't have to specialize dim
   for (int d = dim; d < 3; ++d)
   {
      indices[d][0] = 0;
      indices[d][1] = 0;
   }
}

void RBFFiniteElement::GetGlobalIndices(const Vector &ip,
                                        int (&indices)[3][2]) const
{
   // Make the indices range through all the points
   for (int d = 0; d < dim; ++d)
   {
      indices[d][0] = 0;
      indices[d][1] = numPointsD - 1;
   }
}

void RBFFiniteElement::GetTensorIndices(const Vector &ip,
                                        int (&indices)[3][2]) const
{
   if (isCompact) { GetCompactIndices(ip, indices); }
   else { GetGlobalIndices(ip, indices); }
}

void RBFFiniteElement::DistanceVec(const int i,
                                   const Vector &x,
                                   Vector &y) const
{
   switch (dim)
   {
      case 1:
         y(0) = (x(0) - Nodes.IntPoint(i).x) * hPhysInv;
         break;
      case 2:
         y(0) = (x(0) - Nodes.IntPoint(i).x) * hPhysInv;
         y(1) = (x(1) - Nodes.IntPoint(i).y) * hPhysInv;
         break;
      case 3:
         y(0) = (x(0) - Nodes.IntPoint(i).x) * hPhysInv;
         y(1) = (x(1) - Nodes.IntPoint(i).y) * hPhysInv;
         y(2) = (x(2) - Nodes.IntPoint(i).z) * hPhysInv;
         break;
      default:
         MFEM_ABORT("invalid dimension: " << dim);
   }
}

void RBFFiniteElement::InitializeGeometry()
{
   delta = 1.0 / (static_cast<double>(numPointsD) - 1.0);
   hPhys = delta * h * rbf->HNorm();
   hPhysInv = 1.0 / hPhys;
   radPhys = hPhys * rbf->Radius();
   isCompact = (rbf->CompactSupport() && radPhys < 1.0);
   dimPoints[0] = numPointsD;
   dimPoints[1] = dim > 1 ? numPointsD : 1;
   dimPoints[2] = dim > 2 ? numPointsD : 1;
   switch (dim)
   {
      case 1:
         for (int i = 0; i < numPointsD; ++i)
         {
            Nodes.IntPoint(i).x = delta * static_cast<double>(i);
         }
         break;
      case 2:
         for (int i = 0; i < numPointsD; ++i)
         {
            for (int j = 0; j < numPointsD; ++j)
            {
               int l = j + numPointsD * i;
               Nodes.IntPoint(l).x = delta * static_cast<double>(i);
               Nodes.IntPoint(l).y = delta * static_cast<double>(j);
            }
         }
         break;
      case 3:
         for (int i = 0; i < numPointsD; ++i)
         {
            for (int j = 0; j < numPointsD; ++j)
            {
               for (int k = 0; k < numPointsD; ++k)
               {
                  int l = k + numPointsD * (j + numPointsD * i);
                  Nodes.IntPoint(l).x = delta * static_cast<double>(i);
                  Nodes.IntPoint(l).y = delta * static_cast<double>(j);
                  Nodes.IntPoint(l).z = delta * static_cast<double>(k);
               }
            }
         }
         break;
      default:
         MFEM_ABORT("invalid dimension: " << dim);
   }
}

void RBFFiniteElement::CalcShape(const IntegrationPoint &ip,
                                 Vector &shape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector x_scr(dim); // integration point as vector
   Vector y_scr(dim); // distance vector
   double r_scr; // distance
   int cInd[3][2];
#endif

   IntRuleToVec(ip, x_scr);
   if (isCompact)
   {
      shape = 0.0;
      GetTensorIndices(x_scr, cInd);
      for (int k = cInd[2][0]; k <= cInd[2][1]; ++k)
      {
         for (int j = cInd[1][0]; j <= cInd[1][1]; ++j)
         {
            for (int i = cInd[0][0]; i <= cInd[0][1]; ++i)
            {
               int l = k + dimPoints[2] * (j + dimPoints[1] * i);

               // Get distance vector
               DistanceVec(l, x_scr, y_scr);

               // Get distance
               distance->Distance(y_scr, r_scr);

               // Get value of function
               shape(l) = rbf->BaseFunction(r_scr);
            }
         }
      }
   }
   else
   {
      for (int i = 0; i < dof; ++i)
      {
         // Get distance vector
         DistanceVec(i, x_scr, y_scr);

         // Get distance
         distance->Distance(y_scr, r_scr);

         // Get value of function
         shape(i) = rbf->BaseFunction(r_scr);
      }
   }
}

void RBFFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                  DenseMatrix &dshape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector x_scr(dim); // integration point as vector
   Vector y_scr(dim); // distance vector
   Vector dy_scr(dim); // derivative of distance vector, diagonal
   double r_scr; // distance
   Vector dr_scr(dim); // derivative of distance
   double df_scr; // derivative value of function
   int cInd[3][2];
#endif

   IntRuleToVec(ip, x_scr);
   if (isCompact)
   {
      dshape = 0.0;
      GetTensorIndices(x_scr, cInd);
      for (int k = cInd[2][0]; k <= cInd[2][1]; ++k)
      {
         for (int j = cInd[1][0]; j <= cInd[1][1]; ++j)
         {
            for (int i = cInd[0][0]; i <= cInd[0][1]; ++i)
            {
               int l = k + dimPoints[2] * (j + dimPoints[1] * i);

               // Get distance vector
               DistanceVec(l, x_scr, y_scr);

               // Get distance and its derivative
               distance->Distance(y_scr, r_scr);
               distance->DDistance(y_scr, dr_scr);

               // Get base value of function
               df_scr = rbf->BaseDerivative(r_scr);

               for (int d = 0; d < dim; ++d)
               {
                  dshape(l, d) = dr_scr(d) * df_scr * hPhysInv;
               }
            }
         }
      }
   }
   else
   {
      for (int i = 0; i < dof; ++i)
      {
         // Get distance vector
         DistanceVec(i, x_scr, y_scr);

         // Get distance and its derivative
         distance->Distance(y_scr, r_scr);
         distance->DDistance(y_scr, dr_scr);

         // Get base value of function
         df_scr = rbf->BaseDerivative(r_scr);

         for (int d = 0; d < dim; ++d)
         {
            dshape(i, d) = dr_scr(d) * df_scr * hPhysInv;
         }
      }
   }
}

void RBFFiniteElement::CalcHessian(const IntegrationPoint &ip,
                                   DenseMatrix &hess) const
{
#ifdef MFEM_THREAD_SAFE
   Vector x_scr(dim); // integration point as vector
   Vector y_scr(dim); // distance vector
   Vector dy_scr(dim); // derivative of distance vector, diagonal
   double r_scr; // distance
   Vector dr_scr(dim); // derivative of distance
   DenseMatrix ddr_scr(dim, dim);
   double df_scr; // derivative value of function
   double ddf_scr;
   int cInd[3][2];
#endif
   IntRuleToVec(ip, x_scr);

   if (isCompact)
   {
      hess = 0.0;
      for (int k = cInd[2][0]; k <= cInd[2][1]; ++k)
      {
         for (int j = cInd[1][0]; j <= cInd[1][1]; ++j)
         {
            for (int i = cInd[0][0]; i <= cInd[0][1]; ++i)
            {
               int l = k + dimPoints[2] * (j + dimPoints[1] * i);

               // Get distance vector
               DistanceVec(l, x_scr, y_scr);

               // Get distance and its derivative
               distance->Distance(y_scr, r_scr);
               distance->DDistance(y_scr, dr_scr);
               distance->DDDistance(y_scr, ddr_scr);

               // Get base value of function
               df_scr = rbf->BaseDerivative(r_scr);
               ddf_scr = rbf->BaseDerivative2(r_scr);

               int m = 0;
               for (int d1 = 0; d1 < dim; ++d1)
               {
                  for (int d2 = d1; d2 < dim; ++d2)
                  {
                     hess(l, m) = (dr_scr(d1) * dr_scr(d2) * ddf_scr * hPhysInv
                                   + ddr_scr(d1, d2) * df_scr * hPhysInv * hPhysInv);
                     m += 1;
                  }
               }
            }
         }
      }
   }
   else
   {
      for (int i = 0; i < dof; ++i)
      {
         // Get distance vector
         DistanceVec(i, x_scr, y_scr);

         // Get distance and its derivative
         distance->Distance(y_scr, r_scr);
         distance->DDistance(y_scr, dr_scr);
         distance->DDDistance(y_scr, ddr_scr);

         // Get base value of function
         df_scr = rbf->BaseDerivative(r_scr);
         ddf_scr = rbf->BaseDerivative2(r_scr);

         int k = 0;
         for (int d1 = 0; d1 < dim; ++d1)
         {
            for (int d2 = d1; d2 < dim; ++d2)
            {
               hess(i, k) = (dr_scr(d1) * dr_scr(d2) * ddf_scr * hPhysInv
                             + ddr_scr(d1, d2) * df_scr * hPhysInv * hPhysInv);
               k += 1;
            }
         }
      }
   }
}

RKFiniteElement::RKFiniteElement(const int D,
                                 const int numPointsD,
                                 const double h,
                                 const int rbfType,
                                 const int distNorm,
                                 const int order,
                                 const int intOrder)
   : KernelFiniteElement(D,
                         TensorBasisElement::GetTensorProductGeometry(D),
                         TensorBasisElement::Pow(numPointsD, D),
                         intOrder * numPointsD, // integration order
                         FunctionSpace::Qk),
     polyOrd(order),
     numPoly(RKFiniteElement::GetNumPoly(order, D)),
     numPoly1d(order+1),
     baseFE(new RBFFiniteElement(D, numPointsD, h, rbfType, distNorm, intOrder))
{
   Nodes = baseFE->GetNodes();
#ifndef MFEM_THREAD_SAFE
   x_scr.SetSize(dim);
   y_scr.SetSize(dim);
   g_scr.SetSize(numPoly);
   c_scr.SetSize(numPoly);
   s_scr.SetSize(dof);
   p_scr.SetSize(numPoly);
   df_scr.SetSize(dim);
   q_scr.SetSize(numPoly1d, dim);
   dq_scr.SetSize(numPoly1d, dim);
   M_scr.SetSize(numPoly, numPoly);
   for (int d = 0; d < dim; ++d)
   {
      dM_scr[d].SetSize(numPoly, numPoly);
      dc_scr[d].SetSize(numPoly);
      dp_scr[d].SetSize(numPoly);
   }
#endif
}

void RKFiniteElement::CalcShape(const IntegrationPoint &ip,
                                Vector &shape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector c_scr(numPoly);
   Vector g_scr(numPoly);
   DenseMatrix M_scr(numPoly, numPoly);
   DenseMatrixInverse Minv_scr;
#endif

   // Fill the shape vector with base function values
   baseFE->CalcShape(ip, shape);

   // Calculate M
   GetM(shape, ip, M_scr);
   Minv_scr.Factor(M_scr);

   // Get coefficients
   GetG(g_scr);
   Minv_scr.Mult(g_scr, c_scr);

   // Calculate the values of the functions
   CalculateValues(c_scr, shape, ip, shape);
}

void RKFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                 DenseMatrix &dshape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector c_scr(numPoly);
   Vector g_scr(numPoly);
   Vector s_scr(dof);
   DenseMatrix M_scr(numPoly, numPoly);
   DenseMatrixInverse Minv_scr;
   Vector dc_scr[3];
   DenseMatrix dM_scr[3];
   for (int d = 0; d < dim; ++d)
   {
      dc_scr[d].SetSize(numPoly);
      dM_scr[d].SetSize(numPoly, numPoly);
   }
#endif

   // Fill the shape vector with base function values
   baseFE->CalcShape(ip, s_scr);
   baseFE->CalcDShape(ip, dshape);

   // Calculate M and dM
   GetDM(s_scr, dshape, ip, M_scr, dM_scr);
   Minv_scr.Factor(M_scr);

   // Calculate coefficients
   GetG(g_scr);
   Minv_scr.Mult(g_scr, c_scr);
   for (int d = 0; d < dim; ++d)
   {
      dM_scr[d].Mult(c_scr, g_scr);
      Minv_scr.Mult(g_scr, dc_scr[d]);
      dc_scr[d].Neg();
   }

   // Calculate the values of the functions
   CalculateDValues(c_scr, dc_scr, s_scr, dshape, ip, dshape);
}

int RKFiniteElement::GetNumPoly(int polyOrd, int dim)
{
   double n = polyOrd + dim;
   double num = 1;
   for (int i = 0; i < dim; ++i)
   {
      num *= (n - i) / (i + 1.0);
   }
   return static_cast<int>(round(num));
}

void RKFiniteElement::DistanceVec(const int i,
                                  const Vector &x,
                                  Vector &y) const
{
   switch (dim)
   {
      case 1:
         y(0) = x(0) - Nodes.IntPoint(i).x;
         break;
      case 2:
         y(0) = x(0) - Nodes.IntPoint(i).x;
         y(1) = x(1) - Nodes.IntPoint(i).y;
         break;
      case 3:
         y(0) = x(0) - Nodes.IntPoint(i).x;
         y(1) = x(1) - Nodes.IntPoint(i).y;
         y(2) = x(2) - Nodes.IntPoint(i).z;
         break;
      default:
         MFEM_ABORT("invalid dimension: " << dim);
   }
}

void RKFiniteElement::GetG(Vector &g) const
{
   g(0) = 1.0;
   for (int i = 1; i < numPoly; ++i)
   {
      g(i) = 0.0;
   }
}

void RKFiniteElement::GetPoly(const Vector &x,
                              Vector &p) const
{
#ifdef MFEM_THREAD_SAFE
   DenseMatrix q_scr(numPoly1d, dim);
#endif

   int index = 0;
   switch (dim)
   {
      case 1:
         poly1d.CalcMono(polyOrd, x(0), &p(0));
         break;
      case 2:
         poly1d.CalcMono(polyOrd, x(0), &q_scr(0, 0));
         poly1d.CalcMono(polyOrd, x(1), &q_scr(0, 1));
         for (int i = 0; i < numPoly1d; ++i)
         {
            for (int j = 0; j + i < numPoly1d; ++j)
            {
               p(index) = q_scr(i, 0) * q_scr(j, 1);
               ++index;
            }
         }
         break;
      case 3:
         poly1d.CalcMono(polyOrd, x(0), &q_scr(0, 0));
         poly1d.CalcMono(polyOrd, x(1), &q_scr(0, 1));
         poly1d.CalcMono(polyOrd, x(2), &q_scr(0, 2));
         for (int i = 0; i < numPoly1d; ++i)
         {
            for (int j = 0; j + i < numPoly1d; ++j)
            {
               for (int k = 0; k + j + i < numPoly1d; ++k)
               {
                  p(index) = q_scr(i, 0) * q_scr(j, 1) * q_scr(k, 2);
                  ++index;
               }
            }
         }
         break;
      default:
         MFEM_ABORT("invalid dimension: " << dim);
   }
}

void RKFiniteElement::GetDPoly(const Vector &x,
                               Vector &p,
                               Vector (&dp)[3]) const
{
#ifdef MFEM_THREAD_SAFE
   DenseMatrix q_scr(numPoly1d, dim);
   DenseMatrix dq_scr(numPoly1d, dim);
#endif

   int index = 0;
   switch (dim)
   {
      case 1:
         poly1d.CalcMono(polyOrd, x(0), &p(0), &dp[0](0));
         break;
      case 2:
         poly1d.CalcMono(polyOrd, x(0), &q_scr(0, 0), &dq_scr(0, 0));
         poly1d.CalcMono(polyOrd, x(1), &q_scr(0, 1), &dq_scr(0, 1));
         for (int i = 0; i < numPoly1d; ++i)
         {
            for (int j = 0; j + i < numPoly1d; ++j)
            {
               p(index) = q_scr(i, 0) * q_scr(j, 1);
               dp[0](index) = dq_scr(i, 0) * q_scr(j, 1);
               dp[1](index) = q_scr(i, 0) * dq_scr(j, 1);
               ++index;
            }
         }
         break;
      case 3:
         poly1d.CalcMono(polyOrd, x(0), &q_scr(0, 0), &dq_scr(0, 0));
         poly1d.CalcMono(polyOrd, x(1), &q_scr(0, 1), &dq_scr(0, 1));
         poly1d.CalcMono(polyOrd, x(2), &q_scr(0, 2), &dq_scr(0, 2));
         for (int i = 0; i < numPoly1d; ++i)
         {
            for (int j = 0; j + i < numPoly1d; ++j)
            {
               for (int k = 0; k + j + i < numPoly1d; ++k)
               {
                  p(index) = q_scr(i, 0) * q_scr(j, 1) * q_scr(k, 2);
                  dp[0](index) = dq_scr(i, 0) * q_scr(j, 1) * q_scr(k, 2);
                  dp[1](index) = q_scr(i, 0) * dq_scr(j, 1) * q_scr(k, 2);
                  dp[2](index) = q_scr(i, 0) * q_scr(j, 1) * dq_scr(k, 2);
                  ++index;
               }
            }
         }
         break;
      default:
         MFEM_ABORT("invalid dimension: " << dim);
   }
}

void RKFiniteElement::GetM(const Vector &baseShape,
                           const IntegrationPoint &ip,
                           DenseMatrix &M) const
{
#ifdef MFEM_THREAD_SAFE
   Vector x_scr(dim);
   Vector y_scr(dim);
   Vector p_scr(numPoly);
   int cInd[3][2];
   int dimPoints[3];
#endif

   IntRuleToVec(ip, x_scr);

   // Zero out M
   M = 0.0;

   if (baseFE->IsCompact() && baseFE->TensorIndexed())
   {
      baseFE->GetTensorIndices(x_scr, cInd);
      baseFE->GetTensorNumPoints(dimPoints);
      for (int k = cInd[2][0]; k <= cInd[2][1]; ++k)
      {
         for (int j = cInd[1][0]; j <= cInd[1][1]; ++j)
         {
            for (int i = cInd[0][0]; i <= cInd[0][1]; ++i)
            {
               int l = k + dimPoints[2] * (j + dimPoints[1] * i);

               // Distance vector
               DistanceVec(l, x_scr, y_scr);

               // Polynomials
               GetPoly(y_scr, p_scr);

               // Add values to M
               AddToM(p_scr, baseShape(l), M);
            }
         }
      }
   }
   else
   {
      // Get lower-triangular M matrix
      for (int i = 0; i < dof; ++i)
      {
         // Distance vector
         DistanceVec(i, x_scr, y_scr);

         // Polynomials
         GetPoly(y_scr, p_scr);

         // Add values to M
         AddToM(p_scr, baseShape(i), M);
      }
   }

   // Fill in symmetries
   for (int k = 1; k < numPoly; ++k)
   {
      for (int l = 0; l < k; ++l)
      {
         M(k, l) = M(l, k);
      }
   }
}

void RKFiniteElement::GetDM(const Vector &baseShape,
                            const DenseMatrix &baseDeriv,
                            const IntegrationPoint &ip,
                            DenseMatrix &M,
                            DenseMatrix (&dM)[3]) const
{
#ifdef MFEM_THREAD_SAFE
   double f_scr;
   Vector x_scr(dim);
   Vector y_scr(dim);
   Vector p_scr(numPoly);
   Vector df_scr(dim);
   Vector dp_scr[3];
   for (int d = 0; d < dim; ++d)
   {
      dp_scr[d].SetSize(numPoly);
   }
   int cInd[3][2];
   int dimPoints[3];
#endif
   IntRuleToVec(ip, x_scr);

   // Zero out M and dM
   M = 0.0;
   for (int d = 0; d < dim; ++d)
   {
      dM[d] = 0.0;
   }

   // Get lower-triangular M and dM matrix
   if (baseFE->IsCompact() && baseFE->TensorIndexed())
   {
      baseFE->GetTensorIndices(x_scr, cInd);
      baseFE->GetTensorNumPoints(dimPoints);
      for (int k = cInd[2][0]; k <= cInd[2][1]; ++k)
      {
         for (int j = cInd[1][0]; j <= cInd[1][1]; ++j)
         {
            for (int i = cInd[0][0]; i <= cInd[0][1]; ++i)
            {
               int l = k + dimPoints[2] * (j + dimPoints[1] * i);

               // Distance vector
               DistanceVec(l, x_scr, y_scr);

               // Polynomials
               GetDPoly(y_scr, p_scr, dp_scr);

               // Add values to M
               f_scr = baseShape(l);
               AddToM(p_scr, f_scr, M);

               // Add values to dM
               for (int d = 0; d < dim; ++d)
               {
                  df_scr(d) = baseDeriv(l, d);
               }
               AddToDM(p_scr, dp_scr, f_scr, df_scr, dM);
            }
         }
      }
   }
   else
   {
      for (int i = 0; i < dof; ++i)
      {
         // Distance vector
         DistanceVec(i, x_scr, y_scr);

         // Polynomials
         GetDPoly(y_scr, p_scr, dp_scr);

         // Add values to M
         f_scr = baseShape(i);
         AddToM(p_scr, f_scr, M);

         // Add values to dM
         for (int d = 0; d < dim; ++d)
         {
            df_scr(d) = baseDeriv(i, d);
         }
         AddToDM(p_scr, dp_scr, f_scr, df_scr, dM);
      }
   }

   // Fill in symmetries
   for (int k = 1; k < numPoly; ++k)
   {
      for (int l = 0; l < k; ++l)
      {
         M(k, l) = M(l, k);
      }
   }
   for (int d = 0; d < dim; ++d)
   {
      for (int k = 1; k < numPoly; ++k)
      {
         for (int l = 0; l < k; ++l)
         {
            dM[d](k, l) = dM[d](l, k);
         }
      }
   }
}

void RKFiniteElement::AddToM(const Vector &p,
                             const double &f,
                             DenseMatrix &M) const
{
   // Add to lower triangular part
   for (int k = 0; k < numPoly; ++k)
   {
      for (int l = k; l < numPoly; ++l)
      {
         M(k, l) += p(k) * p(l) * f;
      }
   }
}

void RKFiniteElement::AddToDM(const Vector &p,
                              const Vector (&dp)[3],
                              const double &f,
                              const Vector &df,
                              DenseMatrix (&dM)[3]) const
{
   // Add to lower triangular part
   for (int d = 0; d < dim; ++d)
   {
      const Vector &dpl = dp[d];
      for (int k = 0; k < numPoly; ++k)
      {
         for (int l = k; l < numPoly; ++l)
         {
            dM[d](k, l) += ((dpl(k) * p(l) + p(k) * dpl(l)) * f
                            + p(l) * p(k) * df(d));
         }
      }
   }
}

void RKFiniteElement::CalculateValues(const Vector &c,
                                      const Vector &baseShape,
                                      const IntegrationPoint &ip,
                                      Vector &shape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector x_scr(dim);
   Vector y_scr(dim);
   Vector p_scr(numPoly);
   int cInd[3][2];
   int dimPoints[3];
#endif

   IntRuleToVec(ip, x_scr);

   if (baseFE->IsCompact() && baseFE->TensorIndexed())
   {
      baseFE->GetTensorIndices(x_scr, cInd);
      baseFE->GetTensorNumPoints(dimPoints);
      for (int k = cInd[2][0]; k <= cInd[2][1]; ++k)
      {
         for (int j = cInd[1][0]; j <= cInd[1][1]; ++j)
         {
            for (int i = cInd[0][0]; i <= cInd[0][1]; ++i)
            {
               int l = k + dimPoints[2] * (j + dimPoints[1] * i);

               // Distance vector
               DistanceVec(l, x_scr, y_scr);

               // Polynomials
               GetPoly(y_scr, p_scr);

               // Get shape
               shape(l) = (p_scr * c) * baseShape(l);
            }
         }
      }
   }
   else
   {
      for (int i = 0; i < dof; ++i)
      {
         // Distance vector
         DistanceVec(i, x_scr, y_scr);

         // Polynomials
         GetPoly(y_scr, p_scr);

         // Get shape
         shape(i) = (p_scr * c) * baseShape(i);
      }
   }
}

void RKFiniteElement::CalculateDValues(const Vector &c,
                                       const Vector (&dc)[3],
                                       const Vector &baseShape,
                                       const DenseMatrix &baseDShape,
                                       const IntegrationPoint &ip,
                                       DenseMatrix &dshape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector x_scr(dim);
   Vector y_scr(dim);
   Vector p_scr(numPoly);
   Vector dp_scr[3];
   for (int d = 0; d < dim; ++d)
   {
      dp_scr[d].SetSize(numPoly);
   }
   int cInd[3][2];
   int dimPoints[3];
#endif

   IntRuleToVec(ip, x_scr);

   if (baseFE->IsCompact() && baseFE->TensorIndexed())
   {
      baseFE->GetTensorIndices(x_scr, cInd);
      baseFE->GetTensorNumPoints(dimPoints);
      for (int k = cInd[2][0]; k <= cInd[2][1]; ++k)
      {
         for (int j = cInd[1][0]; j <= cInd[1][1]; ++j)
         {
            for (int i = cInd[0][0]; i <= cInd[0][1]; ++i)
            {
               int l = k + dimPoints[2] * (j + dimPoints[1] * i);

               // Distance vector
               DistanceVec(l, x_scr, y_scr);

               // Polynomials
               GetDPoly(y_scr, p_scr, dp_scr);

               // Get shape
               for (int d = 0; d < dim; ++d)
               {
                  dshape(l, d) = ((dp_scr[d] * c + p_scr * dc[d]) * baseShape(l)
                                  + (p_scr * c) * baseDShape(l, d));
               }
            }
         }
      }
   }
   else
   {
      for (int i = 0; i < dof; ++i)
      {
         // Distance vector
         DistanceVec(i, x_scr, y_scr);

         // Polynomials
         GetDPoly(y_scr, p_scr, dp_scr);

         // Get shape
         for (int d = 0; d < dim; ++d)
         {
            dshape(i, d) = ((dp_scr[d] * c + p_scr * dc[d]) * baseShape(i)
                            + (p_scr * c) * baseDShape(i, d));
         }
      }
   }
}

} // end namespace mfem
