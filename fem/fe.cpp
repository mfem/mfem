// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

// Finite Element classes

#include "fe.hpp"
#include "fe_coll.hpp"
#include "../mesh/nurbs.hpp"
#include "bilininteg.hpp"
#include <cmath>

namespace mfem
{

using namespace std;

FiniteElement::FiniteElement(int D, Geometry::Type G, int Do, int O, int F)
   : Nodes(Do)
{
   Dim = D ; GeomType = G ; Dof = Do ; Order = O ; FuncSpace = F;
   RangeType = SCALAR;
   MapType = VALUE;
   DerivType = NONE;
   DerivRangeType = SCALAR;
   DerivMapType = VALUE;
   OperatorType = FE;
   for (int i = 0; i < Geometry::MaxDim; i++) { Orders[i] = -1; }
#ifndef MFEM_THREAD_SAFE
   vshape.SetSize(Dof, Dim);
#endif
}

void FiniteElement::CalcVShape (
   const IntegrationPoint &ip, DenseMatrix &shape) const
{
   mfem_error ("FiniteElement::CalcVShape (ip, ...)\n"
               "   is not implemented for this class!");
}

void FiniteElement::CalcVShape (
   ElementTransformation &Trans, DenseMatrix &shape) const
{
   mfem_error ("FiniteElement::CalcVShape (trans, ...)\n"
               "   is not implemented for this class!");
}

void FiniteElement::CalcDivShape (
   const IntegrationPoint &ip, Vector &divshape) const
{
   mfem_error ("FiniteElement::CalcDivShape (ip, ...)\n"
               "   is not implemented for this class!");
}

void FiniteElement::CalcPhysDivShape(
   ElementTransformation &Trans, Vector &div_shape) const
{
   CalcDivShape(Trans.GetIntPoint(), div_shape);
   div_shape *= (1.0 / Trans.Weight());
}

void FiniteElement::CalcCurlShape(const IntegrationPoint &ip,
                                  DenseMatrix &curl_shape) const
{
   mfem_error ("FiniteElement::CalcCurlShape (ip, ...)\n"
               "   is not implemented for this class!");
}

void FiniteElement::CalcPhysCurlShape(ElementTransformation &Trans,
                                      DenseMatrix &curl_shape) const
{
   switch (Dim)
   {
      case 3:
      {
#ifdef MFEM_THREAD_SAFE
         DenseMatrix vshape(Dof, Dim);
#endif
         CalcCurlShape(Trans.GetIntPoint(), vshape);
         MultABt(vshape, Trans.Jacobian(), curl_shape);
         curl_shape *= (1.0 / Trans.Weight());
         break;
      }
      case 2:
         // This is valid for both 2x2 and 3x2 Jacobians
         CalcCurlShape(Trans.GetIntPoint(), curl_shape);
         curl_shape *= (1.0 / Trans.Weight());
         break;
      default:
         MFEM_ABORT("Invalid dimension, Dim = " << Dim);
   }
}

void FiniteElement::GetFaceDofs(int face, int **dofs, int *ndofs) const
{
   mfem_error ("FiniteElement::GetFaceDofs (...)");
}

void FiniteElement::CalcHessian (const IntegrationPoint &ip,
                                 DenseMatrix &h) const
{
   mfem_error ("FiniteElement::CalcHessian (...) is not overloaded !");
}

void FiniteElement::GetLocalInterpolation (ElementTransformation &Trans,
                                           DenseMatrix &I) const
{
   mfem_error ("GetLocalInterpolation (...) is not overloaded !");
}

void FiniteElement::GetLocalRestriction(ElementTransformation &,
                                        DenseMatrix &) const
{
   mfem_error("FiniteElement::GetLocalRestriction() is not overloaded !");
}

void FiniteElement::GetTransferMatrix(const FiniteElement &fe,
                                      ElementTransformation &Trans,
                                      DenseMatrix &I) const
{
   MFEM_ABORT("method is not overloaded !");
}

void FiniteElement::Project (
   Coefficient &coeff, ElementTransformation &Trans, Vector &dofs) const
{
   mfem_error ("FiniteElement::Project (...) is not overloaded !");
}

void FiniteElement::Project (
   VectorCoefficient &vc, ElementTransformation &Trans, Vector &dofs) const
{
   mfem_error ("FiniteElement::Project (...) (vector) is not overloaded !");
}

void FiniteElement::ProjectMatrixCoefficient(
   MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
{
   mfem_error("FiniteElement::ProjectMatrixCoefficient() is not overloaded !");
}

void FiniteElement::ProjectDelta(int vertex, Vector &dofs) const
{
   mfem_error("FiniteElement::ProjectDelta(...) is not implemented for "
              "this element!");
}

void FiniteElement::Project(
   const FiniteElement &fe, ElementTransformation &Trans, DenseMatrix &I) const
{
   mfem_error("FiniteElement::Project(...) (fe version) is not implemented "
              "for this element!");
}

void FiniteElement::ProjectGrad(
   const FiniteElement &fe, ElementTransformation &Trans,
   DenseMatrix &grad) const
{
   mfem_error("FiniteElement::ProjectGrad(...) is not implemented for "
              "this element!");
}

void FiniteElement::ProjectCurl(
   const FiniteElement &fe, ElementTransformation &Trans,
   DenseMatrix &curl) const
{
   mfem_error("FiniteElement::ProjectCurl(...) is not implemented for "
              "this element!");
}

void FiniteElement::ProjectDiv(
   const FiniteElement &fe, ElementTransformation &Trans,
   DenseMatrix &div) const
{
   mfem_error("FiniteElement::ProjectDiv(...) is not implemented for "
              "this element!");
}

void FiniteElement::CalcPhysShape(ElementTransformation &Trans,
                                  Vector &shape) const
{
   CalcShape(Trans.GetIntPoint(), shape);
   if (MapType == INTEGRAL)
   {
      shape /= Trans.Weight();
   }
}

void FiniteElement::CalcPhysDShape(ElementTransformation &Trans,
                                   DenseMatrix &dshape) const
{
   MFEM_ASSERT(MapType == VALUE, "");
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(Dof, Dim);
#endif
   CalcDShape(Trans.GetIntPoint(), vshape);
   Mult(vshape, Trans.InverseJacobian(), dshape);
}


void ScalarFiniteElement::NodalLocalInterpolation (
   ElementTransformation &Trans, DenseMatrix &I,
   const ScalarFiniteElement &fine_fe) const
{
   double v[Geometry::MaxDim];
   Vector vv (v, Dim);
   IntegrationPoint f_ip;

#ifdef MFEM_THREAD_SAFE
   Vector c_shape(Dof);
#endif

   MFEM_ASSERT(MapType == fine_fe.GetMapType(), "");

   I.SetSize(fine_fe.Dof, Dof);
   for (int i = 0; i < fine_fe.Dof; i++)
   {
      Trans.Transform(fine_fe.Nodes.IntPoint(i), vv);
      f_ip.Set(v, Dim);
      CalcShape(f_ip, c_shape);
      for (int j = 0; j < Dof; j++)
         if (fabs(I(i,j) = c_shape(j)) < 1.0e-12)
         {
            I(i,j) = 0.0;
         }
   }
   if (MapType == INTEGRAL)
   {
      // assuming Trans is linear; this should be ok for all refinement types
      Trans.SetIntPoint(&Geometries.GetCenter(GeomType));
      I *= Trans.Weight();
   }
}

void ScalarFiniteElement::ScalarLocalInterpolation(
   ElementTransformation &Trans, DenseMatrix &I,
   const ScalarFiniteElement &fine_fe) const
{
   // General "interpolation", defined by L2 projection

   double v[Geometry::MaxDim];
   Vector vv (v, Dim);
   IntegrationPoint f_ip;

   const int fs = fine_fe.GetDof(), cs = this->GetDof();
   I.SetSize(fs, cs);
   Vector fine_shape(fs), coarse_shape(cs);
   DenseMatrix fine_mass(fs), fine_coarse_mass(fs, cs); // initialized with 0
   const int ir_order = GetOrder() + fine_fe.GetOrder();
   const IntegrationRule &ir = IntRules.Get(fine_fe.GetGeomType(), ir_order);

   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      fine_fe.CalcShape(ip, fine_shape);
      Trans.Transform(ip, vv);
      f_ip.Set(v, Dim);
      this->CalcShape(f_ip, coarse_shape);

      AddMult_a_VVt(ip.weight, fine_shape, fine_mass);
      AddMult_a_VWt(ip.weight, fine_shape, coarse_shape, fine_coarse_mass);
   }

   DenseMatrixInverse fine_mass_inv(fine_mass);
   fine_mass_inv.Mult(fine_coarse_mass, I);

   if (MapType == INTEGRAL)
   {
      // assuming Trans is linear; this should be ok for all refinement types
      Trans.SetIntPoint(&Geometries.GetCenter(GeomType));
      I *= Trans.Weight();
   }
}


void NodalFiniteElement::ProjectCurl_2D(
   const FiniteElement &fe, ElementTransformation &Trans,
   DenseMatrix &curl) const
{
   MFEM_ASSERT(GetMapType() == FiniteElement::INTEGRAL, "");

   DenseMatrix curl_shape(fe.GetDof(), 1);

   curl.SetSize(Dof, fe.GetDof());
   for (int i = 0; i < Dof; i++)
   {
      fe.CalcCurlShape(Nodes.IntPoint(i), curl_shape);
      for (int j = 0; j < fe.GetDof(); j++)
      {
         curl(i,j) = curl_shape(j,0);
      }
   }
}

void InvertLinearTrans(ElementTransformation &trans,
                       const IntegrationPoint &pt, Vector &x)
{
   // invert a linear transform with one Newton step
   IntegrationPoint p0;
   p0.Set3(0, 0, 0);
   trans.Transform(p0, x);

   double store[3];
   Vector v(store, x.Size());
   pt.Get(v, x.Size());
   v -= x;

   trans.InverseJacobian().Mult(v, x);
}

void NodalFiniteElement::GetLocalRestriction(ElementTransformation &Trans,
                                             DenseMatrix &R) const
{
   IntegrationPoint ipt;
   Vector pt(&ipt.x, Dim);

#ifdef MFEM_THREAD_SAFE
   Vector c_shape(Dof);
#endif

   Trans.SetIntPoint(&Nodes[0]);

   for (int j = 0; j < Dof; j++)
   {
      InvertLinearTrans(Trans, Nodes[j], pt);
      if (Geometries.CheckPoint(GeomType, ipt)) // do we need an epsilon here?
      {
         CalcShape(ipt, c_shape);
         R.SetRow(j, c_shape);
      }
      else
      {
         // Set the whole row to avoid valgrind warnings in R.Threshold().
         R.SetRow(j, infinity());
      }
   }
   R.Threshold(1e-12);
}

void NodalFiniteElement::Project (
   Coefficient &coeff, ElementTransformation &Trans, Vector &dofs) const
{
   for (int i = 0; i < Dof; i++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(i);
      // some coefficients expect that Trans.IntPoint is the same
      // as the second argument of Eval
      Trans.SetIntPoint(&ip);
      dofs(i) = coeff.Eval (Trans, ip);
      if (MapType == INTEGRAL)
      {
         dofs(i) *= Trans.Weight();
      }
   }
}

void NodalFiniteElement::Project (
   VectorCoefficient &vc, ElementTransformation &Trans, Vector &dofs) const
{
   MFEM_ASSERT(dofs.Size() == vc.GetVDim()*Dof, "");
   Vector x(vc.GetVDim());

   for (int i = 0; i < Dof; i++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(i);
      Trans.SetIntPoint(&ip);
      vc.Eval (x, Trans, ip);
      if (MapType == INTEGRAL)
      {
         x *= Trans.Weight();
      }
      for (int j = 0; j < x.Size(); j++)
      {
         dofs(Dof*j+i) = x(j);
      }
   }
}

void NodalFiniteElement::ProjectMatrixCoefficient(
   MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
{
   // (mc.height x mc.width) @ DOFs -> (Dof x mc.width x mc.height) in dofs
   MFEM_ASSERT(dofs.Size() == mc.GetHeight()*mc.GetWidth()*Dof, "");
   DenseMatrix MQ(mc.GetHeight(), mc.GetWidth());

   for (int k = 0; k < Dof; k++)
   {
      T.SetIntPoint(&Nodes.IntPoint(k));
      mc.Eval(MQ, T, Nodes.IntPoint(k));
      if (MapType == INTEGRAL) { MQ *= T.Weight(); }
      for (int r = 0; r < MQ.Height(); r++)
      {
         for (int d = 0; d < MQ.Width(); d++)
         {
            dofs(k+Dof*(d+MQ.Width()*r)) = MQ(r,d);
         }
      }
   }
}

void NodalFiniteElement::Project(
   const FiniteElement &fe, ElementTransformation &Trans, DenseMatrix &I) const
{
   if (fe.GetRangeType() == SCALAR)
   {
      MFEM_ASSERT(MapType == fe.GetMapType(), "");

      Vector shape(fe.GetDof());

      I.SetSize(Dof, fe.GetDof());
      for (int k = 0; k < Dof; k++)
      {
         fe.CalcShape(Nodes.IntPoint(k), shape);
         for (int j = 0; j < shape.Size(); j++)
         {
            I(k,j) = (fabs(shape(j)) < 1e-12) ? 0.0 : shape(j);
         }
      }
   }
   else
   {
      DenseMatrix vshape(fe.GetDof(), Trans.GetSpaceDim());

      I.SetSize(vshape.Width()*Dof, fe.GetDof());
      for (int k = 0; k < Dof; k++)
      {
         Trans.SetIntPoint(&Nodes.IntPoint(k));
         fe.CalcVShape(Trans, vshape);
         if (MapType == INTEGRAL)
         {
            vshape *= Trans.Weight();
         }
         for (int j = 0; j < vshape.Height(); j++)
            for (int d = 0; d < vshape.Width(); d++)
            {
               I(k+d*Dof,j) = vshape(j,d);
            }
      }
   }
}

void NodalFiniteElement::ProjectGrad(
   const FiniteElement &fe, ElementTransformation &Trans,
   DenseMatrix &grad) const
{
   MFEM_ASSERT(fe.GetMapType() == VALUE, "");
   MFEM_ASSERT(Trans.GetSpaceDim() == Dim, "")

   DenseMatrix dshape(fe.GetDof(), Dim), grad_k(fe.GetDof(), Dim), Jinv(Dim);

   grad.SetSize(Dim*Dof, fe.GetDof());
   for (int k = 0; k < Dof; k++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(k);
      fe.CalcDShape(ip, dshape);
      Trans.SetIntPoint(&ip);
      CalcInverse(Trans.Jacobian(), Jinv);
      Mult(dshape, Jinv, grad_k);
      if (MapType == INTEGRAL)
      {
         grad_k *= Trans.Weight();
      }
      for (int j = 0; j < grad_k.Height(); j++)
         for (int d = 0; d < Dim; d++)
         {
            grad(k+d*Dof,j) = grad_k(j,d);
         }
   }
}

void NodalFiniteElement::ProjectDiv(
   const FiniteElement &fe, ElementTransformation &Trans,
   DenseMatrix &div) const
{
   double detJ;
   Vector div_shape(fe.GetDof());

   div.SetSize(Dof, fe.GetDof());
   for (int k = 0; k < Dof; k++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(k);
      fe.CalcDivShape(ip, div_shape);
      if (MapType == VALUE)
      {
         Trans.SetIntPoint(&ip);
         detJ = Trans.Weight();
         for (int j = 0; j < div_shape.Size(); j++)
         {
            div(k,j) = (fabs(div_shape(j)) < 1e-12) ? 0.0 : div_shape(j)/detJ;
         }
      }
      else
      {
         for (int j = 0; j < div_shape.Size(); j++)
         {
            div(k,j) = (fabs(div_shape(j)) < 1e-12) ? 0.0 : div_shape(j);
         }
      }
   }
}


void PositiveFiniteElement::Project(
   Coefficient &coeff, ElementTransformation &Trans, Vector &dofs) const
{
   for (int i = 0; i < Dof; i++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(i);
      Trans.SetIntPoint(&ip);
      dofs(i) = coeff.Eval(Trans, ip);
   }
}

void PositiveFiniteElement::Project(
   VectorCoefficient &vc, ElementTransformation &Trans, Vector &dofs) const
{
   MFEM_ASSERT(dofs.Size() == vc.GetVDim()*Dof, "");
   Vector x(vc.GetVDim());

   for (int i = 0; i < Dof; i++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(i);
      Trans.SetIntPoint(&ip);
      vc.Eval (x, Trans, ip);
      for (int j = 0; j < x.Size(); j++)
      {
         dofs(Dof*j+i) = x(j);
      }
   }
}

void PositiveFiniteElement::Project(
   const FiniteElement &fe, ElementTransformation &Trans, DenseMatrix &I) const
{
   const NodalFiniteElement *nfe =
      dynamic_cast<const NodalFiniteElement *>(&fe);

   if (nfe && Dof == nfe->GetDof())
   {
      nfe->Project(*this, Trans, I);
      I.Invert();
   }
   else
   {
      // local L2 projection
      DenseMatrix pos_mass, mixed_mass;
      MassIntegrator mass_integ;

      mass_integ.AssembleElementMatrix(*this, Trans, pos_mass);
      mass_integ.AssembleElementMatrix2(fe, *this, Trans, mixed_mass);

      DenseMatrixInverse pos_mass_inv(pos_mass);
      I.SetSize(Dof, fe.GetDof());
      pos_mass_inv.Mult(mixed_mass, I);
   }
}


void VectorFiniteElement::CalcShape (
   const IntegrationPoint &ip, Vector &shape ) const
{
   mfem_error ("Error: Cannot use scalar CalcShape(...) function with\n"
               "   VectorFiniteElements!");
}

void VectorFiniteElement::CalcDShape (
   const IntegrationPoint &ip, DenseMatrix &dshape ) const
{
   mfem_error ("Error: Cannot use scalar CalcDShape(...) function with\n"
               "   VectorFiniteElements!");
}

void VectorFiniteElement::SetDerivMembers()
{
   switch (MapType)
   {
      case H_DIV:
         DerivType = DIV;
         DerivRangeType = SCALAR;
         DerivMapType = INTEGRAL;
         break;
      case H_CURL:
         switch (Dim)
         {
            case 3: // curl: 3D H_CURL -> 3D H_DIV
               DerivType = CURL;
               DerivRangeType = VECTOR;
               DerivMapType = H_DIV;
               break;
            case 2:
               // curl: 2D H_CURL -> INTEGRAL
               DerivType = CURL;
               DerivRangeType = SCALAR;
               DerivMapType = INTEGRAL;
               break;
            case 1:
               DerivType = NONE;
               DerivRangeType = SCALAR;
               DerivMapType = INTEGRAL;
               break;
            default:
               MFEM_ABORT("Invalid dimension, Dim = " << Dim);
         }
         break;
      default:
         MFEM_ABORT("Invalid MapType = " << MapType);
   }
}

void VectorFiniteElement::CalcVShape_RT (
   ElementTransformation &Trans, DenseMatrix &shape) const
{
   MFEM_ASSERT(MapType == H_DIV, "");
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(Dof, Dim);
#endif
   CalcVShape(Trans.GetIntPoint(), vshape);
   MultABt(vshape, Trans.Jacobian(), shape);
   shape *= (1.0 / Trans.Weight());
}

void VectorFiniteElement::CalcVShape_ND (
   ElementTransformation &Trans, DenseMatrix &shape) const
{
   MFEM_ASSERT(MapType == H_CURL, "");
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(Dof, Dim);
#endif
   CalcVShape(Trans.GetIntPoint(), vshape);
   Mult(vshape, Trans.InverseJacobian(), shape);
}

void VectorFiniteElement::Project_RT(
   const double *nk, const Array<int> &d2n,
   VectorCoefficient &vc, ElementTransformation &Trans, Vector &dofs) const
{
   double vk[Geometry::MaxDim];
   const int sdim = Trans.GetSpaceDim();
   MFEM_ASSERT(vc.GetVDim() == sdim, "");
   Vector xk(vk, sdim);
   const bool square_J = (Dim == sdim);

   for (int k = 0; k < Dof; k++)
   {
      Trans.SetIntPoint(&Nodes.IntPoint(k));
      vc.Eval(xk, Trans, Nodes.IntPoint(k));
      // dof_k = nk^t adj(J) xk
      dofs(k) = Trans.AdjugateJacobian().InnerProduct(vk, nk + d2n[k]*Dim);
      if (!square_J) { dofs(k) /= Trans.Weight(); }
   }
}

void VectorFiniteElement::ProjectMatrixCoefficient_RT(
   const double *nk, const Array<int> &d2n,
   MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
{
   // project the rows of the matrix coefficient in an RT space

   const int sdim = T.GetSpaceDim();
   MFEM_ASSERT(mc.GetWidth() == sdim, "");
   const bool square_J = (Dim == sdim);
   DenseMatrix MQ(mc.GetHeight(), mc.GetWidth());
   Vector nk_phys(sdim), dofs_k(MQ.Height());
   MFEM_ASSERT(dofs.Size() == Dof*MQ.Height(), "");

   for (int k = 0; k < Dof; k++)
   {
      T.SetIntPoint(&Nodes.IntPoint(k));
      mc.Eval(MQ, T, Nodes.IntPoint(k));
      // nk_phys = adj(J)^t nk
      T.AdjugateJacobian().MultTranspose(nk + d2n[k]*Dim, nk_phys);
      if (!square_J) { nk_phys /= T.Weight(); }
      MQ.Mult(nk_phys, dofs_k);
      for (int r = 0; r < MQ.Height(); r++)
      {
         dofs(k+Dof*r) = dofs_k(r);
      }
   }
}

void VectorFiniteElement::Project_RT(
   const double *nk, const Array<int> &d2n, const FiniteElement &fe,
   ElementTransformation &Trans, DenseMatrix &I) const
{
   if (fe.GetRangeType() == SCALAR)
   {
      double vk[Geometry::MaxDim];
      Vector shape(fe.GetDof());
      int sdim = Trans.GetSpaceDim();

      I.SetSize(Dof, sdim*fe.GetDof());
      for (int k = 0; k < Dof; k++)
      {
         const IntegrationPoint &ip = Nodes.IntPoint(k);

         fe.CalcShape(ip, shape);
         Trans.SetIntPoint(&ip);
         Trans.AdjugateJacobian().MultTranspose(nk + d2n[k]*Dim, vk);
         if (fe.GetMapType() == INTEGRAL)
         {
            double w = 1.0/Trans.Weight();
            for (int d = 0; d < Dim; d++)
            {
               vk[d] *= w;
            }
         }

         for (int j = 0; j < shape.Size(); j++)
         {
            double s = shape(j);
            if (fabs(s) < 1e-12)
            {
               s = 0.0;
            }
            for (int d = 0; d < sdim; d++)
            {
               I(k,j+d*shape.Size()) = s*vk[d];
            }
         }
      }
   }
   else
   {
      mfem_error("VectorFiniteElement::Project_RT (fe version)");
   }
}

void VectorFiniteElement::ProjectGrad_RT(
   const double *nk, const Array<int> &d2n, const FiniteElement &fe,
   ElementTransformation &Trans, DenseMatrix &grad) const
{
   if (Dim != 2)
   {
      mfem_error("VectorFiniteElement::ProjectGrad_RT works only in 2D!");
   }

   DenseMatrix dshape(fe.GetDof(), fe.GetDim());
   Vector grad_k(fe.GetDof());
   double tk[2];

   grad.SetSize(Dof, fe.GetDof());
   for (int k = 0; k < Dof; k++)
   {
      fe.CalcDShape(Nodes.IntPoint(k), dshape);
      tk[0] = nk[d2n[k]*Dim+1];
      tk[1] = -nk[d2n[k]*Dim];
      dshape.Mult(tk, grad_k);
      for (int j = 0; j < grad_k.Size(); j++)
      {
         grad(k,j) = (fabs(grad_k(j)) < 1e-12) ? 0.0 : grad_k(j);
      }
   }
}

void VectorFiniteElement::ProjectCurl_ND(
   const double *tk, const Array<int> &d2t, const FiniteElement &fe,
   ElementTransformation &Trans, DenseMatrix &curl) const
{
#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(fe.GetDof(), Dim);
   DenseMatrix curlshape_J(fe.GetDof(), Dim);
   DenseMatrix J(Dim, Dim);
#else
   curlshape.SetSize(fe.GetDof(), Dim);
   curlshape_J.SetSize(fe.GetDof(), Dim);
   J.SetSize(Dim, Dim);
#endif

   Vector curl_k(fe.GetDof());

   curl.SetSize(Dof, fe.GetDof());
   for (int k = 0; k < Dof; k++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(k);

      // calculate J^t * J / |J|
      Trans.SetIntPoint(&ip);
      MultAtB(Trans.Jacobian(), Trans.Jacobian(), J);
      J *= 1.0 / Trans.Weight();

      // transform curl of shapes (rows) by J^t * J / |J|
      fe.CalcCurlShape(ip, curlshape);
      Mult(curlshape, J, curlshape_J);

      curlshape_J.Mult(tk + d2t[k]*Dim, curl_k);
      for (int j = 0; j < curl_k.Size(); j++)
      {
         curl(k,j) = (fabs(curl_k(j)) < 1e-12) ? 0.0 : curl_k(j);
      }
   }
}

void VectorFiniteElement::ProjectCurl_RT(
   const double *nk, const Array<int> &d2n, const FiniteElement &fe,
   ElementTransformation &Trans, DenseMatrix &curl) const
{
   DenseMatrix curl_shape(fe.GetDof(), Dim);
   Vector curl_k(fe.GetDof());

   curl.SetSize(Dof, fe.GetDof());
   for (int k = 0; k < Dof; k++)
   {
      fe.CalcCurlShape(Nodes.IntPoint(k), curl_shape);
      curl_shape.Mult(nk + d2n[k]*Dim, curl_k);
      for (int j = 0; j < curl_k.Size(); j++)
      {
         curl(k,j) = (fabs(curl_k(j)) < 1e-12) ? 0.0 : curl_k(j);
      }
   }
}

void VectorFiniteElement::Project_ND(
   const double *tk, const Array<int> &d2t,
   VectorCoefficient &vc, ElementTransformation &Trans, Vector &dofs) const
{
   double vk[Geometry::MaxDim];
   Vector xk(vk, vc.GetVDim());

   for (int k = 0; k < Dof; k++)
   {
      Trans.SetIntPoint(&Nodes.IntPoint(k));

      vc.Eval(xk, Trans, Nodes.IntPoint(k));
      // dof_k = xk^t J tk
      dofs(k) = Trans.Jacobian().InnerProduct(tk + d2t[k]*Dim, vk);
   }
}

void VectorFiniteElement::ProjectMatrixCoefficient_ND(
   const double *tk, const Array<int> &d2t,
   MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
{
   // project the rows of the matrix coefficient in an ND space

   const int sdim = T.GetSpaceDim();
   MFEM_ASSERT(mc.GetWidth() == sdim, "");
   DenseMatrix MQ(mc.GetHeight(), mc.GetWidth());
   Vector tk_phys(sdim), dofs_k(MQ.Height());
   MFEM_ASSERT(dofs.Size() == Dof*MQ.Height(), "");

   for (int k = 0; k < Dof; k++)
   {
      T.SetIntPoint(&Nodes.IntPoint(k));
      mc.Eval(MQ, T, Nodes.IntPoint(k));
      // tk_phys = J tk
      T.Jacobian().Mult(tk + d2t[k]*Dim, tk_phys);
      MQ.Mult(tk_phys, dofs_k);
      for (int r = 0; r < MQ.Height(); r++)
      {
         dofs(k+Dof*r) = dofs_k(r);
      }
   }
}

void VectorFiniteElement::Project_ND(
   const double *tk, const Array<int> &d2t, const FiniteElement &fe,
   ElementTransformation &Trans, DenseMatrix &I) const
{
   if (fe.GetRangeType() == SCALAR)
   {
      int sdim = Trans.GetSpaceDim();
      double vk[Geometry::MaxDim];
      Vector shape(fe.GetDof());

      I.SetSize(Dof, sdim*fe.GetDof());
      for (int k = 0; k < Dof; k++)
      {
         const IntegrationPoint &ip = Nodes.IntPoint(k);

         fe.CalcShape(ip, shape);
         Trans.SetIntPoint(&ip);
         Trans.Jacobian().Mult(tk + d2t[k]*Dim, vk);
         if (fe.GetMapType() == INTEGRAL)
         {
            double w = 1.0/Trans.Weight();
            for (int d = 0; d < sdim; d++)
            {
               vk[d] *= w;
            }
         }

         for (int j = 0; j < shape.Size(); j++)
         {
            double s = shape(j);
            if (fabs(s) < 1e-12)
            {
               s = 0.0;
            }
            for (int d = 0; d < sdim; d++)
            {
               I(k, j + d*shape.Size()) = s*vk[d];
            }
         }
      }
   }
   else
   {
      mfem_error("VectorFiniteElement::Project_ND (fe version)");
   }
}

void VectorFiniteElement::ProjectGrad_ND(
   const double *tk, const Array<int> &d2t, const FiniteElement &fe,
   ElementTransformation &Trans, DenseMatrix &grad) const
{
   MFEM_ASSERT(fe.GetMapType() == VALUE, "");

   DenseMatrix dshape(fe.GetDof(), fe.GetDim());
   Vector grad_k(fe.GetDof());

   grad.SetSize(Dof, fe.GetDof());
   for (int k = 0; k < Dof; k++)
   {
      fe.CalcDShape(Nodes.IntPoint(k), dshape);
      dshape.Mult(tk + d2t[k]*Dim, grad_k);
      for (int j = 0; j < grad_k.Size(); j++)
      {
         grad(k,j) = (fabs(grad_k(j)) < 1e-12) ? 0.0 : grad_k(j);
      }
   }
}

void VectorFiniteElement::LocalInterpolation_RT(
   const VectorFiniteElement &cfe, const double *nk, const Array<int> &d2n,
   ElementTransformation &Trans, DenseMatrix &I) const
{
   MFEM_ASSERT(MapType == cfe.GetMapType(), "");

   double vk[Geometry::MaxDim];
   Vector xk(vk, Dim);
   IntegrationPoint ip;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(cfe.GetDof(), cfe.GetDim());
#else
   DenseMatrix vshape(cfe.vshape.Data(), cfe.GetDof(), cfe.GetDim());
#endif
   I.SetSize(Dof, vshape.Height());

   // assuming Trans is linear; this should be ok for all refinement types
   Trans.SetIntPoint(&Geometries.GetCenter(GeomType));
   const DenseMatrix &adjJ = Trans.AdjugateJacobian();
   for (int k = 0; k < Dof; k++)
   {
      Trans.Transform(Nodes.IntPoint(k), xk);
      ip.Set3(vk);
      cfe.CalcVShape(ip, vshape);
      // xk = |J| J^{-t} n_k
      adjJ.MultTranspose(nk + d2n[k]*Dim, vk);
      // I_k = vshape_k.adj(J)^t.n_k, k=1,...,Dof
      for (int j = 0; j < vshape.Height(); j++)
      {
         double Ikj = 0.;
         for (int i = 0; i < Dim; i++)
         {
            Ikj += vshape(j, i) * vk[i];
         }
         I(k, j) = (fabs(Ikj) < 1e-12) ? 0.0 : Ikj;
      }
   }
}

void VectorFiniteElement::LocalInterpolation_ND(
   const VectorFiniteElement &cfe, const double *tk, const Array<int> &d2t,
   ElementTransformation &Trans, DenseMatrix &I) const
{
   double vk[Geometry::MaxDim];
   Vector xk(vk, Dim);
   IntegrationPoint ip;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(cfe.GetDof(), cfe.GetDim());
#else
   DenseMatrix vshape(cfe.vshape.Data(), cfe.GetDof(), cfe.GetDim());
#endif
   I.SetSize(Dof, vshape.Height());

   // assuming Trans is linear; this should be ok for all refinement types
   Trans.SetIntPoint(&Geometries.GetCenter(GeomType));
   const DenseMatrix &J = Trans.Jacobian();
   for (int k = 0; k < Dof; k++)
   {
      Trans.Transform(Nodes.IntPoint(k), xk);
      ip.Set3(vk);
      cfe.CalcVShape(ip, vshape);
      // xk = J t_k
      J.Mult(tk + d2t[k]*Dim, vk);
      // I_k = vshape_k.J.t_k, k=1,...,Dof
      for (int j = 0; j < vshape.Height(); j++)
      {
         double Ikj = 0.;
         for (int i = 0; i < Dim; i++)
         {
            Ikj += vshape(j, i) * vk[i];
         }
         I(k, j) = (fabs(Ikj) < 1e-12) ? 0.0 : Ikj;
      }
   }
}

void VectorFiniteElement::LocalRestriction_RT(
   const double *nk, const Array<int> &d2n, ElementTransformation &Trans,
   DenseMatrix &R) const
{
   double pt_data[Geometry::MaxDim];
   IntegrationPoint ip;
   Vector pt(pt_data, Dim);

#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(Dof, Dim);
#endif

   Trans.SetIntPoint(&Geometries.GetCenter(GeomType));
   const DenseMatrix &J = Trans.Jacobian();
   const double weight = Trans.Weight();
   for (int j = 0; j < Dof; j++)
   {
      InvertLinearTrans(Trans, Nodes.IntPoint(j), pt);
      ip.Set(pt_data, Dim);
      if (Geometries.CheckPoint(GeomType, ip)) // do we need an epsilon here?
      {
         CalcVShape(ip, vshape);
         J.MultTranspose(nk+Dim*d2n[j], pt_data);
         pt /= weight;
         for (int k = 0; k < Dof; k++)
         {
            double R_jk = 0.0;
            for (int d = 0; d < Dim; d++)
            {
               R_jk += vshape(k,d)*pt_data[d];
            }
            R(j,k) = R_jk;
         }
      }
      else
      {
         // Set the whole row to avoid valgrind warnings in R.Threshold().
         R.SetRow(j, infinity());
      }
   }
   R.Threshold(1e-12);
}

void VectorFiniteElement::LocalRestriction_ND(
   const double *tk, const Array<int> &d2t, ElementTransformation &Trans,
   DenseMatrix &R) const
{
   double pt_data[Geometry::MaxDim];
   IntegrationPoint ip;
   Vector pt(pt_data, Dim);

#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(Dof, Dim);
#endif

   Trans.SetIntPoint(&Geometries.GetCenter(GeomType));
   const DenseMatrix &Jinv = Trans.InverseJacobian();
   for (int j = 0; j < Dof; j++)
   {
      InvertLinearTrans(Trans, Nodes.IntPoint(j), pt);
      ip.Set(pt_data, Dim);
      if (Geometries.CheckPoint(GeomType, ip)) // do we need an epsilon here?
      {
         CalcVShape(ip, vshape);
         Jinv.Mult(tk+Dim*d2t[j], pt_data);
         for (int k = 0; k < Dof; k++)
         {
            double R_jk = 0.0;
            for (int d = 0; d < Dim; d++)
            {
               R_jk += vshape(k,d)*pt_data[d];
            }
            R(j,k) = R_jk;
         }
      }
      else
      {
         // Set the whole row to avoid valgrind warnings in R.Threshold().
         R.SetRow(j, infinity());
      }
   }
   R.Threshold(1e-12);
}


PointFiniteElement::PointFiniteElement()
   : NodalFiniteElement(0, Geometry::POINT, 1, 0)
{
   Nodes.IntPoint(0).x = 0.0;
}

void PointFiniteElement::CalcShape(const IntegrationPoint &ip,
                                   Vector &shape) const
{
   shape(0) = 1.;
}

void PointFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                    DenseMatrix &dshape) const
{
   // dshape is (1 x 0) - nothing to compute
}

Linear1DFiniteElement::Linear1DFiniteElement()
   : NodalFiniteElement(1, Geometry::SEGMENT, 2, 1)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(1).x = 1.0;
}

void Linear1DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                      Vector &shape) const
{
   shape(0) = 1. - ip.x;
   shape(1) = ip.x;
}

void Linear1DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                       DenseMatrix &dshape) const
{
   dshape(0,0) = -1.;
   dshape(1,0) =  1.;
}

Linear2DFiniteElement::Linear2DFiniteElement()
   : NodalFiniteElement(2, Geometry::TRIANGLE, 3, 1)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(2).x = 0.0;
   Nodes.IntPoint(2).y = 1.0;
}

void Linear2DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                      Vector &shape) const
{
   shape(0) = 1. - ip.x - ip.y;
   shape(1) = ip.x;
   shape(2) = ip.y;
}

void Linear2DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                       DenseMatrix &dshape) const
{
   dshape(0,0) = -1.; dshape(0,1) = -1.;
   dshape(1,0) =  1.; dshape(1,1) =  0.;
   dshape(2,0) =  0.; dshape(2,1) =  1.;
}

BiLinear2DFiniteElement::BiLinear2DFiniteElement()
   : NodalFiniteElement(2, Geometry::SQUARE, 4, 1, FunctionSpace::Qk)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(2).x = 1.0;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(3).x = 0.0;
   Nodes.IntPoint(3).y = 1.0;
}

void BiLinear2DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                        Vector &shape) const
{
   shape(0) = (1. - ip.x) * (1. - ip.y) ;
   shape(1) = ip.x * (1. - ip.y) ;
   shape(2) = ip.x * ip.y ;
   shape(3) = (1. - ip.x) * ip.y ;
}

void BiLinear2DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                         DenseMatrix &dshape) const
{
   dshape(0,0) = -1. + ip.y; dshape(0,1) = -1. + ip.x ;
   dshape(1,0) =  1. - ip.y; dshape(1,1) = -ip.x ;
   dshape(2,0) =  ip.y ;     dshape(2,1) = ip.x ;
   dshape(3,0) = -ip.y ;     dshape(3,1) = 1. - ip.x ;
}

void BiLinear2DFiniteElement::CalcHessian(
   const IntegrationPoint &ip, DenseMatrix &h) const
{
   h(0,0) = 0.;   h(0,1) =  1.;   h(0,2) = 0.;
   h(1,0) = 0.;   h(1,1) = -1.;   h(1,2) = 0.;
   h(2,0) = 0.;   h(2,1) =  1.;   h(2,2) = 0.;
   h(3,0) = 0.;   h(3,1) = -1.;   h(3,2) = 0.;
}


GaussLinear2DFiniteElement::GaussLinear2DFiniteElement()
   : NodalFiniteElement(2, Geometry::TRIANGLE, 3, 1, FunctionSpace::Pk)
{
   Nodes.IntPoint(0).x = 1./6.;
   Nodes.IntPoint(0).y = 1./6.;
   Nodes.IntPoint(1).x = 2./3.;
   Nodes.IntPoint(1).y = 1./6.;
   Nodes.IntPoint(2).x = 1./6.;
   Nodes.IntPoint(2).y = 2./3.;
}

void GaussLinear2DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                           Vector &shape) const
{
   const double x = ip.x, y = ip.y;

   shape(0) = 5./3. - 2. * (x + y);
   shape(1) = 2. * (x - 1./6.);
   shape(2) = 2. * (y - 1./6.);
}

void GaussLinear2DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                            DenseMatrix &dshape) const
{
   dshape(0,0) = -2.;  dshape(0,1) = -2.;
   dshape(1,0) =  2.;  dshape(1,1) =  0.;
   dshape(2,0) =  0.;  dshape(2,1) =  2.;
}

void GaussLinear2DFiniteElement::ProjectDelta(int vertex, Vector &dofs) const
{
   dofs(vertex)       = 2./3.;
   dofs((vertex+1)%3) = 1./6.;
   dofs((vertex+2)%3) = 1./6.;
}


// 0.5-0.5/sqrt(3) and 0.5+0.5/sqrt(3)
const double GaussBiLinear2DFiniteElement::p[] =
{ 0.2113248654051871177454256, 0.7886751345948128822545744 };

GaussBiLinear2DFiniteElement::GaussBiLinear2DFiniteElement()
   : NodalFiniteElement(2, Geometry::SQUARE, 4, 1, FunctionSpace::Qk)
{
   Nodes.IntPoint(0).x = p[0];
   Nodes.IntPoint(0).y = p[0];
   Nodes.IntPoint(1).x = p[1];
   Nodes.IntPoint(1).y = p[0];
   Nodes.IntPoint(2).x = p[1];
   Nodes.IntPoint(2).y = p[1];
   Nodes.IntPoint(3).x = p[0];
   Nodes.IntPoint(3).y = p[1];
}

void GaussBiLinear2DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                             Vector &shape) const
{
   const double x = ip.x, y = ip.y;

   shape(0) = 3. * (p[1] - x) * (p[1] - y);
   shape(1) = 3. * (x - p[0]) * (p[1] - y);
   shape(2) = 3. * (x - p[0]) * (y - p[0]);
   shape(3) = 3. * (p[1] - x) * (y - p[0]);
}

void GaussBiLinear2DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                              DenseMatrix &dshape) const
{
   const double x = ip.x, y = ip.y;

   dshape(0,0) = 3. * (y - p[1]);  dshape(0,1) = 3. * (x - p[1]);
   dshape(1,0) = 3. * (p[1] - y);  dshape(1,1) = 3. * (p[0] - x);
   dshape(2,0) = 3. * (y - p[0]);  dshape(2,1) = 3. * (x - p[0]);
   dshape(3,0) = 3. * (p[0] - y);  dshape(3,1) = 3. * (p[1] - x);
}

void GaussBiLinear2DFiniteElement::ProjectDelta(int vertex, Vector &dofs) const
{
#if 1
   dofs(vertex)       = p[1]*p[1];
   dofs((vertex+1)%4) = p[0]*p[1];
   dofs((vertex+2)%4) = p[0]*p[0];
   dofs((vertex+3)%4) = p[0]*p[1];
#else
   dofs = 1.0;
#endif
}


P1OnQuadFiniteElement::P1OnQuadFiniteElement()
   : NodalFiniteElement(2, Geometry::SQUARE, 3, 1, FunctionSpace::Qk)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(2).x = 0.0;
   Nodes.IntPoint(2).y = 1.0;
}

void P1OnQuadFiniteElement::CalcShape(const IntegrationPoint &ip,
                                      Vector &shape) const
{
   shape(0) = 1. - ip.x - ip.y;
   shape(1) = ip.x;
   shape(2) = ip.y;
}

void P1OnQuadFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                       DenseMatrix &dshape) const
{
   dshape(0,0) = -1.; dshape(0,1) = -1.;
   dshape(1,0) =  1.; dshape(1,1) =  0.;
   dshape(2,0) =  0.; dshape(2,1) =  1.;
}


Quad1DFiniteElement::Quad1DFiniteElement()
   : NodalFiniteElement(1, Geometry::SEGMENT, 3, 2)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(2).x = 0.5;
}

void Quad1DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                    Vector &shape) const
{
   double x = ip.x;
   double l1 = 1.0 - x, l2 = x, l3 = 2. * x - 1.;

   shape(0) = l1 * (-l3);
   shape(1) = l2 * l3;
   shape(2) = 4. * l1 * l2;
}

void Quad1DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                     DenseMatrix &dshape) const
{
   double x = ip.x;

   dshape(0,0) = 4. * x - 3.;
   dshape(1,0) = 4. * x - 1.;
   dshape(2,0) = 4. - 8. * x;
}


QuadPos1DFiniteElement::QuadPos1DFiniteElement()
   : PositiveFiniteElement(1, Geometry::SEGMENT, 3, 2)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(2).x = 0.5;
}

void QuadPos1DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                       Vector &shape) const
{
   const double x = ip.x, x1 = 1. - x;

   shape(0) = x1 * x1;
   shape(1) = x * x;
   shape(2) = 2. * x * x1;
}

void QuadPos1DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                        DenseMatrix &dshape) const
{
   const double x = ip.x;

   dshape(0,0) = 2. * x - 2.;
   dshape(1,0) = 2. * x;
   dshape(2,0) = 2. - 4. * x;
}

Quad2DFiniteElement::Quad2DFiniteElement()
   : NodalFiniteElement(2, Geometry::TRIANGLE, 6, 2)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(2).x = 0.0;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(3).x = 0.5;
   Nodes.IntPoint(3).y = 0.0;
   Nodes.IntPoint(4).x = 0.5;
   Nodes.IntPoint(4).y = 0.5;
   Nodes.IntPoint(5).x = 0.0;
   Nodes.IntPoint(5).y = 0.5;
}

void Quad2DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                    Vector &shape) const
{
   double x = ip.x, y = ip.y;
   double l1 = 1.-x-y, l2 = x, l3 = y;

   shape(0) = l1 * (2. * l1 - 1.);
   shape(1) = l2 * (2. * l2 - 1.);
   shape(2) = l3 * (2. * l3 - 1.);
   shape(3) = 4. * l1 * l2;
   shape(4) = 4. * l2 * l3;
   shape(5) = 4. * l3 * l1;
}

void Quad2DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                     DenseMatrix &dshape) const
{
   double x = ip.x, y = ip.y;

   dshape(0,0) =
      dshape(0,1) = 4. * (x + y) - 3.;

   dshape(1,0) = 4. * x - 1.;
   dshape(1,1) = 0.;

   dshape(2,0) = 0.;
   dshape(2,1) = 4. * y - 1.;

   dshape(3,0) = -4. * (2. * x + y - 1.);
   dshape(3,1) = -4. * x;

   dshape(4,0) = 4. * y;
   dshape(4,1) = 4. * x;

   dshape(5,0) = -4. * y;
   dshape(5,1) = -4. * (x + 2. * y - 1.);
}

void Quad2DFiniteElement::CalcHessian (const IntegrationPoint &ip,
                                       DenseMatrix &h) const
{
   h(0,0) = 4.;
   h(0,1) = 4.;
   h(0,2) = 4.;

   h(1,0) = 4.;
   h(1,1) = 0.;
   h(1,2) = 0.;

   h(2,0) = 0.;
   h(2,1) = 0.;
   h(2,2) = 4.;

   h(3,0) = -8.;
   h(3,1) = -4.;
   h(3,2) =  0.;

   h(4,0) = 0.;
   h(4,1) = 4.;
   h(4,2) = 0.;

   h(5,0) =  0.;
   h(5,1) = -4.;
   h(5,2) = -8.;
}

void Quad2DFiniteElement::ProjectDelta(int vertex, Vector &dofs) const
{
#if 0
   dofs = 1.;
#else
   dofs = 0.;
   dofs(vertex) = 1.;
   switch (vertex)
   {
      case 0: dofs(3) = 0.25; dofs(5) = 0.25; break;
      case 1: dofs(3) = 0.25; dofs(4) = 0.25; break;
      case 2: dofs(4) = 0.25; dofs(5) = 0.25; break;
   }
#endif
}


const double GaussQuad2DFiniteElement::p[] =
{ 0.0915762135097707434595714634022015, 0.445948490915964886318329253883051 };

GaussQuad2DFiniteElement::GaussQuad2DFiniteElement()
   : NodalFiniteElement(2, Geometry::TRIANGLE, 6, 2), A(6), D(6,2), pol(6)
{
   Nodes.IntPoint(0).x = p[0];
   Nodes.IntPoint(0).y = p[0];
   Nodes.IntPoint(1).x = 1. - 2. * p[0];
   Nodes.IntPoint(1).y = p[0];
   Nodes.IntPoint(2).x = p[0];
   Nodes.IntPoint(2).y = 1. - 2. * p[0];
   Nodes.IntPoint(3).x = p[1];
   Nodes.IntPoint(3).y = p[1];
   Nodes.IntPoint(4).x = 1. - 2. * p[1];
   Nodes.IntPoint(4).y = p[1];
   Nodes.IntPoint(5).x = p[1];
   Nodes.IntPoint(5).y = 1. - 2. * p[1];

   for (int i = 0; i < 6; i++)
   {
      const double x = Nodes.IntPoint(i).x, y = Nodes.IntPoint(i).y;
      A(0,i) = 1.;
      A(1,i) = x;
      A(2,i) = y;
      A(3,i) = x * x;
      A(4,i) = x * y;
      A(5,i) = y * y;
   }

   A.Invert();
}

void GaussQuad2DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                         Vector &shape) const
{
   const double x = ip.x, y = ip.y;
   pol(0) = 1.;
   pol(1) = x;
   pol(2) = y;
   pol(3) = x * x;
   pol(4) = x * y;
   pol(5) = y * y;

   A.Mult(pol, shape);
}

void GaussQuad2DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                          DenseMatrix &dshape) const
{
   const double x = ip.x, y = ip.y;
   D(0,0) = 0.;      D(0,1) = 0.;
   D(1,0) = 1.;      D(1,1) = 0.;
   D(2,0) = 0.;      D(2,1) = 1.;
   D(3,0) = 2. *  x; D(3,1) = 0.;
   D(4,0) = y;       D(4,1) = x;
   D(5,0) = 0.;      D(5,1) = 2. * y;

   Mult(A, D, dshape);
}


BiQuad2DFiniteElement::BiQuad2DFiniteElement()
   : NodalFiniteElement(2, Geometry::SQUARE, 9, 2, FunctionSpace::Qk)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(2).x = 1.0;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(3).x = 0.0;
   Nodes.IntPoint(3).y = 1.0;
   Nodes.IntPoint(4).x = 0.5;
   Nodes.IntPoint(4).y = 0.0;
   Nodes.IntPoint(5).x = 1.0;
   Nodes.IntPoint(5).y = 0.5;
   Nodes.IntPoint(6).x = 0.5;
   Nodes.IntPoint(6).y = 1.0;
   Nodes.IntPoint(7).x = 0.0;
   Nodes.IntPoint(7).y = 0.5;
   Nodes.IntPoint(8).x = 0.5;
   Nodes.IntPoint(8).y = 0.5;
}

void BiQuad2DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                      Vector &shape) const
{
   double x = ip.x, y = ip.y;
   double l1x, l2x, l3x, l1y, l2y, l3y;

   l1x = (x - 1.) * (2. * x - 1);
   l2x = 4. * x * (1. - x);
   l3x = x * (2. * x - 1.);
   l1y = (y - 1.) * (2. * y - 1);
   l2y = 4. * y * (1. - y);
   l3y = y * (2. * y - 1.);

   shape(0) = l1x * l1y;
   shape(4) = l2x * l1y;
   shape(1) = l3x * l1y;
   shape(7) = l1x * l2y;
   shape(8) = l2x * l2y;
   shape(5) = l3x * l2y;
   shape(3) = l1x * l3y;
   shape(6) = l2x * l3y;
   shape(2) = l3x * l3y;
}

void BiQuad2DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                       DenseMatrix &dshape) const
{
   double x = ip.x, y = ip.y;
   double l1x, l2x, l3x, l1y, l2y, l3y;
   double d1x, d2x, d3x, d1y, d2y, d3y;

   l1x = (x - 1.) * (2. * x - 1);
   l2x = 4. * x * (1. - x);
   l3x = x * (2. * x - 1.);
   l1y = (y - 1.) * (2. * y - 1);
   l2y = 4. * y * (1. - y);
   l3y = y * (2. * y - 1.);

   d1x = 4. * x - 3.;
   d2x = 4. - 8. * x;
   d3x = 4. * x - 1.;
   d1y = 4. * y - 3.;
   d2y = 4. - 8. * y;
   d3y = 4. * y - 1.;

   dshape(0,0) = d1x * l1y;
   dshape(0,1) = l1x * d1y;

   dshape(4,0) = d2x * l1y;
   dshape(4,1) = l2x * d1y;

   dshape(1,0) = d3x * l1y;
   dshape(1,1) = l3x * d1y;

   dshape(7,0) = d1x * l2y;
   dshape(7,1) = l1x * d2y;

   dshape(8,0) = d2x * l2y;
   dshape(8,1) = l2x * d2y;

   dshape(5,0) = d3x * l2y;
   dshape(5,1) = l3x * d2y;

   dshape(3,0) = d1x * l3y;
   dshape(3,1) = l1x * d3y;

   dshape(6,0) = d2x * l3y;
   dshape(6,1) = l2x * d3y;

   dshape(2,0) = d3x * l3y;
   dshape(2,1) = l3x * d3y;
}

void BiQuad2DFiniteElement::ProjectDelta(int vertex, Vector &dofs) const
{
#if 0
   dofs = 1.;
#else
   dofs = 0.;
   dofs(vertex) = 1.;
   switch (vertex)
   {
      case 0: dofs(4) = 0.25; dofs(7) = 0.25; break;
      case 1: dofs(4) = 0.25; dofs(5) = 0.25; break;
      case 2: dofs(5) = 0.25; dofs(6) = 0.25; break;
      case 3: dofs(6) = 0.25; dofs(7) = 0.25; break;
   }
   dofs(8) = 1./16.;
#endif
}

BiQuadPos2DFiniteElement::BiQuadPos2DFiniteElement()
   : PositiveFiniteElement(2, Geometry::SQUARE, 9, 2, FunctionSpace::Qk)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(2).x = 1.0;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(3).x = 0.0;
   Nodes.IntPoint(3).y = 1.0;
   Nodes.IntPoint(4).x = 0.5;
   Nodes.IntPoint(4).y = 0.0;
   Nodes.IntPoint(5).x = 1.0;
   Nodes.IntPoint(5).y = 0.5;
   Nodes.IntPoint(6).x = 0.5;
   Nodes.IntPoint(6).y = 1.0;
   Nodes.IntPoint(7).x = 0.0;
   Nodes.IntPoint(7).y = 0.5;
   Nodes.IntPoint(8).x = 0.5;
   Nodes.IntPoint(8).y = 0.5;
}

void BiQuadPos2DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                         Vector &shape) const
{
   double x = ip.x, y = ip.y;
   double l1x, l2x, l3x, l1y, l2y, l3y;

   l1x = (1. - x) * (1. - x);
   l2x = 2. * x * (1. - x);
   l3x = x * x;
   l1y = (1. - y) * (1. - y);
   l2y = 2. * y * (1. - y);
   l3y = y * y;

   shape(0) = l1x * l1y;
   shape(4) = l2x * l1y;
   shape(1) = l3x * l1y;
   shape(7) = l1x * l2y;
   shape(8) = l2x * l2y;
   shape(5) = l3x * l2y;
   shape(3) = l1x * l3y;
   shape(6) = l2x * l3y;
   shape(2) = l3x * l3y;
}

void BiQuadPos2DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                          DenseMatrix &dshape) const
{
   double x = ip.x, y = ip.y;
   double l1x, l2x, l3x, l1y, l2y, l3y;
   double d1x, d2x, d3x, d1y, d2y, d3y;

   l1x = (1. - x) * (1. - x);
   l2x = 2. * x * (1. - x);
   l3x = x * x;
   l1y = (1. - y) * (1. - y);
   l2y = 2. * y * (1. - y);
   l3y = y * y;

   d1x = 2. * x - 2.;
   d2x = 2. - 4. * x;
   d3x = 2. * x;
   d1y = 2. * y - 2.;
   d2y = 2. - 4. * y;
   d3y = 2. * y;

   dshape(0,0) = d1x * l1y;
   dshape(0,1) = l1x * d1y;

   dshape(4,0) = d2x * l1y;
   dshape(4,1) = l2x * d1y;

   dshape(1,0) = d3x * l1y;
   dshape(1,1) = l3x * d1y;

   dshape(7,0) = d1x * l2y;
   dshape(7,1) = l1x * d2y;

   dshape(8,0) = d2x * l2y;
   dshape(8,1) = l2x * d2y;

   dshape(5,0) = d3x * l2y;
   dshape(5,1) = l3x * d2y;

   dshape(3,0) = d1x * l3y;
   dshape(3,1) = l1x * d3y;

   dshape(6,0) = d2x * l3y;
   dshape(6,1) = l2x * d3y;

   dshape(2,0) = d3x * l3y;
   dshape(2,1) = l3x * d3y;
}

void BiQuadPos2DFiniteElement::GetLocalInterpolation(
   ElementTransformation &Trans, DenseMatrix &I) const
{
   double s[9];
   IntegrationPoint tr_ip;
   Vector xx(&tr_ip.x, 2), shape(s, 9);

   for (int i = 0; i < 9; i++)
   {
      Trans.Transform(Nodes.IntPoint(i), xx);
      CalcShape(tr_ip, shape);
      for (int j = 0; j < 9; j++)
         if (fabs(I(i,j) = s[j]) < 1.0e-12)
         {
            I(i,j) = 0.0;
         }
   }
   for (int i = 0; i < 9; i++)
   {
      double *d = &I(0,i);
      d[4] = 2. * d[4] - 0.5 * (d[0] + d[1]);
      d[5] = 2. * d[5] - 0.5 * (d[1] + d[2]);
      d[6] = 2. * d[6] - 0.5 * (d[2] + d[3]);
      d[7] = 2. * d[7] - 0.5 * (d[3] + d[0]);
      d[8] = 4. * d[8] - 0.5 * (d[4] + d[5] + d[6] + d[7]) -
             0.25 * (d[0] + d[1] + d[2] + d[3]);
   }
}

void BiQuadPos2DFiniteElement::Project(
   Coefficient &coeff, ElementTransformation &Trans, Vector &dofs) const
{
   double *d = dofs;

   for (int i = 0; i < 9; i++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(i);
      Trans.SetIntPoint(&ip);
      d[i] = coeff.Eval(Trans, ip);
   }
   d[4] = 2. * d[4] - 0.5 * (d[0] + d[1]);
   d[5] = 2. * d[5] - 0.5 * (d[1] + d[2]);
   d[6] = 2. * d[6] - 0.5 * (d[2] + d[3]);
   d[7] = 2. * d[7] - 0.5 * (d[3] + d[0]);
   d[8] = 4. * d[8] - 0.5 * (d[4] + d[5] + d[6] + d[7]) -
          0.25 * (d[0] + d[1] + d[2] + d[3]);
}

void BiQuadPos2DFiniteElement::Project (
   VectorCoefficient &vc, ElementTransformation &Trans,
   Vector &dofs) const
{
   double v[3];
   Vector x (v, vc.GetVDim());

   for (int i = 0; i < 9; i++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(i);
      Trans.SetIntPoint(&ip);
      vc.Eval (x, Trans, ip);
      for (int j = 0; j < x.Size(); j++)
      {
         dofs(9*j+i) = v[j];
      }
   }
   for (int j = 0; j < x.Size(); j++)
   {
      double *d = &dofs(9*j);

      d[4] = 2. * d[4] - 0.5 * (d[0] + d[1]);
      d[5] = 2. * d[5] - 0.5 * (d[1] + d[2]);
      d[6] = 2. * d[6] - 0.5 * (d[2] + d[3]);
      d[7] = 2. * d[7] - 0.5 * (d[3] + d[0]);
      d[8] = 4. * d[8] - 0.5 * (d[4] + d[5] + d[6] + d[7]) -
             0.25 * (d[0] + d[1] + d[2] + d[3]);
   }
}


GaussBiQuad2DFiniteElement::GaussBiQuad2DFiniteElement()
   : NodalFiniteElement(2, Geometry::SQUARE, 9, 2, FunctionSpace::Qk)
{
   const double p1 = 0.5*(1.-sqrt(3./5.));

   Nodes.IntPoint(0).x = p1;
   Nodes.IntPoint(0).y = p1;
   Nodes.IntPoint(4).x = 0.5;
   Nodes.IntPoint(4).y = p1;
   Nodes.IntPoint(1).x = 1.-p1;
   Nodes.IntPoint(1).y = p1;
   Nodes.IntPoint(7).x = p1;
   Nodes.IntPoint(7).y = 0.5;
   Nodes.IntPoint(8).x = 0.5;
   Nodes.IntPoint(8).y = 0.5;
   Nodes.IntPoint(5).x = 1.-p1;
   Nodes.IntPoint(5).y = 0.5;
   Nodes.IntPoint(3).x = p1;
   Nodes.IntPoint(3).y = 1.-p1;
   Nodes.IntPoint(6).x = 0.5;
   Nodes.IntPoint(6).y = 1.-p1;
   Nodes.IntPoint(2).x = 1.-p1;
   Nodes.IntPoint(2).y = 1.-p1;
}

void GaussBiQuad2DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                           Vector &shape) const
{
   const double a = sqrt(5./3.);
   const double p1 = 0.5*(1.-sqrt(3./5.));

   double x = a*(ip.x-p1), y = a*(ip.y-p1);
   double l1x, l2x, l3x, l1y, l2y, l3y;

   l1x = (x - 1.) * (2. * x - 1);
   l2x = 4. * x * (1. - x);
   l3x = x * (2. * x - 1.);
   l1y = (y - 1.) * (2. * y - 1);
   l2y = 4. * y * (1. - y);
   l3y = y * (2. * y - 1.);

   shape(0) = l1x * l1y;
   shape(4) = l2x * l1y;
   shape(1) = l3x * l1y;
   shape(7) = l1x * l2y;
   shape(8) = l2x * l2y;
   shape(5) = l3x * l2y;
   shape(3) = l1x * l3y;
   shape(6) = l2x * l3y;
   shape(2) = l3x * l3y;
}

void GaussBiQuad2DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                            DenseMatrix &dshape) const
{
   const double a = sqrt(5./3.);
   const double p1 = 0.5*(1.-sqrt(3./5.));

   double x = a*(ip.x-p1), y = a*(ip.y-p1);
   double l1x, l2x, l3x, l1y, l2y, l3y;
   double d1x, d2x, d3x, d1y, d2y, d3y;

   l1x = (x - 1.) * (2. * x - 1);
   l2x = 4. * x * (1. - x);
   l3x = x * (2. * x - 1.);
   l1y = (y - 1.) * (2. * y - 1);
   l2y = 4. * y * (1. - y);
   l3y = y * (2. * y - 1.);

   d1x = a * (4. * x - 3.);
   d2x = a * (4. - 8. * x);
   d3x = a * (4. * x - 1.);
   d1y = a * (4. * y - 3.);
   d2y = a * (4. - 8. * y);
   d3y = a * (4. * y - 1.);

   dshape(0,0) = d1x * l1y;
   dshape(0,1) = l1x * d1y;

   dshape(4,0) = d2x * l1y;
   dshape(4,1) = l2x * d1y;

   dshape(1,0) = d3x * l1y;
   dshape(1,1) = l3x * d1y;

   dshape(7,0) = d1x * l2y;
   dshape(7,1) = l1x * d2y;

   dshape(8,0) = d2x * l2y;
   dshape(8,1) = l2x * d2y;

   dshape(5,0) = d3x * l2y;
   dshape(5,1) = l3x * d2y;

   dshape(3,0) = d1x * l3y;
   dshape(3,1) = l1x * d3y;

   dshape(6,0) = d2x * l3y;
   dshape(6,1) = l2x * d3y;

   dshape(2,0) = d3x * l3y;
   dshape(2,1) = l3x * d3y;
}

BiCubic2DFiniteElement::BiCubic2DFiniteElement()
   : NodalFiniteElement (2, Geometry::SQUARE, 16, 3, FunctionSpace::Qk)
{
   Nodes.IntPoint(0).x = 0.;
   Nodes.IntPoint(0).y = 0.;
   Nodes.IntPoint(1).x = 1.;
   Nodes.IntPoint(1).y = 0.;
   Nodes.IntPoint(2).x = 1.;
   Nodes.IntPoint(2).y = 1.;
   Nodes.IntPoint(3).x = 0.;
   Nodes.IntPoint(3).y = 1.;
   Nodes.IntPoint(4).x = 1./3.;
   Nodes.IntPoint(4).y = 0.;
   Nodes.IntPoint(5).x = 2./3.;
   Nodes.IntPoint(5).y = 0.;
   Nodes.IntPoint(6).x = 1.;
   Nodes.IntPoint(6).y = 1./3.;
   Nodes.IntPoint(7).x = 1.;
   Nodes.IntPoint(7).y = 2./3.;
   Nodes.IntPoint(8).x = 2./3.;
   Nodes.IntPoint(8).y = 1.;
   Nodes.IntPoint(9).x = 1./3.;
   Nodes.IntPoint(9).y = 1.;
   Nodes.IntPoint(10).x = 0.;
   Nodes.IntPoint(10).y = 2./3.;
   Nodes.IntPoint(11).x = 0.;
   Nodes.IntPoint(11).y = 1./3.;
   Nodes.IntPoint(12).x = 1./3.;
   Nodes.IntPoint(12).y = 1./3.;
   Nodes.IntPoint(13).x = 2./3.;
   Nodes.IntPoint(13).y = 1./3.;
   Nodes.IntPoint(14).x = 1./3.;
   Nodes.IntPoint(14).y = 2./3.;
   Nodes.IntPoint(15).x = 2./3.;
   Nodes.IntPoint(15).y = 2./3.;
}

void BiCubic2DFiniteElement::CalcShape(
   const IntegrationPoint &ip, Vector &shape) const
{
   double x = ip.x, y = ip.y;

   double w1x, w2x, w3x, w1y, w2y, w3y;
   double l0x, l1x, l2x, l3x, l0y, l1y, l2y, l3y;

   w1x = x - 1./3.; w2x = x - 2./3.; w3x = x - 1.;
   w1y = y - 1./3.; w2y = y - 2./3.; w3y = y - 1.;

   l0x = (- 4.5) * w1x * w2x * w3x;
   l1x = ( 13.5) *   x * w2x * w3x;
   l2x = (-13.5) *   x * w1x * w3x;
   l3x = (  4.5) *   x * w1x * w2x;

   l0y = (- 4.5) * w1y * w2y * w3y;
   l1y = ( 13.5) *   y * w2y * w3y;
   l2y = (-13.5) *   y * w1y * w3y;
   l3y = (  4.5) *   y * w1y * w2y;

   shape(0)  = l0x * l0y;
   shape(1)  = l3x * l0y;
   shape(2)  = l3x * l3y;
   shape(3)  = l0x * l3y;
   shape(4)  = l1x * l0y;
   shape(5)  = l2x * l0y;
   shape(6)  = l3x * l1y;
   shape(7)  = l3x * l2y;
   shape(8)  = l2x * l3y;
   shape(9)  = l1x * l3y;
   shape(10) = l0x * l2y;
   shape(11) = l0x * l1y;
   shape(12) = l1x * l1y;
   shape(13) = l2x * l1y;
   shape(14) = l1x * l2y;
   shape(15) = l2x * l2y;
}

void BiCubic2DFiniteElement::CalcDShape(
   const IntegrationPoint &ip, DenseMatrix &dshape) const
{
   double x = ip.x, y = ip.y;

   double w1x, w2x, w3x, w1y, w2y, w3y;
   double l0x, l1x, l2x, l3x, l0y, l1y, l2y, l3y;
   double d0x, d1x, d2x, d3x, d0y, d1y, d2y, d3y;

   w1x = x - 1./3.; w2x = x - 2./3.; w3x = x - 1.;
   w1y = y - 1./3.; w2y = y - 2./3.; w3y = y - 1.;

   l0x = (- 4.5) * w1x * w2x * w3x;
   l1x = ( 13.5) *   x * w2x * w3x;
   l2x = (-13.5) *   x * w1x * w3x;
   l3x = (  4.5) *   x * w1x * w2x;

   l0y = (- 4.5) * w1y * w2y * w3y;
   l1y = ( 13.5) *   y * w2y * w3y;
   l2y = (-13.5) *   y * w1y * w3y;
   l3y = (  4.5) *   y * w1y * w2y;

   d0x = -5.5 + ( 18. - 13.5 * x) * x;
   d1x =  9.  + (-45. + 40.5 * x) * x;
   d2x = -4.5 + ( 36. - 40.5 * x) * x;
   d3x =  1.  + (- 9. + 13.5 * x) * x;

   d0y = -5.5 + ( 18. - 13.5 * y) * y;
   d1y =  9.  + (-45. + 40.5 * y) * y;
   d2y = -4.5 + ( 36. - 40.5 * y) * y;
   d3y =  1.  + (- 9. + 13.5 * y) * y;

   dshape( 0,0) = d0x * l0y;   dshape( 0,1) = l0x * d0y;
   dshape( 1,0) = d3x * l0y;   dshape( 1,1) = l3x * d0y;
   dshape( 2,0) = d3x * l3y;   dshape( 2,1) = l3x * d3y;
   dshape( 3,0) = d0x * l3y;   dshape( 3,1) = l0x * d3y;
   dshape( 4,0) = d1x * l0y;   dshape( 4,1) = l1x * d0y;
   dshape( 5,0) = d2x * l0y;   dshape( 5,1) = l2x * d0y;
   dshape( 6,0) = d3x * l1y;   dshape( 6,1) = l3x * d1y;
   dshape( 7,0) = d3x * l2y;   dshape( 7,1) = l3x * d2y;
   dshape( 8,0) = d2x * l3y;   dshape( 8,1) = l2x * d3y;
   dshape( 9,0) = d1x * l3y;   dshape( 9,1) = l1x * d3y;
   dshape(10,0) = d0x * l2y;   dshape(10,1) = l0x * d2y;
   dshape(11,0) = d0x * l1y;   dshape(11,1) = l0x * d1y;
   dshape(12,0) = d1x * l1y;   dshape(12,1) = l1x * d1y;
   dshape(13,0) = d2x * l1y;   dshape(13,1) = l2x * d1y;
   dshape(14,0) = d1x * l2y;   dshape(14,1) = l1x * d2y;
   dshape(15,0) = d2x * l2y;   dshape(15,1) = l2x * d2y;
}

void BiCubic2DFiniteElement::CalcHessian(
   const IntegrationPoint &ip, DenseMatrix &h) const
{
   double x = ip.x, y = ip.y;

   double w1x, w2x, w3x, w1y, w2y, w3y;
   double l0x, l1x, l2x, l3x, l0y, l1y, l2y, l3y;
   double d0x, d1x, d2x, d3x, d0y, d1y, d2y, d3y;
   double h0x, h1x, h2x, h3x, h0y, h1y, h2y, h3y;

   w1x = x - 1./3.; w2x = x - 2./3.; w3x = x - 1.;
   w1y = y - 1./3.; w2y = y - 2./3.; w3y = y - 1.;

   l0x = (- 4.5) * w1x * w2x * w3x;
   l1x = ( 13.5) *   x * w2x * w3x;
   l2x = (-13.5) *   x * w1x * w3x;
   l3x = (  4.5) *   x * w1x * w2x;

   l0y = (- 4.5) * w1y * w2y * w3y;
   l1y = ( 13.5) *   y * w2y * w3y;
   l2y = (-13.5) *   y * w1y * w3y;
   l3y = (  4.5) *   y * w1y * w2y;

   d0x = -5.5 + ( 18. - 13.5 * x) * x;
   d1x =  9.  + (-45. + 40.5 * x) * x;
   d2x = -4.5 + ( 36. - 40.5 * x) * x;
   d3x =  1.  + (- 9. + 13.5 * x) * x;

   d0y = -5.5 + ( 18. - 13.5 * y) * y;
   d1y =  9.  + (-45. + 40.5 * y) * y;
   d2y = -4.5 + ( 36. - 40.5 * y) * y;
   d3y =  1.  + (- 9. + 13.5 * y) * y;

   h0x = -27. * x + 18.;
   h1x =  81. * x - 45.;
   h2x = -81. * x + 36.;
   h3x =  27. * x -  9.;

   h0y = -27. * y + 18.;
   h1y =  81. * y - 45.;
   h2y = -81. * y + 36.;
   h3y =  27. * y -  9.;

   h( 0,0) = h0x * l0y;   h( 0,1) = d0x * d0y;   h( 0,2) = l0x * h0y;
   h( 1,0) = h3x * l0y;   h( 1,1) = d3x * d0y;   h( 1,2) = l3x * h0y;
   h( 2,0) = h3x * l3y;   h( 2,1) = d3x * d3y;   h( 2,2) = l3x * h3y;
   h( 3,0) = h0x * l3y;   h( 3,1) = d0x * d3y;   h( 3,2) = l0x * h3y;
   h( 4,0) = h1x * l0y;   h( 4,1) = d1x * d0y;   h( 4,2) = l1x * h0y;
   h( 5,0) = h2x * l0y;   h( 5,1) = d2x * d0y;   h( 5,2) = l2x * h0y;
   h( 6,0) = h3x * l1y;   h( 6,1) = d3x * d1y;   h( 6,2) = l3x * h1y;
   h( 7,0) = h3x * l2y;   h( 7,1) = d3x * d2y;   h( 7,2) = l3x * h2y;
   h( 8,0) = h2x * l3y;   h( 8,1) = d2x * d3y;   h( 8,2) = l2x * h3y;
   h( 9,0) = h1x * l3y;   h( 9,1) = d1x * d3y;   h( 9,2) = l1x * h3y;
   h(10,0) = h0x * l2y;   h(10,1) = d0x * d2y;   h(10,2) = l0x * h2y;
   h(11,0) = h0x * l1y;   h(11,1) = d0x * d1y;   h(11,2) = l0x * h1y;
   h(12,0) = h1x * l1y;   h(12,1) = d1x * d1y;   h(12,2) = l1x * h1y;
   h(13,0) = h2x * l1y;   h(13,1) = d2x * d1y;   h(13,2) = l2x * h1y;
   h(14,0) = h1x * l2y;   h(14,1) = d1x * d2y;   h(14,2) = l1x * h2y;
   h(15,0) = h2x * l2y;   h(15,1) = d2x * d2y;   h(15,2) = l2x * h2y;
}


Cubic1DFiniteElement::Cubic1DFiniteElement()
   : NodalFiniteElement(1, Geometry::SEGMENT, 4, 3)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(2).x = 0.33333333333333333333;
   Nodes.IntPoint(3).x = 0.66666666666666666667;
}

void Cubic1DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                     Vector &shape) const
{
   double x = ip.x;
   double l1 = x,
          l2 = (1.0-x),
          l3 = (0.33333333333333333333-x),
          l4 = (0.66666666666666666667-x);

   shape(0) =   4.5 * l2 * l3 * l4;
   shape(1) =   4.5 * l1 * l3 * l4;
   shape(2) =  13.5 * l1 * l2 * l4;
   shape(3) = -13.5 * l1 * l2 * l3;
}

void Cubic1DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                      DenseMatrix &dshape) const
{
   double x = ip.x;

   dshape(0,0) = -5.5 + x * (18. - 13.5 * x);
   dshape(1,0) = 1. - x * (9. - 13.5 * x);
   dshape(2,0) = 9. - x * (45. - 40.5 * x);
   dshape(3,0) = -4.5 + x * (36. - 40.5 * x);
}


Cubic2DFiniteElement::Cubic2DFiniteElement()
   : NodalFiniteElement(2, Geometry::TRIANGLE, 10, 3)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(2).x = 0.0;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(3).x = 0.33333333333333333333;
   Nodes.IntPoint(3).y = 0.0;
   Nodes.IntPoint(4).x = 0.66666666666666666667;
   Nodes.IntPoint(4).y = 0.0;
   Nodes.IntPoint(5).x = 0.66666666666666666667;
   Nodes.IntPoint(5).y = 0.33333333333333333333;
   Nodes.IntPoint(6).x = 0.33333333333333333333;
   Nodes.IntPoint(6).y = 0.66666666666666666667;
   Nodes.IntPoint(7).x = 0.0;
   Nodes.IntPoint(7).y = 0.66666666666666666667;
   Nodes.IntPoint(8).x = 0.0;
   Nodes.IntPoint(8).y = 0.33333333333333333333;
   Nodes.IntPoint(9).x = 0.33333333333333333333;
   Nodes.IntPoint(9).y = 0.33333333333333333333;
}

void Cubic2DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                     Vector &shape) const
{
   double x = ip.x, y = ip.y;
   double l1 = (-1. + x + y),
          lx = (-1. + 3.*x),
          ly = (-1. + 3.*y);

   shape(0) = -0.5*l1*(3.*l1 + 1.)*(3.*l1 + 2.);
   shape(1) =  0.5*x*(lx - 1.)*lx;
   shape(2) =  0.5*y*(-1. + ly)*ly;
   shape(3) =  4.5*x*l1*(3.*l1 + 1.);
   shape(4) = -4.5*x*lx*l1;
   shape(5) =  4.5*x*lx*y;
   shape(6) =  4.5*x*y*ly;
   shape(7) = -4.5*y*l1*ly;
   shape(8) =  4.5*y*l1*(1. + 3.*l1);
   shape(9) = -27.*x*y*l1;
}

void Cubic2DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                      DenseMatrix &dshape) const
{
   double x = ip.x, y = ip.y;

   dshape(0,0) =  0.5*(-11. + 36.*y - 9.*(x*(-4. + 3.*x) + 6.*x*y + 3.*y*y));
   dshape(1,0) =  1. + 4.5*x*(-2. + 3.*x);
   dshape(2,0) =  0.;
   dshape(3,0) =  4.5*(2. + 9.*x*x - 5.*y + 3.*y*y + 2.*x*(-5. + 6.*y));
   dshape(4,0) = -4.5*(1. - 1.*y + x*(-8. + 9.*x + 6.*y));
   dshape(5,0) =  4.5*(-1. + 6.*x)*y;
   dshape(6,0) =  4.5*y*(-1. + 3.*y);
   dshape(7,0) =  4.5*(1. - 3.*y)*y;
   dshape(8,0) =  4.5*y*(-5. + 6.*x + 6.*y);
   dshape(9,0) =  -27.*y*(-1. + 2.*x + y);

   dshape(0,1) =  0.5*(-11. + 36.*y - 9.*(x*(-4. + 3.*x) + 6.*x*y + 3.*y*y));
   dshape(1,1) =  0.;
   dshape(2,1) =  1. + 4.5*y*(-2. + 3.*y);
   dshape(3,1) =  4.5*x*(-5. + 6.*x + 6.*y);
   dshape(4,1) =  4.5*(1. - 3.*x)*x;
   dshape(5,1) =  4.5*x*(-1. + 3.*x);
   dshape(6,1) =  4.5*x*(-1. + 6.*y);
   dshape(7,1) = -4.5*(1. + x*(-1. + 6.*y) + y*(-8. + 9.*y));
   dshape(8,1) =  4.5*(2. + 3.*x*x + y*(-10. + 9.*y) + x*(-5. + 12.*y));
   dshape(9,1) = -27.*x*(-1. + x + 2.*y);
}

void Cubic2DFiniteElement::CalcHessian (const IntegrationPoint &ip,
                                        DenseMatrix &h) const
{
   double x = ip.x, y = ip.y;

   h(0,0) = 18.-27.*(x+y);
   h(0,1) = 18.-27.*(x+y);
   h(0,2) = 18.-27.*(x+y);

   h(1,0) = -9.+27.*x;
   h(1,1) = 0.;
   h(1,2) = 0.;

   h(2,0) = 0.;
   h(2,1) = 0.;
   h(2,2) = -9.+27.*y;

   h(3,0) = -45.+81.*x+54.*y;
   h(3,1) = -22.5+54.*x+27.*y;
   h(3,2) = 27.*x;

   h(4,0) = 36.-81.*x-27.*y;
   h(4,1) = 4.5-27.*x;
   h(4,2) = 0.;

   h(5,0) = 27.*y;
   h(5,1) = -4.5+27.*x;
   h(5,2) = 0.;

   h(6,0) = 0.;
   h(6,1) = -4.5+27.*y;
   h(6,2) = 27.*x;

   h(7,0) = 0.;
   h(7,1) = 4.5-27.*y;
   h(7,2) = 36.-27.*x-81.*y;

   h(8,0) = 27.*y;
   h(8,1) = -22.5+27.*x+54.*y;
   h(8,2) = -45.+54.*x+81.*y;

   h(9,0) = -54.*y;
   h(9,1) = 27.-54.*(x+y);
   h(9,2) = -54.*x;
}


Cubic3DFiniteElement::Cubic3DFiniteElement()
   : NodalFiniteElement(3, Geometry::TETRAHEDRON, 20, 3)
{
   Nodes.IntPoint(0).x = 0;
   Nodes.IntPoint(0).y = 0;
   Nodes.IntPoint(0).z = 0;
   Nodes.IntPoint(1).x = 1.;
   Nodes.IntPoint(1).y = 0;
   Nodes.IntPoint(1).z = 0;
   Nodes.IntPoint(2).x = 0;
   Nodes.IntPoint(2).y = 1.;
   Nodes.IntPoint(2).z = 0;
   Nodes.IntPoint(3).x = 0;
   Nodes.IntPoint(3).y = 0;
   Nodes.IntPoint(3).z = 1.;
   Nodes.IntPoint(4).x = 0.3333333333333333333333333333;
   Nodes.IntPoint(4).y = 0;
   Nodes.IntPoint(4).z = 0;
   Nodes.IntPoint(5).x = 0.6666666666666666666666666667;
   Nodes.IntPoint(5).y = 0;
   Nodes.IntPoint(5).z = 0;
   Nodes.IntPoint(6).x = 0;
   Nodes.IntPoint(6).y = 0.3333333333333333333333333333;
   Nodes.IntPoint(6).z = 0;
   Nodes.IntPoint(7).x = 0;
   Nodes.IntPoint(7).y = 0.6666666666666666666666666667;
   Nodes.IntPoint(7).z = 0;
   Nodes.IntPoint(8).x = 0;
   Nodes.IntPoint(8).y = 0;
   Nodes.IntPoint(8).z = 0.3333333333333333333333333333;
   Nodes.IntPoint(9).x = 0;
   Nodes.IntPoint(9).y = 0;
   Nodes.IntPoint(9).z = 0.6666666666666666666666666667;
   Nodes.IntPoint(10).x = 0.6666666666666666666666666667;
   Nodes.IntPoint(10).y = 0.3333333333333333333333333333;
   Nodes.IntPoint(10).z = 0;
   Nodes.IntPoint(11).x = 0.3333333333333333333333333333;
   Nodes.IntPoint(11).y = 0.6666666666666666666666666667;
   Nodes.IntPoint(11).z = 0;
   Nodes.IntPoint(12).x = 0.6666666666666666666666666667;
   Nodes.IntPoint(12).y = 0;
   Nodes.IntPoint(12).z = 0.3333333333333333333333333333;
   Nodes.IntPoint(13).x = 0.3333333333333333333333333333;
   Nodes.IntPoint(13).y = 0;
   Nodes.IntPoint(13).z = 0.6666666666666666666666666667;
   Nodes.IntPoint(14).x = 0;
   Nodes.IntPoint(14).y = 0.6666666666666666666666666667;
   Nodes.IntPoint(14).z = 0.3333333333333333333333333333;
   Nodes.IntPoint(15).x = 0;
   Nodes.IntPoint(15).y = 0.3333333333333333333333333333;
   Nodes.IntPoint(15).z = 0.6666666666666666666666666667;
   Nodes.IntPoint(16).x = 0.3333333333333333333333333333;
   Nodes.IntPoint(16).y = 0.3333333333333333333333333333;
   Nodes.IntPoint(16).z = 0.3333333333333333333333333333;
   Nodes.IntPoint(17).x = 0;
   Nodes.IntPoint(17).y = 0.3333333333333333333333333333;
   Nodes.IntPoint(17).z = 0.3333333333333333333333333333;
   Nodes.IntPoint(18).x = 0.3333333333333333333333333333;
   Nodes.IntPoint(18).y = 0;
   Nodes.IntPoint(18).z = 0.3333333333333333333333333333;
   Nodes.IntPoint(19).x = 0.3333333333333333333333333333;
   Nodes.IntPoint(19).y = 0.3333333333333333333333333333;
   Nodes.IntPoint(19).z = 0;
}

void Cubic3DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                     Vector &shape) const
{
   double x = ip.x, y = ip.y, z = ip.z;

   shape(0) = -((-1 + x + y + z)*(-2 + 3*x + 3*y + 3*z)*
                (-1 + 3*x + 3*y + 3*z))/2.;
   shape(4) = (9*x*(-1 + x + y + z)*(-2 + 3*x + 3*y + 3*z))/2.;
   shape(5) = (-9*x*(-1 + 3*x)*(-1 + x + y + z))/2.;
   shape(1) = (x*(2 + 9*(-1 + x)*x))/2.;
   shape(6) = (9*y*(-1 + x + y + z)*(-2 + 3*x + 3*y + 3*z))/2.;
   shape(19) = -27*x*y*(-1 + x + y + z);
   shape(10) = (9*x*(-1 + 3*x)*y)/2.;
   shape(7) = (-9*y*(-1 + 3*y)*(-1 + x + y + z))/2.;
   shape(11) = (9*x*y*(-1 + 3*y))/2.;
   shape(2) = (y*(2 + 9*(-1 + y)*y))/2.;
   shape(8) = (9*z*(-1 + x + y + z)*(-2 + 3*x + 3*y + 3*z))/2.;
   shape(18) = -27*x*z*(-1 + x + y + z);
   shape(12) = (9*x*(-1 + 3*x)*z)/2.;
   shape(17) = -27*y*z*(-1 + x + y + z);
   shape(16) = 27*x*y*z;
   shape(14) = (9*y*(-1 + 3*y)*z)/2.;
   shape(9) = (-9*z*(-1 + x + y + z)*(-1 + 3*z))/2.;
   shape(13) = (9*x*z*(-1 + 3*z))/2.;
   shape(15) = (9*y*z*(-1 + 3*z))/2.;
   shape(3) = (z*(2 + 9*(-1 + z)*z))/2.;
}

void Cubic3DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                      DenseMatrix &dshape) const
{
   double x = ip.x, y = ip.y, z = ip.z;

   dshape(0,0) = (-11 + 36*y + 36*z - 9*(3*pow(x,2) + 3*pow(y + z,2) +
                                         x*(-4 + 6*y + 6*z)))/2.;
   dshape(0,1) = (-11 + 36*y + 36*z - 9*(3*pow(x,2) + 3*pow(y + z,2) +
                                         x*(-4 + 6*y + 6*z)))/2.;
   dshape(0,2) = (-11 + 36*y + 36*z - 9*(3*pow(x,2) + 3*pow(y + z,2) +
                                         x*(-4 + 6*y + 6*z)))/2.;
   dshape(4,0) = (9*(9*pow(x,2) + (-1 + y + z)*(-2 + 3*y + 3*z) +
                     2*x*(-5 + 6*y + 6*z)))/2.;
   dshape(4,1) = (9*x*(-5 + 6*x + 6*y + 6*z))/2.;
   dshape(4,2) = (9*x*(-5 + 6*x + 6*y + 6*z))/2.;
   dshape(5,0) = (-9*(1 - y - z + x*(-8 + 9*x + 6*y + 6*z)))/2.;
   dshape(5,1) = (9*(1 - 3*x)*x)/2.;
   dshape(5,2) = (9*(1 - 3*x)*x)/2.;
   dshape(1,0) = 1 + (9*x*(-2 + 3*x))/2.;
   dshape(1,1) = 0;
   dshape(1,2) = 0;
   dshape(6,0) = (9*y*(-5 + 6*x + 6*y + 6*z))/2.;
   dshape(6,1) = (9*(2 + 3*pow(x,2) - 10*y - 5*z + 3*(y + z)*(3*y + z) +
                     x*(-5 + 12*y + 6*z)))/2.;
   dshape(6,2) = (9*y*(-5 + 6*x + 6*y + 6*z))/2.;
   dshape(19,0) = -27*y*(-1 + 2*x + y + z);
   dshape(19,1) = -27*x*(-1 + x + 2*y + z);
   dshape(19,2) = -27*x*y;
   dshape(10,0) = (9*(-1 + 6*x)*y)/2.;
   dshape(10,1) = (9*x*(-1 + 3*x))/2.;
   dshape(10,2) = 0;
   dshape(7,0) = (9*(1 - 3*y)*y)/2.;
   dshape(7,1) = (-9*(1 + x*(-1 + 6*y) - z + y*(-8 + 9*y + 6*z)))/2.;
   dshape(7,2) = (9*(1 - 3*y)*y)/2.;
   dshape(11,0) = (9*y*(-1 + 3*y))/2.;
   dshape(11,1) = (9*x*(-1 + 6*y))/2.;
   dshape(11,2) = 0;
   dshape(2,0) = 0;
   dshape(2,1) = 1 + (9*y*(-2 + 3*y))/2.;
   dshape(2,2) = 0;
   dshape(8,0) = (9*z*(-5 + 6*x + 6*y + 6*z))/2.;
   dshape(8,1) = (9*z*(-5 + 6*x + 6*y + 6*z))/2.;
   dshape(8,2) = (9*(2 + 3*pow(x,2) - 5*y - 10*z + 3*(y + z)*(y + 3*z) +
                     x*(-5 + 6*y + 12*z)))/2.;
   dshape(18,0) = -27*z*(-1 + 2*x + y + z);
   dshape(18,1) = -27*x*z;
   dshape(18,2) = -27*x*(-1 + x + y + 2*z);
   dshape(12,0) = (9*(-1 + 6*x)*z)/2.;
   dshape(12,1) = 0;
   dshape(12,2) = (9*x*(-1 + 3*x))/2.;
   dshape(17,0) = -27*y*z;
   dshape(17,1) = -27*z*(-1 + x + 2*y + z);
   dshape(17,2) = -27*y*(-1 + x + y + 2*z);
   dshape(16,0) = 27*y*z;
   dshape(16,1) = 27*x*z;
   dshape(16,2) = 27*x*y;
   dshape(14,0) = 0;
   dshape(14,1) = (9*(-1 + 6*y)*z)/2.;
   dshape(14,2) = (9*y*(-1 + 3*y))/2.;
   dshape(9,0) = (9*(1 - 3*z)*z)/2.;
   dshape(9,1) = (9*(1 - 3*z)*z)/2.;
   dshape(9,2) = (9*(-1 + x + y + 8*z - 6*(x + y)*z - 9*pow(z,2)))/2.;
   dshape(13,0) = (9*z*(-1 + 3*z))/2.;
   dshape(13,1) = 0;
   dshape(13,2) = (9*x*(-1 + 6*z))/2.;
   dshape(15,0) = 0;
   dshape(15,1) = (9*z*(-1 + 3*z))/2.;
   dshape(15,2) = (9*y*(-1 + 6*z))/2.;
   dshape(3,0) = 0;
   dshape(3,1) = 0;
   dshape(3,2) = 1 + (9*z*(-2 + 3*z))/2.;
}


P0TriangleFiniteElement::P0TriangleFiniteElement()
   : NodalFiniteElement(2, Geometry::TRIANGLE, 1, 0)
{
   Nodes.IntPoint(0).x = 0.333333333333333333;
   Nodes.IntPoint(0).y = 0.333333333333333333;
}

void P0TriangleFiniteElement::CalcShape(const IntegrationPoint &ip,
                                        Vector &shape) const
{
   shape(0) = 1.0;
}

void P0TriangleFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                         DenseMatrix &dshape) const
{
   dshape(0,0) = 0.0;
   dshape(0,1) = 0.0;
}


P0QuadFiniteElement::P0QuadFiniteElement()
   : NodalFiniteElement(2, Geometry::SQUARE, 1, 0, FunctionSpace::Qk)
{
   Nodes.IntPoint(0).x = 0.5;
   Nodes.IntPoint(0).y = 0.5;
}

void P0QuadFiniteElement::CalcShape(const IntegrationPoint &ip,
                                    Vector &shape) const
{
   shape(0) = 1.0;
}

void P0QuadFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                     DenseMatrix &dshape) const
{
   dshape(0,0) = 0.0;
   dshape(0,1) = 0.0;
}


Linear3DFiniteElement::Linear3DFiniteElement()
   : NodalFiniteElement(3, Geometry::TETRAHEDRON, 4, 1)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(0).z = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(1).z = 0.0;
   Nodes.IntPoint(2).x = 0.0;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(2).z = 0.0;
   Nodes.IntPoint(3).x = 0.0;
   Nodes.IntPoint(3).y = 0.0;
   Nodes.IntPoint(3).z = 1.0;
}

void Linear3DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                      Vector &shape) const
{
   shape(0) = 1. - ip.x - ip.y - ip.z;
   shape(1) = ip.x;
   shape(2) = ip.y;
   shape(3) = ip.z;
}

void Linear3DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                       DenseMatrix &dshape) const
{
   if (dshape.Height() == 4)
   {
      double *A = &dshape(0,0);
      A[0] = -1.; A[4] = -1.; A[8]  = -1.;
      A[1] =  1.; A[5] =  0.; A[9]  =  0.;
      A[2] =  0.; A[6] =  1.; A[10] =  0.;
      A[3] =  0.; A[7] =  0.; A[11] =  1.;
   }
   else
   {
      dshape(0,0) = -1.; dshape(0,1) = -1.; dshape(0,2) = -1.;
      dshape(1,0) =  1.; dshape(1,1) =  0.; dshape(1,2) =  0.;
      dshape(2,0) =  0.; dshape(2,1) =  1.; dshape(2,2) =  0.;
      dshape(3,0) =  0.; dshape(3,1) =  0.; dshape(3,2) =  1.;
   }
}

void Linear3DFiniteElement::GetFaceDofs (int face, int **dofs, int *ndofs)
const
{
   static int face_dofs[4][3] = {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}};

   *ndofs = 3;
   *dofs  = face_dofs[face];
}


Quadratic3DFiniteElement::Quadratic3DFiniteElement()
   : NodalFiniteElement(3, Geometry::TETRAHEDRON, 10, 2)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(0).z = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(1).z = 0.0;
   Nodes.IntPoint(2).x = 0.0;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(2).z = 0.0;
   Nodes.IntPoint(3).x = 0.0;
   Nodes.IntPoint(3).y = 0.0;
   Nodes.IntPoint(3).z = 1.0;
   Nodes.IntPoint(4).x = 0.5;
   Nodes.IntPoint(4).y = 0.0;
   Nodes.IntPoint(4).z = 0.0;
   Nodes.IntPoint(5).x = 0.0;
   Nodes.IntPoint(5).y = 0.5;
   Nodes.IntPoint(5).z = 0.0;
   Nodes.IntPoint(6).x = 0.0;
   Nodes.IntPoint(6).y = 0.0;
   Nodes.IntPoint(6).z = 0.5;
   Nodes.IntPoint(7).x = 0.5;
   Nodes.IntPoint(7).y = 0.5;
   Nodes.IntPoint(7).z = 0.0;
   Nodes.IntPoint(8).x = 0.5;
   Nodes.IntPoint(8).y = 0.0;
   Nodes.IntPoint(8).z = 0.5;
   Nodes.IntPoint(9).x = 0.0;
   Nodes.IntPoint(9).y = 0.5;
   Nodes.IntPoint(9).z = 0.5;
}

void Quadratic3DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                         Vector &shape) const
{
   double L0, L1, L2, L3;

   L0 = 1. - ip.x - ip.y - ip.z;
   L1 = ip.x;
   L2 = ip.y;
   L3 = ip.z;

   shape(0) = L0 * ( 2.0 * L0 - 1.0 );
   shape(1) = L1 * ( 2.0 * L1 - 1.0 );
   shape(2) = L2 * ( 2.0 * L2 - 1.0 );
   shape(3) = L3 * ( 2.0 * L3 - 1.0 );
   shape(4) = 4.0 * L0 * L1;
   shape(5) = 4.0 * L0 * L2;
   shape(6) = 4.0 * L0 * L3;
   shape(7) = 4.0 * L1 * L2;
   shape(8) = 4.0 * L1 * L3;
   shape(9) = 4.0 * L2 * L3;
}

void Quadratic3DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                          DenseMatrix &dshape) const
{
   double x, y, z, L0;

   x = ip.x;
   y = ip.y;
   z = ip.z;
   L0 = 1.0 - x - y - z;

   dshape(0,0) = dshape(0,1) = dshape(0,2) = 1.0 - 4.0 * L0;
   dshape(1,0) = -1.0 + 4.0 * x; dshape(1,1) = 0.0; dshape(1,2) = 0.0;
   dshape(2,0) = 0.0; dshape(2,1) = -1.0 + 4.0 * y; dshape(2,2) = 0.0;
   dshape(3,0) = dshape(3,1) = 0.0; dshape(3,2) = -1.0 + 4.0 * z;
   dshape(4,0) = 4.0 * (L0 - x); dshape(4,1) = dshape(4,2) = -4.0 * x;
   dshape(5,0) = dshape(5,2) = -4.0 * y; dshape(5,1) = 4.0 * (L0 - y);
   dshape(6,0) = dshape(6,1) = -4.0 * z; dshape(6,2) = 4.0 * (L0 - z);
   dshape(7,0) = 4.0 * y; dshape(7,1) = 4.0 * x; dshape(7,2) = 0.0;
   dshape(8,0) = 4.0 * z; dshape(8,1) = 0.0; dshape(8,2) = 4.0 * x;
   dshape(9,0) = 0.0; dshape(9,1) = 4.0 * z; dshape(9,2) = 4.0 * y;
}

TriLinear3DFiniteElement::TriLinear3DFiniteElement()
   : NodalFiniteElement(3, Geometry::CUBE, 8, 1, FunctionSpace::Qk)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(0).z = 0.0;

   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(1).z = 0.0;

   Nodes.IntPoint(2).x = 1.0;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(2).z = 0.0;

   Nodes.IntPoint(3).x = 0.0;
   Nodes.IntPoint(3).y = 1.0;
   Nodes.IntPoint(3).z = 0.0;

   Nodes.IntPoint(4).x = 0.0;
   Nodes.IntPoint(4).y = 0.0;
   Nodes.IntPoint(4).z = 1.0;

   Nodes.IntPoint(5).x = 1.0;
   Nodes.IntPoint(5).y = 0.0;
   Nodes.IntPoint(5).z = 1.0;

   Nodes.IntPoint(6).x = 1.0;
   Nodes.IntPoint(6).y = 1.0;
   Nodes.IntPoint(6).z = 1.0;

   Nodes.IntPoint(7).x = 0.0;
   Nodes.IntPoint(7).y = 1.0;
   Nodes.IntPoint(7).z = 1.0;
}

void TriLinear3DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                         Vector &shape) const
{
   double x = ip.x, y = ip.y, z = ip.z;
   double ox = 1.-x, oy = 1.-y, oz = 1.-z;

   shape(0) = ox * oy * oz;
   shape(1) =  x * oy * oz;
   shape(2) =  x *  y * oz;
   shape(3) = ox *  y * oz;
   shape(4) = ox * oy *  z;
   shape(5) =  x * oy *  z;
   shape(6) =  x *  y *  z;
   shape(7) = ox *  y *  z;
}

void TriLinear3DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                          DenseMatrix &dshape) const
{
   double x = ip.x, y = ip.y, z = ip.z;
   double ox = 1.-x, oy = 1.-y, oz = 1.-z;

   dshape(0,0) = - oy * oz;
   dshape(0,1) = - ox * oz;
   dshape(0,2) = - ox * oy;

   dshape(1,0) =   oy * oz;
   dshape(1,1) = -  x * oz;
   dshape(1,2) = -  x * oy;

   dshape(2,0) =    y * oz;
   dshape(2,1) =    x * oz;
   dshape(2,2) = -  x *  y;

   dshape(3,0) = -  y * oz;
   dshape(3,1) =   ox * oz;
   dshape(3,2) = - ox *  y;

   dshape(4,0) = - oy *  z;
   dshape(4,1) = - ox *  z;
   dshape(4,2) =   ox * oy;

   dshape(5,0) =   oy *  z;
   dshape(5,1) = -  x *  z;
   dshape(5,2) =    x * oy;

   dshape(6,0) =    y *  z;
   dshape(6,1) =    x *  z;
   dshape(6,2) =    x *  y;

   dshape(7,0) = -  y *  z;
   dshape(7,1) =   ox *  z;
   dshape(7,2) =   ox *  y;
}


P0SegmentFiniteElement::P0SegmentFiniteElement(int Ord)
   : NodalFiniteElement(1, Geometry::SEGMENT, 1, Ord)   // defaul Ord = 0
{
   Nodes.IntPoint(0).x = 0.5;
}

void P0SegmentFiniteElement::CalcShape(const IntegrationPoint &ip,
                                       Vector &shape) const
{
   shape(0) = 1.0;
}

void P0SegmentFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                        DenseMatrix &dshape) const
{
   dshape(0,0) = 0.0;
}

CrouzeixRaviartFiniteElement::CrouzeixRaviartFiniteElement()
   : NodalFiniteElement(2, Geometry::TRIANGLE, 3, 1)
{
   Nodes.IntPoint(0).x = 0.5;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(1).x = 0.5;
   Nodes.IntPoint(1).y = 0.5;
   Nodes.IntPoint(2).x = 0.0;
   Nodes.IntPoint(2).y = 0.5;
}

void CrouzeixRaviartFiniteElement::CalcShape(const IntegrationPoint &ip,
                                             Vector &shape) const
{
   shape(0) =  1.0 - 2.0 * ip.y;
   shape(1) = -1.0 + 2.0 * ( ip.x + ip.y );
   shape(2) =  1.0 - 2.0 * ip.x;
}

void CrouzeixRaviartFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                              DenseMatrix &dshape) const
{
   dshape(0,0) =  0.0; dshape(0,1) = -2.0;
   dshape(1,0) =  2.0; dshape(1,1) =  2.0;
   dshape(2,0) = -2.0; dshape(2,1) =  0.0;
}

CrouzeixRaviartQuadFiniteElement::CrouzeixRaviartQuadFiniteElement()
// the FunctionSpace should be rotated (45 degrees) Q_1
// i.e. the span of { 1, x, y, x^2 - y^2 }
   : NodalFiniteElement(2, Geometry::SQUARE, 4, 2, FunctionSpace::Qk)
{
   Nodes.IntPoint(0).x = 0.5;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.5;
   Nodes.IntPoint(2).x = 0.5;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(3).x = 0.0;
   Nodes.IntPoint(3).y = 0.5;
}

void CrouzeixRaviartQuadFiniteElement::CalcShape(const IntegrationPoint &ip,
                                                 Vector &shape) const
{
   const double l1 = ip.x+ip.y-0.5, l2 = 1.-l1, l3 = ip.x-ip.y+0.5, l4 = 1.-l3;

   shape(0) = l2 * l3;
   shape(1) = l1 * l3;
   shape(2) = l1 * l4;
   shape(3) = l2 * l4;
}

void CrouzeixRaviartQuadFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                                  DenseMatrix &dshape) const
{
   const double x2 = 2.*ip.x, y2 = 2.*ip.y;

   dshape(0,0) =  1. - x2; dshape(0,1) = -2. + y2;
   dshape(1,0) =       x2; dshape(1,1) =  1. - y2;
   dshape(2,0) =  1. - x2; dshape(2,1) =       y2;
   dshape(3,0) = -2. + x2; dshape(3,1) =  1. - y2;
}


RT0TriangleFiniteElement::RT0TriangleFiniteElement()
   : VectorFiniteElement(2, Geometry::TRIANGLE, 3, 1, H_DIV)
{
   Nodes.IntPoint(0).x = 0.5;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(1).x = 0.5;
   Nodes.IntPoint(1).y = 0.5;
   Nodes.IntPoint(2).x = 0.0;
   Nodes.IntPoint(2).y = 0.5;
}

void RT0TriangleFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                          DenseMatrix &shape) const
{
   double x = ip.x, y = ip.y;

   shape(0,0) = x;
   shape(0,1) = y - 1.;
   shape(1,0) = x;
   shape(1,1) = y;
   shape(2,0) = x - 1.;
   shape(2,1) = y;
}

void RT0TriangleFiniteElement::CalcDivShape(const IntegrationPoint &ip,
                                            Vector &divshape) const
{
   divshape(0) = 2.;
   divshape(1) = 2.;
   divshape(2) = 2.;
}

const double RT0TriangleFiniteElement::nk[3][2] =
{ {0, -1}, {1, 1}, {-1, 0} };

void RT0TriangleFiniteElement::GetLocalInterpolation (
   ElementTransformation &Trans, DenseMatrix &I) const
{
   int k, j;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(Dof, Dim);
   DenseMatrix Jinv(Dim);
#endif

#ifdef MFEM_DEBUG
   for (k = 0; k < 3; k++)
   {
      CalcVShape (Nodes.IntPoint(k), vshape);
      for (j = 0; j < 3; j++)
      {
         double d = vshape(j,0)*nk[k][0]+vshape(j,1)*nk[k][1];
         if (j == k) { d -= 1.0; }
         if (fabs(d) > 1.0e-12)
         {
            mfem::err << "RT0TriangleFiniteElement::GetLocalInterpolation (...)\n"
                      " k = " << k << ", j = " << j << ", d = " << d << endl;
            mfem_error();
         }
      }
   }
#endif

   IntegrationPoint ip;
   ip.x = ip.y = 0.0;
   Trans.SetIntPoint (&ip);
   // Trans must be linear
   // set Jinv = |J| J^{-t} = adj(J)^t
   CalcAdjugateTranspose (Trans.Jacobian(), Jinv);
   double vk[2];
   Vector xk (vk, 2);

   for (k = 0; k < 3; k++)
   {
      Trans.Transform (Nodes.IntPoint (k), xk);
      ip.x = vk[0]; ip.y = vk[1];
      CalcVShape (ip, vshape);
      //  vk = |J| J^{-t} nk
      vk[0] = Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1];
      vk[1] = Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1];
      for (j = 0; j < 3; j++)
         if (fabs (I(k,j) = vshape(j,0)*vk[0]+vshape(j,1)*vk[1]) < 1.0e-12)
         {
            I(k,j) = 0.0;
         }
   }
}

void RT0TriangleFiniteElement::Project (
   VectorCoefficient &vc, ElementTransformation &Trans,
   Vector &dofs) const
{
   double vk[2];
   Vector xk (vk, 2);
#ifdef MFEM_THREAD_SAFE
   DenseMatrix Jinv(Dim);
#endif

   for (int k = 0; k < 3; k++)
   {
      Trans.SetIntPoint (&Nodes.IntPoint (k));
      // set Jinv = |J| J^{-t} = adj(J)^t
      CalcAdjugateTranspose (Trans.Jacobian(), Jinv);

      vc.Eval (xk, Trans, Nodes.IntPoint (k));
      //  xk^t |J| J^{-t} nk
      dofs(k) = (vk[0] * ( Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1] ) +
                 vk[1] * ( Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1] ));
   }
}

RT0QuadFiniteElement::RT0QuadFiniteElement()
   : VectorFiniteElement(2, Geometry::SQUARE, 4, 1, H_DIV, FunctionSpace::Qk)
{
   Nodes.IntPoint(0).x = 0.5;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.5;
   Nodes.IntPoint(2).x = 0.5;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(3).x = 0.0;
   Nodes.IntPoint(3).y = 0.5;
}

void RT0QuadFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                      DenseMatrix &shape) const
{
   double x = ip.x, y = ip.y;

   shape(0,0) = 0;
   shape(0,1) = y - 1.;
   shape(1,0) = x;
   shape(1,1) = 0;
   shape(2,0) = 0;
   shape(2,1) = y;
   shape(3,0) = x - 1.;
   shape(3,1) = 0;
}

void RT0QuadFiniteElement::CalcDivShape(const IntegrationPoint &ip,
                                        Vector &divshape) const
{
   divshape(0) = 1.;
   divshape(1) = 1.;
   divshape(2) = 1.;
   divshape(3) = 1.;
}

const double RT0QuadFiniteElement::nk[4][2] =
{ {0, -1}, {1, 0}, {0, 1}, {-1, 0} };

void RT0QuadFiniteElement::GetLocalInterpolation (
   ElementTransformation &Trans, DenseMatrix &I) const
{
   int k, j;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(Dof, Dim);
   DenseMatrix Jinv(Dim);
#endif

#ifdef MFEM_DEBUG
   for (k = 0; k < 4; k++)
   {
      CalcVShape (Nodes.IntPoint(k), vshape);
      for (j = 0; j < 4; j++)
      {
         double d = vshape(j,0)*nk[k][0]+vshape(j,1)*nk[k][1];
         if (j == k) { d -= 1.0; }
         if (fabs(d) > 1.0e-12)
         {
            mfem::err << "RT0QuadFiniteElement::GetLocalInterpolation (...)\n"
                      " k = " << k << ", j = " << j << ", d = " << d << endl;
            mfem_error();
         }
      }
   }
#endif

   IntegrationPoint ip;
   ip.x = ip.y = 0.0;
   Trans.SetIntPoint (&ip);
   // Trans must be linear (more to have embedding?)
   // set Jinv = |J| J^{-t} = adj(J)^t
   CalcAdjugateTranspose (Trans.Jacobian(), Jinv);
   double vk[2];
   Vector xk (vk, 2);

   for (k = 0; k < 4; k++)
   {
      Trans.Transform (Nodes.IntPoint (k), xk);
      ip.x = vk[0]; ip.y = vk[1];
      CalcVShape (ip, vshape);
      //  vk = |J| J^{-t} nk
      vk[0] = Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1];
      vk[1] = Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1];
      for (j = 0; j < 4; j++)
         if (fabs (I(k,j) = vshape(j,0)*vk[0]+vshape(j,1)*vk[1]) < 1.0e-12)
         {
            I(k,j) = 0.0;
         }
   }
}

void RT0QuadFiniteElement::Project (
   VectorCoefficient &vc, ElementTransformation &Trans,
   Vector &dofs) const
{
   double vk[2];
   Vector xk (vk, 2);
#ifdef MFEM_THREAD_SAFE
   DenseMatrix Jinv(Dim);
#endif

   for (int k = 0; k < 4; k++)
   {
      Trans.SetIntPoint (&Nodes.IntPoint (k));
      // set Jinv = |J| J^{-t} = adj(J)^t
      CalcAdjugateTranspose (Trans.Jacobian(), Jinv);

      vc.Eval (xk, Trans, Nodes.IntPoint (k));
      //  xk^t |J| J^{-t} nk
      dofs(k) = (vk[0] * ( Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1] ) +
                 vk[1] * ( Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1] ));
   }
}

RT1TriangleFiniteElement::RT1TriangleFiniteElement()
   : VectorFiniteElement(2, Geometry::TRIANGLE, 8, 2, H_DIV)
{
   Nodes.IntPoint(0).x = 0.33333333333333333333;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(1).x = 0.66666666666666666667;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(2).x = 0.66666666666666666667;
   Nodes.IntPoint(2).y = 0.33333333333333333333;
   Nodes.IntPoint(3).x = 0.33333333333333333333;
   Nodes.IntPoint(3).y = 0.66666666666666666667;
   Nodes.IntPoint(4).x = 0.0;
   Nodes.IntPoint(4).y = 0.66666666666666666667;
   Nodes.IntPoint(5).x = 0.0;
   Nodes.IntPoint(5).y = 0.33333333333333333333;
   Nodes.IntPoint(6).x = 0.33333333333333333333;
   Nodes.IntPoint(6).y = 0.33333333333333333333;
   Nodes.IntPoint(7).x = 0.33333333333333333333;
   Nodes.IntPoint(7).y = 0.33333333333333333333;
}

void RT1TriangleFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                          DenseMatrix &shape) const
{
   double x = ip.x, y = ip.y;

   shape(0,0) = -2 * x * (-1 + x + 2 * y);
   shape(0,1) = -2 * (-1 + y) * (-1 + x + 2 * y);
   shape(1,0) =  2 * x * (x - y);
   shape(1,1) =  2 * (x - y) * (-1 + y);
   shape(2,0) =  2 * x * (-1 + 2 * x + y);
   shape(2,1) =  2 * y * (-1 + 2 * x + y);
   shape(3,0) =  2 * x * (-1 + x + 2 * y);
   shape(3,1) =  2 * y * (-1 + x + 2 * y);
   shape(4,0) = -2 * (-1 + x) * (x - y);
   shape(4,1) =  2 * y * (-x + y);
   shape(5,0) = -2 * (-1 + x) * (-1 + 2 * x + y);
   shape(5,1) = -2 * y * (-1 + 2 * x + y);
   shape(6,0) = -3 * x * (-2 + 2 * x + y);
   shape(6,1) = -3 * y * (-1 + 2 * x + y);
   shape(7,0) = -3 * x * (-1 + x + 2 * y);
   shape(7,1) = -3 * y * (-2 + x + 2 * y);
}

void RT1TriangleFiniteElement::CalcDivShape(const IntegrationPoint &ip,
                                            Vector &divshape) const
{
   double x = ip.x, y = ip.y;

   divshape(0) = -2 * (-4 + 3 * x + 6 * y);
   divshape(1) =  2 + 6 * x - 6 * y;
   divshape(2) = -4 + 12 * x + 6 * y;
   divshape(3) = -4 + 6 * x + 12 * y;
   divshape(4) =  2 - 6 * x + 6 * y;
   divshape(5) = -2 * (-4 + 6 * x + 3 * y);
   divshape(6) = -9 * (-1 + 2 * x + y);
   divshape(7) = -9 * (-1 + x + 2 * y);
}

const double RT1TriangleFiniteElement::nk[8][2] =
{
   { 0,-1}, { 0,-1},
   { 1, 1}, { 1, 1},
   {-1, 0}, {-1, 0},
   { 1, 0}, { 0, 1}
};

void RT1TriangleFiniteElement::GetLocalInterpolation (
   ElementTransformation &Trans, DenseMatrix &I) const
{
   int k, j;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(Dof, Dim);
   DenseMatrix Jinv(Dim);
#endif

#ifdef MFEM_DEBUG
   for (k = 0; k < 8; k++)
   {
      CalcVShape (Nodes.IntPoint(k), vshape);
      for (j = 0; j < 8; j++)
      {
         double d = vshape(j,0)*nk[k][0]+vshape(j,1)*nk[k][1];
         if (j == k) { d -= 1.0; }
         if (fabs(d) > 1.0e-12)
         {
            mfem::err << "RT1QuadFiniteElement::GetLocalInterpolation (...)\n"
                      " k = " << k << ", j = " << j << ", d = " << d << endl;
            mfem_error();
         }
      }
   }
#endif

   IntegrationPoint ip;
   ip.x = ip.y = 0.0;
   Trans.SetIntPoint (&ip);
   // Trans must be linear (more to have embedding?)
   // set Jinv = |J| J^{-t} = adj(J)^t
   CalcAdjugateTranspose (Trans.Jacobian(), Jinv);
   double vk[2];
   Vector xk (vk, 2);

   for (k = 0; k < 8; k++)
   {
      Trans.Transform (Nodes.IntPoint (k), xk);
      ip.x = vk[0]; ip.y = vk[1];
      CalcVShape (ip, vshape);
      //  vk = |J| J^{-t} nk
      vk[0] = Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1];
      vk[1] = Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1];
      for (j = 0; j < 8; j++)
         if (fabs (I(k,j) = vshape(j,0)*vk[0]+vshape(j,1)*vk[1]) < 1.0e-12)
         {
            I(k,j) = 0.0;
         }
   }
}

void RT1TriangleFiniteElement::Project (
   VectorCoefficient &vc, ElementTransformation &Trans, Vector &dofs) const
{
   double vk[2];
   Vector xk (vk, 2);
#ifdef MFEM_THREAD_SAFE
   DenseMatrix Jinv(Dim);
#endif

   for (int k = 0; k < 8; k++)
   {
      Trans.SetIntPoint (&Nodes.IntPoint (k));
      // set Jinv = |J| J^{-t} = adj(J)^t
      CalcAdjugateTranspose (Trans.Jacobian(), Jinv);

      vc.Eval (xk, Trans, Nodes.IntPoint (k));
      //  xk^t |J| J^{-t} nk
      dofs(k) = (vk[0] * ( Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1] ) +
                 vk[1] * ( Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1] ));
      dofs(k) *= 0.5;
   }
}

RT1QuadFiniteElement::RT1QuadFiniteElement()
   : VectorFiniteElement(2, Geometry::SQUARE, 12, 2, H_DIV, FunctionSpace::Qk)
{
   // y = 0
   Nodes.IntPoint(0).x  = 1./3.;
   Nodes.IntPoint(0).y  = 0.0;
   Nodes.IntPoint(1).x  = 2./3.;
   Nodes.IntPoint(1).y  = 0.0;
   // x = 1
   Nodes.IntPoint(2).x  = 1.0;
   Nodes.IntPoint(2).y  = 1./3.;
   Nodes.IntPoint(3).x  = 1.0;
   Nodes.IntPoint(3).y  = 2./3.;
   // y = 1
   Nodes.IntPoint(4).x  = 2./3.;
   Nodes.IntPoint(4).y  = 1.0;
   Nodes.IntPoint(5).x  = 1./3.;
   Nodes.IntPoint(5).y  = 1.0;
   // x = 0
   Nodes.IntPoint(6).x  = 0.0;
   Nodes.IntPoint(6).y  = 2./3.;
   Nodes.IntPoint(7).x  = 0.0;
   Nodes.IntPoint(7).y  = 1./3.;
   // x = 0.5 (interior)
   Nodes.IntPoint(8).x  = 0.5;
   Nodes.IntPoint(8).y  = 1./3.;
   Nodes.IntPoint(9).x  = 0.5;
   Nodes.IntPoint(9).y  = 2./3.;
   // y = 0.5 (interior)
   Nodes.IntPoint(10).x = 1./3.;
   Nodes.IntPoint(10).y = 0.5;
   Nodes.IntPoint(11).x = 2./3.;
   Nodes.IntPoint(11).y = 0.5;
}

void RT1QuadFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                      DenseMatrix &shape) const
{
   double x = ip.x, y = ip.y;

   // y = 0
   shape(0,0)  = 0;
   shape(0,1)  = -( 1. - 3.*y + 2.*y*y)*( 2. - 3.*x);
   shape(1,0)  = 0;
   shape(1,1)  = -( 1. - 3.*y + 2.*y*y)*(-1. + 3.*x);
   // x = 1
   shape(2,0)  = (-x + 2.*x*x)*( 2. - 3.*y);
   shape(2,1)  = 0;
   shape(3,0)  = (-x + 2.*x*x)*(-1. + 3.*y);
   shape(3,1)  = 0;
   // y = 1
   shape(4,0)  = 0;
   shape(4,1)  = (-y + 2.*y*y)*(-1. + 3.*x);
   shape(5,0)  = 0;
   shape(5,1)  = (-y + 2.*y*y)*( 2. - 3.*x);
   // x = 0
   shape(6,0)  = -(1. - 3.*x + 2.*x*x)*(-1. + 3.*y);
   shape(6,1)  = 0;
   shape(7,0)  = -(1. - 3.*x + 2.*x*x)*( 2. - 3.*y);
   shape(7,1)  = 0;
   // x = 0.5 (interior)
   shape(8,0)  = (4.*x - 4.*x*x)*( 2. - 3.*y);
   shape(8,1)  = 0;
   shape(9,0)  = (4.*x - 4.*x*x)*(-1. + 3.*y);
   shape(9,1)  = 0;
   // y = 0.5 (interior)
   shape(10,0) = 0;
   shape(10,1) = (4.*y - 4.*y*y)*( 2. - 3.*x);
   shape(11,0) = 0;
   shape(11,1) = (4.*y - 4.*y*y)*(-1. + 3.*x);
}

void RT1QuadFiniteElement::CalcDivShape(const IntegrationPoint &ip,
                                        Vector &divshape) const
{
   double x = ip.x, y = ip.y;

   divshape(0)  = -(-3. + 4.*y)*( 2. - 3.*x);
   divshape(1)  = -(-3. + 4.*y)*(-1. + 3.*x);
   divshape(2)  = (-1. + 4.*x)*( 2. - 3.*y);
   divshape(3)  = (-1. + 4.*x)*(-1. + 3.*y);
   divshape(4)  = (-1. + 4.*y)*(-1. + 3.*x);
   divshape(5)  = (-1. + 4.*y)*( 2. - 3.*x);
   divshape(6)  = -(-3. + 4.*x)*(-1. + 3.*y);
   divshape(7)  = -(-3. + 4.*x)*( 2. - 3.*y);
   divshape(8)  = ( 4. - 8.*x)*( 2. - 3.*y);
   divshape(9)  = ( 4. - 8.*x)*(-1. + 3.*y);
   divshape(10) = ( 4. - 8.*y)*( 2. - 3.*x);
   divshape(11) = ( 4. - 8.*y)*(-1. + 3.*x);
}

const double RT1QuadFiniteElement::nk[12][2] =
{
   // y = 0
   {0,-1}, {0,-1},
   // X = 1
   {1, 0}, {1, 0},
   // y = 1
   {0, 1}, {0, 1},
   // x = 0
   {-1,0}, {-1,0},
   // x = 0.5 (interior)
   {1, 0}, {1, 0},
   // y = 0.5 (interior)
   {0, 1}, {0, 1}
};

void RT1QuadFiniteElement::GetLocalInterpolation (
   ElementTransformation &Trans, DenseMatrix &I) const
{
   int k, j;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(Dof, Dim);
   DenseMatrix Jinv(Dim);
#endif

#ifdef MFEM_DEBUG
   for (k = 0; k < 12; k++)
   {
      CalcVShape (Nodes.IntPoint(k), vshape);
      for (j = 0; j < 12; j++)
      {
         double d = vshape(j,0)*nk[k][0]+vshape(j,1)*nk[k][1];
         if (j == k) { d -= 1.0; }
         if (fabs(d) > 1.0e-12)
         {
            mfem::err << "RT1QuadFiniteElement::GetLocalInterpolation (...)\n"
                      " k = " << k << ", j = " << j << ", d = " << d << endl;
            mfem_error();
         }
      }
   }
#endif

   IntegrationPoint ip;
   ip.x = ip.y = 0.0;
   Trans.SetIntPoint (&ip);
   // Trans must be linear (more to have embedding?)
   // set Jinv = |J| J^{-t} = adj(J)^t
   CalcAdjugateTranspose (Trans.Jacobian(), Jinv);
   double vk[2];
   Vector xk (vk, 2);

   for (k = 0; k < 12; k++)
   {
      Trans.Transform (Nodes.IntPoint (k), xk);
      ip.x = vk[0]; ip.y = vk[1];
      CalcVShape (ip, vshape);
      //  vk = |J| J^{-t} nk
      vk[0] = Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1];
      vk[1] = Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1];
      for (j = 0; j < 12; j++)
         if (fabs (I(k,j) = vshape(j,0)*vk[0]+vshape(j,1)*vk[1]) < 1.0e-12)
         {
            I(k,j) = 0.0;
         }
   }
}

void RT1QuadFiniteElement::Project (
   VectorCoefficient &vc, ElementTransformation &Trans, Vector &dofs) const
{
   double vk[2];
   Vector xk (vk, 2);
#ifdef MFEM_THREAD_SAFE
   DenseMatrix Jinv(Dim);
#endif

   for (int k = 0; k < 12; k++)
   {
      Trans.SetIntPoint (&Nodes.IntPoint (k));
      // set Jinv = |J| J^{-t} = adj(J)^t
      CalcAdjugateTranspose (Trans.Jacobian(), Jinv);

      vc.Eval (xk, Trans, Nodes.IntPoint (k));
      //  xk^t |J| J^{-t} nk
      dofs(k) = (vk[0] * ( Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1] ) +
                 vk[1] * ( Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1] ));
   }
}

const double RT2TriangleFiniteElement::M[15][15] =
{
   {
      0, -5.3237900077244501311, 5.3237900077244501311, 16.647580015448900262,
      0, 24.442740046346700787, -16.647580015448900262, -12.,
      -19.118950038622250656, -47.237900077244501311, 0, -34.414110069520051180,
      12., 30.590320061795601049, 15.295160030897800524
   },
   {
      0, 1.5, -1.5, -15., 0, 2.625, 15., 15., -4.125, 30., 0, -14.625, -15.,
      -15., 10.5
   },
   {
      0, -0.67620999227554986889, 0.67620999227554986889, 7.3524199845510997378,
      0, -3.4427400463467007866, -7.3524199845510997378, -12.,
      4.1189500386222506555, -0.76209992275549868892, 0, 7.4141100695200511800,
      12., -6.5903200617956010489, -3.2951600308978005244
   },
   {
      0, 0, 1.5, 0, 0, 1.5, -11.471370023173350393, 0, 2.4713700231733503933,
      -11.471370023173350393, 0, 2.4713700231733503933, 15.295160030897800524,
      0, -3.2951600308978005244
   },
   {
      0, 0, 4.875, 0, 0, 4.875, -16.875, 0, -16.875, -16.875, 0, -16.875, 10.5,
      36., 10.5
   },
   {
      0, 0, 1.5, 0, 0, 1.5, 2.4713700231733503933, 0, -11.471370023173350393,
      2.4713700231733503933, 0, -11.471370023173350393, -3.2951600308978005244,
      0, 15.295160030897800524
   },
   {
      -0.67620999227554986889, 0, -3.4427400463467007866, 0,
      7.3524199845510997378, 0.67620999227554986889, 7.4141100695200511800, 0,
      -0.76209992275549868892, 4.1189500386222506555, -12.,
      -7.3524199845510997378, -3.2951600308978005244, -6.5903200617956010489,
      12.
   },
   {
      1.5, 0, 2.625, 0, -15., -1.5, -14.625, 0, 30., -4.125, 15., 15., 10.5,
      -15., -15.
   },
   {
      -5.3237900077244501311, 0, 24.442740046346700787, 0, 16.647580015448900262,
      5.3237900077244501311, -34.414110069520051180, 0, -47.237900077244501311,
      -19.118950038622250656, -12., -16.647580015448900262, 15.295160030897800524,
      30.590320061795601049, 12.
   },
   { 0, 0, 18., 0, 0, 6., -42., 0, -30., -26., 0, -14., 24., 32., 8.},
   { 0, 0, 6., 0, 0, 18., -14., 0, -26., -30., 0, -42., 8., 32., 24.},
   { 0, 0, -6., 0, 0, -4., 30., 0, 4., 22., 0, 4., -24., -16., 0},
   { 0, 0, -4., 0, 0, -8., 20., 0, 8., 36., 0, 8., -16., -32., 0},
   { 0, 0, -8., 0, 0, -4., 8., 0, 36., 8., 0, 20., 0, -32., -16.},
   { 0, 0, -4., 0, 0, -6., 4., 0, 22., 4., 0, 30., 0, -16., -24.}
};

RT2TriangleFiniteElement::RT2TriangleFiniteElement()
   : VectorFiniteElement(2, Geometry::TRIANGLE, 15, 3, H_DIV)
{
   const double p = 0.11270166537925831148;

   Nodes.IntPoint(0).x = p;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(1).x = 0.5;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(2).x = 1.-p;
   Nodes.IntPoint(2).y = 0.0;
   Nodes.IntPoint(3).x = 1.-p;
   Nodes.IntPoint(3).y = p;
   Nodes.IntPoint(4).x = 0.5;
   Nodes.IntPoint(4).y = 0.5;
   Nodes.IntPoint(5).x = p;
   Nodes.IntPoint(5).y = 1.-p;
   Nodes.IntPoint(6).x = 0.0;
   Nodes.IntPoint(6).y = 1.-p;
   Nodes.IntPoint(7).x = 0.0;
   Nodes.IntPoint(7).y = 0.5;
   Nodes.IntPoint(8).x = 0.0;
   Nodes.IntPoint(8).y = p;
   Nodes.IntPoint(9).x  = 0.25;
   Nodes.IntPoint(9).y  = 0.25;
   Nodes.IntPoint(10).x = 0.25;
   Nodes.IntPoint(10).y = 0.25;
   Nodes.IntPoint(11).x = 0.5;
   Nodes.IntPoint(11).y = 0.25;
   Nodes.IntPoint(12).x = 0.5;
   Nodes.IntPoint(12).y = 0.25;
   Nodes.IntPoint(13).x = 0.25;
   Nodes.IntPoint(13).y = 0.5;
   Nodes.IntPoint(14).x = 0.25;
   Nodes.IntPoint(14).y = 0.5;
}

void RT2TriangleFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                          DenseMatrix &shape) const
{
   double x = ip.x, y = ip.y;

   double Bx[15] = {1., 0., x, 0., y, 0., x*x, 0., x*y, 0., y*y, 0., x*x*x,
                    x*x*y, x*y*y
                   };
   double By[15] = {0., 1., 0., x, 0., y, 0., x*x, 0., x*y, 0., y*y,
                    x*x*y, x*y*y, y*y*y
                   };

   for (int i = 0; i < 15; i++)
   {
      double cx = 0.0, cy = 0.0;
      for (int j = 0; j < 15; j++)
      {
         cx += M[i][j] * Bx[j];
         cy += M[i][j] * By[j];
      }
      shape(i,0) = cx;
      shape(i,1) = cy;
   }
}

void RT2TriangleFiniteElement::CalcDivShape(const IntegrationPoint &ip,
                                            Vector &divshape) const
{
   double x = ip.x, y = ip.y;

   double DivB[15] = {0., 0., 1., 0., 0., 1., 2.*x, 0., y, x, 0., 2.*y,
                      4.*x*x, 4.*x*y, 4.*y*y
                     };

   for (int i = 0; i < 15; i++)
   {
      double div = 0.0;
      for (int j = 0; j < 15; j++)
      {
         div += M[i][j] * DivB[j];
      }
      divshape(i) = div;
   }
}

const double RT2QuadFiniteElement::pt[4] = {0.,1./3.,2./3.,1.};

const double RT2QuadFiniteElement::dpt[3] = {0.25,0.5,0.75};

RT2QuadFiniteElement::RT2QuadFiniteElement()
   : VectorFiniteElement(2, Geometry::SQUARE, 24, 3, H_DIV, FunctionSpace::Qk)
{
   // y = 0 (pt[0])
   Nodes.IntPoint(0).x  = dpt[0];  Nodes.IntPoint(0).y  =  pt[0];
   Nodes.IntPoint(1).x  = dpt[1];  Nodes.IntPoint(1).y  =  pt[0];
   Nodes.IntPoint(2).x  = dpt[2];  Nodes.IntPoint(2).y  =  pt[0];
   // x = 1 (pt[3])
   Nodes.IntPoint(3).x  =  pt[3];  Nodes.IntPoint(3).y  = dpt[0];
   Nodes.IntPoint(4).x  =  pt[3];  Nodes.IntPoint(4).y  = dpt[1];
   Nodes.IntPoint(5).x  =  pt[3];  Nodes.IntPoint(5).y  = dpt[2];
   // y = 1 (pt[3])
   Nodes.IntPoint(6).x  = dpt[2];  Nodes.IntPoint(6).y  =  pt[3];
   Nodes.IntPoint(7).x  = dpt[1];  Nodes.IntPoint(7).y  =  pt[3];
   Nodes.IntPoint(8).x  = dpt[0];  Nodes.IntPoint(8).y  =  pt[3];
   // x = 0 (pt[0])
   Nodes.IntPoint(9).x  =  pt[0];  Nodes.IntPoint(9).y  = dpt[2];
   Nodes.IntPoint(10).x =  pt[0];  Nodes.IntPoint(10).y = dpt[1];
   Nodes.IntPoint(11).x =  pt[0];  Nodes.IntPoint(11).y = dpt[0];
   // x = pt[1] (interior)
   Nodes.IntPoint(12).x =  pt[1];  Nodes.IntPoint(12).y = dpt[0];
   Nodes.IntPoint(13).x =  pt[1];  Nodes.IntPoint(13).y = dpt[1];
   Nodes.IntPoint(14).x =  pt[1];  Nodes.IntPoint(14).y = dpt[2];
   // x = pt[2] (interior)
   Nodes.IntPoint(15).x =  pt[2];  Nodes.IntPoint(15).y = dpt[0];
   Nodes.IntPoint(16).x =  pt[2];  Nodes.IntPoint(16).y = dpt[1];
   Nodes.IntPoint(17).x =  pt[2];  Nodes.IntPoint(17).y = dpt[2];
   // y = pt[1] (interior)
   Nodes.IntPoint(18).x = dpt[0];  Nodes.IntPoint(18).y =  pt[1];
   Nodes.IntPoint(19).x = dpt[1];  Nodes.IntPoint(19).y =  pt[1];
   Nodes.IntPoint(20).x = dpt[2];  Nodes.IntPoint(20).y =  pt[1];
   // y = pt[2] (interior)
   Nodes.IntPoint(21).x = dpt[0];  Nodes.IntPoint(21).y =  pt[2];
   Nodes.IntPoint(22).x = dpt[1];  Nodes.IntPoint(22).y =  pt[2];
   Nodes.IntPoint(23).x = dpt[2];  Nodes.IntPoint(23).y =  pt[2];
}

void RT2QuadFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                      DenseMatrix &shape) const
{
   double x = ip.x, y = ip.y;

   double ax0 =  pt[0] - x;
   double ax1 =  pt[1] - x;
   double ax2 =  pt[2] - x;
   double ax3 =  pt[3] - x;

   double by0 = dpt[0] - y;
   double by1 = dpt[1] - y;
   double by2 = dpt[2] - y;

   double ay0 =  pt[0] - y;
   double ay1 =  pt[1] - y;
   double ay2 =  pt[2] - y;
   double ay3 =  pt[3] - y;

   double bx0 = dpt[0] - x;
   double bx1 = dpt[1] - x;
   double bx2 = dpt[2] - x;

   double A01 =  pt[0] -  pt[1];
   double A02 =  pt[0] -  pt[2];
   double A12 =  pt[1] -  pt[2];
   double A03 =  pt[0] -  pt[3];
   double A13 =  pt[1] -  pt[3];
   double A23 =  pt[2] -  pt[3];

   double B01 = dpt[0] - dpt[1];
   double B02 = dpt[0] - dpt[2];
   double B12 = dpt[1] - dpt[2];

   double tx0 =  (bx1*bx2)/(B01*B02);
   double tx1 = -(bx0*bx2)/(B01*B12);
   double tx2 =  (bx0*bx1)/(B02*B12);

   double ty0 =  (by1*by2)/(B01*B02);
   double ty1 = -(by0*by2)/(B01*B12);
   double ty2 =  (by0*by1)/(B02*B12);

   // y = 0 (p[0])
   shape(0,  0) =  0;
   shape(0,  1) =  (ay1*ay2*ay3)/(A01*A02*A03)*tx0;
   shape(1,  0) =  0;
   shape(1,  1) =  (ay1*ay2*ay3)/(A01*A02*A03)*tx1;
   shape(2,  0) =  0;
   shape(2,  1) =  (ay1*ay2*ay3)/(A01*A02*A03)*tx2;
   // x = 1 (p[3])
   shape(3,  0) =  (ax0*ax1*ax2)/(A03*A13*A23)*ty0;
   shape(3,  1) =  0;
   shape(4,  0) =  (ax0*ax1*ax2)/(A03*A13*A23)*ty1;
   shape(4,  1) =  0;
   shape(5,  0) =  (ax0*ax1*ax2)/(A03*A13*A23)*ty2;
   shape(5,  1) =  0;
   // y = 1 (p[3])
   shape(6,  0) =  0;
   shape(6,  1) =  (ay0*ay1*ay2)/(A03*A13*A23)*tx2;
   shape(7,  0) =  0;
   shape(7,  1) =  (ay0*ay1*ay2)/(A03*A13*A23)*tx1;
   shape(8,  0) =  0;
   shape(8,  1) =  (ay0*ay1*ay2)/(A03*A13*A23)*tx0;
   // x = 0 (p[0])
   shape(9,  0) =  (ax1*ax2*ax3)/(A01*A02*A03)*ty2;
   shape(9,  1) =  0;
   shape(10, 0) =  (ax1*ax2*ax3)/(A01*A02*A03)*ty1;
   shape(10, 1) =  0;
   shape(11, 0) =  (ax1*ax2*ax3)/(A01*A02*A03)*ty0;
   shape(11, 1) =  0;
   // x = p[1] (interior)
   shape(12, 0) =  (ax0*ax2*ax3)/(A01*A12*A13)*ty0;
   shape(12, 1) =  0;
   shape(13, 0) =  (ax0*ax2*ax3)/(A01*A12*A13)*ty1;
   shape(13, 1) =  0;
   shape(14, 0) =  (ax0*ax2*ax3)/(A01*A12*A13)*ty2;
   shape(14, 1) =  0;
   // x = p[2] (interior)
   shape(15, 0) = -(ax0*ax1*ax3)/(A02*A12*A23)*ty0;
   shape(15, 1) =  0;
   shape(16, 0) = -(ax0*ax1*ax3)/(A02*A12*A23)*ty1;
   shape(16, 1) =  0;
   shape(17, 0) = -(ax0*ax1*ax3)/(A02*A12*A23)*ty2;
   shape(17, 1) =  0;
   // y = p[1] (interior)
   shape(18, 0) =  0;
   shape(18, 1) =  (ay0*ay2*ay3)/(A01*A12*A13)*tx0;
   shape(19, 0) =  0;
   shape(19, 1) =  (ay0*ay2*ay3)/(A01*A12*A13)*tx1;
   shape(20, 0) =  0;
   shape(20, 1) =  (ay0*ay2*ay3)/(A01*A12*A13)*tx2;
   // y = p[2] (interior)
   shape(21, 0) =  0;
   shape(21, 1) = -(ay0*ay1*ay3)/(A02*A12*A23)*tx0;
   shape(22, 0) =  0;
   shape(22, 1) = -(ay0*ay1*ay3)/(A02*A12*A23)*tx1;
   shape(23, 0) =  0;
   shape(23, 1) = -(ay0*ay1*ay3)/(A02*A12*A23)*tx2;
}

void RT2QuadFiniteElement::CalcDivShape(const IntegrationPoint &ip,
                                        Vector &divshape) const
{
   double x = ip.x, y = ip.y;

   double a01 =  pt[0]*pt[1];
   double a02 =  pt[0]*pt[2];
   double a12 =  pt[1]*pt[2];
   double a03 =  pt[0]*pt[3];
   double a13 =  pt[1]*pt[3];
   double a23 =  pt[2]*pt[3];

   double bx0 = dpt[0] - x;
   double bx1 = dpt[1] - x;
   double bx2 = dpt[2] - x;

   double by0 = dpt[0] - y;
   double by1 = dpt[1] - y;
   double by2 = dpt[2] - y;

   double A01 =  pt[0] -  pt[1];
   double A02 =  pt[0] -  pt[2];
   double A12 =  pt[1] -  pt[2];
   double A03 =  pt[0] -  pt[3];
   double A13 =  pt[1] -  pt[3];
   double A23 =  pt[2] -  pt[3];

   double A012 = pt[0] + pt[1] + pt[2];
   double A013 = pt[0] + pt[1] + pt[3];
   double A023 = pt[0] + pt[2] + pt[3];
   double A123 = pt[1] + pt[2] + pt[3];

   double B01 = dpt[0] - dpt[1];
   double B02 = dpt[0] - dpt[2];
   double B12 = dpt[1] - dpt[2];

   double tx0 =  (bx1*bx2)/(B01*B02);
   double tx1 = -(bx0*bx2)/(B01*B12);
   double tx2 =  (bx0*bx1)/(B02*B12);

   double ty0 =  (by1*by2)/(B01*B02);
   double ty1 = -(by0*by2)/(B01*B12);
   double ty2 =  (by0*by1)/(B02*B12);

   // y = 0 (p[0])
   divshape(0)  = -(a12 + a13 + a23 - 2.*A123*y + 3.*y*y)/(A01*A02*A03)*tx0;
   divshape(1)  = -(a12 + a13 + a23 - 2.*A123*y + 3.*y*y)/(A01*A02*A03)*tx1;
   divshape(2)  = -(a12 + a13 + a23 - 2.*A123*y + 3.*y*y)/(A01*A02*A03)*tx2;
   // x = 1 (p[3])
   divshape(3)  = -(a01 + a02 + a12 - 2.*A012*x + 3.*x*x)/(A03*A13*A23)*ty0;
   divshape(4)  = -(a01 + a02 + a12 - 2.*A012*x + 3.*x*x)/(A03*A13*A23)*ty1;
   divshape(5)  = -(a01 + a02 + a12 - 2.*A012*x + 3.*x*x)/(A03*A13*A23)*ty2;
   // y = 1 (p[3])
   divshape(6)  = -(a01 + a02 + a12 - 2.*A012*y + 3.*y*y)/(A03*A13*A23)*tx2;
   divshape(7)  = -(a01 + a02 + a12 - 2.*A012*y + 3.*y*y)/(A03*A13*A23)*tx1;
   divshape(8)  = -(a01 + a02 + a12 - 2.*A012*y + 3.*y*y)/(A03*A13*A23)*tx0;
   // x = 0 (p[0])
   divshape(9)  = -(a12 + a13 + a23 - 2.*A123*x + 3.*x*x)/(A01*A02*A03)*ty2;
   divshape(10) = -(a12 + a13 + a23 - 2.*A123*x + 3.*x*x)/(A01*A02*A03)*ty1;
   divshape(11) = -(a12 + a13 + a23 - 2.*A123*x + 3.*x*x)/(A01*A02*A03)*ty0;
   // x = p[1] (interior)
   divshape(12) = -(a02 + a03 + a23 - 2.*A023*x + 3.*x*x)/(A01*A12*A13)*ty0;
   divshape(13) = -(a02 + a03 + a23 - 2.*A023*x + 3.*x*x)/(A01*A12*A13)*ty1;
   divshape(14) = -(a02 + a03 + a23 - 2.*A023*x + 3.*x*x)/(A01*A12*A13)*ty2;
   // x = p[2] (interior)
   divshape(15) =  (a01 + a03 + a13 - 2.*A013*x + 3.*x*x)/(A02*A12*A23)*ty0;
   divshape(16) =  (a01 + a03 + a13 - 2.*A013*x + 3.*x*x)/(A02*A12*A23)*ty1;
   divshape(17) =  (a01 + a03 + a13 - 2.*A013*x + 3.*x*x)/(A02*A12*A23)*ty2;
   // y = p[1] (interior)
   divshape(18) = -(a02 + a03 + a23 - 2.*A023*y + 3.*y*y)/(A01*A12*A13)*tx0;
   divshape(19) = -(a02 + a03 + a23 - 2.*A023*y + 3.*y*y)/(A01*A12*A13)*tx1;
   divshape(20) = -(a02 + a03 + a23 - 2.*A023*y + 3.*y*y)/(A01*A12*A13)*tx2;
   // y = p[2] (interior)
   divshape(21) =  (a01 + a03 + a13 - 2.*A013*y + 3.*y*y)/(A02*A12*A23)*tx0;
   divshape(22) =  (a01 + a03 + a13 - 2.*A013*y + 3.*y*y)/(A02*A12*A23)*tx1;
   divshape(23) =  (a01 + a03 + a13 - 2.*A013*y + 3.*y*y)/(A02*A12*A23)*tx2;
}

const double RT2QuadFiniteElement::nk[24][2] =
{
   // y = 0
   {0,-1}, {0,-1}, {0,-1},
   // x = 1
   {1, 0}, {1, 0}, {1, 0},
   // y = 1
   {0, 1}, {0, 1}, {0, 1},
   // x = 0
   {-1,0}, {-1,0}, {-1,0},
   // x = p[1] (interior)
   {1, 0}, {1, 0}, {1, 0},
   // x = p[2] (interior)
   {1, 0}, {1, 0}, {1, 0},
   // y = p[1] (interior)
   {0, 1}, {0, 1}, {0, 1},
   // y = p[1] (interior)
   {0, 1}, {0, 1}, {0, 1}
};

void RT2QuadFiniteElement::GetLocalInterpolation (
   ElementTransformation &Trans, DenseMatrix &I) const
{
   int k, j;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(Dof, Dim);
   DenseMatrix Jinv(Dim);
#endif

#ifdef MFEM_DEBUG
   for (k = 0; k < 24; k++)
   {
      CalcVShape (Nodes.IntPoint(k), vshape);
      for (j = 0; j < 24; j++)
      {
         double d = vshape(j,0)*nk[k][0]+vshape(j,1)*nk[k][1];
         if (j == k) { d -= 1.0; }
         if (fabs(d) > 1.0e-12)
         {
            mfem::err << "RT2QuadFiniteElement::GetLocalInterpolation (...)\n"
                      " k = " << k << ", j = " << j << ", d = " << d << endl;
            mfem_error();
         }
      }
   }
#endif

   IntegrationPoint ip;
   ip.x = ip.y = 0.0;
   Trans.SetIntPoint (&ip);
   // Trans must be linear (more to have embedding?)
   // set Jinv = |J| J^{-t} = adj(J)^t
   CalcAdjugateTranspose (Trans.Jacobian(), Jinv);
   double vk[2];
   Vector xk (vk, 2);

   for (k = 0; k < 24; k++)
   {
      Trans.Transform (Nodes.IntPoint (k), xk);
      ip.x = vk[0]; ip.y = vk[1];
      CalcVShape (ip, vshape);
      //  vk = |J| J^{-t} nk
      vk[0] = Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1];
      vk[1] = Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1];
      for (j = 0; j < 24; j++)
         if (fabs (I(k,j) = vshape(j,0)*vk[0]+vshape(j,1)*vk[1]) < 1.0e-12)
         {
            I(k,j) = 0.0;
         }
   }
}

void RT2QuadFiniteElement::Project (
   VectorCoefficient &vc, ElementTransformation &Trans, Vector &dofs) const
{
   double vk[2];
   Vector xk (vk, 2);
#ifdef MFEM_THREAD_SAFE
   DenseMatrix Jinv(Dim);
#endif

   for (int k = 0; k < 24; k++)
   {
      Trans.SetIntPoint (&Nodes.IntPoint (k));
      // set Jinv = |J| J^{-t} = adj(J)^t
      CalcAdjugateTranspose (Trans.Jacobian(), Jinv);

      vc.Eval (xk, Trans, Nodes.IntPoint (k));
      //  xk^t |J| J^{-t} nk
      dofs(k) = (vk[0] * ( Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1] ) +
                 vk[1] * ( Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1] ));
   }
}

P1SegmentFiniteElement::P1SegmentFiniteElement()
   : NodalFiniteElement(1, Geometry::SEGMENT, 2, 1)
{
   Nodes.IntPoint(0).x = 0.33333333333333333333;
   Nodes.IntPoint(1).x = 0.66666666666666666667;
}

void P1SegmentFiniteElement::CalcShape(const IntegrationPoint &ip,
                                       Vector &shape) const
{
   double x = ip.x;

   shape(0) = 2. - 3. * x;
   shape(1) = 3. * x - 1.;
}

void P1SegmentFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                        DenseMatrix &dshape) const
{
   dshape(0,0) = -3.;
   dshape(1,0) =  3.;
}


P2SegmentFiniteElement::P2SegmentFiniteElement()
   : NodalFiniteElement(1, Geometry::SEGMENT, 3, 2)
{
   const double p = 0.11270166537925831148;

   Nodes.IntPoint(0).x = p;
   Nodes.IntPoint(1).x = 0.5;
   Nodes.IntPoint(2).x = 1.-p;
}

void P2SegmentFiniteElement::CalcShape(const IntegrationPoint &ip,
                                       Vector &shape) const
{
   const double p = 0.11270166537925831148;
   const double w = 1./((1-2*p)*(1-2*p));
   double x = ip.x;

   shape(0) = (2*x-1)*(x-1+p)*w;
   shape(1) = 4*(x-1+p)*(p-x)*w;
   shape(2) = (2*x-1)*(x-p)*w;
}

void P2SegmentFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                        DenseMatrix &dshape) const
{
   const double p = 0.11270166537925831148;
   const double w = 1./((1-2*p)*(1-2*p));
   double x = ip.x;

   dshape(0,0) = (-3+4*x+2*p)*w;
   dshape(1,0) = (4-8*x)*w;
   dshape(2,0) = (-1+4*x-2*p)*w;
}


Lagrange1DFiniteElement::Lagrange1DFiniteElement(int degree)
   : NodalFiniteElement(1, Geometry::SEGMENT, degree+1, degree)
{
   int i, m = degree;

   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   for (i = 1; i < m; i++)
   {
      Nodes.IntPoint(i+1).x = double(i) / m;
   }

   rwk.SetSize(degree+1);
#ifndef MFEM_THREAD_SAFE
   rxxk.SetSize(degree+1);
#endif

   rwk(0) = 1.0;
   for (i = 1; i <= m; i++)
   {
      rwk(i) = rwk(i-1) * ( (double)(m) / (double)(i) );
   }
   for (i = 0; i < m/2+1; i++)
   {
      rwk(m-i) = ( rwk(i) *= rwk(m-i) );
   }
   for (i = m-1; i >= 0; i -= 2)
   {
      rwk(i) = -rwk(i);
   }
}

void Lagrange1DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                        Vector &shape) const
{
   double w, wk, x = ip.x;
   int i, k, m = GetOrder();

#ifdef MFEM_THREAD_SAFE
   Vector rxxk(m+1);
#endif

   k = (int) floor ( m * x + 0.5 );
   k = k > m ? m : k < 0 ? 0 : k; // clamp k to [0,m]

   wk = 1.0;
   for (i = 0; i <= m; i++)
      if (i != k)
      {
         wk *= ( rxxk(i) = x - (double)(i) / m );
      }
   w = wk * ( rxxk(k) = x - (double)(k) / m );

   if (k != 0)
   {
      shape(0) = w * rwk(0) / rxxk(0);
   }
   else
   {
      shape(0) = wk * rwk(0);
   }
   if (k != m)
   {
      shape(1) = w * rwk(m) / rxxk(m);
   }
   else
   {
      shape(1) = wk * rwk(k);
   }
   for (i = 1; i < m; i++)
      if (i != k)
      {
         shape(i+1) = w * rwk(i) / rxxk(i);
      }
      else
      {
         shape(k+1) = wk * rwk(k);
      }
}

void Lagrange1DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                         DenseMatrix &dshape) const
{
   double s, srx, w, wk, x = ip.x;
   int i, k, m = GetOrder();

#ifdef MFEM_THREAD_SAFE
   Vector rxxk(m+1);
#endif

   k = (int) floor ( m * x + 0.5 );
   k = k > m ? m : k < 0 ? 0 : k; // clamp k to [0,m]

   wk = 1.0;
   for (i = 0; i <= m; i++)
      if (i != k)
      {
         wk *= ( rxxk(i) = x - (double)(i) / m );
      }
   w = wk * ( rxxk(k) = x - (double)(k) / m );

   for (i = 0; i <= m; i++)
   {
      rxxk(i) = 1.0 / rxxk(i);
   }
   srx = 0.0;
   for (i = 0; i <= m; i++)
      if (i != k)
      {
         srx += rxxk(i);
      }
   s = w * srx + wk;

   if (k != 0)
   {
      dshape(0,0) = (s - w * rxxk(0)) * rwk(0) * rxxk(0);
   }
   else
   {
      dshape(0,0) = wk * srx * rwk(0);
   }
   if (k != m)
   {
      dshape(1,0) = (s - w * rxxk(m)) * rwk(m) * rxxk(m);
   }
   else
   {
      dshape(1,0) = wk * srx * rwk(k);
   }
   for (i = 1; i < m; i++)
      if (i != k)
      {
         dshape(i+1,0) = (s - w * rxxk(i)) * rwk(i) * rxxk(i);
      }
      else
      {
         dshape(k+1,0) = wk * srx * rwk(k);
      }
}


P1TetNonConfFiniteElement::P1TetNonConfFiniteElement()
   : NodalFiniteElement(3, Geometry::TETRAHEDRON, 4, 1)
{
   Nodes.IntPoint(0).x = 0.33333333333333333333;
   Nodes.IntPoint(0).y = 0.33333333333333333333;
   Nodes.IntPoint(0).z = 0.33333333333333333333;

   Nodes.IntPoint(1).x = 0.0;
   Nodes.IntPoint(1).y = 0.33333333333333333333;
   Nodes.IntPoint(1).z = 0.33333333333333333333;

   Nodes.IntPoint(2).x = 0.33333333333333333333;
   Nodes.IntPoint(2).y = 0.0;
   Nodes.IntPoint(2).z = 0.33333333333333333333;

   Nodes.IntPoint(3).x = 0.33333333333333333333;
   Nodes.IntPoint(3).y = 0.33333333333333333333;
   Nodes.IntPoint(3).z = 0.0;

}

void P1TetNonConfFiniteElement::CalcShape(const IntegrationPoint &ip,
                                          Vector &shape) const
{
   double L0, L1, L2, L3;

   L1 = ip.x;  L2 = ip.y;  L3 = ip.z;  L0 = 1.0 - L1 - L2 - L3;
   shape(0) = 1.0 - 3.0 * L0;
   shape(1) = 1.0 - 3.0 * L1;
   shape(2) = 1.0 - 3.0 * L2;
   shape(3) = 1.0 - 3.0 * L3;
}

void P1TetNonConfFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                           DenseMatrix &dshape) const
{
   dshape(0,0) =  3.0; dshape(0,1) =  3.0; dshape(0,2) =  3.0;
   dshape(1,0) = -3.0; dshape(1,1) =  0.0; dshape(1,2) =  0.0;
   dshape(2,0) =  0.0; dshape(2,1) = -3.0; dshape(2,2) =  0.0;
   dshape(3,0) =  0.0; dshape(3,1) =  0.0; dshape(3,2) = -3.0;
}


P0TetFiniteElement::P0TetFiniteElement()
   : NodalFiniteElement(3, Geometry::TETRAHEDRON, 1, 0)
{
   Nodes.IntPoint(0).x = 0.25;
   Nodes.IntPoint(0).y = 0.25;
   Nodes.IntPoint(0).z = 0.25;
}

void P0TetFiniteElement::CalcShape(const IntegrationPoint &ip,
                                   Vector &shape) const
{
   shape(0) = 1.0;
}

void P0TetFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                    DenseMatrix &dshape) const
{
   dshape(0,0) =  0.0; dshape(0,1) =  0.0; dshape(0,2) = 0.0;
}


P0HexFiniteElement::P0HexFiniteElement()
   : NodalFiniteElement(3, Geometry::CUBE, 1, 0, FunctionSpace::Qk)
{
   Nodes.IntPoint(0).x = 0.5;
   Nodes.IntPoint(0).y = 0.5;
   Nodes.IntPoint(0).z = 0.5;
}

void P0HexFiniteElement::CalcShape(const IntegrationPoint &ip,
                                   Vector &shape) const
{
   shape(0) = 1.0;
}

void P0HexFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                    DenseMatrix &dshape) const
{
   dshape(0,0) =  0.0; dshape(0,1) =  0.0; dshape(0,2) = 0.0;
}


LagrangeHexFiniteElement::LagrangeHexFiniteElement (int degree)
   : NodalFiniteElement(3, Geometry::CUBE, (degree+1)*(degree+1)*(degree+1),
                        degree, FunctionSpace::Qk)
{
   if (degree == 2)
   {
      I = new int[Dof];
      J = new int[Dof];
      K = new int[Dof];
      // nodes
      I[ 0] = 0; J[ 0] = 0; K[ 0] = 0;
      I[ 1] = 1; J[ 1] = 0; K[ 1] = 0;
      I[ 2] = 1; J[ 2] = 1; K[ 2] = 0;
      I[ 3] = 0; J[ 3] = 1; K[ 3] = 0;
      I[ 4] = 0; J[ 4] = 0; K[ 4] = 1;
      I[ 5] = 1; J[ 5] = 0; K[ 5] = 1;
      I[ 6] = 1; J[ 6] = 1; K[ 6] = 1;
      I[ 7] = 0; J[ 7] = 1; K[ 7] = 1;
      // edges
      I[ 8] = 2; J[ 8] = 0; K[ 8] = 0;
      I[ 9] = 1; J[ 9] = 2; K[ 9] = 0;
      I[10] = 2; J[10] = 1; K[10] = 0;
      I[11] = 0; J[11] = 2; K[11] = 0;
      I[12] = 2; J[12] = 0; K[12] = 1;
      I[13] = 1; J[13] = 2; K[13] = 1;
      I[14] = 2; J[14] = 1; K[14] = 1;
      I[15] = 0; J[15] = 2; K[15] = 1;
      I[16] = 0; J[16] = 0; K[16] = 2;
      I[17] = 1; J[17] = 0; K[17] = 2;
      I[18] = 1; J[18] = 1; K[18] = 2;
      I[19] = 0; J[19] = 1; K[19] = 2;
      // faces
      I[20] = 2; J[20] = 2; K[20] = 0;
      I[21] = 2; J[21] = 0; K[21] = 2;
      I[22] = 1; J[22] = 2; K[22] = 2;
      I[23] = 2; J[23] = 1; K[23] = 2;
      I[24] = 0; J[24] = 2; K[24] = 2;
      I[25] = 2; J[25] = 2; K[25] = 1;
      // element
      I[26] = 2; J[26] = 2; K[26] = 2;
   }
   else if (degree == 3)
   {
      I = new int[Dof];
      J = new int[Dof];
      K = new int[Dof];
      // nodes
      I[ 0] = 0; J[ 0] = 0; K[ 0] = 0;
      I[ 1] = 1; J[ 1] = 0; K[ 1] = 0;
      I[ 2] = 1; J[ 2] = 1; K[ 2] = 0;
      I[ 3] = 0; J[ 3] = 1; K[ 3] = 0;
      I[ 4] = 0; J[ 4] = 0; K[ 4] = 1;
      I[ 5] = 1; J[ 5] = 0; K[ 5] = 1;
      I[ 6] = 1; J[ 6] = 1; K[ 6] = 1;
      I[ 7] = 0; J[ 7] = 1; K[ 7] = 1;
      // edges
      I[ 8] = 2; J[ 8] = 0; K[ 8] = 0;
      I[ 9] = 3; J[ 9] = 0; K[ 9] = 0;
      I[10] = 1; J[10] = 2; K[10] = 0;
      I[11] = 1; J[11] = 3; K[11] = 0;
      I[12] = 2; J[12] = 1; K[12] = 0;
      I[13] = 3; J[13] = 1; K[13] = 0;
      I[14] = 0; J[14] = 2; K[14] = 0;
      I[15] = 0; J[15] = 3; K[15] = 0;
      I[16] = 2; J[16] = 0; K[16] = 1;
      I[17] = 3; J[17] = 0; K[17] = 1;
      I[18] = 1; J[18] = 2; K[18] = 1;
      I[19] = 1; J[19] = 3; K[19] = 1;
      I[20] = 2; J[20] = 1; K[20] = 1;
      I[21] = 3; J[21] = 1; K[21] = 1;
      I[22] = 0; J[22] = 2; K[22] = 1;
      I[23] = 0; J[23] = 3; K[23] = 1;
      I[24] = 0; J[24] = 0; K[24] = 2;
      I[25] = 0; J[25] = 0; K[25] = 3;
      I[26] = 1; J[26] = 0; K[26] = 2;
      I[27] = 1; J[27] = 0; K[27] = 3;
      I[28] = 1; J[28] = 1; K[28] = 2;
      I[29] = 1; J[29] = 1; K[29] = 3;
      I[30] = 0; J[30] = 1; K[30] = 2;
      I[31] = 0; J[31] = 1; K[31] = 3;
      // faces
      I[32] = 2; J[32] = 3; K[32] = 0;
      I[33] = 3; J[33] = 3; K[33] = 0;
      I[34] = 2; J[34] = 2; K[34] = 0;
      I[35] = 3; J[35] = 2; K[35] = 0;
      I[36] = 2; J[36] = 0; K[36] = 2;
      I[37] = 3; J[37] = 0; K[37] = 2;
      I[38] = 2; J[38] = 0; K[38] = 3;
      I[39] = 3; J[39] = 0; K[39] = 3;
      I[40] = 1; J[40] = 2; K[40] = 2;
      I[41] = 1; J[41] = 3; K[41] = 2;
      I[42] = 1; J[42] = 2; K[42] = 3;
      I[43] = 1; J[43] = 3; K[43] = 3;
      I[44] = 3; J[44] = 1; K[44] = 2;
      I[45] = 2; J[45] = 1; K[45] = 2;
      I[46] = 3; J[46] = 1; K[46] = 3;
      I[47] = 2; J[47] = 1; K[47] = 3;
      I[48] = 0; J[48] = 3; K[48] = 2;
      I[49] = 0; J[49] = 2; K[49] = 2;
      I[50] = 0; J[50] = 3; K[50] = 3;
      I[51] = 0; J[51] = 2; K[51] = 3;
      I[52] = 2; J[52] = 2; K[52] = 1;
      I[53] = 3; J[53] = 2; K[53] = 1;
      I[54] = 2; J[54] = 3; K[54] = 1;
      I[55] = 3; J[55] = 3; K[55] = 1;
      // element
      I[56] = 2; J[56] = 2; K[56] = 2;
      I[57] = 3; J[57] = 2; K[57] = 2;
      I[58] = 3; J[58] = 3; K[58] = 2;
      I[59] = 2; J[59] = 3; K[59] = 2;
      I[60] = 2; J[60] = 2; K[60] = 3;
      I[61] = 3; J[61] = 2; K[61] = 3;
      I[62] = 3; J[62] = 3; K[62] = 3;
      I[63] = 2; J[63] = 3; K[63] = 3;
   }
   else
   {
      mfem_error ("LagrangeHexFiniteElement::LagrangeHexFiniteElement");
   }

   fe1d = new Lagrange1DFiniteElement(degree);
   dof1d = fe1d -> GetDof();

#ifndef MFEM_THREAD_SAFE
   shape1dx.SetSize(dof1d);
   shape1dy.SetSize(dof1d);
   shape1dz.SetSize(dof1d);

   dshape1dx.SetSize(dof1d,1);
   dshape1dy.SetSize(dof1d,1);
   dshape1dz.SetSize(dof1d,1);
#endif

   for (int n = 0; n < Dof; n++)
   {
      Nodes.IntPoint(n).x = fe1d -> GetNodes().IntPoint(I[n]).x;
      Nodes.IntPoint(n).y = fe1d -> GetNodes().IntPoint(J[n]).x;
      Nodes.IntPoint(n).z = fe1d -> GetNodes().IntPoint(K[n]).x;
   }
}

void LagrangeHexFiniteElement::CalcShape(const IntegrationPoint &ip,
                                         Vector &shape) const
{
   IntegrationPoint ipy, ipz;
   ipy.x = ip.y;
   ipz.x = ip.z;

#ifdef MFEM_THREAD_SAFE
   Vector shape1dx(dof1d), shape1dy(dof1d), shape1dz(dof1d);
#endif

   fe1d -> CalcShape(ip,  shape1dx);
   fe1d -> CalcShape(ipy, shape1dy);
   fe1d -> CalcShape(ipz, shape1dz);

   for (int n = 0; n < Dof; n++)
   {
      shape(n) = shape1dx(I[n]) *  shape1dy(J[n]) * shape1dz(K[n]);
   }
}

void LagrangeHexFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                          DenseMatrix &dshape) const
{
   IntegrationPoint ipy, ipz;
   ipy.x = ip.y;
   ipz.x = ip.z;

#ifdef MFEM_THREAD_SAFE
   Vector shape1dx(dof1d), shape1dy(dof1d), shape1dz(dof1d);
   DenseMatrix dshape1dx(dof1d,1), dshape1dy(dof1d,1), dshape1dz(dof1d,1);
#endif

   fe1d -> CalcShape(ip,  shape1dx);
   fe1d -> CalcShape(ipy, shape1dy);
   fe1d -> CalcShape(ipz, shape1dz);

   fe1d -> CalcDShape(ip,  dshape1dx);
   fe1d -> CalcDShape(ipy, dshape1dy);
   fe1d -> CalcDShape(ipz, dshape1dz);

   for (int n = 0; n < Dof; n++)
   {
      dshape(n,0) = dshape1dx(I[n],0) * shape1dy(J[n])    * shape1dz(K[n]);
      dshape(n,1) = shape1dx(I[n])    * dshape1dy(J[n],0) * shape1dz(K[n]);
      dshape(n,2) = shape1dx(I[n])    * shape1dy(J[n])    * dshape1dz(K[n],0);
   }
}

LagrangeHexFiniteElement::~LagrangeHexFiniteElement ()
{
   delete fe1d;

   delete [] I;
   delete [] J;
   delete [] K;
}


RefinedLinear1DFiniteElement::RefinedLinear1DFiniteElement()
   : NodalFiniteElement(1, Geometry::SEGMENT, 3, 4)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(2).x = 0.5;
}

void RefinedLinear1DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                             Vector &shape) const
{
   double x = ip.x;

   if (x <= 0.5)
   {
      shape(0) = 1.0 - 2.0 * x;
      shape(1) = 0.0;
      shape(2) = 2.0 * x;
   }
   else
   {
      shape(0) = 0.0;
      shape(1) = 2.0 * x - 1.0;
      shape(2) = 2.0 - 2.0 * x;
   }
}

void RefinedLinear1DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                              DenseMatrix &dshape) const
{
   double x = ip.x;

   if (x <= 0.5)
   {
      dshape(0,0) = - 2.0;
      dshape(1,0) =   0.0;
      dshape(2,0) =   2.0;
   }
   else
   {
      dshape(0,0) =   0.0;
      dshape(1,0) =   2.0;
      dshape(2,0) = - 2.0;
   }
}

RefinedLinear2DFiniteElement::RefinedLinear2DFiniteElement()
   : NodalFiniteElement(2, Geometry::TRIANGLE, 6, 5)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(2).x = 0.0;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(3).x = 0.5;
   Nodes.IntPoint(3).y = 0.0;
   Nodes.IntPoint(4).x = 0.5;
   Nodes.IntPoint(4).y = 0.5;
   Nodes.IntPoint(5).x = 0.0;
   Nodes.IntPoint(5).y = 0.5;
}

void RefinedLinear2DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                             Vector &shape) const
{
   int i;

   double L0, L1, L2;
   L0 = 2.0 * ( 1. - ip.x - ip.y );
   L1 = 2.0 * ( ip.x );
   L2 = 2.0 * ( ip.y );

   // The reference triangle is split in 4 triangles as follows:
   //
   // T0 - 0,3,5
   // T1 - 1,3,4
   // T2 - 2,4,5
   // T3 - 3,4,5

   for (i = 0; i < 6; i++)
   {
      shape(i) = 0.0;
   }

   if (L0 >= 1.0)   // T0
   {
      shape(0) = L0 - 1.0;
      shape(3) =       L1;
      shape(5) =       L2;
   }
   else if (L1 >= 1.0)   // T1
   {
      shape(3) =       L0;
      shape(1) = L1 - 1.0;
      shape(4) =       L2;
   }
   else if (L2 >= 1.0)   // T2
   {
      shape(5) =       L0;
      shape(4) =       L1;
      shape(2) = L2 - 1.0;
   }
   else   // T3
   {
      shape(3) = 1.0 - L2;
      shape(4) = 1.0 - L0;
      shape(5) = 1.0 - L1;
   }
}

void RefinedLinear2DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                              DenseMatrix &dshape) const
{
   int i,j;

   double L0, L1, L2;
   L0 = 2.0 * ( 1. - ip.x - ip.y );
   L1 = 2.0 * ( ip.x );
   L2 = 2.0 * ( ip.y );

   double DL0[2], DL1[2], DL2[2];
   DL0[0] = -2.0; DL0[1] = -2.0;
   DL1[0] =  2.0; DL1[1] =  0.0;
   DL2[0] =  0.0; DL2[1] =  2.0;

   for (i = 0; i < 6; i++)
      for (j = 0; j < 2; j++)
      {
         dshape(i,j) = 0.0;
      }

   if (L0 >= 1.0)   // T0
   {
      for (j = 0; j < 2; j++)
      {
         dshape(0,j) = DL0[j];
         dshape(3,j) = DL1[j];
         dshape(5,j) = DL2[j];
      }
   }
   else if (L1 >= 1.0)   // T1
   {
      for (j = 0; j < 2; j++)
      {
         dshape(3,j) = DL0[j];
         dshape(1,j) = DL1[j];
         dshape(4,j) = DL2[j];
      }
   }
   else if (L2 >= 1.0)   // T2
   {
      for (j = 0; j < 2; j++)
      {
         dshape(5,j) = DL0[j];
         dshape(4,j) = DL1[j];
         dshape(2,j) = DL2[j];
      }
   }
   else   // T3
   {
      for (j = 0; j < 2; j++)
      {
         dshape(3,j) = - DL2[j];
         dshape(4,j) = - DL0[j];
         dshape(5,j) = - DL1[j];
      }
   }
}

RefinedLinear3DFiniteElement::RefinedLinear3DFiniteElement()
   : NodalFiniteElement(3, Geometry::TETRAHEDRON, 10, 4)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(0).z = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(1).z = 0.0;
   Nodes.IntPoint(2).x = 0.0;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(2).z = 0.0;
   Nodes.IntPoint(3).x = 0.0;
   Nodes.IntPoint(3).y = 0.0;
   Nodes.IntPoint(3).z = 1.0;
   Nodes.IntPoint(4).x = 0.5;
   Nodes.IntPoint(4).y = 0.0;
   Nodes.IntPoint(4).z = 0.0;
   Nodes.IntPoint(5).x = 0.0;
   Nodes.IntPoint(5).y = 0.5;
   Nodes.IntPoint(5).z = 0.0;
   Nodes.IntPoint(6).x = 0.0;
   Nodes.IntPoint(6).y = 0.0;
   Nodes.IntPoint(6).z = 0.5;
   Nodes.IntPoint(7).x = 0.5;
   Nodes.IntPoint(7).y = 0.5;
   Nodes.IntPoint(7).z = 0.0;
   Nodes.IntPoint(8).x = 0.5;
   Nodes.IntPoint(8).y = 0.0;
   Nodes.IntPoint(8).z = 0.5;
   Nodes.IntPoint(9).x = 0.0;
   Nodes.IntPoint(9).y = 0.5;
   Nodes.IntPoint(9).z = 0.5;
}

void RefinedLinear3DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                             Vector &shape) const
{
   int i;

   double L0, L1, L2, L3, L4, L5;
   L0 = 2.0 * ( 1. - ip.x - ip.y - ip.z );
   L1 = 2.0 * ( ip.x );
   L2 = 2.0 * ( ip.y );
   L3 = 2.0 * ( ip.z );
   L4 = 2.0 * ( ip.x + ip.y );
   L5 = 2.0 * ( ip.y + ip.z );

   // The reference tetrahedron is split in 8 tetrahedra as follows:
   //
   // T0 - 0,4,5,6
   // T1 - 1,4,7,8
   // T2 - 2,5,7,9
   // T3 - 3,6,8,9
   // T4 - 4,5,6,8
   // T5 - 4,5,7,8
   // T6 - 5,6,8,9
   // T7 - 5,7,8,9

   for (i = 0; i < 10; i++)
   {
      shape(i) = 0.0;
   }

   if (L0 >= 1.0)   // T0
   {
      shape(0) = L0 - 1.0;
      shape(4) =       L1;
      shape(5) =       L2;
      shape(6) =       L3;
   }
   else if (L1 >= 1.0)   // T1
   {
      shape(4) =       L0;
      shape(1) = L1 - 1.0;
      shape(7) =       L2;
      shape(8) =       L3;
   }
   else if (L2 >= 1.0)   // T2
   {
      shape(5) =       L0;
      shape(7) =       L1;
      shape(2) = L2 - 1.0;
      shape(9) =       L3;
   }
   else if (L3 >= 1.0)   // T3
   {
      shape(6) =       L0;
      shape(8) =       L1;
      shape(9) =       L2;
      shape(3) = L3 - 1.0;
   }
   else if ((L4 <= 1.0) && (L5 <= 1.0))   // T4
   {
      shape(4) = 1.0 - L5;
      shape(5) =       L2;
      shape(6) = 1.0 - L4;
      shape(8) = 1.0 - L0;
   }
   else if ((L4 >= 1.0) && (L5 <= 1.0))   // T5
   {
      shape(4) = 1.0 - L5;
      shape(5) = 1.0 - L1;
      shape(7) = L4 - 1.0;
      shape(8) =       L3;
   }
   else if ((L4 <= 1.0) && (L5 >= 1.0))   // T6
   {
      shape(5) = 1.0 - L3;
      shape(6) = 1.0 - L4;
      shape(8) =       L1;
      shape(9) = L5 - 1.0;
   }
   else if ((L4 >= 1.0) && (L5 >= 1.0))   // T7
   {
      shape(5) =       L0;
      shape(7) = L4 - 1.0;
      shape(8) = 1.0 - L2;
      shape(9) = L5 - 1.0;
   }
}

void RefinedLinear3DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                              DenseMatrix &dshape) const
{
   int i,j;

   double L0, L1, L2, L3, L4, L5;
   L0 = 2.0 * ( 1. - ip.x - ip.y - ip.z );
   L1 = 2.0 * ( ip.x );
   L2 = 2.0 * ( ip.y );
   L3 = 2.0 * ( ip.z );
   L4 = 2.0 * ( ip.x + ip.y );
   L5 = 2.0 * ( ip.y + ip.z );

   double DL0[3], DL1[3], DL2[3], DL3[3], DL4[3], DL5[3];
   DL0[0] = -2.0; DL0[1] = -2.0; DL0[2] = -2.0;
   DL1[0] =  2.0; DL1[1] =  0.0; DL1[2] =  0.0;
   DL2[0] =  0.0; DL2[1] =  2.0; DL2[2] =  0.0;
   DL3[0] =  0.0; DL3[1] =  0.0; DL3[2] =  2.0;
   DL4[0] =  2.0; DL4[1] =  2.0; DL4[2] =  0.0;
   DL5[0] =  0.0; DL5[1] =  2.0; DL5[2] =  2.0;

   for (i = 0; i < 10; i++)
      for (j = 0; j < 3; j++)
      {
         dshape(i,j) = 0.0;
      }

   if (L0 >= 1.0)   // T0
   {
      for (j = 0; j < 3; j++)
      {
         dshape(0,j) = DL0[j];
         dshape(4,j) = DL1[j];
         dshape(5,j) = DL2[j];
         dshape(6,j) = DL3[j];
      }
   }
   else if (L1 >= 1.0)   // T1
   {
      for (j = 0; j < 3; j++)
      {
         dshape(4,j) = DL0[j];
         dshape(1,j) = DL1[j];
         dshape(7,j) = DL2[j];
         dshape(8,j) = DL3[j];
      }
   }
   else if (L2 >= 1.0)   // T2
   {
      for (j = 0; j < 3; j++)
      {
         dshape(5,j) = DL0[j];
         dshape(7,j) = DL1[j];
         dshape(2,j) = DL2[j];
         dshape(9,j) = DL3[j];
      }
   }
   else if (L3 >= 1.0)   // T3
   {
      for (j = 0; j < 3; j++)
      {
         dshape(6,j) = DL0[j];
         dshape(8,j) = DL1[j];
         dshape(9,j) = DL2[j];
         dshape(3,j) = DL3[j];
      }
   }
   else if ((L4 <= 1.0) && (L5 <= 1.0))   // T4
   {
      for (j = 0; j < 3; j++)
      {
         dshape(4,j) = - DL5[j];
         dshape(5,j) =   DL2[j];
         dshape(6,j) = - DL4[j];
         dshape(8,j) = - DL0[j];
      }
   }
   else if ((L4 >= 1.0) && (L5 <= 1.0))   // T5
   {
      for (j = 0; j < 3; j++)
      {
         dshape(4,j) = - DL5[j];
         dshape(5,j) = - DL1[j];
         dshape(7,j) =   DL4[j];
         dshape(8,j) =   DL3[j];
      }
   }
   else if ((L4 <= 1.0) && (L5 >= 1.0))   // T6
   {
      for (j = 0; j < 3; j++)
      {
         dshape(5,j) = - DL3[j];
         dshape(6,j) = - DL4[j];
         dshape(8,j) =   DL1[j];
         dshape(9,j) =   DL5[j];
      }
   }
   else if ((L4 >= 1.0) && (L5 >= 1.0))   // T7
   {
      for (j = 0; j < 3; j++)
      {
         dshape(5,j) =   DL0[j];
         dshape(7,j) =   DL4[j];
         dshape(8,j) = - DL2[j];
         dshape(9,j) =   DL5[j];
      }
   }
}


RefinedBiLinear2DFiniteElement::RefinedBiLinear2DFiniteElement()
   : NodalFiniteElement(2, Geometry::SQUARE, 9, 1, FunctionSpace::rQk)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(2).x = 1.0;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(3).x = 0.0;
   Nodes.IntPoint(3).y = 1.0;
   Nodes.IntPoint(4).x = 0.5;
   Nodes.IntPoint(4).y = 0.0;
   Nodes.IntPoint(5).x = 1.0;
   Nodes.IntPoint(5).y = 0.5;
   Nodes.IntPoint(6).x = 0.5;
   Nodes.IntPoint(6).y = 1.0;
   Nodes.IntPoint(7).x = 0.0;
   Nodes.IntPoint(7).y = 0.5;
   Nodes.IntPoint(8).x = 0.5;
   Nodes.IntPoint(8).y = 0.5;
}

void RefinedBiLinear2DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                               Vector &shape) const
{
   int i;
   double x = ip.x, y = ip.y;
   double Lx, Ly;
   Lx = 2.0 * ( 1. - x );
   Ly = 2.0 * ( 1. - y );

   // The reference square is split in 4 squares as follows:
   //
   // T0 - 0,4,7,8
   // T1 - 1,4,5,8
   // T2 - 2,5,6,8
   // T3 - 3,6,7,8

   for (i = 0; i < 9; i++)
   {
      shape(i) = 0.0;
   }

   if ((x <= 0.5) && (y <= 0.5))   // T0
   {
      shape(0) = (Lx - 1.0) * (Ly - 1.0);
      shape(4) = (2.0 - Lx) * (Ly - 1.0);
      shape(8) = (2.0 - Lx) * (2.0 - Ly);
      shape(7) = (Lx - 1.0) * (2.0 - Ly);
   }
   else if ((x >= 0.5) && (y <= 0.5))   // T1
   {
      shape(4) =        Lx  * (Ly - 1.0);
      shape(1) = (1.0 - Lx) * (Ly - 1.0);
      shape(5) = (1.0 - Lx) * (2.0 - Ly);
      shape(8) =        Lx  * (2.0 - Ly);
   }
   else if ((x >= 0.5) && (y >= 0.5))   // T2
   {
      shape(8) =        Lx  *        Ly ;
      shape(5) = (1.0 - Lx) *        Ly ;
      shape(2) = (1.0 - Lx) * (1.0 - Ly);
      shape(6) =        Lx  * (1.0 - Ly);
   }
   else if ((x <= 0.5) && (y >= 0.5))   // T3
   {
      shape(7) = (Lx - 1.0) *        Ly ;
      shape(8) = (2.0 - Lx) *        Ly ;
      shape(6) = (2.0 - Lx) * (1.0 - Ly);
      shape(3) = (Lx - 1.0) * (1.0 - Ly);
   }
}

void RefinedBiLinear2DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                                DenseMatrix &dshape) const
{
   int i,j;
   double x = ip.x, y = ip.y;
   double Lx, Ly;
   Lx = 2.0 * ( 1. - x );
   Ly = 2.0 * ( 1. - y );

   for (i = 0; i < 9; i++)
      for (j = 0; j < 2; j++)
      {
         dshape(i,j) = 0.0;
      }

   if ((x <= 0.5) && (y <= 0.5))   // T0
   {
      dshape(0,0) =  2.0 * (1.0 - Ly);
      dshape(0,1) =  2.0 * (1.0 - Lx);

      dshape(4,0) =  2.0 * (Ly - 1.0);
      dshape(4,1) = -2.0 * (2.0 - Lx);

      dshape(8,0) =  2.0 * (2.0 - Ly);
      dshape(8,1) =  2.0 * (2.0 - Lx);

      dshape(7,0) = -2.0 * (2.0 - Ly);
      dshape(7,0) =  2.0 * (Lx - 1.0);
   }
   else if ((x >= 0.5) && (y <= 0.5))   // T1
   {
      dshape(4,0) = -2.0 * (Ly - 1.0);
      dshape(4,1) = -2.0 * Lx;

      dshape(1,0) =  2.0 * (Ly - 1.0);
      dshape(1,1) = -2.0 * (1.0 - Lx);

      dshape(5,0) =  2.0 * (2.0 - Ly);
      dshape(5,1) =  2.0 * (1.0 - Lx);

      dshape(8,0) = -2.0 * (2.0 - Ly);
      dshape(8,1) =  2.0 * Lx;
   }
   else if ((x >= 0.5) && (y >= 0.5))   // T2
   {
      dshape(8,0) = -2.0 * Ly;
      dshape(8,1) = -2.0 * Lx;

      dshape(5,0) =  2.0 * Ly;
      dshape(5,1) = -2.0 * (1.0 - Lx);

      dshape(2,0) =  2.0 * (1.0 - Ly);
      dshape(2,1) =  2.0 * (1.0 - Lx);

      dshape(6,0) = -2.0 * (1.0 - Ly);
      dshape(6,1) =  2.0 * Lx;
   }
   else if ((x <= 0.5) && (y >= 0.5))   // T3
   {
      dshape(7,0) = -2.0 * Ly;
      dshape(7,1) = -2.0 * (Lx - 1.0);

      dshape(8,0) =  2.0 * Ly ;
      dshape(8,1) = -2.0 * (2.0 - Lx);

      dshape(6,0) = 2.0 * (1.0 - Ly);
      dshape(6,1) = 2.0 * (2.0 - Lx);

      dshape(3,0) = -2.0 * (1.0 - Ly);
      dshape(3,1) =  2.0 * (Lx - 1.0);
   }
}

RefinedTriLinear3DFiniteElement::RefinedTriLinear3DFiniteElement()
   : NodalFiniteElement(3, Geometry::CUBE, 27, 2, FunctionSpace::rQk)
{
   double I[27];
   double J[27];
   double K[27];
   // nodes
   I[ 0] = 0.0; J[ 0] = 0.0; K[ 0] = 0.0;
   I[ 1] = 1.0; J[ 1] = 0.0; K[ 1] = 0.0;
   I[ 2] = 1.0; J[ 2] = 1.0; K[ 2] = 0.0;
   I[ 3] = 0.0; J[ 3] = 1.0; K[ 3] = 0.0;
   I[ 4] = 0.0; J[ 4] = 0.0; K[ 4] = 1.0;
   I[ 5] = 1.0; J[ 5] = 0.0; K[ 5] = 1.0;
   I[ 6] = 1.0; J[ 6] = 1.0; K[ 6] = 1.0;
   I[ 7] = 0.0; J[ 7] = 1.0; K[ 7] = 1.0;
   // edges
   I[ 8] = 0.5; J[ 8] = 0.0; K[ 8] = 0.0;
   I[ 9] = 1.0; J[ 9] = 0.5; K[ 9] = 0.0;
   I[10] = 0.5; J[10] = 1.0; K[10] = 0.0;
   I[11] = 0.0; J[11] = 0.5; K[11] = 0.0;
   I[12] = 0.5; J[12] = 0.0; K[12] = 1.0;
   I[13] = 1.0; J[13] = 0.5; K[13] = 1.0;
   I[14] = 0.5; J[14] = 1.0; K[14] = 1.0;
   I[15] = 0.0; J[15] = 0.5; K[15] = 1.0;
   I[16] = 0.0; J[16] = 0.0; K[16] = 0.5;
   I[17] = 1.0; J[17] = 0.0; K[17] = 0.5;
   I[18] = 1.0; J[18] = 1.0; K[18] = 0.5;
   I[19] = 0.0; J[19] = 1.0; K[19] = 0.5;
   // faces
   I[20] = 0.5; J[20] = 0.5; K[20] = 0.0;
   I[21] = 0.5; J[21] = 0.0; K[21] = 0.5;
   I[22] = 1.0; J[22] = 0.5; K[22] = 0.5;
   I[23] = 0.5; J[23] = 1.0; K[23] = 0.5;
   I[24] = 0.0; J[24] = 0.5; K[24] = 0.5;
   I[25] = 0.5; J[25] = 0.5; K[25] = 1.0;
   // element
   I[26] = 0.5; J[26] = 0.5; K[26] = 0.5;

   for (int n = 0; n < 27; n++)
   {
      Nodes.IntPoint(n).x = I[n];
      Nodes.IntPoint(n).y = J[n];
      Nodes.IntPoint(n).z = K[n];
   }
}

void RefinedTriLinear3DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                                Vector &shape) const
{
   int i, N[8];
   double Lx, Ly, Lz;
   double x = ip.x, y = ip.y, z = ip.z;

   for (i = 0; i < 27; i++)
   {
      shape(i) = 0.0;
   }

   if ((x <= 0.5) && (y <= 0.5) && (z <= 0.5))   // T0
   {
      Lx = 1.0 - 2.0 * x;
      Ly = 1.0 - 2.0 * y;
      Lz = 1.0 - 2.0 * z;

      N[0] =  0;
      N[1] =  8;
      N[2] = 20;
      N[3] = 11;
      N[4] = 16;
      N[5] = 21;
      N[6] = 26;
      N[7] = 24;
   }
   else if ((x >= 0.5) && (y <= 0.5) && (z <= 0.5))   // T1
   {
      Lx = 2.0 - 2.0 * x;
      Ly = 1.0 - 2.0 * y;
      Lz = 1.0 - 2.0 * z;

      N[0] =  8;
      N[1] =  1;
      N[2] =  9;
      N[3] = 20;
      N[4] = 21;
      N[5] = 17;
      N[6] = 22;
      N[7] = 26;
   }
   else if ((x <= 0.5) && (y >= 0.5) && (z <= 0.5))   // T2
   {
      Lx = 2.0 - 2.0 * x;
      Ly = 2.0 - 2.0 * y;
      Lz = 1.0 - 2.0 * z;

      N[0] = 20;
      N[1] =  9;
      N[2] =  2;
      N[3] = 10;
      N[4] = 26;
      N[5] = 22;
      N[6] = 18;
      N[7] = 23;
   }
   else if ((x >= 0.5) && (y >= 0.5) && (z <= 0.5))   // T3
   {
      Lx = 1.0 - 2.0 * x;
      Ly = 2.0 - 2.0 * y;
      Lz = 1.0 - 2.0 * z;

      N[0] = 11;
      N[1] = 20;
      N[2] = 10;
      N[3] =  3;
      N[4] = 24;
      N[5] = 26;
      N[6] = 23;
      N[7] = 19;
   }
   else if ((x <= 0.5) && (y <= 0.5) && (z >= 0.5))   // T4
   {
      Lx = 1.0 - 2.0 * x;
      Ly = 1.0 - 2.0 * y;
      Lz = 2.0 - 2.0 * z;

      N[0] = 16;
      N[1] = 21;
      N[2] = 26;
      N[3] = 24;
      N[4] =  4;
      N[5] = 12;
      N[6] = 25;
      N[7] = 15;
   }
   else if ((x >= 0.5) && (y <= 0.5) && (z >= 0.5))   // T5
   {
      Lx = 2.0 - 2.0 * x;
      Ly = 1.0 - 2.0 * y;
      Lz = 2.0 - 2.0 * z;

      N[0] = 21;
      N[1] = 17;
      N[2] = 22;
      N[3] = 26;
      N[4] = 12;
      N[5] =  5;
      N[6] = 13;
      N[7] = 25;
   }
   else if ((x <= 0.5) && (y >= 0.5) && (z >= 0.5))   // T6
   {
      Lx = 2.0 - 2.0 * x;
      Ly = 2.0 - 2.0 * y;
      Lz = 2.0 - 2.0 * z;

      N[0] = 26;
      N[1] = 22;
      N[2] = 18;
      N[3] = 23;
      N[4] = 25;
      N[5] = 13;
      N[6] =  6;
      N[7] = 14;
   }
   else   // T7
   {
      Lx = 1.0 - 2.0 * x;
      Ly = 2.0 - 2.0 * y;
      Lz = 2.0 - 2.0 * z;

      N[0] = 24;
      N[1] = 26;
      N[2] = 23;
      N[3] = 19;
      N[4] = 15;
      N[5] = 25;
      N[6] = 14;
      N[7] =  7;
   }

   shape(N[0]) = Lx       * Ly       * Lz;
   shape(N[1]) = (1 - Lx) * Ly       * Lz;
   shape(N[2]) = (1 - Lx) * (1 - Ly) * Lz;
   shape(N[3]) = Lx       * (1 - Ly) * Lz;
   shape(N[4]) = Lx       * Ly       * (1 - Lz);
   shape(N[5]) = (1 - Lx) * Ly       * (1 - Lz);
   shape(N[6]) = (1 - Lx) * (1 - Ly) * (1 - Lz);
   shape(N[7]) = Lx       * (1 - Ly) * (1 - Lz);
}

void RefinedTriLinear3DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                                 DenseMatrix &dshape) const
{
   int i, j, N[8];
   double Lx, Ly, Lz;
   double x = ip.x, y = ip.y, z = ip.z;

   for (i = 0; i < 27; i++)
      for (j = 0; j < 3; j++)
      {
         dshape(i,j) = 0.0;
      }

   if ((x <= 0.5) && (y <= 0.5) && (z <= 0.5))   // T0
   {
      Lx = 1.0 - 2.0 * x;
      Ly = 1.0 - 2.0 * y;
      Lz = 1.0 - 2.0 * z;

      N[0] =  0;
      N[1] =  8;
      N[2] = 20;
      N[3] = 11;
      N[4] = 16;
      N[5] = 21;
      N[6] = 26;
      N[7] = 24;
   }
   else if ((x >= 0.5) && (y <= 0.5) && (z <= 0.5))   // T1
   {
      Lx = 2.0 - 2.0 * x;
      Ly = 1.0 - 2.0 * y;
      Lz = 1.0 - 2.0 * z;

      N[0] =  8;
      N[1] =  1;
      N[2] =  9;
      N[3] = 20;
      N[4] = 21;
      N[5] = 17;
      N[6] = 22;
      N[7] = 26;
   }
   else if ((x <= 0.5) && (y >= 0.5) && (z <= 0.5))   // T2
   {
      Lx = 2.0 - 2.0 * x;
      Ly = 2.0 - 2.0 * y;
      Lz = 1.0 - 2.0 * z;

      N[0] = 20;
      N[1] =  9;
      N[2] =  2;
      N[3] = 10;
      N[4] = 26;
      N[5] = 22;
      N[6] = 18;
      N[7] = 23;
   }
   else if ((x >= 0.5) && (y >= 0.5) && (z <= 0.5))   // T3
   {
      Lx = 1.0 - 2.0 * x;
      Ly = 2.0 - 2.0 * y;
      Lz = 1.0 - 2.0 * z;

      N[0] = 11;
      N[1] = 20;
      N[2] = 10;
      N[3] =  3;
      N[4] = 24;
      N[5] = 26;
      N[6] = 23;
      N[7] = 19;
   }
   else if ((x <= 0.5) && (y <= 0.5) && (z >= 0.5))   // T4
   {
      Lx = 1.0 - 2.0 * x;
      Ly = 1.0 - 2.0 * y;
      Lz = 2.0 - 2.0 * z;

      N[0] = 16;
      N[1] = 21;
      N[2] = 26;
      N[3] = 24;
      N[4] =  4;
      N[5] = 12;
      N[6] = 25;
      N[7] = 15;
   }
   else if ((x >= 0.5) && (y <= 0.5) && (z >= 0.5))   // T5
   {
      Lx = 2.0 - 2.0 * x;
      Ly = 1.0 - 2.0 * y;
      Lz = 2.0 - 2.0 * z;

      N[0] = 21;
      N[1] = 17;
      N[2] = 22;
      N[3] = 26;
      N[4] = 12;
      N[5] =  5;
      N[6] = 13;
      N[7] = 25;
   }
   else if ((x <= 0.5) && (y >= 0.5) && (z >= 0.5))   // T6
   {
      Lx = 2.0 - 2.0 * x;
      Ly = 2.0 - 2.0 * y;
      Lz = 2.0 - 2.0 * z;

      N[0] = 26;
      N[1] = 22;
      N[2] = 18;
      N[3] = 23;
      N[4] = 25;
      N[5] = 13;
      N[6] =  6;
      N[7] = 14;
   }
   else   // T7
   {
      Lx = 1.0 - 2.0 * x;
      Ly = 2.0 - 2.0 * y;
      Lz = 2.0 - 2.0 * z;

      N[0] = 24;
      N[1] = 26;
      N[2] = 23;
      N[3] = 19;
      N[4] = 15;
      N[5] = 25;
      N[6] = 14;
      N[7] =  7;
   }

   dshape(N[0],0) = -2.0 * Ly       * Lz      ;
   dshape(N[0],1) = -2.0 * Lx       * Lz      ;
   dshape(N[0],2) = -2.0 * Lx       * Ly      ;

   dshape(N[1],0) =  2.0 * Ly       * Lz      ;
   dshape(N[1],1) = -2.0 * (1 - Lx) * Lz      ;
   dshape(N[1],2) = -2.0 * (1 - Lx) * Ly      ;

   dshape(N[2],0) =  2.0 * (1 - Ly) * Lz      ;
   dshape(N[2],1) =  2.0 * (1 - Lx) * Lz      ;
   dshape(N[2],2) = -2.0 * (1 - Lx) * (1 - Ly);

   dshape(N[3],0) = -2.0 * (1 - Ly) * Lz      ;
   dshape(N[3],1) =  2.0 * Lx       * Lz      ;
   dshape(N[3],2) = -2.0 * Lx       * (1 - Ly);

   dshape(N[4],0) = -2.0 * Ly       * (1 - Lz);
   dshape(N[4],1) = -2.0 * Lx       * (1 - Lz);
   dshape(N[4],2) =  2.0 * Lx       * Ly      ;

   dshape(N[5],0) =  2.0 * Ly       * (1 - Lz);
   dshape(N[5],1) = -2.0 * (1 - Lx) * (1 - Lz);
   dshape(N[5],2) =  2.0 * (1 - Lx) * Ly      ;

   dshape(N[6],0) =  2.0 * (1 - Ly) * (1 - Lz);
   dshape(N[6],1) =  2.0 * (1 - Lx) * (1 - Lz);
   dshape(N[6],2) =  2.0 * (1 - Lx) * (1 - Ly);

   dshape(N[7],0) = -2.0 * (1 - Ly) * (1 - Lz);
   dshape(N[7],1) =  2.0 * Lx       * (1 - Lz);
   dshape(N[7],2) =  2.0 * Lx       * (1 - Ly);
}


Nedelec1HexFiniteElement::Nedelec1HexFiniteElement()
   : VectorFiniteElement(3, Geometry::CUBE, 12, 1, H_CURL, FunctionSpace::Qk)
{
   // not real nodes ...
   Nodes.IntPoint(0).x = 0.5;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(0).z = 0.0;

   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.5;
   Nodes.IntPoint(1).z = 0.0;

   Nodes.IntPoint(2).x = 0.5;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(2).z = 0.0;

   Nodes.IntPoint(3).x = 0.0;
   Nodes.IntPoint(3).y = 0.5;
   Nodes.IntPoint(3).z = 0.0;

   Nodes.IntPoint(4).x = 0.5;
   Nodes.IntPoint(4).y = 0.0;
   Nodes.IntPoint(4).z = 1.0;

   Nodes.IntPoint(5).x = 1.0;
   Nodes.IntPoint(5).y = 0.5;
   Nodes.IntPoint(5).z = 1.0;

   Nodes.IntPoint(6).x = 0.5;
   Nodes.IntPoint(6).y = 1.0;
   Nodes.IntPoint(6).z = 1.0;

   Nodes.IntPoint(7).x = 0.0;
   Nodes.IntPoint(7).y = 0.5;
   Nodes.IntPoint(7).z = 1.0;

   Nodes.IntPoint(8).x = 0.0;
   Nodes.IntPoint(8).y = 0.0;
   Nodes.IntPoint(8).z = 0.5;

   Nodes.IntPoint(9).x = 1.0;
   Nodes.IntPoint(9).y = 0.0;
   Nodes.IntPoint(9).z = 0.5;

   Nodes.IntPoint(10).x= 1.0;
   Nodes.IntPoint(10).y= 1.0;
   Nodes.IntPoint(10).z= 0.5;

   Nodes.IntPoint(11).x= 0.0;
   Nodes.IntPoint(11).y= 1.0;
   Nodes.IntPoint(11).z= 0.5;
}

void Nedelec1HexFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                          DenseMatrix &shape) const
{
   double x = ip.x, y = ip.y, z = ip.z;

   shape(0,0) = (1. - y) * (1. - z);
   shape(0,1) = 0.;
   shape(0,2) = 0.;

   shape(2,0) = y * (1. - z);
   shape(2,1) = 0.;
   shape(2,2) = 0.;

   shape(4,0) = z * (1. - y);
   shape(4,1) = 0.;
   shape(4,2) = 0.;

   shape(6,0) = y * z;
   shape(6,1) = 0.;
   shape(6,2) = 0.;

   shape(1,0) = 0.;
   shape(1,1) = x * (1. - z);
   shape(1,2) = 0.;

   shape(3,0) = 0.;
   shape(3,1) = (1. - x) * (1. - z);
   shape(3,2) = 0.;

   shape(5,0) = 0.;
   shape(5,1) = x * z;
   shape(5,2) = 0.;

   shape(7,0) = 0.;
   shape(7,1) = (1. - x) * z;
   shape(7,2) = 0.;

   shape(8,0) = 0.;
   shape(8,1) = 0.;
   shape(8,2) = (1. - x) * (1. - y);

   shape(9,0) = 0.;
   shape(9,1) = 0.;
   shape(9,2) = x * (1. - y);

   shape(10,0) = 0.;
   shape(10,1) = 0.;
   shape(10,2) = x * y;

   shape(11,0) = 0.;
   shape(11,1) = 0.;
   shape(11,2) = y * (1. - x);

}

void Nedelec1HexFiniteElement::CalcCurlShape(const IntegrationPoint &ip,
                                             DenseMatrix &curl_shape)
const
{
   double x = ip.x, y = ip.y, z = ip.z;

   curl_shape(0,0) = 0.;
   curl_shape(0,1) = y - 1.;
   curl_shape(0,2) = 1. - z;

   curl_shape(2,0) = 0.;
   curl_shape(2,1) = -y;
   curl_shape(2,2) = z - 1.;

   curl_shape(4,0) = 0;
   curl_shape(4,1) = 1. - y;
   curl_shape(4,2) = z;

   curl_shape(6,0) = 0.;
   curl_shape(6,1) = y;
   curl_shape(6,2) = -z;

   curl_shape(1,0) = x;
   curl_shape(1,1) = 0.;
   curl_shape(1,2) = 1. - z;

   curl_shape(3,0) = 1. - x;
   curl_shape(3,1) = 0.;
   curl_shape(3,2) = z - 1.;

   curl_shape(5,0) = -x;
   curl_shape(5,1) = 0.;
   curl_shape(5,2) = z;

   curl_shape(7,0) = x - 1.;
   curl_shape(7,1) = 0.;
   curl_shape(7,2) = -z;

   curl_shape(8,0) = x - 1.;
   curl_shape(8,1) = 1. - y;
   curl_shape(8,2) = 0.;

   curl_shape(9,0) = -x;
   curl_shape(9,1) = y - 1.;
   curl_shape(9,2) = 0;

   curl_shape(10,0) = x;
   curl_shape(10,1) = -y;
   curl_shape(10,2) = 0.;

   curl_shape(11,0) = 1. - x;
   curl_shape(11,1) = y;
   curl_shape(11,2) = 0.;
}

const double Nedelec1HexFiniteElement::tk[12][3] =
{
   {1,0,0}, {0,1,0}, {1,0,0}, {0,1,0},
   {1,0,0}, {0,1,0}, {1,0,0}, {0,1,0},
   {0,0,1}, {0,0,1}, {0,0,1}, {0,0,1}
};

void Nedelec1HexFiniteElement::GetLocalInterpolation (
   ElementTransformation &Trans, DenseMatrix &I) const
{
   int k, j;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(Dof, Dim);
#endif

#ifdef MFEM_DEBUG
   for (k = 0; k < 12; k++)
   {
      CalcVShape (Nodes.IntPoint(k), vshape);
      for (j = 0; j < 12; j++)
      {
         double d = ( vshape(j,0)*tk[k][0] + vshape(j,1)*tk[k][1] +
                      vshape(j,2)*tk[k][2] );
         if (j == k) { d -= 1.0; }
         if (fabs(d) > 1.0e-12)
         {
            mfem::err << "Nedelec1HexFiniteElement::GetLocalInterpolation (...)\n"
                      " k = " << k << ", j = " << j << ", d = " << d << endl;
            mfem_error();
         }
      }
   }
#endif

   IntegrationPoint ip;
   ip.x = ip.y = ip.z = 0.0;
   Trans.SetIntPoint (&ip);
   // Trans must be linear (more to have embedding?)
   const DenseMatrix &J = Trans.Jacobian();
   double vk[3];
   Vector xk (vk, 3);

   for (k = 0; k < 12; k++)
   {
      Trans.Transform (Nodes.IntPoint (k), xk);
      ip.x = vk[0]; ip.y = vk[1]; ip.z = vk[2];
      CalcVShape (ip, vshape);
      //  vk = J tk
      vk[0] = J(0,0)*tk[k][0]+J(0,1)*tk[k][1]+J(0,2)*tk[k][2];
      vk[1] = J(1,0)*tk[k][0]+J(1,1)*tk[k][1]+J(1,2)*tk[k][2];
      vk[2] = J(2,0)*tk[k][0]+J(2,1)*tk[k][1]+J(2,2)*tk[k][2];
      for (j = 0; j < 12; j++)
         if (fabs (I(k,j) = (vshape(j,0)*vk[0]+vshape(j,1)*vk[1]+
                             vshape(j,2)*vk[2])) < 1.0e-12)
         {
            I(k,j) = 0.0;
         }
   }
}

void Nedelec1HexFiniteElement::Project (
   VectorCoefficient &vc, ElementTransformation &Trans,
   Vector &dofs) const
{
   double vk[3];
   Vector xk (vk, 3);

   for (int k = 0; k < 12; k++)
   {
      Trans.SetIntPoint (&Nodes.IntPoint (k));
      const DenseMatrix &J = Trans.Jacobian();

      vc.Eval (xk, Trans, Nodes.IntPoint (k));
      //  xk^t J tk
      dofs(k) =
         vk[0] * ( J(0,0)*tk[k][0]+J(0,1)*tk[k][1]+J(0,2)*tk[k][2] ) +
         vk[1] * ( J(1,0)*tk[k][0]+J(1,1)*tk[k][1]+J(1,2)*tk[k][2] ) +
         vk[2] * ( J(2,0)*tk[k][0]+J(2,1)*tk[k][1]+J(2,2)*tk[k][2] );
   }
}


Nedelec1TetFiniteElement::Nedelec1TetFiniteElement()
   : VectorFiniteElement(3, Geometry::TETRAHEDRON, 6, 1, H_CURL)
{
   // not real nodes ...
   Nodes.IntPoint(0).x = 0.5;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(0).z = 0.0;

   Nodes.IntPoint(1).x = 0.0;
   Nodes.IntPoint(1).y = 0.5;
   Nodes.IntPoint(1).z = 0.0;

   Nodes.IntPoint(2).x = 0.0;
   Nodes.IntPoint(2).y = 0.0;
   Nodes.IntPoint(2).z = 0.5;

   Nodes.IntPoint(3).x = 0.5;
   Nodes.IntPoint(3).y = 0.5;
   Nodes.IntPoint(3).z = 0.0;

   Nodes.IntPoint(4).x = 0.5;
   Nodes.IntPoint(4).y = 0.0;
   Nodes.IntPoint(4).z = 0.5;

   Nodes.IntPoint(5).x = 0.0;
   Nodes.IntPoint(5).y = 0.5;
   Nodes.IntPoint(5).z = 0.5;
}

void Nedelec1TetFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                          DenseMatrix &shape) const
{
   double x = ip.x, y = ip.y, z = ip.z;

   shape(0,0) = 1. - y - z;
   shape(0,1) = x;
   shape(0,2) = x;

   shape(1,0) = y;
   shape(1,1) = 1. - x - z;
   shape(1,2) = y;

   shape(2,0) = z;
   shape(2,1) = z;
   shape(2,2) = 1. - x - y;

   shape(3,0) = -y;
   shape(3,1) = x;
   shape(3,2) = 0.;

   shape(4,0) = -z;
   shape(4,1) = 0.;
   shape(4,2) = x;

   shape(5,0) = 0.;
   shape(5,1) = -z;
   shape(5,2) = y;
}

void Nedelec1TetFiniteElement::CalcCurlShape(const IntegrationPoint &ip,
                                             DenseMatrix &curl_shape)
const
{
   curl_shape(0,0) =  0.;
   curl_shape(0,1) = -2.;
   curl_shape(0,2) =  2.;

   curl_shape(1,0) =  2.;
   curl_shape(1,1) =  0.;
   curl_shape(1,2) = -2.;

   curl_shape(2,0) = -2.;
   curl_shape(2,1) =  2.;
   curl_shape(2,2) =  0.;

   curl_shape(3,0) = 0.;
   curl_shape(3,1) = 0.;
   curl_shape(3,2) = 2.;

   curl_shape(4,0) =  0.;
   curl_shape(4,1) = -2.;
   curl_shape(4,2) =  0.;

   curl_shape(5,0) = 2.;
   curl_shape(5,1) = 0.;
   curl_shape(5,2) = 0.;
}

const double Nedelec1TetFiniteElement::tk[6][3] =
{{1,0,0}, {0,1,0}, {0,0,1}, {-1,1,0}, {-1,0,1}, {0,-1,1}};

void Nedelec1TetFiniteElement::GetLocalInterpolation (
   ElementTransformation &Trans, DenseMatrix &I) const
{
   int k, j;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(Dof, Dim);
#endif

#ifdef MFEM_DEBUG
   for (k = 0; k < 6; k++)
   {
      CalcVShape (Nodes.IntPoint(k), vshape);
      for (j = 0; j < 6; j++)
      {
         double d = ( vshape(j,0)*tk[k][0] + vshape(j,1)*tk[k][1] +
                      vshape(j,2)*tk[k][2] );
         if (j == k) { d -= 1.0; }
         if (fabs(d) > 1.0e-12)
         {
            mfem::err << "Nedelec1TetFiniteElement::GetLocalInterpolation (...)\n"
                      " k = " << k << ", j = " << j << ", d = " << d << endl;
            mfem_error();
         }
      }
   }
#endif

   IntegrationPoint ip;
   ip.x = ip.y = ip.z = 0.0;
   Trans.SetIntPoint (&ip);
   // Trans must be linear
   const DenseMatrix &J = Trans.Jacobian();
   double vk[3];
   Vector xk (vk, 3);

   for (k = 0; k < 6; k++)
   {
      Trans.Transform (Nodes.IntPoint (k), xk);
      ip.x = vk[0]; ip.y = vk[1]; ip.z = vk[2];
      CalcVShape (ip, vshape);
      //  vk = J tk
      vk[0] = J(0,0)*tk[k][0]+J(0,1)*tk[k][1]+J(0,2)*tk[k][2];
      vk[1] = J(1,0)*tk[k][0]+J(1,1)*tk[k][1]+J(1,2)*tk[k][2];
      vk[2] = J(2,0)*tk[k][0]+J(2,1)*tk[k][1]+J(2,2)*tk[k][2];
      for (j = 0; j < 6; j++)
         if (fabs (I(k,j) = (vshape(j,0)*vk[0]+vshape(j,1)*vk[1]+
                             vshape(j,2)*vk[2])) < 1.0e-12)
         {
            I(k,j) = 0.0;
         }
   }
}

void Nedelec1TetFiniteElement::Project (
   VectorCoefficient &vc, ElementTransformation &Trans,
   Vector &dofs) const
{
   double vk[3];
   Vector xk (vk, 3);

   for (int k = 0; k < 6; k++)
   {
      Trans.SetIntPoint (&Nodes.IntPoint (k));
      const DenseMatrix &J = Trans.Jacobian();

      vc.Eval (xk, Trans, Nodes.IntPoint (k));
      //  xk^t J tk
      dofs(k) =
         vk[0] * ( J(0,0)*tk[k][0]+J(0,1)*tk[k][1]+J(0,2)*tk[k][2] ) +
         vk[1] * ( J(1,0)*tk[k][0]+J(1,1)*tk[k][1]+J(1,2)*tk[k][2] ) +
         vk[2] * ( J(2,0)*tk[k][0]+J(2,1)*tk[k][1]+J(2,2)*tk[k][2] );
   }
}

RT0HexFiniteElement::RT0HexFiniteElement()
   : VectorFiniteElement(3, Geometry::CUBE, 6, 1, H_DIV, FunctionSpace::Qk)
{
   // not real nodes ...
   // z = 0, y = 0, x = 1, y = 1, x = 0, z = 1
   Nodes.IntPoint(0).x = 0.5;
   Nodes.IntPoint(0).y = 0.5;
   Nodes.IntPoint(0).z = 0.0;

   Nodes.IntPoint(1).x = 0.5;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(1).z = 0.5;

   Nodes.IntPoint(2).x = 1.0;
   Nodes.IntPoint(2).y = 0.5;
   Nodes.IntPoint(2).z = 0.5;

   Nodes.IntPoint(3).x = 0.5;
   Nodes.IntPoint(3).y = 1.0;
   Nodes.IntPoint(3).z = 0.5;

   Nodes.IntPoint(4).x = 0.0;
   Nodes.IntPoint(4).y = 0.5;
   Nodes.IntPoint(4).z = 0.5;

   Nodes.IntPoint(5).x = 0.5;
   Nodes.IntPoint(5).y = 0.5;
   Nodes.IntPoint(5).z = 1.0;
}

void RT0HexFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                     DenseMatrix &shape) const
{
   double x = ip.x, y = ip.y, z = ip.z;
   // z = 0
   shape(0,0) = 0.;
   shape(0,1) = 0.;
   shape(0,2) = z - 1.;
   // y = 0
   shape(1,0) = 0.;
   shape(1,1) = y - 1.;
   shape(1,2) = 0.;
   // x = 1
   shape(2,0) = x;
   shape(2,1) = 0.;
   shape(2,2) = 0.;
   // y = 1
   shape(3,0) = 0.;
   shape(3,1) = y;
   shape(3,2) = 0.;
   // x = 0
   shape(4,0) = x - 1.;
   shape(4,1) = 0.;
   shape(4,2) = 0.;
   // z = 1
   shape(5,0) = 0.;
   shape(5,1) = 0.;
   shape(5,2) = z;
}

void RT0HexFiniteElement::CalcDivShape(const IntegrationPoint &ip,
                                       Vector &divshape) const
{
   divshape(0) = 1.;
   divshape(1) = 1.;
   divshape(2) = 1.;
   divshape(3) = 1.;
   divshape(4) = 1.;
   divshape(5) = 1.;
}

const double RT0HexFiniteElement::nk[6][3] =
{{0,0,-1}, {0,-1,0}, {1,0,0}, {0,1,0}, {-1,0,0}, {0,0,1}};

void RT0HexFiniteElement::GetLocalInterpolation (
   ElementTransformation &Trans, DenseMatrix &I) const
{
   int k, j;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(Dof, Dim);
   DenseMatrix Jinv(Dim);
#endif

#ifdef MFEM_DEBUG
   for (k = 0; k < 6; k++)
   {
      CalcVShape (Nodes.IntPoint(k), vshape);
      for (j = 0; j < 6; j++)
      {
         double d = ( vshape(j,0)*nk[k][0] + vshape(j,1)*nk[k][1] +
                      vshape(j,2)*nk[k][2] );
         if (j == k) { d -= 1.0; }
         if (fabs(d) > 1.0e-12)
         {
            mfem::err << "RT0HexFiniteElement::GetLocalInterpolation (...)\n"
                      " k = " << k << ", j = " << j << ", d = " << d << endl;
            mfem_error();
         }
      }
   }
#endif

   IntegrationPoint ip;
   ip.x = ip.y = ip.z = 0.0;
   Trans.SetIntPoint (&ip);
   // Trans must be linear
   // set Jinv = |J| J^{-t} = adj(J)^t
   CalcAdjugateTranspose (Trans.Jacobian(), Jinv);
   double vk[3];
   Vector xk (vk, 3);

   for (k = 0; k < 6; k++)
   {
      Trans.Transform (Nodes.IntPoint (k), xk);
      ip.x = vk[0]; ip.y = vk[1]; ip.z = vk[2];
      CalcVShape (ip, vshape);
      //  vk = |J| J^{-t} nk
      vk[0] = Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1]+Jinv(0,2)*nk[k][2];
      vk[1] = Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1]+Jinv(1,2)*nk[k][2];
      vk[2] = Jinv(2,0)*nk[k][0]+Jinv(2,1)*nk[k][1]+Jinv(2,2)*nk[k][2];
      for (j = 0; j < 6; j++)
         if (fabs (I(k,j) = (vshape(j,0)*vk[0]+vshape(j,1)*vk[1]+
                             vshape(j,2)*vk[2])) < 1.0e-12)
         {
            I(k,j) = 0.0;
         }
   }
}

void RT0HexFiniteElement::Project (
   VectorCoefficient &vc, ElementTransformation &Trans,
   Vector &dofs) const
{
   double vk[3];
   Vector xk (vk, 3);
#ifdef MFEM_THREAD_SAFE
   DenseMatrix Jinv(Dim);
#endif

   for (int k = 0; k < 6; k++)
   {
      Trans.SetIntPoint (&Nodes.IntPoint (k));
      // set Jinv = |J| J^{-t} = adj(J)^t
      CalcAdjugateTranspose (Trans.Jacobian(), Jinv);

      vc.Eval (xk, Trans, Nodes.IntPoint (k));
      //  xk^t |J| J^{-t} nk
      dofs(k) =
         vk[0] * ( Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1]+Jinv(0,2)*nk[k][2] ) +
         vk[1] * ( Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1]+Jinv(1,2)*nk[k][2] ) +
         vk[2] * ( Jinv(2,0)*nk[k][0]+Jinv(2,1)*nk[k][1]+Jinv(2,2)*nk[k][2] );
   }
}

RT1HexFiniteElement::RT1HexFiniteElement()
   : VectorFiniteElement(3, Geometry::CUBE, 36, 2, H_DIV, FunctionSpace::Qk)
{
   // z = 0
   Nodes.IntPoint(2).x  = 1./3.;
   Nodes.IntPoint(2).y  = 1./3.;
   Nodes.IntPoint(2).z  = 0.0;
   Nodes.IntPoint(3).x  = 2./3.;
   Nodes.IntPoint(3).y  = 1./3.;
   Nodes.IntPoint(3).z  = 0.0;
   Nodes.IntPoint(0).x  = 1./3.;
   Nodes.IntPoint(0).y  = 2./3.;
   Nodes.IntPoint(0).z  = 0.0;
   Nodes.IntPoint(1).x  = 2./3.;
   Nodes.IntPoint(1).y  = 2./3.;
   Nodes.IntPoint(1).z  = 0.0;
   // y = 0
   Nodes.IntPoint(4).x  = 1./3.;
   Nodes.IntPoint(4).y  = 0.0;
   Nodes.IntPoint(4).z  = 1./3.;
   Nodes.IntPoint(5).x  = 2./3.;
   Nodes.IntPoint(5).y  = 0.0;
   Nodes.IntPoint(5).z  = 1./3.;
   Nodes.IntPoint(6).x  = 1./3.;
   Nodes.IntPoint(6).y  = 0.0;
   Nodes.IntPoint(6).z  = 2./3.;
   Nodes.IntPoint(7).x  = 2./3.;
   Nodes.IntPoint(7).y  = 0.0;
   Nodes.IntPoint(7).z  = 2./3.;
   // x = 1
   Nodes.IntPoint(8).x  = 1.0;
   Nodes.IntPoint(8).y  = 1./3.;
   Nodes.IntPoint(8).z  = 1./3.;
   Nodes.IntPoint(9).x  = 1.0;
   Nodes.IntPoint(9).y  = 2./3.;
   Nodes.IntPoint(9).z  = 1./3.;
   Nodes.IntPoint(10).x = 1.0;
   Nodes.IntPoint(10).y = 1./3.;
   Nodes.IntPoint(10).z = 2./3.;
   Nodes.IntPoint(11).x = 1.0;
   Nodes.IntPoint(11).y = 2./3.;
   Nodes.IntPoint(11).z = 2./3.;
   // y = 1
   Nodes.IntPoint(13).x = 1./3.;
   Nodes.IntPoint(13).y = 1.0;
   Nodes.IntPoint(13).z = 1./3.;
   Nodes.IntPoint(12).x = 2./3.;
   Nodes.IntPoint(12).y = 1.0;
   Nodes.IntPoint(12).z = 1./3.;
   Nodes.IntPoint(15).x = 1./3.;
   Nodes.IntPoint(15).y = 1.0;
   Nodes.IntPoint(15).z = 2./3.;
   Nodes.IntPoint(14).x = 2./3.;
   Nodes.IntPoint(14).y = 1.0;
   Nodes.IntPoint(14).z = 2./3.;
   // x = 0
   Nodes.IntPoint(17).x = 0.0;
   Nodes.IntPoint(17).y = 1./3.;
   Nodes.IntPoint(17).z = 1./3.;
   Nodes.IntPoint(16).x = 0.0;
   Nodes.IntPoint(16).y = 2./3.;
   Nodes.IntPoint(16).z = 1./3.;
   Nodes.IntPoint(19).x = 0.0;
   Nodes.IntPoint(19).y = 1./3.;
   Nodes.IntPoint(19).z = 2./3.;
   Nodes.IntPoint(18).x = 0.0;
   Nodes.IntPoint(18).y = 2./3.;
   Nodes.IntPoint(18).z = 2./3.;
   // z = 1
   Nodes.IntPoint(20).x = 1./3.;
   Nodes.IntPoint(20).y = 1./3.;
   Nodes.IntPoint(20).z = 1.0;
   Nodes.IntPoint(21).x = 2./3.;
   Nodes.IntPoint(21).y = 1./3.;
   Nodes.IntPoint(21).z = 1.0;
   Nodes.IntPoint(22).x = 1./3.;
   Nodes.IntPoint(22).y = 2./3.;
   Nodes.IntPoint(22).z = 1.0;
   Nodes.IntPoint(23).x = 2./3.;
   Nodes.IntPoint(23).y = 2./3.;
   Nodes.IntPoint(23).z = 1.0;
   // x = 0.5 (interior)
   Nodes.IntPoint(24).x = 0.5;
   Nodes.IntPoint(24).y = 1./3.;
   Nodes.IntPoint(24).z = 1./3.;
   Nodes.IntPoint(25).x = 0.5;
   Nodes.IntPoint(25).y = 1./3.;
   Nodes.IntPoint(25).z = 2./3.;
   Nodes.IntPoint(26).x = 0.5;
   Nodes.IntPoint(26).y = 2./3.;
   Nodes.IntPoint(26).z = 1./3.;
   Nodes.IntPoint(27).x = 0.5;
   Nodes.IntPoint(27).y = 2./3.;
   Nodes.IntPoint(27).z = 2./3.;
   // y = 0.5 (interior)
   Nodes.IntPoint(28).x = 1./3.;
   Nodes.IntPoint(28).y = 0.5;
   Nodes.IntPoint(28).z = 1./3.;
   Nodes.IntPoint(29).x = 1./3.;
   Nodes.IntPoint(29).y = 0.5;
   Nodes.IntPoint(29).z = 2./3.;
   Nodes.IntPoint(30).x = 2./3.;
   Nodes.IntPoint(30).y = 0.5;
   Nodes.IntPoint(30).z = 1./3.;
   Nodes.IntPoint(31).x = 2./3.;
   Nodes.IntPoint(31).y = 0.5;
   Nodes.IntPoint(31).z = 2./3.;
   // z = 0.5 (interior)
   Nodes.IntPoint(32).x = 1./3.;
   Nodes.IntPoint(32).y = 1./3.;
   Nodes.IntPoint(32).z = 0.5;
   Nodes.IntPoint(33).x = 1./3.;
   Nodes.IntPoint(33).y = 2./3.;
   Nodes.IntPoint(33).z = 0.5;
   Nodes.IntPoint(34).x = 2./3.;
   Nodes.IntPoint(34).y = 1./3.;
   Nodes.IntPoint(34).z = 0.5;
   Nodes.IntPoint(35).x = 2./3.;
   Nodes.IntPoint(35).y = 2./3.;
   Nodes.IntPoint(35).z = 0.5;
}

void RT1HexFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                     DenseMatrix &shape) const
{
   double x = ip.x, y = ip.y, z = ip.z;
   // z = 0
   shape(2,0)  = 0.;
   shape(2,1)  = 0.;
   shape(2,2)  = -(1. - 3.*z + 2.*z*z)*( 2. - 3.*x)*( 2. - 3.*y);
   shape(3,0)  = 0.;
   shape(3,1)  = 0.;
   shape(3,2)  = -(1. - 3.*z + 2.*z*z)*(-1. + 3.*x)*( 2. - 3.*y);
   shape(0,0)  = 0.;
   shape(0,1)  = 0.;
   shape(0,2)  = -(1. - 3.*z + 2.*z*z)*( 2. - 3.*x)*(-1. + 3.*y);
   shape(1,0)  = 0.;
   shape(1,1)  = 0.;
   shape(1,2)  = -(1. - 3.*z + 2.*z*z)*(-1. + 3.*x)*(-1. + 3.*y);
   // y = 0
   shape(4,0)  = 0.;
   shape(4,1)  = -(1. - 3.*y + 2.*y*y)*( 2. - 3.*x)*( 2. - 3.*z);
   shape(4,2)  = 0.;
   shape(5,0)  = 0.;
   shape(5,1)  = -(1. - 3.*y + 2.*y*y)*(-1. + 3.*x)*( 2. - 3.*z);
   shape(5,2)  = 0.;
   shape(6,0)  = 0.;
   shape(6,1)  = -(1. - 3.*y + 2.*y*y)*( 2. - 3.*x)*(-1. + 3.*z);
   shape(6,2)  = 0.;
   shape(7,0)  = 0.;
   shape(7,1)  = -(1. - 3.*y + 2.*y*y)*(-1. + 3.*x)*(-1. + 3.*z);
   shape(7,2)  = 0.;
   // x = 1
   shape(8,0)  = (-x + 2.*x*x)*( 2. - 3.*y)*( 2. - 3.*z);
   shape(8,1)  = 0.;
   shape(8,2)  = 0.;
   shape(9,0)  = (-x + 2.*x*x)*(-1. + 3.*y)*( 2. - 3.*z);
   shape(9,1)  = 0.;
   shape(9,2)  = 0.;
   shape(10,0) = (-x + 2.*x*x)*( 2. - 3.*y)*(-1. + 3.*z);
   shape(10,1) = 0.;
   shape(10,2) = 0.;
   shape(11,0) = (-x + 2.*x*x)*(-1. + 3.*y)*(-1. + 3.*z);
   shape(11,1) = 0.;
   shape(11,2) = 0.;
   // y = 1
   shape(13,0) = 0.;
   shape(13,1) = (-y + 2.*y*y)*( 2. - 3.*x)*( 2. - 3.*z);
   shape(13,2) = 0.;
   shape(12,0) = 0.;
   shape(12,1) = (-y + 2.*y*y)*(-1. + 3.*x)*( 2. - 3.*z);
   shape(12,2) = 0.;
   shape(15,0) = 0.;
   shape(15,1) = (-y + 2.*y*y)*( 2. - 3.*x)*(-1. + 3.*z);
   shape(15,2) = 0.;
   shape(14,0) = 0.;
   shape(14,1) = (-y + 2.*y*y)*(-1. + 3.*x)*(-1. + 3.*z);
   shape(14,2) = 0.;
   // x = 0
   shape(17,0) = -(1. - 3.*x + 2.*x*x)*( 2. - 3.*y)*( 2. - 3.*z);
   shape(17,1) = 0.;
   shape(17,2) = 0.;
   shape(16,0) = -(1. - 3.*x + 2.*x*x)*(-1. + 3.*y)*( 2. - 3.*z);
   shape(16,1) = 0.;
   shape(16,2) = 0.;
   shape(19,0) = -(1. - 3.*x + 2.*x*x)*( 2. - 3.*y)*(-1. + 3.*z);
   shape(19,1) = 0.;
   shape(19,2) = 0.;
   shape(18,0) = -(1. - 3.*x + 2.*x*x)*(-1. + 3.*y)*(-1. + 3.*z);
   shape(18,1) = 0.;
   shape(18,2) = 0.;
   // z = 1
   shape(20,0) = 0.;
   shape(20,1) = 0.;
   shape(20,2) = (-z + 2.*z*z)*( 2. - 3.*x)*( 2. - 3.*y);
   shape(21,0) = 0.;
   shape(21,1) = 0.;
   shape(21,2) = (-z + 2.*z*z)*(-1. + 3.*x)*( 2. - 3.*y);
   shape(22,0) = 0.;
   shape(22,1) = 0.;
   shape(22,2) = (-z + 2.*z*z)*( 2. - 3.*x)*(-1. + 3.*y);
   shape(23,0) = 0.;
   shape(23,1) = 0.;
   shape(23,2) = (-z + 2.*z*z)*(-1. + 3.*x)*(-1. + 3.*y);
   // x = 0.5 (interior)
   shape(24,0) = (4.*x - 4.*x*x)*( 2. - 3.*y)*( 2. - 3.*z);
   shape(24,1) = 0.;
   shape(24,2) = 0.;
   shape(25,0) = (4.*x - 4.*x*x)*( 2. - 3.*y)*(-1. + 3.*z);
   shape(25,1) = 0.;
   shape(25,2) = 0.;
   shape(26,0) = (4.*x - 4.*x*x)*(-1. + 3.*y)*( 2. - 3.*z);
   shape(26,1) = 0.;
   shape(26,2) = 0.;
   shape(27,0) = (4.*x - 4.*x*x)*(-1. + 3.*y)*(-1. + 3.*z);
   shape(27,1) = 0.;
   shape(27,2) = 0.;
   // y = 0.5 (interior)
   shape(28,0) = 0.;
   shape(28,1) = (4.*y - 4.*y*y)*( 2. - 3.*x)*( 2. - 3.*z);
   shape(28,2) = 0.;
   shape(29,0) = 0.;
   shape(29,1) = (4.*y - 4.*y*y)*( 2. - 3.*x)*(-1. + 3.*z);
   shape(29,2) = 0.;
   shape(30,0) = 0.;
   shape(30,1) = (4.*y - 4.*y*y)*(-1. + 3.*x)*( 2. - 3.*z);
   shape(30,2) = 0.;
   shape(31,0) = 0.;
   shape(31,1) = (4.*y - 4.*y*y)*(-1. + 3.*x)*(-1. + 3.*z);
   shape(31,2) = 0.;
   // z = 0.5 (interior)
   shape(32,0) = 0.;
   shape(32,1) = 0.;
   shape(32,2) = (4.*z - 4.*z*z)*( 2. - 3.*x)*( 2. - 3.*y);
   shape(33,0) = 0.;
   shape(33,1) = 0.;
   shape(33,2) = (4.*z - 4.*z*z)*( 2. - 3.*x)*(-1. + 3.*y);
   shape(34,0) = 0.;
   shape(34,1) = 0.;
   shape(34,2) = (4.*z - 4.*z*z)*(-1. + 3.*x)*( 2. - 3.*y);
   shape(35,0) = 0.;
   shape(35,1) = 0.;
   shape(35,2) = (4.*z - 4.*z*z)*(-1. + 3.*x)*(-1. + 3.*y);
}

void RT1HexFiniteElement::CalcDivShape(const IntegrationPoint &ip,
                                       Vector &divshape) const
{
   double x = ip.x, y = ip.y, z = ip.z;
   // z = 0
   divshape(2)  = -(-3. + 4.*z)*( 2. - 3.*x)*( 2. - 3.*y);
   divshape(3)  = -(-3. + 4.*z)*(-1. + 3.*x)*( 2. - 3.*y);
   divshape(0)  = -(-3. + 4.*z)*( 2. - 3.*x)*(-1. + 3.*y);
   divshape(1)  = -(-3. + 4.*z)*(-1. + 3.*x)*(-1. + 3.*y);
   // y = 0
   divshape(4)  = -(-3. + 4.*y)*( 2. - 3.*x)*( 2. - 3.*z);
   divshape(5)  = -(-3. + 4.*y)*(-1. + 3.*x)*( 2. - 3.*z);
   divshape(6)  = -(-3. + 4.*y)*( 2. - 3.*x)*(-1. + 3.*z);
   divshape(7)  = -(-3. + 4.*y)*(-1. + 3.*x)*(-1. + 3.*z);
   // x = 1
   divshape(8)  = (-1. + 4.*x)*( 2. - 3.*y)*( 2. - 3.*z);
   divshape(9)  = (-1. + 4.*x)*(-1. + 3.*y)*( 2. - 3.*z);
   divshape(10) = (-1. + 4.*x)*( 2. - 3.*y)*(-1. + 3.*z);
   divshape(11) = (-1. + 4.*x)*(-1. + 3.*y)*(-1. + 3.*z);
   // y = 1
   divshape(13) = (-1. + 4.*y)*( 2. - 3.*x)*( 2. - 3.*z);
   divshape(12) = (-1. + 4.*y)*(-1. + 3.*x)*( 2. - 3.*z);
   divshape(15) = (-1. + 4.*y)*( 2. - 3.*x)*(-1. + 3.*z);
   divshape(14) = (-1. + 4.*y)*(-1. + 3.*x)*(-1. + 3.*z);
   // x = 0
   divshape(17) = -(-3. + 4.*x)*( 2. - 3.*y)*( 2. - 3.*z);
   divshape(16) = -(-3. + 4.*x)*(-1. + 3.*y)*( 2. - 3.*z);
   divshape(19) = -(-3. + 4.*x)*( 2. - 3.*y)*(-1. + 3.*z);
   divshape(18) = -(-3. + 4.*x)*(-1. + 3.*y)*(-1. + 3.*z);
   // z = 1
   divshape(20) = (-1. + 4.*z)*( 2. - 3.*x)*( 2. - 3.*y);
   divshape(21) = (-1. + 4.*z)*(-1. + 3.*x)*( 2. - 3.*y);
   divshape(22) = (-1. + 4.*z)*( 2. - 3.*x)*(-1. + 3.*y);
   divshape(23) = (-1. + 4.*z)*(-1. + 3.*x)*(-1. + 3.*y);
   // x = 0.5 (interior)
   divshape(24) = ( 4. - 8.*x)*( 2. - 3.*y)*( 2. - 3.*z);
   divshape(25) = ( 4. - 8.*x)*( 2. - 3.*y)*(-1. + 3.*z);
   divshape(26) = ( 4. - 8.*x)*(-1. + 3.*y)*( 2. - 3.*z);
   divshape(27) = ( 4. - 8.*x)*(-1. + 3.*y)*(-1. + 3.*z);
   // y = 0.5 (interior)
   divshape(28) = ( 4. - 8.*y)*( 2. - 3.*x)*( 2. - 3.*z);
   divshape(29) = ( 4. - 8.*y)*( 2. - 3.*x)*(-1. + 3.*z);
   divshape(30) = ( 4. - 8.*y)*(-1. + 3.*x)*( 2. - 3.*z);
   divshape(31) = ( 4. - 8.*y)*(-1. + 3.*x)*(-1. + 3.*z);
   // z = 0.5 (interior)
   divshape(32) = ( 4. - 8.*z)*( 2. - 3.*x)*( 2. - 3.*y);
   divshape(33) = ( 4. - 8.*z)*( 2. - 3.*x)*(-1. + 3.*y);
   divshape(34) = ( 4. - 8.*z)*(-1. + 3.*x)*( 2. - 3.*y);
   divshape(35) = ( 4. - 8.*z)*(-1. + 3.*x)*(-1. + 3.*y);
}

const double RT1HexFiniteElement::nk[36][3] =
{
   {0, 0,-1}, {0, 0,-1}, {0, 0,-1}, {0, 0,-1},
   {0,-1, 0}, {0,-1, 0}, {0,-1, 0}, {0,-1, 0},
   {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0},
   {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0},
   {-1,0, 0}, {-1,0, 0}, {-1,0, 0}, {-1,0, 0},
   {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1},
   {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0},
   {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0},
   {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}
};

void RT1HexFiniteElement::GetLocalInterpolation (
   ElementTransformation &Trans, DenseMatrix &I) const
{
   int k, j;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(Dof, Dim);
   DenseMatrix Jinv(Dim);
#endif

#ifdef MFEM_DEBUG
   for (k = 0; k < 36; k++)
   {
      CalcVShape (Nodes.IntPoint(k), vshape);
      for (j = 0; j < 36; j++)
      {
         double d = ( vshape(j,0)*nk[k][0] + vshape(j,1)*nk[k][1] +
                      vshape(j,2)*nk[k][2] );
         if (j == k) { d -= 1.0; }
         if (fabs(d) > 1.0e-12)
         {
            mfem::err << "RT0HexFiniteElement::GetLocalInterpolation (...)\n"
                      " k = " << k << ", j = " << j << ", d = " << d << endl;
            mfem_error();
         }
      }
   }
#endif

   IntegrationPoint ip;
   ip.x = ip.y = ip.z = 0.0;
   Trans.SetIntPoint (&ip);
   // Trans must be linear
   // set Jinv = |J| J^{-t} = adj(J)^t
   CalcAdjugateTranspose (Trans.Jacobian(), Jinv);
   double vk[3];
   Vector xk (vk, 3);

   for (k = 0; k < 36; k++)
   {
      Trans.Transform (Nodes.IntPoint (k), xk);
      ip.x = vk[0]; ip.y = vk[1]; ip.z = vk[2];
      CalcVShape (ip, vshape);
      //  vk = |J| J^{-t} nk
      vk[0] = Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1]+Jinv(0,2)*nk[k][2];
      vk[1] = Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1]+Jinv(1,2)*nk[k][2];
      vk[2] = Jinv(2,0)*nk[k][0]+Jinv(2,1)*nk[k][1]+Jinv(2,2)*nk[k][2];
      for (j = 0; j < 36; j++)
         if (fabs (I(k,j) = (vshape(j,0)*vk[0]+vshape(j,1)*vk[1]+
                             vshape(j,2)*vk[2])) < 1.0e-12)
         {
            I(k,j) = 0.0;
         }
   }
}

void RT1HexFiniteElement::Project (
   VectorCoefficient &vc, ElementTransformation &Trans,
   Vector &dofs) const
{
   double vk[3];
   Vector xk (vk, 3);
#ifdef MFEM_THREAD_SAFE
   DenseMatrix Jinv(Dim);
#endif

   for (int k = 0; k < 36; k++)
   {
      Trans.SetIntPoint (&Nodes.IntPoint (k));
      // set Jinv = |J| J^{-t} = adj(J)^t
      CalcAdjugateTranspose (Trans.Jacobian(), Jinv);

      vc.Eval (xk, Trans, Nodes.IntPoint (k));
      //  xk^t |J| J^{-t} nk
      dofs(k) =
         vk[0] * ( Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1]+Jinv(0,2)*nk[k][2] ) +
         vk[1] * ( Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1]+Jinv(1,2)*nk[k][2] ) +
         vk[2] * ( Jinv(2,0)*nk[k][0]+Jinv(2,1)*nk[k][1]+Jinv(2,2)*nk[k][2] );
   }
}

RT0TetFiniteElement::RT0TetFiniteElement()
   : VectorFiniteElement(3, Geometry::TETRAHEDRON, 4, 1, H_DIV)
{
   // not real nodes ...
   Nodes.IntPoint(0).x = 0.33333333333333333333;
   Nodes.IntPoint(0).y = 0.33333333333333333333;
   Nodes.IntPoint(0).z = 0.33333333333333333333;

   Nodes.IntPoint(1).x = 0.0;
   Nodes.IntPoint(1).y = 0.33333333333333333333;
   Nodes.IntPoint(1).z = 0.33333333333333333333;

   Nodes.IntPoint(2).x = 0.33333333333333333333;
   Nodes.IntPoint(2).y = 0.0;
   Nodes.IntPoint(2).z = 0.33333333333333333333;

   Nodes.IntPoint(3).x = 0.33333333333333333333;
   Nodes.IntPoint(3).y = 0.33333333333333333333;
   Nodes.IntPoint(3).z = 0.0;
}

void RT0TetFiniteElement::CalcVShape(const IntegrationPoint &ip,
                                     DenseMatrix &shape) const
{
   double x2 = 2.0*ip.x, y2 = 2.0*ip.y, z2 = 2.0*ip.z;

   shape(0,0) = x2;
   shape(0,1) = y2;
   shape(0,2) = z2;

   shape(1,0) = x2 - 2.0;
   shape(1,1) = y2;
   shape(1,2) = z2;

   shape(2,0) = x2;
   shape(2,1) = y2 - 2.0;
   shape(2,2) = z2;

   shape(3,0) = x2;
   shape(3,1) = y2;
   shape(3,2) = z2 - 2.0;
}

void RT0TetFiniteElement::CalcDivShape(const IntegrationPoint &ip,
                                       Vector &divshape) const
{
   divshape(0) = 6.0;
   divshape(1) = 6.0;
   divshape(2) = 6.0;
   divshape(3) = 6.0;
}

const double RT0TetFiniteElement::nk[4][3] =
{{.5,.5,.5}, {-.5,0,0}, {0,-.5,0}, {0,0,-.5}};

void RT0TetFiniteElement::GetLocalInterpolation (
   ElementTransformation &Trans, DenseMatrix &I) const
{
   int k, j;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(Dof, Dim);
   DenseMatrix Jinv(Dim);
#endif

#ifdef MFEM_DEBUG
   for (k = 0; k < 4; k++)
   {
      CalcVShape (Nodes.IntPoint(k), vshape);
      for (j = 0; j < 4; j++)
      {
         double d = ( vshape(j,0)*nk[k][0] + vshape(j,1)*nk[k][1] +
                      vshape(j,2)*nk[k][2] );
         if (j == k) { d -= 1.0; }
         if (fabs(d) > 1.0e-12)
         {
            mfem::err << "RT0TetFiniteElement::GetLocalInterpolation (...)\n"
                      " k = " << k << ", j = " << j << ", d = " << d << endl;
            mfem_error();
         }
      }
   }
#endif

   IntegrationPoint ip;
   ip.x = ip.y = ip.z = 0.0;
   Trans.SetIntPoint (&ip);
   // Trans must be linear
   // set Jinv = |J| J^{-t} = adj(J)^t
   CalcAdjugateTranspose (Trans.Jacobian(), Jinv);
   double vk[3];
   Vector xk (vk, 3);

   for (k = 0; k < 4; k++)
   {
      Trans.Transform (Nodes.IntPoint (k), xk);
      ip.x = vk[0]; ip.y = vk[1]; ip.z = vk[2];
      CalcVShape (ip, vshape);
      //  vk = |J| J^{-t} nk
      vk[0] = Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1]+Jinv(0,2)*nk[k][2];
      vk[1] = Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1]+Jinv(1,2)*nk[k][2];
      vk[2] = Jinv(2,0)*nk[k][0]+Jinv(2,1)*nk[k][1]+Jinv(2,2)*nk[k][2];
      for (j = 0; j < 4; j++)
         if (fabs (I(k,j) = (vshape(j,0)*vk[0]+vshape(j,1)*vk[1]+
                             vshape(j,2)*vk[2])) < 1.0e-12)
         {
            I(k,j) = 0.0;
         }
   }
}

void RT0TetFiniteElement::Project (
   VectorCoefficient &vc, ElementTransformation &Trans,
   Vector &dofs) const
{
   double vk[3];
   Vector xk (vk, 3);
#ifdef MFEM_THREAD_SAFE
   DenseMatrix Jinv(Dim);
#endif

   for (int k = 0; k < 4; k++)
   {
      Trans.SetIntPoint (&Nodes.IntPoint (k));
      // set Jinv = |J| J^{-t} = adj(J)^t
      CalcAdjugateTranspose (Trans.Jacobian(), Jinv);

      vc.Eval (xk, Trans, Nodes.IntPoint (k));
      //  xk^t |J| J^{-t} nk
      dofs(k) =
         vk[0] * ( Jinv(0,0)*nk[k][0]+Jinv(0,1)*nk[k][1]+Jinv(0,2)*nk[k][2] ) +
         vk[1] * ( Jinv(1,0)*nk[k][0]+Jinv(1,1)*nk[k][1]+Jinv(1,2)*nk[k][2] ) +
         vk[2] * ( Jinv(2,0)*nk[k][0]+Jinv(2,1)*nk[k][1]+Jinv(2,2)*nk[k][2] );
   }
}

RotTriLinearHexFiniteElement::RotTriLinearHexFiniteElement()
   : NodalFiniteElement(3, Geometry::CUBE, 6, 2, FunctionSpace::Qk)
{
   Nodes.IntPoint(0).x = 0.5;
   Nodes.IntPoint(0).y = 0.5;
   Nodes.IntPoint(0).z = 0.0;

   Nodes.IntPoint(1).x = 0.5;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(1).z = 0.5;

   Nodes.IntPoint(2).x = 1.0;
   Nodes.IntPoint(2).y = 0.5;
   Nodes.IntPoint(2).z = 0.5;

   Nodes.IntPoint(3).x = 0.5;
   Nodes.IntPoint(3).y = 1.0;
   Nodes.IntPoint(3).z = 0.5;

   Nodes.IntPoint(4).x = 0.0;
   Nodes.IntPoint(4).y = 0.5;
   Nodes.IntPoint(4).z = 0.5;

   Nodes.IntPoint(5).x = 0.5;
   Nodes.IntPoint(5).y = 0.5;
   Nodes.IntPoint(5).z = 1.0;
}

void RotTriLinearHexFiniteElement::CalcShape(const IntegrationPoint &ip,
                                             Vector &shape) const
{
   double x = 2. * ip.x - 1.;
   double y = 2. * ip.y - 1.;
   double z = 2. * ip.z - 1.;
   double f5 = x * x - y * y;
   double f6 = y * y - z * z;

   shape(0) = (1./6.) * (1. - 3. * z -      f5 - 2. * f6);
   shape(1) = (1./6.) * (1. - 3. * y -      f5 +      f6);
   shape(2) = (1./6.) * (1. + 3. * x + 2. * f5 +      f6);
   shape(3) = (1./6.) * (1. + 3. * y -      f5 +      f6);
   shape(4) = (1./6.) * (1. - 3. * x + 2. * f5 +      f6);
   shape(5) = (1./6.) * (1. + 3. * z -      f5 - 2. * f6);
}

void RotTriLinearHexFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                              DenseMatrix &dshape) const
{
   const double a = 2./3.;

   double xt = a * (1. - 2. * ip.x);
   double yt = a * (1. - 2. * ip.y);
   double zt = a * (1. - 2. * ip.z);

   dshape(0,0) = xt;
   dshape(0,1) = yt;
   dshape(0,2) = -1. - 2. * zt;

   dshape(1,0) = xt;
   dshape(1,1) = -1. - 2. * yt;
   dshape(1,2) = zt;

   dshape(2,0) = 1. - 2. * xt;
   dshape(2,1) = yt;
   dshape(2,2) = zt;

   dshape(3,0) = xt;
   dshape(3,1) = 1. - 2. * yt;
   dshape(3,2) = zt;

   dshape(4,0) = -1. - 2. * xt;
   dshape(4,1) = yt;
   dshape(4,2) = zt;

   dshape(5,0) = xt;
   dshape(5,1) = yt;
   dshape(5,2) = 1. - 2. * zt;
}


Poly_1D::Basis::Basis(const int p, const double *nodes, EvalType etype)
   : etype(etype)
{
   switch (etype)
   {
      case ChangeOfBasis:
      {
         x.SetSize(p + 1);
         w.SetSize(p + 1);
         DenseMatrix A(p + 1);
         for (int i = 0; i <= p; i++)
         {
            CalcBasis(p, nodes[i], A.GetColumn(i));
         }
         Ai.Factor(A);
         // mfem::out << "Poly_1D::Basis(" << p << ",...) : "; Ai.TestInversion();
         break;
      }
      case Barycentric:
      {
         x.SetSize(p + 1);
         w.SetSize(p + 1);
         x = nodes;
         w = 1.0;
         for (int i = 0; i <= p; i++)
         {
            for (int j = 0; j < i; j++)
            {
               double xij = x(i) - x(j);
               w(i) *=  xij;
               w(j) *= -xij;
            }
         }
         for (int i = 0; i <= p; i++)
         {
            w(i) = 1.0/w(i);
         }

#ifdef MFEM_DEBUG
         // Make sure the nodes are increasing
         for (int i = 0; i < p; i++)
         {
            if (x(i) >= x(i+1))
            {
               mfem_error("Poly_1D::Basis::Basis : nodes are not increasing!");
            }
         }
#endif
         break;
      }
      case Positive:
         x.SetDataAndSize(NULL, p + 1); // use x to store (p + 1)
         break;

      default: break;
   }
}

void Poly_1D::Basis::Eval(const double y, Vector &u) const
{
   switch (etype)
   {
      case ChangeOfBasis:
      {
         CalcBasis(Ai.Width() - 1, y, x);
         Ai.Mult(x, u);
         break;
      }
      case Barycentric:
      {
         int i, k, p = x.Size() - 1;
         double l, lk;

         if (p == 0)
         {
            u(0) = 1.0;
            return;
         }

         lk = 1.0;
         for (k = 0; k < p; k++)
         {
            if (y >= (x(k) + x(k+1))/2)
            {
               lk *= y - x(k);
            }
            else
            {
               for (i = k+1; i <= p; i++)
               {
                  lk *= y - x(i);
               }
               break;
            }
         }
         l = lk * (y - x(k));

         for (i = 0; i < k; i++)
         {
            u(i) = l * w(i) / (y - x(i));
         }
         u(k) = lk * w(k);
         for (i++; i <= p; i++)
         {
            u(i) = l * w(i) / (y - x(i));
         }
         break;
      }
      case Positive:
         CalcBernstein(x.Size() - 1, y, u);
         break;

      default: break;
   }
}

void Poly_1D::Basis::Eval(const double y, Vector &u, Vector &d) const
{
   switch (etype)
   {
      case ChangeOfBasis:
      {
         CalcBasis(Ai.Width() - 1, y, x, w);
         Ai.Mult(x, u);
         Ai.Mult(w, d);
         break;
      }
      case Barycentric:
      {
         int i, k, p = x.Size() - 1;
         double l, lp, lk, sk, si;

         if (p == 0)
         {
            u(0) = 1.0;
            d(0) = 0.0;
            return;
         }

         lk = 1.0;
         for (k = 0; k < p; k++)
         {
            if (y >= (x(k) + x(k+1))/2)
            {
               lk *= y - x(k);
            }
            else
            {
               for (i = k+1; i <= p; i++)
               {
                  lk *= y - x(i);
               }
               break;
            }
         }
         l = lk * (y - x(k));

         sk = 0.0;
         for (i = 0; i < k; i++)
         {
            si = 1.0/(y - x(i));
            sk += si;
            u(i) = l * si * w(i);
         }
         u(k) = lk * w(k);
         for (i++; i <= p; i++)
         {
            si = 1.0/(y - x(i));
            sk += si;
            u(i) = l * si * w(i);
         }
         lp = l * sk + lk;

         for (i = 0; i < k; i++)
         {
            d(i) = (lp * w(i) - u(i))/(y - x(i));
         }
         d(k) = sk * u(k);
         for (i++; i <= p; i++)
         {
            d(i) = (lp * w(i) - u(i))/(y - x(i));
         }
         break;
      }
      case Positive:
         CalcBernstein(x.Size() - 1, y, u, d);
         break;

      default: break;
   }
}

const int *Poly_1D::Binom(const int p)
{
   if (binom.NumCols() <= p)
   {
      binom.SetSize(p + 1, p + 1);
      for (int i = 0; i <= p; i++)
      {
         binom(i,0) = binom(i,i) = 1;
         for (int j = 1; j < i; j++)
         {
            binom(i,j) = binom(i-1,j) + binom(i-1,j-1);
         }
      }
   }
   return binom[p];
}

void Poly_1D::ChebyshevPoints(const int p, double *x)
{
   for (int i = 0; i <= p; i++)
   {
      // x[i] = 0.5*(1. + cos(M_PI*(p - i + 0.5)/(p + 1)));
      double s = sin(M_PI_2*(i + 0.5)/(p + 1));
      x[i] = s*s;
   }
}

void Poly_1D::CalcMono(const int p, const double x, double *u)
{
   double xn;
   u[0] = xn = 1.;
   for (int n = 1; n <= p; n++)
   {
      u[n] = (xn *= x);
   }
}

void Poly_1D::CalcMono(const int p, const double x, double *u, double *d)
{
   double xn;
   u[0] = xn = 1.;
   d[0] = 0.;
   for (int n = 1; n <= p; n++)
   {
      d[n] = n * xn;
      u[n] = (xn *= x);
   }
}

void Poly_1D::CalcBinomTerms(const int p, const double x, const double y,
                             double *u)
{
   if (p == 0)
   {
      u[0] = 1.;
   }
   else
   {
      int i;
      const int *b = Binom(p);
      double z = x;

      for (i = 1; i < p; i++)
      {
         u[i] = b[i]*z;
         z *= x;
      }
      u[p] = z;
      z = y;
      for (i--; i > 0; i--)
      {
         u[i] *= z;
         z *= y;
      }
      u[0] = z;
   }
}

void Poly_1D::CalcBinomTerms(const int p, const double x, const double y,
                             double *u, double *d)
{
   if (p == 0)
   {
      u[0] = 1.;
      d[0] = 0.;
   }
   else
   {
      int i;
      const int *b = Binom(p);
      const double xpy = x + y, ptx = p*x;
      double z = 1.;

      for (i = 1; i < p; i++)
      {
         d[i] = b[i]*z*(i*xpy - ptx);
         z *= x;
         u[i] = b[i]*z;
      }
      d[p] = p*z;
      u[p] = z*x;
      z = 1.;
      for (i--; i > 0; i--)
      {
         d[i] *= z;
         z *= y;
         u[i] *= z;
      }
      d[0] = -p*z;
      u[0] = z*y;
   }
}

void Poly_1D::CalcDBinomTerms(const int p, const double x, const double y,
                              double *d)
{
   if (p == 0)
   {
      d[0] = 0.;
   }
   else
   {
      int i;
      const int *b = Binom(p);
      const double xpy = x + y, ptx = p*x;
      double z = 1.;

      for (i = 1; i < p; i++)
      {
         d[i] = b[i]*z*(i*xpy - ptx);
         z *= x;
      }
      d[p] = p*z;
      z = 1.;
      for (i--; i > 0; i--)
      {
         d[i] *= z;
         z *= y;
      }
      d[0] = -p*z;
   }
}

void Poly_1D::CalcLegendre(const int p, const double x, double *u)
{
   // use the recursive definition for [-1,1]:
   // (n+1)*P_{n+1}(z) = (2*n+1)*z*P_n(z)-n*P_{n-1}(z)
   double z;
   u[0] = 1.;
   if (p == 0) { return; }
   u[1] = z = 2.*x - 1.;
   for (int n = 1; n < p; n++)
   {
      u[n+1] = ((2*n + 1)*z*u[n] - n*u[n-1])/(n + 1);
   }
}

void Poly_1D::CalcLegendre(const int p, const double x, double *u, double *d)
{
   // use the recursive definition for [-1,1]:
   // (n+1)*P_{n+1}(z) = (2*n+1)*z*P_n(z)-n*P_{n-1}(z)
   // for the derivative use, z in [-1,1]:
   // P'_{n+1}(z) = (2*n+1)*P_n(z)+P'_{n-1}(z)
   double z;
   u[0] = 1.;
   d[0] = 0.;
   if (p == 0) { return; }
   u[1] = z = 2.*x - 1.;
   d[1] = 2.;
   for (int n = 1; n < p; n++)
   {
      u[n+1] = ((2*n + 1)*z*u[n] - n*u[n-1])/(n + 1);
      d[n+1] = (4*n + 2)*u[n] + d[n-1];
   }
}

void Poly_1D::CalcChebyshev(const int p, const double x, double *u)
{
   // recursive definition, z in [-1,1]
   // T_0(z) = 1,  T_1(z) = z
   // T_{n+1}(z) = 2*z*T_n(z) - T_{n-1}(z)
   double z;
   u[0] = 1.;
   if (p == 0) { return; }
   u[1] = z = 2.*x - 1.;
   for (int n = 1; n < p; n++)
   {
      u[n+1] = 2*z*u[n] - u[n-1];
   }
}

void Poly_1D::CalcChebyshev(const int p, const double x, double *u, double *d)
{
   // recursive definition, z in [-1,1]
   // T_0(z) = 1,  T_1(z) = z
   // T_{n+1}(z) = 2*z*T_n(z) - T_{n-1}(z)
   // T'_n(z) = n*U_{n-1}(z)
   // U_0(z) = 1  U_1(z) = 2*z
   // U_{n+1}(z) = 2*z*U_n(z) - U_{n-1}(z)
   // U_n(z) = z*U_{n-1}(z) + T_n(z) = z*T'_n(z)/n + T_n(z)
   // T'_{n+1}(z) = (n + 1)*(z*T'_n(z)/n + T_n(z))
   double z;
   u[0] = 1.;
   d[0] = 0.;
   if (p == 0) { return; }
   u[1] = z = 2.*x - 1.;
   d[1] = 2.;
   for (int n = 1; n < p; n++)
   {
      u[n+1] = 2*z*u[n] - u[n-1];
      d[n+1] = (n + 1)*(z*d[n]/n + 2*u[n]);
   }
}

void Poly_1D::CalcChebyshev(const int p, const double x, double *u, double *d,
                            double *dd)
{
   // recursive definition, z in [-1,1]
   // T_0(z) = 1,  T_1(z) = z
   // T_{n+1}(z) = 2*z*T_n(z) - T_{n-1}(z)
   // T'_n(z) = n*U_{n-1}(z)
   // U_0(z) = 1  U_1(z) = 2*z
   // U_{n+1}(z) = 2*z*U_n(z) - U_{n-1}(z)
   // U_n(z) = z*U_{n-1}(z) + T_n(z) = z*T'_n(z)/n + T_n(z)
   // T'_{n+1}(z) = (n + 1)*(z*T'_n(z)/n + T_n(z))
   // T''_{n+1}(z) = (n + 1)*(2*(n + 1)*T'_n(z) + z*T''_n(z)) / n
   double z;
   u[0] = 1.;
   d[0] = 0.;
   dd[0]= 0.;
   if (p == 0) { return; }
   u[1] = z = 2.*x - 1.;
   d[1] = 2.;
   dd[1] = 0;
   for (int n = 1; n < p; n++)
   {
      u[n+1] = 2*z*u[n] - u[n-1];
      d[n+1] = (n + 1)*(z*d[n]/n + 2*u[n]);
      dd[n+1] = (n + 1)*(2.*(n + 1)*d[n] + z*dd[n])/n;
   }
}

const double *Poly_1D::GetPoints(const int p, const int btype)
{
   BasisType::Check(btype);
   const int qtype = BasisType::GetQuadrature1D(btype);

   if (qtype == Quadrature1D::Invalid) { return NULL; }

   if (points_container.find(btype) == points_container.end())
   {
      points_container[btype] = new Array<double*>;
   }
   Array<double*> &pts = *points_container[btype];
   if (pts.Size() <= p)
   {
      pts.SetSize(p + 1, NULL);
   }
   if (pts[p] == NULL)
   {
      pts[p] = new double[p + 1];
      quad_func.GivePolyPoints(p+1, pts[p], qtype);
   }
   return pts[p];
}

Poly_1D::Basis &Poly_1D::GetBasis(const int p, const int btype)
{
   BasisType::Check(btype);

   if ( bases_container.find(btype) == bases_container.end() )
   {
      // we haven't been asked for basis or points of this type yet
      bases_container[btype] = new Array<Basis*>;
   }
   Array<Basis*> &bases = *bases_container[btype];
   if (bases.Size() <= p)
   {
      bases.SetSize(p + 1, NULL);
   }
   if (bases[p] == NULL)
   {
      EvalType etype = (btype == BasisType::Positive) ? Positive : Barycentric;
      bases[p] = new Basis(p, GetPoints(p, btype), etype);
   }
   return *bases[p];
}

Poly_1D::~Poly_1D()
{
   for (PointsMap::iterator it = points_container.begin();
        it != points_container.end() ; ++it)
   {
      Array<double*>& pts = *it->second;
      for ( int i = 0 ; i < pts.Size() ; ++i )
      {
         delete [] pts[i];
      }
      delete it->second;
   }

   for (BasisMap::iterator it = bases_container.begin();
        it != bases_container.end() ; ++it)
   {
      Array<Basis*>& bases = *it->second;
      for ( int i = 0 ; i < bases.Size() ; ++i )
      {
         delete bases[i];
      }
      delete it->second;
   }
}

Array2D<int> Poly_1D::binom;
Poly_1D poly1d;


TensorBasisElement::TensorBasisElement(const int dims, const int p,
                                       const int btype, const DofMapType dmtype)
   : b_type(btype),
     basis1d(poly1d.GetBasis(p, b_type))
{
   if (dmtype == H1_DOF_MAP)
   {
      switch (dims)
      {
         case 1:
         {
            dof_map.SetSize(p + 1);
            dof_map[0] = 0;
            dof_map[p] = 1;
            for (int i = 1; i < p; i++)
            {
               dof_map[i] = i+1;
            }
            break;
         }
         case 2:
         {
            const int p1 = p + 1;
            dof_map.SetSize(p1*p1);

            // vertices
            dof_map[0 + 0*p1] = 0;
            dof_map[p + 0*p1] = 1;
            dof_map[p + p*p1] = 2;
            dof_map[0 + p*p1] = 3;

            // edges
            int o = 4;
            for (int i = 1; i < p; i++)
            {
               dof_map[i + 0*p1] = o++;
            }
            for (int i = 1; i < p; i++)
            {
               dof_map[p + i*p1] = o++;
            }
            for (int i = 1; i < p; i++)
            {
               dof_map[(p-i) + p*p1] = o++;
            }
            for (int i = 1; i < p; i++)
            {
               dof_map[0 + (p-i)*p1] = o++;
            }

            // interior
            for (int j = 1; j < p; j++)
            {
               for (int i = 1; i < p; i++)
               {
                  dof_map[i + j*p1] = o++;
               }
            }
            break;
         }
         case 3:
         {
            const int p1 = p + 1;
            dof_map.SetSize(p1*p1*p1);

            // vertices
            dof_map[0 + (0 + 0*p1)*p1] = 0;
            dof_map[p + (0 + 0*p1)*p1] = 1;
            dof_map[p + (p + 0*p1)*p1] = 2;
            dof_map[0 + (p + 0*p1)*p1] = 3;
            dof_map[0 + (0 + p*p1)*p1] = 4;
            dof_map[p + (0 + p*p1)*p1] = 5;
            dof_map[p + (p + p*p1)*p1] = 6;
            dof_map[0 + (p + p*p1)*p1] = 7;

            // edges (see Hexahedron::edges in mesh/hexahedron.cpp)
            int o = 8;
            for (int i = 1; i < p; i++)
            {
               dof_map[i + (0 + 0*p1)*p1] = o++;   // (0,1)
            }
            for (int i = 1; i < p; i++)
            {
               dof_map[p + (i + 0*p1)*p1] = o++;   // (1,2)
            }
            for (int i = 1; i < p; i++)
            {
               dof_map[i + (p + 0*p1)*p1] = o++;   // (3,2)
            }
            for (int i = 1; i < p; i++)
            {
               dof_map[0 + (i + 0*p1)*p1] = o++;   // (0,3)
            }
            for (int i = 1; i < p; i++)
            {
               dof_map[i + (0 + p*p1)*p1] = o++;   // (4,5)
            }
            for (int i = 1; i < p; i++)
            {
               dof_map[p + (i + p*p1)*p1] = o++;   // (5,6)
            }
            for (int i = 1; i < p; i++)
            {
               dof_map[i + (p + p*p1)*p1] = o++;   // (7,6)
            }
            for (int i = 1; i < p; i++)
            {
               dof_map[0 + (i + p*p1)*p1] = o++;   // (4,7)
            }
            for (int i = 1; i < p; i++)
            {
               dof_map[0 + (0 + i*p1)*p1] = o++;   // (0,4)
            }
            for (int i = 1; i < p; i++)
            {
               dof_map[p + (0 + i*p1)*p1] = o++;   // (1,5)
            }
            for (int i = 1; i < p; i++)
            {
               dof_map[p + (p + i*p1)*p1] = o++;   // (2,6)
            }
            for (int i = 1; i < p; i++)
            {
               dof_map[0 + (p + i*p1)*p1] = o++;   // (3,7)
            }

            // faces (see Mesh::GenerateFaces in mesh/mesh.cpp)
            for (int j = 1; j < p; j++)
            {
               for (int i = 1; i < p; i++)
               {
                  dof_map[i + ((p-j) + 0*p1)*p1] = o++;   // (3,2,1,0)
               }
            }
            for (int j = 1; j < p; j++)
            {
               for (int i = 1; i < p; i++)
               {
                  dof_map[i + (0 + j*p1)*p1] = o++;   // (0,1,5,4)
               }
            }
            for (int j = 1; j < p; j++)
            {
               for (int i = 1; i < p; i++)
               {
                  dof_map[p + (i + j*p1)*p1] = o++;   // (1,2,6,5)
               }
            }
            for (int j = 1; j < p; j++)
            {
               for (int i = 1; i < p; i++)
               {
                  dof_map[(p-i) + (p + j*p1)*p1] = o++;   // (2,3,7,6)
               }
            }
            for (int j = 1; j < p; j++)
            {
               for (int i = 1; i < p; i++)
               {
                  dof_map[0 + ((p-i) + j*p1)*p1] = o++;   // (3,0,4,7)
               }
            }
            for (int j = 1; j < p; j++)
            {
               for (int i = 1; i < p; i++)
               {
                  dof_map[i + (j + p*p1)*p1] = o++;   // (4,5,6,7)
               }
            }

            // interior
            for (int k = 1; k < p; k++)
            {
               for (int j = 1; j < p; j++)
               {
                  for (int i = 1; i < p; i++)
                  {
                     dof_map[i + (j + k*p1)*p1] = o++;
                  }
               }
            }
            break;
         }
         default:
            MFEM_ABORT("invalid dimension: " << dims);
            break;
      }
   }
   else if (dmtype == L2_DOF_MAP)
   {
      // leave dof_map empty, indicating that the dofs are ordered
      // lexicographically, i.e. the dof_map is identity
   }
   else
   {
      MFEM_ABORT("invalid DofMapType: " << dmtype);
   }
}


NodalTensorFiniteElement::NodalTensorFiniteElement(const int dims,
                                                   const int p,
                                                   const int btype,
                                                   const DofMapType dmtype)
   : NodalFiniteElement(dims, GetTensorProductGeometry(dims), Pow(p + 1, dims),
                        p, dims > 1 ? FunctionSpace::Qk : FunctionSpace::Pk),
     TensorBasisElement(dims, p, VerifyNodal(btype), dmtype) { }


PositiveTensorFiniteElement::PositiveTensorFiniteElement(
   const int dims, const int p, const DofMapType dmtype)
   : PositiveFiniteElement(dims, GetTensorProductGeometry(dims),
                           Pow(p + 1, dims), p,
                           dims > 1 ? FunctionSpace::Qk : FunctionSpace::Pk),
     TensorBasisElement(dims, p, BasisType::Positive, dmtype) { }


H1_SegmentElement::H1_SegmentElement(const int p, const int btype)
   : NodalTensorFiniteElement(1, p, VerifyClosed(btype), H1_DOF_MAP)
{
   const double *cp = poly1d.ClosedPoints(p, b_type);

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p+1);
   dshape_x.SetSize(p+1);
#endif

   Nodes.IntPoint(0).x = cp[0];
   Nodes.IntPoint(1).x = cp[p];
   for (int i = 1; i < p; i++)
   {
      Nodes.IntPoint(i+1).x = cp[i];
   }
}

void H1_SegmentElement::CalcShape(const IntegrationPoint &ip,
                                  Vector &shape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1);
#endif

   basis1d.Eval(ip.x, shape_x);

   shape(0) = shape_x(0);
   shape(1) = shape_x(p);
   for (int i = 1; i < p; i++)
   {
      shape(i+1) = shape_x(i);
   }
}

void H1_SegmentElement::CalcDShape(const IntegrationPoint &ip,
                                   DenseMatrix &dshape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), dshape_x(p+1);
#endif

   basis1d.Eval(ip.x, shape_x, dshape_x);

   dshape(0,0) = dshape_x(0);
   dshape(1,0) = dshape_x(p);
   for (int i = 1; i < p; i++)
   {
      dshape(i+1,0) = dshape_x(i);
   }
}

void H1_SegmentElement::ProjectDelta(int vertex, Vector &dofs) const
{
   const int p = Order;
   const double *cp = poly1d.ClosedPoints(p, b_type);

   switch (vertex)
   {
      case 0:
         dofs(0) = poly1d.CalcDelta(p, (1.0 - cp[0]));
         dofs(1) = poly1d.CalcDelta(p, (1.0 - cp[p]));
         for (int i = 1; i < p; i++)
         {
            dofs(i+1) = poly1d.CalcDelta(p, (1.0 - cp[i]));
         }
         break;

      case 1:
         dofs(0) = poly1d.CalcDelta(p, cp[0]);
         dofs(1) = poly1d.CalcDelta(p, cp[p]);
         for (int i = 1; i < p; i++)
         {
            dofs(i+1) = poly1d.CalcDelta(p, cp[i]);
         }
         break;
   }
}


H1_QuadrilateralElement::H1_QuadrilateralElement(const int p, const int btype)
   : NodalTensorFiniteElement(2, p, VerifyClosed(btype), H1_DOF_MAP)
{
   const double *cp = poly1d.ClosedPoints(p, b_type);

#ifndef MFEM_THREAD_SAFE
   const int p1 = p + 1;

   shape_x.SetSize(p1);
   shape_y.SetSize(p1);
   dshape_x.SetSize(p1);
   dshape_y.SetSize(p1);
#endif

   int o = 0;
   for (int j = 0; j <= p; j++)
   {
      for (int i = 0; i <= p; i++)
      {
         Nodes.IntPoint(dof_map[o++]).Set2(cp[i], cp[j]);
      }
   }
}

void H1_QuadrilateralElement::CalcShape(const IntegrationPoint &ip,
                                        Vector &shape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1);
#endif

   basis1d.Eval(ip.x, shape_x);
   basis1d.Eval(ip.y, shape_y);

   for (int o = 0, j = 0; j <= p; j++)
      for (int i = 0; i <= p; i++)
      {
         shape(dof_map[o++]) = shape_x(i)*shape_y(j);
      }
}

void H1_QuadrilateralElement::CalcDShape(const IntegrationPoint &ip,
                                         DenseMatrix &dshape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1), dshape_x(p+1), dshape_y(p+1);
#endif

   basis1d.Eval(ip.x, shape_x, dshape_x);
   basis1d.Eval(ip.y, shape_y, dshape_y);

   for (int o = 0, j = 0; j <= p; j++)
   {
      for (int i = 0; i <= p; i++)
      {
         dshape(dof_map[o],0) = dshape_x(i)* shape_y(j);
         dshape(dof_map[o],1) =  shape_x(i)*dshape_y(j);  o++;
      }
   }
}

void H1_QuadrilateralElement::ProjectDelta(int vertex, Vector &dofs) const
{
   const int p = Order;
   const double *cp = poly1d.ClosedPoints(p, b_type);

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1);
#endif

   for (int i = 0; i <= p; i++)
   {
      shape_x(i) = poly1d.CalcDelta(p, (1.0 - cp[i]));
      shape_y(i) = poly1d.CalcDelta(p, cp[i]);
   }

   switch (vertex)
   {
      case 0:
         for (int o = 0, j = 0; j <= p; j++)
            for (int i = 0; i <= p; i++)
            {
               dofs(dof_map[o++]) = shape_x(i)*shape_x(j);
            }
         break;
      case 1:
         for (int o = 0, j = 0; j <= p; j++)
            for (int i = 0; i <= p; i++)
            {
               dofs(dof_map[o++]) = shape_y(i)*shape_x(j);
            }
         break;
      case 2:
         for (int o = 0, j = 0; j <= p; j++)
            for (int i = 0; i <= p; i++)
            {
               dofs(dof_map[o++]) = shape_y(i)*shape_y(j);
            }
         break;
      case 3:
         for (int o = 0, j = 0; j <= p; j++)
            for (int i = 0; i <= p; i++)
            {
               dofs(dof_map[o++]) = shape_x(i)*shape_y(j);
            }
         break;
   }
}


H1_HexahedronElement::H1_HexahedronElement(const int p, const int btype)
   : NodalTensorFiniteElement(3, p, VerifyClosed(btype), H1_DOF_MAP)
{
   const double *cp = poly1d.ClosedPoints(p, b_type);

#ifndef MFEM_THREAD_SAFE
   const int p1 = p + 1;

   shape_x.SetSize(p1);
   shape_y.SetSize(p1);
   shape_z.SetSize(p1);
   dshape_x.SetSize(p1);
   dshape_y.SetSize(p1);
   dshape_z.SetSize(p1);
#endif

   int o = 0;
   for (int k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            Nodes.IntPoint(dof_map[o++]).Set3(cp[i], cp[j], cp[k]);
         }
}

void H1_HexahedronElement::CalcShape(const IntegrationPoint &ip,
                                     Vector &shape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1), shape_z(p+1);
#endif

   basis1d.Eval(ip.x, shape_x);
   basis1d.Eval(ip.y, shape_y);
   basis1d.Eval(ip.z, shape_z);

   for (int o = 0, k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            shape(dof_map[o++]) = shape_x(i)*shape_y(j)*shape_z(k);
         }
}

void H1_HexahedronElement::CalcDShape(const IntegrationPoint &ip,
                                      DenseMatrix &dshape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1),  shape_y(p+1),  shape_z(p+1);
   Vector dshape_x(p+1), dshape_y(p+1), dshape_z(p+1);
#endif

   basis1d.Eval(ip.x, shape_x, dshape_x);
   basis1d.Eval(ip.y, shape_y, dshape_y);
   basis1d.Eval(ip.z, shape_z, dshape_z);

   for (int o = 0, k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            dshape(dof_map[o],0) = dshape_x(i)* shape_y(j)* shape_z(k);
            dshape(dof_map[o],1) =  shape_x(i)*dshape_y(j)* shape_z(k);
            dshape(dof_map[o],2) =  shape_x(i)* shape_y(j)*dshape_z(k);  o++;
         }
}

void H1_HexahedronElement::ProjectDelta(int vertex, Vector &dofs) const
{
   const int p = Order;
   const double *cp = poly1d.ClosedPoints(p,b_type);

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1);
#endif

   for (int i = 0; i <= p; i++)
   {
      shape_x(i) = poly1d.CalcDelta(p, (1.0 - cp[i]));
      shape_y(i) = poly1d.CalcDelta(p, cp[i]);
   }

   switch (vertex)
   {
      case 0:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs(dof_map[o++]) = shape_x(i)*shape_x(j)*shape_x(k);
               }
         break;
      case 1:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs(dof_map[o++]) = shape_y(i)*shape_x(j)*shape_x(k);
               }
         break;
      case 2:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs(dof_map[o++]) = shape_y(i)*shape_y(j)*shape_x(k);
               }
         break;
      case 3:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs(dof_map[o++]) = shape_x(i)*shape_y(j)*shape_x(k);
               }
         break;
      case 4:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs(dof_map[o++]) = shape_x(i)*shape_x(j)*shape_y(k);
               }
         break;
      case 5:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs(dof_map[o++]) = shape_y(i)*shape_x(j)*shape_y(k);
               }
         break;
      case 6:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs(dof_map[o++]) = shape_y(i)*shape_y(j)*shape_y(k);
               }
         break;
      case 7:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs(dof_map[o++]) = shape_x(i)*shape_y(j)*shape_y(k);
               }
         break;
   }
}


H1Pos_SegmentElement::H1Pos_SegmentElement(const int p)
   : PositiveTensorFiniteElement(1, p, H1_DOF_MAP)
{
#ifndef MFEM_THREAD_SAFE
   // thread private versions; see class header.
   shape_x.SetSize(p+1);
   dshape_x.SetSize(p+1);
#endif

   // Endpoints need to be first in the list, so reorder them.
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   for (int i = 1; i < p; i++)
   {
      Nodes.IntPoint(i+1).x = double(i)/p;
   }
}

void H1Pos_SegmentElement::CalcShape(const IntegrationPoint &ip,
                                     Vector &shape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1);
#endif

   Poly_1D::CalcBernstein(p, ip.x, shape_x.GetData() );

   // Endpoints need to be first in the list, so reorder them.
   shape(0) = shape_x(0);
   shape(1) = shape_x(p);
   for (int i = 1; i < p; i++)
   {
      shape(i+1) = shape_x(i);
   }
}

void H1Pos_SegmentElement::CalcDShape(const IntegrationPoint &ip,
                                      DenseMatrix &dshape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), dshape_x(p+1);
#endif

   Poly_1D::CalcBernstein(p, ip.x, shape_x.GetData(), dshape_x.GetData() );

   // Endpoints need to be first in the list, so reorder them.
   dshape(0,0) = dshape_x(0);
   dshape(1,0) = dshape_x(p);
   for (int i = 1; i < p; i++)
   {
      dshape(i+1,0) = dshape_x(i);
   }
}

void H1Pos_SegmentElement::ProjectDelta(int vertex, Vector &dofs) const
{
   dofs = 0.0;
   dofs[vertex] = 1.0;
}


H1Pos_QuadrilateralElement::H1Pos_QuadrilateralElement(const int p)
   : PositiveTensorFiniteElement(2, p, H1_DOF_MAP)
{
#ifndef MFEM_THREAD_SAFE
   const int p1 = p + 1;

   shape_x.SetSize(p1);
   shape_y.SetSize(p1);
   dshape_x.SetSize(p1);
   dshape_y.SetSize(p1);
#endif

   int o = 0;
   for (int j = 0; j <= p; j++)
      for (int i = 0; i <= p; i++)
      {
         Nodes.IntPoint(dof_map[o++]).Set2(double(i)/p, double(j)/p);
      }
}

void H1Pos_QuadrilateralElement::CalcShape(const IntegrationPoint &ip,
                                           Vector &shape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1);
#endif

   Poly_1D::CalcBernstein(p, ip.x, shape_x.GetData() );
   Poly_1D::CalcBernstein(p, ip.y, shape_y.GetData() );

   // Reorder so that vertices are at the beginning of the list
   for (int o = 0, j = 0; j <= p; j++)
      for (int i = 0; i <= p; i++)
      {
         shape(dof_map[o++]) = shape_x(i)*shape_y(j);
      }
}

void H1Pos_QuadrilateralElement::CalcDShape(const IntegrationPoint &ip,
                                            DenseMatrix &dshape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1), dshape_x(p+1), dshape_y(p+1);
#endif

   Poly_1D::CalcBernstein(p, ip.x, shape_x.GetData(), dshape_x.GetData() );
   Poly_1D::CalcBernstein(p, ip.y, shape_y.GetData(), dshape_y.GetData() );

   // Reorder so that vertices are at the beginning of the list
   for (int o = 0, j = 0; j <= p; j++)
      for (int i = 0; i <= p; i++)
      {
         dshape(dof_map[o],0) = dshape_x(i)* shape_y(j);
         dshape(dof_map[o],1) =  shape_x(i)*dshape_y(j);  o++;
      }
}

void H1Pos_QuadrilateralElement::ProjectDelta(int vertex, Vector &dofs) const
{
   dofs = 0.0;
   dofs[vertex] = 1.0;
}


H1Pos_HexahedronElement::H1Pos_HexahedronElement(const int p)
   : PositiveTensorFiniteElement(3, p, H1_DOF_MAP)
{
#ifndef MFEM_THREAD_SAFE
   const int p1 = p + 1;

   shape_x.SetSize(p1);
   shape_y.SetSize(p1);
   shape_z.SetSize(p1);
   dshape_x.SetSize(p1);
   dshape_y.SetSize(p1);
   dshape_z.SetSize(p1);
#endif

   int o = 0;
   for (int k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
            Nodes.IntPoint(dof_map[o++]).Set3(double(i)/p, double(j)/p,
                                              double(k)/p);
}

void H1Pos_HexahedronElement::CalcShape(const IntegrationPoint &ip,
                                        Vector &shape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1), shape_z(p+1);
#endif

   Poly_1D::CalcBernstein(p, ip.x, shape_x.GetData() );
   Poly_1D::CalcBernstein(p, ip.y, shape_y.GetData() );
   Poly_1D::CalcBernstein(p, ip.z, shape_z.GetData() );

   for (int o = 0, k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            shape(dof_map[o++]) = shape_x(i)*shape_y(j)*shape_z(k);
         }
}

void H1Pos_HexahedronElement::CalcDShape(const IntegrationPoint &ip,
                                         DenseMatrix &dshape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1),  shape_y(p+1),  shape_z(p+1);
   Vector dshape_x(p+1), dshape_y(p+1), dshape_z(p+1);
#endif

   Poly_1D::CalcBernstein(p, ip.x, shape_x.GetData(), dshape_x.GetData() );
   Poly_1D::CalcBernstein(p, ip.y, shape_y.GetData(), dshape_y.GetData() );
   Poly_1D::CalcBernstein(p, ip.z, shape_z.GetData(), dshape_z.GetData() );

   for (int o = 0, k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            dshape(dof_map[o],0) = dshape_x(i)* shape_y(j)* shape_z(k);
            dshape(dof_map[o],1) =  shape_x(i)*dshape_y(j)* shape_z(k);
            dshape(dof_map[o],2) =  shape_x(i)* shape_y(j)*dshape_z(k);  o++;
         }
}

void H1Pos_HexahedronElement::ProjectDelta(int vertex, Vector &dofs) const
{
   dofs = 0.0;
   dofs[vertex] = 1.0;
}


H1_TriangleElement::H1_TriangleElement(const int p, const int btype)
   : NodalFiniteElement(2, Geometry::TRIANGLE, ((p + 1)*(p + 2))/2, p,
                        FunctionSpace::Pk)
{
   const double *cp = poly1d.ClosedPoints(p, VerifyNodal(VerifyClosed(btype)));

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p + 1);
   shape_y.SetSize(p + 1);
   shape_l.SetSize(p + 1);
   dshape_x.SetSize(p + 1);
   dshape_y.SetSize(p + 1);
   dshape_l.SetSize(p + 1);
   ddshape_x.SetSize(p + 1);
   ddshape_y.SetSize(p + 1);
   ddshape_l.SetSize(p + 1);
   u.SetSize(Dof);
   du.SetSize(Dof, Dim);
   ddu.SetSize(Dof, (Dim * (Dim + 1)) / 2 );
#else
   Vector shape_x(p + 1), shape_y(p + 1), shape_l(p + 1);
#endif

   // vertices
   Nodes.IntPoint(0).Set2(cp[0], cp[0]);
   Nodes.IntPoint(1).Set2(cp[p], cp[0]);
   Nodes.IntPoint(2).Set2(cp[0], cp[p]);

   // edges
   int o = 3;
   for (int i = 1; i < p; i++)
   {
      Nodes.IntPoint(o++).Set2(cp[i], cp[0]);
   }
   for (int i = 1; i < p; i++)
   {
      Nodes.IntPoint(o++).Set2(cp[p-i], cp[i]);
   }
   for (int i = 1; i < p; i++)
   {
      Nodes.IntPoint(o++).Set2(cp[0], cp[p-i]);
   }

   // interior
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)
      {
         const double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set2(cp[i]/w, cp[j]/w);
      }

   DenseMatrix T(Dof);
   for (int k = 0; k < Dof; k++)
   {
      IntegrationPoint &ip = Nodes.IntPoint(k);
      poly1d.CalcBasis(p, ip.x, shape_x);
      poly1d.CalcBasis(p, ip.y, shape_y);
      poly1d.CalcBasis(p, 1. - ip.x - ip.y, shape_l);

      o = 0;
      for (int j = 0; j <= p; j++)
         for (int i = 0; i + j <= p; i++)
         {
            T(o++, k) = shape_x(i)*shape_y(j)*shape_l(p-i-j);
         }
   }

   Ti.Factor(T);
   // mfem::out << "H1_TriangleElement(" << p << ") : "; Ti.TestInversion();
}

void H1_TriangleElement::CalcShape(const IntegrationPoint &ip,
                                   Vector &shape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p + 1), shape_y(p + 1), shape_l(p + 1), u(Dof);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x);
   poly1d.CalcBasis(p, ip.y, shape_y);
   poly1d.CalcBasis(p, 1. - ip.x - ip.y, shape_l);

   for (int o = 0, j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)
      {
         u(o++) = shape_x(i)*shape_y(j)*shape_l(p-i-j);
      }

   Ti.Mult(u, shape);
}

void H1_TriangleElement::CalcDShape(const IntegrationPoint &ip,
                                    DenseMatrix &dshape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector  shape_x(p + 1),  shape_y(p + 1),  shape_l(p + 1);
   Vector dshape_x(p + 1), dshape_y(p + 1), dshape_l(p + 1);
   DenseMatrix du(Dof, Dim);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x, dshape_x);
   poly1d.CalcBasis(p, ip.y, shape_y, dshape_y);
   poly1d.CalcBasis(p, 1. - ip.x - ip.y, shape_l, dshape_l);

   for (int o = 0, j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)
      {
         int k = p - i - j;
         du(o,0) = ((dshape_x(i)* shape_l(k)) -
                    ( shape_x(i)*dshape_l(k)))*shape_y(j);
         du(o,1) = ((dshape_y(j)* shape_l(k)) -
                    ( shape_y(j)*dshape_l(k)))*shape_x(i);
         o++;
      }

   Ti.Mult(du, dshape);
}

void H1_TriangleElement::CalcHessian(const IntegrationPoint &ip,
                                     DenseMatrix &ddshape) const
{
   const int p = Order;
#ifdef MFEM_THREAD_SAFE
   Vector   shape_x(p + 1),   shape_y(p + 1),   shape_l(p + 1);
   Vector  dshape_x(p + 1),  dshape_y(p + 1),  dshape_l(p + 1);
   Vector ddshape_x(p + 1), ddshape_y(p + 1), ddshape_l(p + 1);
   DenseMatrix ddu(Dof, Dim);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x, dshape_x, ddshape_x);
   poly1d.CalcBasis(p, ip.y, shape_y, dshape_y, ddshape_y);
   poly1d.CalcBasis(p, 1. - ip.x - ip.y, shape_l, dshape_l, ddshape_l);

   for (int o = 0, j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)
      {
         int k = p - i - j;
         // u_xx, u_xy, u_yy
         ddu(o,0) = ((ddshape_x(i) * shape_l(k)) - 2. * (dshape_x(i) * dshape_l(k)) +
                     (shape_x(i) * ddshape_l(k))) * shape_y(j);
         ddu(o,1) = (((shape_x(i) * ddshape_l(k)) - dshape_x(i) * dshape_l(k)) * shape_y(
                        j)) + (((dshape_x(i) * shape_l(k)) - (shape_x(i) * dshape_l(k))) * dshape_y(j));
         ddu(o,2) = ((ddshape_y(j) * shape_l(k)) - 2. * (dshape_y(j) * dshape_l(k)) +
                     (shape_y(j) * ddshape_l(k))) * shape_x(i);
         o++;
      }

   Ti.Mult(ddu, ddshape);
}


H1_TetrahedronElement::H1_TetrahedronElement(const int p, const int btype)
   : NodalFiniteElement(3, Geometry::TETRAHEDRON, ((p + 1)*(p + 2)*(p + 3))/6,
                        p, FunctionSpace::Pk)
{
   const double *cp = poly1d.ClosedPoints(p, VerifyNodal(VerifyClosed(btype)));

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p + 1);
   shape_y.SetSize(p + 1);
   shape_z.SetSize(p + 1);
   shape_l.SetSize(p + 1);
   dshape_x.SetSize(p + 1);
   dshape_y.SetSize(p + 1);
   dshape_z.SetSize(p + 1);
   dshape_l.SetSize(p + 1);
   ddshape_x.SetSize(p + 1);
   ddshape_y.SetSize(p + 1);
   ddshape_z.SetSize(p + 1);
   ddshape_l.SetSize(p + 1);
   u.SetSize(Dof);
   du.SetSize(Dof, Dim);
   ddu.SetSize(Dof, (Dim * (Dim + 1)) / 2);
#else
   Vector shape_x(p + 1), shape_y(p + 1), shape_z(p + 1), shape_l(p + 1);
#endif

   // vertices
   Nodes.IntPoint(0).Set3(cp[0], cp[0], cp[0]);
   Nodes.IntPoint(1).Set3(cp[p], cp[0], cp[0]);
   Nodes.IntPoint(2).Set3(cp[0], cp[p], cp[0]);
   Nodes.IntPoint(3).Set3(cp[0], cp[0], cp[p]);

   // edges (see Tetrahedron::edges in mesh/tetrahedron.cpp)
   int o = 4;
   for (int i = 1; i < p; i++)  // (0,1)
   {
      Nodes.IntPoint(o++).Set3(cp[i], cp[0], cp[0]);
   }
   for (int i = 1; i < p; i++)  // (0,2)
   {
      Nodes.IntPoint(o++).Set3(cp[0], cp[i], cp[0]);
   }
   for (int i = 1; i < p; i++)  // (0,3)
   {
      Nodes.IntPoint(o++).Set3(cp[0], cp[0], cp[i]);
   }
   for (int i = 1; i < p; i++)  // (1,2)
   {
      Nodes.IntPoint(o++).Set3(cp[p-i], cp[i], cp[0]);
   }
   for (int i = 1; i < p; i++)  // (1,3)
   {
      Nodes.IntPoint(o++).Set3(cp[p-i], cp[0], cp[i]);
   }
   for (int i = 1; i < p; i++)  // (2,3)
   {
      Nodes.IntPoint(o++).Set3(cp[0], cp[p-i], cp[i]);
   }

   // faces (see Mesh::GenerateFaces in mesh/mesh.cpp)
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (1,2,3)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(cp[p-i-j]/w, cp[i]/w, cp[j]/w);
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (0,3,2)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(cp[0], cp[j]/w, cp[i]/w);
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (0,1,3)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(cp[i]/w, cp[0], cp[j]/w);
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (0,2,1)
      {
         double w = cp[i] + cp[j] + cp[p-i-j];
         Nodes.IntPoint(o++).Set3(cp[j]/w, cp[i]/w, cp[0]);
      }

   // interior
   for (int k = 1; k < p; k++)
      for (int j = 1; j + k < p; j++)
         for (int i = 1; i + j + k < p; i++)
         {
            double w = cp[i] + cp[j] + cp[k] + cp[p-i-j-k];
            Nodes.IntPoint(o++).Set3(cp[i]/w, cp[j]/w, cp[k]/w);
         }

   DenseMatrix T(Dof);
   for (int m = 0; m < Dof; m++)
   {
      IntegrationPoint &ip = Nodes.IntPoint(m);
      poly1d.CalcBasis(p, ip.x, shape_x);
      poly1d.CalcBasis(p, ip.y, shape_y);
      poly1d.CalcBasis(p, ip.z, shape_z);
      poly1d.CalcBasis(p, 1. - ip.x - ip.y - ip.z, shape_l);

      o = 0;
      for (int k = 0; k <= p; k++)
         for (int j = 0; j + k <= p; j++)
            for (int i = 0; i + j + k <= p; i++)
            {
               T(o++, m) = shape_x(i)*shape_y(j)*shape_z(k)*shape_l(p-i-j-k);
            }
   }

   Ti.Factor(T);
   // mfem::out << "H1_TetrahedronElement(" << p << ") : "; Ti.TestInversion();
}

void H1_TetrahedronElement::CalcShape(const IntegrationPoint &ip,
                                      Vector &shape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p + 1), shape_y(p + 1), shape_z(p + 1), shape_l(p + 1);
   Vector u(Dof);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x);
   poly1d.CalcBasis(p, ip.y, shape_y);
   poly1d.CalcBasis(p, ip.z, shape_z);
   poly1d.CalcBasis(p, 1. - ip.x - ip.y - ip.z, shape_l);

   for (int o = 0, k = 0; k <= p; k++)
      for (int j = 0; j + k <= p; j++)
         for (int i = 0; i + j + k <= p; i++)
         {
            u(o++) = shape_x(i)*shape_y(j)*shape_z(k)*shape_l(p-i-j-k);
         }

   Ti.Mult(u, shape);
}

void H1_TetrahedronElement::CalcDShape(const IntegrationPoint &ip,
                                       DenseMatrix &dshape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector  shape_x(p + 1),  shape_y(p + 1),  shape_z(p + 1),  shape_l(p + 1);
   Vector dshape_x(p + 1), dshape_y(p + 1), dshape_z(p + 1), dshape_l(p + 1);
   DenseMatrix du(Dof, Dim);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x, dshape_x);
   poly1d.CalcBasis(p, ip.y, shape_y, dshape_y);
   poly1d.CalcBasis(p, ip.z, shape_z, dshape_z);
   poly1d.CalcBasis(p, 1. - ip.x - ip.y - ip.z, shape_l, dshape_l);

   for (int o = 0, k = 0; k <= p; k++)
      for (int j = 0; j + k <= p; j++)
         for (int i = 0; i + j + k <= p; i++)
         {
            int l = p - i - j - k;
            du(o,0) = ((dshape_x(i)* shape_l(l)) -
                       ( shape_x(i)*dshape_l(l)))*shape_y(j)*shape_z(k);
            du(o,1) = ((dshape_y(j)* shape_l(l)) -
                       ( shape_y(j)*dshape_l(l)))*shape_x(i)*shape_z(k);
            du(o,2) = ((dshape_z(k)* shape_l(l)) -
                       ( shape_z(k)*dshape_l(l)))*shape_x(i)*shape_y(j);
            o++;
         }

   Ti.Mult(du, dshape);
}

void H1_TetrahedronElement::CalcHessian(const IntegrationPoint &ip,
                                        DenseMatrix &ddshape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector   shape_x(p + 1),   shape_y(p + 1),   shape_z(p + 1),   shape_l(p + 1);
   Vector  dshape_x(p + 1),  dshape_y(p + 1),  dshape_z(p + 1),  dshape_l(p + 1);
   Vector ddshape_x(p + 1), ddshape_y(p + 1), ddshape_z(p + 1), ddshape_l(p + 1);
   DenseMatrix ddu(Dof, ((Dim + 1) * Dim) / 2);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x, dshape_x, ddshape_x);
   poly1d.CalcBasis(p, ip.y, shape_y, dshape_y, ddshape_y);
   poly1d.CalcBasis(p, ip.z, shape_z, dshape_z, ddshape_z);
   poly1d.CalcBasis(p, 1. - ip.x - ip.y - ip.z, shape_l, dshape_l, ddshape_l);

   for (int o = 0, k = 0; k <= p; k++)
      for (int j = 0; j + k <= p; j++)
         for (int i = 0; i + j + k <= p; i++)
         {
            // u_xx, u_xy, u_xz, u_yy, u_yz, u_zz
            int l = p - i - j - k;
            ddu(o,0) = ((ddshape_x(i) * shape_l(l)) - 2. * (dshape_x(i) * dshape_l(l)) +
                        (shape_x(i) * ddshape_l(l))) * shape_y(j) * shape_z(k);
            ddu(o,1) = ((dshape_y(j) * ((dshape_x(i) * shape_l(l)) -
                                        (shape_x(i) * dshape_l(l)))) +
                        (shape_y(j) * ((ddshape_l(l) * shape_x(i)) -
                                       (dshape_x(i) * dshape_l(l)))))* shape_z(k);
            ddu(o,2) = ((dshape_z(k) * ((dshape_x(i) * shape_l(l)) -
                                        (shape_x(i) * dshape_l(l)))) +
                        (shape_z(k) * ((ddshape_l(l) * shape_x(i)) -
                                       (dshape_x(i) * dshape_l(l)))))* shape_y(j);
            ddu(o,3) = ((ddshape_y(j) * shape_l(l)) - 2. * (dshape_y(j) * dshape_l(l)) +
                        (shape_y(j) * ddshape_l(l))) * shape_x(i) * shape_z(k);
            ddu(o,4) = ((dshape_z(k) * ((dshape_y(j) * shape_l(l)) -
                                        (shape_y(j)*dshape_l(l))) ) +
                        (shape_z(k)* ((ddshape_l(l)*shape_y(j)) -
                                      (dshape_y(j) * dshape_l(l)) ) ) )* shape_x(i);
            ddu(o,5) = ((ddshape_z(k) * shape_l(l)) - 2. * (dshape_z(k) * dshape_l(l)) +
                        (shape_z(k) * ddshape_l(l))) * shape_y(j) * shape_x(i);
            o++;
         }
   Ti.Mult(ddu, ddshape);
}

H1Pos_TriangleElement::H1Pos_TriangleElement(const int p)
   : PositiveFiniteElement(2, Geometry::TRIANGLE, ((p + 1)*(p + 2))/2, p,
                           FunctionSpace::Pk)
{
#ifndef MFEM_THREAD_SAFE
   m_shape.SetSize(Dof);
   dshape_1d.SetSize(p + 1);
   m_dshape.SetSize(Dof, Dim);
#endif
   dof_map.SetSize(Dof);

   struct Index
   {
      int p2p3;
      Index(int p) { p2p3 = 2*p + 3; }
      int operator()(int i, int j) { return ((p2p3-j)*j)/2+i; }
   };
   Index idx(p);

   // vertices
   dof_map[idx(0,0)] = 0;
   Nodes.IntPoint(0).Set2(0., 0.);
   dof_map[idx(p,0)] = 1;
   Nodes.IntPoint(1).Set2(1., 0.);
   dof_map[idx(0,p)] = 2;
   Nodes.IntPoint(2).Set2(0., 1.);

   // edges
   int o = 3;
   for (int i = 1; i < p; i++)
   {
      dof_map[idx(i,0)] = o;
      Nodes.IntPoint(o++).Set2(double(i)/p, 0.);
   }
   for (int i = 1; i < p; i++)
   {
      dof_map[idx(p-i,i)] = o;
      Nodes.IntPoint(o++).Set2(double(p-i)/p, double(i)/p);
   }
   for (int i = 1; i < p; i++)
   {
      dof_map[idx(0,p-i)] = o;
      Nodes.IntPoint(o++).Set2(0., double(p-i)/p);
   }

   // interior
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)
      {
         dof_map[idx(i,j)] = o;
         Nodes.IntPoint(o++).Set2(double(i)/p, double(j)/p);
      }
}

// static method
void H1Pos_TriangleElement::CalcShape(
   const int p, const double l1, const double l2, double *shape)
{
   const double l3 = 1. - l1 - l2;

   // The (i,j) basis function is given by: T(i,j,p-i-j) l1^i l2^j l3^{p-i-j},
   // where T(i,j,k) = (i+j+k)! / (i! j! k!)
   // Another expression is given by the terms of the expansion:
   //    (l1 + l2 + l3)^p =
   //       \sum_{j=0}^p \binom{p}{j} l2^j
   //          \sum_{i=0}^{p-j} \binom{p-j}{i} l1^i l3^{p-j-i}
   const int *bp = Poly_1D::Binom(p);
   double z = 1.;
   for (int o = 0, j = 0; j <= p; j++)
   {
      Poly_1D::CalcBinomTerms(p - j, l1, l3, &shape[o]);
      double s = bp[j]*z;
      for (int i = 0; i <= p - j; i++)
      {
         shape[o++] *= s;
      }
      z *= l2;
   }
}

// static method
void H1Pos_TriangleElement::CalcDShape(
   const int p, const double l1, const double l2,
   double *dshape_1d, double *dshape)
{
   const int dof = ((p + 1)*(p + 2))/2;
   const double l3 = 1. - l1 - l2;

   const int *bp = Poly_1D::Binom(p);
   double z = 1.;
   for (int o = 0, j = 0; j <= p; j++)
   {
      Poly_1D::CalcDBinomTerms(p - j, l1, l3, dshape_1d);
      double s = bp[j]*z;
      for (int i = 0; i <= p - j; i++)
      {
         dshape[o++] = s*dshape_1d[i];
      }
      z *= l2;
   }
   z = 1.;
   for (int i = 0; i <= p; i++)
   {
      Poly_1D::CalcDBinomTerms(p - i, l2, l3, dshape_1d);
      double s = bp[i]*z;
      for (int o = i, j = 0; j <= p - i; j++)
      {
         dshape[dof + o] = s*dshape_1d[j];
         o += p + 1 - j;
      }
      z *= l1;
   }
}

void H1Pos_TriangleElement::CalcShape(const IntegrationPoint &ip,
                                      Vector &shape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector m_shape(Dof);
#endif
   CalcShape(Order, ip.x, ip.y, m_shape.GetData());
   for (int i = 0; i < Dof; i++)
   {
      shape(dof_map[i]) = m_shape(i);
   }
}

void H1Pos_TriangleElement::CalcDShape(const IntegrationPoint &ip,
                                       DenseMatrix &dshape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector dshape_1d(Order + 1);
   DenseMatrix m_dshape(Dof, Dim);
#endif
   CalcDShape(Order, ip.x, ip.y, dshape_1d.GetData(), m_dshape.Data());
   for (int d = 0; d < 2; d++)
   {
      for (int i = 0; i < Dof; i++)
      {
         dshape(dof_map[i],d) = m_dshape(i,d);
      }
   }
}


H1Pos_TetrahedronElement::H1Pos_TetrahedronElement(const int p)
   : PositiveFiniteElement(3, Geometry::TETRAHEDRON,
                           ((p + 1)*(p + 2)*(p + 3))/6, p, FunctionSpace::Pk)
{
#ifndef MFEM_THREAD_SAFE
   m_shape.SetSize(Dof);
   dshape_1d.SetSize(p + 1);
   m_dshape.SetSize(Dof, Dim);
#endif
   dof_map.SetSize(Dof);

   struct Index
   {
      int p, dof;
      int tri(int k) { return (k*(k + 1))/2; }
      int tet(int k) { return (k*(k + 1)*(k + 2))/6; }
      Index(int p_) { p = p_; dof = tet(p + 1); }
      int operator()(int i, int j, int k)
      { return dof - tet(p - k) - tri(p + 1 - k - j) + i; }
   };
   Index idx(p);

   // vertices
   dof_map[idx(0,0,0)] = 0;
   Nodes.IntPoint(0).Set3(0., 0., 0.);
   dof_map[idx(p,0,0)] = 1;
   Nodes.IntPoint(1).Set3(1., 0., 0.);
   dof_map[idx(0,p,0)] = 2;
   Nodes.IntPoint(2).Set3(0., 1., 0.);
   dof_map[idx(0,0,p)] = 3;
   Nodes.IntPoint(3).Set3(0., 0., 1.);

   // edges (see Tetrahedron::edges in mesh/tetrahedron.cpp)
   int o = 4;
   for (int i = 1; i < p; i++)  // (0,1)
   {
      dof_map[idx(i,0,0)] = o;
      Nodes.IntPoint(o++).Set3(double(i)/p, 0., 0.);
   }
   for (int i = 1; i < p; i++)  // (0,2)
   {
      dof_map[idx(0,i,0)] = o;
      Nodes.IntPoint(o++).Set3(0., double(i)/p, 0.);
   }
   for (int i = 1; i < p; i++)  // (0,3)
   {
      dof_map[idx(0,0,i)] = o;
      Nodes.IntPoint(o++).Set3(0., 0., double(i)/p);
   }
   for (int i = 1; i < p; i++)  // (1,2)
   {
      dof_map[idx(p-i,i,0)] = o;
      Nodes.IntPoint(o++).Set3(double(p-i)/p, double(i)/p, 0.);
   }
   for (int i = 1; i < p; i++)  // (1,3)
   {
      dof_map[idx(p-i,0,i)] = o;
      Nodes.IntPoint(o++).Set3(double(p-i)/p, 0., double(i)/p);
   }
   for (int i = 1; i < p; i++)  // (2,3)
   {
      dof_map[idx(0,p-i,i)] = o;
      Nodes.IntPoint(o++).Set3(0., double(p-i)/p, double(i)/p);
   }

   // faces (see Mesh::GenerateFaces in mesh/mesh.cpp)
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (1,2,3)
      {
         dof_map[idx(p-i-j,i,j)] = o;
         Nodes.IntPoint(o++).Set3(double(p-i-j)/p, double(i)/p, double(j)/p);
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (0,3,2)
      {
         dof_map[idx(0,j,i)] = o;
         Nodes.IntPoint(o++).Set3(0., double(j)/p, double(i)/p);
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (0,1,3)
      {
         dof_map[idx(i,0,j)] = o;
         Nodes.IntPoint(o++).Set3(double(i)/p, 0., double(j)/p);
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (0,2,1)
      {
         dof_map[idx(j,i,0)] = o;
         Nodes.IntPoint(o++).Set3(double(j)/p, double(i)/p, 0.);
      }

   // interior
   for (int k = 1; k < p; k++)
      for (int j = 1; j + k < p; j++)
         for (int i = 1; i + j + k < p; i++)
         {
            dof_map[idx(i,j,k)] = o;
            Nodes.IntPoint(o++).Set3(double(i)/p, double(j)/p, double(k)/p);
         }
}

// static method
void H1Pos_TetrahedronElement::CalcShape(
   const int p, const double l1, const double l2, const double l3,
   double *shape)
{
   const double l4 = 1. - l1 - l2 - l3;

   // The basis functions are the terms in the expansion:
   //   (l1 + l2 + l3 + l4)^p =
   //      \sum_{k=0}^p \binom{p}{k} l3^k
   //         \sum_{j=0}^{p-k} \binom{p-k}{j} l2^j
   //            \sum_{i=0}^{p-k-j} \binom{p-k-j}{i} l1^i l4^{p-k-j-i}
   const int *bp = Poly_1D::Binom(p);
   double l3k = 1.;
   for (int o = 0, k = 0; k <= p; k++)
   {
      const int *bpk = Poly_1D::Binom(p - k);
      const double ek = bp[k]*l3k;
      double l2j = 1.;
      for (int j = 0; j <= p - k; j++)
      {
         Poly_1D::CalcBinomTerms(p - k - j, l1, l4, &shape[o]);
         double ekj = ek*bpk[j]*l2j;
         for (int i = 0; i <= p - k - j; i++)
         {
            shape[o++] *= ekj;
         }
         l2j *= l2;
      }
      l3k *= l3;
   }
}

// static method
void H1Pos_TetrahedronElement::CalcDShape(
   const int p, const double l1, const double l2, const double l3,
   double *dshape_1d, double *dshape)
{
   const int dof = ((p + 1)*(p + 2)*(p + 3))/6;
   const double l4 = 1. - l1 - l2 - l3;

   // For the x derivatives, differentiate the terms of the expression:
   //   \sum_{k=0}^p \binom{p}{k} l3^k
   //      \sum_{j=0}^{p-k} \binom{p-k}{j} l2^j
   //         \sum_{i=0}^{p-k-j} \binom{p-k-j}{i} l1^i l4^{p-k-j-i}
   const int *bp = Poly_1D::Binom(p);
   double l3k = 1.;
   for (int o = 0, k = 0; k <= p; k++)
   {
      const int *bpk = Poly_1D::Binom(p - k);
      const double ek = bp[k]*l3k;
      double l2j = 1.;
      for (int j = 0; j <= p - k; j++)
      {
         Poly_1D::CalcDBinomTerms(p - k - j, l1, l4, dshape_1d);
         double ekj = ek*bpk[j]*l2j;
         for (int i = 0; i <= p - k - j; i++)
         {
            dshape[o++] = dshape_1d[i]*ekj;
         }
         l2j *= l2;
      }
      l3k *= l3;
   }
   // For the y derivatives, differentiate the terms of the expression:
   //   \sum_{k=0}^p \binom{p}{k} l3^k
   //      \sum_{i=0}^{p-k} \binom{p-k}{i} l1^i
   //         \sum_{j=0}^{p-k-i} \binom{p-k-i}{j} l2^j l4^{p-k-j-i}
   l3k = 1.;
   for (int ok = 0, k = 0; k <= p; k++)
   {
      const int *bpk = Poly_1D::Binom(p - k);
      const double ek = bp[k]*l3k;
      double l1i = 1.;
      for (int i = 0; i <= p - k; i++)
      {
         Poly_1D::CalcDBinomTerms(p - k - i, l2, l4, dshape_1d);
         double eki = ek*bpk[i]*l1i;
         int o = ok + i;
         for (int j = 0; j <= p - k - i; j++)
         {
            dshape[dof + o] = dshape_1d[j]*eki;
            o += p - k - j + 1;
         }
         l1i *= l1;
      }
      l3k *= l3;
      ok += ((p - k + 2)*(p - k + 1))/2;
   }
   // For the z derivatives, differentiate the terms of the expression:
   //   \sum_{j=0}^p \binom{p}{j} l2^j
   //      \sum_{i=0}^{p-j} \binom{p-j}{i} l1^i
   //         \sum_{k=0}^{p-j-i} \binom{p-j-i}{k} l3^k l4^{p-k-j-i}
   double l2j = 1.;
   for (int j = 0; j <= p; j++)
   {
      const int *bpj = Poly_1D::Binom(p - j);
      const double ej = bp[j]*l2j;
      double l1i = 1.;
      for (int i = 0; i <= p - j; i++)
      {
         Poly_1D::CalcDBinomTerms(p - j - i, l3, l4, dshape_1d);
         double eji = ej*bpj[i]*l1i;
         int m = ((p + 2)*(p + 1))/2;
         int n = ((p - j + 2)*(p - j + 1))/2;
         for (int o = i, k = 0; k <= p - j - i; k++)
         {
            // m = ((p - k + 2)*(p - k + 1))/2;
            // n = ((p - k - j + 2)*(p - k - j + 1))/2;
            o += m;
            dshape[2*dof + o - n] = dshape_1d[k]*eji;
            m -= p - k + 1;
            n -= p - k - j + 1;
         }
         l1i *= l1;
      }
      l2j *= l2;
   }
}

void H1Pos_TetrahedronElement::CalcShape(const IntegrationPoint &ip,
                                         Vector &shape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector m_shape(Dof);
#endif
   CalcShape(Order, ip.x, ip.y, ip.z, m_shape.GetData());
   for (int i = 0; i < Dof; i++)
   {
      shape(dof_map[i]) = m_shape(i);
   }
}

void H1Pos_TetrahedronElement::CalcDShape(const IntegrationPoint &ip,
                                          DenseMatrix &dshape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector dshape_1d(Order + 1);
   DenseMatrix m_dshape(Dof, Dim);
#endif
   CalcDShape(Order, ip.x, ip.y, ip.z, dshape_1d.GetData(), m_dshape.Data());
   for (int d = 0; d < 3; d++)
   {
      for (int i = 0; i < Dof; i++)
      {
         dshape(dof_map[i],d) = m_dshape(i,d);
      }
   }
}


H1_WedgeElement::H1_WedgeElement(const int p,
                                 const int btype)
   : NodalFiniteElement(3, Geometry::PRISM, ((p + 1)*(p + 1)*(p + 2))/2,
                        p, FunctionSpace::Qk),
     TriangleFE(p, btype),
     SegmentFE(p, btype)
{
#ifndef MFEM_THREAD_SAFE
   t_shape.SetSize(TriangleFE.GetDof());
   s_shape.SetSize(SegmentFE.GetDof());
   t_dshape.SetSize(TriangleFE.GetDof(), 2);
   s_dshape.SetSize(SegmentFE.GetDof(), 1);
#endif

   t_dof.SetSize(Dof);
   s_dof.SetSize(Dof);

   // Nodal DoFs
   t_dof[0] = 0; s_dof[0] = 0;
   t_dof[1] = 1; s_dof[1] = 0;
   t_dof[2] = 2; s_dof[2] = 0;
   t_dof[3] = 0; s_dof[3] = 1;
   t_dof[4] = 1; s_dof[4] = 1;
   t_dof[5] = 2; s_dof[5] = 1;

   // Edge DoFs
   int ne = p-1;
   for (int i=1; i<p; i++)
   {
      t_dof[5 + 0 * ne + i] = 2 + 0 * ne + i; s_dof[5 + 0 * ne + i] = 0;
      t_dof[5 + 1 * ne + i] = 2 + 1 * ne + i; s_dof[5 + 1 * ne + i] = 0;
      t_dof[5 + 2 * ne + i] = 2 + 2 * ne + i; s_dof[5 + 2 * ne + i] = 0;
      t_dof[5 + 3 * ne + i] = 2 + 0 * ne + i; s_dof[5 + 3 * ne + i] = 1;
      t_dof[5 + 4 * ne + i] = 2 + 1 * ne + i; s_dof[5 + 4 * ne + i] = 1;
      t_dof[5 + 5 * ne + i] = 2 + 2 * ne + i; s_dof[5 + 5 * ne + i] = 1;
      t_dof[5 + 6 * ne + i] = 0;              s_dof[5 + 6 * ne + i] = i + 1;
      t_dof[5 + 7 * ne + i] = 1;              s_dof[5 + 7 * ne + i] = i + 1;
      t_dof[5 + 8 * ne + i] = 2;              s_dof[5 + 8 * ne + i] = i + 1;
   }

   // Triangular Face DoFs
   int k=0;
   int nt = (p-1)*(p-2)/2;
   for (int j=1; j<p; j++)
   {
      for (int i=1; i<p-j; i++)
      {
         int l = j - p + (((2 * p - 1) - i) * i) / 2;
         t_dof[6 + 9 * ne + k]      = 3 * p + l; s_dof[6 + 9 * ne + k]      = 0;
         t_dof[6 + 9 * ne + nt + k] = 3 * p + k; s_dof[6 + 9 * ne + nt + k] = 1;
         k++;
      }
   }

   // Quadrilateral Face DoFs
   k=0;
   int nq = (p-1)*(p-1);
   for (int j=1; j<p; j++)
   {
      for (int i=1; i<p; i++)
      {
         t_dof[6 + 9 * ne + 2 * nt + 0 * nq + k] = 2 + 0 * ne + i;
         t_dof[6 + 9 * ne + 2 * nt + 1 * nq + k] = 2 + 1 * ne + i;
         t_dof[6 + 9 * ne + 2 * nt + 2 * nq + k] = 2 + 2 * ne + i;

         s_dof[6 + 9 * ne + 2 * nt + 0 * nq + k] = 1 + j;
         s_dof[6 + 9 * ne + 2 * nt + 1 * nq + k] = 1 + j;
         s_dof[6 + 9 * ne + 2 * nt + 2 * nq + k] = 1 + j;

         k++;
      }
   }

   // Interior DoFs
   int m=0;
   for (int k=1; k<p; k++)
   {
      int l=0;
      for (int j=1; j<p; j++)
      {
         for (int i=1; i<j; i++)
         {
            t_dof[6 + 9 * ne + 2 * nt + 3 * nq + m] = 3 * p + l;
            s_dof[6 + 9 * ne + 2 * nt + 3 * nq + m] = 1 + k;
            l++; m++;
         }
      }
   }

   // Define Nodes
   const IntegrationRule & t_Nodes = TriangleFE.GetNodes();
   const IntegrationRule & s_Nodes = SegmentFE.GetNodes();
   for (int i=0; i<Dof; i++)
   {
      Nodes.IntPoint(i).x = t_Nodes.IntPoint(t_dof[i]).x;
      Nodes.IntPoint(i).y = t_Nodes.IntPoint(t_dof[i]).y;
      Nodes.IntPoint(i).z = s_Nodes.IntPoint(s_dof[i]).x;
   }
}

void H1_WedgeElement::CalcShape(const IntegrationPoint &ip,
                                Vector &shape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector t_shape(TriangleFE.GetDof());
   Vector s_shape(SegmentFE.GetDof());
#endif

   IntegrationPoint ipz; ipz.x = ip.z; ipz.y = 0.0; ipz.z = 0.0;

   TriangleFE.CalcShape(ip, t_shape);
   SegmentFE.CalcShape(ipz, s_shape);

   for (int i=0; i<Dof; i++)
   {
      shape[i] = t_shape[t_dof[i]] * s_shape[s_dof[i]];
   }
}

void H1_WedgeElement::CalcDShape(const IntegrationPoint &ip,
                                 DenseMatrix &dshape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector      t_shape(TriangleFE.GetDof());
   DenseMatrix t_dshape(TriangleFE.GetDof(), 2);
   Vector      s_shape(SegmentFE.GetDof());
   DenseMatrix s_dshape(SegmentFE.GetDof(), 1);
#endif

   IntegrationPoint ipz; ipz.x = ip.z; ipz.y = 0.0; ipz.z = 0.0;

   TriangleFE.CalcShape(ip, t_shape);
   TriangleFE.CalcDShape(ip, t_dshape);
   SegmentFE.CalcShape(ipz, s_shape);
   SegmentFE.CalcDShape(ipz, s_dshape);

   for (int i=0; i<Dof; i++)
   {
      dshape(i, 0) = t_dshape(t_dof[i],0) * s_shape[s_dof[i]];
      dshape(i, 1) = t_dshape(t_dof[i],1) * s_shape[s_dof[i]];
      dshape(i, 2) = t_shape[t_dof[i]] * s_dshape(s_dof[i],0);
   }
}


H1Pos_WedgeElement::H1Pos_WedgeElement(const int p)
   : PositiveFiniteElement(3, Geometry::PRISM,
                           ((p + 1)*(p + 1)*(p + 2))/2, p, FunctionSpace::Qk),
     TriangleFE(p),
     SegmentFE(p)
{
#ifndef MFEM_THREAD_SAFE
   t_shape.SetSize(TriangleFE.GetDof());
   s_shape.SetSize(SegmentFE.GetDof());
   t_dshape.SetSize(TriangleFE.GetDof(), 2);
   s_dshape.SetSize(SegmentFE.GetDof(), 1);
#endif

   t_dof.SetSize(Dof);
   s_dof.SetSize(Dof);

   // Nodal DoFs
   t_dof[0] = 0; s_dof[0] = 0;
   t_dof[1] = 1; s_dof[1] = 0;
   t_dof[2] = 2; s_dof[2] = 0;
   t_dof[3] = 0; s_dof[3] = 1;
   t_dof[4] = 1; s_dof[4] = 1;
   t_dof[5] = 2; s_dof[5] = 1;

   // Edge DoFs
   int ne = p-1;
   for (int i=1; i<p; i++)
   {
      t_dof[5 + 0 * ne + i] = 2 + 0 * ne + i; s_dof[5 + 0 * ne + i] = 0;
      t_dof[5 + 1 * ne + i] = 2 + 1 * ne + i; s_dof[5 + 1 * ne + i] = 0;
      t_dof[5 + 2 * ne + i] = 2 + 2 * ne + i; s_dof[5 + 2 * ne + i] = 0;
      t_dof[5 + 3 * ne + i] = 2 + 0 * ne + i; s_dof[5 + 3 * ne + i] = 1;
      t_dof[5 + 4 * ne + i] = 2 + 1 * ne + i; s_dof[5 + 4 * ne + i] = 1;
      t_dof[5 + 5 * ne + i] = 2 + 2 * ne + i; s_dof[5 + 5 * ne + i] = 1;
      t_dof[5 + 6 * ne + i] = 0;              s_dof[5 + 6 * ne + i] = i + 1;
      t_dof[5 + 7 * ne + i] = 1;              s_dof[5 + 7 * ne + i] = i + 1;
      t_dof[5 + 8 * ne + i] = 2;              s_dof[5 + 8 * ne + i] = i + 1;
   }

   // Triangular Face DoFs
   int k=0;
   int nt = (p-1)*(p-2)/2;
   for (int j=1; j<p; j++)
   {
      for (int i=1; i<j; i++)
      {
         t_dof[6 + 9 * ne + k]      = 3 * p + k; s_dof[6 + 9 * ne + k]      = 0;
         t_dof[6 + 9 * ne + nt + k] = 3 * p + k; s_dof[6 + 9 * ne + nt + k] = 1;
         k++;
      }
   }

   // Quadrilateral Face DoFs
   k=0;
   int nq = (p-1)*(p-1);
   for (int j=1; j<p; j++)
   {
      for (int i=1; i<p; i++)
      {
         t_dof[6 + 9 * ne + 2 * nt + 0 * nq + k] = 2 + 0 * ne + i;
         t_dof[6 + 9 * ne + 2 * nt + 1 * nq + k] = 2 + 1 * ne + i;
         t_dof[6 + 9 * ne + 2 * nt + 2 * nq + k] = 2 + 2 * ne + i;

         s_dof[6 + 9 * ne + 2 * nt + 0 * nq + k] = 1 + j;
         s_dof[6 + 9 * ne + 2 * nt + 1 * nq + k] = 1 + j;
         s_dof[6 + 9 * ne + 2 * nt + 2 * nq + k] = 1 + j;

         k++;
      }
   }

   // Interior DoFs
   int m=0;
   for (int k=1; k<p; k++)
   {
      int l=0;
      for (int j=1; j<p; j++)
      {
         for (int i=1; i<j; i++)
         {
            t_dof[6 + 9 * ne + 2 * nt + 3 * nq + m] = 3 * p + l;
            s_dof[6 + 9 * ne + 2 * nt + 3 * nq + m] = 1 + k;
            l++; m++;
         }
      }
   }

   // Define Nodes
   const IntegrationRule & t_Nodes = TriangleFE.GetNodes();
   const IntegrationRule & s_Nodes = SegmentFE.GetNodes();
   for (int i=0; i<Dof; i++)
   {
      Nodes.IntPoint(i).x = t_Nodes.IntPoint(t_dof[i]).x;
      Nodes.IntPoint(i).y = t_Nodes.IntPoint(t_dof[i]).y;
      Nodes.IntPoint(i).z = s_Nodes.IntPoint(s_dof[i]).x;
   }
}

void H1Pos_WedgeElement::CalcShape(const IntegrationPoint &ip,
                                   Vector &shape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector t_shape(TriangleFE.GetDof());
   Vector s_shape(SegmentFE.GetDof());
#endif

   IntegrationPoint ipz; ipz.x = ip.z; ipz.y = 0.0; ipz.z = 0.0;

   TriangleFE.CalcShape(ip, t_shape);
   SegmentFE.CalcShape(ipz, s_shape);

   for (int i=0; i<Dof; i++)
   {
      shape[i] = t_shape[t_dof[i]] * s_shape[s_dof[i]];
   }
}

void H1Pos_WedgeElement::CalcDShape(const IntegrationPoint &ip,
                                    DenseMatrix &dshape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector      t_shape(TriangleFE.GetDof());
   DenseMatrix t_dshape(TriangleFE.GetDof(), 2);
   Vector      s_shape(SegmentFE.GetDof());
   DenseMatrix s_dshape(SegmentFE.GetDof(), 1);
#endif

   IntegrationPoint ipz; ipz.x = ip.z; ipz.y = 0.0; ipz.z = 0.0;

   TriangleFE.CalcShape(ip, t_shape);
   TriangleFE.CalcDShape(ip, t_dshape);
   SegmentFE.CalcShape(ipz, s_shape);
   SegmentFE.CalcDShape(ipz, s_dshape);

   for (int i=0; i<Dof; i++)
   {
      dshape(i, 0) = t_dshape(t_dof[i],0) * s_shape[s_dof[i]];
      dshape(i, 1) = t_dshape(t_dof[i],1) * s_shape[s_dof[i]];
      dshape(i, 2) = t_shape[t_dof[i]] * s_dshape(s_dof[i],0);
   }
}


L2_SegmentElement::L2_SegmentElement(const int p, const int btype)
   : NodalTensorFiniteElement(1, p, VerifyOpen(btype), L2_DOF_MAP)
{
   const double *op = poly1d.OpenPoints(p, btype);

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p + 1);
   dshape_x.SetDataAndSize(NULL, p + 1);
#endif

   for (int i = 0; i <= p; i++)
   {
      Nodes.IntPoint(i).x = op[i];
   }
}

void L2_SegmentElement::CalcShape(const IntegrationPoint &ip,
                                  Vector &shape) const
{
   basis1d.Eval(ip.x, shape);
}

void L2_SegmentElement::CalcDShape(const IntegrationPoint &ip,
                                   DenseMatrix &dshape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector shape_x(Dof), dshape_x(dshape.Data(), Dof);
#else
   dshape_x.SetData(dshape.Data());
#endif
   basis1d.Eval(ip.x, shape_x, dshape_x);
}

void L2_SegmentElement::ProjectDelta(int vertex, Vector &dofs) const
{
   const int p = Order;
   const double *op = poly1d.OpenPoints(p, b_type);

   switch (vertex)
   {
      case 0:
         for (int i = 0; i <= p; i++)
         {
            dofs(i) = poly1d.CalcDelta(p,(1.0 - op[i]));
         }
         break;

      case 1:
         for (int i = 0; i <= p; i++)
         {
            dofs(i) = poly1d.CalcDelta(p,op[i]);
         }
         break;
   }
}


L2Pos_SegmentElement::L2Pos_SegmentElement(const int p)
   : PositiveTensorFiniteElement(1, p, L2_DOF_MAP)
{
#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p + 1);
   dshape_x.SetDataAndSize(NULL, p + 1);
#endif

   if (p == 0)
   {
      Nodes.IntPoint(0).x = 0.5;
   }
   else
   {
      for (int i = 0; i <= p; i++)
      {
         Nodes.IntPoint(i).x = double(i)/p;
      }
   }
}

void L2Pos_SegmentElement::CalcShape(const IntegrationPoint &ip,
                                     Vector &shape) const
{
   Poly_1D::CalcBernstein(Order, ip.x, shape);
}

void L2Pos_SegmentElement::CalcDShape(const IntegrationPoint &ip,
                                      DenseMatrix &dshape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector shape_x(Dof), dshape_x(dshape.Data(), Dof);
#else
   dshape_x.SetData(dshape.Data());
#endif
   Poly_1D::CalcBernstein(Order, ip.x, shape_x, dshape_x);
}

void L2Pos_SegmentElement::ProjectDelta(int vertex, Vector &dofs) const
{
   dofs = 0.0;
   dofs[vertex*Order] = 1.0;
}


L2_QuadrilateralElement::L2_QuadrilateralElement(const int p, const int btype)
   : NodalTensorFiniteElement(2, p, VerifyOpen(btype), L2_DOF_MAP)
{
   const double *op = poly1d.OpenPoints(p, b_type);

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p + 1);
   shape_y.SetSize(p + 1);
   dshape_x.SetSize(p + 1);
   dshape_y.SetSize(p + 1);
#endif

   for (int o = 0, j = 0; j <= p; j++)
      for (int i = 0; i <= p; i++)
      {
         Nodes.IntPoint(o++).Set2(op[i], op[j]);
      }
}

void L2_QuadrilateralElement::CalcShape(const IntegrationPoint &ip,
                                        Vector &shape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1);
#endif

   basis1d.Eval(ip.x, shape_x);
   basis1d.Eval(ip.y, shape_y);

   for (int o = 0, j = 0; j <= p; j++)
      for (int i = 0; i <= p; i++)
      {
         shape(o++) = shape_x(i)*shape_y(j);
      }
}

void L2_QuadrilateralElement::CalcDShape(const IntegrationPoint &ip,
                                         DenseMatrix &dshape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1), dshape_x(p+1), dshape_y(p+1);
#endif

   basis1d.Eval(ip.x, shape_x, dshape_x);
   basis1d.Eval(ip.y, shape_y, dshape_y);

   for (int o = 0, j = 0; j <= p; j++)
      for (int i = 0; i <= p; i++)
      {
         dshape(o,0) = dshape_x(i)* shape_y(j);
         dshape(o,1) =  shape_x(i)*dshape_y(j);  o++;
      }
}

void L2_QuadrilateralElement::ProjectDelta(int vertex, Vector &dofs) const
{
   const int p = Order;
   const double *op = poly1d.OpenPoints(p, b_type);

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1);
#endif

   for (int i = 0; i <= p; i++)
   {
      shape_x(i) = poly1d.CalcDelta(p,(1.0 - op[i]));
      shape_y(i) = poly1d.CalcDelta(p,op[i]);
   }

   switch (vertex)
   {
      case 0:
         for (int o = 0, j = 0; j <= p; j++)
            for (int i = 0; i <= p; i++)
            {
               dofs[o++] = shape_x(i)*shape_x(j);
            }
         break;
      case 1:
         for (int o = 0, j = 0; j <= p; j++)
            for (int i = 0; i <= p; i++)
            {
               dofs[o++] = shape_y(i)*shape_x(j);
            }
         break;
      case 2:
         for (int o = 0, j = 0; j <= p; j++)
            for (int i = 0; i <= p; i++)
            {
               dofs[o++] = shape_y(i)*shape_y(j);
            }
         break;
      case 3:
         for (int o = 0, j = 0; j <= p; j++)
            for (int i = 0; i <= p; i++)
            {
               dofs[o++] = shape_x(i)*shape_y(j);
            }
         break;
   }
}


L2Pos_QuadrilateralElement::L2Pos_QuadrilateralElement(const int p)
   : PositiveTensorFiniteElement(2, p, L2_DOF_MAP)
{
#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p + 1);
   shape_y.SetSize(p + 1);
   dshape_x.SetSize(p + 1);
   dshape_y.SetSize(p + 1);
#endif

   if (p == 0)
   {
      Nodes.IntPoint(0).Set2(0.5, 0.5);
   }
   else
   {
      for (int o = 0, j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            Nodes.IntPoint(o++).Set2(double(i)/p, double(j)/p);
         }
   }
}

void L2Pos_QuadrilateralElement::CalcShape(const IntegrationPoint &ip,
                                           Vector &shape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1);
#endif

   Poly_1D::CalcBernstein(p, ip.x, shape_x);
   Poly_1D::CalcBernstein(p, ip.y, shape_y);

   for (int o = 0, j = 0; j <= p; j++)
      for (int i = 0; i <= p; i++)
      {
         shape(o++) = shape_x(i)*shape_y(j);
      }
}

void L2Pos_QuadrilateralElement::CalcDShape(const IntegrationPoint &ip,
                                            DenseMatrix &dshape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1), dshape_x(p+1), dshape_y(p+1);
#endif

   Poly_1D::CalcBernstein(p, ip.x, shape_x, dshape_x);
   Poly_1D::CalcBernstein(p, ip.y, shape_y, dshape_y);

   for (int o = 0, j = 0; j <= p; j++)
      for (int i = 0; i <= p; i++)
      {
         dshape(o,0) = dshape_x(i)* shape_y(j);
         dshape(o,1) =  shape_x(i)*dshape_y(j);  o++;
      }
}

void L2Pos_QuadrilateralElement::ProjectDelta(int vertex, Vector &dofs) const
{
   const int p = Order;

   dofs = 0.0;
   switch (vertex)
   {
      case 0: dofs[0] = 1.0; break;
      case 1: dofs[p] = 1.0; break;
      case 2: dofs[p*(p + 2)] = 1.0; break;
      case 3: dofs[p*(p + 1)] = 1.0; break;
   }
}


L2_HexahedronElement::L2_HexahedronElement(const int p, const int btype)
   : NodalTensorFiniteElement(3, p, VerifyOpen(btype), L2_DOF_MAP)
{
   const double *op = poly1d.OpenPoints(p, btype);

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p + 1);
   shape_y.SetSize(p + 1);
   shape_z.SetSize(p + 1);
   dshape_x.SetSize(p + 1);
   dshape_y.SetSize(p + 1);
   dshape_z.SetSize(p + 1);
#endif

   for (int o = 0, k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            Nodes.IntPoint(o++).Set3(op[i], op[j], op[k]);
         }
}

void L2_HexahedronElement::CalcShape(const IntegrationPoint &ip,
                                     Vector &shape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1), shape_z(p+1);
#endif

   basis1d.Eval(ip.x, shape_x);
   basis1d.Eval(ip.y, shape_y);
   basis1d.Eval(ip.z, shape_z);

   for (int o = 0, k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            shape(o++) = shape_x(i)*shape_y(j)*shape_z(k);
         }
}

void L2_HexahedronElement::CalcDShape(const IntegrationPoint &ip,
                                      DenseMatrix &dshape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1),  shape_y(p+1),  shape_z(p+1);
   Vector dshape_x(p+1), dshape_y(p+1), dshape_z(p+1);
#endif

   basis1d.Eval(ip.x, shape_x, dshape_x);
   basis1d.Eval(ip.y, shape_y, dshape_y);
   basis1d.Eval(ip.z, shape_z, dshape_z);

   for (int o = 0, k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            dshape(o,0) = dshape_x(i)* shape_y(j)* shape_z(k);
            dshape(o,1) =  shape_x(i)*dshape_y(j)* shape_z(k);
            dshape(o,2) =  shape_x(i)* shape_y(j)*dshape_z(k);  o++;
         }
}

void L2_HexahedronElement::ProjectDelta(int vertex, Vector &dofs) const
{
   const int p = Order;
   const double *op = poly1d.OpenPoints(p, b_type);

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1);
#endif

   for (int i = 0; i <= p; i++)
   {
      shape_x(i) = poly1d.CalcDelta(p,(1.0 - op[i]));
      shape_y(i) = poly1d.CalcDelta(p,op[i]);
   }

   switch (vertex)
   {
      case 0:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs[o++] = shape_x(i)*shape_x(j)*shape_x(k);
               }
         break;
      case 1:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs[o++] = shape_y(i)*shape_x(j)*shape_x(k);
               }
         break;
      case 2:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs[o++] = shape_y(i)*shape_y(j)*shape_x(k);
               }
         break;
      case 3:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs[o++] = shape_x(i)*shape_y(j)*shape_x(k);
               }
         break;
      case 4:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs[o++] = shape_x(i)*shape_x(j)*shape_y(k);
               }
         break;
      case 5:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs[o++] = shape_y(i)*shape_x(j)*shape_y(k);
               }
         break;
      case 6:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs[o++] = shape_y(i)*shape_y(j)*shape_y(k);
               }
         break;
      case 7:
         for (int o = 0, k = 0; k <= p; k++)
            for (int j = 0; j <= p; j++)
               for (int i = 0; i <= p; i++)
               {
                  dofs[o++] = shape_x(i)*shape_y(j)*shape_y(k);
               }
         break;
   }
}


L2Pos_HexahedronElement::L2Pos_HexahedronElement(const int p)
   : PositiveTensorFiniteElement(3, p, L2_DOF_MAP)
{
#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p + 1);
   shape_y.SetSize(p + 1);
   shape_z.SetSize(p + 1);
   dshape_x.SetSize(p + 1);
   dshape_y.SetSize(p + 1);
   dshape_z.SetSize(p + 1);
#endif

   if (p == 0)
   {
      Nodes.IntPoint(0).Set3(0.5, 0.5, 0.5);
   }
   else
   {
      for (int o = 0, k = 0; k <= p; k++)
         for (int j = 0; j <= p; j++)
            for (int i = 0; i <= p; i++)
            {
               Nodes.IntPoint(o++).Set3(double(i)/p, double(j)/p, double(k)/p);
            }
   }
}

void L2Pos_HexahedronElement::CalcShape(const IntegrationPoint &ip,
                                        Vector &shape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1), shape_z(p+1);
#endif

   Poly_1D::CalcBernstein(p, ip.x, shape_x);
   Poly_1D::CalcBernstein(p, ip.y, shape_y);
   Poly_1D::CalcBernstein(p, ip.z, shape_z);

   for (int o = 0, k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            shape(o++) = shape_x(i)*shape_y(j)*shape_z(k);
         }
}

void L2Pos_HexahedronElement::CalcDShape(const IntegrationPoint &ip,
                                         DenseMatrix &dshape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1),  shape_y(p+1),  shape_z(p+1);
   Vector dshape_x(p+1), dshape_y(p+1), dshape_z(p+1);
#endif

   Poly_1D::CalcBernstein(p, ip.x, shape_x, dshape_x);
   Poly_1D::CalcBernstein(p, ip.y, shape_y, dshape_y);
   Poly_1D::CalcBernstein(p, ip.z, shape_z, dshape_z);

   for (int o = 0, k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            dshape(o,0) = dshape_x(i)* shape_y(j)* shape_z(k);
            dshape(o,1) =  shape_x(i)*dshape_y(j)* shape_z(k);
            dshape(o,2) =  shape_x(i)* shape_y(j)*dshape_z(k);  o++;
         }
}

void L2Pos_HexahedronElement::ProjectDelta(int vertex, Vector &dofs) const
{
   const int p = Order;

   dofs = 0.0;
   switch (vertex)
   {
      case 0: dofs[0] = 1.0; break;
      case 1: dofs[p] = 1.0; break;
      case 2: dofs[p*(p + 2)] = 1.0; break;
      case 3: dofs[p*(p + 1)] = 1.0; break;
      case 4: dofs[p*(p + 1)*(p + 1)] = 1.0; break;
      case 5: dofs[p + p*(p + 1)*(p + 1)] = 1.0; break;
      case 6: dofs[Dof - 1] = 1.0; break;
      case 7: dofs[Dof - p - 1] = 1.0; break;
   }
}


L2_TriangleElement::L2_TriangleElement(const int p, const int btype)
   : NodalFiniteElement(2, Geometry::TRIANGLE, ((p + 1)*(p + 2))/2, p,
                        FunctionSpace::Pk)
{
   const double *op = poly1d.OpenPoints(p, VerifyNodal(VerifyOpen(btype)));

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p + 1);
   shape_y.SetSize(p + 1);
   shape_l.SetSize(p + 1);
   dshape_x.SetSize(p + 1);
   dshape_y.SetSize(p + 1);
   dshape_l.SetSize(p + 1);
   u.SetSize(Dof);
   du.SetSize(Dof, Dim);
#else
   Vector shape_x(p + 1), shape_y(p + 1), shape_l(p + 1);
#endif

   for (int o = 0, j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)
      {
         double w = op[i] + op[j] + op[p-i-j];
         Nodes.IntPoint(o++).Set2(op[i]/w, op[j]/w);
      }

   DenseMatrix T(Dof);
   for (int k = 0; k < Dof; k++)
   {
      IntegrationPoint &ip = Nodes.IntPoint(k);
      poly1d.CalcBasis(p, ip.x, shape_x);
      poly1d.CalcBasis(p, ip.y, shape_y);
      poly1d.CalcBasis(p, 1. - ip.x - ip.y, shape_l);

      for (int o = 0, j = 0; j <= p; j++)
         for (int i = 0; i + j <= p; i++)
         {
            T(o++, k) = shape_x(i)*shape_y(j)*shape_l(p-i-j);
         }
   }

   Ti.Factor(T);
   // mfem::out << "L2_TriangleElement(" << p << ") : "; Ti.TestInversion();
}

void L2_TriangleElement::CalcShape(const IntegrationPoint &ip,
                                   Vector &shape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p + 1), shape_y(p + 1), shape_l(p + 1), u(Dof);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x);
   poly1d.CalcBasis(p, ip.y, shape_y);
   poly1d.CalcBasis(p, 1. - ip.x - ip.y, shape_l);

   for (int o = 0, j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)
      {
         u(o++) = shape_x(i)*shape_y(j)*shape_l(p-i-j);
      }

   Ti.Mult(u, shape);
}

void L2_TriangleElement::CalcDShape(const IntegrationPoint &ip,
                                    DenseMatrix &dshape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector  shape_x(p + 1),  shape_y(p + 1),  shape_l(p + 1);
   Vector dshape_x(p + 1), dshape_y(p + 1), dshape_l(p + 1);
   DenseMatrix du(Dof, Dim);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x, dshape_x);
   poly1d.CalcBasis(p, ip.y, shape_y, dshape_y);
   poly1d.CalcBasis(p, 1. - ip.x - ip.y, shape_l, dshape_l);

   for (int o = 0, j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)
      {
         int k = p - i - j;
         du(o,0) = ((dshape_x(i)* shape_l(k)) -
                    ( shape_x(i)*dshape_l(k)))*shape_y(j);
         du(o,1) = ((dshape_y(j)* shape_l(k)) -
                    ( shape_y(j)*dshape_l(k)))*shape_x(i);
         o++;
      }

   Ti.Mult(du, dshape);
}

void L2_TriangleElement::ProjectDelta(int vertex, Vector &dofs) const
{
   switch (vertex)
   {
      case 0:
         for (int i = 0; i < Dof; i++)
         {
            const IntegrationPoint &ip = Nodes.IntPoint(i);
            dofs[i] = pow(1.0 - ip.x - ip.y, Order);
         }
         break;
      case 1:
         for (int i = 0; i < Dof; i++)
         {
            const IntegrationPoint &ip = Nodes.IntPoint(i);
            dofs[i] = pow(ip.x, Order);
         }
         break;
      case 2:
         for (int i = 0; i < Dof; i++)
         {
            const IntegrationPoint &ip = Nodes.IntPoint(i);
            dofs[i] = pow(ip.y, Order);
         }
         break;
   }
}


L2Pos_TriangleElement::L2Pos_TriangleElement(const int p)
   : PositiveFiniteElement(2, Geometry::TRIANGLE, ((p + 1)*(p + 2))/2, p,
                           FunctionSpace::Pk)
{
#ifndef MFEM_THREAD_SAFE
   dshape_1d.SetSize(p + 1);
#endif

   if (p == 0)
   {
      Nodes.IntPoint(0).Set2(1./3, 1./3);
   }
   else
   {
      for (int o = 0, j = 0; j <= p; j++)
         for (int i = 0; i + j <= p; i++)
         {
            Nodes.IntPoint(o++).Set2(double(i)/p, double(j)/p);
         }
   }
}

void L2Pos_TriangleElement::CalcShape(const IntegrationPoint &ip,
                                      Vector &shape) const
{
   H1Pos_TriangleElement::CalcShape(Order, ip.x, ip.y, shape.GetData());
}

void L2Pos_TriangleElement::CalcDShape(const IntegrationPoint &ip,
                                       DenseMatrix &dshape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector dshape_1d(Order + 1);
#endif

   H1Pos_TriangleElement::CalcDShape(Order, ip.x, ip.y, dshape_1d.GetData(),
                                     dshape.Data());
}

void L2Pos_TriangleElement::ProjectDelta(int vertex, Vector &dofs) const
{
   dofs = 0.0;
   switch (vertex)
   {
      case 0: dofs[0] = 1.0; break;
      case 1: dofs[Order] = 1.0; break;
      case 2: dofs[Dof-1] = 1.0; break;
   }
}


L2_TetrahedronElement::L2_TetrahedronElement(const int p, const int btype)
   : NodalFiniteElement(3, Geometry::TETRAHEDRON, ((p + 1)*(p + 2)*(p + 3))/6,
                        p, FunctionSpace::Pk)
{
   const double *op = poly1d.OpenPoints(p, VerifyNodal(VerifyOpen(btype)));

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p + 1);
   shape_y.SetSize(p + 1);
   shape_z.SetSize(p + 1);
   shape_l.SetSize(p + 1);
   dshape_x.SetSize(p + 1);
   dshape_y.SetSize(p + 1);
   dshape_z.SetSize(p + 1);
   dshape_l.SetSize(p + 1);
   u.SetSize(Dof);
   du.SetSize(Dof, Dim);
#else
   Vector shape_x(p + 1), shape_y(p + 1), shape_z(p + 1), shape_l(p + 1);
#endif

   for (int o = 0, k = 0; k <= p; k++)
      for (int j = 0; j + k <= p; j++)
         for (int i = 0; i + j + k <= p; i++)
         {
            double w = op[i] + op[j] + op[k] + op[p-i-j-k];
            Nodes.IntPoint(o++).Set3(op[i]/w, op[j]/w, op[k]/w);
         }

   DenseMatrix T(Dof);
   for (int m = 0; m < Dof; m++)
   {
      IntegrationPoint &ip = Nodes.IntPoint(m);
      poly1d.CalcBasis(p, ip.x, shape_x);
      poly1d.CalcBasis(p, ip.y, shape_y);
      poly1d.CalcBasis(p, ip.z, shape_z);
      poly1d.CalcBasis(p, 1. - ip.x - ip.y - ip.z, shape_l);

      for (int o = 0, k = 0; k <= p; k++)
         for (int j = 0; j + k <= p; j++)
            for (int i = 0; i + j + k <= p; i++)
            {
               T(o++, m) = shape_x(i)*shape_y(j)*shape_z(k)*shape_l(p-i-j-k);
            }
   }

   Ti.Factor(T);
   // mfem::out << "L2_TetrahedronElement(" << p << ") : "; Ti.TestInversion();
}

void L2_TetrahedronElement::CalcShape(const IntegrationPoint &ip,
                                      Vector &shape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p + 1), shape_y(p + 1), shape_z(p + 1), shape_l(p + 1);
   Vector u(Dof);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x);
   poly1d.CalcBasis(p, ip.y, shape_y);
   poly1d.CalcBasis(p, ip.z, shape_z);
   poly1d.CalcBasis(p, 1. - ip.x - ip.y - ip.z, shape_l);

   for (int o = 0, k = 0; k <= p; k++)
      for (int j = 0; j + k <= p; j++)
         for (int i = 0; i + j + k <= p; i++)
         {
            u(o++) = shape_x(i)*shape_y(j)*shape_z(k)*shape_l(p-i-j-k);
         }

   Ti.Mult(u, shape);
}

void L2_TetrahedronElement::CalcDShape(const IntegrationPoint &ip,
                                       DenseMatrix &dshape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector  shape_x(p + 1),  shape_y(p + 1),  shape_z(p + 1),  shape_l(p + 1);
   Vector dshape_x(p + 1), dshape_y(p + 1), dshape_z(p + 1), dshape_l(p + 1);
   DenseMatrix du(Dof, Dim);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x, dshape_x);
   poly1d.CalcBasis(p, ip.y, shape_y, dshape_y);
   poly1d.CalcBasis(p, ip.z, shape_z, dshape_z);
   poly1d.CalcBasis(p, 1. - ip.x - ip.y - ip.z, shape_l, dshape_l);

   for (int o = 0, k = 0; k <= p; k++)
      for (int j = 0; j + k <= p; j++)
         for (int i = 0; i + j + k <= p; i++)
         {
            int l = p - i - j - k;
            du(o,0) = ((dshape_x(i)* shape_l(l)) -
                       ( shape_x(i)*dshape_l(l)))*shape_y(j)*shape_z(k);
            du(o,1) = ((dshape_y(j)* shape_l(l)) -
                       ( shape_y(j)*dshape_l(l)))*shape_x(i)*shape_z(k);
            du(o,2) = ((dshape_z(k)* shape_l(l)) -
                       ( shape_z(k)*dshape_l(l)))*shape_x(i)*shape_y(j);
            o++;
         }

   Ti.Mult(du, dshape);
}

void L2_TetrahedronElement::ProjectDelta(int vertex, Vector &dofs) const
{
   switch (vertex)
   {
      case 0:
         for (int i = 0; i < Dof; i++)
         {
            const IntegrationPoint &ip = Nodes.IntPoint(i);
            dofs[i] = pow(1.0 - ip.x - ip.y - ip.z, Order);
         }
         break;
      case 1:
         for (int i = 0; i < Dof; i++)
         {
            const IntegrationPoint &ip = Nodes.IntPoint(i);
            dofs[i] = pow(ip.x, Order);
         }
         break;
      case 2:
         for (int i = 0; i < Dof; i++)
         {
            const IntegrationPoint &ip = Nodes.IntPoint(i);
            dofs[i] = pow(ip.y, Order);
         }
      case 3:
         for (int i = 0; i < Dof; i++)
         {
            const IntegrationPoint &ip = Nodes.IntPoint(i);
            dofs[i] = pow(ip.z, Order);
         }
         break;
   }
}


L2Pos_TetrahedronElement::L2Pos_TetrahedronElement(const int p)
   : PositiveFiniteElement(3, Geometry::TETRAHEDRON,
                           ((p + 1)*(p + 2)*(p + 3))/6, p, FunctionSpace::Pk)
{
#ifndef MFEM_THREAD_SAFE
   dshape_1d.SetSize(p + 1);
#endif

   if (p == 0)
   {
      Nodes.IntPoint(0).Set3(0.25, 0.25, 0.25);
   }
   else
   {
      for (int o = 0, k = 0; k <= p; k++)
         for (int j = 0; j + k <= p; j++)
            for (int i = 0; i + j + k <= p; i++)
            {
               Nodes.IntPoint(o++).Set3(double(i)/p, double(j)/p, double(k)/p);
            }
   }
}

void L2Pos_TetrahedronElement::CalcShape(const IntegrationPoint &ip,
                                         Vector &shape) const
{
   H1Pos_TetrahedronElement::CalcShape(Order, ip.x, ip.y, ip.z,
                                       shape.GetData());
}

void L2Pos_TetrahedronElement::CalcDShape(const IntegrationPoint &ip,
                                          DenseMatrix &dshape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector dshape_1d(Order + 1);
#endif

   H1Pos_TetrahedronElement::CalcDShape(Order, ip.x, ip.y, ip.z,
                                        dshape_1d.GetData(), dshape.Data());
}

void L2Pos_TetrahedronElement::ProjectDelta(int vertex, Vector &dofs) const
{
   dofs = 0.0;
   switch (vertex)
   {
      case 0: dofs[0] = 1.0; break;
      case 1: dofs[Order] = 1.0; break;
      case 2: dofs[(Order*(Order+3))/2] = 1.0; break;
      case 3: dofs[Dof-1] = 1.0; break;
   }
}


L2_WedgeElement::L2_WedgeElement(const int p, const int btype)
   : NodalFiniteElement(3, Geometry::PRISM, ((p + 1)*(p + 1)*(p + 2))/2,
                        p, FunctionSpace::Qk),
     TriangleFE(p, btype),
     SegmentFE(p, btype)
{
#ifndef MFEM_THREAD_SAFE
   t_shape.SetSize(TriangleFE.GetDof());
   s_shape.SetSize(SegmentFE.GetDof());
   t_dshape.SetSize(TriangleFE.GetDof(), 2);
   s_dshape.SetSize(SegmentFE.GetDof(), 1);
#endif

   t_dof.SetSize(Dof);
   s_dof.SetSize(Dof);

   // Interior DoFs
   int m=0;
   for (int k=0; k<=p; k++)
   {
      int l=0;
      for (int j=0; j<=p; j++)
      {
         for (int i=0; i<=j; i++)
         {
            t_dof[m] = l;
            s_dof[m] = k;
            l++; m++;
         }
      }
   }

   // Define Nodes
   const IntegrationRule & t_Nodes = TriangleFE.GetNodes();
   const IntegrationRule & s_Nodes = SegmentFE.GetNodes();
   for (int i=0; i<Dof; i++)
   {
      Nodes.IntPoint(i).x = t_Nodes.IntPoint(t_dof[i]).x;
      Nodes.IntPoint(i).y = t_Nodes.IntPoint(t_dof[i]).y;
      Nodes.IntPoint(i).z = s_Nodes.IntPoint(s_dof[i]).x;
   }
}

void L2_WedgeElement::CalcShape(const IntegrationPoint &ip,
                                Vector &shape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector t_shape(TriangleFE.GetDof());
   Vector s_shape(SegmentFE.GetDof());
#endif

   IntegrationPoint ipz; ipz.x = ip.z; ipz.y = 0.0; ipz.z = 0.0;

   TriangleFE.CalcShape(ip, t_shape);
   SegmentFE.CalcShape(ipz, s_shape);

   for (int i=0; i<Dof; i++)
   {
      shape[i] = t_shape[t_dof[i]] * s_shape[s_dof[i]];
   }
}

void L2_WedgeElement::CalcDShape(const IntegrationPoint &ip,
                                 DenseMatrix &dshape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector      t_shape(TriangleFE.GetDof());
   DenseMatrix t_dshape(TriangleFE.GetDof(), 2);
   Vector      s_shape(SegmentFE.GetDof());
   DenseMatrix s_dshape(SegmentFE.GetDof(), 1);
#endif

   IntegrationPoint ipz; ipz.x = ip.z; ipz.y = 0.0; ipz.z = 0.0;

   TriangleFE.CalcShape(ip, t_shape);
   TriangleFE.CalcDShape(ip, t_dshape);
   SegmentFE.CalcShape(ipz, s_shape);
   SegmentFE.CalcDShape(ipz, s_dshape);

   for (int i=0; i<Dof; i++)
   {
      dshape(i, 0) = t_dshape(t_dof[i],0) * s_shape[s_dof[i]];
      dshape(i, 1) = t_dshape(t_dof[i],1) * s_shape[s_dof[i]];
      dshape(i, 2) = t_shape[t_dof[i]] * s_dshape(s_dof[i],0);
   }
}


L2Pos_WedgeElement::L2Pos_WedgeElement(const int p)
   : PositiveFiniteElement(3, Geometry::PRISM,
                           ((p + 1)*(p + 1)*(p + 2))/2, p, FunctionSpace::Qk),
     TriangleFE(p),
     SegmentFE(p)
{
#ifndef MFEM_THREAD_SAFE
   t_shape.SetSize(TriangleFE.GetDof());
   s_shape.SetSize(SegmentFE.GetDof());
   t_dshape.SetSize(TriangleFE.GetDof(), 2);
   s_dshape.SetSize(SegmentFE.GetDof(), 1);
#endif

   t_dof.SetSize(Dof);
   s_dof.SetSize(Dof);

   // Interior DoFs
   int m=0;
   for (int k=0; k<=p; k++)
   {
      int l=0;
      for (int j=0; j<=p; j++)
      {
         for (int i=0; i<=j; i++)
         {
            t_dof[m] = l;
            s_dof[m] = k;
            l++; m++;
         }
      }
   }

   // Define Nodes
   const IntegrationRule & t_Nodes = TriangleFE.GetNodes();
   const IntegrationRule & s_Nodes = SegmentFE.GetNodes();
   for (int i=0; i<Dof; i++)
   {
      Nodes.IntPoint(i).x = t_Nodes.IntPoint(t_dof[i]).x;
      Nodes.IntPoint(i).y = t_Nodes.IntPoint(t_dof[i]).y;
      Nodes.IntPoint(i).z = s_Nodes.IntPoint(s_dof[i]).x;
   }
}

void L2Pos_WedgeElement::CalcShape(const IntegrationPoint &ip,
                                   Vector &shape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector t_shape(TriangleFE.GetDof());
   Vector s_shape(SegmentFE.GetDof());
#endif

   IntegrationPoint ipz; ipz.x = ip.z; ipz.y = 0.0; ipz.z = 0.0;

   TriangleFE.CalcShape(ip, t_shape);
   SegmentFE.CalcShape(ipz, s_shape);

   for (int i=0; i<Dof; i++)
   {
      shape[i] = t_shape[t_dof[i]] * s_shape[s_dof[i]];
   }
}

void L2Pos_WedgeElement::CalcDShape(const IntegrationPoint &ip,
                                    DenseMatrix &dshape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector      t_shape(TriangleFE.GetDof());
   DenseMatrix t_dshape(TriangleFE.GetDof(), 2);
   Vector      s_shape(SegmentFE.GetDof());
   DenseMatrix s_dshape(SegmentFE.GetDof(), 1);
#endif

   IntegrationPoint ipz; ipz.x = ip.z; ipz.y = 0.0; ipz.z = 0.0;

   TriangleFE.CalcShape(ip, t_shape);
   TriangleFE.CalcDShape(ip, t_dshape);
   SegmentFE.CalcShape(ipz, s_shape);
   SegmentFE.CalcDShape(ipz, s_dshape);

   for (int i=0; i<Dof; i++)
   {
      dshape(i, 0) = t_dshape(t_dof[i],0) * s_shape[s_dof[i]];
      dshape(i, 1) = t_dshape(t_dof[i],1) * s_shape[s_dof[i]];
      dshape(i, 2) = t_shape[t_dof[i]] * s_dshape(s_dof[i],0);
   }
}


const double RT_QuadrilateralElement::nk[8] =
{ 0., -1.,  1., 0.,  0., 1.,  -1., 0. };

RT_QuadrilateralElement::RT_QuadrilateralElement(const int p,
                                                 const int cb_type,
                                                 const int ob_type)
   : VectorFiniteElement(2, Geometry::SQUARE, 2*(p + 1)*(p + 2), p + 1,
                         H_DIV, FunctionSpace::Qk),
     cbasis1d(poly1d.GetBasis(p + 1, VerifyClosed(cb_type))),
     obasis1d(poly1d.GetBasis(p, VerifyOpen(ob_type))),
     dof_map(Dof), dof2nk(Dof)
{
   const double *cp = poly1d.ClosedPoints(p + 1, cb_type);
   const double *op = poly1d.OpenPoints(p, ob_type);
   const int dof2 = Dof/2;

#ifndef MFEM_THREAD_SAFE
   shape_cx.SetSize(p + 2);
   shape_ox.SetSize(p + 1);
   shape_cy.SetSize(p + 2);
   shape_oy.SetSize(p + 1);
   dshape_cx.SetSize(p + 2);
   dshape_cy.SetSize(p + 2);
#endif

   // edges
   int o = 0;
   for (int i = 0; i <= p; i++)  // (0,1)
   {
      dof_map[1*dof2 + i + 0*(p + 1)] = o++;
   }
   for (int i = 0; i <= p; i++)  // (1,2)
   {
      dof_map[0*dof2 + (p + 1) + i*(p + 2)] = o++;
   }
   for (int i = 0; i <= p; i++)  // (2,3)
   {
      dof_map[1*dof2 + (p - i) + (p + 1)*(p + 1)] = o++;
   }
   for (int i = 0; i <= p; i++)  // (3,0)
   {
      dof_map[0*dof2 + 0 + (p - i)*(p + 2)] = o++;
   }

   // interior
   for (int j = 0; j <= p; j++)  // x-components
      for (int i = 1; i <= p; i++)
      {
         dof_map[0*dof2 + i + j*(p + 2)] = o++;
      }
   for (int j = 1; j <= p; j++)  // y-components
      for (int i = 0; i <= p; i++)
      {
         dof_map[1*dof2 + i + j*(p + 1)] = o++;
      }

   // dof orientations
   // x-components
   for (int j = 0; j <= p; j++)
      for (int i = 0; i <= p/2; i++)
      {
         int idx = 0*dof2 + i + j*(p + 2);
         dof_map[idx] = -1 - dof_map[idx];
      }
   if (p%2 == 1)
      for (int j = p/2 + 1; j <= p; j++)
      {
         int idx = 0*dof2 + (p/2 + 1) + j*(p + 2);
         dof_map[idx] = -1 - dof_map[idx];
      }
   // y-components
   for (int j = 0; j <= p/2; j++)
      for (int i = 0; i <= p; i++)
      {
         int idx = 1*dof2 + i + j*(p + 1);
         dof_map[idx] = -1 - dof_map[idx];
      }
   if (p%2 == 1)
      for (int i = 0; i <= p/2; i++)
      {
         int idx = 1*dof2 + i + (p/2 + 1)*(p + 1);
         dof_map[idx] = -1 - dof_map[idx];
      }

   o = 0;
   for (int j = 0; j <= p; j++)
      for (int i = 0; i <= p + 1; i++)
      {
         int idx;
         if ((idx = dof_map[o++]) < 0)
         {
            idx = -1 - idx;
            dof2nk[idx] = 3;
         }
         else
         {
            dof2nk[idx] = 1;
         }
         Nodes.IntPoint(idx).Set2(cp[i], op[j]);
      }
   for (int j = 0; j <= p + 1; j++)
      for (int i = 0; i <= p; i++)
      {
         int idx;
         if ((idx = dof_map[o++]) < 0)
         {
            idx = -1 - idx;
            dof2nk[idx] = 0;
         }
         else
         {
            dof2nk[idx] = 2;
         }
         Nodes.IntPoint(idx).Set2(op[i], cp[j]);
      }
}

void RT_QuadrilateralElement::CalcVShape(const IntegrationPoint &ip,
                                         DenseMatrix &shape) const
{
   const int pp1 = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_cx(pp1 + 1), shape_ox(pp1), shape_cy(pp1 + 1), shape_oy(pp1);
#endif

   cbasis1d.Eval(ip.x, shape_cx);
   obasis1d.Eval(ip.x, shape_ox);
   cbasis1d.Eval(ip.y, shape_cy);
   obasis1d.Eval(ip.y, shape_oy);

   int o = 0;
   for (int j = 0; j < pp1; j++)
      for (int i = 0; i <= pp1; i++)
      {
         int idx, s;
         if ((idx = dof_map[o++]) < 0)
         {
            idx = -1 - idx, s = -1;
         }
         else
         {
            s = +1;
         }
         shape(idx,0) = s*shape_cx(i)*shape_oy(j);
         shape(idx,1) = 0.;
      }
   for (int j = 0; j <= pp1; j++)
      for (int i = 0; i < pp1; i++)
      {
         int idx, s;
         if ((idx = dof_map[o++]) < 0)
         {
            idx = -1 - idx, s = -1;
         }
         else
         {
            s = +1;
         }
         shape(idx,0) = 0.;
         shape(idx,1) = s*shape_ox(i)*shape_cy(j);
      }
}

void RT_QuadrilateralElement::CalcDivShape(const IntegrationPoint &ip,
                                           Vector &divshape) const
{
   const int pp1 = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_cx(pp1 + 1), shape_ox(pp1), shape_cy(pp1 + 1), shape_oy(pp1);
   Vector dshape_cx(pp1 + 1), dshape_cy(pp1 + 1);
#endif

   cbasis1d.Eval(ip.x, shape_cx, dshape_cx);
   obasis1d.Eval(ip.x, shape_ox);
   cbasis1d.Eval(ip.y, shape_cy, dshape_cy);
   obasis1d.Eval(ip.y, shape_oy);

   int o = 0;
   for (int j = 0; j < pp1; j++)
      for (int i = 0; i <= pp1; i++)
      {
         int idx, s;
         if ((idx = dof_map[o++]) < 0)
         {
            idx = -1 - idx, s = -1;
         }
         else
         {
            s = +1;
         }
         divshape(idx) = s*dshape_cx(i)*shape_oy(j);
      }
   for (int j = 0; j <= pp1; j++)
      for (int i = 0; i < pp1; i++)
      {
         int idx, s;
         if ((idx = dof_map[o++]) < 0)
         {
            idx = -1 - idx, s = -1;
         }
         else
         {
            s = +1;
         }
         divshape(idx) = s*shape_ox(i)*dshape_cy(j);
      }
}


const double RT_HexahedronElement::nk[18] =
{ 0.,0.,-1.,  0.,-1.,0.,  1.,0.,0.,  0.,1.,0.,  -1.,0.,0.,  0.,0.,1. };

RT_HexahedronElement::RT_HexahedronElement(const int p,
                                           const int cb_type,
                                           const int ob_type)
   : VectorFiniteElement(3, Geometry::CUBE, 3*(p + 1)*(p + 1)*(p + 2), p + 1,
                         H_DIV, FunctionSpace::Qk),
     cbasis1d(poly1d.GetBasis(p + 1, VerifyClosed(cb_type))),
     obasis1d(poly1d.GetBasis(p, VerifyOpen(ob_type))),
     dof_map(Dof), dof2nk(Dof)
{
   const double *cp = poly1d.ClosedPoints(p + 1, cb_type);
   const double *op = poly1d.OpenPoints(p, ob_type);
   const int dof3 = Dof/3;

#ifndef MFEM_THREAD_SAFE
   shape_cx.SetSize(p + 2);
   shape_ox.SetSize(p + 1);
   shape_cy.SetSize(p + 2);
   shape_oy.SetSize(p + 1);
   shape_cz.SetSize(p + 2);
   shape_oz.SetSize(p + 1);
   dshape_cx.SetSize(p + 2);
   dshape_cy.SetSize(p + 2);
   dshape_cz.SetSize(p + 2);
#endif

   // faces
   int o = 0;
   for (int j = 0; j <= p; j++)  // (3,2,1,0) -- bottom
      for (int i = 0; i <= p; i++)
      {
         dof_map[2*dof3 + i + ((p - j) + 0*(p + 1))*(p + 1)] = o++;
      }
   for (int j = 0; j <= p; j++)  // (0,1,5,4) -- front
      for (int i = 0; i <= p; i++)
      {
         dof_map[1*dof3 + i + (0 + j*(p + 2))*(p + 1)] = o++;
      }
   for (int j = 0; j <= p; j++)  // (1,2,6,5) -- right
      for (int i = 0; i <= p; i++)
      {
         dof_map[0*dof3 + (p + 1) + (i + j*(p + 1))*(p + 2)] = o++;
      }
   for (int j = 0; j <= p; j++)  // (2,3,7,6) -- back
      for (int i = 0; i <= p; i++)
      {
         dof_map[1*dof3 + (p - i) + ((p + 1) + j*(p + 2))*(p + 1)] = o++;
      }
   for (int j = 0; j <= p; j++)  // (3,0,4,7) -- left
      for (int i = 0; i <= p; i++)
      {
         dof_map[0*dof3 + 0 + ((p - i) + j*(p + 1))*(p + 2)] = o++;
      }
   for (int j = 0; j <= p; j++)  // (4,5,6,7) -- top
      for (int i = 0; i <= p; i++)
      {
         dof_map[2*dof3 + i + (j + (p + 1)*(p + 1))*(p + 1)] = o++;
      }

   // interior
   // x-components
   for (int k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 1; i <= p; i++)
         {
            dof_map[0*dof3 + i + (j + k*(p + 1))*(p + 2)] = o++;
         }
   // y-components
   for (int k = 0; k <= p; k++)
      for (int j = 1; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            dof_map[1*dof3 + i + (j + k*(p + 2))*(p + 1)] = o++;
         }
   // z-components
   for (int k = 1; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            dof_map[2*dof3 + i + (j + k*(p + 1))*(p + 1)] = o++;
         }

   // dof orientations
   // for odd p, do not change the orientations in the mid-planes
   // {i = p/2 + 1}, {j = p/2 + 1}, {k = p/2 + 1} in the x, y, z-components
   // respectively.
   // x-components
   for (int k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p/2; i++)
         {
            int idx = 0*dof3 + i + (j + k*(p + 1))*(p + 2);
            dof_map[idx] = -1 - dof_map[idx];
         }
   // y-components
   for (int k = 0; k <= p; k++)
      for (int j = 0; j <= p/2; j++)
         for (int i = 0; i <= p; i++)
         {
            int idx = 1*dof3 + i + (j + k*(p + 2))*(p + 1);
            dof_map[idx] = -1 - dof_map[idx];
         }
   // z-components
   for (int k = 0; k <= p/2; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            int idx = 2*dof3 + i + (j + k*(p + 1))*(p + 1);
            dof_map[idx] = -1 - dof_map[idx];
         }

   o = 0;
   // x-components
   for (int k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p + 1; i++)
         {
            int idx;
            if ((idx = dof_map[o++]) < 0)
            {
               idx = -1 - idx;
               dof2nk[idx] = 4;
            }
            else
            {
               dof2nk[idx] = 2;
            }
            Nodes.IntPoint(idx).Set3(cp[i], op[j], op[k]);
         }
   // y-components
   for (int k = 0; k <= p; k++)
      for (int j = 0; j <= p + 1; j++)
         for (int i = 0; i <= p; i++)
         {
            int idx;
            if ((idx = dof_map[o++]) < 0)
            {
               idx = -1 - idx;
               dof2nk[idx] = 1;
            }
            else
            {
               dof2nk[idx] = 3;
            }
            Nodes.IntPoint(idx).Set3(op[i], cp[j], op[k]);
         }
   // z-components
   for (int k = 0; k <= p + 1; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            int idx;
            if ((idx = dof_map[o++]) < 0)
            {
               idx = -1 - idx;
               dof2nk[idx] = 0;
            }
            else
            {
               dof2nk[idx] = 5;
            }
            Nodes.IntPoint(idx).Set3(op[i], op[j], cp[k]);
         }
}

void RT_HexahedronElement::CalcVShape(const IntegrationPoint &ip,
                                      DenseMatrix &shape) const
{
   const int pp1 = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_cx(pp1 + 1), shape_ox(pp1), shape_cy(pp1 + 1), shape_oy(pp1);
   Vector shape_cz(pp1 + 1), shape_oz(pp1);
#endif

   cbasis1d.Eval(ip.x, shape_cx);
   obasis1d.Eval(ip.x, shape_ox);
   cbasis1d.Eval(ip.y, shape_cy);
   obasis1d.Eval(ip.y, shape_oy);
   cbasis1d.Eval(ip.z, shape_cz);
   obasis1d.Eval(ip.z, shape_oz);

   int o = 0;
   // x-components
   for (int k = 0; k < pp1; k++)
      for (int j = 0; j < pp1; j++)
         for (int i = 0; i <= pp1; i++)
         {
            int idx, s;
            if ((idx = dof_map[o++]) < 0)
            {
               idx = -1 - idx, s = -1;
            }
            else
            {
               s = +1;
            }
            shape(idx,0) = s*shape_cx(i)*shape_oy(j)*shape_oz(k);
            shape(idx,1) = 0.;
            shape(idx,2) = 0.;
         }
   // y-components
   for (int k = 0; k < pp1; k++)
      for (int j = 0; j <= pp1; j++)
         for (int i = 0; i < pp1; i++)
         {
            int idx, s;
            if ((idx = dof_map[o++]) < 0)
            {
               idx = -1 - idx, s = -1;
            }
            else
            {
               s = +1;
            }
            shape(idx,0) = 0.;
            shape(idx,1) = s*shape_ox(i)*shape_cy(j)*shape_oz(k);
            shape(idx,2) = 0.;
         }
   // z-components
   for (int k = 0; k <= pp1; k++)
      for (int j = 0; j < pp1; j++)
         for (int i = 0; i < pp1; i++)
         {
            int idx, s;
            if ((idx = dof_map[o++]) < 0)
            {
               idx = -1 - idx, s = -1;
            }
            else
            {
               s = +1;
            }
            shape(idx,0) = 0.;
            shape(idx,1) = 0.;
            shape(idx,2) = s*shape_ox(i)*shape_oy(j)*shape_cz(k);
         }
}

void RT_HexahedronElement::CalcDivShape(const IntegrationPoint &ip,
                                        Vector &divshape) const
{
   const int pp1 = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_cx(pp1 + 1), shape_ox(pp1), shape_cy(pp1 + 1), shape_oy(pp1);
   Vector shape_cz(pp1 + 1), shape_oz(pp1);
   Vector dshape_cx(pp1 + 1), dshape_cy(pp1 + 1), dshape_cz(pp1 + 1);
#endif

   cbasis1d.Eval(ip.x, shape_cx, dshape_cx);
   obasis1d.Eval(ip.x, shape_ox);
   cbasis1d.Eval(ip.y, shape_cy, dshape_cy);
   obasis1d.Eval(ip.y, shape_oy);
   cbasis1d.Eval(ip.z, shape_cz, dshape_cz);
   obasis1d.Eval(ip.z, shape_oz);

   int o = 0;
   // x-components
   for (int k = 0; k < pp1; k++)
      for (int j = 0; j < pp1; j++)
         for (int i = 0; i <= pp1; i++)
         {
            int idx, s;
            if ((idx = dof_map[o++]) < 0)
            {
               idx = -1 - idx, s = -1;
            }
            else
            {
               s = +1;
            }
            divshape(idx) = s*dshape_cx(i)*shape_oy(j)*shape_oz(k);
         }
   // y-components
   for (int k = 0; k < pp1; k++)
      for (int j = 0; j <= pp1; j++)
         for (int i = 0; i < pp1; i++)
         {
            int idx, s;
            if ((idx = dof_map[o++]) < 0)
            {
               idx = -1 - idx, s = -1;
            }
            else
            {
               s = +1;
            }
            divshape(idx) = s*shape_ox(i)*dshape_cy(j)*shape_oz(k);
         }
   // z-components
   for (int k = 0; k <= pp1; k++)
      for (int j = 0; j < pp1; j++)
         for (int i = 0; i < pp1; i++)
         {
            int idx, s;
            if ((idx = dof_map[o++]) < 0)
            {
               idx = -1 - idx, s = -1;
            }
            else
            {
               s = +1;
            }
            divshape(idx) = s*shape_ox(i)*shape_oy(j)*dshape_cz(k);
         }
}


const double RT_TriangleElement::nk[6] =
{ 0., -1., 1., 1., -1., 0. };

const double RT_TriangleElement::c = 1./3.;

RT_TriangleElement::RT_TriangleElement(const int p)
   : VectorFiniteElement(2, Geometry::TRIANGLE, (p + 1)*(p + 3), p + 1,
                         H_DIV, FunctionSpace::Pk),
     dof2nk(Dof)
{
   const double *iop = (p > 0) ? poly1d.OpenPoints(p - 1) : NULL;
   const double *bop = poly1d.OpenPoints(p);

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p + 1);
   shape_y.SetSize(p + 1);
   shape_l.SetSize(p + 1);
   dshape_x.SetSize(p + 1);
   dshape_y.SetSize(p + 1);
   dshape_l.SetSize(p + 1);
   u.SetSize(Dof, Dim);
   divu.SetSize(Dof);
#else
   Vector shape_x(p + 1), shape_y(p + 1), shape_l(p + 1);
#endif

   // edges
   int o = 0;
   for (int i = 0; i <= p; i++)  // (0,1)
   {
      Nodes.IntPoint(o).Set2(bop[i], 0.);
      dof2nk[o++] = 0;
   }
   for (int i = 0; i <= p; i++)  // (1,2)
   {
      Nodes.IntPoint(o).Set2(bop[p-i], bop[i]);
      dof2nk[o++] = 1;
   }
   for (int i = 0; i <= p; i++)  // (2,0)
   {
      Nodes.IntPoint(o).Set2(0., bop[p-i]);
      dof2nk[o++] = 2;
   }

   // interior
   for (int j = 0; j < p; j++)
      for (int i = 0; i + j < p; i++)
      {
         double w = iop[i] + iop[j] + iop[p-1-i-j];
         Nodes.IntPoint(o).Set2(iop[i]/w, iop[j]/w);
         dof2nk[o++] = 0;
         Nodes.IntPoint(o).Set2(iop[i]/w, iop[j]/w);
         dof2nk[o++] = 2;
      }

   DenseMatrix T(Dof);
   for (int k = 0; k < Dof; k++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(k);
      poly1d.CalcBasis(p, ip.x, shape_x);
      poly1d.CalcBasis(p, ip.y, shape_y);
      poly1d.CalcBasis(p, 1. - ip.x - ip.y, shape_l);
      const double *n_k = nk + 2*dof2nk[k];

      o = 0;
      for (int j = 0; j <= p; j++)
         for (int i = 0; i + j <= p; i++)
         {
            double s = shape_x(i)*shape_y(j)*shape_l(p-i-j);
            T(o++, k) = s*n_k[0];
            T(o++, k) = s*n_k[1];
         }
      for (int i = 0; i <= p; i++)
      {
         double s = shape_x(i)*shape_y(p-i);
         T(o++, k) = s*((ip.x - c)*n_k[0] + (ip.y - c)*n_k[1]);
      }
   }

   Ti.Factor(T);
   // mfem::out << "RT_TriangleElement(" << p << ") : "; Ti.TestInversion();
}

void RT_TriangleElement::CalcVShape(const IntegrationPoint &ip,
                                    DenseMatrix &shape) const
{
   const int p = Order - 1;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p + 1), shape_y(p + 1), shape_l(p + 1);
   DenseMatrix u(Dof, Dim);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x);
   poly1d.CalcBasis(p, ip.y, shape_y);
   poly1d.CalcBasis(p, 1. - ip.x - ip.y, shape_l);

   int o = 0;
   for (int j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)
      {
         double s = shape_x(i)*shape_y(j)*shape_l(p-i-j);
         u(o,0) = s;  u(o,1) = 0;  o++;
         u(o,0) = 0;  u(o,1) = s;  o++;
      }
   for (int i = 0; i <= p; i++)
   {
      double s = shape_x(i)*shape_y(p-i);
      u(o,0) = (ip.x - c)*s;
      u(o,1) = (ip.y - c)*s;
      o++;
   }

   Ti.Mult(u, shape);
}

void RT_TriangleElement::CalcDivShape(const IntegrationPoint &ip,
                                      Vector &divshape) const
{
   const int p = Order - 1;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p + 1),  shape_y(p + 1),  shape_l(p + 1);
   Vector dshape_x(p + 1), dshape_y(p + 1), dshape_l(p + 1);
   Vector divu(Dof);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x, dshape_x);
   poly1d.CalcBasis(p, ip.y, shape_y, dshape_y);
   poly1d.CalcBasis(p, 1. - ip.x - ip.y, shape_l, dshape_l);

   int o = 0;
   for (int j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)
      {
         int k = p - i - j;
         divu(o++) = (dshape_x(i)*shape_l(k) -
                      shape_x(i)*dshape_l(k))*shape_y(j);
         divu(o++) = (dshape_y(j)*shape_l(k) -
                      shape_y(j)*dshape_l(k))*shape_x(i);
      }
   for (int i = 0; i <= p; i++)
   {
      int j = p - i;
      divu(o++) = ((shape_x(i) + (ip.x - c)*dshape_x(i))*shape_y(j) +
                   (shape_y(j) + (ip.y - c)*dshape_y(j))*shape_x(i));
   }

   Ti.Mult(divu, divshape);
}


const double RT_TetrahedronElement::nk[12] =
{ 1,1,1,  -1,0,0,  0,-1,0,  0,0,-1 };
// { .5,.5,.5, -.5,0,0, 0,-.5,0, 0,0,-.5}; // n_F |F|

const double RT_TetrahedronElement::c = 1./4.;

RT_TetrahedronElement::RT_TetrahedronElement(const int p)
   : VectorFiniteElement(3, Geometry::TETRAHEDRON, (p + 1)*(p + 2)*(p + 4)/2,
                         p + 1, H_DIV, FunctionSpace::Pk),
     dof2nk(Dof)
{
   const double *iop = (p > 0) ? poly1d.OpenPoints(p - 1) : NULL;
   const double *bop = poly1d.OpenPoints(p);

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p + 1);
   shape_y.SetSize(p + 1);
   shape_z.SetSize(p + 1);
   shape_l.SetSize(p + 1);
   dshape_x.SetSize(p + 1);
   dshape_y.SetSize(p + 1);
   dshape_z.SetSize(p + 1);
   dshape_l.SetSize(p + 1);
   u.SetSize(Dof, Dim);
   divu.SetSize(Dof);
#else
   Vector shape_x(p + 1), shape_y(p + 1), shape_z(p + 1), shape_l(p + 1);
#endif

   int o = 0;
   // faces (see Mesh::GenerateFaces in mesh/mesh.cpp,
   //        the constructor of H1_TetrahedronElement)
   for (int j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)  // (1,2,3)
      {
         double w = bop[i] + bop[j] + bop[p-i-j];
         Nodes.IntPoint(o).Set3(bop[p-i-j]/w, bop[i]/w, bop[j]/w);
         dof2nk[o++] = 0;
      }
   for (int j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)  // (0,3,2)
      {
         double w = bop[i] + bop[j] + bop[p-i-j];
         Nodes.IntPoint(o).Set3(0., bop[j]/w, bop[i]/w);
         dof2nk[o++] = 1;
      }
   for (int j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)  // (0,1,3)
      {
         double w = bop[i] + bop[j] + bop[p-i-j];
         Nodes.IntPoint(o).Set3(bop[i]/w, 0., bop[j]/w);
         dof2nk[o++] = 2;
      }
   for (int j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)  // (0,2,1)
      {
         double w = bop[i] + bop[j] + bop[p-i-j];
         Nodes.IntPoint(o).Set3(bop[j]/w, bop[i]/w, 0.);
         dof2nk[o++] = 3;
      }

   // interior
   for (int k = 0; k < p; k++)
      for (int j = 0; j + k < p; j++)
         for (int i = 0; i + j + k < p; i++)
         {
            double w = iop[i] + iop[j] + iop[k] + iop[p-1-i-j-k];
            Nodes.IntPoint(o).Set3(iop[i]/w, iop[j]/w, iop[k]/w);
            dof2nk[o++] = 1;
            Nodes.IntPoint(o).Set3(iop[i]/w, iop[j]/w, iop[k]/w);
            dof2nk[o++] = 2;
            Nodes.IntPoint(o).Set3(iop[i]/w, iop[j]/w, iop[k]/w);
            dof2nk[o++] = 3;
         }

   DenseMatrix T(Dof);
   for (int m = 0; m < Dof; m++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(m);
      poly1d.CalcBasis(p, ip.x, shape_x);
      poly1d.CalcBasis(p, ip.y, shape_y);
      poly1d.CalcBasis(p, ip.z, shape_z);
      poly1d.CalcBasis(p, 1. - ip.x - ip.y - ip.z, shape_l);
      const double *nm = nk + 3*dof2nk[m];

      o = 0;
      for (int k = 0; k <= p; k++)
         for (int j = 0; j + k <= p; j++)
            for (int i = 0; i + j + k <= p; i++)
            {
               double s = shape_x(i)*shape_y(j)*shape_z(k)*shape_l(p-i-j-k);
               T(o++, m) = s * nm[0];
               T(o++, m) = s * nm[1];
               T(o++, m) = s * nm[2];
            }
      for (int j = 0; j <= p; j++)
         for (int i = 0; i + j <= p; i++)
         {
            double s = shape_x(i)*shape_y(j)*shape_z(p-i-j);
            T(o++, m) = s*((ip.x - c)*nm[0] + (ip.y - c)*nm[1] +
                           (ip.z - c)*nm[2]);
         }
   }

   Ti.Factor(T);
   // mfem::out << "RT_TetrahedronElement(" << p << ") : "; Ti.TestInversion();
}

void RT_TetrahedronElement::CalcVShape(const IntegrationPoint &ip,
                                       DenseMatrix &shape) const
{
   const int p = Order - 1;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p + 1), shape_y(p + 1), shape_z(p + 1), shape_l(p + 1);
   DenseMatrix u(Dof, Dim);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x);
   poly1d.CalcBasis(p, ip.y, shape_y);
   poly1d.CalcBasis(p, ip.z, shape_z);
   poly1d.CalcBasis(p, 1. - ip.x - ip.y - ip.z, shape_l);

   int o = 0;
   for (int k = 0; k <= p; k++)
      for (int j = 0; j + k <= p; j++)
         for (int i = 0; i + j + k <= p; i++)
         {
            double s = shape_x(i)*shape_y(j)*shape_z(k)*shape_l(p-i-j-k);
            u(o,0) = s;  u(o,1) = 0;  u(o,2) = 0;  o++;
            u(o,0) = 0;  u(o,1) = s;  u(o,2) = 0;  o++;
            u(o,0) = 0;  u(o,1) = 0;  u(o,2) = s;  o++;
         }
   for (int j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)
      {
         double s = shape_x(i)*shape_y(j)*shape_z(p-i-j);
         u(o,0) = (ip.x - c)*s;  u(o,1) = (ip.y - c)*s;  u(o,2) = (ip.z - c)*s;
         o++;
      }

   Ti.Mult(u, shape);
}

void RT_TetrahedronElement::CalcDivShape(const IntegrationPoint &ip,
                                         Vector &divshape) const
{
   const int p = Order - 1;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p + 1),  shape_y(p + 1),  shape_z(p + 1),  shape_l(p + 1);
   Vector dshape_x(p + 1), dshape_y(p + 1), dshape_z(p + 1), dshape_l(p + 1);
   Vector divu(Dof);
#endif

   poly1d.CalcBasis(p, ip.x, shape_x, dshape_x);
   poly1d.CalcBasis(p, ip.y, shape_y, dshape_y);
   poly1d.CalcBasis(p, ip.z, shape_z, dshape_z);
   poly1d.CalcBasis(p, 1. - ip.x - ip.y - ip.z, shape_l, dshape_l);

   int o = 0;
   for (int k = 0; k <= p; k++)
      for (int j = 0; j + k <= p; j++)
         for (int i = 0; i + j + k <= p; i++)
         {
            int l = p - i - j - k;
            divu(o++) = (dshape_x(i)*shape_l(l) -
                         shape_x(i)*dshape_l(l))*shape_y(j)*shape_z(k);
            divu(o++) = (dshape_y(j)*shape_l(l) -
                         shape_y(j)*dshape_l(l))*shape_x(i)*shape_z(k);
            divu(o++) = (dshape_z(k)*shape_l(l) -
                         shape_z(k)*dshape_l(l))*shape_x(i)*shape_y(j);
         }
   for (int j = 0; j <= p; j++)
      for (int i = 0; i + j <= p; i++)
      {
         int k = p - i - j;
         divu(o++) =
            (shape_x(i) + (ip.x - c)*dshape_x(i))*shape_y(j)*shape_z(k) +
            (shape_y(j) + (ip.y - c)*dshape_y(j))*shape_x(i)*shape_z(k) +
            (shape_z(k) + (ip.z - c)*dshape_z(k))*shape_x(i)*shape_y(j);
      }

   Ti.Mult(divu, divshape);
}


const double ND_HexahedronElement::tk[18] =
{ 1.,0.,0.,  0.,1.,0.,  0.,0.,1., -1.,0.,0.,  0.,-1.,0.,  0.,0.,-1. };

ND_HexahedronElement::ND_HexahedronElement(const int p,
                                           const int cb_type, const int ob_type)
   : VectorFiniteElement(3, Geometry::CUBE, 3*p*(p + 1)*(p + 1), p,
                         H_CURL, FunctionSpace::Qk),
     cbasis1d(poly1d.GetBasis(p, VerifyClosed(cb_type))),
     obasis1d(poly1d.GetBasis(p - 1, VerifyOpen(ob_type))),
     dof_map(Dof), dof2tk(Dof)
{
   const double *cp = poly1d.ClosedPoints(p, cb_type);
   const double *op = poly1d.OpenPoints(p - 1, ob_type);
   const int dof3 = Dof/3;

#ifndef MFEM_THREAD_SAFE
   shape_cx.SetSize(p + 1);
   shape_ox.SetSize(p);
   shape_cy.SetSize(p + 1);
   shape_oy.SetSize(p);
   shape_cz.SetSize(p + 1);
   shape_oz.SetSize(p);
   dshape_cx.SetSize(p + 1);
   dshape_cy.SetSize(p + 1);
   dshape_cz.SetSize(p + 1);
#endif

   // edges
   int o = 0;
   for (int i = 0; i < p; i++)  // (0,1)
   {
      dof_map[0*dof3 + i + (0 + 0*(p + 1))*p] = o++;
   }
   for (int i = 0; i < p; i++)  // (1,2)
   {
      dof_map[1*dof3 + p + (i + 0*p)*(p + 1)] = o++;
   }
   for (int i = 0; i < p; i++)  // (3,2)
   {
      dof_map[0*dof3 + i + (p + 0*(p + 1))*p] = o++;
   }
   for (int i = 0; i < p; i++)  // (0,3)
   {
      dof_map[1*dof3 + 0 + (i + 0*p)*(p + 1)] = o++;
   }
   for (int i = 0; i < p; i++)  // (4,5)
   {
      dof_map[0*dof3 + i + (0 + p*(p + 1))*p] = o++;
   }
   for (int i = 0; i < p; i++)  // (5,6)
   {
      dof_map[1*dof3 + p + (i + p*p)*(p + 1)] = o++;
   }
   for (int i = 0; i < p; i++)  // (7,6)
   {
      dof_map[0*dof3 + i + (p + p*(p + 1))*p] = o++;
   }
   for (int i = 0; i < p; i++)  // (4,7)
   {
      dof_map[1*dof3 + 0 + (i + p*p)*(p + 1)] = o++;
   }
   for (int i = 0; i < p; i++)  // (0,4)
   {
      dof_map[2*dof3 + 0 + (0 + i*(p + 1))*(p + 1)] = o++;
   }
   for (int i = 0; i < p; i++)  // (1,5)
   {
      dof_map[2*dof3 + p + (0 + i*(p + 1))*(p + 1)] = o++;
   }
   for (int i = 0; i < p; i++)  // (2,6)
   {
      dof_map[2*dof3 + p + (p + i*(p + 1))*(p + 1)] = o++;
   }
   for (int i = 0; i < p; i++)  // (3,7)
   {
      dof_map[2*dof3 + 0 + (p + i*(p + 1))*(p + 1)] = o++;
   }

   // faces
   // (3,2,1,0) -- bottom
   for (int j = 1; j < p; j++) // x - components
      for (int i = 0; i < p; i++)
      {
         dof_map[0*dof3 + i + ((p - j) + 0*(p + 1))*p] = o++;
      }
   for (int j = 0; j < p; j++) // y - components
      for (int i = 1; i < p; i++)
      {
         dof_map[1*dof3 + i + ((p - 1 - j) + 0*p)*(p + 1)] = -1 - (o++);
      }
   // (0,1,5,4) -- front
   for (int k = 1; k < p; k++) // x - components
      for (int i = 0; i < p; i++)
      {
         dof_map[0*dof3 + i + (0 + k*(p + 1))*p] = o++;
      }
   for (int k = 0; k < p; k++) // z - components
      for (int i = 1; i < p; i++ )
      {
         dof_map[2*dof3 + i + (0 + k*(p + 1))*(p + 1)] = o++;
      }
   // (1,2,6,5) -- right
   for (int k = 1; k < p; k++) // y - components
      for (int j = 0; j < p; j++)
      {
         dof_map[1*dof3 + p + (j + k*p)*(p + 1)] = o++;
      }
   for (int k = 0; k < p; k++) // z - components
      for (int j = 1; j < p; j++)
      {
         dof_map[2*dof3 + p + (j + k*(p + 1))*(p + 1)] = o++;
      }
   // (2,3,7,6) -- back
   for (int k = 1; k < p; k++) // x - components
      for (int i = 0; i < p; i++)
      {
         dof_map[0*dof3 + (p - 1 - i) + (p + k*(p + 1))*p] = -1 - (o++);
      }
   for (int k = 0; k < p; k++) // z - components
      for (int i = 1; i < p; i++)
      {
         dof_map[2*dof3 + (p - i) + (p + k*(p + 1))*(p + 1)] = o++;
      }
   // (3,0,4,7) -- left
   for (int k = 1; k < p; k++) // y - components
      for (int j = 0; j < p; j++)
      {
         dof_map[1*dof3 + 0 + ((p - 1 - j) + k*p)*(p + 1)] = -1 - (o++);
      }
   for (int k = 0; k < p; k++) // z - components
      for (int j = 1; j < p; j++)
      {
         dof_map[2*dof3 + 0 + ((p - j) + k*(p + 1))*(p + 1)] = o++;
      }
   // (4,5,6,7) -- top
   for (int j = 1; j < p; j++) // x - components
      for (int i = 0; i < p; i++)
      {
         dof_map[0*dof3 + i + (j + p*(p + 1))*p] = o++;
      }
   for (int j = 0; j < p; j++) // y - components
      for (int i = 1; i < p; i++)
      {
         dof_map[1*dof3 + i + (j + p*p)*(p + 1)] = o++;
      }

   // interior
   // x-components
   for (int k = 1; k < p; k++)
      for (int j = 1; j < p; j++)
         for (int i = 0; i < p; i++)
         {
            dof_map[0*dof3 + i + (j + k*(p + 1))*p] = o++;
         }
   // y-components
   for (int k = 1; k < p; k++)
      for (int j = 0; j < p; j++)
         for (int i = 1; i < p; i++)
         {
            dof_map[1*dof3 + i + (j + k*p)*(p + 1)] = o++;
         }
   // z-components
   for (int k = 0; k < p; k++)
      for (int j = 1; j < p; j++)
         for (int i = 1; i < p; i++)
         {
            dof_map[2*dof3 + i + (j + k*(p + 1))*(p + 1)] = o++;
         }

   // set dof2tk and Nodes
   o = 0;
   // x-components
   for (int k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i < p; i++)
         {
            int idx;
            if ((idx = dof_map[o++]) < 0)
            {
               dof2tk[idx = -1 - idx] = 3;
            }
            else
            {
               dof2tk[idx] = 0;
            }
            Nodes.IntPoint(idx).Set3(op[i], cp[j], cp[k]);
         }
   // y-components
   for (int k = 0; k <= p; k++)
      for (int j = 0; j < p; j++)
         for (int i = 0; i <= p; i++)
         {
            int idx;
            if ((idx = dof_map[o++]) < 0)
            {
               dof2tk[idx = -1 - idx] = 4;
            }
            else
            {
               dof2tk[idx] = 1;
            }
            Nodes.IntPoint(idx).Set3(cp[i], op[j], cp[k]);
         }
   // z-components
   for (int k = 0; k < p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            int idx;
            if ((idx = dof_map[o++]) < 0)
            {
               dof2tk[idx = -1 - idx] = 5;
            }
            else
            {
               dof2tk[idx] = 2;
            }
            Nodes.IntPoint(idx).Set3(cp[i], cp[j], op[k]);
         }
}

void ND_HexahedronElement::CalcVShape(const IntegrationPoint &ip,
                                      DenseMatrix &shape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_cx(p + 1), shape_ox(p), shape_cy(p + 1), shape_oy(p);
   Vector shape_cz(p + 1), shape_oz(p);
#endif

   cbasis1d.Eval(ip.x, shape_cx);
   obasis1d.Eval(ip.x, shape_ox);
   cbasis1d.Eval(ip.y, shape_cy);
   obasis1d.Eval(ip.y, shape_oy);
   cbasis1d.Eval(ip.z, shape_cz);
   obasis1d.Eval(ip.z, shape_oz);

   int o = 0;
   // x-components
   for (int k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i < p; i++)
         {
            int idx, s;
            if ((idx = dof_map[o++]) < 0)
            {
               idx = -1 - idx, s = -1;
            }
            else
            {
               s = +1;
            }
            shape(idx,0) = s*shape_ox(i)*shape_cy(j)*shape_cz(k);
            shape(idx,1) = 0.;
            shape(idx,2) = 0.;
         }
   // y-components
   for (int k = 0; k <= p; k++)
      for (int j = 0; j < p; j++)
         for (int i = 0; i <= p; i++)
         {
            int idx, s;
            if ((idx = dof_map[o++]) < 0)
            {
               idx = -1 - idx, s = -1;
            }
            else
            {
               s = +1;
            }
            shape(idx,0) = 0.;
            shape(idx,1) = s*shape_cx(i)*shape_oy(j)*shape_cz(k);
            shape(idx,2) = 0.;
         }
   // z-components
   for (int k = 0; k < p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            int idx, s;
            if ((idx = dof_map[o++]) < 0)
            {
               idx = -1 - idx, s = -1;
            }
            else
            {
               s = +1;
            }
            shape(idx,0) = 0.;
            shape(idx,1) = 0.;
            shape(idx,2) = s*shape_cx(i)*shape_cy(j)*shape_oz(k);
         }
}

void ND_HexahedronElement::CalcCurlShape(const IntegrationPoint &ip,
                                         DenseMatrix &curl_shape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_cx(p + 1), shape_ox(p), shape_cy(p + 1), shape_oy(p);
   Vector shape_cz(p + 1), shape_oz(p);
   Vector dshape_cx(p + 1), dshape_cy(p + 1), dshape_cz(p + 1);
#endif

   cbasis1d.Eval(ip.x, shape_cx, dshape_cx);
   obasis1d.Eval(ip.x, shape_ox);
   cbasis1d.Eval(ip.y, shape_cy, dshape_cy);
   obasis1d.Eval(ip.y, shape_oy);
   cbasis1d.Eval(ip.z, shape_cz, dshape_cz);
   obasis1d.Eval(ip.z, shape_oz);

   int o = 0;
   // x-components
   for (int k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i < p; i++)
         {
            int idx, s;
            if ((idx = dof_map[o++]) < 0)
            {
               idx = -1 - idx, s = -1;
            }
            else
            {
               s = +1;
            }
            curl_shape(idx,0) = 0.;
            curl_shape(idx,1) =  s*shape_ox(i)* shape_cy(j)*dshape_cz(k);
            curl_shape(idx,2) = -s*shape_ox(i)*dshape_cy(j)* shape_cz(k);
         }
   // y-components
   for (int k = 0; k <= p; k++)
      for (int j = 0; j < p; j++)
         for (int i = 0; i <= p; i++)
         {
            int idx, s;
            if ((idx = dof_map[o++]) < 0)
            {
               idx = -1 - idx, s = -1;
            }
            else
            {
               s = +1;
            }
            curl_shape(idx,0) = -s* shape_cx(i)*shape_oy(j)*dshape_cz(k);
            curl_shape(idx,1) = 0.;
            curl_shape(idx,2) =  s*dshape_cx(i)*shape_oy(j)* shape_cz(k);
         }
   // z-components
   for (int k = 0; k < p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            int idx, s;
            if ((idx = dof_map[o++]) < 0)
            {
               idx = -1 - idx, s = -1;
            }
            else
            {
               s = +1;
            }
            curl_shape(idx,0) =   s* shape_cx(i)*dshape_cy(j)*shape_oz(k);
            curl_shape(idx,1) =  -s*dshape_cx(i)* shape_cy(j)*shape_oz(k);
            curl_shape(idx,2) = 0.;
         }
}


const double ND_QuadrilateralElement::tk[8] =
{ 1.,0.,  0.,1., -1.,0., 0.,-1. };

ND_QuadrilateralElement::ND_QuadrilateralElement(const int p,
                                                 const int cb_type,
                                                 const int ob_type)
   : VectorFiniteElement(2, Geometry::SQUARE, 2*p*(p + 1), p,
                         H_CURL, FunctionSpace::Qk),
     cbasis1d(poly1d.GetBasis(p, VerifyClosed(cb_type))),
     obasis1d(poly1d.GetBasis(p - 1, VerifyOpen(ob_type))),
     dof_map(Dof), dof2tk(Dof)
{
   const double *cp = poly1d.ClosedPoints(p, cb_type);
   const double *op = poly1d.OpenPoints(p - 1, ob_type);
   const int dof2 = Dof/2;

#ifndef MFEM_THREAD_SAFE
   shape_cx.SetSize(p + 1);
   shape_ox.SetSize(p);
   shape_cy.SetSize(p + 1);
   shape_oy.SetSize(p);
   dshape_cx.SetSize(p + 1);
   dshape_cy.SetSize(p + 1);
#endif

   // edges
   int o = 0;
   for (int i = 0; i < p; i++)  // (0,1)
   {
      dof_map[0*dof2 + i + 0*p] = o++;
   }
   for (int j = 0; j < p; j++)  // (1,2)
   {
      dof_map[1*dof2 + p + j*(p + 1)] = o++;
   }
   for (int i = 0; i < p; i++)  // (2,3)
   {
      dof_map[0*dof2 + (p - 1 - i) + p*p] = -1 - (o++);
   }
   for (int j = 0; j < p; j++)  // (3,0)
   {
      dof_map[1*dof2 + 0 + (p - 1 - j)*(p + 1)] = -1 - (o++);
   }

   // interior
   // x-components
   for (int j = 1; j < p; j++)
      for (int i = 0; i < p; i++)
      {
         dof_map[0*dof2 + i + j*p] = o++;
      }
   // y-components
   for (int j = 0; j < p; j++)
      for (int i = 1; i < p; i++)
      {
         dof_map[1*dof2 + i + j*(p + 1)] = o++;
      }

   // set dof2tk and Nodes
   o = 0;
   // x-components
   for (int j = 0; j <= p; j++)
      for (int i = 0; i < p; i++)
      {
         int idx;
         if ((idx = dof_map[o++]) < 0)
         {
            dof2tk[idx = -1 - idx] = 2;
         }
         else
         {
            dof2tk[idx] = 0;
         }
         Nodes.IntPoint(idx).Set2(op[i], cp[j]);
      }
   // y-components
   for (int j = 0; j < p; j++)
      for (int i = 0; i <= p; i++)
      {
         int idx;
         if ((idx = dof_map[o++]) < 0)
         {
            dof2tk[idx = -1 - idx] = 3;
         }
         else
         {
            dof2tk[idx] = 1;
         }
         Nodes.IntPoint(idx).Set2(cp[i], op[j]);
      }
}

void ND_QuadrilateralElement::CalcVShape(const IntegrationPoint &ip,
                                         DenseMatrix &shape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_cx(p + 1), shape_ox(p), shape_cy(p + 1), shape_oy(p);
#endif

   cbasis1d.Eval(ip.x, shape_cx);
   obasis1d.Eval(ip.x, shape_ox);
   cbasis1d.Eval(ip.y, shape_cy);
   obasis1d.Eval(ip.y, shape_oy);

   int o = 0;
   // x-components
   for (int j = 0; j <= p; j++)
      for (int i = 0; i < p; i++)
      {
         int idx, s;
         if ((idx = dof_map[o++]) < 0)
         {
            idx = -1 - idx, s = -1;
         }
         else
         {
            s = +1;
         }
         shape(idx,0) = s*shape_ox(i)*shape_cy(j);
         shape(idx,1) = 0.;
      }
   // y-components
   for (int j = 0; j < p; j++)
      for (int i = 0; i <= p; i++)
      {
         int idx, s;
         if ((idx = dof_map[o++]) < 0)
         {
            idx = -1 - idx, s = -1;
         }
         else
         {
            s = +1;
         }
         shape(idx,0) = 0.;
         shape(idx,1) = s*shape_cx(i)*shape_oy(j);
      }
}

void ND_QuadrilateralElement::CalcCurlShape(const IntegrationPoint &ip,
                                            DenseMatrix &curl_shape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_cx(p + 1), shape_ox(p), shape_cy(p + 1), shape_oy(p);
   Vector dshape_cx(p + 1), dshape_cy(p + 1);
#endif

   cbasis1d.Eval(ip.x, shape_cx, dshape_cx);
   obasis1d.Eval(ip.x, shape_ox);
   cbasis1d.Eval(ip.y, shape_cy, dshape_cy);
   obasis1d.Eval(ip.y, shape_oy);

   int o = 0;
   // x-components
   for (int j = 0; j <= p; j++)
      for (int i = 0; i < p; i++)
      {
         int idx, s;
         if ((idx = dof_map[o++]) < 0)
         {
            idx = -1 - idx, s = -1;
         }
         else
         {
            s = +1;
         }
         curl_shape(idx,0) = -s*shape_ox(i)*dshape_cy(j);
      }
   // y-components
   for (int j = 0; j < p; j++)
      for (int i = 0; i <= p; i++)
      {
         int idx, s;
         if ((idx = dof_map[o++]) < 0)
         {
            idx = -1 - idx, s = -1;
         }
         else
         {
            s = +1;
         }
         curl_shape(idx,0) =  s*dshape_cx(i)*shape_oy(j);
      }
}


const double ND_TetrahedronElement::tk[18] =
{ 1.,0.,0.,  0.,1.,0.,  0.,0.,1.,  -1.,1.,0.,  -1.,0.,1.,  0.,-1.,1. };

const double ND_TetrahedronElement::c = 1./4.;

ND_TetrahedronElement::ND_TetrahedronElement(const int p)
   : VectorFiniteElement(3, Geometry::TETRAHEDRON, p*(p + 2)*(p + 3)/2, p,
                         H_CURL, FunctionSpace::Pk), dof2tk(Dof)
{
   const double *eop = poly1d.OpenPoints(p - 1);
   const double *fop = (p > 1) ? poly1d.OpenPoints(p - 2) : NULL;
   const double *iop = (p > 2) ? poly1d.OpenPoints(p - 3) : NULL;

   const int pm1 = p - 1, pm2 = p - 2, pm3 = p - 3;

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p);
   shape_y.SetSize(p);
   shape_z.SetSize(p);
   shape_l.SetSize(p);
   dshape_x.SetSize(p);
   dshape_y.SetSize(p);
   dshape_z.SetSize(p);
   dshape_l.SetSize(p);
   u.SetSize(Dof, Dim);
#else
   Vector shape_x(p), shape_y(p), shape_z(p), shape_l(p);
#endif

   int o = 0;
   // edges
   for (int i = 0; i < p; i++) // (0,1)
   {
      Nodes.IntPoint(o).Set3(eop[i], 0., 0.);
      dof2tk[o++] = 0;
   }
   for (int i = 0; i < p; i++) // (0,2)
   {
      Nodes.IntPoint(o).Set3(0., eop[i], 0.);
      dof2tk[o++] = 1;
   }
   for (int i = 0; i < p; i++) // (0,3)
   {
      Nodes.IntPoint(o).Set3(0., 0., eop[i]);
      dof2tk[o++] = 2;
   }
   for (int i = 0; i < p; i++) // (1,2)
   {
      Nodes.IntPoint(o).Set3(eop[pm1-i], eop[i], 0.);
      dof2tk[o++] = 3;
   }
   for (int i = 0; i < p; i++) // (1,3)
   {
      Nodes.IntPoint(o).Set3(eop[pm1-i], 0., eop[i]);
      dof2tk[o++] = 4;
   }
   for (int i = 0; i < p; i++) // (2,3)
   {
      Nodes.IntPoint(o).Set3(0., eop[pm1-i], eop[i]);
      dof2tk[o++] = 5;
   }

   // faces
   for (int j = 0; j <= pm2; j++)  // (1,2,3)
      for (int i = 0; i + j <= pm2; i++)
      {
         double w = fop[i] + fop[j] + fop[pm2-i-j];
         Nodes.IntPoint(o).Set3(fop[pm2-i-j]/w, fop[i]/w, fop[j]/w);
         dof2tk[o++] = 3;
         Nodes.IntPoint(o).Set3(fop[pm2-i-j]/w, fop[i]/w, fop[j]/w);
         dof2tk[o++] = 4;
      }
   for (int j = 0; j <= pm2; j++)  // (0,3,2)
      for (int i = 0; i + j <= pm2; i++)
      {
         double w = fop[i] + fop[j] + fop[pm2-i-j];
         Nodes.IntPoint(o).Set3(0., fop[j]/w, fop[i]/w);
         dof2tk[o++] = 2;
         Nodes.IntPoint(o).Set3(0., fop[j]/w, fop[i]/w);
         dof2tk[o++] = 1;
      }
   for (int j = 0; j <= pm2; j++)  // (0,1,3)
      for (int i = 0; i + j <= pm2; i++)
      {
         double w = fop[i] + fop[j] + fop[pm2-i-j];
         Nodes.IntPoint(o).Set3(fop[i]/w, 0., fop[j]/w);
         dof2tk[o++] = 0;
         Nodes.IntPoint(o).Set3(fop[i]/w, 0., fop[j]/w);
         dof2tk[o++] = 2;
      }
   for (int j = 0; j <= pm2; j++)  // (0,2,1)
      for (int i = 0; i + j <= pm2; i++)
      {
         double w = fop[i] + fop[j] + fop[pm2-i-j];
         Nodes.IntPoint(o).Set3(fop[j]/w, fop[i]/w, 0.);
         dof2tk[o++] = 1;
         Nodes.IntPoint(o).Set3(fop[j]/w, fop[i]/w, 0.);
         dof2tk[o++] = 0;
      }

   // interior
   for (int k = 0; k <= pm3; k++)
      for (int j = 0; j + k <= pm3; j++)
         for (int i = 0; i + j + k <= pm3; i++)
         {
            double w = iop[i] + iop[j] + iop[k] + iop[pm3-i-j-k];
            Nodes.IntPoint(o).Set3(iop[i]/w, iop[j]/w, iop[k]/w);
            dof2tk[o++] = 0;
            Nodes.IntPoint(o).Set3(iop[i]/w, iop[j]/w, iop[k]/w);
            dof2tk[o++] = 1;
            Nodes.IntPoint(o).Set3(iop[i]/w, iop[j]/w, iop[k]/w);
            dof2tk[o++] = 2;
         }

   DenseMatrix T(Dof);
   for (int m = 0; m < Dof; m++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(m);
      const double *tm = tk + 3*dof2tk[m];
      o = 0;

      poly1d.CalcBasis(pm1, ip.x, shape_x);
      poly1d.CalcBasis(pm1, ip.y, shape_y);
      poly1d.CalcBasis(pm1, ip.z, shape_z);
      poly1d.CalcBasis(pm1, 1. - ip.x - ip.y - ip.z, shape_l);

      for (int k = 0; k <= pm1; k++)
         for (int j = 0; j + k <= pm1; j++)
            for (int i = 0; i + j + k <= pm1; i++)
            {
               double s = shape_x(i)*shape_y(j)*shape_z(k)*shape_l(pm1-i-j-k);
               T(o++, m) = s * tm[0];
               T(o++, m) = s * tm[1];
               T(o++, m) = s * tm[2];
            }
      for (int k = 0; k <= pm1; k++)
         for (int j = 0; j + k <= pm1; j++)
         {
            double s = shape_x(pm1-j-k)*shape_y(j)*shape_z(k);
            T(o++, m) = s*((ip.y - c)*tm[0] - (ip.x - c)*tm[1]);
            T(o++, m) = s*((ip.z - c)*tm[0] - (ip.x - c)*tm[2]);
         }
      for (int k = 0; k <= pm1; k++)
      {
         T(o++, m) =
            shape_y(pm1-k)*shape_z(k)*((ip.z - c)*tm[1] - (ip.y - c)*tm[2]);
      }
   }

   Ti.Factor(T);
   // mfem::out << "ND_TetrahedronElement(" << p << ") : "; Ti.TestInversion();
}

void ND_TetrahedronElement::CalcVShape(const IntegrationPoint &ip,
                                       DenseMatrix &shape) const
{
   const int pm1 = Order - 1;

#ifdef MFEM_THREAD_SAFE
   const int p = Order;
   Vector shape_x(p), shape_y(p), shape_z(p), shape_l(p);
   DenseMatrix u(Dof, Dim);
#endif

   poly1d.CalcBasis(pm1, ip.x, shape_x);
   poly1d.CalcBasis(pm1, ip.y, shape_y);
   poly1d.CalcBasis(pm1, ip.z, shape_z);
   poly1d.CalcBasis(pm1, 1. - ip.x - ip.y - ip.z, shape_l);

   int n = 0;
   for (int k = 0; k <= pm1; k++)
      for (int j = 0; j + k <= pm1; j++)
         for (int i = 0; i + j + k <= pm1; i++)
         {
            double s = shape_x(i)*shape_y(j)*shape_z(k)*shape_l(pm1-i-j-k);
            u(n,0) =  s;  u(n,1) = 0.;  u(n,2) = 0.;  n++;
            u(n,0) = 0.;  u(n,1) =  s;  u(n,2) = 0.;  n++;
            u(n,0) = 0.;  u(n,1) = 0.;  u(n,2) =  s;  n++;
         }
   for (int k = 0; k <= pm1; k++)
      for (int j = 0; j + k <= pm1; j++)
      {
         double s = shape_x(pm1-j-k)*shape_y(j)*shape_z(k);
         u(n,0) = s*(ip.y - c);  u(n,1) = -s*(ip.x - c);  u(n,2) =  0.;  n++;
         u(n,0) = s*(ip.z - c);  u(n,1) =  0.;  u(n,2) = -s*(ip.x - c);  n++;
      }
   for (int k = 0; k <= pm1; k++)
   {
      double s = shape_y(pm1-k)*shape_z(k);
      u(n,0) = 0.;  u(n,1) = s*(ip.z - c);  u(n,2) = -s*(ip.y - c);  n++;
   }

   Ti.Mult(u, shape);
}

void ND_TetrahedronElement::CalcCurlShape(const IntegrationPoint &ip,
                                          DenseMatrix &curl_shape) const
{
   const int pm1 = Order - 1;

#ifdef MFEM_THREAD_SAFE
   const int p = Order;
   Vector shape_x(p), shape_y(p), shape_z(p), shape_l(p);
   Vector dshape_x(p), dshape_y(p), dshape_z(p), dshape_l(p);
   DenseMatrix u(Dof, Dim);
#endif

   poly1d.CalcBasis(pm1, ip.x, shape_x, dshape_x);
   poly1d.CalcBasis(pm1, ip.y, shape_y, dshape_y);
   poly1d.CalcBasis(pm1, ip.z, shape_z, dshape_z);
   poly1d.CalcBasis(pm1, 1. - ip.x - ip.y - ip.z, shape_l, dshape_l);

   int n = 0;
   for (int k = 0; k <= pm1; k++)
      for (int j = 0; j + k <= pm1; j++)
         for (int i = 0; i + j + k <= pm1; i++)
         {
            int l = pm1-i-j-k;
            const double dx = (dshape_x(i)*shape_l(l) -
                               shape_x(i)*dshape_l(l))*shape_y(j)*shape_z(k);
            const double dy = (dshape_y(j)*shape_l(l) -
                               shape_y(j)*dshape_l(l))*shape_x(i)*shape_z(k);
            const double dz = (dshape_z(k)*shape_l(l) -
                               shape_z(k)*dshape_l(l))*shape_x(i)*shape_y(j);

            u(n,0) =  0.;  u(n,1) =  dz;  u(n,2) = -dy;  n++;
            u(n,0) = -dz;  u(n,1) =  0.;  u(n,2) =  dx;  n++;
            u(n,0) =  dy;  u(n,1) = -dx;  u(n,2) =  0.;  n++;
         }
   for (int k = 0; k <= pm1; k++)
      for (int j = 0; j + k <= pm1; j++)
      {
         int i = pm1 - j - k;
         // s = shape_x(i)*shape_y(j)*shape_z(k);
         // curl of s*(ip.y - c, -(ip.x - c), 0):
         u(n,0) =  shape_x(i)*(ip.x - c)*shape_y(j)*dshape_z(k);
         u(n,1) =  shape_x(i)*shape_y(j)*(ip.y - c)*dshape_z(k);
         u(n,2) =
            -((dshape_x(i)*(ip.x - c) + shape_x(i))*shape_y(j)*shape_z(k) +
              (dshape_y(j)*(ip.y - c) + shape_y(j))*shape_x(i)*shape_z(k));
         n++;
         // curl of s*(ip.z - c, 0, -(ip.x - c)):
         u(n,0) = -shape_x(i)*(ip.x - c)*dshape_y(j)*shape_z(k);
         u(n,1) = (shape_x(i)*shape_y(j)*(dshape_z(k)*(ip.z - c) + shape_z(k)) +
                   (dshape_x(i)*(ip.x - c) + shape_x(i))*shape_y(j)*shape_z(k));
         u(n,2) = -shape_x(i)*dshape_y(j)*shape_z(k)*(ip.z - c);
         n++;
      }
   for (int k = 0; k <= pm1; k++)
   {
      int j = pm1 - k;
      // curl of shape_y(j)*shape_z(k)*(0, ip.z - c, -(ip.y - c)):
      u(n,0) = -((dshape_y(j)*(ip.y - c) + shape_y(j))*shape_z(k) +
                 shape_y(j)*(dshape_z(k)*(ip.z - c) + shape_z(k)));
      u(n,1) = 0.;
      u(n,2) = 0.;  n++;
   }

   Ti.Mult(u, curl_shape);
}


const double ND_TriangleElement::tk[8] =
{ 1.,0.,  -1.,1.,  0.,-1.,  0.,1. };

const double ND_TriangleElement::c = 1./3.;

ND_TriangleElement::ND_TriangleElement(const int p)
   : VectorFiniteElement(2, Geometry::TRIANGLE, p*(p + 2), p,
                         H_CURL, FunctionSpace::Pk),
     dof2tk(Dof)
{
   const double *eop = poly1d.OpenPoints(p - 1);
   const double *iop = (p > 1) ? poly1d.OpenPoints(p - 2) : NULL;

   const int pm1 = p - 1, pm2 = p - 2;

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p);
   shape_y.SetSize(p);
   shape_l.SetSize(p);
   dshape_x.SetSize(p);
   dshape_y.SetSize(p);
   dshape_l.SetSize(p);
   u.SetSize(Dof, Dim);
   curlu.SetSize(Dof);
#else
   Vector shape_x(p), shape_y(p), shape_l(p);
#endif

   int n = 0;
   // edges
   for (int i = 0; i < p; i++) // (0,1)
   {
      Nodes.IntPoint(n).Set2(eop[i], 0.);
      dof2tk[n++] = 0;
   }
   for (int i = 0; i < p; i++) // (1,2)
   {
      Nodes.IntPoint(n).Set2(eop[pm1-i], eop[i]);
      dof2tk[n++] = 1;
   }
   for (int i = 0; i < p; i++) // (2,0)
   {
      Nodes.IntPoint(n).Set2(0., eop[pm1-i]);
      dof2tk[n++] = 2;
   }

   // interior
   for (int j = 0; j <= pm2; j++)
      for (int i = 0; i + j <= pm2; i++)
      {
         double w = iop[i] + iop[j] + iop[pm2-i-j];
         Nodes.IntPoint(n).Set2(iop[i]/w, iop[j]/w);
         dof2tk[n++] = 0;
         Nodes.IntPoint(n).Set2(iop[i]/w, iop[j]/w);
         dof2tk[n++] = 3;
      }

   DenseMatrix T(Dof);
   for (int m = 0; m < Dof; m++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(m);
      const double *tm = tk + 2*dof2tk[m];
      n = 0;

      poly1d.CalcBasis(pm1, ip.x, shape_x);
      poly1d.CalcBasis(pm1, ip.y, shape_y);
      poly1d.CalcBasis(pm1, 1. - ip.x - ip.y, shape_l);

      for (int j = 0; j <= pm1; j++)
         for (int i = 0; i + j <= pm1; i++)
         {
            double s = shape_x(i)*shape_y(j)*shape_l(pm1-i-j);
            T(n++, m) = s * tm[0];
            T(n++, m) = s * tm[1];
         }
      for (int j = 0; j <= pm1; j++)
      {
         T(n++, m) =
            shape_x(pm1-j)*shape_y(j)*((ip.y - c)*tm[0] - (ip.x - c)*tm[1]);
      }
   }

   Ti.Factor(T);
   // mfem::out << "ND_TriangleElement(" << p << ") : "; Ti.TestInversion();
}

void ND_TriangleElement::CalcVShape(const IntegrationPoint &ip,
                                    DenseMatrix &shape) const
{
   const int pm1 = Order - 1;

#ifdef MFEM_THREAD_SAFE
   const int p = Order;
   Vector shape_x(p), shape_y(p), shape_l(p);
   DenseMatrix u(Dof, Dim);
#endif

   poly1d.CalcBasis(pm1, ip.x, shape_x);
   poly1d.CalcBasis(pm1, ip.y, shape_y);
   poly1d.CalcBasis(pm1, 1. - ip.x - ip.y, shape_l);

   int n = 0;
   for (int j = 0; j <= pm1; j++)
      for (int i = 0; i + j <= pm1; i++)
      {
         double s = shape_x(i)*shape_y(j)*shape_l(pm1-i-j);
         u(n,0) = s;  u(n,1) = 0;  n++;
         u(n,0) = 0;  u(n,1) = s;  n++;
      }
   for (int j = 0; j <= pm1; j++)
   {
      double s = shape_x(pm1-j)*shape_y(j);
      u(n,0) =  s*(ip.y - c);
      u(n,1) = -s*(ip.x - c);
      n++;
   }

   Ti.Mult(u, shape);
}

void ND_TriangleElement::CalcCurlShape(const IntegrationPoint &ip,
                                       DenseMatrix &curl_shape) const
{
   const int pm1 = Order - 1;

#ifdef MFEM_THREAD_SAFE
   const int p = Order;
   Vector shape_x(p), shape_y(p), shape_l(p);
   Vector dshape_x(p), dshape_y(p), dshape_l(p);
   Vector curlu(Dof);
#endif

   poly1d.CalcBasis(pm1, ip.x, shape_x, dshape_x);
   poly1d.CalcBasis(pm1, ip.y, shape_y, dshape_y);
   poly1d.CalcBasis(pm1, 1. - ip.x - ip.y, shape_l, dshape_l);

   int n = 0;
   for (int j = 0; j <= pm1; j++)
      for (int i = 0; i + j <= pm1; i++)
      {
         int l = pm1-i-j;
         const double dx = (dshape_x(i)*shape_l(l) -
                            shape_x(i)*dshape_l(l)) * shape_y(j);
         const double dy = (dshape_y(j)*shape_l(l) -
                            shape_y(j)*dshape_l(l)) * shape_x(i);

         curlu(n++) = -dy;
         curlu(n++) =  dx;
      }

   for (int j = 0; j <= pm1; j++)
   {
      int i = pm1 - j;
      // curl of shape_x(i)*shape_y(j) * (ip.y - c, -(ip.x - c), 0):
      curlu(n++) = -((dshape_x(i)*(ip.x - c) + shape_x(i)) * shape_y(j) +
                     (dshape_y(j)*(ip.y - c) + shape_y(j)) * shape_x(i));
   }

   Vector curl2d(curl_shape.Data(),Dof);
   Ti.Mult(curlu, curl2d);
}


const double ND_SegmentElement::tk[1] = { 1. };

ND_SegmentElement::ND_SegmentElement(const int p, const int ob_type)
   : VectorFiniteElement(1, Geometry::SEGMENT, p, p - 1,
                         H_CURL, FunctionSpace::Pk),
     obasis1d(poly1d.GetBasis(p - 1, VerifyOpen(ob_type))),
     dof2tk(Dof)
{
   const double *op = poly1d.OpenPoints(p - 1, ob_type);

   // set dof2tk and Nodes
   for (int i = 0; i < p; i++)
   {
      dof2tk[i] = 0;
      Nodes.IntPoint(i).x = op[i];
   }
}

void ND_SegmentElement::CalcVShape(const IntegrationPoint &ip,
                                   DenseMatrix &shape) const
{
   Vector vshape(shape.Data(), Dof);

   obasis1d.Eval(ip.x, vshape);
}

void NURBS1DFiniteElement::SetOrder() const
{
   Order = kv[0]->GetOrder();
   Dof = Order + 1;

   weights.SetSize(Dof);
   shape_x.SetSize(Dof);
}

void NURBS1DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                     Vector &shape) const
{
   kv[0]->CalcShape(shape, ijk[0], ip.x);

   double sum = 0.0;
   for (int i = 0; i <= Order; i++)
   {
      sum += (shape(i) *= weights(i));
   }

   shape /= sum;
}

void NURBS1DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                      DenseMatrix &dshape) const
{
   Vector grad(dshape.Data(), Dof);

   kv[0]->CalcShape (shape_x, ijk[0], ip.x);
   kv[0]->CalcDShape(grad,    ijk[0], ip.x);

   double sum = 0.0, dsum = 0.0;
   for (int i = 0; i <= Order; i++)
   {
      sum  += (shape_x(i) *= weights(i));
      dsum += (   grad(i) *= weights(i));
   }

   sum = 1.0/sum;
   add(sum, grad, -dsum*sum*sum, shape_x, grad);
}

void NURBS2DFiniteElement::SetOrder() const
{
   Orders[0] = kv[0]->GetOrder();
   Orders[1] = kv[1]->GetOrder();
   shape_x.SetSize(Orders[0]+1);
   shape_y.SetSize(Orders[1]+1);
   dshape_x.SetSize(Orders[0]+1);
   dshape_y.SetSize(Orders[1]+1);

   Order = max(Orders[0], Orders[1]);
   Dof = (Orders[0] + 1)*(Orders[1] + 1);
   u.SetSize(Dof);
   weights.SetSize(Dof);
}

void NURBS2DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                     Vector &shape) const
{
   kv[0]->CalcShape(shape_x, ijk[0], ip.x);
   kv[1]->CalcShape(shape_y, ijk[1], ip.y);

   double sum = 0.0;
   for (int o = 0, j = 0; j <= Orders[1]; j++)
   {
      const double sy = shape_y(j);
      for (int i = 0; i <= Orders[0]; i++, o++)
      {
         sum += ( shape(o) = shape_x(i)*sy*weights(o) );
      }
   }

   shape /= sum;
}

void NURBS2DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                      DenseMatrix &dshape) const
{
   double sum, dsum[2];

   kv[0]->CalcShape ( shape_x, ijk[0], ip.x);
   kv[1]->CalcShape ( shape_y, ijk[1], ip.y);

   kv[0]->CalcDShape(dshape_x, ijk[0], ip.x);
   kv[1]->CalcDShape(dshape_y, ijk[1], ip.y);

   sum = dsum[0] = dsum[1] = 0.0;
   for (int o = 0, j = 0; j <= Orders[1]; j++)
   {
      const double sy = shape_y(j), dsy = dshape_y(j);
      for (int i = 0; i <= Orders[0]; i++, o++)
      {
         sum += ( u(o) = shape_x(i)*sy*weights(o) );

         dsum[0] += ( dshape(o,0) = dshape_x(i)*sy *weights(o) );
         dsum[1] += ( dshape(o,1) =  shape_x(i)*dsy*weights(o) );
      }
   }

   sum = 1.0/sum;
   dsum[0] *= sum*sum;
   dsum[1] *= sum*sum;

   for (int o = 0; o < Dof; o++)
   {
      dshape(o,0) = dshape(o,0)*sum - u(o)*dsum[0];
      dshape(o,1) = dshape(o,1)*sum - u(o)*dsum[1];
   }
}

//---------------------------------------------------------------------
void NURBS3DFiniteElement::SetOrder() const
{
   Orders[0] = kv[0]->GetOrder();
   Orders[1] = kv[1]->GetOrder();
   Orders[2] = kv[2]->GetOrder();
   shape_x.SetSize(Orders[0]+1);
   shape_y.SetSize(Orders[1]+1);
   shape_z.SetSize(Orders[2]+1);

   dshape_x.SetSize(Orders[0]+1);
   dshape_y.SetSize(Orders[1]+1);
   dshape_z.SetSize(Orders[2]+1);

   Order = max(max(Orders[0], Orders[1]), Orders[2]);
   Dof = (Orders[0] + 1)*(Orders[1] + 1)*(Orders[2] + 1);
   u.SetSize(Dof);
   weights.SetSize(Dof);
}

void NURBS3DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                     Vector &shape) const
{
   kv[0]->CalcShape(shape_x, ijk[0], ip.x);
   kv[1]->CalcShape(shape_y, ijk[1], ip.y);
   kv[2]->CalcShape(shape_z, ijk[2], ip.z);

   double sum = 0.0;
   for (int o = 0, k = 0; k <= Orders[2]; k++)
   {
      const double sz = shape_z(k);
      for (int j = 0; j <= Orders[1]; j++)
      {
         const double sy_sz = shape_y(j)*sz;
         for (int i = 0; i <= Orders[0]; i++, o++)
         {
            sum += ( shape(o) = shape_x(i)*sy_sz*weights(o) );
         }
      }
   }

   shape /= sum;
}

void NURBS3DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                      DenseMatrix &dshape) const
{
   double sum, dsum[3];

   kv[0]->CalcShape ( shape_x, ijk[0], ip.x);
   kv[1]->CalcShape ( shape_y, ijk[1], ip.y);
   kv[2]->CalcShape ( shape_z, ijk[2], ip.z);

   kv[0]->CalcDShape(dshape_x, ijk[0], ip.x);
   kv[1]->CalcDShape(dshape_y, ijk[1], ip.y);
   kv[2]->CalcDShape(dshape_z, ijk[2], ip.z);

   sum = dsum[0] = dsum[1] = dsum[2] = 0.0;
   for (int o = 0, k = 0; k <= Orders[2]; k++)
   {
      const double sz = shape_z(k), dsz = dshape_z(k);
      for (int j = 0; j <= Orders[1]; j++)
      {
         const double  sy_sz  =  shape_y(j)* sz;
         const double dsy_sz  = dshape_y(j)* sz;
         const double  sy_dsz =  shape_y(j)*dsz;
         for (int i = 0; i <= Orders[0]; i++, o++)
         {
            sum += ( u(o) = shape_x(i)*sy_sz*weights(o) );

            dsum[0] += ( dshape(o,0) = dshape_x(i)* sy_sz *weights(o) );
            dsum[1] += ( dshape(o,1) =  shape_x(i)*dsy_sz *weights(o) );
            dsum[2] += ( dshape(o,2) =  shape_x(i)* sy_dsz*weights(o) );
         }
      }
   }

   sum = 1.0/sum;
   dsum[0] *= sum*sum;
   dsum[1] *= sum*sum;
   dsum[2] *= sum*sum;

   for (int o = 0; o < Dof; o++)
   {
      dshape(o,0) = dshape(o,0)*sum - u(o)*dsum[0];
      dshape(o,1) = dshape(o,1)*sum - u(o)*dsum[1];
      dshape(o,2) = dshape(o,2)*sum - u(o)*dsum[2];
   }
}

/// SBP_SegmentElement is a segment element with nodes at Gauss Lobatto
/// points with ordering consistent with SBP_TriangleElement's edges.

//////////////////////////////////////////////////////////////////////////
/// Not currently implemented as collocated SBP type element
//////////////////////////////////////////////////////////////////////////
SBP_SegmentElement::SBP_SegmentElement(const int p)
   : NodalTensorFiniteElement(1, p+1, BasisType::GaussLobatto, H1_DOF_MAP)
{
   const double *cp = poly1d.ClosedPoints(p+1, b_type);

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p+2);
   dshape_x.SetSize(p+2);
#endif

   Nodes.IntPoint(0).x = cp[0];
   Nodes.IntPoint(1).x = cp[p+1];

   switch (p)
   {
      case 1:
         Nodes.IntPoint(2).x = cp[1];
         break;
      case 2:
         Nodes.IntPoint(2).x = cp[1];
         Nodes.IntPoint(3).x = cp[2];
         break;
      case 3:
         Nodes.IntPoint(2).x = cp[2];
         Nodes.IntPoint(3).x = cp[1];
         Nodes.IntPoint(4).x = cp[3];
         break;
      case 4:
         Nodes.IntPoint(2).x = cp[2];
         Nodes.IntPoint(3).x = cp[3];
         Nodes.IntPoint(4).x = cp[1];
         Nodes.IntPoint(5).x = cp[4];
         break;
   }
}

void SBP_SegmentElement::CalcShape(const IntegrationPoint &ip,
                                  Vector &shape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+2);
#endif

   basis1d.Eval(ip.x, shape_x);

   shape(0) = shape_x(0);
   shape(1) = shape_x(p+1);

   switch (p)
   {
      case 1:
         shape(2) = shape_x(1);
         break;
      case 2:
         shape(2) = shape_x(1);
         shape(3) = shape_x(2);
         break;
      case 3:
         shape(2) = shape_x(2);
         shape(3) = shape_x(1);
         shape(4) = shape_x(3);
         break;
      case 4:
         shape(2) = shape_x(2);
         shape(3) = shape_x(3);
         shape(4) = shape_x(1);
         shape(5) = shape_x(4);
         break;
   }
}

void SBP_SegmentElement::CalcDShape(const IntegrationPoint &ip,
                                   DenseMatrix &dshape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+2), dshape_x(p+2);
#endif

   basis1d.Eval(ip.x, shape_x, dshape_x);

   dshape(0,0) = dshape_x(0);
   dshape(1,0) = dshape_x(p+1);

   switch (p)
   {
      case 1:
         dshape(2,0) = dshape_x(1);
         break;
      case 2:
         dshape(2,0) = dshape_x(1);
         dshape(3,0) = dshape_x(2);
         break;
      case 3:
         dshape(2,0) = dshape_x(2);
         dshape(3,0) = dshape_x(1);
         dshape(4,0) = dshape_x(3);
         break;
      case 4:
         dshape(2,0) = dshape_x(2);
         dshape(3,0) = dshape_x(3);
         dshape(4,0) = dshape_x(1);
         dshape(5,0) = dshape_x(4);
         break;
   }
}

// Leftover function from H1_Segment element
// void SBP_SegmentElement::ProjectDelta(int vertex, Vector &dofs) const
// {
//    const int p = Order;
//    const double *cp = poly1d.ClosedPoints(p, b_type);

//    switch (vertex)
//    {
//       case 0:
//          dofs(0) = poly1d.CalcDelta(p, (1.0 - cp[0]));
//          dofs(1) = poly1d.CalcDelta(p, (1.0 - cp[p]));
//          for (int i = 1; i < p; i++)
//          {
//             dofs(i+1) = poly1d.CalcDelta(p, (1.0 - cp[i]));
//          }
//          break;

//       case 1:
//          dofs(0) = poly1d.CalcDelta(p, cp[0]);
//          dofs(1) = poly1d.CalcDelta(p, cp[p]);
//          for (int i = 1; i < p; i++)
//          {
//             dofs(i+1) = poly1d.CalcDelta(p, cp[i]);
//          }
//          break;
//    }
// }

SBP_TriangleElement::SBP_TriangleElement(const int p, const int Do)
   : NodalFiniteElement(2, Geometry::TRIANGLE, Do, p,
                        FunctionSpace::Pk)
{
   // Set operator type to SBP to be used in IntegrationRules::Get()
   OperatorType = SBP;

   // Create Dx and Dy matrixes
   Dx = new DenseMatrix(Dof);
   Dy = new DenseMatrix(Dof);

   // Data to be stored in Dx and Dy matrices depending upon order

   // There is probably a better way to input these and constuct Dx and Dy,
   // I did it this way because it was simple and fast to get the code working.
   const double p0Dx[9] = {-0.9999999999999984,-1.000000000000001,-0.9999999999999988,
         1.000000000000001,0.9999999999999974,0.9999999999999994,
         1.3322676295501878e-15,-1.9984014443252818e-15,9.992007221626409e-16};
   const double p0Dy[9] = {-0.9999999999999974,-0.9999999999999994,-1.0000000000000009,
         1.9984014443252818e-15,-9.992007221626409e-16,-1.7486012637846216e-15,
         1.0000000000000009,0.9999999999999991,0.9999999999999984};
   const double p1Dx[49] = {-3.333333333333333, 0.21647921352995003, 0.10823960676497299, -0.8824293926518367, 0.7863909022744312, -0.051249687578105226, -0.362809297675581,
-0.21647921352995003, 3.333333333333333, -0.10823960676497392, 0.8824293926518368, 0.05124968757810527, -0.7863909022744306, 0.3628092976755813,
-0.10823960676497299, 0.10823960676497392, 0.0, 7.406681257404114e-16, -0.8311797050737315, 0.8311797050737307, -1.9769834191462048e-16,
2.3531450470715645, -2.353145047071565, -1.9751150019744302e-15, 0.0, -0.2351412146963257, 0.23514121469632399, 3.1540140465066577e-17,
-2.09704240606515, -0.13666583354161405, 2.2164792135299507, 0.2351412146963257, 5.0, 0.47028242939265, 1.2743814046488378,
0.13666583354161393, 2.0970424060651482, -2.2164792135299485, -0.23514121469632399, -0.47028242939265, -5.0, -1.2743814046488378,
3.2652836790802304, -3.2652836790802326, 1.7792850772315852e-15, -1.0644797406959973e-16, -4.301037240689829, 4.301037240689829, 0.0};
   const double p1Dy[49] = {-3.333333333333333, 0.10823960676497733, 0.21647921352994814, -0.0512496875781027, 0.7863909022744318, -0.8824293926518388, -0.3628092976755811,
-0.10823960676497733, 0.0, 0.10823960676497837, 0.8311797050737281, -0.8311797050737287, 9.745633233426465e-17, -3.4329014409980615e-16,
-0.21647921352994814, -0.10823960676497837, 3.333333333333333, -0.7863909022744319, 0.051249687578102346, 0.8824293926518396, 0.3628092976755813,
0.1366658335416072, -2.216479213529942, 2.0970424060651514, -5.0, -0.4702824293926414, -0.23514121469633495, -1.274381404648838,
-2.0970424060651514, 2.216479213529943, -0.13666583354160625, 0.4702824293926414, 5.0, 0.235141214696336, 1.2743814046488384,
2.3531450470715702, -2.5988355289137243e-16, -2.3531450470715725, 0.23514121469633495, -0.235141214696336, 0.0, 3.7848168558079887e-16,
3.2652836790802313, 3.0896112968982564e-15, -3.2652836790802326, 4.30103724068983, -4.301037240689832, -1.2773756888351966e-15, 0.0};
   
   const double p2Dx[144] = {-6.654819485608504, -0.2644800033235228, -0.13224000166175998, -1.4588736121976598, 0.16568196946098376, -0.18364010800322192, -0.49629408766969996, 0.18676496181247967, -0.32850861828832895, 0.17877251381228318, -0.18297783134208934, 0.16855922332582518,
0.2644800033235228, 6.654819485608504, 0.13224000166176486, -0.1656819694609847, 1.4588736121976598, 0.3285086182883251, -0.18676496181248323, 0.4962940876696997, 0.18364010800322153, -0.1787725138122836, -0.16855922332582482, 0.18297783134208834,
0.13224000166175998, -0.13224000166176486, 0.0, -0.31265397966647923, 0.3126539796664765, -0.021082992351498762, -1.1303649939093317, 1.1303649939093305, 0.0210829923514981, -3.1850260779507576e-16, 0.010213290486458549, -0.010213290486458991,
6.25007885298757, 0.7098115731146227, 1.3394662911697857, 0.0, -1.0987526276656132, 0.8391459038761706, -0.7493297423228898, -1.0608454694676046, 0.7982718911706455, 0.47711018331982885, -0.20709875867432972, -0.4372738402113571,
-0.7098115731146187, -6.25007885298757, -1.339466291169774, 1.0987526276656132, 0.0, -0.7982718911706481, 1.0608454694676077, 0.7493297423228901, -0.8391459038761684, -0.47711018331982796, 0.4372738402113582, 0.20709875867432867,
0.7867474920341369, -1.407390435347625, 0.09032335875573284, -0.8391459038761706, 0.7982718911706481, 7.766734444360106, 0.549376313832808, -1.8999913733437686, 1.4986594846457786, 0.7173852412251129, -0.7042577654543215, 1.9425023749691277,
2.126213783203917, 0.8001349318703544, 4.842688417639932, 0.7493297423228898, -1.0608454694676077, -0.549376313832808, 7.766734444360106, 1.5965437823412956, -1.8999913733437699, 1.7354036162948008, -0.2271475821344933, 0.2801114010137556,
-0.8001349318703392, -2.1262137832039163, -4.842688417639927, 1.0608454694676046, -0.7493297423228901, 1.8999913733437686, -1.5965437823412956, -7.766734444360106, 0.5493763138328077, -1.7354036162947999, -0.28011140101375603, 0.22714758213449346,
1.4073904353476412, -0.7867474920341352, -0.09032335875573, -0.7982718911706455, 0.8391459038761684, -1.4986594846457786, 1.8999913733437699, -0.5493763138328077, -7.766734444360106, -0.7173852412251128, -1.9425023749691277, 0.7042577654543228,
-3.0482350464695736, 3.0482350464695807, 5.4307610872012924e-15, -1.8988845609884182, 1.8988845609884146, -2.8551722567827746, -6.906855584460779, 6.906855584460776, 2.855172256782774, 0.0, 1.0883892687387082, -1.088389268738702,
3.11993956078517, 2.8740890922806273, -0.1741459541889496, 0.8242469961764219, -1.7403370817278376, 2.8029252875555524, 0.904040726567137, 1.114835175054942, 7.73110258063719, -1.0883892687387082, 0.0, -2.176778537477416,
-2.8740890922806335, -3.119939560785153, 0.17414595418895715, 1.7403370817278334, -0.8242469961764177, -7.73110258063719, -1.1148351750549401, -0.9040407265671375, -2.8029252875555577, 1.088389268738702, 2.176778537477416, 0.0};

   const double p2Dy[144] = {-6.654819485608504, -0.13224000166176303, -0.26448000332352145, -0.3285086182883249, 0.1867649618124839, -0.49629408766970096, -0.18364010800322322, 0.16568196946098196, -1.4588736121976569, 0.16855922332582438, -0.1829778313420883, 0.17877251381228437,
0.13224000166176303, 0.0, -0.13224000166176358, 0.021082992351497333, 1.1303649939093325, -1.13036499390933, -0.0210829923514987, 0.31265397966647596, -0.3126539796664774, -0.010213290486459194, 0.010213290486458717, 1.7550143694830702e-16,
0.26448000332352145, 0.13224000166176358, 6.654819485608504, 0.18364010800321912, 0.49629408766969646, -0.18676496181248112, 0.32850861828832506, 1.4588736121976555, -0.16568196946098368, 0.18297783134208803, -0.16855922332582496, -0.17877251381228357,
1.4073904353476239, -0.09032335875572672, -0.7867474920341248, -7.766734444360106, -0.5493763138328086, 1.8999913733437714, -1.498659484645771, 0.8391459038761694, -0.7982718911706419, 0.7042577654543205, -1.9425023749691286, -0.7173852412251152,
-0.8001349318703573, -4.842688417639936, -2.1262137832039025, 0.5493763138328086, -7.766734444360106, -1.5965437823412931, 1.8999913733437763, -0.7493297423228881, 1.060845469467605, 0.22714758213449282, -0.28011140101375637, -1.7354036162948017,
2.1262137832039216, 4.842688417639926, 0.8001349318703455, -1.8999913733437714, 1.5965437823412931, 7.766734444360106, -0.5493763138328039, -1.0608454694675977, 0.7493297423228871, 0.280111401013755, -0.22714758213449193, 1.7354036162947997,
0.7867474920341424, 0.09032335875573258, -1.4073904353476248, 1.498659484645771, -1.8999913733437763, 0.5493763138328039, 7.766734444360106, 0.7982718911706495, -0.8391459038761694, 1.942502374969129, -0.7042577654543208, 0.7173852412251135,
-0.709811573114611, -1.3394662911697717, -6.250078852987552, -0.8391459038761694, 0.7493297423228881, 1.0608454694675977, -0.7982718911706495, 0.0, 1.0987526276656179, 0.20709875867432867, 0.43727384021135907, -0.47711018331982974,
6.250078852987558, 1.339466291169778, 0.7098115731146183, 0.7982718911706419, -1.060845469467605, -0.7493297423228871, 0.8391459038761694, -1.0987526276656179, 0.0, -0.4372738402113571, -0.20709875867432725, 0.4771101833198306,
-2.8740890922806197, 0.17414595418896062, -3.1199395607851477, -2.8029252875555484, -0.904040726567135, -1.114835175054938, -7.731102580637195, -0.8242469961764177, 1.7403370817278334, 0.0, 2.1767785374774147, 1.0883892687387102,
3.119939560785152, -0.17414595418895248, 2.8740890922806295, 7.731102580637194, 1.1148351750549435, 0.9040407265671315, 2.8029252875555497, -1.7403370817278412, 0.8242469961764122, -2.1767785374774147, 0.0, -1.0883892687387073,
-3.048235046469594, -2.9924601909068347e-15, 3.04823504646958, 2.8551722567827835, 6.906855584460783, -6.906855584460775, -2.855172256782777, 1.8988845609884217, -1.8988845609884253, -1.0883892687387102, 1.0883892687387073, 0.0};

   const double p3Dx[324] = {-10.952585131486583, -0.1691685246371182, -0.08458426231856121, 0.2782043329268357, -2.305479077267717, -0.07465619884488185, 0.08251879096720546, 0.22849304912079002, 0.5302407771816927, 0.28327860677318784, -0.11951297527437506, -0.5003055437372067, 0.12448476066856992, -0.09239160861978224, 0.10362923976480765, -0.16947318256573993, -0.12955658105892584, -0.0720914264099959,
0.1691685246371182, 10.952585131486583, 0.08458426231855752, -0.2782043329268405, 0.0746561988448885, 2.3054790772677194, -0.2832786067731876, 0.5003055437372037, 0.11951297527437463, -0.08251879096721035, -0.5302407771816938, -0.2284930491207896, -0.12448476066856932, -0.10362923976480676, 0.09239160861978266, 0.16947318256574107, 0.07209142640999573, 0.12955658105892595,
0.08458426231856121, -0.08458426231855752, 0.0, 1.1683416656308464e-15, 0.3017477280608998, -0.3017477280609013, -0.005074273846349578, 0.04485677642949094, -1.8051735335305155, 0.0050742738463512614, 1.805173533530519, -0.04485677642949147, 4.205982075638532e-16, 0.020855520903763268, -0.020855520903761818, 7.395777588076851e-16, -0.0973817561557449, 0.09738175615574554,
-1.8896690623176595, 1.8896690623176922, -7.935818527815355e-15, 0.0, 2.0050512471695137, -2.0050512471695145, -0.8926105190374908, 1.6696076717446249, 0.020736328416563414, 0.8926105190375015, -0.020736328416544252, -1.669607671744625, -4.429107086712402e-16, 0.4633133883374836, -0.46331338833748054, 1.783666940029547e-15, 0.7834980551346994, -0.783498055134701,
11.456452196936608, -0.3709837064689789, -1.4994533917700335, -1.4668732071255282, 0.0, 0.43808614602067847, -0.03454227374421423, -0.36820417663301963, 0.6765153661826107, -1.2393619133905875, 1.2485385227460366, 1.2932580655616601, -0.24307113935772914, 0.052425055137118155, 0.44269118420721065, -0.48976278487794334, -0.3768721550070895, 0.17743932374896715,
0.3709837064689459, -11.45645219693662, 1.4994533917700408, 1.4668732071255288, -0.43808614602067847, 0.0, 1.2393619133905824, -1.2932580655616563, -1.2485385227460208, 0.034542273744204106, -0.6765153661826184, 0.36820417663302096, 0.24307113935772948, -0.44269118420720915, -0.052425055137118384, 0.4897627848779442, -0.1774393237489731, 0.3768721550070885,
-0.5604988416610844, 1.924135449667926, 0.03446638735025072, 0.8926105190374908, 0.04721541624352583, -1.6940688111740314, 11.466536000042892, -1.1244968786658345, 0.8805543685036823, 1.785221038074984, -3.363676482918665, 0.026479087826965113, -0.42363907421033664, 2.460011157605557, -0.8869524625478119, 1.3603474091530658, 0.13068371284607608, 0.5768493540183615,
-1.135434157870106, -2.486132536271146, -0.22290356912810128, -1.2214664156593342, 0.36820417663301963, 1.2932580655616563, 0.8226694181207354, 12.000000000000005, -0.21904307301033732, 0.01937180635024931, 1.6167426993790468, -1.3530307323652335, -0.6132774672199158, 0.43432201350751304, -0.11093602189671813, -0.6100947628964166, 0.7813784140227625, 2.7998077074825853,
-2.634887549640154, -0.5938872755970562, 8.97031966066547, -0.01517046739396432, -0.6765153661826107, 1.2485385227460208, -0.6442037890047951, 0.21904307301033732, 12.000000000000005, -2.460828329049918, 2.586516131123318, 1.6167426993790477, -0.0585109667596005, 0.19125087414978273, -0.17058628301270465, 2.4229355524754963, 0.29161562914482175, -0.43265543914744503,
-1.9241354496679275, 0.5604988416611175, -0.034466387350262154, -0.8926105190375015, 1.6940688111740383, -0.047215416243512, -1.785221038074984, -0.026479087826961595, 3.3636764829186583, -11.466536000042892, -0.8805543685036843, 1.1244968786658245, 0.42363907421033115, 0.8869524625478146, -2.4600111576055554, -1.3603474091530607, -0.5768493540183651, -0.13068371284607755,
0.5938872755970585, 2.634887549640159, -8.97031966066549, 0.0151704673939503, -1.2485385227460366, 0.6765153661826184, 2.460828329049923, -1.6167426993790468, -2.586516131123318, 0.6442037890047966, -12.000000000000005, -0.21904307301033812, 0.05851096675959981, 0.17058628301270434, -0.19125087414978267, -2.422935552475496, 0.43265543914744814, -0.29161562914482114,
2.486132536271161, 1.1354341578701037, 0.22290356912810394, 1.2214664156593344, -1.2932580655616601, -0.36820417663302096, -0.019371806350251885, 1.3530307323652335, -1.6167426993790477, -0.822669418120728, 0.21904307301033812, -12.000000000000005, 0.613277467219916, 0.11093602189671813, -0.43432201350751237, 0.6100947628964153, -2.7998077074825867, -0.7813784140227636,
-4.072657651361357, 4.072657651361338, -1.3760339008438063e-14, 2.133320141015979e-15, 1.6003174861462979, -1.6003174861463, 2.0404965421712293, 4.037660156795314, 0.38522106525739636, -2.0404965421712027, -0.38522106525739186, -4.037660156795315, 0.0, -1.4579314260766258, 1.4579314260766234, 2.0161871961592337e-15, 3.273923136675281, -3.2739231366752755,
3.0226944225627097, 3.3903460469074083, -0.6823116044539523, -2.2315915230588144, -0.34515299788282433, 2.914564209151067, -11.84886986676902, -2.859463754812241, -1.2591462686659354, -4.272088065230016, -1.1230959476442317, -0.7303740631402172, 1.4579314260766258, 0.0, 2.9158628521532504, 1.6703256795818853, -0.3883097004508776, -1.603597457093401,
-3.390346046907437, -3.022694422562723, 0.6823116044539049, 2.2315915230587997, -2.914564209151077, 0.34515299788282583, 4.272088065230004, 0.7303740631402172, 1.1230959476442337, 11.84886986676901, 1.2591462686659352, 2.8594637548122366, -1.4579314260766234, -2.9158628521532504, 0.0, -1.6703256795818826, 1.6035974570934062, 0.3883097004508795,
3.825058001824472, -3.8250580018244977, -1.669248067139698e-14, -5.926918282616925e-15, 2.2245074239726854, -2.2245074239726894, -4.520276599333233, 2.771056460992968, -11.004997298094699, 4.520276599333215, 11.004997298094697, -2.7710564609929618, -1.3909328887646931e-15, -1.1523289737701476, 1.1523289737701459, 0.0, -0.9410755641839665, 0.9410755641839751,
2.9241289363066816, -1.6271240279890726, 2.1979339738354153, -2.6034731278339045, 1.7117570640055604, 0.8059311674378296, -0.4342468145397358, -3.5490285023571575, -1.3245210783844836, 1.9168034714993238, -1.9651252935551593, 12.716754362100263, -2.258623293890646, 0.26788818737287284, -1.1062943201205055, 0.9410755641839665, 0.0, 1.8821511283679513,
1.6271240279890764, -2.9241289363066834, -2.1979339738354295, 2.60347312783391, -0.8059311674378026, -1.711757064005556, -1.9168034714993114, -12.716754362100257, 1.9651252935551453, 0.4342468145397406, 1.324521078384481, 3.5490285023571624, 2.258623293890642, 1.106294320120502, -0.2678881873728741, -0.9410755641839751, -1.8821511283679513, 0.0};

   const double p3Dy[324] = {-10.952585131486583, -0.08458426231856092, -0.16916852463712426, 0.28327860677318756, -0.5003055437372038, -0.11951297527437327, 0.0825187909672079, 0.5302407771816892, 0.22849304912079044, 0.2782043329268377, -0.07465619884488649, -2.305479077267721, 0.1036292397648074, -0.09239160861978217, 0.12448476066856967, -0.07209142640999439, -0.12955658105892628, -0.16947318256574065,
0.08458426231856092, 0.0, -0.08458426231857115, 0.00507427384635043, -0.044856776429489494, 1.8051735335305168, -0.005074273846348987, -1.805173533530519, 0.04485677642949065, -2.536797215167323e-16, -0.30174772806090094, 0.3017477280609018, -0.020855520903761887, 0.020855520903762893, 3.895409306444457e-16, 0.09738175615574617, -0.09738175615574511, -2.518495690291951e-17,
0.16916852463712426, 0.08458426231857115, 10.952585131486583, -0.08251879096720664, -0.2284930491207953, -0.5302407771816934, -0.2832786067731896, 0.11951297527436967, 0.5003055437372073, -0.2782043329268386, 2.305479077267721, 0.07465619884488589, 0.09239160861978252, -0.10362923976480695, -0.12448476066856902, 0.12955658105892542, 0.07209142640999575, 0.1694731825657405,
-1.9241354496679255, -0.034466387350256505, 0.5604988416610924, -11.466536000042892, 1.124496878665824, -0.8805543685036857, -1.7852210380749862, 3.3636764829186614, -0.026479087826939977, -0.8926105190374863, -0.04721541624350879, 1.6940688111740283, -2.4600111576055577, 0.8869524625478143, 0.42363907421033403, -0.13068371284608143, -0.5768493540183609, -1.3603474091530663,
2.4861325362711466, 0.2229035691280941, 1.1354341578701321, -0.8226694181207276, -12.000000000000005, 0.21904307301034107, -0.01937180635025535, -1.6167426993790432, 1.3530307323652313, 1.2214664156593333, -0.3682041766330094, -1.2932580655616668, -0.4343220135075129, 0.11093602189671664, 0.6132774672199158, -0.7813784140227632, -2.799807707482588, 0.6100947628964168,
0.5938872755970495, -8.97031966066548, 2.634887549640157, 0.6442037890047976, -0.21904307301034107, -12.000000000000005, 2.4608283290499164, -2.586516131123313, -1.6167426993790395, 0.015170467393960869, 0.6765153661826211, -1.2485385227460257, -0.19125087414978525, 0.17058628301270676, 0.058510966759598895, -0.2916156291448204, 0.43265543914744725, -2.4229355524754985,
-0.560498841661101, 0.03446638735024671, 1.9241354496679397, 1.7852210380749862, 0.026479087826969852, -3.363676482918656, 11.466536000042892, 0.8805543685036855, -1.1244968786658387, 0.8926105190374908, -1.6940688111740407, 0.047215416243521366, -0.8869524625478177, 2.460011157605558, -0.4236390742103285, 0.5768493540183687, 0.13068371284607674, 1.360347409153062,
-2.634887549640136, 8.97031966066549, -0.5938872755970316, -2.4608283290499204, 1.6167426993790432, 2.586516131123313, -0.6442037890047975, 12.000000000000005, 0.2190430730103402, -0.015170467393955334, 1.2485385227460235, -0.6765153661826085, -0.17058628301270545, 0.19125087414978356, -0.05851096675960009, -0.432655439147446, 0.2916156291448159, 2.4229355524755003,
-1.135434157870108, -0.22290356912809986, -2.4861325362711635, 0.019371806350233497, -1.3530307323652313, 1.6167426993790395, 0.8226694181207385, -0.2190430730103402, 12.000000000000005, -1.221466415659336, 1.2932580655616515, 0.3682041766330148, -0.11093602189671982, 0.43432201350751376, -0.6132774672199166, 2.7998077074825862, 0.781378414022764, -0.6100947628964163,
-1.889669062317673, 1.72308862498413e-15, 1.8896690623176793, 0.8926105190374863, -1.6696076717446235, -0.0207363284165587, -0.8926105190374908, 0.020736328416551132, 1.669607671744627, 0.0, -2.005051247169507, 2.005051247169518, -0.46331338833748403, 0.4633133883374836, -7.923728999874815e-16, -0.7834980551347012, 0.7834980551346985, 2.487535508292328e-15,
0.37098370646896894, 1.499453391770039, -11.456452196936628, 0.03454227374420177, 0.3682041766330094, -0.6765153661826211, 1.2393619133905893, -1.2485385227460235, -1.2932580655616515, 1.4668732071255233, 0.0, -0.43808614602068285, -0.0524250551371181, -0.44269118420721065, 0.24307113935773086, 0.37687215500708976, -0.17743932374896845, 0.48976278487794267,
11.456452196936628, -1.4994533917700434, -0.37098370646896595, -1.2393619133905802, 1.2932580655616668, 1.2485385227460257, -0.03454227374421096, 0.6765153661826085, -0.3682041766330148, -1.4668732071255315, 0.43808614602068285, 0.0, 0.44269118420721193, 0.05242505513711802, -0.24307113935772984, 0.17743932374897087, -0.37687215500708626, -0.48976278487794556,
-3.3903460469074287, 0.682311604453907, -3.022694422562719, 11.848869866769022, 2.85946375481224, 1.2591462686659523, 4.272088065230031, 1.123095947644239, 0.7303740631402283, 2.2315915230588166, 0.34515299788282394, -2.9145642091510853, 0.0, -2.9158628521532504, -1.4579314260766258, 0.3883097004508778, 1.6035974570934002, -1.6703256795818757,
3.0226944225627075, -0.68231160445394, 3.3903460469074145, -4.272088065230015, -0.7303740631402075, -1.1230959476442475, -11.848869866769023, -1.259146268665941, -2.859463754812246, -2.2315915230588144, 2.914564209151077, -0.3451529978828235, 2.9158628521532504, 0.0, 1.4579314260766267, -1.6035974570933964, -0.3883097004508765, 1.6703256795818802,
-4.072657651361349, -1.2744265588712169e-14, 4.072657651361328, -2.040496542171217, -4.037660156795314, -0.3852210652573858, 2.04049654217119, 0.3852210652573937, 4.037660156795319, 3.816536908330339e-15, -1.6003174861463092, 1.6003174861463025, 1.4579314260766258, -1.4579314260766267, 0.0, -3.2739231366752795, 3.273923136675281, -1.5485517097979382e-15,
1.6271240279890422, -2.1979339738354438, -2.924128936306672, 0.43424681453975356, 3.5490285023571606, 1.3245210783844776, -1.9168034714993356, 1.9651252935551495, -12.71675436210026, 2.6034731278339103, -1.7117570640055617, -0.8059311674378195, -0.26788818737287295, 1.1062943201204989, 2.2586232938906448, 0.0, -1.8821511283679437, -0.941075564183975,
2.9241289363066914, 2.1979339738354198, -1.627124027989073, 1.9168034714993096, 12.71675436210027, -1.9651252935551553, -0.434246814539738, -1.324521078384457, -3.549028502357164, -2.6034731278339014, 0.8059311674378086, 1.7117570640055457, -1.1062943201205013, 0.26788818737287207, -2.258623293890646, 1.8821511283679437, 0.0, 0.9410755641839769,
3.825058001824488, 5.68431650770163e-16, -3.825058001824485, 4.520276599333235, -2.7710564609929693, 11.00499729809471, -4.5202765993332195, -11.004997298094716, 2.7710564609929667, -8.265791864994906e-15, -2.2245074239726823, 2.2245074239726956, 1.152328973770141, -1.152328973770144, 1.0683192052870465e-15, 0.941075564183975, -0.9410755641839769, 0.0};

   const double p4Dx[729] = {-15.28499617463146, 0.1965429454017319, 0.09827147270087869, 0.4207579292099769, -0.03011871770800639, -3.318562255325795, -0.12660138709292057, -0.20489551789966803, -0.1525100012592784, -0.02465488205886147, -0.44232233201277743, -0.16408537644014787, 0.28709832395121765, 0.06989189152237547, -0.445234967991587, 0.1446267442307163, -0.13342014677484493, 0.04884580398085625, -0.06120785634410805, -0.0065053622920859784, -0.055536979099377465, -0.05310883929020609, 0.137613494649232, 0.09694858537476729, 0.05875484376287877, 0.11266078234486253, -0.10900916485152536,
-0.1965429454017319, 15.28499617463146, -0.09827147270085763, 0.030118717708007003, -0.42075792920997546, 0.12660138709291616, 3.318562255325798, -0.28709832395121665, 0.16408537644014565, 0.4452349679915819, -0.06989189152237295, 0.15251000125928654, 0.20489551789967583, 0.44232233201277626, 0.024654882058862064, -0.14462674423071573, -0.048845803980857666, 0.13342014677484318, 0.06120785634410922, 0.055536979099377264, 0.006505362292085202, -0.13761349464923325, 0.0531088392902052, 0.10900916485152572, -0.11266078234486292, -0.05875484376287771, -0.09694858537476635,
-0.09827147270087869, 0.09827147270085763, 0.0, 0.05238551664039212, -0.05238551664039154, -0.41766744995391064, 0.41766744995391425, 0.13396665873213762, 0.13365960525875686, -0.19649327861528906, -2.87332728733421, -0.1336596052587585, -0.13396665873213948, 2.873327287334207, 0.19649327861529092, 7.071416687303329e-16, 0.0957809402498582, -0.09578094024985911, 5.095198309682238e-16, -0.005670877244732207, 0.005670877244731516, -0.0381937416118884, 0.03819374161188797, 0.024952712304370763, 0.055900325561320505, -0.05590032556132075, -0.02495271230436985,
-2.688383016538256, -0.19243998400255485, -0.3347110618046863, 0.0, -1.8584179757445634, 2.9254109394468375, 0.5544784209743959, -0.09619204170147806, 0.9621335517719756, -0.7640240022365051, -0.17045575971749122, -0.32599503457469536, 1.5819839652391803, -0.1673484738314025, -2.1466117678065375, 0.20084022575022525, 0.8121668775566188, 0.28698839120282377, 0.22242408399761535, 0.1007642564284906, 0.1396610379722831, -0.4047350379804057, -0.19103788701331148, -0.4861845324334061, 0.0708248987102739, -0.5390429074476013, 0.02706866117436579,
0.1924399840025509, 2.688383016538247, 0.3347110618046826, 1.8584179757445634, 0.0, -0.5544784209744025, -2.9254109394468055, -1.5819839652391763, 0.32599503457469153, 2.1466117678065477, 0.1673484738314174, -0.9621335517719789, 0.09619204170146632, 0.1704557597174755, 0.7640240022365011, -0.20084022575022814, -0.28698839120282255, -0.8121668775566167, -0.22242408399761296, -0.13966103797228493, -0.10076425642849256, 0.19103788701331245, 0.40473503798040156, -0.027068661174367476, 0.5390429074475989, -0.07082489871027405, 0.48618453243340803,
16.79909075460827, -0.6408763879659072, 2.114299162464476, -2.3177353997619106, 0.43930042352937404, 0.0, 0.23758827371798064, 0.14587872928125803, -0.026683834684955113, 0.27594927228434313, -0.8508769996321647, 1.733847812899358, -1.9338802842578322, -1.4125593711637086, 1.7702761450361215, 0.49385180688379604, -0.48895648790844665, -0.15872007418498266, -0.09920419307861998, 0.12317929238116793, 0.08585557349611746, 0.09370614907316197, -0.2607460957437406, -0.00372051314937977, -0.04020180712625255, 0.5265453421646704, -0.4929563256972127,
0.6408763879659295, -16.79909075460828, -2.1142991624644942, -0.4393004235293688, 2.3177353997618853, -0.23758827371798064, 0.0, 1.9338802842578542, -1.733847812899381, -1.7702761450361162, 1.4125593711636986, 0.026683834684962007, -0.14587872928127077, 0.8508769996321606, -0.27594927228435256, -0.4938518068837918, 0.15872007418498413, 0.4889564879084464, 0.0992041930786239, -0.08585557349611386, -0.1231792923811642, 0.26074609574374336, -0.09370614907315725, 0.49295632569721404, -0.5265453421646692, 0.040201807126253306, 0.0037205131493783417,
1.3091556741914314, 1.834380779552396, -0.8559641188321434, 0.09619204170147806, 1.5819839652391763, -0.1841259491984436, -2.440914756589403, 19.910396398833793, 0.9292089878722896, -0.6009288150226979, -0.17768908386483087, -0.22980299287322684, -1.9242671035439767, 2.9524607667234655, 0.2041356965652301, -1.1190995177139584, -0.24116380960165718, 0.7245564881296083, -0.49029775323051117, -0.04870816987800388, -0.7132041988172136, -0.454608663355292, 0.08758856822375095, 1.0907550537346067, 4.030854005025503, 0.7748473349352286, 0.07689148379455818,
0.9744446123867609, -1.0484041028347049, -0.8540022369858385, -0.9621335517719756, -0.32599503457469153, 0.03367993684774262, 2.18843676448695, -0.9292089878722896, 19.910396398833793, 0.3767893371095702, 2.3244821244241383, 3.163967930478371, -0.22980299287321784, -4.587526524395932, -0.3514744230298633, 1.53672336568623, -0.04032358385143443, -0.832111126511136, -0.6124399423887177, 0.1737159141196091, -0.3506367152582221, 0.14771638250483257, 0.2886628025018225, 3.839816118012189, 0.6860200157542047, 0.11465722939811557, -0.9936515708028879,
0.12480694029056734, -2.253850330037397, 0.9946802760235417, 0.6053185391404056, -1.7007108357677436, -0.27594927228434313, 1.7702761450361162, 0.47610199597409236, -0.29852147371705867, 17.141859608597226, -0.11879413685898123, -0.27846505302304264, 0.16173198913849687, -1.6885086434480685, 1.7017539992643287, 0.5994748509692391, -0.6941668393814941, 3.674640424508895, -0.09744008746280787, 0.20279671720873968, -0.3963039467276607, -0.7855375215315802, 0.6600584756782935, -0.4640248521828324, 0.7038226731025359, -0.1687199363657259, 0.20287631854527582,
2.2391061027550703, 0.35380388805763474, 14.545240424570853, 0.13504815445355628, -0.13258632374176726, 0.8508769996321647, -1.4125593711636986, 0.14077894981231426, -1.8416334037878173, 0.11879413685898123, 17.141859608597226, -3.634591120025567, 2.339166352039784, 3.5405522900722497, -1.6885086434480694, 3.1856839366004506, -0.20031503249770058, 0.44075477678425123, -0.2731246543464956, 0.10359252413012018, -0.0115845139666891, 0.16267451141902461, -0.1724404495151067, 0.4430765773587932, -0.3703187031096739, 0.1671021499810791, -0.2589921793669097,
1.048404102834719, -0.9744446123868129, 0.8540022369858489, 0.32599503457469536, 0.9621335517719789, -2.1884367644869207, -0.03367993684775132, 0.22980299287322684, -3.163967930478371, 0.3514744230298829, 4.587526524395912, -19.910396398833793, 0.9292089878722776, -2.324482124424116, -0.37678933710957857, -1.536723365686234, 0.8321111265111332, 0.04032358385143004, 0.612439942388719, 0.35063671525822543, -0.17371591411960696, -0.28866280250182014, -0.14771638250483188, 0.9936515708028864, -0.11465722939811639, -0.6860200157542059, -3.8398161180121906,
-1.8343807795524023, -1.3091556741914812, 0.8559641188321553, -1.5819839652391803, -0.09619204170146632, 2.440914756589375, 0.18412594919845968, 1.9242671035439767, 0.22980299287321784, -0.2041356965652155, -2.9524607667234513, -0.9292089878722776, -19.910396398833793, 0.17768908386480717, 0.6009288150227365, 1.1190995177139624, -0.7245564881296058, 0.24116380960166542, 0.49029775323050956, 0.7132041988172101, 0.04870816987800567, -0.08758856822375086, 0.4546086633552912, -0.07689148379455372, -0.7748473349352273, -4.030854005025498, -1.0907550537346087,
-0.35380388805764745, -2.2391061027550645, -14.545240424570837, 0.13258632374175544, -0.13504815445354384, 1.4125593711637086, -0.8508769996321606, -2.3391663520397947, 3.6345911200255827, 1.6885086434480685, -3.5405522900722497, 1.8416334037877997, -0.1407789498122955, -17.141859608597226, -0.11879413685897054, -3.1856839366004435, -0.44075477678425035, 0.20031503249770535, 0.2731246543464969, 0.01158451396669102, -0.10359252413011845, 0.17244044951510712, -0.16267451141902445, 0.25899217936691005, -0.16710214998107684, 0.37031870310967413, -0.44307657735879563,
2.2538503300374226, -0.12480694029057034, -0.994680276023551, 1.7007108357677354, -0.6053185391404025, -1.7702761450361215, 0.27594927228435256, -0.16173198913850842, 0.2784650530230271, -1.7017539992643287, 1.6885086434480694, 0.29852147371706533, -0.476101995974123, 0.11879413685897054, -17.141859608597226, -0.5994748509692417, -3.6746404245088935, 0.6941668393814934, 0.09744008746280555, 0.39630394672766056, -0.2027967172087368, -0.6600584756782925, 0.7855375215315809, -0.20287631854527904, 0.16871993636572646, -0.7038226731025339, 0.4640248521828319,
-3.766522549523058, 3.7665225495230428, -1.8416130814168517e-14, -0.8186227860390011, 0.818622786039013, -2.5406965754889375, 2.5406965754889157, 4.561438634236725, -6.263669333621437, -3.0840905707320223, -16.389240973690264, 6.263669333621453, -4.561438634236742, 16.38924097369023, 3.0840905707320356, 0.0, 0.8064071592627126, -0.8064071592627154, -3.27587819047868e-15, -0.2376302079323201, 0.23763020793230094, -0.5181032452374912, 0.5181032452374866, -2.3969263418417017, 0.8425636809391238, -0.8425636809391335, 2.396926341841694,
3.474668492754472, 1.2720940592425334, -2.494428490280513, -3.3103842098885403, 1.169761861647588, 2.5155118541145702, -0.8165598329744024, 0.9829813174648175, 0.1643585314258267, 3.571248068855471, 1.030551493366546, -3.3916767725891264, 2.953285123732874, 2.267530737757596, 18.904752827804817, -0.8064071592627126, 0.0, -1.6128143185254504, 0.281848015316382, -1.069973391943026, 0.5194782232486872, 1.842445782443736, -3.2355644271100483, -0.020449207033394267, -0.5385524522708879, -0.8386380852683435, 0.9998821015046064,
-1.2720940592424965, -3.474668492754426, 2.4944284902805367, -1.169761861647593, 3.310384209888532, 0.8165598329743948, -2.515511854114569, -2.953285123732884, 3.391676772589138, -18.904752827804824, -2.2675307377576006, -0.1643585314258088, -0.9829813174648512, -1.0305514933665705, -3.571248068855467, 0.8064071592627154, 1.6128143185254504, 0.0, -0.2818480153163751, -0.5194782232487007, 1.0699733919430288, 3.235564427110037, -1.8424457824437297, -0.9998821015046048, 0.8386380852683486, 0.5385524522708905, 0.020449207033400366,
2.2431057877263694, -2.243105787726412, -1.867255202307378e-14, -1.275749958493393, 1.2757499584933791, 0.7181857904438594, -0.7181857904438878, 2.8121834969091495, 3.512750134965057, 0.7054146005697136, 1.977277771072542, -3.5127501349650645, -2.8121834969091406, -1.9772777710725515, -0.705414600569697, 4.609760570128499e-15, -0.39661177621033, 0.3966117762103203, 0.0, 1.473286325955928, -1.4732863259559217, 0.03294149361242836, -0.0329414936124289, -3.953179136944931, -1.408551582879458, 1.4085515828794593, 3.9531791369449287,
0.23840429448465078, -2.035283159571775, 0.20782262815464633, -0.5779499847581375, 0.8010488801122674, -0.8917528051961954, 0.6215488579847753, 0.27937372870475113, -0.996376229788628, -1.468140772361843, -0.749954981917987, -2.0111346167968778, -4.090700119723205, -0.08386574258491672, -2.869030576268716, 0.3343892230125261, 1.5056485212427524, 0.7310009992228542, -1.473286325955928, 0.0, -2.9465726519118554, -0.8795833962368981, 0.19919319766269372, 0.49291247512221925, 0.5258539687346377, 4.152372334607633, 0.5289681866425602,
2.035283159571782, -0.23840429448462236, -0.20782262815462102, -0.801048880112257, 0.5779499847581487, -0.6215488579848013, 0.8917528051961683, 4.090700119723225, 2.0111346167968587, 2.869030576268717, 0.0838657425849028, 0.9963762297886157, -0.27937372870476146, 0.7499549819179745, 1.468140772361822, -0.3343892230124991, -0.7310009992228351, -1.5056485212427564, 1.4732863259559217, 2.9465726519118554, 0.0, -0.19919319766269705, 0.8795833962369017, -0.5289681866425588, -4.152372334607625, -0.5258539687346414, -0.49291247512221426,
1.759402157237792, 4.558892315982919, 1.2652912826383578, 2.0985063083979236, -0.9905102682507421, -0.6132401678100317, -1.706397937502056, 2.357095527641503, -0.7658930694187117, 5.140784956908139, -1.0645890975962542, 1.4966846336620354, 0.4541370173391847, -1.1285002237720632, 4.319613754198333, 0.6590565197748737, -2.3436948454060653, -4.115820254824023, -0.029778238966182636, 0.795120126366912, 0.1800653822873389, 0.0, 3.1163752731045724, 0.15960718397414111, -0.7346658981606436, -0.1973888096528869, -0.7116415547893532,
-4.558892315982878, -1.7594021572377623, -1.2652912826383433, 0.9905102682507371, -2.0985063083979023, 1.7063979375020377, 0.6132401678100009, -0.4541370173391852, -1.4966846336620476, -4.319613754198339, 1.1285002237720605, 0.7658930694187083, -2.357095527641499, 1.064589097596253, -5.140784956908144, -0.6590565197748679, 4.115820254824037, 2.3436948454060573, 0.029778238966183125, -0.1800653822873359, -0.7951201263669153, -3.1163752731045724, 0.0, 0.71164155478935, 0.1973888096528901, 0.7346658981606372, -0.15960718397413862,
-3.2117356080303803, -3.6112813302216713, -0.8266393399665194, 2.520812908731641, 0.14034800771819048, 0.024348115151799948, -3.2260489080343286, -5.655444047050364, -19.909020941069866, 3.0367129695880832, -2.8996213945368607, -5.151973251646951, 0.3986738203099664, -1.6949198009668083, 1.3276813620016823, 3.049025358424062, 0.026012543529633645, 1.271906370130427, 3.57356938337672, -0.4455798406197396, 0.47817325020768053, -0.15960718397414111, -0.71164155478935, 0.0, -1.5581876365522882, -0.3569959936270113, 1.46933179632127,
-1.9464443253920187, 3.7322529760164143, -1.8518791729839088, -0.3672192491087197, -2.794877724005471, 0.26309226440541583, 3.4458651559413247, -20.899531209320617, -3.556937738652459, -4.606019332038912, 2.4234728017780744, 0.594485025057365, 4.017497542393682, 1.0935648461639937, -1.1041521086202575, -1.071788475275629, 0.6850690633045105, -1.0667948963999776, 1.2732933765745933, -0.4753580795859133, 3.753634765664056, 0.7346658981606436, -0.1973888096528901, 1.5581876365522882, 0.0, -1.4232831095787026, -0.35699599362703105,
-3.7322529760164014, 1.9464443253919836, 1.8518791729839168, 2.794877724005484, 0.36721924910872045, -3.4458651559413327, -0.26309226440542083, -4.017497542393689, -0.5944850250573608, 1.1041521086202537, -1.0935648461640088, 3.5569377386524654, 20.89953120932059, -2.423472801778076, 4.6060193320388985, 1.0717884752756415, 1.0667948963999712, -0.6850690633045139, -1.2732933765745946, -3.753634765664064, 0.47535807958591664, 0.1973888096528869, -0.7346658981606372, 0.3569959936270113, 1.4232831095787026, 0.0, -1.5581876365522849,
3.6112813302216593, 3.2117356080303487, 0.8266393399664891, -0.14034800771818173, -2.5208129087316506, 3.2260489080343198, -0.024348115151790604, -0.3986738203099895, 5.151973251646959, -1.3276813620016612, 1.694919800966806, 19.909020941069873, 5.6554440470503735, 2.8996213945368763, -3.03671296958808, -3.0490253584240525, -1.2719063701304292, -0.026012543529641403, -3.5735693833767184, -0.47817325020768164, 0.4455798406197351, 0.7116415547893532, 0.15960718397413862, -1.46933179632127, 0.35699599362703105, 1.5581876365522849, 0.0};

   const double p4Dy[729] = {-15.28499617463146, 0.09827147270086124, 0.19654294540174796, 0.2870983239512125, -0.164085376440144, -0.4452349679915834, 0.06989189152237135, -0.15251000125927563, -0.2048955178996692, -0.44232233201278237, -0.024654882058863847, -0.03011871770800668, 0.42075792920998006, -0.12660138709291216, -3.3185622553257907, 0.048845803980856306, -0.13342014677484385, 0.14462674423071498, -0.05553697909937705, -0.006505362292085907, -0.06120785634410891, -0.10900916485152638, 0.11266078234486171, 0.05875484376287875, 0.09694858537476714, 0.1376134946492328, -0.05310883929020566,
-0.09827147270086124, 0.0, 0.09827147270087347, -0.13396665873213587, -0.13365960525875817, 0.19649327861528584, 2.8733272873342135, 0.1336596052587607, 0.13396665873214064, -2.873327287334207, -0.19649327861528823, -0.052385516640395494, 0.05238551664038741, 0.4176674499539139, -0.41766744995391747, -0.095780940249859, 0.09578094024985785, -9.030965889809071e-16, 0.00567087724473144, -0.0056708772447323225, -2.468104976161084e-16, -0.024952712304370517, -0.05590032556132032, 0.05590032556132042, 0.02495271230436975, 0.03819374161188785, -0.03819374161188903,
-0.19654294540174796, -0.09827147270087347, 15.28499617463146, 0.20489551789967816, 0.15251000125928516, 0.024654882058868614, 0.44232233201278004, 0.16408537644014445, -0.2870983239512186, -0.06989189152236565, 0.445234967991583, -0.42075792920997346, 0.03011871770800419, 3.3185622553257907, 0.12660138709291832, 0.13342014677484448, -0.04884580398085644, -0.14462674423071606, 0.0065053622920860695, 0.055536979099378284, 0.06120785634410899, -0.0969485853747657, -0.058754843762877665, -0.11266078234486313, 0.1090091648515255, 0.05310883929020579, -0.1376134946492322,
-1.8343807795523694, 0.8559641188321322, -1.309155674191496, -19.910396398833793, -0.9292089878722894, 0.6009288150227082, 0.1776890838648202, 0.22980299287321443, 1.9242671035439682, -2.9524607667234357, -0.20413569656523223, -0.09619204170148214, -1.581983965239188, 0.18412594919846403, 2.4409147565893874, 0.24116380960165718, -0.7245564881296103, 1.1190995177139629, 0.048708169878006975, 0.7132041988172096, 0.49029775323050717, -1.0907550537346093, -4.0308540050255, -0.774847334935228, -0.07689148379455714, 0.45460866335528777, -0.0875885682237497,
1.0484041028346944, 0.8540022369858469, -0.9744446123868041, 0.9292089878722894, -19.910396398833793, -0.376789337109576, -2.3244821244241085, -3.163967930478372, 0.22980299287322126, 4.587526524395933, 0.35147442302985465, 0.9621335517719841, 0.32599503457468426, -0.03367993684773616, -2.1884367644869362, 0.0403235838514332, 0.8321111265111371, -1.5367233656862271, -0.17371591411960613, 0.3506367152582226, 0.6124399423887191, -3.8398161180121875, -0.686020015754205, -0.11465722939811551, 0.99365157080289, -0.14771638250483227, -0.2886628025018202,
2.2538503300374044, -0.9946802760235254, -0.1248069402906035, -0.4761019959741005, 0.2985214737170633, -17.141859608597226, 0.1187941368589855, 0.27846505302302493, -0.1617319891384971, 1.6885086434480594, -1.7017539992643365, -0.605318539140416, 1.700710835767743, 0.2759492722843605, -1.7702761450361306, 0.6941668393814986, -3.674640424508889, -0.5994748509692356, -0.20279671720873876, 0.3963039467276609, 0.097440087462807, 0.4640248521828354, -0.703822673102532, 0.16871993636572868, -0.20287631854527582, 0.7855375215315806, -0.6600584756782898,
-0.35380388805762664, -14.545240424570872, -2.2391061027550836, -0.14077894981230582, 1.8416334037877937, -0.1187941368589855, -17.141859608597226, 3.6345911200255987, -2.339166352039789, -3.540552290072251, 1.6885086434480605, -0.135048154453527, 0.13258632374176318, -0.8508769996321724, 1.4125593711637054, 0.20031503249770397, -0.44075477678425157, -3.1856839366004435, -0.103592524130115, 0.01158451396669216, 0.2731246543464982, -0.44307657735879197, 0.37031870310967574, -0.16710214998107722, 0.2589921793669143, -0.16267451141902012, 0.17244044951510798,
0.9744446123867433, -0.854002236985863, -1.048404102834697, -0.22980299287321443, 3.163967930478372, -0.35147442302986054, -4.587526524395952, 19.910396398833793, -0.9292089878722678, 2.324482124424122, 0.37678933710956625, -0.32599503457469653, -0.9621335517719883, 2.188436764486939, 0.03367993684775833, -0.8321111265111345, -0.0403235838514329, 1.5367233656862256, -0.3506367152582263, 0.17371591411960866, -0.612439942388722, -0.9936515708028895, 0.11465722939811439, 0.686020015754204, 3.839816118012191, 0.28866280250181964, 0.1477163825048354,
1.3091556741914387, -0.8559641188321627, 1.8343807795524083, -1.9242671035439682, -0.22980299287322126, 0.20413569656521577, 2.952460766723458, 0.9292089878722678, 19.910396398833793, -0.17768908386481014, -0.6009288150226963, 1.5819839652391892, 0.09619204170147512, -2.440914756589397, -0.18412594919844522, 0.7245564881296075, -0.24116380960166175, -1.119099517713959, -0.7132041988172084, -0.04870816987800363, -0.4902977532305103, 0.0768914837945529, 0.774847334935229, 4.030854005025501, 1.090755053734602, 0.08758856822374894, -0.4546086633552895,
2.2391061027550956, 14.545240424570837, 0.3538038880575977, 2.3391663520397716, -3.634591120025584, -1.6885086434480594, 3.540552290072251, -1.8416334037878042, 0.14077894981229783, 17.141859608597226, 0.1187941368589982, -0.13258632374177015, 0.13504815445355095, -1.4125593711636983, 0.8508769996321621, 0.4407547767842533, -0.200315032497699, 3.1856839366004475, -0.011584513966690685, 0.10359252413011888, -0.27312465434649635, -0.2589921793669082, 0.16710214998107692, -0.37031870310967424, 0.4430765773587957, -0.17244044951510495, 0.16267451141902212,
0.12480694029057937, 0.9946802760235375, -2.253850330037402, 0.16173198913851014, -0.27846505302302027, 1.7017539992643365, -1.6885086434480605, -0.29852147371705556, 0.47610199597409114, -0.1187941368589982, 17.141859608597226, -1.7007108357677325, 0.6053185391404096, 1.7702761450361277, -0.27594927228436483, 3.674640424508897, -0.694166839381494, 0.5994748509692353, -0.39630394672765873, 0.2027967172087418, -0.09744008746280586, 0.2028763185452754, -0.1687199363657282, 0.7038226731025327, -0.4640248521828344, 0.6600584756782918, -0.7855375215315795,
0.1924399840025528, 0.33471106180470783, 2.6883830165382343, 0.09619204170148214, -0.9621335517719841, 0.7640240022365183, 0.17045575971745425, 0.32599503457469653, -1.5819839652391892, 0.16734847383142104, 2.1466117678065335, 0.0, 1.8584179757445463, -2.9254109394468206, -0.5544784209744086, -0.8121668775566203, -0.286988391202826, -0.20084022575023078, -0.10076425642849045, -0.13966103797228321, -0.2224240839976154, 0.4861845324334064, -0.07082489871027524, 0.5390429074475949, -0.027068661174365027, 0.4047350379803992, 0.1910378870133102,
-2.6883830165382765, -0.33471106180465615, -0.19243998400253687, 1.581983965239188, -0.32599503457468426, -2.146611767806547, -0.16734847383141227, 0.9621335517719883, -0.09619204170147512, -0.1704557597174845, -0.7640240022365101, -1.8584179757445463, 0.0, 0.554478420974375, 2.925410939446845, 0.2869883912028199, 0.8121668775566261, 0.20084022575022775, 0.1396610379722875, 0.10076425642848952, 0.22242408399761332, 0.027068661174365592, -0.5390429074475985, 0.07082489871027872, -0.4861845324334045, -0.191037887013308, -0.40473503798040444,
0.640876387965887, -2.114299162464493, -16.799090754608244, -0.14587872928127424, 0.026683834684949993, -0.2759492722843605, 0.8508769996321724, -1.7338478128993726, 1.9338802842578497, 1.4125593711636983, -1.7702761450361277, 2.3177353997618972, -0.43930042352935234, 0.0, -0.23758827371795765, 0.4889564879084486, 0.15872007418498785, -0.4938518068837892, -0.12317929238116428, -0.08585557349611499, 0.09920419307862376, 0.0037205131493781296, 0.04020180712624783, -0.5265453421646701, 0.49295632569721504, -0.09370614907316001, 0.26074609574373936,
16.799090754608244, 2.1142991624645107, -0.6408763879659182, -1.933880284257842, 1.7338478128993704, 1.7702761450361306, -1.4125593711637054, -0.02668383468496756, 0.1458787292812593, -0.8508769996321621, 0.27594927228436483, 0.4393004235293789, -2.317735399761917, 0.23758827371795765, 0.0, -0.15872007418498993, -0.4889564879084488, 0.4938518068837932, 0.08585557349611653, 0.12317929238116535, -0.0992041930786226, -0.4929563256972161, 0.52654534216467, -0.04020180712625487, -0.0037205131493803054, -0.2607460957437367, 0.09370614907315984,
-1.2720940592424979, 2.494428490280534, -3.4746684927544607, -0.9829813174648175, -0.16435853142582169, -3.5712480688554935, -1.0305514933665634, 3.3916767725891317, -2.953285123732881, -2.2675307377576113, -18.904752827804835, 3.310384209888547, -1.1697618616475773, -2.5155118541145804, 0.8165598329744322, 0.0, 1.6128143185254429, 0.806407159262733, 1.0699733919430285, -0.5194782232486954, -0.2818480153163862, 0.020449207033393636, 0.538552452270888, 0.8386380852683486, -0.9998821015046098, -1.842445782443737, 3.2355644271100434,
3.4746684927544433, -2.4944284902805043, 1.2720940592425014, 2.9532851237328925, -3.3916767725891424, 18.904752827804796, 2.2675307377576024, 0.1643585314258205, 0.9829813174648363, 1.0305514933665378, 3.5712480688554704, 1.1697618616476022, -3.3103842098885705, -0.8165598329744215, 2.5155118541145813, -1.6128143185254429, 0.0, -0.8064071592627268, 0.5194782232486914, -1.069973391943029, 0.28184801531637504, 0.9998821015046067, -0.838638085268352, -0.5385524522708939, -0.020449207033397976, -3.2355644271100372, 1.8424457824437361,
-3.7665225495230237, 2.3519395979540512e-14, 3.7665225495230517, -4.561438634236743, 6.263669333621425, 3.084090570732004, 16.38924097369023, -6.263669333621419, 4.561438634236729, -16.38924097369025, -3.0840905707320028, 0.8186227860390236, -0.8186227860390113, 2.5406965754889024, -2.540696575488923, -0.806407159262733, 0.8064071592627268, 0.0, 0.23763020793230993, -0.23763020793231787, 4.300846213266488e-15, 2.39692634184169, -0.8425636809391248, 0.8425636809391247, -2.3969263418416866, 0.5181032452374869, -0.5181032452374932,
2.035283159571767, -0.20782262815461824, -0.2384042944846541, -0.27937372870476895, 0.9963762297886111, 1.4681407723618363, 0.7499549819179495, 2.011134616796883, 4.090700119723195, 0.08386574258491429, 2.8690305762687025, 0.5779499847581366, -0.801048880112282, 0.891752805196169, -0.6215488579847945, -1.5056485212427562, -0.7310009992228411, -0.3343892230125118, 0.0, 2.9465726519118554, 1.4732863259559286, -0.4929124751222108, -0.5258539687346461, -4.1523723346076205, -0.5289681866425575, 0.8795833962369044, -0.1991931976626978,
0.2384042944846482, 0.2078226281546506, -2.0352831595718124, -4.090700119723202, -2.0111346167968613, -2.8690305762687185, -0.08386574258492496, -0.9963762297886256, 0.27937372870474975, -0.7499549819179776, -1.4681407723618582, 0.8010488801122575, -0.5779499847581313, 0.6215488579847835, -0.8917528051961766, 0.7310009992228468, 1.5056485212427566, 0.334389223012523, -2.9465726519118554, 0.0, -1.47328632595593, 0.5289681866425614, 4.152372334607627, 0.5258539687346433, 0.4929124751222133, 0.19919319766269775, -0.8795833962368973,
2.243105787726401, 9.044950905679127e-15, -2.2431057877264036, -2.812183496909127, -3.512750134965065, -0.7054146005697074, -1.9772777710725609, 3.512750134965082, 2.8121834969091446, 1.9772777710725475, 0.7054146005696991, 1.2757499584933931, -1.2757499584933811, -0.7181857904438868, 0.7181857904438784, 0.39661177621033594, -0.39661177621032023, -6.052078294524533e-15, -1.4732863259559286, 1.47328632595593, 0.0, 3.9531791369449296, 1.4085515828794586, -1.4085515828794646, -3.953179136944927, -0.03294149361242823, 0.032941493612420415,
3.6112813302216935, 0.8266393399665112, 3.2117356080303274, 5.655444047050377, 19.909020941069855, -3.0367129695881028, 2.8996213945368523, 5.151973251646967, -0.3986738203099622, 1.6949198009667963, -1.3276813620016585, -2.520812908731642, -0.1403480077181807, -0.024348115151789213, 3.226048908034342, -0.026012543529632844, -1.2719063701304294, -3.049025358424047, 0.445579840619732, -0.4781732502076828, -3.573569383376719, 0.0, 1.5581876365522862, 0.35699599362702117, -1.4693317963212784, 0.159607183974139, 0.7116415547893488,
-3.7322529760163743, 1.8518791729839026, 1.946444325391982, 20.899531209320596, 3.556937738652461, 4.606019332038886, -2.4234728017780864, -0.5944850250573547, -4.01749754239369, -1.0935648461639944, 1.1041521086202688, 0.3672192491087266, 2.7948777240054694, -0.26309226440538497, -3.4458651559413305, -0.6850690633045106, 1.066794896399982, 1.0717884752756304, 0.4753580795859208, -3.7536347656640583, -1.273293376574594, -1.5581876365522862, 0.0, 1.4232831095787029, 0.3569959936270241, -0.7346658981606364, 0.19738880965288536,
-1.9464443253920178, -1.8518791729839061, 3.7322529760164214, 4.017497542393685, 0.5944850250573606, -1.104152108620272, 1.0935648461639964, -3.5569377386524557, -20.8995312093206, 2.4234728017780762, -4.6060193320388905, -2.7948777240054508, -0.36721924910874476, 3.445865155941331, 0.26309226440543104, -1.0667948963999776, 0.685069063304518, -1.0717884752756304, 3.753634765664053, -0.47535807958591836, 1.2732933765745995, -0.35699599362702117, -1.4232831095787029, 0.0, 1.5581876365522802, -0.19738880965288783, 0.73466589816063,
-3.211735608030375, -0.8266393399664858, -3.611281330221664, 0.39867382030998416, -5.15197325164697, 1.3276813620016612, -1.6949198009668363, -19.909020941069876, -5.65544404705034, -2.8996213945368767, 3.0367129695880966, 0.14034800771817776, 2.5208129087316324, -3.226048908034335, 0.024348115151803455, 1.2719063701304334, 0.026012543529638363, 3.0490253584240428, 0.4781732502076793, -0.4455798406197342, 3.573569383376717, 1.4693317963212784, -0.3569959936270241, -1.5581876365522802, 0.0, -0.7116415547893499, -0.1596071839741321,
-4.558892315982905, -1.2652912826383393, -1.7594021572377818, -2.3570955276414813, 0.7658930694187103, -5.140784956908142, 1.0645890975962247, -1.496684633662033, -0.4541370173391748, 1.1285002237720492, -4.319613754198328, -2.09850630839789, 0.9905102682507191, 0.6132401678100189, 1.7063979375020122, 2.3436948454060667, 4.115820254824023, -0.6590565197748682, -0.7951201263669176, -0.18006538228733954, 0.029778238966182518, -0.159607183974139, 0.7346658981606364, 0.19738880965288783, 0.7116415547893499, 0.0, -3.11637527310457,
1.7594021572377778, 1.2652912826383786, 4.5588923159828845, 0.4541370173391787, 1.4966846336620356, 4.319613754198315, -1.1285002237720687, -0.7658930694187266, 2.35709552764149, -1.064589097596238, 5.140784956908135, -0.9905102682507304, 2.098506308397917, -1.7063979375020297, -0.6132401678100178, -4.115820254824031, -2.3436948454060658, 0.6590565197748763, 0.1800653822873396, 0.7951201263669113, -0.029778238966175454, -0.7116415547893488, -0.19738880965288536, -0.73466589816063, 0.1596071839741321, 3.11637527310457, 0.0};

   // Populate the Dx and Dy matrices and create the element's Nodes
   switch (p)
   {
      case 0:
         *Dx=p0Dx;
         *Dy=p0Dy;

         Nodes.IntPoint(0).Set2w(0.0, 0.0, 0.16666666666666666);
         Nodes.IntPoint(1).Set2w(1.0, 0.0, 0.16666666666666666);
         Nodes.IntPoint(2).Set2w(0.0, 1.0, 0.16666666666666666);
         break;
      case 1:
         *Dx=p1Dx;
         *Dy=p1Dy;

         Nodes.IntPoint(0).Set2w(0.0, 0.0, 0.024999999999999998);
         Nodes.IntPoint(1).Set2w(1.0, 0.0, 0.024999999999999998);
         Nodes.IntPoint(2).Set2w(0.0, 1.0, 0.024999999999999998);
         Nodes.IntPoint(3).Set2w(0.5, 0.0, 0.06666666666666667);
         Nodes.IntPoint(4).Set2w(0.5, 0.5, 0.06666666666666667);
         Nodes.IntPoint(5).Set2w(0.0, 0.5, 0.06666666666666667);
         Nodes.IntPoint(6).Set2w(0.3333333333333333, 0.3333333333333333, 0.22500000000000006);
         break;
      case 2:
         *Dx=p2Dx;
         *Dy=p2Dy;
  
         // vertices
         Nodes.IntPoint(0).Set2w(0.0, 0.0, 0.006261126504899741);
         Nodes.IntPoint(1).Set2w(1.0, 0.0, 0.006261126504899741);
         Nodes.IntPoint(2).Set2w(0.0, 1.0, 0.006261126504899741);

         // edges
         Nodes.IntPoint(3).Set2w(0.27639320225002106, 0.0, 0.026823800250389242);
         Nodes.IntPoint(4).Set2w(0.7236067977499789, 0.0, 0.026823800250389242);
         Nodes.IntPoint(5).Set2w(0.7236067977499789, 0.27639320225002106, 0.026823800250389242);
         Nodes.IntPoint(6).Set2w(0.27639320225002106, 0.7236067977499789, 0.026823800250389242);
         Nodes.IntPoint(7).Set2w(0.0, 0.7236067977499789, 0.026823800250389242);
         Nodes.IntPoint(8).Set2w(0.0, 0.27639320225002106, 0.026823800250389242);

         // interior
         Nodes.IntPoint(9).Set2w(0.21285435711180825, 0.5742912857763836, 0.10675793966098839);
         Nodes.IntPoint(10).Set2w(0.21285435711180825, 0.21285435711180825, 0.10675793966098839);
         Nodes.IntPoint(11).Set2w(0.5742912857763836, 0.21285435711180825, 0.10675793966098839);
         break;
      case 3:
         *Dx=p3Dx;
         *Dy=p3Dy;
   
         // vertices
         Nodes.IntPoint(0).Set2w(0.0, 0.0, 0.0022825661430496253);
         Nodes.IntPoint(1).Set2w(1.0, 0.0, 0.0022825661430496253);
         Nodes.IntPoint(2).Set2w(0.0, 1.0, 0.0022825661430496253);

         // edges
         Nodes.IntPoint(3).Set2w(0.5, 0.0, 0.015504052643022513);
         Nodes.IntPoint(4).Set2w(0.17267316464601146, 0.0, 0.011342592592592586);
         Nodes.IntPoint(5).Set2w(0.8273268353539885, 0.0, 0.011342592592592586);

         Nodes.IntPoint(6).Set2w(0.5, 0.5, 0.015504052643022513);
         Nodes.IntPoint(7).Set2w(0.8273268353539885, 0.17267316464601146, 0.011342592592592586);
         Nodes.IntPoint(8).Set2w(0.17267316464601146, 0.8273268353539885, 0.011342592592592586);

         Nodes.IntPoint(9).Set2w(0.0, 0.5, 0.015504052643022513);
         Nodes.IntPoint(10).Set2w(0.0, 0.8273268353539885, 0.011342592592592586);
         Nodes.IntPoint(11).Set2w(0.0, 0.17267316464601146, 0.011342592592592586);

         // interior
         Nodes.IntPoint(12).Set2w(0.4243860251718814, 0.1512279496562372, 0.07467669469983994);
         Nodes.IntPoint(13).Set2w(0.4243860251718814, 0.4243860251718814, 0.07467669469983994);
         Nodes.IntPoint(14).Set2w(0.1512279496562372, 0.4243860251718814, 0.07467669469983994);

         Nodes.IntPoint(15).Set2w(0.14200508409677795, 0.7159898318064442, 0.051518167995569394);
         Nodes.IntPoint(16).Set2w(0.14200508409677795, 0.14200508409677795, 0.051518167995569394);
         Nodes.IntPoint(17).Set2w(0.7159898318064442, 0.14200508409677795, 0.051518167995569394);

         break;
      case 4:
         *Dx=p4Dx;
         *Dy=p4Dy; 

         // vertices
         Nodes.IntPoint(0).Set2w(0.000000000000000000,0.000000000000000000,0.001090393904993471);
         Nodes.IntPoint(1).Set2w(1.000000000000000000,0.000000000000000000,0.001090393904993471);
         Nodes.IntPoint(2).Set2w(0.000000000000000000,1.000000000000000000,0.001090393904993471);

         // edges
         Nodes.IntPoint(3).Set2w(0.357384241759677534,0.000000000000000000,0.006966942871463700);
         Nodes.IntPoint(4).Set2w(0.642615758240322466,0.000000000000000000,0.006966942871463700);
         Nodes.IntPoint(5).Set2w(0.117472338035267576,0.000000000000000000,0.005519747637357106);
         Nodes.IntPoint(6).Set2w(0.882527661964732424,0.000000000000000000,0.005519747637357106);

         Nodes.IntPoint(7).Set2w(0.642615758240322466,0.357384241759677534,0.006966942871463700);
         Nodes.IntPoint(8).Set2w(0.357384241759677534,0.642615758240322466,0.006966942871463700);
         Nodes.IntPoint(9).Set2w(0.882527661964732424,0.117472338035267576,0.005519747637357106);
         Nodes.IntPoint(10).Set2w(0.117472338035267576,0.882527661964732424,0.005519747637357106);

         Nodes.IntPoint(11).Set2w(0.000000000000000000,0.642615758240322466,0.006966942871463700);
         Nodes.IntPoint(12).Set2w(0.000000000000000000,0.357384241759677534,0.006966942871463700);
         Nodes.IntPoint(13).Set2w(0.000000000000000000,0.882527661964732424,0.005519747637357106);
         Nodes.IntPoint(14).Set2w(0.000000000000000000,0.117472338035267576,0.005519747637357106);

         // interior
         Nodes.IntPoint(15).Set2w(0.103677508142805172,0.792644983714389628,0.028397190663911491);
         Nodes.IntPoint(16).Set2w(0.103677508142805172,0.103677508142805172,0.028397190663911491);
         Nodes.IntPoint(17).Set2w(0.792644983714389628,0.103677508142805172,0.028397190663911491);
         Nodes.IntPoint(18).Set2w(0.265331380484209678,0.469337239031580644,0.039960048027851809);
         Nodes.IntPoint(19).Set2w(0.265331380484209678,0.265331380484209678,0.039960048027851809);
         Nodes.IntPoint(20).Set2w(0.469337239031580644,0.265331380484209678,0.039960048027851809);
         Nodes.IntPoint(21).Set2w(0.587085567133367348,0.088273960601581103,0.036122826526134168);
         Nodes.IntPoint(22).Set2w(0.324640472265051494,0.088273960601581103,0.036122826526134168);
         Nodes.IntPoint(23).Set2w(0.324640472265051494,0.587085567133367348,0.036122826526134168);
         Nodes.IntPoint(24).Set2w(0.587085567133367348,0.324640472265051494,0.036122826526134168);
         Nodes.IntPoint(25).Set2w(0.088273960601581103,0.324640472265051494,0.036122826526134168);
         Nodes.IntPoint(26).Set2w(0.088273960601581103,0.587085567133367348,0.036122826526134168);

         break;
      default:
         mfem_error("SBP elements are currently only supported for 0 <= order <= 4");
         break;
   }
   for (int i = 0; i < Dof; i++)
   {
      ipIdxMap[&(Nodes.IntPoint(i))] = i;
   }
}

/// CalcShape outputs ndofx1 vector shape based on Kronecker \delta_{i, ip}
/// where ip is the integration point CalcShape is evaluated at. 
void SBP_TriangleElement::CalcShape(const IntegrationPoint &ip,
                                   Vector &shape) const
{
   int ipIdx;
   try
   {
      ipIdx = ipIdxMap.at(&ip);
   }
   catch (const std::out_of_range& oor)
   // error handling code to handle cases where the pointer to ip is not
   // in the map. Problems arise in GridFunction::SaveVTK(), it seems like
   // an integration rule is constructed on its own and does not use Nodes.
   // I will investigate further, but in the meantime this handles the error
   // and uses the old (slow) linear search approach.
   {
#ifdef MFEM_DEBUG      
      mfem::out << "Integration Point out of range error in unordered map. "
                << "Using linear search.\n";
#endif
      for (int i = 0; i < Dof; i++)
      {
         if (ip.x == Nodes.IntPoint(i).x && ip.y == Nodes.IntPoint(i).y)
         {
            ipIdx = i;
         }
      }
   }
   shape = 0.0;
   shape(ipIdx) = 1.0;
}

/// CalcDShape outputs ndof x ndim DenseMatrix dshape, where the first column
/// is the ith row of Dx, and the second column is the ith row of Dy, where i
/// is the integration point CalcDShape is evaluated at. Since DenseMatrices 
/// are stored a column major we should store the transpose so accessing a row
/// is faster, but this is not done here.
void SBP_TriangleElement::CalcDShape(const IntegrationPoint &ip,
                                    DenseMatrix &dshape) const
{
   int ipIdx;
   try
   {
      ipIdx = ipIdxMap.at(&ip);
   }
   catch (const std::out_of_range& oor)
   // error handling code to handle cases where the pointer to ip is not
   // in the map. Problems arise in GridFunction::SaveVTK(), it seems like
   // an integration rule is constructed on its own and does not use Nodes.
   // I will investigate further, but in the meantime this handles the error
   // and uses the old (slow) linear search approach.
   {
#ifdef MFEM_DEBUG      
      mfem::out << "Integration Point out of range error in unordered map. "
                << "Using linear search.\n";
#endif
      for (int i = 0; i < Dof; i++)
      {
         if (ip.x == Nodes.IntPoint(i).x && ip.y == Nodes.IntPoint(i).y)
         {
            ipIdx = i;
         }
      }
   }
   dshape = 0.0;

   Vector tempVec(Dof);

   // when we switch to storing Dx and Dy transpose so that access to the row we want
   // is faster Dx->GetRow() will be replaced with Dx->GetColumnReference() or 
   // Dx->GetColumn(), whichever is faster
   Dx->GetRow(ipIdx, tempVec);
   dshape.SetCol(0, tempVec);
   Dy->GetRow(ipIdx, tempVec);
   dshape.SetCol(1, tempVec);
}

SBP_TriangleElement::~SBP_TriangleElement()
{
   delete Dx;
   delete Dy;
}

// Global object definitions


// Object declared in mesh/triangle.hpp.
// Defined here to ensure it is constructed before 'Geometries'.
Linear2DFiniteElement TriangleFE;

// Object declared in mesh/tetrahedron.hpp.
// Defined here to ensure it is constructed before 'Geometries'.
Linear3DFiniteElement TetrahedronFE;

// Object declared in mesh/wedge.hpp.
// Defined here to ensure it is constructed after 'poly1d' and before
// 'Geometries'.
H1_WedgeElement WedgeFE(1);

// Object declared in geom.hpp.
// Construct 'Geometries' after 'TriangleFE', 'TetrahedronFE', and 'WedgeFE'.
Geometry Geometries;

}
