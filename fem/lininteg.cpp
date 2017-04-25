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


#include <cmath>
#include "fem.hpp"

namespace mfem
{

void LinearFormIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   mfem_error("LinearFormIntegrator::AssembleRHSElementVect(...)");
}

void LinearFormIntegrator::AssembleRHSElementVect(
   const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, Vector &elvect)
{
   mfem_error("LinearFormIntegrator::AssembleRHSElementVect(...)");
}

void DomainLFIntegrator::AssembleRHSElementVect(const FiniteElement &el,
                                                ElementTransformation &Tr,
                                                Vector &elvect)
{
   int dof = el.GetDof();

   shape.SetSize(dof);       // vector of size dof
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // ir = &IntRules.Get(el.GetGeomType(),
      //                    oa * el.GetOrder() + ob + Tr.OrderW());
      ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint (&ip);
      double val = Tr.Weight() * Q.Eval(Tr, ip);

      el.CalcShape(ip, shape);

      add(elvect, ip.weight * val, shape, elvect);
   }
}

void BoundaryLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dof = el.GetDof();

   shape.SetSize(dof);        // vector of size dof
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = oa * el.GetOrder() + ob;  // <----------
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint (&ip);
      double val = Tr.Weight() * Q.Eval(Tr, ip);

      el.CalcShape(ip, shape);

      add(elvect, ip.weight * val, shape, elvect);
   }
}

void BoundaryNormalLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dim = el.GetDim()+1;
   int dof = el.GetDof();
   Vector nor(dim), Qvec;

   shape.SetSize(dof);
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = oa * el.GetOrder() + ob;  // <----------
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint(&ip);
      CalcOrtho(Tr.Jacobian(), nor);
      Q.Eval(Qvec, Tr, ip);

      el.CalcShape(ip, shape);

      elvect.Add(ip.weight*(Qvec*nor), shape);
   }
}

void BoundaryTangentialLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dim = el.GetDim()+1;
   int dof = el.GetDof();
   Vector tangent(dim), Qvec;

   shape.SetSize(dof);
   elvect.SetSize(dof);
   elvect = 0.0;

   if (dim != 2)
   {
      mfem_error("These methods make sense only in 2D problems.");
   }

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = oa * el.GetOrder() + ob;  // <----------
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint(&ip);
      const DenseMatrix &Jac = Tr.Jacobian();
      tangent(0) =  Jac(0,0);
      tangent(1) = Jac(1,0);

      Q.Eval(Qvec, Tr, ip);

      el.CalcShape(ip, shape);

      add(elvect, ip.weight*(Qvec*tangent), shape, elvect);
   }
}

void VectorDomainLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int vdim = Q.GetVDim();
   int dof  = el.GetDof();

   double val,cf;

   shape.SetSize(dof);       // vector of size dof

   elvect.SetSize(dof * vdim);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = el.GetOrder() + 1;
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint (&ip);
      val = Tr.Weight();

      el.CalcShape(ip, shape);
      Q.Eval (Qvec, Tr, ip);

      for (int k = 0; k < vdim; k++)
      {
         cf = val * Qvec(k);

         for (int s = 0; s < dof; s++)
         {
            elvect(dof*k+s) += ip.weight * cf * shape(s);
         }
      }
   }
}

void VectorBoundaryLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int vdim = Q.GetVDim();
   int dof  = el.GetDof();

   shape.SetSize(dof);
   vec.SetSize(vdim);

   elvect.SetSize(dof * vdim);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = el.GetOrder() + 1;
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Q.Eval(vec, Tr, ip);
      Tr.SetIntPoint (&ip);
      vec *= Tr.Weight() * ip.weight;
      el.CalcShape(ip, shape);
      for (int k = 0; k < vdim; k++)
         for (int s = 0; s < dof; s++)
         {
            elvect(dof*k+s) += vec(k) * shape(s);
         }
   }
}

void VectorBoundaryLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   int vdim = Q.GetVDim();
   int dof  = el.GetDof();

   shape.SetSize(dof);
   vec.SetSize(vdim);

   elvect.SetSize(dof * vdim);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = el.GetOrder() + 1;
      ir = &IntRules.Get(Tr.FaceGeom, intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      IntegrationPoint eip;
      Tr.Loc1.Transform(ip, eip);

      Tr.Face->SetIntPoint(&ip);
      Q.Eval(vec, *Tr.Face, ip);
      vec *= Tr.Face->Weight() * ip.weight;
      el.CalcShape(eip, shape);
      for (int k = 0; k < vdim; k++)
      {
         for (int s = 0; s < dof; s++)
         {
            elvect(dof*k+s) += vec(k) * shape(s);
         }
      }
   }
}


void VectorFEDomainLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dof = el.GetDof();
   int spaceDim = Tr.GetSpaceDim();

   vshape.SetSize(dof,spaceDim);
   vec.SetSize(spaceDim);

   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // int intorder = 2*el.GetOrder() - 1; // ok for O(h^{k+1}) conv. in L2
      int intorder = 2*el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint (&ip);
      el.CalcVShape(Tr, vshape);

      QF.Eval (vec, Tr, ip);
      vec *= ip.weight * Tr.Weight();

      vshape.AddMult (vec, elvect);
   }
}


void VectorBoundaryFluxLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dim = el.GetDim()+1;
   int dof = el.GetDof();

   shape.SetSize (dof);
   nor.SetSize (dim);
   elvect.SetSize (dim*dof);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      ir = &IntRules.Get(el.GetGeomType(), el.GetOrder() + 1);
   }

   elvect = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint (&ip);
      CalcOrtho(Tr.Jacobian(), nor);
      el.CalcShape (ip, shape);
      nor *= Sign * ip.weight * F -> Eval (Tr, ip);
      for (int j = 0; j < dof; j++)
         for (int k = 0; k < dim; k++)
         {
            elvect(dof*k+j) += nor(k) * shape(j);
         }
   }
}


void VectorFEBoundaryFluxLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dof = el.GetDof();

   shape.SetSize(dof);
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = 2*el.GetOrder();  // <----------
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint (&ip);
      double val = ip.weight*F.Eval(Tr, ip);

      el.CalcShape(ip, shape);

      add(elvect, val, shape, elvect);
   }
}


void VectorFEBoundaryTangentLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dof = el.GetDof();
   DenseMatrix vshape(dof, 2);
   Vector f_loc(3);
   Vector f_hat(2);

   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = 2*el.GetOrder();  // <----------
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint(&ip);
      f.Eval(f_loc, Tr, ip);
      Tr.Jacobian().MultTranspose(f_loc, f_hat);
      el.CalcVShape(ip, vshape);

      Swap<double>(f_hat(0), f_hat(1));
      f_hat(0) = -f_hat(0);
      f_hat *= ip.weight;
      vshape.AddMult(f_hat, elvect);
   }
}


void BoundaryFlowIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("BoundaryFlowIntegrator::AssembleRHSElementVect\n"
              "  is not implemented as boundary integrator!\n"
              "  Use LinearForm::AddBdrFaceIntegrator instead of\n"
              "  LinearForm::AddBoundaryIntegrator.");
}

void BoundaryFlowIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   int dim, ndof, order;
   double un, w, vu_data[3], nor_data[3];

   dim  = el.GetDim();
   ndof = el.GetDof();
   Vector vu(vu_data, dim), nor(nor_data, dim);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // Assuming order(u)==order(mesh)
      order = Tr.Elem1->OrderW() + 2*el.GetOrder();
      if (el.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Tr.FaceGeom, order);
   }

   shape.SetSize(ndof);
   elvect.SetSize(ndof);
   elvect = 0.0;

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip;
      Tr.Loc1.Transform(ip, eip);
      el.CalcShape(eip, shape);

      Tr.Face->SetIntPoint(&ip);

      u->Eval(vu, *Tr.Elem1, eip);

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr.Face->Jacobian(), nor);
      }

      un = vu * nor;
      w = 0.5*alpha*un - beta*fabs(un);
      w *= ip.weight*f->Eval(*Tr.Elem1, eip);
      elvect.Add(w, shape);
   }
}


void DGDirichletLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("DGDirichletLFIntegrator::AssembleRHSElementVect");
}

void DGDirichletLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   int dim, ndof;
   bool kappa_is_nonzero = (kappa != 0.);
   double w;

   dim = el.GetDim();
   ndof = el.GetDof();

   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   adjJ.SetSize(dim);
   if (MQ)
   {
      mq.SetSize(dim);
   }

   shape.SetSize(ndof);
   dshape.SetSize(ndof, dim);
   dshape_dn.SetSize(ndof);

   elvect.SetSize(ndof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      int order = 2*el.GetOrder();
      ir = &IntRules.Get(Tr.FaceGeom, order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip;

      Tr.Loc1.Transform(ip, eip);
      Tr.Face->SetIntPoint(&ip);
      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr.Face->Jacobian(), nor);
      }

      el.CalcShape(eip, shape);
      el.CalcDShape(eip, dshape);
      Tr.Elem1->SetIntPoint(&eip);
      // compute uD through the face transformation
      w = ip.weight * uD->Eval(*Tr.Face, ip) / Tr.Elem1->Weight();
      if (!MQ)
      {
         if (Q)
         {
            w *= Q->Eval(*Tr.Elem1, eip);
         }
         ni.Set(w, nor);
      }
      else
      {
         nh.Set(w, nor);
         MQ->Eval(mq, *Tr.Elem1, eip);
         mq.MultTranspose(nh, ni);
      }
      CalcAdjugate(Tr.Elem1->Jacobian(), adjJ);
      adjJ.Mult(ni, nh);

      dshape.Mult(nh, dshape_dn);
      elvect.Add(sigma, dshape_dn);

      if (kappa_is_nonzero)
      {
         elvect.Add(kappa*(ni*nor), shape);
      }
   }
}


void DGElasticityDirichletLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("DGElasticityDirichletLFIntegrator::AssembleRHSElementVect");
}

void DGElasticityDirichletLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   MFEM_ASSERT(Tr.Elem2No < 0, "interior boundary is not supported");

#ifdef MFEM_THREAD_SAFE
   Vector shape;
   DenseMatrix dshape;
   DenseMatrix adjJ;
   DenseMatrix dshape_ps;
   Vector nor;
   Vector dshape_dn;
   Vector dshape_du;
   Vector u_dir;
#endif

   const int dim = el.GetDim();
   const int ndofs = el.GetDof();
   const int nvdofs = dim*ndofs;

   elvect.SetSize(nvdofs);
   elvect = 0.0;

   adjJ.SetSize(dim);
   shape.SetSize(ndofs);
   dshape.SetSize(ndofs, dim);
   dshape_ps.SetSize(ndofs, dim);
   nor.SetSize(dim);
   dshape_dn.SetSize(ndofs);
   dshape_du.SetSize(ndofs);
   u_dir.SetSize(dim);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      const int order = 2*el.GetOrder(); // <-----
      ir = &IntRules.Get(Tr.FaceGeom, order);
   }

   for (int pi = 0; pi < ir->GetNPoints(); ++pi)
   {
      const IntegrationPoint &ip = ir->IntPoint(pi);
      IntegrationPoint eip;
      Tr.Loc1.Transform(ip, eip);
      Tr.Face->SetIntPoint(&ip);
      Tr.Elem1->SetIntPoint(&eip);

      // Evaluate the Dirichlet b.c. using the face transformation.
      uD.Eval(u_dir, *Tr.Face, ip);

      el.CalcShape(eip, shape);
      el.CalcDShape(eip, dshape);

      CalcAdjugate(Tr.Elem1->Jacobian(), adjJ);
      Mult(dshape, adjJ, dshape_ps);

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr.Face->Jacobian(), nor);
      }

      double wL, wM, jcoef;
      {
         const double w = ip.weight / Tr.Elem1->Weight();
         wL = w * lambda->Eval(*Tr.Elem1, eip);
         wM = w * mu->Eval(*Tr.Elem1, eip);
         jcoef = kappa * (wL + 2.0*wM) * (nor*nor);
         dshape_ps.Mult(nor, dshape_dn);
         dshape_ps.Mult(u_dir, dshape_du);
      }

      // alpha < uD, (lambda div(v) I + mu (grad(v) + grad(v)^T)) . n > +
      //   + kappa < h^{-1} (lambda + 2 mu) uD, v >

      // i = idof + ndofs * im
      // v_phi(i,d) = delta(im,d) phi(idof)
      // div(v_phi(i)) = dphi(idof,im)
      // (grad(v_phi(i)))(k,l) = delta(im,k) dphi(idof,l)
      //
      // term 1:
      //   alpha < uD, lambda div(v_phi(i)) n >
      //   alpha lambda div(v_phi(i)) (uD.n) =
      //   alpha lambda dphi(idof,im) (uD.n) --> quadrature -->
      //   ip.weight/det(J1) alpha lambda (uD.nor) dshape_ps(idof,im) =
      //   alpha * wL * (u_dir*nor) * dshape_ps(idof,im)
      // term 2:
      //   < alpha uD, mu grad(v_phi(i)).n > =
      //   alpha mu uD^T grad(v_phi(i)) n =
      //   alpha mu uD(k) delta(im,k) dphi(idof,l) n(l) =
      //   alpha mu uD(im) dphi(idof,l) n(l) --> quadrature -->
      //   ip.weight/det(J1) alpha mu uD(im) dshape_ps(idof,l) nor(l) =
      //   alpha * wM * u_dir(im) * dshape_dn(idof)
      // term 3:
      //   < alpha uD, mu (grad(v_phi(i)))^T n > =
      //   alpha mu n^T grad(v_phi(i)) uD =
      //   alpha mu n(k) delta(im,k) dphi(idof,l) uD(l) =
      //   alpha mu n(im) dphi(idof,l) uD(l) --> quadrature -->
      //   ip.weight/det(J1) alpha mu nor(im) dshape_ps(idof,l) uD(l) =
      //   alpha * wM * nor(im) * dshape_du(idof)
      // term j:
      //   < kappa h^{-1} (lambda + 2 mu) uD, v_phi(i) > =
      //   kappa/h (lambda + 2 mu) uD(k) v_phi(i,k) =
      //   kappa/h (lambda + 2 mu) uD(k) delta(im,k) phi(idof) =
      //   kappa/h (lambda + 2 mu) uD(im) phi(idof) --> quadrature -->
      //      [ 1/h = |nor|/det(J1) ]
      //   ip.weight/det(J1) |nor|^2 kappa (lambda + 2 mu) uD(im) phi(idof) =
      //   jcoef * u_dir(im) * shape(idof)

      wM *= alpha;
      const double t1 = alpha * wL * (u_dir*nor);
      for (int im = 0, i = 0; im < dim; ++im)
      {
         const double t2 = wM * u_dir(im);
         const double t3 = wM * nor(im);
         const double tj = jcoef * u_dir(im);
         for (int idof = 0; idof < ndofs; ++idof, ++i)
         {
            elvect(i) += (t1*dshape_ps(idof,im) + t2*dshape_dn(idof) +
                          t3*dshape_du(idof) + tj*shape(idof));
         }
      }
   }
}




void DGEulerIntegrator::getEulerFlux(const Vector &u, Vector &f)
{
    int var_dim = u.Size();
    int dim     = var_dim - 2;

    double rho  = u(0);
    double u1   = u(1)/rho;
    double u2   = u(2)/rho;
    
    double v_sq = pow(u1, 2) + pow(u2, 2);
    double p    = (u(var_dim - 1) - 0.5*rho*v_sq)*(gamm - 1);

    f(0)             =  u(1); 
    f(1)             =  u(1)*u1 + p;               //rho*u*u + p    
    f(2)             =  u(1)*u2    ;               //rho*u*v
    f(var_dim - 1)   = (u(var_dim - 1) + p)*u1  ;  //(E+p)*u

    f(  var_dim + 0) =  u(2); 
    f(  var_dim + 1) =  u(2)*u1;                  //rho*u*v     
    f(  var_dim + 2) =  u(2)*u2 + p;              //rho*v*v + p
    f(2*var_dim - 1) = (u(var_dim - 1) + p)*u2  ; //(E+p)*v    
}

void DGEulerIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("DGEulerIntegrator::AssembleRHSElementVect");
}

void DGEulerIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   mfem_error("DGEEulerIntegrator::AssembleRHSElementVect");
}



void DGEulerIntegrator::AssembleRHSElementVect(
   const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Trans, Vector &elvect)
{
   int dim, ndof1, ndof2;

   double un, a, b, w;

   Vector shape1, shape2;

   dim = el1.GetDim();
   ndof1 = el1.GetDof();
   ndof2 = el2.GetDof();
   
   Vector vu(dim), nor(dim);

   elvect.SetSize(vDim*(ndof1 + ndof2));
   elvect = 0.0;

   shape1.SetSize(ndof1);
   shape2.SetSize(ndof2);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      if (Trans.Elem2No >= 0)
         order = (std::min(Trans.Elem1->OrderW(), Trans.Elem2->OrderW()) +
                  2*std::max(el1.GetOrder(), el2.GetOrder()));
      else
      {
         order = Trans.Elem1->OrderW() + 2*el1.GetOrder();
      }
      if (el1.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.FaceGeom, order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip1, eip2;
      Trans.Loc1.Transform(ip, eip1);
      if (ndof2)
      {
         Trans.Loc2.Transform(ip, eip2);
      }

      Trans.Face->SetIntPoint(&ip);
      Trans.Elem1->SetIntPoint(&eip1);
      Trans.Elem2->SetIntPoint(&eip2);

      el1.CalcShape(eip1, shape1);
      el2.CalcShape(eip2, shape2);

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Face->Jacobian(), nor);
      }

      Vector u1_dir(vDim), u2_dir(vDim);
      uD.Eval(u1_dir, *Trans.Elem1, eip1);
      uD.Eval(u2_dir, *Trans.Elem2, eip2);

      double rho_L = u1_dir(0);
      double u_L   = u1_dir(1)/rho_L;
      double v_L   = u1_dir(2)/rho_L;
      double E_L   = u1_dir(3);

      double T_L   = (E_L - 0.5*rho_L*(pow(u_L,2) + pow(v_L,2)))/(rho_L*Cv);
      double a_L   = sqrt(gamm * R * T_L);

      double rho_R = u2_dir(0);
      double u_R   = u2_dir(1)/rho_R;
      double v_R   = u2_dir(2)/rho_R;
      double E_R   = u2_dir(3);

      double T_R   = (E_R - 0.5*rho_R*(pow(u_R,2) + pow(v_R,2)))/(rho_R*Cv);
      double a_R   = sqrt(gamm * R * T_R);

      Vector nor_dim(dim);
      double nor_l2 = nor.Norml2();
      nor_dim.Set(1/nor_l2, nor);

      double vnl   = u_L*nor_dim(0) + v_L*nor_dim(1); 
      double vnr   = u_R*nor_dim(0) + v_R*nor_dim(1); 

      double u_max = std::max(a_L + std::abs(vnl), a_R + std::abs(vnr));

      Vector fl_dir(dim*vDim), fr_dir(dim*vDim);
      Vector f_dir(dim*vDim);
      getEulerFlux(u1_dir, fl_dir);
      getEulerFlux(u2_dir, fr_dir);
      add(0.5, fl_dir, fr_dir, f_dir); //Common flux without dissipation

      Vector f1_dir(dim*vDim), f2_dir(dim*vDim);
      fD.Eval(f1_dir, *Trans.Elem1, eip1);
      fD.Eval(f2_dir, *Trans.Elem2, eip2);

      Vector f_diss(dim*vDim); //Rusanov dissipation 
      subtract(-0.5*u_max*nor_dim(0), u2_dir, u1_dir, f_diss); // Left and right depends on normal direction
      for (int i = 0; i < dim; i++)
      {
          for (int j = 0; j < vDim; j++)
          {
              f_diss(i*vDim + j) = -0.5*u_max*nor_dim(i)*(u2_dir(j) - u1_dir(j)); 
          }
      }

      add(f_dir, f_diss, f_dir);

      Vector face_f(vDim), face_f1(vDim), face_f2(vDim); //Face fluxes (dot product with normal)
      face_f = 0.0; face_f1 = 0.0; face_f2 = 0.0;
      for (int i = 0; i < dim; i++)
      {
          for (int j = 0; j < vDim; j++)
          {
              face_f1(j) += f1_dir(i*vDim + j)*nor(i);
              face_f2(j) += f2_dir(i*vDim + j)*nor(i);
              face_f (j) += f_dir (i*vDim + j)*nor(i);
          }
      }

      w = ip.weight * alpha; 

      subtract(face_f, face_f1, face_f1); //u_comm - u1
      for (int j = 0; j < vDim; j++)
      {
          for (int i = 0; i < ndof1; i++)
          {
              elvect(j*ndof1 + i)              += face_f1(j)*w*shape1(i); 
          }
      }

      subtract(face_f, face_f2, face_f2); //ucomm - u2
      for (int j = 0; j < vDim; j++)
      {
          for (int i = 0; i < ndof2; i++)
          {
              elvect(vDim*ndof1 + j*ndof2 + i) -= face_f2(j)*w*shape2(i); 
          }
      }

   }// for ir loop

}



}
