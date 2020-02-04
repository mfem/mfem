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

#include "fem.hpp"
#include "../general/forall.hpp"

namespace mfem
{

void NonlinearFormIntegrator::AssemblePA(const FiniteElementSpace&)
{
   mfem_error ("NonlinearFormIntegrator::AssemblePA(...)\n"
               "   is not implemented for this class.");
}

void NonlinearFormIntegrator::AssemblePA(const FiniteElementSpace &,
                                         const FiniteElementSpace &)
{
   mfem_error ("NonlinearFormIntegrator::AssemblePA(...)\n"
               "   is not implemented for this class.");
}

void NonlinearFormIntegrator::AddMultPA(const Vector &, Vector &) const
{
   mfem_error ("NonlinearFormIntegrator::AddMultPA(...)\n"
               "   is not implemented for this class.");
}

void NonlinearFormIntegrator::AssembleElementVector(
   const FiniteElement &el, ElementTransformation &Tr,
   const Vector &elfun, Vector &elvect)
{
   mfem_error("NonlinearFormIntegrator::AssembleElementVector"
              " is not overloaded!");
}

void NonlinearFormIntegrator::AssembleFaceVector(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Tr, const Vector &elfun, Vector &elvect)
{
   mfem_error("NonlinearFormIntegrator::AssembleFaceVector"
              " is not overloaded!");
}

void NonlinearFormIntegrator::AssembleElementGrad(
   const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun,
   DenseMatrix &elmat)
{
   mfem_error("NonlinearFormIntegrator::AssembleElementGrad"
              " is not overloaded!");
}

void NonlinearFormIntegrator::AssembleFaceGrad(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Tr, const Vector &elfun,
   DenseMatrix &elmat)
{
   mfem_error("NonlinearFormIntegrator::AssembleFaceGrad"
              " is not overloaded!");
}

double NonlinearFormIntegrator::GetElementEnergy(
   const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun)
{
   mfem_error("NonlinearFormIntegrator::GetElementEnergy"
              " is not overloaded!");
   return 0.0;
}


void BlockNonlinearFormIntegrator::AssembleElementVector(
   const Array<const FiniteElement *> &el,
   ElementTransformation &Tr,
   const Array<const Vector *> &elfun,
   const Array<Vector *> &elvec)
{
   mfem_error("BlockNonlinearFormIntegrator::AssembleElementVector"
              " is not overloaded!");
}

void BlockNonlinearFormIntegrator::AssembleFaceVector(
   const Array<const FiniteElement *> &el1,
   const Array<const FiniteElement *> &el2,
   FaceElementTransformations &Tr,
   const Array<const Vector *> &elfun,
   const Array<Vector *> &elvect)
{
   mfem_error("BlockNonlinearFormIntegrator::AssembleFaceVector"
              " is not overloaded!");
}

void BlockNonlinearFormIntegrator::AssembleElementGrad(
   const Array<const FiniteElement*> &el,
   ElementTransformation &Tr,
   const Array<const Vector *> &elfun,
   const Array2D<DenseMatrix *> &elmats)
{
   mfem_error("BlockNonlinearFormIntegrator::AssembleElementGrad"
              " is not overloaded!");
}

void BlockNonlinearFormIntegrator::AssembleFaceGrad(
   const Array<const FiniteElement *>&el1,
   const Array<const FiniteElement *>&el2,
   FaceElementTransformations &Tr,
   const Array<const Vector *> &elfun,
   const Array2D<DenseMatrix *> &elmats)
{
   mfem_error("BlockNonlinearFormIntegrator::AssembleFaceGrad"
              " is not overloaded!");
}

double BlockNonlinearFormIntegrator::GetElementEnergy(
   const Array<const FiniteElement *>&el,
   ElementTransformation &Tr,
   const Array<const Vector *>&elfun)
{
   mfem_error("BlockNonlinearFormIntegrator::GetElementEnergy"
              " is not overloaded!");
   return 0.0;
}


double InverseHarmonicModel::EvalW(const DenseMatrix &J) const
{
   Z.SetSize(J.Width());
   CalcAdjugateTranspose(J, Z);
   return 0.5*(Z*Z)/J.Det();
}

void InverseHarmonicModel::EvalP(const DenseMatrix &J, DenseMatrix &P) const
{
   int dim = J.Width();
   double t;

   Z.SetSize(dim);
   S.SetSize(dim);
   CalcAdjugateTranspose(J, Z);
   MultAAt(Z, S);
   t = 0.5*S.Trace();
   for (int i = 0; i < dim; i++)
   {
      S(i,i) -= t;
   }
   t = J.Det();
   S *= -1.0/(t*t);
   Mult(S, Z, P);
}

void InverseHarmonicModel::AssembleH(
   const DenseMatrix &J, const DenseMatrix &DS, const double weight,
   DenseMatrix &A) const
{
   int dof = DS.Height(), dim = DS.Width();
   double t;

   Z.SetSize(dim);
   S.SetSize(dim);
   G.SetSize(dof, dim);
   C.SetSize(dof, dim);

   CalcAdjugateTranspose(J, Z);
   MultAAt(Z, S);

   t = 1.0/J.Det();
   Z *= t;  // Z = J^{-t}
   S *= t;  // S = |J| (J.J^t)^{-1}
   t = 0.5*S.Trace();

   MultABt(DS, Z, G);  // G = DS.J^{-1}
   Mult(G, S, C);

   // 1.
   for (int i = 0; i < dof; i++)
      for (int j = 0; j <= i; j++)
      {
         double a = 0.0;
         for (int d = 0; d < dim; d++)
         {
            a += G(i,d)*G(j,d);
         }
         a *= weight;
         for (int k = 0; k < dim; k++)
            for (int l = 0; l <= k; l++)
            {
               double b = a*S(k,l);
               A(i+k*dof,j+l*dof) += b;
               if (i != j)
               {
                  A(j+k*dof,i+l*dof) += b;
               }
               if (k != l)
               {
                  A(i+l*dof,j+k*dof) += b;
                  if (i != j)
                  {
                     A(j+l*dof,i+k*dof) += b;
                  }
               }
            }
      }

   // 2.
   for (int i = 1; i < dof; i++)
      for (int j = 0; j < i; j++)
      {
         for (int k = 1; k < dim; k++)
            for (int l = 0; l < k; l++)
            {
               double a =
                  weight*(C(i,l)*G(j,k) - C(i,k)*G(j,l) +
                          C(j,k)*G(i,l) - C(j,l)*G(i,k) +
                          t*(G(i,k)*G(j,l) - G(i,l)*G(j,k)));

               A(i+k*dof,j+l*dof) += a;
               A(j+l*dof,i+k*dof) += a;

               A(i+l*dof,j+k*dof) -= a;
               A(j+k*dof,i+l*dof) -= a;
            }
      }
}


inline void NeoHookeanModel::EvalCoeffs() const
{
   mu = c_mu->Eval(*Ttr, Ttr->GetIntPoint());
   K = c_K->Eval(*Ttr, Ttr->GetIntPoint());
   if (c_g)
   {
      g = c_g->Eval(*Ttr, Ttr->GetIntPoint());
   }
}

double NeoHookeanModel::EvalW(const DenseMatrix &J) const
{
   int dim = J.Width();

   if (have_coeffs)
   {
      EvalCoeffs();
   }

   double dJ = J.Det();
   double sJ = dJ/g;
   double bI1 = pow(dJ, -2.0/dim)*(J*J); // \bar{I}_1

   return 0.5*(mu*(bI1 - dim) + K*(sJ - 1.0)*(sJ - 1.0));
}

void NeoHookeanModel::EvalP(const DenseMatrix &J, DenseMatrix &P) const
{
   int dim = J.Width();

   if (have_coeffs)
   {
      EvalCoeffs();
   }

   Z.SetSize(dim);
   CalcAdjugateTranspose(J, Z);

   double dJ = J.Det();
   double a  = mu*pow(dJ, -2.0/dim);
   double b  = K*(dJ/g - 1.0)/g - a*(J*J)/(dim*dJ);

   P = 0.0;
   P.Add(a, J);
   P.Add(b, Z);
}

void NeoHookeanModel::AssembleH(const DenseMatrix &J, const DenseMatrix &DS,
                                const double weight, DenseMatrix &A) const
{
   int dof = DS.Height(), dim = DS.Width();

   if (have_coeffs)
   {
      EvalCoeffs();
   }

   Z.SetSize(dim);
   G.SetSize(dof, dim);
   C.SetSize(dof, dim);

   double dJ = J.Det();
   double sJ = dJ/g;
   double a  = mu*pow(dJ, -2.0/dim);
   double bc = a*(J*J)/dim;
   double b  = bc - K*sJ*(sJ - 1.0);
   double c  = 2.0*bc/dim + K*sJ*(2.0*sJ - 1.0);

   CalcAdjugateTranspose(J, Z);
   Z *= (1.0/dJ); // Z = J^{-t}

   MultABt(DS, J, C); // C = DS J^t
   MultABt(DS, Z, G); // G = DS J^{-1}

   a *= weight;
   b *= weight;
   c *= weight;

   // 1.
   for (int i = 0; i < dof; i++)
      for (int k = 0; k <= i; k++)
      {
         double s = 0.0;
         for (int d = 0; d < dim; d++)
         {
            s += DS(i,d)*DS(k,d);
         }
         s *= a;

         for (int d = 0; d < dim; d++)
         {
            A(i+d*dof,k+d*dof) += s;
         }

         if (k != i)
            for (int d = 0; d < dim; d++)
            {
               A(k+d*dof,i+d*dof) += s;
            }
      }

   a *= (-2.0/dim);

   // 2.
   for (int i = 0; i < dof; i++)
      for (int j = 0; j < dim; j++)
         for (int k = 0; k < dof; k++)
            for (int l = 0; l < dim; l++)
            {
               A(i+j*dof,k+l*dof) +=
                  a*(C(i,j)*G(k,l) + G(i,j)*C(k,l)) +
                  b*G(i,l)*G(k,j) + c*G(i,j)*G(k,l);
            }
}


double HyperelasticNLFIntegrator::GetElementEnergy(const FiniteElement &el,
                                                   ElementTransformation &Ttr,
                                                   const Vector &elfun)
{
   int dof = el.GetDof(), dim = el.GetDim();
   double energy;

   DSh.SetSize(dof, dim);
   Jrt.SetSize(dim);
   Jpr.SetSize(dim);
   Jpt.SetSize(dim);
   PMatI.UseExternalData(elfun.GetData(), dof, dim);

   const IntegrationRule *ir = IntRule;
   if (!ir)
   {
      ir = &(IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + 3)); // <---
   }

   energy = 0.0;
   model->SetTransformation(Ttr);
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Ttr.SetIntPoint(&ip);
      CalcInverse(Ttr.Jacobian(), Jrt);

      el.CalcDShape(ip, DSh);
      MultAtB(PMatI, DSh, Jpr);
      Mult(Jpr, Jrt, Jpt);

      energy += ip.weight * Ttr.Weight() * model->EvalW(Jpt);
   }

   return energy;
}

void HyperelasticNLFIntegrator::AssembleElementVector(
   const FiniteElement &el, ElementTransformation &Ttr,
   const Vector &elfun, Vector &elvect)
{
   int dof = el.GetDof(), dim = el.GetDim();

   DSh.SetSize(dof, dim);
   DS.SetSize(dof, dim);
   Jrt.SetSize(dim);
   Jpt.SetSize(dim);
   P.SetSize(dim);
   PMatI.UseExternalData(elfun.GetData(), dof, dim);
   elvect.SetSize(dof*dim);
   PMatO.UseExternalData(elvect.GetData(), dof, dim);

   const IntegrationRule *ir = IntRule;
   if (!ir)
   {
      ir = &(IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + 3)); // <---
   }

   elvect = 0.0;
   model->SetTransformation(Ttr);
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Ttr.SetIntPoint(&ip);
      CalcInverse(Ttr.Jacobian(), Jrt);

      el.CalcDShape(ip, DSh);
      Mult(DSh, Jrt, DS);
      MultAtB(PMatI, DS, Jpt);

      model->EvalP(Jpt, P);

      P *= ip.weight * Ttr.Weight();
      AddMultABt(DS, P, PMatO);
   }
}

void HyperelasticNLFIntegrator::AssembleElementGrad(const FiniteElement &el,
                                                    ElementTransformation &Ttr,
                                                    const Vector &elfun,
                                                    DenseMatrix &elmat)
{
   int dof = el.GetDof(), dim = el.GetDim();

   DSh.SetSize(dof, dim);
   DS.SetSize(dof, dim);
   Jrt.SetSize(dim);
   Jpt.SetSize(dim);
   PMatI.UseExternalData(elfun.GetData(), dof, dim);
   elmat.SetSize(dof*dim);

   const IntegrationRule *ir = IntRule;
   if (!ir)
   {
      ir = &(IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + 3)); // <---
   }

   elmat = 0.0;
   model->SetTransformation(Ttr);
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Ttr.SetIntPoint(&ip);
      CalcInverse(Ttr.Jacobian(), Jrt);

      el.CalcDShape(ip, DSh);
      Mult(DSh, Jrt, DS);
      MultAtB(PMatI, DS, Jpt);

      model->AssembleH(Jpt, DS, ip.weight * Ttr.Weight(), elmat);
   }
}

double IncompressibleNeoHookeanIntegrator::GetElementEnergy(
   const Array<const FiniteElement *>&el,
   ElementTransformation &Tr,
   const Array<const Vector *>&elfun)
{
   if (el.Size() != 2)
   {
      mfem_error("IncompressibleNeoHookeanIntegrator::GetElementEnergy"
                 " has incorrect block finite element space size!");
   }

   int dof_u = el[0]->GetDof();
   int dim = el[0]->GetDim();

   DSh_u.SetSize(dof_u, dim);
   J0i.SetSize(dim);
   J1.SetSize(dim);
   J.SetSize(dim);
   PMatI_u.UseExternalData(elfun[0]->GetData(), dof_u, dim);

   int intorder = 2*el[0]->GetOrder() + 3; // <---
   const IntegrationRule &ir = IntRules.Get(el[0]->GetGeomType(), intorder);

   double energy = 0.0;
   double mu = 0.0;

   for (int i = 0; i < ir.GetNPoints(); ++i)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      Tr.SetIntPoint(&ip);
      CalcInverse(Tr.Jacobian(), J0i);

      el[0]->CalcDShape(ip, DSh_u);
      MultAtB(PMatI_u, DSh_u, J1);
      Mult(J1, J0i, J);

      mu = c_mu->Eval(Tr, ip);

      energy += ip.weight*Tr.Weight()*(mu/2.0)*(J*J - 3);
   }

   return energy;
}

void IncompressibleNeoHookeanIntegrator::AssembleElementVector(
   const Array<const FiniteElement *> &el,
   ElementTransformation &Tr,
   const Array<const Vector *> &elfun,
   const Array<Vector *> &elvec)
{
   if (el.Size() != 2)
   {
      mfem_error("IncompressibleNeoHookeanIntegrator::AssembleElementVector"
                 " has finite element space of incorrect block number");
   }

   int dof_u = el[0]->GetDof();
   int dof_p = el[1]->GetDof();

   int dim = el[0]->GetDim();
   int spaceDim = Tr.GetSpaceDim();

   if (dim != spaceDim)
   {
      mfem_error("IncompressibleNeoHookeanIntegrator::AssembleElementVector"
                 " is not defined on manifold meshes");
   }


   DSh_u.SetSize(dof_u, dim);
   DS_u.SetSize(dof_u, dim);
   J0i.SetSize(dim);
   F.SetSize(dim);
   FinvT.SetSize(dim);
   P.SetSize(dim);
   PMatI_u.UseExternalData(elfun[0]->GetData(), dof_u, dim);
   elvec[0]->SetSize(dof_u*dim);
   PMatO_u.UseExternalData(elvec[0]->GetData(), dof_u, dim);

   Sh_p.SetSize(dof_p);
   elvec[1]->SetSize(dof_p);

   int intorder = 2*el[0]->GetOrder() + 3; // <---
   const IntegrationRule &ir = IntRules.Get(el[0]->GetGeomType(), intorder);

   *elvec[0] = 0.0;
   *elvec[1] = 0.0;

   for (int i = 0; i < ir.GetNPoints(); ++i)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      Tr.SetIntPoint(&ip);
      CalcInverse(Tr.Jacobian(), J0i);

      el[0]->CalcDShape(ip, DSh_u);
      Mult(DSh_u, J0i, DS_u);
      MultAtB(PMatI_u, DS_u, F);

      el[1]->CalcShape(ip, Sh_p);

      double pres = Sh_p * *elfun[1];
      double mu = c_mu->Eval(Tr, ip);
      double dJ = F.Det();

      CalcInverseTranspose(F, FinvT);

      P = 0.0;
      P.Add(mu * dJ, F);
      P.Add(-1.0 * pres * dJ, FinvT);
      P *= ip.weight*Tr.Weight();

      AddMultABt(DS_u, P, PMatO_u);

      elvec[1]->Add(ip.weight * Tr.Weight() * (dJ - 1.0), Sh_p);
   }

}

void IncompressibleNeoHookeanIntegrator::AssembleElementGrad(
   const Array<const FiniteElement*> &el,
   ElementTransformation &Tr,
   const Array<const Vector *> &elfun,
   const Array2D<DenseMatrix *> &elmats)
{
   int dof_u = el[0]->GetDof();
   int dof_p = el[1]->GetDof();

   int dim = el[0]->GetDim();

   elmats(0,0)->SetSize(dof_u*dim, dof_u*dim);
   elmats(0,1)->SetSize(dof_u*dim, dof_p);
   elmats(1,0)->SetSize(dof_p, dof_u*dim);
   elmats(1,1)->SetSize(dof_p, dof_p);

   *elmats(0,0) = 0.0;
   *elmats(0,1) = 0.0;
   *elmats(1,0) = 0.0;
   *elmats(1,1) = 0.0;

   DSh_u.SetSize(dof_u, dim);
   DS_u.SetSize(dof_u, dim);
   J0i.SetSize(dim);
   F.SetSize(dim);
   FinvT.SetSize(dim);
   Finv.SetSize(dim);
   P.SetSize(dim);
   PMatI_u.UseExternalData(elfun[0]->GetData(), dof_u, dim);
   Sh_p.SetSize(dof_p);

   int intorder = 2*el[0]->GetOrder() + 3; // <---
   const IntegrationRule &ir = IntRules.Get(el[0]->GetGeomType(), intorder);

   for (int i = 0; i < ir.GetNPoints(); ++i)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      Tr.SetIntPoint(&ip);
      CalcInverse(Tr.Jacobian(), J0i);

      el[0]->CalcDShape(ip, DSh_u);
      Mult(DSh_u, J0i, DS_u);
      MultAtB(PMatI_u, DS_u, F);

      el[1]->CalcShape(ip, Sh_p);
      double pres = Sh_p * *elfun[1];
      double mu = c_mu->Eval(Tr, ip);
      double dJ = F.Det();
      double dJ_FinvT_DS;

      CalcInverseTranspose(F, FinvT);

      // u,u block
      for (int i_u = 0; i_u < dof_u; ++i_u)
      {
         for (int i_dim = 0; i_dim < dim; ++i_dim)
         {
            for (int j_u = 0; j_u < dof_u; ++j_u)
            {
               for (int j_dim = 0; j_dim < dim; ++j_dim)
               {

                  // m = j_dim;
                  // k = i_dim;

                  for (int n=0; n<dim; ++n)
                  {
                     for (int l=0; l<dim; ++l)
                     {
                        (*elmats(0,0))(i_u + i_dim*dof_u, j_u + j_dim*dof_u) +=
                           dJ * (mu * F(i_dim, l) - pres * FinvT(i_dim,l)) *
                           FinvT(j_dim,n) * DS_u(i_u,l) * DS_u(j_u, n) *
                           ip.weight * Tr.Weight();

                        if (j_dim == i_dim && n==l)
                        {
                           (*elmats(0,0))(i_u + i_dim*dof_u, j_u + j_dim*dof_u) +=
                              dJ * mu * DS_u(i_u, l) * DS_u(j_u,n) *
                              ip.weight * Tr.Weight();
                        }

                        // a = n;
                        // b = m;
                        (*elmats(0,0))(i_u + i_dim*dof_u, j_u + j_dim*dof_u) +=
                           dJ * pres * FinvT(i_dim, n) *
                           FinvT(j_dim,l) * DS_u(i_u,l) * DS_u(j_u,n) *
                           ip.weight * Tr.Weight();
                     }
                  }
               }
            }
         }
      }

      // u,p and p,u blocks
      for (int i_p = 0; i_p < dof_p; ++i_p)
      {
         for (int j_u = 0; j_u < dof_u; ++j_u)
         {
            for (int dim_u = 0; dim_u < dim; ++dim_u)
            {
               for (int l=0; l<dim; ++l)
               {
                  dJ_FinvT_DS = dJ * FinvT(dim_u,l) * DS_u(j_u, l) * Sh_p(i_p) *
                                ip.weight * Tr.Weight();
                  (*elmats(1,0))(i_p, j_u + dof_u * dim_u) += dJ_FinvT_DS;
                  (*elmats(0,1))(j_u + dof_u * dim_u, i_p) -= dJ_FinvT_DS;

               }
            }
         }
      }
   }

}

const IntegrationRule&
VectorConvectionNLFIntegrator::GetRule(const FiniteElement &fe,
                                       ElementTransformation &T)
{
   const int order = 2 * fe.GetOrder() + T.OrderGrad(&fe);
   return IntRules.Get(fe.GetGeomType(), order);
}

void VectorConvectionNLFIntegrator::AssembleElementVector(
   const FiniteElement &el,
   ElementTransformation &T,
   const Vector &elfun,
   Vector &elvect)
{
   const int nd = el.GetDof();
   const int dim = el.GetDim();

   shape.SetSize(nd);
   dshape.SetSize(nd, dim);
   elvect.SetSize(nd * dim);
   gradEF.SetSize(dim);

   EF.UseExternalData(elfun.GetData(), nd, dim);
   ELV.UseExternalData(elvect.GetData(), nd, dim);

   Vector vec1(dim), vec2(dim);
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, T);
   ELV = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      T.SetIntPoint(&ip);
      el.CalcShape(ip, shape);
      el.CalcPhysDShape(T, dshape);
      double w = ip.weight * T.Weight();
      if (Q) { w *= Q->Eval(T, ip); }
      MultAtB(EF, dshape, gradEF);
      EF.MultTranspose(shape, vec1);
      gradEF.Mult(vec1, vec2);
      vec2 *= w;
      AddMultVWt(shape, vec2, ELV);
   }
}

void VectorConvectionNLFIntegrator::AssembleElementGrad(
   const FiniteElement &el,
   ElementTransformation &trans,
   const Vector &elfun,
   DenseMatrix &elmat)
{
   int nd = el.GetDof();
   int dim = el.GetDim();

   shape.SetSize(nd);
   dshape.SetSize(nd, dim);
   dshapex.SetSize(nd, dim);
   elmat.SetSize(nd * dim);
   elmat_comp.SetSize(nd);
   gradEF.SetSize(dim);

   EF.UseExternalData(elfun.GetData(), nd, dim);

   double w;
   Vector vec1(dim), vec2(dim), vec3(nd);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      el.CalcShape(ip, shape);
      el.CalcDShape(ip, dshape);

      Mult(dshape, trans.InverseJacobian(), dshapex);

      w = ip.weight;

      if (Q)
      {
         w *= Q->Eval(trans, ip);
      }

      MultAtB(EF, dshapex, gradEF);
      EF.MultTranspose(shape, vec1);

      trans.AdjugateJacobian().Mult(vec1, vec2);

      vec2 *= w;
      dshape.Mult(vec2, vec3);
      MultVWt(shape, vec3, elmat_comp);

      for (int i = 0; i < dim; i++)
      {
         elmat.AddMatrix(elmat_comp, i * nd, i * nd);
      }

      MultVVt(shape, elmat_comp);
      w = ip.weight * trans.Weight();
      if (Q)
      {
         w *= Q->Eval(trans, ip);
      }
      for (int i = 0; i < dim; i++)
      {
         for (int j = 0; j < dim; j++)
         {
            elmat.AddMatrix(w * gradEF(i, j), elmat_comp, i * nd, j * nd);
         }
      }
   }
}

}
