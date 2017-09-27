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

namespace mfem
{

void NonlinearFormIntegrator::AssembleElementGrad(
   const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun,
   DenseMatrix &elmat)
{
   mfem_error("NonlinearFormIntegrator::AssembleElementGrad"
              " is not overloaded!");
}

double NonlinearFormIntegrator::GetElementEnergy(
   const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun)
{
   mfem_error("NonlinearFormIntegrator::GetElementEnergy"
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
   for (int i = 0; i < dof; i++)
      for (int j = 0; j < i; j++)
      {
         for (int k = 0; k < dim; k++)
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
   mu = c_mu->Eval(*T, T->GetIntPoint());
   K = c_K->Eval(*T, T->GetIntPoint());
   if (c_g)
   {
      g = c_g->Eval(*T, T->GetIntPoint());
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
                                                   ElementTransformation &Tr,
                                                   const Vector &elfun)
{
   int dof = el.GetDof(), dim = el.GetDim();
   double energy;

   DSh.SetSize(dof, dim);
   J0i.SetSize(dim);
   J1.SetSize(dim);
   J.SetSize(dim);
   PMatI.UseExternalData(elfun.GetData(), dof, dim);

   int intorder = 2*el.GetOrder() + 3; // <---
   const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), intorder);

   energy = 0.0;
   model->SetTransformation(Tr);
   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      Tr.SetIntPoint(&ip);
      CalcInverse(Tr.Jacobian(), J0i);

      el.CalcDShape(ip, DSh);
      MultAtB(PMatI, DSh, J1);
      Mult(J1, J0i, J);

      energy += ip.weight*Tr.Weight()*model->EvalW(J);
   }

   return energy;
}

void HyperelasticNLFIntegrator::AssembleElementVector(
   const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun,
   Vector &elvect)
{
   int dof = el.GetDof(), dim = el.GetDim();

   DSh.SetSize(dof, dim);
   DS.SetSize(dof, dim);
   J0i.SetSize(dim);
   J.SetSize(dim);
   P.SetSize(dim);
   PMatI.UseExternalData(elfun.GetData(), dof, dim);
   elvect.SetSize(dof*dim);
   PMatO.UseExternalData(elvect.GetData(), dof, dim);

   int intorder = 2*el.GetOrder() + 3; // <---
   const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), intorder);

   elvect = 0.0;
   model->SetTransformation(Tr);
   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      Tr.SetIntPoint(&ip);
      CalcInverse(Tr.Jacobian(), J0i);

      el.CalcDShape(ip, DSh);
      Mult(DSh, J0i, DS);
      MultAtB(PMatI, DS, J);

      model->EvalP(J, P);
 
      P *= ip.weight*Tr.Weight();
      AddMultABt(DS, P, PMatO);
   }
}

void HyperelasticNLFIntegrator::AssembleElementGrad(
   const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun,
   DenseMatrix &elmat)
{
   int dof = el.GetDof(), dim = el.GetDim();

   DSh.SetSize(dof, dim);
   DS.SetSize(dof, dim);
   J0i.SetSize(dim);
   J.SetSize(dim);
   PMatI.UseExternalData(elfun.GetData(), dof, dim);
   elmat.SetSize(dof*dim);

   int intorder = 2*el.GetOrder() + 3; // <---
   const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), intorder);

   elmat = 0.0;
   model->SetTransformation(Tr);
   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      Tr.SetIntPoint(&ip);
      CalcInverse(Tr.Jacobian(), J0i);

      el.CalcDShape(ip, DSh);
      Mult(DSh, J0i, DS);
      MultAtB(PMatI, DS, J);

      model->AssembleH(J, DS, ip.weight*Tr.Weight(), elmat);
   }
}

HyperelasticNLFIntegrator::~HyperelasticNLFIntegrator()
{
   PMatI.ClearExternalData();
   PMatO.ClearExternalData();
}

double BlockNonlinearFormIntegrator::GetElementEnergy(Array<const FiniteElement *>&el,
                                                      ElementTransformation &Tr,
                                                      Array<const Vector *>&elfun)
{
   mfem_error("BlockNonlinearFormIntegrator::GetElementEnergy"
              " is not overloaded!");

   return 0.0;
}



void BlockNonlinearFormIntegrator::AssembleElementVector(Array<const FiniteElement *> &el,
                                                         ElementTransformation &Tr,
                                                         Array<Vector *> &elfun, 
                                                         Array<Vector *> &elvec)
{
   mfem_error("BlockNonlinearFormIntegrator::AssembleElementVector"
              " is not overloaded!");

}

void BlockNonlinearFormIntegrator::AssembleRHSElementVector(Array<const FiniteElement *> &el,
                                         FaceElementTransformations &Tr,
                                         Array<Vector *> &elfun, 
                                         Array<Vector *> &elvec)
{
   mfem_error("BlockNonlinearFormIntegrator::AssembleRHSElementVector"
              " is not overloaded!");
}

void BlockNonlinearFormIntegrator::AssembleElementGrad(Array<const FiniteElement*> &el,
                                    ElementTransformation &Tr,
                                    Array<Vector *> &elfun, 
                                    Array2D<DenseMatrix *> &elmats)
{
   double diff_step = 1.0e-8;
   Array<Vector *> temps(el.Size());
   Array<Vector *> temp_out_1(el.Size());
   Array<Vector *> temp_out_2(el.Size());
   Array<int> dofs(el.Size());

   for (int s1=0; s1<el.Size(); s1++) {
      temps[s1] = new Vector(elfun[s1]->GetData(), elfun[s1]->Size());
      temp_out_1[s1] = new Vector();
      temp_out_2[s1] = new Vector();
      dofs[s1] = elfun[s1]->Size();
   }

   for (int s1=0; s1<el.Size(); s1++) {
      for (int s2=0; s2<el.Size(); s2++) {
         elmats(s1,s2)->SetSize(dofs[s1],dofs[s2]);
      }
   }
   
   for (int s1=0; s1<el.Size(); s1++) {
      for (int j=0; j<temps[s1]->Size(); j++) {
         (*temps[s1])[j] += diff_step;
         AssembleElementVector(el, Tr, temps, temp_out_1);
         (*temps[s1])[j] -= 2.0*diff_step;
         AssembleElementVector(el, Tr, temps, temp_out_2);

         for (int s2=0; s2<el.Size(); s2++) {
            for (int k=0; k<temps[s2]->Size(); k++) {
               (*elmats(s2,s1))(k,j) = ((*temp_out_1[s2])[k] - (*temp_out_2[s2])[k]) / (2.0*diff_step);
            }
         }
         (*temps[s1])[j] = (*elfun[s1])[j];
      }
   }

}

void BlockNonlinearFormIntegrator::AssembleRHSElementGrad(Array<const FiniteElement*> &el,
                                       FaceElementTransformations &Tr,
                                       Array<Vector *> &elfun, 
                                       Array2D<DenseMatrix *> &elmats)
{
   mfem_error("BlockNonlinearFormIntegrator::AssembleRHSElementGrad"
              " is not overloaded!");
}

double IncompressibleNeoHookeanIntegrator::GetElementEnergy(Array<const FiniteElement *>&el,
                                                            ElementTransformation &Tr,
                                                            Array<const Vector *>&elfun)
{
   if (el.Size() != 2) {
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

   for (int i = 0; i < ir.GetNPoints(); i++)
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


void IncompressibleNeoHookeanIntegrator::AssembleElementVector(Array<const FiniteElement *> &el,
                                                         ElementTransformation &Tr,
                                                         Array<Vector *> &elfun, 
                                                         Array<Vector *> &elvec)
{
   if (el.Size() != 2) {
      mfem_error("IncompressibleNeoHookeanIntegrator::AssembleElementVector"
                 " has finite element space of incorrect block number");
   }
   
   int dof_u = el[0]->GetDof();
   int dof_p = el[1]->GetDof();

   int dim = el[0]->GetDim();

   DSh_u.SetSize(dof_u, dim);
   DS_u.SetSize(dof_u, dim);
   J0i.SetSize(dim);
   J.SetSize(dim);
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

   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      Tr.SetIntPoint(&ip);
      CalcInverse(Tr.Jacobian(), J0i);

      el[0]->CalcDShape(ip, DSh_u);
      Mult(DSh_u, J0i, DS_u);
      MultAtB(PMatI_u, DS_u, J);

      el[1]->CalcShape(ip, Sh_p);
    
      double pres = Sh_p * *elfun[1];
      double mu = c_mu->Eval(Tr, ip);      
      double dJ = J.Det();

      F.Transpose(J);
      CalcInverseTranspose(F, FinvT);

      P = 0.0;
      P.Add(mu * dJ, F);
      P.Add(-1.0 * pres * dJ, FinvT);
      P *= ip.weight*Tr.Weight();

      P.Transpose();
      AddMultABt(DS_u, P, PMatO_u);
      
      elvec[1]->Add(ip.weight * Tr.Weight() * (dJ - 1.0), Sh_p);
   }

}

void IncompressibleNeoHookeanIntegrator::AssembleElementGrad(Array<const FiniteElement*> &el,
                                    ElementTransformation &Tr,
                                    Array<Vector *> &elfun, 
                                    Array2D<DenseMatrix *> &elmats)
{

}

}
