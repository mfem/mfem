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

void NonlinearFormIntegrator::Setup(const FiniteElementSpace&)
{
   mfem_error ("NonlinearFormIntegrator::Setup(...)\n"
               "   is not implemented for this class.");
}

void NonlinearFormIntegrator::Setup(const FiniteElementSpace&,
                                    const FiniteElementSpace&)
{
   mfem_error ("NonlinearFormIntegrator::SetupAssembly(...)\n"
               "   is not implemented for this class.");
}

void NonlinearFormIntegrator::AddMultPA(const Vector &, Vector &) const
{
   mfem_error ("NonlinearFormIntegrator::AddMultPA(...)\n"
               "   is not implemented for this class.");
}

void NonlinearFormIntegrator::AddMultTransposePA(const Vector &, Vector &) const
{
   mfem_error ("NonlinearFormIntegrator::AddMultTransposePA(...)\n"
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

void VectorConvectionNLFIntegrator::Setup(const FiniteElementSpace &fes)
{
   MFEM_ASSERT(fes.GetOrdering() == Ordering::byNODES,
               "PA Only supports Ordering::byNODES!");
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el = *fes.GetFE(0);
   ElementTransformation &T = *mesh->GetElementTransformation(0);
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, T);
   dim = mesh->Dimension();
   ne = fes.GetMesh()->GetNE();
   nq = ir->GetNPoints();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   pa_data.SetSize(ne*nq*dim*dim, Device::GetMemoryType());
   double COEFF = 1.0;
   if (Q)
   {
      ConstantCoefficient *cQ = dynamic_cast<ConstantCoefficient*>(Q);
      MFEM_VERIFY(cQ != NULL, "only ConstantCoefficient is supported!");
      COEFF = cQ->constant;
   }
   const int NE = ne;
   const int NQ = nq;
   auto W = ir->GetWeights().Read();
   if (dim==1) { MFEM_ABORT("dim==1 not supported!"); }
   if (dim==2)
   {
      auto J = Reshape(geom->J.Read(), NQ, 2, 2,NE);
      auto G = Reshape(pa_data.Write(), NQ, 2, 2, NE);
      MFEM_FORALL(e, NE,
      {
         for (int q = 0; q < NQ; ++q)
         {
            const double J11 = J(q,0,0,e);
            const double J12 = J(q,0,1,e);
            const double J21 = J(q,1,0,e);
            const double J22 = J(q,1,1,e);
            // Store wq * Q * adj(J)
            G(q,0,0,e) = W[q] * COEFF *  J22; // 1,1
            G(q,0,1,e) = W[q] * COEFF * -J12; // 1,2
            G(q,1,0,e) = W[q] * COEFF * -J21; // 2,1
            G(q,1,1,e) = W[q] * COEFF *  J11; // 2,2
         }
      });
   }
   if (dim==3)
   {
      auto J = Reshape(geom->J.Read(), NQ, 3, 3,NE);
      auto G = Reshape(pa_data.Write(), NQ, 3, 3, NE);
      MFEM_FORALL(e, NE,
      {
         for (int q = 0; q < NQ; ++q)
         {
            const double J11 = J(q,0,0,e);
            const double J21 = J(q,1,0,e);
            const double J31 = J(q,2,0,e);
            const double J12 = J(q,0,1,e);
            const double J22 = J(q,1,1,e);
            const double J32 = J(q,2,1,e);
            const double J13 = J(q,0,2,e);
            const double J23 = J(q,1,2,e);
            const double J33 = J(q,2,2,e);
            const double cw  = W[q] * COEFF;
            // adj(J)
            const double A11 = (J22 * J33) - (J23 * J32);
            const double A12 = (J32 * J13) - (J12 * J33);
            const double A13 = (J12 * J23) - (J22 * J13);
            const double A21 = (J31 * J23) - (J21 * J33);
            const double A22 = (J11 * J33) - (J13 * J31);
            const double A23 = (J21 * J13) - (J11 * J23);
            const double A31 = (J21 * J32) - (J31 * J22);
            const double A32 = (J31 * J12) - (J11 * J32);
            const double A33 = (J11 * J22) - (J12 * J21);
            // Store wq * Q * adj(J)
            G(q,0,0,e) = cw * A11; // 1,1
            G(q,0,1,e) = cw * A12; // 1,2
            G(q,0,2,e) = cw * A13; // 1,3
            G(q,1,0,e) = cw * A21; // 2,1
            G(q,1,1,e) = cw * A22; // 2,2
            G(q,1,2,e) = cw * A23; // 2,3
            G(q,2,0,e) = cw * A31; // 3,1
            G(q,2,1,e) = cw * A32; // 3,2
            G(q,2,2,e) = cw * A33; // 3,3
         }
      });
   }
}

// PA Convection NL 2D kernel
template<int T_D1D = 0, int T_Q1D = 0> static
void PAConvectionNLApply2D(const int NE,
                           const Array<double> &b,
                           const Array<double> &g,
                           const Array<double> &bt,
                           const Vector &q_,
                           const Vector &x_,
                           Vector &y_,
                           const int d1d = 0,
                           const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   auto Bt = Reshape(bt.Read(), D1D, Q1D);
   auto Q = Reshape(q_.Read(), Q1D*Q1D, 2, 2, NE);
   auto x = Reshape(x_.Read(), D1D, D1D, 2, NE);
   auto y = Reshape(y_.ReadWrite(), D1D, D1D, 2, NE);
   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;

      double data[max_Q1D][max_Q1D][2];
      double grad0[max_Q1D][max_Q1D][2];
      double grad1[max_Q1D][max_Q1D][2];
      double Z[max_Q1D][max_Q1D][2];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            data[qy][qx][0] = 0.0;
            data[qy][qx][1] = 0.0;
            grad0[qy][qx][0] = 0.0;
            grad0[qy][qx][1] = 0.0;
            grad1[qy][qx][0] = 0.0;
            grad1[qy][qx][1] = 0.0;
         }
      }
      for (int dy = 0; dy < D1D; ++dy)
      {
         double dataX[max_Q1D][2];
         double gradX0[max_Q1D][2];
         double gradX1[max_Q1D][2];
         for (int qx = 0; qx < Q1D; ++qx)
         {
            dataX[qx][0] = 0.0;
            dataX[qx][1] = 0.0;
            gradX0[qx][0] = 0.0;
            gradX0[qx][1] = 0.0;
            gradX1[qx][0] = 0.0;
            gradX1[qx][1] = 0.0;
         }
         for (int dx = 0; dx < D1D; ++dx)
         {
            const double s0 = x(dx,dy,0,e);
            const double s1 = x(dx,dy,1,e);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double Bx = B(qx,dx);
               const double Gx = G(qx,dx);
               dataX[qx][0] += s0 * Bx;
               dataX[qx][1] += s1 * Bx;
               gradX0[qx][0] += s0 * Gx;
               gradX0[qx][1] += s0 * Bx;
               gradX1[qx][0] += s1 * Gx;
               gradX1[qx][1] += s1 * Bx;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const double By = B(qy,dy);
            const double Gy = G(qy,dy);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               data[qy][qx][0] += dataX[qx][0] * By;
               data[qy][qx][1] += dataX[qx][1] * By;
               grad0[qy][qx][0] += gradX0[qx][0] * By;
               grad0[qy][qx][1] += gradX0[qx][1] * Gy;
               grad1[qy][qx][0] += gradX1[qx][0] * By;
               grad1[qy][qx][1] += gradX1[qx][1] * Gy;
            }
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const int q = qx + qy * Q1D;
            const double u1 = data[qy][qx][0];
            const double u2 = data[qy][qx][1];
            const double grad00 = grad0[qy][qx][0];
            const double grad01 = grad0[qy][qx][1];
            const double grad10 = grad1[qy][qx][0];
            const double grad11 = grad1[qy][qx][1];
            const double Dxu1 = grad00*Q(q,0,0,e) + grad01*Q(q,1,0,e);
            const double Dyu1 = grad00*Q(q,0,1,e) + grad01*Q(q,1,1,e);
            const double Dxu2 = grad10*Q(q,0,0,e) + grad11*Q(q,1,0,e);
            const double Dyu2 = grad10*Q(q,0,1,e) + grad11*Q(q,1,1,e);
            Z[qy][qx][0] = u1 * Dxu1 + u2 * Dyu1;
            Z[qy][qx][1] = u1 * Dxu2 + u2 * Dyu2;
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         double Y[max_D1D][2];
         for (int dx = 0; dx < D1D; ++dx)
         {
            Y[dx][0] = 0.0;
            Y[dx][1] = 0.0;
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double Btx = Bt(dx,qx);
               Y[dx][0] += Btx * Z[qy][qx][0];
               Y[dx][1] += Btx * Z[qy][qx][1];
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double Bty = Bt(dy,qy);
               y(dx,dy,0,e) += Bty * Y[dx][0];
               y(dx,dy,1,e) += Bty * Y[dx][1];
            }
         }
      }
   });
}

// PA Convection NL 3D kernel
template<int T_D1D = 0, int T_Q1D = 0> static
void PAConvectionNLApply3D(const int NE,
                           const Array<double> &b,
                           const Array<double> &g,
                           const Array<double> &bt,
                           const Vector &q_,
                           const Vector &x_,
                           Vector &y_,
                           const int d1d = 0,
                           const int q1d = 0)
{
   constexpr int VDIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");

   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   auto Bt = Reshape(bt.Read(), D1D, Q1D);
   auto Q = Reshape(q_.Read(), Q1D*Q1D*Q1D, VDIM, VDIM, NE);
   auto x = Reshape(x_.Read(), D1D, D1D, D1D, VDIM, NE);
   auto y = Reshape(y_.ReadWrite(), D1D, D1D, D1D, VDIM, NE);

   MFEM_FORALL(e, NE,
   {
      constexpr int VDIM = 3;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;

      double data[max_Q1D][max_Q1D][max_Q1D][VDIM];
      double grad0[max_Q1D][max_Q1D][max_Q1D][VDIM];
      double grad1[max_Q1D][max_Q1D][max_Q1D][VDIM];
      double grad2[max_Q1D][max_Q1D][max_Q1D][VDIM];
      double Z[max_Q1D][max_Q1D][max_Q1D][VDIM];
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               data[qz][qy][qx][0] = 0.0;
               data[qz][qy][qx][1] = 0.0;
               data[qz][qy][qx][2] = 0.0;

               grad0[qz][qy][qx][0] = 0.0;
               grad0[qz][qy][qx][1] = 0.0;
               grad0[qz][qy][qx][2] = 0.0;

               grad1[qz][qy][qx][0] = 0.0;
               grad1[qz][qy][qx][1] = 0.0;
               grad1[qz][qy][qx][2] = 0.0;

               grad2[qz][qy][qx][0] = 0.0;
               grad2[qz][qy][qx][1] = 0.0;
               grad2[qz][qy][qx][2] = 0.0;
            }
         }
      }
      for (int dz = 0; dz < D1D; ++dz)
      {
         double dataXY[max_Q1D][max_Q1D][VDIM];
         double gradXY0[max_Q1D][max_Q1D][VDIM];
         double gradXY1[max_Q1D][max_Q1D][VDIM];
         double gradXY2[max_Q1D][max_Q1D][VDIM];
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               dataXY[qy][qx][0] = 0.0;
               dataXY[qy][qx][1] = 0.0;
               dataXY[qy][qx][2] = 0.0;

               gradXY0[qy][qx][0] = 0.0;
               gradXY0[qy][qx][1] = 0.0;
               gradXY0[qy][qx][2] = 0.0;

               gradXY1[qy][qx][0] = 0.0;
               gradXY1[qy][qx][1] = 0.0;
               gradXY1[qy][qx][2] = 0.0;

               gradXY2[qy][qx][0] = 0.0;
               gradXY2[qy][qx][1] = 0.0;
               gradXY2[qy][qx][2] = 0.0;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            double dataX[max_Q1D][VDIM];
            double gradX0[max_Q1D][VDIM];
            double gradX1[max_Q1D][VDIM];
            double gradX2[max_Q1D][VDIM];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               dataX[qx][0] = 0.0;
               dataX[qx][1] = 0.0;
               dataX[qx][2] = 0.0;

               gradX0[qx][0] = 0.0;
               gradX0[qx][1] = 0.0;
               gradX0[qx][2] = 0.0;

               gradX1[qx][0] = 0.0;
               gradX1[qx][1] = 0.0;
               gradX1[qx][2] = 0.0;

               gradX2[qx][0] = 0.0;
               gradX2[qx][1] = 0.0;
               gradX2[qx][2] = 0.0;
            }
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double s0 = x(dx,dy,dz,0,e);
               const double s1 = x(dx,dy,dz,1,e);
               const double s2 = x(dx,dy,dz,2,e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double Bx = B(qx,dx);
                  const double Gx = G(qx,dx);

                  dataX[qx][0] += s0 * Bx;
                  dataX[qx][1] += s1 * Bx;
                  dataX[qx][2] += s2 * Bx;

                  gradX0[qx][0] += s0 * Gx;
                  gradX0[qx][1] += s0 * Bx;
                  gradX0[qx][2] += s0 * Bx;

                  gradX1[qx][0] += s1 * Gx;
                  gradX1[qx][1] += s1 * Bx;
                  gradX1[qx][2] += s1 * Bx;

                  gradX2[qx][0] += s2 * Gx;
                  gradX2[qx][1] += s2 * Bx;
                  gradX2[qx][2] += s2 * Bx;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double By = B(qy,dy);
               const double Gy = G(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  dataXY[qy][qx][0] += dataX[qx][0] * By;
                  dataXY[qy][qx][1] += dataX[qx][1] * By;
                  dataXY[qy][qx][2] += dataX[qx][2] * By;

                  gradXY0[qy][qx][0] += gradX0[qx][0] * By;
                  gradXY0[qy][qx][1] += gradX0[qx][1] * Gy;
                  gradXY0[qy][qx][2] += gradX0[qx][2] * By;

                  gradXY1[qy][qx][0] += gradX1[qx][0] * By;
                  gradXY1[qy][qx][1] += gradX1[qx][1] * Gy;
                  gradXY1[qy][qx][2] += gradX1[qx][2] * By;

                  gradXY2[qy][qx][0] += gradX2[qx][0] * By;
                  gradXY2[qy][qx][1] += gradX2[qx][1] * Gy;
                  gradXY2[qy][qx][2] += gradX2[qx][2] * By;
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            const double Bz = B(qz,dz);
            const double Gz = G(qz,dz);
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  data[qz][qy][qx][0] += dataXY[qy][qx][0] * Bz;
                  data[qz][qy][qx][1] += dataXY[qy][qx][1] * Bz;
                  data[qz][qy][qx][2] += dataXY[qy][qx][2] * Bz;

                  grad0[qz][qy][qx][0] += gradXY0[qy][qx][0] * Bz;
                  grad0[qz][qy][qx][1] += gradXY0[qy][qx][1] * Bz;
                  grad0[qz][qy][qx][2] += gradXY0[qy][qx][2] * Gz;

                  grad1[qz][qy][qx][0] += gradXY1[qy][qx][0] * Bz;
                  grad1[qz][qy][qx][1] += gradXY1[qy][qx][1] * Bz;
                  grad1[qz][qy][qx][2] += gradXY1[qy][qx][2] * Gz;

                  grad2[qz][qy][qx][0] += gradXY2[qy][qx][0] * Bz;
                  grad2[qz][qy][qx][1] += gradXY2[qy][qx][1] * Bz;
                  grad2[qz][qy][qx][2] += gradXY2[qy][qx][2] * Gz;
               }
            }
         }
      }
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const int q = qx + Q1D * (qy + qz * Q1D);

               const double u1 = data[qz][qy][qx][0];
               const double u2 = data[qz][qy][qx][1];
               const double u3 = data[qz][qy][qx][2];

               const double grad00 = grad0[qz][qy][qx][0];
               const double grad01 = grad0[qz][qy][qx][1];
               const double grad02 = grad0[qz][qy][qx][2];

               const double grad10 = grad1[qz][qy][qx][0];
               const double grad11 = grad1[qz][qy][qx][1];
               const double grad12 = grad1[qz][qy][qx][2];

               const double grad20 = grad2[qz][qy][qx][0];
               const double grad21 = grad2[qz][qy][qx][1];
               const double grad22 = grad2[qz][qy][qx][2];

               const double Dxu1 = grad00*Q(q,0,0,e) + grad01*Q(q,1,0,e) + grad02*Q(q,2,0,e);
               const double Dyu1 = grad00*Q(q,0,1,e) + grad01*Q(q,1,1,e) + grad02*Q(q,2,1,e);
               const double Dzu1 = grad00*Q(q,0,2,e) + grad01*Q(q,1,2,e) + grad02*Q(q,2,2,e);

               const double Dxu2 = grad10*Q(q,0,0,e) + grad11*Q(q,1,0,e) + grad12*Q(q,2,0,e);
               const double Dyu2 = grad10*Q(q,0,1,e) + grad11*Q(q,1,1,e) + grad12*Q(q,2,1,e);
               const double Dzu2 = grad10*Q(q,0,2,e) + grad11*Q(q,1,2,e) + grad12*Q(q,2,2,e);

               const double Dxu3 = grad20*Q(q,0,0,e) + grad21*Q(q,1,0,e) + grad22*Q(q,2,0,e);
               const double Dyu3 = grad20*Q(q,0,1,e) + grad21*Q(q,1,1,e) + grad22*Q(q,2,1,e);
               const double Dzu3 = grad20*Q(q,0,2,e) + grad21*Q(q,1,2,e) + grad22*Q(q,2,2,e);

               Z[qz][qy][qx][0] = u1 * Dxu1 + u2 * Dyu1 + u3 * Dzu1;
               Z[qz][qy][qx][1] = u1 * Dxu2 + u2 * Dyu2 + u3 * Dzu2;
               Z[qz][qy][qx][2] = u1 * Dxu3 + u2 * Dyu3 + u3 * Dzu3;
            }
         }
      }
      for (int qz = 0; qz < Q1D; ++qz)
      {
         double opXY[max_D1D][max_D1D][VDIM];
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               opXY[dy][dx][0] = 0.0;
               opXY[dy][dx][1] = 0.0;
               opXY[dy][dx][2] = 0.0;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            double opX[max_D1D][VDIM];
            for (int dx = 0; dx < D1D; ++dx)
            {
               opX[dx][0] = 0.0;
               opX[dx][1] = 0.0;
               opX[dx][2] = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double Btx = Bt(dx,qx);
                  opX[dx][0] += Btx * Z[qz][qy][qx][0];
                  opX[dx][1] += Btx * Z[qz][qy][qx][1];
                  opX[dx][2] += Btx * Z[qz][qy][qx][2];
               }
            }
            for (int dy = 0; dy < D1D; ++dy)
            {
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const double Bty = Bt(dy,qy);
                  opXY[dy][dx][0] += Bty * opX[dx][0];
                  opXY[dy][dx][1] += Bty * opX[dx][1];
                  opXY[dy][dx][2] += Bty * opX[dx][2];
               }
            }
         }
         for (int dz = 0; dz < D1D; ++dz)
         {
            for (int dy = 0; dy < D1D; ++dy)
            {
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const double Btz = Bt(dz,qz);
                  y(dx,dy,dz,0,e) += Btz * opXY[dy][dx][0];
                  y(dx,dy,dz,1,e) += Btz * opXY[dy][dx][1];
                  y(dx,dy,dz,2,e) += Btz * opXY[dy][dx][2];
               }
            }
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0> static
void SmemPAConvectionNLApply3D(const int NE,
                               const Array<double> &b_,
                               const Array<double> &g_,
                               const Vector &d_,
                               const Vector &x_,
                               Vector &y_,
                               const int d1d = 0,
                               const int q1d = 0)
{
   constexpr int VDIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
   constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
   MFEM_VERIFY(D1D <= MD1, "");
   MFEM_VERIFY(Q1D <= MQ1, "");

   auto b = Reshape(b_.Read(), Q1D, D1D);
   auto g = Reshape(g_.Read(), Q1D, D1D);
   auto D = Reshape(d_.Read(), Q1D*Q1D*Q1D, VDIM, VDIM, NE);
   auto x = Reshape(x_.Read(), D1D, D1D, D1D, VDIM, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, D1D, VDIM, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      MFEM_SHARED double BG[2][MQ1*MD1];
      double (*B)[MD1] = (double (*)[MD1]) (BG+0);
      double (*G)[MD1] = (double (*)[MD1]) (BG+1);
      double (*Bt)[MQ1] = (double (*)[MQ1]) (BG+0);
      MFEM_SHARED double U[2][MQ1][MQ1][MQ1];
      MFEM_SHARED double sm0[3][MQ1*MQ1*MQ1];
      MFEM_SHARED double sm1[3][MQ1*MQ1*MQ1];
      double (*DDQ0)[MD1][MQ1] = (double (*)[MD1][MQ1]) (sm0+0);
      double (*DDQ1)[MD1][MQ1] = (double (*)[MD1][MQ1]) (sm0+1);
      double (*X)[MD1][MD1]    = (double (*)[MD1][MD1]) (sm0+2);
      double (*DQQ0)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+0);
      double (*DQQ1)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+1);
      double (*DQQ2)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+2);
      double (*QQQ0)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm0+0);
      double (*QQQ1)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm0+1);
      double (*QQQ2)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm0+2);
      double (*QQD0)[MQ1][MD1] = (double (*)[MQ1][MD1]) (sm1+0);
      double (*QDD0)[MD1][MD1] = (double (*)[MD1][MD1]) (sm0+0);
      MFEM_SHARED double Z[MQ1][MQ1][MQ1];

      for (int cy = 0; cy < VDIM; ++cy)
      {
         if (tidz == 0)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               MFEM_FOREACH_THREAD(d,y,D1D)
               {
                  B[q][d] = b(q,d);
                  G[q][d] = g(q,d);
               }
            }
         }
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  Z[qz][qy][qx] = 0.0;
               }
            }
         }
         MFEM_SYNC_THREAD;
         for (int c = 0; c < VDIM; ++ c)
         {
            MFEM_FOREACH_THREAD(dz,z,D1D)
            {
               MFEM_FOREACH_THREAD(dy,y,D1D)
               {
                  MFEM_FOREACH_THREAD(dx,x,D1D)
                  {
                     X[dz][dy][dx] = x(dx,dy,dz,cy,e);
                     U[0][dz][dy][dx] = x(dx,dy,dz,c,e);
                  }
               }
            }
            MFEM_SYNC_THREAD;
            MFEM_FOREACH_THREAD(dz,z,D1D)
            {
               MFEM_FOREACH_THREAD(dy,y,D1D)
               {
                  MFEM_FOREACH_THREAD(qx,x,Q1D)
                  {
                     double u = 0.0;
                     double v = 0.0;
                     double z = 0.0;
                     for (int dx = 0; dx < D1D; ++dx)
                     {
                        const double coord = X[dz][dy][dx];
                        const double value = U[0][dz][dy][dx];
                        u += coord * B[qx][dx];
                        v += coord * G[qx][dx];
                        z += value * B[qx][dx];
                     }
                     DDQ0[dz][dy][qx] = u;
                     DDQ1[dz][dy][qx] = v;
                     U[1][dz][dy][qx] = z;
                  }
               }
            }
            MFEM_SYNC_THREAD;
            MFEM_FOREACH_THREAD(dz,z,D1D)
            {
               MFEM_FOREACH_THREAD(qy,y,Q1D)
               {
                  MFEM_FOREACH_THREAD(qx,x,Q1D)
                  {
                     double u = 0.0;
                     double v = 0.0;
                     double w = 0.0;
                     double z = 0.0;
                     for (int dy = 0; dy < D1D; ++dy)
                     {
                        u += DDQ1[dz][dy][qx] * B[qy][dy];
                        v += DDQ0[dz][dy][qx] * G[qy][dy];
                        w += DDQ0[dz][dy][qx] * B[qy][dy];
                        z += U[1][dz][dy][qx] * B[qy][dy];
                     }
                     DQQ0[dz][qy][qx] = u;
                     DQQ1[dz][qy][qx] = v;
                     DQQ2[dz][qy][qx] = w;
                     U[0][dz][qy][qx] = z;
                  }
               }
            }
            MFEM_SYNC_THREAD;
            MFEM_FOREACH_THREAD(qz,z,Q1D)
            {
               MFEM_FOREACH_THREAD(qy,y,Q1D)
               {
                  MFEM_FOREACH_THREAD(qx,x,Q1D)
                  {
                     double u = 0.0;
                     double v = 0.0;
                     double w = 0.0;
                     double z = 0.0;
                     for (int dz = 0; dz < D1D; ++dz)
                     {
                        u += DQQ0[dz][qy][qx] * B[qz][dz];
                        v += DQQ1[dz][qy][qx] * B[qz][dz];
                        w += DQQ2[dz][qy][qx] * G[qz][dz];
                        z += U[0][dz][qy][qx] * B[qz][dz];
                     }
                     QQQ0[qz][qy][qx] = u;
                     QQQ1[qz][qy][qx] = v;
                     QQQ2[qz][qy][qx] = w;
                     U[1][qz][qy][qx] = z;
                  }
               }
            }
            MFEM_SYNC_THREAD;
            MFEM_FOREACH_THREAD(qz,z,Q1D)
            {
               MFEM_FOREACH_THREAD(qy,y,Q1D)
               {
                  MFEM_FOREACH_THREAD(qx,x,Q1D)
                  {
                     const int q = qx + (qy + qz * Q1D) * Q1D;
                     const double z = U[1][qz][qy][qx];
                     const double gX = QQQ0[qz][qy][qx];
                     const double gY = QQQ1[qz][qy][qx];
                     const double gZ = QQQ2[qz][qy][qx];
                     const double d = gX*D(q,0,c,e) + gY*D(q,1,c,e) + gZ*D(q,2,c,e);
                     Z[qz][qy][qx] += z * d;
                  }
               }
            }
            MFEM_SYNC_THREAD;
         } // for each conv component
         if (tidz == 0)
         {
            MFEM_FOREACH_THREAD(d,y,D1D)
            {
               MFEM_FOREACH_THREAD(q,x,Q1D)
               {
                  Bt[d][q] = b(q,d);
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  double u = 0.0;
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     u += Z[qz][qy][qx] * Bt[dx][qx];
                  }
                  QQD0[qz][qy][dx] = u;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  double u = 0.0;
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     u += QQD0[qz][qy][dx] * Bt[dy][qy];
                  }
                  QDD0[qz][dy][dx] = u;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  double u = 0.0;
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     u += QDD0[qz][dy][dx] * Bt[dz][qz];
                  }
                  Y(dx,dy,dz,cy,e) += u;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

void VectorConvectionNLFIntegrator::MultPA(const Vector &x, Vector &y) const
{
   const int NE = ne;
   const int D1D = maps->ndof;
   const int Q1D = maps->nqpt;
   const Vector &Q = pa_data;
   const Array<double> &B = maps->B;
   const Array<double> &G = maps->G;
   const Array<double> &Bt = maps->Bt;
   if (dim == 2)
   {
      return PAConvectionNLApply2D(NE,B,G,Bt,Q,x,y,D1D,Q1D);
   }
   if (dim == 3)
   {
      return SmemPAConvectionNLApply3D(NE,B,G,Q,x,y,D1D,Q1D);
   }
   MFEM_ABORT("Not yet implemented!");
}

}
