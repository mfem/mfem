// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mtop_integrators.hpp"

namespace mfem
{

double ParametricLinearDiffusion::GetElementEnergy(const
                                                   Array<const FiniteElement *> &el,
                                                   const Array<const FiniteElement *> &pel,
                                                   ElementTransformation &Tr,
                                                   const Array<const Vector *> &elfun,
                                                   const Array<const Vector *> &pelfun)
{
   int dof_u0 = el[0]->GetDof();
   int dof_r0 = pel[0]->GetDof();

   int dim = el[0]->GetDim();
   int spaceDim = Tr.GetSpaceDim();
   if (dim != spaceDim)
   {
      mfem::mfem_error("ParametricLinearDiffusion::GetElementEnergy"
                       " is not defined on manifold meshes");
   }

   // shape functions
   Vector shu0(dof_u0);
   Vector shr0(dof_r0);
   DenseMatrix dsu0(dof_u0,dim);
   DenseMatrix B(dof_u0, 4);
   B=0.0;

   double w;

   Vector param(1); param=0.0;
   Vector uu(4); uu=0.0;

   double energy =0.0;

   const IntegrationRule *ir;
   {
      int order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0])
                 +pel[0]->GetOrder();
      ir=&IntRules.Get(Tr.GetGeometryType(),order);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);
      w=Tr.Weight();
      w = ip.weight * w;

      el[0]->CalcPhysDShape(Tr,dsu0);
      el[0]->CalcPhysShape(Tr,shu0);
      pel[0]->CalcPhysShape(Tr,shr0);

      param[0]=shr0*(*pelfun[0]);

      // set the matrix B
      for (int jj=0; jj<dim; jj++)
      {
         B.SetCol(jj,dsu0.GetColumn(jj));
      }
      B.SetCol(3,shu0);
      B.MultTranspose(*elfun[0],uu);
      energy=energy+w * qfun.QEnergy(Tr,ip,param,uu);
   }
   return energy;
}


void ParametricLinearDiffusion::AssembleElementVector(const
                                                      Array<const FiniteElement *> &el,
                                                      const Array<const FiniteElement *> &pel,
                                                      ElementTransformation &Tr,
                                                      const Array<const Vector *> &elfun,
                                                      const Array<const Vector *> &pelfun,
                                                      const Array<Vector *> &elvec)
{
   int dof_u0 = el[0]->GetDof();
   int dof_r0 = pel[0]->GetDof();

   int dim = el[0]->GetDim();

   elvec[0]->SetSize(dof_u0);
   *elvec[0]=0.0;
   int spaceDim = Tr.GetSpaceDim();
   if (dim != spaceDim)
   {
      mfem::mfem_error("ParametricLinearDiffusion::AssembleElementVector"
                       " is not defined on manifold meshes");
   }

   // shape functions
   Vector shu0(dof_u0);
   Vector shr0(dof_r0);
   DenseMatrix dsu0(dof_u0,dim);
   DenseMatrix B(dof_u0, 4);
   B=0.0;

   double w;

   Vector param(1); param=0.0;
   Vector uu(4); uu=0.0;
   Vector rr(4);
   Vector lvec; lvec.SetSize(dof_u0);

   const IntegrationRule *ir = nullptr;
   int order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0])
              +pel[0]->GetOrder();
   ir=&IntRules.Get(Tr.GetGeometryType(),order);


   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);
      w=Tr.Weight();
      w = ip.weight * w;

      el[0]->CalcPhysDShape(Tr,dsu0);
      el[0]->CalcPhysShape(Tr,shu0);
      pel[0]->CalcPhysShape(Tr,shr0);

      param[0]=shr0*(*pelfun[0]);

      // set the matrix B
      for (int jj=0; jj<dim; jj++)
      {
         B.SetCol(jj,dsu0.GetColumn(jj));
      }
      B.SetCol(3,shu0);
      B.MultTranspose(*elfun[0],uu);
      qfun.QResidual(Tr,ip,param, uu, rr);

      B.Mult(rr,lvec);
      elvec[0]->Add(w,lvec);
   }
}

void ParametricLinearDiffusion::AssembleElementGrad(const
                                                    Array<const FiniteElement *> &el,
                                                    const Array<const FiniteElement *> &pel,
                                                    ElementTransformation &Tr,
                                                    const Array<const Vector *> &elfun,
                                                    const Array<const Vector *> &pelfun,
                                                    const Array2D<DenseMatrix *> &elmats)
{
   int dof_u0 = el[0]->GetDof();
   int dof_r0 = pel[0]->GetDof();

   int dim = el[0]->GetDim();

   DenseMatrix* K=elmats(0,0);
   K->SetSize(dof_u0,dof_u0);
   (*K)=0.0;

   int spaceDim = Tr.GetSpaceDim();
   if (dim != spaceDim)
   {
      mfem::mfem_error("ParametricLinearDiffusion::AssembleElementGrad"
                       " is not defined on manifold meshes");
   }

   // shape functions
   Vector shu0(dof_u0);
   Vector shr0(dof_r0);
   DenseMatrix dsu0(dof_u0,dim);
   DenseMatrix B(dof_u0, 4);
   DenseMatrix A(dof_u0, 4);
   B=0.0;
   double w;

   Vector param(1); param=0.0;
   Vector uu(4); uu=0.0;
   DenseMatrix hh(4,4);
   Vector lvec; lvec.SetSize(dof_u0);

   const IntegrationRule *ir = nullptr;
   int order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0])
              +pel[0]->GetOrder();
   ir=&IntRules.Get(Tr.GetGeometryType(),order);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);
      w = Tr.Weight();
      w = ip.weight * w;

      el[0]->CalcPhysDShape(Tr,dsu0);
      el[0]->CalcPhysShape(Tr,shu0);
      pel[0]->CalcPhysShape(Tr,shr0);

      param[0]=shr0*(*pelfun[0]);

      // set the matrix B
      for (int jj=0; jj<dim; jj++)
      {
         B.SetCol(jj,dsu0.GetColumn(jj));
      }
      B.SetCol(3,shu0);
      B.MultTranspose(*elfun[0],uu);
      qfun.QGradResidual(Tr,ip,param,uu,hh);
      Mult(B,hh,A);
      AddMult_a_ABt(w,A,B,*K);
   }
}

void ParametricLinearDiffusion::AssemblePrmElementVector(
   const Array<const FiniteElement *> &el,
   const Array<const FiniteElement *> &pel,
   ElementTransformation &Tr,
   const Array<const Vector *> &elfun,
   const Array<const Vector *> &alfun,
   const Array<const Vector *> &pelfun,
   const Array<Vector *> &elvec)
{
   int dof_u0 = el[0]->GetDof();
   int dof_r0 = pel[0]->GetDof();

   int dim = el[0]->GetDim();
   Vector& e0 = *(elvec[0]);

   e0.SetSize(dof_r0);
   e0=0.0;

   int spaceDim = Tr.GetSpaceDim();
   if (dim != spaceDim)
   {
      mfem::mfem_error("ParametricLinearDiffusion::AssemblePrmElementVector"
                       " is not defined on manifold meshes");
   }

   // shape functions
   Vector shu0(dof_u0);
   Vector shr0(dof_r0);
   DenseMatrix dsu0(dof_u0,dim);
   DenseMatrix B(dof_u0, 4);
   B=0.0;

   double w;

   Vector param(1); param=0.0;
   Vector uu(4); uu=0.0;
   Vector aa(4); aa=0.0;
   Vector rr(1);
   Vector lvec0; lvec0.SetSize(dof_r0);

   const IntegrationRule *ir;
   {
      int order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0])
                 +pel[0]->GetOrder();
      ir=&IntRules.Get(Tr.GetGeometryType(),order);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);
      w=Tr.Weight();
      w = ip.weight * w;

      el[0]->CalcPhysDShape(Tr,dsu0);
      el[0]->CalcPhysShape(Tr,shu0);
      pel[0]->CalcPhysShape(Tr,shr0);

      param[0]=shr0*(*pelfun[0]);

      // set the matrix B
      for (int jj=0; jj<dim; jj++)
      {
         B.SetCol(jj,dsu0.GetColumn(jj));
      }
      B.SetCol(3,shu0);
      B.MultTranspose(*elfun[0],uu);
      B.MultTranspose(*alfun[0],aa);

      qfun.AQResidual(Tr, ip, param, uu, aa, rr);

      lvec0=shr0;
      lvec0*=rr[0];

      e0.Add(w,lvec0);
   }
}

double DiffusionObjIntegrator::GetElementEnergy(const
                                                Array<const FiniteElement *> &el,
                                                ElementTransformation &Tr,
                                                const Array<const Vector *> &elfun)
{
   int dof_u0 = el[0]->GetDof();
   int dim = el[0]->GetDim();
   int spaceDim = Tr.GetSpaceDim();
   if (dim != spaceDim)
   {
      mfem::mfem_error("DiffusionObjIntegrator::GetElementEnergy"
                       " is not defined on manifold meshes");
   }

   // shape functions
   Vector shu0(dof_u0);

   double w;
   double val;

   double energy = 0.0;

   const IntegrationRule *ir;
   {
      int order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0]);
      ir=&IntRules.Get(Tr.GetGeometryType(),order);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);
      w=Tr.Weight();

      w = ip.weight * w;

      el[0]->CalcPhysShape(Tr,shu0);

      val=shu0*(*elfun[0]);
      energy=energy + w * val * val;
   }
   return 0.5*energy;
}

void DiffusionObjIntegrator::AssembleElementVector(const
                                                   Array<const FiniteElement *> &el,
                                                   ElementTransformation &Tr,
                                                   const Array<const Vector *> &elfun,
                                                   const Array<Vector *> &elvec)
{
   int dof_u0 = el[0]->GetDof();
   int dim = el[0]->GetDim();
   int spaceDim = Tr.GetSpaceDim();

   elvec[0]->SetSize(dof_u0);
   *elvec[0]=0.0;

   if (dim != spaceDim)
   {
      mfem::mfem_error("DiffusionObjIntegrator::GetElementEnergy"
                       " is not defined on manifold meshes");
   }

   // shape functions
   Vector shu0(dof_u0);

   double w;
   double val;

   const IntegrationRule *ir;
   {
      int order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0]);
      ir=&IntRules.Get(Tr.GetGeometryType(),order);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);
      w=Tr.Weight();

      w = ip.weight * w;

      el[0]->CalcPhysShape(Tr,shu0);

      val=shu0*(*elfun[0]);

      elvec[0]->Add(w*val,shu0);
   }
}


} // end mfem namespace
