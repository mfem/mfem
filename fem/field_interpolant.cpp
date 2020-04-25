// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Implementation of Field Interpolants and necessary (Vector)QuadratorIntegrators

#include "field_interpolant.hpp"
#include "../linalg/densemat.hpp"
#include "fem.hpp"

namespace mfem
{

void VectorQuadratureIntegrator::AssembleRHSElementVect(const FiniteElement &fe,
                                                        ElementTransformation &Tr,
                                                        Vector &elvect)
{
   const int nqp = IntRule->GetNPoints();
   const int vdim = vqfc.GetVDim();
   const int ndofs = fe.GetDof();
   Vector shape(ndofs);
   Vector temp(vdim);
   elvect.SetSize(vdim * ndofs);
   elvect = 0.0;
   for (int q = 0; q < nqp; q++)
   {
      const IntegrationPoint &ip = IntRule->IntPoint(q);
      Tr.SetIntPoint(&ip);
      const double w = Tr.Weight() * ip.weight;
      vqfc.Eval(temp, Tr, ip);
      fe.CalcShape(ip, shape);
      for (int ind = 0; ind < vdim; ind++)
      {
         for (int nd = 0; nd < ndofs; nd++)
         {
            elvect(nd + ind * ndofs) += w * shape(nd) * temp(ind);
         }
      }
   }
}

void QuadratureIntegrator::AssembleRHSElementVect(const FiniteElement &fe,
                                                  ElementTransformation &Tr,
                                                  Vector &elvect)
{
   const int nqp = IntRule->GetNPoints();
   const int ndofs = fe.GetDof();
   Vector shape(ndofs);
   elvect.SetSize(ndofs);
   elvect = 0.0;
   for (int q = 0; q < nqp; q++)
   {
      const IntegrationPoint &ip = IntRule->IntPoint(q);
      Tr.SetIntPoint (&ip);
      const double w = Tr.Weight() * ip.weight;
      double temp = qfc.Eval(Tr, ip);
      fe.CalcShape(ip, shape);
      shape *= (w * temp);
      elvect += shape;
   }
}

//As a change to this we should have a setup phase where all the inverse matrices are stored off.
//If we do that we don't need to do the inverse and assemble step constantly. We can store the value in a vec
//and then just just use the DenseMatrix UseExternalData function.
//We can therefore provide a set-up phase that is run at the start of this if the vector this is all stored in is null.
//We should also provide a function that clears this.
//We'll need to assume that this is already an L2 space.
//One of the assumptions that we make down below is that our integration scheme is the same across all elements.
//If that isn't the case we might be able to still do things but things will most likely be slower.
void FieldInterpolant::ProjectQuadratureDiscCoefficient(GridFunction &gf,
                                                        VectorQuadratureFunctionCoefficient &vqfc,
                                                        FiniteElementSpace &fes)
{
   int ndofs;
   DenseMatrix mi;
   DenseMatrixInverse inv(&mi);
   const IntegrationRule* ir;
   NE = fes.GetMesh()->GetNE();
   {
      // This is the best way I can think of to make sure the IntegrationRule in the FiniteElementSpace
      // and the QuadratureSpace correspond to the same
      const FiniteElement &el = *fes.GetFE(0);
      ir = &(IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));
      const QuadratureFunction* qf = vqfc.GetQuadFunction();
      const IntegrationRule *ir_qf = &qf->GetSpace()->GetElementIntRule(0);
      MFEM_VERIFY((ir->GetOrder() == ir_qf->GetOrder()) &&
                  (ir->GetNPoints() == ir_qf->GetNPoints()),
                  "IntegrationRule in FiniteElementSpace and in QuadratureFunction appear to be different");
      // This should be the number of nodes available
      ndofs = el.GetDof();
   }
   int vdim = vqfc.GetVDim();

   Vector rhs(ndofs * vdim), rhs_sub(ndofs);
   Vector qfv(ndofs * vdim), qfv_sub(ndofs);

   VectorQuadratureIntegrator qi(vqfc);
   qi.SetIntRule(ir);

   Array<BilinearFormIntegrator*> dbfi = *L2->GetDBFI();

   if (!setup_disc)
   {
      m_all_data.SetSize(ndofs * ndofs * NE);
      double* data = m_all_data.HostReadWrite();
      for (int e = 0; e < NE; e++)
      {
         const FiniteElement &fe = *fes.GetFE(e);
         ElementTransformation &eltr = *fes.GetElementTransformation(e);
         mi.UseExternalData((data + (ndofs * ndofs * e)), ndofs, ndofs);
         dbfi[0]->AssembleElementMatrix(fe, eltr, mi);
      }
      setup_disc = true;
   }

   double* data = m_all_data.HostReadWrite();
   Array<int> dofs;
   for (int e = 0; e < NE; e++)
   {
      qfv = 0.0;
      mi.UseExternalData((data + (ndofs * ndofs * e)), ndofs, ndofs);
      inv.Factor(mi);
      const FiniteElement &fe = *fes.GetFE(e);
      ElementTransformation &eltr = *fes.GetElementTransformation(e);
      qi.AssembleRHSElementVect(fe, eltr, rhs);
      for (int ind = 0; ind < vdim; ind++)
      {
         qfv_sub.MakeRef(qfv, ndofs * ind);
         rhs_sub.MakeRef(rhs, ndofs * ind);
         inv.Mult(rhs_sub, qfv_sub);
      }
      fes.GetElementVDofs(e, dofs);
      gf.SetSubVector(dofs, qfv);
   }
}

void FieldInterpolant::ProjectQuadratureDiscCoefficient(GridFunction &gf,
                                                        QuadratureFunctionCoefficient &qfc,
                                                        FiniteElementSpace &fes)
{
   int ndofs;
   DenseMatrix mi;
   DenseMatrixInverse inv(&mi);
   const IntegrationRule* ir;
   {
      // This is the best way I can think of to make sure the IntegrationRule in the FiniteElementSpace
      // and the QuadratureSpace correspond to the same
      const FiniteElement &el = *fes.GetFE(0);
      ir = &(IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));
      const QuadratureFunction* qf = qfc.GetQuadFunction();
      const IntegrationRule* ir_qf = &qf->GetSpace()->GetElementIntRule(0);
      MFEM_VERIFY((ir->GetOrder() == ir_qf->GetOrder()) &&
                  (ir->GetNPoints() == ir_qf->GetNPoints()),
                  "IntegrationRule in FiniteElementSpace and in QuadratureFunction appear to be different");
      // This should be the number of nodes available
      ndofs = el.GetDof();
   }

   Vector rhs(ndofs);
   Vector qfv(ndofs);
   Array<int> dofs;

   QuadratureIntegrator qi(qfc);
   qi.SetIntRule(ir);
   Array<BilinearFormIntegrator*> dbfi = *L2->GetDBFI();

   if (!setup_disc)
   {
      m_all_data.SetSize(ndofs * ndofs * NE);
      double* data = m_all_data.HostReadWrite();
      for (int e = 0; e < NE; e++)
      {
         const FiniteElement &fe = *fes.GetFE(e);
         ElementTransformation &eltr = *fes.GetElementTransformation(e);
         mi.UseExternalData((data + (ndofs * ndofs * e)), ndofs, ndofs);
         dbfi[0]->AssembleElementMatrix(fe, eltr, mi);
      }
      setup_disc = true;
   }



   double* data = m_all_data.HostReadWrite();
   for (int e = 0; e < NE; e++)
   {
      mi.UseExternalData((data + (ndofs * ndofs * e)), ndofs, ndofs);
      inv.Factor(mi);
      const FiniteElement &fe = *fes.GetFE(e);
      ElementTransformation &eltr = *fes.GetElementTransformation(e);
      qi.AssembleRHSElementVect(fe, eltr, rhs);
      inv.Mult(rhs, qfv);
      fes.GetElementDofs(e, dofs);
      gf.SetSubVector(dofs, qfv);
   }
}

void FieldInterpolant::ProjectQuadratureCoefficient(GridFunction &gf,
                                                    VectorQuadratureFunctionCoefficient &vqfc,
                                                    FiniteElementSpace &fes)
{
   const IntegrationRule* ir;
   {
      // This is the best way I can think of to make sure the IntegrationRule in the FiniteElementSpace
      // and the QuadratureSpace correspond to the same
      const FiniteElement &el = *fes.GetFE(0);
      ir = &(IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));
      const QuadratureFunction* qf = vqfc.GetQuadFunction();
      const IntegrationRule* ir_qf = &qf->GetSpace()->GetElementIntRule(0);
      MFEM_VERIFY((ir->GetOrder() == ir_qf->GetOrder()) &&
                  (ir->GetNPoints() == ir_qf->GetNPoints()),
                  "IntegrationRule in FiniteElementSpace and in QuadratureFunction appear to be different");
   }

   int vdim = vqfc.GetVDim();
   int size = gf.Size() / vdim;

   LinearForm *b = new LinearForm(&fes);
   b->AddDomainIntegrator(new VectorQuadratureIntegrator(vqfc, ir));
   b->Assemble();

   // If our FES is byVDIM then we're going to rearrange b to be in byNodes order
   if (fes.GetOrdering() == Ordering::byVDIM)
   {
      Vector tmp = *b;
      double* data = b->HostReadWrite();
      for (int i = 0; i < vdim; i++)
      {
         for (int j = 0; j < size; j++)
         {
            data[j + i * size] = tmp(i + j * vdim);
         }
      }
   }

   // L2->Assemble();

   GridFunction x(&fes);
   x = 0.0;
   OperatorPtr A;
   Vector B, b_sub, X_sub, X;

   Array<int> ess_tdof_list;

   for (int ind = 0; ind < vdim; ind++)
   {
      int offset = ind * size;
      b_sub.MakeRef(*b, offset, size);
      X_sub.MakeRef(x, offset, size);
      L2->FormLinearSystem(ess_tdof_list, X_sub, b_sub, A, X, B);
      // Fix this to be more efficient;
      cg->SetOperator(*A);
      cg->Mult(B, X);
      // Recover the solution as a finite element grid function.
      L2->RecoverFEMSolution(X, *b, X_sub);
   }

   if (fes.GetOrdering() == Ordering::byNODES)
   {
      gf = x;
   }
   else
   {
      for (int i = 0; i < vdim; i++)
      {
         for (int j = 0; j < size; j++)
         {
            gf(i + j * vdim) = x(i * size + j);
         }
      }
   }

   delete b;
}
void FieldInterpolant::ProjectQuadratureCoefficient(GridFunction &gf,
                                                    QuadratureFunctionCoefficient &qfc,
                                                    FiniteElementSpace &fes)
{
   const IntegrationRule* ir;
   {
      // This is the best way I can think of to make sure the IntegrationRule in the FiniteElementSpace
      // and the QuadratureSpace correspond to the same
      const FiniteElement &el = *fes.GetFE(0);
      ir = &(IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));
      const QuadratureFunction* qf = qfc.GetQuadFunction();
      const IntegrationRule* ir_qf = &qf->GetSpace()->GetElementIntRule(0);
      MFEM_VERIFY((ir->GetOrder() == ir_qf->GetOrder()) &&
                  (ir->GetNPoints() == ir_qf->GetNPoints()),
                  "IntegrationRule in FiniteElementSpace and in QuadratureFunction appear to be different");
   }
   LinearForm *b = new LinearForm(&fes);
   b->AddDomainIntegrator(new QuadratureIntegrator(qfc, ir));
   b->Assemble();

   // L2->Assemble();

   GridFunction x(&fes);
   x = 0.0;
   OperatorPtr A;
   Vector B, X;
   Array<int> ess_tdof_list;

   L2->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
   cg->SetOperator(*A);
   cg->Mult(B, X);
   // Recover the solution as a finite element grid function.
   L2->RecoverFEMSolution(X, *b, x);
   gf = x;

   delete b;
}

#ifdef MFEM_USE_MPI
// This function takes a vector quadrature function coefficient and projects it onto a GridFunction of the same space as vector
// quadrature function coefficient.
void ParFieldInterpolant::ProjectQuadratureCoefficient(ParGridFunction &gf,
                                                       VectorQuadratureFunctionCoefficient &vqfc,
                                                       ParFiniteElementSpace &fes)
{
   const IntegrationRule* ir;
   {
      // This is the best way I can think of to make sure the IntegrationRule in the FiniteElementSpace
      // and the QuadratureSpace correspond to the same
      const FiniteElement &el = *fes.GetFE(0);
      ir = &(IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));
      const QuadratureFunction* qf = vqfc.GetQuadFunction();
      const IntegrationRule* ir_qf = &qf->GetSpace()->GetElementIntRule(0);
      MFEM_VERIFY((ir->GetOrder() == ir_qf->GetOrder()) &&
                  (ir->GetNPoints() == ir_qf->GetNPoints()),
                  "IntegrationRule in FiniteElementSpace and in QuadratureFunction appear to be different");
   }

   int vdim = vqfc.GetVDim();
   int size = gf.Size() / vdim;

   ParLinearForm *b = new ParLinearForm(&fes);
   b->AddDomainIntegrator(new VectorQuadratureIntegrator(vqfc, ir));
   b->Assemble();

   // If our FES is byVDIM then we're going to rearrange b to be in byNodes order
   if (fes.GetOrdering() == Ordering::byVDIM)
   {
      Vector tmp = *b;
      double* data = b->HostReadWrite();
      for (int i = 0; i < vdim; i++)
      {
         for (int j = 0; j < size; j++)
         {
            data[j + i * size] = tmp(i + j * vdim);
         }
      }
   }

   // ParL2->Assemble();

   ParGridFunction x(&fes);
   x = 0.0;
   OperatorPtr A;
   Vector B, b_sub, X_sub, X;

   Array<int> ess_tdof_list;

   for (int ind = 0; ind < vdim; ind++)
   {
      int offset = ind * size;
      b_sub.MakeRef(*b, offset, size);
      X_sub.MakeRef(x, offset, size);
      ParL2->FormLinearSystem(ess_tdof_list, X_sub, b_sub, A, X, B);
      // Recover the solution as a finite element grid function.
      cg->SetOperator(*A);
      cg->Mult(B, X);
      ParL2->RecoverFEMSolution(X, *b, X_sub);
   }

   if (fes.GetOrdering() == Ordering::byNODES)
   {
      gf = x;
   }
   else
   {
      for (int i = 0; i < vdim; i++)
      {
         for (int j = 0; j < size; j++)
         {
            gf(i + j * vdim) = x(i * size + j);
         }
      }
   }

   delete b;
}
// This function takes a quadrature function coefficient and projects it onto a GridFunction of the same space as
// quadrature function coefficient.
void ParFieldInterpolant::ProjectQuadratureCoefficient(ParGridFunction &gf,
                                                       QuadratureFunctionCoefficient &qfc,
                                                       ParFiniteElementSpace &fes)
{
   const IntegrationRule* ir;
   {
      // This is the best way I can think of to make sure the IntegrationRule in the FiniteElementSpace
      // and the QuadratureSpace correspond to the same
      const FiniteElement &el = *fes.GetFE(0);
      ir = &(IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));
      const QuadratureFunction* qf = qfc.GetQuadFunction();
      const IntegrationRule* ir_qf = &qf->GetSpace()->GetElementIntRule(0);
      MFEM_VERIFY((ir->GetOrder() == ir_qf->GetOrder()) &&
                  (ir->GetNPoints() == ir_qf->GetNPoints()),
                  "IntegrationRule in FiniteElementSpace and in QuadratureFunction appear to be different");
   }
   ParLinearForm *b = new ParLinearForm(&fes);
   b->AddDomainIntegrator(new QuadratureIntegrator(qfc, ir));
   b->Assemble();

   // ParL2->Assemble();

   ParGridFunction x(&fes);
   x = 0.0;
   OperatorPtr A;
   Vector B, X;
   Array<int> ess_tdof_list;

   ParL2->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
   // Fix this to be more efficient
   cg->SetOperator(*A);
   cg->Mult(B, X);
   // Recover the solution as a finite element grid function.
   ParL2->RecoverFEMSolution(X, *b, x);
   gf = x;

   delete b;
}
#endif

}