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

void Quad2FieldInterpolant::ProjectQuadratureDiscCoefficient(GridFunction &gf,
                                                        VectorQuadratureFunctionCoefficient &vqfc)
{
   {
      const FiniteElementSpace *fes = gf.FESpace();
      MFEM_VERIFY(fes->GetVDim() == vqfc.GetVDim(),
      "FiniteElementSpace corresponding to this GridFunction should have the \
       same vdim of the VectorQuadratureFunctionCoefficient");
      
      const FiniteElement &el = *fes->GetFE(0);
      const IntegrationRule* ir = &(IntRules.Get(el.GetGeomType(),
                                    2 * el.GetOrder() + 1));
      const QuadratureFunction* qf = vqfc.GetQuadFunction();
      const IntegrationRule* ir_qf = &qf->GetSpace()->GetElementIntRule(0);
      
      MFEM_VERIFY((ir->GetOrder() == ir_qf->GetOrder()) &&
                  (ir->GetNPoints() == ir_qf->GetNPoints()),
                  "IntegrationRule for GridFunction and in QuadratureFunction \
                  appear to be different");
   }
   gf.HostReadWrite();
   // Later on we might be able to swap this over to something that can run on
   // on the gpu.
   gf.ProjectDiscCoefficient(vqfc, GridFunction::ARITHMETIC);
}

void Quad2FieldInterpolant::ProjectQuadratureDiscCoefficient(GridFunction &gf,
                                                        QuadratureFunctionCoefficient &qfc)
{
   {
      const FiniteElementSpace *fes = gf.FESpace();
      MFEM_VERIFY(fes->GetVDim() == 1,
      "FiniteElementSpace corresponding to this GridFunction should have a vdim\
       of 1");
      
      const FiniteElement &el = *fes->GetFE(0);
      const IntegrationRule* ir = &(IntRules.Get(el.GetGeomType(),
                                    2 * el.GetOrder() + 1));
      const QuadratureFunction* qf = qfc.GetQuadFunction();
      const IntegrationRule* ir_qf = &qf->GetSpace()->GetElementIntRule(0);
      
      MFEM_VERIFY((ir->GetOrder() == ir_qf->GetOrder()) &&
                  (ir->GetNPoints() == ir_qf->GetNPoints()),
                  "IntegrationRule for GridFunction and in QuadratureFunction \
                  appear to be different");
   }
   gf.HostReadWrite();
   // Later on we might be able to swap this over to something that can run on
   // on the gpu.
   gf.ProjectDiscCoefficient(qfc, GridFunction::ARITHMETIC);
}

void Quad2FieldInterpolant::ProjectQuadratureCoefficient(GridFunction &gf,
                                                    VectorQuadratureFunctionCoefficient &vqfc)
{
   FiniteElementSpace *fes = gf.FESpace();
   {
      MFEM_VERIFY(fes->GetVDim() == vqfc.GetVDim(),
      "FiniteElementSpace corresponding to this GridFunction should have the \
       same vdim of the VectorQuadratureFunctionCoefficient");

      // This is the best way I can think of to make sure the IntegrationRule in
      // the FiniteElementSpace and the QuadratureSpace correspond to the same
      const FiniteElement &el = *fes->GetFE(0);
      const IntegrationRule* ir = &(IntRules.Get(el.GetGeomType(),
                                    2 * el.GetOrder() + 1));
      const QuadratureFunction* qf = vqfc.GetQuadFunction();
      const IntegrationRule* ir_qf = &qf->GetSpace()->GetElementIntRule(0);
      MFEM_VERIFY((ir->GetOrder() == ir_qf->GetOrder()) &&
                  (ir->GetNPoints() == ir_qf->GetNPoints()),
                  "IntegrationRule in FiniteElementSpace and in QuadratureFunction \
                  appear to be different");
   }

   int vdim = vqfc.GetVDim();
   int size = gf.Size() / vdim;

   LinearForm *b = new LinearForm(fes);
   b->AddDomainIntegrator(new VectorQuadratureLFIntegrator(vqfc,
                          &vqfc.GetQuadFunction()->GetSpace()->GetElementIntRule(0)));
   b->Assemble();

   // If our FES is byVDIM then we're going to rearrange b to be in byNodes order
   if (fes->GetOrdering() == Ordering::byVDIM)
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

   GridFunction x(fes);
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

   if (fes->GetOrdering() == Ordering::byNODES)
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
void Quad2FieldInterpolant::ProjectQuadratureCoefficient(GridFunction &gf,
                                                    QuadratureFunctionCoefficient &qfc)
{
   FiniteElementSpace *fes = gf.FESpace();
   {
      MFEM_VERIFY(fes->GetVDim() == 1,
      "FiniteElementSpace corresponding to this GridFunction should have a \
       vdim of 1");

      // This is the best way I can think of to make sure the
      // IntegrationRule in the FiniteElementSpace
      // and the QuadratureSpace correspond to the same
      const FiniteElement &el = *fes->GetFE(0);
      const IntegrationRule* ir = &(IntRules.Get(el.GetGeomType(),
                                    2 * el.GetOrder() + 1));
      const QuadratureFunction* qf = qfc.GetQuadFunction();
      const IntegrationRule* ir_qf = &qf->GetSpace()->GetElementIntRule(0);
      MFEM_VERIFY((ir->GetOrder() == ir_qf->GetOrder()) &&
                  (ir->GetNPoints() == ir_qf->GetNPoints()),
                  "IntegrationRule in FiniteElementSpace and in QuadratureFunction \
                  appear to be different");
   }
   LinearForm *b = new LinearForm(fes);
   b->AddDomainIntegrator(new QuadratureLFIntegrator(qfc, 
                          &qfc.GetQuadFunction()->GetSpace()->GetElementIntRule(0)));
   b->Assemble();

   // L2->Assemble();

   GridFunction x(fes);
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
void ParQuad2FieldInterpolant::ProjectQuadratureCoefficient(ParGridFunction &gf,
                                                       VectorQuadratureFunctionCoefficient &vqfc)
{
   ParFiniteElementSpace *fes = gf.ParFESpace();
   {
      MFEM_VERIFY(fes->GetVDim() == vqfc.GetVDim(),
      "FiniteElementSpace corresponding to this GridFunction should have the \
       same vdim of the VectorQuadratureFunctionCoefficient");

      // This is the best way I can think of to make sure the IntegrationRule in
      // the FiniteElementSpace and the QuadratureSpace correspond to the same
      const FiniteElement &el = *fes->GetFE(0);
      const IntegrationRule* ir = &(IntRules.Get(el.GetGeomType(),
                                    2 * el.GetOrder() + 1));
      const QuadratureFunction* qf = vqfc.GetQuadFunction();
      const IntegrationRule* ir_qf = &qf->GetSpace()->GetElementIntRule(0);
      MFEM_VERIFY((ir->GetOrder() == ir_qf->GetOrder()) &&
                  (ir->GetNPoints() == ir_qf->GetNPoints()),
                  "IntegrationRule in FiniteElementSpace and in QuadratureFunction \
                  appear to be different");
   }

   int vdim = vqfc.GetVDim();
   int size = gf.Size() / vdim;

   ParLinearForm *b = new ParLinearForm(fes);
   b->AddDomainIntegrator(new VectorQuadratureLFIntegrator(vqfc, 
                          &vqfc.GetQuadFunction()->GetSpace()->GetElementIntRule(0)));
   b->Assemble();

   // If our FES is byVDIM then we're going to rearrange b to be in byNodes order
   if (fes->GetOrdering() == Ordering::byVDIM)
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

   ParGridFunction x(fes);
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

   if (fes->GetOrdering() == Ordering::byNODES)
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
void ParQuad2FieldInterpolant::ProjectQuadratureCoefficient(ParGridFunction &gf,
                                                       QuadratureFunctionCoefficient &qfc)
{
   ParFiniteElementSpace *fes = gf.ParFESpace();
   {
      MFEM_VERIFY(fes->GetVDim() == 1,
      "FiniteElementSpace corresponding to this GridFunction should have a \
       a vdim of 1");

      // This is the best way I can think of to make sure the
      // IntegrationRule in the FiniteElementSpace
      // and the QuadratureSpace correspond to the same
      const FiniteElement &el = *fes->GetFE(0);
      const IntegrationRule* ir = &(IntRules.Get(el.GetGeomType(),
                                    2 * el.GetOrder() + 1));
      const QuadratureFunction* qf = qfc.GetQuadFunction();
      const IntegrationRule* ir_qf = &qf->GetSpace()->GetElementIntRule(0);
      MFEM_VERIFY((ir->GetOrder() == ir_qf->GetOrder()) &&
                  (ir->GetNPoints() == ir_qf->GetNPoints()),
                  "IntegrationRule in FiniteElementSpace and in QuadratureFunction \
                  appear to be different");
   }
   ParLinearForm *b = new ParLinearForm(fes);
   b->AddDomainIntegrator(new QuadratureLFIntegrator(qfc, 
                          &qfc.GetQuadFunction()->GetSpace()->GetElementIntRule(0)));
   b->Assemble();

   ParGridFunction x(fes);
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