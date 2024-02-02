// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "shear_stress_evaluator.hpp"
#include "shear_stress_evaluator_kernels.hpp"

using namespace mfem;
using namespace navier;

ShearStressEvaluator::ShearStressEvaluator(const ParFiniteElementSpace &kvfes,
                                           ParFiniteElementSpace &ufes,
                                           const IntegrationRule &ir)
   : ir(ir), dim(kvfes.GetParMesh()->Dimension()), ne(ufes.GetNE()), ufes(&ufes)
{
   ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   geom = ufes.GetParMesh()->GetGeometricFactors(
             ir, GeometricFactors::JACOBIANS | GeometricFactors::DETERMINANTS);
   maps = &ufes.GetFE(0)->GetDofToQuad(ir, DofToQuad::TENSOR);

   Pkv = kvfes.GetProlongationMatrix();
   Pu = ufes.GetProlongationMatrix();

   Rkv = kvfes.GetElementRestriction(ordering);
   Ru = ufes.GetElementRestriction(ordering);

   kv_l.SetSize(Pkv->Height());
   kv_e.SetSize(Rkv->Height());
   dkv_qp.SetSize(ir.GetNPoints() * ne * dim);

   u_l.SetSize(Pu->Height());
   u_e.SetSize(Ru->Height());

   y_l.SetSize(Pu->Height());
   y_e.SetSize(Ru->Height());

   qi = kvfes.GetQuadratureInterpolator(ir);
   qi->SetOutputLayout(QVectorLayout::byNODES);
}

void ShearStressEvaluator::Apply(const Vector &kv, const Vector &u,
                                 Vector &y)
{
   const int d1d = maps->ndof, q1d = maps->nqpt;

   // T -> L
   Pkv->Mult(kv, kv_l);
   Pu->Mult(u, u_l);

   // L -> E
   Rkv->Mult(kv_l, kv_e);
   Ru->Mult(u_l, u_e);

   qi->Derivatives(kv_e, dkv_qp);

   y_l = 0.0;

   if (dim == 2)
   {
      const int id = (d1d << 4) | q1d;
      switch (id)
      {
         case 0x22:
         {
            ShearStressEvaluatorApply2D<2, 2>(ne, maps->B, maps->G, ir.GetWeights(),
                                              geom->J,
                                              geom->detJ, u_e, y_e, dkv_qp);
            break;
         }
         case 0x33:
         {
            ShearStressEvaluatorApply2D<3, 3>(ne, maps->B, maps->G, ir.GetWeights(),
                                              geom->J,
                                              geom->detJ, u_e, y_e, dkv_qp);
            break;
         }
         case 0x55:
         {
            ShearStressEvaluatorApply2D<5, 5>(ne, maps->B, maps->G, ir.GetWeights(),
                                              geom->J,
                                              geom->detJ, u_e, y_e, dkv_qp);
            break;
         }
         case 0x66:
         {
            ShearStressEvaluatorApply2D<6, 6>(ne, maps->B, maps->G, ir.GetWeights(),
                                              geom->J,
                                              geom->detJ, u_e, y_e, dkv_qp);
            break;
         }
         case 0x77:
         {
            ShearStressEvaluatorApply2D<7, 7>(ne, maps->B, maps->G, ir.GetWeights(),
                                              geom->J,
                                              geom->detJ, u_e, y_e, dkv_qp);
            break;
         }
         default:
            MFEM_ABORT("unknown kernel");
      }
   }
   else if (dim == 3)
   {
      const int id = (d1d << 4) | q1d;
      switch (id)
      {
         case 0x22:
         {
            ShearStressEvaluatorApply3D<2, 2>(ne, maps->B, maps->G, ir.GetWeights(),
                                              geom->J,
                                              geom->detJ, u_e, y_e, dkv_qp);
            break;
         }
         case 0x33:
         {
            ShearStressEvaluatorApply3D<3, 3>(ne, maps->B, maps->G, ir.GetWeights(),
                                              geom->J,
                                              geom->detJ, u_e, y_e, dkv_qp);
            break;
         }
         case 0x55:
         {
            ShearStressEvaluatorApply3D<5, 5>(ne, maps->B, maps->G, ir.GetWeights(),
                                              geom->J,
                                              geom->detJ, u_e, y_e, dkv_qp);
            break;
         }
         case 0x77:
         {
            ShearStressEvaluatorApply3D<7, 7>(ne, maps->B, maps->G, ir.GetWeights(),
                                              geom->J,
                                              geom->detJ, u_e, y_e, dkv_qp);
            break;
         }
         default:
            MFEM_ABORT("unknown kernel");
      }
   }
   else
   {
      MFEM_ABORT("unknown kernel");
   }

   // TODO: Need to average or similar to preserve smoothness

   // E -> L
   Ru->MultTranspose(y_e, y_l);

   ParGridFunction ygf(ufes, y_l.GetData()), yavggf(ufes);
   VectorGridFunctionCoefficient ygf_coeff(&ygf);
   yavggf.ProjectDiscCoefficient(ygf_coeff);

   // L -> T
   Pu->MultTranspose(y_l, y);
}
