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

#include "stress_evaluator.hpp"
#include "stress_evaluator_kernels.hpp"

using namespace mfem;
using namespace navier;

StressEvaluator::StressEvaluator(const ParFiniteElementSpace &kvfes,
                                 const ParFiniteElementSpace &ufes,
                                 const IntegrationRule &ir)
   : kvfes(kvfes), ufes(ufes), ir(ir), dim(kvfes.GetParMesh()->Dimension()),
     ne(ufes.GetNE())
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
   u_e.SetSize(Pu->Height());

   y_l.SetSize(Pu->Height());
   y_e.SetSize(Ru->Height());

   qi = kvfes.GetQuadratureInterpolator(ir);
   qi->SetOutputLayout(QVectorLayout::byNODES);
}

void StressEvaluator::Apply(const Vector &kv, const Vector &u,
                            Vector &y)
{
   const int d1d = maps->ndof, q1d = maps->nqpt;

   MFEM_ASSERT(y.Size() == Pu->Height(), "y wrong size");

   // T -> L
   Pkv->Mult(kv, kv_l);
   Pu->Mult(u, u_l);
   // L -> E
   Rkv->Mult(kv_l, kv_e);
   Ru->Mult(u_l, u_e);

   qi->Derivatives(kv_e, dkv_qp);

   if (dim == 2)
   {
      const int id = (d1d << 4) | q1d;
      switch (id)
      {
         case 0x3:
         {
            StressEvaluatorApply2D<3, 3>(ne, maps->B, maps->G, ir.GetWeights(),
                                         geom->J,
                                         geom->detJ, u_e, y_e, dkv_qp);
            break;
         }
         case 0x77:
         {
            StressEvaluatorApply2D<7, 7>(ne, maps->B, maps->G, ir.GetWeights(),
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
      MFEM_ABORT("unknown kernel");
   }

   // E -> L
   Rkv->MultTranspose(y_e, y_l);
   // L -> T
   Pkv->MultTranspose(y_l, y);
}
