// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mean_evaluator.hpp"
#include "general/forall.hpp"

using namespace mfem;
using namespace navier;

MeanEvaluator::MeanEvaluator(ParFiniteElementSpace &fes_,
                             const IntegrationRule &ir_)
   : fes(fes_), ir(ir_), quad_interp(fes, ir)
{
   ParMesh &mesh = *fes.GetParMesh();
   geom = mesh.GetGeometricFactors(ir, GeometricFactors::DETERMINANTS);
   x_qvec.SetSize(ir.Size()*mesh.GetNE());
   quad_interp.SetOutputLayout(
      QVectorLayout::byNODES); // (nq, ne), same layout as geom->detJ

   // Quadrature weights times Jacobian determinants, layout (nq, ne)
   const int nq = ir.GetNPoints(), ne = mesh.GetNE();
   w_detJ.SetSize(nq*ne);
   const real_t *d_detJ = geom->detJ.HostRead();
   real_t *d_w = w_detJ.HostWrite();
   for (int e = 0; e < ne; e++)
   {
      for (int q = 0; q < nq; q++)
      {
         d_w[q + nq*e] = ir.IntPoint(q).weight * d_detJ[q + nq*e];
      }
   }

   Vector ones(fes.GetTrueVSize());
   ones = 1.0;
   volume = ComputeIntegral(ones);
}

real_t MeanEvaluator::ComputeIntegral(const Vector &x) const
{
   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *P = fes.GetProlongationMatrix();
   const Operator *el_restr = fes.GetElementRestriction(ordering);

   x_lvec.SetSize(P->Height());
   P->Mult(x, x_lvec);

   x_evec.SetSize(el_restr->Height());
   el_restr->Mult(x_lvec, x_evec);

   quad_interp.Values(x_evec, x_qvec);

   real_t integ = x_qvec*w_detJ; // sum_q w_q detJ_q x(x_q)
   MPI_Allreduce(MPI_IN_PLACE, &integ, 1, MFEM_MPI_REAL_T, MPI_SUM, fes.GetComm());
   return integ;
}

void MeanEvaluator::MakeMeanZero(Vector &x) const
{
   x -= (ComputeIntegral(x)/volume);
}
