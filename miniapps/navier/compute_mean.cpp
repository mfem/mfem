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

#include "compute_mean.hpp"
#include "../../general/forall.hpp"

using namespace mfem;
using namespace navier;

MeanEvaluator::MeanEvaluator(ParFiniteElementSpace &fes_,
                             const IntegrationRule &ir_)
   : fes(fes_), ir(ir_), quad_interp(fes, ir)
{
   ParMesh &mesh = *fes.GetParMesh();
   geom = mesh.GetGeometricFactors(ir, GeometricFactors::DETERMINANTS);
   x_qvec.SetSize(ir.Size()*mesh.GetNE());

   Vector ones(fes.GetTrueVSize());
   ones = 1.0;
   volume = ComputeIntegral(ones);
}

double MeanEvaluator::ComputeIntegral(const Vector &x) const
{
   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *el_restr = fes.GetElementRestriction(ordering);

   x_evec.SetSize(el_restr->Height());
   el_restr->Mult(x, x_evec);
   quad_interp.Values(x_evec, x_qvec);

   double integ = x_evec*geom->detJ;
   MPI_Allreduce(MPI_IN_PLACE, &integ, 1, MPI_DOUBLE, MPI_SUM, fes.GetComm());
   return integ;
}

void MeanEvaluator::MakeMeanZero(Vector &x) const
{
   x -= (ComputeIntegral(x)/volume);
}
