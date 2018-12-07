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
#include "kernels/kIntMass.hpp"

namespace mfem
{

// *****************************************************************************
KMassIntegrator::KMassIntegrator(const FiniteElementSpace *f,
                                 const IntegrationRule *i)
   :op(),
    maps(NULL),
    fes(f),
    ir(i)
{
   push();
   assert(i);
   assert(fes);
}

// *****************************************************************************
void KMassIntegrator::Assemble()
{
   push();
   assert(ir);
   maps = kDofQuadMaps::Get(*fes, *fes, *ir);
}

// *****************************************************************************
void KMassIntegrator::SetOperator(Vector &v)
{
   push();
   op.SetSize(v.Size());
   op = v;
}

// *****************************************************************************
void KMassIntegrator::MultAdd(Vector &x, Vector &y)
{
   //push();
   Mesh *mesh = fes->GetMesh();
   const int dim = mesh->Dimension();
   const int quad1D = IntRules.Get(Geometry::SEGMENT,ir->GetOrder()).GetNPoints();
   const int dofs1D = fes->GetFE(0)->GetOrder() + 1;
   kMassMultAdd(dim,
                dofs1D,
                quad1D,
                mesh->GetNE(),
                maps->dofToQuad,
                maps->dofToQuadD,
                maps->quadToDof,
                maps->quadToDofD,
                op, x, y);
}

} // namespace mfem
