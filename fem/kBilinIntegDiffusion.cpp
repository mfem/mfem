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
#include "kBilinIntegDiffusion.hpp"
#include "kernels/kGeometry.hpp"
#include "kernels/kIntDiffusion.hpp"

// *****************************************************************************
MFEM_NAMESPACE

// *****************************************************************************
KDiffusionIntegrator::KDiffusionIntegrator(const FiniteElementSpace *f,
                                           const IntegrationRule *i)
   :vec(),
    maps(NULL),
    fes(f),
    ir(i) {assert(i); assert(fes);}

// *****************************************************************************
void KDiffusionIntegrator::Assemble()
{
   const FiniteElement &fe = *(fes->GetFE(0));
   const Mesh *mesh = fes->GetMesh();
   const int dim = mesh->Dimension();
   const int dims = fe.GetDim();
   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   const int elements = fes->GetNE();
   assert(elements==mesh->GetNE());
   const int quadraturePoints = ir->GetNPoints();
   const int quad1D = IntRules.Get(Geometry::SEGMENT,ir->GetOrder()).GetNPoints();
   const int dofs1D = fes->GetFE(0)->GetOrder() + 1;
   dbg("dofs1D=%d",dofs1D);
   dbg("quad1D=%d",quad1D);
   const int size = symmDims * quadraturePoints * elements;
   vec.SetSize(size);
   kGeometry *geo = kGeometry::Get(*fes, *ir);
   maps = kDofQuadMaps::Get(*fes, *fes, *ir);
   kIntDiffusionAssemble(dim,
                         quad1D,
                         elements,
                         maps->quadWeights,
                         geo->J,
                         1.0,//COEFF
                         vec);
   //dbg("vec=");vec.Print();
   //assert(false);
}

// *****************************************************************************
void KDiffusionIntegrator::MultAdd(Vector &x, Vector &y)
{
   const Mesh *mesh = fes->GetMesh();
   const int dim = mesh->Dimension();
   const int quad1D = IntRules.Get(Geometry::SEGMENT,ir->GetOrder()).GetNPoints();
   //const FiniteElementSpace *f = fes->GetMesh()->GetNodes()->FESpace();
   const FiniteElementSpace *f = fes;
   const FiniteElement *fe = f->GetFE(0);
   //const int dofsND = fe->GetDof();
   const int dofs1D = fe->GetOrder() + 1;
   dbg("dofs1D=%d",dofs1D);
   dbg("quad1D=%d",quad1D);
   kIntDiffusionMultAdd(dim,
                        dofs1D,
                        quad1D,
                        fes->GetMesh()->GetNE(),
                        maps->dofToQuad,
                        maps->dofToQuadD,
                        maps->quadToDof,
                        maps->quadToDofD,
                        vec,
                        x,
                        y);
}

// *****************************************************************************
MFEM_NAMESPACE_END
