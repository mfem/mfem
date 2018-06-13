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

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#include "../raja.hpp"

namespace mfem
{

namespace raja
{

// *****************************************************************************
RajaDiffusionIntegrator::RajaDiffusionIntegrator(const RajaCoefficient &coeff_)
   :
   RajaIntegrator(coeff_.RajaEngine()),
   coeff(coeff_),
   assembledOperator(*(new Layout(coeff_.RajaEngine(), 0)))
{
   push();
   coeff.SetName("COEFF");
   pop();
}

// *****************************************************************************
RajaDiffusionIntegrator::~RajaDiffusionIntegrator() {}

// *****************************************************************************
std::string RajaDiffusionIntegrator::GetName()
{
   return "DiffusionIntegrator";
}

// *****************************************************************************
void RajaDiffusionIntegrator::SetupIntegrationRule()
{
   push();
   const FiniteElement &trialFE = *(trialFESpace->GetFE(0));
   const FiniteElement &testFE  = *(testFESpace->GetFE(0));
   ir = &mfem::DiffusionIntegrator::GetRule(trialFE, testFE);
   pop();
}

// *****************************************************************************
void RajaDiffusionIntegrator::Setup()
{
   push();
   pop();
}

// *****************************************************************************
void RajaDiffusionIntegrator::Assemble()
{
   push();
   const mfem::FiniteElement &fe = *(trialFESpace->GetFE(0));
   const int dim = mesh->Dimension();
   const int dims = fe.GetDim();
   assert(dim==dims);

   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   const int elements = trialFESpace->GetNE();
   assert(elements==mesh->GetNE());

   const int quadraturePoints = ir->GetNPoints();
   const int quad1D = IntRules.Get(Geometry::SEGMENT,ir->GetOrder()).GetNPoints();
   //assert(quad1D==quadraturePoints);

   RajaGeometry *geo = GetGeometry(RajaGeometry::Jacobian);
   assert(geo);

   assembledOperator.Resize<double>(symmDims * quadraturePoints * elements,NULL);

   /*   dbg("maps->quadWeights.Size()=%d",maps->quadWeights.Size());
      dbg("geom->J.Size()=%d",geom->J.Size());
      dbg("assembledOperator.Size()=%d",assembledOperator.Size());
      dbg("\t\033[35mCOEFF=%f",1.0);
      dbg("\t\033[35mquad1D=%d",quad1D);
      dbg("\t\033[35mmesh->GetNE()=%d",mesh->GetNE());
      for(size_t i=0;i<maps->quadWeights.Size();i+=1)
         printf("\n\t\033[35m[Assemble] quadWeights[%ld]=%f",i, maps->quadWeights[i]);
      //for(size_t i=0;i<geo->J.Size();i+=1) printf("\n\t\033[35m[Assemble] J[%ld]=%f",i, geo->J[i]);
      */
   rDiffusionAssemble(dim,
                      quad1D,
                      mesh->GetNE(),
                      maps->quadWeights,
                      geo->J,
                      1.0,//COEFF
                      (double*)assembledOperator.RajaMem().ptr());
   /*   for(size_t i=0;i<assembledOperator.Size();i+=1)
         printf("\n\t\033[35m[Assemble] assembledOperator[%ld]=%f",i,
         ((double*)assembledOperator.RajaMem().ptr())[i]);
   */
   pop();
}

// *****************************************************************************
void RajaDiffusionIntegrator::MultAdd(Vector &x, Vector &y)
{
   push();
   const int dim = mesh->Dimension();
   const int quad1D = IntRules.Get(Geometry::SEGMENT,ir->GetOrder()).GetNPoints();
   const int dofs1D = trialFESpace->GetFE(0)->GetOrder() + 1;
   // Note: x and y are E-vectors
   /*   for(size_t i=0;i<x.Size();i+=1)
         printf("\n\t\033[36m[MultAdd] x[%ld]=%f",i, ((double*)x.RajaMem().ptr())[i]);
      for(size_t i=0;i<maps->dofToQuad.Size();i+=1)
         printf("\n\t\033[36m[MultAdd] dofToQuad[%ld]=%f",i, maps->dofToQuad[i]);
      for(size_t i=0;i<maps->dofToQuadD.Size();i+=1)
         printf("\n\t\033[36m[MultAdd] dofToQuadD[%ld]=%f",i, maps->dofToQuadD[i]);
      for(size_t i=0;i<maps->quadToDof.Size();i+=1)
         printf("\n\t\033[36m[MultAdd] quadToDof[%ld]=%f",i, maps->quadToDof[i]);
      for(size_t i=0;i<maps->quadToDofD.Size();i+=1)
         printf("\n\t\033[36m[MultAdd] quadToDofD[%ld]=%f",i, maps->quadToDofD[i]);
      for(size_t i=0;i<assembledOperator.Size();i+=1)
         printf("\n\t\033[36m[MultAdd] assembledOperator[%ld]=%f",i, ((double*)assembledOperator.RajaMem().ptr())[i]);
   */
   rDiffusionMultAdd(dim,
                     dofs1D,
                     quad1D,
                     mesh->GetNE(),
                     maps->dofToQuad,
                     maps->dofToQuadD,
                     maps->quadToDof,
                     maps->quadToDofD,
                     (double*)assembledOperator.RajaMem().ptr(),
                     (const double*)x.RajaMem().ptr(),
                     (double*)y.RajaMem().ptr());
   /*   for(size_t i=0;i<y.Size();i+=1)
         printf("\n\t\033[36m[MultAdd] y[%ld]=%f",i, ((double*)y.RajaMem().ptr())[i]);
   */
   pop();
}

} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)
