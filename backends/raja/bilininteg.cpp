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

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#include "bilininteg.hpp"
#include "../../fem/fem.hpp"

namespace mfem
{

namespace raja
{

std::map<std::string, RajaDofQuadMaps> RajaDofQuadMaps::AllDofQuadMaps;

RajaGeometry RajaGeometry::Get(raja::device device,
                               FiniteElementSpace &ofespace,
                               const mfem::IntegrationRule &ir,
                               const int flags)
{
   RajaGeometry geom;

   mfem::Mesh &mesh = *(ofespace.GetMesh());
   if (!mesh.GetNodes())
   {
      mesh.SetCurvature(1, false, -1, mfem::Ordering::byVDIM);
   }
   mfem::GridFunction &nodes = *(mesh.GetNodes());
   const mfem::FiniteElementSpace &fespace = *(nodes.FESpace());
   const mfem::FiniteElement &fe = *(fespace.GetFE(0));

   const int dims     = fe.GetDim();
   const int elements = fespace.GetNE();
   const int numDofs  = fe.GetDof();
   const int numQuad  = ir.GetNPoints();

   MFEM_ASSERT(dims == mesh.SpaceDimension(), "");

   assert(false);
   //geom.meshNodes.allocate(device,dims, numDofs, elements);

   const mfem::Table &e2dTable = fespace.GetElementToDofTable();
   const int *elementMap = e2dTable.GetJ();
   nodes.Pull();
   for (int e = 0; e < elements; ++e)
   {
      for (int dof = 0; dof < numDofs; ++dof)
      {
         const int gid = elementMap[dof + numDofs*e];
         for (int dim = 0; dim < dims; ++dim)
         {
            geom.meshNodes(dim, dof, e) = nodes[fespace.DofToVDof(gid,dim)];
         }
      }
   }
   //geom.meshNodes.keepInDevice();

   if (flags & Jacobian)
   {
      assert(false);
      //geom.J.allocate(device,dims, dims, numQuad, elements);
   }
   else
   {
      assert(false);/*
                      geom.J.allocate(device, 1);*/
   }
   if (flags & JacobianInv)
   {
      assert(false);/*
      geom.invJ.allocate(device,
      dims, dims, numQuad, elements);*/
   }
   else
   {
       assert(false);/*
                       geom.invJ.allocate(device, 1);*/
   }
   if (flags & JacobianDet)
   {
      assert(false);/*
      geom.detJ.allocate(device,
      numQuad, elements);*/
   }
   else
   {
      assert(false);/*
                      geom.detJ.allocate(device, 1);*/
   }

   //geom.J.stopManaging();
   //geom.invJ.stopManaging();
   //geom.detJ.stopManaging();

   RajaDofQuadMaps &maps = RajaDofQuadMaps::GetSimplexMaps(device, fe, ir);
   assert(false);/*
   init(elements,
        maps.dofToQuadD,
        geom.meshNodes,
        geom.J, geom.invJ, geom.detJ);*/

   return geom;
}

RajaDofQuadMaps::RajaDofQuadMaps() :
   hash() {}

RajaDofQuadMaps::RajaDofQuadMaps(const RajaDofQuadMaps &maps)
{
   *this = maps;
}

RajaDofQuadMaps& RajaDofQuadMaps::operator = (const RajaDofQuadMaps &maps)
{
   hash = maps.hash;
   dofToQuad   = maps.dofToQuad;
   dofToQuadD  = maps.dofToQuadD;
   quadToDof   = maps.quadToDof;
   quadToDofD  = maps.quadToDofD;
   quadWeights = maps.quadWeights;
   return *this;
}

RajaDofQuadMaps& RajaDofQuadMaps::Get(raja::device device,
                                      const FiniteElementSpace &fespace,
                                      const mfem::IntegrationRule &ir,
                                      const bool transpose)
{
   return Get(device,
              *fespace.GetFE(0),
              *fespace.GetFE(0),
              ir,
              transpose);
}

RajaDofQuadMaps& RajaDofQuadMaps::Get(raja::device device,
                                      const mfem::FiniteElement &fe,
                                      const mfem::IntegrationRule &ir,
                                      const bool transpose)
{
   return Get(device, fe, fe, ir, transpose);
}

RajaDofQuadMaps& RajaDofQuadMaps::Get(raja::device device,
                                      const FiniteElementSpace &trialFESpace,
                                      const FiniteElementSpace &testFESpace,
                                      const mfem::IntegrationRule &ir,
                                      const bool transpose)
{
   return Get(device,
              *trialFESpace.GetFE(0),
              *testFESpace.GetFE(0),
              ir,
              transpose);
}

RajaDofQuadMaps& RajaDofQuadMaps::Get(raja::device device,
                                      const mfem::FiniteElement &trialFE,
                                      const mfem::FiniteElement &testFE,
                                      const mfem::IntegrationRule &ir,
                                      const bool transpose)
{
   return (dynamic_cast<const mfem::TensorBasisElement*>(&trialFE)
           ? GetTensorMaps(device, trialFE, testFE, ir, transpose)
           : GetSimplexMaps(device, trialFE, testFE, ir, transpose));
}

RajaDofQuadMaps& RajaDofQuadMaps::GetTensorMaps(raja::device device,
                                                const mfem::FiniteElement &fe,
                                                const mfem::IntegrationRule &ir,
                                                const bool transpose)
{
   return GetTensorMaps(device,
                        fe, fe,
                        ir, transpose);
}

RajaDofQuadMaps& RajaDofQuadMaps::GetTensorMaps(raja::device device,
                                                const mfem::FiniteElement &trialFE,
                                                const mfem::FiniteElement &testFE,
                                                const mfem::IntegrationRule &ir,
                                                const bool transpose)
{
   const mfem::TensorBasisElement &trialTFE =
      dynamic_cast<const mfem::TensorBasisElement&>(trialFE);
   const mfem::TensorBasisElement &testTFE =
      dynamic_cast<const mfem::TensorBasisElement&>(testFE);

   assert(false);
   static RajaDofQuadMaps maps;
   return maps;
}

RajaDofQuadMaps RajaDofQuadMaps::GetD2QTensorMaps(raja::device device,
                                                  const mfem::FiniteElement &fe,
                                                  const mfem::IntegrationRule &ir,
                                                  const bool transpose)
{
   const mfem::TensorBasisElement &tfe =
      dynamic_cast<const mfem::TensorBasisElement&>(fe);

   const mfem::Poly_1D::Basis &basis = tfe.GetBasis1D();
   const int order = fe.GetOrder();
   // [MISSING] Get 1D dofs
   const int dofs = order + 1;
   const int dims = fe.GetDim();

   // Create the dof -> quadrature point map
   const mfem::IntegrationRule &ir1D =
      mfem::IntRules.Get(mfem::Geometry::SEGMENT, ir.GetOrder());
   const int quadPoints = ir1D.GetNPoints();
   const int quadPoints2D = quadPoints*quadPoints;
   const int quadPoints3D = quadPoints2D*quadPoints;
   const int quadPointsND = ((dims == 1) ? quadPoints :
                             ((dims == 2) ? quadPoints2D : quadPoints3D));

   RajaDofQuadMaps maps;/*
   // Initialize the dof -> quad mapping
   maps.dofToQuad.allocate(device,
                           quadPoints, dofs);
   maps.dofToQuadD.allocate(device,
                            quadPoints, dofs);

   double *quadWeights1DData = NULL;

   if (transpose)
   {
      maps.dofToQuad.reindex(1,0);
      maps.dofToQuadD.reindex(1,0);
      // Initialize quad weights only for transpose
      maps.quadWeights.allocate(device,
                                quadPointsND);
      quadWeights1DData = new double[quadPoints];
   }

   mfem::Vector d2q(dofs);
   mfem::Vector d2qD(dofs);
   for (int q = 0; q < quadPoints; ++q)
   {
      const mfem::IntegrationPoint &ip = ir1D.IntPoint(q);
      basis.Eval(ip.x, d2q, d2qD);
      if (transpose)
      {
         quadWeights1DData[q] = ip.weight;
      }
      for (int d = 0; d < dofs; ++d)
      {
         maps.dofToQuad(q, d)  = d2q[d];
         maps.dofToQuadD(q, d) = d2qD[d];
      }
   }

   maps.dofToQuad.keepInDevice();
   maps.dofToQuadD.keepInDevice();

   if (transpose)
   {
      for (int q = 0; q < quadPointsND; ++q)
      {
         const int qx = q % quadPoints;
         const int qz = q / quadPoints2D;
         const int qy = (q - qz*quadPoints2D) / quadPoints;
         double w = quadWeights1DData[qx];
         if (dims > 1)
         {
            w *= quadWeights1DData[qy];
         }
         if (dims > 2)
         {
            w *= quadWeights1DData[qz];
         }
         maps.quadWeights[q] = w;
      }
      maps.quadWeights.keepInDevice();
      delete [] quadWeights1DData;
   }
                        */
   return maps;
}

RajaDofQuadMaps& RajaDofQuadMaps::GetSimplexMaps(raja::device device,
                                                 const mfem::FiniteElement &fe,
                                                 const mfem::IntegrationRule &ir,
                                                 const bool transpose)
{
   return GetSimplexMaps(device,
                         fe, fe,
                         ir, transpose);
}

RajaDofQuadMaps& RajaDofQuadMaps::GetSimplexMaps(raja::device device,
                                                 const mfem::FiniteElement &trialFE,
                                                 const mfem::FiniteElement &testFE,
                                                 const mfem::IntegrationRule &ir,
                                                 const bool transpose)
{
   assert(false);
   static RajaDofQuadMaps maps;// = AllDofQuadMaps[hash];
/*
   if (!maps.hash.size())
   {
      // Create the dof-quad maps
      maps.hash = hash;

      RajaDofQuadMaps trialMaps = GetD2QSimplexMaps(device, trialFE, ir);
      RajaDofQuadMaps testMaps  = GetD2QSimplexMaps(device, testFE , ir, true);

      maps.dofToQuad   = trialMaps.dofToQuad;
      maps.dofToQuadD  = trialMaps.dofToQuadD;
      maps.quadToDof   = testMaps.dofToQuad;
      maps.quadToDofD  = testMaps.dofToQuadD;
      maps.quadWeights = testMaps.quadWeights;
   }*/
   return maps;
}

RajaDofQuadMaps RajaDofQuadMaps::GetD2QSimplexMaps(raja::device device,
                                                   const mfem::FiniteElement &fe,
                                                   const mfem::IntegrationRule &ir,
                                                   const bool transpose)
{
   const int dims = fe.GetDim();
   const int numDofs = fe.GetDof();
   const int numQuad = ir.GetNPoints();

   RajaDofQuadMaps maps;
   // Initialize the dof -> quad mapping
   assert(false);
   //maps.dofToQuad.allocate(device,numQuad, numDofs);
   //maps.dofToQuadD.allocate(device,dims, numQuad, numDofs);

   if (transpose)
   {
      //maps.dofToQuad.reindex(1,0);
      //maps.dofToQuadD.reindex(1,0);
      // Initialize quad weights only for transpose
      assert(false);/*
                      maps.quadWeights.allocate(device,numQuad);*/
   }

   mfem::Vector d2q(numDofs);
   mfem::DenseMatrix d2qD(numDofs, dims);
   for (int q = 0; q < numQuad; ++q)
   {
      const mfem::IntegrationPoint &ip = ir.IntPoint(q);
      if (transpose)
      {
         maps.quadWeights[q] = ip.weight;
      }
      fe.CalcShape(ip, d2q);
      fe.CalcDShape(ip, d2qD);
      for (int d = 0; d < numDofs; ++d)
      {
         const double w = d2q[d];
         maps.dofToQuad(q, d) = w;
         for (int dim = 0; dim < dims; ++dim)
         {
            const double wD = d2qD(d, dim);
            maps.dofToQuadD(dim, q, d) = wD;
         }
      }
   }

   //maps.dofToQuad.keepInDevice();
   //maps.dofToQuadD.keepInDevice();
   if (transpose)
   {
      //maps.quadWeights.keepInDevice();
   }

   return maps;
}

//---[ Integrator Defines ]-----------
std::string stringWithDim(const std::string &s, const int dim)
{
   std::string ret = s;
   ret += ('0' + (char) dim);
   ret += 'D';
   return ret;
}

int closestWarpBatchTo(const int value)
{
   return ((value + 31) / 32) * 32;
}

int closestMultipleWarpBatch(const int multiple, const int maxSize)
{
   if (multiple > maxSize)
   {
      return maxSize;
   }
   int batch = (32 / multiple);
   int minDiff = 32 - (multiple * batch);
   for (int i = 64; i <= maxSize; i += 32)
   {
      const int newDiff = i - (multiple * (i / multiple));
      if (newDiff < minDiff)
      {
         batch = (i / multiple);
         minDiff = newDiff;
      }
   }
   return batch;
}

void SetProperties(FiniteElementSpace &fespace,
                   const mfem::IntegrationRule &ir)
{
   SetProperties(fespace, fespace, ir);
}

void SetProperties(FiniteElementSpace &trialFESpace,
                   FiniteElementSpace &testFESpace,
                   const mfem::IntegrationRule &ir)
{
   //props["defines/TRIAL_VDIM"] = trialFESpace.GetVDim();
   //props["defines/TEST_VDIM"]  = testFESpace.GetVDim();
   //props["defines/NUM_DIM"]    = trialFESpace.GetDim();

   if (trialFESpace.hasTensorBasis())
   {
      SetTensorProperties(trialFESpace, testFESpace, ir);
   }
   else
   {
      SetSimplexProperties(trialFESpace, testFESpace, ir);
   }
}

void SetTensorProperties(FiniteElementSpace &fespace,
                         const mfem::IntegrationRule &ir)
{
   SetTensorProperties(fespace, fespace, ir);
}

void SetTensorProperties(FiniteElementSpace &trialFESpace,
                         FiniteElementSpace &testFESpace,
                         const mfem::IntegrationRule &ir)
{
   assert(false);
}

void SetSimplexProperties(FiniteElementSpace &fespace,
                          const mfem::IntegrationRule &ir)
{
   SetSimplexProperties(fespace, fespace, ir);
}

void SetSimplexProperties(FiniteElementSpace &trialFESpace,
                          FiniteElementSpace &testFESpace,
                          const mfem::IntegrationRule &ir)
{
 
}


//---[ Base Integrator ]--------------
RajaIntegrator::RajaIntegrator(const Engine &e)
   : engine(&e),
     bform(),
     mesh(),
     otrialFESpace(),
     otestFESpace(),
     trialFESpace(),
     testFESpace(),
     itype(DomainIntegrator),
     ir(NULL),
     hasTensorBasis(false) { }

RajaIntegrator::~RajaIntegrator() {}

void RajaIntegrator::SetupMaps()
{
   maps = RajaDofQuadMaps::Get(GetDevice(),
                               *otrialFESpace,
                               *otestFESpace,
                               *ir);

   mapsTranspose = RajaDofQuadMaps::Get(GetDevice(),
                                        *otestFESpace,
                                        *otrialFESpace,
                                        *ir);
}

FiniteElementSpace& RajaIntegrator::GetTrialRajaFESpace() const
{
   return *otrialFESpace;
}

FiniteElementSpace& RajaIntegrator::GetTestRajaFESpace() const
{
   return *otestFESpace;
}

mfem::FiniteElementSpace& RajaIntegrator::GetTrialFESpace() const
{
   return *trialFESpace;
}

mfem::FiniteElementSpace& RajaIntegrator::GetTestFESpace() const
{
   return *testFESpace;
}

void RajaIntegrator::SetIntegrationRule(const mfem::IntegrationRule &ir_)
{
   ir = &ir_;
}

const mfem::IntegrationRule& RajaIntegrator::GetIntegrationRule() const
{
   return *ir;
}

RajaDofQuadMaps& RajaIntegrator::GetDofQuadMaps()
{
   return maps;
}

void RajaIntegrator::SetupIntegrator(RajaBilinearForm &bform_,
                                     const RajaIntegratorType itype_)
{
   MFEM_ASSERT(engine == &bform_.RajaEngine(), "");
   bform     = &bform_;
   mesh      = &(bform_.GetMesh());

   otrialFESpace = &(bform_.GetTrialRajaFESpace());
   otestFESpace  = &(bform_.GetTestRajaFESpace());

   trialFESpace = &(bform_.GetTrialFESpace());
   testFESpace  = &(bform_.GetTestFESpace());

   hasTensorBasis = otrialFESpace->hasTensorBasis();

   itype = itype_;

   if (ir == NULL)
   {
      SetupIntegrationRule();
   }

   SetupMaps();

   SetProperties(*otrialFESpace,
                 *otestFESpace,
                 *ir);

   Setup();
}

RajaGeometry RajaIntegrator::GetGeometry(const int flags)
{
   return RajaGeometry::Get(GetDevice(), *otrialFESpace, *ir, flags);
}


//====================================


//---[ Diffusion Integrator ]---------
RajaDiffusionIntegrator::RajaDiffusionIntegrator(const RajaCoefficient &coeff_)
   :
   RajaIntegrator(coeff_.RajaEngine()),
   coeff(coeff_),
   assembledOperator(*(new Layout(coeff_.RajaEngine(), 0)))
{
   coeff.SetName("COEFF");
}

RajaDiffusionIntegrator::~RajaDiffusionIntegrator() {}


std::string RajaDiffusionIntegrator::GetName()
{
   return "DiffusionIntegrator";
}

void RajaDiffusionIntegrator::SetupIntegrationRule()
{
   const FiniteElement &trialFE = *(trialFESpace->GetFE(0));
   const FiniteElement &testFE  = *(testFESpace->GetFE(0));
   ir = &mfem::DiffusionIntegrator::GetRule(trialFE, testFE);
}

void RajaDiffusionIntegrator::Setup()
{
   assert(false);
}

void RajaDiffusionIntegrator::Assemble()
{
   assert(false);
}

void RajaDiffusionIntegrator::MultAdd(Vector &x, Vector &y)
{
   assert(false);
   // Note: x and y are E-vectors
/*
   multKernel((int) mesh->GetNE(),
              maps.dofToQuad,
              maps.dofToQuadD,
              maps.quadToDof,
              maps.quadToDofD,
              assembledOperator.RajaMem(),
              x.RajaMem(), y.RajaMem());*/
}
//====================================


//---[ Mass Integrator ]--------------
RajaMassIntegrator::RajaMassIntegrator(const RajaCoefficient &coeff_) :
   RajaIntegrator(coeff_.RajaEngine()),
   coeff(coeff_),
   assembledOperator(*(new Layout(coeff_.RajaEngine(), 0)))
{
   coeff.SetName("COEFF");
}

RajaMassIntegrator::~RajaMassIntegrator() {}

std::string RajaMassIntegrator::GetName()
{
   return "MassIntegrator";
}

void RajaMassIntegrator::SetupIntegrationRule()
{
   const mfem::FiniteElement &trialFE = *(trialFESpace->GetFE(0));
   const mfem::FiniteElement &testFE  = *(testFESpace->GetFE(0));
   mfem::ElementTransformation &T = *trialFESpace->GetElementTransformation(0);
   ir = &mfem::MassIntegrator::GetRule(trialFE, testFE, T);
}

void RajaMassIntegrator::Setup()
{
   assert(false);/*
   ::raja::properties kernelProps = props;

   coeff.Setup(*this, kernelProps);

   // Setup assemble and mult kernels
   assembleKernel = GetAssembleKernel(kernelProps);
   multKernel     = GetMultAddKernel(kernelProps);*/
}

void RajaMassIntegrator::Assemble()
{
    assert(false);/*
  if (assembledOperator.Size())
   {
      return;
   }

   const int elements = trialFESpace->GetNE();
   const int quadraturePoints = ir->GetNPoints();

   RajaGeometry geom = GetGeometry(RajaGeometry::Jacobian);

   assembledOperator.Resize<double>(quadraturePoints * elements, NULL);

   assembleKernel((int) mesh->GetNE(),
                  maps.quadWeights,
                  geom.J,
                  coeff,
                  assembledOperator.RajaMem());*/
}

void RajaMassIntegrator::SetOperator(Vector &v)
{
   assembledOperator = v;
}

void RajaMassIntegrator::MultAdd(Vector &x, Vector &y)
{
     assert(false);/*
 multKernel((int) mesh->GetNE(),
              maps.dofToQuad,
              maps.dofToQuadD,
              maps.quadToDof,
              maps.quadToDofD,
              assembledOperator.RajaMem(),
              x.RajaMem(), y.RajaMem());*/
}
//====================================


//---[ Vector Mass Integrator ]--------------
RajaVectorMassIntegrator::RajaVectorMassIntegrator(const RajaCoefficient &
                                                   coeff_)
   :
   RajaIntegrator(coeff_.RajaEngine()),
   coeff(coeff_),
   assembledOperator(*(new Layout(coeff_.RajaEngine(), 0)))
{
   coeff.SetName("COEFF");
}

RajaVectorMassIntegrator::~RajaVectorMassIntegrator() {}

std::string RajaVectorMassIntegrator::GetName()
{
   return "VectorMassIntegrator";
}

void RajaVectorMassIntegrator::SetupIntegrationRule()
{
   const mfem::FiniteElement &trialFE = *(trialFESpace->GetFE(0));
   const mfem::FiniteElement &testFE  = *(testFESpace->GetFE(0));
   mfem::ElementTransformation &T = *trialFESpace->GetElementTransformation(0);
   ir = &mfem::MassIntegrator::GetRule(trialFE, testFE, T);
}

void RajaVectorMassIntegrator::Setup()
{
     assert(false);/*
 ::raja::properties kernelProps = props;

   coeff.Setup(*this, kernelProps);

   // Setup assemble and mult kernels
   assembleKernel = GetAssembleKernel(kernelProps);
   multKernel     = GetMultAddKernel(kernelProps);*/
}

void RajaVectorMassIntegrator::Assemble()
{
      assert(false);/*
const int elements = trialFESpace->GetNE();
   const int quadraturePoints = ir->GetNPoints();

   RajaGeometry geom = GetGeometry(RajaGeometry::Jacobian);

   assembledOperator.Resize<double>(quadraturePoints * elements, NULL);

   assembleKernel((int) mesh->GetNE(),
                  maps.quadWeights,
                  geom.J,
                  coeff,
                  assembledOperator.RajaMem());*/
}

void RajaVectorMassIntegrator::MultAdd(Vector &x, Vector &y)
{
     assert(false);/*
 multKernel((int) mesh->GetNE(),
              maps.dofToQuad,
              maps.dofToQuadD,
              maps.quadToDof,
              maps.quadToDofD,
              assembledOperator.RajaMem(),
              x.RajaMem(), y.RajaMem());*/
}

} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)
