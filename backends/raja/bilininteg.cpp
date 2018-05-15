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
// *****************************************************************************
static RajaGeometry *geom=NULL;

// ***************************************************************************
// * ~ RajaGeometry
// ***************************************************************************
RajaGeometry::~RajaGeometry(){
  push(SteelBlue);
  free(geom->meshNodes);
  free(geom->J);
  free(geom->invJ);
  free(geom->detJ);
  delete[] geom;
  pop();
}

// *****************************************************************************
// * RajaGeometry Get: use this one to fetch nodes from vector Sx
// *****************************************************************************
RajaGeometry* RajaGeometry::GetV(FiniteElementSpace& fes,
                                const IntegrationRule& ir,
                                const RajaVector& Sx) {
  push(SteelBlue);
  const Mesh *mesh = fes.GetMesh();
  const GridFunction *nodes = mesh->GetNodes();
  const mfem::FiniteElementSpace *fespace = nodes->FESpace();
  const mfem::FiniteElement *fe = fespace->GetFE(0);
  const int dims     = fe->GetDim();
  const int numDofs  = fe->GetDof();
  const int numQuad  = ir.GetNPoints();
  const int elements = fespace->GetNE();
  const int ndofs    = fespace->GetNDofs();
  const RajaDofQuadMaps* maps = RajaDofQuadMaps::GetSimplexMaps(*fe, ir);
  push(rNodeCopyByVDim,SteelBlue);
  rNodeCopyByVDim(elements,numDofs,ndofs,dims,geom->eMap,Sx,geom->meshNodes);
  pop(rNodeCopyByVDim);
  push(rIniGeom,SteelBlue);
  rIniGeom(dims,numDofs,numQuad,elements,
           maps->dofToQuadD,
           geom->meshNodes,
           geom->J,
           geom->invJ,
           geom->detJ);
  pop(rIniGeom);
  pop();
  return geom;
}
  

// *****************************************************************************
RajaGeometry* RajaGeometry::Get(FiniteElementSpace& fes,
                                const IntegrationRule& ir) {
  push(SteelBlue);
  Mesh& mesh = *(fes.GetMesh());
  const bool geom_to_allocate =
    (!geom) || rconfig::Get().GeomNeedsUpdate(mesh.GetSequence());
  if (geom_to_allocate) geom=new RajaGeometry();
  if (!mesh.GetNodes()) mesh.SetCurvature(1, false, -1, Ordering::byVDIM);
  GridFunction& nodes = *(mesh.GetNodes());
  const mfem::FiniteElementSpace& fespace = *(nodes.FESpace());
  const mfem::FiniteElement& fe = *(fespace.GetFE(0));
  const int dims     = fe.GetDim();
  const int elements = fespace.GetNE();
  const int numDofs  = fe.GetDof();
  const int numQuad  = ir.GetNPoints();
  const bool orderedByNODES = (fespace.GetOrdering() == Ordering::byNODES);
 
  if (orderedByNODES) ReorderByVDim(nodes);
  const int asize = dims*numDofs*elements;
  mfem::Array<double> meshNodes(asize);
  const Table& e2dTable = fespace.GetElementToDofTable();
  const int* elementMap = e2dTable.GetJ();
  mfem::Array<int> eMap(numDofs*elements);
  {
    push(cpynodes,SteelBlue);
    for (int e = 0; e < elements; ++e) {
      for (int d = 0; d < numDofs; ++d) {
        const int lid = d+numDofs*e;
        const int gid = elementMap[lid];
        eMap[lid]=gid;
        for (int v = 0; v < dims; ++v) {
          const int moffset = v+dims*lid;
          const int xoffset = v+dims*gid;
           meshNodes[moffset] = nodes[xoffset];
        }
      }
    }
    pop();
  }
  if (geom_to_allocate){
    geom->meshNodes.allocate(dims, numDofs, elements);
    geom->eMap.allocate(numDofs, elements);
  }
  {
    push(H2D:cpyMeshNodes,SteelBlue);
    geom->meshNodes = meshNodes;
    geom->eMap = eMap;
    pop();
  }
  // Reorder the original gf back
  if (orderedByNODES) ReorderByNodes(nodes);
  if (geom_to_allocate){
    geom->J.allocate(dims, dims, numQuad, elements);
    geom->invJ.allocate(dims, dims, numQuad, elements);
    geom->detJ.allocate(numQuad, elements);
  }
    
  const RajaDofQuadMaps* maps = RajaDofQuadMaps::GetSimplexMaps(fe, ir);
  {
    push(rIniGeom,SteelBlue);
    rIniGeom(dims,numDofs,numQuad,elements,
             maps->dofToQuadD,
             geom->meshNodes,
             geom->J,
             geom->invJ,
             geom->detJ);
    pop();
  }
  pop();
  return geom;
}

// ***************************************************************************
void RajaGeometry::ReorderByVDim(GridFunction& nodes){
  push(SteelBlue);
  const mfem::FiniteElementSpace *fes=nodes.FESpace();
  const int size = nodes.Size();
  const int vdim = fes->GetVDim();
  const int ndofs = fes->GetNDofs();
  double *data = nodes.GetData();
  double *temp = new double[size];
  int k=0;
  for (int d = 0; d < ndofs; d++)
    for (int v = 0; v < vdim; v++)
      temp[k++] = data[d+v*ndofs];
  for (int i = 0; i < size; i++)
    data[i] = temp[i];
  delete [] temp;
  pop();
}

// ***************************************************************************
void RajaGeometry::ReorderByNodes(GridFunction& nodes){
  push(SteelBlue);
  const mfem::FiniteElementSpace *fes=nodes.FESpace();
  const int size = nodes.Size();
  const int vdim = fes->GetVDim();
  const int ndofs = fes->GetNDofs();
  double *data = nodes.GetData();
  double *temp = new double[size];
  int k = 0;
  for (int j = 0; j < ndofs; j++)
    for (int i = 0; i < vdim; i++)
      temp[j+i*ndofs] = data[k++];
  for (int i = 0; i < size; i++)
    data[i] = temp[i];
  delete [] temp;
  pop();
}

// ***************************************************************************
// * RajaDofQuadMaps
// ***************************************************************************
  static std::map<std::string, RajaDofQuadMaps* > AllDofQuadMaps;

  // ***************************************************************************
  RajaDofQuadMaps::~RajaDofQuadMaps(){}

// *****************************************************************************
void RajaDofQuadMaps::delRajaDofQuadMaps(){
  for(std::map<std::string,
        RajaDofQuadMaps*>::iterator itr = AllDofQuadMaps.begin();
      itr != AllDofQuadMaps.end();
      itr++) {
    delete itr->second;
  }
}

// *****************************************************************************
RajaDofQuadMaps* RajaDofQuadMaps::Get(const FiniteElementSpace& fespace,
                                      const mfem::IntegrationRule& ir,
                                      const bool transpose) {
  return Get(*fespace.GetFE(0),*fespace.GetFE(0),ir,transpose);
}

RajaDofQuadMaps* RajaDofQuadMaps::Get(const FiniteElementSpace&
                                      trialFESpace,
                                      const FiniteElementSpace& testFESpace,
                                      const mfem::IntegrationRule& ir,
                                      const bool transpose) {
  return Get(*trialFESpace.GetFE(0),*testFESpace.GetFE(0),ir,transpose);
}

RajaDofQuadMaps* RajaDofQuadMaps::Get(const mfem::FiniteElement& trialFE,
                                      const mfem::FiniteElement& testFE,
                                      const mfem::IntegrationRule& ir,
                                      const bool transpose) {
  return GetTensorMaps(trialFE, testFE, ir, transpose);
}

// ***************************************************************************
RajaDofQuadMaps* RajaDofQuadMaps::GetTensorMaps(const mfem::FiniteElement& trialFE,
                                                const mfem::FiniteElement& testFE,
                                                const mfem::IntegrationRule& ir,
                                                const bool transpose) {
  const TensorBasisElement& trialTFE =
    dynamic_cast<const TensorBasisElement&>(trialFE);
  const TensorBasisElement& testTFE =
    dynamic_cast<const TensorBasisElement&>(testFE);
  std::stringstream ss;
  ss << "TensorMap:"
     << " O1:"  << trialFE.GetOrder()
     << " O2:"  << testFE.GetOrder()
     << " BT1:" << trialTFE.GetBasisType()
     << " BT2:" << testTFE.GetBasisType()
     << " Q:"   << ir.GetNPoints();
  std::string hash = ss.str();
  // If we've already made the dof-quad maps, reuse them
  if (AllDofQuadMaps.find(hash)!=AllDofQuadMaps.end())
    return AllDofQuadMaps[hash];
  // Otherwise, build them
  RajaDofQuadMaps *maps = new RajaDofQuadMaps();
  AllDofQuadMaps[hash]=maps;
  maps->hash = hash;
  push();
  const RajaDofQuadMaps* trialMaps = GetD2QTensorMaps(trialFE, ir);
  const RajaDofQuadMaps* testMaps  = GetD2QTensorMaps(testFE, ir, true);
  maps->dofToQuad   = trialMaps->dofToQuad;
  maps->dofToQuadD  = trialMaps->dofToQuadD;
  maps->quadToDof   = testMaps->dofToQuad;
  maps->quadToDofD  = testMaps->dofToQuadD;
  maps->quadWeights = testMaps->quadWeights;
  pop();
  return maps;
}
  
// ***************************************************************************
RajaDofQuadMaps* RajaDofQuadMaps::GetD2QTensorMaps(const mfem::FiniteElement& fe,
                                                   const mfem::IntegrationRule& ir,
                                                   const bool transpose) {
  const mfem::TensorBasisElement& tfe = dynamic_cast<const TensorBasisElement&>(fe);
  const Poly_1D::Basis& basis = tfe.GetBasis1D();
  const int order = fe.GetOrder();
  const int dofs = order + 1;
  const int dims = fe.GetDim();
  const mfem::IntegrationRule& ir1D = IntRules.Get(Geometry::SEGMENT, ir.GetOrder());
  const int quadPoints = ir1D.GetNPoints();
  const int quadPoints2D = quadPoints*quadPoints;
  const int quadPoints3D = quadPoints2D*quadPoints;
  const int quadPointsND = ((dims == 1) ? quadPoints :
                            ((dims == 2) ? quadPoints2D : quadPoints3D));
  std::stringstream ss ;
  ss << "D2QTensorMap:"
     << " order:" << order
     << " dofs:" << dofs
     << " dims:" << dims
     << " quadPoints:"<<quadPoints
     << " transpose:"  << (transpose?"T":"F");
  std::string hash = ss.str();
  if (AllDofQuadMaps.find(hash)!=AllDofQuadMaps.end())
    return AllDofQuadMaps[hash];

  push(SteelBlue);
  RajaDofQuadMaps *maps = new RajaDofQuadMaps();
  AllDofQuadMaps[hash]=maps;
  maps->hash = hash;
  
  maps->dofToQuad.allocate(quadPoints, dofs,1,1,transpose);
  maps->dofToQuadD.allocate(quadPoints, dofs,1,1,transpose);
  double* quadWeights1DData = NULL;
  if (transpose) {
    // Initialize quad weights only for transpose
    maps->quadWeights.allocate(quadPointsND);
    quadWeights1DData = ::new double[quadPoints];
  }
  mfem::Vector d2q(dofs); 
  mfem::Vector d2qD(dofs);
  mfem::Array<double> dofToQuad(quadPoints*dofs);
  mfem::Array<double> dofToQuadD(quadPoints*dofs);
  for (int q = 0; q < quadPoints; ++q) {
    const IntegrationPoint& ip = ir1D.IntPoint(q);
    basis.Eval(ip.x, d2q, d2qD);
    if (transpose) {
      quadWeights1DData[q] = ip.weight;
    }
    for (int d = 0; d < dofs; ++d) {
      dofToQuad[maps->dofToQuad.dim()[0]*q + maps->dofToQuad.dim()[1]*d] = d2q[d];
      dofToQuadD[maps->dofToQuad.dim()[0]*q + maps->dofToQuad.dim()[1]*d] = d2qD[d];
    }
  }
  maps->dofToQuad = dofToQuad;
  maps->dofToQuadD = dofToQuadD;
  if (transpose) {
    mfem::Array<double> quadWeights(quadPointsND);
    for (int q = 0; q < quadPointsND; ++q) {
      const int qx = q % quadPoints;
      const int qz = q / quadPoints2D;
      const int qy = (q - qz*quadPoints2D) / quadPoints;
      double w = quadWeights1DData[qx];
      if (dims > 1) {
        w *= quadWeights1DData[qy];
      }
      if (dims > 2) {
        w *= quadWeights1DData[qz];
      }
      quadWeights[q] = w;
    }
    maps->quadWeights = quadWeights;
    ::delete [] quadWeights1DData;
  }
  assert(maps);
  pop();
  return maps;
}

// ***************************************************************************
RajaDofQuadMaps* RajaDofQuadMaps::GetSimplexMaps(const mfem::FiniteElement& fe,
                                                 const mfem::IntegrationRule& ir,
                                                 const bool transpose) {
  return GetSimplexMaps(fe, fe, ir, transpose);
}

// *****************************************************************************
RajaDofQuadMaps* RajaDofQuadMaps::GetSimplexMaps(const mfem::FiniteElement& trialFE,
                                                 const mfem::FiniteElement& testFE,
                                                 const mfem::IntegrationRule& ir,
                                                 const bool transpose) {
  std::stringstream ss;
  ss << "SimplexMap:"
     << " O1:" << trialFE.GetOrder()
     << " O2:" << testFE.GetOrder()
     << " Q:"  << ir.GetNPoints();
  std::string hash = ss.str();
  // If we've already made the dof-quad maps, reuse them
  if (AllDofQuadMaps.find(hash)!=AllDofQuadMaps.end())
    return AllDofQuadMaps[hash];
  push(SteelBlue);
  RajaDofQuadMaps *maps = new RajaDofQuadMaps();
  AllDofQuadMaps[hash]=maps;
  maps->hash = hash;
  const RajaDofQuadMaps* trialMaps = GetD2QSimplexMaps(trialFE, ir);
  const RajaDofQuadMaps* testMaps  = GetD2QSimplexMaps(testFE, ir, true);
  maps->dofToQuad   = trialMaps->dofToQuad;
  maps->dofToQuadD  = trialMaps->dofToQuadD;
  maps->quadToDof   = testMaps->dofToQuad;
  maps->quadToDofD  = testMaps->dofToQuadD;
  maps->quadWeights = testMaps->quadWeights;
  pop();
  return maps;
}

// ***************************************************************************
RajaDofQuadMaps* RajaDofQuadMaps::GetD2QSimplexMaps(const mfem::FiniteElement& fe,
                                                    const mfem::IntegrationRule& ir,
                                                    const bool transpose) {
  const int dims = fe.GetDim();
  const int numDofs = fe.GetDof();
  const int numQuad = ir.GetNPoints();
  std::stringstream ss ;
  ss << "D2QSimplexMap:"
     << " Dim:" << dims
     << " numDofs:" << numDofs
     << " numQuad:" << numQuad
     << " transpose:"  << (transpose?"T":"F");
  std::string hash = ss.str();
  if (AllDofQuadMaps.find(hash)!=AllDofQuadMaps.end())
    return AllDofQuadMaps[hash];
  RajaDofQuadMaps* maps = new RajaDofQuadMaps();
  AllDofQuadMaps[hash]=maps;
  maps->hash = hash;  
  push(SteelBlue);
  // Initialize the dof -> quad mapping
  maps->dofToQuad.allocate(numQuad, numDofs,1,1,transpose);
  maps->dofToQuadD.allocate(dims, numQuad, numDofs,1,transpose);
  if (transpose) // Initialize quad weights only for transpose
    maps->quadWeights.allocate(numQuad);
  mfem::Vector d2q(numDofs);
  mfem::DenseMatrix d2qD(numDofs, dims);
  mfem::Array<double> quadWeights(numQuad);
  mfem::Array<double> dofToQuad(numQuad*numDofs);
  mfem::Array<double> dofToQuadD(dims*numQuad*numDofs);  
  for (int q = 0; q < numQuad; ++q) {
    const IntegrationPoint& ip = ir.IntPoint(q);
    if (transpose) {
      quadWeights[q] = ip.weight;
    }
    fe.CalcShape(ip, d2q);
    fe.CalcDShape(ip, d2qD);
    for (int d = 0; d < numDofs; ++d) {
      const double w = d2q[d];
      dofToQuad[maps->dofToQuad.dim()[0]*q +
                maps->dofToQuad.dim()[1]*d] = w;
      for (int dim = 0; dim < dims; ++dim) {
        const double wD = d2qD(d, dim);
        dofToQuadD[maps->dofToQuadD.dim()[0]*dim +
                   maps->dofToQuadD.dim()[1]*q +
                   maps->dofToQuadD.dim()[2]*d] = wD;
      }
    }
  }
  if (transpose) 
    maps->quadWeights = quadWeights;
  maps->dofToQuad = dofToQuad;
  maps->dofToQuadD = dofToQuadD;
  pop();
  return maps;
}

/*
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
      assert(false);//                      geom.J.allocate(device, 1);
   }
   if (flags & JacobianInv)
   {
      assert(false);
      geom.invJ.allocate(device,
      dims, dims, numQuad, elements);
   }
   else
   {
       assert(false);
                       geom.invJ.allocate(device, 1);
   }
   if (flags & JacobianDet)
   {
      assert(false);
      geom.detJ.allocate(device,
      numQuad, elements);
   }
   else
   {
      assert(false);
                      geom.detJ.allocate(device, 1);
   }

   //geom.J.stopManaging();
   //geom.invJ.stopManaging();
   //geom.detJ.stopManaging();

   RajaDofQuadMaps &maps = RajaDofQuadMaps::GetSimplexMaps(device, fe, ir);
   assert(false);
   init(elements,
        maps.dofToQuadD,
        geom.meshNodes,
        geom.J, geom.invJ, geom.detJ);

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

   RajaDofQuadMaps maps;
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
   }
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
      assert(false);
                      maps.quadWeights.allocate(device,numQuad);
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
*/

//---[ Integrator Defines ]-----------
/*std::string stringWithDim(const std::string &s, const int dim)
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

*/
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
   maps = RajaDofQuadMaps::Get(*otrialFESpace,
                               *otestFESpace,
                               *ir);

   mapsTranspose = RajaDofQuadMaps::Get(*otestFESpace,
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

RajaDofQuadMaps *RajaIntegrator::GetDofQuadMaps()
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

   //SetProperties(*otrialFESpace,*otestFESpace,*ir);

   Setup();
}

RajaGeometry *RajaIntegrator::GetGeometry(const int flags)
{
   push();
   pop();
   return RajaGeometry::Get(*otrialFESpace, *ir/*, flags*/);
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
}

void RajaDiffusionIntegrator::Assemble()
{
   push(SteelBlue);
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
                      assembledOperator.RajaMem());
/*   for(size_t i=0;i<assembledOperator.Size();i+=1)
      printf("\n\t\033[35m[Assemble] assembledOperator[%ld]=%f",i,
      ((double*)assembledOperator.RajaMem().ptr())[i]);
*/
    pop();
}

void RajaDiffusionIntegrator::MultAdd(Vector &x, Vector &y)
{
   push(SteelBlue);
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
                     assembledOperator.RajaMem(),
                     x.RajaMem(),
                     y.RajaMem());
/*   for(size_t i=0;i<y.Size();i+=1)
      printf("\n\t\033[36m[MultAdd] y[%ld]=%f",i, ((double*)y.RajaMem().ptr())[i]);
*/
/*
  multKernel((int) mesh->GetNE(),
  maps.dofToQuad,
              maps.dofToQuadD,
              maps.quadToDof,
              maps.quadToDofD,
              assembledOperator.RajaMem(),
              x.RajaMem(), y.RajaMem());*/
  pop();
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
