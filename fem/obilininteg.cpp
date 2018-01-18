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

#include "../config/config.hpp"

#ifdef MFEM_USE_OCCA
#include <iostream>
#include <cmath>

#include "obilininteg.hpp"

namespace mfem {
  std::map<std::string, OccaDofQuadMaps> OccaDofQuadMaps::AllDofQuadMaps;

  OccaGeometry OccaGeometry::Get(occa::device device,
                                 OccaFiniteElementSpace &ofespace,
                                 const IntegrationRule &ir,
                                 const int flags) {

    OccaGeometry geom;

    Mesh &mesh = *(ofespace.GetMesh());
    if (!mesh.GetNodes()) {
      mesh.SetCurvature(1, false, -1, Ordering::byVDIM);
    }
    GridFunction &nodes = *(mesh.GetNodes());
    const FiniteElementSpace &fespace = *(nodes.FESpace());
    const FiniteElement &fe = *(fespace.GetFE(0));

    const int dims     = fe.GetDim();
    const int elements = fespace.GetNE();
    const int numDofs  = fe.GetDof();
    const int numQuad  = ir.GetNPoints();

    Ordering::Type originalOrdering = fespace.GetOrdering();
    if (originalOrdering == Ordering::byNODES) {
      nodes.ReorderByVDim(true);
    }

    geom.meshNodes.allocate(device,
                            dims, numDofs, elements);

    const Table &e2dTable = fespace.GetElementToDofTable();
    const int *elementMap = e2dTable.GetJ();
    for (int e = 0; e < elements; ++e) {
      for (int dof = 0; dof < numDofs; ++dof) {
        const int gid = elementMap[dof + numDofs*e];
        for (int dim = 0; dim < dims; ++dim) {
          geom.meshNodes(dim, dof, e) = nodes[dim + gid*dims];
        }
      }
    }
    geom.meshNodes.keepInDevice();

    // Reorder the original gf back
    if (originalOrdering == Ordering::byNODES) {
      nodes.ReorderByNodes(true);
    }

    if (flags & Jacobian) {
      geom.J.allocate(device,
                      dims, dims, numQuad, elements);
    } else {
      geom.J.allocate(device, 1);
    }
    if (flags & JacobianInv) {
      geom.invJ.allocate(device,
                         dims, dims, numQuad, elements);
    } else {
      geom.invJ.allocate(device, 1);
    }
    if (flags & JacobianDet) {
      geom.detJ.allocate(device,
                         numQuad, elements);
    } else {
      geom.detJ.allocate(device, 1);
    }

    geom.J.stopManaging();
    geom.invJ.stopManaging();
    geom.detJ.stopManaging();

    OccaDofQuadMaps &maps = OccaDofQuadMaps::GetSimplexMaps(device, fe, ir);

    occa::properties props;
    props["defines/NUM_DOFS"] = numDofs;
    props["defines/NUM_QUAD"] = numQuad;
    props["defines/STORE_JACOBIAN"]     = (flags & Jacobian);
    props["defines/STORE_JACOBIAN_INV"] = (flags & JacobianInv);
    props["defines/STORE_JACOBIAN_DET"] = (flags & JacobianDet);

    occa::kernel init = device.buildKernel("occa://mfem/fem/geometry.okl",
                                           stringWithDim("InitGeometryInfo", fe.GetDim()),
                                           props);
    init(elements,
         maps.dofToQuadD,
         geom.meshNodes,
         geom.J, geom.invJ, geom.detJ);

    return geom;
  }

  OccaDofQuadMaps::OccaDofQuadMaps() :
    hash() {}

  OccaDofQuadMaps::OccaDofQuadMaps(const OccaDofQuadMaps &maps) {
    *this = maps;
  }

  OccaDofQuadMaps& OccaDofQuadMaps::operator = (const OccaDofQuadMaps &maps) {
    hash = maps.hash;
    dofToQuad   = maps.dofToQuad;
    dofToQuadD  = maps.dofToQuadD;
    quadToDof   = maps.quadToDof;
    quadToDofD  = maps.quadToDofD;
    quadWeights = maps.quadWeights;
    return *this;
  }

  OccaDofQuadMaps& OccaDofQuadMaps::Get(occa::device device,
                                        const OccaFiniteElementSpace &fespace,
                                        const IntegrationRule &ir,
                                        const bool transpose) {

    return Get(device,
               *fespace.GetFE(0),
               *fespace.GetFE(0),
               ir,
               transpose);
  }

  OccaDofQuadMaps& OccaDofQuadMaps::Get(occa::device device,
                                        const FiniteElement &fe,
                                        const IntegrationRule &ir,
                                        const bool transpose) {

    return Get(device, fe, fe, ir, transpose);
  }

  OccaDofQuadMaps& OccaDofQuadMaps::Get(occa::device device,
                                        const OccaFiniteElementSpace &trialFESpace,
                                        const OccaFiniteElementSpace &testFESpace,
                                        const IntegrationRule &ir,
                                        const bool transpose) {

    return Get(device,
               *trialFESpace.GetFE(0),
               *testFESpace.GetFE(0),
               ir,
               transpose);
  }

  OccaDofQuadMaps& OccaDofQuadMaps::Get(occa::device device,
                                        const FiniteElement &trialFE,
                                        const FiniteElement &testFE,
                                        const IntegrationRule &ir,
                                        const bool transpose) {

    return (dynamic_cast<const TensorBasisElement*>(&trialFE)
            ? GetTensorMaps(device, trialFE, testFE, ir, transpose)
            : GetSimplexMaps(device, trialFE, testFE, ir, transpose));
  }

  OccaDofQuadMaps& OccaDofQuadMaps::GetTensorMaps(occa::device device,
                                                  const FiniteElement &fe,
                                                  const IntegrationRule &ir,
                                                  const bool transpose) {

    return GetTensorMaps(device,
                         fe, fe,
                         ir, transpose);
  }

  OccaDofQuadMaps& OccaDofQuadMaps::GetTensorMaps(occa::device device,
                                                  const FiniteElement &trialFE,
                                                  const FiniteElement &testFE,
                                                  const IntegrationRule &ir,
                                                  const bool transpose) {
    const TensorBasisElement &trialTFE =
      dynamic_cast<const TensorBasisElement&>(trialFE);
    const TensorBasisElement &testTFE =
      dynamic_cast<const TensorBasisElement&>(testFE);

    std::stringstream ss;
    ss << occa::hash(device)
       << "Tensor"
       << "O1:"  << trialFE.GetOrder()
       << "O2:"  << testFE.GetOrder()
       << "BT1:" << trialTFE.GetBasisType()
       << "BT2:" << testTFE.GetBasisType()
       << "Q:"   << ir.GetNPoints();
    std::string hash = ss.str();

    // If we've already made the dof-quad maps, reuse them
    OccaDofQuadMaps &maps = AllDofQuadMaps[hash];
    if (!maps.hash.size()) {
      // Create the dof-quad maps
      maps.hash = hash;

      OccaDofQuadMaps trialMaps = GetD2QTensorMaps(device, trialFE, ir);
      OccaDofQuadMaps testMaps  = GetD2QTensorMaps(device, testFE , ir, true);

      maps.dofToQuad   = trialMaps.dofToQuad;
      maps.dofToQuadD  = trialMaps.dofToQuadD;
      maps.quadToDof   = testMaps.dofToQuad;
      maps.quadToDofD  = testMaps.dofToQuadD;
      maps.quadWeights = testMaps.quadWeights;
    }
    return maps;
  }

  OccaDofQuadMaps OccaDofQuadMaps::GetD2QTensorMaps(occa::device device,
                                                    const FiniteElement &fe,
                                                    const IntegrationRule &ir,
                                                    const bool transpose) {

    const TensorBasisElement &tfe = dynamic_cast<const TensorBasisElement&>(fe);

    const Poly_1D::Basis &basis = tfe.GetBasis1D();
    const int order = fe.GetOrder();
    // [MISSING] Get 1D dofs
    const int dofs = order + 1;
    const int dims = fe.GetDim();

    // Create the dof -> quadrature point map
    const IntegrationRule &ir1D = IntRules.Get(Geometry::SEGMENT, ir.GetOrder());
    const int quadPoints = ir1D.GetNPoints();
    const int quadPoints2D = quadPoints*quadPoints;
    const int quadPoints3D = quadPoints2D*quadPoints;
    const int quadPointsND = ((dims == 1) ? quadPoints :
                              ((dims == 2) ? quadPoints2D : quadPoints3D));

    OccaDofQuadMaps maps;
    // Initialize the dof -> quad mapping
    maps.dofToQuad.allocate(device,
                            quadPoints, dofs);
    maps.dofToQuadD.allocate(device,
                             quadPoints, dofs);

    double *quadWeights1DData = NULL;

    if (transpose) {
      maps.dofToQuad.reindex(1,0);
      maps.dofToQuadD.reindex(1,0);
      // Initialize quad weights only for transpose
      maps.quadWeights.allocate(device,
                                quadPointsND);
      quadWeights1DData = new double[quadPoints];
    }

    mfem::Vector d2q(dofs);
    mfem::Vector d2qD(dofs);
    for (int q = 0; q < quadPoints; ++q) {
      const IntegrationPoint &ip = ir1D.IntPoint(q);
      basis.Eval(ip.x, d2q, d2qD);
      if (transpose) {
        quadWeights1DData[q] = ip.weight;
      }
      for (int d = 0; d < dofs; ++d) {
        maps.dofToQuad(q, d)  = d2q[d];
        maps.dofToQuadD(q, d) = d2qD[d];
      }
    }

    maps.dofToQuad.keepInDevice();
    maps.dofToQuadD.keepInDevice();

    if (transpose) {
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
        maps.quadWeights[q] = w;
      }
      maps.quadWeights.keepInDevice();
      delete [] quadWeights1DData;
    }

    return maps;
  }

  OccaDofQuadMaps& OccaDofQuadMaps::GetSimplexMaps(occa::device device,
                                                   const FiniteElement &fe,
                                                   const IntegrationRule &ir,
                                                   const bool transpose) {

    return GetSimplexMaps(device,
                          fe, fe,
                          ir, transpose);
  }

  OccaDofQuadMaps& OccaDofQuadMaps::GetSimplexMaps(occa::device device,
                                                   const FiniteElement &trialFE,
                                                   const FiniteElement &testFE,
                                                   const IntegrationRule &ir,
                                                   const bool transpose) {

    std::stringstream ss;
    ss << occa::hash(device)
       << "Simplex"
       << "O1:" << trialFE.GetOrder()
       << "O2:" << testFE.GetOrder()
       << "Q:"  << ir.GetNPoints();
    std::string hash = ss.str();

    // If we've already made the dof-quad maps, reuse them
    OccaDofQuadMaps &maps = AllDofQuadMaps[hash];
    if (!maps.hash.size()) {
      // Create the dof-quad maps
      maps.hash = hash;

      OccaDofQuadMaps trialMaps = GetD2QSimplexMaps(device, trialFE, ir);
      OccaDofQuadMaps testMaps  = GetD2QSimplexMaps(device, testFE , ir, true);

      maps.dofToQuad   = trialMaps.dofToQuad;
      maps.dofToQuadD  = trialMaps.dofToQuadD;
      maps.quadToDof   = testMaps.dofToQuad;
      maps.quadToDofD  = testMaps.dofToQuadD;
      maps.quadWeights = testMaps.quadWeights;
    }
    return maps;
  }

  OccaDofQuadMaps OccaDofQuadMaps::GetD2QSimplexMaps(occa::device device,
                                                     const FiniteElement &fe,
                                                     const IntegrationRule &ir,
                                                     const bool transpose) {
    const int dims = fe.GetDim();
    const int numDofs = fe.GetDof();
    const int numQuad = ir.GetNPoints();

    OccaDofQuadMaps maps;
    // Initialize the dof -> quad mapping
    maps.dofToQuad.allocate(device,
                            numQuad, numDofs);
    maps.dofToQuadD.allocate(device,
                             dims, numQuad, numDofs);

    if (transpose) {
      maps.dofToQuad.reindex(1,0);
      maps.dofToQuadD.reindex(1,0);
      // Initialize quad weights only for transpose
      maps.quadWeights.allocate(device,
                                numQuad);
    }

    Vector d2q(numDofs);
    DenseMatrix d2qD(numDofs, dims);
    for (int q = 0; q < numQuad; ++q) {
      const IntegrationPoint &ip = ir.IntPoint(q);
      if (transpose) {
        maps.quadWeights[q] = ip.weight;
      }
      fe.CalcShape(ip, d2q);
      fe.CalcDShape(ip, d2qD);
      for (int d = 0; d < numDofs; ++d) {
        const double w = d2q[d];
        maps.dofToQuad(q, d) = w;
        for (int dim = 0; dim < dims; ++dim) {
          const double wD = d2qD(d, dim);
          maps.dofToQuadD(dim, q, d) = wD;
        }
      }
    }

    maps.dofToQuad.keepInDevice();
    maps.dofToQuadD.keepInDevice();
    if (transpose) {
      maps.quadWeights.keepInDevice();
    }

    return maps;
  }

  //---[ Integrator Defines ]-----------
  std::string stringWithDim(const std::string &s, const int dim) {
    std::string ret = s;
    ret += ('0' + (char) dim);
    ret += 'D';
    return ret;
  }

  int closestWarpBatchTo(const int value) {
    return ((value + 31) / 32) * 32;
  }

  int closestMultipleWarpBatch(const int multiple, const int maxSize) {
    if (multiple > maxSize) {
      return maxSize;
    }
    int batch = (32 / multiple);
    int minDiff = 32 - (multiple * batch);
    for (int i = 64; i <= maxSize; i += 32) {
      const int newDiff = i - (multiple * (i / multiple));
      if (newDiff < minDiff) {
        batch = (i / multiple);
        minDiff = newDiff;
      }
    }
    return batch;
  }

  void SetProperties(OccaFiniteElementSpace &fespace,
                     const IntegrationRule &ir,
                     occa::properties &props) {

    SetProperties(fespace, fespace, ir, props);
  }

  void SetProperties(OccaFiniteElementSpace &trialFESpace,
                     OccaFiniteElementSpace &testFESpace,
                     const IntegrationRule &ir,
                     occa::properties &props) {

    const FiniteElement &trialFE = *(trialFESpace.GetFE(0));
    const FiniteElement &testFE  = *(testFESpace.GetFE(0));

    props["defines/TRIAL_VDIM"] = trialFESpace.GetVDim();
    props["defines/TEST_VDIM"]  = testFESpace.GetVDim();
    props["defines/NUM_DIM"]    = trialFESpace.GetDim();

    if (trialFESpace.hasTensorBasis()) {
      SetTensorProperties(trialFESpace, testFESpace, ir, props);
    } else {
      SetSimplexProperties(trialFESpace, testFESpace, ir, props);
    }
  }

  void SetTensorProperties(OccaFiniteElementSpace &fespace,
                           const IntegrationRule &ir,
                           occa::properties &props) {

    SetTensorProperties(fespace, fespace, ir, props);
  }

  void SetTensorProperties(OccaFiniteElementSpace &trialFESpace,
                           OccaFiniteElementSpace &testFESpace,
                           const IntegrationRule &ir,
                           occa::properties &props) {

    const FiniteElement &trialFE = *(trialFESpace.GetFE(0));
    const FiniteElement &testFE = *(testFESpace.GetFE(0));

    const IntegrationRule &ir1D = IntRules.Get(Geometry::SEGMENT, ir.GetOrder());

    const int trialDofs = trialFE.GetDof();
    const int testDofs  = testFE.GetDof();
    const int numQuad  = ir.GetNPoints();

    const int trialDofs1D = trialFE.GetOrder() + 1;
    const int testDofs1D  = testFE.GetOrder() + 1;
    const int quad1D  = ir1D.GetNPoints();
    int trialDofsND = trialDofs1D;
    int testDofsND  = testDofs1D;
    int quadND  = quad1D;

    const bool trialByVDIM = (trialFESpace.GetOrdering() == Ordering::byVDIM);
    const bool testByVDIM  = (testFESpace.GetOrdering()  == Ordering::byVDIM);

    props["defines/ORDERING_BY_NODES"] = 0;
    props["defines/ORDERING_BY_VDIM"]  = 1;
    props["defines/VDIM_ORDERING"]  = (int) trialByVDIM;
    props["defines/TRIAL_ORDERING"] = (int) trialByVDIM;
    props["defines/TEST_ORDERING"]  = (int) testByVDIM;

    props["defines/USING_TENSOR_OPS"] = 1;
    props["defines/NUM_DOFS"]  = trialDofs;
    props["defines/NUM_QUAD"]  = numQuad;

    props["defines/TRIAL_DOFS"] = trialDofs;
    props["defines/TEST_DOFS"]  = testDofs;

    for (int d = 1; d <= 3; ++d) {
      if (d > 1) {
        trialDofsND *= trialDofs1D;
        testDofsND  *= testDofs1D;
        quadND *= quad1D;
      }
      props["defines"][stringWithDim("NUM_DOFS_", d)] = trialDofsND;
      props["defines"][stringWithDim("NUM_QUAD_", d)] = quadND;

      props["defines"][stringWithDim("TRIAL_DOFS_", d)] = trialDofsND;
      props["defines"][stringWithDim("TEST_DOFS_" , d)] = testDofsND;
    }

    // 1D Defines
    const int m1InnerBatch = 32 * ((quad1D + 31) / 32);
    props["defines/A1_ELEMENT_BATCH"]       = closestMultipleWarpBatch(quad1D, 512);
    props["defines/M1_OUTER_ELEMENT_BATCH"] = closestMultipleWarpBatch(m1InnerBatch, 512);
    props["defines/M1_INNER_ELEMENT_BATCH"] = m1InnerBatch;

    // 2D Defines
    props["defines/A2_ELEMENT_BATCH"] = 1;
    props["defines/A2_QUAD_BATCH"]    = 1;
    props["defines/M2_ELEMENT_BATCH"] = 32;

    // 3D Defines
    const int a3QuadBatch = closestMultipleWarpBatch(quadND, 512);
    props["defines/A3_ELEMENT_BATCH"] = closestMultipleWarpBatch(a3QuadBatch, 512);
    props["defines/A3_QUAD_BATCH"]    = a3QuadBatch;
  }

  void SetSimplexProperties(OccaFiniteElementSpace &fespace,
                            const IntegrationRule &ir,
                            occa::properties &props) {

    SetSimplexProperties(fespace, fespace, ir, props);
  }

  void SetSimplexProperties(OccaFiniteElementSpace &trialFESpace,
                            OccaFiniteElementSpace &testFESpace,
                            const IntegrationRule &ir,
                            occa::properties &props) {


    const FiniteElement &trialFE = *(trialFESpace.GetFE(0));
    const FiniteElement &testFE = *(testFESpace.GetFE(0));

    const int trialDofs = trialFE.GetDof();
    const int testDofs  = testFE.GetDof();
    const int numQuad = ir.GetNPoints();
    const int maxDQ   = std::max(std::max(trialDofs, testDofs), numQuad);

    const bool trialByVDIM = (trialFESpace.GetOrdering() == Ordering::byVDIM);
    const bool testByVDIM  = (testFESpace.GetOrdering()  == Ordering::byVDIM);

    props["defines/ORDERING_BY_NODES"] = 0;
    props["defines/ORDERING_BY_VDIM"]  = 1;
    props["defines/VDIM_ORDERING"]  = (int) trialByVDIM;
    props["defines/TRIAL_ORDERING"] = (int) trialByVDIM;
    props["defines/TEST_ORDERING"]  = (int) testByVDIM;

    props["defines/USING_TENSOR_OPS"] = 0;
    props["defines/NUM_DOFS"]  = trialDofs;
    props["defines/NUM_QUAD"]  = numQuad;

    props["defines/TRIAL_DOFS"] = trialDofs;
    props["defines/TEST_DOFS"]  = testDofs;

    // 2D Defines
    const int quadBatch = closestWarpBatchTo(numQuad);
    props["defines/A2_ELEMENT_BATCH"] = closestMultipleWarpBatch(quadBatch, 2048);
    props["defines/A2_QUAD_BATCH"]    = quadBatch;
    props["defines/M2_INNER_BATCH"]   = closestWarpBatchTo(maxDQ);

    // 3D Defines
    props["defines/A3_ELEMENT_BATCH"] = closestMultipleWarpBatch(quadBatch, 2048);
    props["defines/A3_QUAD_BATCH"]    = quadBatch;
    props["defines/M3_INNER_BATCH"]   = closestWarpBatchTo(maxDQ);
  }

  //---[ Base Integrator ]--------------
  OccaIntegrator::OccaIntegrator() :
    ir(NULL) {}

  OccaIntegrator::~OccaIntegrator() {}

  void OccaIntegrator::SetupMaps() {
    maps = OccaDofQuadMaps::Get(device,
                                *otrialFESpace,
                                *otestFESpace,
                                *ir);

    mapsTranspose = OccaDofQuadMaps::Get(device,
                                         *otestFESpace,
                                         *otrialFESpace,
                                         *ir);
  }

  occa::device OccaIntegrator::GetDevice() {
    return device;
  }

  OccaFiniteElementSpace& OccaIntegrator::GetTrialOccaFESpace() const {
    return *otrialFESpace;
  }

  OccaFiniteElementSpace& OccaIntegrator::GetTestOccaFESpace() const {
    return *otestFESpace;
  }

  FiniteElementSpace& OccaIntegrator::GetTrialFESpace() const {
    return *trialFESpace;
  }

  FiniteElementSpace& OccaIntegrator::GetTestFESpace() const {
    return *testFESpace;
  }

  void OccaIntegrator::SetIntegrationRule(const IntegrationRule &ir_) {
    ir = &ir_;
  }

  const IntegrationRule& OccaIntegrator::GetIntegrationRule() const {
    return *ir;
  }

  OccaDofQuadMaps& OccaIntegrator::GetDofQuadMaps() {
    return maps;
  }

  void OccaIntegrator::SetupIntegrator(OccaBilinearForm &bform_,
                                       const occa::properties &props_,
                                       const OccaIntegratorType itype_) {
    device    = bform_.device;
    bform     = &bform_;
    mesh      = &(bform_.GetMesh());

    otrialFESpace = &(bform_.GetTrialOccaFESpace());
    otestFESpace  = &(bform_.GetTestOccaFESpace());

    trialFESpace = &(bform_.GetTrialFESpace());
    testFESpace  = &(bform_.GetTestFESpace());

    hasTensorBasis = otrialFESpace->hasTensorBasis();

    props = props_;
    itype = itype_;

    if (ir == NULL) {
      SetupIntegrationRule();
    }

    SetupMaps();

    SetProperties(*otrialFESpace,
                  *otestFESpace,
                  *ir,
                  props);

    Setup();
  }

  OccaGeometry OccaIntegrator::GetGeometry(const int flags) {
    return OccaGeometry::Get(device, *otrialFESpace, *ir, flags);
  }

  occa::kernel OccaIntegrator::GetAssembleKernel(const occa::properties &props) {
    const FiniteElement &fe = *(trialFESpace->GetFE(0));
    return GetKernel(stringWithDim("Assemble", fe.GetDim()),
                     props);
  }

  occa::kernel OccaIntegrator::GetMultAddKernel(const occa::properties &props) {
    const FiniteElement &fe = *(trialFESpace->GetFE(0));
    return GetKernel(stringWithDim("MultAdd", fe.GetDim()),
                     props);
  }

  occa::kernel OccaIntegrator::GetKernel(const std::string &kernelName,
                                         const occa::properties &props) {
    const std::string filename = GetName() + ".okl";
    return device.buildKernel("occa://mfem/fem/" + filename,
                              kernelName,
                              props);
  }
  //====================================

  //---[ Diffusion Integrator ]---------
  OccaDiffusionIntegrator::OccaDiffusionIntegrator(const OccaCoefficient &coeff_) :
    OccaIntegrator(),
    coeff(coeff_) {
    coeff.SetName("COEFF");
  }

  OccaDiffusionIntegrator::~OccaDiffusionIntegrator() {}

  std::string OccaDiffusionIntegrator::GetName() {
    return "DiffusionIntegrator";
  }

  void OccaDiffusionIntegrator::SetupIntegrationRule() {
    const FiniteElement &trialFE = *(trialFESpace->GetFE(0));
    const FiniteElement &testFE  = *(testFESpace->GetFE(0));
    ir = &(GetDiffusionIntegrationRule(trialFE, testFE));
  }

  void OccaDiffusionIntegrator::Setup() {
    occa::properties kernelProps = props;

    coeff.Setup(*this, kernelProps);

    // Setup assemble and mult kernels
    assembleKernel = GetAssembleKernel(kernelProps);
    multKernel     = GetMultAddKernel(kernelProps);
  }

  void OccaDiffusionIntegrator::Assemble() {
    const FiniteElement &fe = *(trialFESpace->GetFE(0));

    const int dims = fe.GetDim();
    const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6

    const int elements = trialFESpace->GetNE();
    const int quadraturePoints = ir->GetNPoints();

    OccaGeometry geom = GetGeometry(OccaGeometry::Jacobian);

    assembledOperator.SetSize(device,
                              symmDims * quadraturePoints * elements);

    assembleKernel((int) mesh->GetNE(),
                   maps.quadWeights,
                   geom.J,
                   coeff,
                   assembledOperator);
  }

  void OccaDiffusionIntegrator::MultAdd(OccaVector &x, OccaVector &y) {
    multKernel((int) mesh->GetNE(),
               maps.dofToQuad,
               maps.dofToQuadD,
               maps.quadToDof,
               maps.quadToDofD,
               assembledOperator,
               x, y);
  }
  //====================================

  //---[ Mass Integrator ]--------------
  OccaMassIntegrator::OccaMassIntegrator(const OccaCoefficient &coeff_) :
    OccaIntegrator(),
    coeff(coeff_) {
    coeff.SetName("COEFF");
  }

  OccaMassIntegrator::~OccaMassIntegrator() {}

  std::string OccaMassIntegrator::GetName() {
    return "MassIntegrator";
  }

  void OccaMassIntegrator::SetupIntegrationRule() {
    const FiniteElement &trialFE = *(trialFESpace->GetFE(0));
    const FiniteElement &testFE  = *(testFESpace->GetFE(0));
    ir = &(GetMassIntegrationRule(trialFE, testFE));
  }

  void OccaMassIntegrator::Setup() {
    occa::properties kernelProps = props;

    coeff.Setup(*this, kernelProps);

    // Setup assemble and mult kernels
    assembleKernel = GetAssembleKernel(kernelProps);
    multKernel     = GetMultAddKernel(kernelProps);
  }

  void OccaMassIntegrator::Assemble() {
    if (assembledOperator.Size()) {
      return;
    }

    const int elements = trialFESpace->GetNE();
    const int quadraturePoints = ir->GetNPoints();

    OccaGeometry geom = GetGeometry(OccaGeometry::Jacobian);

    assembledOperator.SetSize(device,
                              quadraturePoints * elements);

    assembleKernel((int) mesh->GetNE(),
                   maps.quadWeights,
                   geom.J,
                   coeff,
                   assembledOperator);
  }

  void OccaMassIntegrator::SetOperator(OccaVector &v) {
    assembledOperator = v;
  }

  void OccaMassIntegrator::MultAdd(OccaVector &x, OccaVector &y) {
    multKernel((int) mesh->GetNE(),
               maps.dofToQuad,
               maps.dofToQuadD,
               maps.quadToDof,
               maps.quadToDofD,
               assembledOperator,
               x, y);
  }
  //====================================

  //---[ Vector Mass Integrator ]--------------
  OccaVectorMassIntegrator::OccaVectorMassIntegrator(const OccaCoefficient &coeff_) :
    OccaIntegrator(),
    coeff(coeff_) {
    coeff.SetName("COEFF");
  }

  OccaVectorMassIntegrator::~OccaVectorMassIntegrator() {}

  std::string OccaVectorMassIntegrator::GetName() {
    return "VectorMassIntegrator";
  }

  void OccaVectorMassIntegrator::SetupIntegrationRule() {
    const FiniteElement &trialFE = *(trialFESpace->GetFE(0));
    const FiniteElement &testFE  = *(testFESpace->GetFE(0));
    ir = &(GetMassIntegrationRule(trialFE, testFE));
  }

  void OccaVectorMassIntegrator::Setup() {
    occa::properties kernelProps = props;

    coeff.Setup(*this, kernelProps);

    // Setup assemble and mult kernels
    assembleKernel = GetAssembleKernel(kernelProps);
    multKernel     = GetMultAddKernel(kernelProps);
  }

  void OccaVectorMassIntegrator::Assemble() {
    const int elements = trialFESpace->GetNE();
    const int quadraturePoints = ir->GetNPoints();

    OccaGeometry geom = GetGeometry(OccaGeometry::Jacobian);

    assembledOperator.SetSize(device,
                              quadraturePoints * elements);

    assembleKernel((int) mesh->GetNE(),
                   maps.quadWeights,
                   geom.J,
                   coeff,
                   assembledOperator);
  }

  void OccaVectorMassIntegrator::MultAdd(OccaVector &x, OccaVector &y) {
    multKernel((int) mesh->GetNE(),
               maps.dofToQuad,
               maps.dofToQuadD,
               maps.quadToDof,
               maps.quadToDofD,
               assembledOperator,
               x, y);
  }
  //====================================
}

#endif
