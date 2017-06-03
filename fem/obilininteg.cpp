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
    nodes.ReorderByVDim();
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
      nodes.ReorderByNodes();
    } else {
      nodes.ReorderByVDim();
    }

    if (flags & Jacobian) {
      geom.J.allocate(device,
                      dims*dims, numQuad, elements);
    } else {
      geom.J.allocate(device, 1);
    }
    if (flags & JacobianInv) {
      geom.invJ.allocate(device,
                         dims*dims, numQuad, elements);
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

  OccaDofQuadMaps& OccaDofQuadMaps::GetTensorMaps(occa::device device,
                                                  const FiniteElement &fe,
                                                  const TensorBasisElement &tfe,
                                                  const IntegrationRule &ir,
                                                  const bool transpose) {

    return GetTensorMaps(device, fe, fe, tfe, tfe, ir, transpose);
  }

  OccaDofQuadMaps& OccaDofQuadMaps::GetTensorMaps(occa::device device,
                                                  const FiniteElement &fe,
                                                  const FiniteElement &fe2,
                                                  const TensorBasisElement &tfe,
                                                  const TensorBasisElement &tfe2,
                                                  const IntegrationRule &ir,
                                                  const bool transpose) {
    std::stringstream ss;
    ss << occa::hash(device)
       << "Tensor"
       << "O:"   << fe.GetOrder()
       << "O2:"  << fe2.GetOrder()
       << "BT:"  << tfe.GetBasisType()
       << "BT2:" << tfe2.GetBasisType()
       << "Q:"   << ir.GetNPoints();
    std::string hash = ss.str();

    // If we've already made the dof-quad maps, reuse them
    OccaDofQuadMaps &maps = AllDofQuadMaps[hash];
    if (!maps.hash.size()) {
      // Create the dof-quad maps
      maps.hash = hash;

      OccaDofQuadMaps d2qMaps  = GetD2QTensorMaps(device, fe , tfe , ir);
      OccaDofQuadMaps d2qMaps2 = GetD2QTensorMaps(device, fe2, tfe2, ir, true);

      maps.dofToQuad   = d2qMaps.dofToQuad;
      maps.dofToQuadD  = d2qMaps.dofToQuadD;
      maps.quadToDof   = d2qMaps2.dofToQuad;
      maps.quadToDofD  = d2qMaps2.dofToQuadD;
      maps.quadWeights = d2qMaps2.quadWeights;
    }
    return maps;
  }

  OccaDofQuadMaps OccaDofQuadMaps::GetD2QTensorMaps(occa::device device,
                                                    const FiniteElement &fe,
                                                    const TensorBasisElement &tfe,
                                                    const IntegrationRule &ir,
                                                    const bool transpose) {

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

    return GetSimplexMaps(device, fe, fe, ir, transpose);
  }

  OccaDofQuadMaps& OccaDofQuadMaps::GetSimplexMaps(occa::device device,
                                                   const FiniteElement &fe,
                                                   const FiniteElement &fe2,
                                                   const IntegrationRule &ir,
                                                   const bool transpose) {
    std::stringstream ss;
    ss << occa::hash(device)
       << "Simplex"
       << "O:"  << fe.GetOrder()
       << "O2:" << fe.GetOrder()
       << "Q:"  << ir.GetNPoints();
    std::string hash = ss.str();

    // If we've already made the dof-quad maps, reuse them
    OccaDofQuadMaps &maps = AllDofQuadMaps[hash];
    if (!maps.hash.size()) {
      // Create the dof-quad maps
      maps.hash = hash;

      OccaDofQuadMaps d2qMaps  = GetD2QSimplexMaps(device, fe , ir);
      OccaDofQuadMaps d2qMaps2 = GetD2QSimplexMaps(device, fe2, ir, true);

      maps.dofToQuad   = d2qMaps.dofToQuad;
      maps.dofToQuadD  = d2qMaps.dofToQuadD;
      maps.quadToDof   = d2qMaps2.dofToQuad;
      maps.quadToDofD  = d2qMaps2.dofToQuadD;
      maps.quadWeights = d2qMaps2.quadWeights;
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

  void setTensorProperties(const FiniteElement &fe,
                           const IntegrationRule &ir,
                           occa::properties &props) {

    setTensorProperties(fe, fe, ir, props);
  }

  void setTensorProperties(const FiniteElement &fe,
                           const FiniteElement &fe2,
                           const IntegrationRule &ir,
                           occa::properties &props) {

    const IntegrationRule &ir1D = IntRules.Get(Geometry::SEGMENT, ir.GetOrder());

    const int numDofs  = fe.GetDof();
    const int numDofs2 = fe2.GetDof();
    const int numQuad  = ir.GetNPoints();

    const int dofs1D  = fe.GetOrder() + 1;
    const int dofs21D = fe2.GetOrder() + 1;
    const int quad1D  = ir1D.GetNPoints();
    int dofsND  = dofs1D;
    int dofs2ND = dofs21D;
    int quadND  = quad1D;

    props["defines/USING_TENSOR_OPS"] = 1;
    props["defines/NUM_DOFS"]  = numDofs;
    props["defines/NUM_DOFS2"] = numDofs2;
    props["defines/NUM_QUAD"]  = numQuad;

    for (int d = 1; d <= 3; ++d) {
      if (d > 1) {
        dofsND  *= dofs1D;
        dofs2ND *= dofs21D;
        quadND  *= quad1D;
      }
      props["defines"][stringWithDim("NUM_DOFS_" , d)] = dofsND;
      props["defines"][stringWithDim("NUM_DOFS2_", d)] = dofsND;
      props["defines"][stringWithDim("NUM_QUAD_" , d)] = quadND;
    }

    // 1D Defines
    const int m1InnerBatch = 32 * ((quad1D + 31) / 32);
    props["defines/A1_ELEMENT_BATCH"]       = closestMultipleWarpBatch(quad1D, 2048);
    props["defines/M1_OUTER_ELEMENT_BATCH"] = closestMultipleWarpBatch(m1InnerBatch, 2048);
    props["defines/M1_INNER_ELEMENT_BATCH"] = m1InnerBatch;

    // 2D Defines
    props["defines/A2_ELEMENT_BATCH"] = 1;
    props["defines/A2_QUAD_BATCH"]    = 1;
    props["defines/M2_ELEMENT_BATCH"] = 32;

    // 3D Defines
    const int a3QuadBatch = closestMultipleWarpBatch(quadND, 2048);
    props["defines/A3_ELEMENT_BATCH"] = closestMultipleWarpBatch(a3QuadBatch, 2048);
    props["defines/A3_QUAD_BATCH"]    = a3QuadBatch;
  }

  void setSimplexProperties(const FiniteElement &fe,
                            const IntegrationRule &ir,
                            occa::properties &props) {

    setSimplexProperties(fe, fe, ir, props);
  }

  void setSimplexProperties(const FiniteElement &fe,
                            const FiniteElement &fe2,
                            const IntegrationRule &ir,
                            occa::properties &props) {

    const int numDofs  = fe.GetDof();
    const int numDofs2 = fe2.GetDof();
    const int numQuad = ir.GetNPoints();
    const int maxDQ   = numDofs > numQuad ? numDofs : numQuad;

    props["defines/USING_TENSOR_OPS"] = 0;
    props["defines/NUM_DOFS"]  = numDofs;
    props["defines/NUM_DOFS2"] = numDofs2;
    props["defines/NUM_QUAD"]  = numQuad;

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
  OccaIntegrator::OccaIntegrator() {}
  OccaIntegrator::~OccaIntegrator() {}

  void OccaIntegrator::SetupMaps() {
    const FiniteElement &fe = *(fespace->GetFE(0));
    const TensorBasisElement *el = dynamic_cast<const TensorBasisElement*>(&fe);

    const FiniteElement &fe2 = *(fespace2->GetFE(0));
    const TensorBasisElement *el2 = dynamic_cast<const TensorBasisElement*>(&fe2);

    OccaDofQuadMaps maps2;

    if (el) {
      maps = OccaDofQuadMaps::GetTensorMaps(device,
                                            fe, fe2,
                                            *el, *el2,
                                            *ir);
      hasTensorBasis = true;
    } else {
      maps = OccaDofQuadMaps::GetSimplexMaps(device,
                                             fe , fe2,
                                             *ir);
      hasTensorBasis = false;
    }
  }

  void OccaIntegrator::SetupProperties(occa::properties &props) {
    const FiniteElement &fe  = *(fespace->GetFE(0));
    const FiniteElement &fe2 = *(fespace2->GetFE(0));
    if (hasTensorBasis) {
      setTensorProperties(fe, fe2, *ir, props);
    } else {
      setSimplexProperties(fe, fe2, *ir, props);
    }
  }

  occa::device OccaIntegrator::GetDevice() {
    return device;
  }

  FiniteElementSpace& OccaIntegrator::GetFESpace() {
    return *fespace;
  }

  FiniteElementSpace& OccaIntegrator::GetFESpace2() {
    return *fespace2;
  }

  const IntegrationRule& OccaIntegrator::GetIntegrationRule() {
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

    ofespace  = &(bform_.GetOccaFESpace());
    ofespace2 = &(bform_.GetOccaFESpace2());

    fespace  = &(bform_.GetFESpace());
    fespace2 = &(bform_.GetFESpace2());

    props = props_;
    itype = itype_;

    SetupIntegrationRule();
    SetupMaps();
    SetupProperties(props);

    Setup();
  }

  OccaGeometry OccaIntegrator::GetGeometry(const int flags) {
    return OccaGeometry::Get(device, *ofespace, *ir, flags);
  }

  occa::kernel OccaIntegrator::GetAssembleKernel(const occa::properties &props) {
    const FiniteElement &fe = *(fespace->GetFE(0));
    return GetKernel(stringWithDim("Assemble", fe.GetDim()),
                     props);
  }

  occa::kernel OccaIntegrator::GetMultKernel(const occa::properties &props) {
    const FiniteElement &fe = *(fespace->GetFE(0));
    return GetKernel(stringWithDim("Mult", fe.GetDim()),
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
    coeff(coeff_) {
    coeff.SetName("COEFF");
  }

  OccaDiffusionIntegrator::~OccaDiffusionIntegrator() {}

  std::string OccaDiffusionIntegrator::GetName() {
    return "DiffusionIntegrator";
  }

  void OccaDiffusionIntegrator::SetupIntegrationRule() {
    const FiniteElement &fe  = *(fespace->GetFE(0));
    const FiniteElement &fe2 = *(fespace2->GetFE(0));
    ir = &(GetDiffusionIntegrationRule(fe, fe2));
  }

  void OccaDiffusionIntegrator::Setup() {
    occa::properties kernelProps = props;

    const FiniteElement &fe = *(fespace->GetFE(0));

    const int dims = fe.GetDim();
    const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6

    const int elements = fespace->GetNE();
    const int quadraturePoints = ir->GetNPoints();

    assembledOperator.allocate(symmDims, quadraturePoints, elements);

    OccaGeometry geom = GetGeometry(OccaGeometry::Jacobian);
    jacobian = geom.J;

    coeff.Setup(*this, kernelProps);

    // Setup assemble and mult kernels
    assembleKernel = GetAssembleKernel(kernelProps);
    multKernel     = GetMultKernel(kernelProps);
  }

  void OccaDiffusionIntegrator::Assemble() {
    assembleKernel((int) mesh->GetNE(),
                   maps.quadWeights,
                   jacobian,
                   coeff,
                   assembledOperator);
  }

  void OccaDiffusionIntegrator::Mult(OccaVector &x, OccaVector &y) {
    multKernel((int) mesh->GetNE(),
               maps.dofToQuad,
               maps.dofToQuadD,
               maps.quadToDof,
               maps.quadToDofD,
               assembledOperator,
               x);
  }
  //====================================

  //---[ Mass Integrator ]--------------
  OccaMassIntegrator::OccaMassIntegrator(const OccaCoefficient &coeff_) :
    coeff(coeff_) {
    coeff.SetName("COEFF");
  }

  OccaMassIntegrator::~OccaMassIntegrator() {}

  std::string OccaMassIntegrator::GetName() {
    return "MassIntegrator";
  }

  void OccaMassIntegrator::SetupIntegrationRule() {
    const FiniteElement &fe  = *(fespace->GetFE(0));
    const FiniteElement &fe2 = *(fespace2->GetFE(0));
    ir = &(GetMassIntegrationRule(fe, fe2));
  }

  void OccaMassIntegrator::Setup() {
    occa::properties kernelProps = props;

    const int elements = fespace->GetNE();
    const int quadraturePoints = ir->GetNPoints();

    assembledOperator.allocate(quadraturePoints, elements);

    OccaGeometry geom = GetGeometry(OccaGeometry::Jacobian);
    jacobian = geom.J;

    coeff.Setup(*this, kernelProps);

    // Setup assemble and mult kernels
    assembleKernel = GetAssembleKernel(kernelProps);
    multKernel     = GetMultKernel(kernelProps);
  }

  void OccaMassIntegrator::Assemble() {
    assembleKernel((int) mesh->GetNE(),
                   maps.quadWeights,
                   jacobian,
                   coeff,
                   assembledOperator);
  }

  void OccaMassIntegrator::Mult(OccaVector &x, OccaVector &y) {
    multKernel((int) mesh->GetNE(),
               maps.dofToQuad,
               maps.dofToQuadD,
               maps.quadToDof,
               maps.quadToDofD,
               assembledOperator,
               x);
  }
  //====================================

  //---[ Vector Mass Integrator ]--------------
  OccaVectorMassIntegrator::OccaVectorMassIntegrator(const OccaCoefficient &coeff_) :
    coeff(coeff_) {
    coeff.SetName("COEFF");
  }

  OccaVectorMassIntegrator::~OccaVectorMassIntegrator() {}

  std::string OccaVectorMassIntegrator::GetName() {
    return "VectorMassIntegrator";
  }

  void OccaVectorMassIntegrator::SetupIntegrationRule() {
    const FiniteElement &fe  = *(fespace->GetFE(0));
    const FiniteElement &fe2 = *(fespace2->GetFE(0));
    ir = &(GetMassIntegrationRule(fe, fe2));
  }

  void OccaVectorMassIntegrator::Setup() {
    occa::properties kernelProps = props;

    const int elements = fespace->GetNE();
    const int quadraturePoints = ir->GetNPoints();

    assembledOperator.allocate(quadraturePoints, elements);

    OccaGeometry geom = GetGeometry(OccaGeometry::Jacobian);
    jacobian = geom.J;

    coeff.Setup(*this, kernelProps);

    // Setup assemble and mult kernels
    assembleKernel = GetAssembleKernel(kernelProps);
    multKernel     = GetMultKernel(kernelProps);
  }

  void OccaVectorMassIntegrator::Assemble() {
    assembleKernel((int) mesh->GetNE(),
                   maps.quadWeights,
                   jacobian,
                   coeff,
                   assembledOperator);
  }

  void OccaVectorMassIntegrator::Mult(OccaVector &x, OccaVector &y) {
    multKernel((int) mesh->GetNE(),
               maps.dofToQuad,
               maps.dofToQuadD,
               maps.quadToDof,
               maps.quadToDofD,
               assembledOperator,
               x);
  }
  //====================================

}

#endif
