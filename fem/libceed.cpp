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

#include "libceed.hpp"
#include "ceed.h"
#include "../general/device.hpp"

namespace mfem
{

//***********
// LibCEED
//***********

void initCeedCoeff(Coefficient* Q, CeedData* ptr){
    if (ConstantCoefficient* coeff = dynamic_cast<ConstantCoefficient*>(Q))
    {
      CeedConstCoeff* ceedCoeff = new CeedConstCoeff{coeff->constant};
      ptr->coeff_type = Const;
      ptr->coeff = (void*)ceedCoeff;
    } else if (GridFunctionCoefficient* coeff = dynamic_cast<GridFunctionCoefficient*>(Q)) {
      CeedGridCoeff* ceedCoeff = new CeedGridCoeff;
      ceedCoeff->coeff = coeff->GetGridFunction();
      ptr->coeff_type = Grid;
      ptr->coeff = (void*)ceedCoeff;
    } else {
      MFEM_ABORT("This type of Coefficient is not supported.");
    }  
}

/// libCEED Q-function for building quadrature data for a diffusion operator
static int f_build_diff_const(void *ctx, CeedInt Q,
                              const CeedScalar *const *in, CeedScalar *const *out) {
  BuildContext *bc = (BuildContext*)ctx;
  // in[0] is Jacobians with shape [dim, nc=dim, Q]
  // in[1] is quadrature weights, size (Q)
  //
  // At every quadrature point, compute qw/det(J).adj(J).adj(J)^T and store
  // the symmetric part of the result.
  const CeedScalar coeff = bc->coeff;
  const CeedScalar *J = in[0], *qw = in[1];
  CeedScalar *qd = out[0];
  switch (bc->dim + 10 * bc->space_dim) {
  case 11:
    for (CeedInt i = 0; i < Q; i++) {
      qd[i] = coeff * qw[i] / J[i];
    }
    break;
  case 22:
    for (CeedInt i = 0; i < Q; i++) {
      // J: 0 2   qd: 0 1   adj(J):  J22 -J12
      //    1 3       1 2           -J21  J11
      const CeedScalar J11 = J[i + Q * 0];
      const CeedScalar J21 = J[i + Q * 1];
      const CeedScalar J12 = J[i + Q * 2];
      const CeedScalar J22 = J[i + Q * 3];
      const CeedScalar w = qw[i] / (J11 * J22 - J21 * J12);
      qd[i + Q * 0] =   coeff * w * (J12 * J12 + J22 * J22);
      qd[i + Q * 1] = - coeff * w * (J11 * J12 + J21 * J22);
      qd[i + Q * 2] =   coeff * w * (J11 * J11 + J21 * J21);
    }
    break;
  case 33:
    for (CeedInt i = 0; i < Q; i++) {
      // J: 0 3 6   qd: 0 1 2
      //    1 4 7       1 3 4
      //    2 5 8       2 4 5
      const CeedScalar J11 = J[i + Q * 0];
      const CeedScalar J21 = J[i + Q * 1];
      const CeedScalar J31 = J[i + Q * 2];
      const CeedScalar J12 = J[i + Q * 3];
      const CeedScalar J22 = J[i + Q * 4];
      const CeedScalar J32 = J[i + Q * 5];
      const CeedScalar J13 = J[i + Q * 6];
      const CeedScalar J23 = J[i + Q * 7];
      const CeedScalar J33 = J[i + Q * 8];
      const CeedScalar A11 = J22 * J33 - J23 * J32;
      const CeedScalar A12 = J13 * J32 - J12 * J33;
      const CeedScalar A13 = J12 * J23 - J13 * J22;
      const CeedScalar A21 = J23 * J31 - J21 * J33;
      const CeedScalar A22 = J11 * J33 - J13 * J31;
      const CeedScalar A23 = J13 * J21 - J11 * J23;
      const CeedScalar A31 = J21 * J32 - J22 * J31;
      const CeedScalar A32 = J12 * J31 - J11 * J32;
      const CeedScalar A33 = J11 * J22 - J12 * J21;
      const CeedScalar w = qw[i] / (J11 * A11 + J21 * A12 + J31 * A13);
      qd[i + Q * 0] = coeff * w * (A11 * A11 + A12 * A12 + A13 * A13);
      qd[i + Q * 1] = coeff * w * (A11 * A21 + A12 * A22 + A13 * A23);
      qd[i + Q * 2] = coeff * w * (A11 * A31 + A12 * A32 + A13 * A33);
      qd[i + Q * 3] = coeff * w * (A21 * A21 + A22 * A22 + A23 * A23);
      qd[i + Q * 4] = coeff * w * (A21 * A31 + A22 * A32 + A23 * A33);
      qd[i + Q * 5] = coeff * w * (A31 * A31 + A32 * A32 + A33 * A33);
    }
    break;
  default:
    return CeedError(NULL, 1, "dim=%d, space_dim=%d is not supported",
                     bc->dim, bc->space_dim);
  }
  return 0;
}

static int f_build_diff_grid(void *ctx, CeedInt Q,
                             const CeedScalar *const *in, CeedScalar *const *out) {
  BuildContext *bc = (BuildContext *)ctx;
  // in[1] is Jacobians with shape [dim, nc=dim, Q]
  // in[2] is quadrature weights, size (Q)
  //
  // At every quadrature point, compute qw/det(J).adj(J).adj(J)^T and store
  // the symmetric part of the result.
  const CeedScalar *c = in[0], *J = in[1], *qw = in[2];
  CeedScalar *qd = out[0];
  switch (bc->dim + 10 * bc->space_dim) {
  case 11:
    for (CeedInt i = 0; i < Q; i++) {
      qd[i] = c[i] * qw[i] / J[i];
    }
    break;
  case 22:
    for (CeedInt i = 0; i < Q; i++) {
      // J: 0 2   qd: 0 1   adj(J):  J22 -J12
      //    1 3       1 2           -J21  J11
      const CeedScalar coeff = c[i];
      const CeedScalar J11 = J[i + Q * 0];
      const CeedScalar J21 = J[i + Q * 1];
      const CeedScalar J12 = J[i + Q * 2];
      const CeedScalar J22 = J[i + Q * 3];
      const CeedScalar w = qw[i] / (J11 * J22 - J21 * J12);
      qd[i + Q * 0] =   coeff * w * (J12 * J12 + J22 * J22);
      qd[i + Q * 1] = - coeff * w * (J11 * J12 + J21 * J22);
      qd[i + Q * 2] =   coeff * w * (J11 * J11 + J21 * J21);
    }
    break;
  case 33:
    for (CeedInt i = 0; i < Q; i++) {
      // J: 0 3 6   qd: 0 1 2
      //    1 4 7       1 3 4
      //    2 5 8       2 4 5
      const CeedScalar coeff = c[i];
      const CeedScalar J11 = J[i + Q * 0];
      const CeedScalar J21 = J[i + Q * 1];
      const CeedScalar J31 = J[i + Q * 2];
      const CeedScalar J12 = J[i + Q * 3];
      const CeedScalar J22 = J[i + Q * 4];
      const CeedScalar J32 = J[i + Q * 5];
      const CeedScalar J13 = J[i + Q * 6];
      const CeedScalar J23 = J[i + Q * 7];
      const CeedScalar J33 = J[i + Q * 8];
      const CeedScalar A11 = J22 * J33 - J23 * J32;
      const CeedScalar A12 = J13 * J32 - J12 * J33;
      const CeedScalar A13 = J12 * J23 - J13 * J22;
      const CeedScalar A21 = J23 * J31 - J21 * J33;
      const CeedScalar A22 = J11 * J33 - J13 * J31;
      const CeedScalar A23 = J13 * J21 - J11 * J23;
      const CeedScalar A31 = J21 * J32 - J22 * J31;
      const CeedScalar A32 = J12 * J31 - J11 * J32;
      const CeedScalar A33 = J11 * J22 - J12 * J21;
      const CeedScalar w = qw[i] / (J11 * A11 + J21 * A12 + J31 * A13);
      qd[i + Q * 0] = coeff * w * (A11 * A11 + A12 * A12 + A13 * A13);
      qd[i + Q * 1] = coeff * w * (A11 * A21 + A12 * A22 + A13 * A23);
      qd[i + Q * 2] = coeff * w * (A11 * A31 + A12 * A32 + A13 * A33);
      qd[i + Q * 3] = coeff * w * (A21 * A21 + A22 * A22 + A23 * A23);
      qd[i + Q * 4] = coeff * w * (A21 * A31 + A22 * A32 + A23 * A33);
      qd[i + Q * 5] = coeff * w * (A31 * A31 + A32 * A32 + A33 * A33);
    }
    break;
  default:
    return CeedError(NULL, 1, "dim=%d, space_dim=%d is not supported",
                     bc->dim, bc->space_dim);
  }
  return 0;
}

/// libCEED Q-function for applying a diff operator
static int f_apply_diff(void *ctx, CeedInt Q,
                        const CeedScalar *const *in, CeedScalar *const *out) {
  BuildContext *bc = (BuildContext *)ctx;
  // in[0], out[0] have shape [dim, nc=1, Q]
  const CeedScalar *ug = in[0], *qd = in[1];
  CeedScalar *vg = out[0];
  switch (bc->dim) {
  case 1:
    for (CeedInt i = 0; i < Q; i++) {
      vg[i] = ug[i] * qd[i];
    }
    break;
  case 2:
    for (CeedInt i = 0; i < Q; i++) {
      const CeedScalar ug0 = ug[i + Q * 0];
      const CeedScalar ug1 = ug[i + Q * 1];
      vg[i + Q * 0] = qd[i + Q * 0] * ug0 + qd[i + Q * 1] * ug1;
      vg[i + Q * 1] = qd[i + Q * 1] * ug0 + qd[i + Q * 2] * ug1;
    }
    break;
  case 3:
    for (CeedInt i = 0; i < Q; i++) {
      const CeedScalar ug0 = ug[i + Q * 0];
      const CeedScalar ug1 = ug[i + Q * 1];
      const CeedScalar ug2 = ug[i + Q * 2];
      vg[i + Q * 0] = qd[i + Q * 0] * ug0 + qd[i + Q * 1] * ug1 + qd[i + Q * 2] * ug2;
      vg[i + Q * 1] = qd[i + Q * 1] * ug0 + qd[i + Q * 3] * ug1 + qd[i + Q * 4] * ug2;
      vg[i + Q * 2] = qd[i + Q * 2] * ug0 + qd[i + Q * 4] * ug1 + qd[i + Q * 5] * ug2;
    }
    break;
  default:
    return CeedError(NULL, 1, "topo_dim=%d is not supported", bc->dim);
  }
  return 0;
}

static void FESpace2Ceed(const mfem::FiniteElementSpace &fes,
                         const mfem::IntegrationRule &ir,
                         Ceed ceed, CeedBasis *basis,
                         CeedElemRestriction *restr) {
   mfem::Mesh *mesh = fes.GetMesh();
   const mfem::FiniteElement *fe = fes.GetFE(0);
   const int order = fes.GetOrder(0);
   mfem::Array<int> dof_map;
   switch (mesh->Dimension()) {
      case 1: {
         const mfem::H1_SegmentElement *h1_fe =
         dynamic_cast<const mfem::H1_SegmentElement *>(fe);
         MFEM_VERIFY(h1_fe, "invalid FE");
         h1_fe->GetDofMap().Copy(dof_map);
         break;
      }
      case 2: {
         const mfem::H1_QuadrilateralElement *h1_fe =
         dynamic_cast<const mfem::H1_QuadrilateralElement *>(fe);
         MFEM_VERIFY(h1_fe, "invalid FE");
         h1_fe->GetDofMap().Copy(dof_map);
         break;
      }
      case 3: {
         const mfem::H1_HexahedronElement *h1_fe =
         dynamic_cast<const mfem::H1_HexahedronElement *>(fe);
         MFEM_VERIFY(h1_fe, "invalid FE");
         h1_fe->GetDofMap().Copy(dof_map);
         break;
      }
   }
   const mfem::FiniteElement *fe1d =
   fes.FEColl()->FiniteElementForGeometry(mfem::Geometry::SEGMENT);
   mfem::DenseMatrix shape1d(fe1d->GetDof(), ir.GetNPoints());
   mfem::DenseMatrix grad1d(fe1d->GetDof(), ir.GetNPoints());
   mfem::Vector qref1d(ir.GetNPoints()), qweight1d(ir.GetNPoints());
   mfem::Vector shape_i(shape1d.Height());
   mfem::DenseMatrix grad_i(grad1d.Height(), 1);
   const mfem::H1_SegmentElement *h1_fe1d =
   dynamic_cast<const mfem::H1_SegmentElement *>(fe1d);
   MFEM_VERIFY(h1_fe1d, "invalid FE");
   const mfem::Array<int> &dof_map_1d = h1_fe1d->GetDofMap();
   for (int i = 0; i < ir.GetNPoints(); i++) {
      const mfem::IntegrationPoint &ip = ir.IntPoint(i);
      qref1d(i) = ip.x;
      qweight1d(i) = ip.weight;
      fe1d->CalcShape(ip, shape_i);
      fe1d->CalcDShape(ip, grad_i);
      for (int j = 0; j < shape1d.Height(); j++) {
         shape1d(j, i) = shape_i(dof_map_1d[j]);
         grad1d(j, i) = grad_i(dof_map_1d[j], 0);
      }
   }
   CeedBasisCreateTensorH1(ceed, mesh->Dimension(), fes.GetVDim(), order + 1,
      ir.GetNPoints(), shape1d.GetData(),
      grad1d.GetData(), qref1d.GetData(),
      qweight1d.GetData(), basis);

   const mfem::Table &el_dof = fes.GetElementToDofTable();
   mfem::Array<int> tp_el_dof(el_dof.Size_of_connections());
   for (int i = 0; i < mesh->GetNE(); i++) {
      const int el_offset = fe->GetDof() * i;
      for (int j = 0; j < fe->GetDof(); j++) {
         tp_el_dof[j + el_offset] = el_dof.GetJ()[dof_map[j] + el_offset];
      }
   }
   CeedElemRestrictionCreate(ceed, mesh->GetNE(), fe->GetDof(),
      fes.GetNDofs(), fes.GetVDim(), CEED_MEM_HOST, CEED_COPY_VALUES,
      tp_el_dof.GetData(), restr);
}

namespace internal { extern Ceed ceed; }
void CeedPADiffusionAssemble(const FiniteElementSpace &fes, CeedData& ceedData)
{
   Ceed ceed(internal::ceed);
   const bool dev_enabled = Device::IsEnabled();
   mfem::Mesh *mesh = fes.GetMesh();
   const int order = fes.GetOrder(0);
   const int ir_order = 2 * (order + 2) - 1; // <-----
   const mfem::IntegrationRule &ir =
   mfem::IntRules.Get(mfem::Geometry::SEGMENT, ir_order);
   CeedInt nqpts, nelem = mesh->GetNE(), dim = mesh->SpaceDimension();
   if (dev_enabled) { Device::Disable(); }   
   mesh->EnsureNodes();
   FESpace2Ceed(fes, ir, ceed, &ceedData.basis, &ceedData.restr);

   const mfem::FiniteElementSpace *mesh_fes = mesh->GetNodalFESpace();
   MFEM_VERIFY(mesh_fes, "the Mesh has no nodal FE space");
   FESpace2Ceed(*mesh_fes, ir, ceed, &ceedData.mesh_basis, &ceedData.mesh_restr);
   if (dev_enabled) { Device::Enable(); }
   CeedBasisGetNumQuadraturePoints(ceedData.basis, &nqpts);

   CeedElemRestrictionCreateIdentity(ceed, nelem, nqpts * dim * (dim + 1) / 2,
      nqpts * nelem * dim * (dim + 1) / 2, 1, &ceedData.restr_i);
   CeedElemRestrictionCreateIdentity(ceed, nelem, nqpts,
      nqpts * nelem, 1, &ceedData.mesh_restr_i);

   CeedVectorCreate(ceed, mesh->GetNodes()->Size(), &ceedData.node_coords);
   CeedVectorSetArray(ceedData.node_coords, CEED_MEM_HOST, CEED_USE_POINTER,
      mesh->GetNodes()->GetData());

   CeedVectorCreate(ceed, nelem * nqpts * dim * (dim + 1) / 2, &ceedData.rho);

   // Context data to be passed to the 'f_build_diff' Q-function.
   ceedData.build_ctx.dim = mesh->Dimension();
   ceedData.build_ctx.space_dim = mesh->SpaceDimension();

   // Create the Q-function that builds the diff operator (i.e. computes its
   // quadrature data) and set its context data.
   switch(ceedData.coeff_type){
      case Const:
      CeedQFunctionCreateInterior(ceed, 1, f_build_diff_const,
         MFEM_SOURCE_DIR"/fem/libceed.okl:f_build_diff_const", &ceedData.build_qfunc);
      ceedData.build_ctx.coeff = ((CeedConstCoeff*)ceedData.coeff)->val;
      break;
      case Grid:
      CeedQFunctionCreateInterior(ceed, 1, f_build_diff_grid,
         MFEM_SOURCE_DIR"/fem/libceed.okl:f_build_diff_grid", &ceedData.build_qfunc);
      CeedQFunctionAddInput(ceedData.build_qfunc, "coeff", 1, CEED_EVAL_INTERP);
      break;
      default:
      MFEM_ABORT("This coeff_type is not handled");
   }
   CeedQFunctionAddInput(ceedData.build_qfunc, "dx", dim, CEED_EVAL_GRAD);
   CeedQFunctionAddInput(ceedData.build_qfunc, "weights", 1, CEED_EVAL_WEIGHT);
   CeedQFunctionAddOutput(ceedData.build_qfunc, "rho", dim * (dim + 1) / 2, CEED_EVAL_NONE);
   CeedQFunctionSetContext(ceedData.build_qfunc, &ceedData.build_ctx, sizeof(ceedData.build_ctx));

   // Create the operator that builds the quadrature data for the diff operator.
   CeedOperatorCreate(ceed, ceedData.build_qfunc, NULL, NULL, &ceedData.build_oper);
   CeedTransposeMode lmode = CEED_NOTRANSPOSE;
   if (mesh_fes->GetOrdering()==Ordering::byVDIM)
   {
      lmode = CEED_TRANSPOSE;
   }
   if (ceedData.coeff_type==Grid)
   {
      CeedGridCoeff* ceedCoeff = (CeedGridCoeff*)ceedData.coeff;
      if (dev_enabled) { Device::Disable(); }
      FESpace2Ceed(*ceedCoeff->coeff->FESpace(), ir, ceed, &ceedCoeff->basis, &ceedCoeff->restr);
      if (dev_enabled) { Device::Enable(); }
      CeedVectorCreate(ceed, ceedCoeff->coeff->FESpace()->GetNDofs(), &ceedCoeff->coeffVector);
      CeedVectorSetArray(ceedCoeff->coeffVector, CEED_MEM_HOST, CEED_USE_POINTER,
         ceedCoeff->coeff->GetData());
      CeedOperatorSetField(ceedData.build_oper, "coeff", ceedCoeff->restr, lmode,
         ceedCoeff->basis, ceedCoeff->coeffVector);
   }
   CeedOperatorSetField(ceedData.build_oper, "dx", ceedData.mesh_restr, lmode,
      ceedData.mesh_basis, CEED_VECTOR_ACTIVE);
   CeedOperatorSetField(ceedData.build_oper, "weights", ceedData.mesh_restr_i, CEED_NOTRANSPOSE,
      ceedData.mesh_basis, CEED_VECTOR_NONE);
   CeedOperatorSetField(ceedData.build_oper, "rho", ceedData.restr_i, CEED_NOTRANSPOSE,
      CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

   // Compute the quadrature data for the diff operator.
   CeedOperatorApply(ceedData.build_oper, ceedData.node_coords, ceedData.rho,
      CEED_REQUEST_IMMEDIATE);

   // Create the Q-function that defines the action of the diff operator.
   CeedQFunctionCreateInterior(ceed, 1, f_apply_diff,
      MFEM_SOURCE_DIR"/fem/libceed.okl:f_apply_diff", &ceedData.apply_qfunc);
   CeedQFunctionAddInput(ceedData.apply_qfunc, "u", 1, CEED_EVAL_GRAD);
   CeedQFunctionAddInput(ceedData.apply_qfunc, "rho", dim * (dim + 1) / 2, CEED_EVAL_NONE);
   CeedQFunctionAddOutput(ceedData.apply_qfunc, "v", 1, CEED_EVAL_GRAD);
   CeedQFunctionSetContext(ceedData.apply_qfunc, &ceedData.build_ctx, sizeof(ceedData.build_ctx));

   // Create the diff operator.
   CeedOperatorCreate(ceed, ceedData.apply_qfunc, NULL, NULL, &ceedData.oper);
   CeedOperatorSetField(ceedData.oper, "u", ceedData.restr, CEED_NOTRANSPOSE,
      ceedData.basis, CEED_VECTOR_ACTIVE);
   CeedOperatorSetField(ceedData.oper, "rho", ceedData.restr_i, CEED_NOTRANSPOSE,
      CEED_BASIS_COLLOCATED, ceedData.rho);
   CeedOperatorSetField(ceedData.oper, "v", ceedData.restr, CEED_NOTRANSPOSE,
      ceedData.basis, CEED_VECTOR_ACTIVE);

   CeedVectorCreate(ceed, fes.GetNDofs(), &ceedData.u);
   CeedVectorCreate(ceed, fes.GetNDofs(), &ceedData.v);
   
}

/// libCEED Q-function for building quadrature data for a mass operator
static int f_build_mass_const(void *ctx, CeedInt Q,
                        const CeedScalar *const *in, CeedScalar *const *out) {
  // in[0] is Jacobians with shape [dim, nc=dim, Q]
  // in[1] is quadrature weights, size (Q)
  BuildContext *bc = (BuildContext *)ctx;
  const CeedScalar coeff = bc->coeff;
  const CeedScalar *J = in[0], *qw = in[1];
  CeedScalar *rho = out[0];
  switch (bc->dim + 10*bc->space_dim) {
  case 11:
    for (CeedInt i=0; i<Q; i++) {
      rho[i] = coeff * J[i] * qw[i];
    }
    break;
  case 22:
    for (CeedInt i=0; i<Q; i++) {
      // 0 2
      // 1 3
      rho[i] = coeff * (J[i+Q*0]*J[i+Q*3] - J[i+Q*1]*J[i+Q*2]) * qw[i];
    }
    break;
  case 33:
    for (CeedInt i=0; i<Q; i++) {
      // 0 3 6
      // 1 4 7
      // 2 5 8
      rho[i] = (J[i+Q*0]*(J[i+Q*4]*J[i+Q*8] - J[i+Q*5]*J[i+Q*7]) -
                J[i+Q*1]*(J[i+Q*3]*J[i+Q*8] - J[i+Q*5]*J[i+Q*6]) +
                J[i+Q*2]*(J[i+Q*3]*J[i+Q*7] - J[i+Q*4]*J[i+Q*6])) * coeff * qw[i];
    }
    break;
  default:
    return CeedError(NULL, 1, "dim=%d, space_dim=%d is not supported",
                     bc->dim, bc->space_dim);
  }
  return 0;
}

static int f_build_mass_grid(void *ctx, CeedInt Q,
                        const CeedScalar *const *in, CeedScalar *const *out) {
  // in[0] is Jacobians with shape [dim, nc=dim, Q]
  // in[1] is quadrature weights, size (Q)
  BuildContext *bc = (BuildContext *)ctx;
  const CeedScalar *c = in[0], *J = in[1], *qw = in[2];
  CeedScalar *rho = out[0];
  switch (bc->dim + 10*bc->space_dim) {
  case 11:
    for (CeedInt i=0; i<Q; i++) {
      rho[i] = c[i] * J[i] * qw[i];
    }
    break;
  case 22:
    for (CeedInt i=0; i<Q; i++) {
      // 0 2
      // 1 3
      rho[i] = c[i] * (J[i+Q*0]*J[i+Q*3] - J[i+Q*1]*J[i+Q*2]) * qw[i];
    }
    break;
  case 33:
    for (CeedInt i=0; i<Q; i++) {
      // 0 3 6
      // 1 4 7
      // 2 5 8
      rho[i] = (J[i+Q*0]*(J[i+Q*4]*J[i+Q*8] - J[i+Q*5]*J[i+Q*7]) -
                J[i+Q*1]*(J[i+Q*3]*J[i+Q*8] - J[i+Q*5]*J[i+Q*6]) +
                J[i+Q*2]*(J[i+Q*3]*J[i+Q*7] - J[i+Q*4]*J[i+Q*6])) * c[i] * qw[i];
    }
    break;
  default:
    return CeedError(NULL, 1, "dim=%d, space_dim=%d is not supported",
                     bc->dim, bc->space_dim);
  }
  return 0;
}

/// libCEED Q-function for applying a mass operator
static int f_apply_mass(void *ctx, CeedInt Q,
                        const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *u = in[0], *w = in[1];
  CeedScalar *v = out[0];
  for (CeedInt i=0; i<Q; i++) {
    v[i] = w[i] * u[i];
  }
  return 0;
}

void CeedPAMassAssemble(const FiniteElementSpace &fes, CeedData& ceedData)
{
  Ceed ceed(internal::ceed);
  mfem::Mesh *mesh = fes.GetMesh();
  const int order = fes.GetOrder(0);
  const int ir_order = 2*(order + 2) - 1; // <-----
  const mfem::IntegrationRule &ir =
    mfem::IntRules.Get(mfem::Geometry::SEGMENT, ir_order);
  CeedInt nqpts, nelem = mesh->GetNE();

  FESpace2Ceed(fes, ir, ceed, &ceedData.basis, &ceedData.restr);
  mesh->EnsureNodes();
  const mfem::FiniteElementSpace *mesh_fes = mesh->GetNodalFESpace();
  MFEM_VERIFY(mesh_fes, "the Mesh has no nodal FE space");
  FESpace2Ceed(*mesh_fes, ir, ceed, &ceedData.mesh_basis, &ceedData.mesh_restr);
  CeedBasisGetNumQuadraturePoints(ceedData.basis, &nqpts);

  CeedElemRestrictionCreateIdentity(ceed, nelem, nqpts,
                                    nqpts*nelem, 1, &ceedData.restr_i);
  CeedElemRestrictionCreateIdentity(ceed, nelem, nqpts,
                                    nqpts*nelem, 1, &ceedData.mesh_restr_i);

  CeedVectorCreate(ceed, mesh->GetNodes()->Size(), &ceedData.node_coords);
  CeedVectorSetArray(ceedData.node_coords, CEED_MEM_HOST, CEED_USE_POINTER,
                     mesh->GetNodes()->GetData());

  CeedVectorCreate(ceed, nelem*nqpts, &ceedData.rho);

  // Context data to be passed to the 'f_build_mass' Q-function.
  ceedData.build_ctx.dim = mesh->Dimension();
  ceedData.build_ctx.space_dim = mesh->SpaceDimension();

  // Create the Q-function that builds the mass operator (i.e. computes its
  // quadrature data) and set its context data.
  switch(ceedData.coeff_type){
    case Const:
      CeedQFunctionCreateInterior(ceed, 1, f_build_mass_const,
         MFEM_SOURCE_DIR"/fem/libceed.okl:f_build_mass_const", &ceedData.build_qfunc);
      ceedData.build_ctx.coeff = ((CeedConstCoeff*)ceedData.coeff)->val;
      break;
    case Grid:
      CeedQFunctionCreateInterior(ceed, 1, f_build_mass_grid,
         MFEM_SOURCE_DIR"/fem/libceed.okl:f_build_mass_grid", &ceedData.build_qfunc);
      CeedQFunctionAddInput(ceedData.build_qfunc, "coeff", 1, CEED_EVAL_INTERP);
      break;
    default:
      MFEM_ABORT("This coeff_type is not handled");
  }
  CeedQFunctionAddInput(ceedData.build_qfunc, "dx", mesh->SpaceDimension(),
                        CEED_EVAL_GRAD);
  CeedQFunctionAddInput(ceedData.build_qfunc, "weights", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(ceedData.build_qfunc, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionSetContext(ceedData.build_qfunc, &ceedData.build_ctx, sizeof(ceedData.build_ctx));

  // Create the operator that builds the quadrature data for the mass operator.
  CeedOperatorCreate(ceed, ceedData.build_qfunc, NULL, NULL, &ceedData.build_oper);
  CeedTransposeMode lmode = CEED_NOTRANSPOSE;
  if (mesh_fes->GetOrdering()==Ordering::byVDIM)
  {
    lmode = CEED_TRANSPOSE;
  }
  if (ceedData.coeff_type==Grid)
  {
    CeedGridCoeff* ceedCoeff = (CeedGridCoeff*)ceedData.coeff;
    FESpace2Ceed(*ceedCoeff->coeff->FESpace(), ir, ceed, &ceedCoeff->basis, &ceedCoeff->restr);
    CeedVectorCreate(ceed, ceedCoeff->coeff->FESpace()->GetNDofs(), &ceedCoeff->coeffVector);
    CeedVectorSetArray(ceedCoeff->coeffVector, CEED_MEM_HOST, CEED_USE_POINTER,
                       ceedCoeff->coeff->GetData());
    CeedOperatorSetField(ceedData.build_oper, "coeff", ceedCoeff->restr, lmode,
                         ceedCoeff->basis, ceedCoeff->coeffVector);
  }
  CeedOperatorSetField(ceedData.build_oper, "dx", ceedData.mesh_restr, lmode,
                       ceedData.mesh_basis, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(ceedData.build_oper, "weights", ceedData.mesh_restr_i, CEED_NOTRANSPOSE,
                       ceedData.mesh_basis, CEED_VECTOR_NONE);
  CeedOperatorSetField(ceedData.build_oper, "rho", ceedData.restr_i, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Compute the quadrature data for the mass operator.
  CeedOperatorApply(ceedData.build_oper, ceedData.node_coords, ceedData.rho,
                    CEED_REQUEST_IMMEDIATE);

  // Create the Q-function that defines the action of the mass operator.
  CeedQFunctionCreateInterior(ceed, 1, f_apply_mass,
   MFEM_SOURCE_DIR"/fem/libceed.okl:f_apply_mass", &ceedData.apply_qfunc);
  CeedQFunctionAddInput(ceedData.apply_qfunc, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(ceedData.apply_qfunc, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(ceedData.apply_qfunc, "v", 1, CEED_EVAL_INTERP);

  // Create the mass operator.
  CeedOperatorCreate(ceed, ceedData.apply_qfunc, NULL, NULL, &ceedData.oper);
  CeedOperatorSetField(ceedData.oper, "u", ceedData.restr, CEED_NOTRANSPOSE,
                       ceedData.basis, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(ceedData.oper, "rho", ceedData.restr_i, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, ceedData.rho);
  CeedOperatorSetField(ceedData.oper, "v", ceedData.restr, CEED_NOTRANSPOSE,
                       ceedData.basis, CEED_VECTOR_ACTIVE);

  CeedVectorCreate(ceed, fes.GetNDofs(), &ceedData.u);
  CeedVectorCreate(ceed, fes.GetNDofs(), &ceedData.v);
}

//***********
// LibCEED
//***********

} // namespace mfem
