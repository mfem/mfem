// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "ceed.hpp"

#include "../../general/device.hpp"
#include "../../fem/gridfunc.hpp"
#include "../../linalg/dtensor.hpp"

#include <sys/types.h>
#include <sys/stat.h>
#ifndef _WIN32
typedef struct stat struct_stat;
#else
#define stat(dir, buf) _stat(dir, buf)
#define S_ISDIR(mode) _S_IFDIR(mode)
typedef struct _stat struct_stat;
#endif

namespace mfem
{

namespace internal
{

#ifdef MFEM_USE_CEED
extern Ceed ceed;

std::string ceed_path;

extern CeedBasisMap ceed_basis_map;
extern CeedRestrMap ceed_restr_map;
#endif

}

void InitCeedCoeff(Coefficient *Q, Mesh &mesh,
                   const IntegrationRule &ir, CeedData *ptr)
{
#ifdef MFEM_USE_CEED
   if ( Q == nullptr )
   {
      CeedConstCoeff *ceedCoeff = new CeedConstCoeff{1.0};
      ptr->coeff_type = CeedCoeff::Const;
      ptr->coeff = static_cast<void*>(ceedCoeff);
   }
   else if (ConstantCoefficient *coeff = dynamic_cast<ConstantCoefficient*>(Q))
   {
      CeedConstCoeff *ceedCoeff = new CeedConstCoeff{coeff->constant};
      ptr->coeff_type = CeedCoeff::Const;
      ptr->coeff = static_cast<void*>(ceedCoeff);
   }
   else if (GridFunctionCoefficient* coeff =
               dynamic_cast<GridFunctionCoefficient*>(Q))
   {
      CeedGridCoeff *ceedCoeff = new CeedGridCoeff;
      ceedCoeff->coeff = coeff->GetGridFunction();
      InitCeedVector(*ceedCoeff->coeff, ceedCoeff->coeffVector);
      ptr->coeff_type = CeedCoeff::Grid;
      ptr->coeff = static_cast<void*>(ceedCoeff);
   }
   else if (QuadratureFunctionCoefficient *cQ =
               dynamic_cast<QuadratureFunctionCoefficient*>(Q))
   {
      CeedQuadCoeff *ceedCoeff = new CeedQuadCoeff;
      const int ne = mesh.GetNE();
      const int nq = ir.GetNPoints();
      const QuadratureFunction &qFun = cQ->GetQuadFunction();
      MFEM_VERIFY(qFun.Size() == nq * ne,
                  "Incompatible QuadratureFunction dimension \n");

      MFEM_VERIFY(&ir == &qFun.GetSpace()->GetElementIntRule(0),
                  "IntegrationRule used within integrator and in"
                  " QuadratureFunction appear to be different");
      qFun.Read();
      ceedCoeff->coeff.MakeRef(const_cast<QuadratureFunction &>(qFun),0);
      InitCeedVector(ceedCoeff->coeff, ceedCoeff->coeffVector);
      ptr->coeff_type = CeedCoeff::Quad;
      ptr->coeff = static_cast<void*>(ceedCoeff);
   }
   else
   {
      CeedQuadCoeff *ceedCoeff = new CeedQuadCoeff;
      const int ne = mesh.GetNE();
      const int nq = ir.GetNPoints();
      ceedCoeff->coeff.SetSize(nq * ne);
      auto C = Reshape(ceedCoeff->coeff.HostWrite(), nq, ne);
      for (int e = 0; e < ne; ++e)
      {
         ElementTransformation &T = *mesh.GetElementTransformation(e);
         for (int q = 0; q < nq; ++q)
         {
            C(q,e) = Q->Eval(T, ir.IntPoint(q));
         }
      }
      InitCeedVector(ceedCoeff->coeff, ceedCoeff->coeffVector);
      ptr->coeff_type = CeedCoeff::Quad;
      ptr->coeff = static_cast<void*>(ceedCoeff);
   }
#else
   mfem_error("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

void CeedPAAssemble(const CeedPAOperator& op,
                    CeedData& ceedData)
{
#ifdef MFEM_USE_CEED
   const FiniteElementSpace &fes = op.fes;
   const mfem::IntegrationRule &irm = op.ir;
   Ceed ceed(internal::ceed);
   mfem::Mesh *mesh = fes.GetMesh();
   CeedInt nqpts, nelem = mesh->GetNE();
   CeedInt dim = mesh->SpaceDimension(), vdim = fes.GetVDim();

   mesh->EnsureNodes();
   InitCeedBasisAndRestriction(fes, irm, ceed, &ceedData.basis, &ceedData.restr);

   const mfem::FiniteElementSpace *mesh_fes = mesh->GetNodalFESpace();
   MFEM_VERIFY(mesh_fes, "the Mesh has no nodal FE space");
   InitCeedBasisAndRestriction(*mesh_fes, irm, ceed, &ceedData.mesh_basis,
                               &ceedData.mesh_restr);

   CeedBasisGetNumQuadraturePoints(ceedData.basis, &nqpts);

   const int qdatasize = op.qdatasize;
   InitCeedStridedRestriction(*mesh_fes, nelem, nqpts, qdatasize,
                              CEED_STRIDES_BACKEND,
                              &ceedData.restr_i);

   InitCeedVector(*mesh->GetNodes(), ceedData.node_coords);

   CeedVectorCreate(ceed, nelem * nqpts * qdatasize, &ceedData.rho);

   // Context data to be passed to the 'f_build_diff' Q-function.
   ceedData.build_ctx_data.dim = mesh->Dimension();
   ceedData.build_ctx_data.space_dim = mesh->SpaceDimension();
   ceedData.build_ctx_data.vdim = fes.GetVDim();

   std::string qf_file = GetCeedPath() + op.header;
   std::string qf;

   // Create the Q-function that builds the operator (i.e. computes its
   // quadrature data) and set its context data.
   switch (ceedData.coeff_type)
   {
      case CeedCoeff::Const:
         qf = qf_file + op.const_func;
         CeedQFunctionCreateInterior(ceed, 1, op.const_qf,
                                     qf.c_str(),
                                     &ceedData.build_qfunc);
         ceedData.build_ctx_data.coeff = ((CeedConstCoeff*)ceedData.coeff)->val;
         break;
      case CeedCoeff::Grid:
         qf = qf_file + op.quad_func;
         CeedQFunctionCreateInterior(ceed, 1, op.quad_qf,
                                     qf.c_str(),
                                     &ceedData.build_qfunc);
         CeedQFunctionAddInput(ceedData.build_qfunc, "coeff", 1, CEED_EVAL_INTERP);
         break;
      case CeedCoeff::Quad:
         qf = qf_file + op.quad_func;
         CeedQFunctionCreateInterior(ceed, 1, op.quad_qf,
                                     qf.c_str(),
                                     &ceedData.build_qfunc);
         CeedQFunctionAddInput(ceedData.build_qfunc, "coeff", 1, CEED_EVAL_NONE);
         break;
   }
   CeedQFunctionAddInput(ceedData.build_qfunc, "dx", dim * dim, CEED_EVAL_GRAD);
   CeedQFunctionAddInput(ceedData.build_qfunc, "weights", 1, CEED_EVAL_WEIGHT);
   CeedQFunctionAddOutput(ceedData.build_qfunc, "qdata", qdatasize,
                          CEED_EVAL_NONE);

   CeedQFunctionContextCreate(ceed, &ceedData.build_ctx);
   CeedQFunctionContextSetData(ceedData.build_ctx, CEED_MEM_HOST, CEED_USE_POINTER,
                               sizeof(ceedData.build_ctx_data),
                               &ceedData.build_ctx_data);
   CeedQFunctionSetContext(ceedData.build_qfunc, ceedData.build_ctx);

   // Create the operator that builds the quadrature data for the operator.
   CeedOperatorCreate(ceed, ceedData.build_qfunc, NULL, NULL,
                      &ceedData.build_oper);
   switch (ceedData.coeff_type)
   {
      case CeedCoeff::Const:
         break;
      case CeedCoeff::Grid:
      {
         CeedGridCoeff* gridCoeff = (CeedGridCoeff*)ceedData.coeff;
         InitCeedBasisAndRestriction(*gridCoeff->coeff->FESpace(), irm, ceed,
                                     &gridCoeff->basis,
                                     &gridCoeff->restr);
         CeedOperatorSetField(ceedData.build_oper, "coeff", gridCoeff->restr,
                              gridCoeff->basis, gridCoeff->coeffVector);
      }
      break;
      case CeedCoeff::Quad:
      {
         CeedQuadCoeff* quadCoeff = (CeedQuadCoeff*)ceedData.coeff;
         const int ncomp = 1;
         CeedInt strides[3] = {1, nqpts, ncomp*nqpts};
         InitCeedStridedRestriction(*mesh_fes, nelem, nqpts, ncomp, strides,
                                    &quadCoeff->restr);
         CeedOperatorSetField(ceedData.build_oper, "coeff", quadCoeff->restr,
                              CEED_BASIS_COLLOCATED, quadCoeff->coeffVector);
      }
      break;
   }
   CeedOperatorSetField(ceedData.build_oper, "dx", ceedData.mesh_restr,
                        ceedData.mesh_basis, CEED_VECTOR_ACTIVE);
   CeedOperatorSetField(ceedData.build_oper, "weights", CEED_ELEMRESTRICTION_NONE,
                        ceedData.mesh_basis, CEED_VECTOR_NONE);
   CeedOperatorSetField(ceedData.build_oper, "qdata", ceedData.restr_i,
                        CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

   // Compute the quadrature data for the operator.
   CeedOperatorApply(ceedData.build_oper, ceedData.node_coords, ceedData.rho,
                     CEED_REQUEST_IMMEDIATE);

   // Create the Q-function that defines the action of the operator.
   qf = qf_file + op.apply_func;
   CeedQFunctionCreateInterior(ceed, 1, op.apply_qf,
                               qf.c_str(),
                               &ceedData.apply_qfunc);
   CeedInt dimU = vdim*(op.trial_op==CEED_EVAL_GRAD ? dim : 1);
   CeedInt dimV = vdim*(op.test_op==CEED_EVAL_GRAD ? dim : 1);
   CeedQFunctionAddInput(ceedData.apply_qfunc, "u", dimU, op.trial_op);
   CeedQFunctionAddInput(ceedData.apply_qfunc, "qdata", qdatasize,
                         CEED_EVAL_NONE);
   CeedQFunctionAddOutput(ceedData.apply_qfunc, "v", dimV, op.test_op);
   CeedQFunctionSetContext(ceedData.apply_qfunc, ceedData.build_ctx);

   // Create the operator.
   CeedOperatorCreate(ceed, ceedData.apply_qfunc, NULL, NULL, &ceedData.oper);
   CeedOperatorSetField(ceedData.oper, "u", ceedData.restr, ceedData.basis,
                        CEED_VECTOR_ACTIVE);
   CeedOperatorSetField(ceedData.oper, "qdata", ceedData.restr_i,
                        CEED_BASIS_COLLOCATED, ceedData.rho);
   CeedOperatorSetField(ceedData.oper, "v", ceedData.restr, ceedData.basis,
                        CEED_VECTOR_ACTIVE);

   CeedVectorCreate(ceed, vdim*fes.GetNDofs(), &ceedData.u);
   CeedVectorCreate(ceed, vdim*fes.GetNDofs(), &ceedData.v);
#else
   mfem_error("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

void CeedMFAssemble(const CeedMFOperator& op,
                    CeedData& ceedData)
{
#ifdef MFEM_USE_CEED
   const FiniteElementSpace &fes = op.fes;
   const mfem::IntegrationRule &irm = op.ir;
   Ceed ceed(internal::ceed);
   mfem::Mesh *mesh = fes.GetMesh();
   CeedInt nqpts, nelem = mesh->GetNE();
   CeedInt dim = mesh->SpaceDimension(), vdim = fes.GetVDim();

   mesh->EnsureNodes();
   InitCeedBasisAndRestriction(fes, irm, ceed, &ceedData.basis, &ceedData.restr);

   const mfem::FiniteElementSpace *mesh_fes = mesh->GetNodalFESpace();
   MFEM_VERIFY(mesh_fes, "the Mesh has no nodal FE space");
   InitCeedBasisAndRestriction(*mesh_fes, irm, ceed, &ceedData.mesh_basis,
                               &ceedData.mesh_restr);

   CeedBasisGetNumQuadraturePoints(ceedData.basis, &nqpts);

   InitCeedVector(*mesh->GetNodes(), ceedData.node_coords);

   // Context data to be passed to the Q-function.
   ceedData.build_ctx_data.dim = mesh->Dimension();
   ceedData.build_ctx_data.space_dim = mesh->SpaceDimension();
   ceedData.build_ctx_data.vdim = fes.GetVDim();

   std::string qf_file = GetCeedPath() + op.header;
   std::string qf;

   // Create the Q-function that builds the operator (i.e. computes its
   // quadrature data) and set its context data.
   CeedInt dimU = vdim*(op.trial_op==CEED_EVAL_GRAD ? dim : 1);
   CeedInt dimV = vdim*(op.test_op==CEED_EVAL_GRAD ? dim : 1);
   switch (ceedData.coeff_type)
   {
      case CeedCoeff::Const:
         qf = qf_file + op.const_func;
         CeedQFunctionCreateInterior(ceed, 1, op.const_qf,
                                     qf.c_str(),
                                     &ceedData.apply_qfunc);
         ceedData.build_ctx_data.coeff = ((CeedConstCoeff*)ceedData.coeff)->val;
         break;
      case CeedCoeff::Grid:
         qf = qf_file + op.quad_func;
         CeedQFunctionCreateInterior(ceed, 1, op.quad_qf,
                                     qf.c_str(),
                                     &ceedData.apply_qfunc);
         CeedQFunctionAddInput(ceedData.apply_qfunc, "coeff", 1, CEED_EVAL_INTERP);
         break;
      case CeedCoeff::Quad:
         qf = qf_file + op.quad_func;
         CeedQFunctionCreateInterior(ceed, 1, op.quad_qf,
                                     qf.c_str(),
                                     &ceedData.apply_qfunc);
         CeedQFunctionAddInput(ceedData.apply_qfunc, "coeff", 1, CEED_EVAL_NONE);
         break;
   }
   CeedQFunctionAddInput(ceedData.apply_qfunc, "u", dimU, op.trial_op);
   CeedQFunctionAddInput(ceedData.apply_qfunc, "dx", dim * dim, CEED_EVAL_GRAD);
   CeedQFunctionAddInput(ceedData.apply_qfunc, "weights", 1, CEED_EVAL_WEIGHT);
   CeedQFunctionAddOutput(ceedData.apply_qfunc, "v", dimV, op.test_op);

   CeedQFunctionContextCreate(ceed, &ceedData.build_ctx);
   CeedQFunctionContextSetData(ceedData.build_ctx, CEED_MEM_HOST, CEED_USE_POINTER,
                               sizeof(ceedData.build_ctx_data),
                               &ceedData.build_ctx_data);
   CeedQFunctionSetContext(ceedData.apply_qfunc, ceedData.build_ctx);

   // Create the operator.
   CeedOperatorCreate(ceed, ceedData.apply_qfunc, NULL, NULL, &ceedData.oper);
   CeedOperatorSetField(ceedData.oper, "u", ceedData.restr, ceedData.basis,
                        CEED_VECTOR_ACTIVE);
   switch (ceedData.coeff_type)
   {
      case CeedCoeff::Const:
         break;
      case CeedCoeff::Grid:
      {
         CeedGridCoeff* gridCoeff = (CeedGridCoeff*)ceedData.coeff;
         InitCeedBasisAndRestriction(*gridCoeff->coeff->FESpace(), irm, ceed,
                                     &gridCoeff->basis,
                                     &gridCoeff->restr);
         CeedOperatorSetField(ceedData.oper, "coeff", gridCoeff->restr,
                              gridCoeff->basis, gridCoeff->coeffVector);
      }
      break;
      case CeedCoeff::Quad:
      {
         CeedQuadCoeff* quadCoeff = (CeedQuadCoeff*)ceedData.coeff;
         const int ncomp = 1;
         CeedInt strides[3] = {1, nqpts, ncomp*nqpts};
         InitCeedStridedRestriction(*mesh->GetNodalFESpace(),
                                    nelem, nqpts, ncomp, strides,
                                    &quadCoeff->restr);
         CeedOperatorSetField(ceedData.oper, "coeff", quadCoeff->restr,
                              CEED_BASIS_COLLOCATED, quadCoeff->coeffVector);
      }
      break;
   }
   CeedOperatorSetField(ceedData.oper, "dx", ceedData.mesh_restr,
                        ceedData.mesh_basis, ceedData.node_coords);
   CeedOperatorSetField(ceedData.oper, "weights", CEED_ELEMRESTRICTION_NONE,
                        ceedData.mesh_basis, CEED_VECTOR_NONE);
   CeedOperatorSetField(ceedData.oper, "v", ceedData.restr, ceedData.basis,
                        CEED_VECTOR_ACTIVE);

   CeedVectorCreate(ceed, vdim*fes.GetNDofs(), &ceedData.u);
   CeedVectorCreate(ceed, vdim*fes.GetNDofs(), &ceedData.v);
#else
   mfem_error("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

void CeedAddMult(const CeedData *ceedDataPtr,
                 const Vector &x,
                 Vector &y)
{
#ifdef MFEM_USE_CEED
   const CeedScalar *x_ptr;
   CeedScalar *y_ptr;
   CeedMemType mem;
   CeedGetPreferredMemType(internal::ceed, &mem);
   if ( Device::Allows(Backend::DEVICE_MASK) && mem==CEED_MEM_DEVICE )
   {
      x_ptr = x.Read();
      y_ptr = y.ReadWrite();
   }
   else
   {
      x_ptr = x.HostRead();
      y_ptr = y.HostReadWrite();
      mem = CEED_MEM_HOST;
   }
   CeedVectorSetArray(ceedDataPtr->u, mem, CEED_USE_POINTER,
                      const_cast<CeedScalar*>(x_ptr));
   CeedVectorSetArray(ceedDataPtr->v, mem, CEED_USE_POINTER, y_ptr);

   CeedOperatorApplyAdd(ceedDataPtr->oper, ceedDataPtr->u, ceedDataPtr->v,
                        CEED_REQUEST_IMMEDIATE);

   CeedVectorTakeArray(ceedDataPtr->u, mem, const_cast<CeedScalar**>(&x_ptr));
   CeedVectorTakeArray(ceedDataPtr->v, mem, &y_ptr);
#else
   mfem_error("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

void CeedAssembleDiagonal(const CeedData *ceedDataPtr,
                          Vector &diag)
{
#ifdef MFEM_USE_CEED
   CeedScalar *d_ptr;
   CeedMemType mem;
   CeedGetPreferredMemType(internal::ceed, &mem);
   if ( Device::Allows(Backend::DEVICE_MASK) && mem==CEED_MEM_DEVICE )
   {
      d_ptr = diag.ReadWrite();
   }
   else
   {
      d_ptr = diag.HostReadWrite();
      mem = CEED_MEM_HOST;
   }
   CeedVectorSetArray(ceedDataPtr->v, mem, CEED_USE_POINTER, d_ptr);

   CeedOperatorLinearAssembleAddDiagonal(ceedDataPtr->oper, ceedDataPtr->v,
                                         CEED_REQUEST_IMMEDIATE);

   CeedVectorTakeArray(ceedDataPtr->v, mem, &d_ptr);
#else
   mfem_error("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

void RemoveCeedBasisAndRestriction(const FiniteElementSpace *fes)
{
#ifdef MFEM_USE_CEED
   auto itb = internal::ceed_basis_map.begin();
   while (itb != internal::ceed_basis_map.end())
   {
      if (std::get<0>(itb->first)==fes)
      {
         CeedBasisDestroy(&itb->second);
         itb = internal::ceed_basis_map.erase(itb);
      }
      else
      {
         itb++;
      }
   }
   auto itr = internal::ceed_restr_map.begin();
   while (itr != internal::ceed_restr_map.end())
   {
      if (std::get<0>(itr->first)==fes)
      {
         CeedElemRestrictionDestroy(&itr->second);
         itr = internal::ceed_restr_map.erase(itr);
      }
      else
      {
         itr++;
      }
   }
#endif
}

#ifdef MFEM_USE_CEED
void InitCeedVector(const Vector &v, CeedVector &cv)
{
   CeedVectorCreate(internal::ceed, v.Size(), &cv);
   CeedScalar *cv_ptr;
   CeedMemType mem;
   CeedGetPreferredMemType(internal::ceed, &mem);
   if ( Device::Allows(Backend::DEVICE_MASK) && mem==CEED_MEM_DEVICE )
   {
      cv_ptr = const_cast<CeedScalar*>(v.Read());
   }
   else
   {
      cv_ptr = const_cast<CeedScalar*>(v.HostRead());
      mem = CEED_MEM_HOST;
   }
   CeedVectorSetArray(cv, mem, CEED_USE_POINTER, cv_ptr);
}

static CeedElemTopology GetCeedTopology(Geometry::Type geom)
{
   switch (geom)
   {
      case Geometry::SEGMENT:
         return CEED_LINE;
      case Geometry::TRIANGLE:
         return CEED_TRIANGLE;
      case Geometry::SQUARE:
         return CEED_QUAD;
      case Geometry::TETRAHEDRON:
         return CEED_TET;
      case Geometry::CUBE:
         return CEED_HEX;
      case Geometry::PRISM:
         return CEED_PRISM;
      default:
         MFEM_ABORT("This type of element is not supported");
         return CEED_PRISM;
   }
}

static void InitCeedNonTensorBasis(const FiniteElementSpace &fes,
                                   const IntegrationRule &ir,
                                   Ceed ceed, CeedBasis *basis)
{
   const DofToQuad &maps = fes.GetFE(0)->GetDofToQuad(ir, DofToQuad::FULL);
   Mesh *mesh = fes.GetMesh();
   const int dim = mesh->Dimension();
   const int ndofs = maps.ndof;
   const int nqpts = maps.nqpt;
   DenseMatrix qX(dim,nqpts);
   Vector qW(nqpts);
   for (int i = 0; i < nqpts; i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      qX(0,i) = ip.x;
      if (dim>1) { qX(1,i) = ip.y; }
      if (dim>2) { qX(2,i) = ip.z; }
      qW(i) = ip.weight;
   }
   CeedBasisCreateH1(ceed, GetCeedTopology(fes.GetFE(0)->GetGeomType()),
                     fes.GetVDim(), ndofs, nqpts,
                     maps.Bt.GetData(), maps.Gt.GetData(),
                     qX.GetData(), qW.GetData(), basis);
}

static void InitCeedNonTensorRestriction(const FiniteElementSpace &fes,
                                         Ceed ceed, CeedElemRestriction *restr)
{
   Mesh *mesh = fes.GetMesh();
   const FiniteElement *fe = fes.GetFE(0);
   const int P = fe->GetDof();
   CeedInt compstride = fes.GetOrdering()==Ordering::byVDIM ? 1 : fes.GetNDofs();
   const Table &el_dof = fes.GetElementToDofTable();
   Array<int> tp_el_dof(el_dof.Size_of_connections());
   const TensorBasisElement * tfe =
      dynamic_cast<const TensorBasisElement *>(fe);
   const int stride = compstride == 1 ? fes.GetVDim() : 1;
   if (tfe) // Lexicographic ordering using dof_map
   {
      const Array<int>& dof_map = tfe->GetDofMap();
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         const int el_offset = P * i;
         for (int j = 0; j < P; j++)
         {
            tp_el_dof[j+el_offset] = stride*el_dof.GetJ()[dof_map[j]+el_offset];
         }
      }
   }
   else  // Native ordering
   {
      for (int e = 0; e < mesh->GetNE(); e++)
      {
         for (int i = 0; i < P; i++)
         {
            tp_el_dof[i + e*P] = stride*el_dof.GetJ()[i + e*P];
         }
      }
   }
   CeedElemRestrictionCreate(ceed, mesh->GetNE(), P, fes.GetVDim(),
                             compstride, (fes.GetVDim())*(fes.GetNDofs()),
                             CEED_MEM_HOST, CEED_COPY_VALUES,
                             tp_el_dof.GetData(), restr);
}

static void InitCeedTensorBasis(const FiniteElementSpace &fes,
                                const IntegrationRule &ir,
                                Ceed ceed, CeedBasis *basis)
{
   const DofToQuad &maps = fes.GetFE(0)->GetDofToQuad(ir, DofToQuad::TENSOR);
   Mesh *mesh = fes.GetMesh();
   const int ndofs = maps.ndof;
   const int nqpts = maps.nqpt;
   Vector qX(nqpts), qW(nqpts);
   const IntegrationRule &ir1d = IntRules.Get(Geometry::SEGMENT, ir.GetOrder());
   for (int i = 0; i < nqpts; i++)
   {
      const IntegrationPoint &ip = ir1d.IntPoint(i);
      qX(i) = ip.x;
      qW(i) = ip.weight;
   }
   CeedBasisCreateTensorH1(ceed, mesh->Dimension(), fes.GetVDim(), ndofs,
                           nqpts, maps.Bt.GetData(),
                           maps.Gt.GetData(), qX.GetData(),
                           qW.GetData(), basis);
}

static void InitCeedTensorRestriction(const FiniteElementSpace &fes,
                                      Ceed ceed, CeedElemRestriction *restr)
{
   Mesh *mesh = fes.GetMesh();
   const FiniteElement *fe = fes.GetFE(0);
   const TensorBasisElement * tfe =
      dynamic_cast<const TensorBasisElement *>(fe);
   MFEM_VERIFY(tfe, "invalid FE");
   const Array<int>& dof_map = tfe->GetDofMap();

   CeedInt compstride = fes.GetOrdering()==Ordering::byVDIM ? 1 : fes.GetNDofs();
   const Table &el_dof = fes.GetElementToDofTable();
   Array<int> tp_el_dof(el_dof.Size_of_connections());
   const int dof = fe->GetDof();
   const int stride = compstride == 1 ? fes.GetVDim() : 1;
   if (dof_map.Size()>0)
   {
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         const int el_offset = dof * i;
         for (int j = 0; j < dof; j++)
         {
            tp_el_dof[j+el_offset] = stride*el_dof.GetJ()[dof_map[j]+el_offset];
         }
      }
   }
   else // dof_map.Size == 0, means dof_map[j]==j;
   {
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         const int el_offset = dof * i;
         for (int j = 0; j < dof; j++)
         {
            tp_el_dof[j+el_offset] = stride*el_dof.GetJ()[j+el_offset];
         }
      }
   }
   CeedElemRestrictionCreate(ceed, mesh->GetNE(), dof, fes.GetVDim(),
                             compstride, (fes.GetVDim())*(fes.GetNDofs()),
                             CEED_MEM_HOST, CEED_COPY_VALUES,
                             tp_el_dof.GetData(), restr);
}

void InitCeedStridedRestriction(const FiniteElementSpace &fes,
                                CeedInt nelem, CeedInt nqpts, CeedInt qdatasize,
                                const CeedInt *strides,
                                CeedElemRestriction *restr)
{
   CeedRestrKey restr_key(&fes, nelem, nqpts, qdatasize);
   auto restr_itr = internal::ceed_restr_map.find(restr_key);
   if (restr_itr == internal::ceed_restr_map.end())
   {
      CeedElemRestrictionCreateStrided(internal::ceed, nelem, nqpts, qdatasize,
                                       nelem*nqpts*qdatasize,
                                       strides,
                                       restr);
      internal::ceed_restr_map[restr_key] = *restr;
   }
   else
   {
      *restr = restr_itr->second;
   }
}

void InitCeedBasisAndRestriction(const FiniteElementSpace &fes,
                                 const IntegrationRule &irm,
                                 Ceed ceed, CeedBasis *basis,
                                 CeedElemRestriction *restr)
{
   // Check for FES -> basis, restriction in hash tables
   const Mesh *mesh = fes.GetMesh();
   const FiniteElement *fe = fes.GetFE(0);
   const int P = fe->GetDof();
   const int Q = irm.GetNPoints();
   const int nelem = mesh->GetNE();
   const int ncomp = fes.GetVDim();
   CeedBasisKey basis_key(&fes, &irm, ncomp, P, Q);
   auto basis_itr = internal::ceed_basis_map.find(basis_key);
   CeedRestrKey restr_key(&fes, nelem, P, ncomp);
   auto restr_itr = internal::ceed_restr_map.find(restr_key);

   // Init or retreive key values
   if (basis_itr == internal::ceed_basis_map.end())
   {
      if (UsesTensorBasis(fes))
      {
         InitCeedTensorBasis(fes, irm, ceed, basis);
      }
      else
      {
         InitCeedNonTensorBasis(fes, irm, ceed, basis);
      }
      internal::ceed_basis_map[basis_key] = *basis;
   }
   else
   {
      *basis = basis_itr->second;
   }
   if (restr_itr == internal::ceed_restr_map.end())
   {
      if (UsesTensorBasis(fes))
      {
         InitCeedTensorRestriction(fes, ceed, restr);
      }
      else
      {
         InitCeedNonTensorRestriction(fes, ceed, restr);
      }
      internal::ceed_restr_map[restr_key] = *restr;
   }
   else
   {
      *restr = restr_itr->second;
   }
}

const std::string &GetCeedPath()
{
   if (internal::ceed_path.empty())
   {
      const char *install_dir = MFEM_INSTALL_DIR "/include/mfem/fem/libceed";
      const char *source_dir = MFEM_SOURCE_DIR "/fem/libceed";
      struct_stat m_stat;
      if (stat(install_dir, &m_stat) == 0 && S_ISDIR(m_stat.st_mode))
      {
         internal::ceed_path = install_dir;
      }
      else if (stat(source_dir, &m_stat) == 0 && S_ISDIR(m_stat.st_mode))
      {
         internal::ceed_path = source_dir;
      }
      else
      {
         MFEM_ABORT("Cannot find libCEED kernels in MFEM_INSTALL_DIR or "
                    "MFEM_SOURCE_DIR");
      }
      // Could be useful for debugging:
      // mfem::out << "Using libCEED dir: " << internal::ceed_path << std::endl;
   }
   return internal::ceed_path;
}

#endif // MFEM_USE_CEED

} // namespace mfem
