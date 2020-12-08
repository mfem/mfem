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

#ifndef MFEM_LIBCEED_PAINTEG
#define MFEM_LIBCEED_PAINTEG

#include <ceed.h>
#include "operator.hpp"
#include "mf_integrator.hpp"
#include "util.hpp"
#include "coefficient.hpp"
#include "../fespace.hpp"
#include "../gridfunc.hpp"

namespace mfem
{

/** This structure contains the data to assemble a PA operator with libCEED.
    See libceed/mass.cpp or libceed/diffusion.cpp for examples. */
struct CeedPAOperator
{
   /** The finite element space for the trial and test functions. */
   const FiniteElementSpace &fes;
   /** The Integration Rule to use to compute the operator. */
   const IntegrationRule &ir;
   /** The number of quadrature data at each quadrature point. */
   int qdatasize;
   /** The path to the header containing the functions for libCEED. */
   std::string header;
   /** The name of the Qfunction to build the quadrature data with a constant
       coefficient.*/
   std::string const_func;
   /** The Qfunction to build the quadrature data with constant coefficient. */
   CeedQFunctionUser const_qf;
   /** The name of the Qfunction to build the quadrature data with a coefficient
       evaluated at quadrature points. */
   std::string quad_func;
   /** The Qfunction to build the quad. data with a coefficient. */
   CeedQFunctionUser quad_qf;
   /** The name of the Qfunction to build the quadrature data with a constant
       vector coefficient.*/
   std::string vec_const_func;
   /** The Qfunction to build the quadrature data with constant vector
       coefficient. */
   CeedQFunctionUser vec_const_qf;
   /** The name of the Qfunction to build the quadrature data with a vector
       coefficient evaluated at quadrature points. */
   std::string vec_quad_func;
   /** The Qfunction to build the quad. data with a vector coefficient. */
   CeedQFunctionUser vec_quad_qf;
   /** The name of the Qfunction to apply the operator. */
   std::string apply_func;
   /** The Qfunction to apply the operator. */
   CeedQFunctionUser apply_qf;
   /** The evaluation mode to apply to the trial function (CEED_EVAL_INTERP,
       CEED_EVAL_GRAD, etc.) */
   EvalMode trial_op;
   /** The evaluation mode to apply to the test function ( CEED_EVAL_INTERP,
       CEED_EVAL_GRAD, etc.)*/
   EvalMode test_op;
};

class CeedPAIntegrator : public CeedMFIntegrator
{
protected:
   CeedOperator build_oper;
   CeedQFunction build_qfunc;

public:
   CeedPAIntegrator()
   : CeedMFIntegrator(),  build_oper(nullptr), build_qfunc(nullptr) { }

   template <typename Context>
   void Assemble(CeedPAOperator &op, Context &ctx)
   {
      const FiniteElementSpace &fes = op.fes;
      const IntegrationRule &irm = op.ir;
      Ceed ceed(internal::ceed);
      Mesh *mesh = fes.GetMesh();
      CeedInt nqpts, nelem = mesh->GetNE();
      CeedInt dim = mesh->SpaceDimension(), vdim = fes.GetVDim();

      mesh->EnsureNodes();
      InitCeedBasisAndRestriction(fes, irm, ceed, &basis, &restr);

      const FiniteElementSpace *mesh_fes = mesh->GetNodalFESpace();
      MFEM_VERIFY(mesh_fes, "the Mesh has no nodal FE space");
      InitCeedBasisAndRestriction(*mesh_fes, irm, ceed, &mesh_basis,
                                  &mesh_restr);

      CeedBasisGetNumQuadraturePoints(basis, &nqpts);

      const int qdatasize = op.qdatasize;
      InitCeedStridedRestriction(*mesh_fes, nelem, nqpts, qdatasize,
                                 CEED_STRIDES_BACKEND,
                                 &restr_i);

      InitCeedVector(*mesh->GetNodes(), node_coords);

      CeedVectorCreate(ceed, nelem * nqpts * qdatasize, &qdata);

      // Context data to be passed to the Q-function.
      ctx.dim = mesh->Dimension();
      ctx.space_dim = mesh->SpaceDimension();
      ctx.vdim = fes.GetVDim();

      std::string qf_file = GetCeedPath() + op.header;
      std::string qf;

      // Create the Q-function that builds the operator (i.e. computes its
      // quadrature data) and set its context data.
      switch (coeff_type)
      {
         case CeedCoeff::Const:
            qf = qf_file + op.const_func;
            CeedQFunctionCreateInterior(ceed, 1, op.const_qf,
                                        qf.c_str(),
                                        &build_qfunc);
            ctx.coeff[0] = ((CeedConstCoeff*)coeff)->val;
            break;
         case CeedCoeff::Grid:
            qf = qf_file + op.quad_func;
            CeedQFunctionCreateInterior(ceed, 1, op.quad_qf,
                                        qf.c_str(),
                                        &build_qfunc);
            CeedQFunctionAddInput(build_qfunc, "coeff", 1, CEED_EVAL_INTERP);
            break;
         case CeedCoeff::Quad:
            qf = qf_file + op.quad_func;
            CeedQFunctionCreateInterior(ceed, 1, op.quad_qf,
                                        qf.c_str(),
                                        &build_qfunc);
            CeedQFunctionAddInput(build_qfunc, "coeff", 1, CEED_EVAL_NONE);
            break;
         case CeedCoeff::VecConst:
         {
            qf = qf_file + op.vec_const_func;
            CeedQFunctionCreateInterior(ceed, 1, op.vec_const_qf,
                                       qf.c_str(),
                                       &build_qfunc);
            CeedVecConstCoeff *vcoeff = static_cast<CeedVecConstCoeff*>(coeff);
            for (int i = 0; i < dim; i++)
            {
               ctx.coeff[i] = vcoeff->val[i];
            }
         }
         break;
         case CeedCoeff::VecGrid:
            qf = qf_file + op.vec_quad_func;
            CeedQFunctionCreateInterior(ceed, 1, op.vec_quad_qf,
                                       qf.c_str(),
                                       &build_qfunc);
            CeedQFunctionAddInput(build_qfunc, "coeff", dim, CEED_EVAL_INTERP);
            break;
         case CeedCoeff::VecQuad:
            qf = qf_file + op.vec_quad_func;
            CeedQFunctionCreateInterior(ceed, 1, op.vec_quad_qf,
                                       qf.c_str(),
                                       &build_qfunc);
            CeedQFunctionAddInput(build_qfunc, "coeff", dim, CEED_EVAL_NONE);
            break;
      }
      CeedQFunctionAddInput(build_qfunc, "dx", dim * dim, CEED_EVAL_GRAD);
      CeedQFunctionAddInput(build_qfunc, "weights", 1, CEED_EVAL_WEIGHT);
      CeedQFunctionAddOutput(build_qfunc, "qdata", qdatasize, CEED_EVAL_NONE);

      CeedQFunctionContextCreate(ceed, &build_ctx);
      CeedQFunctionContextSetData(build_ctx, CEED_MEM_HOST,
                                  CEED_COPY_VALUES,
                                  sizeof(ctx),
                                  &ctx);
      CeedQFunctionSetContext(build_qfunc, build_ctx);

      // Create the operator that builds the quadrature data for the operator.
      CeedOperatorCreate(ceed, build_qfunc, NULL, NULL, &build_oper);
      switch (coeff_type)
      {
         case CeedCoeff::Const:
            break;
         case CeedCoeff::Grid:
         {
            CeedGridCoeff* gridCoeff = (CeedGridCoeff*)coeff;
            InitCeedBasisAndRestriction(*gridCoeff->coeff->FESpace(), irm, ceed,
                                        &gridCoeff->basis, &gridCoeff->restr);
            CeedOperatorSetField(build_oper, "coeff", gridCoeff->restr,
                                 gridCoeff->basis, gridCoeff->coeffVector);
         }
         break;
         case CeedCoeff::Quad:
         {
            CeedQuadCoeff* quadCoeff = (CeedQuadCoeff*)coeff;
            const int ncomp = 1;
            CeedInt strides[3] = {1, nqpts, ncomp*nqpts};
            InitCeedStridedRestriction(*mesh_fes, nelem, nqpts, ncomp, strides,
                                       &quadCoeff->restr);
            CeedOperatorSetField(build_oper, "coeff", quadCoeff->restr,
                                 CEED_BASIS_COLLOCATED, quadCoeff->coeffVector);
         }
         break;
         case CeedCoeff::VecConst:
            break;
         case CeedCoeff::VecGrid:
         {
            CeedVecGridCoeff* gridCoeff = (CeedVecGridCoeff*)coeff;
            InitCeedBasisAndRestriction(*gridCoeff->coeff->FESpace(), irm, ceed,
                                        &gridCoeff->basis, &gridCoeff->restr);
            CeedOperatorSetField(build_oper, "coeff", gridCoeff->restr,
                                 gridCoeff->basis, gridCoeff->coeffVector);
         }
         break;
         case CeedCoeff::VecQuad:
         {
            CeedVecQuadCoeff* quadCoeff = (CeedVecQuadCoeff*)coeff;
            const int ncomp = dim;
            CeedInt strides[3] = {ncomp, 1, ncomp*nqpts};
            InitCeedStridedRestriction(*mesh_fes, nelem, nqpts, ncomp, strides,
                                       &quadCoeff->restr);
            CeedOperatorSetField(build_oper, "coeff", quadCoeff->restr,
                                 CEED_BASIS_COLLOCATED, quadCoeff->coeffVector);
         }
         break;
      }
      CeedOperatorSetField(build_oper, "dx", mesh_restr,
                           mesh_basis, CEED_VECTOR_ACTIVE);
      CeedOperatorSetField(build_oper, "weights", CEED_ELEMRESTRICTION_NONE,
                           mesh_basis, CEED_VECTOR_NONE);
      CeedOperatorSetField(build_oper, "qdata", restr_i,
                           CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

      // Compute the quadrature data for the operator.
      CeedOperatorApply(build_oper, node_coords, qdata, CEED_REQUEST_IMMEDIATE);

      // Create the Q-function that defines the action of the operator.
      qf = qf_file + op.apply_func;
      CeedQFunctionCreateInterior(ceed, 1, op.apply_qf, qf.c_str(), &apply_qfunc);
      // input
      switch (op.trial_op)
      {
         case EvalMode::None:
            CeedQFunctionAddInput(apply_qfunc, "u", vdim, CEED_EVAL_NONE);
            break;
         case EvalMode::Interp:
            CeedQFunctionAddInput(apply_qfunc, "u", vdim, CEED_EVAL_INTERP);
            break;
         case EvalMode::Grad:
            CeedQFunctionAddInput(apply_qfunc, "gu", vdim*dim, CEED_EVAL_GRAD);
            break;
         case EvalMode::InterpAndGrad:
            CeedQFunctionAddInput(apply_qfunc, "u", vdim, CEED_EVAL_INTERP);
            CeedQFunctionAddInput(apply_qfunc, "gu", vdim*dim, CEED_EVAL_GRAD);
            break;
      }
      // qdata
      CeedQFunctionAddInput(apply_qfunc, "qdata", qdatasize, CEED_EVAL_NONE);
      // output
      switch (op.test_op)
      {
         case EvalMode::None:
            CeedQFunctionAddOutput(apply_qfunc, "v", vdim, CEED_EVAL_NONE);
            break;
         case EvalMode::Interp:
            CeedQFunctionAddOutput(apply_qfunc, "v", vdim, CEED_EVAL_INTERP);
            break;
         case EvalMode::Grad:
            CeedQFunctionAddOutput(apply_qfunc, "gv", vdim*dim, CEED_EVAL_GRAD);
            break;
         case EvalMode::InterpAndGrad:
            CeedQFunctionAddOutput(apply_qfunc, "v", vdim, CEED_EVAL_INTERP);
            CeedQFunctionAddOutput(apply_qfunc, "gv", vdim*dim, CEED_EVAL_GRAD);
            break;
      }
      CeedQFunctionSetContext(apply_qfunc, build_ctx);

      // Create the operator.
      CeedOperatorCreate(ceed, apply_qfunc, NULL, NULL, &oper);
      // input
      switch (op.trial_op)
      {
         case EvalMode::None:
            CeedOperatorSetField(oper, "u", restr,
                                 CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
            break;
         case EvalMode::Interp:
            CeedOperatorSetField(oper, "u", restr, basis, CEED_VECTOR_ACTIVE);
            break;
         case EvalMode::Grad:
            CeedOperatorSetField(oper, "gu", restr, basis, CEED_VECTOR_ACTIVE);
            break;
         case EvalMode::InterpAndGrad:
            CeedOperatorSetField(oper, "u", restr, basis, CEED_VECTOR_ACTIVE);
            CeedOperatorSetField(oper, "gu", restr, basis, CEED_VECTOR_ACTIVE);
            break;
      }
      // qdata
      CeedOperatorSetField(oper, "qdata", restr_i, CEED_BASIS_COLLOCATED, qdata);
      // output
      switch (op.test_op)
      {
         case EvalMode::None:
            CeedOperatorSetField(oper, "v", restr,
                                 CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
            break;
         case EvalMode::Interp:
            CeedOperatorSetField(oper, "v", restr, basis, CEED_VECTOR_ACTIVE);
            break;
         case EvalMode::Grad:
            CeedOperatorSetField(oper, "gv", restr, basis, CEED_VECTOR_ACTIVE);
            break;
         case EvalMode::InterpAndGrad:
            CeedOperatorSetField(oper, "v", restr, basis, CEED_VECTOR_ACTIVE);
            CeedOperatorSetField(oper, "gv", restr, basis, CEED_VECTOR_ACTIVE);
            break;
      }

      CeedVectorCreate(ceed, vdim*fes.GetNDofs(), &u);
      CeedVectorCreate(ceed, vdim*fes.GetNDofs(), &v);
   }

   ~CeedPAIntegrator()
   {
      // ~CeedMFIntegrator();
      CeedOperatorDestroy(&build_oper);
      CeedQFunctionDestroy(&build_qfunc);
   }
};

} // namespace mfem

#endif // MFEM_LIBCEED_PAINTEG
