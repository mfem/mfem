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

#ifndef MFEM_LIBCEED_MFINTEG
#define MFEM_LIBCEED_MFINTEG

#include "operator.hpp"
#include "../../config/config.hpp"
#include "coefficient.hpp"
#include "../fespace.hpp"
#include "../gridfunc.hpp"

namespace mfem
{

#ifdef MFEM_USE_CEED
/** This structure contains the data to assemble a MF operator with libCEED.
    See libceed/mass.cpp or libceed/diffusion.cpp for examples. */
struct CeedMFOperator
{
   /** The finite element space for the trial and test functions. */
   const FiniteElementSpace &fes;
   /** The Integration Rule to use to compote the operator. */
   const IntegrationRule &ir;
   /** The path to the header containing the functions for libCEED. */
   std::string header;
   /** The name of the Qfunction to apply the quadrature data with a constant
       coefficient.*/
   std::string const_func;
   /** The Qfunction to apply the quadrature data with constant coefficient. */
   CeedQFunctionUser const_qf;
   /** The name of the Qfunction to apply the quadrature data with a coefficient
       evaluated at quadrature point. */
   std::string quad_func;
   /** The Qfunction to apply the quad. data with a coefficient evaluated at
       quadrature point. */
   CeedQFunctionUser quad_qf;
   /** The name of the Qfunction to apply the quadrature data with a constant
       vector coefficient.*/
   std::string vec_const_func;
   /** The Qfunction to apply the quadrature data with constant vector
       coefficient. */
   CeedQFunctionUser vec_const_qf;
   /** The name of the Qfunction to apply the quadrature data with a vector
       coefficient evaluated at quadrature point. */
   std::string vec_quad_func;
   /** The Qfunction to apply the quad. data with a vector coefficient evaluated
       at quadrature point. */
   CeedQFunctionUser vec_quad_qf;
   /** The evaluation mode to apply to the trial function (CEED_EVAL_INTERP,
       CEED_EVAL_GRAD, etc.) */
   EvalMode trial_op;
   /** The evaluation mode to apply to the test function ( CEED_EVAL_INTERP,
       CEED_EVAL_GRAD, etc.) */
   EvalMode test_op;
};
#endif

class CeedMFIntegrator : public MFEMCeedOperator
{
#ifdef MFEM_USE_CEED
protected:
   CeedBasis basis, mesh_basis;
   CeedElemRestriction restr, mesh_restr, restr_i, mesh_restr_i;
   CeedQFunction apply_qfunc;
   CeedVector node_coords, qdata;
   CeedCoeff *coeff;
   CeedQFunctionContext build_ctx;

public:
   CeedMFIntegrator()
      : MFEMCeedOperator(), basis(nullptr), mesh_basis(nullptr),
        restr(nullptr), mesh_restr(nullptr),
        restr_i(nullptr), mesh_restr_i(nullptr),
        apply_qfunc(nullptr), node_coords(nullptr),
        qdata(nullptr), coeff(nullptr), build_ctx(nullptr) { }

   template <typename Context>
   void Assemble(CeedMFOperator &op, Context &ctx)
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

      InitCeedVector(*mesh->GetNodes(), node_coords);

      // Context data to be passed to the Q-function.
      ctx.dim = mesh->Dimension();
      ctx.space_dim = mesh->SpaceDimension();
      ctx.vdim = fes.GetVDim();

      std::string qf_file = GetCeedPath() + op.header;
      std::string qf;

      // Create the Q-function that builds the operator (i.e. computes its
      // quadrature data) and set its context data.
      switch (coeff->coeff_type)
      {
         case CeedCoeffType::Const:
            qf = qf_file + op.const_func;
            CeedQFunctionCreateInterior(ceed, 1, op.const_qf, qf.c_str(),
                                        &apply_qfunc);
            break;
         case CeedCoeffType::Grid:
            qf = qf_file + op.quad_func;
            CeedQFunctionCreateInterior(ceed, 1, op.quad_qf, qf.c_str(),
                                        &apply_qfunc);
            CeedQFunctionAddInput(apply_qfunc, "coeff", 1, CEED_EVAL_INTERP);
            break;
         case CeedCoeffType::Quad:
            qf = qf_file + op.quad_func;
            CeedQFunctionCreateInterior(ceed, 1, op.quad_qf, qf.c_str(),
                                        &apply_qfunc);
            CeedQFunctionAddInput(apply_qfunc, "coeff", 1, CEED_EVAL_NONE);
            break;
         case CeedCoeffType::VecConst:
            qf = qf_file + op.vec_const_func;
            CeedQFunctionCreateInterior(ceed, 1, op.vec_const_qf, qf.c_str(),
                                        &apply_qfunc);
            break;
         case CeedCoeffType::VecGrid:
            qf = qf_file + op.vec_quad_func;
            CeedQFunctionCreateInterior(ceed, 1, op.vec_quad_qf, qf.c_str(),
                                        &apply_qfunc);
            CeedQFunctionAddInput(apply_qfunc, "coeff", dim, CEED_EVAL_INTERP);
            break;
         case CeedCoeffType::VecQuad:
            qf = qf_file + op.vec_quad_func;
            CeedQFunctionCreateInterior(ceed, 1, op.vec_quad_qf, qf.c_str(),
                                        &apply_qfunc);
            CeedQFunctionAddInput(apply_qfunc, "coeff", dim, CEED_EVAL_NONE);
            break;
      }
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
      CeedQFunctionAddInput(apply_qfunc, "dx", dim * dim, CEED_EVAL_GRAD);
      CeedQFunctionAddInput(apply_qfunc, "weights", 1, CEED_EVAL_WEIGHT);
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

      CeedQFunctionContextCreate(ceed, &build_ctx);
      CeedQFunctionContextSetData(build_ctx, CEED_MEM_HOST,
                                  CEED_COPY_VALUES,
                                  sizeof(ctx),
                                  &ctx);
      CeedQFunctionSetContext(apply_qfunc, build_ctx);

      // Create the operator.
      CeedOperatorCreate(ceed, apply_qfunc, NULL, NULL, &oper);
      // coefficient
      switch (coeff->coeff_type)
      {
         case CeedCoeffType::Const:
         case CeedCoeffType::VecConst:
            break;
         case CeedCoeffType::Grid:
         case CeedCoeffType::VecGrid:
         {
            CeedGridCoeff *gridCoeff = (CeedGridCoeff*)coeff;
            InitCeedBasisAndRestriction(*gridCoeff->coeff->FESpace(), irm, ceed,
                                        &gridCoeff->basis, &gridCoeff->restr);
            CeedOperatorSetField(oper, "coeff", gridCoeff->restr,
                                 gridCoeff->basis, gridCoeff->coeffVector);
         }
         break;
         case CeedCoeffType::Quad:
         case CeedCoeffType::VecQuad:
         {
            CeedQuadCoeff *quadCoeff = (CeedQuadCoeff*)coeff;
            const int ncomp = quadCoeff->ncomp;
            CeedInt strides[3] = {ncomp, nqpts, ncomp*nqpts};
            InitCeedStridedRestriction(*mesh->GetNodalFESpace(),
                                       nelem, nqpts, ncomp, strides,
                                       &quadCoeff->restr);
            CeedOperatorSetField(oper, "coeff", quadCoeff->restr,
                                 CEED_BASIS_COLLOCATED, quadCoeff->coeffVector);
         }
         break;
      }
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
      CeedOperatorSetField(oper, "dx", mesh_restr,
                           mesh_basis, node_coords);
      CeedOperatorSetField(oper, "weights", CEED_ELEMRESTRICTION_NONE,
                           mesh_basis, CEED_VECTOR_NONE);
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

   virtual ~CeedMFIntegrator()
   {
      CeedQFunctionDestroy(&apply_qfunc);
      CeedQFunctionContextDestroy(&build_ctx);
      CeedVectorDestroy(&node_coords);
      CeedVectorDestroy(&qdata);
      delete coeff;
   }
#endif
};

} // namespace mfem

#endif // MFEM_LIBCEED_MFINTEG
