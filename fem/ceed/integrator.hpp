// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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

#include "operator.hpp"
#include "../../config/config.hpp"
#include "coefficient.hpp"
#include "../fespace.hpp"
#include "../gridfunc.hpp"

namespace mfem
{

namespace ceed
{

/** The different evaluation modes available for PA and MF CeedIntegrator. */
enum class EvalMode { None, Interp, Grad, InterpAndGrad };

#ifdef MFEM_USE_CEED
/** This structure is a template interface for the Assemble methods of
    PAIntegrator and MFIntegrator. See ceed/mass.cpp for an example. */
struct OperatorInfo
{
   /** The path to the qFunction header. */
   const char *header;
   /** The name of the qFunction to build a partially assembled CeedOperator
       with a constant Coefficient. */
   const char *build_func_const;
   /** The qFunction to build a partially assembled CeedOperator with a constant
       Coefficient. */
   CeedQFunctionUser build_qf_const;
   /** The name of the qFunction to build a partially assembled CeedOperator
       with a variable Coefficient. */
   const char *build_func_quad;
   /** The qFunction to build a partially assembled CeedOperator with a variable
       Coefficient. */
   CeedQFunctionUser build_qf_quad;
   /** The name of the qFunction to apply a partially assembled CeedOperator. */
   const char *apply_func;
   /** The qFunction to apply a partially assembled CeedOperator. */
   CeedQFunctionUser apply_qf;
   /** The name of the qFunction to apply a matrix-free CeedOperator with a
       constant Coefficient. */
   const char *apply_func_mf_const;
   /** The qFunction to apply a matrix-free CeedOperator with a constant
       Coefficient. */
   CeedQFunctionUser apply_qf_mf_const;
   /** The name of the qFunction to apply a matrix-free CeedOperator with a
       variable Coefficient. */
   const char *apply_func_mf_quad;
   /** The qFunction to apply a matrix-free CeedOperator with a variable
       Coefficient. */
   CeedQFunctionUser apply_qf_mf_quad;
   /** The EvalMode on the trial basis functions. */
   EvalMode trial_op;
   /** The EvalMode on the test basis functions. */
   EvalMode test_op;
   /** The size of the data at each quadrature point. */
   int qdatasize;
};
#endif

/** This class represent a partially assembled operator using libCEED. */
class PAIntegrator : public ceed::Operator
{
#ifdef MFEM_USE_CEED
protected:
   CeedBasis basis, mesh_basis;
   CeedElemRestriction restr, mesh_restr, restr_i;
   CeedQFunction build_qfunc, apply_qfunc;
   CeedVector node_coords, qdata;
   Coefficient *coeff;
   CeedQFunctionContext build_ctx;
   CeedOperator build_oper;

   PAIntegrator()
      : Operator(), basis(nullptr), mesh_basis(nullptr),
        restr(nullptr), mesh_restr(nullptr),
        restr_i(nullptr),
        build_qfunc(nullptr), apply_qfunc(nullptr), node_coords(nullptr),
        qdata(nullptr), coeff(nullptr), build_ctx(nullptr), build_oper(nullptr)
   { }

public:
   /** This method assembles the PAIntegrator with the given CeedOperatorInfo
       @a info, an mfem::FiniteElementSpace @a fes, an mfem::IntegrationRule
       @a ir, and mfem::Coefficient or mfem::VectorCoefficient @a Q.
       The CeedOperatorInfo type is expected to inherit from OperatorInfo and
       contain a Context type relevant to the qFunctions. */
   template <typename CeedOperatorInfo, typename CoeffType>
   void Assemble(CeedOperatorInfo &info,
                 const mfem::FiniteElementSpace &fes,
                 const mfem::IntegrationRule &irm,
                 CoeffType *Q)
   {
      Ceed ceed(internal::ceed);
      mfem::Mesh &mesh = *fes.GetMesh();
      InitCoefficient(Q, mesh, irm, coeff, info.ctx);
      bool const_coeff = coeff->IsConstant();
      std::string build_func = const_coeff ? info.build_func_const
                               : info.build_func_quad;
      CeedQFunctionUser build_qf = const_coeff ? info.build_qf_const
                                   : info.build_qf_quad;
      PAOperator op {info.qdatasize, info.header,
                     build_func, build_qf,
                     info.apply_func, info.apply_qf,
                     info.trial_op,
                     info.test_op
                    };
      CeedInt nqpts, nelem = mesh.GetNE();
      CeedInt dim = mesh.SpaceDimension(), vdim = fes.GetVDim();

      mesh.EnsureNodes();
      InitBasisAndRestriction(fes, irm, ceed, &basis, &restr);

      const mfem::FiniteElementSpace *mesh_fes = mesh.GetNodalFESpace();
      MFEM_VERIFY(mesh_fes, "the Mesh has no nodal FE space");
      InitBasisAndRestriction(*mesh_fes, irm, ceed, &mesh_basis,
                              &mesh_restr);

      CeedBasisGetNumQuadraturePoints(basis, &nqpts);

      const int qdatasize = op.qdatasize;
      InitStridedRestriction(*mesh_fes, nelem, nqpts, qdatasize,
                             CEED_STRIDES_BACKEND,
                             &restr_i);

      InitVector(*mesh.GetNodes(), node_coords);

      CeedVectorCreate(ceed, nelem * nqpts * qdatasize, &qdata);

      // Context data to be passed to the Q-function.
      info.ctx.dim = mesh.Dimension();
      info.ctx.space_dim = mesh.SpaceDimension();
      info.ctx.vdim = fes.GetVDim();

      std::string qf_file = GetCeedPath() + op.header;
      std::string qf = qf_file + op.build_func;
      CeedQFunctionCreateInterior(ceed, 1, op.build_qf, qf.c_str(),
                                  &build_qfunc);

      // Create the Q-function that builds the operator (i.e. computes its
      // quadrature data) and set its context data.
      if (VariableCoefficient *var_coeff =
             dynamic_cast<VariableCoefficient*>(coeff))
      {
         CeedQFunctionAddInput(build_qfunc, "coeff", coeff->ncomp,
                               var_coeff->emode);
      }
      CeedQFunctionAddInput(build_qfunc, "dx", dim * dim, CEED_EVAL_GRAD);
      CeedQFunctionAddInput(build_qfunc, "weights", 1, CEED_EVAL_WEIGHT);
      CeedQFunctionAddOutput(build_qfunc, "qdata", qdatasize, CEED_EVAL_NONE);

      CeedQFunctionContextCreate(ceed, &build_ctx);
      CeedQFunctionContextSetData(build_ctx, CEED_MEM_HOST,
                                  CEED_COPY_VALUES,
                                  sizeof(info.ctx),
                                  &info.ctx);
      CeedQFunctionSetContext(build_qfunc, build_ctx);

      // Create the operator that builds the quadrature data for the operator.
      CeedOperatorCreate(ceed, build_qfunc, NULL, NULL, &build_oper);
      if (GridCoefficient *gridCoeff = dynamic_cast<GridCoefficient*>(coeff))
      {
         InitBasisAndRestriction(*gridCoeff->gf.FESpace(), irm, ceed,
                                 &gridCoeff->basis, &gridCoeff->restr);
         CeedOperatorSetField(build_oper, "coeff", gridCoeff->restr,
                              gridCoeff->basis, gridCoeff->coeffVector);
      }
      else if (QuadCoefficient *quadCoeff =
                  dynamic_cast<QuadCoefficient*>(coeff))
      {
         const int ncomp = quadCoeff->ncomp;
         CeedInt strides[3] = {ncomp, 1, ncomp*nqpts};
         InitStridedRestriction(*mesh_fes, nelem, nqpts, ncomp, strides,
                                &quadCoeff->restr);
         CeedOperatorSetField(build_oper, "coeff", quadCoeff->restr,
                              CEED_BASIS_COLLOCATED, quadCoeff->coeffVector);
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
      CeedQFunctionCreateInterior(ceed, 1, op.apply_qf, qf.c_str(),
                                  &apply_qfunc);
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
      CeedOperatorSetField(oper, "qdata", restr_i, CEED_BASIS_COLLOCATED,
                           qdata);
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

   virtual ~PAIntegrator()
   {
      CeedQFunctionDestroy(&build_qfunc);
      CeedQFunctionDestroy(&apply_qfunc);
      CeedQFunctionContextDestroy(&build_ctx);
      CeedVectorDestroy(&node_coords);
      CeedVectorDestroy(&qdata);
      delete coeff;
      CeedOperatorDestroy(&build_oper);
   }

private:
   /** This structure contains the data to assemble a partially assembled
       operator with libCEED. */
   struct PAOperator
   {
      /** The number of quadrature data at each quadrature point. */
      int qdatasize;
      /** The path to the header containing the functions for libCEED. */
      std::string header;
      /** The name of the Qfunction to build the quadrature data. */
      std::string build_func;
      /** The Qfunction to build the quadrature data. */
      CeedQFunctionUser build_qf;
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
#endif
};

/** This class represent a matrix-free operator using libCEED. */
class MFIntegrator : public ceed::Operator
{
#ifdef MFEM_USE_CEED
protected:
   CeedBasis basis, mesh_basis;
   CeedElemRestriction restr, mesh_restr, restr_i;
   CeedQFunction apply_qfunc;
   CeedVector node_coords, qdata;
   Coefficient *coeff;
   CeedQFunctionContext build_ctx;

   MFIntegrator()
      : Operator(), basis(nullptr), mesh_basis(nullptr),
        restr(nullptr), mesh_restr(nullptr),
        restr_i(nullptr),
        apply_qfunc(nullptr), node_coords(nullptr),
        qdata(nullptr), coeff(nullptr), build_ctx(nullptr) { }

public:
   /** This method assembles the MFIntegrator with the given CeedOperatorInfo
       @a info, an mfem::FiniteElementSpace @a fes, an mfem::IntegrationRule
       @a ir, and mfem::Coefficient or mfem::VectorCoefficient @a Q.
       The CeedOperatorInfo type is expected to inherit from OperatorInfo and
       contain a Context type relevant to the qFunctions. */
   template <typename CeedOperatorInfo, typename CoeffType>
   void Assemble(CeedOperatorInfo &info,
                 const mfem::FiniteElementSpace &fes,
                 const mfem::IntegrationRule &irm,
                 CoeffType *Q)
   {
      Ceed ceed(internal::ceed);
      Mesh &mesh = *fes.GetMesh();
      InitCoefficient(Q, mesh, irm, coeff, info.ctx);
      bool const_coeff = coeff->IsConstant();
      std::string apply_func = const_coeff ? info.apply_func_mf_const
                               : info.apply_func_mf_quad;
      CeedQFunctionUser apply_qf = const_coeff ? info.apply_qf_mf_const
                                   : info.apply_qf_mf_quad;
      MFOperator op {info.header,
                     apply_func, apply_qf,
                     info.trial_op,
                     info.test_op
                    };
      CeedInt nqpts, nelem = mesh.GetNE();
      CeedInt dim = mesh.SpaceDimension(), vdim = fes.GetVDim();

      mesh.EnsureNodes();
      InitBasisAndRestriction(fes, irm, ceed, &basis, &restr);

      const mfem::FiniteElementSpace *mesh_fes = mesh.GetNodalFESpace();
      MFEM_VERIFY(mesh_fes, "the Mesh has no nodal FE space");
      InitBasisAndRestriction(*mesh_fes, irm, ceed, &mesh_basis,
                              &mesh_restr);

      CeedBasisGetNumQuadraturePoints(basis, &nqpts);

      InitVector(*mesh.GetNodes(), node_coords);

      // Context data to be passed to the Q-function.
      info.ctx.dim = mesh.Dimension();
      info.ctx.space_dim = mesh.SpaceDimension();
      info.ctx.vdim = fes.GetVDim();

      std::string qf_file = GetCeedPath() + op.header;
      std::string qf = qf_file + op.apply_func;
      CeedQFunctionCreateInterior(ceed, 1, op.apply_qf, qf.c_str(),
                                  &apply_qfunc);

      // Create the Q-function that builds the operator (i.e. computes its
      // quadrature data) and set its context data.
      if (VariableCoefficient *var_coeff =
             dynamic_cast<VariableCoefficient*>(coeff))
      {
         CeedQFunctionAddInput(apply_qfunc, "coeff", coeff->ncomp,
                               var_coeff->emode);
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
                                  sizeof(info.ctx),
                                  &info.ctx);
      CeedQFunctionSetContext(apply_qfunc, build_ctx);

      // Create the operator.
      CeedOperatorCreate(ceed, apply_qfunc, NULL, NULL, &oper);
      // coefficient
      if (GridCoefficient *gridCoeff = dynamic_cast<GridCoefficient*>(coeff))
      {
         InitBasisAndRestriction(*gridCoeff->gf.FESpace(), irm, ceed,
                                 &gridCoeff->basis, &gridCoeff->restr);
         CeedOperatorSetField(oper, "coeff", gridCoeff->restr,
                              gridCoeff->basis, gridCoeff->coeffVector);
      }
      else if (QuadCoefficient *quadCoeff =
                  dynamic_cast<QuadCoefficient*>(coeff))
      {
         const int ncomp = quadCoeff->ncomp;
         CeedInt strides[3] = {ncomp, 1, ncomp*nqpts};
         InitStridedRestriction(*mesh.GetNodalFESpace(),
                                nelem, nqpts, ncomp, strides,
                                &quadCoeff->restr);
         CeedOperatorSetField(oper, "coeff", quadCoeff->restr,
                              CEED_BASIS_COLLOCATED, quadCoeff->coeffVector);
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

   virtual ~MFIntegrator()
   {
      CeedQFunctionDestroy(&apply_qfunc);
      CeedQFunctionContextDestroy(&build_ctx);
      CeedVectorDestroy(&node_coords);
      CeedVectorDestroy(&qdata);
      delete coeff;
   }

private:
   /** This structure contains the data to assemble a matrix-free operator with
       libCEED. */
   struct MFOperator
   {
      /** The path to the header containing the functions for libCEED. */
      std::string header;
      /** The name of the Qfunction to apply the operator. */
      std::string apply_func;
      /** The Qfunction to apply the operator. */
      CeedQFunctionUser apply_qf;
      /** The evaluation mode to apply to the trial function (CEED_EVAL_INTERP,
          CEED_EVAL_GRAD, etc.) */
      EvalMode trial_op;
      /** The evaluation mode to apply to the test function ( CEED_EVAL_INTERP,
          CEED_EVAL_GRAD, etc.) */
      EvalMode test_op;
   };
#endif
};

} // namespace ceed

} // namespace mfem

#endif // MFEM_LIBCEED_PAINTEG
