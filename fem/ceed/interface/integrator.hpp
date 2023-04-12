// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_LIBCEED_INTEG
#define MFEM_LIBCEED_INTEG

#include "../../fespace.hpp"
#include "../../gridfunc.hpp"
#include "operator.hpp"
#include "coefficient.hpp"
#include "restriction.hpp"
#include "util.hpp"
#include "ceed.hpp"

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
   /** The path to the QFunction header. */
   const char *header;
   /** The name of the QFunction to build a partially assembled CeedOperator
       with a constant Coefficient. */
   const char *build_func_const;
   /** The QFunction to build a partially assembled CeedOperator with a constant
       Coefficient. */
   CeedQFunctionUser build_qf_const;
   /** The name of the QFunction to build a partially assembled CeedOperator
       with a variable Coefficient. */
   const char *build_func_quad;
   /** The QFunction to build a partially assembled CeedOperator with a variable
       Coefficient. */
   CeedQFunctionUser build_qf_quad;
   /** The name of the QFunction to apply a partially assembled CeedOperator. */
   const char *apply_func;
   /** The QFunction to apply a partially assembled CeedOperator. */
   CeedQFunctionUser apply_qf;
   /** The name of the QFunction to apply a matrix-free CeedOperator with a
       constant Coefficient. */
   const char *apply_func_mf_const;
   /** The QFunction to apply a matrix-free CeedOperator with a constant
       Coefficient. */
   CeedQFunctionUser apply_qf_mf_const;
   /** The name of the QFunction to apply a matrix-free CeedOperator with a
       variable Coefficient. */
   const char *apply_func_mf_quad;
   /** The QFunction to apply a matrix-free CeedOperator with a variable
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

/** This class represents a matrix-free or partially assembled operator using
    libCEED. */
class Integrator : public Operator
{
#ifdef MFEM_USE_CEED
protected:
   CeedBasis trial_basis, test_basis, mesh_basis;
   CeedElemRestriction trial_restr, test_restr, mesh_restr, qdata_restr;
   CeedQFunction apply_qfunc;
   CeedQFunctionContext apply_ctx;
   CeedVector node_coords, qdata;
   Coefficient *coeff;

public:
   Integrator()
      : Operator(),
        trial_basis(nullptr), test_basis(nullptr), mesh_basis(nullptr),
        trial_restr(nullptr), test_restr(nullptr), mesh_restr(nullptr),
        qdata_restr(nullptr),
        apply_qfunc(nullptr), apply_ctx(nullptr),
        node_coords(nullptr), qdata(nullptr), coeff(nullptr) {}

   /** @brief This method assembles the `Integrator` with the given
       `CeedOperatorInfo` @a info, an `mfem::FiniteElementSpace` @a fes, an
       `mfem::IntegrationRule` @a ir, and `mfem::Coefficient`,
       `mfem::VectorCoefficient`, or `mfem::MatrixCoefficient` @a Q.
       The `CeedOperatorInfo` type is expected to inherit from `OperatorInfo`,
       and contain a `Context` type relevant to the QFunctions.

       @param[in] info is the structure describing the CeedOperator to assemble.
       @param[in] fes is the finite element space.
       @param[in] ir is the integration rule for the operator.
       @param[in] Q is the coefficient from the `Integrator`.
       @param[in] use_bdr controls whether to construct the operator for the domain
                          or domain boundary.
       @param[in] use_mf controls whether to construct a matrix-free or partially
                         assembled operator. */
   template <typename CeedOperatorInfo, typename CoeffType>
   void Assemble(CeedOperatorInfo &info,
                 const mfem::FiniteElementSpace &fes,
                 const mfem::IntegrationRule &ir,
                 CoeffType *Q,
                 const bool use_bdr = false,
                 const bool use_mf = false)
   {
      Assemble(info, fes, ir, use_bdr ? fes.GetNBE() : fes.GetNE(),
               nullptr, Q, use_bdr, use_mf);
   }

   /** @brief This method assembles the `Integrator` with the given
       `CeedOperatorInfo` @a info, an `mfem::FiniteElementSpace` @a fes, an
       `mfem::IntegrationRule` @a ir, and `mfem::Coefficient`,
       `mfem::VectorCoefficient`, or `mfem::MatrixCoefficient` @a Q for the
       elements given by the indices @a indices.
       The `CeedOperatorInfo` type is expected to inherit from `OperatorInfo`,
       and contain a `Context` type relevant to the QFunctions.

       @param[in] info is the structure describing the CeedOperator to assemble.
       @param[in] fes is the finite element space.
       @param[in] ir is the integration rule for the operator.
       @param[in] nelem The number of elements.
       @param[in] indices The indices of the elements of same type in the
                          `FiniteElementSpace`. If `indices == nullptr`, assumes
                          that the `FiniteElementSpace` is not mixed.
       @param[in] Q is the coefficient from the `Integrator`.
       @param[in] use_bdr controls whether to construct the operator for the domain
                          or domain boundary.
       @param[in] use_mf controls whether to construct a matrix-free or partially
                         assembled operator. */
   template <typename CeedOperatorInfo, typename CoeffType>
   void Assemble(CeedOperatorInfo &info,
                 const mfem::FiniteElementSpace &fes,
                 const mfem::IntegrationRule &ir,
                 int nelem,
                 const int *indices,
                 CoeffType *Q,
                 const bool use_bdr = false,
                 const bool use_mf = false)
   {
      Assemble(info, fes, fes, ir, nelem, indices, Q, use_bdr, use_mf);
   }

   /** This method assembles the `Integrator` for mixed forms.

       @param[in] info the `CeedOperatorInfo` describing the `CeedOperator`,
                       the `CeedOperatorInfo` type is expected to inherit from
                       `OperatorInfo` and contain a `Context` type relevant to
                       the QFunctions.
       @param[in] trial_fes the trial `FiniteElementSpace` for the form,
       @param[in] test_fes the test `FiniteElementSpace` for the form,
       @param[in] ir the `IntegrationRule` for the numerical integration,
       @param[in] Q `Coefficient`, `VectorCoefficient`, or
                    `MatrixCoefficient`.
       @param[in] use_bdr controls whether to construct the operator for the domain
                          or domain boundary.
       @param[in] use_mf controls whether to construct a matrix-free or partially
                         assembled operator. */
   template <typename CeedOperatorInfo, typename CoeffType>
   void Assemble(CeedOperatorInfo &info,
                 const mfem::FiniteElementSpace &trial_fes,
                 const mfem::FiniteElementSpace &test_fes,
                 const mfem::IntegrationRule &ir,
                 CoeffType *Q,
                 const bool use_bdr = false,
                 const bool use_mf = false)
   {
      Assemble(info, trial_fes, test_fes, ir,
               use_bdr ? trial_fes.GetNBE() : trial_fes.GetNE(),
               nullptr, Q, use_bdr, use_mf);
   }

   /** This method assembles the `Integrator` for mixed forms on mixed meshes.

       @param[in] info the `CeedOperatorInfo` describing the `CeedOperator`,
                       the `CeedOperatorInfo` type is expected to inherit from
                       `OperatorInfo` and contain a `Context` type relevant to
                       the QFunctions.
       @param[in] trial_fes the trial `FiniteElementSpace` for the form,
       @param[in] test_fes the test `FiniteElementSpace` for the form,
       @param[in] ir the `IntegrationRule` for the numerical integration,
       @param[in] nelem The number of elements,
       @param[in] indices The indices of the elements of same type in the
                          `FiniteElementSpace`. If `indices == nullptr`, assumes
                          that the `FiniteElementSpace` is not mixed,
       @param[in] Q `Coefficient`, `VectorCoefficient`, or
                    `MatrixCoefficient`.
       @param[in] use_bdr controls whether to construct the operator for the domain
                          or domain boundary.
       @param[in] use_mf controls whether to construct a matrix-free or partially
                         assembled operator. */
   template <typename CeedOperatorInfo, typename CoeffType>
   void Assemble(CeedOperatorInfo &info,
                 const mfem::FiniteElementSpace &trial_fes,
                 const mfem::FiniteElementSpace &test_fes,
                 const mfem::IntegrationRule &ir,
                 int nelem,
                 const int *indices,
                 CoeffType *Q,
                 const bool use_bdr = false,
                 const bool use_mf = false)
   {
      Ceed ceed(internal::ceed);
      mfem::Mesh &mesh = *trial_fes.GetMesh();
      CeedInt dim = mesh.Dimension() - use_bdr;
      CeedInt space_dim = mesh.SpaceDimension();
      CeedInt trial_vdim = trial_fes.GetVDim();
      CeedInt test_vdim = test_fes.GetVDim();
      MFEM_VERIFY(!(!indices && mesh.GetNumGeometries(dim) > 1),
                  "Use ceed::MixedIntegrator on mixed meshes.");
      InitCoefficient(Q, mesh, ir, use_bdr, nelem, indices, coeff);

      if (&trial_fes == &test_fes)
      {
         InitBasisAndRestriction(trial_fes, ir, use_bdr, nelem, indices, ceed,
                                 &trial_basis, &trial_restr);
         CeedBasisReferenceCopy(trial_basis, &test_basis);
         CeedElemRestrictionReferenceCopy(trial_restr, &test_restr);
      }
      else
      {
         InitBasisAndRestriction(trial_fes, ir, use_bdr, nelem, indices, ceed,
                                 &trial_basis, &trial_restr);
         InitBasisAndRestriction(test_fes, ir, use_bdr, nelem, indices, ceed,
                                 &test_basis, &test_restr);
      }

      CeedInt trial_nqpts, test_nqpts;
      CeedBasisGetNumQuadraturePoints(trial_basis, &trial_nqpts);
      CeedBasisGetNumQuadraturePoints(test_basis, &test_nqpts);
      MFEM_VERIFY(trial_nqpts == test_nqpts,
                  "Trial and test basis must have the same number of quadrature"
                  " points.");
      const CeedInt nqpts = trial_nqpts;

      mesh.EnsureNodes();
      const mfem::FiniteElementSpace *mesh_fes = mesh.GetNodalFESpace();
      MFEM_VERIFY(mesh_fes, "The Mesh has no nodal FE space");
      InitBasisAndRestriction(*mesh_fes, ir, use_bdr, nelem, indices, ceed,
                              &mesh_basis, &mesh_restr);
      InitVector(*mesh.GetNodes(), node_coords);

      CeedQFunctionContextCreate(ceed, &apply_ctx);
      CeedQFunctionContextSetData(apply_ctx, CEED_MEM_HOST,
                                  CEED_COPY_VALUES,
                                  sizeof(info.ctx),
                                  &info.ctx);

      if (!use_mf)
      {
         const int qdatasize = info.qdatasize;
         InitStridedRestriction(*mesh_fes, nelem, nqpts, qdatasize,
                                CEED_STRIDES_BACKEND, ceed,
                                &qdata_restr);
         CeedVectorCreate(ceed, nelem * nqpts * qdatasize, &qdata);

         std::string build_func = coeff ? info.build_func_quad
                                  : info.build_func_const;
         CeedQFunctionUser build_qf = coeff ? info.build_qf_quad
                                      : info.build_qf_const;

         // Create the Q-function that builds the operator (i.e. computes its
         // quadrature data) and set its context data.
         CeedQFunction build_qfunc;
         std::string qf_file = GetCeedPath() + info.header;
         std::string qf = qf_file + build_func;
         CeedQFunctionCreateInterior(ceed, 1, build_qf, qf.c_str(),
                                     &build_qfunc);
         if (coeff)
         {
            CeedQFunctionAddInput(build_qfunc, "coeff", coeff->ncomp, coeff->emode);
         }
         CeedQFunctionAddInput(build_qfunc, "dx", dim * space_dim, CEED_EVAL_GRAD);
         CeedQFunctionAddInput(build_qfunc, "weights", 1, CEED_EVAL_WEIGHT);
         CeedQFunctionAddOutput(build_qfunc, "qdata", qdatasize, CEED_EVAL_NONE);
         CeedQFunctionSetContext(build_qfunc, apply_ctx);

         // Create the operator that builds the quadrature data for the operator.
         CeedOperator build_oper;
         CeedOperatorCreate(ceed, build_qfunc, NULL, NULL, &build_oper);
         if (GridCoefficient *grid_coeff = dynamic_cast<GridCoefficient *>(coeff))
         {
            InitBasisAndRestriction(*grid_coeff->gf.FESpace(), ir,
                                    use_bdr, nelem, indices, ceed,
                                    &grid_coeff->basis, &grid_coeff->restr);
            CeedOperatorSetField(build_oper, "coeff", grid_coeff->restr,
                                 grid_coeff->basis, grid_coeff->coeff_vector);
         }
         else if (QuadCoefficient *quad_coeff = dynamic_cast<QuadCoefficient *>(coeff))
         {
            const int ncomp = quad_coeff->ncomp;
            CeedInt strides[3] = {ncomp, 1, ncomp * nqpts};
            InitStridedRestriction(*mesh_fes, nelem, nqpts, ncomp,
                                   strides, ceed,
                                   &quad_coeff->restr);
            CeedOperatorSetField(build_oper, "coeff", quad_coeff->restr,
                                 CEED_BASIS_COLLOCATED, quad_coeff->coeff_vector);
         }
         CeedOperatorSetField(build_oper, "dx", mesh_restr,
                              mesh_basis, CEED_VECTOR_ACTIVE);
         CeedOperatorSetField(build_oper, "weights", CEED_ELEMRESTRICTION_NONE,
                              mesh_basis, CEED_VECTOR_NONE);
         CeedOperatorSetField(build_oper, "qdata", qdata_restr,
                              CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

         // Compute the quadrature data for the operator.
         CeedOperatorApply(build_oper, node_coords, qdata, CEED_REQUEST_IMMEDIATE);

         CeedOperatorDestroy(&build_oper);
         CeedQFunctionDestroy(&build_qfunc);
      }

      std::string apply_func = !use_mf ? info.apply_func :
                               (coeff ? info.apply_func_mf_quad
                                : info.apply_func_mf_const);
      CeedQFunctionUser apply_qf = !use_mf ? info.apply_qf :
                                   (coeff ? info.apply_qf_mf_quad
                                    : info.apply_qf_mf_const);

      // Create the Q-function that defines the action of the operator.
      std::string qf_file = GetCeedPath() + info.header;
      std::string qf = qf_file + apply_func;
      CeedQFunctionCreateInterior(ceed, 1, apply_qf, qf.c_str(),
                                  &apply_qfunc);
      // input
      switch (info.trial_op)
      {
         case EvalMode::None:
            CeedQFunctionAddInput(apply_qfunc, "u", trial_vdim, CEED_EVAL_NONE);
            break;
         case EvalMode::Interp:
            CeedQFunctionAddInput(apply_qfunc, "u", trial_vdim, CEED_EVAL_INTERP);
            break;
         case EvalMode::Grad:
            CeedQFunctionAddInput(apply_qfunc, "gu", trial_vdim * dim, CEED_EVAL_GRAD);
            break;
         case EvalMode::InterpAndGrad:
            CeedQFunctionAddInput(apply_qfunc, "u", trial_vdim, CEED_EVAL_INTERP);
            CeedQFunctionAddInput(apply_qfunc, "gu", trial_vdim * dim, CEED_EVAL_GRAD);
            break;
      }
      if (use_mf)
      {
         if (coeff)
         {
            // coefficient
            CeedQFunctionAddInput(apply_qfunc, "coeff", coeff->ncomp, coeff->emode);
         }
         CeedQFunctionAddInput(apply_qfunc, "dx", dim * space_dim, CEED_EVAL_GRAD);
         CeedQFunctionAddInput(apply_qfunc, "weights", 1, CEED_EVAL_WEIGHT);
      }
      else
      {
         // qdata
         CeedQFunctionAddInput(apply_qfunc, "qdata", info.qdatasize, CEED_EVAL_NONE);
      }
      // output
      switch (info.test_op)
      {
         case EvalMode::None:
            CeedQFunctionAddOutput(apply_qfunc, "v", test_vdim, CEED_EVAL_NONE);
            break;
         case EvalMode::Interp:
            CeedQFunctionAddOutput(apply_qfunc, "v", test_vdim, CEED_EVAL_INTERP);
            break;
         case EvalMode::Grad:
            CeedQFunctionAddOutput(apply_qfunc, "gv", test_vdim * dim, CEED_EVAL_GRAD);
            break;
         case EvalMode::InterpAndGrad:
            CeedQFunctionAddOutput(apply_qfunc, "v", test_vdim, CEED_EVAL_INTERP);
            CeedQFunctionAddOutput(apply_qfunc, "gv", test_vdim * dim, CEED_EVAL_GRAD);
            break;
      }
      CeedQFunctionSetContext(apply_qfunc, apply_ctx);

      // Create the operator.
      CeedOperatorCreate(ceed, apply_qfunc, NULL, NULL, &oper);
      // input
      switch (info.trial_op)
      {
         case EvalMode::None:
            CeedOperatorSetField(oper, "u", trial_restr,
                                 CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
            break;
         case EvalMode::Interp:
            CeedOperatorSetField(oper, "u", trial_restr, trial_basis, CEED_VECTOR_ACTIVE);
            break;
         case EvalMode::Grad:
            CeedOperatorSetField(oper, "gu", trial_restr, trial_basis, CEED_VECTOR_ACTIVE);
            break;
         case EvalMode::InterpAndGrad:
            CeedOperatorSetField(oper, "u", trial_restr, trial_basis, CEED_VECTOR_ACTIVE);
            CeedOperatorSetField(oper, "gu", trial_restr, trial_basis, CEED_VECTOR_ACTIVE);
            break;
      }
      if (use_mf)
      {
         // coefficient
         if (GridCoefficient *grid_coeff = dynamic_cast<GridCoefficient *>(coeff))
         {
            InitBasisAndRestriction(*grid_coeff->gf.FESpace(), ir,
                                    use_bdr, nelem, indices, ceed,
                                    &grid_coeff->basis, &grid_coeff->restr);
            CeedOperatorSetField(oper, "coeff", grid_coeff->restr,
                                 grid_coeff->basis, grid_coeff->coeff_vector);
         }
         else if (QuadCoefficient *quad_coeff = dynamic_cast<QuadCoefficient *>(coeff))
         {
            const int ncomp = quad_coeff->ncomp;
            CeedInt strides[3] = {ncomp, 1, ncomp * nqpts};
            InitStridedRestriction(*mesh_fes, nelem, nqpts, ncomp,
                                   strides, ceed,
                                   &quad_coeff->restr);
            CeedOperatorSetField(oper, "coeff", quad_coeff->restr,
                                 CEED_BASIS_COLLOCATED, quad_coeff->coeff_vector);
         }
         CeedOperatorSetField(oper, "dx", mesh_restr, mesh_basis, node_coords);
         CeedOperatorSetField(oper, "weights", CEED_ELEMRESTRICTION_NONE,
                              mesh_basis, CEED_VECTOR_NONE);
      }
      else
      {
         // qdata
         CeedOperatorSetField(oper, "qdata", qdata_restr, CEED_BASIS_COLLOCATED, qdata);
      }
      // output
      switch (info.test_op)
      {
         case EvalMode::None:
            CeedOperatorSetField(oper, "v", test_restr,
                                 CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
            break;
         case EvalMode::Interp:
            CeedOperatorSetField(oper, "v", test_restr, test_basis, CEED_VECTOR_ACTIVE);
            break;
         case EvalMode::Grad:
            CeedOperatorSetField(oper, "gv", test_restr, test_basis, CEED_VECTOR_ACTIVE);
            break;
         case EvalMode::InterpAndGrad:
            CeedOperatorSetField(oper, "v", test_restr, test_basis, CEED_VECTOR_ACTIVE);
            CeedOperatorSetField(oper, "gv", test_restr, test_basis, CEED_VECTOR_ACTIVE);
            break;
      }

      CeedVectorCreate(ceed, trial_vdim * trial_fes.GetNDofs(), &u);
      CeedVectorCreate(ceed, test_vdim * test_fes.GetNDofs(), &v);
   }

   virtual ~Integrator()
   {
      // All basis and restriction objects are destroyed by fes destructor
      CeedQFunctionDestroy(&apply_qfunc);
      CeedQFunctionContextDestroy(&apply_ctx);
      CeedVectorDestroy(&node_coords);
      CeedVectorDestroy(&qdata);
      delete coeff;
   }
#endif
};

} // namespace ceed

} // namespace mfem

#endif // MFEM_LIBCEED_INTEG
