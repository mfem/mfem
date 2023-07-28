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

#ifndef MFEM_LIBCEED_INTEGRATOR
#define MFEM_LIBCEED_INTEGRATOR

#include "../../fespace.hpp"
#include "../../gridfunc.hpp"
#include "basis.hpp"
#include "coefficient.hpp"
#include "operator.hpp"
#include "restriction.hpp"
#include "util.hpp"
#include "ceed.hpp"
#include <string>

namespace mfem
{

namespace ceed
{

/** The different evaluation modes available for PA and MF CeedIntegrator. */
enum class EvalMode { None, Interp, Grad, Div, Curl,
                      InterpAndGrad, InterpAndDiv, InterpAndCurl
                    };

#ifdef MFEM_USE_CEED
/** This structure is a template interface for the Assemble methods of
    PAIntegrator and MFIntegrator. See ceed/mass.cpp for an example. */
struct OperatorInfo
{
   /** The path to the QFunction header. */
   const char *header;
   /** The name of the QFunction to build a partially assembled CeedOperator. */
   const char *build_func;
   /** The QFunction to build a partially assembled CeedOperator. */
   CeedQFunctionUser build_qf;
   /** The name of the QFunction to apply a partially assembled CeedOperator. */
   const char *apply_func;
   /** The QFunction to apply a partially assembled CeedOperator. */
   CeedQFunctionUser apply_qf;
   /** The name of the QFunction to apply a matrix-free CeedOperator. */
   const char *apply_func_mf;
   /** The QFunction to apply a matrix-free CeedOperator. */
   CeedQFunctionUser apply_qf_mf;
   /** The EvalMode on the trial basis functions. */
   EvalMode trial_op;
   /** The EvalMode on the test basis functions. */
   EvalMode test_op;
   /** The size of the data at each quadrature point. */
   int qdatasize;
};
#endif

/** This class represents a matrix-free or partially assembled bilinear,
    mixed bilinear, or nonlinear form operator using libCEED. */
class Integrator
{
#ifdef MFEM_USE_CEED
protected:
   CeedOperator oper;
   CeedBasis trial_basis, test_basis, mesh_basis;
   CeedElemRestriction trial_restr, test_restr, mesh_restr, qdata_restr;
   CeedQFunction apply_qfunc;
   CeedQFunctionContext apply_ctx;
   CeedVector node_coords, qdata;
   Coefficient *coeff1, *coeff2;

public:
   Integrator()
      : oper(nullptr),
        trial_basis(nullptr), test_basis(nullptr), mesh_basis(nullptr),
        trial_restr(nullptr), test_restr(nullptr), mesh_restr(nullptr),
        qdata_restr(nullptr),
        apply_qfunc(nullptr), apply_ctx(nullptr),
        node_coords(nullptr), qdata(nullptr),
        coeff1(nullptr), coeff2(nullptr) {}

   CeedOperator GetOperator() const { return oper; }
   CeedOperator GetTransposeOperator() const { return nullptr; }

   template <typename CeedOperatorInfo, typename CoeffType1, typename CoeffType2>
   void Assemble(Ceed ceed,
                 CeedOperatorInfo &info,
                 const mfem::FiniteElementSpace &trial_fes,
                 const mfem::FiniteElementSpace &test_fes,
                 const mfem::IntegrationRule &ir,
                 CoeffType1 *Q1,
                 CoeffType2 *Q2,
                 const bool use_bdr = false,
                 const bool use_mf = false)
   {
      Assemble(ceed, info, trial_fes, test_fes, ir,
               use_bdr ? trial_fes.GetNBE() : trial_fes.GetNE(),
               nullptr, Q1, Q2, use_bdr, use_mf);
   }

   /** This method assembles the `Integrator` for mixed forms on mixed meshes.

       @param[in] ceed The Ceed object.
       @param[in] info The `CeedOperatorInfo` describing the `CeedOperator`,
                       the `CeedOperatorInfo` type is expected to inherit from
                       `OperatorInfo` and contain a `Context` type relevant to
                       the QFunctions.
       @param[in] trial_fes The trial `FiniteElementSpace` for the form.
       @param[in] test_fes The test `FiniteElementSpace` for the form.
       @param[in] ir The `IntegrationRule` for the numerical integration.
       @param[in] nelem The number of elements.
       @param[in] indices The indices of the elements of same type in the
                          `FiniteElementSpace`. If `indices == nullptr`, assumes
                          that the `FiniteElementSpace` is not mixed.
       @param[in] Q1 `Coefficient`, `VectorCoefficient`, or
                     `MatrixCoefficient`.
       @param[in] Q2 A second `Coefficient`, `VectorCoefficient`, or
                     `MatrixCoefficient` for certain integrators.
       @param[in] use_bdr Controls whether to construct the operator for the
                          domain or domain boundary.
       @param[in] use_mf Controls whether to construct a matrix-free or
                         partially assembled operator. */
   template <typename CeedOperatorInfo, typename CoeffType1, typename CoeffType2>
   void Assemble(Ceed ceed,
                 CeedOperatorInfo &info,
                 const mfem::FiniteElementSpace &trial_fes,
                 const mfem::FiniteElementSpace &test_fes,
                 const mfem::IntegrationRule &ir,
                 int nelem,
                 const int *indices,
                 CoeffType1 *Q1,
                 CoeffType2 *Q2,
                 const bool use_bdr = false,
                 const bool use_mf = false)
   {
      mfem::Mesh &mesh = *trial_fes.GetMesh();
      CeedInt dim = mesh.Dimension() - use_bdr;
      CeedInt space_dim = mesh.SpaceDimension();
      CeedInt curl_dim = (dim < 3) ? 1 : dim;
      CeedInt trial_vdim = trial_fes.GetVDim();
      CeedInt test_vdim = test_fes.GetVDim();
      bool trial_vectorfe =
         (trial_fes.FEColl()->GetRangeType(dim) == mfem::FiniteElement::VECTOR);
      bool test_vectorfe =
         (test_fes.FEColl()->GetRangeType(dim) == mfem::FiniteElement::VECTOR);
      MFEM_VERIFY(!(!indices && mesh.GetNumGeometries(dim) > 1),
                  "Use ceed::MixedOperator<ceed::Integrator> on mixed meshes.");
      if (Q1) { InitCoefficient(*Q1, mesh, ir, use_bdr, nelem, indices, ceed, coeff1); }
      if (Q2) { InitCoefficient(*Q2, mesh, ir, use_bdr, nelem, indices, ceed, coeff2); }

      if (&trial_fes == &test_fes)
      {
         InitBasis(trial_fes, ir, use_bdr, indices, ceed,
                   &trial_basis);
         InitRestriction(trial_fes, use_bdr, nelem, indices, ceed,
                         &trial_restr);
         CeedBasisReferenceCopy(trial_basis, &test_basis);
         CeedElemRestrictionReferenceCopy(trial_restr, &test_restr);
      }
      else
      {
         InitBasis(trial_fes, ir, use_bdr, indices, ceed,
                   &trial_basis);
         InitBasis(test_fes, ir, use_bdr, indices, ceed,
                   &test_basis);
         InitRestriction(trial_fes, use_bdr, nelem, indices, ceed,
                         &trial_restr);
         InitRestriction(test_fes, use_bdr, nelem, indices, ceed,
                         &test_restr);
      }

      CeedInt trial_nqpts, test_nqpts;
      CeedBasisGetNumQuadraturePoints(trial_basis, &trial_nqpts);
      CeedBasisGetNumQuadraturePoints(test_basis, &test_nqpts);
      MFEM_VERIFY(trial_nqpts == test_nqpts,
                  "Trial and test basis must have the same number of quadrature"
                  " points.");
      const CeedInt nqpts = trial_nqpts;
#ifdef MFEM_USE_OPENMP
      #pragma omp critical (MeshNodes)
#endif
      {
         mesh.EnsureNodes();
         InitVector(*mesh.GetNodes(), ceed, node_coords);
      }
      const mfem::FiniteElementSpace *mesh_fes = mesh.GetNodalFESpace();
      MFEM_VERIFY(mesh_fes, "The mesh has no nodal FE space.");
      InitBasis(*mesh_fes, ir, use_bdr, indices, ceed, &mesh_basis);
      InitRestriction(*mesh_fes, use_bdr, nelem, indices, ceed, &mesh_restr);

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

         // Create the QFunction that builds the operator (i.e. computes its
         // quadrature data) and set its context data.
         CeedQFunction build_qfunc;
         std::string qf = GetCeedPath() + info.header + info.build_func;
         CeedQFunctionCreateInterior(ceed, 1, info.build_qf, qf.c_str(),
                                     &build_qfunc);
         if (coeff1)
         {
            CeedQFunctionAddInput(build_qfunc, "coeff1", coeff1->ncomp, coeff1->emode);
         }
         if (coeff2)
         {
            CeedQFunctionAddInput(build_qfunc, "coeff2", coeff2->ncomp, coeff2->emode);
         }
         CeedQFunctionAddInput(build_qfunc, "dx", dim * space_dim, CEED_EVAL_GRAD);
         CeedQFunctionAddInput(build_qfunc, "weights", 1, CEED_EVAL_WEIGHT);
         CeedQFunctionAddOutput(build_qfunc, "qdata", qdatasize, CEED_EVAL_NONE);
         CeedQFunctionSetContext(build_qfunc, apply_ctx);

         // Create the operator that builds the quadrature data for the operator.
         CeedOperator build_oper;
         CeedOperatorCreate(ceed, build_qfunc, NULL, NULL, &build_oper);
         if (coeff1)
         {
            if (GridCoefficient *grid_coeff = dynamic_cast<GridCoefficient *>(coeff1))
            {
               const mfem::FiniteElementSpace *coeff_fes = grid_coeff->gf.FESpace();
               InitBasis(*coeff_fes, ir, use_bdr, indices, ceed,
                         &grid_coeff->basis);
               InitRestriction(*coeff_fes, use_bdr, nelem, indices, ceed,
                               &grid_coeff->restr);
               CeedOperatorSetField(build_oper, "coeff1", grid_coeff->restr,
                                    grid_coeff->basis, grid_coeff->coeff_vector);
            }
            else if (QuadCoefficient *quad_coeff = dynamic_cast<QuadCoefficient *>(coeff1))
            {
               const int ncomp = quad_coeff->ncomp;
               CeedInt strides[3] = {ncomp, 1, ncomp * nqpts};
               InitStridedRestriction(*mesh_fes, nelem, nqpts, ncomp, strides, ceed,
                                      &quad_coeff->restr);
               CeedOperatorSetField(build_oper, "coeff1", quad_coeff->restr,
                                    CEED_BASIS_NONE, quad_coeff->coeff_vector);
            }
            else
            {
               MFEM_ABORT("Unknown coefficient type in QFunction setup!");
            }
         }
         if (coeff2)
         {
            if (GridCoefficient *grid_coeff = dynamic_cast<GridCoefficient *>(coeff2))
            {
               const mfem::FiniteElementSpace *coeff_fes = grid_coeff->gf.FESpace();
               InitBasis(*coeff_fes, ir, use_bdr, indices, ceed,
                         &grid_coeff->basis);
               InitRestriction(*coeff_fes, use_bdr, nelem, indices, ceed,
                               &grid_coeff->restr);
               CeedOperatorSetField(build_oper, "coeff2", grid_coeff->restr,
                                    grid_coeff->basis, grid_coeff->coeff_vector);
            }
            else if (QuadCoefficient *quad_coeff = dynamic_cast<QuadCoefficient *>(coeff2))
            {
               const int ncomp = quad_coeff->ncomp;
               CeedInt strides[3] = {ncomp, 1, ncomp * nqpts};
               InitStridedRestriction(*mesh_fes, nelem, nqpts, ncomp, strides, ceed,
                                      &quad_coeff->restr);
               CeedOperatorSetField(build_oper, "coeff2", quad_coeff->restr,
                                    CEED_BASIS_NONE, quad_coeff->coeff_vector);
            }
            else
            {
               MFEM_ABORT("Unknown coefficient type in QFunction setup!");
            }
         }
         CeedOperatorSetField(build_oper, "dx", mesh_restr,
                              mesh_basis, CEED_VECTOR_ACTIVE);
         CeedOperatorSetField(build_oper, "weights", CEED_ELEMRESTRICTION_NONE,
                              mesh_basis, CEED_VECTOR_NONE);
         CeedOperatorSetField(build_oper, "qdata", qdata_restr,
                              CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);
         CeedOperatorCheckReady(build_oper);

         // Compute the quadrature data for the operator.
         CeedOperatorApply(build_oper, node_coords, qdata, CEED_REQUEST_IMMEDIATE);

         CeedOperatorDestroy(&build_oper);
         CeedQFunctionDestroy(&build_qfunc);

         CeedVectorDestroy(&node_coords);
         node_coords = nullptr;
         delete coeff1;
         delete coeff2;
         coeff1 = coeff2 = nullptr;
      }

      // Create the QFunction that defines the action of the operator.
      std::string qf = GetCeedPath() + info.header + info.apply_func;
      CeedQFunctionCreateInterior(ceed, 1, info.apply_qf, qf.c_str(),
                                  &apply_qfunc);
      switch (info.trial_op)
      {
         case EvalMode::None:
            CeedQFunctionAddInput(apply_qfunc, "u", trial_vdim, CEED_EVAL_NONE);
            break;
         case EvalMode::Interp:
            CeedQFunctionAddInput(apply_qfunc, "u", trial_vdim * (trial_vectorfe ? dim : 1),
                                  CEED_EVAL_INTERP);
            break;
         case EvalMode::Grad:
            CeedQFunctionAddInput(apply_qfunc, "gu", trial_vdim * dim, CEED_EVAL_GRAD);
            break;
         case EvalMode::Div:
            CeedQFunctionAddInput(apply_qfunc, "du", trial_vdim, CEED_EVAL_DIV);
            break;
         case EvalMode::Curl:
            CeedQFunctionAddInput(apply_qfunc, "cu", trial_vdim * curl_dim, CEED_EVAL_CURL);
            break;
         case EvalMode::InterpAndGrad:
            MFEM_VERIFY(!trial_vectorfe,
                        "EvalMode::InterpAndGrad is not intended for vector FE.");
            CeedQFunctionAddInput(apply_qfunc, "u", trial_vdim, CEED_EVAL_INTERP);
            CeedQFunctionAddInput(apply_qfunc, "gu", trial_vdim * dim, CEED_EVAL_GRAD);
            break;
         case EvalMode::InterpAndDiv:
            MFEM_VERIFY(trial_vectorfe,
                        "EvalMode::InterpAndDiv is only intended for vector FE.");
            CeedQFunctionAddInput(apply_qfunc, "u", trial_vdim * dim, CEED_EVAL_INTERP);
            CeedQFunctionAddInput(apply_qfunc, "du", trial_vdim, CEED_EVAL_DIV);
            break;
         case EvalMode::InterpAndCurl:
            MFEM_VERIFY(trial_vectorfe,
                        "EvalMode::InterpAndCurl is only intended for vector FE.");
            CeedQFunctionAddInput(apply_qfunc, "u", trial_vdim * dim, CEED_EVAL_INTERP);
            CeedQFunctionAddInput(apply_qfunc, "cu", trial_vdim * curl_dim, CEED_EVAL_CURL);
            break;
      }
      if (use_mf)
      {
         if (coeff1)
         {
            CeedQFunctionAddInput(apply_qfunc, "coeff1", coeff1->ncomp, coeff1->emode);
         }
         if (coeff2)
         {
            CeedQFunctionAddInput(apply_qfunc, "coeff2", coeff2->ncomp, coeff2->emode);
         }
         CeedQFunctionAddInput(apply_qfunc, "dx", dim * space_dim, CEED_EVAL_GRAD);
         CeedQFunctionAddInput(apply_qfunc, "weights", 1, CEED_EVAL_WEIGHT);
      }
      else
      {
         CeedQFunctionAddInput(apply_qfunc, "qdata", info.qdatasize, CEED_EVAL_NONE);
      }
      switch (info.test_op)
      {
         case EvalMode::None:
            CeedQFunctionAddOutput(apply_qfunc, "v", test_vdim, CEED_EVAL_NONE);
            break;
         case EvalMode::Interp:
            CeedQFunctionAddOutput(apply_qfunc, "v", test_vdim * (test_vectorfe ? dim : 1),
                                   CEED_EVAL_INTERP);
            break;
         case EvalMode::Grad:
            CeedQFunctionAddOutput(apply_qfunc, "gv", test_vdim * dim, CEED_EVAL_GRAD);
            break;
         case EvalMode::Div:
            CeedQFunctionAddOutput(apply_qfunc, "dv", test_vdim, CEED_EVAL_DIV);
            break;
         case EvalMode::Curl:
            CeedQFunctionAddOutput(apply_qfunc, "cv", test_vdim * curl_dim, CEED_EVAL_CURL);
            break;
         case EvalMode::InterpAndGrad:
            MFEM_VERIFY(!test_vectorfe,
                        "EvalMode::InterpAndGrad is not intended for vector FE.");
            CeedQFunctionAddOutput(apply_qfunc, "v", test_vdim, CEED_EVAL_INTERP);
            CeedQFunctionAddOutput(apply_qfunc, "gv", test_vdim * dim, CEED_EVAL_GRAD);
            break;
         case EvalMode::InterpAndDiv:
            MFEM_VERIFY(test_vectorfe,
                        "EvalMode::InterpAndDiv is only intended for vector FE.");
            CeedQFunctionAddOutput(apply_qfunc, "v", test_vdim * dim, CEED_EVAL_INTERP);
            CeedQFunctionAddOutput(apply_qfunc, "dv", test_vdim, CEED_EVAL_DIV);
            break;
         case EvalMode::InterpAndCurl:
            MFEM_VERIFY(test_vectorfe,
                        "EvalMode::InterpAndCurl is only intended for vector FE.");
            CeedQFunctionAddOutput(apply_qfunc, "v", test_vdim * dim, CEED_EVAL_INTERP);
            CeedQFunctionAddOutput(apply_qfunc, "cv", test_vdim * curl_dim, CEED_EVAL_CURL);
            break;
      }
      CeedQFunctionSetContext(apply_qfunc, apply_ctx);

      // Create the operator.
      CeedOperatorCreate(ceed, apply_qfunc, NULL, NULL, &oper);
      switch (info.trial_op)
      {
         case EvalMode::None:
            CeedOperatorSetField(oper, "u", trial_restr,
                                 CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);
            break;
         case EvalMode::Interp:
            CeedOperatorSetField(oper, "u", trial_restr, trial_basis, CEED_VECTOR_ACTIVE);
            break;
         case EvalMode::Grad:
            CeedOperatorSetField(oper, "gu", trial_restr, trial_basis, CEED_VECTOR_ACTIVE);
            break;
         case EvalMode::Div:
            CeedOperatorSetField(oper, "du", trial_restr, trial_basis, CEED_VECTOR_ACTIVE);
            break;
         case EvalMode::Curl:
            CeedOperatorSetField(oper, "cu", trial_restr, trial_basis, CEED_VECTOR_ACTIVE);
            break;
         case EvalMode::InterpAndGrad:
            CeedOperatorSetField(oper, "u", trial_restr, trial_basis, CEED_VECTOR_ACTIVE);
            CeedOperatorSetField(oper, "gu", trial_restr, trial_basis, CEED_VECTOR_ACTIVE);
            break;
         case EvalMode::InterpAndDiv:
            CeedOperatorSetField(oper, "u", trial_restr, trial_basis, CEED_VECTOR_ACTIVE);
            CeedOperatorSetField(oper, "du", trial_restr, trial_basis, CEED_VECTOR_ACTIVE);
            break;
         case EvalMode::InterpAndCurl:
            CeedOperatorSetField(oper, "u", trial_restr, trial_basis, CEED_VECTOR_ACTIVE);
            CeedOperatorSetField(oper, "cu", trial_restr, trial_basis, CEED_VECTOR_ACTIVE);
            break;
      }
      if (use_mf)
      {
         if (coeff1)
         {
            if (GridCoefficient *grid_coeff = dynamic_cast<GridCoefficient *>(coeff1))
            {
               const mfem::FiniteElementSpace *coeff_fes = grid_coeff->gf.FESpace();
               InitBasis(*coeff_fes, ir, use_bdr, indices, ceed,
                         &grid_coeff->basis);
               InitRestriction(*coeff_fes, use_bdr, nelem, indices, ceed,
                               &grid_coeff->restr);
               CeedOperatorSetField(oper, "coeff1", grid_coeff->restr,
                                    grid_coeff->basis, grid_coeff->coeff_vector);
            }
            else if (QuadCoefficient *quad_coeff = dynamic_cast<QuadCoefficient *>(coeff1))
            {
               const int ncomp = quad_coeff->ncomp;
               CeedInt strides[3] = {ncomp, 1, ncomp * nqpts};
               InitStridedRestriction(*mesh_fes, nelem, nqpts, ncomp, strides, ceed,
                                      &quad_coeff->restr);
               CeedOperatorSetField(oper, "coeff1", quad_coeff->restr,
                                    CEED_BASIS_NONE, quad_coeff->coeff_vector);
            }
            else
            {
               MFEM_ABORT("Unknown coefficient type in QFunction setup!");
            }
         }
         if (coeff2)
         {
            if (GridCoefficient *grid_coeff = dynamic_cast<GridCoefficient *>(coeff2))
            {
               const mfem::FiniteElementSpace *coeff_fes = grid_coeff->gf.FESpace();
               InitBasis(*coeff_fes, ir, use_bdr, indices, ceed,
                         &grid_coeff->basis);
               InitRestriction(*coeff_fes, use_bdr, nelem, indices, ceed,
                               &grid_coeff->restr);
               CeedOperatorSetField(oper, "coeff2", grid_coeff->restr,
                                    grid_coeff->basis, grid_coeff->coeff_vector);
            }
            else if (QuadCoefficient *quad_coeff = dynamic_cast<QuadCoefficient *>(coeff2))
            {
               const int ncomp = quad_coeff->ncomp;
               CeedInt strides[3] = {ncomp, 1, ncomp * nqpts};
               InitStridedRestriction(*mesh_fes, nelem, nqpts, ncomp, strides, ceed,
                                      &quad_coeff->restr);
               CeedOperatorSetField(oper, "coeff2", quad_coeff->restr,
                                    CEED_BASIS_NONE, quad_coeff->coeff_vector);
            }
            else
            {
               MFEM_ABORT("Unknown coefficient type in QFunction setup!");
            }
         }
         CeedOperatorSetField(oper, "dx", mesh_restr, mesh_basis, node_coords);
         CeedOperatorSetField(oper, "weights", CEED_ELEMRESTRICTION_NONE,
                              mesh_basis, CEED_VECTOR_NONE);
      }
      else
      {
         CeedOperatorSetField(oper, "qdata", qdata_restr, CEED_BASIS_NONE, qdata);
      }
      switch (info.test_op)
      {
         case EvalMode::None:
            CeedOperatorSetField(oper, "v", test_restr,
                                 CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);
            break;
         case EvalMode::Interp:
            CeedOperatorSetField(oper, "v", test_restr, test_basis, CEED_VECTOR_ACTIVE);
            break;
         case EvalMode::Grad:
            CeedOperatorSetField(oper, "gv", test_restr, test_basis, CEED_VECTOR_ACTIVE);
            break;
         case EvalMode::Div:
            CeedOperatorSetField(oper, "dv", test_restr, test_basis, CEED_VECTOR_ACTIVE);
            break;
         case EvalMode::Curl:
            CeedOperatorSetField(oper, "cv", test_restr, test_basis, CEED_VECTOR_ACTIVE);
            break;
         case EvalMode::InterpAndGrad:
            CeedOperatorSetField(oper, "v", test_restr, test_basis, CEED_VECTOR_ACTIVE);
            CeedOperatorSetField(oper, "gv", test_restr, test_basis, CEED_VECTOR_ACTIVE);
            break;
         case EvalMode::InterpAndDiv:
            CeedOperatorSetField(oper, "v", test_restr, test_basis, CEED_VECTOR_ACTIVE);
            CeedOperatorSetField(oper, "dv", test_restr, test_basis, CEED_VECTOR_ACTIVE);
            break;
         case EvalMode::InterpAndCurl:
            CeedOperatorSetField(oper, "v", test_restr, test_basis, CEED_VECTOR_ACTIVE);
            CeedOperatorSetField(oper, "cv", test_restr, test_basis, CEED_VECTOR_ACTIVE);
            break;
      }
      CeedOperatorCheckReady(oper);
   }

   ~Integrator()
   {
      // All basis and restriction objects are destroyed by fes destructor
      CeedOperatorDestroy(&oper);
      CeedQFunctionDestroy(&apply_qfunc);
      CeedQFunctionContextDestroy(&apply_ctx);
      CeedVectorDestroy(&node_coords);
      CeedVectorDestroy(&qdata);
      delete coeff1;
      delete coeff2;
   }
#endif
};

/** This class represents a matrix-free or partially assembled discrete linear
    operator using libCEED. */
class Interpolator
{
#ifdef MFEM_USE_CEED
protected:
   CeedOperator oper, oper_t;
   CeedBasis basis_ctof;
   CeedElemRestriction trial_restr, test_restr;
   CeedQFunction apply_qfunc, apply_qfunc_t;

public:
   Interpolator()
      : oper(nullptr), oper_t(nullptr),
        basis_ctof(nullptr),
        trial_restr(nullptr), test_restr(nullptr),
        apply_qfunc(nullptr), apply_qfunc_t(nullptr) {}

   CeedOperator GetOperator() const { return oper; }
   CeedOperator GetTransposeOperator() const { return oper_t; }

   template <typename CeedOperatorInfo, typename CoeffType1, typename CoeffType2>
   void Assemble(Ceed ceed,
                 CeedOperatorInfo &info,
                 const mfem::FiniteElementSpace &trial_fes,
                 const mfem::FiniteElementSpace &test_fes,
                 const mfem::IntegrationRule &ir,
                 CoeffType1 *Q1,
                 CoeffType2 *Q2,
                 const bool use_bdr = false,
                 const bool use_mf = false)
   {
      Assemble(ceed, info, trial_fes, test_fes, ir,
               use_bdr ? trial_fes.GetNBE() : trial_fes.GetNE(),
               nullptr, Q1, Q2, use_bdr, use_mf);
   }

   /** This method assembles the `Interpolator` on mixed meshes. Its signature
       matches that for `Integrator`.

       @param[in] ceed The Ceed object.
       @param[in] info The `CeedOperatorInfo` describing the `CeedOperator`,
                       the `CeedOperatorInfo` type is expected to inherit from
                       `OperatorInfo` and contain a `Context` type relevant to
                       the QFunctions.
       @param[in] trial_fes The trial `FiniteElementSpace` for the form.
       @param[in] test_fes The test `FiniteElementSpace` for the form.
       @param[in] ir Not supported by `Interpolator`.
       @param[in] nelem The number of elements.
       @param[in] indices The indices of the elements of same type in the
                          `FiniteElementSpace`. If `indices == nullptr`, assumes
                          that the `FiniteElementSpace` is not mixed.
       @param[in] Q1 Not supported by `Interpolator`.
       @param[in] Q2 Not supported by `Interpolator`.
       @param[in] use_bdr Not supported by `Interpolator`.
       @param[in] use_mf Controls whether to construct a matrix-free or
                         partially assembled operator. */
   template <typename CeedOperatorInfo, typename CoeffType1, typename CoeffType2>
   void Assemble(Ceed ceed,
                 CeedOperatorInfo &info,
                 const mfem::FiniteElementSpace &trial_fes,
                 const mfem::FiniteElementSpace &test_fes,
                 const mfem::IntegrationRule &ir,
                 int nelem,
                 const int *indices,
                 CoeffType1 *Q1,
                 CoeffType2 *Q2,
                 const bool use_bdr = false,
                 const bool use_mf = false)
   {
      CeedInt trial_vdim = trial_fes.GetVDim();
      CeedInt test_vdim = test_fes.GetVDim();
      MFEM_VERIFY(!Q1 && !Q2, "ceed:Interpolator does not support coefficients.");
      MFEM_VERIFY(!use_bdr,
                  "ceed:Interpolator does not support boundary interpolators.");
      MFEM_VERIFY(trial_vdim == 1 && test_vdim == 1,
                  "ceed:Interpolator does not support spaces with vdim > 1.");

      InitInterpolatorBasis(trial_fes, test_fes, indices, ceed, &basis_ctof);
      InitRestriction(trial_fes, use_bdr, true, false, nelem, indices,
                      ceed, &trial_restr);
      InitRestriction(test_fes, use_bdr, true, true, nelem, indices,
                      ceed, &test_restr);
      MFEM_VERIFY(info.trial_op == EvalMode::Interp,
                  "ceed:Interpolator only supports trial_op == Interp.");
      MFEM_VERIFY(info.test_op == EvalMode::None,
                  "ceed:Interpolator only supports test_op == None.");

      // Create the QFunction that defines the action of the operator
      // (only an identity as element dof multiplicity is handled outside of libCEED)
      CeedQFunctionCreateIdentity(ceed, trial_vdim, CEED_EVAL_INTERP, CEED_EVAL_NONE,
                                  &apply_qfunc);
      CeedQFunctionCreateIdentity(ceed, trial_vdim, CEED_EVAL_NONE, CEED_EVAL_INTERP,
                                  &apply_qfunc_t);

      // Create the operator
      CeedOperatorCreate(ceed, apply_qfunc, NULL, NULL, &oper);
      CeedOperatorSetField(oper, "input", trial_restr, basis_ctof,
                           CEED_VECTOR_ACTIVE);
      CeedOperatorSetField(oper, "output", test_restr, CEED_BASIS_NONE,
                           CEED_VECTOR_ACTIVE);
      CeedOperatorCheckReady(oper);

      // Create the transpose operator
      CeedOperatorCreate(ceed, apply_qfunc_t, NULL, NULL, &oper_t);
      CeedOperatorSetField(oper_t, "input", test_restr, CEED_BASIS_NONE,
                           CEED_VECTOR_ACTIVE);
      CeedOperatorSetField(oper_t, "output", trial_restr, basis_ctof,
                           CEED_VECTOR_ACTIVE);
      CeedOperatorCheckReady(oper_t);
   }

   ~Interpolator()
   {
      // All basis and restriction objects are destroyed by fes destructor
      CeedOperatorDestroy(&oper);
      CeedOperatorDestroy(&oper_t);
      CeedQFunctionDestroy(&apply_qfunc);
      CeedQFunctionDestroy(&apply_qfunc_t);
   }
#endif
};

} // namespace ceed

} // namespace mfem

#endif // MFEM_LIBCEED_INTEGRATOR
