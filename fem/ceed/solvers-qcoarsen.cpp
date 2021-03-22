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

#include "solvers-qcoarsen.hpp"
#include "util.hpp"

#ifdef MFEM_USE_CEED
#include <ceed-backend.h>

// todo: should probably use Ceed memory wrappers instead of calloc/free?
#include <stdlib.h>
#include "linear_qf.h"

#include <math.h>  // for fabs(), which I actualy don't like

#include <fstream>

namespace mfem
{

namespace ceed
{

/** @brief Coarsen the rows (integration points) of a CeedBasis

    Originally thought this would be like CeedBasisATPMGCoarsen(),
    but the "interpolation" in quadrature points is not quite
    analogous, so I think I'm going to do something simpler but less
    algebraic.

    We will need something like qbasisctof in order to coarsen the
    linearassembled vector, in fact qbasisctof is essentially P_Q
    and is essential to Pazner's theorem.

    For the first coarsening, fine_quadmode = CEED_GAUSS (almost certainly)
    You can decide the coarse_quadmode, CEED_GAUSS_LOBATTO leads to collocated B operators
    For later coarsenings, fine_quadmode has to match what you did before for coarse

    @param[in] basisin   the CeedBasis, already p-coarsened
    @param[out] basisout new CeedBasis, same width, but shorter
    @param[out] qbasisctof describes interpolation of integration points (P_Q)
    @param[in] order_reduction  amount to coarsen
    @param[in] collocated_coarse  whether to use collocated quadrature on coarser level
    @param[in] fine_quadmode  points for fine quadrature rule
*/
int CeedBasisQCoarsen(CeedBasis basisin, CeedBasis* basisout,
                      CeedBasis* qbasisctof,
                      int order_reduction,
                      CeedQuadMode fine_quadmode,
                      CeedQuadMode coarse_quadmode)
{
   int ierr;
   Ceed ceed;
   ierr = CeedBasisGetCeed(basisin, &ceed); CeedChk(ierr);

   CeedInt dim, ncomp, P1d, Q1d;
   ierr = CeedBasisGetDimension(basisin, &dim); CeedChk(ierr);
   ierr = CeedBasisGetNumComponents(basisin, &ncomp); CeedChk(ierr);
   ierr = CeedBasisGetNumNodes1D(basisin, &P1d); CeedChk(ierr);
   ierr = CeedBasisGetNumQuadraturePoints1D(basisin, &Q1d); CeedChk(ierr);

   CeedInt coarse_Q1d = Q1d - order_reduction;

   // the "grad" part of qbasisctof will be meaningless, we only use the
   // "interp" part.
   if (coarse_quadmode == CEED_GAUSS)
   {
      ierr = CeedBasisCreateTensorH1Gauss(ceed, dim, ncomp, coarse_Q1d, Q1d,
                                          fine_quadmode, qbasisctof); CeedChk(ierr);
   }
   else if (coarse_quadmode == CEED_GAUSS_LOBATTO)
   {
      ierr = CeedBasisCreateTensorH1Lagrange(ceed, dim, ncomp, coarse_Q1d, Q1d,
                                             fine_quadmode, qbasisctof); CeedChk(ierr);
   }
   else
   {
      return CeedError(ceed, 1, "Bad quadrature mode!");
   }

   // the Ceed reference element is [-1, 1], while the MFEM element is [0, 1]
   // which means with order_reduction=0 we get different gradients in the
   // bases; should actually interpolate or something...
   ierr = CeedBasisCreateMFEMTensorH1Lagrange(ceed, dim, ncomp, P1d, coarse_Q1d,
                                              coarse_quadmode, basisout); CeedChk(ierr);

   return 0;
}

/** Given the (CeedVector) output of CeedOperatorLinearAssembleQFunction,
    coarsen it according to the local integration-interpolation qbasisctof
    and the numbering encoded in rstr_q, returning a (smaller)
    CeedVector with the coarsened D operator. */
int CeedQFunctionCoarsenAssembledVector(CeedVector assembledqf,
                                        CeedElemRestriction rstr_q,
                                        CeedBasis qbasisctof,
                                        CeedVector* coarse_assembledqf_out)
{
   int ierr;
   Ceed ceed;
   ierr = CeedVectorGetCeed(assembledqf, &ceed); CeedChk(ierr);

   CeedInt layout[3];
   ierr = CeedElemRestrictionGetELayout(rstr_q, &layout); CeedChk(ierr);
   CeedInt qflength;
   ierr = CeedVectorGetLength(assembledqf, &qflength); CeedChk(ierr);

   // may want to ignore layout *entirely* here
   int elemsize, lsize, ncomp, nelem;
   ierr = CeedElemRestrictionGetLVectorSize(rstr_q, &lsize); CeedChk(ierr);
   ierr = CeedElemRestrictionGetNumElements(rstr_q, &nelem); CeedChk(ierr);
   ierr = CeedElemRestrictionGetElementSize(rstr_q, &elemsize); CeedChk(ierr);
   ierr = CeedElemRestrictionGetNumComponents(rstr_q, &ncomp); CeedChk(ierr);

   CeedInt coarse_qflength;
   CeedInt P, Q, basis_ncomp;
   ierr = CeedBasisGetNumNodes(qbasisctof, &P); CeedChk(ierr);
   ierr = CeedBasisGetNumQuadraturePoints(qbasisctof, &Q); CeedChk(ierr);
   ierr = CeedBasisGetNumComponents(qbasisctof, &basis_ncomp); CeedChk(ierr);

   if (qflength != ncomp * nelem * Q)
   {
      return CeedError(ceed, 1, "original qfunction vector does not match rstr_q!");
   }

   // note well ncomp (not basis_ncomp) on line below (they are different)
   coarse_qflength = ncomp * nelem * P;
   if (Q != elemsize)
   {
      return CeedError(ceed, 1, "qbasisctof does not match rstr_q!");
   }

   CeedVector coarse_assembledqf;
   ierr = CeedVectorCreate(ceed, coarse_qflength, &coarse_assembledqf);
   CeedChk(ierr);
   ierr = CeedVectorSetValue(coarse_assembledqf, 0.0); CeedChk(ierr);

   const CeedScalar* finedata;
   CeedScalar* coarsedata;
   ierr = CeedVectorGetArrayRead(assembledqf, CEED_MEM_HOST, &finedata);
   CeedChk(ierr);
   ierr = CeedVectorGetArray(coarse_assembledqf, CEED_MEM_HOST, &coarsedata);
   CeedChk(ierr);

   // rows associated with fine, cols associated with coarse
   // they're both quadpoints, but interface thinks rows(fine) are quad, cols(coarse) are basis
   // also note I am applying the *transpose*
   const CeedScalar* ctof_interp;
   ierr = CeedBasisGetInterp(qbasisctof, &ctof_interp); CeedChk(ierr);
   for (int k = 0; k < coarse_qflength; ++k)
   {
      coarsedata[k] = 0.0;
   }
   const int d_per_fineelem = ncomp * Q;
   const int d_per_coarseelem = ncomp * P;

   for (int e = 0; e < nelem; ++e)
   {
      for (int j = 0; j < P; ++j)  // associated with coarse
      {
         for (int c = 0; c < ncomp; ++c)
         {
            const int output_index = e*d_per_coarseelem + c*P + j;
            for (int i = 0; i < Q; ++i) // associated with fine
            {
               const int input_index = e*d_per_fineelem + c*Q + i;
               coarsedata[output_index] +=
                  ctof_interp[i*P + j] * finedata[input_index];
            }
         }
      }
   }

   ierr = CeedVectorRestoreArrayRead(assembledqf, &finedata); CeedChk(ierr);
   ierr = CeedVectorRestoreArray(coarse_assembledqf, &coarsedata); CeedChk(ierr);

   *coarse_assembledqf_out = coarse_assembledqf;

   return 0;
}

/** Given an ElemRestriction rstr_q and a local integration-interpolation,
    return a CeedElemRestriction with the integration points coarsened.

    @param[in] rstr_q
    @param[in] qbasisctof (only used for dimensions/sizes!)
    @param[out] coarse_rstr_q
    @param[out] ncomp
*/
int CeedElementRestrictionQCoarsen(CeedElemRestriction rstr_q,
                                   CeedBasis qbasisctof,
                                   CeedElemRestriction* coarse_rstr_q,
                                   CeedInt* ncomp)
{
   int ierr;
   Ceed ceed;
   ierr = CeedElemRestrictionGetCeed(rstr_q, &ceed); CeedChk(ierr);

   // layout and strides are different; we may only care about strides
   CeedInt layout[3];
   ierr = CeedElemRestrictionGetELayout(rstr_q, &layout); CeedChk(ierr);
   CeedInt strides[3];
   ierr = CeedElemRestrictionGetStrides(rstr_q, &strides); CeedChk(ierr);

   CeedInt coarse_lsize;
   CeedInt coarse_strides[3];

   // some of these are only used for sanity checking
   CeedInt q_nelem, q_elemsize, q_lsize, q_ncomp;
   ierr = CeedElemRestrictionGetNumElements(rstr_q, &q_nelem); CeedChk(ierr);
   ierr = CeedElemRestrictionGetElementSize(rstr_q, &q_elemsize); CeedChk(ierr);
   ierr = CeedElemRestrictionGetLVectorSize(rstr_q, &q_lsize); CeedChk(ierr);
   ierr = CeedElemRestrictionGetNumComponents(rstr_q, &q_ncomp); CeedChk(ierr);
   *ncomp = q_ncomp;

   // Q is *fine* quadpoints, coarse_elemsize is *coarse* quadpoints
   CeedInt coarse_elemsize, Q;
   ierr = CeedBasisGetNumNodes(qbasisctof, &coarse_elemsize); CeedChk(ierr);
   ierr = CeedBasisGetNumQuadraturePoints(qbasisctof, &Q); CeedChk(ierr);

   coarse_lsize = q_nelem * coarse_elemsize * q_ncomp;

   coarse_strides[0] = strides[0];  // always 1, as far as I can tell
   if (strides[1] == q_elemsize && strides[2] == q_elemsize * q_ncomp)
   {
      // characteristic of host layout/strides?
      coarse_strides[1] = coarse_elemsize;
      coarse_strides[2] = q_ncomp * coarse_elemsize;
   }
   else if (strides[1] == q_elemsize * q_nelem && strides[2] == q_elemsize)
   {
      // characteristic of device layout/strides?
      coarse_strides[1] = q_nelem * coarse_elemsize;
      coarse_strides[2] = coarse_elemsize;
   }
   else
   {
      return CeedError(ceed, 1, "I do not understand");
   }
   ierr = CeedElemRestrictionCreateStrided(ceed, q_nelem, coarse_elemsize, q_ncomp,
                                           coarse_lsize, coarse_strides,
                                           coarse_rstr_q); CeedChk(ierr);

   return 0;
}

int CeedSingleOperatorGetHeuristics(CeedOperator oper, CeedScalar* minq,
                                    CeedScalar* maxq, CeedScalar* absmin)
{
   int ierr;
   CeedQFunction qfin;
   ierr = CeedOperatorGetQFunction(oper, &qfin); CeedChk(ierr);

   CeedVector assembledqf;
   CeedElemRestriction rstr_q;
   ierr = CeedOperatorLinearAssembleQFunction(
             oper, &assembledqf, &rstr_q, CEED_REQUEST_IMMEDIATE); CeedChk(ierr);

   CeedInt assembledqf_len;
   ierr = CeedVectorGetLength(assembledqf, &assembledqf_len); CeedChk(ierr);
   const CeedScalar * tempdata;
   ierr = CeedVectorGetArrayRead(assembledqf, CEED_MEM_HOST, &tempdata);
   CeedChk(ierr);

   *minq = 1.e+12;
   *maxq = -1.e+12;
   *absmin = 1.e+12;
   for (int i = 0; i < assembledqf_len; ++i)
   {
      *maxq = std::max(*maxq, tempdata[i]);
      *minq = std::min(*minq, tempdata[i]);
      *absmin = std::min(*absmin, fabs(tempdata[i]));
   }
   ierr = CeedVectorRestoreArrayRead(assembledqf, &tempdata); CeedChk(ierr);

   ierr = CeedElemRestrictionDestroy(&rstr_q); CeedChk(ierr);
   ierr = CeedVectorDestroy(&assembledqf); CeedChk(ierr);
   return 0;
}

int CeedOperatorGetHeuristics(CeedOperator oper, CeedScalar* minq,
                              CeedScalar* maxq, CeedScalar* absmin)
{
   int ierr;
   bool isComposite;
   ierr = CeedOperatorIsComposite(oper, &isComposite); CeedChk(ierr);
   if (!isComposite)
   {
      return CeedSingleOperatorGetHeuristics(oper, minq, maxq, absmin);
   }

   *minq = 1.e+12;
   *maxq = -1.e+12;
   *absmin = 1.e+12;

   int nsub;
   ierr = CeedOperatorGetNumSub(oper, &nsub); CeedChk(ierr);
   CeedOperator *subops;
   ierr = CeedOperatorGetSubList(oper, &subops); CeedChk(ierr);
   for (int isub=0; isub<nsub; ++isub)
   {
      CeedOperator subop = subops[isub];
      CeedScalar lminq, lmaxq, labsmin;
      ierr = CeedSingleOperatorGetHeuristics(subop, &lminq, &lmaxq,
                                             &labsmin); CeedChk(ierr);
      *minq = std::min(lminq, *minq);
      *maxq = std::max(lmaxq, *maxq);
      *absmin = std::min(*absmin, labsmin);
   }

   return 0;
}

int CeedQFunctionQCoarsen(CeedOperator oper, CeedInt qorder_reduction,
                          CeedVector* coarse_assembledqf,
                          CeedElemRestriction* coarse_rstr_q, CeedBasis* qcoarse_basis,
                          CeedQFunction* qfout, CeedQFunctionContext* context_ptr,
                          CeedQuadMode fine_qmode, CeedQuadMode coarse_qmode)
{
   int ierr;
   Ceed ceed;
   ierr = CeedOperatorGetCeed(oper, &ceed); CeedChk(ierr);

   CeedQFunction qfin;
   ierr = CeedOperatorGetQFunction(oper, &qfin); CeedChk(ierr);
   CeedInt vlength;
   ierr = CeedQFunctionGetVectorLength(qfin, &vlength); CeedChk(ierr);

   CeedVector assembledqf;
   CeedElemRestriction rstr_q;
   ierr = CeedOperatorLinearAssembleQFunction(
             oper, &assembledqf, &rstr_q, CEED_REQUEST_IMMEDIATE); CeedChk(ierr);

   CeedBasis qbasisctof; // P_Q
   CeedBasis fine_basis;
   ierr = CeedOperatorGetActiveBasis(oper, &fine_basis); CeedChk(ierr);
   ierr = CeedBasisQCoarsen(fine_basis, qcoarse_basis, &qbasisctof,
                            qorder_reduction, fine_qmode, coarse_qmode); CeedChk(ierr);

   int ncomp_rstr; // components in ElementRestriction rstr_q
   ierr = CeedElementRestrictionQCoarsen(rstr_q, qbasisctof, coarse_rstr_q,
                                         &ncomp_rstr); CeedChk(ierr);
   CeedInt coarse_vlength = vlength; /// this looks wrong but is probably right

   // coarsen: coarsen the vector assembledqf itself (using basisctof == P_Q)
   ierr = CeedQFunctionCoarsenAssembledVector(assembledqf, rstr_q, qbasisctof,
                                              coarse_assembledqf); CeedChk(ierr);

   CeedInt coarse_layout[3];
   ierr = CeedElemRestrictionGetELayout(*coarse_rstr_q, &coarse_layout);
   CeedChk(ierr);

   // not entirely sure of the next kernel magic
   std::string qf_file = GetCeedPath() + "/linear_qf.h";
   std::string qf = qf_file + ":qcoarsen_linearfunc";
   ierr = CeedQFunctionCreateInterior(ceed, coarse_vlength, qcoarsen_linearfunc,
                                      qf.c_str(), qfout); CeedChk(ierr);

   struct LinearQFunctionContext * context =
      (struct LinearQFunctionContext *) calloc(1,
                                               sizeof(struct LinearQFunctionContext));
   context->dim = -1;
   context->ncomp = ncomp_rstr;
   for (int i = 0; i < 3; ++i)
   {
      context->layout[i] = coarse_layout[i];
   }
   CeedQFunctionContext qf_context;
   ierr = CeedQFunctionContextCreate(ceed, &qf_context); CeedChk(ierr);
   ierr = CeedQFunctionContextSetData(qf_context, CEED_MEM_HOST, CEED_COPY_VALUES,
                                      sizeof(*context), context); CeedChk(ierr);
   ierr = CeedQFunctionSetContext(*qfout, qf_context); CeedChk(ierr);
   *context_ptr = qf_context;
   free(context);

   ierr = CeedElemRestrictionDestroy(&rstr_q); CeedChk(ierr);
   ierr = CeedBasisDestroy(&qbasisctof); CeedChk(ierr);
   ierr = CeedVectorDestroy(&assembledqf); CeedChk(ierr);

   return 0;
}

/** @brief given a CeedOperator, use the "assembled" qfunction to create your
    own qfunction that has the same action (works for linear operators),
    and build a new CeedOperator around that.

    oper is in, qorder_reduction is in, everything else is out
*/
int CeedOperatorQCoarsen(CeedOperator oper, int qorder_reduction,
                         CeedOperator* out, CeedVector* coarse_assembledqf,
                         CeedQFunctionContext* context_ptr,
                         CeedQuadMode fine_qmode, CeedQuadMode coarse_qmode)
{
   int ierr;
   Ceed ceed;
   ierr = CeedOperatorGetCeed(oper, &ceed); CeedChk(ierr);

   CeedQFunction qfin;
   ierr = CeedOperatorGetQFunction(oper, &qfin); CeedChk(ierr);
   CeedElemRestriction coarse_rstr_q;
   CeedQFunction qfout;
   CeedBasis qcoarse_basis;
   ierr = CeedQFunctionQCoarsen(oper, qorder_reduction, coarse_assembledqf,
                                &coarse_rstr_q, &qcoarse_basis, &qfout,
                                context_ptr,
                                fine_qmode, coarse_qmode); CeedChk(ierr);

   CeedInt numinputfields, numoutputfields;
   ierr = CeedQFunctionGetNumArgs(qfin, &numinputfields, &numoutputfields);
   CeedChk(ierr);
   CeedQFunctionField *inputqfields, *outputqfields;
   ierr = CeedQFunctionGetFields(qfin, &inputqfields, &outputqfields);
   CeedChk(ierr);
   CeedOperatorField *inputfields, *outputfields;
   ierr = CeedOperatorGetFields(oper, &inputfields, &outputfields); CeedChk(ierr);

   // Determine active input basis, get dimension, numemodein
   CeedInt size;
   char * fieldname;
   CeedInt numemodein = 0, dim = 1;
   CeedEvalMode emodein;
   CeedBasis basisin = NULL;
   CeedVector vec;
   for (CeedInt i=0; i<numinputfields; i++)
   {
      ierr = CeedOperatorFieldGetVector(inputfields[i], &vec); CeedChk(ierr);
      if (vec == CEED_VECTOR_ACTIVE)
      {
         ierr = CeedOperatorFieldGetBasis(inputfields[i], &basisin); CeedChk(ierr);
         ierr = CeedBasisGetDimension(basisin, &dim); CeedChk(ierr);
         ierr = CeedQFunctionFieldGetEvalMode(inputqfields[i], &emodein); CeedChk(ierr);
         switch (emodein)
         {
            case CEED_EVAL_NONE:
            case CEED_EVAL_INTERP:
               numemodein += 1;
               break;
            case CEED_EVAL_GRAD:
               numemodein += dim;
               break;
            case CEED_EVAL_WEIGHT:
            case CEED_EVAL_DIV:
            case CEED_EVAL_CURL:
               break; // Caught by QF Assembly
         }
         ierr = CeedQFunctionFieldGetName(inputqfields[i], &fieldname); CeedChk(ierr);
         ierr = CeedQFunctionFieldGetSize(inputqfields[i], &size); CeedChk(ierr);
         ierr = CeedQFunctionAddInput(qfout, fieldname, size, emodein); CeedChk(ierr);
      }
      else
      {
         // don't do anything for inactive fields
         // maybe count them to make sure there's exactly one?
      }
   }

   ierr = CeedQFunctionAddInput(qfout, "assembled", numemodein*numemodein,
                                CEED_EVAL_NONE); CeedChk(ierr);

   // Determine active output basis, count emodeout
   CeedInt numemodeout = 0;
   CeedEvalMode emodeout;
   for (CeedInt i=0; i<numoutputfields; i++)
   {
      ierr = CeedOperatorFieldGetVector(outputfields[i], &vec); CeedChk(ierr);
      if (vec == CEED_VECTOR_ACTIVE)
      {
         ierr = CeedQFunctionFieldGetEvalMode(outputqfields[i], &emodeout);
         CeedChk(ierr);
         switch (emodeout)
         {
            case CEED_EVAL_NONE:
            case CEED_EVAL_INTERP:
               numemodeout += 1;
               break;
            case CEED_EVAL_GRAD:
               numemodeout += dim;
               break;
            case CEED_EVAL_WEIGHT:
            case CEED_EVAL_DIV:
            case CEED_EVAL_CURL:
               break; // Caught by QF Assembly
         }
         ierr = CeedQFunctionFieldGetName(outputqfields[i], &fieldname); CeedChk(ierr);
         ierr = CeedQFunctionFieldGetSize(outputqfields[i], &size); CeedChk(ierr);
         ierr = CeedQFunctionAddOutput(qfout, fieldname, size, emodeout); CeedChk(ierr);
      }
      else
      {
         // don't do anything for inactive fields
      }
   }

   CeedOperator qcoper;
   ierr = CeedOperatorCreate(ceed, qfout, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                             &qcoper); CeedChk(ierr);

   bool linear_thing_set = false;
   CeedBasis basis;
   CeedElemRestriction er_input;
   for (int i = 0; i < numinputfields; ++i)
   {
      ierr = CeedQFunctionFieldGetName(inputqfields[i], &fieldname); CeedChk(ierr);
      ierr = CeedOperatorFieldGetVector(inputfields[i], &vec); CeedChk(ierr);
      ierr = CeedOperatorFieldGetBasis(inputfields[i], &basis); CeedChk(ierr);
      ierr = CeedOperatorFieldGetElemRestriction(inputfields[i], &er_input);
      CeedChk(ierr);
      if (vec == CEED_VECTOR_ACTIVE)
      {
         ierr = CeedOperatorSetField(qcoper, fieldname, er_input, qcoarse_basis,
                                     CEED_VECTOR_ACTIVE); CeedChk(ierr);
      }
      else
      {
         if (linear_thing_set)
         {
            return CeedError(ceed, 1, "Only know how to do one non-active vector!");
         }
         ierr = CeedOperatorSetField(qcoper, "assembled", coarse_rstr_q,
                                     CEED_BASIS_COLLOCATED,
                                     *coarse_assembledqf); CeedChk(ierr);
         linear_thing_set = true;
      }
   }
   if (!linear_thing_set)
   {
      return CeedError(ceed, 1, "Did not find active vector!");
   }
   CeedElemRestriction er_output;
   for (int i = 0; i < numoutputfields; ++i)
   {
      ierr = CeedQFunctionFieldGetName(outputqfields[i], &fieldname); CeedChk(ierr);
      ierr = CeedOperatorFieldGetVector(outputfields[i], &vec); CeedChk(ierr);
      ierr = CeedOperatorFieldGetBasis(outputfields[i], &basis); CeedChk(ierr);
      ierr = CeedOperatorFieldGetElemRestriction(outputfields[i], &er_output);
      CeedChk(ierr);
      if (vec == CEED_VECTOR_ACTIVE)
      {
         ierr = CeedOperatorSetField(qcoper, fieldname, er_output, qcoarse_basis,
                                     CEED_VECTOR_ACTIVE); CeedChk(ierr);
      }
      else
      {
         return CeedError(ceed, 1, "Don't think this should happen!");
      }
   }

   // the following probably do not really get destroyed, but their refcounts
   // get reduced, so now they are owned by qcoper
   ierr = CeedElemRestrictionDestroy(&coarse_rstr_q); CeedChk(ierr);
   ierr = CeedQFunctionDestroy(&qfout); CeedChk(ierr);
   ierr = CeedBasisDestroy(&qcoarse_basis); CeedChk(ierr);

   *out = qcoper;
   return 0;
}

}

}

#endif // MFEM_USE_CEED
