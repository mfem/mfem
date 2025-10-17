// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "full-assembly.hpp"

#ifdef MFEM_USE_CEED

#include "../../../linalg/sparsemat.hpp"
#include "../interface/util.hpp"
#include "../interface/ceed.hpp"

namespace mfem
{

namespace ceed
{

int CeedHackReallocArray(size_t n, size_t unit, void *p)
{
   *(void **)p = realloc(*(void **)p, n*unit);
   if (n && unit && !*(void **)p)
      return CeedError(NULL, 1, "realloc failed to allocate %zd members of size "
                       "%zd\n", n, unit);
   return 0;
}

#define CeedHackRealloc(n, p) CeedHackReallocArray((n), sizeof(**(p)), p)

int CeedHackFree(void *p)
{
   free(*(void **)p);
   *(void **)p = NULL;
   return 0;
}

int CeedSingleOperatorFullAssemble(CeedOperator op, SparseMatrix *out)
{
   int ierr;
   Ceed ceed;
   ierr = CeedOperatorGetCeed(op, &ceed); PCeedChk(ierr);

   // Assemble QFunction
   CeedQFunction qf;
   ierr = CeedOperatorGetQFunction(op, &qf); PCeedChk(ierr);
   CeedInt numinputfields, numoutputfields;
   PCeedChk(ierr);
   CeedVector assembledqf;
   CeedElemRestriction rstr_q;
   ierr = CeedOperatorLinearAssembleQFunction(
             op, &assembledqf, &rstr_q, CEED_REQUEST_IMMEDIATE); PCeedChk(ierr);

   CeedSize qflength;
   ierr = CeedVectorGetLength(assembledqf, &qflength); PCeedChk(ierr);

   CeedOperatorField *input_fields;
   CeedOperatorField *output_fields;
   ierr = CeedOperatorGetFields(op, &numinputfields, &input_fields,
                                &numoutputfields, &output_fields);
   PCeedChk(ierr);

   // Determine active input basis
   CeedQFunctionField *qffields;
   ierr = CeedQFunctionGetFields(qf, &numinputfields, &qffields,
                                 &numoutputfields, NULL);
   PCeedChk(ierr);
   CeedInt numemodein = 0, ncomp, dim = 1;
   CeedEvalMode *emodein = NULL;
   CeedBasis basisin = NULL;
   CeedElemRestriction rstrin = NULL;
   for (CeedInt i=0; i<numinputfields; i++)
   {
      CeedVector vec;
      ierr = CeedOperatorFieldGetVector(input_fields[i], &vec); PCeedChk(ierr);
      if (vec == CEED_VECTOR_ACTIVE)
      {
         CeedBasis basis;
         ierr = CeedOperatorFieldGetBasis(input_fields[i], &basis); PCeedChk(ierr);
         if (!basisin)
         {
            ierr = CeedBasisReferenceCopy(basis, &basisin); PCeedChk(ierr);
         }
#if CEED_VERSION_GE(0, 13, 0)
         ierr = CeedBasisDestroy(&basis); PCeedChk(ierr);
#endif
         ierr = CeedBasisGetNumComponents(basisin, &ncomp); PCeedChk(ierr);
         ierr = CeedBasisGetDimension(basisin, &dim); PCeedChk(ierr);
         CeedElemRestriction rstr;
         ierr = CeedOperatorFieldGetElemRestriction(input_fields[i], &rstr);
         PCeedChk(ierr);
         if (!rstrin)
         {
            ierr = CeedElemRestrictionReferenceCopy(rstr, &rstrin); PCeedChk(ierr);
         }
#if CEED_VERSION_GE(0, 13, 0)
         ierr = CeedElemRestrictionDestroy(&rstr); PCeedChk(ierr);
#endif
         CeedEvalMode emode;
         ierr = CeedQFunctionFieldGetEvalMode(qffields[i], &emode);
         PCeedChk(ierr);
         switch (emode)
         {
            case CEED_EVAL_NONE:
            case CEED_EVAL_INTERP:
               ierr = CeedHackRealloc(numemodein + 1, &emodein); PCeedChk(ierr);
               emodein[numemodein] = emode;
               numemodein += 1;
               break;
            case CEED_EVAL_GRAD:
               ierr = CeedHackRealloc(numemodein + dim, &emodein); PCeedChk(ierr);
               for (CeedInt d=0; d<dim; d++)
               {
                  emodein[numemodein+d] = emode;
               }
               numemodein += dim;
               break;
            case CEED_EVAL_WEIGHT:
            case CEED_EVAL_DIV:
            case CEED_EVAL_CURL:
               break; // Caught by QF Assembly
         }
      }
#if CEED_VERSION_GE(0, 13, 0)
      ierr = CeedVectorDestroy(&vec); PCeedChk(ierr);
#endif
   }

   // Determine active output basis
   ierr = CeedQFunctionGetFields(qf, &numinputfields, NULL, &numoutputfields,
                                 &qffields); PCeedChk(ierr);
   CeedInt numemodeout = 0;
   CeedEvalMode *emodeout = NULL;
   CeedBasis basisout = NULL;
   CeedElemRestriction rstrout = NULL;
   for (CeedInt i=0; i<numoutputfields; i++)
   {
      CeedVector vec;
      ierr = CeedOperatorFieldGetVector(output_fields[i], &vec); PCeedChk(ierr);
      if (vec == CEED_VECTOR_ACTIVE)
      {
         CeedBasis basis;
         ierr = CeedOperatorFieldGetBasis(output_fields[i], &basis); PCeedChk(ierr);
         if (!basisout)
         {
            ierr = CeedBasisReferenceCopy(basis, &basisout); PCeedChk(ierr);
         }
#if CEED_VERSION_GE(0, 13, 0)
         ierr = CeedBasisDestroy(&basis); PCeedChk(ierr);
#endif
         CeedElemRestriction rstr;
         ierr = CeedOperatorFieldGetElemRestriction(output_fields[i], &rstr);
         PCeedChk(ierr);
         if (!rstrout)
         {
            ierr = CeedElemRestrictionReferenceCopy(rstr, &rstrout); PCeedChk(ierr);
         }
#if CEED_VERSION_GE(0, 13, 0)
         ierr = CeedElemRestrictionDestroy(&rstr); PCeedChk(ierr);
#endif
         CeedEvalMode emode;
         ierr = CeedQFunctionFieldGetEvalMode(qffields[i], &emode);
         PCeedChk(ierr);
         switch (emode)
         {
            case CEED_EVAL_NONE:
            case CEED_EVAL_INTERP:
               ierr = CeedHackRealloc(numemodeout + 1, &emodeout); PCeedChk(ierr);
               emodeout[numemodeout] = emode;
               numemodeout += 1;
               break;
            case CEED_EVAL_GRAD:
               ierr = CeedHackRealloc(numemodeout + dim, &emodeout); PCeedChk(ierr);
               for (CeedInt d=0; d<dim; d++)
               {
                  emodeout[numemodeout+d] = emode;
               }
               numemodeout += dim;
               break;
            case CEED_EVAL_WEIGHT:
            case CEED_EVAL_DIV:
            case CEED_EVAL_CURL:
               break; // Caught by QF Assembly
         }
      }
#if CEED_VERSION_GE(0, 13, 0)
      ierr = CeedVectorDestroy(&vec); PCeedChk(ierr);
#endif
   }

   CeedInt nelem, elemsize, nqpts;
   CeedSize nnodes;
   ierr = CeedElemRestrictionGetNumElements(rstrin, &nelem); PCeedChk(ierr);
   ierr = CeedElemRestrictionGetElementSize(rstrin, &elemsize); PCeedChk(ierr);
   ierr = CeedElemRestrictionGetLVectorSize(rstrin, &nnodes); PCeedChk(ierr);
   ierr = CeedBasisGetNumQuadraturePoints(basisin, &nqpts); PCeedChk(ierr);

   // Determine elem_dof relation
   CeedVector index_vec;
   ierr = CeedVectorCreate(ceed, nnodes, &index_vec); PCeedChk(ierr);
   CeedScalar *array;
   ierr = CeedVectorGetArrayWrite(index_vec, CEED_MEM_HOST, &array);
   PCeedChk(ierr);
   for (CeedSize i = 0; i < nnodes; ++i)
   {
      array[i] = i;
   }
   ierr = CeedVectorRestoreArray(index_vec, &array); PCeedChk(ierr);
   CeedVector elem_dof;
   ierr = CeedVectorCreate(ceed, nelem * elemsize, &elem_dof); PCeedChk(ierr);
   ierr = CeedVectorSetValue(elem_dof, 0.0); PCeedChk(ierr);
   CeedElemRestrictionApply(rstrin, CEED_NOTRANSPOSE, index_vec,
                            elem_dof, CEED_REQUEST_IMMEDIATE); PCeedChk(ierr);
   const CeedScalar * elem_dof_a;
   ierr = CeedVectorGetArrayRead(elem_dof, CEED_MEM_HOST, &elem_dof_a);
   PCeedChk(ierr);
   ierr = CeedVectorDestroy(&index_vec); PCeedChk(ierr);

   // loop over elements and put in SparseMatrix
   // SparseMatrix * out = new SparseMatrix(nnodes, nnodes);
   MFEM_ASSERT(out->Height() == nnodes, "Sizes don't match!");
   MFEM_ASSERT(out->Width() == nnodes, "Sizes don't match!");
   const CeedScalar *interpin, *gradin;
   ierr = CeedBasisGetInterp(basisin, &interpin); PCeedChk(ierr);
   ierr = CeedBasisGetGrad(basisin, &gradin); PCeedChk(ierr);

   const CeedScalar * assembledqfarray;
   ierr = CeedVectorGetArrayRead(assembledqf, CEED_MEM_HOST, &assembledqfarray);
   PCeedChk(ierr);

   CeedInt layout[3];
#if CEED_VERSION_GE(0, 13, 0)
   ierr = CeedElemRestrictionGetELayout(rstr_q, layout); PCeedChk(ierr);
#else
   ierr = CeedElemRestrictionGetELayout(rstr_q, &layout); PCeedChk(ierr);
#endif
   ierr = CeedElemRestrictionDestroy(&rstr_q); PCeedChk(ierr);

   // enforce structurally symmetric for later elimination
   const int skip_zeros = 0;
   MFEM_ASSERT(numemodein == numemodeout,
               "Ceed full assembly not implemented for this case.");
   for (int e = 0; e < nelem; ++e)
   {
      // get Array<int> for use in SparseMatrix::AddSubMatrix()
      Array<int> rows(elemsize);
      for (int i = 0; i < elemsize; ++i)
      {
         rows[i] = elem_dof_a[e * elemsize + i];
      }

      // form element matrix itself
      DenseMatrix Bmat(nqpts * numemodein, elemsize);
      Bmat = 0.0;
      // Store block-diagonal D matrix as collection of small dense blocks
      DenseTensor Dmat(numemodeout, numemodein, nqpts);
      Dmat = 0.0;
      DenseMatrix elem_mat(elemsize, elemsize);
      elem_mat = 0.0;
      for (int q = 0; q < nqpts; ++q)
      {
         for (int n = 0; n < elemsize; ++n)
         {
            CeedInt din = -1;
            for (int ein = 0; ein < numemodein; ++ein)
            {
               if (emodein[ein] == CEED_EVAL_INTERP)
               {
                  Bmat(numemodein * q + ein, n) += interpin[q * elemsize + n];
               }
               else if (emodein[ein] == CEED_EVAL_GRAD)
               {
                  din += 1;
                  Bmat(numemodein * q + ein, n) += gradin[(din*nqpts+q) * elemsize + n];
               }
               else
               {
                  MFEM_ASSERT(false, "Not implemented!");
               }
            }
         }
         for (int ei = 0; ei < numemodein; ++ei)
         {
            for (int ej = 0; ej < numemodein; ++ej)
            {
               const int comp = ei * numemodein + ej;
               const int index = q*layout[0] + comp*layout[1] + e*layout[2];
               Dmat(ei, ej, q) += assembledqfarray[index];
            }
         }
      }
      DenseMatrix BTD(elemsize, nqpts*numemodein);
      // Compute B^T*D
      BTD = 0.0;
      for (int j=0; j<elemsize; ++j)
      {
         for (int q=0; q<nqpts; ++q)
         {
            int qq = numemodein*q;
            for (int ei = 0; ei < numemodein; ++ei)
            {
               for (int ej = 0; ej < numemodein; ++ej)
               {
                  BTD(j,qq+ei) += Bmat(qq+ej,j)*Dmat(ej,ei,q);
               }
            }
         }
      }

      Mult(BTD, Bmat, elem_mat);

      // put element matrix in sparsemat
      out->AddSubMatrix(rows, rows, elem_mat, skip_zeros);
   }

   ierr = CeedVectorRestoreArrayRead(elem_dof, &elem_dof_a); PCeedChk(ierr);
   ierr = CeedVectorDestroy(&elem_dof); PCeedChk(ierr);
   ierr = CeedVectorRestoreArrayRead(assembledqf, &assembledqfarray);
   PCeedChk(ierr);
   ierr = CeedVectorDestroy(&assembledqf); PCeedChk(ierr);
   ierr = CeedElemRestrictionDestroy(&rstrin); PCeedChk(ierr);
   ierr = CeedElemRestrictionDestroy(&rstrout); PCeedChk(ierr);
   ierr = CeedBasisDestroy(&basisin); PCeedChk(ierr);
   ierr = CeedBasisDestroy(&basisout); PCeedChk(ierr);
   ierr = CeedHackFree(&emodein); PCeedChk(ierr);
   ierr = CeedHackFree(&emodeout); PCeedChk(ierr);

   return 0;
}

int CeedOperatorFullAssemble(CeedOperator op, SparseMatrix **mat)
{
   int ierr;

   CeedSize in_len, out_len;
   ierr = CeedOperatorGetActiveVectorLengths(op, &in_len, &out_len);
   PCeedChk(ierr);
   const int nnodes = in_len;
   MFEM_VERIFY(in_len == out_len, "not a square CeedOperator");
   MFEM_VERIFY(in_len == nnodes, "size overflow");

   SparseMatrix *out = new SparseMatrix(nnodes, nnodes);

   bool isComposite;
   ierr = CeedOperatorIsComposite(op, &isComposite); PCeedChk(ierr);
   if (isComposite)
   {
      CeedInt numsub;
      CeedOperator *subops;
      ierr = CeedOperatorCompositeGetNumSub(op, &numsub); PCeedChk(ierr);
      ierr = CeedOperatorCompositeGetSubList(op, &subops); PCeedChk(ierr);
      for (int i = 0; i < numsub; ++i)
      {
         ierr = CeedSingleOperatorFullAssemble(subops[i], out); PCeedChk(ierr);
      }
   }
   else
   {
      ierr = CeedSingleOperatorFullAssemble(op, out); PCeedChk(ierr);
   }
   // enforce structurally symmetric for later elimination
   const int skip_zeros = 0;
   out->Finalize(skip_zeros);
   *mat = out;

   return 0;
}

} // namespace ceed

} // namespace mfem

#endif // MFEM_USE_CEED
