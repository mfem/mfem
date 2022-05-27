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

#include "full-assembly.hpp"

#ifdef MFEM_USE_CEED
#include "util.hpp"

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
   ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);

   // Assemble QFunction
   CeedQFunction qf;
   ierr = CeedOperatorGetQFunction(op, &qf); CeedChk(ierr);
   CeedInt numinputfields, numoutputfields;
   ierr= CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
   CeedChk(ierr);
   CeedVector assembledqf;
   CeedElemRestriction rstr_q;
   ierr = CeedOperatorLinearAssembleQFunction(
             op, &assembledqf, &rstr_q, CEED_REQUEST_IMMEDIATE); CeedChk(ierr);

   CeedInt qflength;
   ierr = CeedVectorGetLength(assembledqf, &qflength); CeedChk(ierr);

   CeedOperatorField *input_fields;
   CeedOperatorField *output_fields;
   ierr = CeedOperatorGetFields(op, &input_fields, &output_fields); CeedChk(ierr);

   // Determine active input basis
   CeedQFunctionField *qffields;
   ierr = CeedQFunctionGetFields(qf, &qffields, NULL); CeedChk(ierr);
   CeedInt numemodein = 0, ncomp, dim = 1;
   CeedEvalMode *emodein = NULL;
   CeedBasis basisin = NULL;
   CeedElemRestriction rstrin = NULL;
   for (CeedInt i=0; i<numinputfields; i++)
   {
      CeedVector vec;
      ierr = CeedOperatorFieldGetVector(input_fields[i], &vec); CeedChk(ierr);
      if (vec == CEED_VECTOR_ACTIVE)
      {
         ierr = CeedOperatorFieldGetBasis(input_fields[i], &basisin);
         CeedChk(ierr);
         ierr = CeedBasisGetNumComponents(basisin, &ncomp); CeedChk(ierr);
         ierr = CeedBasisGetDimension(basisin, &dim); CeedChk(ierr);
         ierr = CeedOperatorFieldGetElemRestriction(input_fields[i], &rstrin);
         CeedChk(ierr);
         CeedEvalMode emode;
         ierr = CeedQFunctionFieldGetEvalMode(qffields[i], &emode);
         CeedChk(ierr);
         switch (emode)
         {
            case CEED_EVAL_NONE:
            case CEED_EVAL_INTERP:
               ierr = CeedHackRealloc(numemodein + 1, &emodein); CeedChk(ierr);
               emodein[numemodein] = emode;
               numemodein += 1;
               break;
            case CEED_EVAL_GRAD:
               ierr = CeedHackRealloc(numemodein + dim, &emodein); CeedChk(ierr);
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
   }

   // Determine active output basis
   ierr = CeedQFunctionGetFields(qf, NULL, &qffields); CeedChk(ierr);
   CeedInt numemodeout = 0;
   CeedEvalMode *emodeout = NULL;
   CeedBasis basisout = NULL;
   CeedElemRestriction rstrout = NULL;
   for (CeedInt i=0; i<numoutputfields; i++)
   {
      CeedVector vec;
      ierr = CeedOperatorFieldGetVector(output_fields[i], &vec); CeedChk(ierr);
      if (vec == CEED_VECTOR_ACTIVE)
      {
         ierr = CeedOperatorFieldGetBasis(output_fields[i], &basisout);
         CeedChk(ierr);
         ierr = CeedOperatorFieldGetElemRestriction(output_fields[i], &rstrout);
         CeedChk(ierr);
         CeedChk(ierr);
         CeedEvalMode emode;
         ierr = CeedQFunctionFieldGetEvalMode(qffields[i], &emode);
         CeedChk(ierr);
         switch (emode)
         {
            case CEED_EVAL_NONE:
            case CEED_EVAL_INTERP:
               ierr = CeedHackRealloc(numemodeout + 1, &emodeout); CeedChk(ierr);
               emodeout[numemodeout] = emode;
               numemodeout += 1;
               break;
            case CEED_EVAL_GRAD:
               ierr = CeedHackRealloc(numemodeout + dim, &emodeout); CeedChk(ierr);
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
   }

   CeedInt nnodes, nelem, elemsize, nqpts;
   ierr = CeedElemRestrictionGetNumElements(rstrin, &nelem); CeedChk(ierr);
   ierr = CeedElemRestrictionGetElementSize(rstrin, &elemsize); CeedChk(ierr);
   ierr = CeedElemRestrictionGetLVectorSize(rstrin, &nnodes); CeedChk(ierr);
   ierr = CeedBasisGetNumQuadraturePoints(basisin, &nqpts); CeedChk(ierr);

   // Determine elem_dof relation
   CeedVector index_vec;
   ierr = CeedVectorCreate(ceed, nnodes, &index_vec); CeedChk(ierr);
   CeedScalar *array;
   ierr = CeedVectorGetArray(index_vec, CEED_MEM_HOST, &array); CeedChk(ierr);
   for (CeedInt i = 0; i < nnodes; ++i)
   {
      array[i] = i;
   }
   ierr = CeedVectorRestoreArray(index_vec, &array); CeedChk(ierr);
   CeedVector elem_dof;
   ierr = CeedVectorCreate(ceed, nelem * elemsize, &elem_dof); CeedChk(ierr);
   ierr = CeedVectorSetValue(elem_dof, 0.0); CeedChk(ierr);
   CeedElemRestrictionApply(rstrin, CEED_NOTRANSPOSE, index_vec,
                            elem_dof, CEED_REQUEST_IMMEDIATE); CeedChk(ierr);
   const CeedScalar * elem_dof_a;
   ierr = CeedVectorGetArrayRead(elem_dof, CEED_MEM_HOST, &elem_dof_a);
   CeedChk(ierr);
   ierr = CeedVectorDestroy(&index_vec); CeedChk(ierr);

   // loop over elements and put in SparseMatrix
   // SparseMatrix * out = new SparseMatrix(nnodes, nnodes);
   MFEM_ASSERT(out->Height() == nnodes, "Sizes don't match!");
   MFEM_ASSERT(out->Width() == nnodes, "Sizes don't match!");
   const CeedScalar *interpin, *gradin;
   ierr = CeedBasisGetInterp(basisin, &interpin); CeedChk(ierr);
   ierr = CeedBasisGetGrad(basisin, &gradin); CeedChk(ierr);

   const CeedScalar * assembledqfarray;
   ierr = CeedVectorGetArrayRead(assembledqf, CEED_MEM_HOST, &assembledqfarray);
   CeedChk(ierr);

   CeedInt layout[3];
   ierr = CeedElemRestrictionGetELayout(rstr_q, &layout); CeedChk(ierr);
   ierr = CeedElemRestrictionDestroy(&rstr_q); CeedChk(ierr);

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

   ierr = CeedVectorRestoreArrayRead(elem_dof, &elem_dof_a); CeedChk(ierr);
   ierr = CeedVectorDestroy(&elem_dof); CeedChk(ierr);
   ierr = CeedVectorRestoreArrayRead(assembledqf, &assembledqfarray);
   CeedChk(ierr);
   ierr = CeedVectorDestroy(&assembledqf); CeedChk(ierr);
   ierr = CeedHackFree(&emodein); CeedChk(ierr);
   ierr = CeedHackFree(&emodeout); CeedChk(ierr);

   return 0;
}

int CeedOperatorFullAssemble(CeedOperator op, SparseMatrix **mat)
{
   int ierr;

   CeedElemRestriction er;
   ierr = CeedOperatorGetActiveElemRestriction(op, &er); CeedChk(ierr);
   CeedInt nnodes;
   ierr = CeedElemRestrictionGetLVectorSize(er, &nnodes); CeedChk(ierr);

   SparseMatrix *out = new SparseMatrix(nnodes, nnodes);

   bool isComposite;
   ierr = CeedOperatorIsComposite(op, &isComposite); CeedChk(ierr);
   if (isComposite)
   {
      CeedInt numsub;
      CeedOperator *subops;
      CeedOperatorGetNumSub(op, &numsub);
      ierr = CeedOperatorGetSubList(op, &subops); CeedChk(ierr);
      for (int i = 0; i < numsub; ++i)
      {
         ierr = CeedSingleOperatorFullAssemble(subops[i], out); CeedChk(ierr);
      }
   }
   else
   {
      ierr = CeedSingleOperatorFullAssemble(op, out); CeedChk(ierr);
   }
   // enforce structurally symmetric for later elimination
   const int skip_zeros = 0;
   out->Finalize(skip_zeros);
   *mat = out;

   return 0;
}

} // namespace ceed

} // namespace mfem

#endif
