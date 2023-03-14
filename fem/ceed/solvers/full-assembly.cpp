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

#include "full-assembly.hpp"

#ifdef MFEM_USE_CEED
#include <ceed/backend.h>
#endif

#ifdef MFEM_USE_CEED

namespace mfem
{

namespace ceed
{

int CeedInternalReallocArray(size_t n, size_t unit, void *p)
{
   *(void **)p = realloc(*(void **)p, n*unit);
   if (n && unit && !*(void **)p)
      return CeedError(NULL, 1, "realloc failed to allocate %zd members of size "
                       "%zd\n", n, unit);
   return 0;
}

#define CeedInternalRealloc(n, p) CeedInternalReallocArray((n), sizeof(**(p)), p)

int CeedInternalFree(void *p)
{
   free(*(void **)p);
   *(void **)p = NULL;
   return 0;
}

int CeedSingleOperatorFullAssemble(CeedOperator op, SparseMatrix &out)
{
   int ierr;
   Ceed ceed;
   ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);

   // Assemble QFunction
   CeedQFunction qf;
   ierr = CeedOperatorGetQFunction(op, &qf); CeedChk(ierr);
   CeedInt num_input_fields, num_output_fields; CeedChk(ierr);
   CeedVector assembled_qf;
   CeedElemRestriction rstr_q;
   ierr = CeedOperatorLinearAssembleQFunction(op, &assembled_qf, &rstr_q,
                                              CEED_REQUEST_IMMEDIATE); CeedChk(ierr);

   CeedSize qf_length;
   ierr = CeedVectorGetLength(assembled_qf, &qf_length); CeedChk(ierr);

   CeedOperatorField *input_fields;
   CeedOperatorField *output_fields;
   ierr = CeedOperatorGetFields(op, &num_input_fields, &input_fields,
                                &num_output_fields, &output_fields); CeedChk(ierr);

   // Determine active input basis
   CeedQFunctionField *qf_fields;
   ierr = CeedQFunctionGetFields(qf, &num_input_fields, &qf_fields,
                                 &num_output_fields, NULL); CeedChk(ierr);
   CeedInt num_emode_in = 0, dim, ncomp, qcomp;
   CeedEvalMode *emode_in = NULL;
   CeedBasis basis_in = NULL;
   CeedElemRestriction rstr_in = NULL;
   for (CeedInt i = 0; i < num_input_fields; i++)
   {
      CeedVector vec;
      ierr = CeedOperatorFieldGetVector(input_fields[i], &vec); CeedChk(ierr);
      if (vec == CEED_VECTOR_ACTIVE)
      {
         CeedEvalMode emode;
         ierr = CeedQFunctionFieldGetEvalMode(qf_fields[i], &emode); CeedChk(ierr);
         ierr = CeedOperatorFieldGetBasis(input_fields[i], &basis_in); CeedChk(ierr);
         ierr = CeedOperatorFieldGetElemRestriction(input_fields[i], &rstr_in);
         CeedChk(ierr);
         ierr = CeedBasisGetDimension(basis_in, &dim); CeedChk(ierr);
         ierr = CeedBasisGetNumComponents(basis_in, &ncomp); CeedChk(ierr);
         ierr = CeedBasisGetNumQuadratureComponents(basis_in, emode, &qcomp);
         CeedChk(ierr);
         if (emode != CEED_EVAL_WEIGHT)
         {
            ierr = CeedInternalRealloc(num_emode_in + qcomp, &emode_in); CeedChk(ierr);
            for (CeedInt d = 0; d < qcomp; d++)
            {
               emode_in[num_emode_in + d] = emode;
            }
            num_emode_in += qcomp;
         }
      }
   }

   // Determine active output basis
   ierr = CeedQFunctionGetFields(qf, &num_input_fields, NULL, &num_output_fields,
                                 &qf_fields); CeedChk(ierr);
   CeedInt num_emode_out = 0;
   CeedEvalMode *emode_out = NULL;
   CeedBasis basis_out = NULL;
   CeedElemRestriction rstr_out = NULL;
   for (CeedInt i = 0; i < num_output_fields; i++)
   {
      CeedVector vec;
      ierr = CeedOperatorFieldGetVector(output_fields[i], &vec); CeedChk(ierr);
      if (vec == CEED_VECTOR_ACTIVE)
      {
         CeedEvalMode emode;
         ierr = CeedQFunctionFieldGetEvalMode(qf_fields[i], &emode); CeedChk(ierr);
         ierr = CeedOperatorFieldGetBasis(output_fields[i], &basis_out); CeedChk(ierr);
         ierr = CeedOperatorFieldGetElemRestriction(output_fields[i], &rstr_out);
         CeedChk(ierr);
         ierr = CeedBasisGetNumQuadratureComponents(basis_out, emode, &qcomp);
         CeedChk(ierr);
         if (emode != CEED_EVAL_WEIGHT)
         {
            ierr = CeedInternalRealloc(num_emode_out + qcomp, &emode_out); CeedChk(ierr);
            for (CeedInt d = 0; d < qcomp; d++)
            {
               emode_out[num_emode_out + d] = emode;
            }
            num_emode_out += qcomp;
         }
      }
   }

   CeedInt nelem, elemsize, nqpts;
   CeedSize nnodes;
   ierr = CeedElemRestrictionGetNumElements(rstr_in, &nelem); CeedChk(ierr);
   ierr = CeedElemRestrictionGetElementSize(rstr_in, &elemsize); CeedChk(ierr);
   ierr = CeedElemRestrictionGetLVectorSize(rstr_in, &nnodes); CeedChk(ierr);
   ierr = CeedBasisGetNumQuadraturePoints(basis_in, &nqpts); CeedChk(ierr);

   // Determine elem_dof relation
   CeedVector index_vec;
   ierr = CeedVectorCreate(ceed, nnodes, &index_vec); CeedChk(ierr);
   CeedScalar *array;
   ierr = CeedVectorGetArrayWrite(index_vec, CEED_MEM_HOST, &array); CeedChk(ierr);
   for (CeedSize i = 0; i < nnodes; ++i)
   {
      array[i] = i;
   }
   ierr = CeedVectorRestoreArray(index_vec, &array); CeedChk(ierr);
   CeedVector elem_dof;
   ierr = CeedVectorCreate(ceed, nelem * elemsize, &elem_dof); CeedChk(ierr);
   ierr = CeedVectorSetValue(elem_dof, 0.0); CeedChk(ierr);
   CeedElemRestrictionApply(rstr_in, CEED_NOTRANSPOSE, index_vec,
                            elem_dof, CEED_REQUEST_IMMEDIATE); CeedChk(ierr);
   const CeedScalar *elem_dof_a;
   ierr = CeedVectorGetArrayRead(elem_dof, CEED_MEM_HOST, &elem_dof_a);
   CeedChk(ierr);
   ierr = CeedVectorDestroy(&index_vec); CeedChk(ierr);

   // loop over elements and put in SparseMatrix
   MFEM_ASSERT(out.Height() == nnodes, "Sizes don't match!");
   MFEM_ASSERT(out.Width() == nnodes, "Sizes don't match!");

   const CeedScalar *assembled_qf_array;
   ierr = CeedVectorGetArrayRead(assembled_qf, CEED_MEM_HOST, &assembled_qf_array);
   CeedChk(ierr);

   CeedInt layout[3];
   ierr = CeedElemRestrictionGetELayout(rstr_q, &layout); CeedChk(ierr);
   ierr = CeedElemRestrictionDestroy(&rstr_q); CeedChk(ierr);

   // Enforce structurally symmetric for later elimination
   const int skip_zeros = 0;
   MFEM_ASSERT(num_emode_in == num_emode_out,
               "Ceed full assembly not implemented for this case.");
   for (int e = 0; e < nelem; ++e)
   {
      // Get Array<int> for use in SparseMatrix::AddSubMatrix()
      Array<int> rows(elemsize);
      for (int i = 0; i < elemsize; ++i)
      {
         rows[i] = elem_dof_a[e * elemsize + i];
      }

      // Form element matrix itself, storing block-diagonal D matrix as a
      // collection of small dense blocks
      DenseTensor Dmat(num_emode_out, num_emode_in, nqpts);
      Dmat = 0.0;
      DenseMatrix Bmat(nqpts * num_emode_in, elemsize);
      Bmat = 0.0;
      for (int q = 0; q < nqpts; ++q)
      {
         for (int n = 0; n < elemsize; ++n)
         {
            CeedInt d_in = 0;
            CeedEvalMode emode_prev = CEED_EVAL_NONE;
            for (int e_in = 0; e_in < num_emode_in; ++e_in)
            {
               const CeedScalar *B;
               switch (emode_in[e_in])
               {
                  case CEED_EVAL_INTERP:
                     ierr = CeedBasisGetInterp(basis_in, &B); CeedChk(ierr);
                     break;
                  case CEED_EVAL_GRAD:
                     ierr = CeedBasisGetGrad(basis_in, &B); CeedChk(ierr);
                     break;
                  case CEED_EVAL_DIV:
                     ierr = CeedBasisGetDiv(basis_in, &B); CeedChk(ierr);
                     break;
                  case CEED_EVAL_CURL:
                     ierr = CeedBasisGetCurl(basis_in, &B); CeedChk(ierr);
                     break;
                  case CEED_EVAL_NONE:
                  default:
                     MFEM_ABORT("CeedEvalMode is not implemented.");
               }
               ierr = CeedBasisGetNumQuadratureComponents(basis_in, emode_in[e_in], &qcomp);
               CeedChk(ierr);
               if (qcomp > 1)
               {
                  if (e_in == 0 || emode_in[e_in] != emode_prev)
                  {
                     d_in = 0;
                  }
                  else
                  {
                     d_in++;
                  }
               }
               Bmat(num_emode_in * q + e_in, n) += B[(d_in * nqpts + q) * elemsize + n];
               emode_prev = emode_in[e_in];
            }
         }
         for (int ei = 0; ei < num_emode_in; ++ei)
         {
            for (int ej = 0; ej < num_emode_in; ++ej)
            {
               const int comp = ei * num_emode_in + ej;
               const int index = q * layout[0] + comp * layout[1] + e * layout[2];
               Dmat(ei, ej, q) += assembled_qf_array[index];
            }
         }
      }

      // Compute B^T * D
      DenseMatrix BtD(elemsize, nqpts * num_emode_in);
      BtD = 0.0;
      for (int j = 0; j < elemsize; ++j)
      {
         for (int q = 0; q < nqpts; ++q)
         {
            int qq = num_emode_in * q;
            for (int ei = 0; ei < num_emode_in; ++ei)
            {
               for (int ej = 0; ej < num_emode_in; ++ej)
               {
                  BtD(j, qq + ei) += Bmat(qq + ej, j) * Dmat(ej, ei, q);
               }
            }
         }
      }

      // Put element matrix in sparsemat
      DenseMatrix elem_mat(elemsize, elemsize);
      elem_mat = 0.0;
      Mult(BtD, Bmat, elem_mat);
      out.AddSubMatrix(rows, rows, elem_mat, skip_zeros);
   }

   ierr = CeedVectorRestoreArrayRead(elem_dof, &elem_dof_a); CeedChk(ierr);
   ierr = CeedVectorDestroy(&elem_dof); CeedChk(ierr);
   ierr = CeedVectorRestoreArrayRead(assembled_qf, &assembled_qf_array);
   CeedChk(ierr);
   ierr = CeedVectorDestroy(&assembled_qf); CeedChk(ierr);
   ierr = CeedInternalFree(&emode_in); CeedChk(ierr);
   ierr = CeedInternalFree(&emode_out); CeedChk(ierr);

   return 0;
}

int CeedOperatorFullAssemble(CeedOperator op, SparseMatrix **mat)
{
   int ierr;

   CeedSize in_len, out_len;
   ierr = CeedOperatorGetActiveVectorLengths(op, &in_len, &out_len); CeedChk(ierr);
   const int nnodes = in_len;
   MFEM_VERIFY(in_len == out_len, "Not a square CeedOperator.");
   MFEM_VERIFY(in_len == nnodes, "size overflow");

   SparseMatrix *out = new SparseMatrix(nnodes, nnodes);

   bool isComposite;
   ierr = CeedOperatorIsComposite(op, &isComposite); CeedChk(ierr);
   if (isComposite)
   {
      CeedInt numsub;
      CeedOperator *subops;
#if CEED_VERSION_GE(0, 10, 2)
      CeedCompositeOperatorGetNumSub(op, &numsub);
      ierr = CeedCompositeOperatorGetSubList(op, &subops); CeedChk(ierr);
#else
      CeedOperatorGetNumSub(op, &numsub);
      ierr = CeedOperatorGetSubList(op, &subops); CeedChk(ierr);
#endif
      for (int i = 0; i < numsub; ++i)
      {
         ierr = CeedSingleOperatorFullAssemble(subops[i], *out); CeedChk(ierr);
      }
   }
   else
   {
      ierr = CeedSingleOperatorFullAssemble(op, *out); CeedChk(ierr);
   }
   // Enforce structurally symmetric for later elimination
   const int skip_zeros = 0;
   out->Finalize(skip_zeros);
   *mat = out;

   return 0;
}

} // namespace ceed

} // namespace mfem

#endif
