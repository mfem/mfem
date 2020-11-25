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

#include "ceedsolvers-utility.h"

#include "../../general/forall.hpp"
using namespace mfem;

#ifdef MFEM_USE_CEED

/// a = a (pointwise*) b
/// @todo: using MPI_FORALL in this Ceed-like function is ugly
int CeedVectorPointwiseMult(CeedVector a, const CeedVector b)
{
   int ierr;
   Ceed ceed;
   CeedVectorGetCeed(a, &ceed);

   int length, length2;
   ierr = CeedVectorGetLength(a, &length); CeedChk(ierr);
   ierr = CeedVectorGetLength(b, &length2); CeedChk(ierr);
   if (length != length2)
   {
      return CeedError(ceed, 1, "Vector sizes don't match");
   }

   CeedMemType mem;
   if (Device::Allows(Backend::DEVICE_MASK))
   {
      mem = CEED_MEM_DEVICE;
   }
   else
   {
      mem = CEED_MEM_HOST;
   }
   CeedScalar *a_data;
   const CeedScalar *b_data;
   ierr = CeedVectorGetArray(a, mem, &a_data); CeedChk(ierr);
   ierr = CeedVectorGetArrayRead(b, mem, &b_data); CeedChk(ierr);
   MFEM_FORALL(i, length,
   {a_data[i] *= b_data[i];});

   ierr = CeedVectorRestoreArray(a, &a_data); CeedChk(ierr);
   ierr = CeedVectorRestoreArrayRead(b, &b_data); CeedChk(ierr);

   return 0;
}

/// assumes a tensor-product operator with one active field
int CeedOperatorGetActiveField(CeedOperator oper, CeedOperatorField *field)
{
   int ierr;
   Ceed ceed;
   ierr = CeedOperatorGetCeed(oper, &ceed); CeedChk(ierr);

   CeedQFunction qf;
   bool isComposite;
   ierr = CeedOperatorIsComposite(oper, &isComposite); CeedChk(ierr);
   CeedOperator *subops;
   if (isComposite)
   {
      ierr = CeedOperatorGetSubList(oper, &subops); CeedChk(ierr);
      ierr = CeedOperatorGetQFunction(subops[0], &qf); CeedChk(ierr);
   }
   else
   {
      ierr = CeedOperatorGetQFunction(oper, &qf); CeedChk(ierr);
   }
   CeedInt numinputfields, numoutputfields;
   ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
   CeedOperatorField *inputfields;
   if (isComposite)
   {
      ierr = CeedOperatorGetFields(subops[0], &inputfields, NULL); CeedChk(ierr);
   }
   else
   {
      ierr = CeedOperatorGetFields(oper, &inputfields, NULL); CeedChk(ierr);
   }

   CeedVector if_vector;
   bool found = false;
   int found_index = -1;
   for (int i = 0; i < numinputfields; ++i)
   {
      ierr = CeedOperatorFieldGetVector(inputfields[i], &if_vector); CeedChk(ierr);
      if (if_vector == CEED_VECTOR_ACTIVE)
      {
         if (found)
         {
            return CeedError(ceed, 1, "Multiple active vectors in CeedOperator!");
         }
         found = true;
         found_index = i;
      }
   }
   if (!found)
   {
      return CeedError(ceed, 1, "No active vector in CeedOperator!");
   }
   *field = inputfields[found_index];

   return 0;
}

/// (a better design splits this into CeedOperatorGetActiveBasis() and
/// CeedOperatorGetOrder, which calls the basis one)
/// TODO: unit test
int CeedOperatorGetOrder(CeedOperator oper, CeedInt * order)
{
   int ierr;

   CeedOperatorField active_field;
   ierr = CeedOperatorGetActiveField(oper, &active_field); CeedChk(ierr);
   CeedBasis basis;
   ierr = CeedOperatorFieldGetBasis(active_field, &basis); CeedChk(ierr);
   int P1d;
   ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChk(ierr);
   *order = P1d - 1;

   return 0;
}

int CeedOperatorGetActiveElemRestriction(CeedOperator oper,
                                         CeedElemRestriction* restr_out)
{
   int ierr;

   CeedOperatorField active_field;
   ierr = CeedOperatorGetActiveField(oper, &active_field); CeedChk(ierr);
   CeedElemRestriction er;
   ierr = CeedOperatorFieldGetElemRestriction(active_field, &er); CeedChk(ierr);
   *restr_out = er;

   return 0;
}

/// assumes a square operator (you could do rectangular, you'd have
/// to find separate active input and output fields/restrictions)
int CeedOperatorGetSize(CeedOperator oper, CeedInt * size)
{
   int ierr;
   CeedElemRestriction er;
   ierr = CeedOperatorGetActiveElemRestriction(oper, &er); CeedChk(ierr);
   ierr = CeedElemRestrictionGetLVectorSize(er, size); CeedChk(ierr);
   return 0;
}

/** Just like CeedBasisCreateTensorH1Lagrange but with Legendre dofs
    (basically only one line changed, which seems wasteful) */
int CeedBasisCreateTensorH1Gauss(Ceed ceed, CeedInt dim, CeedInt ncomp,
                                 CeedInt P, CeedInt Q, CeedQuadMode qmode,
                                 CeedBasis *basis) 
{
   // Allocate
   int ierr, i, j, k;
   CeedScalar c1, c2, c3, c4, dx, *nodes, *interp1d, *grad1d, 
      *qref1d, *qweight1d, *dummyweights;

   if (dim<1)
   {
      return CeedError(ceed, 1, "Basis dimension must be a positive value");
   }

   // CeedCalloc replaced below
   interp1d = (CeedScalar*) calloc(P * Q, sizeof(CeedScalar));
   grad1d = (CeedScalar*) calloc(P * Q, sizeof(CeedScalar));
   nodes = (CeedScalar*) calloc(P, sizeof(CeedScalar));
   dummyweights = (CeedScalar*) calloc(P, sizeof(CeedScalar));
   qref1d = (CeedScalar*) calloc(Q, sizeof(CeedScalar));
   qweight1d = (CeedScalar*) calloc(Q, sizeof(CeedScalar));

   // Get Nodes and Weights
   ierr = CeedGaussQuadrature(P, nodes, dummyweights); CeedChk(ierr);
   switch (qmode)
   {
   case CEED_GAUSS:
      ierr = CeedGaussQuadrature(Q, qref1d, qweight1d); CeedChk(ierr);
      break;
   case CEED_GAUSS_LOBATTO:
      ierr = CeedLobattoQuadrature(Q, qref1d, qweight1d); CeedChk(ierr);
      break;
   }
   // Build B, D matrix
   // Fornberg, 1998
   for (i = 0; i  < Q; i++)
   {
      c1 = 1.0;
      c3 = nodes[0] - qref1d[i];
      interp1d[i*P+0] = 1.0;
      for (j = 1; j < P; j++)
      {
         c2 = 1.0;
         c4 = c3;
         c3 = nodes[j] - qref1d[i];
         for (k = 0; k < j; k++)
         {
            dx = nodes[j] - nodes[k];
            c2 *= dx;
            if (k == j - 1)
            {
               grad1d[i*P + j] = c1*(interp1d[i*P + k] - c4*grad1d[i*P + k]) / c2;
               interp1d[i*P + j] = - c1*c4*interp1d[i*P + k] / c2;
            }
            grad1d[i*P + k] = (c3*grad1d[i*P + k] - interp1d[i*P + k]) / dx;
            interp1d[i*P + k] = c3*interp1d[i*P + k] / dx;
         }
         c1 = c2;
      }
   }
   //  // Pass to CeedBasisCreateTensorH1
   ierr = CeedBasisCreateTensorH1(ceed, dim, ncomp, P, Q, interp1d, grad1d, qref1d,
                                  qweight1d, basis); CeedChk(ierr);
   /*
     ierr = CeedFree(&interp1d); CeedChk(ierr);
     ierr = CeedFree(&grad1d); CeedChk(ierr);
     ierr = CeedFree(&nodes); CeedChk(ierr);
     ierr = CeedFree(&dummyweights); CeedChk(ierr);
     ierr = CeedFree(&qref1d); CeedChk(ierr);
     ierr = CeedFree(&qweight1d); CeedChk(ierr);
   */
   free(interp1d);
   free(grad1d);
   free(nodes);
   free(dummyweights);
   free(qref1d);
   free(qweight1d);
   return 0;
}

/** Ugly hacky copy/paste from CeedBasisCreateTensorH1Lagrange to deal with
    different reference elements for MFEM/Ceed.

    The correct way to do this is to actually interpolate, but I am lazy. */
int CeedBasisCreateMFEMTensorH1Lagrange(Ceed ceed, CeedInt dim, CeedInt ncomp,
                                        CeedInt P, CeedInt Q, CeedQuadMode qmode,
                                        CeedBasis *basis)
{
   // Allocate
   int ierr, i, j, k;
   CeedScalar c1, c2, c3, c4, dx, *nodes, *interp1d, *grad1d, *qref1d, *qweight1d;

   if (dim<1)
   {
      return CeedError(ceed, 1, "Basis dimension must be a positive value");
   }

   interp1d = (CeedScalar*) calloc(P*Q, sizeof(CeedScalar));
   grad1d = (CeedScalar*) calloc(P*Q, sizeof(CeedScalar));
   nodes = (CeedScalar*) calloc(P, sizeof(CeedScalar));
   qref1d = (CeedScalar*) calloc(Q, sizeof(CeedScalar));
   qweight1d = (CeedScalar*) calloc(Q, sizeof(CeedScalar));

   // Get Nodes and Weights
   ierr = CeedLobattoQuadrature(P, nodes, NULL); CeedChk(ierr);
   switch (qmode)
   {
   case CEED_GAUSS:
      ierr = CeedGaussQuadrature(Q, qref1d, qweight1d); CeedChk(ierr);
      break;
   case CEED_GAUSS_LOBATTO:
      ierr = CeedLobattoQuadrature(Q, qref1d, qweight1d); CeedChk(ierr);
      break;
   }

   /// modification for MFEM reference element
   for (int j = 0; j < P; ++j)
   {
      nodes[j] = 0.5 + 0.5*nodes[j];
   }
   for (int q = 0; q < Q; ++q)
   {
      qref1d[q] = 0.5 + 0.5*qref1d[q];
      qweight1d[q] *= 0.5;
   }

   // Build B, D matrix
   // Fornberg, 1998
   for (i = 0; i < Q; i++)
   {
      c1 = 1.0;
      c3 = nodes[0] - qref1d[i];
      interp1d[i*P+0] = 1.0;
      for (j = 1; j < P; j++)
      {
         c2 = 1.0;
         c4 = c3;
         c3 = nodes[j] - qref1d[i];
         for (k = 0; k < j; k++)
         {
            dx = nodes[j] - nodes[k];
            c2 *= dx;
            if (k == j - 1)
            {
               grad1d[i*P + j] = c1*(interp1d[i*P + k] - c4*grad1d[i*P + k]) / c2;
               interp1d[i*P + j] = - c1*c4*interp1d[i*P + k] / c2;
            }
            grad1d[i*P + k] = (c3*grad1d[i*P + k] - interp1d[i*P + k]) / dx;
            interp1d[i*P + k] = c3*interp1d[i*P + k] / dx;
         }
         c1 = c2;
      }
   }
   //  // Pass to CeedBasisCreateTensorH1
   ierr = CeedBasisCreateTensorH1(ceed, dim, ncomp, P, Q, interp1d, grad1d, qref1d,
                                  qweight1d, basis); CeedChk(ierr);
  
   /*
     ierr = CeedFree(&interp1d); CeedChk(ierr);
     ierr = CeedFree(&grad1d); CeedChk(ierr);
     ierr = CeedFree(&nodes); CeedChk(ierr);
     ierr = CeedFree(&qref1d); CeedChk(ierr);
     ierr = CeedFree(&qweight1d); CeedChk(ierr);
   */
   free(interp1d);
   free(grad1d);
   free(nodes);
   free(qref1d);
   free(qweight1d);
   return 0;
}

int CeedOperatorGetActiveBasis(CeedOperator oper, CeedBasis *basis)
{
   int ierr;
   Ceed ceed;
   ierr = CeedOperatorGetCeed(oper, &ceed); CeedChk(ierr);
   CeedQFunction qf;
   ierr = CeedOperatorGetQFunction(oper, &qf); CeedChk(ierr);
   CeedInt numinputfields, numoutputfields;
   ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
   CeedOperatorField *inputfields, *outputfields;
   ierr = CeedOperatorGetFields(oper, &inputfields, &outputfields); CeedChk(ierr);

   *basis = NULL;
   for (int i = 0; i < numinputfields; ++i)
   {
      CeedVector if_vector;
      CeedBasis basis_in;
      ierr = CeedOperatorFieldGetVector(inputfields[i], &if_vector); CeedChk(ierr);
      ierr = CeedOperatorFieldGetBasis(inputfields[i], &basis_in); CeedChk(ierr);
      if (if_vector == CEED_VECTOR_ACTIVE)
      {
         if (*basis == NULL)
         {
            *basis = basis_in;
         }
         else if (*basis != basis_in)
         {
            return CeedError(ceed, 1, "Two different active input basis!");
         }
      }
   }
   for (int i = 0; i < numoutputfields; ++i)
   {
      CeedVector of_vector;
      CeedBasis basis_out;
      ierr = CeedOperatorFieldGetVector(outputfields[i], &of_vector); CeedChk(ierr);
      ierr = CeedOperatorFieldGetBasis(outputfields[i], &basis_out); CeedChk(ierr);
      if (of_vector == CEED_VECTOR_ACTIVE)
      {
         if (*basis != basis_out)
         {
            return CeedError(ceed, 1, "Input and output basis do not match!");
         }
      }
   }
   return 0;
}

#endif // MFEM_USE_CEED
