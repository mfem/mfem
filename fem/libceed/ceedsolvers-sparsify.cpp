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

#include "ceedsolvers-sparsify.h"
#include "ceedsolvers-utility.h"

#ifdef MFEM_USE_CEED
#include <ceed-backend.h>

#include <math.h>
// todo: should probably use Ceed memory wrappers instead of calloc/free?
#include <stdlib.h>

typedef struct {
  int index;
  double basis_value;
  double grad_value;
} basis_element;

/// for C (not C++) sorting using stdlib.h qsort()
/// sort by basis_value
int basis_element_compare_abs(const void *a, const void *b) {
  basis_element *x = (basis_element *) a;
  basis_element *y = (basis_element *) b;

  // it is actually not so clear that absolute comparison
  // is the right thing here
  double xv = fabs(x->basis_value);
  double yv = fabs(y->basis_value);
  if (xv > yv) {
    return  1;
  }
  if (xv < yv) {
    return -1;
  }
  return 0;
}

/// sort by grad_value
int basis_element_compare_grad_abs(const void *a, const void *b) {
  basis_element *x = (basis_element *) a;
  basis_element *y = (basis_element *) b;

  double xv = fabs(x->grad_value);
  double yv = fabs(y->grad_value);
  if (xv > yv) {
    return  1;
  }
  if (xv < yv) {
    return -1;
  }
  return 0;
}

int basis_element_compare(const void *a, const void *b) {
  basis_element *x = (basis_element *) a;
  basis_element *y = (basis_element *) b;

  double xv = x->basis_value;
  double yv = y->basis_value;
  if (xv > yv) {
    return  1;
  }
  if (xv < yv) {
    return -1;
  }
  return 0;
}

/// this guy puts the largest element of row at the end,
/// and then the ones closest to it in *index* next to it
/// at the end
int near_largest(basis_element * row, int size) {
  if (size < 4) return 0;

  int largest_index = -1;
  double largest_value = -1.e+10;
  for (int i = 0; i < size; i++) {
    double v = row[i].basis_value;
    if (v > largest_value) {
      largest_value = v;
      largest_index = i;
    }
  }
  int near_index_a = largest_index - 1;
  if (near_index_a < 0) {
    near_index_a = largest_index + 2;
  }
  int near_index_b = largest_index + 1;
  if (near_index_b > size - 1) {
    near_index_b = largest_index - 2;
  }

  basis_element temp = row[size - 1];
  row[size - 1] = row[largest_index];
  row[largest_index] = temp;

  temp = row[size - 2];
  row[size - 2] = row[near_index_a];
  row[near_index_a] = temp;
  
  temp = row[size - 3];
  row[size - 3] = row[near_index_b];
  row[near_index_b] = temp;

  return 0;
}

typedef struct {
  double dist;
  int index;
} distindex;

int compare_distindex(const void *a, const void *b) {
  distindex *x = (distindex *) a;
  distindex *y = (distindex *) b;

  if (x->dist > y->dist) {
    return  1;
  }
  if (x->dist < y->dist) {
    return -1;
  }
  return 0;
}

/**
   Replacement for near_largest, assume Gauss-Lobatto for nodes and
   Gauss-Legendre for quad points, sorts by physical proximity

   row has size P1d, qrefx is the coordinate of the quadpoint
   corresponding to the row.
*/
int nearby_physical(basis_element * row, int P1d, double qrefx)
{
  int ierr;

  CeedScalar * nodes = (CeedScalar*) calloc(P1d, sizeof(CeedScalar));
  ierr = CeedLobattoQuadrature(P1d, nodes, NULL); CeedChk(ierr);
  // so these nodes are in [-1, 1]
  // but qrefx is in [0, 1], but why?

  // I am just going to copy the row because I don't care about efficiency
  basis_element old_row[P1d];
  distindex distance_row[P1d];
  for (int i = 0; i < P1d; ++i) {
    const double node_loc = 0.5 * (nodes[i] + 1.0);
    old_row[i] = row[i];
    distance_row[i].dist = fabs(node_loc - qrefx);
    distance_row[i].index = i;
  }
  qsort(distance_row, P1d, sizeof(distindex), compare_distindex);
  for (int i = 0; i < P1d; ++i) {
    // we want closest ones at the end, so reverse
    // of the result from qsort
    row[P1d - i - 1] = old_row[distance_row[i].index];
  }
  free(nodes);

  return 0;
}

int CeedBasisSparsifyScaling(CeedBasis basisin, CeedBasis* basisout,
                             SparsifySelectionStrategy sel_strategy,
                             int parameter) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basisin, &ceed); CeedChk(ierr);
  if (parameter < 0) {
    return CeedError(ceed, 1, "Bad parameter value!");
  }

  CeedInt dim, ncomp, P1d, Q1d;
  ierr = CeedBasisGetDimension(basisin, &dim); CeedChk(ierr);
  ierr = CeedBasisGetNumComponents(basisin, &ncomp); CeedChk(ierr);
  ierr = CeedBasisGetNumNodes1D(basisin, &P1d); CeedChk(ierr);
  ierr = CeedBasisGetNumQuadraturePoints1D(basisin, &Q1d); CeedChk(ierr);

  const CeedScalar *interp1d;
  ierr = CeedBasisGetInterp1D(basisin, &interp1d); CeedChk(ierr);
  const CeedScalar * grad1d;
  ierr = CeedBasisGetGrad1D(basisin, &grad1d); CeedChk(ierr);
  const CeedScalar * qref1d;
  ierr = CeedBasisGetQRef(basisin, &qref1d); CeedChk(ierr);
  const CeedScalar * qweight1d;
  ierr = CeedBasisGetQWeights(basisin, &qweight1d); CeedChk(ierr);

  CeedScalar * sparse_interp1d = (CeedScalar*) calloc(
    P1d * Q1d, sizeof(CeedScalar));
  CeedScalar * sparse_grad1d = (CeedScalar*) calloc(
    P1d * Q1d, sizeof(CeedScalar));

  basis_element * row = (basis_element*) calloc(P1d, sizeof(basis_element));
  int sc = (P1d < parameter + 1) ? P1d : parameter + 1;
    
  for (int q = 0; q < Q1d; ++q) {
    int basis_nonzeros = 0;
    for (int i = 0; i < P1d; ++i) {
      sparse_interp1d[q * P1d + i] = 0.0;
      sparse_grad1d[q * P1d + i] = 0.0;

      row[i].index = i;
      row[i].basis_value = interp1d[q * P1d + i];
      row[i].grad_value = grad1d[q * P1d + i];
      if (fabs(row[i].basis_value) > 1.e-12) {
        basis_nonzeros++;
      }
    }
    if (sel_strategy == SPARSIFY_LARGEST_GRAD_ABS ||
        basis_nonzeros < sc) {
      qsort(row, P1d, sizeof(basis_element), basis_element_compare_grad_abs);
    } else if (sel_strategy == SPARSIFY_LARGEST_ABS) {
      qsort(row, P1d, sizeof(basis_element), basis_element_compare_abs);
    } else if (sel_strategy == SPARSIFY_LARGEST_POSITIVE) {
      qsort(row, P1d, sizeof(basis_element), basis_element_compare);
    } else if (sel_strategy == SPARSIFY_NEARBY) {
      // near_largest(row, P1d);
      nearby_physical(row, P1d, qref1d[q]);
    } else {
      return CeedError(ceed, 1, "Bad strategy!");
    }

    double basis_sum = 0.0;
    double grad_sum_plus = 0.0;
    double grad_sum_minus = 0.0;
    for (int k = 0; k < sc; k++) {
      int reverse_index = P1d - 1 - k;
      double basis_value = row[reverse_index].basis_value;
      double grad_value = row[reverse_index].grad_value;
      basis_sum += basis_value;
      if (grad_value > 0.0) {
        grad_sum_plus += grad_value;
      } else {
        grad_sum_minus += (-grad_value);
      }
    }
    double grad_scale_plus = (grad_sum_plus + grad_sum_minus) /
      (2.0 * grad_sum_plus);
    if (grad_sum_plus == 0.0) {
      grad_scale_plus = 1.0;
    }
    double grad_scale_minus = (grad_sum_plus + grad_sum_minus) /
      (2.0 * grad_sum_minus);
    if (grad_sum_minus == 0.0) {
      grad_scale_minus = 1.0;
    }
    for (int k = 0; k < sc; k++) {
      int reverse_index = P1d - 1 - k;
      int i = row[reverse_index].index;
      double basis_value = row[reverse_index].basis_value / basis_sum;
      double grad_value = row[reverse_index].grad_value;
      sparse_interp1d[q * P1d + i] = basis_value;
      if (grad_value > 0.0) {
        sparse_grad1d[q * P1d + i] = grad_value * grad_scale_plus;
      } else {
        sparse_grad1d[q * P1d + i] = grad_value * grad_scale_minus;
      }
    }
  }
  free(row);

  ierr = CeedBasisCreateTensorH1(ceed, dim, ncomp,
                                 P1d, Q1d, sparse_interp1d, sparse_grad1d,
                                 qref1d, qweight1d, basisout); CeedChk(ierr);

  const bool debug = false;
  if (debug) {
    double frob = 0.0;
    for (int q = 0; q < Q1d; q++) {
      for (int i = 0; i < P1d; i++) {
        printf("  interp1d[%d,%d] = %f, sparse = %f\n",
               q, i, interp1d[q * P1d + i], sparse_interp1d[q * P1d + i]);
        frob += (interp1d[q * P1d + i] -  sparse_interp1d[q * P1d + i]) *
          (interp1d[q * P1d + i] -  sparse_interp1d[q * P1d + i]);
      }
    }
    printf("interp1d frob diff = %e\n", sqrt(frob));
    frob = 0.0;
    for (int q = 0; q < Q1d; q++) {
      for (int i = 0; i < P1d; i++) {
        printf("  grad1d[%d,%d] = %f, sparse = %f\n",
               q, i, grad1d[q * P1d + i], sparse_grad1d[q * P1d + i]);
        frob += (grad1d[q * P1d + i] -  sparse_grad1d[q * P1d + i]) *
          (grad1d[q * P1d + i] -  sparse_grad1d[q * P1d + i]);
      }
    }
    printf("grad1d frob diff = %e\n", sqrt(frob));
  }

  free(sparse_interp1d);
  free(sparse_grad1d);

  return 0;
}

int CeedSparsifySingleOperator(CeedOperator oper, int sparse_parameter,
                               SparsifySelectionStrategy sel_strategy,
                               int (*basis_sparsify)(CeedBasis, CeedBasis*, SparsifySelectionStrategy, int),
                               CeedBasis* sparse_basis_out,
                               CeedOperator* out) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(oper, &ceed); CeedChk(ierr);

  CeedQFunction qf;
  bool isComposite;
  ierr = CeedOperatorIsComposite(oper, &isComposite); CeedChk(ierr);
  if (isComposite) {
     return CeedError(ceed, 1, "This function does not make sense for composite operator!");
  }
  ierr = CeedOperatorGetQFunction(oper, &qf); CeedChk(ierr);
  CeedInt numinputfields, numoutputfields;
  ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedQFunctionField *inputqfields, *outputqfields;
  ierr = CeedQFunctionGetFields(qf, &inputqfields, &outputqfields); CeedChk(ierr);
  CeedOperatorField *inputfields, *outputfields;
  ierr = CeedOperatorGetFields(oper, &inputfields, &outputfields); CeedChk(ierr);

  CeedElemRestriction * er_input =
    (CeedElemRestriction*) calloc(numinputfields, sizeof(CeedElemRestriction));
  CeedElemRestriction * er_output =
    (CeedElemRestriction*) calloc(numoutputfields, sizeof(CeedElemRestriction));
  CeedVector * if_vector = 
    (CeedVector*) calloc(numinputfields, sizeof(CeedVector));
  CeedVector * of_vector = 
    (CeedVector*) calloc(numoutputfields, sizeof(CeedVector));
  CeedBasis * basis_input =
    (CeedBasis*) calloc(numinputfields, sizeof(CeedBasis));
  CeedBasis * basis_output = 
    (CeedBasis*) calloc(numoutputfields, sizeof(CeedBasis));
  CeedBasis sbasis;
  int active_input_basis = -1;
  for (int i = 0; i < numinputfields; ++i) {
    ierr = CeedOperatorFieldGetElemRestriction(inputfields[i],
                                               &er_input[i]); CeedChk(ierr);
    ierr = CeedOperatorFieldGetVector(inputfields[i], &if_vector[i]); CeedChk(ierr);
    ierr = CeedOperatorFieldGetBasis(inputfields[i], &basis_input[i]); CeedChk(ierr);
    if (if_vector[i] == CEED_VECTOR_ACTIVE) {
      if (active_input_basis < 0) {
        // sparsify!
        ierr = (*basis_sparsify)(basis_input[i], &sbasis, sel_strategy, sparse_parameter); CeedChk(ierr);
        active_input_basis = i;
      } else {
        // should already be sparsified
        if (basis_input[i] != basis_input[active_input_basis]) {
          return CeedError(ceed, 1, "Two different active input basis!");
        }
      }
    }
  }
  for (int i = 0; i < numoutputfields; ++i) {
    ierr = CeedOperatorFieldGetElemRestriction(outputfields[i],
                                               &er_output[i]); CeedChk(ierr);
    ierr = CeedOperatorFieldGetVector(outputfields[i], &of_vector[i]); CeedChk(ierr);
    ierr = CeedOperatorFieldGetBasis(outputfields[i], &basis_output[i]); CeedChk(ierr);
    if (of_vector[i] == CEED_VECTOR_ACTIVE) {
      // should already be sparsified
      if (basis_output[i] != basis_input[active_input_basis]) {
        return CeedError(ceed, 1, "Input and output basis do not match!");
      }
    }
  }

  CeedOperator soper;
  ierr = CeedOperatorCreate(ceed, qf, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                            &soper); CeedChk(ierr);

  for (int i = 0; i < numinputfields; ++i) {
    char * fieldname;
    ierr = CeedQFunctionFieldGetName(inputqfields[i], &fieldname); CeedChk(ierr);
    if (if_vector[i] == CEED_VECTOR_ACTIVE) {
      ierr = CeedOperatorSetField(soper, fieldname, er_input[i], sbasis, 
                                  if_vector[i]); CeedChk(ierr);
    } else {
      ierr = CeedOperatorSetField(soper, fieldname, er_input[i], basis_input[i], 
                                  if_vector[i]); CeedChk(ierr);
    }
  }
  for (int i = 0; i < numoutputfields; ++i) {
    char * fieldname;
    ierr = CeedQFunctionFieldGetName(outputqfields[i], &fieldname); CeedChk(ierr);
    if (of_vector[i] == CEED_VECTOR_ACTIVE) {
      ierr = CeedOperatorSetField(soper, fieldname, er_output[i], sbasis,
                                  of_vector[i]); CeedChk(ierr);
    } else {
      ierr = CeedOperatorSetField(soper, fieldname, er_output[i], basis_output[i],
                                  of_vector[i]); CeedChk(ierr);
    }
  }
  free(er_input);
  free(er_output);
  free(if_vector);
  free(of_vector);
  free(basis_input);
  free(basis_output);

  *sparse_basis_out = sbasis;
  *out = soper;
  return 0;
}

int CeedSparsifyH1Operator(CeedOperator oper, int sparse_parameter,
                           SparsifySelectionStrategy sel_strategy,
                           int (*basis_sparsify)(CeedBasis, CeedBasis*, SparsifySelectionStrategy, int),
                           CeedBasis* sparse_basis_out,
                           CeedOperator* out) {
   int ierr;
   Ceed ceed;
   ierr = CeedOperatorGetCeed(oper, &ceed); CeedChk(ierr);

   CeedInt numsub;
   CeedOperator *subops;
   bool isComposite;
   ierr = CeedOperatorIsComposite(oper, &isComposite); CeedChk(ierr);
   if (isComposite) {
      ierr = CeedOperatorGetNumSub(oper, &numsub); CeedChk(ierr);
      ierr = CeedOperatorGetSubList(oper, &subops); CeedChk(ierr);
      if (numsub != 1) {
         // implementing this case is pretty easy but I am lazy
         return CeedError(ceed, 1, "Not implemented for multiple suboperators!");
      }
      ierr = CeedSparsifySingleOperator(subops[0], sparse_parameter,
                                        sel_strategy, basis_sparsify,
                                        sparse_basis_out, out);
   } else {
      ierr = CeedSparsifySingleOperator(oper,  sparse_parameter,
                                        sel_strategy, basis_sparsify,
                                        sparse_basis_out, out);
   }      
   return 0;
}

int CeedSparsifySimple(CeedOperator oper, CeedBasis* sparse_basis_out,
                       CeedOperator* out) {
   int ierr;
   ierr = CeedSparsifyH1Operator(oper, 1, SPARSIFY_LARGEST_POSITIVE,
                                 CeedBasisSparsifyScaling, sparse_basis_out,
                                 out); CeedChk(ierr);
   return 0;
}

#endif
