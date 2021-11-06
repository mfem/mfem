// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more inforAion and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_TENSOR_ELEM_OP
#define MFEM_TENSOR_ELEM_OP

#include "quadrature_operator.hpp"

namespace mfem
{

/// Represent a local E-vector level finite element operator, dofs to dofs.
template <typename QData, typename TrialBasis, typename TestBasis>
struct ElementOperator
{
   const QData& qdata;
   const TrialBasis& trial_basis;
   const TestBasis& test_basis;

   ElementOperator(const QData& qdata,
                      const TrialBasis& trial_basis,
                      const TestBasis& test_basis)
   : qdata(qdata), trial_basis(trial_basis), test_basis(test_basis)
   { }
};

template <typename QData, typename TrialBasis, typename TestBasis>
MFEM_HOST_DEVICE
auto operator*(const TestBasis& test_basis,
               const QuadratureOperator<QData,TrialBasis>& q_op)
{
   return ElementOperator<QData,TrialBasis,TestBasis>(q_op.qdata, q_op.basis, test_basis);
}

template <typename QData, typename TrialBasis, typename TestBasis>
MFEM_HOST_DEVICE
auto operator*(const TransposeQuadratureOperator<QData,TestBasis>& q_op,
               const TrialBasis& trial_basis)
{
   return ElementOperator<QData,TrialBasis,TestBasis>(q_op.qdata, trial_basis, q_op.basis);
}

template <typename QData, typename TrialBasis, typename TestBasis, typename Dofs>
MFEM_HOST_DEVICE
auto operator*(const ElementOperator<QData,TrialBasis,TestBasis>& op,
               const Dofs& u)
{
   return op.test_basis * (op.qdata * ( op.trial_basis * u ) );
}

/// get_matrix_rows


} // namespace mfem

#endif // MFEM_TENSOR_ELEM_OP
