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

#ifndef MFEM_TENSOR_QUAD_OP
#define MFEM_TENSOR_QUAD_OP

namespace mfem
{

/// Represent an operator that goes from dofs to data at quadrature points.
template <typename QData, typename Basis>
struct QuadratureOperator
{
   const QData& qdata;
   const Basis& basis;

   QuadratureOperator(const QData& qdata, const Basis& basis)
   : qdata(qdata), basis(basis)
   { }
};

template <typename QData,
          typename Basis,
          std::enable_if_t<
             is_qdata<QData> &&
             is_basis<Basis>,
             bool> = true >
auto operator*(const QData& qdata, const Basis& basis)
{
   return QuadratureOperator<QData,Basis>(qdata, basis);
}

/// Represent an operator that goes from data at quadrature points to dofs.
template <typename QData, typename Basis>
struct TransposeQuadratureOperator
{
   const QData& qdata;
   const Basis& basis;

   TransposeQuadratureOperator(const QData& qdata, const Basis& basis)
   : qdata(qdata), basis(basis)
   { }
};

template <typename QData,
          typename Basis,
          std::enable_if_t<
             is_qdata<QData> &&
             is_basis<Basis>,
             bool> = true >
auto operator*(const Basis& basis, const QData& qdata)
{
   return TransposeQuadratureOperator<QData,Basis>(qdata, basis);
}

} // namespace mfem

#endif // MFEM_TENSOR_QUAD_OP
