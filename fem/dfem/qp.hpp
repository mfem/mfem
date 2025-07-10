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
#pragma once

#include "tuple.hpp"
#include "../linalg/tensor.hpp"

using namespace mfem::future;
using mfem::future::tensor;

// Helper to add dimension to tensor type
template<typename T, int qp>
struct AddQPDimension;

// Specialization for tensor<real_t, dim>
template<typename real_t, int dim, int qp>
struct AddQPDimension<tensor<real_t, dim>, qp>
{
   using type = tensor<real_t, dim, qp>;
};

// Specialization for tensor<real_t, dim, dim>
template<typename real_t, int dim, int qp>
struct AddQPDimension<tensor<real_t, dim, dim>, qp>
{
   using type = tensor<real_t, dim, dim, qp>;
};

// Specialization for real_t (transforms to tensor<real_t, qp>)
template<typename real_t, int qp>
struct AddQPDimension
{
   using type = tensor<real_t, qp>;
};

// Helper to transform tuple
template<typename Tuple, int qp>
struct TransformTupleQP {};

// Specialization for mfem::future::tuple
template<int qp, typename... Types>
struct TransformTupleQP<mfem::future::tuple<Types...>, qp>
{
   using type = mfem::future::tuple<typename AddQPDimension<Types, qp>::type...>;
};

template<int qp, typename... Types>
struct TransformTupleQP<std::tuple<Types...>, qp>
{
   using type = std::tuple<typename AddQPDimension<Types, qp>::type...>;
};

// Function to transform tuple type with qp dimension
template<int qp, typename qf_param_ts>
struct add_qp_dimension
{
   using type = typename TransformTupleQP<qf_param_ts, qp>::type;
};

// Helper alias template for cleaner usage
template<int qp, typename qf_param_ts>
using add_qp_dimension_t = typename add_qp_dimension<qp, qf_param_ts>::type;

// ...AddDomainIntegrator...
// {
//    constexpr int Q1D = 4;
//    using qf_param_augmentd_ts = add_qp_dimension_t<Q1D, decay_tuple<qf_param_ts>>;
// }