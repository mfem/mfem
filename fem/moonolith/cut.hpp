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

#ifndef MFEM_TRANSFER_CUT_HPP
#define MFEM_TRANSFER_CUT_HPP

#include "../fem.hpp"
#include <memory>

namespace mfem
{

/*!
 * @brief All subclasses of Cut will implement intersection routines and
 * quadrature point generation within the cut in the intersection of two
 * elements. Although, this class is designed to support MortarAssembler and
 * ParMortarAssembler, it can be used for any problem requiring to perform
 * Petrov-Galerkin formulations on non-matching elements.
 */
class Cut
{
public:
   virtual ~Cut() = default;

   /*!
    * @brief This method creates/updates the quadrature on the reference element
    * based on the integration order
    * @param order the order of the quadrature rule
    */
   virtual void SetIntegrationOrder(const int order) = 0;

   /*!
    * @brief This computes the intersection of the finite element geometries and
    * generates quadratures formulas for the two reference configurations
    * @param from_space the space from which we want to transfer a function with
    * MortarAssembler and ParMortarAssembler
    * @param from_elem_idx the index of the element of the from_space
    * @param to_space the space to which we want to transfer a function with
    * MortarAssembler and ParMortarAssembler
    * @param to_elem_idx the index of the element of the to_space
    * @param[out] from_quadrature the quadrature rule in the reference coordinate
    * system of the from element
    * @param[out] to_quadrature the quadrature rule in the reference coordinate
    * system of the to element
    * @return true if the two element intersected and if the output must be used
    * or ignored.
    */
   virtual bool BuildQuadrature(const FiniteElementSpace &from_space,
                                const int from_elem_idx,
                                const FiniteElementSpace &to_space,
                                const int to_elem_idx,
                                IntegrationRule &from_quadrature,
                                IntegrationRule &to_quadrature) = 0;

   /*!
    * @brief Method for printing information to the command line
    */
   virtual void Describe() const {}

protected:
   virtual void SetQuadratureRule(const IntegrationRule &ir) = 0;
};

/// Create a new cut object based on the spatial dimension
/// @param dim the spatial dimension
std::shared_ptr<Cut> NewCut(const int dim);

} // namespace mfem

#endif // MFEM_TRANSFER_CUT_HPP
