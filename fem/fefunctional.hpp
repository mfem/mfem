// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_FEFUNCTIONAL_HPP
#define MFEM_FEFUNCTIONAL_HPP

#include "../config/config.hpp"
#include "../linalg/functional.hpp"
#include "bilinearform.hpp"
#include "linearform.hpp"
#ifdef MFEM_USE_MPI
#include "pbilinearform.hpp"
#include "plinearform.hpp"
#endif

namespace mfem
{

namespace optimization
{

/// @brief Bilinear functional f(u) = a(u,u)/2 - l(u)
///
/// @note Input is always tdof, and essential boundary conditions
/// are handled at problem level. See mfem::optimization::Problem.
///
/// @note Assemble() must be called before Eval(), ComponentDerivative(),
/// and ComponentHessian().
class BilinearFunctional : public QuadraticFunctional
{
   BilinearForm *blf;
   bool own_blf;
   LinearForm *lf;
   bool own_lf;
   OperatorHandle matrix;
   Vector B;
#ifdef MFEM_USE_MPI
   ParBilinearForm *par_blf = nullptr;
   ParLinearForm *par_lf = nullptr;
#endif
public:
   BilinearFunctional(BilinearForm *blf_, LinearForm *lf_ = nullptr,
                      bool own_blf_ = false, bool own_lf_ = false);

   ~BilinearFunctional() override;

   void Assemble(int skip_zeros=0);

   BilinearForm *GetBilinearForm() const { return blf; }
   LinearForm *GetLinearForm() const { return lf; }

};

} // namespace optimization

} // namespace mfem

#endif // MFEM_FEFUNCTIONAL_HPP
