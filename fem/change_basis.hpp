// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_CHANGE_BASIS_HPP
#define MFEM_CHANGE_BASIS_HPP

#include "../linalg/operator.hpp"
#include "fespace.hpp"

namespace mfem
{

class ChangeOfBasis : public Operator
{
public:
   enum
   {
      LEGENDRE            = BasisType::NumBasisTypes + 1,
      INTEGRATED_LEGENDRE = BasisType::NumBasisTypes + 2
   };
protected:
   FiniteElementSpace &fes;
   mutable Vector x_e, y_e;

   mutable DofToQuad dof2quad;
   DenseMatrix T1D;
   DenseMatrix T1D_inv;

   void Mult_(const DenseMatrix &B1D, const Vector &x, Vector &y) const;

public:
   ChangeOfBasis(FiniteElementSpace &fes_, int dest_btype);
   void Mult(const Vector &x, Vector &y) const override;
   void MultInverse(const Vector &x, Vector &y) const;
};

} // namespace mfem

#endif
