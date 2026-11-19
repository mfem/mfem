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

#ifndef MFEM_CONTACT_NEO_HOOKEAN_HPP
#define MFEM_CONTACT_NEO_HOOKEAN_HPP

#include "mfem.hpp"

namespace mfem
{

/// CPU/device evaluation of the miniapp's compressible Neo-Hookean model (g = 1).
/// The reference mesh and material coefficients must stay fixed. Elements
/// must have one geometry and polynomial order, with dimension 2 or 3.
class DeviceNeoHookean
{
private:
   ParFiniteElementSpace &fes;
   QuadratureSpace quadrature;
   CoefficientVector shear, bulk;
   const ElementRestriction &restriction;
   const DofToQuad &maps;
   const GeometricFactors &geometry;
   Vector physical_gradients, weights;
   Vector local_state, element_state, local_gradient, element_gradient;
   Vector material_data, projected_gradients, energy_data, determinants;
   Vector element_hessian;
   // On ordinary HYPRE builds, local_matrix owns the CSR storage and
   // local_hessian is its non-owning assembly view. Declare the owner first
   // so the view is destroyed before the storage it references.
   std::unique_ptr<HypreParMatrix> local_matrix;
   SparseMatrix local_hessian;
   bool matrix_initialized = false;

public:
   DeviceNeoHookean(ParFiniteElementSpace &fes, Coefficient &shear,
                    Coefficient &bulk);

   ~DeviceNeoHookean()
   {
#ifdef HYPRE_BIGINT
      // The BIGINT wrapper borrows local_hessian's values instead of owning
      // them; release it before the source storage is destroyed.
      local_matrix.reset();
#endif
   }

   /// Evaluate energy and optionally its gradient and assembled true-dof Hessian.
   /// Coordinates are the deformed configuration in true dofs. Essential
   /// entries of the gradient are zero, and the Hessian uses DIAG_ONE.
   /// A null output pointer skips that output, including Hessian assembly.
   /// With identity prolongation the Hessian handle borrows local_matrix;
   /// it remains valid until the next Hessian evaluation or this object's
   /// destruction. Otherwise the handle owns the parallel-assembled matrix.
   void Evaluate(const Vector &coordinates, const Array<int> &ess_tdofs,
                 real_t &energy, Vector *gradient = nullptr,
                 OperatorHandle *hessian = nullptr);
};

}

#endif
