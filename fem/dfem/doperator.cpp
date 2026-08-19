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

#include "doperator.hpp"

#ifdef MFEM_USE_MPI

using namespace mfem;
using namespace mfem::future;

void DifferentiableOperator::SetParameters(std::vector<Vector *> p) const
{
   MFEM_ASSERT(parameters.size() == p.size(),
               "number of parameters doesn't match descriptors");
   for (size_t i = 0; i < parameters.size(); i++)
   {
      p[i]->Read();
      parameters_l[i] = *p[i];
   }
}

DifferentiableOperator::DifferentiableOperator(
   const std::vector<FieldDescriptor> &solutions,
   const std::vector<FieldDescriptor> &parameters,
   const ParMesh &mesh) :
   mesh(mesh),
   solutions(solutions),
   parameters(parameters)
{
   fields.resize(solutions.size() + parameters.size());
   fields_e.resize(fields.size());
   solutions_l.resize(solutions.size());
   parameters_l.resize(parameters.size());

   for (size_t i = 0; i < solutions.size(); i++)
   {
      fields[i] = solutions[i];
   }

   for (size_t i = 0; i < parameters.size(); i++)
   {
      fields[i + solutions.size()] = parameters[i];
   }
}


void FDJacobian::Mult(const Vector &v, Vector &y) const
{
   // See [1] for choice of eps.
   //
   // [1] Woodward, C.S., Gardner, D.J. and Evans, K.J., 2015. On the use of
   // finite difference matrix-vector products in Newton-Krylov solvers for
   // implicit climate dynamics with spectral elements. Procedia Computer
   // Science, 51, pp.2036-2045.
   real_t eps;
   if (fixed_eps > 0.0)
   {
      eps = fixed_eps;
   }
   else
   {
      const real_t vnorm_local = v.Norml2();
      real_t vnorm;
      MPI_Allreduce(&vnorm_local, &vnorm, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                    MPI_COMM_WORLD);
      eps = lambda * (lambda + xnorm / vnorm);
   }

   // x + eps * v
   {
      const auto d_v = v.Read();
      const auto d_x = x.Read();
      auto d_xpev = xpev.Write();
      mfem::forall(x.Size(), [=] MFEM_HOST_DEVICE (int i)
      {
         d_xpev[i] = d_x[i] + eps * d_v[i];
      });
   }

   // y = f(x + eps * v)
   op.Mult(xpev, y);

   // y = (f(x + eps * v) - f(x)) / eps
   {
      const auto d_f = f.Read();
      auto d_y = y.ReadWrite();
      mfem::forall(f.Size(), [=] MFEM_HOST_DEVICE (int i)
      {
         d_y[i] = (d_y[i] - d_f[i]) / eps;
      });
   }
}

#endif // MFEM_USE_MPI
