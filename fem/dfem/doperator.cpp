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

#endif // MFEM_USE_MPI
