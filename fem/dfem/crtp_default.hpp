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

#include "crtp_base.hpp"

#ifdef MFEM_USE_MPI

#include <string>
#include <utility>

#ifdef NVTX_DEBUG_HPP
#undef NVTX_COLOR
#define NVTX_COLOR ::nvtx::kGreen
#include NVTX_DEBUG_HPP
#else
#define dbg(...)
#endif

namespace mfem::future
{

class DefaultDifferentiableOperator final :
   public BackendOperator<DefaultDifferentiableOperator>
{
public:
   DefaultDifferentiableOperator(
      const std::vector<FieldDescriptor> &solutions,
      const std::vector<FieldDescriptor> &parameters,
      const ParMesh &mesh):
      BackendOperator<DefaultDifferentiableOperator>
      ((dbg("\n"),solutions), parameters, mesh)
   {
   }

   void impl_SetName(std::string n) { name = std::move(n); }

   void impl_SetBlocks(int a) { blocks = a; }

   void impl_Print() const { dbg("Backend:{}, blocks:{}", name, blocks); }

   void impl_Initialize(Vector &residual_e)
   {
      dbg("residual_e.Size() = {}", residual_e.Size());
      residual_e = 0.0;
   }

   void impl_Interpolate() { dbg("map_fields_to_quadrature_data"); }

   void impl_Qfunction() { dbg("call_qfunction"); }

   void impl_Integrate() { dbg("map_quadrature_data_to_fields"); }
};

} // namespace mfem::future

#endif // MFEM_USE_MPI