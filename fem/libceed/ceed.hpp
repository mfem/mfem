// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_LIBCEED_HPP
#define MFEM_LIBCEED_HPP

#include "../../config/config.hpp"

#ifdef MFEM_USE_CEED
#include "../../general/device.hpp"
#include <ceed.h>

namespace mfem
{

class FiniteElementSpace;
class GridFunction;
class IntegrationRule;
class Coefficient;

namespace internal { extern Ceed ceed; } // defined in device.cpp

/// A structure used to pass additional data to f_build_diff and f_apply_diff
struct BuildContext { CeedInt dim, space_dim; CeedScalar coeff; };

enum class CeedCoeff { Const, Grid };

struct CeedConstCoeff
{
   double val;
};

struct CeedGridCoeff
{
   GridFunction* coeff;
   CeedBasis basis;
   CeedElemRestriction restr;
   CeedVector coeffVector;
};

struct CeedData
{
   CeedOperator build_oper, oper;
   CeedBasis basis, mesh_basis;
   CeedElemRestriction restr, mesh_restr, restr_i, mesh_restr_i;
   CeedQFunction apply_qfunc, build_qfunc;
   CeedVector node_coords, rho;
   CeedCoeff coeff_type;
   void* coeff;
   BuildContext build_ctx;

   CeedVector u, v;

   ~CeedData()
   {
      CeedOperatorDestroy(&build_oper);
      CeedOperatorDestroy(&oper);
      CeedBasisDestroy(&basis);
      CeedBasisDestroy(&mesh_basis);
      CeedElemRestrictionDestroy(&restr);
      CeedElemRestrictionDestroy(&mesh_restr);
      CeedElemRestrictionDestroy(&restr_i);
      CeedElemRestrictionDestroy(&mesh_restr_i);
      CeedQFunctionDestroy(&apply_qfunc);
      CeedQFunctionDestroy(&build_qfunc);
      CeedVectorDestroy(&node_coords);
      CeedVectorDestroy(&rho);
      if (coeff_type==CeedCoeff::Grid)
      {
         CeedGridCoeff* c = (CeedGridCoeff*)coeff;
         CeedBasisDestroy(&c->basis);
         CeedElemRestrictionDestroy(&c->restr);
         CeedVectorDestroy(&c->coeffVector);
         delete c;
      }
      else
      {
         delete (CeedConstCoeff*)coeff;
      }
      CeedVectorDestroy(&u);
      CeedVectorDestroy(&v);
   }

};


/** @brief Identifies the type of coefficient of the Integrator to initialize
    accordingly the CeedData. */
void InitCeedCoeff(Coefficient* Q, CeedData* ptr);

/// Initialize a CeedBasis and a CeedElemRestriction
void InitCeedBasisAndRestriction(const FiniteElementSpace &fes,
                                 const IntegrationRule &ir,
                                 Ceed ceed, CeedBasis *basis,
                                 CeedElemRestriction *restr);

/// Return the path to the libCEED q-function headers.
const std::string &GetCeedPath();

/** @brief Function that determines if a CEED kernel should be used, based on
    the current mfem::Device configuration. */
inline bool DeviceCanUseCeed()
{
   return Device::Allows(Backend::CEED_CUDA) ||
          (Device::Allows(Backend::CEED_CPU) &&
           !Device::Allows(Backend::DEVICE_MASK|Backend::OMP_MASK));
}

} // namespace mfem

#else // MFEM_USE_CEED

namespace mfem
{
inline bool DeviceCanUseCeed()
{
   return false;
}

} // namespace mfem

#endif // MFEM_USE_CEED

#endif // MFEM_LIBCEED_HPP
