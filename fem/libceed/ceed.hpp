// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_LIBCEED_HPP
#define MFEM_LIBCEED_HPP

#include "../gridfunc.hpp"
#include "../fespace.hpp"

#ifdef MFEM_USE_CEED
#include <ceed.h>
#else
typedef void* Ceed;
typedef int CeedInt;
typedef double CeedScalar;
#define CEED_QFUNCTION(name) int name
#endif

namespace mfem
{

#ifdef MFEM_USE_CEED

namespace internal { extern Ceed ceed; }

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
};

/// Identifies the type of coefficient of the Integrator to initialize accordingly the CeedData
void InitCeedCoeff(Coefficient* Q, CeedData* ptr);

/// Initialize a tensor CeedBasis and a CeedElemRestriction
void InitCeedTensorBasisAndRestriction(const mfem::FiniteElementSpace &fes,
                                       const mfem::IntegrationRule &ir,
                                       Ceed ceed, CeedBasis *basis,
                                       CeedElemRestriction *restr);

const std::string &GetCeedPath();

#endif

}

#endif // MFEM_LIBCEED_HPP
