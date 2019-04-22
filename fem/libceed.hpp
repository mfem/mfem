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


#include "gridfunc.hpp"
#include "fespace.hpp"
#include "../general/ceed.hpp"

namespace mfem
{

#ifdef MFEM_USE_CEED
/// A structure used to pass additional data to f_build_diff and f_apply_diff
struct BuildContext { CeedInt dim, space_dim; CeedScalar coeff; };

// struct BuildContextConstCoeff { CeedInt dim, space_dim; CeedScalar coeff; };

enum CeedCoeff { Const, Grid };

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

void initCeedCoeff(Coefficient* Q, CeedData* ptr);

void CeedPADiffusionAssemble(const FiniteElementSpace &fes, CeedData& ceedData);

void CeedPAMassAssemble(const FiniteElementSpace &fes, CeedData& ceedData);
#else
typedef void* CeedData;
#endif

}
