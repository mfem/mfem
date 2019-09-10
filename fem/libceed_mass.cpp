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

#include "libceed.hpp"
#include "../general/ceed.hpp"
#include "../general/device.hpp"

#include "libceed_mass.h"

namespace mfem
{

#ifdef MFEM_USE_CEED

void CeedPAMassAssemble(const FiniteElementSpace &fes,
                        const mfem::IntegrationRule &irm, CeedData& ceedData)
{
   Ceed ceed(internal::ceed);
   mfem::Mesh *mesh = fes.GetMesh();
   const int order = fes.GetOrder(0);
   const int ir_order = irm.GetOrder();
   const mfem::IntegrationRule &ir =
      mfem::IntRules.Get(mfem::Geometry::SEGMENT, ir_order);
   CeedInt nqpts, nelem = mesh->GetNE();

   FESpace2Ceed(fes, ir, ceed, &ceedData.basis, &ceedData.restr);
   mesh->EnsureNodes();
   const mfem::FiniteElementSpace *mesh_fes = mesh->GetNodalFESpace();
   MFEM_VERIFY(mesh_fes, "the Mesh has no nodal FE space");
   FESpace2Ceed(*mesh_fes, ir, ceed, &ceedData.mesh_basis, &ceedData.mesh_restr);
   CeedBasisGetNumQuadraturePoints(ceedData.basis, &nqpts);

   CeedElemRestrictionCreateIdentity(ceed, nelem, nqpts,
                                     nqpts*nelem, 1, &ceedData.restr_i);
   CeedElemRestrictionCreateIdentity(ceed, nelem, nqpts,
                                     nqpts*nelem, 1, &ceedData.mesh_restr_i);

   CeedVectorCreate(ceed, mesh->GetNodes()->Size(), &ceedData.node_coords);
   CeedVectorSetArray(ceedData.node_coords, CEED_MEM_HOST, CEED_USE_POINTER,
                      mesh->GetNodes()->GetData());

   CeedVectorCreate(ceed, nelem*nqpts, &ceedData.rho);

   // Context data to be passed to the 'f_build_mass' Q-function.
   ceedData.build_ctx.dim = mesh->Dimension();
   ceedData.build_ctx.space_dim = mesh->SpaceDimension();

   // Create the Q-function that builds the mass operator (i.e. computes its
   // quadrature data) and set its context data.
   switch (ceedData.coeff_type)
   {
      case Const:
         CeedQFunctionCreateInterior(ceed, 1, f_build_mass_const,
                                     MFEM_SOURCE_DIR"/fem/libceed_mass.okl:f_build_mass_const",
                                     &ceedData.build_qfunc);
         ceedData.build_ctx.coeff = ((CeedConstCoeff*)ceedData.coeff)->val;
         break;
      case Grid:
         CeedQFunctionCreateInterior(ceed, 1, f_build_mass_grid,
                                     MFEM_SOURCE_DIR"/fem/libceed_mass.okl:f_build_mass_grid",
                                     &ceedData.build_qfunc);
         CeedQFunctionAddInput(ceedData.build_qfunc, "coeff", 1, CEED_EVAL_INTERP);
         break;
      default:
         MFEM_ABORT("This coeff_type is not handled");
   }
   CeedQFunctionAddInput(ceedData.build_qfunc, "dx", mesh->SpaceDimension()*mesh->SpaceDimension(),
                         CEED_EVAL_GRAD);
   CeedQFunctionAddInput(ceedData.build_qfunc, "weights", 1, CEED_EVAL_WEIGHT);
   CeedQFunctionAddOutput(ceedData.build_qfunc, "rho", 1, CEED_EVAL_NONE);
   CeedQFunctionSetContext(ceedData.build_qfunc, &ceedData.build_ctx,
                           sizeof(ceedData.build_ctx));

   // Create the operator that builds the quadrature data for the mass operator.
   CeedOperatorCreate(ceed, ceedData.build_qfunc, NULL, NULL,
                      &ceedData.build_oper);
   CeedTransposeMode lmode = CEED_NOTRANSPOSE;
   if (mesh_fes->GetOrdering()==Ordering::byVDIM)
   {
      lmode = CEED_TRANSPOSE;
   }
   if (ceedData.coeff_type==Grid)
   {
      CeedGridCoeff* ceedCoeff = (CeedGridCoeff*)ceedData.coeff;
      FESpace2Ceed(*ceedCoeff->coeff->FESpace(), ir, ceed, &ceedCoeff->basis,
                   &ceedCoeff->restr);
      CeedVectorCreate(ceed, ceedCoeff->coeff->FESpace()->GetNDofs(),
                       &ceedCoeff->coeffVector);
      CeedVectorSetArray(ceedCoeff->coeffVector, CEED_MEM_HOST, CEED_USE_POINTER,
                         ceedCoeff->coeff->GetData());
      CeedOperatorSetField(ceedData.build_oper, "coeff", ceedCoeff->restr, lmode,
                           ceedCoeff->basis, ceedCoeff->coeffVector);
   }
   CeedOperatorSetField(ceedData.build_oper, "dx", ceedData.mesh_restr, lmode,
                        ceedData.mesh_basis, CEED_VECTOR_ACTIVE);
   CeedOperatorSetField(ceedData.build_oper, "weights", ceedData.mesh_restr_i,
                        CEED_NOTRANSPOSE,
                        ceedData.mesh_basis, CEED_VECTOR_NONE);
   CeedOperatorSetField(ceedData.build_oper, "rho", ceedData.restr_i,
                        CEED_NOTRANSPOSE,
                        CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

   // Compute the quadrature data for the mass operator.
   CeedOperatorApply(ceedData.build_oper, ceedData.node_coords, ceedData.rho,
                     CEED_REQUEST_IMMEDIATE);

   // Create the Q-function that defines the action of the mass operator.
   CeedQFunctionCreateInterior(ceed, 1, f_apply_mass,
                               MFEM_SOURCE_DIR"/fem/libceed_mass.okl:f_apply_mass", &ceedData.apply_qfunc);
   CeedQFunctionAddInput(ceedData.apply_qfunc, "u", 1, CEED_EVAL_INTERP);
   CeedQFunctionAddInput(ceedData.apply_qfunc, "rho", 1, CEED_EVAL_NONE);
   CeedQFunctionAddOutput(ceedData.apply_qfunc, "v", 1, CEED_EVAL_INTERP);

   // Create the mass operator.
   CeedOperatorCreate(ceed, ceedData.apply_qfunc, NULL, NULL, &ceedData.oper);
   CeedOperatorSetField(ceedData.oper, "u", ceedData.restr, CEED_NOTRANSPOSE,
                        ceedData.basis, CEED_VECTOR_ACTIVE);
   CeedOperatorSetField(ceedData.oper, "rho", ceedData.restr_i, CEED_NOTRANSPOSE,
                        CEED_BASIS_COLLOCATED, ceedData.rho);
   CeedOperatorSetField(ceedData.oper, "v", ceedData.restr, CEED_NOTRANSPOSE,
                        ceedData.basis, CEED_VECTOR_ACTIVE);

   CeedVectorCreate(ceed, fes.GetNDofs(), &ceedData.u);
   CeedVectorCreate(ceed, fes.GetNDofs(), &ceedData.v);
}

#endif

}
