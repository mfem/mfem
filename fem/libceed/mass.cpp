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

#include "mass.hpp"

#ifdef MFEM_USE_CEED
#include "../../general/device.hpp"
#include "../../mesh/mesh.hpp"
#include "../../fem/gridfunc.hpp"
#include "ceed.hpp"
#include "mass.h"

namespace mfem
{

void CeedPAMassAssemble(const FiniteElementSpace &fes,
                        const mfem::IntegrationRule &irm, CeedData& ceedData)
{
   Ceed ceed(internal::ceed);
   mfem::Mesh *mesh = fes.GetMesh();
   const int ir_order = irm.GetOrder();
   CeedInt nqpts, nelem = mesh->GetNE(), dim = mesh->SpaceDimension();

   mesh->EnsureNodes();
   InitCeedBasisAndRestriction(fes, irm, ceed, &ceedData.basis, &ceedData.restr);

   const mfem::FiniteElementSpace *mesh_fes = mesh->GetNodalFESpace();
   MFEM_VERIFY(mesh_fes, "the Mesh has no nodal FE space");
   InitCeedBasisAndRestriction(*mesh_fes, irm, ceed, &ceedData.mesh_basis,
                               &ceedData.mesh_restr);

   CeedBasisGetNumQuadraturePoints(ceedData.basis, &nqpts);

   const int qdatasize = 1;
   CeedElemRestrictionCreateStrided(ceed, nelem, nqpts, nelem*nqpts, qdatasize,
                                    CEED_STRIDES_BACKEND, &ceedData.restr_i);

   CeedVectorCreate(ceed, mesh->GetNodes()->Size(), &ceedData.node_coords);
   CeedVectorSetArray(ceedData.node_coords, CEED_MEM_HOST, CEED_USE_POINTER,
                      mesh->GetNodes()->GetData());

   CeedVectorCreate(ceed, nelem*nqpts, &ceedData.rho);

   // Context data to be passed to the 'f_build_mass' Q-function.
   ceedData.build_ctx.dim = mesh->Dimension();
   ceedData.build_ctx.space_dim = mesh->SpaceDimension();

   std::string mass_qf_file = GetCeedPath() + "/mass.h";
   std::string mass_qf;

   // Create the Q-function that builds the mass operator (i.e. computes its
   // quadrature data) and set its context data.
   switch (ceedData.coeff_type)
   {
      case CeedCoeff::Const:
         mass_qf = mass_qf_file + ":f_build_mass_const";
         CeedQFunctionCreateInterior(ceed, 1, f_build_mass_const,
                                     mass_qf.c_str(),
                                     &ceedData.build_qfunc);
         ceedData.build_ctx.coeff = ((CeedConstCoeff*)ceedData.coeff)->val;
         break;
      case CeedCoeff::Grid:
         mass_qf = mass_qf_file + ":f_build_mass_grid";
         CeedQFunctionCreateInterior(ceed, 1, f_build_mass_grid,
                                     mass_qf.c_str(),
                                     &ceedData.build_qfunc);
         CeedQFunctionAddInput(ceedData.build_qfunc, "coeff", 1, CEED_EVAL_INTERP);
         break;
      default:
         MFEM_ABORT("This coeff_type is not handled");
   }
   CeedQFunctionAddInput(ceedData.build_qfunc, "dx",
                         mesh->SpaceDimension()*mesh->SpaceDimension(),
                         CEED_EVAL_GRAD);
   CeedQFunctionAddInput(ceedData.build_qfunc, "weights", 1, CEED_EVAL_WEIGHT);
   CeedQFunctionAddOutput(ceedData.build_qfunc, "rho", 1, CEED_EVAL_NONE);
   CeedQFunctionSetContext(ceedData.build_qfunc, &ceedData.build_ctx,
                           sizeof(ceedData.build_ctx));

   // Create the operator that builds the quadrature data for the mass operator.
   CeedOperatorCreate(ceed, ceedData.build_qfunc, NULL, NULL,
                      &ceedData.build_oper);
   if (ceedData.coeff_type==CeedCoeff::Grid)
   {
      CeedGridCoeff* ceedCoeff = (CeedGridCoeff*)ceedData.coeff;
      InitCeedBasisAndRestriction(*ceedCoeff->coeff->FESpace(), irm, ceed,
                                  &ceedCoeff->basis,
                                  &ceedCoeff->restr);
      CeedVectorCreate(ceed, ceedCoeff->coeff->FESpace()->GetNDofs(),
                       &ceedCoeff->coeffVector);
      CeedVectorSetArray(ceedCoeff->coeffVector, CEED_MEM_HOST, CEED_USE_POINTER,
                         ceedCoeff->coeff->GetData());
      CeedOperatorSetField(ceedData.build_oper, "coeff", ceedCoeff->restr,
                           ceedCoeff->basis, ceedCoeff->coeffVector);
   }
   CeedOperatorSetField(ceedData.build_oper, "dx", ceedData.mesh_restr,
                        ceedData.mesh_basis, CEED_VECTOR_ACTIVE);
   CeedOperatorSetField(ceedData.build_oper, "weights", CEED_ELEMRESTRICTION_NONE,
                        ceedData.mesh_basis, CEED_VECTOR_NONE);
   CeedOperatorSetField(ceedData.build_oper, "rho", ceedData.restr_i,
                        CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

   // Compute the quadrature data for the mass operator.
   CeedOperatorApply(ceedData.build_oper, ceedData.node_coords, ceedData.rho,
                     CEED_REQUEST_IMMEDIATE);

   // Create the Q-function that defines the action of the mass operator.
   mass_qf = mass_qf_file + ":f_apply_mass";
   CeedQFunctionCreateInterior(ceed, 1, f_apply_mass,
                               mass_qf.c_str(), &ceedData.apply_qfunc);
   CeedQFunctionAddInput(ceedData.apply_qfunc, "u", 1, CEED_EVAL_INTERP);
   CeedQFunctionAddInput(ceedData.apply_qfunc, "rho", 1, CEED_EVAL_NONE);
   CeedQFunctionAddOutput(ceedData.apply_qfunc, "v", 1, CEED_EVAL_INTERP);

   // Create the mass operator.
   CeedOperatorCreate(ceed, ceedData.apply_qfunc, NULL, NULL, &ceedData.oper);
   CeedOperatorSetField(ceedData.oper, "u", ceedData.restr,
                        ceedData.basis, CEED_VECTOR_ACTIVE);
   CeedOperatorSetField(ceedData.oper, "rho", ceedData.restr_i,
                        CEED_BASIS_COLLOCATED, ceedData.rho);
   CeedOperatorSetField(ceedData.oper, "v", ceedData.restr,
                        ceedData.basis, CEED_VECTOR_ACTIVE);

   CeedVectorCreate(ceed, fes.GetNDofs(), &ceedData.u);
   CeedVectorCreate(ceed, fes.GetNDofs(), &ceedData.v);
}

} // namespace mfem

#endif // MFEM_USE_CEED
