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

#include "ceed.hpp"
#include "../../general/device.hpp"

#include "mechanics.h"

namespace mfem
{

#ifdef MFEM_USE_CEED

void CeedPAMechanicsAssemble(const FiniteElementSpace &fes,
                             const mfem::IntegrationRule &irm,
                             const double* ktan_ptr,
                             CeedData& ceedData)
{
   Ceed ceed(internal::ceed);
   mfem::Mesh *mesh = fes.GetMesh();
   const int ir_order = irm.GetOrder();
   const mfem::IntegrationRule &ir =
      mfem::IntRules.Get(mfem::Geometry::SEGMENT, ir_order);
   CeedInt nqpts, nelem = mesh->GetNE(), dim = mesh->SpaceDimension();
   mesh->EnsureNodes();
   InitCeedTensorBasisAndRestriction(fes, ir, ceed, &ceedData.basis, &ceedData.restr);

   const mfem::FiniteElementSpace *mesh_fes = mesh->GetNodalFESpace();
   MFEM_VERIFY(mesh_fes, "the Mesh has no nodal FE space");
   InitCeedTensorBasisAndRestriction(*mesh_fes, ir, ceed, &ceedData.mesh_basis, &ceedData.mesh_restr);
   CeedBasisGetNumQuadraturePoints(ceedData.basis, &nqpts);

   CeedElemRestrictionCreateIdentity(ceed, nelem, nqpts,
                                     nqpts * nelem, dim * dim * dim * dim, &ceedData.restr_i);
   CeedElemRestrictionCreateIdentity(ceed, nelem, nqpts,
                                     nqpts * nelem, 1, &ceedData.mesh_restr_i);

   CeedVectorCreate(ceed, mesh->GetNodes()->Size(), &ceedData.node_coords);
   CeedVectorSetArray(ceedData.node_coords, CEED_MEM_HOST, CEED_USE_POINTER,
                      mesh->GetNodes()->GetData());

   CeedVectorCreate(ceed, nelem * nqpts * dim * dim * dim * dim, &ceedData.rho);

   // Context data to be passed to the 'f_build_diff' Q-function.
   ceedData.build_ctx.dim = mesh->Dimension();
   ceedData.build_ctx.space_dim = mesh->SpaceDimension();

   std::string mech_qf_file = GetCeedPath() + "/mechanics.h";
   std::string mech_qf;

   // Create the Q-function that builds the diff operator (i.e. computes its
   // quadrature data) and set its context data.
   mech_qf = mech_qf_file + ":f_build_mech";
   CeedQFunctionCreateInterior(ceed, 1, f_build_mech,
                               mech_qf.c_str(),
                               &ceedData.build_qfunc);
   CeedQFunctionAddInput(ceedData.build_qfunc, "ktan", dim * dim * dim * dim,
                          CEED_EVAL_NONE);
   CeedQFunctionAddInput(ceedData.build_qfunc, "dx", dim * dim, CEED_EVAL_GRAD);
   CeedQFunctionAddInput(ceedData.build_qfunc, "weights", 1, CEED_EVAL_WEIGHT);
   CeedQFunctionAddOutput(ceedData.build_qfunc, "qd", dim * dim * dim * dim,
                          CEED_EVAL_NONE);
   CeedQFunctionSetContext(ceedData.build_qfunc, &ceedData.build_ctx,
                           sizeof(ceedData.build_ctx));

   // Create the operator that builds the quadrature data for the diff operator.
   CeedOperatorCreate(ceed, ceedData.build_qfunc, NULL, NULL,
                      &ceedData.build_oper);
   CeedTransposeMode lmode = CEED_NOTRANSPOSE;
   if (mesh_fes->GetOrdering()==Ordering::byVDIM)
   {
      lmode = CEED_TRANSPOSE;
   }
   CeedVector ktan;
   CeedVectorCreate(ceed, nelem * nqpts * dim * dim * dim * dim, &ktan);
   CeedVectorSetArray(ktan, CEED_MEM_DEVICE, CEED_USE_POINTER, const_cast<double*>(ktan_ptr));
   CeedOperatorSetField(ceedData.build_oper, "ktan", ceedData.restr_i,
                        CEED_NOTRANSPOSE,
                        CEED_BASIS_COLLOCATED, ktan);
   CeedOperatorSetField(ceedData.build_oper, "dx", ceedData.mesh_restr, lmode,
                        ceedData.mesh_basis, CEED_VECTOR_ACTIVE);
   CeedOperatorSetField(ceedData.build_oper, "weights", ceedData.mesh_restr_i,
                        CEED_NOTRANSPOSE,
                        ceedData.mesh_basis, CEED_VECTOR_NONE);
   CeedOperatorSetField(ceedData.build_oper, "qd", ceedData.restr_i,
                        CEED_NOTRANSPOSE,
                        CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

   // Compute the quadrature data for the diff operator.
   CeedOperatorApply(ceedData.build_oper, ceedData.node_coords, ceedData.rho,
                     CEED_REQUEST_IMMEDIATE);

   // Create the Q-function that defines the action of the diff operator.
   mech_qf = mech_qf_file + ":f_apply_mech";
   CeedQFunctionCreateInterior(ceed, 1, f_apply_mech,
                               mech_qf.c_str(),
                               &ceedData.apply_qfunc);
   CeedQFunctionAddInput(ceedData.apply_qfunc, "u", dim * dim, CEED_EVAL_GRAD);
   CeedQFunctionAddInput(ceedData.apply_qfunc, "qd", dim * dim * dim * dim,
                         CEED_EVAL_NONE);
   CeedQFunctionAddOutput(ceedData.apply_qfunc, "v", dim * dim, CEED_EVAL_GRAD);
   CeedQFunctionSetContext(ceedData.apply_qfunc, &ceedData.build_ctx,
                           sizeof(ceedData.build_ctx));

   // Create the diff operator.
   CeedOperatorCreate(ceed, ceedData.apply_qfunc, NULL, NULL, &ceedData.oper);
   CeedOperatorSetField(ceedData.oper, "u", ceedData.restr, CEED_NOTRANSPOSE,
                        ceedData.basis, CEED_VECTOR_ACTIVE);
   CeedOperatorSetField(ceedData.oper, "qd", ceedData.restr_i, CEED_NOTRANSPOSE,
                        CEED_BASIS_COLLOCATED, ceedData.rho);
   CeedOperatorSetField(ceedData.oper, "v", ceedData.restr, CEED_NOTRANSPOSE,
                        ceedData.basis, CEED_VECTOR_ACTIVE);

   CeedVectorCreate(ceed, fes.GetNDofs(), &ceedData.u);
   CeedVectorCreate(ceed, fes.GetNDofs(), &ceedData.v);
}

#endif

}
