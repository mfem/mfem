// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mesh_test_utils.hpp"

namespace mfem
{

int CheckPoisson(Mesh &mesh, int order, int disabled_boundary_attribute)
{
   constexpr int dim = 3;

   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);

   GridFunction sol(&fes);

   ConstantCoefficient one(1.0);
   BilinearForm a(&fes);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   a.Assemble();

   LinearForm b(&fes);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // Add in essential boundary conditions
   Array<int> ess_tdof_list;
   REQUIRE(mesh.bdr_attributes.Max() > 0);

   // Mark all boundaries essential
   Array<int> bdr_attr_is_ess(mesh.bdr_attributes.Max());
   bdr_attr_is_ess = 1;
   if (disabled_boundary_attribute >= 0)
   {
      bdr_attr_is_ess[mesh.bdr_attributes.Find(disabled_boundary_attribute)] = 0;
   }

   fes.GetEssentialTrueDofs(bdr_attr_is_ess, ess_tdof_list);
   REQUIRE(ess_tdof_list.Size() > 0);

   ConstantCoefficient zero(0.0);
   sol.ProjectCoefficient(zero);
   Vector B, X;
   OperatorPtr A;
   a.FormLinearSystem(ess_tdof_list, sol, b, A, X, B);

   // Solve the system
   CG(*A, B, X, 2, 1000, 1e-20, 0.0);

   // Recover the solution
   a.RecoverFEMSolution(X, b, sol);

   // Check that X solves the system A X = B.
   A->AddMult(X, B, -1.0);
   auto residual_norm = B.Norml2();
   bool satisfy_system = residual_norm < 1e-10;
   CAPTURE(residual_norm);
   CHECK(satisfy_system);

   bool satisfy_bc = true;
   Vector tvec;
   sol.GetTrueDofs(tvec);
   for (auto dof : ess_tdof_list)
   {
      if (tvec[dof] != 0.0)
      {
         satisfy_bc = false;
         break;
      }
   }
   CHECK(satisfy_bc);
   return ess_tdof_list.Size();
};

#ifdef MFEM_USE_MPI

void CheckPoisson(ParMesh &pmesh, int order,
                  int disabled_boundary_attribute)
{
   constexpr int dim = 3;

   H1_FECollection fec(order, dim);
   ParFiniteElementSpace pfes(&pmesh, &fec);

   ParGridFunction sol(&pfes);

   ConstantCoefficient one(1.0);
   ParBilinearForm a(&pfes);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   a.Assemble();
   ParLinearForm b(&pfes);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // Add in essential boundary conditions
   Array<int> ess_tdof_list;
   REQUIRE(pmesh.bdr_attributes.Max() > 0);

   Array<int> bdr_attr_is_ess(pmesh.bdr_attributes.Max());
   bdr_attr_is_ess = 1;
   if (disabled_boundary_attribute >= 0)
   {
      CAPTURE(disabled_boundary_attribute);
      bdr_attr_is_ess[pmesh.bdr_attributes.Find(disabled_boundary_attribute)] = 0;
   }

   pfes.GetEssentialTrueDofs(bdr_attr_is_ess, ess_tdof_list);
   int num_ess_dof = ess_tdof_list.Size();
   MPI_Allreduce(MPI_IN_PLACE, &num_ess_dof, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   REQUIRE(num_ess_dof > 0);


   ConstantCoefficient zero(0.0);
   sol.ProjectCoefficient(zero);
   Vector B, X;
   OperatorPtr A;
   const bool copy_interior = true; // interior(sol) --> interior(X)
   a.FormLinearSystem(ess_tdof_list, sol, b, A, X, B, copy_interior);

   // Solve the system
   CGSolver cg(MPI_COMM_WORLD);
   // HypreBoomerAMG preconditioner;
   cg.SetMaxIter(2000);
   cg.SetRelTol(1e-12);
   cg.SetPrintLevel(0);
   cg.SetOperator(*A);
   // cg.SetPreconditioner(preconditioner);
   cg.Mult(B, X);
   // Recover the solution
   a.RecoverFEMSolution(X, b, sol);

   // Check that X solves the system A X = B.
   A->AddMult(X, B, -1.0);
   auto residual_norm = B.Norml2();
   bool satisfy_system = residual_norm < 1e-10;
   CAPTURE(residual_norm);
   CHECK(satisfy_system);

   // Initialize the bdr_dof to be checked
   Vector tvec;
   sol.GetTrueDofs(tvec);
   bool satisfy_bc = true;
   for (auto dof : ess_tdof_list)
   {
      if (tvec[dof] != 0.0)
      {
         satisfy_bc = false;
         break;
      }
   }
   CHECK(satisfy_bc);
};

std::unique_ptr<ParMesh> CheckParMeshNBE(Mesh &smesh,
                                         const std::unique_ptr<int[]> &partition)
{
   auto pmesh = std::unique_ptr<ParMesh>(new ParMesh(MPI_COMM_WORLD, smesh,
                                                     partition.get()));

   int nbe = pmesh->GetNBE();
   MPI_Allreduce(MPI_IN_PLACE, &nbe, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   CHECK(nbe == smesh.GetNBE());
   return pmesh;
};

bool CheckFaceInternal(ParMesh& pmesh, int f,
                       const std::map<int, int> &local_to_shared)
{
   int e1, e2;
   pmesh.GetFaceElements(f, &e1, &e2);
   int inf1, inf2, ncface;
   pmesh.GetFaceInfos(f, &inf1, &inf2, &ncface);

   if (e2 < 0 && inf2 >=0)
   {
      // Shared face on processor boundary -> Need to discover the neighbor
      // attributes
      auto FET = pmesh.GetSharedFaceTransformations(local_to_shared.at(f));

      if (FET->Elem1->Attribute != FET->Elem2->Attribute && f < pmesh.GetNumFaces())
      {
         // shared face on domain attribute boundary, which this rank owns
         return true;
      }
   }

   if (e2 >= 0 && pmesh.GetAttribute(e1) != pmesh.GetAttribute(e2))
   {
      // local face on domain attribute boundary
      return true;
   }
   return false;
};

#endif

} // namespace mfem
