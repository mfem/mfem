// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "fem/intrules.hpp"
#include "lib/navier_solver.hpp"
#include "kernels/boundary_normal_stress_integrator.hpp"
#include "kernels/boundary_normal_stress_evaluator.hpp"
#include <fstream>

using namespace mfem;
using namespace navier;

void analytical_velocity(const Vector &coords, Vector &u)
{
   const double x = coords(0);
   const double y = coords(1);

   u(0) = x * y;
   u(1) = 2 * x * y;
}

double analytical_pressure(const Vector &coords)
{
   const double x = coords(0);
   const double y = coords(1);

   return x + y;
}

void analytical_stress(const Vector &coords, Vector &sigma_ne)
{
   const double x = coords(0);
   const double y = coords(1);

   //   y - x  x + 2y
   //  x + 2y  3x - y

   DenseMatrix sigma(2,2);
   sigma(0, 0) = y - x;
   sigma(1, 0) = x + 2*y;
   sigma(0, 1) = x + 2*y;
   sigma(1, 1) = 3*x - y;

   Vector ne(2);
   ne(0) = 1.0;
   ne(1) = 0.0;

   sigma_ne.SetSize(2);
   sigma.Mult(ne, sigma_ne);
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   const int polynomial_order = 2;

   Mesh mesh = Mesh::LoadFromFile("two_domain_test.mesh");
   mesh.EnsureNodes();

   const int dimension = mesh.Dimension();
   const double density = 1.0;

   Array<int> left_domain(mesh.attributes.Max());
   left_domain = 0;
   left_domain[0] = 1;

   Array<int> right_domain(mesh.attributes.Max());
   right_domain = 0;
   right_domain[1] = 1;

   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   ParSubMesh fluid_mesh = ParSubMesh::CreateFromDomain(pmesh, left_domain);
   NavierSolver navier(&fluid_mesh, polynomial_order, 1.0);

   auto u = navier.GetCurrentVelocity();
   VectorFunctionCoefficient analytical_velocity_coeff(2, analytical_velocity);
   u->ProjectCoefficient(analytical_velocity_coeff);

   auto p = navier.GetCurrentPressure();
   FunctionCoefficient analytical_pressure_coeff(analytical_pressure);
   p->ProjectCoefficient(analytical_pressure_coeff);

   auto mu = navier.GetVariableViscosity();

   Array<int> interface_marker(fluid_mesh.bdr_attributes.Max());
   interface_marker = 0;
   interface_marker[2] = 1;

   auto bdr_stress_lfi = new BoundaryNormalStressIntegrator(*u, *p, *mu);
   bdr_stress_lfi->SetIntRule(&navier.gll_ir_face);

   ParLinearForm bdr_f(u->ParFESpace());
   bdr_f.AddBoundaryIntegrator(bdr_stress_lfi, interface_marker);
   bdr_f.Assemble();

   ParBilinearForm mass_blf(u->ParFESpace());
   auto vmass_blfi = new VectorMassIntegrator;
   vmass_blfi->SetIntRule(&navier.gll_ir_face);
   mass_blf.AddBoundaryIntegrator(vmass_blfi, interface_marker);
   mass_blf.Assemble();
   mass_blf.Finalize();
   auto M = mass_blf.ParallelAssemble();

   ParGridFunction analytical_sigma_ne(u->ParFESpace());
   VectorFunctionCoefficient analytical_sigma_ne_coeff(2, analytical_stress);
   analytical_sigma_ne.ProjectBdrCoefficient(analytical_sigma_ne_coeff,
                                             interface_marker);

   Vector analytical_sigma_ne_dual(M->Height());
   M->Mult(analytical_sigma_ne, analytical_sigma_ne_dual);

   Vector diff(bdr_f);
   diff -= analytical_sigma_ne_dual;
   MFEM_ASSERT(diff.Norml2() < 1e-12, "");

   ParGridFunction sigma_ne(u->ParFESpace());
   BoundaryNormalStressEvaluator(*u, *p, *mu, interface_marker, navier.gll_ir_face,
                                 sigma_ne);

   sigma_ne -= analytical_sigma_ne;
   MFEM_ASSERT(sigma_ne.Norml2() < 1e-12, "");

   ParSubMesh solid_mesh = ParSubMesh::CreateFromDomain(pmesh, right_domain);

   H1_FECollection h1_fec(polynomial_order);
   ParFiniteElementSpace h1_vfes(&solid_mesh, &h1_fec, dimension);

   // Using a GridFunction here is signaling that we are transferring a Vdof
   // vector. Although the values represent a "dual" of a GridFunction.
   ParGridFunction boundary_traction_integrated_fluid(
      navier.GetCurrentVelocity()->ParFESpace());
   boundary_traction_integrated_fluid = bdr_f;
   ParGridFunction boundary_traction_integrated_solid(&h1_vfes);

   ParSubMesh::Transfer(boundary_traction_integrated_fluid,
                        boundary_traction_integrated_solid);

   diff = boundary_traction_integrated_fluid;
   diff -= boundary_traction_integrated_solid;
   MFEM_ASSERT(diff.Norml2() < 1e-12, "");

   return 0;
}
