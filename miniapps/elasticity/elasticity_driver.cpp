// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// This miniapp solves a quasistatic solid mechanics problem assuming an elastic
// material and no body forces.
//
// The equation
//                   ∇⋅σ(∇u) = 0
//
// with stress σ is solved for displacement u.
//
//             +----------+----------+
//   fixed --->|                     |<--- constant displacement
//             |                     |
//             +----------+----------+
//
// This miniapp uses an elasticity operator that allows for a custom material.
// By default the NeoHookeanMaterial is used. A linear elastic material is also
// provided. Based on these examples, other materials could be implemented.
//
// The implementation of NeoHookeanMaterial also demenstrates the use of
// automatic differentiation using either a native dual number implementation or
// leveraging the Enzyme third party library.

#include <mfem.hpp>

#include "materials/linear_elastic.hpp"
#include "materials/neohookean.hpp"
#include "operators/elasticity_gradient_operator.hpp"
#include "operators/elasticity_operator.hpp"
#include "preconditioners/diagonal_preconditioner.hpp"

using namespace std;
using namespace mfem;

/// This example only works in 3D.
constexpr int dimension = 3;

int main(int argc, char *argv[])
{
   MPI_Session mpi;
   int myid = mpi.WorldRank();

   int order = 1;
   const char *device_config = "cpu";
   int diagpc_type = ElasticityDiagonalPreconditioner::Type::Diagonal;
   int serial_refinement_levels = 0;
   bool paraview = false;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&diagpc_type, "-pc", "--pctype",
                  "Select diagonal preconditioner type"
                  " (0:Diagonal, 1:BlockDiagonal).");
   args.AddOption(&serial_refinement_levels, "-rs", "--ref-serial",
                  "Number of uniform refinements on the serial mesh.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv",
                  "--no-paraview",
                  "Enable or disable ParaView DataCollection.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   Device device(device_config);
   if (myid == 0)
   {
      device.Print();
   }

   auto mesh =
      Mesh::MakeCartesian3D(8, 2, 2, Element::HEXAHEDRON, 8.0, 1.0, 1.0);
   if (mesh.Dimension() != dimension)
   {
      MFEM_ABORT("This example only works in 3D.");
   }
   mesh.EnsureNodes();

   for (int l = 0; l < serial_refinement_levels; l++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Create the elasticity operator on the parallel mesh.
   ElasticityOperator elasticity_op(pmesh, order);

   // Create and set the material type. We define it's GradientType during
   // instantiation.

   // As seen in materials/gradient_type.hpp there is a choice of the
   // GradientType with either
   // * Symbolic
   // * EnzymeFwd
   // * EnzymeRev
   // * FiniteDiff
   // * DualNumbers
   const NeoHookeanMaterial<dimension, GradientType::DualNumbers> material{};
   elasticity_op.SetMaterial(material);

   // Define all essential boundaries. In this specific example, this includes
   // all fixed and statically displaced degrees of freedom on mesh entities in
   // the defined attributes.
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_attr(pmesh.bdr_attributes.Max());
      ess_attr = 0;
      ess_attr[4] = 1;
      ess_attr[2] = 1;
      elasticity_op.SetEssentialAttributes(ess_attr);
   }

   // Define all statically displaced mesh attributes. On these degrees of
   // freedom (determined from the mesh attributes), a fixed displacement is
   // prescribed.
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> displaced_attr(pmesh.bdr_attributes.Max());
      displaced_attr = 0;
      displaced_attr[2] = 1;
      elasticity_op.SetDisplacedAttributes(displaced_attr);
   }

   ParGridFunction U_gf(&elasticity_op.h1_fes_);
   U_gf = 0.0;

   Vector U;
   U_gf.GetTrueDofs(U);

   // Prescribe a fixed displacement to the displaced degrees of freedom.
   U.SetSubVector(elasticity_op.GetDisplacedTDofs(), 1.0e-2);

   // Define the type of preconditioner to use for the linear solver.
   ElasticityDiagonalPreconditioner diagonal_pc(
      static_cast<ElasticityDiagonalPreconditioner::Type>(diagpc_type));

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-1);
   cg.SetMaxIter(10000);
   cg.SetPrintLevel(2);
   cg.SetPreconditioner(diagonal_pc);

   NewtonSolver newton(MPI_COMM_WORLD);
   newton.SetSolver(cg);
   newton.SetOperator(elasticity_op);
   newton.SetRelTol(1e-6);
   newton.SetMaxIter(10);
   newton.SetPrintLevel(1);

   Vector zero;
   newton.Mult(zero, U);

   U_gf.Distribute(U);

   if (paraview)
   {
      ParaViewDataCollection *pd = NULL;
      pd = new ParaViewDataCollection("elasticity_output", &pmesh);
      pd->RegisterField("solution", &U_gf);
      pd->SetLevelsOfDetail(order);
      pd->SetDataFormat(VTKFormat::BINARY);
      pd->SetHighOrderOutput(true);
      pd->SetCycle(0);
      pd->SetTime(0.0);
      pd->Save();
      delete pd;
   }

   return 0;
}
