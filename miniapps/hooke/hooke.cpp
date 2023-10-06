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
// The implementation of NeoHookeanMaterial also demonstrates the use of
// automatic differentiation using either a native dual number forward mode
// implementation or leveraging the Enzyme third party library.

#include <mfem.hpp>

#include "materials/linear_elastic.hpp"
#include "materials/neohookean.hpp"
#include "operators/elasticity_gradient_operator.hpp"
#include "operators/elasticity_operator.hpp"
#include "preconditioners/diagonal_preconditioner.hpp"

using namespace std;
using namespace mfem;

/// This example only works in 3D. Kernels for 2D are not implemented.
constexpr int dimension = 3;

void display_banner(ostream& os)
{
   os << R"(
         ___ ___ ________   ________   ____  __.___________
        /   |   \\_____  \  \_____  \ |    |/ _|\_   _____/
       /    ~    \/   |   \  /   |   \|      <   |    __)_ 
       \    Y    /    |    \/    |    \    |  \  |        \
        \___|_  /\_______  /\_______  /____|__ \/_______  /
              \/         \/         \/        \/        \/ 
      )"
      << endl << flush;
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();

   int order = 1;
   const char *device_config = "cpu";
   int diagpc_type = ElasticityDiagonalPreconditioner::Type::Diagonal;
   int serial_refinement_levels = 0;
   bool visualization = true;
   bool paraview = false;

   if (Mpi::Root())
   {
      display_banner(out);
   }

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
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv",
                  "--no-paraview",
                  "Enable or disable ParaView DataCollection output.");
   args.ParseCheck();

   Device device(device_config);
   if (Mpi::Root())
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

   // Create and set the material type. We define its GradientType during
   // instantiation.

   // As seen in materials/gradient_type.hpp there is a choice of the
   // GradientType with either
   // * Symbolic (Manually derived)
   // * EnzymeFwd
   // * EnzymeRev
   // * FiniteDiff
   // * InternalFwd
   const NeoHookeanMaterial<dimension, GradientType::InternalFwd> material{};
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

   // Define all statically displaced degrees of freedom on mesh entities in the
   // defined attributes. On these degrees of freedom (determined from the mesh
   // attributes), a fixed displacement is prescribed.
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> displaced_attr(pmesh.bdr_attributes.Max());
      displaced_attr = 0;
      displaced_attr[2] = 1;
      elasticity_op.SetPrescribedDisplacement(displaced_attr);
   }

   ParGridFunction U_gf(&elasticity_op.h1_fes_);
   U_gf = 0.0;

   Vector U;
   U_gf.GetTrueDofs(U);

   // Prescribe a fixed displacement to the displaced degrees of freedom.
   U.SetSubVector(elasticity_op.GetPrescribedDisplacementTDofs(), 1.0e-2);

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

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << U_gf << flush;
   }

   if (paraview)
   {
      ParaViewDataCollection pd("elasticity_output", &pmesh);
      pd.RegisterField("solution", &U_gf);
      pd.SetLevelsOfDetail(order);
      pd.SetDataFormat(VTKFormat::BINARY);
      pd.SetHighOrderOutput(true);
      pd.SetCycle(0);
      pd.SetTime(0.0);
      pd.Save();
   }

   return 0;
}
