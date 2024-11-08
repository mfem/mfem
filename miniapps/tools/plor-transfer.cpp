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
//
//   -----------------------------------------------------------------------
//   Parallel LOR Transfer Miniapp:  Map functions between HO and LOR spaces
//   -----------------------------------------------------------------------
//
// This miniapp visualizes the maps between a high-order (HO) finite element
// space, typically using high-order functions on a high-order mesh, and a
// low-order refined (LOR) finite element space, typically defined by 0th or 1st
// order functions on a low-order refinement of the HO mesh.
//
// The grid transfer operators are represented using either
// InterpolationGridTransfer or L2ProjectionGridTransfer (depending on the
// options requested by the user). The two transfer operators are then:
//
//  1. R: HO -> LOR, defined by GridTransfer::ForwardOperator
//  2. P: LOR -> HO, defined by GridTransfer::BackwardOperator
//
// While defined generally, these operators have some nice properties for
// particular finite element spaces. For example they satisfy PR=I, plus mass
// conservation in both directions for L2 fields.
//
// Compile with: make plor-transfer
//
// Sample runs:  plor-transfer
//               plor-transfer -h1
//               plor-transfer -t
//               plor-transfer -m ../../data/star-q2.mesh -lref 5 -p 4
//               plor-transfer -m ../../data/star-mixed.mesh -lref 3 -p 2
//               plor-transfer -lref 4 -o 4 -lo 0 -p 1
//               plor-transfer -lref 5 -o 4 -lo 0 -p 1
//               plor-transfer -lref 5 -o 4 -lo 3 -p 2
//               plor-transfer -lref 5 -o 4 -lo 0 -p 3

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int problem = 1; // problem type

int Wx = 0, Wy = 0; // window position
int Ww = 350, Wh = 350; // window size
int offx = Ww+5, offy = Wh+25; // window offsets

string space;
string direction;

// Exact functions to project
real_t RHO_exact(const Vector &x);

// Helper functions
void visualize(VisItDataCollection &, string, int, int, int /* visport */);
real_t compute_mass(ParFiniteElementSpace *, real_t, VisItDataCollection &,
                    string);

int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();

   // Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int order = 2;
   int lref = order+1;
   int lorder = 0;
   bool vis = true;
   bool useH1 = false;
   int visport = 19916;
   bool use_pointwise_transfer = false;
   const char *device_config = "cpu";
   bool use_ea       = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem type (see the RHO_exact function).");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&lref, "-lref", "--lor-ref-level", "LOR refinement level.");
   args.AddOption(&lorder, "-lo", "--lor-order",
                  "LOR space order (polynomial degree, zero by default).");
   args.AddOption(&vis, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&useH1, "-h1", "--use-h1", "-l2", "--use-l2",
                  "Use H1 spaces instead of L2.");
   args.AddOption(&use_pointwise_transfer, "-t", "--use-pointwise-transfer",
                  "-no-t", "--dont-use-pointwise-transfer",
                  "Use pointwise transfer operators instead of L2 projection.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&use_ea, "-ea", "--ea-version", "-no-ea",
                  "--no-ea-version", "Use element assembly version.");
   args.ParseCheck();

   // Configure device
   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   // Read the mesh from the given mesh file.
   Mesh serial_mesh(mesh_file, 1, 1);
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();
   int dim = mesh.Dimension();

   // Make initial refinement on serial mesh.
   for (int l = 0; l < 4; l++)
   {
      mesh.UniformRefinement();
   }

   // Create the low-order refined mesh
   int basis_lor = BasisType::GaussLobatto; // BasisType::ClosedUniform;
   ParMesh mesh_lor = ParMesh::MakeRefined(mesh, lref, basis_lor);

   // Create spaces
   FiniteElementCollection *fec, *fec_lor;
   if (useH1)
   {
      space = "H1";
      if (lorder == 0)
      {
         lorder = 1;
         if (Mpi::Root())
         {
            cerr << "Switching the H1 LOR space order from 0 to 1\n";
         }
      }
      fec = new H1_FECollection(order, dim);
      fec_lor = new H1_FECollection(lorder, dim);
   }
   else
   {
      space = "L2";
      fec = new L2_FECollection(order, dim);
      fec_lor = new L2_FECollection(lorder, dim);
   }

   ParFiniteElementSpace fespace(&mesh, fec);
   ParFiniteElementSpace fespace_lor(&mesh_lor, fec_lor);

   ParGridFunction rho(&fespace);
   ParGridFunction rho_lor(&fespace_lor);

   // Data collections for vis/analysis
   VisItDataCollection HO_dc(MPI_COMM_WORLD, "HO", &mesh);
   HO_dc.RegisterField("density", &rho);
   VisItDataCollection LOR_dc(MPI_COMM_WORLD, "LOR", &mesh_lor);
   LOR_dc.RegisterField("density", &rho_lor);

   ParBilinearForm M_ho(&fespace);
   M_ho.AddDomainIntegrator(new MassIntegrator);
   M_ho.Assemble();
   M_ho.Finalize();
   HypreParMatrix* M_ho_tdof = M_ho.ParallelAssemble();

   ParBilinearForm M_lor(&fespace_lor);
   M_lor.AddDomainIntegrator(new MassIntegrator);
   M_lor.Assemble();
   M_lor.Finalize();
   HypreParMatrix* M_lor_tdof = M_lor.ParallelAssemble();

   // HO projections
   direction = "HO -> LOR @ HO";
   FunctionCoefficient RHO(RHO_exact);
   rho.ProjectCoefficient(RHO);
   // Make sure AMR constraints are satisfied
   rho.SetTrueVector();
   rho.SetFromTrueVector();

   real_t ho_mass = compute_mass(&fespace, -1.0, HO_dc, "HO       ");
   if (vis) { visualize(HO_dc, "HO", Wx, Wy, visport); Wx += offx; }

   GridTransfer *gt;
   if (use_pointwise_transfer)
   {
      gt = new InterpolationGridTransfer(fespace, fespace_lor);
   }
   else
   {
      gt = new L2ProjectionGridTransfer(fespace, fespace_lor);
   }

   // Configure element assembly for device acceleration
   gt->UseEA(use_ea);

   const Operator &R = gt->ForwardOperator();

   // HO->LOR restriction
   direction = "HO -> LOR @ LOR";
   R.Mult(rho, rho_lor);
   compute_mass(&fespace_lor, ho_mass, LOR_dc, "R(HO)    ");
   if (vis) { visualize(LOR_dc, "R(HO)", Wx, Wy, visport); Wx += offx; }
   auto global_max = [](const Vector& v)
   {
      real_t max = v.Normlinf();
      MPI_Allreduce(MPI_IN_PLACE, &max, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MAX, MPI_COMM_WORLD);
      return max;
   };

   if (gt->SupportsBackwardsOperator())
   {
      const Operator &P = gt->BackwardOperator();
      // LOR->HO prolongation
      direction = "HO -> LOR @ HO";
      ParGridFunction rho_prev = rho;
      P.Mult(rho_lor, rho);
      compute_mass(&fespace, ho_mass, HO_dc, "P(R(HO)) ");
      if (vis) { visualize(HO_dc, "P(R(HO))", Wx, Wy, visport); Wx = 0; Wy += offy; }

      rho_prev -= rho;
      Vector rho_prev_true(fespace.GetTrueVSize());
      rho_prev.GetTrueDofs(rho_prev_true);
      real_t l_inf = global_max(rho_prev_true);
      if (Mpi::Root())
      {
         cout.precision(12);
         cout << "|HO - P(R(HO))|_∞   = " << l_inf << endl;
      }
   }

   // HO* to LOR* dual fields
   ParLinearForm M_rho(&fespace), M_rho_lor(&fespace_lor);
   auto global_sum = [](const Vector& v)
   {
      real_t sum = v.Sum();
      MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, MPI_COMM_WORLD);
      return sum;
   };
   if (!use_pointwise_transfer && gt->SupportsBackwardsOperator())
   {
      Vector M_rho_true(fespace.GetTrueVSize());
      M_ho_tdof->Mult(rho.GetTrueVector(), M_rho_true);
      fespace.GetRestrictionOperator()->MultTranspose(M_rho_true, M_rho);
      const Operator &P = gt->BackwardOperator();
      P.MultTranspose(M_rho, M_rho_lor);
      real_t ho_dual_mass = global_sum(M_rho);
      real_t lor_dual_mass = global_sum(M_rho_lor);
      if (Mpi::Root())
      {
         cout << "HO -> LOR dual field: " << abs(ho_dual_mass - lor_dual_mass) << "\n\n";
      }
   }

   // LOR projections
   direction = "LOR -> HO @ LOR";
   rho_lor.ProjectCoefficient(RHO);
   ParGridFunction rho_lor_prev = rho_lor;
   real_t lor_mass = compute_mass(&fespace_lor, -1.0, LOR_dc, "LOR      ");
   if (vis) { visualize(LOR_dc, "LOR", Wx, Wy, visport); Wx += offx; }

   if (gt->SupportsBackwardsOperator())
   {
      const Operator &P = gt->BackwardOperator();
      // Prolongate to HO space
      direction = "LOR -> HO @ HO";
      P.Mult(rho_lor, rho);
      compute_mass(&fespace, lor_mass, HO_dc, "P(LOR)   ");
      if (vis) { visualize(HO_dc, "P(LOR)", Wx, Wy, visport); Wx += offx; }

      // Restrict back to LOR space. This won't give the original function because
      // the rho_lor doesn't necessarily live in the range of R.
      direction = "LOR -> HO @ LOR";
      R.Mult(rho, rho_lor);
      compute_mass(&fespace_lor, lor_mass, LOR_dc, "R(P(LOR))");
      if (vis) { visualize(LOR_dc, "R(P(LOR))", Wx, Wy, visport); }

      rho_lor_prev -= rho_lor;
      Vector rho_lor_prev_true(fespace_lor.GetTrueVSize());
      rho_lor_prev.GetTrueDofs(rho_lor_prev_true);
      real_t l_inf = global_max(rho_lor_prev_true);
      if (Mpi::Root())
      {
         cout.precision(12);
         cout << "|LOR - R(P(LOR))|_∞ = " << l_inf << endl;
      }
   }

   // LOR* to HO* dual fields
   if (!use_pointwise_transfer)
   {
      Vector M_rho_lor_true(fespace_lor.GetTrueVSize());
      M_lor_tdof->Mult(rho_lor.GetTrueVector(), M_rho_lor_true);
      fespace_lor.GetRestrictionOperator()->MultTranspose(M_rho_lor_true,
                                                          M_rho_lor);
      R.MultTranspose(M_rho_lor, M_rho);
      real_t ho_dual_mass = global_sum(M_rho);
      real_t lor_dual_mass = global_sum(M_rho_lor);

      if (Mpi::Root())
      {
         cout << "lor dual mass = " << lor_dual_mass << '\n';
         cout << "ho dual mass = " << ho_dual_mass << '\n';
         cout << "LOR -> HO dual field: " << abs(ho_dual_mass - lor_dual_mass) << '\n';
      }
   }

   delete fec;
   delete fec_lor;
   delete M_ho_tdof;
   delete M_lor_tdof;
   delete gt;

   return 0;
}


real_t RHO_exact(const Vector &x)
{
   switch (problem)
   {
      case 1: // smooth field
         return x(1)+0.25*cos(2*M_PI*x.Norml2());
      case 2: // cubic function
         return x(1)*x(1)*x(1) + 2*x(0)*x(1) + x(0);
      case 3: // sharp gradient
         return M_PI/2-atan(5*(2*x.Norml2()-1));
      case 4: // basis function
         return (x.Norml2() < 0.1) ? 1 : 0;
      default:
         return 1.0;
   }
}


void visualize(VisItDataCollection &dc, string prefix, int x, int y,
               int visport)
{
   int w = Ww, h = Wh;

   char vishost[] = "localhost";

   socketstream sol_sockL2(vishost, visport);
   sol_sockL2 << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() <<
              "\n";
   sol_sockL2.precision(8);
   sol_sockL2 << "solution\n" << *dc.GetMesh() << *dc.GetField("density")
              << "window_geometry " << x << " " << y << " " << w << " " << h
              << "plot_caption '" << space << " " << prefix << " Density'"
              << "window_title '" << direction << "'" << flush;
}


real_t compute_mass(ParFiniteElementSpace *L2, real_t massL2,
                    VisItDataCollection &dc, string prefix)
{
   ConstantCoefficient one(1.0);
   ParLinearForm lf(L2);
   lf.AddDomainIntegrator(new DomainLFIntegrator(one));
   lf.Assemble();

   real_t newmass = lf(*dc.GetParField("density"));
   if (Mpi::Root())
   {
      cout.precision(18);
      cout << space << " " << prefix << " mass   = " << newmass;
      if (massL2 >= 0)
      {
         cout.precision(4);
         cout << " ("  << fabs(newmass-massL2)*100/massL2 << "%)";
      }
      cout << endl;
   }
   return newmass;
}
