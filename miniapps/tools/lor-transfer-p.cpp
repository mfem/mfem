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
//       --------------------------------------------------------------
//       LOR Transfer Miniapp:  Map functions between HO and LOR spaces
//       --------------------------------------------------------------
//       --------------------------------------------------------------
//       PARALLEL VERSION
//       --------------------------------------------------------------
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
// Compile with: make lor-transfer-p
//
// Sample runs:  lor-transfer-p
//               lor-transfer-p -h1
//               lor-transfer-p -d 'cuda'
//               lor-transfer-p -d 'hip'
//               lor-transfer-p -t
//               lor-transfer-p -m ../../data/star-q2.mesh -lref 5 -p 4
//               lor-transfer-p -m ../../data/star-mixed.mesh -lref 3 -p 2
//               lor-transfer-p -lref 4 -o 4 -lo 0 -p 1
//               lor-transfer-p -lref 5 -o 4 -lo 0 -p 1
//               lor-transfer-p -lref 5 -o 4 -lo 3 -p 2
//               lor-transfer-p -lref 5 -o 4 -lo 0 -p 3

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <chrono>

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
void visualize(VisItDataCollection &, string, int, int);
real_t compute_mass(ParFiniteElementSpace *, real_t, VisItDataCollection &,
                    string, int);

void report_time(std::chrono::duration<double>, string, int);

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // Parse command-line options.
   const char *mesh_file = "../../data/inline-hex.mesh";
   int order = 2;
   int lref = order+1;
   int lorder = 0;
   int ref_levels = 4;
   bool vis = true;
   bool useH1 = false;
   bool use_pointwise_transfer = false;
   bool use_fast_version       = true;
   bool verify_solution      = false;
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem type (see the RHO_exact function).");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
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
   args.AddOption(&use_fast_version, "-fast", "--fast-version", "-no-fast",
                  "--no-fast-version",
                  "Use fast / device friendly version.");
   args.AddOption(&verify_solution, "-verify", "--verify-solution", "-no-verify",
                  "--no-verify-solution",
                  "Verify against non-device code.");
   args.ParseCheck();

   //Helper timers
   std::chrono::time_point<std::chrono::system_clock> start, end;

   // Configure device
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // Read the mesh from the given mesh file.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // Make initial refinement on serial mesh.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Create the low-order refined mesh
   int basis_lor = BasisType::ClosedUniform; //BasisType::GaussLobatto; //
   ParMesh pmesh_lor = ParMesh::MakeRefined(pmesh, lref, basis_lor);

   // Create spaces
   FiniteElementCollection *fec, *fec_lor;
   if (useH1)
   {
      space = "H1";
      if (lorder == 0)
      {
         lorder = 1;
         if (myid == 0) {cerr << "Switching the H1 LOR space order from 0 to 1\n";}
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

   ParFiniteElementSpace fespace(&pmesh, fec);
   ParFiniteElementSpace fespace_lor(&pmesh_lor, fec_lor);

   ParGridFunction rho(&fespace);
   ParGridFunction rho_lor(&fespace_lor);

   // Data collections for vis/analysis
   VisItDataCollection HO_dc("HO", &pmesh);
   HO_dc.RegisterField("density", &rho);
   VisItDataCollection LOR_dc("LOR", &pmesh_lor);
   LOR_dc.RegisterField("density", &rho_lor);

   ParBilinearForm M_ho(&fespace);
   M_ho.AddDomainIntegrator(new MassIntegrator);
   M_ho.Assemble();
   M_ho.Finalize();

   ParBilinearForm M_lor(&fespace_lor);
   M_lor.AddDomainIntegrator(new MassIntegrator);
   M_lor.Assemble();
   M_lor.Finalize();

   // HO projections
   direction = "HO -> LOR @ HO";
   FunctionCoefficient RHO(RHO_exact);
   rho.ProjectCoefficient(RHO);

   // Make sure AMR constraints are satisfied
   rho.SetTrueVector();
   rho.SetFromTrueVector();

   real_t ho_mass = compute_mass(&fespace, -1.0, HO_dc, "HO       ",myid);
   if (vis) { visualize(HO_dc, "HO", Wx, Wy); Wx += offx; }

   GridTransfer *gt;
   if (use_pointwise_transfer)
   {
      gt = new InterpolationGridTransfer(fespace, fespace_lor);
   }
   else
   {
      gt = new L2ProjectionGridTransfer(fespace, fespace_lor, false);
   }

   gt->UseDevice(use_fast_version);
   gt->VerifySolution(verify_solution);

   start = std::chrono::system_clock::now();
   const Operator &R = gt->ForwardOperator();
   end = std::chrono::system_clock::now();

   std::chrono::duration<double> R_fwd_elapsed = end - start;
   report_time(R_fwd_elapsed, "R fwd elapsed time: ", myid);


   // HO->LOR restriction
   direction = "HO -> LOR @ LOR";
   start = std::chrono::system_clock::now();
   R.Mult(rho, rho_lor);
   end = std::chrono::system_clock::now();

   std::chrono::duration<double> R_fwd_mult_elapsed = end - start;
   report_time(R_fwd_mult_elapsed,"R fwd mult elapsed time: ", myid);

   compute_mass(&fespace_lor, ho_mass, LOR_dc, "R(HO)    ", myid);
   if (vis) { visualize(LOR_dc, "R(HO)", Wx, Wy); Wx += offx; }

   if (gt->SupportsBackwardsOperator())
   {
      start = std::chrono::system_clock::now();
      const Operator &P = gt->BackwardOperator();
      end = std::chrono::system_clock::now();
      std::chrono::duration<double> P_bwd_elapsed = end - start;

      report_time(P_bwd_elapsed,"P bwd elapsed time: ", myid);

      // LOR->HO prolongation
      direction = "HO -> LOR @ HO";
      ParGridFunction rho_prev = rho;
      start = std::chrono::system_clock::now();
      P.Mult(rho_lor, rho);
      end = std::chrono::system_clock::now();

      std::chrono::duration<double> P_bwd_mult_elapsed = end - start;
      report_time(P_bwd_mult_elapsed,"P bwd mult elapsed time: ", myid);

      compute_mass(&fespace, ho_mass, HO_dc, "P(R(HO)) ", myid);
      if (vis) { visualize(HO_dc, "P(R(HO))", Wx, Wy); Wx = 0; Wy += offy; }

      rho_prev -= rho;
      cout.precision(12);

      double rho_prev_inf = rho_prev.Normlinf();
      MPI_Allreduce(MPI_IN_PLACE, &rho_prev_inf, 1, MPI_DOUBLE, MPI_MAX,
                    MPI_COMM_WORLD);
      if (myid == 0) { cout << "|HO - P(R(HO))|_∞   = " << rho_prev_inf << endl; }
   }
   // exit(0);

   // HO* to LOR* dual fields
   ParLinearForm M_rho(&fespace), M_rho_lor(&fespace_lor);
   if (!use_pointwise_transfer && gt->SupportsBackwardsOperator())
   {
      start = std::chrono::system_clock::now();
      const Operator &P = gt->BackwardOperator();
      end = std::chrono::system_clock::now();

      std::chrono::duration<double> P_bwd_elapsed = end - start;
      report_time(P_bwd_elapsed,"P bwd elapsed time: ", myid);

      M_ho.Mult(rho, M_rho);

      start = std::chrono::system_clock::now();
      P.MultTranspose(M_rho, M_rho_lor);
      end = std::chrono::system_clock::now();
      std::chrono::duration<double> P_bwd_multT_elapsed = end - start;
      report_time(P_bwd_multT_elapsed,"P bwd multT elapsed elapsed time: ", myid);

      double M_rho_lor_diff = abs(M_rho.Sum()-M_rho_lor.Sum());
      MPI_Allreduce(MPI_IN_PLACE, &M_rho_lor_diff, 1, MPI_DOUBLE, MPI_MAX,
                    MPI_COMM_WORLD);
      if (myid == 0) { cout << "HO -> LOR dual field: " << abs(M_rho.Sum()-M_rho_lor.Sum()) << "\n\n"; }
   }

   // LOR projections
   direction = "LOR -> HO @ LOR";
   rho_lor.ProjectCoefficient(RHO);
   ParGridFunction rho_lor_prev = rho_lor;
   real_t lor_mass = compute_mass(&fespace_lor, -1.0, LOR_dc, "LOR      ",myid);
   if (vis) { visualize(LOR_dc, "LOR", Wx, Wy); Wx += offx; }

   if (gt->SupportsBackwardsOperator())
   {
      start = std::chrono::system_clock::now();
      const Operator &P = gt->BackwardOperator();
      end = std::chrono::system_clock::now();

      std::chrono::duration<double> P_bwd_elapsed = end - start;
      report_time(P_bwd_elapsed,"P bwd elapsed time: ", myid);

      // Prolongate to HO space
      direction = "LOR -> HO @ HO";
      start = std::chrono::system_clock::now();
      P.Mult(rho_lor, rho);
      end = std::chrono::system_clock::now();

      std::chrono::duration<double> P_bwd_mult_elapsed = end - start;
      report_time(P_bwd_mult_elapsed,"P bwd mult elapsed time: ", myid);

      compute_mass(&fespace, lor_mass, HO_dc, "P(LOR)   ",myid);
      if (vis) { visualize(HO_dc, "P(LOR)", Wx, Wy); Wx += offx; }

      // Restrict back to LOR space. This won't give the original function because
      // the rho_lor doesn't necessarily live in the range of R.
      direction = "LOR -> HO @ LOR";

      start = std::chrono::system_clock::now();
      R.Mult(rho, rho_lor);
      end = std::chrono::system_clock::now();
      std::chrono::duration<double> R_fwd_mult_elapsed = end - start;
      report_time(R_fwd_mult_elapsed,"R fwd mult elapsed time: ", myid);

      compute_mass(&fespace_lor, lor_mass, LOR_dc, "R(P(LOR))",myid);
      if (vis) { visualize(LOR_dc, "R(P(LOR))", Wx, Wy); }

      rho_lor_prev -= rho_lor;

      double rho_lor_prev_inf = rho_lor_prev.Normlinf();
      MPI_Allreduce(MPI_IN_PLACE, &rho_lor_prev, 1, MPI_DOUBLE, MPI_MAX,
                    MPI_COMM_WORLD);
      cout.precision(12);
      if (myid == 0) { cout << "|LOR - R(P(LOR))|_∞ = " << rho_lor_prev_inf << endl; }
   }

   // LOR* to HO* dual fields
   if (!use_pointwise_transfer)
   {
      M_lor.Mult(rho_lor, M_rho_lor);
      start = std::chrono::system_clock::now();
      R.MultTranspose(M_rho_lor, M_rho);
      end = std::chrono::system_clock::now();
      std::chrono::duration<double> R_fwd_multT_elapsed = end - start;
      report_time(R_fwd_multT_elapsed,"R fwd multT elapsed time: ", myid);

      if (myid ==0) { cout << "LOR -> HO dual field: " << abs(M_rho.Sum() - M_rho_lor.Sum()) << '\n'; }
   }

   delete fec;
   delete fec_lor;
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


void visualize(VisItDataCollection &dc, string prefix, int x, int y)
{
   int w = Ww, h = Wh;

   char vishost[] = "localhost";
   int  visport   = 19916;

   socketstream sol_sockL2(vishost, visport);
   sol_sockL2.precision(8);
   sol_sockL2 << "solution\n" << *dc.GetMesh() << *dc.GetField("density")
              << "window_geometry " << x << " " << y << " " << w << " " << h
              << "plot_caption '" << space << " " << prefix << " Density'"
              << "window_title '" << direction << "'" << flush;
}


real_t compute_mass(ParFiniteElementSpace *L2, real_t massL2,
                    VisItDataCollection &dc, string prefix, int myid)
{
   ConstantCoefficient one(1.0);
   ParLinearForm lf(L2);
   lf.AddDomainIntegrator(new DomainLFIntegrator(one));
   lf.Assemble();

   ParGridFunction* pdensity = dynamic_cast<ParGridFunction *>
                               (dc.GetField("density"));
   real_t newmass = lf(*pdensity);
   cout.precision(18);

   MPI_Allreduce(MPI_IN_PLACE, &newmass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

   if (myid == 0)
   {
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


void report_time(std::chrono::duration<double> elapsed_time, string name,
                 int myid)
{
   double elapsed_val = elapsed_time.count();
   double walltime = elapsed_val;
   MPI_Allreduce(MPI_IN_PLACE, &elapsed_val, 1, MPI_DOUBLE, MPI_SUM,
                 MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &walltime, 1, MPI_DOUBLE, MPI_MAX,
                 MPI_COMM_WORLD);

   if (myid == 0)
   {
      mfem::out << "wall time " << name << walltime << "s\n";
   }

}
