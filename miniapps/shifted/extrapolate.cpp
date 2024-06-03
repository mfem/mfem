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
//             ----------------------------------------------
//             Extrapolation Miniapp: PDE-based extrapolation
//             ----------------------------------------------
//
// This miniapp extrapolates a finite element function from a set of elements
// (known values) to the rest of the domain. The set of elements that contains
// the known values is specified by the positive values of a level set
// Coefficient. The known values are not modified. The miniapp supports two
// PDE-based approaches [1, 2], both of which rely on solving a sequence of
// advection problems in the direction of the unknown parts of the domain.
// The extrapolation can be constant (1st order), linear (2nd order), or
// quadratic (3rd order). These formal orders hold for a limited band around
// the zero level set, see the given references for more info.
//
// [1] Aslam, "A Partial Differential Equation Approach to Multidimensional
// Extrapolation", JCP 193(1), 2004.
// [2] Bochkov, Gibou, "PDE-Based Multidimensional Extrapolation of Scalar
// Fields over Interfaces with Kinks and High Curvatures", SISC 42(4), 2020.
//
// Compile with: make extrapolate
//
// Sample runs:
//     mpirun -np 4 extrapolate -m "../../data/inline-segment.mesh" -rs 6 -ed 2
//     mpirun -np 4 extrapolate -rs 5 -p 0 -ed 2
//     mpirun -np 4 extrapolate -rs 5 -p 1 -ed 2
//     mpirun -np 4 extrapolate -rs 5 -p 1 -et 1 -ed 1 -dg 1
//     mpirun -np 4 extrapolate -m "../../data/inline-hex.mesh" -ed 1 -rs 1
//     mpirun -np 4 extrapolate -m "../../data/inline-hex.mesh" -p 1 -ed 1 -rs 1

#include "extrapolator.hpp"

using namespace std;
using namespace mfem;

int problem = 0;

real_t domainLS(const Vector &coord)
{
   // Map from [0,1] to [-1,1].
   const int dim = coord.Size();
   const real_t x = coord(0)*2.0 - 1.0,
                y = (dim > 1) ? coord(1)*2.0 - 1.0 : 0.0,
                z = (dim > 2) ? coord(2)*2.0 - 1.0 : 0.0;

   switch (problem)
   {
      case 0:
      {
         // Sphere.
         return 0.75 - sqrt(x*x + y*y + z*z + 1e-12);
      }
      case 1:
      {
         // Star.
         MFEM_VERIFY(dim > 1, "Problem 1 is not applicable to 1D.");

         return 0.60 - sqrt(x*x + y*y + z*z + 1e-12) +
                0.25 * (y*y*y*y*y + 5.0*x*x*x*x*y - 10.0*x*x*y*y*y) /
                pow(x*x + y*y + z*z + 1e-12, 2.5) *
                std::cos(0.5*M_PI * z / 0.6);
      }
      default: MFEM_ABORT("Bad option for --problem!"); return 0.0;
   }
}

real_t solution0(const Vector &coord)
{
   // Map from [0,1] to [-1,1].
   const int dim = coord.Size();
   const real_t x = coord(0)*2.0 - 1.0 + 0.25,
                y = (dim > 1) ? coord(1)*2.0 - 1.0 : 0.0,
                z = (dim > 2) ? coord(2)*2.0 - 1.0 : 0.0;

   return std::cos(M_PI * x) * std::cos(M_PI * y) * std::cos(M_PI * z);
}

void PrintNorm(int myid, Vector &v, std::string text)
{
   real_t norm = v.Norml1();
   MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                 MPI_COMM_WORLD);
   if (myid == 0)
   {
      std::cout << std::setprecision(12) << std::fixed
                << text << norm << std::endl;
   }
}

void PrintIntegral(int myid, ParGridFunction &g, std::string text)
{
   ConstantCoefficient zero(0.0);
   real_t norm = g.ComputeL1Error(zero);
   if (myid == 0)
   {
      std::cout << std::setprecision(12) << std::fixed
                << text << norm << std::endl;
   }
}

int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int rs_levels = 2;
   Extrapolator::XtrapType ex_type   = Extrapolator::ASLAM;
   AdvectionOper::AdvectionMode dg_mode = AdvectionOper::HO;
   int ex_degree = 1;
   int order = 2;
   real_t distance = 0.35;
   bool vis_on = true;
   int vis_steps_cnt = 50;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption((int*)&ex_type, "-et", "--extrap-type",
                  "Extrapolation type: Aslam (0) or Bochkov (1).");
   args.AddOption((int*)&dg_mode, "-dg", "--dg-mode",
                  "DG advection mode: 0 - Standard High-Order,\n\t"
                  "                   1 - Low-Order Upwind Diffusion.");
   args.AddOption(&ex_degree, "-ed", "--extrap-degree",
                  "Extrapolation degree: 0/1/2 for constant/linear/quadratic.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&distance, "-d", "--distance",
                  "Extrapolation distance.");
   args.AddOption(&problem, "-p", "--problem",
                  "0 - 2D circle,\n\t"
                  "1 - 2D star");
   args.AddOption(&vis_on, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps_cnt, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   // Refine the mesh and distribute.
   Mesh mesh(mesh_file, 1, 1);
   for (int lev = 0; lev < rs_levels; lev++) { mesh.UniformRefinement(); }
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   const int dim = pmesh.Dimension();

   // Input function.
   L2_FECollection fec_L2(order, dim);
   ParFiniteElementSpace pfes_L2(&pmesh, &fec_L2);
   ParGridFunction u(&pfes_L2);
   FunctionCoefficient u0_coeff(solution0);
   u.ProjectCoefficient(u0_coeff);

   // Extrapolate.
   Extrapolator xtrap;
   xtrap.xtrap_type     = ex_type;
   xtrap.advection_mode = dg_mode;
   xtrap.xtrap_degree   = ex_degree;
   xtrap.visualization  = vis_on;
   xtrap.vis_steps      = vis_steps_cnt;
   FunctionCoefficient ls_coeff(domainLS);
   ParGridFunction ux(&pfes_L2);
   xtrap.Extrapolate(ls_coeff, u, distance, ux);

   PrintNorm(myid, ux, "Solution l1 norm: ");
   PrintIntegral(myid, ux, "Solution L1 norm: ");

   GridFunctionCoefficient u_exact_coeff(&u);
   real_t err_L1 = ux.ComputeL1Error(u_exact_coeff),
          err_L2 = ux.ComputeL2Error(u_exact_coeff);
   if (myid == 0)
   {
      std::cout << "Global L1 error: " << err_L1 << std::endl
                << "Global L2 error: " << err_L2 << std::endl;
   }
   real_t loc_error_L1, loc_error_L2, loc_error_LI;
   xtrap.ComputeLocalErrors(ls_coeff, u, ux,
                            loc_error_L1, loc_error_L2, loc_error_LI);
   if (myid == 0)
   {
      std::cout << "Local  L1 error: " << loc_error_L1 << std::endl
                << "Local  L2 error: " << loc_error_L2 << std::endl
                << "Local  Li error: " << loc_error_LI << std::endl;
   }

   // ParaView output.
   ParGridFunction ls_gf(&pfes_L2);
   ls_gf.ProjectCoefficient(ls_coeff);
   ParaViewDataCollection dacol("ParaViewExtrapolate", &pmesh);
   dacol.SetLevelsOfDetail(order);
   dacol.RegisterField("Level Set Function", &ls_gf);
   dacol.RegisterField("Extrapolated Solution", &ux);
   dacol.SetTime(1.0);
   dacol.SetCycle(1);
   dacol.Save();

   return 0;
}
