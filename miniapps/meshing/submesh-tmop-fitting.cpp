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
//
//    --------------------------------------------------------------
//              Boundary and Interface Fitting Miniapp
//    --------------------------------------------------------------
//
// This miniapp performs mesh optimization for controlling mesh quality and
// aligning a selected set of nodes to boundary and/or interface of interest
// defined using a level-set function. The mesh quality aspect is based on a
// variational formulation of the Target-Matrix Optimization Paradigm (TMOP).
// Boundary/interface alignment is weakly enforced using a penalization term
// that moves a selected set of nodes towards the zero level set of a signed
// smooth discrete function. See the following papers for more details:
// (1) "Adaptive Surface Fitting and Tangential Relaxation for High-Order Mesh Optimization" by
//     Knupp, Kolev, Mittal, Tomov.
// (2) "Implicit High-Order Meshing using Boundary and Interface Fitting" by
//     Barrera, Kolev, Mittal, Tomov.
// (3) "The target-matrix optimization paradigm for high-order meshes" by
//     Dobrev, Knupp, Kolev, Mittal, Tomov.

// Compile with: make tmop-fitting
// Sample runs:
//  Interface fitting:
//    mpirun -np 4 tmop-fitting -m square01.mesh -o 3 -rs 1 -mid 58 -tid 1 -ni 200 -vl 1 -sfc 5e4 -rtol 1e-5
//    mpirun -np 4 tmop-fitting -m square01-tri.mesh -o 3 -rs 0 -mid 58 -tid 1 -ni 200 -vl 1 -sfc 1e4 -rtol 1e-5
//  Surface fitting with weight adaptation and termination based on fitting error
//    mpirun -np 4 tmop-fitting -m square01.mesh -o 2 -rs 1 -mid 2 -tid 1 -ni 100 -vl 2 -sfc 10 -rtol 1e-20 -st 0 -sfa 10.0 -sft 1e-5
//  Fitting to Fischer-Tropsch reactor like domain
//  * mpirun -np 6 tmop-fitting -m square01.mesh -o 2 -rs 4 -mid 2 -tid 1 -vl 2 -sfc 100 -rtol 1e-12 -ni 100 -ae 1 -bnd -sbgmesh -slstype 2 -smtype 0 -sfa 10.0 -sft 1e-4 -amriter 7 -dist

//Submesh
// square (squircle) inside a circle
// make submesh-tmop-fitting -j && time mpirun -np 1 submesh-tmop-fitting -m ../../data/inline-tri.mesh -o 1 -rs 4 -mid 2 -tid 1 -ni 100 -sfc 10 -rtol 1e-12 -ae 1 -sfa 100 -st 0 -qo 8 -marking -sft 1e-4 -deact 2 -vis -trim -smtype 1 -fix-bnd -sbgmesh -amriter 4 -slstype 3 -vl 0
// make submesh-tmop-fitting -j && time mpirun -np 6 submesh-tmop-fitting -m ../../data/inline-tri.mesh -o 1 -rs 4 -mid 2 -tid 1 -ni 100 -sfc 100 -rtol 1e-12 -ae 1 -sfa 100 -st 0 -qo 8 -marking -sft 1e-3 -deact 2 -no-vis -trim -smtype 1 -fix-bnd -sbgmesh -amriter 4 -slstype 3 -vl 0
// make submesh-tmop-fitting -j && time mpirun -np 1 submesh-tmop-fitting -m ../../data/inline-tri.mesh -o 1 -rs 4 -mid 2 -tid 1 -ni 100 -sfc 10 -rtol 1e-12 -ae 1 -sfa 100 -st 0 -qo 8 -marking -sft 1e-4 -deact 4 -vis -smtype 0 -fix-bnd -sbgmesh -amriter 4 -slstype 5 -vl 0 -htot -2
// 3D version
// make submesh-tmop-fitting -j && time mpirun -np 6 submesh-tmop-fitting -m ../../data/inline-tet.mesh -o 1 -rs 3 -mid 303 -tid 1 -ni 100 -sfc 10 -rtol 1e-12 -ae 1 -sfa 100 -st 0 -qo 8 -marking -sft 1e-4 -deact 0 -vis -trim -smtype 1 -fix-bnd -sbgmesh -amriter 4 -slstype 4 -vl 2
// make submesh-tmop-fitting -j && time mpirun -np 6 submesh-tmop-fitting -m ../../data/inline-tet.mesh -o 1 -rs 3 -mid 303 -tid 3 -ni 100 -sfc 100 -rtol 1e-12 -ae 1 -sfa 100 -st 0 -qo 8 -marking -sft 1e-4 -deact 2 -vis -trim -smtype 1 -fix-bnd -sbgmesh -amriter 5 -slstype 4 -vl 2
// make submesh-tmop-fitting -j && time mpirun -np 6 submesh-tmop-fitting -o 1 -rs 2 -mid 303 -tid 3 -ni 100 -sfc 100 -rtol 1e-12 -ae 1 -sfa 10 -st 0 -qo 8 -marking -sft 10e-5 -deact 0 -vis -trim -smtype 1 -fix-bnd -amriter 4 -slstype 4 -vl 2 -sbgmesh

// Boundary fitting
// make submesh-tmop-fitting -j && time mpirun -np 6 submesh-tmop-fitting -o 1 -rs 4 -mid 303 -tid 4 -ni 100 -sfc 100 -rtol 1e-12 -ae 1 -sfa 10 -st 0 -qo 8 -marking -sft 10e-5 -deact 0 -vis -trim -smtype 1 -fix-bnd -amriter 4 -slstype 4 -vl 2 -sbgmesh -htot 0 -m ../../data/inline-tet.mesh
// make submesh-tmop-fitting -j && time mpirun -np 6 submesh-tmop-fitting -o 1 -rs 4 -mid 303 -tid 4 -ni 100 -sfc 100 -rtol 1e-12 -ae 1 -sfa 10 -st 0 -qo 8 -marking -sft 10e-5 -deact 0 -vis -trim -smtype 1 -fix-bnd -amriter 4 -slstype 4 -vl 2 -sbgmesh -htot 2

// Interface
// make submesh-tmop-fitting -j && time mpirun -np 2 submesh-tmop-fitting -o 1 -rs 3 -mid 303 -tid 4 -ni 100 -sfc 100 -rtol 1e-12 -ae 1 -sfa 10 -st 0 -qo 8 -marking -sft 10e-5 -deact 0 -vis -smtype 0 -fix-bnd -amriter 4 -slstype 4 -vl 2 -sbgmesh -htot 2
// make submesh-tmop-fitting -j && time mpirun -np 6 submesh-tmop-fitting -o 1 -rs 2 -mid 303 -tid 4 -ni 100 -sfc 100 -rtol 1e-12 -ae 1 -sfa 10 -st 0 -qo 8 -marking -sft 10e-5 -deact 2 -vis -smtype 0 -fix-bnd -amriter 4 -slstype 4 -vl 2 -sbgmesh -htot 2
// make submesh-tmop-fitting -j && time mpirun -np 6 submesh-tmop-fitting -o 1 -rs 3 -mid 303 -tid 4 -ni 100 -sfc 100 -rtol 1e-12 -ae 1 -sfa 10 -st 0 -qo 8 -marking -sft 10e-5 -deact 2 -vis -smtype 0 -fix-bnd -amriter 4 -slstype 4 -vl 2 -sbgmesh -htot 2
// smooth
// 2D - make submesh-tmop-fitting -j && time mpirun -np 1 submesh-tmop-fitting -o 1 -rs 6 -mid 2 -tid 1 -ni 100 -sfc 10 -rtol 1e-12 -ae 1 -sfa 100 -st 0 -qo 8 -marking -sft 1e-4 -vis -fix-bnd -sbgmesh -amriter 4 -slstype 7 -vl 2 -htot -2 -deact 4
// 3D - make submesh-tmop-fitting -j && time mpirun -np 6 submesh-tmop-fitting -o 1 -rs 2 -mid 303 -tid 4 -ni 100 -sfc 1000 -rtol 1e-12 -ae 1 -sfa 10 -st 0 -qo 8 -marking -sft 10e-5 -deact 2 -vis -smtype 0 -fix-bnd -amriter 4 -slstype 7 -vl 2 -sbgmesh -htot 2


// 3D - cube+sphere:
// make submesh-tmop-fitting -j && time mpirun -np 6 submesh-tmop-fitting -o 1 -rs 4 -mid 303 -tid 4 -ni 100 -sfc 100 -rtol 1e-12 -ae 1 -sfa 10 -st 0 -qo 8 -marking -sft 10e-5 -vis -smtype 0 -fix-bnd -amriter 3 -slstype 4 -vl 2 -sbgmesh -htot 2 -deact 2
#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include <iostream>
#include <fstream>
#include "tmop-fitting.hpp"

using namespace mfem;
using namespace std;

void ExtendRefinementListToNeighbors(ParMesh &pmesh, Array<int> &intel)
{
   mfem::L2_FECollection l2fec(0, pmesh.Dimension());
   mfem::ParFiniteElementSpace l2fespace(&pmesh, &l2fec);
   mfem::ParGridFunction el_to_refine(&l2fespace);
   const int quad_order = 4;

   el_to_refine = 0.0;

   for (int i = 0; i < intel.Size(); i++)
   {
      el_to_refine(intel[i]) = 1.0;
   }

   mfem::H1_FECollection lhfec(1, pmesh.Dimension());
   mfem::ParFiniteElementSpace lhfespace(&pmesh, &lhfec);
   mfem::ParGridFunction lhx(&lhfespace);

   el_to_refine.ExchangeFaceNbrData();
   GridFunctionCoefficient field_in_dg(&el_to_refine);
   lhx.ProjectDiscCoefficient(field_in_dg, GridFunction::ARITHMETIC);

   IntegrationRules irRules = IntegrationRules(0, Quadrature1D::GaussLobatto);
   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      Array<int> dofs;
      Vector x_vals;
      lhfespace.GetElementDofs(e, dofs);
      const IntegrationRule &ir =
         irRules.Get(pmesh.GetElementGeometry(e), quad_order);
      lhx.GetValues(e, ir, x_vals);
      double max_val = x_vals.Max();
      if (max_val > 0)
      {
         intel.Append(e);
      }
   }

   intel.Sort();
   intel.Unique();
}

void GetBoundaryElements(ParMesh *pmesh, ParGridFunction &mat,
                         Array<int> &intel, int attr_to_match)
{
   intel.SetSize(0);
   mat.ExchangeFaceNbrData();
   const int NElem = pmesh->GetNE();
   MFEM_VERIFY(mat.Size() == NElem, "Material GridFunction should be a piecewise"
               "constant function over the mesh.");
   for (int f = 0; f < pmesh->GetNBE(); f++ )
   {
      int el;
      int info;
      pmesh->GetBdrElementAdjacentElement(f, el, info);
      if (pmesh->GetBdrAttribute(f) == attr_to_match)
      {
         intel.Append(el);
      }
   }
}

void GetMaterialInterfaceElements(ParMesh *pmesh, ParGridFunction &mat,
                                  Array<int> &intel)
{
   intel.SetSize(0);
   mat.ExchangeFaceNbrData();
   const int NElem = pmesh->GetNE();
   MFEM_VERIFY(mat.Size() == NElem, "Material GridFunction should be a piecewise"
               "constant function over the mesh.");
   for (int f = 0; f < pmesh->GetNumFaces(); f++ )
   {
      Array<int> nbrs;
      pmesh->GetFaceAdjacentElements(f,nbrs);
      Vector matvals;
      Array<int> vdofs;
      Vector vec;
      Array<int> els;
      //if there is more than 1 element across the face.
      if (nbrs.Size() > 1)
      {
         matvals.SetSize(nbrs.Size());
         for (int j = 0; j < nbrs.Size(); j++)
         {
            if (nbrs[j] < NElem)
            {
               matvals(j) = mat(nbrs[j]);
               els.Append(nbrs[j]);
            }
            else
            {
               const int Elem2NbrNo = nbrs[j] - NElem;
               mat.ParFESpace()->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs);
               mat.FaceNbrData().GetSubVector(vdofs, vec);
               matvals(j) = vec(0);
            }
         }
         if (matvals(0) != matvals(1))
         {
            intel.Append(els);
         }
      }
   }
}

void MakeMaterialConsistentForElementGroups(ParGridFunction &mat,
                                            ParGridFunction &pgl_el_num,
                                            int nel_per_group)
{
   ParFiniteElementSpace *pfespace = mat.ParFESpace();
   ParMesh *pmesh = pfespace->GetParMesh();
   GSLIBGroupCommunicator gslib = GSLIBGroupCommunicator(pmesh->GetComm());

   pmesh->GetGlobalElementNum(0);//To compute global offset
   Array<long long> ids(pmesh->GetNE());
   for (int e = 0; e < pmesh->GetNE(); e++)
   {
      long long gl_el_num = pgl_el_num(e);
      long long group_num = (gl_el_num - gl_el_num % nel_per_group)/nel_per_group + 1;
      ids[e] = (long long)group_num;
   }

   gslib.Setup(ids);
   gslib.GOP(mat, 2);
   gslib.FreeData();
}

int main (int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   Hypre::Init();
   int nranks = Mpi::WorldSize();

   // 1. Set the method's default parameters.
   const char *mesh_file = "icf.mesh";
   int mesh_poly_deg     = 1;
   int rs_levels         = 0;
   int rp_levels         = 0;
   int metric_id         = 1;
   int target_id         = 1;
   double surface_fit_const = 0.0;
   int quad_type         = 1;
   int quad_order        = 8;
   int solver_type       = 0;
   int solver_iter       = 20;
   double solver_rtol    = 1e-10;
   int solver_art_type   = 0;
   int lin_solver        = 2;
   int max_lin_iter      = 100;
   bool move_bnd         = true;
   bool visualization    = true;
   int verbosity_level   = 0;
   int adapt_eval        = 0;
   const char *devopt    = "cpu";
   double surface_fit_adapt = 0.0;
   double surface_fit_threshold = -10;
   bool adapt_marking     = false;
   bool surf_bg_mesh     = false;
   bool comp_dist     = false;
   int surf_ls_type      = 1;
   int marking_type      = 0;
   bool mod_bndr_attr    = false;
   bool material         = false;
   int mesh_node_ordering = 0;
   int amr_iters         = 0;
   int int_amr_iters     = 0;
   int deactivation_layers = 0;
   bool twopass            = false;
   bool trim_mesh        = false;
   int hex_to_tet_split_type = 0;
   bool mesh_save        = false;

   // 2. Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&metric_id, "-mid", "--metric-id",
                  "Mesh optimization metric:\n\t"
                  "T-metrics\n\t"
                  "1  : |T|^2                          -- 2D shape\n\t"
                  "2  : 0.5|T|^2/tau-1                 -- 2D shape (condition number)\n\t"
                  "7  : |T-T^-t|^2                     -- 2D shape+size\n\t"
                  "9  : tau*|T-T^-t|^2                 -- 2D shape+size\n\t"
                  "14 : |T-I|^2                        -- 2D shape+size+orientation\n\t"
                  "22 : 0.5(|T|^2-2*tau)/(tau-tau_0)   -- 2D untangling\n\t"
                  "50 : 0.5|T^tT|^2/tau^2-1            -- 2D shape\n\t"
                  "55 : (tau-1)^2                      -- 2D size\n\t"
                  "56 : 0.5(sqrt(tau)-1/sqrt(tau))^2   -- 2D size\n\t"
                  "58 : |T^tT|^2/(tau^2)-2*|T|^2/tau+2 -- 2D shape\n\t"
                  "77 : 0.5(tau-1/tau)^2               -- 2D size\n\t"
                  "80 : (1-gamma)mu_2 + gamma mu_77    -- 2D shape+size\n\t"
                  "85 : |T-|T|/sqrt(2)I|^2             -- 2D shape+orientation\n\t"
                  "98 : (1/tau)|T-I|^2                 -- 2D shape+size+orientation\n\t"
                  // "211: (tau-1)^2-tau+sqrt(tau^2+eps)  -- 2D untangling\n\t"
                  // "252: 0.5(tau-1)^2/(tau-tau_0)       -- 2D untangling\n\t"
                  "301: (|T||T^-1|)/3-1              -- 3D shape\n\t"
                  "302: (|T|^2|T^-1|^2)/9-1          -- 3D shape\n\t"
                  "303: (|T|^2)/3*tau^(2/3)-1        -- 3D shape\n\t"
                  // "311: (tau-1)^2-tau+sqrt(tau^2+eps)-- 3D untangling\n\t"
                  "313: (|T|^2)(tau-tau0)^(-2/3)/3   -- 3D untangling\n\t"
                  "315: (tau-1)^2                    -- 3D size\n\t"
                  "316: 0.5(sqrt(tau)-1/sqrt(tau))^2 -- 3D size\n\t"
                  "321: |T-T^-t|^2                   -- 3D shape+size\n\t"
                  // "352: 0.5(tau-1)^2/(tau-tau_0)     -- 3D untangling\n\t"
                  "A-metrics\n\t"
                  "11 : (1/4*alpha)|A-(adjA)^T(W^TW)/omega|^2 -- 2D shape\n\t"
                  "36 : (1/alpha)|A-W|^2                      -- 2D shape+size+orientation\n\t"
                  "107: (1/2*alpha)|A-|A|/|W|W|^2             -- 2D shape+orientation\n\t"
                  "126: (1-gamma)nu_11 + gamma*nu_14a         -- 2D shape+size\n\t"
                 );
   args.AddOption(&target_id, "-tid", "--target-id",
                  "Target (ideal element) type:\n\t"
                  "1: Ideal shape, unit size\n\t"
                  "2: Ideal shape, equal size\n\t"
                  "3: Ideal shape, initial size\n\t"
                  "4: Given full analytic Jacobian (in physical space)\n\t"
                  "5: Ideal shape, given size (in physical space)");
   args.AddOption(&surface_fit_const, "-sfc", "--surface-fit-const",
                  "Surface preservation constant.");
   args.AddOption(&quad_type, "-qt", "--quad-type",
                  "Quadrature rule type:\n\t"
                  "1: Gauss-Lobatto\n\t"
                  "2: Gauss-Legendre\n\t"
                  "3: Closed uniform points");
   args.AddOption(&quad_order, "-qo", "--quad_order",
                  "Order of the quadrature rule.");
   args.AddOption(&solver_type, "-st", "--solver-type",
                  " Type of solver: (default) 0: Newton, 1: LBFGS");
   args.AddOption(&solver_iter, "-ni", "--newton-iters",
                  "Maximum number of Newton iterations.");
   args.AddOption(&solver_rtol, "-rtol", "--newton-rel-tolerance",
                  "Relative tolerance for the Newton solver.");
   args.AddOption(&solver_art_type, "-art", "--adaptive-rel-tol",
                  "Type of adaptive relative linear solver tolerance:\n\t"
                  "0: None (default)\n\t"
                  "1: Eisenstat-Walker type 1\n\t"
                  "2: Eisenstat-Walker type 2");
   args.AddOption(&lin_solver, "-ls", "--lin-solver",
                  "Linear solver:\n\t"
                  "0: l1-Jacobi\n\t"
                  "1: CG\n\t"
                  "2: MINRES\n\t"
                  "3: MINRES + Jacobi preconditioner\n\t"
                  "4: MINRES + l1-Jacobi preconditioner");
   args.AddOption(&max_lin_iter, "-li", "--lin-iter",
                  "Maximum number of iterations in the linear solve.");
   args.AddOption(&move_bnd, "-bnd", "--move-boundary", "-fix-bnd",
                  "--fix-boundary",
                  "Enable motion along horizontal and vertical boundaries.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&verbosity_level, "-vl", "--verbosity-level",
                  "Set the verbosity level - 0, 1, or 2.");
   args.AddOption(&adapt_eval, "-ae", "--adaptivity-evaluator",
                  "0 - Advection based (DEFAULT), 1 - GSLIB.");
   args.AddOption(&devopt, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&surface_fit_adapt, "-sfa", "--adaptive-surface-fit",
                  "Enable or disable adaptive surface fitting.");
   args.AddOption(&surface_fit_threshold, "-sft", "--surf-fit-threshold",
                  "Set threshold for surface fitting. TMOP solver will"
                  "terminate when max surface fitting error is below this limit");
   args.AddOption(&adapt_marking, "-marking", "--adaptive-marking", "-no-amarking",
                  "--no-adaptive-marking",
                  "Enable or disable adaptive marking surface fitting.");
   args.AddOption(&surf_bg_mesh, "-sbgmesh", "--surf-bg-mesh",
                  "-no-sbgmesh","--no-surf-bg-mesh", "Use background mesh for surface fitting.");
   args.AddOption(&comp_dist, "-dist", "--comp-dist",
                  "-no-dist","--no-comp-dist", "Compute distance from 0 level set or not.");
   args.AddOption(&surf_ls_type, "-slstype", "--surf-ls-type",
                  "1 - Circle (DEFAULT), 2 - Squircle, 3 - Butterfly.");
   args.AddOption(&marking_type, "-smtype", "--surf-marking-type",
                  "0 - Interface (DEFAULT), >0 - Boundary attribute.");
   args.AddOption(&mod_bndr_attr, "-mod-bndr-attr", "--modify-boundary-attribute",
                  "-fix-bndr-attr", "--fix-boundary-attribute",
                  "Change boundary attribue based on alignment with Cartesian axes.");
   args.AddOption(&material, "-mat", "--mat",
                  "-no-mat","--no-mat", "Use default material attributes.");
   args.AddOption(&mesh_node_ordering, "-mno", "--mesh_node_ordering",
                  "Ordering of mesh nodes."
                  "0 (default): byNodes, 1: byVDIM");
   args.AddOption(&amr_iters, "-amriter", "--amr-iter",
                  "Number of amr iterations on background mesh");
   args.AddOption(&int_amr_iters, "-iamriter", "--int-amr-iter",
                  "Number of amr iterations around interface on mesh");
   args.AddOption(&deactivation_layers, "-deact", "--deact-layers",
                  "Number of layers of elements around the interface to consider for TMOP solver");
   args.AddOption(&twopass, "-twopass", "--twopass", "-no-twopass",
                  "--no-twopass",
                  "Enable 2nd pass for smoothing volume elements when some elements"
                  "are deactivated in 1st pass with surface fitting.");
   args.AddOption(&trim_mesh, "-trim", "--trim",
                  "-no-trim","--no-trim", "trim the mesh or not.");
   args.AddOption(&hex_to_tet_split_type, "-htot", "--hex_to_tet_split_type",
                  "Split Hex Mesh Into Tets");
   args.AddOption(&mesh_save, "-ms", "--meshsave", "-no-ms",
                  "--no-meshsave",
                  "Save original and optimized mesh.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   Device device(devopt);
   if (myid == 0) { device.Print();}

   // 3. Initialize and refine the starting mesh.
   Mesh *mesh = NULL;
   if (hex_to_tet_split_type > 0)
   {
      int res = std::max(1, rs_levels);
      //SPLIT TYPE == 1 - 12 TETS, 2 = 24 TETS
      mesh = new Mesh(Mesh::MakeHexTo24or12TetMesh(8*res,8*res,8*res,
                                                   1.0, 1.0, 1.0,
                                                   hex_to_tet_split_type)); //24tet
   }
   else if (hex_to_tet_split_type == 0)
   {
      mesh = new Mesh(mesh_file, 1, 1, false);
      for (int lev = 0; lev < rs_levels; lev++)
      {
         mesh->UniformRefinement();
      }
   }
   else
   {
      int res = std::max(1, rs_levels);
      //SPLIT TYPE == 1 - 12 TETS, 2 = 24 TETS
      mesh = new Mesh(Mesh::MakeQuadTo4TriMesh(8*res,8*res, 1.0, 1.0));
   }
   //   mesh->EnsureNCMesh();
   const int dim = mesh->Dimension();

   // Define level-set coefficient
   FunctionCoefficient *ls_coeff = NULL;
   if (surf_ls_type == 1) //Circle
   {
      ls_coeff = new FunctionCoefficient(circle_level_set);
   }
   else if (surf_ls_type == 2) // reactor
   {
      ls_coeff = new FunctionCoefficient(reactor);
   }
   else if (surf_ls_type == 3) // squircle_inside_circle_level_set
   {
      ls_coeff = new FunctionCoefficient(squircle_inside_circle_level_set);
   }
   else if (surf_ls_type == 4) // cube inside sphere
   {
      ls_coeff = new FunctionCoefficient(csg_cubecylsph_smooth);
   }
   else if (surf_ls_type == 5) // squircle 2D
   {
      ls_coeff = new FunctionCoefficient(squircle_level_set);
   }
   else if (surf_ls_type == 6) // 3D shape
   {
      ls_coeff = new FunctionCoefficient(csg_cubecylsph);
   }
   else if (surf_ls_type == 7) // cube inside sphere
   {
      ls_coeff = new FunctionCoefficient(kabaria_smooth);
   }
   else
   {
      MFEM_ABORT("Surface fitting level set type not implemented yet.")
   }
   mesh->EnsureNCMesh();

   // Trim the mesh based on material attribute and set boundary attribute for fitting to 3
   StopWatch TimeMeshTrim;
   TimeMeshTrim.Start();
   if (trim_mesh)
   {
      //      Mesh *mesh_trim = TrimMesh(*mesh, *ls_coeff, mesh_poly_deg, 2, 1, 1);
      Array<int> attr_to_keep(1);
      attr_to_keep = 1;
      Mesh *mesh_trim = TrimMeshAsSubMesh(*mesh, *ls_coeff, mesh_poly_deg,
                                          attr_to_keep, hex_to_tet_split_type);
      //      Mesh *mesh_trim = TrimMesh(*mesh, *ls_coeff, mesh_poly_deg, 1, 1, 3);
      if (myid == 0)
      {
         std::cout << "Original mesh size: " << mesh->GetNE() << " " << std::endl <<
                   "Trimmer mesh size: " << mesh_trim->GetNE() << " " <<
                   std::endl;
      }
      delete mesh;
      mesh = new Mesh(*mesh_trim);
      delete mesh_trim;
      if (mesh_save)
      {
         ostringstream mesh_name;
         mesh_name << "TrimmedInitialMesh.mesh";
         ofstream mesh_ofs(mesh_name.str().c_str());
         mesh_ofs.precision(8);
         mesh->Print(mesh_ofs);
      }
      MFEM_VERIFY(marking_type > 0, "Trimmed mesh + marking type = 0 (i.e. interface)"
                  " does not make sense\n");
   }
   TimeMeshTrim.Stop();

   L2_FECollection gl_el_coll(0, dim);
   FiniteElementSpace gl_el_fes(mesh, &gl_el_coll);
   GridFunction gl_el_num(&gl_el_fes);
   for (int e = 0; e < mesh->GetNE(); e++)
   {
      gl_el_num(e) = e;
   }
   int *partitioning = mesh->GeneratePartitioning(Mpi::WorldSize());
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   ParGridFunction pgl_el_num(pmesh, &gl_el_num, partitioning);

   delete mesh;

   for (int lev = 0; lev < rp_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   HRefUpdater HRUpdater = HRefUpdater();

   // 4. Define a finite element space on the mesh. Here we use vector finite
   //    elements which are tensor products of quadratic finite elements. The
   //    number of components in the vector finite element space is specified by
   //    the last parameter of the FiniteElementSpace constructor.
   FiniteElementCollection *fec;
   if (mesh_poly_deg <= 0)
   {
      fec = new QuadraticPosFECollection;
      mesh_poly_deg = 2;
   }
   else { fec = new H1_FECollection(mesh_poly_deg, dim); }
   ParFiniteElementSpace *pfespace = new ParFiniteElementSpace(pmesh, fec, dim,
                                                               mesh_node_ordering);

   // 5. Make the mesh curved based on the above finite element space. This
   //    means that we define the mesh elements through a fespace-based
   //    transformation of the reference element.
   pmesh->SetNodalFESpace(pfespace);

   // 6. Set up an empty right-hand side vector b, which is equivalent to b=0.
   Vector b(0);

   // 7. Get the mesh nodes (vertices and other degrees of freedom in the finite
   //    element space) as a finite element grid function in fespace. Note that
   //    changing x automatically changes the shapes of the mesh elements.
   ParGridFunction x(pfespace);
   pmesh->SetNodalGridFunction(&x);
   x.SetTrueVector();
   HRUpdater.AddFESpaceForUpdate(pfespace);
   HRUpdater.AddGridFunctionForUpdate(&x);

   // Modify boundary attribute for surface node movement
   // Sets attributes of a boundary element to 1/2/3 if it is parallel to x/y/z.
   if (mod_bndr_attr)
   {
      ModifyBoundaryAttributesForNodeMovement(pmesh, x);
      pmesh->SetAttributes();
   }
   pmesh->ExchangeFaceNbrData();

   // Do AMR Refinements around the mesh
   // Surface fitting.
   L2_FECollection mat_coll(0, dim);
   H1_FECollection surf_fit_fec(mesh_poly_deg, dim);
   ParFiniteElementSpace surf_fit_fes(pmesh, &surf_fit_fec);
   ParFiniteElementSpace mat_fes(pmesh, &mat_coll);
   ParGridFunction mat(&mat_fes);
   ParGridFunction surf_fit_gf0(&surf_fit_fes);
   HRUpdater.AddFESpaceForUpdate(&surf_fit_fes);
   HRUpdater.AddFESpaceForUpdate(&mat_fes);
   HRUpdater.AddGridFunctionForUpdate(&mat);
   HRUpdater.AddGridFunctionForUpdate(&surf_fit_gf0);
   Array<bool> surf_fit_marker(0);

   Array<int> vdofs;
   Array<int> orig_pmesh_attributes(0);
   ParMesh *psubmesh = NULL;
   ParFiniteElementSpace *psub_pfespace = NULL;
   ParGridFunction *psub_x = NULL;
   StopWatch TimeMeshDist;
   if (surface_fit_const > 0.0)
   {
      surf_fit_gf0.ProjectCoefficient(*ls_coeff);
      TimeMeshDist.Start();
      if (comp_dist && !surf_bg_mesh)
      {
         ComputeScalarDistanceFromLevelSet(*pmesh, *ls_coeff, surf_fit_gf0);
      }
      TimeMeshDist.Stop();

      // Set material gridfunction
      for (int i = 0; i < pmesh->GetNE(); i++)
      {
         if (material || trim_mesh)
         {
            mat(i) = pmesh->GetAttribute(i)-1;
         }
         else
         {
            mat(i) = material_id(i, surf_fit_gf0);
            pmesh->SetAttribute(i, mat(i) + 1);
         }
      }
      // TODO: Make materials consistent using gslib communicator
      if (mesh_save)
      {
         ostringstream mesh_name;
         mesh_name << "perturbed.mesh";
         ofstream mesh_ofs(mesh_name.str().c_str());
         mesh_ofs.precision(8);
         pmesh->PrintAsSerial(mesh_ofs);
      }
      if (hex_to_tet_split_type > 0)
      {
         MakeMaterialConsistentForElementGroups(mat, pgl_el_num,
                                                12*hex_to_tet_split_type);
      }
      else if (hex_to_tet_split_type < 0)
      {
         MakeMaterialConsistentForElementGroups(mat, pgl_el_num,
                                                4);
      }
      for (int i = 0; i < pmesh->GetNE(); i++)
      {
         pmesh->SetAttribute(i, mat(i) + 1);
      }
      pmesh->SetAttributes();

      mat.ExchangeFaceNbrData();

      // Adapt attributes for marking such that if all but 1 face of an element
      // are marked, the element attribute is switched.
      if (adapt_marking && !material && !trim_mesh && hex_to_tet_split_type == 0)
      {
         ModifyAttributeForMarkingDOFS(pmesh, mat, 0);
         ModifyAttributeForMarkingDOFS(pmesh, mat, 1);
      }

      // Set DOFs for fitting
      // Strategy 1: Automatically choose face between elements of different attribute.
      if (marking_type == 0)
      {
         //need to check material consistency for AMR meshes
         int matcheck = CheckMaterialConsistency(pmesh, mat);
         MFEM_VERIFY(matcheck, "Not all children at the interface have same material.");
      }

      // Refine elements near fitting boundary
      if (int_amr_iters)
      {
         Array<int> refinements;
         for (int i = 0; i < 3; i++)
         {
            if (marking_type > 0)
            {
               GetBoundaryElements(pmesh, mat, refinements, marking_type);
            }
            else
            {
               GetMaterialInterfaceElements(pmesh, mat, refinements);
            }
            //              GetMaterialInterfaceElements(pmesh, mat, refinements);
            refinements.Sort();
            refinements.Unique();
            {
               ExtendRefinementListToNeighbors(*pmesh, refinements);
            }
            pmesh->GeneralRefinement(refinements, -1);
            HRUpdater.Update();
         }
         if (!pmesh->Conforming())
         {
            pmesh->Rebalance();
            HRUpdater.Update();
         }
      }
   }

   int num_active_glob = 0,
       neglob = pmesh->GetGlobalNE();
   if (myid == 0)
   {
      std::cout << "Number of elements in the mesh: " << neglob <<  endl;
   }

   int max_el_attr = pmesh->attributes.Max();
   int max_bdr_el_attr = pmesh->bdr_attributes.Max();
   Array<int> deactivate_list(0);

   StopWatch TimeSubMeshTrim;
   TimeSubMeshTrim.Start();
   if (deactivation_layers > 0)
   {
      Array<int> active_list;
      // Deactivate  elements away from interface/boundary to fit
      if (marking_type > 0)
      {
         GetBoundaryElements(pmesh, mat, active_list, marking_type);
      }
      else
      {
         GetMaterialInterfaceElements(pmesh, mat, active_list);
      }
      active_list.Sort();
      active_list.Unique();

      for (int i = 0; i < deactivation_layers; i++)
      {
         ExtendRefinementListToNeighbors(*pmesh, active_list);
      }
      active_list.Sort();
      active_list.Unique();
      int num_active_loc = active_list.Size();
      num_active_glob = num_active_loc;
      MPI_Allreduce(&num_active_loc, &num_active_glob, 1, MPI_INT, MPI_SUM,
                    pmesh->GetComm());
      deactivate_list.SetSize(pmesh->GetNE());
      if (myid == 0)
      {
         std::cout << "Number of elements in the submesh: " << num_active_glob <<  endl;
      }
      if (neglob == num_active_glob)
      {
         deactivate_list = 0;
      }
      else
      {
         deactivate_list = 1;
         for (int i = 0; i < active_list.Size(); i++)
         {
            deactivate_list[active_list[i]] = 0;
         }
         orig_pmesh_attributes.SetSize(pmesh->GetNE());
         for (int i = 0; i < pmesh->GetNE(); i++)
         {
            orig_pmesh_attributes[i] = pmesh->GetAttribute(i);
            if (deactivate_list[i])
            {
               pmesh->SetAttribute(i, max_el_attr+1);
            }
         }
      }

      Array<int> domain_attributes(max_el_attr);
      if (marking_type == 0)   //interface fitting
      {
         // In this case, we will keep attributes 1 and 2 around the interface
         domain_attributes[0] = 1;
         domain_attributes[1] = 2;
      }
      else
      {
         //for boundary fitting, there should be only 1 element attribute
         // in the mesh.
         domain_attributes[0] = 1;
      }

      psubmesh = new ParSubMesh(ParSubMesh::CreateFromDomain(*pmesh,
                                                             domain_attributes));
      psub_x = dynamic_cast<ParGridFunction *>(psubmesh->GetNodes());
      psub_pfespace = psub_x->ParFESpace();
      psubmesh->SetAttributes();
      num_active_glob = psubmesh->GetGlobalNE();
      if (myid == 0)
      {
         std::cout << "Number of elements in the submesh 2: " << num_active_glob <<
                   endl;
      }

      //Fix boundary attribues of submesh
      int n_bdr_el_attr_psub = psubmesh->bdr_attributes.Size();
      int max_bdr_el_attr_psub = psubmesh->bdr_attributes.Max();
      int set_new_bdr_attr = max_bdr_el_attr+1;
      for (int i = 0; i < psubmesh->GetNBE(); i++)
      {
         if (psubmesh->GetBdrAttribute(i) > max_bdr_el_attr)
         {
            psubmesh->SetBdrAttribute(i, set_new_bdr_attr);
         }
      }
      psubmesh->SetAttributes();
   }
   else
   {
      psubmesh = pmesh;
      psub_pfespace = pfespace;
      psub_x = &x;
   }
   psub_x->SetTrueVector();
   TimeSubMeshTrim.Stop();

   ParFiniteElementSpace psub_surf_fit_fes(psubmesh, &surf_fit_fec);
   ParFiniteElementSpace psub_mat_fes(psubmesh, &mat_coll);
   ParGridFunction psub_mat(&psub_mat_fes);
   ParGridFunction psub_surf_fit_gf0(&psub_surf_fit_fes);
   if (deactivation_layers > 0)
   {
      auto tmap1 = ParSubMesh::CreateTransferMap(mat, psub_mat);
      tmap1.Transfer(mat, psub_mat);

      auto tmap2 = ParSubMesh::CreateTransferMap(surf_fit_gf0, psub_surf_fit_gf0);
      tmap2.Transfer(surf_fit_gf0, psub_surf_fit_gf0);
   }
   else
   {
      psub_mat = mat;
      psub_surf_fit_gf0 = surf_fit_gf0;
   }

   psub_mat.ExchangeFaceNbrData();
   if (mesh_save)
   {
      ostringstream mesh_name;
      mesh_name << "perturbed_submesh.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      //      psubmesh->PrintAsOne(mesh_ofs);
      psubmesh->PrintAsSerial(mesh_ofs);
   }

   // Setup background mesh for surface fitting
   ParMesh *pmesh_surf_fit_bg = NULL;
   if (surf_bg_mesh)
   {
      Mesh *mesh_surf_fit_bg = NULL;
      if (dim == 2)
      {
         mesh_surf_fit_bg = new Mesh(Mesh::MakeCartesian2D(8, 8, Element::QUADRILATERAL,
                                                           true));
      }
      else if (dim == 3)
      {
         mesh_surf_fit_bg = new Mesh(Mesh::MakeCartesian3D(8, 8, 8, Element::HEXAHEDRON,
                                                           true));
      }
      mesh_surf_fit_bg->EnsureNCMesh();
      pmesh_surf_fit_bg = new ParMesh(MPI_COMM_WORLD, *mesh_surf_fit_bg);
      delete mesh_surf_fit_bg;
   }

   // 10. Save the starting (prior to the optimization) mesh to a file. This
   //     output can be viewed later using GLVis: "glvis -m perturbed -np
   //     num_mpi_tasks".
   if (mesh_save)
   {
      ostringstream mesh_name;
      mesh_name << "perturbed.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      //      pmesh->PrintAsOne(mesh_ofs);
      pmesh->PrintAsSerial(mesh_ofs);
   }


   // 11. Store the starting (prior to the optimization) positions.
   ParGridFunction x0(pfespace);
   x0 = x;

   // 12. Form the integrator that uses the chosen metric and target.
   double tauval = -0.1;
   TMOP_QualityMetric *metric = NULL;
   switch (metric_id)
   {
      // T-metrics
      case 1: metric = new TMOP_Metric_001; break;
      case 2: metric = new TMOP_Metric_002; break;
      case 55: metric = new TMOP_Metric_055; break;
      case 58: metric = new TMOP_Metric_058; break;
      case 80: metric = new TMOP_Metric_080(0.5); break;
      case 300: metric = new TMOP_Metric_300; break;
      case 301: metric = new TMOP_Metric_301; break;
      case 302: metric = new TMOP_Metric_302; break;
      case 303: metric = new TMOP_Metric_303; break;
      case 315: metric = new TMOP_Metric_315; break;
      case 316: metric = new TMOP_Metric_316; break;
      case 321: metric = new TMOP_Metric_321; break;
      case 328: metric = new TMOP_Metric_328(0.5); break;
      case 332: metric = new TMOP_Metric_332(0.5); break;
      case 333: metric = new TMOP_Metric_333(0.5); break;
      case 334: metric = new TMOP_Metric_334(0.5); break;
      default:
         if (myid == 0) { cout << "Unknown metric_id: " << metric_id << endl; }
         return 3;
   }

   if (metric_id < 300)
   {
      MFEM_VERIFY(dim == 2, "Incompatible metric for 3D meshes");
   }
   if (metric_id >= 300)
   {
      MFEM_VERIFY(dim == 3, "Incompatible metric for 2D meshes");
   }

   TargetConstructor::TargetType target_t;
   TargetConstructor *target_c = NULL;
   switch (target_id)
   {
      case 1: target_t = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE; break;
      case 2: target_t = TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE; break;
      case 3: target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE; break;
      case 4: target_t = TargetConstructor::GIVEN_SHAPE_AND_SIZE; break;
      default:
         if (myid == 0) { cout << "Unknown target_id: " << target_id << endl; }
         return 3;
   }

   if (target_c == NULL)
   {
      target_c = new TargetConstructor(target_t, MPI_COMM_WORLD);
   }
   target_c->SetNodes(x0);
   TMOP_Integrator *tmop_integ = new TMOP_Integrator(metric, target_c);

   // Setup the quadrature rules for the TMOP integrator.
   IntegrationRules *irules = NULL;
   switch (quad_type)
   {
      case 1: irules = &IntRulesLo; break;
      case 2: irules = &IntRules; break;
      case 3: irules = &IntRulesCU; break;
      default:
         if (myid == 0) { cout << "Unknown quad_type: " << quad_type << endl; }
         return 3;
   }
   tmop_integ->SetIntegrationRules(*irules, quad_order);
   if (myid == 0 && dim == 2)
   {
      cout << "Triangle quadrature points: "
           << irules->Get(Geometry::TRIANGLE, quad_order).GetNPoints()
           << "\nQuadrilateral quadrature points: "
           << irules->Get(Geometry::SQUARE, quad_order).GetNPoints() << endl;
   }
   if (myid == 0 && dim == 3)
   {
      cout << "Tetrahedron quadrature points: "
           << irules->Get(Geometry::TETRAHEDRON, quad_order).GetNPoints()
           << "\nHexahedron quadrature points: "
           << irules->Get(Geometry::CUBE, quad_order).GetNPoints()
           << "\nPrism quadrature points: "
           << irules->Get(Geometry::PRISM, quad_order).GetNPoints() << endl;
   }

   // Background mesh FECollection, FESpace, and GridFunction
   H1_FECollection *surf_fit_bg_fec = NULL;
   ParFiniteElementSpace *surf_fit_bg_fes = NULL;
   ParGridFunction *surf_fit_bg_gf0 = NULL;
   ParFiniteElementSpace *surf_fit_bg_grad_fes = NULL;
   ParGridFunction *surf_fit_bg_grad = NULL;
   ParFiniteElementSpace *surf_fit_bg_hess_fes = NULL;
   ParGridFunction *surf_fit_bg_hess = NULL;

   // If a background mesh is used, we interpolate the Gradient and Hessian
   // from that mesh to the current mesh being optimized.
   ParFiniteElementSpace *surf_fit_grad_fes = NULL;
   ParGridFunction *surf_fit_grad = NULL;
   ParFiniteElementSpace *surf_fit_hess_fes = NULL;
   ParGridFunction *surf_fit_hess = NULL;

   StopWatch TimeBGMeshDist, TimeBGMeshAMR, TimeBGMeshDer;
   if (surf_bg_mesh)
   {
      pmesh_surf_fit_bg->SetCurvature(mesh_poly_deg, false, -1, 0);

      Vector p_min(dim), p_max(dim);
      psubmesh->GetBoundingBox(p_min, p_max);
      GridFunction &x_bg = *pmesh_surf_fit_bg->GetNodes();
      const int num_nodes = x_bg.Size() / dim;
      for (int i = 0; i < num_nodes; i++)
      {
         for (int d = 0; d < dim; d++)
         {
            double length_d = p_max(d) - p_min(d),
                   extra_d = 0.2 * length_d;
            x_bg(i + d*num_nodes) = p_min(d) - extra_d +
                                    x_bg(i + d*num_nodes) * (length_d + 2*extra_d);
         }
      }
      surf_fit_bg_fec = new H1_FECollection(mesh_poly_deg, dim);
      surf_fit_bg_fes = new ParFiniteElementSpace(pmesh_surf_fit_bg, surf_fit_bg_fec);
      surf_fit_bg_gf0 = new ParGridFunction(surf_fit_bg_fes);

      TimeBGMeshAMR.Start();
      if (myid == 0) { std::cout << "Do AMR on Background mesh\n"; }
      OptimizeMeshWithAMRAroundZeroLevelSet(*pmesh_surf_fit_bg, *ls_coeff, amr_iters,
                                            *surf_fit_bg_gf0);
      if (myid == 0) { std::cout << "Done AMR on Background mesh\n"; }
      {
         int ne_glob_bg = pmesh_surf_fit_bg->GetGlobalNE();
         if (myid == 0)
         {
            std::cout << "Number of elements in background mesh: " <<
                      ne_glob_bg << endl;
         }
      }

      TimeBGMeshAMR.Stop();

      pmesh_surf_fit_bg->Rebalance();
      surf_fit_bg_fes->Update();
      surf_fit_bg_gf0->Update();
      TimeBGMeshDist.Start();
      if (comp_dist)
      {
         if (myid == 0) { std::cout << "Compute Distance on Background mesh\n"; }
         ComputeScalarDistanceFromLevelSet(*pmesh_surf_fit_bg, *ls_coeff,
                                           *surf_fit_bg_gf0);
         if (myid == 0) { std::cout << "Done ComputeDistance on Background mesh\n"; }
      }
      else
      {
         surf_fit_bg_gf0->ProjectCoefficient(*ls_coeff);
      }
      TimeBGMeshDist.Stop();


      surf_fit_bg_grad_fes = new ParFiniteElementSpace(pmesh_surf_fit_bg,
                                                       surf_fit_bg_fec,
                                                       pmesh_surf_fit_bg->Dimension());
      surf_fit_bg_grad = new ParGridFunction(surf_fit_bg_grad_fes);


      int n_hessian_bg = pow(pmesh_surf_fit_bg->Dimension(), 2);
      surf_fit_bg_hess_fes = new ParFiniteElementSpace(pmesh_surf_fit_bg,
                                                       surf_fit_bg_fec,
                                                       n_hessian_bg);
      surf_fit_bg_hess = new ParGridFunction(surf_fit_bg_hess_fes);

      TimeBGMeshDer.Start();
      //Setup gradient of the background mesh
      for (int d = 0; d < pmesh_surf_fit_bg->Dimension(); d++)
      {
         ParGridFunction surf_fit_bg_grad_comp(surf_fit_bg_fes,
                                               surf_fit_bg_grad->GetData()+d*surf_fit_bg_gf0->Size());
         surf_fit_bg_gf0->GetDerivative(1, d, surf_fit_bg_grad_comp);
      }

      //Setup Hessian on background mesh
      int id = 0;
      for (int d = 0; d < pmesh_surf_fit_bg->Dimension(); d++)
      {
         for (int idir = 0; idir < pmesh_surf_fit_bg->Dimension(); idir++)
         {
            ParGridFunction surf_fit_bg_grad_comp(surf_fit_bg_fes,
                                                  surf_fit_bg_grad->GetData()+d*surf_fit_bg_gf0->Size());
            ParGridFunction surf_fit_bg_hess_comp(surf_fit_bg_fes,
                                                  surf_fit_bg_hess->GetData()+id*surf_fit_bg_gf0->Size());
            surf_fit_bg_grad_comp.GetDerivative(1, idir, surf_fit_bg_hess_comp);
            id++;
         }
      }
      TimeBGMeshDer.Stop();
   }

   Array<int> el_face_marked_count(pmesh->GetNE());
   el_face_marked_count = 0;

   // Surface fitting.
   AdaptivityEvaluator *adapt_surface = NULL;
   AdaptivityEvaluator *adapt_grad_surface = NULL;
   AdaptivityEvaluator *adapt_hess_surface = NULL;
   Array<bool> psub_surf_fit_marker(psub_surf_fit_gf0.Size());
   ConstantCoefficient surf_fit_coeff(surface_fit_const);
   ParGridFunction surf_fit_mat_gf(&psub_surf_fit_fes);
   if (surface_fit_const > 0.0)
   {
      psub_surf_fit_gf0.ProjectCoefficient(*ls_coeff);
      if (surf_bg_mesh)
      {
         surf_fit_grad_fes = new ParFiniteElementSpace(psubmesh, &surf_fit_fec,
                                                       psubmesh->Dimension());
         surf_fit_grad = new ParGridFunction(surf_fit_grad_fes);

         surf_fit_hess_fes = new ParFiniteElementSpace(psubmesh, &surf_fit_fec,
                                                       psubmesh->Dimension()*psubmesh->Dimension());
         surf_fit_hess = new ParGridFunction(surf_fit_hess_fes);
      }

      GridFunctionCoefficient coeff_mat(&psub_mat);
      surf_fit_mat_gf.ProjectDiscCoefficient(coeff_mat, GridFunction::ARITHMETIC);
      surf_fit_mat_gf.SetTrueVector();
      surf_fit_mat_gf.SetFromTrueVector();

      // Set DOFs for fitting
      // Strategy 1: Automatically choose face between elements of different attribute.
      if (marking_type == 0)
      {
         Array<int> intfaces;
         GetMaterialInterfaceFaces(psubmesh, psub_mat, intfaces);

         psub_surf_fit_marker.SetSize(psub_surf_fit_gf0.Size());
         for (int j = 0; j < psub_surf_fit_marker.Size(); j++)
         {
            psub_surf_fit_marker[j] = false;
         }
         surf_fit_mat_gf = 0.0;

         Array<int> dof_list;
         Array<int> dofs;
         for (int i = 0; i < intfaces.Size(); i++)
         {
            psub_surf_fit_gf0.ParFESpace()->GetFaceDofs(intfaces[i], dofs);
            dof_list.Append(dofs);
         }
         for (int i = 0; i < dof_list.Size(); i++)
         {
            psub_surf_fit_marker[dof_list[i]] = true;
            surf_fit_mat_gf(dof_list[i]) = 1.0;
         }

         // do the same for actual mesh for the second pass
         if (deactivation_layers > 0 && twopass)
         {
            dof_list.SetSize(0);
            Array<int> intfaces2;
            GetMaterialInterfaceFaces(pmesh, mat, intfaces2);

            surf_fit_marker.SetSize(x.Size());
            for (int j = 0; j < surf_fit_marker.Size(); j++)
            {
               surf_fit_marker[j] = false;
            }

            for (int i = 0; i < intfaces2.Size(); i++)
            {
               x.ParFESpace()->GetFaceVDofs(intfaces2[i], dofs);
               dof_list.Append(dofs);
            }
            for (int i = 0; i < dof_list.Size(); i++)
            {
               surf_fit_marker[dof_list[i]] = true;
            }
         }
      }
      // Strategy 2: Mark all boundaries with attribute marking_type
      else if (marking_type > 0)
      {
         psub_surf_fit_marker.SetSize(psub_surf_fit_gf0.Size());
         for (int j = 0; j < psub_surf_fit_marker.Size(); j++)
         {
            psub_surf_fit_marker[j] = false;
         }
         surf_fit_mat_gf = 0.0;
         for (int i = 0; i < psubmesh->GetNBE(); i++)
         {
            const int attr = psubmesh->GetBdrElement(i)->GetAttribute();
            int elno, info;
            psubmesh->GetBdrElementAdjacentElement(i, elno, info);
            el_face_marked_count[elno] += 1;
            if (attr == marking_type)
            {
               psub_surf_fit_fes.GetBdrElementVDofs(i, vdofs);
               for (int j = 0; j < vdofs.Size(); j++)
               {
                  psub_surf_fit_marker[vdofs[j]] = true;
                  surf_fit_mat_gf(vdofs[j]) = 1.0;
               }
            }
         }
      }

      ParGridFunction flagm(mat);
      flagm = 0.0;
      int n_c_count = 0;
      for (int i = 0; i < el_face_marked_count.Size(); i++)
      {
         if (el_face_marked_count[i] > 1)
         {
            n_c_count++;
            flagm(i) = 1;
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, &n_c_count, 1, MPI_INT, MPI_SUM, pmesh->GetComm());
      if (myid == 0)
      {
         std::cout << "TOTAL ELEMENTS WITH MORE THAN 1 FACE: " << n_c_count << " k101\n";
      }

      if (visualization && n_c_count > 0)
      {
         socketstream vis1, vis2, vis3, vis4, vis5;
         common::VisualizeField(vis1, "localhost", 19916, flagm, "marked els",
                                300, 600, 300, 300);
      }

      // Set AdaptivityEvaluators for transferring information from initial
      // mesh to current mesh as it moves during adaptivity.
      if (adapt_eval == 0)
      {
         adapt_surface = new AdvectorCG;
         MFEM_ASSERT(!surf_bg_mesh, "Background meshes require GSLIB.");
      }
      else if (adapt_eval == 1)
      {
#ifdef MFEM_USE_GSLIB
         adapt_surface = new InterpolatorFP;
         if (surf_bg_mesh)
         {
            adapt_grad_surface = new InterpolatorFP;
            adapt_hess_surface = new InterpolatorFP;
         }
#else
         MFEM_ABORT("MFEM is not built with GSLIB support!");
#endif
      }
      else { MFEM_ABORT("Bad interpolation option."); }

      if (!surf_bg_mesh)
      {
         tmop_integ->EnableSurfaceFitting(psub_surf_fit_gf0, psub_surf_fit_marker,
                                          surf_fit_coeff,
                                          *adapt_surface);
      }
      else
      {
         tmop_integ->EnableSurfaceFittingFromSource(*surf_fit_bg_gf0,
                                                    psub_surf_fit_gf0,
                                                    psub_surf_fit_marker,
                                                    surf_fit_coeff,
                                                    *adapt_surface,
                                                    *surf_fit_bg_grad,
                                                    *surf_fit_grad,
                                                    *adapt_grad_surface,
                                                    *surf_fit_bg_hess,
                                                    *surf_fit_hess,
                                                    *adapt_hess_surface);
      }

      if (visualization)
      {
         socketstream vis1, vis2, vis3, vis4, vis5;
         common::VisualizeField(vis1, "localhost", 19916, psub_surf_fit_gf0,
                                "Level Set 0",
                                300, 600, 300, 300);
         common::VisualizeField(vis2, "localhost", 19916, psub_mat, "Materials",
                                600, 600, 300, 300);
         common::VisualizeField(vis3, "localhost", 19916, surf_fit_mat_gf,
                                "Dofs to Move",
                                900, 600, 300, 300);
         if (surf_bg_mesh)
         {
            common::VisualizeField(vis4, "localhost", 19916, *surf_fit_bg_gf0,
                                   "Level Set 0 Source",
                                   300, 600, 300, 300);

            common::VisualizeField(vis5, "localhost", 19916, *surf_fit_bg_grad,
                                   "Level Set 0 Source grad",
                                   300, 600, 300, 300);
         }
      }
   }


   //   MFEM_ABORT("k10aborting1");

   // 13. Setup the final NonlinearForm (which defines the integral of interest,
   //     its first and second derivatives). Here we can use a combination of
   //     metrics, i.e., optimize the sum of two integrals, where both are
   //     scaled by used-defined space-dependent weights.  Note that there are
   //     no command-line options for the weights and the type of the second
   //     metric; one should update those in the code.
   ParNonlinearForm a(psub_pfespace);
   ConstantCoefficient *metric_coeff1 = NULL;
   a.AddDomainIntegrator(tmop_integ);

   // Compute the minimum det(J) of the starting mesh (TODO: WHOLE MESH or PARTIAL MESH).
   tauval = infinity();
   const int NE = pmesh->GetNE();
   for (int i = 0; i < NE; i++)
   {
      const IntegrationRule &ir =
         irules->Get(pfespace->GetFE(i)->GetGeomType(), quad_order);
      ElementTransformation *transf = pmesh->GetElementTransformation(i);
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         transf->SetIntPoint(&ir.IntPoint(j));
         tauval = min(tauval, transf->Jacobian().Det());
      }
   }
   double minJ0;
   MPI_Allreduce(&tauval, &minJ0, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   tauval = minJ0;
   if (myid == 0)
   { cout << "Minimum det(J) of the original mesh is " << tauval << endl; }

   if (tauval < 0.0 && metric_id != 22 && metric_id != 211 && metric_id != 252
       && metric_id != 311 && metric_id != 313 && metric_id != 352)
   {
      MFEM_ABORT("The input mesh is inverted! Try an untangling metric.");
   }
   if (tauval < 0.0)
   {
      MFEM_VERIFY(target_t == TargetConstructor::IDEAL_SHAPE_UNIT_SIZE,
                  "Untangling is supported only for ideal targets.");

      const DenseMatrix &Wideal =
         Geometries.GetGeomToPerfGeomJac(pfespace->GetFE(0)->GetGeomType());
      tauval /= Wideal.Det();
   }

   const double init_energy = a.GetParGridFunctionEnergy(*psub_x);
   double init_metric_energy;
   if (surface_fit_const > 0.0)
   {
      surf_fit_coeff.constant   = 0.0;
      init_metric_energy = a.GetParGridFunctionEnergy(x);
      surf_fit_coeff.constant  = surface_fit_const;
   }

   // Visualize the starting mesh and metric values.
   // Note that for combinations of metrics, this only shows the first metric.
   if (visualization)
   {
      char title[] = "Initial metric values";
      vis_tmop_metric_p(mesh_poly_deg, *metric, *target_c, *psubmesh, title, 0);
   }


   Array<int> ess_deactivate_dofs;
   if (deactivate_list.Size() > 0)
   {
      for (int i = 0; i < pmesh->GetNE(); i++)
      {
         if (deactivate_list[i])
         {
            pfespace->GetElementVDofs(i, vdofs);
            ess_deactivate_dofs.Append(vdofs);
         }
      }
   }

   // 14. Fix all boundary nodes, or fix only a given component depending on the
   //     boundary attributes of the given mesh.  Attributes 1/2/3 correspond to
   //     fixed x/y/z components of the node.  Attribute dim+1 corresponds to
   //     an entirely fixed node.
   Array<int> ess_dofs_bdr;
   if (move_bnd == false)
   {
      Array<int> ess_bdr(psubmesh->bdr_attributes.Max());
      ess_bdr = 1;
      if (marking_type > 0)
      {
         ess_bdr[marking_type-1] = 0;
      }
      a.SetEssentialBC(ess_bdr);
   }
   else
   {
      int n = 0;
      for (int i = 0; i < psubmesh->GetNBE(); i++)
      {
         const int nd = psub_pfespace->GetBE(i)->GetDof();
         const int attr = psubmesh->GetBdrElement(i)->GetAttribute();
         //         MFEM_VERIFY(!(dim == 2 && attr == 3),
         //                     "Boundary attribute 3 must be used only for 3D meshes. "
         //                     "Adjust the attributes (1/2/3/4 for fixed x/y/z/all "
         //                     "components, rest for free nodes), or use -fix-bnd.");
         if (attr == 1 || attr == 2 || (attr == 3 && dim == 3)) { n += nd; }
         if (attr > dim) { n += nd * dim; }
      }
      Array<int> ess_vdofs(n);
      n = 0;
      for (int i = 0; i < psubmesh->GetNBE(); i++)
      {
         const int nd = psub_pfespace->GetBE(i)->GetDof();
         const int attr = psubmesh->GetBdrElement(i)->GetAttribute();
         psub_pfespace->GetBdrElementVDofs(i, vdofs);
         if (attr == 1) // Fix x components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j]; }
         }
         else if (attr == 2) // Fix y components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j+nd]; }
         }
         else if (attr == 3 && dim == 3) // Fix z components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j+2*nd]; }
         }
         else if (attr > dim) // Fix all components.
         {
            for (int j = 0; j < vdofs.Size(); j++)
            { ess_vdofs[n++] = vdofs[j]; }
         }
      }
      a.SetEssentialVDofs(ess_vdofs);
   }

   // 15. As we use the Newton method to solve the resulting nonlinear system,
   //     here we setup the linear solver for the system's Jacobian.
   Solver *S = NULL, *S_prec = NULL;
   const double linsol_rtol = 1e-12;
   if (lin_solver == 0)
   {
      S = new DSmoother(1, 1.0, max_lin_iter);
   }
   else if (lin_solver == 1)
   {
      CGSolver *cg = new CGSolver(MPI_COMM_WORLD);
      cg->SetMaxIter(max_lin_iter);
      cg->SetRelTol(linsol_rtol);
      cg->SetAbsTol(0.0);
      cg->SetPrintLevel(verbosity_level >= 2 ? 3 : -1);
      S = cg;
   }
   else
   {
      MINRESSolver *minres = new MINRESSolver(MPI_COMM_WORLD);
      minres->SetMaxIter(max_lin_iter);
      minres->SetRelTol(linsol_rtol);
      minres->SetAbsTol(0.0);
      if (verbosity_level > 2) { minres->SetPrintLevel(1); }
      else { minres->SetPrintLevel(verbosity_level == 2 ? 3 : -1); }
      if (lin_solver == 3 || lin_solver == 4)
      {
         auto hs = new HypreSmoother;
         hs->SetType((lin_solver == 3) ? HypreSmoother::Jacobi
                     /* */             : HypreSmoother::l1Jacobi, 1);
         hs->SetPositiveDiagonal(true);
         S_prec = hs;
         minres->SetPreconditioner(*S_prec);
      }
      S = minres;
   }

   if (surface_fit_const > 0.0)
   {
      double err_avg, err_max;
      tmop_integ->GetSurfaceFittingErrors(err_avg, err_max);
      if (myid == 0)
      {
         std::cout << "Avg fitting error Pre-Optimization: " << err_avg << std::endl
                   << "Max fitting error Pre-Optimization: " << err_max << std::endl;
      }
   }

   // Perform the nonlinear optimization.
   const IntegrationRule &ir =
      irules->Get(psub_pfespace->GetFE(0)->GetGeomType(), quad_order);
   TMOPNewtonSolver solver(psub_pfespace->GetComm(), ir, solver_type);
   if (surface_fit_adapt > 0.0)
   {
      solver.SetAdaptiveSurfaceFittingScalingFactor(surface_fit_adapt);
   }
   if (surface_fit_threshold > 0)
   {
      solver.SetTerminationWithMaxSurfaceFittingError(surface_fit_threshold);
   }
   solver.SetAdaptiveSurfaceFittingRelativeChangeThreshold(0.01);
   // Provide all integration rules in case of a mixed mesh.
   solver.SetIntegrationRules(*irules, quad_order);
   if (solver_type == 0)
   {
      // Specify linear solver when we use a Newton-based solver.
      solver.SetPreconditioner(*S);
   }
   // For untangling, the solver will update the min det(T) values.
   if (tauval < 0.0) { solver.SetMinDetPtr(&tauval); }
   solver.SetMaxIter(solver_iter);
   solver.SetRelTol(solver_rtol);
   solver.SetAbsTol(0.0);
   solver.SetMinimumDeterminantThreshold(0.001*tauval);
   if (solver_art_type > 0)
   {
      solver.SetAdaptiveLinRtol(solver_art_type, 0.5, 0.9);
   }
   solver.SetPrintLevel(verbosity_level >= 1 ? 1 : -1);
   solver.SetOperator(a);
   StopWatch TimeSolve;
   TimeSolve.Start();
   solver.Mult(b, psub_x->GetTrueVector());
   psub_x->SetFromTrueVector();
   TimeSolve.Stop();


   if (deactivation_layers > 0)
   {
      auto tmap = ParSubMesh::CreateTransferMap(*psub_x, x);
      tmap.Transfer(*psub_x, x);
   }

   // 16. Save the optimized mesh to a file. This output can be viewed later
   //     using GLVis: "glvis -m optimized -np num_mpi_tasks".
   if (mesh_save)
   {
      ostringstream mesh_name;
      mesh_name << "optimized.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      //      pmesh->PrintAsOne(mesh_ofs);
      pmesh->PrintAsSerial(mesh_ofs);
   }

   // Compute the final energy of the functional.
   const double fin_energy = a.GetParGridFunctionEnergy(*psub_x);
   double fin_metric_energy = fin_energy;
   if (surface_fit_const > 0.0)
   {
      surf_fit_coeff.constant  = 0.0;
      fin_metric_energy  = a.GetParGridFunctionEnergy(x);
      surf_fit_coeff.constant  = surface_fit_const;
   }

   if (myid == 0)
   {
      std::cout << std::scientific << std::setprecision(4);
      cout << "Initial strain energy: " << init_energy
           << " = metrics: " << init_metric_energy
           << " + extra terms: " << init_energy - init_metric_energy << endl;
      cout << "  Final strain energy: " << fin_energy
           << " = metrics: " << fin_metric_energy
           << " + extra terms: " << fin_energy - fin_metric_energy << endl;
      cout << "The strain energy decreased by: "
           << (init_energy - fin_energy) * 100.0 / init_energy << " %." << endl;
   }


   if (myid == 0)
   {
      std::cout << "k10-Number of ranks: " << nranks << std::endl;
      std::cout << "k10-Timings for Mesh Optimization:" << std::endl;
      std::cout << "k10-Time to Trim: " << TimeMeshTrim.RealTime() << std::endl;
      std::cout << "k10-Time to generate submesh: " << TimeSubMeshTrim.RealTime() <<
                std::endl;
      std::cout << "k10-Time To do AMR on bg: " << TimeBGMeshAMR.RealTime() <<
                std::endl;
      std::cout << "k10-Time To generate distance on bg: " <<
                TimeBGMeshDist.RealTime() <<
                std::endl;
      std::cout << "k10-Time To get grad on bg: " << TimeBGMeshDer.RealTime() <<
                std::endl;
      std::cout << "k10-Time for TMOP Solve: " << TimeSolve.RealTime() << std::endl;
      std::cout << "k10-Number of elements in the mesh: " << neglob <<  endl;
      std::cout << "k10-Number of elements in the sub-mesh: " << num_active_glob <<
                endl;
   }

   // 18. Visualize the final mesh and metric values.
   if (visualization)
   {
      char title[] = "Final metric values";
      vis_tmop_metric_p(mesh_poly_deg, *metric, *target_c, *pmesh, title, 600);
   }

   if (surface_fit_const > 0.0)
   {
      if (visualization)
      {
         socketstream vis2, vis3;
         common::VisualizeField(vis2, "localhost", 19916, psub_mat,
                                "Materials", 600, 900, 300, 300);
         common::VisualizeField(vis3, "localhost", 19916, surf_fit_mat_gf,
                                "Surface dof", 900, 900, 300, 300);
      }
      double err_avg, err_max;
      tmop_integ->GetSurfaceFittingErrors(err_avg, err_max);
      if (myid == 0)
      {
         std::cout << "Avg fitting error: " << err_avg << std::endl
                   << "Max fitting error: " << err_max << std::endl;

         std::cout << "Last active surface fitting constant: " <<
                   tmop_integ->GetLastActiveSurfaceFittingWeight() <<
                   std::endl;
      }
   }

   ParGridFunction x1(x0);
   if (visualization)
   {
      x1 -= x;
      socketstream sock;
      if (myid == 0)
      {
         sock.open("localhost", 19916);
         sock << "solution\n";
      }
      pmesh->PrintAsOne(sock);
      x1.SaveAsOne(sock);
      if (myid == 0)
      {
         sock << "window_title 'Displacements pre'\n"
              << "window_geometry "
              << 1200 << " " << 0 << " " << 600 << " " << 600 << "\n"
              << "keys jRmclA" << endl;
      }
      x1 = x;
   }

   {
      DataCollection *dc = NULL;
      dc = new VisItDataCollection("Optimized", pmesh);
      dc->RegisterField("solution", &x1);
      surf_fit_gf0.ProjectCoefficient(*ls_coeff);
      dc->RegisterField("level-set-projected", &surf_fit_gf0);

      dc->SetCycle(0);
      dc->SetTime(0.0);
      dc->Save();
   }

   // Don't do second pass
   //   if (deactivate_list.Size() > 0 && twopass) {
   //       ParNonlinearForm a2(pfespace);
   //       tmop_integ->DisableSurfaceFitting();
   //       a2.AddDomainIntegrator(tmop_integ);
   //       {
   //           if (move_bnd == false)
   //           {
   //              Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   //              ess_bdr = 1;
   //              if (marking_type > 0)
   //              {
   //                 ess_bdr[marking_type-1] = 0;
   //              }
   //              a2.SetEssentialBC(ess_bdr);
   //           }
   //           else
   //           {
   //              int n = 0;
   //              for (int i = 0; i < pmesh->GetNBE(); i++)
   //              {
   //                 const int nd = pfespace->GetBE(i)->GetDof();
   //                 const int attr = pmesh->GetBdrElement(i)->GetAttribute();
   //                 MFEM_VERIFY(!(dim == 2 && attr == 3),
   //                             "Boundary attribute 3 must be used only for 3D meshes. "
   //                             "Adjust the attributes (1/2/3/4 for fixed x/y/z/all "
   //                             "components, rest for free nodes), or use -fix-bnd.");
   //                 if (attr == 1 || attr == 2 || attr == 3) { n += nd; }
   //                 if (attr == 4) { n += nd * dim; }
   //              }
   //              Array<int> ess_vdofs(n);
   //              n = 0;
   //              for (int i = 0; i < pmesh->GetNBE(); i++)
   //              {
   //                 const int nd = pfespace->GetBE(i)->GetDof();
   //                 const int attr = pmesh->GetBdrElement(i)->GetAttribute();
   //                 pfespace->GetBdrElementVDofs(i, vdofs);
   //                 if (attr == 1) // Fix x components.
   //                 {
   //                    for (int j = 0; j < nd; j++)
   //                    { ess_vdofs[n++] = vdofs[j]; }
   //                 }
   //                 else if (attr == 2) // Fix y components.
   //                 {
   //                    for (int j = 0; j < nd; j++)
   //                    { ess_vdofs[n++] = vdofs[j+nd]; }
   //                 }
   //                 else if (attr == 3) // Fix z components.
   //                 {
   //                    for (int j = 0; j < nd; j++)
   //                    { ess_vdofs[n++] = vdofs[j+2*nd]; }
   //                 }
   //                 else if (attr == 4) // Fix all components.
   //                 {
   //                    for (int j = 0; j < vdofs.Size(); j++)
   //                    { ess_vdofs[n++] = vdofs[j]; }
   //                 }
   //              }
   ////              int old_size = ess_vdofs.Size();
   //              for (int i = 0; i < surf_fit_marker.Size(); i++) {
   //                  if (surf_fit_marker[i]) {
   //                      ess_vdofs.Append(i);
   //                  }
   //              }
   //              a2.SetEssentialVDofs(ess_vdofs);
   //           }
   //       }
   //       TMOPNewtonSolver solver2(pfespace->GetComm(), ir, solver_type);
   //       solver2.SetIntegrationRules(*irules, quad_order);
   //       if (solver_type == 0)
   //       {
   //          // Specify linear solver when we use a Newton-based solver.
   //          solver2.SetPreconditioner(*S);
   //       }
   //       if (tauval < 0.0) { solver2.SetMinDetPtr(&tauval); }
   //       solver2.SetMaxIter(solver_iter);
   //       solver2.SetRelTol(1e-7);
   //       solver2.SetAbsTol(0.0);
   //       solver2.SetMinimumDeterminantThreshold(0.001*tauval);
   //       if (solver_art_type > 0)
   //       {
   //          solver2.SetAdaptiveLinRtol(solver_art_type, 0.5, 0.9);
   //       }
   //       solver2.SetPrintLevel(verbosity_level >= 1 ? 1 : -1);
   //       solver2.SetOperator(a2);
   //       Vector b2(0);
   //       x.SetTrueVector();
   //       solver2.Mult(b2, x.GetTrueVector());
   //       x.SetFromTrueVector();
   //   }

   // 19. Visualize the mesh displacement.
   //   if (visualization)
   //   {
   //      x1 -= x;
   //      socketstream sock;
   //      if (myid == 0)
   //      {
   //         sock.open("localhost", 19916);
   //         sock << "solution\n";
   //      }
   //      pmesh->PrintAsOne(sock);
   //      x1.SaveAsOne(sock);
   //      if (myid == 0)
   //      {
   //         sock << "window_title 'Displacements'\n"
   //              << "window_geometry "
   //              << 1200 << " " << 0 << " " << 600 << " " << 600 << "\n"
   //              << "keys jRmclA" << endl;
   //      }
   //   }

   // 20. Free the used memory.
   delete S;
   delete S_prec;
   delete metric_coeff1;
   delete adapt_surface;
   delete adapt_grad_surface;
   delete adapt_hess_surface;
   delete surf_fit_hess;
   delete surf_fit_hess_fes;
   delete surf_fit_bg_hess;
   delete surf_fit_bg_hess_fes;
   delete surf_fit_grad;
   delete surf_fit_grad_fes;
   delete surf_fit_bg_grad;
   delete surf_fit_bg_grad_fes;
   delete surf_fit_bg_gf0;
   delete surf_fit_bg_fes;
   delete surf_fit_bg_fec;
   delete target_c;
   delete metric;
   delete pfespace;
   delete fec;
   delete pmesh_surf_fit_bg;
   if (psubmesh == pmesh || psubmesh->GetGlobalNE() == pmesh->GetGlobalNE())
   {
      delete pmesh;
   }
   else
   {
      delete psubmesh;
      delete pmesh;
   }
   delete ls_coeff;

   return 0;
}
