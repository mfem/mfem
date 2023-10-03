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

// make tmop-fitting -j && mpirun -np 4 tmop-fitting -m ../../data/inline-tri.mesh -o 1 -rs 1 -ni 200 -vl 1 -rtol 1e-10 -mod-bndr-attr -sfc 1.0 -sfa 2.0 -sft 1e-10  -tid 1 -mid 2
// make tmop-fitting -j && mpirun -np 4 tmop-fitting -m ../../data/inline-tri.mesh -o 1 -rs 1 -ni 200 -vl 1 -rtol 1e-10 -mod-bndr-attr -sfc 1.0 -sfa 2.0 -sft 1e-10  -tid 4 -mid 80

// incline level-set. low-initial weight - sfa = 1.0
// 201 - make tmop-fitting -j && mpirun -np 6 tmop-fitting -m ../../data/inline-tri.mesh -o 1 -rs 1 -ni 100 -rtol 1e-10 -mod-bndr-attr -sfc 10 -sft 1e-6 -tid 1 -mid 2 -ls 4 -ctot 0 -vl 1 -li 1000 -ae 1 -slstype 7  -adw 0 -sfa 0 -jid 201


// 3D
// make tmop-fitting -j && mpirun -np 6 tmop-fitting -m ../../data/inline-hex.mesh -o 2 -rs 2 -ni 100 -rtol 1e-10 -mod-bndr-attr -sft 1e-08 -tid 1 -mid 303 -ls 4 -ctot 0 -cus-mat -vl 1 -li 1000 -ae 1 -slstype 1 -sfc 1 -sfa 2 -jid 400 -sfcmax 1e10 -sfcjac 1e-5
// make tmop-fitting -j && mpirun -np 6 tmop-fitting -m cube.mesh -o 2 -rs 2 -ni 100 -rtol 1e-10 -mod-bndr-attr -sft 1e-08 -tid 1 -mid 303 -ls 4 -ctot 2 -cus-mat -vl 1 -li 1000 -ae 1 -slstype 1 -sfc 1 -sfa 2 -jid 400 -sfcmax 1e10 -sfcjac 1e-5

#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include <iostream>
#include <fstream>
#include "tmop-fitting.hpp"

using namespace mfem;
using namespace std;

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

double GetMinDet(ParMesh *pmesh, ParFiniteElementSpace *pfespace,
                 IntegrationRules *irules, int quad_order)
{
   double tauval = infinity();
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

      const IntegrationRule &ir2 = pfespace->GetFE(i)->GetNodes();
      for (int j = 0; j < ir2.GetNPoints(); j++)
      {
         transf->SetIntPoint(&ir2.IntPoint(j));
         tauval = min(tauval, transf->Jacobian().Det());
      }
   }
   double minJ0;
   MPI_Allreduce(&tauval, &minJ0, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   tauval = minJ0;
   return tauval;
}

double GetWorstJacobianSkewness(DenseMatrix &J)
{
   double size;
   double skew = -100000;
   const int dim = J.Size();
   if (dim == 2)
   {
      Vector col1, col2;
      J.GetColumn(0, col1);
      J.GetColumn(1, col2);

      skew = std::atan2(J.Det(), col1 * col2);
      skew = std::fabs(skew);
   }
   else
   {
      Vector skewv(dim);
      Vector col1, col2, col3;
      J.GetColumn(0, col1);
      J.GetColumn(1, col2);
      J.GetColumn(2, col3);
      double len1 = col1.Norml2(),
             len2 = col2.Norml2(),
             len3 = col3.Norml2();

      Vector col1unit = col1,
             col2unit = col2,
             col3unit = col3;
      col1unit *= 1.0/len1;
      col2unit *= 1.0/len2;
      col3unit *= 1.0/len3;

      Vector crosscol12, crosscol13, crosscol23;
      col1.cross3D(col2, crosscol12);
      col1.cross3D(col3, crosscol13);
      col2.cross3D(col3, crosscol23);
      skewv(0) = std::atan2(crosscol12.Norml2(),col1*col2);
      skewv(1) = std::atan2(crosscol13.Norml2(),col1*col3);
      skewv(2) = std::atan2(col1*crosscol23,crosscol12*crosscol13);
      for (int d = 0; d < dim; d++)
      {
         skew = std::max(skew, std::fabs(skewv(d)));
      }
   }

   return skew;
}

double GetWorstSkewness(ParMesh *pmesh, ParFiniteElementSpace *pfespace,
                        IntegrationRules *irules, int quad_order)
{
   double skewval = -1000;
   const int NE = pmesh->GetNE();
   const int dim = pmesh->Dimension();
   for (int i = 0; i < NE; i++)
   {
      const IntegrationRule &ir =
         irules->Get(pfespace->GetFE(i)->GetGeomType(), quad_order);
      ElementTransformation *transf = pmesh->GetElementTransformation(i);
      DenseMatrix Jac(dim);

      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         transf->SetIntPoint(&ir.IntPoint(j));
         pmesh->GetElementJacobian(i, Jac, &ir.IntPoint(j));
         double skew_q = GetWorstJacobianSkewness(Jac);
         //         skewval = std::max(skewval, skew_q);
      }

      const IntegrationRule &ir2 = pfespace->GetFE(i)->GetNodes();
      for (int j = 0; j < ir2.GetNPoints(); j++)
      {
         transf->SetIntPoint(&ir2.IntPoint(j));
         pmesh->GetElementJacobian(i, Jac, &ir.IntPoint(j));
         double skew_q = GetWorstJacobianSkewness(Jac);
         skewval = std::max(skewval, skew_q);
      }
   }
   double skewvalglobal;
   MPI_Allreduce(&skewval, &skewvalglobal, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   return skewvalglobal;
}

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

int main (int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   const int num_procs = Mpi::WorldSize();
   Hypre::Init();

   // 1. Set the method's default parameters.
   const char *mesh_file = "icf.mesh";
   int mesh_poly_deg     = 1;
   int rs_levels         = 0;
   int rp_levels         = 0;
   double jitter         = 0.0;
   int metric_id         = 1;
   int target_id         = 1;
   double surface_fit_const = 0.0;
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
   int adapt_eval        = 1;
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
   bool mu_linearization  = false;
   int adaptive_weight   = 0;
   int custom_split_mesh = 0;
   bool custom_material   = false;
   int jobid  = 0;
   double surf_inc_trigger_rel_thresold = 0.01;
   double surf_fit_const_max = 1e20;
   double surf_fit_min_det_threshold = 0.0;
   bool normalization    = false;
   double worst_skew = 0.0;

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
   args.AddOption(&jitter, "-ji", "--jitter",
                  "Random perturbation scaling factor.");
   args.AddOption(&metric_id, "-mid", "--metric-id",
                  "Mesh optimization metric:\n\t");
   args.AddOption(&target_id, "-tid", "--target-id",
                  "Target (ideal element) type:\n\t"
                  "1: Ideal shape, unit size\n\t"
                  "2: Ideal shape, equal size\n\t"
                  "3: Ideal shape, initial size\n\t"
                  "4: Given full analytic Jacobian (in physical space)\n\t"
                  "5: Ideal shape, given size (in physical space)");
   args.AddOption(&surface_fit_const, "-sfc", "--surface-fit-const",
                  "Surface preservation constant.");
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
                  "1 - Interface (DEFAULT), 2 - Boundary attribute.");
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
   args.AddOption(&mu_linearization, "-mulin", "--mu-linearization", "-no-mulin",
                  "--no-mu-linearization",
                  "Linearized form of metric.");
   args.AddOption(&adaptive_weight, "-adw", "--adw",
                  "0 - initial weight is not adaptive\n"
                  "1 - initial weight is set adaptively - constant\n"
                  "2 - pointwise weight updated after each Newton iteration\n");
   args.AddOption(&custom_split_mesh, "-ctot", "--custom_split_mesh",
                  "Split Mesh Into Tets/Tris/Quads for consistent materials");
   args.AddOption(&custom_material, "-cus-mat", "--custom-material",
                  "-no-cus-mat", "--no-custom-material",
                  "When true, sets the material based on predetermined logic instead of level-set");
   args.AddOption(&jobid, "-jid", "--jid",
                  "job id used for visit  save files");
   args.AddOption(&surf_inc_trigger_rel_thresold, "-sfct",
                  "--surf-fit-change-threshold",
                  "Set threshold for surface fitting increase based on relative error decrease");
   args.AddOption(&surf_fit_const_max, "-sfcmax", "--surf-fit-const-max",
                  "Max surface fitting weight allowed");
   args.AddOption(&surf_fit_min_det_threshold, "-sfcjac", "--surf-fit-jac",
                  "Threshold on minimum determinant for deciding how long surf fit const can be increased");
   args.AddOption(&normalization, "-nor", "--normalization", "-no-nor",
                  "--no-normalization",
                  "Make all terms in the optimization functional unitless.");
   args.AddOption(&worst_skew, "-skewmax", "--skew-max",
                  "worst skewness in degrees.. between 90 and 180");
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
   if (custom_split_mesh > 0)
   {
      int res = std::max(1, rs_levels);
      //SPLIT TYPE == 1 - 12 TETS, 2 = 24 TETS
      mesh = new Mesh(Mesh::MakeHexTo24or12TetMesh(2*res,2*res,2*res,
                                                   1.0, 1.0, 1.0,
                                                   custom_split_mesh)); //24tet
   }
   else if (custom_split_mesh == 0)
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
      //SPLIT TYPE == -1 => 1 quad to 4 tris
      if (custom_split_mesh == -1)
      {
         mesh = new Mesh(Mesh::MakeQuadTo4TriMesh(2*res,2*res, 1.0, 1.0));
      }
      else   //1 quad to 5 quads
      {
         mesh = new Mesh(Mesh::MakeQuadTo5QuadMesh(2*res,2*res, 1.0, 1.0));

      }
   }
   if (custom_split_mesh != 0)
   {
      for (int e = 0; e < mesh->GetNE(); e++)
      {
         mesh->SetAttribute(e, e);
      }
   }

   if (myid == 0)
   {
      std::cout << "Mesh read/setup\n";
   }

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
   else if (surf_ls_type == 3) //Circle
   {
      ls_coeff = new FunctionCoefficient(squircle_level_set);
   }
   else if (surf_ls_type == 6) // 3D shape
   {
      ls_coeff = new FunctionCoefficient(csg_cubecylsph);
   }
   else if (surf_ls_type == 7) // 3D shape
   {
      ls_coeff = new FunctionCoefficient(inclined_plane);
   }
   else if (surf_ls_type == 8)
   {
      ls_coeff = new FunctionCoefficient(csg_cubecylsph_smooth);
   }
   else if (surf_ls_type == 9)
   {
      ls_coeff = new FunctionCoefficient(flat_plane);
   }
   else if (surf_ls_type == 10)
   {
      ls_coeff = new FunctionCoefficient(sinusoidal);
   }
   else
   {
      MFEM_ABORT("Surface fitting level set type not implemented yet.")
   }
   if (int_amr_iters > 0)
   {
      mesh->EnsureNCMesh(true);
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   if (myid == 0)
   {
      std::cout << "ParMesh setup\n";
   }

   for (int lev = 0; lev < rp_levels; lev++)
   {
      pmesh->UniformRefinement();
   }
   int neglob = pmesh->GetGlobalNE();
   int neglob_preamr = neglob;
   if (myid == 0)
   {
      std::cout << "k10-Number of elements: " << neglob << std::endl;
   }

   HRefUpdater HRUpdater = HRefUpdater();

   // Set Mesh Nodal Space and GridFunction
   FiniteElementCollection *fec = new H1_FECollection(mesh_poly_deg, dim);
   ParFiniteElementSpace *pfespace = new ParFiniteElementSpace(pmesh, fec, dim,
                                                               mesh_node_ordering);
   pmesh->SetNodalFESpace(pfespace);

   // Get the mesh nodes (vertices and other degrees of freedom in the finite
   //    element space) as a finite element grid function in fespace.
   ParGridFunction x(pfespace);
   pmesh->SetNodalGridFunction(&x);
   x.SetTrueVector();
   HRUpdater.AddFESpaceForUpdate(pfespace);
   HRUpdater.AddGridFunctionForUpdate(&x);

   // 11. Store the starting (prior to the optimization) positions.
   ParGridFunction x0(pfespace);
   x0 = x;
   HRUpdater.AddGridFunctionForUpdate(&x0);

   if (surface_fit_const == 0.0)
   {
      DataCollection *dc = NULL;
      dc = new VisItDataCollection("Perturbed_"+std::to_string(jobid), pmesh);
      dc->SetCycle(0);
      dc->SetTime(0.0);
      dc->Save();
      delete dc;
   }

   // Setup background mesh for surface fitting
   // Define relevant spaces and gridfunctions
   ParMesh *pmesh_surf_fit_bg = NULL;
   AdaptivityEvaluator *remap_from_bg = NULL;
   FiniteElementCollection *surf_fit_bg_fec = NULL;
   ParFiniteElementSpace *surf_fit_bg_fes = NULL;
   ParGridFunction *surf_fit_bg_gf0 = NULL;

   if (surf_bg_mesh)
   {
      if (myid == 0)
      {
         std::cout << "Setup Background Mesh\n";
      }
      MFEM_VERIFY(surface_fit_const > 0,
                  "Fitting is not active. Why background mesh?");
      Mesh *mesh_surf_fit_bg = NULL;
      if (dim == 2)
      {
         mesh_surf_fit_bg = new Mesh(Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL,
                                                           true));
      }
      else if (dim == 3)
      {
         mesh_surf_fit_bg = new Mesh(Mesh::MakeCartesian3D(4, 4, 4, Element::HEXAHEDRON,
                                                           true));
      }
      mesh_surf_fit_bg->EnsureNCMesh(true);
      pmesh_surf_fit_bg = new ParMesh(MPI_COMM_WORLD, *mesh_surf_fit_bg);
      delete mesh_surf_fit_bg;

      pmesh_surf_fit_bg->SetCurvature(1);
      Vector p_min(dim), p_max(dim);
      pmesh->GetBoundingBox(p_min, p_max);
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
      surf_fit_bg_fec = new H1_FECollection(mesh_poly_deg+1, dim);
      surf_fit_bg_fes = new ParFiniteElementSpace(pmesh_surf_fit_bg, surf_fit_bg_fec);
      surf_fit_bg_gf0 = new ParGridFunction(surf_fit_bg_fes);

      surf_fit_bg_gf0->ProjectCoefficient(*ls_coeff);
      OptimizeMeshWithAMRAroundZeroLevelSet(*pmesh_surf_fit_bg, *ls_coeff, amr_iters,
                                            *surf_fit_bg_gf0);
      pmesh_surf_fit_bg->Rebalance();
      surf_fit_bg_fes->Update();
      surf_fit_bg_gf0->Update();

      if (comp_dist)
      {
         ComputeScalarDistanceFromLevelSet(*pmesh_surf_fit_bg, *ls_coeff,
                                           *surf_fit_bg_gf0);
      }
      else
      {
         surf_fit_bg_gf0->ProjectCoefficient(*ls_coeff);
      }

      {
         DataCollection *dc = NULL;
         dc = new VisItDataCollection("bg_"+std::to_string(jobid), pmesh_surf_fit_bg);
         dc->RegisterField("level-set", surf_fit_bg_gf0);
         dc->SetCycle(0);
         dc->SetTime(0.0);
         dc->SetFormat(DataCollection::SERIAL_FORMAT);
         dc->Save();
         delete dc;
      }

#ifdef MFEM_USE_GSLIB
      remap_from_bg = new InterpolatorFP;
#else
      MFEM_ABORT("MFEM is not built with GSLIB support!");
#endif

      remap_from_bg->SetParMetaInfo(*pmesh_surf_fit_bg, *surf_fit_bg_fes);
      remap_from_bg->SetInitialField(*pmesh_surf_fit_bg->GetNodes(),
                                     *surf_fit_bg_gf0);
   }

   if (myid == 0)
   {
      std::cout << "Background Mesh setup done\n";
   }


   // Surface fitting.
   L2_FECollection mat_coll(0, dim);
   H1_FECollection surf_fit_fec(mesh_poly_deg, dim);
   ParFiniteElementSpace surf_fit_fes(pmesh, &surf_fit_fec);
   ParFiniteElementSpace mat_fes(pmesh, &mat_coll);
   ParGridFunction mat(&mat_fes);
   ParGridFunction pgl_el_num(&mat_fes);
   ParGridFunction surf_fit_gf0(&surf_fit_fes);
   ParGridFunction NumFaces(&mat_fes);
   HRUpdater.AddFESpaceForUpdate(&surf_fit_fes);
   HRUpdater.AddFESpaceForUpdate(&mat_fes);
   HRUpdater.AddGridFunctionForUpdate(&mat);
   HRUpdater.AddGridFunctionForUpdate(&surf_fit_gf0);
   HRUpdater.AddGridFunctionForUpdate(&pgl_el_num);
   HRUpdater.AddGridFunctionForUpdate(&NumFaces);
   Array<bool> surf_fit_marker(0);

   for (int e = 0; e < pmesh->GetNE(); e++)
   {
      pgl_el_num(e) = pmesh->GetAttribute(e);
   }

   Array<int> vdofs;
   Array<int> orig_pmesh_attributes(0);
   ParMesh *psubmesh = NULL;
   ParFiniteElementSpace *psub_pfespace = NULL;
   ParGridFunction *psub_x = NULL;
   ParGridFunction *psub_x0 = NULL;
   if (surface_fit_const > 0.0)
   {
      surf_fit_gf0.ProjectCoefficient(*ls_coeff);
      if (comp_dist && !surf_bg_mesh)
      {
         ComputeScalarDistanceFromLevelSet(*pmesh, *ls_coeff, surf_fit_gf0);
      }
      else if (surf_bg_mesh)
      {
         remap_from_bg->ComputeAtNewPosition(*pmesh->GetNodes(),
                                             surf_fit_gf0,
                                             pmesh->GetNodalFESpace()->GetOrdering());
      }

      // Set material gridfunction
      for (int i = 0; i < pmesh->GetNE(); i++)
      {
         if (material)
         {
            mat(i) = pmesh->GetAttribute(i)-1;
         }
         else
         {
            if (custom_material)
            {
               Vector center(pmesh->Dimension());
               pmesh->GetElementCenter(i, center);
               mat(i) = 1;
               if (surf_ls_type == 10)
               {
                  mat(i) = center(1) > 0.5 ? 0 : 1;
               }
               else
               {
                  if (center(0) > 0.25 && center(0) < 0.75 && center(1) > 0.25 &&
                      center(1) < 0.75)
                  {
                     if (dim == 2 || (dim == 3 && center(2) > 0.25 && center(2) < 0.75))
                     {
                        mat(i) = 0;
                     }
                  }
               }
            }
            else
            {
               mat(i) = material_id(i, surf_fit_gf0);
            }
            pmesh->SetAttribute(i, mat(i) + 1);
         }
      }
      mat.ExchangeFaceNbrData();

      if (custom_split_mesh > 0)
      {
         MakeMaterialConsistentForElementGroups(mat, pgl_el_num,
                                                12*custom_split_mesh);
      }
      else if (custom_split_mesh < 0)
      {
         MakeMaterialConsistentForElementGroups(mat, pgl_el_num,
                                                custom_split_mesh == -1 ? 4 : 5);
      }

      mat.ExchangeFaceNbrData();

      for (int i = 0; i < pmesh->GetNE(); i++)
      {
         pmesh->SetAttribute(i, mat(i) + 1);
      }
      pmesh->SetAttributes();

      // Adapt attributes for marking such that if all but 1 face of an element
      // are marked, the element attribute is switched.
      MakeGridFunctionWithNumberOfInterfaceFaces(pmesh, mat, NumFaces);
      if (adapt_marking && custom_split_mesh == 0)
      {
         ModifyAttributeForMarkingDOFS(pmesh, mat, 0);
         MakeGridFunctionWithNumberOfInterfaceFaces(pmesh, mat, NumFaces);
         ModifyAttributeForMarkingDOFS(pmesh, mat, 1);
         MakeGridFunctionWithNumberOfInterfaceFaces(pmesh, mat, NumFaces);
         ModifyTetAttributeForMarking(pmesh, mat, 0);
         MakeGridFunctionWithNumberOfInterfaceFaces(pmesh, mat, NumFaces);
         ModifyTetAttributeForMarking(pmesh, mat, 1);
         MakeGridFunctionWithNumberOfInterfaceFaces(pmesh, mat, NumFaces);
      }

      MakeGridFunctionWithNumberOfInterfaceFaces(pmesh, mat, NumFaces);

      if (marking_type == 0)
      {
         //need to check material consistency for AMR meshes
         int matcheck = CheckMaterialConsistency(pmesh, mat);
         MFEM_VERIFY(matcheck, "Not all children at the interface have same material.");
      }

      // Refine elements near fitting boundary
      if (int_amr_iters)
      {
         for (int i = 0; i < int_amr_iters; i++)
         {
            Array<int> refinements;
            if (marking_type > 0)
            {
               GetBoundaryElements(pmesh, mat, refinements, marking_type);
            }
            else
            {
               //               GetMaterialInterfaceElements(pmesh, mat, refinements);
               ParGridFunction surf_fit_mat_gf_temp(&surf_fit_fes);
               Array<int> intdofs, dofs;
               GetMaterialInterfaceDofs(pmesh, mat, surf_fit_mat_gf_temp, intdofs);
               surf_fit_mat_gf_temp = 0.0;
               for (int ii = 0; ii < intdofs.Size(); ii++)
               {
                  surf_fit_mat_gf_temp(intdofs[ii]) = 1.0;
               }
               surf_fit_mat_gf_temp.GroupCommunicatorOp(2);
               surf_fit_mat_gf_temp.ExchangeFaceNbrData();
               Vector datavec;
               for (int e = 0; e < pmesh->GetNE(); e++)
               {
                  surf_fit_fes.GetElementDofs(e, dofs);
                  surf_fit_mat_gf_temp.GetSubVector(dofs, datavec);
                  //optional for run for paper
                  //                    if (surf_ls_type == 3)
                  //                    {
                  //                        Vector center(pmesh->Dimension());
                  //                        pmesh->GetElementCenter(e, center);
                  //                        if (center(0) >= 0.375 && center(0) <= 0.625) {
                  //                            datavec = 0.0;
                  //                        }
                  //                        if (center(1) >= 0.375 && center(1) <= 0.625) {
                  //                            datavec = 0.0;
                  //                        }
                  //                    }
                  if (datavec.Max() == 1.0) { refinements.Append(e); }
               }
            }
            refinements.Sort();
            refinements.Unique();
            //            if (i >= int_amr_iters-2)
            {
               //               ExtendRefinementListToNeighbors(*pmesh, refinements);
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
      neglob_preamr = neglob;
      neglob = pmesh->GetGlobalNE();
      if (myid == 0)
      {
         std::cout << "k10-Number of elements after AMR: " << neglob << std::endl;
      }
   }
   int ntdofsglob = x.ParFESpace()->GlobalTrueVSize();
   int ndofsglob = x.ParFESpace()->GlobalVSize();
   if (myid == 0)
   {
      std::cout << "k10-NE,NDofs,NDofsTrue InputMesh: " << neglob << " " <<
                ndofsglob << " " << ntdofsglob << std::endl;
   }

   if (mod_bndr_attr)
   {
      ModifyBoundaryAttributesForNodeMovement(pmesh, x);
      pmesh->SetAttributes();
   }
   pmesh->ExchangeFaceNbrData();

   if (surface_fit_const > 0.0 && visualization)
   {
      socketstream vis1, vis2, vis3, vis4;

      common::VisualizeField(vis1, "localhost", 19916, surf_fit_gf0,
                             "Surface dof", 900, 900, 300, 300);
      common::VisualizeField(vis2, "localhost", 19916, mat,
                             "Materials", 600, 900, 300, 300);
      if (surf_bg_mesh)
      {
         common::VisualizeField(vis4, "localhost", 19916, *surf_fit_bg_gf0,
                                "Level Set 0 Source",
                                300, 600, 300, 300);
      }
   }

   int num_active_glob = neglob;
   int num_procs_submesh = num_procs; //will initialize to 0 later
   int ndofsglobsubmesh = ndofsglob;
   int ntdofsglobsubmesh = ntdofsglob;

   int max_el_attr = pmesh->attributes.Max();
   int max_bdr_el_attr = pmesh->bdr_attributes.Max();
   Array<int> deactivate_list(0);

   if (deactivation_layers > 0)
   {
      num_procs_submesh = 0;
      MFEM_VERIFY(!int_amr_iters, "Submesh does not support nonconforming meshes.");
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
         std::cout << "k10-Number of elements in the submesh: " << num_active_glob <<
                   endl;
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
               pmesh->SetAttribute(i, max_el_attr+orig_pmesh_attributes[i]);
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
      psub_x0 = new ParGridFunction(psub_pfespace);
      *psub_x0 = *psub_x;
      psubmesh->SetAttributes();
      num_active_glob = psubmesh->GetGlobalNE();
      num_procs_submesh = psubmesh->GetNE() > 0 ? 1 : 0;
      ndofsglobsubmesh = psub_x0->ParFESpace()->GlobalVSize();
      ntdofsglobsubmesh = psub_x0->ParFESpace()->GlobalTrueVSize();

      MPI_Allreduce(MPI_IN_PLACE, &num_procs_submesh, 1, MPI_INT, MPI_SUM,
                    pmesh->GetComm());

      if (myid == 0)
      {
         std::cout << "k10-Number of elements in the submesh 2: " << num_active_glob <<
                   endl;
         std::cout << "k10-Number of ranks for submesh: " << num_procs_submesh <<
                   std::endl;
         std::cout << "k10-NE,NDofs,NDofsTrue submesh: " << num_active_glob << " " <<
                   ndofsglobsubmesh << " " << ntdofsglobsubmesh << std::endl;
      }

      //Fix boundary attribues of submesh
      //      int n_bdr_el_attr_psub = psubmesh->bdr_attributes.Size();
      //      int max_bdr_el_attr_psub = psubmesh->bdr_attributes.Max();
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
      psub_x0 = &x0;
   }
   psub_x->SetTrueVector();

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
   {
      ostringstream mesh_name;
      mesh_name << "perturbed_submesh.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      if (int_amr_iters)
      {
         //  psubmesh->PrintAsOne(mesh_ofs);
      }
      else
      {
         //  psubmesh->PrintAsSerial(mesh_ofs);
      }
   }


   Vector b(0);

   // 12. Form the integrator that uses the chosen metric and target.
   double tauval = -0.1;
   TMOP_QualityMetric *metric = NULL;
   switch (metric_id)
   {
      // T-metrics
      case 0: metric = new TMOP_Metric_000; break;
      case 1: metric = new TMOP_Metric_001; break;
      case 2: metric = new TMOP_Metric_002; break;
      case 55: metric = new TMOP_Metric_055; break;
      case 58: metric = new TMOP_Metric_058; break;
      case 80: metric = new TMOP_Metric_080(0.5); break;
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
   if (mu_linearization)
   {
      metric->EnableLinearization();
      //       metric->EnablePartialLinearization();
   }


   if (metric_id > 0 && metric_id < 300)
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
   target_c->SetNodes(*psub_x0);


   TMOP_Integrator *tmop_integ = new TMOP_Integrator(metric, target_c);

   // Setup the quadrature rules for the TMOP integrator.
   IntegrationRules *irules = &IntRulesLo;
   tmop_integ->SetIntegrationRules(*irules, quad_order);

   pmesh->ExchangeFaceNbrData();
   psubmesh->ExchangeFaceNbrData();

   // Background mesh FECollection, FESpace, and GridFunction
   ParFiniteElementSpace *surf_fit_bg_grad_fes = NULL;
   ParGridFunction *surf_fit_bg_grad = NULL;
   ParFiniteElementSpace *surf_fit_bg_hess_fes = NULL;
   ParGridFunction *surf_fit_bg_hess = NULL;

   if (surf_bg_mesh)
   {
      surf_fit_bg_grad_fes = new ParFiniteElementSpace(pmesh_surf_fit_bg,
                                                       surf_fit_bg_fec,
                                                       pmesh_surf_fit_bg->Dimension());
      surf_fit_bg_grad = new ParGridFunction(surf_fit_bg_grad_fes);


      int n_hessian_bg = pow(pmesh_surf_fit_bg->Dimension(), 2);
      surf_fit_bg_hess_fes = new ParFiniteElementSpace(pmesh_surf_fit_bg,
                                                       surf_fit_bg_fec,
                                                       n_hessian_bg);
      surf_fit_bg_hess = new ParGridFunction(surf_fit_bg_hess_fes);

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
   }


   // If a background mesh is used, we interpolate the Gradient and Hessian
   // from that mesh to the current mesh being optimized.
   ParFiniteElementSpace *surf_fit_grad_fes = NULL;
   ParGridFunction *surf_fit_grad = NULL;
   ParFiniteElementSpace *surf_fit_hess_fes = NULL;
   ParGridFunction *surf_fit_hess = NULL;

   ParGridFunction *jacobian_init = NULL;
   ParGridFunction *jacobian_current = NULL;
   ParGridFunction jacobian_ratio(&psub_surf_fit_fes);

   double surf_fit_err_avg, surf_fit_err_max;
   double init_surf_fit_err_avg, init_surf_fit_err_max;

   // Surface fitting.
   AdaptivityEvaluator *adapt_surface = NULL;
   AdaptivityEvaluator *adapt_grad_surface = NULL;
   AdaptivityEvaluator *adapt_hess_surface = NULL;
   Array<bool> psub_surf_fit_marker(psub_surf_fit_gf0.Size());
   ConstantCoefficient surf_fit_coeff(surface_fit_const);
   ParGridFunction surf_fit_mat_gf(&psub_surf_fit_fes);
   Array<int> fitting_face_list;

   if (surface_fit_const > 0.0)
   {
      surf_fit_grad_fes = new ParFiniteElementSpace(psubmesh, &surf_fit_fec,
                                                    psubmesh->Dimension());
      surf_fit_grad = new ParGridFunction(surf_fit_grad_fes);

      surf_fit_hess_fes = new ParFiniteElementSpace(psubmesh, &surf_fit_fec,
                                                    psubmesh->Dimension()*psubmesh->Dimension());
      surf_fit_hess = new ParGridFunction(surf_fit_hess_fes);

      GridFunctionCoefficient coeff_mat(&psub_mat);
      surf_fit_mat_gf.ProjectDiscCoefficient(coeff_mat, GridFunction::ARITHMETIC);
      surf_fit_mat_gf.SetTrueVector();
      surf_fit_mat_gf.SetFromTrueVector();

      // Set DOFs for fitting
      // Strategy 1: Automatically choose face between elements of different attribute.
      if (marking_type == 0)
      {
         GetMaterialInterfaceFaces(psubmesh, psub_mat, fitting_face_list);

         psub_surf_fit_marker.SetSize(psub_surf_fit_gf0.Size());
         for (int j = 0; j < psub_surf_fit_marker.Size(); j++)
         {
            psub_surf_fit_marker[j] = false;
         }
         surf_fit_mat_gf = 0.0;

         Array<int> dof_list;
         Array<int> dofs;
         for (int i = 0; i < fitting_face_list.Size(); i++)
         {
            psub_surf_fit_gf0.ParFESpace()->GetFaceDofs(fitting_face_list[i], dofs);
            dof_list.Append(dofs);
         }
         for (int i = 0; i < dof_list.Size(); i++)
         {
            psub_surf_fit_marker[dof_list[i]] = true;
            surf_fit_mat_gf(dof_list[i]) = 1.0;
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
            if (attr == marking_type)
            {
               fitting_face_list.Append(psubmesh->GetBdrFace(i));
               psub_surf_fit_fes.GetBdrElementVDofs(i, vdofs);
               for (int j = 0; j < vdofs.Size(); j++)
               {
                  psub_surf_fit_marker[vdofs[j]] = true;
                  surf_fit_mat_gf(vdofs[j]) = 1.0;
               }
            }
         }
      }

      // Unify across processor boundaries
      surf_fit_mat_gf.GroupCommunicatorOp(2);
      surf_fit_mat_gf.ExchangeFaceNbrData();

      for (int i = 0; i < surf_fit_mat_gf.Size(); i++)
      {
         psub_surf_fit_marker[i] = surf_fit_mat_gf(i) == 1.0;
      }

      if (adapt_eval == 0)
      {
         adapt_surface = new AdvectorCG;
         adapt_grad_surface = new AdvectorCG;
         adapt_hess_surface = new AdvectorCG;
         MFEM_VERIFY(!surf_bg_mesh, "Background meshes require GSLIB.");
      }
      else if (adapt_eval == 1)
      {
#ifdef MFEM_USE_GSLIB
         adapt_surface = new InterpolatorFP;
         adapt_grad_surface = new InterpolatorFP;
         adapt_hess_surface = new InterpolatorFP;
#else
         MFEM_ABORT("MFEM is not built with GSLIB support!");
#endif
      }
      else { MFEM_ABORT("Bad interpolation option."); }

      if (!surf_bg_mesh)
      {
         if (num_procs_submesh != num_procs)
         {
            MFEM_ABORT("Empty MPI rank not currently supported without background mesh.");
         }
         tmop_integ->EnableSurfaceFitting(psub_surf_fit_gf0, psub_surf_fit_marker,
                                          surf_fit_coeff,
                                          *adapt_surface, adapt_grad_surface,
                                          adapt_hess_surface, true);
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

      jacobian_init = new ParGridFunction(&psub_surf_fit_fes);
      jacobian_current = new ParGridFunction(&psub_surf_fit_fes);
      psub_mat.ExchangeFaceNbrData();
   }

   // Set initial fitting weight automatically
   if (surface_fit_const > 0)
   {
      tmop_integ->ComputePointWiseJacobian(*psub_x, *jacobian_init, 0);
      jacobian_ratio = 1.0;
   }
   if (adaptive_weight > 0)
   {
      if (surface_fit_adapt == 0.0) { surface_fit_adapt = 1.0; }
      if (adaptive_weight == 1)
      {
         tmop_integ->SetFittingWeightScaleWithPointWiseJacobian(*psub_x);
      }
      if (adaptive_weight == 2)
      {
         double new_weight = tmop_integ->GetInitialFittingWeightUsingGradientNorm(
                                *psub_x);
         surface_fit_const = new_weight;
         surf_fit_coeff.constant = surface_fit_const;
         tmop_integ->UpdateSurfaceFittingCoefficient(surf_fit_coeff);
      }

      if (myid == 0)
      {
         std::cout << "Initial fitting weight will be: " <<
                   surf_fit_coeff.constant << std::endl;
      }
   }


   if (surface_fit_const > 0.0)
   {
      tmop_integ->GetSurfaceFittingErrors(surf_fit_err_avg, surf_fit_err_max);
      init_surf_fit_err_avg = surf_fit_err_avg;
      init_surf_fit_err_max = surf_fit_err_max;
      if (myid == 0)
      {
         std::cout << "Initial mesh - Avg fitting error: " << surf_fit_err_avg <<
                   std::endl
                   << "Initial mesh - Max fitting error: " << surf_fit_err_max << std::endl;
         std::cout << "Avg/Max surface fitting error: " << surf_fit_err_avg << " " <<
                   surf_fit_err_max << std::endl;
         std::cout << "Min/Max surface fitting weight: " << surface_fit_const << " " <<
                   surface_fit_const << std::endl;
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
         }
         common::VisualizeField(vis5, "localhost", 19916, *jacobian_init,
                                "Jacobian Initial",
                                900, 600, 300, 300);
      }
   }


   if (surface_fit_const > 0.0)
   {
      ParGridFunction pmesh_surf_fit_mat_gf(&surf_fit_fes);
      if (deactivation_layers > 0)
      {
         auto tmap1 = ParSubMesh::CreateTransferMap(surf_fit_mat_gf,
                                                    pmesh_surf_fit_mat_gf);
         tmap1.Transfer(surf_fit_mat_gf, pmesh_surf_fit_mat_gf);
      }

      DataCollection *dc = NULL;
      dc = new VisItDataCollection("Perturbed_"+std::to_string(jobid), pmesh);
      dc->RegisterField("level-set", &surf_fit_gf0);
      dc->RegisterField("Marker", &pmesh_surf_fit_mat_gf);
      dc->RegisterField("NumFaces", &NumFaces);
      dc->RegisterField("mat", &mat); //for nc meshes
      dc->SetCycle(0);
      dc->SetTime(0.0);
      dc->SetFormat(DataCollection::SERIAL_FORMAT);
      dc->Save();
      delete dc;
   }

   if (normalization)
   {
      surf_fit_coeff.constant   = 1.0;
      tmop_integ->ParEnableNormalization(x0);
      surf_fit_coeff.constant  = surface_fit_const;
   }

   Vector integrated_face_error(fitting_face_list.Size());
   double tot_init_integ_error = 0.0;
   double tot_final_integ_error = 0.0;
   if (surface_fit_const > 0.0)
   {
      tot_init_integ_error = ComputeIntegrateErroronInterfaces(psubmesh,
                                                               ls_coeff,
                                                               fitting_face_list,
                                                               surf_fit_bg_gf0,
                                                               remap_from_bg,
                                                               integrated_face_error, 0);
   }


   ParNonlinearForm a(psub_pfespace);
   ConstantCoefficient *metric_coeff1 = NULL;
   a.AddDomainIntegrator(tmop_integ);

   // Compute the minimum det(J) of the starting mesh.
   tauval = GetMinDet(psubmesh, psub_pfespace, irules, quad_order);
   if (myid == 0)
   { cout << "Minimum det(J) of the original mesh is " << tauval << endl; }
   double init_min_det = tauval;
   double skewval = GetWorstSkewness(psubmesh, psub_pfespace, irules, quad_order);
   if (myid == 0)
   { cout << "Worst skewness of the original mesh is " << skewval*180.0/M_PI << endl; }
   double init_skew = skewval;

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
   double init_metric_energy = 0.0;
   if (surface_fit_const > 0.0)
   {
      surf_fit_coeff.constant   = 0.0;
      init_metric_energy = a.GetParGridFunctionEnergy(*psub_x);
      surf_fit_coeff.constant  = surface_fit_const;
   }

   // Visualize the starting mesh and metric values.
   // Note that for combinations of metrics, this only shows the first metric.
   if (visualization)
   {
      char title[] = "Initial metric values";
      TargetConstructor target_c2(target_t, MPI_COMM_WORLD);
      target_c2.SetNodes(x0);
      vis_tmop_metric_p(mesh_poly_deg, *metric, target_c2, *pmesh, title, 0);
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
         for (int d = 0; d < dim; d++)
         {
            if (attr == d+1)
            {
               for (int j = 0; j < nd; j++)
               { ess_vdofs[n++] = vdofs[j+d*nd]; }
            }
         }
         if (attr > dim) // Fix all components.
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
   if (lin_solver >= 2)
   {
      MINRESSolver *minres = new MINRESSolver(MPI_COMM_WORLD);
      //      GMRESSolver *minres = new GMRESSolver(MPI_COMM_WORLD);
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
   else
   {
      MFEM_ABORT("Invalid lin_solver");
   }

   {
      ParGridFunction pmesh_surf_fit_mat_gf(&surf_fit_fes);
      ParGridFunction pmesh_jacobian_ratio(&surf_fit_fes);
      if (deactivation_layers > 0)
      {
         auto tmap1 = ParSubMesh::CreateTransferMap(surf_fit_mat_gf,
                                                    pmesh_surf_fit_mat_gf);
         tmap1.Transfer(surf_fit_mat_gf, pmesh_surf_fit_mat_gf);
         tmap1.Transfer(jacobian_ratio, pmesh_jacobian_ratio);
      }

      DataCollection *dc = NULL;
      dc = new VisItDataCollection("Optimized_"+std::to_string(jobid), pmesh);
      dc->RegisterField("mat", &mat);
      dc->RegisterField("level-set", &surf_fit_gf0);
      dc->RegisterField("Marker", &pmesh_surf_fit_mat_gf);
      dc->RegisterField("gradratio", &pmesh_jacobian_ratio);
      dc->SetCycle(0);
      dc->SetTime(0.0);
      dc->Save();
      delete dc;
   }

   // Perform the nonlinear optimization.
   const IntegrationRule &ir =
      irules->Get(pfespace->GetFE(0)->GetGeomType(), quad_order);
   TMOPNewtonSolver solver(psub_pfespace->GetComm(), ir, solver_type);
   // For untangling, the solver will update the min det(T) values.
   if (tauval < 0.0) { solver.SetMinDetPtr(&tauval); }

   if (surface_fit_adapt > 0.0)
   {
      solver.SetAdaptiveSurfaceFittingScalingFactor(surface_fit_adapt);
      solver.SetAdaptiveSurfaceFittingRelativeChangeThreshold(
         surf_inc_trigger_rel_thresold);
      solver.SetTerminationWithMaxSurfaceFittingError(solver_rtol*surf_fit_err_max);
      if (surface_fit_adapt == 1.0)
      {
         solver.SetMaxNumberofIncrementsForAdaptiveFitting(std::abs(solver_iter));
      }
   }
   if (surface_fit_threshold > 0)
   {
      solver.SetTerminationWithMaxSurfaceFittingError(surface_fit_threshold);
   }
   solver.SetIntegrationRules(*irules, quad_order);
   if (solver_type == 0)
   {
      // Specify linear solver when we use a Newton-based solver.
      solver.SetPreconditioner(*S);
   }
   if (surf_fit_min_det_threshold > 0)
   {
      solver.SetMinimumDeterminantThreshold(surf_fit_min_det_threshold*tauval);
   }
   solver.SetMaximumFittingWeightLimit(surf_fit_const_max);
   if (worst_skew > 0.0)
   {
      solver.SetWorstSkewnessLimit(worst_skew*M_PI/180.0);
   }

   StopWatch TimeSolver;
   TimeSolver.Clear();

   double solvertime = TimeSolver.RealTime(),
          vectortime = solver.GetAssembleElementVectorTime(),
          gradtime   = solver.GetAssembleElementGradTime(),
          prectime   = solver.GetPrecMultTime(),
          processnewstatetime = solver.GetProcessNewStateTime(),
          scalefactortime = solver.GetComputeScalingTime();
   int NewtonIters = 0;
   int PrecIters = 0;

   if (solver_iter < 0)
   {
      for (int iter = 0; iter < -solver_iter; iter++)
      {
         solver.SetMaxIter(1);
         //           solver.SetRelTol(solver_rtol);
         solver.SetAbsTol(solver_rtol);
         solver.SetAbsTol(0.0);
         if (solver_art_type > 0)
         {
            solver.SetAdaptiveLinRtol(solver_art_type, 0.5, 0.9);
         }
         solver.SetPrintLevel(verbosity_level >= 1 ? 1 : -1);
         solver.SetOperator(a);
         TimeSolver.Start();
         solver.Mult(b, psub_x->GetTrueVector());
         TimeSolver.Stop();
         psub_x->SetFromTrueVector();

         solvertime += TimeSolver.RealTime();
         vectortime += solver.GetAssembleElementVectorTime(),
                       gradtime   += solver.GetAssembleElementGradTime(),
                                     prectime   += solver.GetPrecMultTime(),
                                                   processnewstatetime += solver.GetProcessNewStateTime(),
                                                                          scalefactortime += solver.GetComputeScalingTime();
         NewtonIters += 1;
         PrecIters += solver.GetTotalNumberOfLinearIterations();


         {
            tauval = GetMinDet(psubmesh, psub_pfespace, irules, quad_order);
            if (myid == 0)
            { cout << "Minimum det(J) of the mesh is " << tauval << endl; }
         }

         if (surface_fit_const > 0.0)
         {
            tmop_integ->ReMapSurfaceFittingLevelSet(psub_surf_fit_gf0);
            tmop_integ->ComputePointWiseJacobian(*psub_x, *jacobian_current, 1);
            for (int i = 0; i < jacobian_ratio.Size(); i++)
            {
               jacobian_ratio(i) = (*jacobian_current)(i)/(*jacobian_init)(i);
            }
         }

         if (deactivation_layers > 0)
         {
            auto tmap1 = ParSubMesh::CreateTransferMap(*psub_x, x);
            tmap1.Transfer(*psub_x, x);

            auto tmap2 = ParSubMesh::CreateTransferMap(psub_surf_fit_gf0, surf_fit_gf0);
            tmap2.Transfer(psub_surf_fit_gf0, surf_fit_gf0);
         }

         if (surface_fit_const > 0.0)
         {
            DataCollection *dc = NULL;
            dc = new VisItDataCollection("Optimized_"+std::to_string(jobid), pmesh);
            dc->RegisterField("mat", &mat);
            dc->RegisterField("level-set", &surf_fit_gf0);
            dc->RegisterField("Marker", &surf_fit_mat_gf);
            dc->RegisterField("gradratio", &jacobian_ratio);
            dc->SetCycle(iter+1);
            dc->SetTime(iter+1);
            dc->Save();
            delete dc;
         }
      }
   }
   else
   {
      solver.SetMaxIter(solver_iter);
      solver.SetRelTol(solver_rtol);
      solver.SetAbsTol(0.0);
      if (solver_art_type > 0)
      {
         solver.SetAdaptiveLinRtol(solver_art_type, 0.5, 0.9);
      }
      solver.SetPrintLevel(verbosity_level >= 1 ? 1 : -1);
      solver.SetOperator(a);
      TimeSolver.Start();
      solver.Mult(b, psub_x->GetTrueVector());
      TimeSolver.Stop();
      psub_x->SetFromTrueVector();

      solvertime = TimeSolver.RealTime();
      vectortime = solver.GetAssembleElementVectorTime(),
      gradtime   = solver.GetAssembleElementGradTime(),
      prectime   = solver.GetPrecMultTime(),
      processnewstatetime = solver.GetProcessNewStateTime(),
      scalefactortime = solver.GetComputeScalingTime();
      NewtonIters = solver.GetNumIterations();
      PrecIters = solver.GetTotalNumberOfLinearIterations();

      if (surface_fit_const > 0.0)
      {
         tmop_integ->ReMapSurfaceFittingLevelSet(psub_surf_fit_gf0);
      }
      {
         tauval = GetMinDet(psubmesh, psub_pfespace, irules, quad_order);
         if (myid == 0)
         { cout << "Minimum det(J) of the optimized mesh is " << tauval << endl; }
      }
      {
         skewval = GetWorstSkewness(psubmesh, psub_pfespace, irules, quad_order);
         if (myid == 0)
         { cout << "Worst skewness of the optimized mesh is " << skewval*180.0/M_PI << endl; }
      }
      if (deactivation_layers > 0)
      {
         auto tmap1 = ParSubMesh::CreateTransferMap(*psub_x, x);
         tmap1.Transfer(*psub_x, x);

         auto tmap2 = ParSubMesh::CreateTransferMap(psub_surf_fit_gf0, surf_fit_gf0);
         tmap2.Transfer(psub_surf_fit_gf0, surf_fit_gf0);
      }
   }

   if (myid == 0)
   {
      std::cout << solver.GetNumIterations() << " " <<
                solver.GetTotalNumberOfLinearIterations() << " " <<
                " k10iterinfo\n";
   }

   // 16. Save the optimized mesh to a file. This output can be viewed later
   //     using GLVis: "glvis -m optimized -np num_mpi_tasks".
   {
      ostringstream mesh_name;
      mesh_name << "optimized.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      if (int_amr_iters == 0)
      {
         //  pmesh->PrintAsSerial(mesh_ofs);
      }
      else
      {
         //  pmesh->PrintAsOne(mesh_ofs);
      }
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

   // 18. Visualize the final mesh and metric values.
   if (visualization)
   {
      char title[] = "Final metric values";
      TargetConstructor target_c2(target_t, MPI_COMM_WORLD);
      target_c2.SetNodes(x);
      vis_tmop_metric_p(mesh_poly_deg, *metric, target_c2, *pmesh, title, 600);
   }

   if (surface_fit_const > 0.0)
   {
      tmop_integ->GetSurfaceFittingErrors(surf_fit_err_avg, surf_fit_err_max);
      if (myid == 0)
      {
         std::cout << "Avg fitting error: " << surf_fit_err_avg << std::endl
                   << "Max fitting error: " << surf_fit_err_max << std::endl;
         std::cout << "Last active surface fitting constant: " <<
                   tmop_integ->GetLastActiveSurfaceFittingWeight() <<
                   std::endl;
      }
   }

   if (surface_fit_const > 0.0)
   {
      tot_final_integ_error = ComputeIntegrateErroronInterfaces(psubmesh,
                                                                ls_coeff,
                                                                fitting_face_list,
                                                                surf_fit_bg_gf0,
                                                                remap_from_bg,
                                                                integrated_face_error, 0);
   }


   if (myid == 0)
   {
      std::cout << "Monitoring info      :" << endl
                << "Job-id               :" << jobid << endl
                << "Number of elements   :" << neglob << endl
                << "Number of procs      :" << num_procs << endl
                << "Polynomial degree    :" << mesh_poly_deg << endl
                << "Metric          :" << metric_id << endl
                << "Target          :" << target_id << endl
                << "Total NDofs          :" << ndofsglob << endl
                << "Total TDofs          :" << ntdofsglob << endl
                << std::setprecision(4)
                << "Total Iterations     :" << NewtonIters << endl
                << "Total Prec Iterations:" << PrecIters << endl
                << "Total Solver Time (%):" << solvertime << " "
                << (solvertime*100/solvertime) << endl
                << "Assemble Vector Time :" << vectortime << " "
                << (vectortime*100/solvertime) << endl
                << "Assemble Grad Time   :" << gradtime << " "
                << gradtime*100/solvertime <<  endl
                << "Prec Solve Time      :" << prectime << " "
                << prectime*100/solvertime <<  endl
                << "ProcessNewState Time :" << processnewstatetime << " "
                << (processnewstatetime*100/solvertime) <<  endl
                << "ComputeScale Time    :" << scalefactortime << " "
                << (scalefactortime*100/solvertime) <<  "  " << endl
                << "Initial mindet: " << init_min_det << endl
                << "Final mindet: " << tauval << endl
                << "Initial skew: " << init_skew*180.0/M_PI << endl
                << "Final skew: " << skewval*180.0/M_PI << endl
                << "Initial energy: " << init_energy << endl
                << "Final energy: " << fin_energy << endl
                << "Initial metric energy: " << init_metric_energy << endl
                << "Final metric energy: " << fin_metric_energy << endl
                << "Initial avg fitting error: " << init_surf_fit_err_avg << endl
                << "Initial max fitting error: " << init_surf_fit_err_max << endl
                << "Final avg fitting error: " << surf_fit_err_avg << endl
                << "Final max fitting error: " << surf_fit_err_max << endl
                << "Initial fitting weight: " << surface_fit_const << endl
                << "Final fitting weight: " << tmop_integ->GetLastActiveSurfaceFittingWeight()
                << endl
                << "Surface fit scaling factor: " << surface_fit_adapt << endl
                << "Surface fit termination threshold: " << surface_fit_threshold << endl
                << "Surface fit inc rel threshold: " << surf_inc_trigger_rel_thresold << endl
                << "Surface fit max weight: " <<surf_fit_const_max << endl
                << "surf fit min det threshold: " << surf_fit_min_det_threshold << endl
                << "Submesh layers: " << deactivation_layers << endl
                << "Submesh NE: " << num_active_glob << endl
                << "Submesh NP: " << num_procs_submesh << endl
                << "Submesh NDOFs: " << ndofsglobsubmesh << endl
                << "Submesh NTDOFs: " << ntdofsglobsubmesh << endl
                << "AMRIter: " << int_amr_iters << endl
                << "NE Pre AMRIter: " << neglob_preamr << endl
                << "InitIntegError: " << tot_init_integ_error << endl
                << "FinIntegError: " << tot_final_integ_error << endl
                << endl;

      std::cout << "TMOPFittingInfo: " <<
                "jobid,ne,np,order,metric,target,ndofs,tdofs,"
                "niter,preciter,totalsolvetime,vectime,gradtime,multtime,procnewstatetime,computescaletime,"
                "initmindet,finalmindet,initskew,finalskew,initenergy,finalenergy,"
                "initmetricenergy,finalmetricenergy,"
                "initavgfiterr,initmaxfiterr,finavgfiterr,finmaxfiterr,"
                "initfitwt,finalfitwt,sfa,sft,sfct,sfcmax,sfcjac,"
                "sublayer,subne,subnp,subndofs,subtdofs,amriter,nepreiter,"
                "InitIntegError,FinalIntegError " <<
                jobid << "," << neglob << "," << num_procs << "," <<
                mesh_poly_deg << "," << metric_id << "," << target_id << "," <<
                ndofsglob << "," << ntdofsglob << "," <<
                NewtonIters << "," << PrecIters << "," <<
                solvertime << "," << vectortime << "," << gradtime << "," <<
                prectime << "," << processnewstatetime << "," << scalefactortime << "," <<
                init_min_det << "," << tauval << "," <<
                init_skew*180.0/M_PI << "," << skewval*180.0/M_PI << "," <<
                init_energy << "," << fin_energy << "," <<
                init_metric_energy << "," << fin_metric_energy << "," <<
                init_surf_fit_err_avg << "," << init_surf_fit_err_max << "," <<
                surf_fit_err_avg << "," << surf_fit_err_max << "," <<
                surface_fit_const << "," << tmop_integ->GetLastActiveSurfaceFittingWeight() <<
                "," <<
                surface_fit_adapt << "," << surface_fit_threshold << "," <<
                surf_inc_trigger_rel_thresold << "," << surf_fit_const_max << "," <<
                surf_fit_min_det_threshold << "," << deactivation_layers << "," <<
                num_active_glob << "," << num_procs_submesh << "," <<
                ndofsglobsubmesh << "," << ntdofsglobsubmesh << "," <<
                int_amr_iters << "," << neglob_preamr << "," <<
                tot_init_integ_error << "," << tot_final_integ_error <<
                std::endl;
   }

   GridFunction dx(x);
   dx -= x0;

   if (solver_iter > 0 && surface_fit_const > 0.0)
   {
      tmop_integ->ComputePointWiseJacobian(*psub_x, *jacobian_current, 1);
      for (int i = 0; i < jacobian_ratio.Size(); i++)
      {
         jacobian_ratio(i) = (*jacobian_current)(i)/(*jacobian_init)(i);
      }

      {
         ParGridFunction pmesh_surf_fit_mat_gf(&surf_fit_fes);
         ParGridFunction pmesh_jacobian_ratio(&surf_fit_fes);
         if (deactivation_layers > 0)
         {
            auto tmap1 = ParSubMesh::CreateTransferMap(surf_fit_mat_gf,
                                                       pmesh_surf_fit_mat_gf);
            tmap1.Transfer(surf_fit_mat_gf, pmesh_surf_fit_mat_gf);
            tmap1.Transfer(jacobian_ratio, pmesh_jacobian_ratio);
         }
         else
         {
            pmesh_surf_fit_mat_gf = surf_fit_mat_gf;
            pmesh_jacobian_ratio = jacobian_ratio;
         }

         DataCollection *dc = NULL;
         dc = new VisItDataCollection("Optimized_"+std::to_string(jobid), pmesh);
         dc->RegisterField("mat", &mat);
         dc->RegisterField("level-set", &surf_fit_gf0);
         dc->RegisterField("Marker", &pmesh_surf_fit_mat_gf);
         dc->RegisterField("gradratio", &pmesh_jacobian_ratio);
         dc->RegisterField("Displacement", &dx);
         dc->SetCycle(0);
         dc->SetTime(0.0);
         dc->Save();
         delete dc;
      }
   }
   else
   {
      DataCollection *dc = NULL;
      dc = new VisItDataCollection("Optimized_"+std::to_string(jobid), pmesh);
      dc->SetCycle(1);
      dc->SetTime(1.0);
      dc->Save();
      delete dc;
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

   // 20. Free the used memory.
   delete S;
   delete S_prec;
   delete metric_coeff1;
   delete adapt_surface;
   delete adapt_grad_surface;
   delete adapt_hess_surface;
   delete ls_coeff;
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
   if (psubmesh == pmesh || psubmesh->GetGlobalNE() == pmesh->GetGlobalNE())
   {
      delete pmesh;
   }
   else
   {
      delete psubmesh;
      delete pmesh;
   }

   return 0;
}
