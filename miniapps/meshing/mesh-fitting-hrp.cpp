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
//            --------------------------------------------------
//            Mesh Optimizer Miniapp: Optimize high-order meshes
//            --------------------------------------------------
//
// This miniapp performs mesh optimization using the Target-Matrix Optimization
// Paradigm (TMOP) by P.Knupp et al., and a global variational minimization
// approach. It minimizes the quantity sum_T int_T mu(J(x)), where T are the
// target (ideal) elements, J is the Jacobian of the transformation from the
// target to the physical element, and mu is the mesh quality metric. This
// metric can measure shape, size or alignment of the region around each
// quadrature point. The combination of targets & quality metrics is used to
// optimize the physical node positions, i.e., they must be as close as possible
// to the shape / size / alignment of their targets. This code also demonstrates
// a possible use of nonlinear operators (the class TMOP_QualityMetric, defining
// mu(J), and the class TMOP_Integrator, defining int mu(J)), as well as their
// coupling to Newton methods for solving minimization problems. Note that the
// utilized Newton methods are oriented towards avoiding invalid meshes with
// negative Jacobian determinants. Each Newton step requires the inversion of a
// Jacobian matrix, which is done through an inner linear solver.
//
// Compile with: make mesh-fitting
//
// make mesh-fitting-hrp -j && ./mesh-fitting-hrp -m square01.mesh -rs 1 -o 1 -oi 2 -sbgmesh -vl 2 -mo 4 -mi 3 -preft 5e-14 -lsf 1 -vis -bgamr 2 -det 2 -pderef -1e-4 -href -no-ro -ni 10
// make mesh-fitting-hrp -j && ./mesh-fitting-hrp -m square01-tri.mesh -rs 2 -o 1 -oi 2 -sbgmesh -vl 2 -mo 6 -mi 3 -preft 5e-14 -lsf 0 -bgamr 4 -bgo 6 -det 2 -pderef -1e-4 -href -no-ro -ni 100 -mid 2 -sft 1e-20 -et 0

// make mesh-fitting-hrp -j && ./mesh-fitting-hrp -m square01-tri.mesh -rs 1 -o 1 -oi 2 -sbgmesh -vl 2 -mo 3 -mi 4 -preft 5e-14 -lsf 0 -bgamr 4 -bgo 6 -det 2 -pderef -1e-4 -href -no-ro -ni 100 -mid 2 -sft 1e-20 -et 0
#include "../../mfem.hpp"
#include "../common/mfem-common.hpp"
#include <fstream>
#include <iostream>
#include <chrono>
#include "mesh-fitting.hpp"
#include "mesh-fitting-pref.hpp"

using namespace mfem;
using namespace std;

struct InterfaceFaceEdgeConnectivity
{
   Array<int> inter_faces; //holds face element index for interface
   Array<int> inter_face_el1, inter_face_el2; //holds indices of adjacent els
   Array<int> intfdofs; // holds indices of all interface dofs
   Array<int> inter_face_el_all; // union of inter_face_el1 and el2.
   std::map<int, vector<int>> face_to_els;
   std::map<int, vector<int>> els_to_face;
   std::map<int, vector<int>> group_to_els;
   std::map<int, int> els_to_group;
   std::map<int, vector<int>> group_to_face;
   std::map<int, int> face_to_group;
   std::map<int, vector<int>> group_to_edg_els;
   // Some 3D info for elements whose edges are on the interface
   Array<Array<int> *> edge_to_el;
   Array<Array<int> *> el_to_el_edge;
   Array<int> intedges;
   Array<int> intedgels;
   // Error for each face
   std::map<int, double> face_error_map;
   int ngroups;
   int nfaces;
};

map<int, double> MapErrorToFaces(Array<int> &inter_faces, Vector &face_error)
{
   map<int, double> face_error_map;
   for (int i = 0; i < inter_faces.Size(); i++)
   {
      face_error_map[inter_faces[i]] = face_error(i);
   }
   return face_error_map;
}

void SetGroupNumGf(InterfaceFaceEdgeConnectivity &ifec,
                   GridFunction &groupnum)
{
   for (int i = 0; i < groupnum.FESpace()->GetMesh()->GetNE(); i++)
   {
      if (ifec.els_to_group.find(i) == ifec.els_to_group.end())
      {
         groupnum(i) = -1;
         continue;
      }
      int group = ifec.els_to_group[i];
      groupnum(i) = group;
   }
}

void GetGroupInfo(Mesh *mesh, GridFunction &mat,
                  GridFunction &surf_fit_gf0,
                  InterfaceFaceEdgeConnectivity &ifec)
{
   ifec.group_to_edg_els.clear();
   ifec.face_error_map.clear();

   int ngroups = GetMaterialInterfaceEntities2(mesh, mat, surf_fit_gf0,
                                               ifec.inter_faces,
                                               ifec.intfdofs, ifec.inter_face_el1, ifec.inter_face_el2,
                                               ifec.inter_face_el_all, ifec.face_to_els, ifec.els_to_face,
                                               ifec.group_to_els, ifec.els_to_group, ifec.group_to_face,
                                               ifec.face_to_group);

   const int dim = mesh->Dimension();

   if (dim == 3)
   {
      ifec.edge_to_el.SetSize(mesh->GetNEdges());
      ifec.el_to_el_edge.SetSize(mesh->GetNE());

      for (int i = 0; i < ifec.edge_to_el.Size(); i++)
      {
         Array<int> *temp = new Array<int>;
         ifec.edge_to_el[i] = temp;
      }
      for (int e = 0; e < mesh->GetNE(); e++)
      {
         Array<int> edges, cors;
         mesh->GetElementEdges(e, edges, cors);
         for (int ee = 0; ee < edges.Size(); ee++)
         {
            int edgenum = edges[ee];
            ifec.edge_to_el[edgenum]->Append(e);
         }
      }

      for (int e = 0; e < ifec.el_to_el_edge.Size(); e++)
      {
         ifec.el_to_el_edge[e] = NULL;
      }

      GetMaterialInterfaceEdgeDofs(mesh, mat, surf_fit_gf0,
                                   ifec.edge_to_el,
                                   ifec.inter_face_el_all,
                                   ifec.intfdofs,
                                   ifec.intedges,
                                   ifec.intedgels,
                                   ifec.el_to_el_edge);

      // Verify all the elements that an edge-element depends on are part of the same
      // group and add it to that group
      for (int e = 0; e < ifec.el_to_el_edge.Size(); e++)
      {
         int elnum = e;
         Array<int> *eldeps = ifec.el_to_el_edge[e];
         if (!eldeps)
         {
            continue;
         }

         int el0 = (*eldeps)[0];
         int group0 = ifec.els_to_group[el0];
         // add element to the group
         ifec.els_to_group[elnum] = group0;
         ifec.group_to_edg_els[group0].push_back(elnum);

         int nomatch = 0;
         for (int i = 1; i < eldeps->Size(); i++)
         {
            int eldep = (*eldeps)[i];
            int group = ifec.els_to_group[eldep];
            if (!nomatch)
            {
               nomatch = group != group0;
            }
         }
         MFEM_VERIFY(nomatch == 0, "Inconsistent group for edge elements");
      }
   }

   ifec.ngroups = ngroups;
   ifec.nfaces = ifec.inter_faces.Size();
}

void GetGroupError(Mesh *mesh,
                   InterfaceFaceEdgeConnectivity &ifec,
                   Vector &current_group_max_error,
                   Vector &current_group_error,
                   Vector &current_group_dofs,
                   Array<Array<int> *> *group_sibling_dofs = nullptr)
{
   int ngroups = ifec.ngroups;
   const FiniteElementSpace *fespace = mesh->GetNodalFESpace();
   current_group_max_error.SetSize(ngroups);
   current_group_max_error = 0.0;

   current_group_error.SetSize(ngroups);
   current_group_error = 0.0;

   current_group_dofs.SetSize(ngroups);
   current_group_dofs = 0.0;

   // For meshes that are h-refined, we need to get DOFs from all the siblings.
   NCMesh *ncm = mesh->ncmesh;
   if (group_sibling_dofs)
   {
      MFEM_VERIFY(ncm, "This mesh is not an NCMesh. Cannot get sibling dofs.");
      //first delete all internal arrays
      for (int i = 0; i < group_sibling_dofs->Size(); i++)
      {
         delete (*group_sibling_dofs)[i];
      }

      group_sibling_dofs->SetSize(0);
      for (int i = 0; i < ngroups; i++)
      {
         group_sibling_dofs->Append(new Array<int>);
      }
   }

   for (int i = 0; i < ngroups; i++)
   {
      int groupnum = i;
      vector<int> faces = ifec.group_to_face[groupnum];
      Array<int> temp;
      for (int f = 0; f < faces.size(); f++)
      {
         int facenum = faces[f];
         double error_bg_face = ifec.face_error_map[facenum];
         current_group_error[groupnum] += error_bg_face;
         current_group_max_error[groupnum] =
            std::max(current_group_max_error[groupnum], error_bg_face);
      }

      vector<int> groupels = ifec.group_to_els[groupnum];
      Array<int> dofs;
      for (int e = 0; e < groupels.size(); e++)
      {
         int elnum = groupels[e];
         // std::cout << i << " " << elnum << " k10-group-elnum\n";
         if (!group_sibling_dofs)
         {
            fespace->GetElementVDofs(elnum, dofs);
            temp.Append(dofs);
         }
         else
         {
            Array<int> siblings = ncm->GetElementSiblings(elnum);
            for (int k = 0; k < siblings.Size(); k++)
            {
               fespace->GetElementVDofs(siblings[k], dofs);
               temp.Append(dofs);
            }
         }
      }
      for (int e = 0; e < ifec.group_to_edg_els[groupnum].size(); e++)
      {
         int elnum = ifec.group_to_edg_els[groupnum][e];
         if (!group_sibling_dofs)
         {
            fespace->GetElementVDofs(elnum, dofs);
            temp.Append(dofs);
         }
         else
         {
            Array<int> siblings = ncm->GetElementSiblings(elnum);
            for (int k = 0; k < siblings.Size(); k++)
            {
               fespace->GetElementVDofs(siblings[k], dofs);
               temp.Append(dofs);
            }
         }
      }

      temp.Sort();
      temp.Unique();
      current_group_dofs[groupnum] += temp.Size();
      if (group_sibling_dofs)
      {
         ((*group_sibling_dofs)[groupnum])->Append(temp);
      }
   }

   for (int i = 0; i < ngroups; i++)
   {
      if (group_sibling_dofs)
      {
         (*group_sibling_dofs)[i]->Sort();
         (*group_sibling_dofs)[i]->Unique();
         current_group_dofs[i] = (*group_sibling_dofs)[i]->Size();
      }
   }
}


// Invert coarse_to_fine table to get fine_to_coarse table
Table MakeFineToCoarseTable(Table &coarse_to_fine)
{
   Array<Connection> list;
   for (int i = 0; i < coarse_to_fine.Size(); i++)
   {
      const int *row = coarse_to_fine.GetRow(i);
      for (int j = 0; j < coarse_to_fine.RowSize(i); j++)
      {
         list.Append(Connection(row[j], i));
      }
   }

   Table fine_to_coarse;
   fine_to_coarse.MakeFromList(list.Size(), list);
   return fine_to_coarse;
}

int main(int argc, char *argv[])
{
   // 0. Set the method's default parameters.
   const char *mesh_file = "icf.mesh";
   int mesh_poly_deg     = 1;
   int rs_levels         = 0;
   double jitter         = 0.0;
   int metric_id         = 2;
   int target_id         = 1;
   double surface_fit_const = 0.1;
   int quad_order        = 8;
   int solver_type       = 0;
   int solver_iter       = 200;
   double solver_rtol    = 1e-10;
   int lin_solver        = 2;
   int max_lin_iter      = 100;
   bool move_bnd         = true;
   bool visualization    = true;
   int verbosity_level   = 2;
   double surface_fit_adapt = 10.0;
   double surface_fit_threshold = 1e-14;
   int mesh_node_ordering = 0;
   bool prefine            = true;
   int pref_order_increase = 2;
   int pref_max_order      = 4;
   int pref_max_iter       = 2;
   double pref_tol         = 1e-13;
   bool surf_bg_mesh       = true;
   bool reduce_order       = true;
   const char *bg_mesh_file = "NULL";
   const char *bg_ls_file = "NULL";
   bool custom_material   = true;
   const char *custom_material_file = "NULL"; //material coming from a gf file
   bool adapt_marking = false;
   int bg_amr_iter = 0;
   int ls_function = 0;
   int bg_rs_levels = 1;
   int jobid  = 0;
   bool visit    = true;
   int adjeldiff = 1;
   double pderef = 0.01;
   int bgo = 4;
   int custom_split_mesh = 0;
   bool mod_bndr_attr    = false;
   int exceptions    = 0; //to do some special stuff.
   int error_type = 0;
   int deref_error_type = 0;
   bool hrefine = false;
   int href_max_depth = 4;
   bool relaxed_hp = false;

   // the error types for ref and deref are: 0 - L2squared, 1 - length, 2 - L2, 3 - L2/length

   // 1. Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&bg_rs_levels, "-bgrs", "--bg-refine-serial",
                  "Number of times to refine the background mesh uniformly in serial.");
   args.AddOption(&jitter, "-ji", "--jitter",
                  "Random perturbation scaling factor.");
   args.AddOption(&metric_id, "-mid", "--metric-id",
                  "Mesh optimization metric. See list in mesh-optimizer.");
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
   args.AddOption(&surface_fit_adapt, "-sfa", "--adaptive-surface-fit",
                  "Enable or disable adaptive surface fitting.");
   args.AddOption(&surface_fit_threshold, "-sft", "--surf-fit-threshold",
                  "Set threshold for surface fitting. TMOP solver will"
                  "terminate when max surface fitting error is below this limit");
   args.AddOption(&mesh_node_ordering, "-mno", "--mesh_node_ordering",
                  "Ordering of mesh nodes."
                  "0 (default): byNodes, 1: byVDIM");
   args.AddOption(&prefine, "-pref", "--pref", "-no-pref",
                  "--no-pref",
                  "Randomly p-refine the mesh.");
   args.AddOption(&pref_order_increase, "-oi", "--preforderincrease",
                  "How much polynomial order to increase for p-refinement.");
   args.AddOption(&pref_max_order, "-mo", "--prefmaxorder",
                  "Maximum polynomial order for p-refinement.");
   args.AddOption(&pref_max_iter, "-mi", "--prefmaxiter",
                  "Maximum number of iteration");
   args.AddOption(&pref_tol, "-preft", "--preftol",
                  "Error tolerance on a face.");
   args.AddOption(&pderef, "-pderef", "--pderef",
                  "Tolerance for derefinement set as pderef*pref_tol.");
   args.AddOption(&surf_bg_mesh, "-sbgmesh", "--surf-bg-mesh",
                  "-no-sbgmesh","--no-surf-bg-mesh",
                  "Use background mesh for surface fitting.");
   args.AddOption(&reduce_order, "-ro", "--reduce-order",
                  "-no-ro","--no-reduce-order",
                  "Reduce the order of elements around the interface.");
   args.AddOption(&bg_mesh_file, "-bgm", "--bgm",
                  "Background Mesh file to use.");
   args.AddOption(&bg_ls_file, "-bgls", "--bgls",
                  "Background level set gridfunction file to use.");
   args.AddOption(&custom_material, "-cus-mat", "--custom-material",
                  "-no-cus-mat", "--no-custom-material",
                  "When true, sets the material based on predetermined logic instead of level-set");
   args.AddOption(&custom_material_file, "-cmf", "--cmf",
                  "0 order L2 Gridfunction to specify material.");
   args.AddOption(&adapt_marking, "-marking", "--adaptive-marking", "-no-marking",
                  "--no-adaptive-marking",
                  "Enable or disable adaptive marking surface fitting.");
   args.AddOption(&bg_amr_iter, "-bgamr", "--bgamr",
                  "Number of times to AMR refine the background mesh.");
   args.AddOption(&ls_function, "-lsf", "--ls-function",
                  "Choice of level set function.");
   args.AddOption(&jobid, "-jid", "--jid",
                  "job id used for visit  save files");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit",
                  "--no-visit",
                  "Enable or disable VISIT output");
   args.AddOption(&adjeldiff, "-diff", "--diff",
                  "Difference in p of adjacent elements.");
   args.AddOption(&bgo, "-bgo", "--bg-order",
                  "Polynomial degree of mesh finite element space on bg mesh.");
   args.AddOption(&custom_split_mesh, "-ctot", "--custom_split_mesh",
                  "Split Mesh Into Tets/Tris/Quads for consistent materials");
   args.AddOption(&mod_bndr_attr, "-mod-bndr-attr", "--modify-boundary-attribute",
                  "-fix-bndr-attr", "--fix-boundary-attribute",
                  "Change boundary attribue based on alignment with Cartesian axes.");
   args.AddOption(&exceptions, "-exc", "--exc",
                  "Do some special things for some cases.");
   args.AddOption(&error_type, "-et", "--error_type",
                  "Error type.");
   args.AddOption(&deref_error_type, "-det", "--deref_error_type",
                  "derefinement Error type.");
   args.AddOption(&hrefine, "-href", "--h-ref",
                  "-no-href", "--no-h-ref",
                  "Do h-ref.");
   args.AddOption(&href_max_depth, "-md", "--hrefmaxdepth",
                  "Maximum number of hrefinement.");
   args.AddOption(&relaxed_hp, "-relax", "--relax-hp",
                  "-no-relax", "--no-relax-hp",
                  "Do relaxed hp.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);
   if (pref_max_order < 0)
   {
      pref_max_order = mesh_poly_deg+pref_order_increase;
   }

   const char *vis_keys = "Rjaamc";
   Array<Mesh *> surf_el_meshes = SetupSurfaceMeshes();

   FunctionCoefficient ls_coeff(circle_level_set);
   if (ls_function==1)
   {
      ls_coeff = FunctionCoefficient(squircle_level_set);
   }
   else if (ls_function==2)
   {
      ls_coeff = FunctionCoefficient(apollo_level_set);
   }
   else if (ls_function==3)
   {
      ls_coeff = FunctionCoefficient(csg_cubecylsph_smooth);
   }

   // 2. Initialize and refine the starting mesh.
   Mesh *mesh = NULL;
   if (custom_split_mesh == 0)
   {
      mesh = new Mesh(mesh_file, 1, 1, false);
      for (int lev = 0; lev < rs_levels; lev++)
      {
         mesh->UniformRefinement();
      }
   }
   else if (custom_split_mesh > 0)
   {
      int res = std::max(1, rs_levels);
      //SPLIT TYPE == 1 - 12 TETS, 2 = 24 TETS
      mesh = new Mesh(Mesh::MakeCartesian3DWith24TetsPerHex(2*res,2*res,2*res,
                                                            1.0, 1.0, 1.0)); //24tet
   }
   else
   {
      int res = std::max(1, rs_levels);
      //SPLIT TYPE == -1 => 1 quad to 4 tris
      if (custom_split_mesh == -1)
      {
         if (exceptions == 4)
         {
            mesh = new Mesh(Mesh::MakeCartesian2DWith4TrisPerQuad(6, 5, 1.2, 1.0));
            for (int lev = 0; lev < rs_levels; lev++)
            {
               mesh->UniformRefinement();
            }
         }
         else
         {
            mesh = new Mesh(Mesh::MakeCartesian2DWith4TrisPerQuad(2*res,2*res, 1.0, 1.0));
         }
      }
      else   //1 quad to 5 quads
      {
         mesh = new Mesh(Mesh::MakeCartesian2DWith5QuadsPerQuad(2*res,2*res, 1.0, 1.0));
      }
   }

   const int dim = mesh->Dimension();
   std::cout << "Number of elements: " << mesh->GetNE() << std::endl;
   if (prefine || hrefine) { mesh->EnsureNCMesh(true); }
   FindPointsGSLIB finder;

   // Setup background mesh for surface fitting.
   // If the user has specified a mesh name, use that.
   // Otherwise use mesh to be morphed and refine it.
   Mesh *mesh_surf_fit_bg = NULL;
   if (surf_bg_mesh)
   {
      if (strcmp(bg_mesh_file, "NULL") != 0) //user specified background mesh
      {
         mesh_surf_fit_bg = new Mesh(bg_mesh_file, 1, 1, false);
      }
      else
      {
         if (dim == 2)
         {
            mesh_surf_fit_bg = new Mesh(Mesh::MakeCartesian2D(4, 4,
                                                              Element::QUADRILATERAL,
                                                              true));
         }
         else if (dim == 3)
         {
            mesh_surf_fit_bg = new Mesh(Mesh::MakeCartesian3D(4, 4, 4,
                                                              Element::HEXAHEDRON));

         }
         if (bg_amr_iter == 0)
         {
            for (int ref = 0; ref < bg_rs_levels; ref++)
            {
               mesh_surf_fit_bg->UniformRefinement();
            }
         }
      }
      GridFunction *mesh_surf_fit_bg_nodes = mesh_surf_fit_bg->GetNodes();
      if (mesh_surf_fit_bg_nodes == NULL)
      {
         std::cout << "Background mesh does not have nodes. Setting curvature\n";
         mesh_surf_fit_bg->SetCurvature(1, 0, -1, 0);
      }
      else
      {
         const TensorBasisElement *tbe =
            dynamic_cast<const TensorBasisElement *>
            (mesh_surf_fit_bg_nodes->FESpace()->GetFE(0));
         int order = mesh_surf_fit_bg_nodes->FESpace()->GetFE(0)->GetOrder();
         if (tbe == NULL)
         {
            std::cout << "Background mesh does not have tensor basis nodes. "
                      "Setting tensor basis\n";
            mesh_surf_fit_bg->SetCurvature(order, 0, -1, 0);
         }
      }
      finder.Setup(*mesh_surf_fit_bg);
   }
   else
   {
      MFEM_ABORT("p-adaptivity is not supported without background mesh");
   }

   HRefUpdater hrefup = HRefUpdater();

   // Define a finite element space on the mesh-> Here we use vector finite
   // elements.
   MFEM_VERIFY(mesh_poly_deg >= 1,"Mesh order should at-least be 1.");
   // Use an H1 space for mesh nodes
   FiniteElementCollection *fec = new H1_FECollection(mesh_poly_deg, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec, dim,
                                                        mesh_node_ordering);

   // Define L2 space for storing the order of elements (piecewise constant).
   L2_FECollection l2zero_coll(0, dim);
   FiniteElementSpace l2zero_fes(mesh, &l2zero_coll);
   GridFunction order_gf(&l2zero_fes);
   GridFunction int_marker(&l2zero_fes);
   GridFunction done_els(&l2zero_fes);
   GridFunction done_elsp(&l2zero_fes);
   GridFunction done_elsh(&l2zero_fes);
   GridFunction mat(&l2zero_fes);
   GridFunction NumFaces(&l2zero_fes);
   GridFunction fitting_error_gf(&l2zero_fes);
   GridFunction groupnum_gf(&l2zero_fes);

   order_gf = mesh_poly_deg*1.0;
   int_marker = 0.0;
   done_els = 1.0;
   done_elsp = 1.0;
   done_elsh = 1.0;
   mat = 0.0;
   NumFaces = 0.0;
   fitting_error_gf = pref_tol/100;
   groupnum_gf = -1.0;

   hrefup.AddFESpaceForUpdate(&l2zero_fes);
   hrefup.AddGridFunctionForUpdate(&order_gf);
   hrefup.AddGridFunctionForUpdate(&int_marker);
   hrefup.AddGridFunctionForUpdate(&done_els);
   hrefup.AddGridFunctionForUpdate(&done_elsp);
   hrefup.AddGridFunctionForUpdate(&done_elsh);
   hrefup.AddGridFunctionForUpdate(&mat);
   hrefup.AddGridFunctionForUpdate(&NumFaces);
   hrefup.AddGridFunctionForUpdate(&fitting_error_gf);
   hrefup.AddGridFunctionForUpdate(&groupnum_gf);


   fespace->SetRelaxedHpConformity(relaxed_hp);

   // Curve the mesh based on the (optionally p-refined) finite element space.
   GridFunction x(fespace);
   mesh->SetNodalGridFunction(&x);
   hrefup.AddFESpaceForUpdate(fespace);
   hrefup.AddGridFunctionForUpdate(&x);

   if (mod_bndr_attr)
   {
      ModifyBoundaryAttributesForNodeMovement(mesh, x);
      mesh->SetAttributes();
   }

   // Define a gridfunction to save the mesh at maximum order when some of the
   // elements in the mesh are p-refined. We need this for now because some of
   // mfem's output functions do not work for p-refined spaces.
   GridFunction *x_max_order = NULL;
   delete x_max_order;
   x_max_order = ProlongToMaxOrder(&x, 0);
   mesh->SetNodalGridFunction(x_max_order);
   mesh->SetNodalGridFunction(&x);
   delete x_max_order;

   // Define a vector representing the minimal local mesh size in the mesh
   // nodes. We index the nodes using the scalar version of the degrees of
   // freedom in fespace. Note: this is partition-dependent.
   // In addition, compute average mesh size and total volume.
   Vector h0(fespace->GetNDofs());
   h0 = infinity();
   double mesh_volume = 0.0;
   Array<int> dofs;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      // Get the local scalar element degrees of freedom in dofs.
      fespace->GetElementDofs(i, dofs);
      // Adjust the value of h0 in dofs based on the local mesh size.
      const double hi = mesh->GetElementSize(i);
      for (int j = 0; j < dofs.Size(); j++)
      {
         h0(dofs[j]) = min(h0(dofs[j]), hi);
      }
      mesh_volume += mesh->GetElementVolume(i);
   }

   // 8. Add a random perturbation to the nodes in the interior of the domain.
   //    We define a random grid function of fespace and make sure that it is
   //    zero on the boundary and its values are locally of the order of h0.
   //    The latter is based on the DofToVDof() method which maps the scalar to
   //    the vector degrees of freedom in fespace.
   GridFunction rdm(fespace);
   if (jitter != 0.0)
   {
      rdm.Randomize();
      rdm -= 0.25; // Shift to random values in [-0.5,0.5].
      rdm *= jitter;
      rdm.HostReadWrite();
      // Scale the random values to be of order of the local mesh size.
      for (int i = 0; i < fespace->GetNDofs(); i++)
      {
         for (int d = 0; d < dim; d++)
         {
            rdm(fespace->DofToVDof(i,d)) *= h0(i);
         }
      }
      Array<int> vdofs;
      for (int i = 0; i < fespace->GetNBE(); i++)
      {
         // Get the vector degrees of freedom in the boundary element.
         fespace->GetBdrElementVDofs(i, vdofs);
         // Set the boundary values to zero.
         for (int j = 0; j < vdofs.Size(); j++) { rdm(vdofs[j]) = 0.0; }
      }
      x -= rdm;
   }

   // For parallel runs, we define the true-vector. This makes sure the data is
   // consistent across processor boundaries.
   x.SetTrueVector();
   x.SetFromTrueVector();

   // 9. Save the starting (prior to the optimization) mesh to a file. This
   //    output can be viewed later using GLVis: "glvis -m perturbed.mesh".
   if (!prefine)
   {
      ofstream mesh_ofs("perturbed.mesh");
      mesh->Print(mesh_ofs);
   }

   // 10. Store the starting (prior to the optimization) positions.
   GridFunction x0(fespace);
   x0 = x;
   hrefup.AddGridFunctionForUpdate(&x0);

   // 11. Form the integrator that uses the chosen metric and target.
   // First pick a metric
   double min_detJ = -0.1;
   TMOP_QualityMetric *metric = NULL;
   switch (metric_id)
   {
      // T-metrics
      case 1: metric = new TMOP_Metric_001; break; //shape-metric
      case 2: metric = new TMOP_Metric_002; break; //shape-metric
      case 58: metric = new TMOP_Metric_058; break; // shape-metric
      case 80: metric = new TMOP_Metric_080(0.5); break; //shape+size
      case 303: metric = new TMOP_Metric_303; break; //shape
      case 328: metric = new TMOP_Metric_328(); break; //shape+size
      default:
         cout << "Unknown metric_id: " << metric_id << endl;
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

   // Next, select a target.
   TargetConstructor::TargetType target_t;
   TargetConstructor *target_c = NULL;
   switch (target_id)
   {
      case 1: target_t = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE; break;
      case 2: target_t = TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE; break;
      case 3: target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE; break;
      case 4: target_t = TargetConstructor::GIVEN_SHAPE_AND_SIZE; break;
      default: cout << "Unknown target_id: " << target_id << endl; return 3;
   }
   if (target_c == NULL)
   {
      target_c = new TargetConstructor(target_t);
   }
   target_c->SetNodes(x0);

   // Setup the quadrature rules for the TMOP integrator.
   IntegrationRules *irules = &IntRulesLo;

   // Surface fitting.
   H1_FECollection surf_fit_fec(mesh_poly_deg, dim);
   FiniteElementSpace surf_fit_fes(mesh, &surf_fit_fec);
   // Elevate to the same space as mesh for prefinement
   surf_fit_fes.CopySpaceElementOrders(*fespace);
   surf_fit_fes.SetRelaxedHpConformity(fespace->GetRelaxedHpConformity());
   GridFunction surf_fit_mat_gf(&surf_fit_fes);
   GridFunction surf_fit_gf0(&surf_fit_fes);
   Array<bool> surf_fit_marker(surf_fit_gf0.Size());
   ConstantCoefficient surf_fit_coeff(surface_fit_const);
   AdaptivityEvaluator *adapt_surface = NULL;
   AdaptivityEvaluator *adapt_grad_surface = NULL;
   AdaptivityEvaluator *adapt_hess_surface = NULL;

   GridFunction *surf_fit_gf0_max_order = &surf_fit_gf0;
   GridFunction *surf_fit_mat_gf_max_order = &surf_fit_mat_gf;

   // Background mesh FECollection, FESpace, and GridFunction
   FiniteElementCollection *surf_fit_bg_fec = NULL;
   FiniteElementSpace *surf_fit_bg_fes = NULL;
   GridFunction *surf_fit_bg_gf0 = NULL;
   FiniteElementSpace *surf_fit_bg_grad_fes = NULL;
   GridFunction *surf_fit_bg_grad = NULL;
   FiniteElementSpace *surf_fit_bg_hess_fes = NULL;
   GridFunction *surf_fit_bg_hess = NULL;

   hrefup.AddFESpaceForUpdate(&surf_fit_fes);
   hrefup.AddGridFunctionForUpdate(&surf_fit_mat_gf);
   hrefup.AddGridFunctionForUpdate(&surf_fit_gf0);

   // If a background mesh is used, we interpolate the Gradient and Hessian
   // from that mesh to the current mesh being optimized.
   FiniteElementSpace *surf_fit_grad_fes = NULL;
   GridFunction *surf_fit_grad = NULL;
   FiniteElementSpace *surf_fit_hess_fes = NULL;
   GridFunction *surf_fit_hess = NULL;

   if (surf_bg_mesh)
   {
      //if the user specified a gridfunction file, use that
      if (strcmp(bg_ls_file, "NULL") != 0) //user specified background mesh
      {
         ifstream bg_ls_stream(bg_ls_file);
         surf_fit_bg_gf0 = new GridFunction(mesh_surf_fit_bg, bg_ls_stream);
         surf_fit_bg_fes = surf_fit_bg_gf0->FESpace();
         surf_fit_bg_fec = const_cast<FiniteElementCollection *>
                           (surf_fit_bg_fes->FEColl());
         if (exceptions == 1)
         {
            *surf_fit_bg_gf0 -= 0.1; //Apollo
         }
         finder.Interpolate(x, *surf_fit_bg_gf0, surf_fit_gf0,
                            x.FESpace()->GetOrdering());
      }
      else
      {
         // Init the FEC, FES and GridFunction of uniform order = 6
         // for the background ls function
         surf_fit_bg_fec = new H1_FECollection(bgo, dim);
         surf_fit_bg_fes = new FiniteElementSpace(mesh_surf_fit_bg, surf_fit_bg_fec);
         surf_fit_bg_gf0 = new GridFunction(surf_fit_bg_fes);
         surf_fit_bg_gf0->ProjectCoefficient(ls_coeff);
         if (bg_amr_iter > 0)
         {
            std::cout << "Doing AMR on the bg mesh\n";
            OptimizeMeshWithAMRAroundZeroLevelSet(*mesh_surf_fit_bg,
                                                  ls_coeff,
                                                  bg_amr_iter,
                                                  *surf_fit_bg_gf0);
            std::cout << "Done AMR on the bg mesh\n";
         }
         finder.Setup(*mesh_surf_fit_bg);
         std::cout << "Done finder Setup on the bg mesh\n";
         surf_fit_bg_gf0->ProjectCoefficient(ls_coeff);
      }

      if (visit)
      {
         DataCollection *dc = NULL;
         dc = new VisItDataCollection("Background_"+std::to_string(jobid),
                                      mesh_surf_fit_bg);
         dc->RegisterField("Level-set", surf_fit_bg_gf0);
         dc->SetCycle(0);
         dc->SetTime(0.0);
         dc->Save();
         delete dc;
      }

      surf_fit_bg_grad_fes =
         new FiniteElementSpace(mesh_surf_fit_bg, surf_fit_bg_fec, dim);
      surf_fit_bg_grad = new GridFunction(surf_fit_bg_grad_fes);

      surf_fit_bg_hess_fes =
         new FiniteElementSpace(mesh_surf_fit_bg, surf_fit_bg_fec, dim * dim);
      surf_fit_bg_hess = new GridFunction(surf_fit_bg_hess_fes);

      //Setup gradient of the background mesh
      const int size_bg = surf_fit_bg_gf0->Size();
      for (int d = 0; d < mesh_surf_fit_bg->Dimension(); d++)
      {
         GridFunction surf_fit_bg_grad_comp(
            surf_fit_bg_fes, surf_fit_bg_grad->GetData() + d * size_bg);
         surf_fit_bg_gf0->GetDerivative(1, d, surf_fit_bg_grad_comp);
      }
      std::cout << "Done Setup gradient on the bg mesh\n";

      //Setup Hessian on background mesh
      int id = 0;
      for (int d = 0; d < mesh_surf_fit_bg->Dimension(); d++)
      {
         for (int idir = 0; idir < mesh_surf_fit_bg->Dimension(); idir++)
         {
            GridFunction surf_fit_bg_grad_comp(
               surf_fit_bg_fes, surf_fit_bg_grad->GetData() + d * size_bg);
            GridFunction surf_fit_bg_hess_comp(
               surf_fit_bg_fes, surf_fit_bg_hess->GetData()+ id * size_bg);
            surf_fit_bg_grad_comp.GetDerivative(1, idir,
                                                surf_fit_bg_hess_comp);
            id++;
         }
      }
      std::cout << "Done Setup Hessian on the bg mesh\n";
   }

   if (surface_fit_const > 0.0)
   {
      finder.Interpolate(x, *surf_fit_bg_gf0, surf_fit_gf0,
                         x.FESpace()->GetOrdering());
      std::cout << "Done remap from bg mesh\n";

      GridFunction *mat_file = NULL;
      if (custom_material && strcmp(custom_material_file, "NULL") != 0)
      {
         ifstream cg_mat_stream(custom_material_file);
         mat_file = new GridFunction(mesh, cg_mat_stream);
         MFEM_VERIFY(mat_file->FESpace()->GetMesh()->GetNE() == mesh->GetNE(),
                     "Invalid material file. Not compatible");
      }

      for (int i = 0; i < mesh->GetNE(); i++)
      {
         mat(i) = material_id(i, surf_fit_gf0);
         if (exceptions == 2 || exceptions == 3)
         {
            mat(i) = material_id2(i, surf_fit_gf0);
         }
         else if (exceptions == 4)
         {
            mat(i) = material_id3(i, surf_fit_gf0);
            Vector center(mesh->Dimension());
            mesh->GetElementCenter(i, center);
            if (center(0) < 0.1 || center(0) > 0.9 || center(1) < 0.1 || center(1) > 0.9)
            {
               mat(i) = 1;
            }
         }

         if (custom_material)
         {
            Vector center(mesh->Dimension());
            mesh->GetElementCenter(i, center);
            mat(i) = 1;
            if (mesh->Dimension() == 2)
            {
               if (center(0) > 0.25 && center(0) < 0.75 && center(1) > 0.25 &&
                   center(1) < 0.75)
               {
                  mat(i) = 0;
               }
            }
            else if (mesh->Dimension() == 3)
            {
               if (center(0) > 0.25 && center(0) < 0.75
                   && center(1) > 0.25 && center(1) < 0.75
                   && center(2) > 0.25 && center(2) < 0.75)
               {
                  mat(i) = 0;
               }
            }
            if (mat_file) { mat(i) = (*mat_file)(i); }
         }
         mesh->SetAttribute(i, static_cast<int>(mat(i) + 1));
      }
      mesh->SetAttributes();

      delete mat_file;
      std::cout << "Done marking\n";
      if (custom_split_mesh > 0)
      {
         MakeMaterialConsistentForElementGroups(mat, 24);
      }
      else if (custom_split_mesh < 0 && exceptions != 4)
      {
         MakeMaterialConsistentForElementGroups(mat,
                                                custom_split_mesh == -1 ? 4 : 5);
      }

      surf_fit_gf0_max_order = ProlongToMaxOrder(&surf_fit_gf0, 0);
      // if (prefine)
      // {
      //    surf_fit_gf0_max_order = ProlongToMaxOrder(&surf_fit_gf0, 0);
      // }

      MakeGridFunctionWithNumberOfInterfaceFaces(mesh, mat, NumFaces);
      if (adapt_marking && (custom_split_mesh == 0 || exceptions == 4))
      {
         ModifyAttributeForMarkingDOFS(mesh, mat, 0);
         MakeGridFunctionWithNumberOfInterfaceFaces(mesh, mat, NumFaces);
         ModifyAttributeForMarkingDOFS(mesh, mat, 1);
         MakeGridFunctionWithNumberOfInterfaceFaces(mesh, mat, NumFaces);
         ModifyAttributeForMarkingDOFS(mesh, mat, 0);
         MakeGridFunctionWithNumberOfInterfaceFaces(mesh, mat, NumFaces);
         ModifyAttributeForMarkingDOFS(mesh, mat, 1);
      }
   }

   InterfaceFaceEdgeConnectivity ifec0;

   GetGroupInfo(mesh, mat, surf_fit_gf0, ifec0);
   SetGroupNumGf(ifec0, groupnum_gf);

   if (visualization)
   {
      x_max_order = ProlongToMaxOrder(&x, 0);
      mesh->SetNodalGridFunction(x_max_order);
      socketstream vis1;
      common::VisualizeField(vis1, "localhost", 19916, groupnum_gf,
                             "Initial group numbers",
                             0, 325, 300, 300, vis_keys);
      mesh->SetNodalGridFunction(&x);
      delete x_max_order;
   }
   // MFEM_ABORT(" ");

   // Output the connectivity info
   for (int i = 0; i < ifec0.inter_faces.Size(); i++)
   {
      int fnum = ifec0.inter_faces[i];
      vector<int> els = ifec0.face_to_els[fnum];
      std::cout << "Face: " << fnum << " ";
      std::cout << "Els: " << els[0] << " " << els[1] << std::endl;
   }

   // Output the group info
   for (int i = 0; i < ifec0.ngroups; i++)
   {
      vector<int> els = ifec0.group_to_els[i];
      std::cout << "Group: " << i << "\n";
      std::cout << "Elements: " << " ";
      for (int j = 0; j < els.size(); j++)
      {
         std::cout << els[j] << " ";
      }
      std::cout << std::endl;
      vector<int> fnums = ifec0.group_to_face[i];
      std::cout << "Faces: " << " ";
      for (int j = 0; j < fnums.size(); j++)
      {
         std::cout << fnums[j] << " ";
      }
      std::cout << std::endl;

      std::cout << "Edg els: " << " ";
      vector<int> edgels = ifec0.group_to_edg_els[i];
      for (int j = 0; j < edgels.size(); j++)
      {
         std::cout << edgels[j] << " ";
      }
      std::cout << std::endl;
   }


   // Set done_els to 0 for all the elements in different groups
   for (int i = 0; i < ifec0.ngroups; i++)
   {
      vector<int> els = ifec0.group_to_els[i];
      for (int j = 0; j < els.size(); j++)
      {
         done_els(els[j]) = 0.0;
         done_elsh(els[j]) = 0.0;
         done_elsp(els[j]) = 0.0;
         int_marker(ifec0.inter_face_el_all[i]) = 1.0;
      }

      vector<int> edgels = ifec0.group_to_edg_els[i];
      for (int j = 0; j < edgels.size(); j++)
      {
         done_els(edgels[j]) = 0.0;
         done_elsh(edgels[j]) = 0.0;
         done_elsp(edgels[j]) = 0.0;
      }
   }

   // Vector initial_face_error(ifec0.inter_faces.Size());
   double max_initial_face_error, max_current_face_error;
   double sum_current_face_error;
   // Vector current_face_error(ifec0.inter_faces.Size());
   const int ninterfaces = ifec0.inter_faces.Size();
   std::cout << "Total number of faces for fitting: " << ifec0.inter_faces.Size()
             <<
             std::endl;


   if (visualization)
   {
      x_max_order = ProlongToMaxOrder(&x, 0);
      mesh->SetNodalGridFunction(x_max_order);
      socketstream vis1;
      common::VisualizeField(vis1, "localhost", 19916, order_gf,
                             "Initial polynomial order",
                             0, 325, 300, 300, vis_keys);
      mesh->SetNodalGridFunction(&x);
      delete x_max_order;
   }

   std::cout << "Max Order        : " << pref_max_order << std::endl;
   std::cout << "Init Order       : " << mesh_poly_deg << std::endl;
   std::cout << "Increase Order by: " << pref_order_increase << std::endl;
   std::cout << "# of iterations  : " << pref_max_iter << std::endl;

   Array<int> adjacent_el_diff(0);
   if (adjeldiff >= 0)
   {
      adjacent_el_diff.SetSize(1);
      adjacent_el_diff=adjeldiff;
   }

   int iter_pref = 0;
   bool faces_to_update = true;

   // // make table of element and its neighbors
   // const Table &eltoeln = mesh->ElementToElementTable();

   if (visit)
   {
      order_gf = x.FESpace()->GetElementOrdersV();

      DataCollection *dc = NULL;
      dc = new VisItDataCollection("Initial_"+std::to_string(jobid), mesh);
      dc->RegisterField("orders", &order_gf);
      dc->RegisterField("Level-set", surf_fit_gf0_max_order);
      dc->RegisterField("intmarker", &int_marker);
      dc->SetCycle(0);
      dc->SetTime(0.0);
      dc->Save();
      delete dc;
   }


   double prederef_error = 0.0;
   int predef_ndofs = 0;
   int predef_tdofs = 0;
   int predef_nsurfdofs = 0;
   Vector current_group_error;
   Vector current_group_dofs;
   Vector current_group_max_error;
   Array<int> prior_el_order; // store element order of all elements at beginning
   // of each iteration.

   // Randomly set some elements to done for p and some to done for h.
   // for (int i = 0; i < ifec0.ngroups; i++)
   // {
   //    vector<int> els = ifec0.group_to_els[i];
   //    if (rand_real() < 0.3)
   //    {
   //       for (int j = 0; j < els.size(); j++)
   //       {
   //          done_elsh(els[j]) = 1.0;
   //       }
   //       std::cout << i << " done for h\n";
   //    }
   //    if (rand_real() < 0.3)
   //    {
   //       for (int j = 0; j < els.size(); j++)
   //       {
   //          done_elsp(els[j]) = 1.0;
   //       }
   //       std::cout << i << " done for p\n";
   //    }
   // }

   while (iter_pref < pref_max_iter && faces_to_update)
   {
      std::cout << "hrp-adaptivity iteration: " << iter_pref << std::endl;
      // Compute the minimum det(J) of the starting mesh.
      min_detJ = GetMinDet(mesh, fespace, irules, quad_order);
      cout << "Minimum det(J) of the mesh " << min_detJ << endl;

      if (iter_pref > 0)
      {
         if (exceptions == 4)
         {
            surface_fit_const = 1e-5;
         }
      }

      const Table &eltoeln = mesh->ElementToElementTable();

      // Define a TMOPIntegrator based on the metric and target.
      target_c->SetNodes(x0);
      TMOP_Integrator *tmop_integ = new TMOP_Integrator(metric, target_c);
      tmop_integ->SetIntegrationRules(*irules, quad_order);

      // Interpolate GridFunction from background mesh
      finder.Interpolate(x, *surf_fit_bg_gf0, surf_fit_gf0,
                         x.FESpace()->GetOrdering());

      if (iter_pref == 0 && visit)
      {
         order_gf = x.FESpace()->GetElementOrdersV();
         // if (prefine) {
         delete surf_fit_gf0_max_order;
         surf_fit_gf0_max_order = ProlongToMaxOrder(&surf_fit_gf0, 0);
         // }
         DataCollection *dc = NULL;
         dc = new VisItDataCollection("Optimized_"+std::to_string(jobid), mesh);
         dc->RegisterField("orders", &order_gf);
         dc->RegisterField("mat", &mat);
         dc->RegisterField("Level-set", surf_fit_gf0_max_order);
         dc->RegisterField("intmarker", &int_marker);
         dc->SetCycle(0);
         dc->SetTime(0.0);
         dc->Save();
         delete dc;
      }

      // Define a transfer operator for updating gridfunctions after the mesh
      // has been p-refined
      PRefinementTransfer preft_fespace = PRefinementTransfer(*fespace);
      PRefinementTransfer preft_surf_fit_fes = PRefinementTransfer(surf_fit_fes);
      int max_order = fespace->GetMaxElementOrder();

      x_max_order = ProlongToMaxOrder(&x, 0);
      mesh->SetNodalGridFunction(x_max_order);
      surf_fit_gf0_max_order = ProlongToMaxOrder(&surf_fit_gf0, 0);

      // Get current error information
      {
         double min_face_error = std::numeric_limits<double>::max();
         double max_face_error = std::numeric_limits<double>::min();
         // Compute integrated error for each face
         // current_face_error.SetSize(ifec0.inter_faces.Size());
         ComputeIntegratedErrorBGonFaces(x_max_order->FESpace(),
                                         surf_fit_bg_gf0,
                                         ifec0.inter_faces,
                                         surf_fit_gf0_max_order,
                                         finder,
                                         ifec0.face_error_map,
                                         error_type);

         max_current_face_error = -INFINITY;
         double error_sum = 0.0;
         for (int i=0; i < ifec0.inter_faces.Size(); i++)
         {
            int facenum = ifec0.inter_faces[i];
            double error_bg_face = ifec0.face_error_map[facenum];
            error_sum += error_bg_face;
            max_current_face_error = std::max(max_current_face_error, error_bg_face);
            min_face_error = std::min(min_face_error, error_bg_face);
            max_face_error = std::max(max_face_error, error_bg_face);

            fitting_error_gf(ifec0.inter_face_el1[i]) = std::max(error_bg_face,
                                                                 fitting_error_gf(ifec0.inter_face_el1[i]));
            fitting_error_gf(ifec0.inter_face_el2[i]) = std::max(error_bg_face,
                                                                 fitting_error_gf(ifec0.inter_face_el2[i]));
         } // i = inter_faces.size()
         sum_current_face_error = error_sum;

         if (iter_pref == 0)
         {
            max_initial_face_error = max_current_face_error;

            if (visualization)
            {
               socketstream vis1;
               common::VisualizeField(vis1, "localhost", 19916, fitting_error_gf,
                                      "Fitting Error before any fitting",
                                      0, 650, 300, 300, vis_keys);
            }
         }

         std::cout << "Integrate fitting error on BG: " << error_sum << " " <<
                   std::endl;
         std::cout << "Min/Max face error: " << min_face_error << " " <<
                   max_face_error << std::endl;
         std::cout << "Max order || NDOFS || Integrate fitting error on BG" <<
                   std::endl;
         std::cout << fespace->GetMaxElementOrder() << " "
                   << fespace->GetVSize() << " " << error_sum
                   << std::endl;
      }
      mesh->SetNodalGridFunction(&x);
      delete x_max_order;
      delete surf_fit_gf0_max_order;

      // For each group, compute number of TDOFs, total fitting error,
      // max face error
      GetGroupError(mesh, ifec0, current_group_max_error,
                    current_group_error, current_group_dofs);

      // check if the polynomial order and refinement level is consistent
      // for all elements in a group.
      for (int g = 0; g < ifec0.ngroups; g++)
      {
         int groupnum = g;
         vector<int> els = ifec0.group_to_els[groupnum];
         int maxdepth = mesh->ncmesh->GetElementDepth(els[0]);
         int maxorder = fespace->GetElementOrder(els[0]);
         int doel = done_els(els[0]);
         int doelp = done_elsp(els[0]);
         int doelh = done_elsh(els[0]);
         for (int e = 1; e < els.size(); e++)
         {
            int elnum = els[e];
            MFEM_VERIFY(maxdepth == mesh->ncmesh->GetElementDepth(elnum),
                        "Inconsistent refinement levels in group");
            MFEM_VERIFY(maxorder == fespace->GetElementOrder(elnum),
                        "Inconsistent polynomial orders in group");
            MFEM_VERIFY(doel == done_els(elnum),
                        "Inconsistent done_els in group");
            MFEM_VERIFY(doelp == done_elsp(elnum),
                        "Inconsistent done_elsp in group");
            MFEM_VERIFY(doelh == done_elsh(elnum),
                        "Inconsistent done_elsh in group");
         }
         vector<int> edgels = ifec0.group_to_edg_els[groupnum];
         for (int e = 0; e < edgels.size(); e++)
         {
            int elnum = edgels[e];
            MFEM_VERIFY(maxdepth == mesh->ncmesh->GetElementDepth(elnum),
                        "Inconsistent refinement levels in group-edges");
            MFEM_VERIFY(maxorder == fespace->GetElementOrder(elnum),
                        "Inconsistent polynomial orders in group-edges");
            MFEM_VERIFY(doel == done_els(elnum),
                        "Inconsistent done_els in group-edges");
            MFEM_VERIFY(doelp == done_elsp(elnum),
                        "Inconsistent done_els in group-edges");
            MFEM_VERIFY(doelh == done_elsh(elnum),
                        "Inconsistent done_els in group-edges");
         }
      }

      // Make list of all elements around the interface that are to be
      // tested for h/p-refinement.
      // Criterion - element should not have been marked done at prior iteration,
      // the current error should be above threshold, AND current order/depth
      // should be less than the threshold for p/h-refinement.
      int n_href_count = 0;
      int n_pref_count = 0;
      Array<int> h_candidate_els;
      Array<int> p_candidate_els;
      for (int g = 0; g < ifec0.ngroups; g++)
      {
         vector<int> els = ifec0.group_to_els[g];
         vector<int> edgels = ifec0.group_to_edg_els[g];
         // combine els and edgels in to one vector
         vector<int> combined_els = els;
         combined_els.insert(combined_els.end(), edgels.begin(), edgels.end());

         for (int e = 0; e < combined_els.size(); e++)
         {
            int elnum = combined_els[e];
            int current_order = fespace->GetElementOrder(elnum);
            if (done_els[elnum] == 0.0 &&
                current_group_max_error[g] > pref_tol)
            {
               if (prefine &&
                   current_order < pref_max_order &&
                   done_elsp[elnum] == 0.0)
               {
                  p_candidate_els.Append(elnum);
                  n_pref_count++;
               }
               if (hrefine &&
                   mesh->ncmesh->GetElementDepth(elnum) < href_max_depth &&
                   done_elsh[elnum] == 0.0)
               {
                  h_candidate_els.Append(elnum);
                  n_href_count++;
               }
            }
         }
      }

      std::cout << "===================================" << endl;
      cout << "Number of elements eligible for h- and p- test refinements: " <<
           n_href_count << " " << n_pref_count << endl;
      std::cout << "===================================" << endl;

      prior_el_order.SetSize(mesh->GetNE());
      for (int e = 0; e < mesh->GetNE(); e++)
      {
         prior_el_order[e] = fespace->GetElementOrder(e);
      }

      InterfaceFaceEdgeConnectivity ifecref = ifec0;

      // If this is not the first iteration, refine both in h- and p-.
      if (iter_pref > 0)
      {
         {
            socketstream vis1;
            x_max_order = ProlongToMaxOrder(&x, 0);
            mesh->SetNodalGridFunction(x_max_order);
            order_gf = x.FESpace()->GetElementOrdersV();
            // common::VisualizeField(vis1, "localhost", 19916, order_gf,
            //                        "Orders just before test p-refined mesh",
            //                        1000, 0, 300, 300, vis_keys);
            mesh->SetNodalGridFunction(&x);
            delete x_max_order;
         }
         // first we do p-refinement
         if (n_pref_count)
         {
            for (int i = 0; i < p_candidate_els.Size(); i++)
            {
               int elnum = p_candidate_els[i];
               int current_order = fespace->GetElementOrder(elnum);
               int done_h_status = done_elsh[elnum];
               // If we we will also be doing h-refinement, then we want to
               // increase so that increase in DOFs due to h-refinement is
               // similar.. otherwise we can increase by user input.
               int target_order = hrefine && !done_h_status ?
                                  2*current_order :
                                  current_order + pref_order_increase;
               int set_order = std::min(pref_max_order, target_order);
               order_gf(elnum) = set_order;
            }

            Array<int> new_orders;
            PropogateOrders(order_gf, ifec0.inter_face_el_all,
                            adjacent_el_diff,
                            eltoeln, new_orders, 0);

            for (int e = 0; e < mesh->GetNE(); e++)
            {
               order_gf(e) = new_orders[e];
               fespace->SetElementOrder(e, order_gf(e));
            }

            if (fespace->DidOrderChange())
            {
               fespace->Update(false);
               surf_fit_fes.CopySpaceElementOrders(*fespace);
               preft_fespace.Transfer(x);
               preft_fespace.Transfer(x0);
               preft_surf_fit_fes.Transfer(surf_fit_mat_gf);
               preft_surf_fit_fes.Transfer(surf_fit_gf0);
               surf_fit_marker.SetSize(surf_fit_gf0.Size());
            }

            // x.SetTrueVector();
            // x.SetFromTrueVector();

            // mesh->SetNodalGridFunction(&x);
            // x_max_order = ProlongToMaxOrder(&x, 0);
            finder.Interpolate(x, *surf_fit_bg_gf0, surf_fit_gf0,
                               x.FESpace()->GetOrdering());
            // mesh->SetNodalGridFunction(&x);
            // delete x_max_order;
         }

         // Compute the minimum det(J) of the starting mesh.
         min_detJ = GetMinDet(mesh, fespace, irules, quad_order);
         cout << "Minimum det(J) of the original mesh after test p-ref is " << min_detJ
              << endl;
         if (min_detJ <= 0.0)
         {
            socketstream vis1;
            x_max_order = ProlongToMaxOrder(&x, 0);
            mesh->SetNodalGridFunction(x_max_order);
            order_gf = x.FESpace()->GetElementOrdersV();
            common::VisualizeField(vis1, "localhost", 19916, order_gf,
                                   "Orders on (test) p-refined mesh",
                                   1000, 0, 300, 300, vis_keys);
            mesh->SetNodalGridFunction(&x);
            delete x_max_order;
            // if (iter_pref == 2) { MFEM_ABORT(" "); }
         }
         MFEM_VERIFY(min_detJ > 0.0, "Non-positive Jacobian after test p-refinement");

         // Now do h-refinement
         if (n_href_count)
         {
            mesh->GeneralRefinement(h_candidate_els, 1, 0);
            hrefup.Update(true);
            // Also update the face information
            GetGroupInfo(mesh, mat, surf_fit_gf0, ifecref);
            surf_fit_marker.SetSize(surf_fit_gf0.Size());
         }
         min_detJ = GetMinDet(mesh, fespace, irules, quad_order);
         cout << "Minimum det(J) of the original mesh after test hp-ref is " << min_detJ
              << endl;

         x_max_order = ProlongToMaxOrder(&x, 1);
         mesh->SetNodalGridFunction(x_max_order);
         if (visualization)
         {
            socketstream vis1, vis2;
            // common::VisualizeField(vis1, "localhost", 19916, order_gf,
            //                        "Orders on (test) hp-refined mesh",
            //                        325*(iter_pref+1), 0, 300, 300, vis_keys);
            // common::VisualizeField(vis2, "localhost", 19916, groupnum_gf,
            //                        "Group numbers on (test) hp-refined mesh",
            //                        325*(iter_pref+1), 0, 300, 300, vis_keys);
         }
         mesh->SetNodalGridFunction(&x);
         delete x_max_order;
      } //if (iter_pref > 0) - end of test h/p refinement


      // Setup things for fitting
      if (surf_bg_mesh)
      {
         delete surf_fit_grad_fes;
         delete surf_fit_grad;
         surf_fit_grad_fes =
            new FiniteElementSpace(mesh, &surf_fit_fec, dim);
         surf_fit_grad_fes->CopySpaceElementOrders(*fespace);
         surf_fit_grad_fes->SetRelaxedHpConformity(fespace->GetRelaxedHpConformity());
         surf_fit_grad = new GridFunction(surf_fit_grad_fes);

         delete surf_fit_hess_fes;
         delete surf_fit_hess;
         surf_fit_hess_fes =
            new FiniteElementSpace(mesh, &surf_fit_fec, dim * dim);
         surf_fit_hess_fes->CopySpaceElementOrders(*fespace);
         surf_fit_hess_fes->SetRelaxedHpConformity(fespace->GetRelaxedHpConformity());
         surf_fit_hess = new GridFunction(surf_fit_hess_fes);
      }
      for (int j = 0; j < surf_fit_marker.Size(); j++)
      {
         surf_fit_marker[j] = false;
      }
      surf_fit_mat_gf = 0.0;

      Array<int> dof_list;
      Array<int> dofs;
      for (int i = 0; i < ifecref.inter_faces.Size(); i++)
      {
         int fnum = ifecref.inter_faces[i];
         if (dim == 2)
         {
            surf_fit_gf0.FESpace()->GetEdgeDofs(fnum, dofs);
         }
         else
         {
            surf_fit_gf0.FESpace()->GetFaceDofs(fnum, dofs);
         }
         dof_list.Append(dofs);
      }
      for (int i = 0; i < dof_list.Size(); i++)
      {
         surf_fit_marker[dof_list[i]] = true;
         surf_fit_mat_gf(dof_list[i]) = 1.0;
      }
      if (iter_pref != 0 && prefine)
      {
         delete surf_fit_mat_gf_max_order;
      }

#ifdef MFEM_USE_GSLIB
      delete adapt_surface;
      adapt_surface = new InterpolatorFP;
      if (surf_bg_mesh)
      {
         delete adapt_grad_surface;
         delete adapt_hess_surface;
         adapt_grad_surface = new InterpolatorFP;
         adapt_hess_surface = new InterpolatorFP;
      }
#else
      MFEM_ABORT("p-adaptivity requires MFEM with GSLIB support!");
#endif

      if (!surf_bg_mesh)
      {
         tmop_integ->EnableSurfaceFitting(surf_fit_gf0, surf_fit_marker,
                                          surf_fit_coeff, *adapt_surface);
      }
      else
      {
         tmop_integ->EnableSurfaceFittingFromSource(
            *surf_fit_bg_gf0, surf_fit_gf0,
            surf_fit_marker, surf_fit_coeff, *adapt_surface,
            *surf_fit_bg_grad, *surf_fit_grad, *adapt_grad_surface,
            *surf_fit_bg_hess, *surf_fit_hess, *adapt_hess_surface);
      }

      // if (prefine)
      {
         x_max_order = ProlongToMaxOrder(&x, 0);
         mesh->SetNodalGridFunction(x_max_order);
         surf_fit_gf0_max_order = ProlongToMaxOrder(&surf_fit_gf0, 0);
         surf_fit_mat_gf_max_order = ProlongToMaxOrder(&surf_fit_mat_gf, 0);
      }

      if (visualization && iter_pref==0)
      {
         socketstream vis1, vis2;
         common::VisualizeField(vis1, "localhost", 19916, mat,
                                "Materials for initial mesh",
                                325, 0, 300, 300, vis_keys);
      }
      mesh->SetNodalGridFunction(&x);
      delete x_max_order;
      delete surf_fit_gf0_max_order;

      // Setup the final NonlinearForm
      NonlinearForm a(fespace);
      a.AddDomainIntegrator(tmop_integ);

      // Compute the minimum det(J) of the starting mesh.
      min_detJ = GetMinDet(mesh, fespace, irules, quad_order);
      cout << "Minimum det(J) of the original mesh is " << min_detJ << endl;

      if (min_detJ < 0.0)
      {
         MFEM_ABORT("The input mesh is inverted! Try an untangling metric.");
      }

      const double init_energy = a.GetGridFunctionEnergy(x);
      double init_metric_energy = init_energy;
      if (surface_fit_const > 0.0)
      {
         surf_fit_coeff.constant   = 0.0;
         init_metric_energy = a.GetGridFunctionEnergy(x);
         surf_fit_coeff.constant   = surface_fit_const;
      }
      x_max_order = ProlongToMaxOrder(&x, 0);
      mesh->SetNodalGridFunction(x_max_order);
      if (visualization && iter_pref==0)
      {
         char title[] = "Initial metric values";
         vis_tmop_metric_s(mesh_poly_deg, *metric, *target_c, *mesh, title, 0, 0, 300,
                           300);
      }
      mesh->SetNodalGridFunction(&x);
      delete x_max_order;

      // 13. Fix nodes
      if (move_bnd == false)
      {
         Array<int> ess_bdr(mesh->bdr_attributes.Max());
         ess_bdr = 1;
         a.SetEssentialBC(ess_bdr);
      }
      else
      {
         Array<int> ess_vdofs;
         Array<int> vdofs;
         for (int i = 0; i < mesh->GetNBE(); i++)
         {
            const int nd = fespace->GetBE(i)->GetDof();
            const int attr = mesh->GetBdrElement(i)->GetAttribute();
            fespace->GetBdrElementVDofs(i, vdofs);
            for (int d = 0; d < dim; d++)
            {
               if (attr == d+1)
               {
                  for (int j = 0; j < nd; j++)
                  { ess_vdofs.Append(vdofs[j+d*nd]); }
                  continue;
               }
            }
            if (attr == 4) // Fix all components.
            {
               ess_vdofs.Append(vdofs);
            }
         }
         a.SetEssentialVDofs(ess_vdofs);
      }

      // Setup solver
      Solver *S = NULL, *S_prec = NULL;
      const double linsol_rtol = 1e-12;
      if (lin_solver == 0)
      {
         S = new DSmoother(1, 1.0, max_lin_iter);
      }
      else if (lin_solver == 1)
      {
         CGSolver *cg = new CGSolver;
         cg->SetMaxIter(max_lin_iter);
         cg->SetRelTol(linsol_rtol);
         cg->SetAbsTol(0.0);
         cg->SetPrintLevel(verbosity_level >= 2 ? 3 : -1);
         S = cg;
      }
      else
      {
         MINRESSolver *minres = new MINRESSolver;
         minres->SetMaxIter(max_lin_iter);
         minres->SetRelTol(linsol_rtol);
         minres->SetAbsTol(0.0);
         if (verbosity_level > 2) { minres->SetPrintLevel(1); }
         minres->SetPrintLevel(verbosity_level == 2 ? 3 : -1);
         if (lin_solver == 3 || lin_solver == 4)
         {
            auto ds = new DSmoother((lin_solver == 3) ? 0 : 1, 1.0, 1);
            ds->SetPositiveDiagonal(true);
            S_prec = ds;
            minres->SetPreconditioner(*S_prec);
         }
         S = minres;
      }

      // Set up an empty right-hand side vector b, which is equivalent to b=0.
      // We use this later when we solve the TMOP problem
      Vector b(0);

      // Perform the nonlinear optimization.
      const IntegrationRule &ir =
         irules->Get(fespace->GetFE(0)->GetGeomType(), quad_order);
      TMOPNewtonSolver solver(ir, solver_type);
      if (surface_fit_const > 0.0 && surface_fit_adapt)
      {
         solver.SetAdaptiveSurfaceFittingScalingFactor(surface_fit_adapt);
         solver.SetAdaptiveSurfaceFittingRelativeChangeThreshold(0.01);
      }
      if (surface_fit_const > 0.0 && surface_fit_threshold > 0)
      {
         solver.SetTerminationWithMaxSurfaceFittingError(surface_fit_threshold);
      }
      // Provide all integration rules in case of a mixed mesh.
      solver.SetIntegrationRules(*irules, quad_order);
      if (solver_type == 0)
      {
         solver.SetPreconditioner(*S);
      }
      solver.SetMaxIter(solver_iter);
      if (exceptions == 100 && iter_pref == 0)
      {
         solver.SetMaxIter(30);
      }
      solver.SetRelTol(solver_rtol);
      solver.SetAbsTol(0.0);
      solver.SetPrintLevel(verbosity_level >= 1 ? 1 : -1);

      solver.SetOperator(a);
      solver.Mult(b, x.GetTrueVector());
      x.SetFromTrueVector();

      x_max_order = ProlongToMaxOrder(&x, 0);
      mesh->SetNodalGridFunction(x_max_order);

      // Save the optimized mesh to a file. This output can be viewed later
      // using GLVis: "glvis -m optimized.mesh".
      {
         ofstream mesh_ofs("optimizedsub_"+std::to_string(iter_pref)+"_"+std::to_string(
                              jobid)+ ".mesh");
         mesh_ofs.precision(14);
         mesh->Print(mesh_ofs);
      }
      {
         ofstream gf_ofs("optimizedmatsub_"+std::to_string(iter_pref)+"_"+std::to_string(
                            jobid)+ ".gf");
         gf_ofs.precision(14);
         mat.Save(gf_ofs);
      }
      mesh->SetNodalGridFunction(&x);
      // Don't delete x_max_order yet

      // Report the final energy of the functional.
      const double fin_energy = a.GetGridFunctionEnergy(x);
      double fin_metric_energy = fin_energy;
      if (surface_fit_const > 0.0)
      {
         surf_fit_coeff.constant  = 0.0;
         fin_metric_energy  = a.GetGridFunctionEnergy(x);
         surf_fit_coeff.constant  = surface_fit_const;
      }

      std::cout << std::scientific << std::setprecision(4);
      cout << "Initial strain energy: " << init_energy
           << " = metrics: " << init_metric_energy
           << " + extra terms: " << init_energy - init_metric_energy << endl;
      cout << "  Final strain energy: " << fin_energy
           << " = metrics: " << fin_metric_energy
           << " + extra terms: " << fin_energy - fin_metric_energy << endl;
      cout << "The strain energy decreased by: "
           << (init_energy - fin_energy) * 100.0 / init_energy << " %." << endl;

      // Set nodal gridfunction for visualization and fitting error computation.
      mesh->SetNodalGridFunction(x_max_order);

      // Visualize the final mesh and metric values.
      if (visualization && iter_pref==pref_max_iter-1)
      {
         char title[] = "Final metric values";
         vis_tmop_metric_s(mesh_poly_deg, *metric, *target_c, *mesh, title, 325, 0, 300,
                           300);
      }
      {
         double err_avg, err_max;
         Vector temp(0);
         tmop_integ->GetSurfaceFittingErrors(temp,err_avg, err_max);
         std::cout << "Avg fitting error: " << err_avg << std::endl
                   << "Max fitting error: " << err_max << std::endl;

         tmop_integ->RemapSurfFittingGridFunction(x, *x.FESpace());
         tmop_integ->CopyFittingGridFunction(surf_fit_gf0);

         surf_fit_gf0_max_order = ProlongToMaxOrder(&surf_fit_gf0, 0);
         if (visit)
         {
            order_gf = x.FESpace()->GetElementOrdersV();
            DataCollection *dc = NULL;
            dc = new VisItDataCollection("Optimized_"+std::to_string(jobid), mesh);
            dc->RegisterField("orders", &order_gf);
            dc->RegisterField("Level-set", surf_fit_gf0_max_order);
            dc->RegisterField("mat", &mat);
            dc->RegisterField("intmarker", &int_marker);
            dc->SetCycle(2*iter_pref+1);
            dc->SetTime(2.0*iter_pref+1.0);
            dc->Save();
            delete dc;
         }
         double error_sum = 0.0;
         fitting_error_gf = pref_tol/100.0;
         double min_face_error = std::numeric_limits<double>::max();
         double max_face_error = std::numeric_limits<double>::min();

         ComputeIntegratedErrorBGonFaces(x_max_order->FESpace(),
                                         surf_fit_bg_gf0,
                                         ifecref.inter_faces,
                                         surf_fit_gf0_max_order,
                                         finder,
                                         ifecref.face_error_map,
                                         error_type);
         error_sum = 0.0;
         for (int i=0; i < ifecref.inter_faces.Size(); i++)
         {
            int fnum = ifecref.inter_faces[i];
            double faceerror = ifecref.face_error_map[fnum];
            error_sum += faceerror;
            fitting_error_gf(ifecref.inter_face_el1[i]) = faceerror;
            fitting_error_gf(ifecref.inter_face_el2[i]) = faceerror;
            min_face_error = std::min(min_face_error, faceerror);
            max_face_error = std::max(max_face_error, faceerror);
         }
         predef_ndofs = fespace->GetVSize();
         predef_tdofs = fespace->GetTrueVSize();
         surf_fit_mat_gf = 0.0;
         surf_fit_mat_gf.SetTrueVector();
         surf_fit_mat_gf.SetFromTrueVector();
         {
            // Get number of surface DOFs
            for (int i = 0; i < surf_fit_mat_gf.Size(); i++)
            {
               surf_fit_mat_gf[i] = surf_fit_marker[i];
            }
            surf_fit_mat_gf.SetTrueVector();
            Vector surf_fit_mat_gf_tvec = surf_fit_mat_gf.GetTrueVector();
            predef_nsurfdofs = 0;
            for (int i = 0; i < surf_fit_mat_gf_tvec.Size(); i++)
            {
               predef_nsurfdofs += (int)surf_fit_mat_gf_tvec(i);
            }
         }

         order_gf = x.FESpace()->GetElementOrdersV();
         if (visualization)
         {
            socketstream vis1, vis2;
            common::VisualizeField(vis1, "localhost", 19916, fitting_error_gf,
                                   "Fitting Error after fitting",
                                   325*(iter_pref+1), 100, 300, 300, vis_keys);
         }

         std::cout << "NDofs & Integrated Error Post Fitting: " <<
                   fespace->GetVSize() << " " <<
                   error_sum << " " << std::endl;
         std::cout << "Min/Max face error Post fitting: " << min_face_error << " " <<
                   max_face_error << std::endl;
         min_detJ = GetMinDet(mesh, fespace, irules, quad_order);
         cout << "Minimum det(J) of the mesh Post Fitting is " << min_detJ << endl;

         {
            Array<int> el_by_order(x.FESpace()->GetMaxElementOrder()+1);
            el_by_order = 0;
            for (int e = 0; e < mesh->GetNE(); e++)
            {
               int el_order = x.FESpace()->GetElementOrder(e);
               el_by_order[el_order] += 1;
            }
            if (exceptions == 4)
            {
               el_by_order = 0;
               for (int e = 0; e < mesh->GetNE(); e++)
               {
                  int el_order = x.FESpace()->GetElementOrder(e);
                  if (mat(e) == 1.0)
                  {
                     el_by_order[el_order] += 1;
                  }
               }
            }
            std::cout << "Print number of elements by order (0 to max from lTOr)\n";
            el_by_order.Print();
            std::cout << "Total elements: " << el_by_order.Sum() << std::endl;
         }
      }
      mesh->SetNodalGridFunction(&x);
      delete x_max_order;
      delete surf_fit_gf0_max_order;

      // Now we compare h- and p-refinements to keep the better of the two.
      // If there are no h-refinements, we can skip this step as we will test
      // for p-derefinements in a separate section.
      // If there are no p-refinements, we can skip this step since there will
      // be no testing needed.
      if (iter_pref > 0 && n_href_count && n_pref_count)
      {
         // Let's first test by derefining in h to get an idea of the impact
         // of p-refinements
         Vector group_error_p = current_group_error;
         Vector group_dofs_p = current_group_dofs;
         if (n_pref_count)
         {
            GridFunction *x_max_order_p = NULL;
            GridFunction *surf_fit_gf0_max_order_p = NULL;
            Mesh mesh_p(*mesh);
            GridFunction *x_p = mesh_p.GetNodes();
            FiniteElementSpace *fespace_p = x_p->FESpace();
            // we turn off ownership for now so that gridfunction, fespace, and fec
            // are not deleted when we set x_max_order.
            mesh_p.SetNodesOwner(false);

            FiniteElementSpace l2zero_fes_p(&mesh_p, &l2zero_coll);
            GridFunction mat_p(&l2zero_fes_p);
            GridFunction fitting_error_p(&l2zero_fes_p);
            GridFunction groupnum_gf_p(&l2zero_fes_p);
            GridFunction done_elsp_p(&l2zero_fes_p);
            GridFunction done_elsh_p(&l2zero_fes_p);
            mat_p = mat;
            groupnum_gf_p = groupnum_gf;
            done_elsp_p = done_elsp;
            done_elsh_p = done_elsh;

            FiniteElementSpace surf_fit_fes_p(&mesh_p, &surf_fit_fec);
            surf_fit_fes_p.CopySpaceElementOrders(*fespace_p);
            surf_fit_fes_p.SetRelaxedHpConformity(fespace_p->GetRelaxedHpConformity());
            GridFunction surf_fit_gf0_p(&surf_fit_fes_p);
            surf_fit_gf0_p = surf_fit_gf0;

            // const CoarseFineTransformations &cf_tr =
            //    mesh->ncmesh->GetRefinementTransforms();
            // Table coarse_to_fine;
            // cf_tr.MakeCoarseToFineTable(coarse_to_fine);
            // Table fine_to_coarse = MakeFineToCoarseTable(coarse_to_fine);
            // // Numrows = numparents, each row has child elements from current mesh

            // First derefine all the elements that were most recently refined
            Vector error_fine_mesh(mesh_p.GetNE());
            error_fine_mesh = done_els;

            double threshold = 0.5;
            mesh_p.DerefineByError(error_fine_mesh, threshold, 0, 1);

            // Update
            surf_fit_gf0_p.FESpace()->Update();
            surf_fit_gf0_p.Update();
            l2zero_fes_p.Update();
            mat_p.Update();
            groupnum_gf_p.Update();
            done_elsp_p.Update();
            done_elsh_p.Update();

            // Get connectivity information
            InterfaceFaceEdgeConnectivity ifec_p;
            GetGroupInfo(&mesh_p, mat_p, surf_fit_gf0_p, ifec_p);

            // Compute error for all interfaces
            x_max_order_p = ProlongToMaxOrder(x_p, 0);
            surf_fit_gf0_max_order_p = ProlongToMaxOrder(&surf_fit_gf0_p, 0);
            mesh_p.SetNodalGridFunction(x_max_order_p);

            Vector face_error_p(ifec_p.inter_faces.Size());
            ComputeIntegratedErrorBGonFaces(x_max_order_p->FESpace(),
                                            surf_fit_bg_gf0,
                                            ifec_p.inter_faces,
                                            surf_fit_gf0_max_order_p,
                                            finder,
                                            ifec_p.face_error_map,
                                            error_type);
            mesh_p.SetNodalGridFunction(x_p);

            delete x_max_order_p;
            delete surf_fit_gf0_max_order_p;

            if (true || visualization)
            {
               fitting_error_p = pref_tol/100.0;
               for (int i=0; i < ifec_p.inter_faces.Size(); i++)
               {
                  int facenum = ifec_p.inter_faces[i];
                  double error_bg_face = ifec_p.face_error_map[facenum];
                  fitting_error_p(ifec_p.inter_face_el1[i]) = std::max(error_bg_face,
                                                                       fitting_error_gf(ifec_p.inter_face_el1[i]));
                  fitting_error_p(ifec_p.inter_face_el2[i]) = std::max(error_bg_face,
                                                                       fitting_error_gf(ifec_p.inter_face_el2[i]));
               } // i = inter_faces.size()

               x_max_order_p = ProlongToMaxOrder(x_p, 0);
               mesh_p.SetNodalGridFunction(x_max_order_p);
               socketstream vis1, vis2;
               // common::VisualizeField(vis1, "localhost", 19916, fitting_error_p,
               //                        "Fitting Error for p-ref only",
               //                        1000, 400, 300, 300, vis_keys);
               // common::VisualizeField(vis2, "localhost", 19916, groupnum_gf_p,
               //                      "Group numbers on p-ref only",
               //                      1000, 400, 300, 300, vis_keys);
               mesh_p.SetNodalGridFunction(x_p);
               delete x_max_order_p;
            }

            // // Get derefinement transforms to map info
            // const CoarseFineTransformations &dtrans =
            //    mesh_p.ncmesh->GetDerefinementTransforms();
            // Table coarse_to_fine2;
            // dtrans.MakeCoarseToFineTable(coarse_to_fine2);
            // // Size is number of current elements, and each row has info for
            // // previous fine mesh.

            Vector group_max_error_temp, group_error_temp, group_dofs_temp;
            GetGroupError(&mesh_p, ifec_p, group_max_error_temp,
                          group_error_temp, group_dofs_temp);

            // map error from this mesh to previous mesh and then to mesh
            // before that.
            for (int i = 0; i < group_error_temp.Size(); i++)
            {
               double fine_error = group_error_temp(i);
               double fine_dofs = group_dofs_temp(i);
               int groupnum = i;
               int current_element = ifec_p.group_to_els[groupnum][0];
               int orig_groupnum = groupnum_gf_p(current_element);
               if (done_elsp_p(current_element) == 0.0)
               {
                  // if candidate for p
                  if (done_elsh_p(current_element) == 0.0)
                  {
                     // if candidate for h
                     group_error_p(orig_groupnum) = fine_error;
                     group_dofs_p(orig_groupnum) = fine_dofs;
                  }
                  else // not a candidate for h. i.e. for p-only
                  {
                     // we zero out error to enforce that p-wins here.
                     group_error_p(orig_groupnum) = fine_error*0.0;
                     group_dofs_p(orig_groupnum) = fine_dofs;
                  }
               }
               // std::cout << i << " " << orig_groupnum << " " <<
               // group_error_p(orig_groupnum) << " " << group_dofs_p(orig_groupnum) << " k10-p-error\n";
            }

            mesh_p.SetNodesOwner(true);
         }

         // Now we derefine in p and get numbers for h
         Vector group_error_h = current_group_error;
         Vector group_dofs_h = current_group_dofs;
         if (n_href_count)
         {
            GridFunction *x_max_order_h = NULL;

            Mesh mesh_h(*mesh);
            GridFunction *x_h = mesh_h.GetNodes();
            FiniteElementSpace *fespace_h = x_h->FESpace();
            mesh_h.SetNodesOwner(false);

            FiniteElementSpace l2zero_fes_h(&mesh_h, &l2zero_coll);
            GridFunction mat_h(&l2zero_fes_h);
            GridFunction fitting_error_h(&l2zero_fes_h);
            GridFunction groupnum_gf_h(&l2zero_fes_h);
            GridFunction done_elsp_h(&l2zero_fes_h);
            GridFunction done_elsh_h(&l2zero_fes_h);
            mat_h = mat;
            groupnum_gf_h = groupnum_gf;
            done_elsp_h = done_elsp;
            done_elsh_h = done_elsh;

            FiniteElementSpace surf_fit_fes_h(&mesh_h, &surf_fit_fec);
            surf_fit_fes_h.CopySpaceElementOrders(*fespace_h);
            surf_fit_fes_h.SetRelaxedHpConformity(fespace_h->GetRelaxedHpConformity());
            GridFunction surf_fit_gf0_h(&surf_fit_fes_h);
            surf_fit_gf0_h = surf_fit_gf0;

            PRefinementTransfer preft_h = PRefinementTransfer(*fespace_h);
            PRefinementTransfer preft_surf_h = PRefinementTransfer(surf_fit_fes_h);

            const CoarseFineTransformations &cf_tr =
               mesh->ncmesh->GetRefinementTransforms();
            Table coarse_to_fine;
            cf_tr.MakeCoarseToFineTable(coarse_to_fine);
            Table fine_to_coarse = MakeFineToCoarseTable(coarse_to_fine);
            // Numrows = numparents, each row has child elements from current mesh

            // if (visualization)
            // {
            //    x_max_order_h = ProlongToMaxOrder(x_h, 0);
            //    mesh_h.SetNodalGridFunction(x_max_order_h);
            //    socketstream vis1;
            //    common::VisualizeField(vis1, "localhost", 19916, surf_fit_gf0_h,
            //                         "Gridfunction on fitted mesh",
            //                         650, 650, 300, 300, vis_keys);
            //    mesh_h.SetNodalGridFunction(x_h);
            //    delete x_max_order_h;
            // }

            // First derefine all the elements that were most recently refined
            for (int i = 0; i < coarse_to_fine.Size(); i++)
            {
               Array<int> tabrow;
               coarse_to_fine.GetRow(i, tabrow);
               const int parent = i;
               const int parent_order = prior_el_order[parent];
               for (int j = 0; j < tabrow.Size(); j++)
               {
                  int fine_el = tabrow[j];
                  fespace_h->SetElementOrder(fine_el, parent_order);
               }
            }

            // Update
            fespace_h->Update(false);
            surf_fit_fes_h.CopySpaceElementOrders(*fespace_h);
            surf_fit_fes_h.SetRelaxedHpConformity(fespace_h->GetRelaxedHpConformity());
            preft_h.Transfer(*x_h);
            preft_surf_h.Transfer(surf_fit_gf0_h);

            // Get connectivity information
            InterfaceFaceEdgeConnectivity ifec_h = ifecref;

            // Compute error for all interfaces
            x_max_order_h = ProlongToMaxOrder(x_h, 0);
            GridFunction *surf_fit_gf0_max_order_h = NULL;
            surf_fit_gf0_max_order_h = ProlongToMaxOrder(&surf_fit_gf0_h, 0);

            Vector face_error_h(ifec_h.inter_faces.Size());
            mesh_h.SetNodalGridFunction(x_max_order_h);
            ComputeIntegratedErrorBGonFaces(x_max_order_h->FESpace(),
                                            surf_fit_bg_gf0,
                                            ifec_h.inter_faces,
                                            surf_fit_gf0_max_order_h,
                                            finder,
                                            ifec_h.face_error_map,
                                            error_type);
            mesh_h.SetNodalGridFunction(x_h);

            delete x_max_order_h;
            delete surf_fit_gf0_max_order_h;

            if (visualization || true)
            {
               fitting_error_h = pref_tol/100.0;
               for (int i=0; i < ifec_h.inter_faces.Size(); i++)
               {
                  int facenum = ifec_h.inter_faces[i];
                  double error_bg_face = ifec_h.face_error_map[facenum];
                  fitting_error_h(ifec_h.inter_face_el1[i]) = std::max(error_bg_face,
                                                                       fitting_error_gf(ifec_h.inter_face_el1[i]));
                  fitting_error_h(ifec_h.inter_face_el2[i]) = std::max(error_bg_face,
                                                                       fitting_error_gf(ifec_h.inter_face_el2[i]));
               } // i = inter_faces.size()

               x_max_order_h = ProlongToMaxOrder(x_h, 0);
               mesh_h.SetNodalGridFunction(x_max_order_h);
               socketstream vis1;
               // common::VisualizeField(vis1, "localhost", 19916, fitting_error_h,
               //                        "Fitting Error for h-ref only",
               //                        1000, 800, 300, 300, vis_keys);
               mesh_h.SetNodalGridFunction(x_h);
               delete x_max_order_h;
            }

            auto group_sibling_dofs = new Array<Array<int> *>;

            Vector group_max_error_temp, group_error_temp, group_dofs_temp;
            GetGroupError(&mesh_h, ifec_h, group_max_error_temp,
                          group_error_temp,
                          group_dofs_temp, group_sibling_dofs);

            // The DOF numbers are inaccurate as we need number of DOFs for the
            // siblings as well.
            // Note - group_sibling-dofs has ngroups based on h-refined mesh.
            Array<Array<int> *> parent_dofs(ifec0.ngroups);
            for (int i = 0; i < parent_dofs.Size(); i++)
            {
               parent_dofs[i] = new Array<int>;
            }
            for (int i = 0; i < group_sibling_dofs->Size(); i++)
            {
               int groupnum = i;
               for (int e = 0; e < ifec_h.group_to_els[groupnum].size(); e++)
               {
                  int elnum = ifec_h.group_to_els[groupnum][e];
                  int parent_group = groupnum_gf_h(elnum);

                  // int parent = fine_to_coarse.GetRow(elnum)[0];
                  // int parent_group = ifec0.els_to_group[parent];
                  Array<int> *eldofs = (*group_sibling_dofs)[groupnum];
                  parent_dofs[parent_group]->Append(*eldofs);
               }
            }
            for (int i = 0; i < ifec0.ngroups; i++)
            {
               parent_dofs[i]->Sort();
               parent_dofs[i]->Unique();
               group_dofs_h(i) = parent_dofs[i]->Size();
            }

            group_error_h = 0.0;
            // map error from this mesh to previous mesh
            for (int i = 0; i < group_error_temp.Size(); i++)
            {
               double fine_error = group_error_temp(i);
               // double fine_dofs = group_dofs_temp(i);
               int groupnum = i;
               int current_element = ifec_h.group_to_els[groupnum][0];
               int orig_groupnum = groupnum_gf_h(current_element);
               // int parent_element = fine_to_coarse.GetRow(current_element)[0];
               // int orig_groupnum = ifec0.els_to_group[parent_element];
               if (done_elsh_h(current_element) == 0.0)
               {
                  if (done_elsp_h(current_element) == 0.0)
                  {
                     group_error_h(orig_groupnum) += fine_error;
                  }
                  else
                  {
                     group_error_h(orig_groupnum) = 0.0;
                  }
               }
               // std::cout << i << " " << orig_groupnum << " " << group_error_h(orig_groupnum) << " " << group_dofs_h(orig_groupnum) << " k10-h-error\n";
            }
         }

         // output errors for current_group_error, group_error_p, group_error_h
         // output dofs for current_group_dofs, group_dofs_p, group_dofs_h
         for (int i = 0; i < current_group_error.Size(); i++)
         {
            std::cout << "Group " << i << " errors and dofs - orig, p, h\n";
            std::cout << current_group_error(i) << " " << group_error_p(
                         i) << " " << group_error_h(i) << std::endl;
            std::cout << current_group_dofs(i) << " " << group_dofs_p(
                         i) << " " << group_dofs_h(i) << std::endl;
         }

         // now we implement the actual derefinements
         {
            const CoarseFineTransformations &cf_tr =
               mesh->ncmesh->GetRefinementTransforms();
            Table coarse_to_fine;
            cf_tr.MakeCoarseToFineTable(coarse_to_fine);
            Table fine_to_coarse = MakeFineToCoarseTable(coarse_to_fine);
            // coarse_to_fine.Print();
            // fine_to_coarse.Print();

            // first derefine in p
            {
               preft_fespace.SetSourceFESpace(*fespace);
               preft_surf_fit_fes.SetSourceFESpace(surf_fit_fes);

               for (int i = 0; i < ifec0.ngroups; i++)
               {
                  double p_ref_benefit = (current_group_error(i) - group_error_p(i)) /
                                         (group_dofs_p(i) - current_group_dofs(i));
                  double h_ref_benefit = (current_group_error(i) - group_error_h(i)) /
                                         (group_dofs_h(i)-current_group_dofs(i));

                  // if (i % 2 == 0)
                  // std::cout << i << " " << (p_ref_benefit < h_ref_benefit) << " k10-p-h\n";
                  if (p_ref_benefit <= h_ref_benefit)
                  {
                     vector<int> els = ifec0.group_to_els[i];
                     for (int j = 0; j < els.size(); j++)
                     {
                        int parent_el = els[j];
                        Array<int> tabrow;
                        coarse_to_fine.GetRow(parent_el, tabrow);
                        int parent_order = prior_el_order[parent_el];
                        for (int k = 0; k < tabrow.Size(); k++)
                        {
                           int fine_el = tabrow[k];
                           if (done_elsp(fine_el) == 0.0)
                           {
                              fespace->SetElementOrder(fine_el, parent_order);
                           }
                        }
                        // if (done_elsp(el) == 0.0)
                        // {

                        //    int parent = fine_to_coarse.GetRow(el)[0];
                        //    int parent_order = prior_el_order[parent];
                        //    for (int k = 0; k < tabrow.Size(); k++)
                        //    {
                        //       int fine_el = tabrow[k];
                        //       fespace->SetElementOrder(fine_el, parent_order);
                        //    }
                        // }
                     }
                  }
               }

               // Update
               if (fespace->DidOrderChange())
               {
                  fespace->Update(false);
                  surf_fit_fes.CopySpaceElementOrders(*fespace);
                  surf_fit_fes.SetRelaxedHpConformity(fespace->GetRelaxedHpConformity());
                  preft_fespace.Transfer(x);
                  preft_fespace.Transfer(x0);
                  preft_surf_fit_fes.Transfer(surf_fit_mat_gf);
                  preft_surf_fit_fes.Transfer(surf_fit_gf0);
                  surf_fit_marker.SetSize(surf_fit_gf0.Size());
               }
            }

            order_gf = x.FESpace()->GetElementOrdersV();

            // visualize
            if (visualization)
            {
               x_max_order = ProlongToMaxOrder(&x, 0);
               mesh->SetNodalGridFunction(x_max_order);
               socketstream vis1;
               // common::VisualizeField(vis1, "localhost", 19916, surf_fit_gf0,
               //                      "Gridfunction on optimal mesh",
               //                      325*(iter_pref+1), 650, 300, 300, vis_keys);

               // common::VisualizeField(vis1, "localhost", 19916, order_gf,
               //                        "Orders on p-derefined mesh",
               //                        325*(iter_pref+1), 650, 300, 300, vis_keys);

               mesh->SetNodalGridFunction(&x);
               delete x_max_order;
            }
            // if (iter_pref == 2) { MFEM_ABORT(" "); }

            // Now derefine in h
            {
               Vector error_fine_mesh(mesh->GetNE());
               error_fine_mesh = 1.0;

               for (int i = 0; i < ifec0.ngroups; i++)
               {
                  double p_ref_benefit = (current_group_error(i) - group_error_p(i)) /
                                         (group_dofs_p(i) - current_group_dofs(i));
                  double h_ref_benefit = (current_group_error(i) - group_error_h(i)) /
                                         (group_dofs_h(i)-current_group_dofs(i));

                  if (p_ref_benefit >= h_ref_benefit)
                  {
                     vector<int> els = ifec0.group_to_els[i];
                     for (int j = 0; j < els.size(); j++)
                     {
                        int el = els[j];
                        Array<int> tabrow;
                        coarse_to_fine.GetRow(el, tabrow);
                        for (int k = 0; k < tabrow.Size(); k++)
                        {
                           int fine_el = tabrow[k];
                           if (done_elsh(fine_el) == 0.0)
                           {
                              error_fine_mesh(fine_el) = 0.0;
                           }
                        }
                     }
                  }
               }

               double threshold = 0.5;
               mesh->DerefineByError(error_fine_mesh, threshold, 0, 1);

               // Update
               hrefup.Update(true);
            }
            if (visualization)
            {
               x_max_order = ProlongToMaxOrder(&x, 0);
               mesh->SetNodalGridFunction(x_max_order);
               socketstream vis1;
               // common::VisualizeField(vis1, "localhost", 19916, surf_fit_gf0,
               //                      "Gridfunction on optimal mesh",
               //                      325*(iter_pref+1), 650, 300, 300, vis_keys);

               // common::VisualizeField(vis1, "localhost", 19916, order_gf,
               //                        "Orders on h-derefined mesh",
               //                        325*(iter_pref+1), 650, 300, 300, vis_keys);

               mesh->SetNodalGridFunction(&x);
               delete x_max_order;
            }

            order_gf = x.FESpace()->GetElementOrdersV();
         }

         min_detJ = GetMinDet(mesh, fespace, irules, quad_order);
         cout << "Minimum det(J) after implementing hp-derefinements is " << min_detJ <<
              endl;
         MFEM_VERIFY(min_detJ > 0.0, "Non-positive Jacobian after hp-derefinements");
      }

      // Now we mark elements as done if there error is below threshold
      {
         GetGroupInfo(mesh, mat, surf_fit_gf0, ifec0);
         SetGroupNumGf(ifec0, groupnum_gf);
         x_max_order = ProlongToMaxOrder(&x, 0);
         mesh->SetNodalGridFunction(x_max_order);
         surf_fit_gf0_max_order = ProlongToMaxOrder(&surf_fit_gf0, 0);
         ComputeIntegratedErrorBGonFaces(x_max_order->FESpace(),
                                         surf_fit_bg_gf0,
                                         ifec0.inter_faces,
                                         surf_fit_gf0_max_order,
                                         finder,
                                         ifec0.face_error_map,
                                         error_type);

         GetGroupError(mesh, ifec0, current_group_max_error,
                       current_group_error, current_group_dofs);

         // mark all elements as done
         done_els = 1.0;
         done_elsh = 1.0;
         done_elsp = 1.0;

         // now mark all elements at interface as done with error below threshold
         for (int i = 0; i < ifec0.ngroups; i++)
         {
            auto group_faces = ifec0.group_to_face[i];
            double max_face_error = 0.0;
            for (int f = 0; f < group_faces.size(); f++)
            {
               int fnum = group_faces[f];
               max_face_error = std::max(max_face_error, ifec0.face_error_map[fnum]);
            }
            vector<int> els = ifec0.group_to_els[i];
            for (int j = 0; j < els.size(); j++)
            {
               int elnum = els[j];
               if (max_face_error < pref_tol)
               {
                  done_els[elnum] = 1.0;
                  done_elsp[elnum] = 1.0;
                  done_elsh[elnum] = 1.0;
               }
               else
               {
                  done_els[elnum] = 0.0;
                  done_elsp[elnum] = 0.0;
                  done_elsh[elnum] = 0.0;
               }
               // done_els[elnum] = max_face_error < pref_tol ? 1.0 : 0.0;
               // mark an element done for p-refinement if it is at max order
               int current_order = fespace->GetElementOrder(elnum);
               int current_depth = mesh->ncmesh->GetElementDepth(elnum);
               if (current_order == pref_max_order || !prefine)
               {
                  done_elsp(elnum) = 1.0;
               }
               if (current_depth == href_max_depth || !hrefine)
               {
                  done_elsh(elnum) = 1.0;
               }
               done_els(elnum) = min(done_elsp(elnum), done_elsh(elnum));
            }
         }
         mesh->SetNodalGridFunction(&x);
         delete x_max_order;
         delete surf_fit_gf0_max_order;

         // visualize
         if (visualization)
         {
            x_max_order = ProlongToMaxOrder(&x, 0);
            mesh->SetNodalGridFunction(x_max_order);
            socketstream vis1, vis2;
            common::VisualizeField(vis1, "localhost", 19916, done_els,
                                   "Done elements",
                                   325*(iter_pref+1), 650, 300, 300, vis_keys);
            // common::VisualizeField(vis2, "localhost", 19916, done_elsp,
            //                        "p-Done elements",
            //                        325*(iter_pref+1), 650, 300, 300, vis_keys);
            mesh->SetNodalGridFunction(&x);
            delete x_max_order;
         }
      }

      // visualize
      if (visualization)
      {
         x_max_order = ProlongToMaxOrder(&x, 0);
         mesh->SetNodalGridFunction(x_max_order);
         socketstream vis1, vis2;
         // common::VisualizeField(vis1, "localhost", 19916, surf_fit_gf0,
         //                      "Gridfunction on optimal mesh",
         //                      325*(iter_pref+1), 650, 300, 300, vis_keys);

         common::VisualizeField(vis1, "localhost", 19916, order_gf,
                                "Orders on optimal mesh",
                                325*(iter_pref+1), 650, 300, 300, vis_keys);
         common::VisualizeField(vis2, "localhost", 19916, mat,
                                "Final materials on optimal mesh",
                                325*(iter_pref+1), 325, 300, 300, vis_keys);


         mesh->SetNodalGridFunction(&x);
         delete x_max_order;
      }

      std::cout << done_elsh.Sum() << " " << done_elsp.Sum() << " " << mesh->GetNE()
                << " k101\n";
      if (done_elsh.Sum() == mesh->GetNE() && done_elsp.Sum() == mesh->GetNE())
      {
         faces_to_update = false;
      }

      // {
      //    // Reduce the orders of the elements if needed
      //    if (reduce_order && iter_pref > 0 && pref_order_increase > 1)
      //    {
      //       MFEM_ABORT("trying to derefine in order increase\n");
      //       int compt_updates(0);
      //       for (int i=0; i < ifecref.inter_faces.Size(); i++)
      //       {
      //          int el1 = ifecref.inter_face_el1[i];
      //          int el2 = ifecref.inter_face_el2[i];

      //          Array<int> els;
      //          int el_order = fespace->GetElementOrder(el1);
      //          double interface_error = ComputeIntegrateErrorBG(x_max_order->FESpace(),
      //                                                           surf_fit_bg_gf0,
      //                                                           ifecref.inter_faces[i],
      //                                                           surf_fit_gf0_max_order,
      //                                                           finder,
      //                                                           deref_error_type);

      //          double coarsened_face_error;
      //          int orig_order = el_order;
      //          int target_order = el_order;
      //          bool trycoarsening = true;
      //          while (el_order >= mesh_poly_deg+1 && trycoarsening &&
      //                 ( (pref_tol > 0 && pderef > 0 && interface_error < pref_tol) ||
      //                   (pref_tol < 0 && pderef > 0 &&
      //                    interface_error < (-pref_tol)*max_initial_face_error) ||
      //                   (pderef < 0) ) )
      //          {
      //             coarsened_face_error = InterfaceElementOrderReduction(mesh,
      //                                                                   ifecref.inter_faces[i], el_order-1, surf_el_meshes,
      //                                                                   surf_fit_bg_gf0, finder, deref_error_type);
      //             trycoarsening = false;
      //             double ref_threshold = pref_tol > 0 ?
      //                                    pref_tol :
      //                                    (-pref_tol)*max_initial_face_error;
      //             double deref_threshold = pderef*ref_threshold;

      //             // when derefined, the length of the element will decrease and
      //             // the level set function integral will increase
      //             if ( (pderef > 0 && coarsened_face_error < deref_threshold) ||
      //                  (pderef < 0 && deref_error_type == 2 &&
      //                   coarsened_face_error < (1-pderef)*(interface_error)) ||
      //                  //relative to change in error due to refinement
      //                  (pderef < 0 && deref_error_type == 4 &&
      //                   coarsened_face_error < -pderef*pref_tol) ||
      //                  (pderef < 0 && deref_error_type == 1 &&
      //                   coarsened_face_error > (1+pderef)*(interface_error)) //relative change in length
      //                )
      //             {
      //                trycoarsening = CheckElementValidityAtOrder(mesh, el1, el_order);
      //                if  (trycoarsening)
      //                {
      //                   trycoarsening = CheckElementValidityAtOrder(mesh, el2, el_order);
      //                }
      //                if (trycoarsening)
      //                {
      //                   el_order -= 1;
      //                   target_order = el_order;
      //                }
      //             }
      //          }

      //          if (target_order != orig_order)
      //          {
      //             order_gf(el1) = target_order;
      //             order_gf(el2) = target_order;
      //             compt_updates++;
      //          }
      //       } //i < inter_faces.Size()

      //       // Update the FES and GridFunctions only if some orders have been changed
      //       if (compt_updates > 0)
      //       {
      //          std::cout << "=======================================\n";
      //          std::cout << "# Derefinements: " << compt_updates << std::endl;
      //          std::cout << "=======================================\n";
      //          // Propogate error first
      //          Array<int> new_orders;
      //          PropogateOrders(order_gf, ifecref.inter_face_el_all,
      //                          adjacent_el_diff,
      //                          eltoeln, new_orders, 1);

      //          for (int e = 0; e < mesh->GetNE(); e++)
      //          {
      //             bool validity = CheckElementValidityAtOrder(mesh, e, new_orders[e]);
      //             if (validity)
      //             {
      //                order_gf(e) = new_orders[e];
      //                fespace->SetElementOrder(e, order_gf(e));
      //             }
      //             else
      //             {
      //                std::cout << e << " invalid derefinement\n";
      //             }
      //          }

      //          // make consistent for edges in 3D
      //          if (dim == 3)
      //          {
      //             for (int e = 0; e < mesh->GetNE(); e++)
      //             {
      //                double max_order = -1.0;
      //                Array<int> *temp = ifecref.el_to_el_edge[e];
      //                if (temp)
      //                {

      //                   for (int ee = 0; ee < temp->Size(); ee++)
      //                   {
      //                      max_order = std::max(max_order,
      //                                           order_gf((*temp)[ee]));
      //                   }
      //                }
      //                if (max_order != -1.0)
      //                {
      //                   fespace->SetElementOrder(e, int(max_order));
      //                }
      //             }
      //          }

      //          min_detJ = GetMinDet(mesh, fespace, irules, quad_order);
      //          cout << "Minimum det(J) of the mesh after coarsening is " << min_detJ << endl;
      //          MFEM_VERIFY(min_detJ > 0, "Mesh has somehow become inverted "
      //                      "due to coarsening");
      //          {
      //             ofstream mesh_ofs("optimized_" +std::to_string(iter_pref)+std::to_string(
      //                                  jobid)+ ".mesh");
      //             mesh_ofs.precision(14);
      //             mesh->Print(mesh_ofs);
      //          }

      //          PRefinementTransfer preft_fespace = PRefinementTransfer(*fespace);
      //          PRefinementTransfer preft_surf_fit_fes = PRefinementTransfer(surf_fit_fes);
      //          // Updates if we increase the order of at least one element
      //          fespace->Update(false);
      //          surf_fit_fes.CopySpaceElementOrders(*fespace);
      //          surf_fit_fes.SetRelaxedHpConformity(fespace->GetRelaxedHpConformity());
      //          preft_fespace.Transfer(x);
      //          preft_fespace.Transfer(x0);
      //          preft_surf_fit_fes.Transfer(surf_fit_mat_gf);
      //          preft_surf_fit_fes.Transfer(surf_fit_gf0);
      //          surf_fit_marker.SetSize(surf_fit_gf0.Size());

      //          x.SetTrueVector();
      //          x.SetFromTrueVector();

      //          mesh->SetNodalGridFunction(&x);

      //          delete x_max_order;
      //          x_max_order = ProlongToMaxOrder(&x, 0);
      //          delete surf_fit_gf0_max_order;
      //          surf_fit_gf0_max_order = ProlongToMaxOrder(&surf_fit_gf0, 0);

      //          mesh->SetNodalGridFunction(x_max_order);
      //          ComputeIntegratedErrorBGonFaces(x_max_order->FESpace(),
      //                                          surf_fit_bg_gf0,
      //                                          ifecref.inter_faces,
      //                                          surf_fit_gf0_max_order,
      //                                          finder,
      //                                          ifecref.face_error_map);
      //          double error_sum = 0.0;
      //          for (const auto& pair : ifecref.face_error_map)
      //          {
      //             error_sum += pair.second;
      //          }
      //          sum_current_face_error = error_sum;

      //          std::cout << "NDofs & Integrated Error Post Derefinement: " <<
      //                    fespace->GetVSize() << " " <<
      //                    error_sum << " " << std::endl;
      //          mesh->SetNodalGridFunction(&x);
      //       } //compt_updates
      //    } //reduction
      // } //check error and reduction after fitting

      order_gf = x.FESpace()->GetElementOrdersV();
      if (visit)
      {
         mesh->SetNodalGridFunction(&x);
         x_max_order = ProlongToMaxOrder(&x, 0);
         mesh->SetNodalGridFunction(x_max_order);
         surf_fit_gf0_max_order = ProlongToMaxOrder(&surf_fit_gf0, 0);

         DataCollection *dc = NULL;
         dc = new VisItDataCollection("Optimized_"+std::to_string(jobid), mesh);
         dc->RegisterField("orders", &order_gf);
         dc->RegisterField("Level-set", surf_fit_gf0_max_order);
         dc->RegisterField("mat", &mat);
         dc->RegisterField("intmarker", &int_marker);
         dc->SetCycle(2*iter_pref+2);
         dc->SetTime(2.0*iter_pref+2.0);
         dc->Save();
         delete dc;
      }

      if (visualization && iter_pref > 0)
      {
         // socketstream vis1, vis2;
         // common::VisualizeField(vis2, "localhost", 19916, order_gf,
         //                        "Polynomial order after deref",
         //                        325*(iter_pref+1), 0, 300, 300, vis_keys);
      }

      mesh->SetNodalGridFunction(&x); // Need this for the loop
      delete x_max_order;
      delete surf_fit_gf0_max_order;

      iter_pref++; // Update the loop iterator

      delete S;
      delete S_prec;
      std::cout << "=======================================\n";
   }

   // Get number of surface DOFs
   surf_fit_mat_gf = 0.0;
   surf_fit_mat_gf.SetTrueVector();
   surf_fit_mat_gf.SetFromTrueVector();
   for (int i = 0; i < surf_fit_mat_gf.Size(); i++)
   {
      surf_fit_mat_gf[i] = surf_fit_marker[i];
   }
   surf_fit_mat_gf.SetTrueVector();
   Vector surf_fit_mat_gf_tvec = surf_fit_mat_gf.GetTrueVector();
   //   surf_fit_mat_gf_tvec.Print();
   int nsurfdofs = 0;
   for (int i = 0; i < surf_fit_mat_gf_tvec.Size(); i++)
   {
      nsurfdofs += (int)surf_fit_mat_gf_tvec(i);
   }

   int type = (pref_max_order > mesh_poly_deg && pref_max_iter > 1);
   if (type == 1 && !reduce_order)
   {
      type = 2;
   }
   std::cout <<
             "Final info: Type,rs,order,NDofs,TDofs,Error,PreNDofs,PreTDofs" <<
             ",PreError,pref_tol,mo,dp,nelem,maxNDofs,error_type,pref_tol,deref_error_type,pderef,PreNSurfDofs,NSurfDofs : "
             <<
             type << "," <<
             rs_levels << "," <<
             mesh_poly_deg << "," <<
             fespace->GetVSize() << "," <<
             fespace->GetTrueVSize() << "," <<
             sum_current_face_error << "," <<
             predef_ndofs << "," <<
             predef_tdofs << "," <<
             prederef_error << "," <<
             pref_tol << "," <<
             pref_max_order << "," <<
             adjeldiff << "," <<
             mesh->GetNE() << "," <<
             x_max_order->Size() << "," <<
             error_type << "," <<
             pref_tol << "," <<
             deref_error_type << "," <<
             pderef << "," <<
             predef_nsurfdofs << "," <<
             nsurfdofs << "," <<
             std::endl;

   Array<int> el_by_order(x.FESpace()->GetMaxElementOrder()+1);
   el_by_order = 0;
   for (int e = 0; e < mesh->GetNE(); e++)
   {
      int el_order = x.FESpace()->GetElementOrder(e);
      el_by_order[el_order] += 1;
   }
   if (exceptions == 4)
   {
      el_by_order = 0;
      for (int e = 0; e < mesh->GetNE(); e++)
      {
         int el_order = x.FESpace()->GetElementOrder(e);
         if (mat(e) == 1.0)
         {
            el_by_order[el_order] += 1;
         }
      }
   }
   std::cout << "Print number of elements by order\n";
   el_by_order.Print();
   std::cout << "Total elements: " << el_by_order.Sum() << std::endl;

   for (int i = 0; i < el_by_order.Size(); i++)
   {
      el_by_order[i] = 100*el_by_order[i]/el_by_order.Sum();
   }
   std::cout << "Print % number of elements by order\n";
   el_by_order.Print();


   if (visualization)
   {
      x_max_order = ProlongToMaxOrder(&x, 0);
      mesh->SetNodalGridFunction(x_max_order);
      socketstream vis1, vis2, vis3;
      order_gf = x.FESpace()->GetElementOrdersV();
      // common::VisualizeField(vis1, "localhost", 19916, order_gf,
      //                           "Polynomial order after p-refinement",
      //                           1100, 650, 300, 300, vis_keys);
      // common::VisualizeField(vis2, "localhost", 19916, mat, "Materials after fitting",
      //                        1200, 650, 300, 300, vis_keys);

      mesh->SetNodalGridFunction(&x);
      delete x_max_order;
      if (surf_bg_mesh)
      {
         common::VisualizeField(vis2, "localhost", 19916, *surf_fit_bg_grad,
                                "Background Mesh - Level Set Gradrient",
                                0, 00, 300, 300, vis_keys);
         common::VisualizeField(vis3, "localhost", 19916, *surf_fit_bg_gf0,
                                "Background Mesh - Level Set",
                                0, 0, 300, 300, vis_keys);
      }
   }

   mesh->SetNodalGridFunction(&x);

   finder.FreeData();

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
   delete fespace;
   delete fec;
   delete mesh_surf_fit_bg;
   delete mesh;

   return 0;
}
