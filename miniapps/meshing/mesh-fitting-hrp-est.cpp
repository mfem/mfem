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
// make mesh-fitting-hrp-est -j && ./mesh-fitting-hrp-est -m square01-tri.mesh -rs 1 -o 1 -sbgmesh -vl 2 -mo 4 -mi 4 -preft 5e-14 -lsf 1 -bgamr 4 -bgo 6 -det 0 -pderef 1e-4 -href -no-ro -ni 100 -mid 2 -sft 1e-20 -et 0
// make mesh-fitting-hrp-est -j && ./mesh-fitting-hrp-est -m quadsplit.mesh -rs 0 -o 1 -sbgmesh -vl 2 -mo 6 -mi 4 -preft 1e-14 -lsf 1 -bgamr 5 -bgo 6 -det 0 -pderef 1e-4 -href -no-ro -ni 100 -mid 80 -tid 4 -sft 1e-20 -et 0 -cus-mat

// reactor
// make mesh-fitting-hrp-est -j && ./mesh-fitting-hrp-est -m square01-tri.mesh -rs 2 -o 1 -sbgmesh -vl 2 -mo 4 -mi 3 -preft 1e-10 -lsf 5 -bgamr 0 -bgrs 4 -bgo 4 -det 0 -pderef -1e-4 -href -no-ro -ni 100 -mid 2 -tid 1 -sft 1e-20 -et 0 -sfc 1e1 -exc 5 -hpt 0.001
// make mesh-fitting-hrp-est -j && ./mesh-fitting-hrp-est -m square01-tri.mesh -rs 2 -o 1 -sbgmesh -vl 2 -mo 4 -mi 5 -preft 1e-10 -lsf 5 -bgamr 4 -bgrs 2 -bgo 4 -det 0 -pderef -1e-4 -href -no-ro -ni 200 -mid 2 -tid 1 -sft 1e-20 -et 0 -sfc 1e3 -sfa 10 -exc 5 -hpt 0.2 -refint 1 -remark -no-vis

// mickey-mouse
// make mesh-fitting-hrp-est -j && ./mesh-fitting-hrp-est -m square01-tri.mesh -rs 2 -o 1 -sbgmesh -vl 2 -mo 4 -mi 3 -preft 1e-10 -lsf 7 -bgamr 0 -bgrs 4 -bgo 4 -det 0 -pderef -1e-4 -href -no-ro -ni 200 -mid 2 -tid 1 -sft 1e-20 -et 0 -sfc 0.01 -sfa 5 -exc 5 -hpt 0.01 -no-int
#include "../../mfem.hpp"
#include "../common/mfem-common.hpp"
#include <fstream>
#include <iostream>
#include <chrono>
#include "mesh-fitting.hpp"
#include "mesh-fitting-pref.hpp"

// Error types:
// 0 - L2squared of level-set function.
// 1 - element size/length
// 2 - square root of L2squared of level-set function.
// 3 - square root of length
// 4 - same as 0
// 5 - max pointwise error
// 6 - sum of pointwise error

using namespace mfem;
using namespace std;

char vishost[] = "localhost";
int  visport   = 19916;
int  wsize     = 350;

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



void GetElementNodalLocations(GridFunction *gf, int el_id, Vector &nodes)
{
   const FiniteElementSpace *fes = gf->FESpace();
   const FiniteElement *fe = fes->GetFE(el_id);
   const IntegrationRule &ir = fe->GetNodes();
   ElementTransformation *trans = fes->GetElementTransformation(el_id);
   Mesh *mesh = gf->FESpace()->GetMesh();
   const int npts = ir.GetNPoints();
   const int dim = fes->GetMesh()->SpaceDimension();
   const int ndofs = fe->GetDof();
   nodes.SetSize(ndofs*dim);
   DenseMatrix nodes_mat(nodes.GetData(), ndofs, dim);
   mesh->GetNodes()->GetVectorValues(*trans, ir, nodes_mat);
}

void GetElementQuadPointLocations(GridFunction *gf, int el_id,
                                  IntegrationRule &ir,
                                  Vector &nodes)
{
   const FiniteElementSpace *fes = gf->FESpace();
   const FiniteElement *fe = fes->GetFE(el_id);
   ElementTransformation *trans = fes->GetElementTransformation(el_id);
   Mesh *mesh = gf->FESpace()->GetMesh();
   const int npts = ir.GetNPoints();
   const int dim = fes->GetMesh()->SpaceDimension();
   nodes.SetSize(npts*dim);
   DenseMatrix nodes_mat(nodes.GetData(), npts, dim);
   mesh->GetNodes()->GetVectorValues(*trans, ir, nodes_mat);
}

void MapJumpEstimateFromBackgroundMesh(GridFunction *bg_jump,
                                       IntegrationRules *irules,
                                       int quad_order,
                                       InterfaceFaceEdgeConnectivity &ifec0,
                                       GridFunction &jump_estimator)
{
   Mesh *mesh = jump_estimator.FESpace()->GetMesh();
   Mesh *mesh_surf_fit_bg = bg_jump->FESpace()->GetMesh();
   FindPointsGSLIB finder_current;
   finder_current.Setup(*mesh);
   // First map jump to all elements
   for (int i = 0; i < mesh_surf_fit_bg->GetNE(); i++)
   {
      Vector elnodes;
      Vector intvals;
      Array<int> eldofs;
      IntegrationRule iruleqp = irules->Get(bg_jump->FESpace()->GetFE(i)->GetGeomType(),
                                             quad_order);
      GetElementQuadPointLocations(bg_jump, i, iruleqp, elnodes);
      const FiniteElement *fe = bg_jump->FESpace()->GetFE(i);
      bg_jump->GetValues(i, iruleqp, intvals);
      finder_current.FindPoints(elnodes, Ordering::byVDIM);
      Array<int> mfem_elem = finder_current.GetElem();
      for (int e = 0; e < mfem_elem.Size(); e++)
      {
         int elem = mfem_elem[e];
         jump_estimator(elem) = std::max(jump_estimator(elem), intvals(e));
      }
   }

   H1_FECollection fec_lin_H1(1, mesh->Dimension());
   FiniteElementSpace  fes_lin_h1(mesh, &fec_lin_H1);
   GridFunction lin_h1(&fes_lin_h1);
   lin_h1 = 0.0;

   // now map jump to all DOFs
   for (int e = 0; e < mesh->GetNE(); e++)
   {
      Array<int> dofs;
      fes_lin_h1.GetElementDofs(e, dofs);
      for (int i = 0; i < dofs.Size(); i++)
      {
         lin_h1(dofs[i]) = std::max(lin_h1(dofs[i]), jump_estimator(e));
      }
   }

   // now map back to all elements
   for (int e = 0; e < mesh->GetNE(); e++)
   {
      Array<int> dofs;
      fes_lin_h1.GetElementDofs(e, dofs);
      for (int i = 0; i < dofs.Size(); i++)
      {
         jump_estimator(e) = std::max(lin_h1(dofs[i]), jump_estimator(e));
      }
   }

   for (int e = 0; e < mesh->GetNE(); e++)
   {
      if (ifec0.inter_face_el_all.Find(e) == -1 &&
         ifec0.intedgels.Find(e) == -1)
      {
         jump_estimator(e) = 0.0;
      }
   }


   // Make error consistent for each group
   for (int i = 0; i < ifec0.ngroups; i++)
   {
      vector<int> els = ifec0.group_to_els[i];
      double max_group_err = 0.0;
      for (int e = 0; e < els.size(); e++)
      {
         max_group_err = std::max(max_group_err, jump_estimator(els[e]));
      }
      for (int e = 0; e < els.size(); e++)
      {
         jump_estimator(els[e]) = max_group_err;
      }
   }

   finder_current.FreeData();
}

void ExtendRefinementListToNeighbors(Mesh &mesh, Array<int> &intel)
{
   mfem::L2_FECollection l2fec(0, mesh.Dimension());
   mfem::FiniteElementSpace l2fespace(&mesh, &l2fec);
   mfem::GridFunction el_to_refine(&l2fespace);
   const int quad_order = 4;

   el_to_refine = 0.0;

   for (int i = 0; i < intel.Size(); i++)
   {
      el_to_refine(intel[i]) = 1.0;
   }

   mfem::H1_FECollection lhfec(1, mesh.Dimension());
   mfem::FiniteElementSpace lhfespace(&mesh, &lhfec);
   mfem::GridFunction lhx(&lhfespace);

   // el_to_refine.ExchangeFaceNbrData();
   GridFunctionCoefficient field_in_dg(&el_to_refine);
   lhx.ProjectDiscCoefficient(field_in_dg, GridFunction::ARITHMETIC);

   IntegrationRules irRules = IntegrationRules(0, Quadrature1D::GaussLobatto);
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      Array<int> dofs;
      Vector x_vals;
      lhfespace.GetElementDofs(e, dofs);
      const IntegrationRule &ir =
         irRules.Get(mesh.GetElementGeometry(e), quad_order);
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

// Propogate orders all members of group.
void PropogateOrdersToGroup(GridFunction &ordergf, //current orders
                            InterfaceFaceEdgeConnectivity &intfc,
                            Array<int> &int_el_list)
{
   int_el_list.SetSize(0);
   // First set orders based on element groups.
   Array<int> inter_face_el_all = intfc.inter_face_el_all;
   for (int g = 0; g < intfc.ngroups; g++)
   {
      for (int i = 0; i < intfc.group_to_els[g].size(); i++)
      {
         int_el_list.Append(intfc.group_to_els[g][i]);
      }
      int elnum = intfc.group_to_els[g][0];
      int group_order = ordergf(elnum);
      for (int i = 0; i < intfc.group_to_edg_els[g].size(); i++)
      {
         int edgel = intfc.group_to_edg_els[g][i];
         ordergf(edgel) = group_order;
         int_el_list.Append(edgel);
      }
   }
}

// Prolongs mesh nodes to max orders and sets the mesh nodes. Also gives mesh
// the ownership so that they are automatically deleted later.
GridFunction* ProlongToMaxOrderAndSetMeshNodes(Mesh *mesh)
{
   GridFunction *x = mesh->GetNodes();
   auto tempx = ProlongToMaxOrder(x, 0);
   mesh->NewNodes(*tempx, true);
   return mesh->GetNodes();
}

void MeasureJumps(const GridFunction &u, GridFunction &jumps, bool vis)
{
   const FiniteElementSpace &pfes_L2 = *(u.FESpace());
   const int dim   = pfes_L2.GetMesh()->Dimension();
   const int order = pfes_L2.FEColl()->GetOrder();


   // Normalize to [0, 1].
   GridFunction u_n(u);
   double u_min = u_n.Min();
   u_n += fabs(u_min);
   double u_max = u_n.Max();
   u_n /= u_max;

   if (vis)
   {
      socketstream sock_un;
      common::VisualizeField(sock_un, vishost, visport, u_n, "u normalized",
                             wsize, wsize, wsize, wsize, "Rjmc**");
   }

   // Form min/max at each CG dof, considering element overlaps.
   const L2_FECollection *fecl2 = dynamic_cast<const L2_FECollection *>(pfes_L2.FEColl());
   H1_FECollection fec_H1(order, dim);

   FiniteElementSpace pfes_H1(pfes_L2.GetMesh(), &fec_H1);

   // std::cout << fecl2->GetBasisType() << " " << fec_H1.GetBasisType() << " "
   // << order << " " << dim << " k10-basis-type\n";
   const IntegrationRule &irulel2 = pfes_L2.GetFE(0)->GetNodes();
   const IntegrationRule &iruleh1 = pfes_H1.GetFE(0)->GetNodes();
   // std::cout << irulel2.GetNPoints() << " " << iruleh1.GetNPoints() << " k10np\n";
   for (int i = 0; i < iruleh1.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = iruleh1.IntPoint(i);
      const IntegrationPoint &ip2 = irulel2.IntPoint(i);
      // std::cout << ip.x << " " << ip.y << " " <<
      // ip2.x << " " << ip2.y << " k10-ip\n";
   }


   GridFunction u_min_H1(&pfes_H1), u_max_H1(&pfes_H1);
   u_min_H1 =   std::numeric_limits<double>::infinity();
   u_max_H1 = - std::numeric_limits<double>::infinity();
   const int NE = pfes_H1.GetNE();
   Array<int> dofsCG;
   Array<int> dofsDCG;
   const TensorBasisElement *fe_tensor =
      dynamic_cast<const TensorBasisElement *>(pfes_H1.GetFE(0));
   const NodalFiniteElement *fe_nodal =
      dynamic_cast<const NodalFiniteElement *>(pfes_H1.GetFE(0));
   // std::cout << fe_tensor << " " << fe_nodal << " k10-fetensor-fenodal\n";
   MFEM_VERIFY(fe_tensor || fe_nodal, "TODO - implement for triangles");
   const Array<int> &dof_map2 = fe_nodal->GetLexicographicOrdering();
   const Array<int> &dof_map = fe_tensor->GetDofMap();
   const int ndofs = dof_map2.Size();
   // dof_map2.Print();
   for (int k = 0; k < NE; k++)
   {
      pfes_H1.GetElementDofs(k, dofsCG);
      pfes_L2.GetElementDofs(k, dofsDCG);
      for (int i = 0; i < ndofs; i++)
      {
         // if (k == 0) {
         //    std::cout << i << " " << dof_map2[i] << " k10map2\n";
         // }
         u_min_H1(dofsCG[dof_map2[i]]) = fmin(u_min_H1(dofsCG[dof_map2[i]]),
                                             u_n(k*ndofs + i));
         u_max_H1(dofsCG[dof_map2[i]]) = fmax(u_max_H1(dofsCG[dof_map2[i]]),
                                             u_n(k*ndofs + i));
      }
   }

   // Compute jumps (and reuse the min H1 function).
   GridFunction &u_jump_H1 = u_min_H1;
   for (int i = 0; i < u_jump_H1.Size(); i++)
   {
      u_jump_H1(i) = fabs(u_max_H1(i) - u_min_H1(i));
   }
   if (vis)
   {
      socketstream sock_j;
      common::VisualizeField(sock_j, vishost, visport, u_jump_H1, "u jumps HO",
                             2*wsize, wsize, wsize, wsize, "Rjmc**");
   }

   // Project the jumps to Q1.
   jumps.ProjectGridFunction(u_jump_H1);
   for (int i = 0; i < jumps.Size(); i++) { jumps(i) = fmax(jumps(i), 0.0); }
}

class GradientCompCoefficient : public Coefficient
{
private:
   const GridFunction &u;
   const int comp;

public:
   GradientCompCoefficient(const GridFunction &u_gf, int c)
      : u(u_gf), comp(c) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

double GradientCompCoefficient::Eval(ElementTransformation &T,
                                     const IntegrationPoint &ip)
{
   Vector grad_u(T.GetDimension());
   u.GetGradient(T, grad_u);
   return grad_u(comp);
}

int main(int argc, char *argv[])
{
   // 0. Set the method's default parameters.
   const char *mesh_file = "icf.mesh";
   int mesh_poly_deg     = 1;
   int rs_levels         = 0;
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
   int pref_order_increase = -1;
   int pref_max_order      = 4;
   int pref_max_iter       = 2;
   double pref_tol         = 1e-13;
   bool surf_bg_mesh       = true;
   bool reduce_order       = true;
   const char *bg_mesh_file = "NULL";
   const char *bg_ls_file = "NULL";
   bool custom_material   = false;
   const char *custom_material_file = "NULL"; //material coming from a gf file
   bool adapt_marking = true;
   int bg_amr_iter = 0;
   int ls_function = 0;
   int bg_rs_levels = 1;
   int jobid  = 0;
   bool visit    = true;
   int adjeldiff = 1;
   int bgo = 4;
   int custom_split_mesh = 0;
   bool mod_bndr_attr    = false;
   int exceptions    = 0; //to do some special stuff.
   int error_type = 0;
   double pderef = 0.0;
   int deref_error_type = 0;
   bool hrefine = false;
   int href_max_depth = 4;
   bool relaxed_hp = false;
   double hpthreshold = 0.9;
   bool integral = false;
   int ref_int_neighbors = 0;
   bool amr_remarking = false;

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
   args.AddOption(&pderef, "-pderef", "--pderef",
                  "If greater than 0, we try to p-derefine based on the deref_error_type "
                  "crietion");
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
   args.AddOption(&hpthreshold, "-hpt", "--hpt",
                  "Threshold to choose for h vs p refinement");
   args.AddOption(&integral, "-int", "--integral",
                  "-no-int", "--no-integral",
                  "Use integral term for fitting term.");
   args.AddOption(&ref_int_neighbors, "-refint", "--refint",
                  "Layers of neighbors to refine");
   args.AddOption(&amr_remarking, "-remark", "--re-mark", "-no-remark",
                  "--no-re-mark",
                  "Remark after adaptive mesh refinement.");
   // if pderef > 0, we check for derefinement based on change in length (deref_error_type = 1),
   // change in L2 error (deref_error_type = 0), and relative to pref_tol (deref_error_type = 4)
   // It is best to keep error_type = 0 so that criterion is based on L2 error.

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);
   MFEM_VERIFY(pref_max_order > 0, "pref_max_order should be greater than 0.");
   MFEM_VERIFY(pref_order_increase < 0,
               "For now we assume that pref_order_increase "
               "is negative so that we can set it internally "
               " to max_order - base_order");
   MFEM_VERIFY(pref_tol > 0, "P-refinement tolerance must be positive.\n");
   MFEM_VERIFY(error_type == 0,"Error type should be 0 for now.\n");

   if (pref_order_increase < 0)
   {
      pref_order_increase = pref_max_order - mesh_poly_deg;
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
   else if (ls_function == 4)
   {
      ls_coeff = FunctionCoefficient(reactor);
   }
   else if (ls_function == 5)
   {
      ls_coeff = FunctionCoefficient(reactoranalytic);
   }
   else if (ls_function == 6)
   {
      ls_coeff = FunctionCoefficient(squarewithcorners);
   }
   else if (ls_function == 7)
   {
      ls_coeff = FunctionCoefficient(mickymouseanalytic);
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

   // Define L2 space for storing piecewise constant functions.
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
   GridFunction previous_order_gf(&l2zero_fes);
   GridFunction jump_estimator(&l2zero_fes);
   GridFunction do_elsp(&l2zero_fes);
   GridFunction do_elsh(&l2zero_fes);

   order_gf = mesh_poly_deg*1.0;
   int_marker = 0.0;
   done_els = 1.0;
   done_elsp = 1.0;
   done_elsh = 1.0;
   mat = 0.0;
   NumFaces = 0.0;
   fitting_error_gf = pref_tol/100;
   groupnum_gf = -1.0;
   do_elsp = 0.0;
   do_elsh = 0.0;

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
   hrefup.AddGridFunctionForUpdate(&previous_order_gf);
   hrefup.AddGridFunctionForUpdate(&jump_estimator);
   hrefup.AddGridFunctionForUpdate(&do_elsp);
   hrefup.AddGridFunctionForUpdate(&do_elsh);


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
   x_max_order = ProlongToMaxOrder(&x, 0);
   delete x_max_order;


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


   H1_FECollection fec_lin_H1(1, dim);
   FiniteElementSpace *bg_fes_lin_H1 = NULL;
   GridFunction *bg_grad_jump = NULL;
   GridFunction *bg_hess_jump = NULL;
   GridFunction *bg_jump = NULL;

   hrefup.AddFESpaceForUpdate(&surf_fit_fes);
   hrefup.AddGridFunctionForUpdate(&surf_fit_mat_gf);
   hrefup.AddGridFunctionForUpdate(&surf_fit_gf0);

   // If a background mesh is used, we interpolate the Gradient and Hessian
   // from that mesh to the current mesh being optimized.
   FiniteElementSpace *surf_fit_grad_fes = NULL;
   GridFunction *surf_fit_grad = NULL;
   FiniteElementSpace *surf_fit_hess_fes = NULL;
   GridFunction *surf_fit_hess = NULL;
   FiniteElementSpace *fes_lin_H1 = NULL;
   GridFunction *grad_jump_lin_H1 = NULL;

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
         // DiffuseField(*surf_fit_bg_gf0, 1);
         if (bg_amr_iter > 0)
         {
            std::cout << "Doing AMR on the bg mesh\n";
            OptimizeMeshWithAMRAroundZeroLevelSet(*mesh_surf_fit_bg,
                                                  ls_coeff,
                                                  bg_amr_iter,
                                                  *surf_fit_bg_gf0);
            std::cout << "Done AMR on the bg mesh\n";
         }
         // DiffuseField(*surf_fit_bg_gf0, 1);
         finder.Setup(*mesh_surf_fit_bg);
         std::cout << "Done finder Setup on the bg mesh\n";
         surf_fit_bg_gf0->ProjectCoefficient(ls_coeff);
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

      // Compute jumps
      L2_FECollection fec_bg_grad_L2_temp(bgo, dim, BasisType::GaussLobatto);
      FiniteElementSpace fes_bg_grad_L2_temp(mesh_surf_fit_bg, &fec_bg_grad_L2_temp);
      GridFunction du_comp(&fes_bg_grad_L2_temp);
      GridFunction dudu_comp(&fes_bg_grad_L2_temp);

      bg_fes_lin_H1 = new FiniteElementSpace(mesh_surf_fit_bg, &fec_lin_H1);
      bg_grad_jump = new GridFunction(bg_fes_lin_H1);
      bg_hess_jump = new GridFunction(bg_fes_lin_H1);
      bg_jump = new GridFunction(bg_fes_lin_H1);
      GridFunction comp_jumps(bg_fes_lin_H1);
      GridFunction comp2_jumps(bg_fes_lin_H1);
      *bg_grad_jump = 0.0;
      *bg_hess_jump = 0.0;
      for (int d = 0; d < dim; d++)
      {
         GradientCompCoefficient du_comp_coeff(*surf_fit_bg_gf0, d);
         du_comp.ProjectCoefficient(du_comp_coeff);
         for (int c = 0; c < dim; c++)
         {
            GradientCompCoefficient d2u_comp_coeff(du_comp, c);
            dudu_comp.ProjectCoefficient(d2u_comp_coeff);
            MeasureJumps(dudu_comp, comp_jumps, false);
            for (int i = 0; i < bg_hess_jump->Size(); i++)
            {
               (*bg_hess_jump)(i) = fmax((*bg_hess_jump)(i), comp_jumps(i));
            }
         }
         MeasureJumps(du_comp, comp_jumps, false);
         for (int i = 0; i < bg_grad_jump->Size(); i++)
         {
            (*bg_grad_jump)(i) = fmax((*bg_grad_jump)(i), comp_jumps(i));
         }
      }
      DiffuseField(*bg_hess_jump, 4);
      DiffuseField(*bg_grad_jump, 4);

      if (visualization)
      {
         socketstream vis1, vis2, vis3, vis4;
         if (surf_bg_mesh)
         {
            common::VisualizeField(vis1, "localhost", 19916, *surf_fit_bg_gf0,
                                 "Background Mesh - Level Set",
                                 1300, 0, 300, 300, vis_keys);
            common::VisualizeField(vis2, "localhost", 19916, *surf_fit_bg_grad,
                                 "Background Mesh - Level Set Gradrient",
                                 1300, 300, 300, 300, vis_keys);
            common::VisualizeField(vis3, "localhost", 19916, *bg_grad_jump,
                                 "Background Mesh - Gradient jump",
                                 1300, 600, 300, 300, vis_keys);
            common::VisualizeField(vis4, "localhost", 19916, *bg_hess_jump,
                                 "Background Mesh - Hessian jump",
                                 1300, 900, 300, 300, vis_keys);
         }
      }

      *bg_jump = *bg_grad_jump;
      *bg_jump += *bg_hess_jump;
      double jumpmin = bg_jump->Min();
      double jumpmax = bg_jump->Max();
      *bg_jump -= jumpmin;
      *bg_jump /= (jumpmax - jumpmin);


      if (visit)
      {
         DataCollection *dc = NULL;
         dc = new VisItDataCollection("Background_"+std::to_string(jobid),
                                      mesh_surf_fit_bg);
         dc->RegisterField("Level-set", surf_fit_bg_gf0);
         dc->RegisterField("LS-Gradient", surf_fit_bg_grad);
         dc->RegisterField("LS-Gradient-Jump", bg_grad_jump);
         dc->RegisterField("LS-Hessian-Jump", bg_hess_jump);
         dc->RegisterField("LS-Final-Jump-Field", bg_jump);
         dc->SetCycle(0);
         dc->SetTime(0.0);
         dc->Save();
         delete dc;
      }
      // MFEM_ABORT(" ");
   }


   GridFunction *mat_file = NULL;
   if (surface_fit_const > 0.0)
   {
      finder.Interpolate(x, *surf_fit_bg_gf0, surf_fit_gf0,
                         x.FESpace()->GetOrdering());
      std::cout << "Done remap from bg mesh\n";

      if (custom_material && strcmp(custom_material_file, "NULL") != 0)
      {
         ifstream cg_mat_stream(custom_material_file);
         mat_file = new GridFunction(mesh, cg_mat_stream);
         MFEM_VERIFY(mat_file->FESpace()->GetMesh()->GetNE() == mesh->GetNE(),
                     "Invalid material file. Not compatible");
      }

      // Set the material grid function
      {
         SetMaterialGridFunction(mesh, surf_fit_gf0, mat, custom_material,
                              custom_split_mesh, exceptions, 1, mat_file);

         surf_fit_gf0_max_order = ProlongToMaxOrder(&surf_fit_gf0, 0);

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
   }

   InterfaceFaceEdgeConnectivity ifec0;

   GetGroupInfo(mesh, mat, surf_fit_gf0, ifec0);
   SetGroupNumGf(ifec0, groupnum_gf);

   if (visualization)
   {
      x_max_order = ProlongToMaxOrderAndSetMeshNodes(mesh);
      socketstream vis1;
      common::VisualizeField(vis1, "localhost", 19916, groupnum_gf,
                             "Initial group numbers",
                             0, 325, 300, 300, vis_keys);
      mesh->NewNodes(x, false);
   }

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
      x_max_order = ProlongToMaxOrderAndSetMeshNodes(mesh);
      socketstream vis1;
      common::VisualizeField(vis1, "localhost", 19916, order_gf,
                             "Initial polynomial order",
                             0, 325, 300, 300, vis_keys);
      mesh->NewNodes(x, false);
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

   while (iter_pref < pref_max_iter && faces_to_update)
   {
      std::cout << "hrp-adaptivity iteration: " << iter_pref << std::endl;
      // Compute the minimum det(J) of the starting mesh.
      min_detJ = GetMinDet(mesh, fespace, irules, quad_order);
      cout << "Minimum det(J) of the mesh " << min_detJ << endl;

      if (iter_pref > 0)
      {
         if (exceptions == 4 || exceptions == 2)
         {
            surface_fit_const = 1e-5;
         }
         else if (exceptions == 5 && iter_pref == 2)
         {
            // surface_fit_const = 1e5;
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
         jump_estimator = 0.0;
         dc->RegisterField("jumpestimator", &jump_estimator);
         dc->RegisterField("donemarker", &done_els);
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

      x_max_order = ProlongToMaxOrderAndSetMeshNodes(mesh);
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

         if (visit)
         {
            order_gf = x.FESpace()->GetElementOrdersV();
            DataCollection *dc = NULL;
            dc = new VisItDataCollection("base"+std::to_string(jobid), mesh);
            dc->RegisterField("orders", &order_gf);
            dc->RegisterField("mat", &mat);
            dc->RegisterField("fitting-error", &fitting_error_gf);
            dc->RegisterField("groupnum", &groupnum_gf);
            dc->SetCycle(2.0*iter_pref);
            dc->SetTime(2.0*iter_pref);
            dc->Save();
            delete dc;
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
      mesh->NewNodes(x, false);
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

      // Transfer max jump estimate from background mesh to current mesh
      if (surf_bg_mesh)
      {
         delete fes_lin_H1;
         delete grad_jump_lin_H1;
         fes_lin_H1 =  new FiniteElementSpace(mesh, &fec_lin_H1);
         grad_jump_lin_H1 = new GridFunction(fes_lin_H1);

         GridFunction comp_jumps(fes_lin_H1);
         L2_FECollection fec_grad_L2_temp(bgo, dim, BasisType::GaussLobatto);
         FiniteElementSpace fes_grad_L2_temp(mesh, &fec_grad_L2_temp);
         GridFunction du_comp(&fes_grad_L2_temp);

         *grad_jump_lin_H1 = 0.0;

         for (int d = 0; d < dim; d++)
         {
            GradientCompCoefficient du_comp_coeff(surf_fit_gf0, d);
            du_comp.ProjectCoefficient(du_comp_coeff);
            MeasureJumps(du_comp, comp_jumps, false);
            for (int i = 0; i < grad_jump_lin_H1->Size(); i++)
            {
               (*grad_jump_lin_H1)(i) = fmax((*grad_jump_lin_H1)(i), comp_jumps(i));
            }
         }

         MapJumpEstimateFromBackgroundMesh(bg_jump, irules, quad_order, ifec0,
                                           jump_estimator);
      }

      double max_jump_error = jump_estimator.Max();
      std::cout << max_jump_error << " " << hpthreshold << " k10maxjumperror and threshold\n";
      if (visualization)
      {
         socketstream vis1, vis2;
         x_max_order = ProlongToMaxOrderAndSetMeshNodes(mesh);
         common::VisualizeField(vis1, "localhost", 19916, jump_estimator,
                                 "Jump estimator",
                                 1000, 0, 300, 300, vis_keys);
         common::VisualizeField(vis2, "localhost", 19916, *grad_jump_lin_H1,
                              "Jump field",
                              1000, 300, 300, 300, vis_keys);
         mesh->NewNodes(x, false);
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
      do_elsp = 0.0;
      do_elsh = 0.0;
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
                   done_elsp[elnum] == 0.0 &&
                   (jump_estimator(elnum) < hpthreshold*max_jump_error || !hrefine))
               {
                  p_candidate_els.Append(elnum);
                  n_pref_count++;
                  do_elsp(elnum) = 1.0;
               }
               else if (hrefine &&
                   mesh->ncmesh->GetElementDepth(elnum) < href_max_depth &&
                   done_elsh[elnum] == 0.0 &&
                   jump_estimator(elnum) >= hpthreshold*max_jump_error)
               {
                  h_candidate_els.Append(elnum);
                  n_href_count++;
                  do_elsh(elnum) = 1.0;
               }
            }
         }
      }

      if (ref_int_neighbors)
      {
         // Refine neighbors of elements that are marked for refinement
         for (int iter = 0; iter < ref_int_neighbors; iter++)
         {
            ExtendRefinementListToNeighbors(*mesh, h_candidate_els);
         }

         // If an element is marked, mark other elements in its group as well
         Array<int> h_temp;
         for (int i = 0; i < h_candidate_els.Size(); i++)
         {
            int elem = h_candidate_els[i];
            int groupnum = ifec0.els_to_group.find(elem) == ifec0.els_to_group.end() ?
                           -1 : ifec0.els_to_group[elem];
            if (groupnum >= 0)
            {
                  for (int e = 0; e < ifec0.group_to_els[groupnum].size(); e++)
                  {
                     int elg = ifec0.group_to_els[groupnum][e];
                     h_temp.Append(elg);
                  }
            }
         }
         h_temp.Sort();
         h_temp.Unique();
         h_candidate_els.Append(h_temp);
         h_candidate_els.Sort();
         h_candidate_els.Unique();

         // Remove elements from p-refinement list if they are marked for h-refinement
         for (int i = 0; i < h_candidate_els.Size(); i++)
         {
            int elem = h_candidate_els[i];
            if (p_candidate_els.Find(elem) != -1)
            {
               int groupnum = ifec0.els_to_group[elem];
               for (int e = 0; e < ifec0.group_to_els[groupnum].size(); e++)
               {
                  int elg = ifec0.group_to_els[groupnum][e];
                  if (p_candidate_els.Find(elg) != -1)
                  {
                     p_candidate_els.DeleteFirst(elg);
                  }
               }
            }
         }
      }

      n_href_count = h_candidate_els.Size();
      n_pref_count = p_candidate_els.Size();

      std::cout << "===================================" << endl;
      cout << "Number of elements eligible for h- and p- test refinements: " <<
           n_href_count << " " << n_pref_count << endl;
      std::cout << "===================================" << endl;


      prior_el_order.SetSize(mesh->GetNE());
      for (int e = 0; e < mesh->GetNE(); e++)
      {
         prior_el_order[e] = fespace->GetElementOrder(e);
         previous_order_gf(e) =  fespace->GetElementOrder(e);
      }

      InterfaceFaceEdgeConnectivity ifecref = ifec0;

      // If this is not the first iteration, do h/p refinement
      if (iter_pref > 0)
      {
         // Do p-refinement
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
               // int target_order = hrefine && !done_h_status ?
               //                    2*current_order :
               //                    current_order + pref_order_increase;
               // int target_order = current_order + pref_order_increase;

               int target_order = current_order + 1;
               int set_order = std::min(pref_max_order, target_order);
               order_gf(elnum) = set_order;
            }

            Array<int> int_el_list;
            PropogateOrdersToGroup(order_gf, ifec0, int_el_list);

            Array<int> new_orders;
            PropogateOrders(order_gf, int_el_list,
                            adjacent_el_diff,
                            eltoeln, new_orders, 1);

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

            finder.Interpolate(x, *surf_fit_bg_gf0, surf_fit_gf0,
                               x.FESpace()->GetOrdering());
         }

         // Compute the minimum det(J) of the starting mesh.
         min_detJ = GetMinDet(mesh, fespace, irules, quad_order);
         cout << "Minimum det(J) of the original mesh after test p-ref is " << min_detJ
              << endl;
         if (min_detJ <= 0.0)
         {
            socketstream vis1;
            x_max_order = ProlongToMaxOrderAndSetMeshNodes(mesh);
            order_gf = x.FESpace()->GetElementOrdersV();
            common::VisualizeField(vis1, "localhost", 19916, order_gf,
                                   "Orders on (test) p-refined mesh",
                                   1000, 0, 300, 300, vis_keys);
            mesh->NewNodes(x, false);
            // if (iter_pref == 2) { MFEM_ABORT(" "); }
         }
         MFEM_VERIFY(min_detJ > 0.0, "Non-positive Jacobian after test p-refinement");

         // Do h-refinement
         if (n_href_count)
         {
            mesh->GeneralRefinement(h_candidate_els, 1, 0);
            hrefup.Update(true);

            GetGroupInfo(mesh, mat, surf_fit_gf0, ifecref);
            std::cout << ifecref.nfaces << " k10ifecrefsize\n";

            //
            if (amr_remarking && mat_file == NULL)
            {
               finder.Interpolate(x, *surf_fit_bg_gf0, surf_fit_gf0,
                                  x.FESpace()->GetOrdering());
               // Set the material grid function
               {
                  SetMaterialGridFunction(mesh, surf_fit_gf0, mat, custom_material,
                                          custom_split_mesh, exceptions,
                                          iter_pref % 2 + 1, mat_file);

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
            }

            // Also update the face information
            GetGroupInfo(mesh, mat, surf_fit_gf0, ifecref);
            SetGroupNumGf(ifecref, groupnum_gf);
            surf_fit_marker.SetSize(surf_fit_gf0.Size());
            std::cout << ifecref.nfaces << " k10ifecrefsize\n";
            // MFEM_ABORT(" ");
         }
         min_detJ = GetMinDet(mesh, fespace, irules, quad_order);
         cout << "Minimum det(J) of the original mesh after test hp-ref is " << min_detJ
              << endl;

         x_max_order = ProlongToMaxOrderAndSetMeshNodes(mesh);
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
         mesh->NewNodes(x, false);

         // Set orders properly if we are doing only h-refinement so that elements
         // with highest order stay at interface. This should only affect h-refinement
         // on p-refined meshes.
         if (n_href_count && !n_pref_count && x.FESpace()->IsVariableOrder())
         {
            order_gf = x.FESpace()->GetElementOrdersV();
            const Table &eltoelntemp = mesh->ElementToElementTable();
            PRefinementTransfer preft_fespacetemp = PRefinementTransfer(*fespace);
            PRefinementTransfer preft_surf_fit_festemp = PRefinementTransfer(surf_fit_fes);

            order_gf = x.FESpace()->GetElementOrdersV();

            Array<int> int_el_list_temp;
            PropogateOrdersToGroup(order_gf, ifecref, int_el_list_temp);

            Array<int> new_orders;
            PropogateOrders(order_gf, int_el_list_temp,
                            adjacent_el_diff,
                            eltoelntemp, new_orders, 1);

            for (int e = 0; e < mesh->GetNE(); e++)
            {
               order_gf(e) = new_orders[e];
               fespace->SetElementOrder(e, order_gf(e));
            }

            if (fespace->DidOrderChange())
            {
               fespace->Update(false);
               surf_fit_fes.CopySpaceElementOrders(*fespace);
               preft_fespacetemp.Transfer(x);
               preft_fespacetemp.Transfer(x0);
               preft_surf_fit_festemp.Transfer(surf_fit_mat_gf);
               preft_surf_fit_festemp.Transfer(surf_fit_gf0);
               surf_fit_marker.SetSize(surf_fit_gf0.Size());
            }
            order_gf = x.FESpace()->GetElementOrdersV();
         }
      } //if (iter_pref > 0) - done h/p refinement


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
      if (integral)
      {
         tmop_integ->EnableSurfaceFittingFaceIntegral(mat);
      }

      // if (prefine)
      {
         x_max_order = ProlongToMaxOrderAndSetMeshNodes(mesh);
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

      if (visit && iter_pref!=0)
      {
         order_gf = x.FESpace()->GetElementOrdersV();
         DataCollection *dc = NULL;
         dc = new VisItDataCollection("Optimized_"+std::to_string(jobid), mesh);
         dc->RegisterField("orders", &order_gf);
         dc->RegisterField("Level-set", surf_fit_gf0_max_order);
         dc->RegisterField("mat", &mat);
         dc->RegisterField("intmarker", surf_fit_mat_gf_max_order);
         dc->RegisterField("jumpestimator", &jump_estimator);
         dc->RegisterField("donemarker", &done_els);
         dc->SetCycle(3*iter_pref+1);
         dc->SetTime(iter_pref);
         dc->Save();
         delete dc;
      }

      mesh->NewNodes(x, false);
      delete surf_fit_gf0_max_order;

      // Setup the final NonlinearForm
      NonlinearForm a(fespace);
      a.AddDomainIntegrator(tmop_integ);
      if (integral) { a.AddInteriorFaceIntegrator(tmop_integ); }

      // Compute the minimum det(J) of the starting mesh.
      min_detJ = GetMinDet(mesh, fespace, irules, quad_order);
      cout << "Minimum det(J) of the original mesh is " << min_detJ << endl;

      if (min_detJ < 0.0)
      {
         MFEM_ABORT("The input mesh is inverted! Try an untangling metric.");
      }

      const double init_energy = a.GetGridFunctionEnergy(x);
      // std::cout << init_energy << " k10init\n";
      // MFEM_ABORT(" ");
      double init_metric_energy = init_energy;
      if (surface_fit_const > 0.0)
      {
         surf_fit_coeff.constant   = 0.0;
         init_metric_energy = a.GetGridFunctionEnergy(x);
         surf_fit_coeff.constant   = surface_fit_const;
      }
      x_max_order = ProlongToMaxOrderAndSetMeshNodes(mesh);
      if (visualization && iter_pref==0)
      {
         char title[] = "Initial metric values";
         vis_tmop_metric_s(mesh_poly_deg, *metric, *target_c, *mesh, title, 0, 0, 300,
                           300);
      }
      mesh->NewNodes(x, false);

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

         // also fix dofs for elements that are done
         // Array<int> vdofs_surf;
         // for (int e = 0; e < mesh->GetNE(); e++)
         // {
         //    if (done_els[e] == 1)
         //    {
         //       surf_fit_gf0.FESpace()->GetElementVDofs(e, vdofs);
         //       for (int i = 0; i < vdofs.Size(); i++)
         //       {
         //          if (surf_fit_marker[vdofs[i]]) {
         //             vdofs_surf.Append(vdofs[i]);
         //          }
         //       }
         //    }
         // }
         // vdofs_surf.Sort();
         // vdofs_surf.Unique();
         // // remove corner vertices associated with elements that might not be done
         // for (int e = 0; e < mesh->GetNE(); e++)
         // {
         //    if (done_els[e] == 0 &&
         //       (ifec0.inter_face_el_all.Find(e) != -1 ||
         //        ifec0.intedgels.Find(e) != -1))
         //    {
         //       surf_fit_gf0.FESpace()->GetElementVDofs(e, vdofs);
         //       for (int i = 0; i < vdofs.Size(); i++)
         //       {
         //          if (surf_fit_marker[vdofs[i]] &&
         //              vdofs_surf.Find(vdofs[i]) != -1)
         //          {
         //             vdofs_surf.DeleteFirst(vdofs[i]);
         //          }
         //       }
         //    }
         // }
         // fespace->DofsToVDofs(vdofs_surf);
         // ess_vdofs.Append(vdofs_surf);
         // a.SetEssentialVDofs(ess_vdofs);
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

      x_max_order = ProlongToMaxOrderAndSetMeshNodes(mesh);

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
      mesh->NewNodes(x, false);
      // Don't delete x_max_order yet

      // Report the final energy of the functional.
      // const double fin_energy = a.GetGridFunctionEnergy(x);
      // double fin_metric_energy = fin_energy;
      // if (surface_fit_const > 0.0)
      // {
      //    surf_fit_coeff.constant  = 0.0;
      //    fin_metric_energy  = a.GetGridFunctionEnergy(x);
      //    surf_fit_coeff.constant  = surface_fit_const;
      // }

      // std::cout << std::scientific << std::setprecision(4);
      // cout << "Initial strain energy: " << init_energy
      //      << " = metrics: " << init_metric_energy
      //      << " + extra terms: " << init_energy - init_metric_energy << endl;
      // cout << "  Final strain energy: " << fin_energy
      //      << " = metrics: " << fin_metric_energy
      //      << " + extra terms: " << fin_energy - fin_metric_energy << endl;
      // cout << "The strain energy decreased by: "
      //      << (init_energy - fin_energy) * 100.0 / init_energy << " %." << endl;

      // Set nodal gridfunction for visualization and fitting error computation.
      x_max_order = ProlongToMaxOrderAndSetMeshNodes(mesh);

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
            dc->RegisterField("intmarker", surf_fit_mat_gf_max_order);
            jump_estimator = 0.0;
            dc->RegisterField("jumpestimator", &jump_estimator);
            dc->RegisterField("donemarker", &done_els);
            dc->SetCycle(3*iter_pref+2);
            dc->SetTime(iter_pref);
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
            surf_fit_mat_gf_max_order = ProlongToMaxOrder(&surf_fit_mat_gf, 0);
            common::VisualizeField(vis2, "localhost", 19916, *surf_fit_mat_gf_max_order,
                                   "Markers for fitting",
                                   325*(iter_pref+1), 150, 300, 300, vis_keys);
         }

         if (visit)
         {
            order_gf = x.FESpace()->GetElementOrdersV();
            DataCollection *dc = NULL;
            dc = new VisItDataCollection("base"+std::to_string(jobid), mesh);
            dc->RegisterField("orders", &order_gf);
            dc->RegisterField("mat", &mat);
            dc->RegisterField("fitting-error", &fitting_error_gf);
            dc->RegisterField("groupnum", &groupnum_gf);
            dc->SetCycle(2.0*iter_pref+1.0);
            dc->SetTime(2.0*iter_pref+1.0);
            dc->Save();
            delete dc;
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
      mesh->NewNodes(x, false);
      delete surf_fit_gf0_max_order;

      // Now we mark elements as done if there error is below threshold
      {
         GetGroupInfo(mesh, mat, surf_fit_gf0, ifec0);
         SetGroupNumGf(ifec0, groupnum_gf);
         x_max_order = ProlongToMaxOrderAndSetMeshNodes(mesh);
         surf_fit_gf0_max_order = ProlongToMaxOrder(&surf_fit_gf0, 0);
         ComputeIntegratedErrorBGonFaces(x_max_order->FESpace(),
                                         surf_fit_bg_gf0,
                                         ifec0.inter_faces,
                                         surf_fit_gf0_max_order,
                                         finder,
                                         ifec0.face_error_map,
                                         error_type);

         if (iter_pref == pref_max_iter-1 && visit)
         {
            order_gf = x.FESpace()->GetElementOrdersV();
            DataCollection *dc = NULL;
            dc = new VisItDataCollection("base"+std::to_string(jobid), mesh);
            dc->RegisterField("orders", &order_gf);
            dc->RegisterField("mat", &mat);
            dc->RegisterField("fitting-error", &fitting_error_gf);
            dc->RegisterField("groupnum", &groupnum_gf);
            dc->SetCycle(2.0*iter_pref+2.0);
            dc->SetTime(2.0*iter_pref+2.0);
            dc->Save();
            delete dc;
         }

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
               int current_order = fespace->GetElementOrder(elnum);
               int current_depth = mesh->ncmesh->GetElementDepth(elnum);
               if (max_face_error > pref_tol)
               {
                  if (current_order < pref_max_order && prefine) {
                     done_elsp[elnum] = 0.0;
                  }
                  if (current_depth < href_max_depth && hrefine)
                  {
                     done_elsh(elnum) = 0.0;
                  }
                  done_els(elnum) = min(done_elsp(elnum), done_elsh(elnum));
               }
            }
         }
         mesh->NewNodes(x, false);
         delete surf_fit_gf0_max_order;

         // visualize
         if (visualization)
         {
            x_max_order = ProlongToMaxOrderAndSetMeshNodes(mesh);
            socketstream vis1, vis2;
            common::VisualizeField(vis1, "localhost", 19916, done_els,
                                   "Done elements",
                                   325*(iter_pref+1), 650, 300, 300, vis_keys);
            // common::VisualizeField(vis2, "localhost", 19916, done_elsp,
            //                        "p-Done elements",
            //                        325*(iter_pref+1), 650, 300, 300, vis_keys);
            mesh->NewNodes(x, false);
         }
      }

      // Now we propogate orders based on the final mesh
      if (iter_pref > 0 && n_href_count && n_pref_count)
      {
         const Table &eltoelntemp = mesh->ElementToElementTable();
         PRefinementTransfer preft_fespacetemp = PRefinementTransfer(*fespace);
         PRefinementTransfer preft_surf_fit_festemp = PRefinementTransfer(surf_fit_fes);

         order_gf = x.FESpace()->GetElementOrdersV();

         Array<int> int_el_list;
         PropogateOrdersToGroup(order_gf, ifec0, int_el_list);

         Array<int> new_orders;
         PropogateOrders(order_gf, int_el_list,
                         adjacent_el_diff,
                         eltoelntemp, new_orders, 1);

         for (int e = 0; e < mesh->GetNE(); e++)
         {
            order_gf(e) = new_orders[e];
            fespace->SetElementOrder(e, order_gf(e));
         }

         if (fespace->DidOrderChange())
         {
            fespace->Update(false);
            surf_fit_fes.CopySpaceElementOrders(*fespace);
            preft_fespacetemp.Transfer(x);
            preft_fespacetemp.Transfer(x0);
            preft_surf_fit_festemp.Transfer(surf_fit_mat_gf);
            preft_surf_fit_festemp.Transfer(surf_fit_gf0);
            surf_fit_marker.SetSize(surf_fit_gf0.Size());
         }
         order_gf = x.FESpace()->GetElementOrdersV();
      }

      // visualize
      if (visualization)
      {
         x_max_order = ProlongToMaxOrderAndSetMeshNodes(mesh);
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


         mesh->NewNodes(x, false);
      }

      if (done_elsh.Sum() == mesh->GetNE() && done_elsp.Sum() == mesh->GetNE())
      {
         std::cout << "****************\n";
         std::cout << "All elements are marked as done after " << iter_pref <<
                   " hp-iterations\n";
         std::cout << "****************\n";
         faces_to_update = false;
      }

      // Now we derefine the elements that were unnecessarily p-refined
      if (iter_pref > 0)
      {
         int compt_updates = 0;
         x_max_order = ProlongToMaxOrderAndSetMeshNodes(mesh);
         surf_fit_gf0_max_order = ProlongToMaxOrder(&surf_fit_gf0, 0);

         for (int i=0; i < ifec0.inter_faces.Size(); i++)
         {
            int el1 = ifec0.inter_face_el1[i];
            int el2 = ifec0.inter_face_el2[i];
            int fnum = ifec0.inter_faces[i];

            Array<int> els;
            int el_order = fespace->GetElementOrder(el1);
            int prior_order = previous_order_gf(el1);
            //  This element was not p-refined if the below is true... so we skip
            if (el_order == prior_order) { continue; }

            double interface_error = ifec0.face_error_map[fnum];
            double interface_deref_error = ComputeIntegrateErrorBG(x_max_order->FESpace(),
                                                                   surf_fit_bg_gf0,
                                                                   fnum,
                                                                   surf_fit_gf0_max_order,
                                                                   finder,
                                                                   deref_error_type);

            int orig_order = el_order;
            int target_order = el_order;
            bool trycoarsening = true;
            // derefine the element if the error of the element is below threshold.
            // Criterion 1: the error stays below threshold after derefinement.
            // Criterion 2: the change in length/error is less than a given %.
            while (el_order > prior_order+1 && trycoarsening && done_els(el1) == 1)
            {
               double coarsened_face_error = InterfaceElementOrderReduction(mesh, fnum,
                                                                            el_order-1,
                                                                            surf_el_meshes,
                                                                            surf_fit_bg_gf0, finder, error_type);
               trycoarsening = false;
               if (coarsened_face_error < pref_tol)
               {
                  trycoarsening = CheckElementValidityAtOrder(mesh, el1, el_order-1);
                  if  (trycoarsening)
                  {
                     trycoarsening = CheckElementValidityAtOrder(mesh, el2, el_order-1);
                  }
                  if (trycoarsening)
                  {
                     el_order -= 1;
                     target_order = el_order;
                  }
               }
            }

            trycoarsening = false;
            while (done_els(el1) == 1 && el_order > prior_order+1 &&
                   trycoarsening && pderef > 0 )
            {
               double coarsened_face_deref_error = InterfaceElementOrderReduction(mesh, fnum,
                                                                                  el_order-1, surf_el_meshes,
                                                                                  surf_fit_bg_gf0, finder, deref_error_type);
               trycoarsening = false;

               if ( (deref_error_type == 1 &&
                     coarsened_face_deref_error > (1-pderef)*(interface_deref_error)) ||
                    //relative to decrease in length due to refinement
                    (deref_error_type == 4 &&
                     coarsened_face_deref_error < pderef*pref_tol) || // hysteresis
                    (deref_error_type == 0 &&
                     coarsened_face_deref_error < (1+pderef)*
                     (interface_deref_error)) //relative to increase in L2 error
                  )
               {
                  trycoarsening = CheckElementValidityAtOrder(mesh, el1, el_order-1);
                  if  (trycoarsening)
                  {
                     trycoarsening = CheckElementValidityAtOrder(mesh, el2, el_order-1);
                  }
                  if (trycoarsening)
                  {
                     el_order -= 1;
                     target_order = el_order;
                  }
               }
            }

            if (target_order != orig_order)
            {
               order_gf(el1) = target_order;
               order_gf(el2) = target_order;
               compt_updates++;
            }
         } //i < inter_faces.Size()

         std::cout << "=======================================\n";
         std::cout << "# Derefinements: " << compt_updates << std::endl;
         std::cout << "=======================================\n";

         // Update the FES and GridFunctions only if some orders have been changed
         if (compt_updates > 0)
         {
            const Table &eltoelntemp = mesh->ElementToElementTable();
            PRefinementTransfer preft_fespacetemp = PRefinementTransfer(*fespace);
            PRefinementTransfer preft_surf_fit_festemp = PRefinementTransfer(surf_fit_fes);

            // Propogate error first
            Array<int> int_el_list;
            PropogateOrdersToGroup(order_gf, ifec0, int_el_list);
            Array<int> new_orders;
            PropogateOrders(order_gf, int_el_list,
                            adjacent_el_diff,
                            eltoelntemp, new_orders, 1);

            for (int e = 0; e < mesh->GetNE(); e++)
            {
               bool validity = CheckElementValidityAtOrder(mesh, e, new_orders[e]);
               if (validity)
               {
                  order_gf(e) = new_orders[e];
                  fespace->SetElementOrder(e, order_gf(e));
               }
               else
               {
                  std::cout << e << " invalid derefinement\n";
               }
            }

            // Updates if we increase the order of at least one element
            fespace->Update(false);
            surf_fit_fes.CopySpaceElementOrders(*fespace);
            surf_fit_fes.SetRelaxedHpConformity(fespace->GetRelaxedHpConformity());
            preft_fespacetemp.Transfer(x);
            preft_fespacetemp.Transfer(x0);
            preft_surf_fit_festemp.Transfer(surf_fit_mat_gf);
            preft_surf_fit_festemp.Transfer(surf_fit_gf0);
            surf_fit_marker.SetSize(surf_fit_gf0.Size());

            {
               ofstream mesh_ofs("optimized_" +std::to_string(iter_pref)+std::to_string(
                                    jobid)+ ".mesh");
               mesh_ofs.precision(14);
               mesh->Print(mesh_ofs);
            }
            mesh->NewNodes(x, false);
            min_detJ = GetMinDet(mesh, fespace, irules, quad_order);
            cout << "Minimum det(J) of the mesh after coarsening is " << min_detJ << endl;
            MFEM_VERIFY(min_detJ > 0, "Mesh has somehow become inverted "
                        "due to coarsening");

            x_max_order = ProlongToMaxOrder(&x, 0);
            delete surf_fit_gf0_max_order;
            surf_fit_gf0_max_order = ProlongToMaxOrder(&surf_fit_gf0, 0);

            x_max_order = ProlongToMaxOrderAndSetMeshNodes(mesh);
            ComputeIntegratedErrorBGonFaces(x_max_order->FESpace(),
                                            surf_fit_bg_gf0,
                                            ifec0.inter_faces,
                                            surf_fit_gf0_max_order,
                                            finder,
                                            ifec0.face_error_map);
            double error_sum = 0.0;
            for (const auto& pair : ifec0.face_error_map)
            {
               error_sum += pair.second;
            }
            sum_current_face_error = error_sum;

            std::cout << "NDofs & Integrated Error Post Derefinement: " <<
                      fespace->GetVSize() << " " <<
                      error_sum << " " << std::endl;
            mesh->NewNodes(x, false);
            delete surf_fit_gf0_max_order;

            if (visualization)
            {
               x_max_order = ProlongToMaxOrderAndSetMeshNodes(mesh);
               socketstream vis1, vis2;
               common::VisualizeField(vis1, "localhost", 19916, order_gf,
                                      "Orders on optimal mesh after p-deref",
                                      325*(iter_pref+1), 700, 300, 300, vis_keys);


               {
                  ofstream mesh_ofs("optimized.mesh");
                  mesh_ofs.precision(14);
                  mesh->Print(mesh_ofs);
               }

               mesh->NewNodes(x, false);
            }
         } //compt_updates

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
         x_max_order = ProlongToMaxOrderAndSetMeshNodes(mesh);

         // Visualize the final mesh and metric values.
         if (visualization && iter_pref==pref_max_iter-1)
         {
            char title[] = "Final metric values";
            vis_tmop_metric_s(mesh_poly_deg, *metric, *target_c, *mesh, title, 325, 0, 300,
                              300);
         }
         mesh->NewNodes(x, false);
      } //iter_pref > 0 (basically derefinement start)

      order_gf = x.FESpace()->GetElementOrdersV();
      if (visit)
      {
         mesh->NewNodes(x, false);
         x_max_order = ProlongToMaxOrderAndSetMeshNodes(mesh);
         surf_fit_gf0_max_order = ProlongToMaxOrder(&surf_fit_gf0, 0);

         DataCollection *dc = NULL;
         dc = new VisItDataCollection("Optimized_"+std::to_string(jobid), mesh);
         dc->RegisterField("orders", &order_gf);
         dc->RegisterField("Level-set", surf_fit_gf0_max_order);
         dc->RegisterField("mat", &mat);
         dc->RegisterField("intmarker", surf_fit_mat_gf_max_order);
         jump_estimator = 0.0;
         dc->RegisterField("jumpestimator", &jump_estimator);
         dc->RegisterField("donemarker", &done_els);
         dc->SetCycle(3*iter_pref+3);
         dc->SetTime(iter_pref);
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

      mesh->NewNodes(x, false);
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
