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

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "../common/mfem-common.hpp"
#include "mesh-fitting.hpp"
using namespace std;
using namespace mfem;

TargetConstructor *GetTargetConstructor(int target_id, ParGridFunction &x0)
{
   TargetConstructor *target_c = NULL;
   TargetConstructor::TargetType target_t;
   switch (target_id)
   {
      case 1: target_t = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE; break;
      case 2: target_t = TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE; break;
      case 3: target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE; break;
      default:
         MFEM_ABORT("Unknown target_id"); break;
   }
   if (target_c == NULL)
   {
      target_c = new TargetConstructor(target_t, x0.ParFESpace()->GetComm());
   }
   target_c->SetNodes(x0);
   return target_c;
}

TMOP_QualityMetric *GetMetric(int metric_id)
{
   TMOP_QualityMetric *metric = NULL;
   switch (metric_id)
   {
      case 1: metric = new TMOP_Metric_001; break;
      case 2: metric = new TMOP_Metric_002; break;
      case 4: metric = new TMOP_Metric_004; break;
      case 14: metric = new TMOP_Metric_014; break;
      case 49: metric = new TMOP_AMetric_049(0.6); break;
      case 50: metric = new TMOP_Metric_050; break;
      case 55: metric = new TMOP_Metric_055; break;
      case 58: metric = new TMOP_Metric_058; break;
      case 66: metric = new TMOP_Metric_066(0.1); break;
      case 80: metric = new TMOP_Metric_080(0.25); break;
      case 301: metric = new TMOP_Metric_301; break;
      case 302: metric = new TMOP_Metric_302; break;
      case 303: metric = new TMOP_Metric_303; break;
      case 322: metric = new TMOP_Metric_322; break;
      case 360: metric = new TMOP_Metric_360; break;
      default:
         MFEM_ABORT("Unknown metric_id"); break;
   }
   return metric;
}

class MeshOptimizer
{
private:
   TMOP_QualityMetric *metric = nullptr;
   TargetConstructor *target_c = nullptr;
   ParNonlinearForm *nlf = nullptr;
   IterativeSolver *lin_solver = nullptr;
   TMOPNewtonSolver *solver = nullptr;
   TMOP_Integrator *tmop_integ = nullptr; // owned by nlf
   ParGridFunction *x_nodes; // ptr to mesh nodes
   ParFiniteElementSpace *dist_pfespace = nullptr;
   ParGridFunction *lim_dist = nullptr;
   ConstantCoefficient *lim_coeff;
   ParGridFunction *x_0;

   ParFiniteElementSpace *pfes_nodes_scalar = nullptr;
   ParMesh *pmesh;
#ifdef MFEM_USE_GSLIB
   Array<FindPointsGSLIB *> finder_arr;
   Array<Array<int> *> tang_dofs_arr;
   Array<GridFunction *> nodes0_arr;
#endif

public:
   double init_energy, final_energy;
   MeshOptimizer(ParMesh *pmesh_): pmesh(pmesh_) { }

   ~MeshOptimizer()
   {
      delete pfes_nodes_scalar;
      delete solver;
      delete lin_solver;
      delete nlf;
      delete target_c;
      delete metric;
   }

   // Must be called before optimization.
   void Setup(ParGridFunction &x,
              double *min_det_ptr,
              int quad_order,
              int metric_id, int target_id,
              PLBound *plb,
              ParGridFunction *detgf, int solver_iter,
              bool move_bnd, Array<int> surf_mesh_attr,
              Array<int> aux_ess_dofs, int solver_type,
              real_t limiting = 0.0);

   // Optimizes the node positions given in x.
   // When we enter, x contains the initial node positions.
   // When we exit, x contains the optimized node positions.
   // The underlying mesh of x remains unchanged (its positions don't change).
   void OptimizeNodes(ParGridFunction &x);

#ifdef MFEM_USE_GSLIB
   void SetupTangentialRelaxationForFacEdg(ParFiniteElementSpace *pfespace,
                                           Array<int> *fdofs,
                                           FindPointsGSLIB *finder,
                                           GridFunction *nodes0);
   void EnableTangentialRelaxation()
   {
      solver->SetTangentialRelaxationFlag(true);
      tmop_integ->EnableTangentialRelaxation(finder_arr, tang_dofs_arr,
                                             nodes0_arr);
   }
#endif

   TMOP_Integrator *GetTMOPIntegrator() { return tmop_integ; }
};

// Go over the dofs and set bitwise flags in attr_marker based on boundary
// attribute the dof touches.
// attr_count tells how many attributes each dof touches.
// works for both serial and vector spaces.
void SetupDofAttributes(ParGridFunction &attr_count, Array<int> &attr_marker)
{
   ParFiniteElementSpace *pfesmarker = attr_count.ParFESpace();
   ParMesh *pmesh = pfesmarker->GetParMesh();
   attr_marker.SetSize(attr_count.Size());
   attr_marker = 0;
   Array<int> dofs;
   int nbdr_faces = pmesh->GetNFbyType(FaceType::Boundary);
   for (int f = 0; f < nbdr_faces; f++)
   {
      int attrib = pmesh->GetBdrAttribute(f);
      pfesmarker->GetBdrElementVDofs(f, dofs);
      for (int i = 0; i < dofs.Size(); i++)
      {
         int val = attr_marker[dofs[i]];
         attr_marker[dofs[i]] = val | (1 << (attrib-1));
      }
   }

   GroupCommunicator &gcomm = attr_count.ParFESpace()->GroupComm();
   gcomm.Reduce<int>(attr_marker, GroupCommunicator::BitOR);
   gcomm.Bcast(attr_marker);

   auto countBits = [](int val)
   {
      int count = 0;
      while (val)
      {
         count += val & 1;
         val >>= 1;
      }
      return count;
   };

   for (int i = 0; i < attr_count.Size(); i++)
   {
      attr_count(i) = countBits(attr_marker[i]);
   }
}

// Mark only the dofs that are on interfaces between different attributes.
// The attribute for such dofs is element attribute + max_bdr_attr

void RemoveDofsFromBdrFaceDofs(Array<int> &bdr_face_dofs,
                               Array<int> &bdr_edge_dofs)
{
   // Remove the edge dofs from the face dofs.
   for (int i = 0; i < bdr_face_dofs.Size(); i++)
   {
      int dof = bdr_face_dofs[i];
      if (bdr_edge_dofs.Find(dof) != -1)
      {
         bdr_face_dofs[i] = -1; // mark for removal
      }
   }
   bdr_face_dofs.Sort();
   bdr_face_dofs.Unique(); // remove duplicates
   // Remove the marked dof
   bdr_face_dofs.DeleteFirst(-1);
}

// Get indices of all dofs that have more than or equal to `dim` attributes.
// Essentially dofs that should not be allowed to move.
// Works for both scalar and vector spaces
Array<int> IdentifyAuxiliaryEssentialDofs(ParGridFunction &attr_count)
{
   ParFiniteElementSpace *pfespace = attr_count.ParFESpace();
   Array<int> aux_dofs;
   int dim = pfespace->GetParMesh()->SpaceDimension();
   for (int i = 0; i < attr_count.Size(); i++)
   {
      if (attr_count[i] >= dim)
      {
         aux_dofs.Append(i);
      }
   }
   return aux_dofs;
}

// Get indices of all dofs that have only the required attribute marked.
Array<int> *GetDofMatchingBdrFaceAttributes(ParGridFunction &attr_count,
                                            Array<int> &attr_marker,
                                            int attr)
{
   ParFiniteElementSpace *pfespace = attr_count.ParFESpace();
   Array<int> *facedofs = new Array<int>;
   int vdim = pfespace->GetVDim();
   MFEM_VERIFY(vdim == 1, "Only scalar spaces should be used.");

   // check if jth bit is set
   auto hasBitSet = [](int val, int j)
   {
      int mask = (1 << (j-1));
      return (val & mask) == mask;
   };

   for (int i = 0; i < attr_count.Size(); i++)
   {
      if (attr_count[i] == 1)
      {
         int val = attr_marker[i];
         if (hasBitSet(val, attr))
         {
            facedofs->Append(i);
         }
      }
   }
   facedofs->Sort();
   facedofs->Unique();
   return facedofs;
}

// Get indices of all dofs that have only the required attribute marked.
Array<int> *GetDofMatchingBdrEdgeAttributes(ParGridFunction &attr_count,
                                            Array<int> &attr_marker,
                                            int attr1, int attr2)
{
   ParFiniteElementSpace *pfespace = attr_count.ParFESpace();
   Array<int> *edgedofs = new Array<int>;
   int vdim = pfespace->GetVDim();
   MFEM_VERIFY(vdim == 1, "Only scalar spaces should be used.");

   // check if jth and kth bits are set
   auto hasAtLeastTwoBitsSet = [](int val, int j, int k)
   {
      int mask = (1 << (j-1)) | (1 << (k-1));
      return (val & mask) == mask;
   };

   for (int i = 0; i < attr_count.Size(); i++)
   {
      if (attr_count[i] == 2)
      {
         int val = attr_marker[i];
         if (hasAtLeastTwoBitsSet(val, attr1, attr2))
         {
            edgedofs->Append(i);
         }
      }
   }
   edgedofs->Sort();
   edgedofs->Unique();
   return edgedofs;
}

// Get indices of all dofs that have only the required attribute marked.

void SetupSerialDofAttributes(GridFunction &attr_count, Array<int> &attr_marker)
{
   FiniteElementSpace *fesmarker = attr_count.FESpace();
   Mesh *mesh = fesmarker->GetMesh();
   attr_marker.SetSize(attr_count.Size());
   attr_marker = 0;
   Array<int> dofs;
   int nbdr_faces = mesh->GetNFbyType(FaceType::Boundary);
   for (int f = 0; f < nbdr_faces; f++)
   {
      int attrib = mesh->GetBdrAttribute(f);
      fesmarker->GetBdrElementVDofs(f, dofs);
      for (int i = 0; i < dofs.Size(); i++)
      {
         int val = attr_marker[dofs[i]];
         attr_marker[dofs[i]] = val | (1 << (attrib-1));
      }
   }

   auto countBits = [](int val)
   {
      int count = 0;
      while (val)
      {
         count += val & 1;
         val >>= 1;
      }
      return count;
   };

   for (int i = 0; i < attr_count.Size(); i++)
   {
      attr_count(i) = countBits(attr_marker[i]);
   }
}

Mesh *SetupEdgeMesh3D(Mesh *mesh, GridFunction &attr_count_ser,
                      Array<int> &attr_marker_ser,
                      int attr1, int attr2)
{
   Array<int> facedofs(0);
   int spaceDim = mesh->SpaceDimension();
   MFEM_VERIFY(spaceDim == 3, "Only 2D meshes supported right now.");
   Array<int> edofs;
   GridFunction *x = mesh->GetNodes();
   MFEM_VERIFY(x, "Mesh nodal space not set\n");
   const FiniteElementSpace *fes = mesh->GetNodalFESpace();
   int mesh_poly_deg = fes->GetMaxElementOrder();
   int nedges = mesh->GetNEdges();

   FiniteElementSpace *fespace_attr = attr_count_ser.FESpace();

   Array<int> ev, dofs;
   int attr_marker_dofs[2], attr_count_dofs[2];
   Array<int> edge_to_include;
   for (int ei = 0; ei < nedges; ei++)
   {
      mesh->GetEdgeVertices(ei, ev);
      MFEM_VERIFY(ev.Size(), "Could not get edge vertices.");
      for (int j = 0; j < 2; j++)
      {
         int v_id = ev[j];
         fespace_attr->GetVertexDofs(v_id, dofs);
         attr_marker_dofs[j] = attr_marker_ser[dofs[0]];
         attr_count_dofs[j] = attr_count_ser[dofs[0]];
      }
      auto hasAtLeastTwoBitsSet = [](int val, int j, int k)
      {
         int mask = (1 << (j-1)) | (1 << (k-1));
         return (val & mask) == mask;
      };
      bool dof1_mask2 = hasAtLeastTwoBitsSet(attr_marker_dofs[0], attr1, attr2);
      bool dof2_mask2 = hasAtLeastTwoBitsSet(attr_marker_dofs[1], attr1, attr2);
      if ( (dof1_mask2 && dof2_mask2))
      {
         edge_to_include.Append(ei);
      }
   }

   // Setup a mesh with dummy vertices
   int nel = edge_to_include.Size();
   Vector vals;
   Mesh *intmesh = new Mesh(1, nel*2, nel, 0, spaceDim);
   {
      for (int i = 0; i < nel; i++)
      {
         for (int j = 0; j < 2; j++) // 2 vertices per element
         {
            Vector vert(spaceDim);
            vert = 0.5;
            intmesh->AddVertex(vert.GetData());
         }
         Array<int> verts(2);
         for (int d = 0; d < 2; d++)
         {
            verts[d] = i*2+d;
         }
         intmesh->AddSegment(verts, 1);
      }
      intmesh->Finalize(true, true);
      intmesh->FinalizeTopology(false);
      intmesh->SetCurvature(mesh_poly_deg, false, -1, 0);
   }

   const FiniteElementSpace *intnodespace = intmesh->GetNodalFESpace();
   GridFunction *intnodes = intmesh->GetNodes();

   for (int i = 0; i < nel; i++)
   {
      int ei = edge_to_include[i];
      fes->GetEdgeVDofs(ei, dofs);
      x->GetSubVector(dofs, vals);
      Array<int> edofs;
      intnodespace->GetElementVDofs(i, edofs);
      intnodes->SetSubVector(edofs, vals);
   }

   return intmesh;
}

// Get indices of all edges that have both the required attributes marked.

Mesh *SetupFaceMesh3D(Mesh *mesh, int attr)
{
   Array<int> facedofs(0);
   int spaceDim = mesh->SpaceDimension();
   MFEM_VERIFY(spaceDim == 3, "Only 2D meshes supported right now.");
   Array<int> fdofs;
   GridFunction *x = mesh->GetNodes();
   MFEM_VERIFY(x, "Mesh nodal space not set\n");
   const FiniteElementSpace *fes = mesh->GetNodalFESpace();
   int mesh_poly_deg = fes->GetMaxElementOrder();

   int nbdr_faces = mesh->GetNFbyType(FaceType::Boundary);
   Array<int> faces_to_include;
   for (int f = 0; f < nbdr_faces; f++)
   {
      int attrib = mesh->GetBdrAttribute(f);
      if (attrib == attr)
      {
         faces_to_include.Append(mesh->GetBdrElementFaceIndex(f));
      }
   }

   // Setup a mesh with dummy vertices
   int nel = faces_to_include.Size();
   Vector vals;
   Mesh *intmesh = new Mesh(2, nel*4, nel, 0, spaceDim);
   {
      for (int i = 0; i < nel; i++)
      {
         for (int j = 0; j < 4; j++) // 4 vertices per element
         {
            Vector vert(spaceDim);
            vert = 0.5;
            intmesh->AddVertex(vert.GetData());
         }
         Array<int> verts(4);
         for (int d = 0; d < 4; d++)
         {
            verts[d] = i*4+d;
         }
         intmesh->AddQuad(verts, 1);
      }
      intmesh->Finalize(true, true);
      intmesh->FinalizeTopology(false);
      intmesh->SetCurvature(mesh_poly_deg, false, -1, 0);
   }

   const FiniteElementSpace *intnodespace = intmesh->GetNodalFESpace();
   GridFunction *intnodes = intmesh->GetNodes();

   for (int i = 0; i < nel; i++)
   {
      int fi = faces_to_include[i];
      fes->GetFaceVDofs(fi, fdofs);
      x->GetSubVector(fdofs, vals);
      intnodespace->GetElementVDofs(i, fdofs);
      intnodes->SetSubVector(fdofs, vals);
   }

   return intmesh;
}

#ifdef MFEM_USE_GSLIB
void MeshOptimizer::SetupTangentialRelaxationForFacEdg(ParFiniteElementSpace
                                                       *pfespace,
                                                       Array<int> *fdofs,
                                                       FindPointsGSLIB *finder,
                                                       GridFunction *nodes0)
{
   finder_arr.Append(finder);
   tang_dofs_arr.Append(fdofs);
   nodes0_arr.Append(nodes0);
}
#endif

void MeshOptimizer::Setup(ParGridFunction &x,
                          double *min_det_ptr,
                          int quad_order,
                          int metric_id, int target_id,
                          PLBound *plb,
                          ParGridFunction *detgf, int solver_iter,
                          bool move_bnd, Array<int> surf_mesh_attr,
                          Array<int> aux_ess_dofs, int solver_type,
                          real_t limiting)
{
   ParFiniteElementSpace &pfes = *x.ParFESpace();
   x_nodes = &x;
   const int dim = pfes.GetMesh()->Dimension();
   x_0 = new ParGridFunction(&pfes);
   *x_0 = *x_nodes;

   // Metric.
   metric = GetMetric(metric_id);

   // Target.
   target_c = GetTargetConstructor(target_id, x);

   // Integrator.
   tmop_integ = new TMOP_Integrator(metric, target_c, nullptr);
   tmop_integ->SetIntegrationRules(IntRulesLo, quad_order);
   if (plb)
   {
      tmop_integ->EnableDeterminantPLBounds(detgf, 3, 2);
   }

   // Nonlinear form.
   nlf = new ParNonlinearForm(&pfes);
   nlf->AddDomainIntegrator(tmop_integ);

   // Boundary.
   Array<int> ess_bdr(pfes.GetParMesh()->bdr_attributes.Max());
   int n = 0;
   if (!move_bnd)
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      nlf->SetEssentialBC(ess_bdr);
   }
   else
   {
      for (int i = 0; i < pmesh->GetNBE(); i++)
      {
         const int nd = pfes.GetBE(i)->GetDof();
         const int attr = pmesh->GetBdrElement(i)->GetAttribute();
         if (surf_mesh_attr.Find(attr) == -1)
         {
            if (attr == 1 || attr == 2 || (attr == 3 && dim == 3)) { n += nd; }
            if (attr >= dim+1) { n += nd * dim; }
         }
      }
      Array<int> ess_vdofs(n);
      n = 0;
      Array<int> vdofs;
      for (int i = 0; i < pmesh->GetNBE(); i++)
      {
         const int nd = pfes.GetBE(i)->GetDof();
         const int attr = pmesh->GetBdrElement(i)->GetAttribute();
         pfes.GetBdrElementVDofs(i, vdofs);
         if (surf_mesh_attr.Find(attr) == -1)
         {
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
            else if (attr >= dim+1) // Fix all components.
            {
               for (int j = 0; j < vdofs.Size(); j++)
               { ess_vdofs[n++] = vdofs[j]; }
            }
         }
      }
      for (int i = 0; i < aux_ess_dofs.Size(); i++)
      {
         ess_vdofs.Append(aux_ess_dofs[i]);
      }
      nlf->SetEssentialVDofs(ess_vdofs);
   }

   // limiting
   if (limiting > 0.0)
   {
      dist_pfespace = new ParFiniteElementSpace(pmesh, pfes.FEColl()); // scalar space
      lim_dist = new ParGridFunction(dist_pfespace);
      *lim_dist = 1.0;
      lim_coeff = new ConstantCoefficient(limiting);
      tmop_integ->EnableLimiting(*x_0, *lim_dist, *lim_coeff);
   }

   // Linear solver.
   lin_solver = new MINRESSolver(pfes.GetComm());
   lin_solver->SetMaxIter(100);
   lin_solver->SetRelTol(1e-12);
   lin_solver->SetAbsTol(0.0);
   IterativeSolver::PrintLevel minres_pl;
   lin_solver->SetPrintLevel(minres_pl.FirstAndLast().Summary());

   // Nonlinear solver.
   const IntegrationRule &ir =
      IntRulesLo.Get(pfes.GetFE(0)->GetGeomType(), quad_order);
   solver = new TMOPNewtonSolver(pfes.GetComm(), ir, solver_type);
   solver->SetIntegrationRules(IntRulesLo, quad_order);
   if (plb) { solver->EnsurePositiveDeterminantBound(); }
   solver->SetOperator(*nlf);
   if (solver_type == 0)
   {
      solver->SetPreconditioner(*lin_solver);
   }
   if (min_det_ptr)
   {
      solver->SetMinDetPtr(min_det_ptr);
   }
   solver->SetMaxIter(solver_iter);
   solver->SetRelTol(1e-8);
   solver->SetAbsTol(0.0);
   IterativeSolver::PrintLevel newton_pl;
   solver->SetPrintLevel(newton_pl.Iterations().Summary());
}

void MeshOptimizer::OptimizeNodes(ParGridFunction &x)
{
   MFEM_VERIFY(solver, "Setup() has not been called.");
   init_energy = nlf->GetParGridFunctionEnergy(x);

   // Optimize.
   x.SetTrueVector();
   Vector b;
   solver->Mult(b, x.GetTrueVector());
   x.SetFromTrueVector();

   final_energy = nlf->GetParGridFunctionEnergy(x);
   if (x.ParFESpace()->GetMyRank() == 0)
   {
      std::cout << "Initial energy: " << init_energy << endl
                << "Final energy:   " << final_energy << endl;
   }
}

Mesh *SetupEdgeMesh2D(Mesh *mesh, GridFunction &attr_count_ser,
                      Array<int> &attr_marker_ser, int attr)
{
   Array<int> facedofs(0);
   int spaceDim = mesh->SpaceDimension();
   MFEM_VERIFY(spaceDim == 2, "Only 2D meshes supported right now.");
   Array<int> edofs;
   GridFunction *x = mesh->GetNodes();
   MFEM_VERIFY(x, "Mesh nodal space not set\n");
   const FiniteElementSpace *fes = mesh->GetNodalFESpace();
   int mesh_poly_deg = fes->GetMaxElementOrder();
   int nedges = mesh->GetNEdges();

   FiniteElementSpace *fespace_attr = attr_count_ser.FESpace();

   Array<int> ev, dofs;
   int attr_marker_dofs[2], attr_count_dofs[2];
   Array<int> edge_to_include;
   for (int ei = 0; ei < nedges; ei++)
   {
      mesh->GetEdgeVertices(ei, ev);
      MFEM_VERIFY(ev.Size(), "Could not get edge vertices.");
      for (int j = 0; j < 2; j++)
      {
         int v_id = ev[j];
         fespace_attr->GetVertexDofs(v_id, dofs);
         attr_marker_dofs[j] = attr_marker_ser[dofs[0]];
         attr_count_dofs[j] = attr_count_ser[dofs[0]];
      }
      auto hasAtLeastOneBitSet = [](int val, int j)
      {
         int mask = (1 << (j-1));
         return (val & mask) == mask;
      };
      bool dof1_mask2 = hasAtLeastOneBitSet(attr_marker_dofs[0], attr);
      bool dof2_mask2 = hasAtLeastOneBitSet(attr_marker_dofs[1], attr);
      if ( (dof1_mask2 && dof2_mask2))
      {
         edge_to_include.Append(ei);
      }
   }

   // Setup a mesh with dummy vertices
   int nel = edge_to_include.Size();
   Vector vals;
   Mesh *intmesh = new Mesh(1, nel*2, nel, 0, spaceDim);
   {
      for (int i = 0; i < nel; i++)
      {
         for (int j = 0; j < 2; j++) // 2 vertices per element
         {
            Vector vert(spaceDim);
            vert = 0.5;
            intmesh->AddVertex(vert.GetData());
         }
         Array<int> verts(2);
         for (int d = 0; d < 2; d++)
         {
            verts[d] = i*2+d;
         }
         intmesh->AddSegment(verts, 1);
      }
      intmesh->Finalize(true, true);
      intmesh->FinalizeTopology(false);
      intmesh->SetCurvature(mesh_poly_deg, false, -1, 0);
   }

   const FiniteElementSpace *intnodespace = intmesh->GetNodalFESpace();
   GridFunction *intnodes = intmesh->GetNodes();

   for (int i = 0; i < nel; i++)
   {
      int ei = edge_to_include[i];
      fes->GetEdgeVDofs(ei, dofs);
      x->GetSubVector(dofs, vals);
      Array<int> edofs;
      intnodespace->GetElementVDofs(i, edofs);
      intnodes->SetSubVector(edofs, vals);
   }

   return intmesh;
}

real_t GetMinDet(ParMesh *pmesh,
                 ParGridFunction &x,
                 int quad_order)
{
   int NE = pmesh->GetNE();
   ParFiniteElementSpace &pfes = *(x.ParFESpace());
   real_t min_det = infinity();
   for (int e = 0; e < NE; e++)
   {
      const IntegrationRule &ir =
         IntRulesLo.Get(pfes.GetFE(e)->GetGeomType(), quad_order);
      ElementTransformation *transf = pmesh->GetElementTransformation(e);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         transf->SetIntPoint(&ir.IntPoint(q));
         real_t det = transf->Jacobian().Det();
         min_det = fmin(min_det, det);
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &min_det, 1, MPI_DOUBLE,
                 MPI_MIN, pfes.GetComm());
   return min_det;
}