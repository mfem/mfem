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
using namespace std;
using namespace mfem;

void rotationTransformation(const Vector &p, Vector &v)
{
   double alpha = 30.0*M_PI/180.0;
   double beta   = 45.0*M_PI/180.0;
   double gamma   = 15.0*M_PI/180.0;

   DenseMatrix R(3);
   double ca = cos(alpha), sa = sin(alpha);
   double cb = cos(beta), sb = sin(beta);
   double cg = cos(gamma), sg = sin(gamma);

   R(0,0) = ca * cb;
   R(0,1) = ca * sb * sg - sa * cg;
   R(0,2) = ca * sb * cg + sa * sg;

   R(1,0) = sa * cb;
   R(1,1) = sa * sb * sg + ca * cg;
   R(1,2) = sa * sb * cg - ca * sg;

   R(2,0) = -sb;
   R(2,1) = cb * sg;
   R(2,2) = cb * cg;

   R.Mult(p, v);
}

void warpingTransformation(const Vector &p, Vector &v)
{
   double a = 0.2;   // interior warping
   double b = 0.5;   // diagonal stretching
   double c = 1.5;   // boundary curvature control
   double d = 1.25;
   double x = p(0),
          y = p(1),
          z = p(2);

   // --- Non-uniform spacing adjustment ---
   x = x + x*(1 - x)*0.4;
   y = y + y*(1 - y)*0.4;
   z = z + z*(1 - z)*0.4;

   v(0) = x + a * sin(0.5 * M_PI * x) * sin(c * M_PI * y)   +
          b * x * y - 0.1*sin(d*M_PI*z*x*y);
   v(1) = y + a * sin(c * M_PI * x)   * sin(0.5 * M_PI * y) +
          b * x * y - 0.1*sin(d*M_PI*z*x*y);
   v(2) = z;
}

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
              GridFunction::PLBound *plb,
              ParGridFunction *detgf, int solver_iter,
              bool move_bnd, Array<int> surf_mesh_attr,
              Array<int> aux_ess_dofs, int solver_type, bool adapt_qp,
              real_t limiting = 0.0);

   // Optimizes the node positions given in x.
   // When we enter, x contains the initial node positions.
   // When we exit, x contains the optimized node positions.
   // The underlying mesh of x remains unchanged (its positions don't change).
   void OptimizeNodes(ParGridFunction &x);

#ifdef MFEM_USE_GSLIB
   void SetupTangentialRelaxationFor2DEdge(ParFiniteElementSpace *pfespace,
                                           int bdr_attr,
                                           FindPointsGSLIB *finder,
                                           GridFunction *nodes0);

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

   void EnableVisualization(DataCollection *dc, int vis_cycle, int vis_freq)
   {
      tmop_integ->SetVisualization(dc, vis_freq, vis_cycle, x_nodes);
   }

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

void SetupBdrDofAttributes(ParFiniteElementSpace *pfesmarker,
                           Array<int> &attr_marker)
{
   ParMesh *pmesh = pfesmarker->GetParMesh();
   MFEM_VERIFY(attr_marker.Size() == pfesmarker->GetVSize(),
               "attr_marker size mismatch");
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

   GroupCommunicator &gcomm = pfesmarker->GroupComm();
   gcomm.Reduce<int>(attr_marker, GroupCommunicator::BitOR);
   gcomm.Bcast(attr_marker);
}

// Mark only the dofs that are on interfaces between different attributes.
// The attribute for such dofs is element attribute + max_bdr_attr
void SetupInterfaceAttributes(ParFiniteElementSpace *pfesmarker,
                              Array<int> &attr_marker)
{
   ParMesh *pmesh = pfesmarker->GetParMesh();
   int nbdr_attr = pmesh->bdr_attributes.Max();
   MFEM_VERIFY(attr_marker.Size() == pfesmarker->GetVSize(),
               "attr_marker size mismatch");

   L2_FECollection l2_fec(0, pmesh->Dimension());
   ParFiniteElementSpace l2_fes(pmesh, &l2_fec);
   ParGridFunction elattr(&l2_fes);
   for (int e = 0; e < pmesh->GetNE(); e++)
   {
      int attr = pmesh->GetAttribute(e);
      elattr(e) = attr;
   }
   elattr.ExchangeFaceNbrData();
   if (true)
   {
      socketstream vis;
      common::VisualizeField(vis, "localhost", 19916, elattr,
                              "attribute", 300, 0, 300, 300, "jRmclA");
   }
   Array<int> dofs;

   int int_face_count = 0;
   for (int f = 0; f < pmesh->GetNumFaces(); f++)
   {
      Mesh::FaceInformation fi = pmesh->GetFaceInformation(f);
      if (!fi.IsInterior())
      {
         continue;
      }
      int_face_count++;
      int attr1 = pmesh->GetAttribute(fi.element[0].index);
      int e1 = fi.element[0].index;
      int e2 = fi.element[1].index;
      int attr2;
      if (fi.IsShared())
      {
         e2 += pmesh->GetNE();
         Vector vals;
         elattr.GetElementDofValues(e2, vals);
         attr2 = vals(0);
      }
      else
      {
         attr2 = pmesh->GetAttribute(e2);
      }
      // if ((attr1 == 2 && attr2 == 3) ||
      //     (attr1 == 3 && attr2 == 2))
      // {
      //    std::cout << e1 << " " << e2 << " k10 found interface between 2 and 3\n";
      // }
      if (attr1 != attr2)
      {
         int a1 = attr1 < attr2 ? attr1 : attr2;
         int a2 = attr1 < attr2 ? attr2 : attr1;
         a1 += nbdr_attr;
         a2 += nbdr_attr;
         // if (a1 == 4 && a2 == 5)
         // {
         //    std::cout << " k10 found interface between 4 and 5\n";
         // }

         pfesmarker->GetFaceVDofs(f, dofs);
         for (int i = 0; i < dofs.Size(); i++)
         {
            attr_marker[dofs[i]] = attr_marker[dofs[i]] | (1 << (a1-1));
            attr_marker[dofs[i]] = attr_marker[dofs[i]] | (1 << (a2-1));
         }
      }
   }

   GroupCommunicator &gcomm = pfesmarker->GroupComm();
   gcomm.Reduce<int>(attr_marker, GroupCommunicator::BitOR);
   gcomm.Bcast(attr_marker);
}

void SetupDofAttributesBdrandInt(ParGridFunction &attr_count,
                                 Array<int> &attr_marker)
{
   ParFiniteElementSpace *pfesmarker = attr_count.ParFESpace();
   ParMesh *pmesh = pfesmarker->GetParMesh();
   pmesh->SetAttributes();
   int nbdr_attr = pmesh->bdr_attributes.Max();
   attr_marker.SetSize(attr_count.Size());
   attr_marker = 0;

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

   SetupBdrDofAttributes(pfesmarker, attr_marker); // this marks boundaries
   SetupInterfaceAttributes(pfesmarker, attr_marker); // this marks interfaces
   for (int i = 0; i < attr_count.Size(); i++)
   {
      attr_count(i) = countBits(attr_marker[i]);
   }
}

void SetupDofAttributes(
   std::map<std::tuple<int, int>, int> &interface_attrs,
   ParGridFunction &attr_count, Array<int> &attr_marker)
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

   L2_FECollection l2_fec(0, pmesh->Dimension());
   ParFiniteElementSpace l2_fes(pmesh, &l2_fec);
   ParGridFunction elattr(&l2_fes);
   for (int e = 0; e < pmesh->GetNE(); e++)
   {
      int attr = pmesh->GetAttribute(e);
      elattr(e) = attr;
   }
   elattr.ExchangeFaceNbrData();

   for (int f = 0; f < pmesh->GetNumFaces(); f++)
   {
      Mesh::FaceInformation fi = pmesh->GetFaceInformation(f);
      if (!fi.IsInterior())
      {
         continue;
      }
      int attr1 = pmesh->GetAttribute(fi.element[0].index);
      int e2 = fi.element[1].index;
      if (fi.IsShared())
      {
         e2 += pmesh->GetNE();
      }
      int attr2 = -1;
      if (e2 >= 0)
      {
         // This is an interior face, so we have two attributes.
         // We need to check if the second element has an attribute.
         attr2 = pmesh->GetAttribute(e2);
      }
      else
      {
         Vector vals;
         elattr.GetElementDofValues(pmesh->GetNE() + (-1-e2), vals);
         attr2 = vals(0);
      }
      if (attr1 != attr2)
      {
         int a1 = attr1 < attr2 ? attr1 : attr2;
         int a2 = attr1 < attr2 ? attr2 : attr1;
         auto it = interface_attrs.find(std::make_tuple(a1, a2));
         MFEM_VERIFY(it != interface_attrs.end(), "Interface attribute pair not found");

         pfesmarker->GetFaceVDofs(f, dofs);
         for (int i = 0; i < dofs.Size(); i++)
         {
            int val = attr_marker[dofs[i]];
            int attrib = interface_attrs[std::make_tuple(a1, a2)];
            attr_marker[dofs[i]] = val | (1 << (attrib-1));
         }
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

Array<int> *GetBdrFaceDofsForAttr(ParGridFunction &attr_count,
                                  Array<int> &attr_marker,
                                  int attr, bool scalar)
{
   ParFiniteElementSpace *pfesmarker = attr_count.ParFESpace();
   ParMesh *pmesh = pfesmarker->GetParMesh();
   int nbdr_faces = pmesh->GetNFbyType(FaceType::Boundary);
   Array<int> *facedofs = new Array<int>;

   // check if jth bit is set
   auto hasBitSet = [](int val, int j)
   {
      int mask = (1 << (j-1));
      return (val & mask) == mask;
   };
   Array<int> dofs;

   for (int f = 0; f < nbdr_faces; f++)
   {
      if (scalar)
      {
         pfesmarker->GetBdrElementDofs(f, dofs);
      }
      else
      {
         pfesmarker->GetBdrElementVDofs(f, dofs);
      }
      for (int i = 0; i < dofs.Size(); i++)
      {
         int val = attr_marker[dofs[i]];
         if (hasBitSet(val, attr))
         {
            facedofs->Append(dofs[i]);
         }
      }
   }
   facedofs->Sort();
   facedofs->Unique();
   return facedofs;
}

Array<int> *GetBdrEdgeDofsForAttr(ParGridFunction &attr_count,
                                  Array<int> &attr_marker,
                                  int attr1, int attr2, bool scalar)
{
   ParFiniteElementSpace *pfesmarker = attr_count.ParFESpace();
   ParMesh *pmesh = pfesmarker->GetParMesh();
   Array<int> *edgedofs = new Array<int>;

   // check if jth and kth bits are set
   auto hasAtLeastTwoBitsSet = [](int val, int j, int k)
   {
      int mask = (1 << (j-1)) | (1 << (k-1));
      return (val & mask) == mask;
   };
   int nedges = pmesh->GetNEdges();

   Array<int> dofs;
   for (int ei = 0; ei < nedges; ei++)
   {
      if (scalar)
      {
         pfesmarker->GetEdgeDofs(ei, dofs);
      }
      else
      {
         pfesmarker->GetEdgeVDofs(ei, dofs);
      }
      for (int i = 0; i < dofs.Size(); i++)
      {
         int val = attr_marker[dofs[i]];
         bool dof1_mask2 = hasAtLeastTwoBitsSet(val, attr1, attr2);
         bool dof2_mask2 = hasAtLeastTwoBitsSet(val, attr1, attr2);
         if ( (dof1_mask2 && dof2_mask2))
         {
            edgedofs->Append(dofs[i]);
         }
      }
   }
   edgedofs->Sort();
   edgedofs->Unique();
   return edgedofs;
}

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
Array<int> IdentifyAuxiliaryEssentialDofs2(ParGridFunction &attr_count)
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

// Get indices of all dofs that have more than or equal to `dim` attributes.
// Essentially dofs that should not be allowed to move.
// Works for both scalar and vector spaces
Array<int> IdentifyAuxiliaryEssentialDofs3(ParFiniteElementSpace *pfespace,
                                           Array<int> &attr_marker)
{
   Array<int> aux_dofs;
   int dim = pfespace->GetParMesh()->SpaceDimension();
   ParMesh *pmesh = pfespace->GetParMesh();
   int nbdr_attr = pmesh->bdr_attributes.Max();
   int nel_attr = pmesh->attributes.Max();
   int ntot_attr = nbdr_attr + nel_attr;
   // any dof with 2 boundary attributes or 1 boundary and 1 interface attribute

   auto countBitsTill = [](int val, int till)
   {
      int count = 0;
      int iter = 0;
      while (val && iter < till)
      {
         count += val & 1;
         val >>= 1;
         iter++;
      }
      return count;
   };

   if (dim == 2)
   {
      for (int i = 0; i < attr_marker.Size(); i++)
      {
         int bdr_attr_count = countBitsTill(attr_marker[i], nbdr_attr);
         int total_attr_count = countBitsTill(attr_marker[i], ntot_attr);
         int interface_attr_count = total_attr_count - bdr_attr_count;
         if (bdr_attr_count >= 2 || (bdr_attr_count >= 1 && interface_attr_count >= 1))
         {
            aux_dofs.Append(i);
         }
      }
   }
   else
   {
      MFEM_ABORT("Not implemented for 3D yet");
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
   int dim = pfespace->GetParMesh()->SpaceDimension();
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
      // std::cout << "attr_count[" << i << "] = " << attr_count[i] << std::endl;
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
   int dim = pfespace->GetParMesh()->SpaceDimension();
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
Array<int> *GetDofMatchingInterfaceAttributes(ParGridFunction &attr_count,
                                              Array<int> &attr_marker,
                                              int attr1, int attr2)
{
   ParFiniteElementSpace *pfespace = attr_count.ParFESpace();
   Array<int> *facedofs = new Array<int>;
   int dim = pfespace->GetParMesh()->SpaceDimension();
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
      int val = attr_marker[i];
      if (hasBitSet(val, attr1) && hasBitSet(val, attr2))
      {
         facedofs->Append(i);
      }
   }
   facedofs->Sort();
   facedofs->Unique();
   return facedofs;
}

std::map<std::tuple<int, int>, int> Get2DInterfaceAttributePairs(Mesh *mesh)
{
   int max_bdr_attr = mesh->bdr_attributes.Max();
   int max_el_attr = mesh->attributes.Max();
   int min_el_attr = mesh->attributes.Min();
   std::map<std::tuple<int, int>, int> interface_attrs;
   if (max_el_attr == min_el_attr)
   {
      return interface_attrs;
   }
   int flag = max_bdr_attr+1;

   for (int f = 0; f < mesh->GetNumFaces(); f++)
   {
      Mesh::FaceInformation fi = mesh->GetFaceInformation(f);
      if (!fi.IsInterior())
      {
         continue;
      }
      int attr1 = mesh->GetAttribute(fi.element[0].index);
      int attr2 = mesh->GetAttribute(fi.element[1].index);
      if (attr1 != attr2)
      {
         int a1 = attr1 < attr2 ? attr1 : attr2;
         int a2 = attr1 < attr2 ? attr2 : attr1;
         auto it = interface_attrs.find(std::make_tuple(a1, a2));
         if (it == interface_attrs.end())
         {
            interface_attrs[std::make_tuple(a1, a2)] = flag++;
         }
      }
   }
   return interface_attrs;
}

void SetupSerialDofAttributes(
   std::map<std::tuple<int, int>, int> &interface_attrs,
                              GridFunction &attr_count,
                              Array<int> &attr_marker)
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

   for (int f = 0; f < mesh->GetNumFaces(); f++)
   {
      Mesh::FaceInformation fi = mesh->GetFaceInformation(f);
      if (!fi.IsInterior())
      {
         continue;
      }
      int attr1 = mesh->GetAttribute(fi.element[0].index);
      int attr2 = mesh->GetAttribute(fi.element[1].index);
      if (attr1 != attr2)
      {
         int a1 = attr1 < attr2 ? attr1 : attr2;
         int a2 = attr1 < attr2 ? attr2 : attr1;
         auto it = interface_attrs.find(std::make_tuple(a1, a2));
         MFEM_VERIFY(it != interface_attrs.end(), "Interface attribute pair not found");


         fesmarker->GetFaceVDofs(f, dofs);
         for (int i = 0; i < dofs.Size(); i++)
         {
            int val = attr_marker[dofs[i]];
            int attrib = interface_attrs[std::make_tuple(a1, a2)];
            attr_marker[dofs[i]] = val | (1 << (attrib-1));
         }
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

Mesh *SetupEdgeMesh3D(ParMesh *pmesh, ParGridFunction &attr_count,
                       Array<int> &attr_marker,
                       int attr1, int attr2)
{
   Array<int> facedofs(0);
   int spaceDim = pmesh->SpaceDimension();
   MFEM_VERIFY(spaceDim == 3, "Only 3D meshes supported right now.");
   Array<int> edofs;
   GridFunction *x = pmesh->GetNodes();
   MFEM_VERIFY(x, "Mesh nodal space not set\n");
   const FiniteElementSpace *fes = pmesh->GetNodalFESpace();
   int mesh_poly_deg = fes->GetMaxElementOrder();
   int nedges = pmesh->GetNEdges();
   ParFiniteElementSpace *pfespace_attr = attr_count.ParFESpace();

   Array<int> ev, dofs;
   int attr_marker_dofs[2], attr_count_dofs[2];
   Array<int> edge_to_include;
   for (int ei = 0; ei < nedges; ei++)
   {
      pmesh->GetEdgeVertices(ei, ev);
      MFEM_VERIFY(ev.Size(), "Could not get edge vertices.");
      for (int j = 0; j < 2; j++)
      {
         int v_id = ev[j];
         pfespace_attr->GetVertexDofs(v_id, dofs);
         attr_marker_dofs[j] = attr_marker[dofs[0]];
         attr_count_dofs[j] = attr_count[dofs[0]];
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
      intmesh->SetCurvature(fes->FEColl()->GetOrder(),
                            fes->IsDGSpace(),
                            spaceDim,
                            fes->GetOrdering());
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
      // vals.Print();
      Array<int> edofs;
      intnodespace->GetElementVDofs(i, edofs);
      // edofs.Print();
      // vals.Print();
      intnodes->SetSubVector(edofs, vals);
   }
   // intnodes->Print();

   return intmesh;
}

// Get indices of all edges that have both the required attributes marked.
Array<int> GetBdrEdgeIndices(ParMesh *pmesh,
                             ParGridFunction &attr_count,
                             Array<int> &attr_marker,
                             int attr1, int attr2)
{
   ParFiniteElementSpace *pfesmarker = attr_count.ParFESpace();
   int nedges = pmesh->GetNEdges();
   Array<int> edge_to_include;

   ParFiniteElementSpace *pfespace_attr = attr_count.ParFESpace();

   Array<int> ev, dofs;
   int attr_marker_dofs[2];
   for (int ei = 0; ei < nedges; ei++)
   {
      pmesh->GetEdgeVertices(ei, ev);
      MFEM_VERIFY(ev.Size(), "Could not get edge vertices.");
      for (int j = 0; j < 2; j++)
      {
         int v_id = ev[j];
         pfespace_attr->GetVertexDofs(v_id, dofs);
         attr_marker_dofs[j] = attr_marker[dofs[0]];
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
   return edge_to_include;
}

Array<int> GetBdrEdgeAttrList3D(ParMesh *pmesh, ParGridFunction &attr_count,
                                Array<int> &attr_marker)
{
   pmesh->SetAttributes();
   int max_bdr_attr = pmesh->bdr_attributes.Max();
   int max_el_attr = pmesh->attributes.Max();
   Array<int> surf_mesh_edge_attr;

   auto setTwoBits = [](int j, int k)
   {
      return (1 << (j-1)) | (1 << (k-1));
   };

   // find which pairs of boundary attributes are present on edges
   for (int i = 0; i < max_bdr_attr; i++)
   {
      for (int j = i+1; j < max_bdr_attr; j++)
      {
         Array<int> edge_to_include = GetBdrEdgeIndices(pmesh, attr_count,
                                                        attr_marker, i+1, j+1);
         int nedges = edge_to_include.Size();
         int nedges_glob = nedges;
         MPI_Allreduce(MPI_IN_PLACE, &nedges_glob, 1, MPI_INT, MPI_SUM,
                        pmesh->GetComm());
         if (nedges_glob == 0) { continue; }
         surf_mesh_edge_attr.Append(setTwoBits(i+1, j+1));
      }
   }

   return surf_mesh_edge_attr;
}

Array<int> GetInterfaceEdgeList2D(ParMesh *pmesh, ParGridFunction &attr_count,
                                  Array<int> &attr_marker)
{
   pmesh->SetAttributes();
   int max_bdr_attr = pmesh->bdr_attributes.Max();
   int max_el_attr = pmesh->attributes.Max();
   Array<int> surf_mesh_edge_attr;

   auto setTwoBits = [](int j, int k)
   {
      return (1 << (j-1)) | (1 << (k-1));
   };

   // find which pairs of boundary attributes are present on edges
   for (int i = max_bdr_attr; i < max_bdr_attr+max_el_attr; i++)
   {
      for (int j = i+1; j < max_bdr_attr+max_el_attr; j++)
      {
         Array<int> edge_to_include = GetBdrEdgeIndices(pmesh, attr_count,
                                                        attr_marker, i+1, j+1);
         int nedges = edge_to_include.Size();
         int nedges_glob = nedges;
         MPI_Allreduce(MPI_IN_PLACE, &nedges_glob, 1, MPI_INT, MPI_SUM,
                        pmesh->GetComm());
         if (nedges_glob == 0) { continue; }
         surf_mesh_edge_attr.Append(setTwoBits(i+1, j+1));
      }
   }
   return surf_mesh_edge_attr;
}

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
void MeshOptimizer::SetupTangentialRelaxationFor2DEdge(ParFiniteElementSpace
                                                       *pfespace,
                                                       int bdr_attr,
                                                       FindPointsGSLIB *finder,
                                                       GridFunction *nodes0)
{
   Array<int> *fdofs = new Array<int>;
   ParMesh *pmesh = pfespace->GetParMesh();

   Array<int> dofs;
   int nbdr_faces = pmesh->GetNFbyType(FaceType::Boundary);
   // int nfaces = 0;
   for (int f = 0; f < nbdr_faces; f++)
   {
      int attrib = pmesh->GetBdrAttribute(f);
      if (attrib == bdr_attr)
      {
         // nfaces += 1;
         pfespace->GetBdrElementDofs(f, dofs);
         fdofs->Append(dofs);
      }
   }
   fdofs->Sort();
   fdofs->Unique();
   // std::cout << "Found " << fdofs->Size() << " dofs for bdr attr "
   //  << bdr_attr << std::endl;
   finder_arr.Append(finder);
   tang_dofs_arr.Append(fdofs);
   nodes0_arr.Append(nodes0);
}

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

Array<int> IdentifyAuxiliaryEssentialDofs(ParFiniteElementSpace *pfespace)
{
   ParMesh *pmesh = pfespace->GetParMesh();
   Array<int> dof_flags(pfespace->GetVSize());
   Array<int> dof_attr(pfespace->GetVSize());
   Array<long long> dof_id(pfespace->GetVSize());
   int dim = pmesh->Dimension();
   MFEM_VERIFY(dim == 2, "Only 2D meshes supported ffor aux dofs.");

   dof_flags = 0; // not an essential dof
   dof_attr = 0; // no attribute assigned yet

   Array<int> dofs;
   int nbdr_faces = pmesh->GetNFbyType(FaceType::Boundary);
   // int nfaces = 0;
   for (int f = 0; f < nbdr_faces; f++)
   {
      int attrib = pmesh->GetBdrAttribute(f);
      pfespace->GetBdrElementVDofs(f, dofs);
      for (int d = 0; d < dofs.Size(); d++)
      {
         int dof_idx = dofs[d];
         if (dof_flags[dof_idx] == 0)
         {
            if (dof_attr[dof_idx] == 0)
            {
               dof_attr[dof_idx] = attrib;
            }
            else if (dof_attr[dof_idx] != attrib)
            {
               dof_flags[dof_idx] = 1; // not an essential dof
            }
         }
      }
   }
   for (int i = 0; i < pfespace->GetVSize(); i++)
   {
      dof_id[i] = (long long)(pfespace->GetGlobalTDofNumber(i));
   }

   GSOPGSLIB gsop(pfespace->GetComm(), dof_id);
   gsop.GS(dof_flags, GSOPGSLIB::MAX);

   Array<int> dof_attr_min = dof_attr;
   Array<int> dof_attr_max = dof_attr;

   gsop.GS(dof_attr_min, GSOPGSLIB::MIN);
   gsop.GS(dof_attr_max, GSOPGSLIB::MAX);

   for (int i = 0; i < dof_attr_min.Size(); i++)
   {
      if (dof_attr_min[i] != dof_attr_max[i])
      {
         dof_flags[i] = 1; // not an essential dof
      }
   }

   Array<int> aux_dofs;
   for (int i = 0; i < dof_flags.Size(); i++)
   {
      if (dof_flags[i] == 1)
      {
         aux_dofs.Append(i);
      }
   }

   return aux_dofs;
}
#endif

void MeshOptimizer::Setup(ParGridFunction &x,
                          double *min_det_ptr,
                          int quad_order,
                          int metric_id, int target_id,
                          GridFunction::PLBound *plb,
                          ParGridFunction *detgf, int solver_iter,
                          bool move_bnd, Array<int> surf_mesh_attr,
                          Array<int> aux_ess_dofs, int solver_type,
                          bool adapt_qp, real_t limiting)
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
   int max_quad_order = 400;
   if (plb)
   {
      tmop_integ->SetPLBoundsForDeterminant(plb, detgf);
      if (adapt_qp)
      {
         tmop_integ->EnableAdaptiveIntegrationRule(pfes.GetNE(), quad_order,
         max_quad_order);
      }
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
   // Solver *S_prec = NULL;
   lin_solver = new MINRESSolver(pfes.GetComm());
   lin_solver->SetMaxIter(100);
   lin_solver->SetRelTol(1e-12);
   lin_solver->SetAbsTol(0.0);
   IterativeSolver::PrintLevel minres_pl;
   lin_solver->SetPrintLevel(minres_pl.FirstAndLast().Summary());
   {
      // auto hs = new HypreSmoother;
      // hs->SetType(HypreSmoother::l1Jacobi, 1);
      // hs->SetPositiveDiagonal(true);
      // S_prec = hs;
      // lin_solver->SetPreconditioner(*S_prec);
   }

   // Nonlinear solver.
   const IntegrationRule &ir =
      IntRulesLo.Get(pfes.GetFE(0)->GetGeomType(), quad_order);
   solver = new TMOPNewtonSolver(pfes.GetComm(), ir, solver_type);
   solver->SetIntegrationRules(IntRulesLo, quad_order);
   solver->SetOperator(*nlf);
   if (solver_type == 0)
   {
      solver->SetPreconditioner(*lin_solver);
   }
   if (min_det_ptr)
   {
      solver->SetMinDetPtr(min_det_ptr);
   }
   if (plb) { solver->SetDeterminantBound(true); }
   solver->SetMaxIter(solver_iter);
   solver->SetRelTol(1e-8);
   solver->SetAbsTol(0.0);
   IterativeSolver::PrintLevel newton_pl;
   solver->SetPrintLevel(newton_pl.Iterations().Summary());
   if (plb && adapt_qp)
   {
      solver->EnableQuadOrderIncrement(max_quad_order, 5.0); /// new
   }
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

void GetMinDet(ParMesh *pmesh, ParGridFunction &x,
               int quad_order,
               real_t &min_det, real_t &volume)
{
   int NE = pmesh->GetNE();
   // int dim = pmesh->Dimension();
   ParFiniteElementSpace &pfes = *(x.ParFESpace());
   min_det = infinity();
   volume = 0.0;
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
         volume += ir.IntPoint(q).weight * det;
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &min_det, 1, MPI_DOUBLE,
                 MPI_MIN, pfes.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &volume, 1, MPI_DOUBLE, MPI_SUM, pfes.GetComm());
}

void GetMeshStats(ParMesh *pmesh, ParGridFunction &x,
                  TMOP_QualityMetric *metric,  TargetConstructor *target_c,
                  int quad_order,
                  double &min_det, double &min_muT, double &max_muT,
                  double &avg_muT, double &volume,
                  Array<int> *quad_order_list = nullptr)
{
   int NE = pmesh->GetNE();
   int dim = pmesh->Dimension();
   ParFiniteElementSpace &pfes = *(x.ParFESpace());
   min_det = infinity();
   min_muT = infinity();
   max_muT = -infinity();
   avg_muT = 0.0;
   volume = 0.0;

   for (int e = 0; e < NE; e++)
   {
      int qo = quad_order_list ? (*quad_order_list)[e] : quad_order;
      const IntegrationRule &ir =
         IntRulesLo.Get(pfes.GetFE(e)->GetGeomType(), qo);
      ElementTransformation *transf = pmesh->GetElementTransformation(e);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         transf->SetIntPoint(&ir.IntPoint(q));
         min_det = fmin(min_det, transf->Jacobian().Det());
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &min_det, 1, MPI_DOUBLE,
                 MPI_MIN, pfes.GetComm());

   double integral_mu = 0.0;
   for (int i = 0; i < NE; i++)
   {
      int qo = quad_order_list != nullptr ? (*quad_order_list)[i] : quad_order;
      const FiniteElement &fe_pos = *x.FESpace()->GetFE(i);
      const IntegrationRule &ir = IntRulesLo.Get(fe_pos.GetGeomType(), qo);
      const int nsp = ir.GetNPoints(), dof = fe_pos.GetDof();

      DenseMatrix dshape(dof, dim);
      DenseMatrix pos(dof, dim);
      pos.SetSize(dof, dim);
      Vector posV(pos.Data(), dof * dim);

      Array<int> pos_dofs;
      x.FESpace()->GetElementVDofs(i, pos_dofs);
      x.GetSubVector(pos_dofs, posV);

      DenseTensor W(dim, dim, nsp);
      DenseMatrix Winv(dim), T(dim), A(dim);
      target_c->ComputeElementTargets(i, fe_pos, ir, posV, W);

      for (int j = 0; j < nsp; j++)
      {
         const DenseMatrix &Wj = W(j);
         metric->SetTargetJacobian(Wj);
         CalcInverse(Wj, Winv);

         const IntegrationPoint &ip = ir.IntPoint(j);
         fe_pos.CalcDShape(ip, dshape);
         MultAtB(pos, dshape, A);
         Mult(A, Winv, T);

         const double mu = metric->EvalW(T);
         max_muT = fmax(mu, max_muT);
         min_muT = fmin(mu, min_muT);
         integral_mu += mu * ip.weight * T.Det();
         volume += ip.weight * A.Det();
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &min_muT, 1, MPI_DOUBLE, MPI_MIN, pfes.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &max_muT, 1, MPI_DOUBLE, MPI_MAX, pfes.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &integral_mu, 1, MPI_DOUBLE, MPI_SUM,
                 pfes.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &volume, 1, MPI_DOUBLE, MPI_SUM, pfes.GetComm());
   avg_muT = integral_mu / volume;
}

void GetElementWiseMinDetAtQPs(ParMesh *pmesh, ParGridFunction &x,
                  int quad_order, Vector &min_el_det)
{
   int NE = pmesh->GetNE();
   int dim = pmesh->Dimension();
   ParFiniteElementSpace &pfes = *(x.ParFESpace());
   min_el_det.SetSize(NE);

   for (int e = 0; e < NE; e++)
   {
      const IntegrationRule &ir =
         IntRulesLo.Get(pfes.GetFE(e)->GetGeomType(), quad_order);
      ElementTransformation *transf = pmesh->GetElementTransformation(e);
      double min_det = infinity();
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         transf->SetIntPoint(&ir.IntPoint(q));
         min_det = fmin(min_det, transf->Jacobian().Det());
      }
      min_el_det(e) = min_det;
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

Mesh *SetupInterfaceEdgeMesh2D(Mesh *mesh, GridFunction &attr_count_ser,
                               Array<int> &attr_marker_ser,
                               int attr1, int attr2)
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
      bool dof1_mask1 = hasAtLeastOneBitSet(attr_marker_dofs[0], attr1);
      if (!dof1_mask1) { continue; }
      bool dof2_mask1 = hasAtLeastOneBitSet(attr_marker_dofs[1], attr1);
      if (!dof2_mask1) { continue; }
      bool dof1_mask2 = hasAtLeastOneBitSet(attr_marker_dofs[0], attr2);
      if (!dof1_mask2) { continue; }
      bool dof2_mask2 = hasAtLeastOneBitSet(attr_marker_dofs[1], attr2);
      if (!dof2_mask2) { continue; }
      edge_to_include.Append(ei);
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

// Mesh *SetupEdgeMesh2D(Mesh *mesh, int attr)
// {
//    Array<int> facedofs(0);
//    int spaceDim = mesh->SpaceDimension();
//    MFEM_VERIFY(spaceDim == 2, "Only 2D meshes supported right now.");
//    Array<int> edofs;
//    GridFunction *x = mesh->GetNodes();
//    MFEM_VERIFY(x, "Mesh nodal space not set\n");
//    const FiniteElementSpace *fes = mesh->GetNodalFESpace();
//    int mesh_poly_deg = fes->GetMaxElementOrder();
//    Array<int> edge_to_include;
//    for (int f = 0; f < mesh->GetNBE(); f++)
//    {
//       int attrib = mesh->GetBdrAttribute(f);
//       if (attrib == attr)
//       {
//          edge_to_include.Append(mesh->GetBdrElementFaceIndex(f));
//       }
//    }

//    int nfaces = edge_to_include.Size();
//    Mesh *intmesh = new Mesh(1, nfaces*2, nfaces, 0, spaceDim);
//    {
//       for (int i = 0; i < nfaces; i++)
//       {
//          for (int j = 0; j < 2; j++) // 2 vertices per element
//          {
//             Vector vert(spaceDim);
//             vert = 0.5;
//             intmesh->AddVertex(vert.GetData());
//          }
//          Array<int> verts(spaceDim);
//          for (int d = 0; d < spaceDim; d++)
//          {
//             verts[d] = i*spaceDim+d;
//          }
//          intmesh->AddSegment(verts, 1);
//       }
//       intmesh->Finalize(true, true);
//       intmesh->FinalizeTopology();
//       intmesh->SetCurvature(mesh_poly_deg, false, -1, 0);
//    }

//    const FiniteElementSpace *intnodespace = intmesh->GetNodalFESpace();
//    GridFunction *intnodes = intmesh->GetNodes();

//    Vector vals;
//    for (int i = 0; i < nfaces; i++)
//    {
//       int ei = edge_to_include[i];
//       fes->GetEdgeVDofs(ei, edofs);
//       x->GetSubVector(edofs, vals);
//       intnodespace->GetElementVDofs(i, edofs);
//       intnodes->SetSubVector(edofs, vals);
//    }

   // FaceElementTransformations *face_elem_transf;
   // int count = 0;
   // Vector vect;
   // Vector nodeval(spaceDim);
   // for (int f = 0; f < mesh->GetNBE(); f++)
   // {
   //    int attrib = mesh->GetBdrAttribute(f);
   //    int fnum = mesh->GetBdrElementFaceIndex(f);
   //    if (attrib == attr)
   //    {
   //       intnodespace->GetElementVDofs(count, edofs);
   //       const FiniteElement *fe = intnodespace->GetFE(count);
   //       face_elem_transf = mesh->GetFaceElementTransformations(fnum);
   //       IntegrationRule irule = fe->GetNodes();
   //       int npts = irule.GetNPoints();
   //       vect.SetSize(npts*spaceDim);
   //       for (int q = 0; q < npts; q++)
   //       {
   //          IntegrationPoint &ip = irule.IntPoint(q);
   //          IntegrationPoint eip;
   //          face_elem_transf->Loc1.Transform(ip, eip);
   //          x->GetVectorValue(face_elem_transf->Elem1No, eip, nodeval);
   //          for (int d = 0; d < spaceDim; d++)
   //          {
   //             vect(q + npts*d) = nodeval(d);
   //          }
   //       }
   //       intnodes->SetSubVector(edofs, vect);
   //       count++;
   //    }
   // }
   // MFEM_ABORT(" ");

//    return intmesh;
// }
