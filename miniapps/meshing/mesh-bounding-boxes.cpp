// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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
//    ---------------------------------------------------------------------
//    Bounding Boxes Miniapp: Construct Bounding Boxes of Quad/Hex Meshes
//    ---------------------------------------------------------------------
//
// This miniapp computes bounding boxes for each element in a given mesh, and
// also computes the bounds on the determinant of the Jacobian of the
// transformation for each element. The bounding approach is based on the
// method described in:
//
// (1) Section 3 of Mittal et al., "General Field Evaluation in High-Order
//     Meshes on GPUs"
// and
// (2) Dzanic et al., "A method for bounding high-order finite element
//     functions: Applications to mesh validity and bounds-preserving limiters".
//
//
// Compile with: make mesh-bounding-boxes
//
// Sample runs:
//  mpirun -np 4 mesh-bounding-boxes -m ../../data/klein-bottle.mesh
//  mpirun -np 4 mesh-bounding-boxes -m ../gslib/triple-pt-1.mesh
//  mpirun -np 4 mesh-bounding-boxes -m ../../data/star-surf.mesh
//  mpirun -np 4 mesh-bounding-boxes -m ../../data/fichera-q2.mesh

#include "mfem.hpp"
#include <iostream>
#include <fstream>

using namespace mfem;
using namespace std;

Mesh MakeBoundingBoxMesh(Mesh &mesh, GridFunction &nodal_bb_gf);
void GetDeterminantJacobianGF(ParMesh *mesh, ParGridFunction *detgf);
void VisualizeBB(Mesh &mesh, char *title, int pos_x, int pos_y);
void VisualizeField(ParMesh &pmesh, ParGridFunction &input,
                    char *title, int pos_x, int pos_y);

int main (int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();

   // Set the method's default parameters.
   const char *mesh_file = "../../data/klein-bottle.mesh";
   int mesh_poly_deg     = 2;
   bool visualization    = true;
   bool visit            = false;
   bool jacobian         = true;

   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit",
                  "--no-visit",
                  "Enable or disable VisIt output.");
   args.AddOption(&jacobian, "-jac", "--jacobian", "-no-jac",
                  "--no-jacobian",
                  "Compute bounds on determinant of mesh Jacobian");
   args.ParseCheck();

   // Initialize and refine the starting mesh.
   Mesh mesh(mesh_file, 1, 1, false);
   const int rdim = mesh.Dimension();
   const int sdim = mesh.SpaceDimension();

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   if (pmesh.GetNodes() == NULL) { pmesh.SetCurvature(mesh_poly_deg); }
   else { mesh_poly_deg = pmesh.GetNodes()->FESpace()->GetMaxElementOrder(); }
   mesh.Clear();

   // Setup finite element space and gridfunction to store bounding box
   // x/y/z min & max for each element.
   L2_FECollection fec_pc(0, rdim);
   ParFiniteElementSpace fes_l2_bb(&pmesh, &fec_pc, sdim*2, Ordering::byVDIM);
   ParGridFunction nodal_bb(&fes_l2_bb);
   Array<int> vdofs;

   GridFunction *nodes = pmesh.GetNodes();
   int nelem = pmesh.GetNE();

   // Compute bounds on nodal positions and save in nodal_bb gridfunction.
   Vector lower, upper;
   nodes->GetElementBounds(lower, upper, 2, -1);
   for (int e = 0; e < nelem; e++)
   {
      fes_l2_bb.GetElementVDofs(e, vdofs);
      Vector lower_upper(vdofs.Size());
      for (int d = 0; d < sdim; d++)
      {
         lower_upper(d) = lower(e + d*nelem);
         lower_upper(d+sdim) = upper(e + d*nelem);
      }
      nodal_bb.SetSubVector(vdofs, lower_upper);
   }

   // Make a mesh of bounding boxes to output.
   Mesh pmesh_ser = pmesh.GetSerialMesh(0);
   GridFunction nodal_bb_ser = nodal_bb.GetSerialGridFunction(0, pmesh_ser);
   Mesh meshbb = MakeBoundingBoxMesh(pmesh_ser, nodal_bb_ser);

   // Output in GLVis and VisIt
   if (visualization && Mpi::Root())
   {
      char title1[] = "Input mesh";
      VisualizeBB(pmesh_ser, title1, 0, 0);
      char title2[] = "Bounding box mesh";
      VisualizeBB(meshbb, title2, 400, 0);
   }
   if (visit && Mpi::Root())
   {
      VisItDataCollection visit_dc("bounding-box-input", &pmesh_ser);
      visit_dc.SetFormat(DataCollection::SERIAL_FORMAT);
      visit_dc.Save();

      VisItDataCollection visit_dc_bb("bounding-box", &meshbb);
      visit_dc_bb.SetFormat(DataCollection::SERIAL_FORMAT);
      visit_dc_bb.Save();
   }

   // Print min and max bound of nodal gridfunction
   int ref_factor = 4;
   nodes->GetBounds(lower, upper, ref_factor);
   if (Mpi::Root())
   {
      out << "Nodal position minimum bounds:" << endl;
      lower.Print();
      out << "Nodal position maximum bounds:" << endl;
      upper.Print();
   }

   if (!jacobian) { return 0; }

   // Setup gridfunction for the determinant of the Jacobian.
   // Note: determinant order = rdim*mesh_order - 1 for quads/hexes
   int det_order = rdim*mesh_poly_deg-1;
   L2_FECollection fec_det(det_order, rdim, BasisType::GaussLobatto);
   ParFiniteElementSpace fespace_det(&pmesh, &fec_det);
   ParGridFunction detgf(&fespace_det);
   GetDeterminantJacobianGF(&pmesh, &detgf);

   // Setup piecewise constant gridfunction to save bounds on the determinant
   // of the Jacobian
   L2_FECollection fec_det_pc(0, rdim);
   ParFiniteElementSpace fes_det_pc(&pmesh, &fec_det_pc);
   ParGridFunction bounds_detgf_lower(&fes_det_pc);
   ParGridFunction bounds_detgf_upper(&fes_det_pc);

   // Compute bounds
   detgf.GetElementBounds(bounds_detgf_lower, bounds_detgf_upper, ref_factor);

   // GLVis Visualization
   if (visualization)
   {
      char title1[] = "Determinant of Jacobian (det J)";
      VisualizeField(pmesh, detgf, title1, 0, 465);
      char title2[] = "Element-wise lower bound on det J";
      VisualizeField(pmesh, bounds_detgf_lower, title2, 400, 465);
      char title3[] = "Element-wise upper bound on det J";
      VisualizeField(pmesh, bounds_detgf_upper, title3, 800, 465);
   }
   // Visit Visualization
   if (visit)
   {
      VisItDataCollection visit_dc("jacobian-determinant-bounds", &pmesh);
      visit_dc.SetFormat(DataCollection::PARALLEL_FORMAT);
      visit_dc.RegisterField("determinant", &detgf);
      visit_dc.RegisterField("det-lower-bound", &bounds_detgf_lower);
      visit_dc.RegisterField("det-upper-bound", &bounds_detgf_upper);
      visit_dc.Save();
   }

   // Print min and max bound of determinant gridfunction
   detgf.GetBounds(lower, upper, ref_factor);
   if (Mpi::Root())
   {
      out << "Jacobian determinant minimum bound: " << lower(0) << endl;
      out << "Jacobian determinant maximum bound: " << upper(0) << endl;
   }

   return 0;
}

Mesh MakeBoundingBoxMesh(Mesh &mesh, GridFunction &nodal_bb_gf)
{
   int nelem = mesh.GetNE();
   int sdim = mesh.SpaceDimension();
   int nverts = pow(2,sdim)*nelem;
   Mesh meshbb(sdim, nverts, nelem, 0, sdim);
   int eidx = 0;
   int vidx = 0;
   for (int e = 0; e < nelem; e++)
   {
      Vector xyzminmax_el;
      nodal_bb_gf.GetElementDofValues(e, xyzminmax_el);
      if (sdim == 2)
      {
         Vector xyz(2);
         xyz(0) = xyzminmax_el(0);
         xyz(1) = xyzminmax_el(1);
         meshbb.AddVertex(xyz);

         xyz(0) = xyzminmax_el(2);
         xyz(1) = xyzminmax_el(1);
         meshbb.AddVertex(xyz);

         xyz(0) = xyzminmax_el(2);
         xyz(1) = xyzminmax_el(3);
         meshbb.AddVertex(xyz);

         xyz(0) = xyzminmax_el(0);
         xyz(1) = xyzminmax_el(3);
         meshbb.AddVertex(xyz);

         const int inds[4] = {vidx++, vidx++, vidx++, vidx++};
         int attr = eidx+1;
         meshbb.AddQuad(inds, attr);
         eidx++;
      }
      else if (sdim == 3)
      {
         Vector xyz(3);
         xyz(0) = xyzminmax_el(0);
         xyz(1) = xyzminmax_el(1);
         xyz(2) = xyzminmax_el(2);
         meshbb.AddVertex(xyz);

         xyz(0) = xyzminmax_el(3);
         xyz(1) = xyzminmax_el(1);
         xyz(2) = xyzminmax_el(2);
         meshbb.AddVertex(xyz);

         xyz(0) = xyzminmax_el(3);
         xyz(1) = xyzminmax_el(4);
         xyz(2) = xyzminmax_el(2);
         meshbb.AddVertex(xyz);

         xyz(0) = xyzminmax_el(0);
         xyz(1) = xyzminmax_el(4);
         xyz(2) = xyzminmax_el(2);
         meshbb.AddVertex(xyz);

         xyz(0) = xyzminmax_el(0);
         xyz(1) = xyzminmax_el(1);
         xyz(2) = xyzminmax_el(5);
         meshbb.AddVertex(xyz);

         xyz(0) = xyzminmax_el(3);
         xyz(1) = xyzminmax_el(1);
         xyz(2) = xyzminmax_el(5);
         meshbb.AddVertex(xyz);

         xyz(0) = xyzminmax_el(3);
         xyz(1) = xyzminmax_el(4);
         xyz(2) = xyzminmax_el(5);
         meshbb.AddVertex(xyz);

         xyz(0) = xyzminmax_el(0);
         xyz(1) = xyzminmax_el(4);
         xyz(2) = xyzminmax_el(5);
         meshbb.AddVertex(xyz);

         const int inds[8] = {vidx++, vidx++, vidx++, vidx++,
                              vidx++, vidx++, vidx++, vidx++
                             };
         meshbb.AddHex(inds, (eidx++)+1);
      }
   }
   if (sdim == 2)
   {
      meshbb.FinalizeQuadMesh(1, 1, true);
   }
   else
   {
      meshbb.FinalizeHexMesh(1, 1, true);
   }
   return meshbb;
}

IntegrationRule PermuteIR(const IntegrationRule &irule,
                          const Array<int> ordering)
{
   const int np = irule.GetNPoints();
   MFEM_VERIFY(np == ordering.Size(), "Invalid permutation size");
   IntegrationRule ir(np);
   ir.SetOrder(irule.GetOrder());

   for (int i = 0; i < np; i++)
   {
      IntegrationPoint &ip_new = ir.IntPoint(i);
      const IntegrationPoint &ip_old = irule.IntPoint(ordering[i]);
      ip_new.Set(ip_old.x, ip_old.y, ip_old.z, ip_old.weight);
   }

   return ir;
}

void GetDeterminantJacobianGF(ParMesh *mesh, ParGridFunction *detgf)
{
   int dim = mesh->Dimension();
   FiniteElementSpace *fespace = detgf->FESpace();
   Array<int> dofs;

   for (int e = 0; e < mesh->GetNE(); e++)
   {
      const FiniteElement *fe = fespace->GetFE(e);
      const IntegrationRule ir = fe->GetNodes();
      ElementTransformation *transf = mesh->GetElementTransformation(e);
      DenseMatrix Jac(fe->GetDim());
      const NodalFiniteElement *nfe = dynamic_cast<const NodalFiniteElement*>
                                      (fe);
      const Array<int> &irordering = nfe->GetLexicographicOrdering();
      IntegrationRule ir2 = irordering.Size() ?
                            PermuteIR(ir, irordering) :
                            ir;

      Vector detvals(ir2.GetNPoints());
      Vector loc(dim);
      for (int q = 0; q < ir2.GetNPoints(); q++)
      {
         IntegrationPoint ip = ir2.IntPoint(q);
         transf->SetIntPoint(&ip);
         transf->Transform(ip, loc);
         Jac = transf->Jacobian();
         detvals(q) = Jac.Weight();
      }

      fespace->GetElementDofs(e, dofs);
      if (irordering.Size())
      {
         for (int i = 0; i < dofs.Size(); i++)
         {
            (*detgf)(dofs[i]) = detvals(irordering[i]);
         }
      }
      else
      {
         detgf->SetSubVector(dofs, detvals);
      }
   }
}

void VisualizeBB(Mesh &mesh, char *title, int pos_x, int pos_y)
{
   socketstream sock;
   sock.open("localhost", 19916);
   sock << "mesh\n";
   mesh.Print(sock);
   std::string keystrokes = mesh.SpaceDimension() == 2 ? "keys em" : "keys )";
   sock << "window_title '"<< title << "'\n"
        << "window_geometry "
        << pos_x << " " << pos_y << " " << 400 << " " << 400 << "\n"
        //   << "keys jRmclA//]]]]]]]]" << endl;
        << keystrokes << endl;
}

void VisualizeField(ParMesh &pmesh, ParGridFunction &input,
                    char *title, int pos_x, int pos_y)
{
   socketstream sock;
   if (pmesh.GetMyRank() == 0)
   {
      sock.open("localhost", 19916);
      sock << "solution\n";
   }
   pmesh.PrintAsOne(sock);
   input.SaveAsOne(sock);
   if (pmesh.GetMyRank() == 0)
   {
      sock << "window_title '"<< title << "'\n"
           << "window_geometry "
           << pos_x << " " << pos_y << " " << 400 << " " << 400 << "\n"
           << "keys jRmclApppppppppppp//]]]]]]]]" << endl;
   }
}
