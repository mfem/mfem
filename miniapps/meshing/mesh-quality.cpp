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
//                          Mesh Quality Miniapp
//            --------------------------------------------------
//
// This miniapp extracts geometrics parameters from the Jacobian transformation
// at each degree-of-freedom of the mesh, and visualizes it. The geometric
// parameters size, skewness, and aspect-ratio can help assess the
// quality of a mesh.
//
// Compile with: make mesh-quality
// Sample runs:
// Visualize using GLVis and VisIt the geometric parameters for blade.mesh
// mpirun -np 4 mesh-quality -m blade.mesh -size -aspr -skew -vis -visit
// mpirun -np 4 mesh-quality -m ../../data/square-disc.mesh  -o 2 -size -aspr -skew -vis

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "../common/mfem-common.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   Hypre::Init();
   // 1. Parse command-line options.
   const char *mesh_file = "../data/inline-quad.mesh";
   int order          = -1;
   int ref_levels     = 0;
   bool visualization = true;
   bool visit         = false;
   bool size          = true;
   bool aspect_ratio  = true;
   bool skewness      = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order for visualization"
                  "(defaults to order of the mesh).");
   args.AddOption(&ref_levels, "-r", "--ref-levels",
                  "Number of initial uniform refinement levels.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit",
                  "--no-visit",
                  "Enable or disable VisIt.");
   args.AddOption(&size, "-size", "--size", "-no-size",
                  "--no-size",
                  "Visualize size parameter.");
   args.AddOption(&aspect_ratio, "-aspr", "--aspect-ratio", "-no-aspr",
                  "--no-aspect-ratio",
                  "Visualize aspect-ratio parameter.");
   args.AddOption(&skewness, "-skew", "--skew", "-no-skew",
                  "--no-skew",
                  "Visualize skewness parameter.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }
   const int dim = mesh->Dimension();

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   const int nParams = dim == 2 ? 4 : 12;
   if (order < 0)
   {
      order = pmesh->GetNodes() == NULL ? 1 :
              pmesh->GetNodes()->FESpace()->GetFE(0)->GetOrder();
   }

   // Define a GridFunction for geometric parameters associated with the mesh.
   L2_FECollection l2fec(order, pmesh->Dimension());
   ParFiniteElementSpace fespace(pmesh, &l2fec, nParams);
   ParGridFunction quality(&fespace);

   DenseMatrix jacobian(dim);
   Vector geomParams;
   Array<int> vdofs;
   Vector vals;
   // Compute the geometric parameter at the dofs of each element.
   for (int e = 0; e < pmesh->GetNE(); e++)
   {
      const FiniteElement *fe = fespace.GetFE(e);
      const IntegrationRule &ir = fe->GetNodes();
      fespace.GetElementVDofs(e, vdofs);
      vals.SetSize(vdofs.Size());
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         pmesh->GetElementJacobian(e, jacobian, &ip);
         pmesh->GetGeometricParametersFromJacobian(jacobian, geomParams);
         for (int n = 0; n < nParams; n++)
         {
            vals(q + n*ir.GetNPoints()) = geomParams(n);
         }
      }
      quality.SetSubVector(vdofs, vals);
   }

   VisItDataCollection visit_dc("quality", pmesh);

   // Reorder data by nodes for convenience.
   ParGridFunction qualityCopy = quality;
   if (fespace.GetOrdering() == Ordering::byVDIM)
   {
      qualityCopy.ReorderByNodes();
   }

   // Visualize different parameters
   int visw = 500;
   int vish = 500;
   int cx = 0;
   int cy = 0;
   int gap = 20;
   ParFiniteElementSpace scfespace(pmesh, &l2fec);
   int ndofs = scfespace.GetNDofs();
   if (dim == 2)
   {
      int idx = 0;
      ParGridFunction size_gf, aspr_gf, skew_gf;
      socketstream vis1, vis2, vis3;
      if (size)
      {
         size_gf.MakeRef(&scfespace, qualityCopy.GetData() + (idx++)*ndofs);
         if (visit) { visit_dc.RegisterField("Size", &size_gf); }
         if (visualization)
         {
            common::VisualizeField(vis1, "localhost", 19916, size_gf,
                                   "Size", cx, cy, visw, vish, "Rjmc");
         }
         double min_size = size_gf.Min(),
                max_size = size_gf.Max();
         MPI_Allreduce(MPI_IN_PLACE, &min_size, 1,
                       MPI_DOUBLE, MPI_MIN, pmesh->GetComm());
         MPI_Allreduce(MPI_IN_PLACE, &max_size, 1,
                       MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
         if (myid == 0)
         {
            cout << "Min size:           " << min_size << endl;
            cout << "Max size:           " << max_size << endl;
         }

      }
      if (aspect_ratio)
      {
         aspr_gf.MakeRef(&scfespace, qualityCopy.GetData() + (idx++)*ndofs);
         if (visit) { visit_dc.RegisterField("Aspect-Ratio", &aspr_gf); }
         if (visualization)
         {
            cx += gap+visw;
            common::VisualizeField(vis2, "localhost", 19916, aspr_gf,
                                   "Aspect-Ratio", cx, cy, visw, vish, "Rjmc");
         }
         double min_aspr = aspr_gf.Min(),
                max_aspr = aspr_gf.Max();
         MPI_Allreduce(MPI_IN_PLACE, &min_aspr, 1,
                       MPI_DOUBLE, MPI_MIN, pmesh->GetComm());
         MPI_Allreduce(MPI_IN_PLACE, &max_aspr, 1,
                       MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
         max_aspr = max(1.0/min_aspr, max_aspr);
         if (myid == 0)
         {
            cout << "Worst aspect-ratio: " << max_aspr << endl;
            cout << "(in any direction)" << endl;
         }
      }
      if (skewness)
      {
         skew_gf.MakeRef(&scfespace, qualityCopy.GetData() + (idx++)*ndofs);
         if (visit) { visit_dc.RegisterField("Skewness", &skew_gf); }
         if (visualization)
         {
            cx += gap+visw;
            common::VisualizeField(vis3, "localhost", 19916, skew_gf,
                                   "Skewness (radians)", cx, cy, visw, vish, "Rjmc");
         }
         double min_skew = skew_gf.Min(),
                max_skew = skew_gf.Max();
         MPI_Allreduce(MPI_IN_PLACE, &min_skew, 1,
                       MPI_DOUBLE, MPI_MIN, pmesh->GetComm());
         MPI_Allreduce(MPI_IN_PLACE, &max_skew, 1,
                       MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
         if (myid == 0)
         {
            cout << "Min skew (in deg):  " << min_skew*180/M_PI << endl;
            cout << "Max skew (in deg):  " << max_skew*180/M_PI << endl;
         }
      }
      if (visit)
      {
         visit_dc.SetFormat(DataCollection::SERIAL_FORMAT);
         visit_dc.Save();
      }
   }
   else if (dim == 3)
   {
      int idx = 0;
      ParGridFunction size_gf, aspr_gf1, aspr_gf2,
                      skew_gf1, skew_gf2, skew_gf3;
      socketstream vis1, vis2, vis3, vis4, vis5, vis6;
      if (size)
      {
         size_gf.MakeRef(&scfespace, qualityCopy.GetData() + (idx++)*ndofs);
         if (visit) { visit_dc.RegisterField("Size", &size_gf); }
         if (visualization)
         {
            common::VisualizeField(vis1, "localhost", 19916, size_gf,
                                   "Size", cx, cy, visw, vish, "Rjmc");
         }
         double min_size = size_gf.Min(),
                max_size = size_gf.Max();
         MPI_Allreduce(MPI_IN_PLACE, &min_size, 1,
                       MPI_DOUBLE, MPI_MIN, pmesh->GetComm());
         MPI_Allreduce(MPI_IN_PLACE, &max_size, 1,
                       MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
         if (myid == 0)
         {
            cout << "Min size:            " << min_size << endl;
            cout << "Max size:            " << max_size << endl;
         }
      }
      if (aspect_ratio)
      {
         aspr_gf1.MakeRef(&scfespace, qualityCopy.GetData() + (idx++)*ndofs);
         aspr_gf2.MakeRef(&scfespace, qualityCopy.GetData() + (idx++)*ndofs);
         if (visit)
         {
            visit_dc.RegisterField("Aspect-Ratio", &aspr_gf1);
            visit_dc.RegisterField("Aspect-Ratio2", &aspr_gf2);
         }
         if (visualization)
         {
            cx += gap+visw;
            common::VisualizeField(vis2, "localhost", 19916, aspr_gf1,
                                   "Aspect-Ratio", cx, cy, visw, vish, "Rjmc");
            cx += gap+visw;
            common::VisualizeField(vis3, "localhost", 19916, aspr_gf2,
                                   "Aspect-Ratio2", cx, cy, visw, vish, "Rjmc");
         }
         double min_aspr1 = aspr_gf1.Min(),
                max_aspr1 = aspr_gf1.Max();
         MPI_Allreduce(MPI_IN_PLACE, &min_aspr1, 1,
                       MPI_DOUBLE, MPI_MIN, pmesh->GetComm());
         MPI_Allreduce(MPI_IN_PLACE, &max_aspr1, 1,
                       MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
         max_aspr1 = max(1.0/min_aspr1, max_aspr1);

         double min_aspr2 = aspr_gf2.Min(),
                max_aspr2 = aspr_gf2.Max();
         MPI_Allreduce(MPI_IN_PLACE, &min_aspr2, 1,
                       MPI_DOUBLE, MPI_MIN, pmesh->GetComm());
         MPI_Allreduce(MPI_IN_PLACE, &max_aspr2, 1,
                       MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
         max_aspr2 = max(1.0/min_aspr2, max_aspr2);

         double max_aspr = max(max_aspr1, max_aspr2);

         Vector aspr_gf3(aspr_gf1.Size());
         for (int i = 0; i < aspr_gf1.Size(); i++)
         {
            aspr_gf3(i) = 1.0/(aspr_gf1(i)*aspr_gf2(i));
         }
         double min_aspr3 = aspr_gf3.Min(),
                max_aspr3 = aspr_gf3.Max();
         MPI_Allreduce(MPI_IN_PLACE, &min_aspr3, 1,
                       MPI_DOUBLE, MPI_MIN, pmesh->GetComm());
         MPI_Allreduce(MPI_IN_PLACE, &max_aspr3, 1,
                       MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
         max_aspr3 = max(1.0/min_aspr3, max_aspr3);

         max_aspr = max(max_aspr, max_aspr3);

         if (myid == 0)
         {
            cout << "Worst aspect-ratio:  " << max_aspr << endl;
            cout << "(in any direction)" << endl;
         }
      }
      if (skewness)
      {
         skew_gf1.MakeRef(&scfespace, qualityCopy.GetData() + (idx++)*ndofs);
         skew_gf2.MakeRef(&scfespace, qualityCopy.GetData() + (idx++)*ndofs);
         skew_gf3.MakeRef(&scfespace, qualityCopy.GetData() + (idx++)*ndofs);
         if (visit)
         {
            visit_dc.RegisterField("Skewness (radians)", &skew_gf1);
            visit_dc.RegisterField("Skewness2 (radians)", &skew_gf2);
            visit_dc.RegisterField("Dihedral (radians)", &skew_gf3);
         }
         if (visualization)
         {
            cx = 0;
            cy += 10*gap+vish;
            common::VisualizeField(vis4, "localhost", 19916, skew_gf1,
                                   "Skewness", cx, cy, visw, vish, "Rjmc");
            cx += gap+visw;
            common::VisualizeField(vis5, "localhost", 19916, skew_gf2,
                                   "Skewness2", cx, cy, visw, vish, "Rjmc");
            cx += gap+visw;
            common::VisualizeField(vis6, "localhost", 19916, skew_gf3,
                                   "Dihedral", cx, cy, visw, vish, "Rjmc");
         }
         double min_skew1 = skew_gf1.Min(),
                max_skew1 = skew_gf1.Max();
         MPI_Allreduce(MPI_IN_PLACE, &min_skew1, 1,
                       MPI_DOUBLE, MPI_MIN, pmesh->GetComm());
         MPI_Allreduce(MPI_IN_PLACE, &max_skew1, 1,
                       MPI_DOUBLE, MPI_MAX, pmesh->GetComm());

         double min_skew2 = skew_gf2.Min(),
                max_skew2 = skew_gf2.Max();
         MPI_Allreduce(MPI_IN_PLACE, &min_skew2, 1,
                       MPI_DOUBLE, MPI_MIN, pmesh->GetComm());
         MPI_Allreduce(MPI_IN_PLACE, &max_skew2, 1,
                       MPI_DOUBLE, MPI_MAX, pmesh->GetComm());

         double min_skew3 = skew_gf3.Min(),
                max_skew3 = skew_gf3.Max();
         MPI_Allreduce(MPI_IN_PLACE, &min_skew3, 1,
                       MPI_DOUBLE, MPI_MIN, pmesh->GetComm());
         MPI_Allreduce(MPI_IN_PLACE, &max_skew3, 1,
                       MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
         if (myid == 0)
         {
            cout << "Min skew 1 (in deg): " << min_skew1*180/M_PI << endl;
            cout << "Max skew 1 (in deg): " << max_skew1*180/M_PI << endl;

            cout << "Min skew 2 (in deg): " << min_skew2*180/M_PI << endl;
            cout << "Max skew 2 (in deg): " << max_skew2*180/M_PI << endl;

            cout << "Min skew 3 (in deg): " << min_skew3*180/M_PI << endl;
            cout << "Max skew 3 (in deg): " << max_skew3*180/M_PI << endl;
         }

      }
      if (visit)
      {
         visit_dc.SetFormat(DataCollection::SERIAL_FORMAT);
         visit_dc.Save();
      }
   }

   delete pmesh;
   return 0;
}
