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
//            Compute bounds of the given grid-function
//    ---------------------------------------------------------------------
//
// This miniapp computes piecewise linear bounds on a given gridfunction, and
// visualizes the lower and upper bound for each element. The bounding approach
// is based on the method described in:
//
// (1) Section 3 of Mittal et al., "General Field Evaluation in High-Order
//     Meshes on GPUs"
// and
// (2) Dzanic et al., "A method for bounding high-order finite element
//     functions: Applications to mesh validity and bounds-preserving limiters".
//
//
// Compile with: make gridfunction-bounds
//
// Sample runs:
//  mpirun -np 4 gridfunction-bounds
//  mpirun -np 4 gridfunction-bounds -nb 100 -ref 5 -bt 2 -l2

#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>

using namespace mfem;
using namespace std;

void VisualizeField(ParMesh &pmesh, ParGridFunction &input,
                    char *title, int pos_x, int pos_y);

int main (int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();

   // Set the method's default parameters.
   const char *mesh_file = "../gslib/triple-pt-1.mesh";
   const char *sltn_file = "../gslib/triple-pt-1.gf";
   int  ref              = 2;
   bool visualization    = true;
   bool visit            = false;
   int  b_type           = -1;
   bool continuous       = true;
   int  nbrute           = 0;

   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&sltn_file, "-s", "--sltn",
                  "Solution file to use.");
   args.AddOption(&ref, "-ref", "--piecewise-linear-ref-factor",
                  "Scaling factor for resolution of piecewise linear bounds."
                  " If less than 2, the resolution is picked automatically");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit",
                  "--no-visit",
                  "Enable or disable VisIt output.");
   args.AddOption(&b_type, "-bt", "--basis-type",
                  "Project input function to a different bases. "
                  "-1 = don't project (default)."
                  "0 = Gauss-Legendre nodes. "
                  "1 = Gauss-Lobatto nodes. "
                  "2 = uniformly spaced nodes. ");
   args.AddOption(&continuous, "-h1", "--h1", "-l2", "--l2",
                  "Use continuous or discontinuous space.");
   args.AddOption(&nbrute, "-nb", "--nbrute",
                  "Brute force search for minimum in an array of nxnxn points "
                  "in each element.");
   args.ParseCheck();

   Mesh mesh(mesh_file, 1, 1, false);
   const int dim = mesh.Dimension();
   if (continuous && b_type != -1)
   {
      MFEM_VERIFY(b_type > 0, "Continuous space do not support GL nodes. "
                  "Please use basis type: 1 for Lagrange interpolants on GLL "
                  " nodes 2 for positive bases on uniformly spaced nodes.");
   }

   std::unique_ptr<int[]> partition(
      mesh.GeneratePartitioning(Mpi::WorldSize())
   );

   ifstream mat_stream_1(sltn_file);
   std::unique_ptr<GridFunction> func(new GridFunction(&mesh, mat_stream_1));

   ParMesh pmesh(MPI_COMM_WORLD, mesh, partition.get());
   ParGridFunction pfunc(&pmesh, func.get(), partition.get());
   int func_order = func->FESpace()->GetMaxElementOrder();
   int vdim     = pfunc.FESpace()->GetVDim();
   int nel      = pmesh.GetNE();

   func.reset();
   mesh.Clear();
   partition.reset();

   // Project input function based on user input
   ParGridFunction *pfunc_proj = NULL;
   if (b_type >= 0)
   {
      FiniteElementCollection *fec = NULL;
      if (continuous)
      {
         fec = new H1_FECollection(func_order, dim, b_type);
      }
      else
      {
         fec = new L2_FECollection(func_order, dim, b_type);
      }
      int ordering = pfunc.FESpace()->GetOrdering();
      ParFiniteElementSpace *fes = new ParFiniteElementSpace(&pmesh, fec,
                                                             vdim, ordering);
      pfunc_proj = new ParGridFunction(fes);
      pfunc_proj->MakeOwner(fec);
      pfunc_proj->ProjectGridFunction(pfunc);
      if (Mpi::Root())
      {
         cout << "fec name orig: " << pfunc.FESpace()->FEColl()->Name() <<
              endl;
         cout << "fec name: " << fec->Name() << endl;
      }
   }
   else
   {
      pfunc_proj = &pfunc;
      if (Mpi::Root())
      {
         cout << "fec name: " << pfunc.FESpace()->FEColl()->Name() << endl;
      }
   }

   L2_FECollection fec_pc(0, dim);
   ParFiniteElementSpace fes_pc(&pmesh, &fec_pc, vdim, Ordering::byNODES);
   ParGridFunction lowerb(&fes_pc), upperb(&fes_pc);

   // Compute bounds
   pfunc_proj->GetElementBounds(lowerb, upperb, ref);

   Vector bound_min(vdim), bound_max(vdim);
   for (int d = 0; d < vdim; d++)
   {
      Vector lowerT(lowerb.GetData() + d*nel, nel);
      Vector upperT(upperb.GetData() + d*nel, nel);
      bound_min(d) = lowerT.Min();
      bound_max(d) = upperT.Max();
   }

   MPI_Allreduce(MPI_IN_PLACE, bound_min.GetData(), vdim,
                 MPITypeMap<real_t>::mpi_type, MPI_MIN, pmesh.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, bound_max.GetData(), vdim,
                 MPITypeMap<real_t>::mpi_type, MPI_MAX, pmesh.GetComm());

   // GLVis Visualization
   if (visualization)
   {
      char title1[] = "Input gridfunction";
      VisualizeField(pmesh, pfunc, title1, 0, 0);
      if (b_type >= 0)
      {
         char title1p[] = "Projected gridfunction";
         VisualizeField(pmesh, *pfunc_proj, title1p, 0, 400);
      }
      char title2[] = "Element-wise lower bound";
      VisualizeField(pmesh, lowerb, title2, 400, 0);
      char title3[] = "Element-wise upper bound";
      VisualizeField(pmesh, upperb, title3, 800, 0);
   }

   // Visit Visualization
   if (visit)
   {
      VisItDataCollection visit_dc("jacobian-determinant-bounds", &pmesh);
      visit_dc.SetFormat(DataCollection::PARALLEL_FORMAT);
      visit_dc.RegisterField("input-function", &pfunc);
      if (b_type >= 0)
      {
         visit_dc.RegisterField("projected-function", pfunc_proj);
      }
      visit_dc.RegisterField("lower-bound", &lowerb);
      visit_dc.RegisterField("upper-bound", &upperb);
      visit_dc.Save();
   }

   if (nbrute > 0)
   {
      Vector global_min(vdim), global_max(vdim);
      global_min = numeric_limits<real_t>::max();
      global_max = numeric_limits<real_t>::min();
      // search for the minimum value of pfunc_proj in each element at
      // an array of integration points
      for (int e = 0; e < pmesh.GetNE(); e++)
      {
         IntegrationPoint ip;
         for (int k = 0; k < (dim > 2 ? nbrute : 1); k++)
         {
            ip.z = k/(nbrute-1.0);
            for (int j = 0; j < (dim > 1 ? nbrute : 1); j++)
            {
               ip.y = j/(nbrute-1.0);
               for (int i = 0; i < nbrute; i++)
               {
                  ip.x = i/(nbrute-1.0);
                  for (int d = 0; d < vdim; d++)
                  {
                     real_t val = pfunc_proj->GetValue(e, ip, d+1);
                     global_min(d) = min(global_min(d), val);
                     global_max(d) = max(global_max(d), val);
                  }
               }
            }
         }
      }

      MPI_Allreduce(MPI_IN_PLACE, global_min.GetData(), vdim,
                    MPITypeMap<real_t>::mpi_type, MPI_MIN, pmesh.GetComm());
      MPI_Allreduce(MPI_IN_PLACE, global_max.GetData(), vdim,
                    MPITypeMap<real_t>::mpi_type, MPI_MAX, pmesh.GetComm());
      if (Mpi::Root())
      {
         for (int d = 0; d < vdim; d++)
         {
            cout << "Brute force and bounding comparison for component " <<
                 d << endl;
            cout << "Brute force minimum and minimum bound: " << global_min(d)
                 << " " <<  bound_min(d) << endl;

            cout << "Brute force maximum and maximum bound: " << global_max(d)
                 << " " <<  bound_max(d) << endl;

            cout << "The difference in bounds is: " <<
                 global_min(d)-bound_min(d) << " " <<
                 bound_max(d)-global_max(d) << endl;
         }
      }
   }

   if (nbrute == 0 && Mpi::Root())
   {
      for (int d = 0; d < vdim; d++)
      {
         cout << "Minimum bound for component " << d << " is " <<
              bound_min(d) << endl;
         cout << "Maximum bound for component " << d << " is " <<
              bound_max(d) << endl;
      }
   }

   if (b_type >= 0)
   {
      delete pfunc_proj;
   }
   return 0;
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
