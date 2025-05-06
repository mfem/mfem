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
//                     -----------------------
//                     DG Agglomeration Solver
//                     -----------------------
//

#include "mfem.hpp"
#include <iostream>
#include <memory>

#include "partition.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *mesh_file = "../../data/star.mesh";
   int ncoarse = 4;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ncoarse, "-n", "--n-coarse", "Number of coarse elements.");
   args.ParseCheck();

   Mesh mesh(mesh_file);

   DG_FECollection fec(0, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction p(&fes);

   ParaViewDataCollection pv("Agglomeration", &mesh);
   pv.SetPrefixPath("ParaView");
   pv.RegisterField("p", &p);

   p = 0;
   pv.SetCycle(0);
   pv.SetTime(0.0);
   pv.Save();

   Array<int> partitioning = PartitionMesh(mesh, 4);
   for (int i = 0; i < p.Size(); ++i)
   {
      p[i] = partitioning[i];
   }

   pv.SetCycle(1);
   pv.SetTime(1.0);
   pv.Save();

   return 0;
}
