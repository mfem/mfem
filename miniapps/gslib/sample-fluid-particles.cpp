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

#include "mfem.hpp"
#include "../common/particles_extras.hpp"

using namespace std;
using namespace mfem;
using namespace mfem::common;


int main (int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int size = Mpi::WorldSize();
   int rank = Mpi::WorldRank();
   Hypre::Init();
   
   // Initialize domain with just vertices
   int NP = 10;
   int SD = 3;
   Mesh m(1, NP, 0, 0, SD);
   Vector coords(10*SD);
   coords.Randomize(17);

   for (int p = 0; p < NP; p++)
   {
      Vector p_coord(coords, p*SD, SD);
      m.AddVertex(p_coord);
   }
   m.Finalize();

   m.Print();

   m.EnsureNodes();
   L2_FECollection l2fec(1, 1);
   FiniteElementSpace fespace(&m, &l2fec);
   
   
   m.SetNodalFESpace(&fespace);
   cout << "Printing:\n";
   m.GetNodes()->Print(mfem::out, SD);

   return 0;
}