// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
//                          MFEM NURBS knot vector example
//
// Compile with: make nurbs_curveint
//
// Sample runs:  nurbs_curveint
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple KnotVector and print its corresponding shape functions.

#include <iostream>
#include "mfem.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   OptionsParser args(argc, argv);
   bool visualization;

   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization. Dummy option to allow testing.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }

   KnotVector kv(2, 7);

   kv[0] = 0;
   kv[1] = 0;
   kv[2] = 0;
   kv[3] = 0.25;
   kv[4] = 0.5;
   kv[5] = 0.5; // Repeated knot
   kv[6] = 0.75;
   kv[7] = 1;
   kv[8] = 1;
   kv[9] = 1;

   cout << "Printing knotvector:" << endl;
   kv.Print(cout);

   // Count number of elements, required for printing of shapes
   kv.GetElements();

   cout << "\nPrinting shapefunctions:" << endl;
   kv.PrintFunctions(cout);
}
