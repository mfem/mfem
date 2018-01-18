// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
//
//      ----------------------------------------------------------------
//      Display Basis Miniapp:  Visualize finite element basis functions
//      ----------------------------------------------------------------
//
// This miniapp visualizes various types of finite element basis functions on a
// single mesh element in 1D, 2D and 3D. The order and the type of finite
// element space can be changed, and the mesh element is either the reference
// one, or a simple transformation of it. Dynamic creation and interaction with
// multiple GLVis windows is demonstrated.
//
// Compile with: make display-basis
//
// Sample runs:  display-basis
//               display_basis -e 2 -b 3 -o 3
//               display-basis -e 5 -b 1 -o 1

#include <iostream>

#include "mfem.hpp"

mfem::Vector ReadGF(const char *filename) {
  std::ifstream in(filename);
  std::string line;

  for (int i = 0; i < 5; ++i) {
    std::getline(in, line);
  }

  std::vector<double> v;
  double entry;
  while (in >> entry) {
    v.push_back(entry);
  }

  const int entries = v.size();
  double *data = new double[entries];
  for (int i = 0; i < entries; ++i) {
    data[i] = v[i];
  }

  return mfem::Vector(data, entries);
}

int main(const int argc, const char **argv) {
  if (argc < 2) {
    std::cout << argv[0] << ": Must take 1 or 2 sol.gf file inputs to compare\n";
    // mfem-test doesn't take custom arguments
    // Return 0 to make Travis CI happy
    return 0;
  }

  mfem::Vector diff = ReadGF(argv[1]);
  if (argc > 2) {
    diff -= ReadGF(argv[2]);
  }

  std::cout << "Max       : " << diff.Max() << '\n'
            << "Min       : " << diff.Min() << '\n'
            << "L1 Norm   : " << diff.Norml1() << '\n'
            << "L2 Norm   : " << diff.Norml2() << '\n'
            << "L3 Norm   : " << diff.Normlp(3) << '\n'
            << "LInf Norm : " << diff.Normlinf() << '\n';

  return 0;
}