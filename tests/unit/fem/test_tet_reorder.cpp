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

#include "mfem.hpp"
#include "catch.hpp"

using namespace mfem;

namespace tet_reorder
{

TEST_CASE("Tetrahedron Reordering")
{

  Mesh mesh(3, 4, 1);

  double c[3];
  c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
  mesh.AddVertex(c);
  c[0] = 1.0; c[1] = 0.0; c[2] = 0.0;
  mesh.AddVertex(c);
  c[0] = 0.0; c[1] = 2.0; c[2] = 0.0;
  mesh.AddVertex(c);
  c[0] = 0.0; c[1] = 0.0; c[2] = 3.0;
  mesh.AddVertex(c);

  int v[4];
  v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3;
  mesh.AddTet(v);
  mesh.FinalizeMesh(0, false);

  mesh.SetCurvature(5);
  std::cout << "Element Volume: " << mesh.GetElementVolume(0) << std::endl;

  mesh.Finalize(true, true);
  std::cout << "Element Volume: " << mesh.GetElementVolume(0) << std::endl;
  mesh.Finalize(true, true);
  std::cout << "Element Volume: " << mesh.GetElementVolume(0) << std::endl;
  //mesh.SetCurvature(5);
  //std::cout << "Element Volume: " << mesh.GetElementVolume(0) << std::endl;
  /*
  int * vi = mesh.GetElement(0)->GetVertices();
  std::swap(vi[2],vi[3]);
  std::cout << "Element Volume: " << mesh.GetElementVolume(0) << std::endl;
  */
}
  
} // namespace tet_reorder
