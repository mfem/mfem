// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Finite Element classes

#include "fe.hpp"
#include "fe_coll.hpp"
#include "bilininteg.hpp"
#include <cmath>

namespace mfem
{

using namespace std;


// Global object definitions

// Object declared in mesh/triangle.hpp.
// Defined here to ensure it is constructed before 'Geometries'.
Linear2DFiniteElement TriangleFE;

// Object declared in mesh/tetrahedron.hpp.
// Defined here to ensure it is constructed before 'Geometries'.
Linear3DFiniteElement TetrahedronFE;

// Object declared in mesh/wedge.hpp.
// Defined here to ensure it is constructed after 'poly1d' and before
// 'Geometries'.
// TODO: define as thread_local to prevent race conditions in GLVis, because
// there is no "LinearWedgeFiniteElement" and WedgeFE is in turn used from two
// different threads for different things in GLVis. We also don't want to turn
// MFEM_THREAD_SAFE on globally. (See PR #731)
H1_WedgeElement WedgeFE(1);

// Object declared in geom.hpp.
// Construct 'Geometries' after 'TriangleFE', 'TetrahedronFE', and 'WedgeFE'.
Geometry Geometries;

}
