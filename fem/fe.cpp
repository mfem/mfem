// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "fe.hpp"

namespace mfem
{

// Global object definitions

// Static member data declared in fe_base.hpp
// Defined here to ensure it is constructed before WedgeFE
Array2D<int> Poly_1D::binom;
Poly_1D poly1d;

// Object declared in mesh/point.hpp.
// Defined here to ensure it is constructed before 'Geometries'.
PointFiniteElement PointFE;

// Object declared in mesh/segment.hpp.
// Defined here to ensure it is constructed before 'Geometries'.
Linear1DFiniteElement SegmentFE;

// Object declared in mesh/triangle.hpp.
// Defined here to ensure it is constructed before 'Geometries'.
Linear2DFiniteElement TriangleFE;

// Object declared in mesh/quadrilateral.hpp.
// Defined here to ensure it is constructed before 'Geometries'.
BiLinear2DFiniteElement QuadrilateralFE;

// Object declared in mesh/tetrahedron.hpp.
// Defined here to ensure it is constructed before 'Geometries'.
Linear3DFiniteElement TetrahedronFE;

// Object declared in mesh/hexahedron.hpp.
// Defined here to ensure it is constructed before 'Geometries'.
TriLinear3DFiniteElement HexahedronFE;

// Object declared in mesh/wedge.hpp.
// Defined here to ensure it is constructed before 'Geometries'.
LinearWedgeFiniteElement WedgeFE;

// Object declared in mesh/pyramid.hpp.
// Defined here to ensure it is constructed before 'Geometries'.
LinearPyramidFiniteElement PyramidFE;

// Object declared in geom.hpp.
Geometry Geometries;

}
