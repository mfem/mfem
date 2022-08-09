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

#ifndef MFEM_MESH_EXTRAS
#define MFEM_MESH_EXTRAS

#include "mfem.hpp"
#include <sstream>

namespace mfem
{

namespace common
{

class ElementMeshStream : public std::stringstream
{
public:
   ElementMeshStream(Element::Type e);
};

double ComputeVolume(const Mesh &mesh, int ir_order);
double ComputeVolume(const Mesh &mesh, const Array<int> &attr_marker,
		     int ir_order);
double ComputeSurfaceArea(const Mesh &mesh, int ir_order);
double ComputeSurfaceArea(const Mesh &mesh, const Array<int> &bdr_attr_marker,
			  int ir_order);

double ComputeZerothMoment(const Mesh &mesh, Coefficient &rho,
                           int ir_order);
double ComputeFirstMoment(const Mesh &mesh, Coefficient &rho,
                          int ir_order, Vector &mom);
double ComputeSecondMoment(const Mesh &mesh, Coefficient &rho,
                           const Vector &center,
                           int ir_order, DenseMatrix &mom);

inline void ComputeNormalizedFirstMoment(const Mesh &mesh, Coefficient &rho,
                                         int ir_order, Vector &mom)
{
   double mom0 = ComputeFirstMoment(mesh, rho, ir_order, mom);
   mom /= mom0;
}

inline void ComputeNormalizedSecondMoment(const Mesh &mesh, Coefficient &rho,
                                          const Vector &center,
                                          int ir_order, DenseMatrix &mom)
{
   double mom0 = ComputeSecondMoment(mesh, rho, center, ir_order, mom);
   mom *= 1.0 / mom0;
}

void ComputeElementZerothMoments(const Mesh &mesh, Coefficient &rho,
                                 int ir_order, GridFunction &m);

void ComputeElementCentersOfMass(const Mesh &mesh, Coefficient &rho,
                                 int ir_order, GridFunction &c);

/// Merges vertices which lie at the same location
void MergeMeshNodes(Mesh * mesh, int logging);

/// Convert a set of attribute numbers to a marker array
/** The marker array will be of size max_attr and it will contain only zeroes
    and ones. Ones indicate which attribute numbers are present in the attrs
    array. In the special case when attrs has a single entry equal to -1 the
    marker array will contain all ones. */
void AttrToMarker(int max_attr, const Array<int> &attrs, Array<int> &marker);

} // namespace common

} // namespace mfem

#endif
