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

#ifndef MFEM_MESH_EXTRAS
#define MFEM_MESH_EXTRAS

#include "mfem.hpp"
#include <sstream>

namespace mfem
{

namespace miniapps
{

class ElementMeshStream : public std::stringstream
{
public:
   ElementMeshStream(Element::Type e);
};

double ComputeVolume(Mesh &mesh, int ir_order);
double ComputeSurfaceArea(Mesh &mesh, int ir_order);

double ComputeZerothMoment(Mesh &mesh, Coefficient &rho,
                           int ir_order);
double ComputeFirstMoment(Mesh &mesh, Coefficient &rho,
                          int ir_order, Vector &mom);
double ComputeSecondMoment(Mesh &mesh, Coefficient &rho,
                           const Vector &center,
                           int ir_order, DenseMatrix &mom);

void ComputeNormalizedFirstMoment(Mesh &mesh, Coefficient &rho,
                                  int ir_order, Vector &mom)
{
   double mom0 = ComputeFirstMoment(mesh, rho, ir_order, mom);
   mom /= mom0;
}

void ComputeNormalizedSecondMoment(Mesh &mesh, Coefficient &rho,
                                   const Vector &center,
                                   int ir_order, DenseMatrix &mom)
{
   double mom0 = ComputeSecondMoment(mesh, rho, center, ir_order, mom);
   mom *= 1.0 / mom0;
}

void ComputeElementZerothMoments(Mesh &mesh, Coefficient &rho,
                                 int ir_order, GridFunction &m);

void ComputeElementCentersOfMass(Mesh &mesh, Coefficient &rho,
                                 int ir_order, GridFunction &c);

} // namespace miniapps

} // namespace mfem

#endif
