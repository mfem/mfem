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

#ifndef MFEM_PMESH_EXTRAS
#define MFEM_PMESH_EXTRAS

#include "mfem.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

namespace common
{

double ComputeVolume(const ParMesh &pmesh, int ir_order);
double ComputeVolume(const ParMesh &mesh, const Array<int> &attr_marker,
		     int ir_order);
double ComputeSurfaceArea(const ParMesh &pmesh, int ir_order);
double ComputeSurfaceArea(const ParMesh &mesh,
			  const Array<int> &bdr_attr_marker,
			  int ir_order);

double ComputeZerothMoment(const ParMesh &pmesh, Coefficient &rho,
                           int ir_order);
double ComputeFirstMoment(const ParMesh &pmesh, Coefficient &rho,
                          int ir_order, Vector &mom);
double ComputeSecondMoment(const ParMesh &pmesh, Coefficient &rho,
                           const Vector &center,
                           int ir_order, DenseMatrix &mom);

inline void ComputeNormalizedFirstMoment(const ParMesh &pmesh,
					 Coefficient &rho,
                                         int ir_order, Vector &mom)
{
   double mom0 = ComputeFirstMoment(pmesh, rho, ir_order, mom);
   mom /= mom0;
}

inline void ComputeNormalizedSecondMoment(const ParMesh &pmesh,
					  Coefficient &rho,
                                          const Vector &center,
                                          int ir_order, DenseMatrix &mom)
{
   double mom0 = ComputeSecondMoment(pmesh, rho, center, ir_order, mom);
   mom *= 1.0 / mom0;
}

} // namespace common

} // namespace mfem

#endif // MFEM_USE_MPI
#endif
