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

#include "pmesh_extras.hpp"
#include "mesh_extras.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{

namespace common
{

double ComputeVolume(const ParMesh &pmesh, const Array<int> &attr_marker,
		     int ir_order)
{
   double loc_vol = ComputeVolume(dynamic_cast<const Mesh&>(pmesh),
				  attr_marker, ir_order);
   double glb_vol = 0.0;
   MPI_Allreduce(&loc_vol, &glb_vol, 1, MPI_DOUBLE, MPI_SUM, pmesh.GetComm());
   return glb_vol;
}

double ComputeVolume(const ParMesh &pmesh, int ir_order)
{
   double loc_vol = ComputeVolume(dynamic_cast<const Mesh&>(pmesh), ir_order);
   double glb_vol = 0.0;
   MPI_Allreduce(&loc_vol, &glb_vol, 1, MPI_DOUBLE, MPI_SUM, pmesh.GetComm());
   return glb_vol;
}

double ComputeSurfaceArea(const ParMesh &pmesh, int ir_order)
{
   double loc_area = ComputeSurfaceArea(dynamic_cast<const Mesh&>(pmesh),
					ir_order);
   double glb_area = 0.0;
   MPI_Allreduce(&loc_area, &glb_area, 1, MPI_DOUBLE, MPI_SUM, pmesh.GetComm());
   return glb_area;
}

double ComputeSurfaceArea(const ParMesh &pmesh,
			  const Array<int> &bdr_attr_marker, int ir_order)
{
   double loc_area = ComputeSurfaceArea(dynamic_cast<const Mesh&>(pmesh),
					bdr_attr_marker, ir_order);
   double glb_area = 0.0;
   MPI_Allreduce(&loc_area, &glb_area, 1, MPI_DOUBLE, MPI_SUM, pmesh.GetComm());
   return glb_area;
}

double ComputeZerothMoment(const ParMesh &pmesh, Coefficient &rho,
                           int ir_order)
{
   double loc_mom = ComputeZerothMoment(dynamic_cast<const Mesh&>(pmesh),
                                        rho, ir_order);
   double glb_mom = 0.0;
   MPI_Allreduce(&loc_mom, &glb_mom, 1, MPI_DOUBLE, MPI_SUM, pmesh.GetComm());
   return glb_mom;
}

double ComputeFirstMoment(const ParMesh &pmesh, Coefficient &rho,
                          int ir_order, Vector &mom)
{
   int sdim = pmesh.SpaceDimension();
   mom.SetSize(sdim);
   double loc_mom_data[3];
   Vector loc_mom(loc_mom_data, sdim);
   double loc_mom0 = ComputeFirstMoment(dynamic_cast<const Mesh&>(pmesh),
                                        rho, ir_order, loc_mom);
   double glb_mom0 = 0.0;
   MPI_Allreduce(&loc_mom0, &glb_mom0, 1, MPI_DOUBLE, MPI_SUM, pmesh.GetComm());
   MPI_Allreduce(loc_mom_data, mom.GetData(), sdim, MPI_DOUBLE, MPI_SUM,
                 pmesh.GetComm());
   return glb_mom0;
}

double ComputeSecondMoment(const ParMesh &pmesh, Coefficient &rho,
                           const Vector &center,
                           int ir_order, DenseMatrix &mom)
{
   int sdim = pmesh.SpaceDimension();
   mom.SetSize(sdim);
   double loc_mom_data[9];
   DenseMatrix loc_mom(loc_mom_data, sdim, sdim);
   double loc_mom0 = ComputeSecondMoment(dynamic_cast<const Mesh&>(pmesh),
                                         rho, center, ir_order, loc_mom);
   double glb_mom0 = 0.0;
   MPI_Allreduce(&loc_mom0, &glb_mom0, 1, MPI_DOUBLE, MPI_SUM, pmesh.GetComm());
   MPI_Allreduce(loc_mom_data, mom.GetData(), sdim * sdim, MPI_DOUBLE, MPI_SUM,
                 pmesh.GetComm());
   return glb_mom0;
}

} // namespace common

} // namespace mfem

#endif
