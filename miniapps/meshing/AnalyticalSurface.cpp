// Copyright (c) 2017, Lawrence LivermoreA National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include "AnalyticalSurface.hpp"

namespace mfem
{

AnalyticalSurface::AnalyticalSurface(int geometryType,  ParFiniteElementSpace &pfes_mesh, const ParGridFunction & coord, const ParMesh & pmesh, Array<int> & ess_vdofs) : 
   geometryType(geometryType),
   pfes_mesh(pfes_mesh),
   geometry(NULL),
   coord(coord),
   pmesh(pmesh),
   distance_gf(&pfes_mesh)
{
   switch (geometryType)
   {
      // create geometry, currently only for square or rectangular shapes
      case 1: geometry = new Square(pfes_mesh, distance_gf, pmesh, coord, ess_vdofs); break;
      default:
         out << "Unknown geometry type: " << geometryType << '\n';
         break;
   }
   distance_gf = 0.0;
   geometry->ComputeDistances(coord, pmesh, pfes_mesh);
}

void AnalyticalSurface::ConvertPhysicalCoordinatesToParametric(ParGridFunction &coord)
{
   Array<int> vdofs;
   const int dim = pmesh.Dimension();
   std::vector<int> touchedNodes;
   for (int e = 0; e < pmesh.GetNBE(); e++)
   {
      const int nd = pfes_mesh.GetBE(e)->GetDof();
      const int attr = pmesh.GetBdrElement(e)->GetAttribute();
      pfes_mesh.GetBdrElementVDofs(e, vdofs);
      // std::cout << " vdofs " << vdofs.Size() << std::endl;
      for (int i = 0; i < nd; i++)
      {
         // 2D for now
         int j_x = vdofs[i], j_y = vdofs[nd+i];
         int j_z = (dim == 2) ? 0.0 : vdofs[2*nd + i];
         Vector coordsX(dim);
         Vector coordsT(dim);
         coordsX = 0.0;
         coordsT = 0.0;
         Vector dist(dim);
         dist = 0.0;
         dist(0) = distance_gf(j_x);
         dist(1) = distance_gf(j_y);
	 if (dim == 3) { dist(2) = distance_gf(j_z); }
	 if ( std::find(touchedNodes.begin(), touchedNodes.end(), j_x) == touchedNodes.end() )
         {
            coordsX[0] = coord(j_x);
            coordsX[1] = coord(j_y);
	    if (dim == 3) { coordsX[2] = coord(j_z); }
	    geometry->GetTFromX(coordsT, coordsX, j_x, dist);
            coord(j_x) = coordsT[0];
            coord(j_y) = coordsT[1];
	    if (dim == 3) { coord(j_z) = coordsT[2]; }
	    touchedNodes.push_back(j_x);
         }
      }
   }
}

void AnalyticalSurface::ConvertParametricCoordinatesToPhysical(ParGridFunction &coord)
{
   Array<int> vdofs;
   const int dim = pmesh.Dimension();
   std::vector<int> touchedNodes;
   for (int e = 0; e < pmesh.GetNBE(); e++)
   {
      const int nd = pfes_mesh.GetBE(e)->GetDof();
      const int attr = pmesh.GetBdrElement(e)->GetAttribute();
      pfes_mesh.GetBdrElementVDofs(e, vdofs);
      for (int i = 0; i < nd; i++)
      {
         int j_x = vdofs[i], j_y = vdofs[nd+i];
         int j_z = (dim == 2) ? 0.0 : vdofs[2*nd + i];
         Vector coordsX(dim);
         Vector coordsT(dim);
         coordsT = 0.0;
         coordsX = 0.0;
         Vector dist(dim);
         dist = 0.0;
         dist(0) = distance_gf(j_x);
         dist(1) = distance_gf(j_y);
         if (dim == 3) { dist(2) = distance_gf(j_z); }
         if ( std::find(touchedNodes.begin(), touchedNodes.end(), j_x) == touchedNodes.end() )
         {
            coordsT[0] = coord(j_x);
            coordsT[1] = coord(j_y);
	    if (dim == 3) { coordsT[2] = coord(j_z); }
	    geometry->GetXFromT(coordsX, coordsT, j_x, dist);
            coord(j_x) = coordsX[0];
            coord(j_y) = coordsX[1];
            if (dim == 3) { coord(j_z) = coordsX[2]; }
	    touchedNodes.push_back(j_x);
         }
      }
   }
}

void AnalyticalSurface::SetScaleMatrix(const Vector &elfun, const Array<int> & vdofs, int i, int j, DenseMatrix &Pmat_scale)
{
   geometry->SetScaleMatrix(elfun, vdofs, i, j, Pmat_scale);
}

void AnalyticalSurface::SetScaleMatrixFourthOrder(const Vector &elfun, const Array<int> & vdofs, DenseMatrix &Pmat_scale)
{
   geometry->SetScaleMatrixFourthOrder(elfun, vdofs, Pmat_scale);
}

void AnalyticalSurface::SetHessianScaleMatrix(const Vector &elfun, const Array<int> & vdofs, int i, int idim, int j, int jdim, DenseMatrix &Pmat_hessian)
{
   geometry->SetHessianScaleMatrix(elfun, vdofs, i, idim, j, jdim, Pmat_hessian);
}

void AnalyticalSurface::convertToPhysical(const Array<int> & vdofs,const Vector &elfun, Vector &convertedX)
{
   geometry->convertToPhysical(vdofs, elfun, convertedX);
}
AnalyticalSurface::~AnalyticalSurface()
{
   delete geometry;
}

}
