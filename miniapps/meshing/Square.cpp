// Copyright (c) 2017, Lawrence Livermore National Security, OALLC. Produced at
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


#include "Square.hpp"
#include <algorithm>
namespace mfem
{
Square::Square(ParFiniteElementSpace &pfes_mesh, ParGridFunction &distance_gf, const ParMesh &pmesh, const ParGridFunction &coord, Array<int> & ess_vdofs): AnalyticalGeometricShape(pfes_mesh, distance_gf, pmesh, coord, ess_vdofs), width(2.0), depth(1.0), center(2), attr_marker(pfes_mesh.GetNDofs())
{
   // create a rectangle with:
   // width = 2, depth = 1 and centered at (1.0,0.5) 
   width = 2.0;
   depth = 1.0;
   center(0) = 1.0;
   center(1) = 0.5;
   Array<int> vdofs;
   attr_marker = 0;
   // store the corner coordinates of the rectangle 
   cornersX.push_back(center(0)-width/2);
   cornersX.push_back(center(0)+width/2);
   cornersY.push_back(center(1)-depth/2);
   cornersY.push_back(center(1)+depth/2);

   // set the ess_vdofs to size 0 (empty)
   ess_vdofs.SetSize(0);

   // populates the attr_marker with marker of each dof   
   for (int i = 0; i < pmesh.GetNBE(); i++)
   {
      const int nd = pfes_mesh.GetBE(i)->GetDof();
      const int attr = pmesh.GetBdrElement(i)->GetAttribute();
      pfes_mesh.GetBdrElementVDofs(i, vdofs);
      for (int j = 0; j < nd; j++)
      {
         int j_x = vdofs[j], j_y = vdofs[nd+j];
         attr_marker[pfes_mesh.VDofToDof(j_x)] = attr;
      }
   }
}

Square::~Square()
{
}
void Square::GetTFromX(Vector &coordsT, const Vector &coordsX, const int& j_x, const Vector &dist)
{
   int attr = attr_marker[pfes_mesh.VDofToDof(j_x)];
   // get the parametrized coordinates given the:
   // physical coordinates and attribute
   if (attr == 1)
   {
      coordsT[0] = (coordsX[0] + dist(0) - (center(0) - width/2) ) / width;
      coordsT[1] = center(1) + depth/2;
   }
   else if (attr == 2)
   {
      coordsT[0] = center(0) + width/2;
      coordsT[1] = (coordsX[1] + dist(1) - (center(1) - depth/2) ) / depth;
   }
   else if (attr == 3)
   {
      coordsT[0] = (coordsX[0] + dist(0) - (center(0) - width/2) ) / width;
      coordsT[1] = center(1) - depth/2;
   }
   else if (attr == 4)
   {
      coordsT[0] = center(0) - width/2;
      coordsT[1] = (coordsX[1] + dist(1) - (center(1) - depth/2) ) / depth;
   }
}

void Square::GetXFromT(Vector &coordsX, const Vector &coordsT, const int &j_x, const Vector &dist)
{
   int attr = attr_marker[pfes_mesh.VDofToDof(j_x)];
   // get the physical coordinates given the:
   // parametric coordinates and attribute  
   if (attr == 1)
   {
      coordsX[0] = width * coordsT[0] + (center(0) - width/2) - dist(0);
      coordsX[1] = coordsT[1] - dist(1);
   }
   else if (attr == 2)
   {
      coordsX[0] = coordsT[0] - dist(0);
      coordsX[1] = depth * coordsT[1] + (center(1) - depth/2) - dist(1);
   }
   else if (attr == 3)
   {
      coordsX[0] = width * coordsT[0] + (center(0) - width/2) - dist(0);
      coordsX[1] = coordsT[1] - dist(1);
   }
   else if (attr == 4)
   {
      coordsX[0] = coordsT[0] - dist(0);
      coordsX[1] = depth * coordsT[1] + (center(1) - depth/2) - dist(1);
   }
}
void Square::ComputeDistances(const ParGridFunction & coord, const ParMesh & pmesh, const ParFiniteElementSpace &pfes_mesh)
{
   Array<int> vdofs;
   const int dim = pmesh.Dimension();
   // Compute the distance vector for every dof given the:
   // attribute marker and physical coordinates
   for (int e = 0; e < pmesh.GetNBE(); e++)
   {
      const int nd = pfes_mesh.GetBE(e)->GetDof();
      pfes_mesh.GetBdrElementVDofs(e, vdofs);
      for (int i = 0; i < nd; i++)
      {
         int j_x = vdofs[i], j_y = vdofs[nd+i];
         int attr = attr_marker[pfes_mesh.VDofToDof(j_x)];
         if (attr == 1)
         {
            distance_gf(j_x) = 0;
            distance_gf(j_y) = depth/2 + center(1) -  coord(j_y);
         }
         else if (attr == 2)
         {
            int j_x = vdofs[i], j_y = vdofs[nd+i];
            distance_gf(j_y) = 0;
            distance_gf(j_x) = width/2 + center(0) -  coord(j_x);
         }
         else if (attr == 3)
         {
            int j_x = vdofs[i], j_y = vdofs[nd+i];
            distance_gf(j_x) = 0;
            distance_gf(j_y) = center(1) - depth/2 -  coord(j_y);
         }
        else if (attr == 4)
         {
            int j_x = vdofs[i], j_y = vdofs[nd+i];
            distance_gf(j_y) = 0;
            distance_gf(j_x) = center(0) - width/2 -  coord(j_x);
         }
      }
   }
}

void Square::SetScaleMatrix(const Vector &elfun, const Array<int> & vdofs, int i, int j, DenseMatrix &Pmat_scale)
{
   int attr = attr_marker[vdofs[i]];
   int j_x = vdofs[i];
   int j_y = vdofs[i+vdofs.Size()/2];
   // Set the scale matrix, M = dx(t)_{i,j}/dt, for a given i and j,
   // M multiplies the RHS
   // M is stored as (i,j)
   if ( (std::find(cornersX.begin(), cornersX.end(), coord(j_x)) == cornersX.end()) || (std::find(cornersY.begin(), cornersY.end(), coord(j_y)) == cornersY.end() ) )
   {
      if (attr == 1)
      {
         if (j == 0)
         {
            Pmat_scale(i,j) = width;
        }
      }
      else if (attr == 2)
      {
         if (j == 1)
         {
            Pmat_scale(i,j) = depth;
         }
      }
      else if (attr == 3)
      {
         if (j == 0)
         {
            Pmat_scale(i,j) = width;
         }
      }
      else if (attr == 4)
      {
         if (j == 1)
         {
            Pmat_scale(i,j) = depth;
         }
      }
      else
      {
         Pmat_scale(i,j) = 1.0;
      }
   }
}

void Square::SetScaleMatrixFourthOrder(const Vector &elfun, const Array<int> & vdofs, DenseMatrix &Pmat_scale)
{
   // Set the scale matrix, M = dx(t)_{i,j}/dt_{a,b}
   // that multiplies the first piece of the Hessian of the metric's strain energy density function
   // M is stored as (i + j * vdofs.Size()/2, i + j * vdofs.Size()/2)
   for (int i = 0; i < vdofs.Size()/2; i++)
   {
      int attr = attr_marker[vdofs[i]];
      int j_x = vdofs[i];
      int j_y = vdofs[i+vdofs.Size()/2];
      if ( (std::find(cornersX.begin(), cornersX.end(), coord(j_x)) == cornersX.end()) || (std::find(cornersY.begin(), cornersY.end(), coord(j_y)) == cornersY.end() ) )
      {
         for (int j = 0; j < pmesh.Dimension(); j++)
         {
            if (attr == 1)
            {
               if (j == 0)
               {
                  Pmat_scale(i + j * vdofs.Size()/2, i + j * vdofs.Size()/2) = width;
               }
            }
            else if (attr == 2)
            {
               if (j == 1)
               {
                  Pmat_scale(i + j * vdofs.Size()/2, i + j * vdofs.Size()/2) = depth;
               }
            }
            else if (attr == 3)
            {
               if (j == 0)
               {
                  Pmat_scale(i + j * vdofs.Size()/2, i + j * vdofs.Size()/2) = width;
               }
            }
            else if (attr == 4)
            {
               if (j == 1)
               {
               Pmat_scale(i + j * vdofs.Size()/2, i + j * vdofs.Size()/2) = depth;
               }
            }
            else
            {
               Pmat_scale(i + j * vdofs.Size()/2, i + j * vdofs.Size()/2) = 1.0;
            }
         }
      }
   }
}

void Square::SetHessianScaleMatrix(const Vector &elfun, const Array<int> & vdofs, int i, int idim, int j, int jdim, DenseMatrix &Pmat_hessian)
{
   // Set the scaled matrix, (\partial K / \partial t) = (\partial M / \partial t)
   // that multiplies second piece of the Hessian of the metric's strain energy density function 
   Pmat_hessian = 0.0;
}

void Square::convertToPhysical(const Array<int> & vdofs,const Vector &elfun, Vector &convertedX)
{
   for (int i = 0; i < vdofs.Size()/2; i++)
   {
      int attr = attr_marker[pfes_mesh.VDofToDof(vdofs[i])];
      if (attr == 1)
      {
         convertedX[i] = width * elfun[i] + (center(0) - width/2);
         convertedX[i+vdofs.Size()/2] = elfun[i+vdofs.Size()/2] - distance_gf(vdofs[i+vdofs.Size()/2]);
      }
      else if (attr == 2)
      {
         convertedX[i] = elfun[i] - distance_gf(vdofs[i]);
         convertedX[i+vdofs.Size()/2] = depth * elfun[i+vdofs.Size()/2] + (center(1) - depth/2);
      }
      else if (attr == 3)
      {
         convertedX[i] = width * elfun[i] + (center(0) - width/2);
         convertedX[i+vdofs.Size()/2] = elfun[i+vdofs.Size()/2] - distance_gf(vdofs[i+vdofs.Size()/2]);
      }
      else if (attr == 4)
      {
         convertedX[i] = elfun[i] - distance_gf(vdofs[i]);
         convertedX[i+vdofs.Size()/2] = depth * elfun[i+vdofs.Size()/2] + (center(1) - depth/2);
      }
    }
  }
}
