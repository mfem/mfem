// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mesh_extras.hpp"

using namespace std;

namespace mfem
{

namespace common
{

ElementMeshStream::ElementMeshStream(Element::Type e)
{
   *this << "MFEM mesh v1.0" << endl;
   switch (e)
   {
      case Element::SEGMENT:
         *this << "dimension" << endl << 1 << endl
               << "elements" << endl << 1 << endl
               << "1 1 0 1" << endl
               << "boundary" << endl << 2 << endl
               << "1 0 0" << endl
               << "1 0 1" << endl
               << "vertices" << endl
               << 2 << endl
               << 1 << endl
               << 0 << endl
               << 1 << endl;
         break;
      case Element::TRIANGLE:
         *this << "dimension" << endl << 2 << endl
               << "elements" << endl << 1 << endl
               << "1 2 0 1 2" << endl
               << "boundary" << endl << 3 << endl
               << "1 1 0 1" << endl
               << "1 1 1 2" << endl
               << "1 1 2 0" << endl
               << "vertices" << endl
               << "3" << endl
               << "2" << endl
               << "0 0" << endl
               << "1 0" << endl
               << "0 1" << endl;
         break;
      case Element::QUADRILATERAL:
         *this << "dimension" << endl << 2 << endl
               << "elements" << endl << 1 << endl
               << "1 3 0 1 2 3" << endl
               << "boundary" << endl << 4 << endl
               << "1 1 0 1" << endl
               << "1 1 1 2" << endl
               << "1 1 2 3" << endl
               << "1 1 3 0" << endl
               << "vertices" << endl
               << "4" << endl
               << "2" << endl
               << "0 0" << endl
               << "1 0" << endl
               << "1 1" << endl
               << "0 1" << endl;
         break;
      case Element::TETRAHEDRON:
         *this << "dimension" << endl << 3 << endl
               << "elements" << endl << 1 << endl
               << "1 4 0 1 2 3" << endl
               << "boundary" << endl << 4 << endl
               << "1 2 0 2 1" << endl
               << "1 2 1 2 3" << endl
               << "1 2 2 0 3" << endl
               << "1 2 0 1 3" << endl
               << "vertices" << endl
               << "4" << endl
               << "3" << endl
               << "0 0 0" << endl
               << "1 0 0" << endl
               << "0 1 0" << endl
               << "0 0 1" << endl;
         break;
      case Element::HEXAHEDRON:
         *this << "dimension" << endl << 3 << endl
               << "elements" << endl << 1 << endl
               << "1 5 0 1 2 3 4 5 6 7" << endl
               << "boundary" << endl << 6 << endl
               << "1 3 0 3 2 1" << endl
               << "1 3 4 5 6 7" << endl
               << "1 3 0 1 5 4" << endl
               << "1 3 1 2 6 5" << endl
               << "1 3 2 3 7 6" << endl
               << "1 3 3 0 4 7" << endl
               << "vertices" << endl
               << "8" << endl
               << "3" << endl
               << "0 0 0" << endl
               << "1 0 0" << endl
               << "1 1 0" << endl
               << "0 1 0" << endl
               << "0 0 1" << endl
               << "1 0 1" << endl
               << "1 1 1" << endl
               << "0 1 1" << endl;
         break;
      case Element::WEDGE:
         *this << "dimension" << endl << 3 << endl
               << "elements" << endl << 1 << endl
               << "1 6 0 1 2 3 4 5" << endl
               << "boundary" << endl << 5 << endl
               << "1 2 2 1 0" << endl
               << "1 2 3 4 5" << endl
               << "1 3 0 1 4 3" << endl
               << "1 3 1 2 5 4" << endl
               << "1 3 2 0 3 5" << endl
               << "vertices" << endl
               << "6" << endl
               << "3" << endl
               << "0 0 0" << endl
               << "1 0 0" << endl
               << "0 1 0" << endl
               << "0 0 1" << endl
               << "1 0 1" << endl
               << "0 1 1" << endl;
         break;
      case Element::PYRAMID:
         *this << "dimension" << endl << 3 << endl
               << "elements" << endl << 1 << endl
               << "1 7 0 1 2 3 4" << endl
               << "boundary" << endl << 5 << endl
               << "1 3 3 2 1 0" << endl
               << "1 2 0 1 4" << endl
               << "1 2 1 2 4" << endl
               << "1 2 3 4 2" << endl
               << "1 2 0 4 3" << endl
               << "vertices" << endl
               << "5" << endl
               << "3" << endl
               << "0 0 0" << endl
               << "1 0 0" << endl
               << "1 1 0" << endl
               << "0 1 0" << endl
               << "0 0 1" << endl;
         break;
      default:
         mfem_error("Invalid element type!");
         break;
   }

}

void
MergeMeshNodes(Mesh * mesh, int logging)
{
   int dim  = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

   real_t h_min, h_max, k_min, k_max;
   mesh->GetCharacteristics(h_min, h_max, k_min, k_max);

   // Set tolerance for merging vertices
   real_t tol = 1.0e-8 * h_min;

   if ( logging > 0 )
      cout << "Euler Number of Initial Mesh:  "
           << ((dim==3)?mesh->EulerNumber() :
               ((dim==2)?mesh->EulerNumber2D() :
                mesh->GetNV() - mesh->GetNE())) << endl;

   vector<int> v2v(mesh->GetNV());

   Vector vd(sdim);

   for (int i = 0; i < mesh->GetNV(); i++)
   {
      Vector vi(mesh->GetVertex(i), sdim);

      v2v[i] = -1;

      for (int j = 0; j < i; j++)
      {
         Vector vj(mesh->GetVertex(j), sdim);
         add(vi, -1.0, vj, vd);

         if ( vd.Norml2() < tol )
         {
            v2v[i] = j;
            break;
         }
      }
      if ( v2v[i] < 0 ) { v2v[i] = i; }
   }

   // renumber elements
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      Element *el = mesh->GetElement(i);
      int *v = el->GetVertices();
      int nv = el->GetNVertices();
      for (int j = 0; j < nv; j++)
      {
         v[j] = v2v[v[j]];
      }
   }
   // renumber boundary elements
   for (int i = 0; i < mesh->GetNBE(); i++)
   {
      Element *el = mesh->GetBdrElement(i);
      int *v = el->GetVertices();
      int nv = el->GetNVertices();
      for (int j = 0; j < nv; j++)
      {
         v[j] = v2v[v[j]];
      }
   }

   mesh->RemoveUnusedVertices();

   if ( logging > 0 )
   {
      cout << "Euler Number of Final Mesh:    "
           << ((dim==3) ? mesh->EulerNumber() :
               ((dim==2) ? mesh->EulerNumber2D() :
                mesh->GetNV() - mesh->GetNE()))
           << endl;
   }
}

void AttrToMarker(int max_attr, const Array<int> &attrs, Array<int> &marker)
{
   MFEM_ASSERT(attrs.Max() <= max_attr, "Invalid attribute number present.");

   marker.SetSize(max_attr);
   if (attrs.Size() == 1 && attrs[0] == -1)
   {
      marker = 1;
   }
   else
   {
      marker = 0;
      for (int j=0; j<attrs.Size(); j++)
      {
         int attr = attrs[j];
         MFEM_VERIFY(attr > 0, "Attribute number less than one!");
         marker[attr-1] = 1;
      }
   }
}

void AffineTransformation::Eval(Vector &V, ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   V = 0.0;
   T.Transform(ip, x);

   if (A.Height() == vdim)
   {
      A.Mult(x, V);
   }

   if (b.Size() == vdim)
   {
      V.Add(1.0, b);
   }
}

void KershawTransformation::Eval(Vector &V, ElementTransformation &T,
                                 const IntegrationPoint &ip)
{
   V = 0.0;
   Vector pos(dim);
   T.Transform(ip, pos);
   real_t x = pos(0), y = pos(1), z = dim == 3 ? pos(2) : 0;
   real_t X, Y, Z;

   X = x;

   int layer = x*6.0;
   real_t lambda = (x-layer/6.0)*6;

   // The x-range is split in 6 layers going from left-to-left, left-to-right,
   // right-to-left (2 layers), left-to-right and right-to-right yz-faces.
   switch (layer)
   {
      case 0:
         Y = left(epsy, y);
         Z = left(epsz, z);
         break;
      case 1:
      case 4:
         Y = step(left(epsy, y), right(epsy, y), lambda);
         Z = step(left(epsz, z), right(epsz, z), lambda);
         break;
      case 2:
         Y = step(right(epsy, y), left(epsy, y), lambda/2);
         Z = step(right(epsz, z), left(epsz, z), lambda/2);
         break;
      case 3:
         Y = step(right(epsy, y), left(epsy, y), (1+lambda)/2);
         Z = step(right(epsz, z), left(epsz, z), (1+lambda)/2);
         break;
      default:
         Y = right(epsy, y);
         Z = right(epsz, z);
         break;
   }
   V.SetSize(dim);
   V(0) = X;
   V(1) = Y;
   if (dim == 3) { V(2) = Z; }
}

void SpiralTransformation::Eval(Vector &V, ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   Vector pos(dim);
   T.Transform(ip, pos);
   real_t x = pos(0), y = pos(1), z = dim == 3 ? pos(2) : 0;

   real_t theta = 2.0*M_PI*turns*x;
   real_t r_min = (0.5-0.5*width) + (gap+width)*turns*x;
   real_t r_xyz = r_min + (width)*y;

   V.SetSize(dim);
   V(0) = r_xyz*std::cos(theta);
   V(1) = r_xyz*std::sin(theta);
   if (dim == 3) { V(2) = z*width + x*height; }
}

} // namespace common

} // namespace mfem
