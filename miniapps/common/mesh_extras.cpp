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
      default:
         mfem_error("Invalid element type!");
         break;
   }

}

} // namespace common

} // namespace mfem
