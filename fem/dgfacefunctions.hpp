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

//This file contains useful functions to compute fluxes for DG methods.


#ifndef MFEM_DGFACEFUNC
#define MFEM_DGFACEFUNC

using std::vector;
using std::pair;

namespace mfem
{

/**
*	Returns the canonical coordinate vectors e_1 and e_2.
*/
void getBaseVector2D(Vector& e1, Vector& e2);

/**
*	Returns the canonical coordinate vectors e_1, e_2 and e_3.
*/
void getBaseVector3D(Vector& e1, Vector& e2, Vector& e3);

/** A function that initialize the local coordinate base for a face with
*   indice face_ind.
*   This returns the local face coordinate base expressed in reference
*   element coordinate.
*/
// Highly dependent of the node ordering from geom.cpp
void InitFaceCoord2D(const int face_id, IntMatrix& base);

// Highly dependent of the node ordering from geom.cpp
void InitFaceCoord3D(const int face_id, IntMatrix& base);

/** Maps the coordinate vectors of the first face to the coordinate vectors of the second face.
*   nb_rot is the number of rotation to opperate so that the first node of each face match.
*   The result map contains pairs of int, where the first int is a direction cofficient,
*	 and the second int is the indice of the second face vector.
*/
// There shouldn't be any rotation in 2D.
void GetLocalCoordMap2D(vector<pair<int,int> >& map, const int nb_rot = 0);

// Default parameter nb_rot=0 should be only use with a structured mesh.
// Rotations follow the ordering of the nodes.
void GetLocalCoordMap3D(vector<pair<int,int> >& map, const int nb_rot = 0);

/**
*	Returns the change of matrix P from base_K2 to base_K1 according to the mapping map.
*/
void GetChangeOfBasis(const IntMatrix& base_K1, IntMatrix& base_K2,
								const vector<pair<int,int> >& map, IntMatrix& P);

/**
*	Returns the face_id that identifies the face on the reference element, and nb_rot the
*  "rotations" the face did between reference to physical spaces.
*/
void GetIdRotInfo(const int face_info, int& face_id, int& nb_rot);

}

#endif // MFEM_DGFACEFUNC