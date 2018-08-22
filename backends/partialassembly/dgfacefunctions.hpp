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


#ifndef MFEM_PA_DGFACEFUNC
#define MFEM_PA_DGFACEFUNC
#include "tensor.hpp"
#include <vector>
#include <utility>
#include "../../linalg/vector.hpp"
#include "../../mesh/mesh.hpp"
#include "tensorialfunctions.hpp"

using std::vector;
using std::pair;

namespace mfem
{

namespace pa
{

/**
*	Returns the canonical coordinate vectors e_1 and e_2.
*/
void getBaseVector2D(mfem::Vector& e1, mfem::Vector& e2);

/**
*	Returns the canonical coordinate vectors e_1, e_2 and e_3.
*/
void getBaseVector3D(mfem::Vector& e1, mfem::Vector& e2, mfem::Vector& e3);

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

// Rotations follow the ordering of the nodes.
void GetLocalCoordMap3D(vector<pair<int,int> >& map, const int nb_rot);

/**
*	Returns the change of matrix P from base_K2 to base_K1 according to the mapping map.
*/
void GetChangeOfBasis(const IntMatrix& base_K1, IntMatrix& base_K2,
								const vector<pair<int,int> >& map, IntMatrix& P);

void GetChangeOfBasis(const int permutation, IntMatrix& P);

/**
*	Returns the change of coordinate from second element to first element on a 2D face.
*/
void GetChangeOfBasis2D(const int face_id1, const int face_id2, IntMatrix& P);

/**
*	Returns the indices, face ID, and number of rotations, of the two element sharing a face.
*  The number of rotations is relative to the element 1, so nb_rot1 is always 0.
*/
void GetFaceInfo(const Mesh* mesh, const int face,
						int& ind_elt1, int& ind_elt2,
						int& face_id1, int& face_id2,
						int& nb_rot1, int& nb_rot2);

/**
*	Returns the face_id that identifies the face on the reference element, and nb_rot the
*  "rotations" the face did between reference to physical spaces.
*/
void GetIdRotInfo(const int face_info, int& face_id, int& nb_rot);

/**
*	Returns an integer identifying the permutation to apply to be in structured-
*  like configuration for 2D hex meshes.
*/
int Permutation2D(const int face_id_trial, const int face_id_test);

/**
*	Returns an integer identifying the permutation to apply to be in structured-
*  like configuration for 3D hex meshes.
*/
void Permutation3D(const int face_id1, const int face_id2, const int orientation, int& perm1, int& perm2);

/**
*	Returns an integer identifying the permutation to apply to be in structured-
*  like configuration.
*/
void GetPermutation(const int dim, const int face_id1, const int face_id2, const int orientation, int& perm1, int& perm2);

int GetFaceQuadIndex3D(const int face_id, const int orientation, const int qind, const int quads, Tensor<1,int>& ind_f);

/**
*	Returns the indices of a quadrature point on the face of an hex element relative to the index of the quadrature
*  point on the reference face.
*/
int GetFaceQuadIndex(const int dim, const int face_id, const int orientation, const int qind, const int quads, Tensor<1,int>& ind_f);

/**
*	Returns the indices of a quadrature point on the element relative to the index of the quadrature
*  point on the reference face.
*/
const int GetGlobalQuadIndex(const int dim, const int face_id, const int quads, Tensor<1,int>& ind_f);

}

}

#endif // MFEM_DGFACEFUNC