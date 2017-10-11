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


/** A function that initialize the local coordinate base for a face with
*   indice face_ind.
*   This returns the local face coordinate base expressed in reference
*   element coordinate.
*/
template <int Dim>
void InitFaceCoord(const int face_ind, DenseMatrix& base)

// Highly dependent of node ordering from geom.cpp
template <>
void InitFaceCoord<2>(const int face_ind, DenseMatrix& base)
{
	switch(face_ind)
	{
		case 0://SOUTH
			base.col(1) = e1;
			base.col(2) = -e2;
			break;
		case 1://EAST
			base.col(1) = e2;
			base.col(2) = e1;
			break;
		case 2://NORTH
			base.col(1) = -e1;
			base.col(2) = e2;
			break;
		case 3://WEST
			base.col(1) = -e2;
			base.col(2) = e1;
			break;
		default:
			mfem_error("The face_ind exceeds the nu;ber of faces in this dimension.");
			break;
	}
}

// Highly dependent of node ordering from geom.cpp
template <>
void InitFaceCoord<3>(const int face_ind, DenseMatrix& base)
{
	switch(face_ind)
	{
		case 0://BOTTOM
			base.col(1) =  e1;
			base.col(2) = -e2;
			base.col(3) = -e3;
			break;		
		case 1://SOUTH
			base.col(1) =  e1;
			base.col(2) =  e3;
			base.col(3) = -e2;
			break;
		case 2://EAST
			base.col(1) =  e2;
			base.col(2) =  e3;
			base.col(3) =  e1;
			break;
		case 3://NORTH
			base.col(1) = -e1;
			base.col(2) =  e3;
			base.col(3) =  e2;
			break;
		case 4://WEST
			base.col(1) = -e2;
			base.col(2) =  e3;
			base.col(3) = -e1;
			break;
		case 5://TOP
			base.col(1) =  e1;
			base.col(2) =  e2;
			base.col(3) =  e3;
			break;
		default:
			mfem_error("The face_ind exceeds the nu;ber of faces in this dimension.");
			break;
	}
}

/** Maps the coordinate vectors of the first face to the coordinate vectors of the second face.
*   nb_rot is the number of rotation to opperate so that the first node of each face match.
*   The result map contains pairs of int, where the first int is the cofficient, and the 
*   second int is the indice of the second face vector.
*/
template <int Dim>
void GetLocalCoordMap(vector<pair<int,int>>& map, const int nb_rot);

// There shouldn't be any rotation in 2D.
template <>
void GetLocalCoordMap<2>(vector<pair<int,int>>& map, const int nb_rot = 0)
{
	map.resize(2);
	//First and second coordinate vectors should always be of opposite direction in 2D.
	map[0] = pair(-1,1);
	map[1] = pair(-1,2);
}

// Default parameter nb_rot=0 should be only use with a structured mesh.
// Rotations follow the ordering of the nodes.
template <>
void GetLocalCoordMap<3>(vector<pair<int,int>>& map, const int nb_rot = 0)
{
	map.resize(3);
	// Normal to the face are always of opposite direction
	map[2] = pair(-1,3);
	// nb_rot determines how local coordinates are oriented from one face to the other.
	// See case 2 for an example.
	switch(nb_rot)
	{
		case 0:
			map[0] = pair( 1,2);
			map[1] = pair( 1,1);
			break;
		case 1:
			map[0] = pair(-1,1);
			map[1] = pair( 1,2);
			break;
		case 2:
			//first vector equals -1 times the second vector of the other face coordinates
			map[0] = pair(-1,2);
			//second vector equals -1 times the first vector of the other face coordinates
			map[1] = pair(-1,1);
			break;
		case 3:
			map[0] = pair( 1,1);
			map[1] = pair(-1,2);
			break;
		default:
			mfem_error("There shouldn't be that many rotations.");
			break;
	}
}

/**
*	Returns the change of matrix P from base_K2 to base_K1 according to the mapping map.
*/
template <int Dim>
void getChangeOfBasis(const DenseMatrix& base_K1, const DenseMatrix& base_K2,
								const vector<pair<int,int>>& map, DenseMatrix& P)
{
	for (int j = 0; j < Dim; j++)
	{
		int i = 0;
		//we look if the vector is colinear with e_j
		while (base_K2.col(i)[j]!=0) i++;
		int coeff = map[i].first;
		int ind = map[i].second;
		P.col(j) = coeff * base_K1.col(ind);
	}
}