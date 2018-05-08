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

#include <vector>
#include "fem.hpp"
#include "dalg.hpp"

using std::vector;
using std::pair;

namespace mfem
{

/**
*	Returns the canonical coordinate vectors e_1 and e_2.
*/
void getBaseVector2D(Vector& e1, Vector& e2)
{
	e1.SetSize(2);
	e1(0) = 1;
	e1(1) = 0;
	e2.SetSize(2);
	e2(0) = 0;
	e2(1) = 1;
}

/**
*	Returns the canonical coordinate vectors e_1, e_2 and e_3.
*/
void getBaseVector3D(Vector& e1, Vector& e2, Vector& e3)
{
	e1.SetSize(3);
	e1(0) = 1;
	e1(1) = 0;
	e1(2) = 0;
	e2.SetSize(3);
	e2(0) = 0;
	e2(1) = 1;
	e2(2) = 0;
	e3.SetSize(3);
	e3(0) = 0;
	e3(1) = 0;
	e3(2) = 1;
}

/** 
*   A function that initialize the local coordinate base for a face with
*   indice face_ind.
*   This returns the local face coordinate base expressed in reference
*   element coordinate.
*/
// Highly dependent of the node ordering from geom.cpp
void InitFaceCoord2D(const int face_id, IntMatrix& base)
{
	//Vector e1,e2;
	//getBaseVector2D(e1,e2);
	base.Zero();
	switch(face_id)
	{
		case 0://SOUTH
			base(0,0)= 1;//base.SetCol(0, e1);
			base(1,1)=-1;//base.SetCol(1,-e2);
			break;
		case 1://EAST
			base(1,0)= 1;//base.SetCol(0, e2);
			base(0,1)= 1;//base.SetCol(1, e1);
			break;
		case 2://NORTH
			base(0,0)=-1;//base.SetCol(0,-e1);
			base(1,1)= 1;//base.SetCol(1, e2);
			break;
		case 3://WEST
			base(1,0)=-1;//base.SetCol(0,-e2);
			base(0,1)= 1;//base.SetCol(1, e1);
			break;
		default:
			mfem_error("The face_ind exceeds the number of faces in this dimension.");
			break;
	}
}

// Highly dependent of the node ordering from geom.cpp
void InitFaceCoord3D(const int face_id, IntMatrix& base)
{
	//Vector e1,e2,e3;
	//getBaseVector3D(e1,e2,e3);
	base.Zero();
	switch(face_id)
	{
		case 0://BOTTOM
			base(0,0)= 1;//base.SetCol(0, e1);
			base(1,1)=-1;//base.SetCol(1,-e2);
			base(2,2)=-1;//base.SetCol(2,-e3);
			break;		
		case 1://SOUTH
			base(0,0)= 1;//base.SetCol(0, e1);
			base(2,1)= 1;//base.SetCol(1, e3);
			base(1,2)=-1;//base.SetCol(2,-e2);
			break;
		case 2://EAST
			base(1,0)= 1;//base.SetCol(0, e2);
			base(2,1)= 1;//base.SetCol(1, e3);
			base(0,2)= 1;//base.SetCol(2, e1);
			break;
		case 3://NORTH
			base(0,0)=-1;//base.SetCol(0,-e1);
			base(2,1)= 1;//base.SetCol(1, e3);
			base(1,2)= 1;//base.SetCol(2, e2);
			break;
		case 4://WEST
			base(1,0)=-1;//base.SetCol(0,-e2);
			base(2,1)= 1;//base.SetCol(1, e3);
			base(0,2)=-1;//base.SetCol(2,-e1);
			break;
		case 5://TOP
			base(0,0)= 1;//base.SetCol(0, e1);
			base(1,1)= 1;//base.SetCol(1, e2);
			base(2,2)= 1;//base.SetCol(2, e3);
			break;
		default:
			mfem_error("The face_ind exceeds the number of faces in this dimension.");
			break;
	}
}

/** Maps the coordinate vectors of the first face to the coordinate vectors of the second face.
*   nb_rot is the number of rotation to opperate so that the first node of each face match.
*   The result map contains pairs of int, where the first int is the cofficient, and the 
*   second int is the indice of the second face vector.
*/
// There shouldn't be any rotation in 2D.
void GetLocalCoordMap2D(vector<pair<int,int> >& map, const int nb_rot)
{
	map.resize(2);
	//First and second coordinate vectors should always be of opposite direction in 2D.
	//TODO Maybe not
	map[0] = pair<int,int>(-1,0);
	map[1] = pair<int,int>(-1,1);
}

// Default parameter nb_rot=0 should be only use with a structured mesh.
// Rotations follow the ordering of the nodes.
/*void GetLocalCoordMap3D(vector<pair<int,int> >& map, const int nb_rot)
{
	map.resize(3);
	// Normal to the face are always of opposite direction
	map[2] = pair<int,int>(-1,2);
	// nb_rot determines how local coordinates are oriented from one face to the other.
	// See case 2 for an example.
	switch(nb_rot)
	{
		case 0:
			map[0] = pair<int,int>( 1,1);
			map[1] = pair<int,int>( 1,0);
			break;
		case 1:
			map[0] = pair<int,int>(-1,0);
			map[1] = pair<int,int>( 1,1);
			break;
		case 2:
			//first vector equals -1 times the second vector of the other face coordinates
			map[0] = pair<int,int>(-1,1);
			//second vector equals -1 times the first vector of the other face coordinates
			map[1] = pair<int,int>(-1,0);
			break;
		case 3:
			map[0] = pair<int,int>( 1,0);
			map[1] = pair<int,int>(-1,1);
			break;
		default:
			mfem_error("There shouldn't be that many rotations.");
			break;
	}
}*/

void GetLocalCoordMap3D(vector< pair<int,int> >& map, const int orientation)
{
	map.resize(3);
	// orientation determines how local coordinates are oriented from one face to the other.
	// See case 2 for an example.
	switch(orientation)
	{
		case 0://{0, 1, 2, 3}
			map[0] = pair<int,int>( 1,0);
			map[1] = pair<int,int>( 1,1);
			map[2] = pair<int,int>( 1,2);
			break;
		case 1://{0, 3, 2, 1}
			map[0] = pair<int,int>( 1,1);
			map[1] = pair<int,int>( 1,0);
			map[2] = pair<int,int>(-1,2);
			break;
		case 2://{1, 2, 3, 0}
			//first vector equals -1 times the second vector of the other face coordinates
			map[0] = pair<int,int>(-1,1);
			//second vector equals -1 times the first vector of the other face coordinates
			map[1] = pair<int,int>( 1,0);
			//third vector equals -1 times the third vector of the other face coordinates
			map[2] = pair<int,int>( 1,2);
			break;
		case 3://{1, 0, 3, 2}
			map[0] = pair<int,int>(-1,0);
			map[1] = pair<int,int>( 1,1);
			map[2] = pair<int,int>(-1,2);
			break;
		case 4://{2, 3, 0, 1}
			map[0] = pair<int,int>(-1,0);
			map[1] = pair<int,int>(-1,1);
			map[2] = pair<int,int>( 1,2);
			break;
		case 5://{2, 1, 0, 3}
			map[0] = pair<int,int>(-1,1);
			map[1] = pair<int,int>(-1,0);
			map[2] = pair<int,int>(-1,2);
			break;
		case 6://{3, 0, 1, 2}
			map[0] = pair<int,int>( 1,1);
			map[1] = pair<int,int>(-1,0);
			map[2] = pair<int,int>( 1,2);
			break;
		case 7://{3, 2, 1, 0}
			map[0] = pair<int,int>( 1,0);
			map[1] = pair<int,int>(-1,1);
			map[2] = pair<int,int>(-1,2);
			break;
		default:
			mfem_error("There shouldn't be that many orientations.");
			break;
	}
}

/**
*	Returns the change of matrix P from base_K2 to base_K1 according to the mapping map.
*/
void GetChangeOfBasis(const IntMatrix& base_K1, IntMatrix& base_K2,
								const vector<pair<int,int> >& map, IntMatrix& P)
{
/*	int dim = map.size();
	for (int j = 0; j < dim; j++)
	{
		int i = 0;
		//we look if the vector is colinear with e_j
		// Can be replaced by base_K2(j,i)!=0
		while (base_K2(j,i)!=0) i++;
		int coeff = map[i].first;
		int ind = map[i].second;
		for (int k = 0; k < dim; ++k)
		{
			P(k,j) = coeff * base_K1(k,ind);
		}
	}*/
	//TODO make it valid for 3D!!!
	int dim = base_K1.Height();
	// for (int i = 0; i < dim; ++i)
	// {
	// 	int coeff = map[i].first;
	// 	int ind   = map[i].second;
	// 	for (int j = 0; j < dim; ++j)
	// 	{
	// 		int sum = 0;
	// 		for (int k = 0; k < dim; ++k)
	// 		{
	// 			sum += coeff*base_K1(i,k)*base_K2(j,k);
	// 		}
	// 		P(ind,j) =  sum;
	// 	}
	// }
	int i,j,ind;
	double coeff;
	for (int n = 0; n < dim; ++n)
	{
		i = 0;
		while( base_K1(i,n)   == 0 ) ++i;
		j = 0;
		ind = map[n].second;
		while( base_K2(j,ind) == 0 ) ++j;
		coeff = map[n].first;
		P(i,j) = coeff * base_K1(i,n) * base_K2(j,ind);
	}
}

void GetChangeOfBasis2D(const int face_id1, const int face_id2, IntMatrix& P)
{
	// We add 8 because of C++ stupid definition of modulo
	int nb_rot = (8 + face_id2 - face_id1 - 2)%4;
	// if (face_id2!=-1)
	// {
		// cout << "face_id1=" << face_id1 << ", face_id2=" << face_id2 << ", nb_rot=" << nb_rot << endl;
	// }
	P.Zero();
	switch(nb_rot)
	{
	case 0://Id=R^4
		P(0,0) = 1;
		P(1,1) = 1;
		break;
	case 1://R
		P(1,0) = 1;
		P(0,1) =-1;
		break;
	case 2://R²
		P(0,0) =-1;
		P(1,1) =-1;
		break;
	case 3://R³
		P(1,0) =-1;
		P(0,1) = 1;
		break;
	default:mfem_error("C++ modulo error in GetChangeOfBasis2D");
	}
}

void GetChangeOfBasis(const int permutation, IntMatrix& P)
{
	int code1 = permutation/100;
	int ind1  = code1/2;
	int val1  = code1%2==0?-1:1;
	int code2 = (permutation%100)/10;
	int ind2  = code2/2;
	int val2  = code2%2==0?-1:1;
	int code3 = permutation%10;
	int ind3  = code3/2;
	int val3  = code3%2==0?-1:1;
	P.Zero();
	P(ind1,0) = val1;
	P(ind2,1) = val2;
	P(ind3,2) = val3;
}

/**
*	Returns the face_id that identifies the face on the reference element, and nb_rot the
*  "rotations" the face did between reference to physical spaces.
*/
void GetIdRotInfo(const int face_info, int& face_id, int& nb_rot){
	int orientation = face_info % 64;
	face_id = face_info / 64;
	// Test if my understanding of mfem code is correct, error if not
	//MFEM_ASSERT(orientation % 2 == 0, "Unexpected inside out face");
	nb_rot = orientation;// / 2;
}

void GetFaceInfo(const Mesh* mesh, const int face, int& ind_elt1, int& ind_elt2, int& face_id1, int& face_id2, int& nb_rot1, int& nb_rot2)
{
	// We collect the indices of the two elements on the face, element1 is the master element,
	// the one that defines the normal to the face.
	mesh->GetFaceElements(face,&ind_elt1,&ind_elt2);
	int info_elt1, info_elt2;
	// We collect the informations on the face for the two elements.
	mesh->GetFaceInfos(face,&info_elt1,&info_elt2);
	GetIdRotInfo(info_elt1,face_id1,nb_rot1);//nb_rot1 is always 0 by convention
	GetIdRotInfo(info_elt2,face_id2,nb_rot2);
}

/**
*	Returns the permutation id, so that we can permute dofs to be in a structured case.
*/
int Permutation2D(const int face_id_trial, const int face_id_test)
{
	int perm = face_id_trial - face_id_test - 2;
	perm = perm < 0 ? perm+4 : perm;
	return perm;
}

/**
*  Returns an integer that encrypts P.
*/
void Permutation3D(const int face_id1, const int face_id2, const int orientation, int& perm1, int& perm2)
{
	IntMatrix K1(3,3);
	K1.Zero();
	InitFaceCoord3D(face_id1, K1);
	IntMatrix K2(3,3);
	K2.Zero();
	InitFaceCoord3D(face_id2, K2);
	vector< pair<int,int> > map;
	GetLocalCoordMap3D(map, orientation);
	IntMatrix P(3,3);
	P.Zero();
	GetChangeOfBasis(K1, K2, map, P);
	// cout << "orientation=" << orientation << endl;
	// cout << P(0,0) << ", " << P(0,1) << ", " << P(0,2) << endl;
	// cout << P(1,0) << ", " << P(1,1) << ", " << P(1,2) << endl;
	// cout << P(2,0) << ", " << P(2,1) << ", " << P(2,2) << endl;
	perm1  = 0;
	// Encrypts first column
	perm1 += 100*(0*(P(0,0)==-1) + 1*(P(0,0)==1) + 2*(P(1,0)==-1) + 3*(P(1,0)==1) + 4*(P(2,0)==-1) + 5*(P(2,0)==1));
	// Encrypts second column
	perm1 += 10 *(0*(P(0,1)==-1) + 1*(P(0,1)==1) + 2*(P(1,1)==-1) + 3*(P(1,1)==1) + 4*(P(2,1)==-1) + 5*(P(2,1)==1));
	// Encrypts third column
	perm1 +=     (0*(P(0,2)==-1) + 1*(P(0,2)==1) + 2*(P(1,2)==-1) + 3*(P(1,2)==1) + 4*(P(2,2)==-1) + 5*(P(2,2)==1));
	// Encrypts the transposed permutation matrix in a second integer.
	perm2  = 0;
	perm2 += 100*(0*(P(0,0)==-1) + 1*(P(0,0)==1) + 2*(P(0,1)==-1) + 3*(P(0,1)==1) + 4*(P(0,2)==-1) + 5*(P(0,2)==1));
	perm2 += 10 *(0*(P(1,0)==-1) + 1*(P(1,0)==1) + 2*(P(1,1)==-1) + 3*(P(1,1)==1) + 4*(P(1,2)==-1) + 5*(P(1,2)==1));
	perm2 +=     (0*(P(2,0)==-1) + 1*(P(2,0)==1) + 2*(P(2,1)==-1) + 3*(P(2,1)==1) + 4*(P(2,2)==-1) + 5*(P(2,2)==1));
}

void GetPermutation(const int dim, const int face_id1, const int face_id2, const int orientation, int& perm1, int& perm2)
{
	switch(dim){
		case 1:
			mfem_error("Not yet implemented");
			break;
		case 2:
			perm1 = Permutation2D(face_id1, face_id2);
			perm2 = Permutation2D(face_id2, face_id1);
			break;
		case 3:
			Permutation3D(face_id1, face_id2, orientation, perm1, perm2);
			break;
		default:
			mfem_error("Dimension of the problem too high.");
			break;
	}
}

/**
*	Hardcoded permutation due to arbitrary hardcoded orientation in geom.cpp.
*   Will break if geom.cpp changes.
*   This function could be improved by returning the 'permutation' parameters once,
*   instead of recomputing them for every quadrature point...
*/
int GetFaceQuadIndex3D(const int face_id, const int orientation, const int qind, const int quads, Tensor<1,int>& ind_f)
{
	// cout << "orientation=" << orientation << endl;
	int& k1 = ind_f(0);
	int& k2 = ind_f(1);
	int kf1,kf2;
	kf1 = qind%quads;
	kf2 = qind/quads;
	switch(face_id)
	{
		case 0://BOTTOM
			switch(orientation)
			{
				case 0://{0, 1, 2, 3}
					k1 = kf1;
					k2 = quads-1-kf2;
					break;
				case 1://{0, 3, 2, 1}
					k1 = quads-1-kf2;
					k2 = kf1;
					break;
				case 2://{1, 2, 3, 0}
					k1 = quads-1-kf2;
					k2 = quads-1-kf1;
					break;
				case 3://{1, 0, 3, 2}
					k1 = quads-1-kf1;
					k2 = quads-1-kf2;
					break;
				case 4://{2, 3, 0, 1}
					k1 = quads-1-kf1;
					k2 = kf2;
					break;
				case 5://{2, 1, 0, 3}
					k1 = kf2;
					k2 = quads-1-kf1;
					break;
				case 6://{3, 0, 1, 2}
					k1 = kf2;
					k2 = kf1;
					break;
				case 7://{3, 2, 1, 0}
					k1 = kf1;
					k2 = kf2;
					break;
				default:
					mfem_error("This orientation does not exist in 3D");
					break;
			}
			break;
		case 1://SOUTH
			switch(orientation)
			{
				case 0://{0, 1, 2, 3}
					k1 = kf1;
					k2 = kf2;
					break;
				case 1://{0, 3, 2, 1}
					k1 = kf2;
					k2 = kf1;
					break;
				case 2://{1, 2, 3, 0}
					k1 = kf2;
					k2 = quads-1-kf1;
					break;
				case 3://{1, 0, 3, 2}
					k1 = quads-1-kf1;
					k2 = kf2;
					break;
				case 4://{2, 3, 0, 1}
					k1 = quads-1-kf1;
					k2 = quads-1-kf2;
					break;
				case 5://{2, 1, 0, 3}
					k1 = quads-1-kf2;
					k2 = quads-1-kf1;
					break;
				case 6://{3, 0, 1, 2}
					k1 = quads-1-kf2;
					k2 = kf1;
					break;
				case 7://{3, 2, 1, 0}
					k1 = kf1;
					k2 = quads-1-kf2;
					break;
				default:
					mfem_error("This orientation does not exist in 3D");
					break;
			}
			break;
		case 2://EAST
			switch(orientation)
			{
				case 0://{0, 1, 2, 3}
					k1 = kf1;
					k2 = kf2;
					break;
				case 1://{0, 3, 2, 1}
					k1 = kf2;
					k2 = kf1;
					break;
				case 2://{1, 2, 3, 0}
					k1 = kf2;
					k2 = quads-1-kf1;
					break;
				case 3://{1, 0, 3, 2}
					k1 = quads-1-kf1;
					k2 = kf2;
					break;
				case 4://{2, 3, 0, 1}
					k1 = quads-1-kf1;
					k2 = quads-1-kf2;
					break;
				case 5://{2, 1, 0, 3}
					k1 = quads-1-kf2;
					k2 = quads-1-kf1;
					break;
				case 6://{3, 0, 1, 2}
					k1 = quads-1-kf2;
					k2 = kf1;
					break;
				case 7://{3, 2, 1, 0}
					k1 = kf1;
					k2 = quads-1-kf2;
					break;
				default:
					mfem_error("This orientation does not exist in 3D");
					break;
			}
			break;
		case 3://NORTH
			switch(orientation)
			{
				case 0://{0, 1, 2, 3}
					k1 = quads-1-kf1;
					k2 = kf2;
					break;
				case 1://{0, 3, 2, 1}
					k1 = kf2;
					k2 = quads-1-kf1;
					break;
				case 2://{1, 2, 3, 0}
					k1 = kf2;
					k2 = kf1;
					break;
				case 3://{1, 0, 3, 2}
					k1 = kf1;
					k2 = kf2;
					break;
				case 4://{2, 3, 0, 1}
					k1 = kf1;
					k2 = quads-1-kf2;
					break;
				case 5://{2, 1, 0, 3}
					k1 = quads-1-kf2;
					k2 = kf1;
					break;
				case 6://{3, 0, 1, 2}
					k1 = quads-1-kf2;
					k2 = quads-1-kf1;
					break;
				case 7://{3, 2, 1, 0}
					k1 = quads-1-kf1;
					k2 = quads-1-kf2;
					break;
				default:
					mfem_error("This orientation does not exist in 3D");
					break;
			}
			break;
		case 4://WEST
			switch(orientation)
			{
				case 0://{0, 1, 2, 3}
					k1 = quads-1-kf1;
					k2 = kf2;
					break;
				case 1://{0, 3, 2, 1}
					k1 = kf2;
					k2 = quads-1-kf1;
					break;
				case 2://{1, 2, 3, 0}
					k1 = kf2;
					k2 = kf1;
					break;
				case 3://{1, 0, 3, 2}
					k1 = kf1;
					k2 = kf2;
					break;
				case 4://{2, 3, 0, 1}
					k1 = kf1;
					k2 = quads-1-kf2;
					break;
				case 5://{2, 1, 0, 3}
					k1 = quads-1-kf2;
					k2 = kf1;
					break;
				case 6://{3, 0, 1, 2}
					k1 = quads-1-kf2;
					k2 = quads-1-kf1;
					break;
				case 7://{3, 2, 1, 0}
					k1 = quads-1-kf1;
					k2 = quads-1-kf2;
					break;
				default:
					mfem_error("This orientation does not exist in 3D");
					break;
			}
			break;
		case 5://TOP
			switch(orientation)
			{
				case 0://{0, 1, 2, 3}
					k1 = kf1;
					k2 = kf2;
					break;
				case 1://{0, 3, 2, 1}
					k1 = kf2;
					k2 = kf1;
					break;
				case 2://{1, 2, 3, 0}
					k1 = kf2;
					k2 = quads-1-kf1;
					break;
				case 3://{1, 0, 3, 2}
					k1 = quads-1-kf1;
					k2 = kf2;
					break;
				case 4://{2, 3, 0, 1}
					k1 = quads-1-kf1;
					k2 = quads-1-kf2;
					break;
				case 5://{2, 1, 0, 3}
					k1 = quads-1-kf2;
					k2 = quads-1-kf1;
					break;
				case 6://{3, 0, 1, 2}
					k1 = quads-1-kf2;
					k2 = kf1;
					break;
				case 7://{3, 2, 1, 0}
					k1 = kf1;
					k2 = quads-1-kf2;
					break;
				default:
					mfem_error("This orientation does not exist in 3D");
					break;
			}
			break;
		default:
			mfem_error("This face_id does not exist in 3D");
			break;	
	}
	return k1 + quads*k2;
}

int GetFaceQuadIndex(const int dim, const int face_id, const int orientation, const int qind, const int quads, Tensor<1,int>& ind_f)
{
	int res = 0;
	switch(dim)
	{
		case 1:
			break;
		case 2:
            if(face_id<=1){//SOUTH or EAST (canonical ordering)
               res = ind_f(0) = qind;
            }else{//NORTH or WEST (counter-canonical ordering)
               res = ind_f(0) = quads-1-qind;
            }
            break;
        case 3:
			res = GetFaceQuadIndex3D(face_id, orientation, qind, quads, ind_f);
			break;
		default:
			mfem_error("Dimension too high.");
			break;
	}
	return res;
}

const int GetGlobalQuadIndex(const int dim, const int face_id, const int quads, Tensor<1,int>& ind_f)
{
   switch(dim)
   {
      case 1:
         if (face_id==0)//WEST
         {
            return 0;
         }else{//EAST
            return quads-1;
         }
      case 2:
            switch(face_id)
            {
               case 0://SOUTH
                  return ind_f(0);
               case 1://EAST
                  return quads-1 + ind_f(0)*quads;
               case 2://NORTH
                  return ind_f(0) + (quads-1)*quads;
               case 3://WEST
                  return ind_f(0)*quads;
            }
        case 3:
         switch(face_id)
         {
            case 0://BOTTOM
               return ind_f(0) + ind_f(1)*quads;
            case 1://SOUTH
               return ind_f(0) + ind_f(1)*quads*quads;
            case 2://EAST
               return (quads-1) + ind_f(0)*quads + ind_f(1)*quads*quads;
            case 3://NORTH
               return ind_f(0) + (quads-1)*quads + ind_f(1)*quads*quads;
            case 4://WEST
               return ind_f(0)*quads + ind_f(1)*quads*quads;
            case 5://TOP
               return ind_f(0) + ind_f(1)*quads + (quads-1)*quads*quads;
         }
      default:
         mfem_error("Dimension too high.");
         break;
   }
   return -1;
}

}