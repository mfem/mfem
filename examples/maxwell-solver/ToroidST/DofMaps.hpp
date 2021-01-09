#pragma once
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void E_exact(const Vector &x, Vector &E);


void FindPtsGetCommonElements(Mesh & mesh0, Mesh & mesh1, 
                              Array<int> & elems0, Array<int> & elems1);

void GetCommonIndices(const Array<int> & list0, const Array<int> & list1, Array<int> & idx0, Array<int> & idx1);

// Given two FiniteElementSpaces and an ElementMap compute
// the dof map between fes0 and fes1
void GetDofMaps(const FiniteElementSpace &fes0, const FiniteElementSpace &fes1, 
                Array<int> & dofs0, Array<int> & dofs1,  
                const Array<int> * elems0_ = nullptr, const Array<int> * elems1_ = nullptr);

// Partition the given mesh to nrsubmeshes with overlap given by ovlp
// ElemMaps: For each subdomain the element indices of the global mesh
// Dofmap0[i]: dof indices of fes in the shared region with subdomain i
// Dofmap1[i]: dof indices of fespaces[i] in the shared region with fes
// OvlpMaps0 : dof indices of fespaces[i] in overlapping region i
// OvlpMaps1 : dof indices of fespaces[i+1] in overlapping region i
void PartitionFE(const FiniteElementSpace * fes, int nrsubmeshes, double ovlp, 
                 Array<FiniteElementSpace*> & fespaces, 
                 Array<Array<int> * > & ElemMaps,
                 Array<Array<int> * > & DofMaps0,
                 Array<Array<int> * > & DofMaps1,
                 Array<Array<int> * > & OvlpMaps0, 
                 Array<Array<int> * > & OvlpMaps1);


void GetRestrictionDofs(FiniteElementSpace &fes, int direction, double ovlp, Array<int> & rdofs);
void RestrictDofs(const Array<int> & rdofs, int tsize, Vector & x);

// direction:  1 left (anti-clockwise)
//            -1 right (clockwise)
//             0 both the above
// ovlp     :  given in degrees         
void GetElements(Mesh &mesh, double ovlp, int direction, Array<int> & elems);

void MapDofs(const Array<int> & dmap0, const Array<int> & dmap1,
             const Vector &gf0, Vector &gf1);

void AddMapDofs(const Array<int> & dmap0, const Array<int> & dmap1,
                const Vector &gf0, Vector &gf1);

void DofMapTests(FiniteElementSpace &fes0, FiniteElementSpace &fes1,
                 const Array<int> & dmap0, const Array<int> & dmap1);                 

void DofMapOvlpTest(FiniteElementSpace &fes, const Array<int> & dmap);
                       