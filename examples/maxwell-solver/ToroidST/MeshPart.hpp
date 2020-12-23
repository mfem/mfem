#pragma once
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


int get_angle_range(double angle, Array<double> angles);

void SetMeshAttributes(Mesh * mesh, int subdivisions, double ovlp);

// Partition mesh to nrsubmeshes (equally spaced in the azimuthal direction)
void PartitionMesh(Mesh * mesh, int nrsubmeshes, double ovlp, 
                   Array<Mesh*> SubMeshes, Array<Array<int> *>elems);


// Partition mesh according to Attributes
// @input: mesh0 : the mesh to get trimmed
//         attr  : Attributes to remove or leave depending on the complement flag
Mesh * GetPartMesh(const Mesh * mesh0, const Array<int> & attr, 
                    Array<int> & elem_map, bool complement = false);