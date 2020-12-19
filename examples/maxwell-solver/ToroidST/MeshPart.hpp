
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Partition mesh according to Attributes
// @input: mesh0 : the mesh to get trimmed
//         attr  : Attributes to remove or leave depending on the complement flag
Mesh * GetPartMesh(const Mesh * mesh0, const Array<int> & attr, bool complement = false);