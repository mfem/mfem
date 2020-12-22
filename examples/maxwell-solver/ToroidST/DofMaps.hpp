
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


void GetCommonIndices(const Array<int> & list0, const Array<int> & list1, Array<int> & idx0, Array<int> & idx1);

// Given two FiniteElementSpaces and an ElementMap compute
// the dof map between fes0 and fes1
Array2D<int> * GetDofMap(const FiniteElementSpace &fes0, const FiniteElementSpace &fes1, 
               const Array<int> * elems0_ = nullptr, const Array<int> * elems1_ = nullptr);