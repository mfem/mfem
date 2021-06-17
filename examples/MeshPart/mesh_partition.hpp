#pragma once
#include "mfem.hpp"
#include <fstream>
#include <iostream>
using namespace std;
using namespace mfem;


class SubMesh
{
private:
   Array<int> element_map;
public:
   SubMesh(const Mesh & mesh_, const Array<int> & elems);
   Mesh *mesh=nullptr;
   void GetElementMap(Array<int> & element_map_)
   {
      element_map_ = element_map;
   }
   ~SubMesh(){ };
};

// void AddElementToMesh(Mesh * mesh,mfem::Element::Type elem_type,int * ind);
// void GetNumVertices(int type, mfem::Element::Type & elem_type, int & nrvert);
// void PrintElementMap();