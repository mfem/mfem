#pragma once
#include "mfem.hpp"
#include <fstream>
#include <iostream>
using namespace std;
using namespace mfem;

double GetUniformMeshElementSize(Mesh * mesh);
Mesh * ExtendMesh(Mesh * mesh, const Array<int> & directions);

class CartesianMeshPartition 
{
private:
   Mesh *mesh=nullptr;
public:
   int nrpatch;
   int nx, ny, nz;
   std::vector<Array<int>> element_map;
   // constructor
   CartesianMeshPartition(Mesh * mesh_);
   ~CartesianMeshPartition() {};
};

class OverlappingCartesianMeshPartition 
{
private:
   Mesh *mesh=nullptr;
public:
   int nrpatch;
   int nx, ny, nz;
   std::vector<Array<int>> element_map;
   // constructor
   OverlappingCartesianMeshPartition(Mesh * mesh_);
   ~OverlappingCartesianMeshPartition() {};
};

class STPOverlappingCartesianMeshPartition // Special layered partition for STP
{
private:
   Mesh *mesh=nullptr;
public:
   int nrpatch;
   int nx, ny, nz;
   std::vector<Array<int>> element_map;
   // constructor
   STPOverlappingCartesianMeshPartition(Mesh * mesh_);
   ~STPOverlappingCartesianMeshPartition() {};
};

class MeshPartition
{
private:
   Mesh *mesh=nullptr;
   void AddElementToMesh(Mesh * mesh,mfem::Element::Type elem_type,int * ind);
   void GetNumVertices(int type, mfem::Element::Type & elem_type, int & nrvert);
   void PrintElementMap();
public:
   int nrpatch;
   int nx, ny, nz;
   std::vector<Array<int>> element_map;
   Array<Mesh *> patch_mesh;
   int partition_kind;
   // constructor
   MeshPartition(Mesh * mesh_, int part);
   ~MeshPartition();
};

void SaveMeshPartition(Array<Mesh * > meshes, 
                       string mfilename="output/mesh.",
                       string sfilename="output/sol.");
