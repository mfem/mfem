#pragma once
#include "mfem.hpp"
#include <fstream>
#include <iostream>
using namespace std;
using namespace mfem;


struct UniqueIndexGenerator
{
   int counter = 0;
   std::unordered_map<int,int> idx;
   int Get(int i)
   {
      std::unordered_map<int,int>::iterator f = idx.find(i);
      if (f == idx.end())
      {
         idx[i] = counter;
         return counter++;
      }
      else
      {
         return (*f).second;
      }
   }
   void Reset()
   {
      counter = 0;
      idx.clear();
   }
};


double GetUniformMeshElementSize(Mesh * mesh);
Mesh * ExtendMesh(Mesh * mesh, const Array<int> & directions);

class CartesianMeshPartition 
{
private:
   Mesh *mesh=nullptr;
public:
   int nrpatch;
   int nxyz[3];
   double MeshSize;
   std::vector<Array<int>> element_map;
   Array3D<int>subdomains;
   // constructor
   CartesianMeshPartition(Mesh * mesh_,int & nx, int & ny, int & nz);
   ~CartesianMeshPartition() {};
};

class OverlappingCartesianMeshPartition 
{
private:
   Mesh *mesh=nullptr;
public:
   int nrpatch;
   double MeshSize;
   int nxyz[3];
   std::vector<Array<int>> element_map;
   Array3D<int> subdomains;
   // constructor
   OverlappingCartesianMeshPartition(Mesh * mesh_,int & nx, int & ny, int & nz);
   OverlappingCartesianMeshPartition(Mesh * mesh_,int & nx, int & ny, int & nz, int ovlp_nlayers);
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
   double MeshSize;
   std::vector<Array<int>> element_map;
   Array3D<int> subdomains;
   Array<Mesh *> patch_mesh;
   int partition_kind;
   int nxyz[3];
   // constructor
   MeshPartition(Mesh * mesh_, int part, int mx=1, int my=1, int mz=1, int ovl_nlayers=0);
   ~MeshPartition();
};

void SaveMeshPartition(Array<Mesh * > meshes, 
                       string mfilename="output/mesh.",
                       string sfilename="output/sol.");


#ifdef MFEM_USE_MPI

class CartesianParMeshPartition 
{
private:
   ParMesh *pmesh=nullptr;
public:
   int nrsubdomains;
   int nxyz[3];
   double MeshSize;
   std::vector<Array<int>> local_element_map;
   Array<int> subdomain_rank;
   Array3D<int>subdomains;
   // constructor
   CartesianParMeshPartition(ParMesh * pmesh_,int & nx, int & ny, int & nz, 
                             int ovlp_nlayers);
   ~CartesianParMeshPartition() {};
};

class ParMeshPartition
{
private:
   MPI_Comm comm;
   ParMesh *pmesh=nullptr;
   void AddElementToMesh(Mesh * mesh,mfem::Element::Type elem_type,int * ind);
   void GetNumVertices(int type, mfem::Element::Type & elem_type, int & nrvert);
   void PrintElementMap();
public:
   int nrsubdomains;
   int OvlpNlayers;
   int myelem_offset = 0;
   double MeshSize;
   std::vector<Array<int>> element_map;
   std::vector<Array<int>> local_element_map;
   Array3D<int> subdomains;
   Array<Mesh *> subdomain_mesh;
   Array<int> subdomain_rank;
   int partition_kind;
   int nxyz[3];
   // constructor
   ParMeshPartition(ParMesh * pmesh_, int mx=1, int my=1, int mz=1, int ovl_nlayers=0);
   void SaveMeshPartition();
   ~ParMeshPartition();
};

#endif