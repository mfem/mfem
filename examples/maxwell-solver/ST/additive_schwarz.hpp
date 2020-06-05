#pragma once
#include "mfem.hpp"
#include <fstream>
#include <iostream>
using namespace std;
using namespace mfem;


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


class CartesianMeshPartition // for now every vertex defines a patch
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

class VertexMeshPartition // for now every vertex defines a patch
{
private:
   Mesh *mesh=nullptr;
public:
   int nrpatch;
   // map local (patch) element to global (original mesh) element
   std::vector<Array<int>> element_map; 
   // constructor
   VertexMeshPartition(Mesh * mesh_);
   ~VertexMeshPartition() {};
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


class PatchAssembly // for now every vertex defines a patch
{
   FiniteElementSpace *fespace=nullptr;
   BilinearForm *bf=nullptr;
   void print_patch_dof_map();
public:
   int nrpatch;
   Array<FiniteElementSpace *> patch_fespaces;
   std::vector<Array<int>> patch_dof_map;
   Array<SparseMatrix *> patch_mat;
   Array<KLUSolver * > patch_mat_inv;
   std::vector<Array<int>> ess_tdof_list;
   std::vector<Array<int>> ess_int_tdofs;

   // constructor
   PatchAssembly(BilinearForm * bf_, Array<int> & ess_tdofs, int part);
   ~PatchAssembly();
};

class AddSchwarz : public Solver//
{
private:
   int nrpatch;
   int maxit = 1;
   int part;
   double theta = 0.5;
   PatchAssembly * p;
   const Operator * A;
public:
   AddSchwarz(BilinearForm * bf_, Array<int> & ess_tdofs, int i = 0);
   void SetNumSmoothSteps(const int iter)
   {
      maxit = iter;
   }
   void SetDumpingParam(const double dump_param)
   {
      theta = dump_param;
   }
   virtual void SetOperator(const Operator &op)
   {
      A = &op;
   }
   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~AddSchwarz();
};

Mesh * ExtendMesh(Mesh * mesh, const Array<int> & directions);

double GetUniformMeshElementSize(Mesh * mesh);
