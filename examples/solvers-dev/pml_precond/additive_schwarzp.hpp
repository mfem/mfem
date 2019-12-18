#pragma once
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <unordered_map>

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


class ParMeshPartition // for now every vertex defines a patch 
{
private:
   ParMesh *pmesh=nullptr;
   void AddElementToMesh(Mesh * mesh,mfem::Element::Type elem_type,int * ind);
   void GetNumVertices(int type, mfem::Element::Type & elem_type, int & nrvert);
   void save_mesh_partition();
   void print_element_map(){};
public:
   int nrpatch;
   std::vector<Array<int>> element_map; 
   Array<Mesh *> patch_mesh;
   // constructor
   ParMeshPartition(ParMesh * pmesh_);
   ~ParMeshPartition(){}
};


class ParPatchAssembly // for now every vertex defines a patch 
{
   ParFiniteElementSpace *fespace=nullptr;
   ParBilinearForm *bf=nullptr;
   void print_patch_dof_map(){};
public:
   int nrpatch;
   Array<FiniteElementSpace *> patch_fespaces;
   std::vector<Array<int>> patch_dof_map;
   Array<SparseMatrix *> patch_mat;
   Array<KLUSolver * > patch_mat_inv;
   std::vector<Array<int>> ess_tdof_list;

   // constructor
   ParPatchAssembly(ParBilinearForm * bf_);
   ~ParPatchAssembly();
};

class ParAddSchwarz : public Solver// 
{
private:
   int nrpatch;
   int maxit = 1;
   double theta = 0.5;
   FiniteElementSpace *fespace=nullptr;
   ParPatchAssembly * p;
   const Operator * A;
   BilinearForm * bf;
public:
   ParAddSchwarz(ParBilinearForm * bf_);
   void SetNumSmoothSteps(const int iter) {maxit = iter;}
   void SetDumpingParam(const double dump_param) {theta = dump_param;}
   virtual void SetOperator(const Operator &op) {A = &op;}
   virtual void Mult(const Vector &r, Vector &z) const {} 
   virtual ~ParAddSchwarz(){};
};


