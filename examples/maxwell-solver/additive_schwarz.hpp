#pragma once
#include "mfem.hpp"
#include <fstream>
#include <iostream>
using namespace std;
using namespace mfem;


class mesh_partition // for now every vertex defines a patch 
{
private:
   Mesh *mesh=nullptr;
   void AddElementToMesh(Mesh * mesh,mfem::Element::Type elem_type,int * ind);
   void print_element_map();
   void save_mesh_partition();
public:
   int nrpatch;
   std::vector<Array<int>> element_map; // map local (patch) element to global (original mesh) element
   Array<Mesh *> patch_mesh;
   // constructor
   mesh_partition(Mesh * mesh_);
   ~mesh_partition();
};


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

   // constructor
   PatchAssembly(BilinearForm * bf_);
   ~PatchAssembly();
};

class AddSchwarz : public Solver// 
{
private:
   int nrpatch;
   int maxit = 1;
   double theta = 0.5;
   FiniteElementSpace *fespace=nullptr;
   PatchAssembly * p;
   const Operator * A;
   BilinearForm * bf;
public:
   AddSchwarz(BilinearForm * bf_);
   void SetNumSmoothSteps(const int iter) {maxit = iter;}
   void SetDumpingParam(const double dump_param) {theta = dump_param;}
   virtual void SetOperator(const Operator &op) {A = &op;}
   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~AddSchwarz();
};


