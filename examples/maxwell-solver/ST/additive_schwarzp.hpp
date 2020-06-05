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


class CartesianParMeshPartition // for now every vertex defines a patch
{
private:
   ParMesh *pmesh=nullptr;
public:
   int nrpatch;
   Array<int> patch_rank;
   std::vector<Array<int>> local_element_map;
   // constructor
   CartesianParMeshPartition(ParMesh * pmesh_);
   ~CartesianParMeshPartition() {};
};

class VertexParMeshPartition
{
private:
   ParMesh *pmesh=nullptr;
public:
   int nrpatch;
   Array<int> patch_rank;
   std::vector<Array<int>> local_element_map;
   // constructor
   VertexParMeshPartition(ParMesh * pmesh_);
   ~VertexParMeshPartition() {};
};

class ParMeshPartition
{
private:
   MPI_Comm comm;
   ParMesh *pmesh=nullptr;
   void AddElementToMesh(Mesh * mesh,mfem::Element::Type elem_type,int * ind);
   void GetNumVertices(int type, mfem::Element::Type & elem_type, int & nrvert);
   void SaveMeshPartition();
public:
   int nrpatch;
   int myelem_offset = 0;
   Array<int> patch_rank;
   std::vector<Array<int>> element_map;
   std::vector<Array<int>> local_element_map;
   Array<Mesh *> patch_mesh;
   // constructor
   ParMeshPartition(ParMesh * pmesh_, int part);
   ~ParMeshPartition();
};

class ParPatchDofInfo
{
public:
   MPI_Comm comm = MPI_COMM_WORLD;
   int nrpatch;
   Array<int> patch_rank;
   vector<Array<int>> PatchGlobalTrueDofs; // list of all the true dofs in a patch
   vector<Array<int>> PatchTrueDofs; // list of only
   Array<FiniteElementSpace *> patch_fespaces;
   std::vector<Array<int>> patch_dof_map;
   ParMeshPartition * p;
   // constructor
   ParPatchDofInfo(ParFiniteElementSpace *fespace, int part);
   // void Print();
   ~ParPatchDofInfo();
};



class ParPatchAssembly // for now every vertex defines a patch
{
private:
   std::vector<int> tdof_offsets;
   ParBilinearForm *bf=nullptr;
   void compute_trueoffsets();
   void AssemblePatchMatrices(ParPatchDofInfo * p);
   void print_patch_dof_map() {};
public:
   MPI_Comm comm;
   int nrpatch;
   ParFiniteElementSpace *fespace=nullptr;
   Array<int> patch_rank;
   std::vector<Array<int>> patch_true_dofs;
   std::vector<Array<int>> patch_local_dofs;

   Array<SparseMatrix *> patch_mat;
   Array<BilinearForm * > patch_bilinear_forms;
   Array<KLUSolver * > patch_mat_inv;
   std::vector<Array<int>> ess_tdof_list;

   // constructor
   ParPatchAssembly(ParBilinearForm * bf_, int part);
   int get_rank(int tdof);
   ~ParPatchAssembly();
};


class ParPatchRestriction
{
private:
   MPI_Comm comm;
   int num_procs, myid;
   Array<int> patch_rank;
   ParPatchAssembly * P;
   int nrpatch;
   Array<int> send_count;
   Array<int> send_displ;
   Array<int> recv_count;
   Array<int> recv_displ;
   int sbuff_size, rbuff_size;
public:
   ParPatchRestriction(ParPatchAssembly * P_);
   // void Mult(const Vector & r , Array<BlockVector *> & res);
   void Mult(const Vector & r , std::vector<Vector  > & res);
   // void MultTranspose(const Array<BlockVector*> & sol, Vector & z);
   void MultTranspose(const std::vector<Vector  > & sol, Vector & z);
   virtual ~ParPatchRestriction() {}
};


class ParAddSchwarz : public Solver//
{
private:
   MPI_Comm comm;
   int nrpatch;
   int part;
   int maxit = 1;
   double theta = 0.5;
   ParPatchAssembly * p;
   const Operator * A;
   ParPatchRestriction * R;
public:
   ParAddSchwarz(ParBilinearForm * bf_, int i = 0);

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
   virtual ~ParAddSchwarz();
};


