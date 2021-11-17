#pragma once
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

bool is_a_patch(int iv, Array<int> patch_ids);

bool owned(int tdof, int * offs);

SparseMatrix * GetLocalRestriction(const Array<int> & tdof_i, const int * row_start, 
                                   int num_rows, int num_cols);                              

void GetLocal2GlobalMap(const Array<int> & tdof_i, const int * row_start, 
                        int num_rows, int num_cols, Array<int> & l2gmap);

void GetOffdColumnValues(const Array<int> & tdof_i, const Array<int> & tdof_j, SparseMatrix & offd, const int * cmap, 
                         const int * row_start, SparseMatrix * PatchMat);

void GetArrayIntersection(const Array<int> & A, const Array<int> & B, Array<int>  & C); 



int GetNumColumns(const int tdof_i, const Array<int> & tdof_j, SparseMatrix & diag,
SparseMatrix & offd, const int * cmap, const int * row_start);

void GetColumnValues(int tdof_i,const Array<int> & tdof_j, SparseMatrix & diag ,
SparseMatrix & offd, const int *cmap, const int * row_start, Array<int> &cols, Array<double> &vals);


class VertexPatchInfo 
{
private:
   ParMesh * pmesh = nullptr;
   int ref_levels=0;
public:
   int mynrpatch;
   int nrpatch;
   std::vector<Array<int>> vert_contr;
   std::vector<Array<int>> edge_contr;
   std::vector<Array<int>> face_contr;
   std::vector<Array<int>> elem_contr;
   Array<int> host_rank;
   Array<int> patch_natural_order_idx;
   Array<int> patch_global_dofs_ids;

   VertexPatchInfo(ParMesh * pmesh_, int ref_levels_);
   ~VertexPatchInfo() {}

};

class PatchDofInfo 
{
public:
   MPI_Comm comm = MPI_COMM_WORLD;
   int nrpatch;
   Array<int> host_rank;
   std::vector<Array<int>> patch_tdofs;
   std::vector<Array<int>> patch_local_tdofs;
   PatchDofInfo(ParMesh * cpmesh_, int ref_levels_,ParFiniteElementSpace *fespace);
   ~PatchDofInfo(){};
};

class PatchAssembly
{
public:
   MPI_Comm comm;
   int nrpatch;
   std::vector<int>tdof_offsets;
   std::vector<Array<int>> patch_other_tdofs;
   std::vector<Array<int>> patch_owned_other_tdofs;
   std::vector<Array<int>> l2gmaps; // patch to global maps for the dofs owned by the processor
   Array<SparseMatrix* > PatchMat;
   PatchDofInfo *patch_tdof_info=nullptr;   
   Array<int> host_rank; 
   HypreParMatrix * A = nullptr;
   int get_rank(int tdof);
   // constructor
   PatchAssembly(ParMesh * cpmesh_, int ref_levels_,ParFiniteElementSpace *fespace_, HypreParMatrix * A_);
   ~PatchAssembly();
private:
   void compute_trueoffsets();
   ParFiniteElementSpace *fespace=nullptr;
};

class PatchRestriction  {
private:
   MPI_Comm comm;
   int num_procs, myid;
   Array<int> host_rank;
   PatchAssembly * P;
   int nrpatch;
   Array<int> send_count;
   Array<int> send_displ;
   Array<int> recv_count;
   Array<int> recv_displ;
   int sbuff_size;
   int rbuff_size;
public:
   PatchRestriction(PatchAssembly * P_);
   void Mult(const Vector & r , Array<BlockVector *> & res);
   void MultTranspose(const Array<BlockVector *> & sol, Vector & z);
   virtual ~PatchRestriction() {}
};

class SchwarzSmoother : public Solver {
private:
   MPI_Comm comm;
   int nrpatch;
   int maxit = 1;
   double theta = 1.0;
   Array<int> host_rank;
#ifdef MFEM_USE_SUITESPARSE
   Array<UMFPackSolver * > PatchInv;
#else   
   Array<GMRESSolver * > PatchInv;
#endif   /// The linear system matrix
   HypreParMatrix * A;
   PatchAssembly * P;
   PatchRestriction * R= nullptr;
public:
   SchwarzSmoother(ParMesh * cpmesh_, int ref_levels_,ParFiniteElementSpace *fespace_, HypreParMatrix * A_);
   void SetNumSmoothSteps(const int iter) {maxit = iter;}
   void SetDumpingParam(const double dump_param) {theta = dump_param;}
   virtual void SetOperator(const Operator &op) {}
   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~SchwarzSmoother();
};


class ComplexSchwarzSmoother : public Solver {
private:
   MPI_Comm comm;
   int nrpatch;
   int maxit = 1;
   double theta = 1.0;
   Array<int> host_rank;
#ifdef MFEM_USE_SUITESPARSE
   Array<UMFPackSolver * > PatchInv;
#else   
   Array<GMRESSolver * > PatchInv;
#endif   /// The linear system matrix
   ComplexHypreParMatrix * A;
   PatchAssembly * P_r;
   PatchAssembly * P_i;
   PatchRestriction * R_r= nullptr;
   PatchRestriction * R_i= nullptr;
public:
   ComplexSchwarzSmoother(ParMesh * cpmesh_, int ref_levels_,ParFiniteElementSpace *fespace_, ComplexHypreParMatrix * A_);
   void SetNumSmoothSteps(const int iter) {maxit = iter;}
   void SetDumpingParam(const double dump_param) {theta = dump_param;}
   virtual void SetOperator(const Operator &op) {}
   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~ComplexSchwarzSmoother();
};


