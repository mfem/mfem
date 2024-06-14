#pragma once
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

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

class LinePatchInfo
{
private:
   ParMesh * pmesh = nullptr;
   VectorCoefficient &BCoef;
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
   //Array<int> patch_global_dofs_ids;

   LinePatchInfo(ParMesh * pmesh_, VectorCoefficient & BCoef_, int ref_levels_);
   ~LinePatchInfo() {}
};

class PatchDofInfo
{
public:
   MPI_Comm comm = MPI_COMM_WORLD;
   int nrpatch;
   Array<int> host_rank;
   std::vector<Array<HYPRE_BigInt>> patch_tdofs;
   std::vector<Array<HYPRE_BigInt>> patch_local_tdofs;
   PatchDofInfo(ParMesh * cpmesh_,
                VectorCoefficient & BCoef_,
                int ref_levels_,
                ParFiniteElementSpace *fespace);
   ~PatchDofInfo() {};
};

class PatchAssembly
{
public:
   MPI_Comm comm;
   int nrpatch;
   std::vector<HYPRE_BigInt> tdof_offsets;
   std::vector<Array<HYPRE_BigInt>> patch_other_tdofs;
   std::vector<Array<HYPRE_BigInt>> patch_owned_other_tdofs;
   std::vector<Array<int>>
                        l2gmaps; // patch to global maps for the dofs owned by the processor
   Array<SparseMatrix* > PatchMat;
   PatchDofInfo *patch_tdof_info=nullptr;
   Array<int> host_rank;
   HypreParMatrix * A = nullptr;
   int get_rank(HYPRE_BigInt tdof);
   // constructor
   PatchAssembly(ParMesh * cpmesh_, VectorCoefficient & BCoef_, int ref_levels_,
                 ParFiniteElementSpace *fespace_, HypreParMatrix * A_);
   ~PatchAssembly();
private:
   void compute_trueoffsets();
   ParFiniteElementSpace *fespace=nullptr;
};

class PatchRestriction
{
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
   void Mult(const Vector & r, Array<BlockVector *> & res);
   void MultTranspose(const Array<BlockVector *> & sol, Vector & z);
   virtual ~PatchRestriction() {}
};

class SparseBooleanMatrix
{
public:
   SparseBooleanMatrix(int dim) : n(dim), a(dim) { };

   SparseBooleanMatrix(SparseBooleanMatrix const& M);

   void SetEntry(int row, int col);

   int GetSize() const { return n; }

   const std::vector<int>& GetRow(int row) const { return a[row]; }

   SparseBooleanMatrix* Mult(SparseBooleanMatrix const& M) const;

   SparseBooleanMatrix* Transpose() const;

private:
   const int n;

   std::vector<std::vector<int>> a;
};

class SchwarzSmoother : public Solver
{
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
   SchwarzSmoother(ParMesh * cpmesh_, VectorCoefficient & BCoef_,
                   int ref_levels_, ParFiniteElementSpace * fespace_,
                   HypreParMatrix * A_);
   void SetNumSmoothSteps(const int iter) {maxit = iter;}
   void SetDampingParam(const double dump_param) {theta = dump_param;}
   virtual void SetOperator(const Operator &op) {}
   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~SchwarzSmoother();
};

//void TotBFunc(const Vector &x, Vector &B);

