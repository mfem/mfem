#pragma once
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "additive_schwarzp.hpp"
using namespace std;
using namespace mfem;



class ComplexParPatchAssembly
{
   // std::vector<int> tdof_offsets;
   ParSesquilinearForm * bf=nullptr;
   void compute_trueoffsets();
   void AssemblePatchMatrices(ParPatchDofInfo * p);
public:
   // MPI_Comm comm;
   // int nrpatch;
   // ParFiniteElementSpace *fespace=nullptr;
   // Array<int> patch_rank;
   // std::vector<Array<int>> patch_true_dofs;
   // std::vector<Array<int>> patch_local_dofs;
   // Array<SparseMatrix *> patch_mat;
   // Array<SesquilinearForm * > patch_bilinear_forms;
   // Array<KLUSolver * > patch_mat_inv;
   // std::vector<Array<int>> ess_tdof_list;

   // constructor
   ComplexParPatchAssembly(ParSesquilinearForm * bf_);
   int get_rank(int tdof);
   ~ComplexParPatchAssembly();
};


class ComplexParPatchRestriction
{
private:
   // MPI_Comm comm;
   // int num_procs, myid;
   // Array<int> patch_rank;
   // ParPatchAssembly * P;
   // int nrpatch;
   // Array<int> send_count;
   // Array<int> send_displ;
   // Array<int> recv_count;
   // Array<int> recv_displ;
   // int sbuff_size, rbuff_size;
public:
   ComplexParPatchRestriction(ComplexParPatchAssembly * P_);
   void Mult(const Vector & r , std::vector<Vector > & res);
   void MultTranspose(const std::vector<Vector > & sol, Vector & z);
   virtual ~ComplexParPatchRestriction() {}
};


class ComplexParAddSchwarz : public Solver//
{
private:
   // MPI_Comm comm;
   // int nrpatch;
   // int maxit = 1;
   // double theta = 0.5;
   // FiniteElementSpace *fespace=nullptr;
   // ComplexParPatchAssembly * p;
   // const Operator * A;
   // ParSesquilinearForm * pbf;
   // ComplexParPatchRestriction * R;
public:
   ComplexParAddSchwarz(ParSesquilinearForm * pbf_);
   void SetNumSmoothSteps(const int iter)
   {
      // maxit = iter;
   }
   void SetDumpingParam(const double dump_param)
   {
      // theta = dump_param;
   }
   virtual void SetOperator(const Operator &op)
   {
      // A = &op;
   }
   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~ComplexParAddSchwarz();
};


