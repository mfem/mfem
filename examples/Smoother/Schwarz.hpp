#include "mfem.hpp"
#include "util.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

namespace Schwarz
{
   enum SmootherType{ADDITIVE, MULTIPLICATIVE, SYM_MULTIPLICATIVE};
}

struct patch_nod_info 
{
   int nrpatch;
   vector<Array<int>> vertex_contr;
   vector<Array<int>> edge_contr;
   vector<Array<int>> face_contr;
   vector<Array<int>> elem_contr;

   // constructor
   patch_nod_info(Mesh * mesh_, int ref_levels_);
private:
   int ref_levels=0;;
   Mesh *mesh=nullptr;
};

struct patch_assembly 
{
   int nrpatch;
   int ref_levels;
   Mesh cmesh;
   Array<SparseMatrix *> Pid; 
   // constructor
   patch_assembly(Mesh * cmesh_, int ref_levels_,FiniteElementSpace *fespace);
};

class SchwarzSmoother : virtual public Solver {
private:
   int nrpatch;
   /// The linear system matrix
   SparseMatrix * A;
   patch_assembly * P;
   Array<SparseMatrix  *> A_local;
   Array<UMFPackSolver *> invA_local;
   Array<int>vert_dofs;
   Schwarz::SmootherType sType=Schwarz::SmootherType::ADDITIVE; 
   vector<int> patch_ids;
   int maxit = 1;
   double theta = 0.5;
public:
   SchwarzSmoother(Mesh * cmesh_, int ref_levels_, FiniteElementSpace *fespace,SparseMatrix *A_, Array<int> ess_bdr);
   // SchwarzSmoother(Mesh *cmesh_, int ref_levels_, FiniteElementSpace *fespace_, SparseMatrix *A_);



   void SetType(const Schwarz::SmootherType Type) {sType = Type;}
   void SetNumSmoothSteps(const int iter) {maxit = iter;}
   void SetDumpingParam(const double dump_param) {theta = dump_param;}
   virtual void SetOperator(const Operator &op) {}
   virtual void Mult(const Vector &r, Vector &z) const;
   void GetNonEssentialPatches(Mesh * cmesh, const Array<int> &ess_bdr, vector <int> & patch_ids);
   virtual ~SchwarzSmoother() {}
};


class BlkSchwarzSmoother : public Solver {
private:
   int nrpatch;
   /// The linear system matrix
   SparseMatrix * A;
   patch_assembly * P;
   Array<SparseMatrix  *> A_local;
   Array<UMFPackSolver *> invA_local;
   Array<int>vert_dofs;
   Schwarz::SmootherType sType=Schwarz::SmootherType::ADDITIVE; 
   vector<int> patch_ids;
   int maxit = 1;
   double theta = 0.5;
public:
   BlkSchwarzSmoother(Mesh *cmesh_, int ref_levels_, FiniteElementSpace *fespace_, SparseMatrix *A_);
   void SetType(const Schwarz::SmootherType Type) {sType = Type;}
   void SetNumSmoothSteps(const int iter) {maxit = iter;}
   void SetDumpingParam(const double dump_param) {theta = dump_param;}
   virtual void SetOperator(const Operator &op) {}
   virtual void Mult(const Vector &r, Vector &z) const;
   // void GetNonEssentialPatches(Mesh * cmesh, const Array<int> &ess_bdr, vector <int> & patch_ids);
   virtual ~BlkSchwarzSmoother() {}
};

