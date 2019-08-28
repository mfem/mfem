//                                MFEM Example 1
//
// Compile with: make ex1
//

#include "mfem.hpp"

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
   patch_nod_info(Mesh * mesh_, int ref_levels_,Array<int> ess_dof_list);
private:
   int ref_levels=0;;
   Mesh *mesh=nullptr;
};
// constructor


struct patch_assembly 
{
   int nrpatch;
   int ref_levels;
   Mesh cmesh;
   Array<SparseMatrix *> Pid; 
   // constructor
   patch_assembly(Mesh * cmesh_, int ref_levels_,FiniteElementSpace *fespace,Array<int> ess_dof_list);
};

class SchwarzSmoother : public Solver {
private:
   int nrpatch;
   /// The linear system matrix
   SparseMatrix * A;
   patch_assembly * P;
   Array<SparseMatrix  *> A_local;
   Array<UMFPackSolver *> invA_local;
   Array<int>vert_dofs;
   Schwarz::SmootherType sType=Schwarz::SmootherType::ADDITIVE; 
public:
   SchwarzSmoother(Mesh * cmesh_, int ref_levels_, FiniteElementSpace *fespace,SparseMatrix *A_, Array<int> ess_dof_list);

   void SetType(const Schwarz::SmootherType Type) {sType = Type;}
   virtual void SetOperator(const Operator &op) {}
   virtual void Mult(const Vector &r, Vector &z) const;
   bool IsEssential(int idof, Array<int> ess_dof_list);
   virtual ~SchwarzSmoother() {}
};
