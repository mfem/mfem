#include "mfem.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


struct par_patch_nod_info 
{
   int mynrpatch;
   int nrpatch;
   vector<Array<int>> vert_contr;
   vector<Array<int>> edge_contr;
   vector<Array<int>> face_contr;
   vector<Array<int>> elem_contr;
   Array<int> patch_natural_order_idx;
   Array<int> patch_global_dofs_ids;
   // constructor
   par_patch_nod_info(ParMesh * cpmesh_, int ref_levels_);
   // Print
   void Print(int rank_id);
   ~par_patch_nod_info() 
   {
      delete aux_fespace;
   }
private:
   int ref_levels=0;
   ParMesh pmesh;
   ParFiniteElementSpace *aux_fespace=nullptr;
};
struct par_patch_dof_info 
{
   MPI_Comm comm = MPI_COMM_WORLD;
   int mynrpatch;
   int nrpatch;
   vector<Array<int>> patch_tdofs;
   vector<Array<int>> patch_local_tdofs;
   // constructor
   par_patch_dof_info(ParMesh * cpmesh_, int ref_levels_,ParFiniteElementSpace *fespace);
   void Print();
};

struct par_patch_assembly
{
   MPI_Comm comm;
   int mynrpatch;
   int nrpatch;
   vector<int>tdof_offsets;
   vector<Array<int>> patch_other_tdofs;
   vector<Array<int>> patch_owned_other_tdofs;
   Array<SparseMatrix * > PatchMat;
   Array<SparseMatrix * > Prl;
   HypreParMatrix * A = nullptr;
   par_patch_dof_info *patch_tdof_info=nullptr;   
   ParFiniteElementSpace *fespace=nullptr;
   Array<int> host_rank; 
   int get_rank(int tdof);
   // constructor
   par_patch_assembly(ParMesh * cpmesh_, int ref_levels_,ParFiniteElementSpace *fespace_, HypreParMatrix * A_);
   ~par_patch_assembly()
   {
      delete patch_tdof_info;
   }
private:
   void compute_trueoffsets();
};


class ParSchwarzSmoother : virtual public Solver {
private:
   MPI_Comm comm;
   int nrpatch;
   int maxit = 1;
   double theta = 0.5;
   Array<int> host_rank;
   Array<UMFPackSolver * > PatchInv;
   /// The linear system matrix
   HypreParMatrix * A;
   par_patch_assembly * P;
public:
   ParSchwarzSmoother(ParMesh * cpmesh_, int ref_levels_,ParFiniteElementSpace *fespace_, HypreParMatrix * A_);
   void SetNumSmoothSteps(const int iter) {maxit = iter;}
   void SetDumpingParam(const double dump_param) {theta = dump_param;}
  
   virtual void SetOperator(const Operator &op) {}
   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~ParSchwarzSmoother() {}


};


bool its_a_patch(int iv, Array<int> patch_ids);
SparseMatrix * GetDiagColumnValues(const Array<int> & tdof_i, SparseMatrix & diag,
const int * row_start);

SparseMatrix * GetLocalProlongation(const Array<int> & tdof_i, const int * row_start, 
                              const int num_rows, const int num_cols);

void GetOffdColumnValues(const Array<int> & tdof_i, const Array<int> & tdof_j, SparseMatrix & offd, const int * cmap, 
                         const int * row_start, SparseMatrix * PatchMat);

void GetArrayIntersection(const Array<int> & A, const Array<int> & B, Array<int>  & C); 



int GetNumColumns(const int tdof_i, const Array<int> & tdof_j, SparseMatrix & diag,
SparseMatrix & offd, const int * cmap, const int * row_start);

void GetColumnValues(const int tdof_i,const Array<int> & tdof_j, SparseMatrix & diag ,
SparseMatrix & offd, const int *cmap, const int * row_start, Array<int> &cols, Array<double> &vals);


// void GetColumnValues2(const int tdof_i,const Array<int> & tdof_j, SparseMatrix & diag ,
// SparseMatrix & offd, const int *cmap, const int * row_start, Array<int> &cols, Array<double> &vals);
