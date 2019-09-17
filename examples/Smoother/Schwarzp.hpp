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
private:
   int ref_levels=0;;
   ParMesh pmesh;
   FiniteElementCollection *aux_fec=nullptr;
   ParFiniteElementSpace *aux_fespace=nullptr;
};
struct par_patch_dof_info 
{
   MPI_Comm comm = MPI_COMM_WORLD;
   int mynrpatch;
   int nrpatch;
   vector<Array<int>> patch_tdofs;
   vector<Array<int>> patch_ldofs;
   // constructor
   par_patch_dof_info(ParMesh * cpmesh_, int ref_levels_,ParFiniteElementSpace *fespace);
   void Print();
};

struct par_patch_assembly
{
   MPI_Comm comm;
   int mynrpatch;
   int nrpatch;
   Array<int>tdof_offsets;
   vector<Array<int>> patch_tdofs;
   HypreParMatrix * A = nullptr;
   ParFiniteElementSpace *fespace=nullptr;
   // constructor
   par_patch_assembly(ParMesh * cpmesh_, int ref_levels_,ParFiniteElementSpace *fespace_, HypreParMatrix * A_);
   int get_rank(int tdof);
   int compute_trueoffsets();

};


bool its_a_patch(int iv, Array<int> patch_ids);
void GetColumnValues(int tdof_i,Array<int> tdof_j, HypreParMatrix * A, 
                     Array<int> & cols, Array<double> & vals);

