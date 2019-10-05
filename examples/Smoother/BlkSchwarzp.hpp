#include "mfem.hpp"

#include <fstream>
#include <iostream>
#include "Schwarzp.hpp"

using namespace std;
using namespace mfem;


class BlkParSchwarzSmoother : virtual public Solver {
private:
   MPI_Comm comm;
   int nrpatch;
   int maxit = 1;
   double theta = 0.5;
   Array<int> host_rank;
   Array<UMFPackSolver * > PatchInv;
   Array<SparseMatrix * > PatchMat;
   /// The linear system matrix
   Array2D<HypreParMatrix*> blockA;
   Array2D<par_patch_assembly *> P;
   Array2D<PatchRestriction *> R;
   BlockOperator * blk = nullptr;
public:
   BlkParSchwarzSmoother(ParMesh * cpmesh_, int ref_levels_,ParFiniteElementSpace *fespace_,
                         Array2D<HypreParMatrix*> blockA_,Array<int> ess_tdof_list);
   void SetNumSmoothSteps(const int iter) {maxit = iter;}
   void SetDumpingParam(const double dump_param) {theta = dump_param;}
  
   virtual void SetOperator(const Operator &op) {}
   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~BlkParSchwarzSmoother() 
   {
      for (int ip=0; ip<nrpatch; ip++)
      {
         delete PatchMat[ip];
      }
   }
};