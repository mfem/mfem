#include "mfem.hpp"

namespace mfem
{

class EllipticSolver
{
private:
   BilinearForm &a;
   Array<int> ess_tdof_list;
   std::unique_ptr<CGSolver> solver;
   std::unique_ptr<GSSmoother> prec;
   OperatorHandle A;
   Vector B;
   Vector X;
   bool parallel;
#ifdef MFEM_USE_MPI
   ParBilinearForm *par_a;
   MPI_Comm comm;
   std::unique_ptr<HyprePCG> par_solver;
   std::unique_ptr<HypreBoomerAMG> par_prec;
   HypreParVector par_B;
#endif
public:
private:
   void BuildEssTdofList();
   void SetupSolver();
public:
   EllipticSolver(BilinearForm &a, Array<int> &ess_bdr);
   EllipticSolver(BilinearForm &a, Array2D<int> &ess_bdr);
   void UseElasticityOption();
   void Solve(LinearForm &b, GridFunction &x);
   // This must be called after reassembly
   void Update();
};

}// namespace mfem
