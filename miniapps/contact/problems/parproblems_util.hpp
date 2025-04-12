
#include "mfem.hpp"


using namespace std;
using namespace mfem;

#include "axom/slic.hpp"

#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"

int get_rank(int tdof, std::vector<int> & tdof_offsets);
void ComputeTdofOffsets(const ParFiniteElementSpace * pfes,
                        std::vector<int> & tdof_offsets);
void ComputeTdofOffsets(MPI_Comm comm, int mytoffset, std::vector<int> & tdof_offsets);
void ComputeTdofs(MPI_Comm comm, int mytoffs, std::vector<int> & tdofs);


// Performs Pᵀ * A * P for BlockOperator  P (with blocks as HypreParMatrices)
// and A a HypreParMatrix, i.e., this handles the special case 
// where P = [P₁ P₂ ⋅⋅⋅ Pₙ] 
void RAP(const HypreParMatrix & A, const BlockOperator & P, BlockOperator & C);
void ParAdd(const BlockOperator & A, const BlockOperator & B, BlockOperator & C);

class GeneralSolutionMonitor : public IterativeSolverMonitor
{
public:
   GeneralSolutionMonitor(ParFiniteElementSpace * fes_, HypreParMatrix * A, Vector & B, int output_rate);

   void MonitorResidual(int it, real_t norm, const Vector &r, bool final) override;
   void MonitorSolution(int it, real_t norm, const Vector &x, bool final) override;

   ~GeneralSolutionMonitor()
   {
      delete pgf;
      delete error_gf;
      delete true_gf;
      delete paraview_dc;
   }
private:

    ParFiniteElementSpace * fes = nullptr;
    ParGridFunction * true_gf = nullptr;
    ParGridFunction * error_gf = nullptr;
    ParGridFunction * pgf = nullptr;
    ParaViewDataCollection * paraview_dc = nullptr;
    int counter = 0;
    int output_rate;
};

