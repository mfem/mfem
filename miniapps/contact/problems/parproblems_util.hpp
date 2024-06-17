
#include "mfem.hpp"


using namespace std;
using namespace mfem;

#include "axom/slic.hpp"

#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"
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