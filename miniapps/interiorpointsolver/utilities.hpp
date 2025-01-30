#ifndef UTILITY_FUNCTIONS
#define UTILITY_FUNCTIONS

#include "mfem.hpp"

// using namespace mfem;

namespace mfem {

void HypreToMfemOffsets(HYPRE_BigInt * offsets);

HypreParMatrix * GenerateHypreParMatrixFromSparseMatrix(HYPRE_BigInt * colOffsetsloc, HYPRE_BigInt * rowOffsetsloc, SparseMatrix * Asparse);

HypreParMatrix * GenerateHypreParMatrixFromDiagonal(HYPRE_BigInt * offsetsloc, 
		mfem::Vector & diag);


HypreParMatrix * GenerateProjector(HYPRE_BigInt * offsets, HYPRE_BigInt * reduced_offsets, HYPRE_Int * mask);

HypreParMatrix * GenerateProjector(HYPRE_BigInt * offsets, HYPRE_BigInt * reduced_offsets, const HypreParVector & mask);


HYPRE_BigInt * offsetsFromLocalSizes(int n);

}


#endif
