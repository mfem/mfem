
#include "mfem.hpp"
#include "problems_util.hpp"
#include "../util/mpicomm.hpp"
#include "axom/slic.hpp"

#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"
// Coordinates in xyz are assumed to be ordered as [X, Y, Z]
// where X is the list of x-coordinates for all points and so on.
// conn: connectivity of the target surface elements
// xi: surface reference cooridnates for the cloest point, involves a linear transformation from [0,1] to [-1,1]
void FindPointsInMesh(Mesh & mesh, const Array<int> & gvert, const Vector & xyz, const Array<int> & s_conn, Array<int>& conn,
                      Vector & xyz2, Array<int> & s_conn2, Vector& xi, DenseMatrix & coords);

// somewhat simplified version of the above
void FindPointsInMesh(ParMesh & mesh, const Array<int> & gvert, Array<int> & s_conn, const Vector &x1, Vector & xyz, Array<int>& conn,
                      Vector& xi, DenseMatrix & coords, bool singlemesh = false);                   

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