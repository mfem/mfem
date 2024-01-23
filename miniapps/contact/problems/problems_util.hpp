#include "mfem.hpp"
#include "../util/mpicomm.hpp"

using namespace std;
using namespace mfem;

void BasisEval(const Vector xi, Vector &N, DenseMatrix &dNdxi); // dNdxi is 2*4
void BasisEvalDerivs(const Vector xi, Vector& N, DenseMatrix& dNdxi,
                     DenseMatrix& dN2dxi);
// returns the vector and matrix form of the shape functions and its derivative
void BasisVectorDerivs(const Vector xi, DenseMatrix& N, DenseMatrix& dNdxi,
                       DenseMatrix& ddNdxi);
void cross(const Vector a, const Vector b, Vector& c);
// a outer b
void outer(const Vector a, const Vector b, DenseMatrix& c);
// dphidxi 2*4
// coords 4*3
void ComputeNormal(const DenseMatrix& dphidxi, const DenseMatrix& coords,
                   Vector& normal, double& nnorm);
void SlaveToMaster(const DenseMatrix& m_coords, const Vector& s_x, Vector& xi);

// m_coords is expected to be 4 * 3
void  ComputeGapJacobian(const Vector x_s, const Vector xi,
                         const DenseMatrix m_coords,
                         double& gap, Vector& normal, Vector& dgdxm, Vector& dgdxs);

void ComputeGapHessian(const Vector x_s, const Vector xi,
                       const DenseMatrix m_coords,
                       DenseMatrix& dg2dx);
void NodeSegConPairs(const Vector x1, const Vector xi2,
                     const DenseMatrix coords2,
                     double& node_g, Vector& node_dg, DenseMatrix& node_dg2);
// coordsm : (npoints*4, 3) use what class?
// m_conn: (npoints*4)
void Assemble_Contact(const Vector x_s,
                      const Vector xi, const DenseMatrix coordsm, const Array<int> s_conn,
                      const Array<int> m_conn, Vector& g, SparseMatrix& M,
                      Array<SparseMatrix *> & dM);

void Assemble_Contact(const Vector x_s,
                      const Vector xi, const DenseMatrix coordsm, const Array<int> s_conn,
                      const Array<int> m_conn, Vector & g, SparseMatrix & M1, SparseMatrix & M2, 
                      Array<SparseMatrix *> & dM11,
                      Array<SparseMatrix *> & dM12,
                      Array<SparseMatrix *> & dM21,
                      Array<SparseMatrix *> & dM22);
void Assemble_Contact(const Vector x_s,
                      const Vector xi, const DenseMatrix coordsm, const Array<int> s_conn,
                      const Array<int> m_conn, Vector & g, SparseMatrix & M1, SparseMatrix & M2,const Array<int> & points_map); 

void FindSurfaceToProject(Mesh& mesh, const int elem, int& cbdrface);

Vector GetNormalVector(Mesh & mesh, const int elem, const double *ref,
                       int & refFace, int & refNormal, bool & interior);
int GetHexVertex(int cdim, int c, int fa, int fb, Vector & refCrd);

// Coordinates in xyz are assumed to be ordered as [X, Y, Z]
// where X is the list of x-coordinates for all points and so on.
// conn: connectivity of the target surface elements
// xi: surface reference cooridnates for the cloest point, involves a linear transformation from [0,1] to [-1,1]
void FindPointsInMesh(Mesh & mesh, Vector const& xyz, Array<int>& conn, Vector& xi);

#ifdef MFEM_USE_MPI
// Coordinates in xyz are assumed to be ordered as [X, Y, Z]
// where X is the list of x-coordinates for all points and so on.
// conn: connectivity of the target surface elements
// xi: surface reference cooridnates for the cloest point, involves a linear transformation from [0,1] to [-1,1]
void FindPointsInMesh(Mesh & mesh, const Array<int> & gvert, const Vector & xyz, const Array<int> & s_conn, Array<int>& conn,
                      Vector & xyz2, Array<int> & s_conn2, Vector& xi, DenseMatrix & coords);

// somewhat simplified version of the above
void FindPointsInMesh(Mesh & mesh, const Array<int> & gvert, Array<int> & s_conn, const Vector &x1, Vector & xyz, Array<int>& conn,
                      Vector& xi, DenseMatrix & coords);                   

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

#endif
