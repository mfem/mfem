
#include "mfem.hpp"

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
