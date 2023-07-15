
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
void Assemble_Contact(const int m, 
                      const Vector x_s,
                      const Vector xi, const DenseMatrix coordsm, const Array<int> s_conn,
                      const Array<int> m_conn, Vector& g, SparseMatrix& M,
                      Array<SparseMatrix *> & dM);

void FindSurfaceToProject(Mesh& mesh, const int elem, int& cbdrface);

Vector GetNormalVector(Mesh & mesh, const int elem, const double *ref,
                       int & refFace, int & refNormal, bool & interior);
int GetHexVertex(int cdim, int c, int fa, int fb, Vector & refCrd);
