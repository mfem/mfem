#include "mfem.hpp"
using namespace mfem;
extern "C" void
dgelss_(int *, int *, int *, double *, int *, double *, int *, double *,
        double *, int *, double *, int *, int *);
extern "C" void
dgels_(char *, int *, int *, int *, double *, int *, double *, int *, double *,
       int *, int *);

extern "C" void
dgelsy_(int *, int *, int *, double *, int *, double *, int *, int *, double *,
       int *, double *, int *, int *);
void buildLSInterpolation(int dim, int degree, const Vector &cent, const DenseMatrix &x_merge,
                          const DenseMatrix &x_quad, DenseMatrix &interp);

void buildInterpolation(int dim, int degree, int output,
                        const DenseMatrix &x_center, const DenseMatrix &x_quad,
                        DenseMatrix &interp);

/// Abstract class for Galerkin difference method using patch construction
class CellMerging
{
protected:
   /// mesh object
   Mesh *mesh;
   /// mesh dimension
   int dim;
   /// number of elements in mesh
   int nEle;
   /// degree of lagrange interpolation
   int degree;
   /// number of unknowns per dof
   int vdim;
   /// the size of cut element
   double scale;
   /// finite element collection
   const mfem::FiniteElementCollection *fec; // not owned
   /// finite element space
   const mfem::FiniteElementSpace *fespace;
public:
   /// Class constructor. 
   CellMerging(mfem::Mesh *m, int di, int ne, const mfem::FiniteElementCollection *f, 
               const mfem::FiniteElementSpace *fes, int vdi= 1, 
               int de = 0, double s = 1)
               : mesh(m), dim(di), nEle(ne), fec(f), fespace(fes), vdim(vdi), 
               degree(de), scale(s){}
   /// An overload function for build the densmatrix
   void BuildNeighbourMat(const Array<int> &els_id,
                          DenseMatrix &mat_cent,
                          DenseMatrix &mat_quad) const;

   /// prolongation operator for cell merging.
   mfem::SparseMatrix getProlongationOperator() const;
   /// provides the center (barycenter) of an element
   /// \param[in]  id - the id of the element for which we need barycenter
   /// \param[out] cent - the vector of coordinates of center of an element
   void GetElementCenter(int id, mfem::Vector &cent) const;

};
